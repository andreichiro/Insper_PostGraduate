#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analytics_v2_common import (
    DEFAULT_BASE_DIR,
    V2Config,
    build_config,
    connect_duckdb,
    persist_df_to_duckdb,
    safe_auc,
    safe_average_precision,
    setup_logging,
    top_decile_lift,
    utc_now_iso,
    write_df_bundle,
    write_json,
    write_markdown,
)


PROFILE_CONTROL_VAR = "currentsubject_group"
RECENT_WINDOW_MONTHS = 6
MIN_PROFILE_SLICE_ROWS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Etapa 09 v2: predição de abandono/retorno e auditoria de drift.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def persist_output(conn: Any, cfg: V2Config, name: str, df: pd.DataFrame) -> None:
    persist_df_to_duckdb(conn, name, df)
    write_df_bundle(cfg.output_dir, name, df)


def require_tables(conn: Any, table_names: Sequence[str]) -> None:
    existing = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
    missing = [name for name in table_names if name not in existing]
    if missing:
        raise RuntimeError(
            "Tabelas obrigatorias ausentes para etapa_09_prediction_drift_v2.py: "
            + ", ".join(sorted(missing))
        )


def normalize_text(series: pd.Series, default: str = "missing") -> pd.Series:
    out = series.fillna(default).astype(str)
    out = out.replace({"<missing>": default, "None": default, "nan": default}).str.strip()
    return out.replace("", default)


def load_prediction_base(conn: Any) -> pd.DataFrame:
    query = """
    SELECT
      tm.teacher_unique_id,
      tm.month,
      tm.active_user_flag,
      tm.next_month_observed_flag,
      tm.returned_active_m1,
      tm.returned_any_download_m1,
      tm.session_count_month,
      tm.total_session_minutes_month,
      tm.avg_session_minutes_month,
      tm.interaction_rows_month,
      tm.activity_events_month,
      tm.active_days_month,
      tm.aula_events_month,
      tm.plano_events_month,
      tm.prova_events_month,
      tm.ia_events_month,
      tm.strict_download_count_month,
      tm.content_views_month,
      tm.other_activity_non_download_events_month,
      tm.mapped_lessons_month,
      tm.strict_value_flag,
      tm.viewed_aula_flag,
      tm.viewed_plano_flag,
      tm.viewed_prova_flag,
      tm.used_ia_flag,
      tm.used_desktop_flag,
      tm.used_mobile_flag,
      tm.no_download_flag,
      tm.no_download_view_only_flag,
      tm.no_download_view_plus_action_flag,
      tm.no_download_action_only_flag,
      tm.session_exposed_no_download_flag,
      tm.session_exposed_no_activity_no_download_flag,
      tm.session_exposed_activity_no_download_flag,
      tm.lifetime_active_months,
      tm.lifetime_active_minutes_total,
      tm.active_streak_current_months,
      tm.strict_streak_current_months,
      COALESCE(dt.estado, 'missing') AS estado_group,
      COALESCE(dt.currentsubject_group, 'missing') AS currentsubject_group,
      COALESCE(dt.utm_group, 'missing') AS utm_group,
      COALESCE(dt.population_status, 'missing') AS population_status
    FROM fct_teacher_month tm
    INNER JOIN dim_teacher dt
      ON tm.teacher_unique_id = dt.teacher_unique_id
    WHERE COALESCE(tm.active_user_flag, 0) = 1
      AND COALESCE(tm.next_month_observed_flag, 0) = 1
    ORDER BY tm.month, tm.teacher_unique_id
    """
    df = conn.execute(query).fetchdf()
    if df.empty:
        return df
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    numeric_cols = [
        "active_user_flag",
        "next_month_observed_flag",
        "returned_active_m1",
        "returned_any_download_m1",
        "session_count_month",
        "total_session_minutes_month",
        "avg_session_minutes_month",
        "interaction_rows_month",
        "activity_events_month",
        "active_days_month",
        "aula_events_month",
        "plano_events_month",
        "prova_events_month",
        "ia_events_month",
        "strict_download_count_month",
        "content_views_month",
        "other_activity_non_download_events_month",
        "mapped_lessons_month",
        "strict_value_flag",
        "viewed_aula_flag",
        "viewed_plano_flag",
        "viewed_prova_flag",
        "used_ia_flag",
        "used_desktop_flag",
        "used_mobile_flag",
        "no_download_flag",
        "no_download_view_only_flag",
        "no_download_view_plus_action_flag",
        "no_download_action_only_flag",
        "session_exposed_no_download_flag",
        "session_exposed_no_activity_no_download_flag",
        "session_exposed_activity_no_download_flag",
        "lifetime_active_months",
        "lifetime_active_minutes_total",
        "active_streak_current_months",
        "strict_streak_current_months",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["estado_group", "currentsubject_group", "utm_group", "population_status"]:
        df[col] = normalize_text(df[col])
    return df


def add_heavy_month_flag(base: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return base.copy()
    heavy = base.copy()
    feature_cols = [
        "activity_events_month",
        "active_days_month",
        "total_session_minutes_month",
        "strict_download_count_month",
    ]
    transformed: Dict[str, pd.Series] = {}
    for col in feature_cols:
        transformed[col] = np.log1p(pd.to_numeric(heavy[col], errors="coerce").fillna(0))
    score = 0.0
    for col in feature_cols:
        series = transformed[col]
        std = float(series.std(ddof=0))
        denom = std if std > 0 else 1.0
        score = score + (series - float(series.mean())) / denom
    heavy["heavy_intensity_score"] = pd.to_numeric(score, errors="coerce")
    threshold = float(heavy["heavy_intensity_score"].quantile(0.90)) if not heavy.empty else float("nan")
    heavy["heavy_month_flag"] = (heavy["heavy_intensity_score"] >= threshold).astype(int)
    heavy["heavy_month_threshold"] = threshold
    return heavy


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return num / den.replace(0, np.nan)


def engineer_prediction_features(base: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_mart = add_heavy_month_flag(base)
    if feature_mart.empty:
        return feature_mart, pd.DataFrame()

    feature_mart["events_per_session"] = safe_divide(feature_mart["activity_events_month"], feature_mart["session_count_month"])
    feature_mart["views_per_session"] = safe_divide(feature_mart["content_views_month"], feature_mart["session_count_month"])
    feature_mart["downloads_per_session"] = safe_divide(feature_mart["strict_download_count_month"], feature_mart["session_count_month"])
    feature_mart["minutes_per_session"] = safe_divide(feature_mart["total_session_minutes_month"], feature_mart["session_count_month"])
    feature_mart["events_per_active_day"] = safe_divide(feature_mart["activity_events_month"], feature_mart["active_days_month"])
    feature_mart["views_per_active_day"] = safe_divide(feature_mart["content_views_month"], feature_mart["active_days_month"])
    feature_mart["downloads_per_active_day"] = safe_divide(feature_mart["strict_download_count_month"], feature_mart["active_days_month"])
    feature_mart["mobile_only_flag"] = (
        (feature_mart["used_mobile_flag"].fillna(0) == 1) & (feature_mart["used_desktop_flag"].fillna(0) == 0)
    ).astype(int)
    feature_mart["desktop_only_flag"] = (
        (feature_mart["used_desktop_flag"].fillna(0) == 1) & (feature_mart["used_mobile_flag"].fillna(0) == 0)
    ).astype(int)
    feature_mart["mixed_device_flag"] = (
        (feature_mart["used_mobile_flag"].fillna(0) == 1) & (feature_mart["used_desktop_flag"].fillna(0) == 1)
    ).astype(int)
    feature_mart["target_churn_m1"] = (feature_mart["returned_active_m1"].fillna(0) == 0).astype(int)
    feature_mart["target_return_active_m1"] = (feature_mart["returned_active_m1"].fillna(0) == 1).astype(int)

    feature_catalog = pd.DataFrame(
        [
            ("session_count_month", "behavior", "teacher_month", "Quantidade de sessoes no mes."),
            ("total_session_minutes_month", "behavior", "teacher_month", "Tempo total de sessao no mes."),
            ("activity_events_month", "behavior", "teacher_month", "Eventos de atividade no mes."),
            ("active_days_month", "behavior", "teacher_month", "Dias ativos no mes."),
            ("content_views_month", "behavior", "teacher_month", "Visualizacoes de conteudo no mes."),
            ("strict_download_count_month", "behavior", "teacher_month", "Downloads strict no mes."),
            ("other_activity_non_download_events_month", "behavior", "teacher_month", "Outras acoes sem download."),
            ("mapped_lessons_month", "behavior", "teacher_month", "Aulas unicas mapeadas no mes."),
            ("aula_events_month", "behavior", "teacher_month", "Eventos de aula."),
            ("plano_events_month", "behavior", "teacher_month", "Eventos de plano."),
            ("prova_events_month", "behavior", "teacher_month", "Eventos de prova."),
            ("ia_events_month", "behavior", "teacher_month", "Eventos de IA."),
            ("lifetime_active_months", "history", "teacher_month", "Meses ativos acumulados."),
            ("lifetime_active_minutes_total", "history", "teacher_month", "Tempo acumulado historico."),
            ("active_streak_current_months", "history", "teacher_month", "Sequencia ativa corrente."),
            ("strict_streak_current_months", "history", "teacher_month", "Sequencia strict corrente."),
            ("events_per_session", "engineered_ratio", "feature_mart", "Eventos por sessao."),
            ("views_per_session", "engineered_ratio", "feature_mart", "Views por sessao."),
            ("downloads_per_session", "engineered_ratio", "feature_mart", "Downloads por sessao."),
            ("minutes_per_session", "engineered_ratio", "feature_mart", "Minutos por sessao."),
            ("events_per_active_day", "engineered_ratio", "feature_mart", "Eventos por dia ativo."),
            ("views_per_active_day", "engineered_ratio", "feature_mart", "Views por dia ativo."),
            ("downloads_per_active_day", "engineered_ratio", "feature_mart", "Downloads por dia ativo."),
            ("heavy_month_flag", "engineered_state", "feature_mart", "Mes heavy pelo score p90."),
            ("mobile_only_flag", "engineered_state", "feature_mart", "Uso so mobile."),
            ("desktop_only_flag", "engineered_state", "feature_mart", "Uso so desktop."),
            ("mixed_device_flag", "engineered_state", "feature_mart", "Uso mobile e desktop."),
            (PROFILE_CONTROL_VAR, "profile_control", "dim_teacher", "Variavel de perfil usada para controle nos modelos."),
        ],
        columns=["feature_name", "feature_group", "source_layer", "definition"],
    )
    return feature_mart, feature_catalog


def window_subset(df: pd.DataFrame, window_label: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.copy()
    months = sorted(work["month"].dropna().unique().tolist())
    if window_label == "all_history":
        return work
    if window_label == "recent_6m":
        keep = set(months[-RECENT_WINDOW_MONTHS:])
        return work[work["month"].isin(keep)].copy()
    raise ValueError(f"window_label desconhecido: {window_label}")


def temporal_train_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.Timestamp], List[pd.Timestamp]]:
    months = sorted(df["month"].dropna().unique().tolist())
    if len(months) < 3:
        return pd.DataFrame(), pd.DataFrame(), [], []
    split_idx = max(1, min(len(months) - 1, int(np.floor(len(months) * 0.67))))
    train_months = months[:split_idx]
    test_months = months[split_idx:]
    train = df[df["month"].isin(train_months)].copy()
    test = df[df["month"].isin(test_months)].copy()
    return train, test, train_months, test_months


def build_preprocessor(numeric_features: Sequence[str], categorical_features: Sequence[str]) -> ColumnTransformer:
    transformers: List[Tuple[str, Pipeline, Sequence[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                list(numeric_features),
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(categorical_features),
            )
        )
    return ColumnTransformer(transformers=transformers)


def top_feature_rows(
    pipeline: Pipeline,
    model_name: str,
    target_label: str,
    window_label: str,
    feature_spec: str,
    max_rows: int = 25,
) -> List[Dict[str, Any]]:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()
    rows: List[Dict[str, Any]] = []
    if model_name == "profile_only_logistic" or model_name == "behavior_plus_profile_logistic":
        values = model.coef_[0]
        sorted_pairs = sorted(zip(feature_names, values), key=lambda item: abs(item[1]), reverse=True)[:max_rows]
        for feature_name, coef in sorted_pairs:
            rows.append(
                {
                    "target": target_label,
                    "window_label": window_label,
                    "feature_spec": feature_spec,
                    "model_name": model_name,
                    "feature_name": feature_name,
                    "importance_value": float(coef),
                    "direction": "positivo" if coef > 0 else "negativo",
                    "importance_type": "logistic_coefficient",
                }
            )
    else:
        values = model.feature_importances_
        sorted_pairs = sorted(zip(feature_names, values), key=lambda item: item[1], reverse=True)[:max_rows]
        for feature_name, importance in sorted_pairs:
            rows.append(
                {
                    "target": target_label,
                    "window_label": window_label,
                    "feature_spec": feature_spec,
                    "model_name": model_name,
                    "feature_name": feature_name,
                    "importance_value": float(importance),
                    "direction": "na",
                    "importance_type": "random_forest_importance",
                }
            )
    return rows


def profile_slice_performance(
    test: pd.DataFrame,
    score: np.ndarray,
    target_col: str,
    target_label: str,
    window_label: str,
    model_name: str,
    feature_spec: str,
) -> pd.DataFrame:
    work = test[[PROFILE_CONTROL_VAR, target_col]].copy()
    work["score"] = score
    top_groups = (
        normalize_text(work[PROFILE_CONTROL_VAR])
        .value_counts(dropna=False)
        .head(6)
        .index.astype(str)
        .tolist()
    )
    work[PROFILE_CONTROL_VAR] = normalize_text(work[PROFILE_CONTROL_VAR]).where(
        normalize_text(work[PROFILE_CONTROL_VAR]).isin(top_groups),
        "other",
    )
    rows: List[Dict[str, Any]] = []
    for profile_value, group in work.groupby(PROFILE_CONTROL_VAR, dropna=False):
        if len(group) < MIN_PROFILE_SLICE_ROWS or group[target_col].nunique() < 2:
            continue
        rows.append(
            {
                "target": target_label,
                "window_label": window_label,
                "model_name": model_name,
                "feature_spec": feature_spec,
                "profile_control_variable": PROFILE_CONTROL_VAR,
                "profile_control_value": str(profile_value),
                "test_rows": int(len(group)),
                "positive_rate_test": float(group[target_col].mean()),
                "roc_auc": safe_auc(group[target_col], group["score"]),
                "average_precision": safe_average_precision(group[target_col], group["score"]),
            }
        )
    return pd.DataFrame(rows)


def fit_models_for_target(
    feature_mart: pd.DataFrame,
    target_col: str,
    target_label: str,
    window_label: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = window_subset(feature_mart, window_label)
    train, test, train_months, test_months = temporal_train_test_split(base)
    if train.empty or test.empty or train[target_col].nunique() < 2 or test[target_col].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    behavior_numeric = [
        "session_count_month",
        "total_session_minutes_month",
        "avg_session_minutes_month",
        "interaction_rows_month",
        "activity_events_month",
        "active_days_month",
        "aula_events_month",
        "plano_events_month",
        "prova_events_month",
        "ia_events_month",
        "strict_download_count_month",
        "content_views_month",
        "other_activity_non_download_events_month",
        "mapped_lessons_month",
        "strict_value_flag",
        "viewed_aula_flag",
        "viewed_plano_flag",
        "viewed_prova_flag",
        "used_ia_flag",
        "used_desktop_flag",
        "used_mobile_flag",
        "no_download_flag",
        "no_download_view_only_flag",
        "no_download_view_plus_action_flag",
        "no_download_action_only_flag",
        "session_exposed_no_download_flag",
        "session_exposed_no_activity_no_download_flag",
        "session_exposed_activity_no_download_flag",
        "lifetime_active_months",
        "lifetime_active_minutes_total",
        "active_streak_current_months",
        "strict_streak_current_months",
        "events_per_session",
        "views_per_session",
        "downloads_per_session",
        "minutes_per_session",
        "events_per_active_day",
        "views_per_active_day",
        "downloads_per_active_day",
        "heavy_month_flag",
        "mobile_only_flag",
        "desktop_only_flag",
        "mixed_device_flag",
    ]
    profile_categorical = [PROFILE_CONTROL_VAR]

    model_specs = [
        {
            "model_name": "profile_only_logistic",
            "feature_spec": "profile_only",
            "numeric_features": [],
            "categorical_features": profile_categorical,
            "model": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        },
        {
            "model_name": "behavior_plus_profile_logistic",
            "feature_spec": "behavior_plus_profile",
            "numeric_features": behavior_numeric,
            "categorical_features": profile_categorical,
            "model": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        },
        {
            "model_name": "behavior_plus_profile_random_forest",
            "feature_spec": "behavior_plus_profile",
            "numeric_features": behavior_numeric,
            "categorical_features": profile_categorical,
            "model": RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=40,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        },
    ]

    perf_rows: List[Dict[str, Any]] = []
    feat_rows: List[Dict[str, Any]] = []
    slice_frames: List[pd.DataFrame] = []

    for spec in model_specs:
        preprocess = build_preprocessor(spec["numeric_features"], spec["categorical_features"])
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", spec["model"]),
            ]
        )
        train_cols = spec["numeric_features"] + spec["categorical_features"]
        X_train = train[train_cols].copy()
        y_train = train[target_col].astype(int)
        X_test = test[train_cols].copy()
        y_test = test[target_col].astype(int)

        pipeline.fit(X_train, y_train)
        score = pipeline.predict_proba(X_test)[:, 1]
        perf_rows.append(
            {
                "target": target_label,
                "window_label": window_label,
                "model_name": spec["model_name"],
                "feature_spec": spec["feature_spec"],
                "profile_control_variable": PROFILE_CONTROL_VAR,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "positive_rate_test": float(y_test.mean()),
                "roc_auc": safe_auc(y_test, score),
                "average_precision": safe_average_precision(y_test, score),
                "top_decile_lift": top_decile_lift(y_test, score),
                "train_month_start": str(min(train_months)),
                "train_month_end": str(max(train_months)),
                "test_month_start": str(min(test_months)),
                "test_month_end": str(max(test_months)),
            }
        )
        feat_rows.extend(
            top_feature_rows(
                pipeline,
                spec["model_name"],
                target_label,
                window_label,
                spec["feature_spec"],
            )
        )
        slice_df = profile_slice_performance(
            test,
            score,
            target_col,
            target_label,
            window_label,
            spec["model_name"],
            spec["feature_spec"],
        )
        if not slice_df.empty:
            slice_frames.append(slice_df)

    perf_df = pd.DataFrame(perf_rows)
    feat_df = pd.DataFrame(feat_rows)
    slice_perf = pd.concat(slice_frames, ignore_index=True) if slice_frames else pd.DataFrame()
    return perf_df, feat_df, slice_perf


def build_target_rates(feature_mart: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for window_label in ["all_history", "recent_6m"]:
        subset = window_subset(feature_mart, window_label)
        if subset.empty:
            continue
        rows.append(
            {
                "window_label": window_label,
                "rows": int(len(subset)),
                "teachers": int(subset["teacher_unique_id"].nunique()),
                "month_start": str(subset["month"].min()),
                "month_end": str(subset["month"].max()),
                "target_churn_m1_rate": float(subset["target_churn_m1"].mean()),
                "target_return_active_m1_rate": float(subset["target_return_active_m1"].mean()),
            }
        )
    return pd.DataFrame(rows)


def psi_numeric(baseline: pd.Series, recent: pd.Series, bins: int = 10) -> float:
    base = pd.to_numeric(baseline, errors="coerce").dropna()
    rec = pd.to_numeric(recent, errors="coerce").dropna()
    if base.empty or rec.empty:
        return np.nan
    quantiles = np.unique(np.nanquantile(base, np.linspace(0, 1, bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    edges = [-np.inf] + list(quantiles[1:-1]) + [np.inf]
    base_counts = pd.cut(base, bins=edges, include_lowest=True).value_counts(sort=False)
    rec_counts = pd.cut(rec, bins=edges, include_lowest=True).value_counts(sort=False)
    base_share = np.clip(base_counts / max(1, len(base)), 1e-6, None)
    rec_share = np.clip(rec_counts / max(1, len(rec)), 1e-6, None)
    return float(np.sum((rec_share - base_share) * np.log(rec_share / base_share)))


def numeric_drift_level(psi: float, smd: float) -> str:
    if pd.isna(psi) or pd.isna(smd):
        return "insufficient_data"
    if psi >= 0.25 or abs(smd) >= 0.50:
        return "high_drift"
    if psi >= 0.10 or abs(smd) >= 0.25:
        return "medium_drift"
    return "low_drift"


def build_numeric_drift(feature_mart: pd.DataFrame) -> pd.DataFrame:
    if feature_mart.empty:
        return pd.DataFrame()
    months = sorted(feature_mart["month"].dropna().unique().tolist())
    baseline_months = months[:RECENT_WINDOW_MONTHS]
    recent_months = months[-RECENT_WINDOW_MONTHS:]
    baseline = feature_mart[feature_mart["month"].isin(baseline_months)].copy()
    recent = feature_mart[feature_mart["month"].isin(recent_months)].copy()
    features = [
        "session_count_month",
        "total_session_minutes_month",
        "activity_events_month",
        "active_days_month",
        "content_views_month",
        "strict_download_count_month",
        "lifetime_active_months",
        "events_per_session",
        "views_per_session",
        "downloads_per_session",
        "heavy_month_flag",
        "used_mobile_flag",
        "target_churn_m1",
        "target_return_active_m1",
    ]
    rows: List[Dict[str, Any]] = []
    for feature in features:
        base = pd.to_numeric(baseline[feature], errors="coerce")
        rec = pd.to_numeric(recent[feature], errors="coerce")
        base_mean = float(base.mean()) if not base.dropna().empty else np.nan
        rec_mean = float(rec.mean()) if not rec.dropna().empty else np.nan
        base_median = float(base.median()) if not base.dropna().empty else np.nan
        rec_median = float(rec.median()) if not rec.dropna().empty else np.nan
        pooled_std = float(
            np.sqrt(
                (
                    np.nanvar(base, ddof=0) + np.nanvar(rec, ddof=0)
                ) / 2.0
            )
        )
        smd = (rec_mean - base_mean) / pooled_std if pooled_std and not np.isnan(pooled_std) else np.nan
        psi = psi_numeric(base, rec)
        rows.append(
            {
                "feature_name": feature,
                "baseline_month_start": str(min(baseline_months)) if baseline_months else None,
                "baseline_month_end": str(max(baseline_months)) if baseline_months else None,
                "recent_month_start": str(min(recent_months)) if recent_months else None,
                "recent_month_end": str(max(recent_months)) if recent_months else None,
                "baseline_rows": int(base.notna().sum()),
                "recent_rows": int(rec.notna().sum()),
                "baseline_mean": base_mean,
                "recent_mean": rec_mean,
                "baseline_median": base_median,
                "recent_median": rec_median,
                "mean_delta": rec_mean - base_mean if pd.notna(base_mean) and pd.notna(rec_mean) else np.nan,
                "mean_delta_pct": (rec_mean - base_mean) / base_mean if pd.notna(base_mean) and base_mean not in {0, np.nan} else np.nan,
                "standardized_mean_diff": smd,
                "psi": psi,
                "drift_level": numeric_drift_level(psi, smd),
            }
        )
    return pd.DataFrame(rows).sort_values(["drift_level", "psi", "standardized_mean_diff"], ascending=[True, False, False]).reset_index(drop=True)


def categorical_feature_drift_level(total_variation: float, max_share_diff_pp: float) -> str:
    if pd.isna(total_variation) or pd.isna(max_share_diff_pp):
        return "insufficient_data"
    if total_variation >= 0.15 or max_share_diff_pp >= 10.0:
        return "high_drift"
    if total_variation >= 0.08 or max_share_diff_pp >= 5.0:
        return "medium_drift"
    return "low_drift"


def build_categorical_drift(feature_mart: pd.DataFrame) -> pd.DataFrame:
    if feature_mart.empty:
        return pd.DataFrame()
    months = sorted(feature_mart["month"].dropna().unique().tolist())
    baseline_months = months[:RECENT_WINDOW_MONTHS]
    recent_months = months[-RECENT_WINDOW_MONTHS:]
    baseline = feature_mart[feature_mart["month"].isin(baseline_months)].copy()
    recent = feature_mart[feature_mart["month"].isin(recent_months)].copy()
    features = ["estado_group", "currentsubject_group", "utm_group", "population_status"]
    rows: List[Dict[str, Any]] = []
    for feature in features:
        base_series = normalize_text(baseline[feature])
        rec_series = normalize_text(recent[feature])
        categories = sorted(set(base_series.value_counts().head(8).index.astype(str)).union(set(rec_series.value_counts().head(8).index.astype(str))))
        if "other" not in categories:
            categories.append("other")
        base_top = base_series.where(base_series.isin(categories), "other")
        rec_top = rec_series.where(rec_series.isin(categories), "other")
        base_share = base_top.value_counts(normalize=True, dropna=False)
        rec_share = rec_top.value_counts(normalize=True, dropna=False)
        all_cats = sorted(set(base_share.index.astype(str)).union(set(rec_share.index.astype(str))))
        total_variation = 0.5 * float(
            sum(abs(float(base_share.get(cat, 0.0)) - float(rec_share.get(cat, 0.0))) for cat in all_cats)
        )
        max_diff = max(abs((float(rec_share.get(cat, 0.0)) - float(base_share.get(cat, 0.0))) * 100) for cat in all_cats) if all_cats else np.nan
        drift_level = categorical_feature_drift_level(total_variation, max_diff)
        for cat in all_cats:
            rows.append(
                {
                    "feature_name": feature,
                    "category_value": str(cat),
                    "baseline_month_start": str(min(baseline_months)) if baseline_months else None,
                    "baseline_month_end": str(max(baseline_months)) if baseline_months else None,
                    "recent_month_start": str(min(recent_months)) if recent_months else None,
                    "recent_month_end": str(max(recent_months)) if recent_months else None,
                    "baseline_share": float(base_share.get(cat, 0.0)),
                    "recent_share": float(rec_share.get(cat, 0.0)),
                    "share_diff_pp": (float(rec_share.get(cat, 0.0)) - float(base_share.get(cat, 0.0))) * 100,
                    "feature_total_variation": total_variation,
                    "feature_max_share_diff_pp": max_diff,
                    "drift_level": drift_level,
                }
            )
    return pd.DataFrame(rows).sort_values(["feature_name", "share_diff_pp"], ascending=[True, False]).reset_index(drop=True)


def build_summary_payload(
    target_rates: pd.DataFrame,
    performance: pd.DataFrame,
    top_features: pd.DataFrame,
    numeric_drift: pd.DataFrame,
    categorical_drift: pd.DataFrame,
) -> Dict[str, Any]:
    best_models = performance.sort_values(["target", "window_label", "roc_auc"], ascending=[True, True, False]).groupby(
        ["target", "window_label"], as_index=False
    ).head(1) if not performance.empty else pd.DataFrame()
    top_numeric = numeric_drift.sort_values(["drift_level", "psi"], ascending=[True, False]).head(6) if not numeric_drift.empty else pd.DataFrame()
    top_cat = (
        categorical_drift.sort_values("share_diff_pp", key=lambda s: s.abs(), ascending=False).head(8)
        if not categorical_drift.empty
        else pd.DataFrame()
    )
    top_feat = top_features[
        (top_features["window_label"] == "all_history")
        & (top_features["feature_spec"] == "behavior_plus_profile")
    ].copy()
    top_feat = top_feat.groupby(["target", "model_name"], as_index=False).head(8) if not top_feat.empty else pd.DataFrame()
    return {
        "generated_at_utc": utc_now_iso(),
        "profile_control_variable": PROFILE_CONTROL_VAR,
        "recent_window_months": RECENT_WINDOW_MONTHS,
        "target_rates": target_rates.to_dict(orient="records"),
        "best_models": best_models[
            ["target", "window_label", "model_name", "feature_spec", "roc_auc", "average_precision", "top_decile_lift"]
        ].to_dict(orient="records")
        if not best_models.empty
        else [],
        "top_features_all_history": top_feat[
            ["target", "model_name", "feature_name", "importance_value", "importance_type", "direction"]
        ].to_dict(orient="records")
        if not top_feat.empty
        else [],
        "top_numeric_drift": top_numeric[
            ["feature_name", "drift_level", "baseline_mean", "recent_mean", "standardized_mean_diff", "psi"]
        ].to_dict(orient="records")
        if not top_numeric.empty
        else [],
        "top_categorical_drift": top_cat[
            ["feature_name", "category_value", "share_diff_pp", "feature_total_variation", "drift_level"]
        ].to_dict(orient="records")
        if not top_cat.empty
        else [],
    }


def write_summary_markdown(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Prediction and drift v2",
        "",
        f"- Gerado em UTC: {summary['generated_at_utc']}",
        f"- Variavel de controle de perfil: {summary['profile_control_variable']}",
        f"- Janela recente: {summary['recent_window_months']} meses",
        "",
        "## Best Models",
    ]
    if not summary["best_models"]:
        lines.append("- none")
    else:
        for row in summary["best_models"]:
            lines.append(
                f"- `{row['target']}` | `{row['window_label']}` | `{row['model_name']}` | auc={row['roc_auc']:.4f} | ap={row['average_precision']:.4f} | lift={row['top_decile_lift']:.4f}"
            )
    lines.append("")
    lines.append("## Top Numeric Drift")
    if not summary["top_numeric_drift"]:
        lines.append("- none")
    else:
        for row in summary["top_numeric_drift"]:
            lines.append(
                f"- `{row['feature_name']}` | `{row['drift_level']}` | mean_old={row['baseline_mean']:.4f} | mean_recent={row['recent_mean']:.4f} | smd={row['standardized_mean_diff']:.4f} | psi={row['psi']:.4f}"
            )
    lines.append("")
    lines.append("## Top Categorical Drift")
    if not summary["top_categorical_drift"]:
        lines.append("- none")
    else:
        for row in summary["top_categorical_drift"]:
            lines.append(
                f"- `{row['feature_name']}::{row['category_value']}` | `{row['drift_level']}` | diff_pp={row['share_diff_pp']:.2f} | tv={row['feature_total_variation']:.4f}"
            )
    write_markdown(path, lines)


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg: V2Config = build_config(args.base_dir, args.data_dir, args.output_dir)
    conn = connect_duckdb(cfg)
    try:
        require_tables(conn, ["fct_teacher_month", "dim_teacher"])
        base = load_prediction_base(conn)
        feature_mart, feature_catalog = engineer_prediction_features(base)
        target_rates = build_target_rates(feature_mart)

        perf_frames: List[pd.DataFrame] = []
        feat_frames: List[pd.DataFrame] = []
        slice_frames: List[pd.DataFrame] = []
        for target_col, target_label in [
            ("target_churn_m1", "abandonar_m1"),
            ("target_return_active_m1", "retornar_ativo_m1"),
        ]:
            for window_label in ["all_history", "recent_6m"]:
                perf_df, feat_df, slice_df = fit_models_for_target(feature_mart, target_col, target_label, window_label)
                if not perf_df.empty:
                    perf_frames.append(perf_df)
                if not feat_df.empty:
                    feat_frames.append(feat_df)
                if not slice_df.empty:
                    slice_frames.append(slice_df)

        performance = pd.concat(perf_frames, ignore_index=True) if perf_frames else pd.DataFrame()
        top_features = pd.concat(feat_frames, ignore_index=True) if feat_frames else pd.DataFrame()
        slice_performance = pd.concat(slice_frames, ignore_index=True) if slice_frames else pd.DataFrame()
        numeric_drift = build_numeric_drift(feature_mart)
        categorical_drift = build_categorical_drift(feature_mart)

        outputs: Dict[str, pd.DataFrame] = {
            "mart_teacher_month_prediction_features_v2": feature_mart,
            "analytics_prediction_feature_catalog_v2": feature_catalog,
            "analytics_prediction_target_rates_v2": target_rates,
            "analytics_prediction_model_performance_v2": performance,
            "analytics_prediction_model_top_features_v2": top_features,
            "analytics_prediction_profile_slice_performance_v2": slice_performance,
            "analytics_prediction_drift_numeric_v2": numeric_drift,
            "analytics_prediction_drift_categorical_v2": categorical_drift,
        }
        for name, df in outputs.items():
            persist_output(conn, cfg, name, df)

        summary = build_summary_payload(target_rates, performance, top_features, numeric_drift, categorical_drift)
        write_json(cfg.output_dir / "json" / "prediction_drift_summary_v2.json", summary)
        write_summary_markdown(cfg.output_dir / "audit" / "prediction_drift_summary_v2.md", summary)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
