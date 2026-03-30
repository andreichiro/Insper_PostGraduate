#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analytics_v2_common import (
    DEFAULT_BASE_DIR,
    V2Config,
    build_config,
    connect_duckdb,
    ensure_output_dirs,
    fmt_num,
    month_diff,
    safe_auc,
    safe_average_precision,
    setup_logging,
    top_decile_lift,
    utc_now_iso,
    write_df_bundle,
    write_json,
    write_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Etapa 04 v2: analytics de usuários, retenção, clusters e modelos.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def add_heavy_month_flag(teacher_month: pd.DataFrame) -> pd.DataFrame:
    df = teacher_month.copy()
    active = df[df["active_user_flag"] == 1].copy()
    feature_cols = [
        "activity_events_month",
        "active_days_month",
        "total_session_minutes_month",
        "strict_download_count_month",
    ]
    for col in feature_cols:
        active[col] = np.log1p(pd.to_numeric(active[col], errors="coerce").fillna(0))
    score = sum((active[col] - active[col].mean()) / (active[col].std(ddof=0) or 1.0) for col in feature_cols)
    active["heavy_intensity_score"] = score
    threshold = float(active["heavy_intensity_score"].quantile(0.90)) if not active.empty else float("nan")
    active["heavy_month_flag"] = (active["heavy_intensity_score"] >= threshold).astype(int)
    df = df.merge(
        active[["teacher_unique_id", "month", "heavy_intensity_score", "heavy_month_flag"]],
        on=["teacher_unique_id", "month"],
        how="left",
    )
    df["heavy_intensity_score"] = pd.to_numeric(df["heavy_intensity_score"], errors="coerce")
    df["heavy_month_flag"] = pd.to_numeric(df["heavy_month_flag"], errors="coerce").fillna(0).astype(int)
    return df


def compute_missing_rate(series: pd.Series) -> float:
    if pd.api.types.is_numeric_dtype(series):
        return float(pd.to_numeric(series, errors="coerce").isna().mean())
    normalized = series.astype("string")
    missing_mask = (
        normalized.isna()
        | normalized.str.strip().eq("")
        | normalized.str.lower().isin(["missing", "<missing>", "nan", "none"])
    )
    return float(missing_mask.mean())


def compute_monthly_core_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for month, group in df.groupby("month", dropna=False):
        active_mask = pd.to_numeric(group["active_user_flag"], errors="coerce").fillna(0) == 1
        strict_mask = pd.to_numeric(group["strict_value_flag"], errors="coerce").fillna(0) == 1
        next_obs_mask = pd.to_numeric(group["next_month_observed_flag"], errors="coerce").fillna(0) == 1
        eligible_active_mask = active_mask & next_obs_mask
        eligible_strict_mask = strict_mask & next_obs_mask
        strict_users = int(pd.to_numeric(group["strict_user_flag"], errors="coerce").fillna(0).sum())
        strict_return_value_users = int(pd.to_numeric(group["strict_return_value_m1"], errors="coerce").fillna(0).sum())
        rows.append(
            {
                "month": month,
                "teacher_month_rows": int(group["teacher_unique_id"].count()),
                "active_users": int(active_mask.sum()),
                "strict_value_users": int(strict_mask.sum()),
                "eligible_active_users_next_month": int(eligible_active_mask.sum()),
                "eligible_strict_value_users_next_month": int(eligible_strict_mask.sum()),
                "strict_users": strict_users,
                "strict_return_value_users": strict_return_value_users,
                "avg_strict_downloads": float(pd.to_numeric(group["strict_download_count_month"], errors="coerce").fillna(0).mean()),
                "avg_sessions": float(pd.to_numeric(group["session_count_month"], errors="coerce").fillna(0).mean()),
                "avg_session_minutes": float(pd.to_numeric(group["total_session_minutes_month"], errors="coerce").fillna(0).mean()),
            }
        )
    out = pd.DataFrame(rows).sort_values("month").reset_index(drop=True)
    out["strict_value_rate"] = out["strict_value_users"] / out["active_users"].replace(0, np.nan)
    out["strict_user_rate"] = out["strict_users"] / out["eligible_strict_value_users_next_month"].replace(0, np.nan)
    out["strict_return_value_rate"] = (
        out["strict_return_value_users"] / out["eligible_strict_value_users_next_month"].replace(0, np.nan)
    )
    return out


def stratified_effect(
    df: pd.DataFrame,
    exposed_col: str,
    outcome_col: str,
    strata_cols: Sequence[str],
    label: str,
) -> Dict[str, Any]:
    base = df[list(strata_cols) + [exposed_col, outcome_col]].copy()
    base[exposed_col] = pd.to_numeric(base[exposed_col], errors="coerce")
    base[outcome_col] = pd.to_numeric(base[outcome_col], errors="coerce")
    base = base.dropna(subset=[exposed_col, outcome_col])
    grouped_rows: List[Dict[str, Any]] = []
    for strata_key, g in base.groupby(list(strata_cols), dropna=False, sort=False):
        if not isinstance(strata_key, tuple):
            strata_key = (strata_key,)
        row = {col: value for col, value in zip(strata_cols, strata_key)}
        row.update(
            {
                "n_total": len(g),
                "n_exposed": int((g[exposed_col] == 1).sum()),
                "n_unexposed": int((g[exposed_col] == 0).sum()),
                "p_exposed": float(g.loc[g[exposed_col] == 1, outcome_col].mean()) if (g[exposed_col] == 1).any() else np.nan,
                "p_unexposed": float(g.loc[g[exposed_col] == 0, outcome_col].mean()) if (g[exposed_col] == 0).any() else np.nan,
            }
        )
        grouped_rows.append(row)
    grouped = pd.DataFrame(grouped_rows)
    grouped = grouped[(grouped["n_exposed"] > 0) & (grouped["n_unexposed"] > 0)].copy()
    if grouped.empty:
        return {
            "hypothesis_id": label,
            "status": "not_testable",
            "effect": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "strata_used": 0,
            "n_obs": 0,
            "note": "Sem estratos com expostos e não expostos.",
        }
    grouped["diff"] = grouped["p_exposed"] - grouped["p_unexposed"]
    grouped["weight"] = grouped["n_total"] / grouped["n_total"].sum()
    grouped["var"] = (
        grouped["p_exposed"] * (1 - grouped["p_exposed"]) / grouped["n_exposed"]
        + grouped["p_unexposed"] * (1 - grouped["p_unexposed"]) / grouped["n_unexposed"]
    )
    effect = float((grouped["diff"] * grouped["weight"]).sum())
    se = float(np.sqrt(((grouped["weight"] ** 2) * grouped["var"]).sum()))
    ci_low = effect - 1.96 * se
    ci_high = effect + 1.96 * se
    status = "validated" if ci_low > 0 or ci_high < 0 else "inconclusive"
    return {
        "hypothesis_id": label,
        "status": status,
        "effect": effect,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "strata_used": int(len(grouped)),
        "n_obs": int(grouped["n_total"].sum()),
        "note": "Efeito estratificado por mês, tenure e baseline de uso.",
    }


def build_abandonment_gap_curve(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    active = df[df["active_user_flag"] == 1].copy()
    active = active.sort_values(["teacher_unique_id", "month"])
    active["next_active_month"] = active.groupby("teacher_unique_id")["month"].shift(-1)
    active["next_month_observed"] = active.groupby("teacher_unique_id")["next_month_observed_flag"].shift(0)
    active["gap_months"] = active.apply(
        lambda row: month_diff(pd.Timestamp(row["next_active_month"]), pd.Timestamp(row["month"]))
        if pd.notna(row["next_active_month"])
        else np.nan,
        axis=1,
    )
    eligible = active[active["next_month_observed_flag"] == 1].copy()
    eligible["gap_months"] = pd.to_numeric(eligible["gap_months"], errors="coerce")
    max_horizon = int(min(12, np.nanmax(eligible["gap_months"])) if not eligible["gap_months"].dropna().empty else 0)
    rows: List[Dict[str, Any]] = []
    returned_before = pd.Series(False, index=eligible.index)
    cumulative = 0.0
    suggested = None
    for horizon in range(1, max_horizon + 1):
        at_risk_mask = returned_before == 0
        at_risk = eligible[at_risk_mask]
        events = at_risk["gap_months"] == horizon
        n_at_risk = int(at_risk.shape[0])
        n_events = int(events.sum())
        hazard = float(n_events / n_at_risk) if n_at_risk else np.nan
        cumulative += hazard if not np.isnan(hazard) else 0.0
        rows.append(
            {
                "horizon_month": horizon,
                "n_at_risk": n_at_risk,
                "n_events_first_return": n_events,
                "hazard": hazard,
                "cumulative_return": cumulative,
            }
        )
        returned_before = returned_before | (eligible["gap_months"] == horizon)
        if suggested is None and n_at_risk >= 500 and not np.isnan(hazard) and hazard < 0.05:
            suggested = horizon
    curve = pd.DataFrame(rows)
    meta = {
        "suggested_abandonment_gap_months": suggested,
        "note": "Threshold sugerido = primeiro gap com hazard < 5% e base em risco >= 500.",
    }
    return curve, meta


def build_strict_cohort_curve(df: pd.DataFrame) -> pd.DataFrame:
    strict = df[df["strict_value_flag"] == 1].copy()
    if strict.empty:
        return pd.DataFrame()
    strict["cohort_month"] = strict.groupby("teacher_unique_id")["month"].transform("min")
    strict["horizon_m"] = strict.apply(lambda row: month_diff(pd.Timestamp(row["month"]), pd.Timestamp(row["cohort_month"])), axis=1)
    out = (
        strict.groupby(["cohort_month", "horizon_m"], dropna=False)
        .agg(
            cohort_size=("teacher_unique_id", "nunique"),
            active_rows=("teacher_unique_id", "count"),
            returned_active_rate=("returned_active_m1", "mean"),
            returned_value_rate=("returned_any_download_m1", "mean"),
        )
        .reset_index()
        .sort_values(["cohort_month", "horizon_m"])
    )
    return out


def cluster_teacher_months(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    active = df[df["active_user_flag"] == 1].copy()
    if active.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    feature_quality: List[Dict[str, Any]] = []
    admitted: List[str] = []
    for col in feature_cols:
        series = pd.to_numeric(active[col], errors="coerce")
        missing_rate = float(series.isna().mean())
        nonzero_share = float((series.fillna(0) > 0).mean())
        std = float(series.fillna(0).std(ddof=0))
        admitted_flag = missing_rate <= 0.40 and nonzero_share >= 0.01 and std > 0
        feature_quality.append(
            {
                "feature": col,
                "missing_rate": missing_rate,
                "nonzero_share": nonzero_share,
                "std": std,
                "admitted": int(admitted_flag),
                "semantic_rule": "completude<=40%, suporte>=1%, variância>0",
            }
        )
        if admitted_flag:
            admitted.append(col)
    if len(admitted) < 3:
        diagnostics = pd.DataFrame(
            [
                {
                    "k": np.nan,
                    "silhouette": np.nan,
                    "stability_ari_mean": np.nan,
                    "blended_score": np.nan,
                    "sample_rows": int(len(active)),
                    "admitted_feature_count": int(len(admitted)),
                    "selected_k": 0,
                    "status": "insufficient_features",
                }
            ]
        )
        return diagnostics, pd.DataFrame(), pd.DataFrame(feature_quality)

    sample = active.sample(min(len(active), 50_000), random_state=random_seed).copy()
    sample_matrix = np.log1p(sample[admitted].fillna(0))
    scaler = StandardScaler()
    sample_scaled = scaler.fit_transform(sample_matrix)
    diagnostics: List[Dict[str, Any]] = []
    best_k = 3
    best_score = -1e9
    for k in range(3, min(6, len(sample)) + 1):
        models = [KMeans(n_clusters=k, random_state=random_seed + seed, n_init=10) for seed in range(3)]
        labels = [model.fit_predict(sample_scaled) for model in models]
        silhouette = float(silhouette_score(sample_scaled, labels[0])) if len(np.unique(labels[0])) > 1 else np.nan
        aris = [adjusted_rand_score(a, b) for a, b in itertools.combinations(labels, 2)]
        stability = float(np.mean(aris)) if aris else np.nan
        blended = (silhouette if not np.isnan(silhouette) else -1.0) + (stability if not np.isnan(stability) else 0.0)
        diagnostics.append(
            {
                "k": k,
                "silhouette": silhouette,
                "stability_ari_mean": stability,
                "blended_score": blended,
                "sample_rows": len(sample),
            }
        )
        if blended > best_score:
            best_score = blended
            best_k = k

    full_matrix = np.log1p(active[admitted].fillna(0))
    full_scaled = StandardScaler().fit_transform(full_matrix)
    final_model = KMeans(n_clusters=best_k, random_state=random_seed, n_init=20)
    active["behavior_cluster_id"] = final_model.fit_predict(full_scaled)
    profile = (
        active.groupby("behavior_cluster_id", dropna=False)
        .agg(
            teacher_month_rows=("teacher_unique_id", "size"),
            teachers=("teacher_unique_id", "nunique"),
            strict_value_rate=("strict_value_flag", "mean"),
            returned_active_rate=("returned_active_m1", "mean"),
            returned_download_rate=("returned_any_download_m1", "mean"),
            heavy_month_rate=("heavy_month_flag", "mean"),
            avg_strict_downloads=("strict_download_count_month", "mean"),
            avg_session_minutes=("total_session_minutes_month", "mean"),
            avg_active_days=("active_days_month", "mean"),
            avg_content_views=("content_views_month", "mean"),
            avg_other_actions=("other_activity_non_download_events_month", "mean"),
        )
        .reset_index()
        .sort_values("teacher_month_rows", ascending=False)
    )
    diagnostics_df = pd.DataFrame(diagnostics)
    diagnostics_df["admitted_feature_count"] = int(len(admitted))
    diagnostics_df["admitted_features"] = ", ".join(admitted)
    diagnostics_df["selected_k"] = (diagnostics_df["k"] == best_k).astype(int)
    diagnostics_df["status"] = "ok"
    return diagnostics_df, profile, pd.DataFrame(feature_quality)


def build_heavy_usage_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    active = df[(df["active_user_flag"] == 1) & (df["next_month_observed_flag"] == 1)].copy()
    out = (
        active.groupby("heavy_month_flag", dropna=False)
        .agg(
            teacher_month_rows=("teacher_unique_id", "size"),
            teachers=("teacher_unique_id", "nunique"),
            avg_intensity_score=("heavy_intensity_score", "mean"),
            return_active_rate=("returned_active_m1", "mean"),
            return_download_rate=("returned_any_download_m1", "mean"),
            avg_strict_downloads=("strict_download_count_month", "mean"),
            avg_session_minutes=("total_session_minutes_month", "mean"),
        )
        .reset_index()
        .sort_values("heavy_month_flag", ascending=False)
    )
    out["segment"] = out["heavy_month_flag"].map({1: "heavy_month", 0: "base_active_month"})
    return out


def build_model_datasets(df: pd.DataFrame, dim_teacher: pd.DataFrame) -> pd.DataFrame:
    base = df.merge(
        dim_teacher[
            [
                "teacher_unique_id",
                "estado",
                "currentsubject_group",
                "utm_group",
                "population_status",
                "is_estado_missing",
                "is_utm_missing",
            ]
        ],
        on="teacher_unique_id",
        how="left",
    )
    base["estado_group"] = base["estado"].fillna("missing").replace("", "missing")
    return base


def fit_predictive_models(df: pd.DataFrame, target_col: str, target_label: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = df.copy()
    numeric_features = [
        "session_count_month",
        "active_days_month",
        "total_session_minutes_month",
        "strict_download_count_month",
        "content_views_month",
        "other_activity_non_download_events_month",
        "mapped_lessons_month",
        "lifetime_active_months",
        "lifetime_active_minutes_total",
        "active_streak_current_months",
        "strict_streak_current_months",
        "used_desktop_flag",
        "used_mobile_flag",
    ]
    categorical_features = ["estado_group", "currentsubject_group", "utm_group"]
    cols = ["month", target_col] + numeric_features + categorical_features
    base = base[cols].copy()
    base[target_col] = pd.to_numeric(base[target_col], errors="coerce")
    base = base.dropna(subset=[target_col])
    if base.empty or base[target_col].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame()

    base["month"] = pd.to_datetime(base["month"], errors="coerce")
    months = sorted(base["month"].dropna().unique().tolist())
    split_idx = max(1, int(len(months) * 0.70))
    train_months = set(months[:split_idx])
    test_months = set(months[split_idx:])
    train = base[base["month"].isin(train_months)].copy()
    test = base[base["month"].isin(test_months)].copy()
    if train.empty or test.empty:
        return pd.DataFrame(), pd.DataFrame()

    X_train = train[numeric_features + categorical_features]
    y_train = train[target_col].astype(int)
    X_test = test[numeric_features + categorical_features]
    y_test = test[target_col].astype(int)

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    logistic = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )
    forest = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=50,
                random_state=42,
                n_jobs=-1,
            )),
        ]
    )

    results: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []
    for model_name, pipeline in [("logistic_regression", logistic), ("random_forest", forest)]:
        pipeline.fit(X_train, y_train)
        score = pipeline.predict_proba(X_test)[:, 1]
        results.append(
            {
                "target": target_label,
                "model_name": model_name,
                "train_rows": len(train),
                "test_rows": len(test),
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
        if model_name == "logistic_regression":
            model = pipeline.named_steps["model"]
            feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
            for feature_name, coef in sorted(zip(feature_names, model.coef_[0]), key=lambda item: abs(item[1]), reverse=True)[:20]:
                feature_rows.append(
                    {
                        "target": target_label,
                        "model_name": model_name,
                        "feature_name": feature_name,
                        "coefficient": float(coef),
                        "direction": "positivo" if coef > 0 else "negativo",
                    }
                )
    return pd.DataFrame(results), pd.DataFrame(feature_rows)


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg: V2Config = build_config(args.base_dir, args.data_dir, args.output_dir)
    ensure_output_dirs(cfg.output_dir)
    conn = connect_duckdb(cfg, read_only=True)
    try:
        teacher_month = conn.execute("SELECT * FROM fct_teacher_month").fetchdf()
        dim_teacher = conn.execute("SELECT * FROM dim_teacher").fetchdf()
        if teacher_month.empty:
            raise RuntimeError("fct_teacher_month está vazio. Execute etapa_02_star_schema_v2.py antes.")

        teacher_month["month"] = pd.to_datetime(teacher_month["month"], errors="coerce")
        teacher_month = add_heavy_month_flag(teacher_month)
        heavy_next = teacher_month[
            ["teacher_unique_id", "month", "heavy_month_flag"]
        ].rename(columns={"month": "next_month", "heavy_month_flag": "next_month_heavy_month_flag"})
        teacher_month = teacher_month.merge(heavy_next, on=["teacher_unique_id", "next_month"], how="left")
        teacher_month["next_month_heavy_month_flag"] = pd.to_numeric(
            teacher_month["next_month_heavy_month_flag"], errors="coerce"
        )

        outputs: Dict[str, pd.DataFrame] = {}
        outputs["analytics_monthly_core_metrics"] = compute_monthly_core_metrics(teacher_month)

        active_observed = teacher_month[(teacher_month["active_user_flag"] == 1) & (teacher_month["next_month_observed_flag"] == 1)].copy()
        active_observed["tenure_band"] = pd.qcut(
            active_observed["lifetime_active_months"].rank(method="first"),
            q=min(4, max(2, active_observed["lifetime_active_months"].nunique())),
            duplicates="drop",
            labels=False,
        )
        active_observed["baseline_band"] = pd.qcut(
            active_observed["activity_events_month"].rank(method="first"),
            q=min(4, max(2, active_observed["activity_events_month"].nunique())),
            duplicates="drop",
            labels=False,
        )

        outputs["analytics_download_return_comparison"] = (
            active_observed.groupby("strict_value_flag", dropna=False)
            .agg(
                teacher_month_rows=("teacher_unique_id", "size"),
                teachers=("teacher_unique_id", "nunique"),
                return_active_rate=("returned_active_m1", "mean"),
                return_download_rate=("returned_any_download_m1", "mean"),
                avg_downloads=("strict_download_count_month", "mean"),
                avg_session_minutes=("total_session_minutes_month", "mean"),
            )
            .reset_index()
            .sort_values("strict_value_flag", ascending=False)
        )
        outputs["analytics_download_return_comparison"]["segment"] = outputs["analytics_download_return_comparison"]["strict_value_flag"].map(
            {1: "fez_strict_value", 0: "nao_fez_strict_value"}
        )

        outputs["analytics_no_download_path_outcomes"] = (
            active_observed[active_observed["no_download_flag"] == 1]
            .assign(
                path_category=lambda df: np.select(
                    [
                        df["no_download_view_only_flag"] == 1,
                        df["no_download_view_plus_action_flag"] == 1,
                        df["no_download_action_only_flag"] == 1,
                    ],
                    [
                        "visualizou_sem_baixar",
                        "visualizou_e_fez_outras_acoes_sem_baixar",
                        "fez_acoes_sem_visualizar_sem_baixar",
                    ],
                    default="outro_sem_download",
                )
            )
            .groupby("path_category", dropna=False)
            .agg(
                teacher_month_rows=("teacher_unique_id", "size"),
                teachers=("teacher_unique_id", "nunique"),
                return_active_rate=("returned_active_m1", "mean"),
                return_download_rate=("returned_any_download_m1", "mean"),
                avg_session_minutes=("total_session_minutes_month", "mean"),
            )
            .reset_index()
            .sort_values("teacher_month_rows", ascending=False)
        )

        outputs["analytics_session_exposure_outcomes"] = (
            teacher_month[
                (teacher_month["session_count_month"] > 0) & (teacher_month["next_month_observed_flag"] == 1)
            ]
            .assign(
                exposure_category=lambda df: np.select(
                    [
                        df["session_exposed_no_activity_no_download_flag"] == 1,
                        df["session_exposed_activity_no_download_flag"] == 1,
                        df["strict_value_flag"] == 1,
                    ],
                    [
                        "acessou_sem_atividade_sem_download",
                        "acessou_com_atividade_sem_download",
                        "acessou_com_strict_value",
                    ],
                    default="acessou_outro",
                )
            )
            .groupby("exposure_category", dropna=False)
            .agg(
                teacher_month_rows=("teacher_unique_id", "size"),
                teachers=("teacher_unique_id", "nunique"),
                return_active_rate=("returned_active_m1", "mean"),
                return_download_rate=("returned_any_download_m1", "mean"),
                avg_session_minutes=("total_session_minutes_month", "mean"),
            )
            .reset_index()
            .sort_values("teacher_month_rows", ascending=False)
        )

        outputs["analytics_heavy_usage_outcomes"] = build_heavy_usage_outcomes(teacher_month)

        abandonment_curve, abandonment_meta = build_abandonment_gap_curve(teacher_month)
        outputs["analytics_abandonment_gap_curve"] = abandonment_curve
        outputs["analytics_strict_cohort_curve"] = build_strict_cohort_curve(teacher_month)

        cluster_feature_candidates = [
            "session_count_month",
            "active_days_month",
            "total_session_minutes_month",
            "strict_download_count_month",
            "content_views_month",
            "other_activity_non_download_events_month",
            "aula_events_month",
            "plano_events_month",
            "prova_events_month",
            "ia_events_month",
        ]
        cluster_diagnostics, cluster_profiles, cluster_feature_quality = cluster_teacher_months(
            teacher_month,
            cluster_feature_candidates,
        )
        outputs["analytics_cluster_diagnostics"] = cluster_diagnostics
        outputs["analytics_cluster_profiles"] = cluster_profiles
        outputs["analytics_cluster_feature_quality"] = cluster_feature_quality

        model_base = build_model_datasets(teacher_month, dim_teacher)
        churn_base = model_base[(model_base["active_user_flag"] == 1) & (model_base["next_month_observed_flag"] == 1)].copy()
        churn_base["target_stop_using_m1"] = (pd.to_numeric(churn_base["returned_active_m1"], errors="coerce").fillna(0) == 0).astype(int)
        heavy_base = model_base[(model_base["active_user_flag"] == 1) & (model_base["next_month_observed_flag"] == 1)].copy()
        heavy_base["target_heavy_next_m1"] = (pd.to_numeric(heavy_base["next_month_heavy_month_flag"], errors="coerce").fillna(0) == 1).astype(int)

        model_perf_frames: List[pd.DataFrame] = []
        model_feat_frames: List[pd.DataFrame] = []
        for frame, target_col, target_label in [
            (churn_base, "target_stop_using_m1", "parar_de_usar_m1"),
            (heavy_base, "target_heavy_next_m1", "heavy_user_m1"),
        ]:
            perf_df, feat_df = fit_predictive_models(frame, target_col, target_label)
            if not perf_df.empty:
                model_perf_frames.append(perf_df)
            if not feat_df.empty:
                model_feat_frames.append(feat_df)
        outputs["analytics_model_performance"] = pd.concat(model_perf_frames, ignore_index=True) if model_perf_frames else pd.DataFrame()
        outputs["analytics_model_top_features"] = pd.concat(model_feat_frames, ignore_index=True) if model_feat_frames else pd.DataFrame()

        hypotheses_rows: List[Dict[str, Any]] = []
        strata_cols = ["month", "tenure_band", "baseline_band"]
        hypotheses_rows.append(stratified_effect(active_observed, "strict_value_flag", "returned_active_m1", strata_cols, "download_vs_no_download_return_active"))
        hypotheses_rows.append(
            stratified_effect(
                active_observed.assign(high_download_intensity=(active_observed["strict_download_count_month"] >= active_observed["strict_download_count_month"].quantile(0.75)).astype(int)),
                "high_download_intensity",
                "returned_any_download_m1",
                strata_cols,
                "alta_intensidade_download_retorna_com_download",
            )
        )
        no_download_subset = active_observed[
            (active_observed["no_download_view_only_flag"] == 1) | (active_observed["no_download_action_only_flag"] == 1)
        ].copy()
        if not no_download_subset.empty:
            no_download_subset["view_only_exposed"] = (no_download_subset["no_download_view_only_flag"] == 1).astype(int)
            hypotheses_rows.append(
                stratified_effect(
                    no_download_subset,
                    "view_only_exposed",
                    "returned_active_m1",
                    strata_cols,
                    "visualizacao_sem_download_vs_acao_sem_visualizacao",
                )
            )
        session_subset = teacher_month[
            (teacher_month["next_month_observed_flag"] == 1)
            & (
                (teacher_month["session_exposed_no_activity_no_download_flag"] == 1)
                | (teacher_month["session_exposed_activity_no_download_flag"] == 1)
            )
        ].copy()
        if not session_subset.empty:
            session_subset["activity_without_download_exposed"] = (
                session_subset["session_exposed_activity_no_download_flag"] == 1
            ).astype(int)
            session_subset["tenure_band"] = pd.qcut(
                session_subset["lifetime_active_months"].rank(method="first"),
                q=min(4, max(2, session_subset["lifetime_active_months"].nunique())),
                duplicates="drop",
                labels=False,
            )
            session_subset["baseline_band"] = pd.qcut(
                session_subset["session_count_month"].rank(method="first"),
                q=min(4, max(2, session_subset["session_count_month"].nunique())),
                duplicates="drop",
                labels=False,
            )
            hypotheses_rows.append(
                stratified_effect(
                    session_subset,
                    "activity_without_download_exposed",
                    "returned_active_m1",
                    ["month", "tenure_band", "baseline_band"],
                    "atividade_sem_download_vs_sessao_sem_atividade",
                )
            )
        outputs["analytics_hypotheses"] = pd.DataFrame(hypotheses_rows)

        outputs["analytics_feature_admission"] = pd.DataFrame(
            [
                {
                    "feature": col,
                    "definition": definition,
                    "missing_rate": compute_missing_rate(model_base[col]) if col in model_base.columns else np.nan,
                    "admitted_models": int(col in [
                        "session_count_month",
                        "active_days_month",
                        "total_session_minutes_month",
                        "strict_download_count_month",
                        "content_views_month",
                        "other_activity_non_download_events_month",
                        "mapped_lessons_month",
                        "lifetime_active_months",
                        "lifetime_active_minutes_total",
                        "active_streak_current_months",
                        "strict_streak_current_months",
                        "used_desktop_flag",
                        "used_mobile_flag",
                    ]),
                    "note": note,
                }
                for col, definition, note in [
                    ("strict_download_count_month", "contagem de strict_value no mês", "Mantida como contínua; bandas só na apresentação."),
                    ("session_count_month", "contagem de sessões limpas no mês", "Elegível para frequência."),
                    ("total_session_minutes_month", "tempo total limpo no mês", "Elegível para intensidade."),
                    ("active_streak_current_months", "sequência ativa corrente", "Elegível e interpretável."),
                    ("strict_streak_current_months", "sequência strict corrente", "Elegível e interpretável."),
                    ("estado_group", "UF com missing explícito", "Usada para interpretação/modelos, com caveat de missing."),
                    ("currentsubject_group", "grupo de disciplina do cadastro", "Usada com caveat de qualidade do cadastro."),
                ]
            ]
        )

        for name, df in outputs.items():
            write_df_bundle(cfg.output_dir, name, df)

        summary = {
            "generated_at_utc": utc_now_iso(),
            "heavy_month_threshold": float(teacher_month["heavy_intensity_score"].dropna().quantile(0.90))
            if teacher_month["heavy_intensity_score"].dropna().shape[0] > 0
            else None,
            "abandonment_meta": abandonment_meta,
            "cluster_rows": int(len(outputs["analytics_cluster_profiles"])),
            "model_targets": outputs["analytics_model_performance"]["target"].tolist()
            if not outputs["analytics_model_performance"].empty
            else [],
        }
        write_json(cfg.output_dir / "json" / "analytics_users_summary_v2.json", summary)

        md_lines = [
            "# Analytics de usuários v2",
            "",
            f"- Gerado em UTC: {summary['generated_at_utc']}",
            f"- Threshold heavy_month (p90 score): {fmt_num(summary['heavy_month_threshold'], 3)}",
            f"- Gap sugerido para abandono (meses): {summary['abandonment_meta'].get('suggested_abandonment_gap_months')}",
            f"- Nota abandono: {summary['abandonment_meta'].get('note')}",
            "",
            "## Artefatos",
        ]
        for key in outputs:
            md_lines.append(f"- `{key}`")
        write_markdown(cfg.output_dir / "audit" / "analytics_users_summary_v2.md", md_lines)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
