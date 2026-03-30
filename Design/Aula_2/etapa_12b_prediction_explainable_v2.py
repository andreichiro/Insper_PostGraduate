#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from explainable_drift_prediction_common_v2 import (
    CONTROL_MIN_GROUP_ROWS,
    PROFILE_CONTROL_VAR,
    RECENT_WINDOW_MONTHS,
    attach_reference,
    build_config,
    build_feature_candidates,
    build_input_map,
    build_output_reference,
    build_preprocessor,
    chi_square_with_cramers_v,
    connect_output,
    connect_source,
    load_public_tables,
    normalize_text,
    persist_table,
    prepare_model_population,
    safe_auc,
    safe_average_precision,
    safe_brier,
    safe_log_loss,
    strip_reference_cols,
    top_decile_lift,
    write_json,
    write_markdown,
    Config,
)

PRIMARY_RECENT_WINDOW = "recent_12m"
ANALYSIS_WINDOWS: Dict[str, int | None] = {
    "all_history": None,
    "recent_12m": 12,
    "recent_6m": RECENT_WINDOW_MONTHS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part 2: explainable prediction review from the relevant base-modelada tables.")
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def build_assumptions_table() -> pd.DataFrame:
    rows = [
        {
            "step_number": 1,
            "topic": "target_definition",
            "assumption": "The prediction target is defined at month t+1: either `target_churn_m1` or `target_return_active_m1`.",
            "why_this_is_sound": "This is the operational question the product team cares about: what happens next month after observing month t.",
            "what_changes_if_it_fails": "If the target is not aligned to month t+1, the model no longer answers the intended retention/churn question.",
        },
        {
            "step_number": 2,
            "topic": "control_variable",
            "assumption": f"`{PROFILE_CONTROL_VAR}` is only used as a control after checking coverage, group size, and association with the target.",
            "why_this_is_sound": "A weak or mostly missing control variable creates noise instead of rigor.",
            "what_changes_if_it_fails": "The comparison between profile-only and behavior-plus-profile models becomes hard to interpret.",
        },
        {
            "step_number": 3,
            "topic": "feature_admission",
            "assumption": "Only features available at month t are admitted as predictors; future-derived fields are leakage and are excluded.",
            "why_this_is_sound": "A valid prediction model cannot look into the future.",
            "what_changes_if_it_fails": "Performance becomes inflated and non-deployable.",
        },
        {
            "step_number": 4,
            "topic": "evaluation",
            "assumption": "Models are evaluated with temporal train/test splits, not random row splits.",
            "why_this_is_sound": "This is the closest approximation to real deployment and directly exposes time drift risk.",
            "what_changes_if_it_fails": "Random splits overestimate performance by mixing future patterns into training.",
        },
        {
            "step_number": 5,
            "topic": "comparison_windows",
            "assumption": "The model is tested on all available history, on the most recent 12 months as the primary recent regime, and on the most recent 6 months as a thinner sensitivity check.",
            "why_this_is_sound": "This keeps a larger recent holdout as the main answer while still checking whether the very latest slice tells a different story.",
            "what_changes_if_it_fails": "We would miss whether recent distribution shift degraded the model.",
        },
    ]
    df = pd.DataFrame(rows)
    return attach_reference(
        df,
        source_tables=["mart_teacher_month_persona_ready", "audit_base_modelada_validation", "audit_persona_feature_readiness"],
        build_summary="Step-by-step assumptions that define target construction, feature admission, control validation, and evaluation design for the prediction analysis.",
        rebuild_from_raw="No raw rebuild is needed for this table itself; it documents the assumptions used by etapa_12b_prediction_explainable_v2.py after raw_para_base_modelada_v4.py has built the relevant-table layer.",
    )


def build_control_variable_validity(population: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    months = sorted(population["month"].dropna().unique().tolist())
    windows = {
        window_label: (
            population.copy()
            if month_count is None
            else population[population["month"].isin(months[-month_count:])].copy()
        )
        for window_label, month_count in ANALYSIS_WINDOWS.items()
    }
    validity_rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []

    for window_label, subset in windows.items():
        feature = normalize_text(subset[PROFILE_CONTROL_VAR])
        counts = feature.value_counts(dropna=False)
        missing_rate = float((feature == "missing").mean())
        min_group_rows = int(counts.min()) if not counts.empty else 0
        share_groups_ge_min = float((counts >= CONTROL_MIN_GROUP_ROWS).mean()) if not counts.empty else 0.0

        for target_col, target_label in [
            ("target_churn_m1", "abandonar_m1"),
            ("target_return_active_m1", "retornar_ativo_m1"),
        ]:
            p_value, cramers_v, n_groups, n_rows = chi_square_with_cramers_v(subset, PROFILE_CONTROL_VAR, target_col)
            is_supported = int(
                pd.notna(p_value)
                and p_value < 0.05
                and min_group_rows >= CONTROL_MIN_GROUP_ROWS
                and missing_rate <= 0.20
            )
            validity_rows.append(
                {
                    "window_label": window_label,
                    "target": target_label,
                    "control_variable": PROFILE_CONTROL_VAR,
                    "rows": int(n_rows),
                    "distinct_groups": int(n_groups),
                    "missing_rate": missing_rate,
                    "min_group_rows": min_group_rows,
                    "share_groups_ge_min_rows": share_groups_ge_min,
                    "chi_square_p_value": p_value,
                    "cramers_v": cramers_v,
                    "is_statistically_supported": is_supported,
                    "plain_english_readout": "usable control" if is_supported else "weak or unstable control",
                }
            )

            group_view = subset[[PROFILE_CONTROL_VAR, target_col]].copy()
            group_view[PROFILE_CONTROL_VAR] = normalize_text(group_view[PROFILE_CONTROL_VAR])
            grouped = (
                group_view.groupby(PROFILE_CONTROL_VAR, dropna=False)[target_col]
                .agg(rows="size", outcome_rate="mean")
                .reset_index()
                .rename(columns={PROFILE_CONTROL_VAR: "control_group"})
            )
            grouped["window_label"] = window_label
            grouped["target"] = target_label
            group_rows.extend(grouped.to_dict(orient="records"))

    validity_df = pd.DataFrame(validity_rows)
    validity_df = attach_reference(
        validity_df,
        source_tables=["mart_teacher_month_persona_ready", "dim_teacher", "audit_base_modelada_validation"],
        build_summary="Statistical validity check of the chosen control variable using coverage, group size, and association with the targets.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, build the modeling population from mart_teacher_month_persona_ready, then validate teacher_currentsubject_group against each target with missingness, group-size, chi-square, and Cramer's V checks.",
    )

    group_df = pd.DataFrame(group_rows).sort_values(["window_label", "target", "rows"], ascending=[True, True, False]).reset_index(drop=True)
    group_df = attach_reference(
        group_df,
        source_tables=["mart_teacher_month_persona_ready", "dim_teacher"],
        build_summary="Outcome-rate breakdown for each control-variable group, used to show why the control variable is or is not empirically useful.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, build the modeling population from mart_teacher_month_persona_ready, group by teacher_currentsubject_group, and compute rows plus outcome rates for each target and time window.",
    )
    return validity_df, group_df


def build_feature_screening(
    population: pd.DataFrame,
    feature_candidates: pd.DataFrame,
) -> pd.DataFrame:
    features = feature_candidates[feature_candidates["include_in_model"] == 1]["feature_name"].tolist()
    rows: List[Dict[str, Any]] = []

    for feature_name in features:
        if feature_name not in population.columns:
            continue
        series = pd.to_numeric(population[feature_name], errors="coerce")
        if series.notna().sum() < 100:
            continue
        for target_col, target_label in [
            ("target_churn_m1", "abandonar_m1"),
            ("target_return_active_m1", "retornar_ativo_m1"),
        ]:
            y = population[target_col].astype(int)
            valid = series.notna() & y.notna()
            if valid.sum() < 100 or y[valid].nunique() < 2:
                continue
            x = series[valid]
            y_valid = y[valid]
            raw_auc = safe_auc(y_valid, x)
            separation_auc = max(raw_auc, 1 - raw_auc) if pd.notna(raw_auc) else float("nan")
            pos = x[y_valid == 1]
            neg = x[y_valid == 0]
            if pos.empty or neg.empty:
                p_value = float("nan")
            else:
                try:
                    p_value = float(mannwhitneyu(pos, neg, alternative="two-sided").pvalue)
                except ValueError:
                    p_value = float("nan")
            rows.append(
                {
                    "feature_name": feature_name,
                    "target": target_label,
                    "rows_used": int(valid.sum()),
                    "positive_class_mean": float(pos.mean()) if not pos.empty else float("nan"),
                    "negative_class_mean": float(neg.mean()) if not neg.empty else float("nan"),
                    "effect_direction": "higher_in_positive_class" if pos.mean() >= neg.mean() else "lower_in_positive_class",
                    "raw_auc": raw_auc,
                    "separation_auc": separation_auc,
                    "mann_whitney_p_value": p_value,
                    "is_univariate_signal": int(
                        pd.notna(p_value) and p_value < 0.05 and pd.notna(separation_auc) and separation_auc >= 0.55
                    ),
                }
            )

    df = pd.DataFrame(rows).sort_values(["target", "separation_auc"], ascending=[True, False]).reset_index(drop=True)
    return attach_reference(
        df,
        source_tables=["mart_teacher_month_persona_ready", "audit_persona_feature_readiness"],
        build_summary="Univariate screening of each admitted numeric feature against each target using separation AUC and Mann-Whitney significance.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, build the modeling population from mart_teacher_month_persona_ready, screen each admitted numeric feature against each target, and compute separation_auc plus Mann-Whitney p-values.",
    )


def enrich_prediction_population(
    tables: Dict[str, pd.DataFrame],
    population: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    enriched = population.copy()
    formation = tables["fct_formation_clean"][["teacher_unique_id", "formation_month", "is_complete_status"]].copy()
    formation["formation_month"] = pd.to_datetime(formation["formation_month"], errors="coerce")
    formation = formation.dropna(subset=["formation_month"])

    if formation.empty:
        enriched["formation_events_to_month"] = 0.0
        enriched["formation_complete_events_to_month"] = 0.0
        enriched["had_formation_to_month"] = 0
    else:
        formation_monthly = (
            formation.groupby(["teacher_unique_id", "formation_month"], as_index=False)
            .agg(
                formation_events_month=("teacher_unique_id", "size"),
                formation_complete_events_month=("is_complete_status", "sum"),
            )
            .sort_values(["teacher_unique_id", "formation_month"])
        )
        formation_monthly["formation_events_to_month"] = formation_monthly.groupby("teacher_unique_id")[
            "formation_events_month"
        ].cumsum()
        formation_monthly["formation_complete_events_to_month"] = formation_monthly.groupby("teacher_unique_id")[
            "formation_complete_events_month"
        ].cumsum()
        enriched = enriched.sort_values(["teacher_unique_id", "month"]).reset_index(drop=True)
        formation_monthly = formation_monthly.rename(columns={"formation_month": "month"})
        enriched = enriched.merge(
            formation_monthly[
                [
                    "teacher_unique_id",
                    "month",
                    "formation_events_month",
                    "formation_complete_events_month",
                ]
            ],
            on=["teacher_unique_id", "month"],
            how="left",
        )
        for col in ["formation_events_month", "formation_complete_events_month"]:
            enriched[col] = pd.to_numeric(enriched[col], errors="coerce").fillna(0)
        enriched["formation_events_to_month"] = enriched.groupby("teacher_unique_id")["formation_events_month"].cumsum()
        enriched["formation_complete_events_to_month"] = enriched.groupby("teacher_unique_id")[
            "formation_complete_events_month"
        ].cumsum()
        enriched["had_formation_to_month"] = (enriched["formation_events_to_month"] > 0).astype(int)
        enriched = enriched.drop(columns=["formation_events_month", "formation_complete_events_month"])

    interactions = tables["fct_interaction_clean"][
        ["teacher_unique_id", "interaction_month", "event_family", "event_action"]
    ].copy()
    interactions["interaction_month"] = pd.to_datetime(interactions["interaction_month"], errors="coerce")
    interactions = interactions.dropna(subset=["interaction_month"])
    interactions["event_family"] = normalize_text(interactions["event_family"])
    interactions["event_action"] = normalize_text(interactions["event_action"])

    if interactions.empty:
        for col in [
            "raw_interaction_rows_month",
            "aula_share_month",
            "plano_share_month",
            "ia_share_month",
            "prova_share_month",
            "view_share_month",
            "download_share_month",
            "navigation_share_month",
            "other_action_share_month",
            "zero_interaction_month_flag",
            "navigation_without_activity_flag",
        ]:
            enriched[col] = 0.0
    else:
        interaction_monthly = (
            interactions.groupby(["teacher_unique_id", "interaction_month"], as_index=False)
            .agg(
                raw_interaction_rows_month=("teacher_unique_id", "size"),
                aula_rows_month=("event_family", lambda s: int((s == "aula").sum())),
                plano_rows_month=("event_family", lambda s: int((s == "plano").sum())),
                ia_rows_month=("event_family", lambda s: int((s == "ia").sum())),
                prova_rows_month=("event_family", lambda s: int((s == "prova").sum())),
                view_rows_month=("event_action", lambda s: int((s == "view").sum())),
                download_rows_month=("event_action", lambda s: int((s == "download").sum())),
                navigation_rows_month=("event_action", lambda s: int((s == "navigation").sum())),
                other_rows_month=("event_action", lambda s: int((~s.isin(["view", "download", "navigation", "missing"])).sum())),
            )
            .rename(columns={"interaction_month": "month"})
        )
        enriched = enriched.merge(interaction_monthly, on=["teacher_unique_id", "month"], how="left")
        for col in [
            "raw_interaction_rows_month",
            "aula_rows_month",
            "plano_rows_month",
            "ia_rows_month",
            "prova_rows_month",
            "view_rows_month",
            "download_rows_month",
            "navigation_rows_month",
            "other_rows_month",
        ]:
            enriched[col] = pd.to_numeric(enriched[col], errors="coerce").fillna(0.0)
        interaction_base = enriched["raw_interaction_rows_month"].where(enriched["raw_interaction_rows_month"] > 0, 1.0)
        enriched["aula_share_month"] = enriched["aula_rows_month"] / interaction_base
        enriched["plano_share_month"] = enriched["plano_rows_month"] / interaction_base
        enriched["ia_share_month"] = enriched["ia_rows_month"] / interaction_base
        enriched["prova_share_month"] = enriched["prova_rows_month"] / interaction_base
        enriched["view_share_month"] = enriched["view_rows_month"] / interaction_base
        enriched["download_share_month"] = enriched["download_rows_month"] / interaction_base
        enriched["navigation_share_month"] = enriched["navigation_rows_month"] / interaction_base
        enriched["other_action_share_month"] = enriched["other_rows_month"] / interaction_base
        enriched["zero_interaction_month_flag"] = (enriched["raw_interaction_rows_month"] == 0).astype(int)
        enriched["navigation_without_activity_flag"] = (
            (enriched["raw_interaction_rows_month"] == 0)
            | (enriched["clean_entry_exposed_no_activity_no_download_flag"].fillna(0) == 1)
            | (
                (enriched["navigation_share_month"] >= 0.50)
                & (enriched["download_share_month"] == 0)
                & (enriched["other_rows_month"] == 0)
            )
        ).astype(int)
        enriched = enriched.drop(
            columns=[
                "aula_rows_month",
                "plano_rows_month",
                "ia_rows_month",
                "prova_rows_month",
                "view_rows_month",
                "download_rows_month",
                "navigation_rows_month",
                "other_rows_month",
            ]
        )

    def safe_ratio(numerator_col: str, denominator_col: str) -> pd.Series:
        numerator = pd.to_numeric(enriched[numerator_col], errors="coerce").fillna(0.0)
        denominator = pd.to_numeric(enriched[denominator_col], errors="coerce").fillna(0.0)
        return numerator / denominator.where(denominator > 0, 1.0)

    enriched["downloads_per_view_month"] = safe_ratio("download_count_month", "content_views_month")
    enriched["strict_downloads_per_view_month"] = safe_ratio("strict_download_count_month", "content_views_month")
    enriched["content_views_per_session_month"] = safe_ratio("content_views_month", "clean_entry_session_count_month")
    enriched["session_minutes_per_active_day_month"] = safe_ratio(
        "clean_entry_total_session_minutes_month",
        "active_days_month",
    )
    enriched["lifetime_minutes_per_active_month"] = safe_ratio(
        "lifetime_clean_entry_minutes_total",
        "lifetime_active_months",
    )
    enriched["formation_completion_rate_to_month"] = safe_ratio(
        "formation_complete_events_to_month",
        "formation_events_to_month",
    )

    supplemental_specs = [
        {
            "feature_name": "lifetime_active_months",
            "feature_level": "teacher_month",
            "feature_role": "history_tenure",
            "definition": "number of active months accumulated up to month t",
            "caveat": "time-safe tenure proxy; do not replace it with full-dataset totals that leak future information",
            "include_in_model": 1,
            "model_usage": "predictor",
        },
        {
            "feature_name": "lifetime_clean_entry_minutes_total",
            "feature_level": "teacher_month",
            "feature_role": "history_usage",
            "definition": "cumulative clean-entry minutes accumulated up to month t",
            "caveat": "time-safe history feature built from month-t and earlier only",
            "include_in_model": 1,
            "model_usage": "predictor",
        },
        {
            "feature_name": "active_streak_current_months",
            "feature_level": "teacher_month",
            "feature_role": "history_usage",
            "definition": "current active streak length up to month t",
            "caveat": "captures consecutive activity; can be sensitive to gaps in observation",
            "include_in_model": 1,
            "model_usage": "predictor",
        },
        {
            "feature_name": "strict_streak_current_months",
            "feature_level": "teacher_month",
            "feature_role": "history_usage",
            "definition": "current strict-value streak length up to month t",
            "caveat": "captures repeated strict value; should be read together with overall activity",
            "include_in_model": 1,
            "model_usage": "predictor",
        },
        {
            "feature_name": "download_aula_count_month",
            "feature_level": "teacher_month",
            "feature_role": "behavior_core",
            "definition": "downloads of aulas in month t",
            "caveat": "one specific download subtype; complements total download_count_month",
            "include_in_model": 1,
            "model_usage": "predictor",
        },
        {
            "feature_name": "download_plano_count_month",
            "feature_level": "teacher_month",
            "feature_role": "behavior_core",
            "definition": "downloads of plano materials in month t",
            "caveat": "one specific download subtype; complements total download_count_month",
            "include_in_model": 1,
            "model_usage": "predictor",
        },
        {
            "feature_name": "formation_events_to_month",
            "feature_level": "teacher_month",
            "feature_role": "teacher_formation",
            "definition": "cumulative count of formation events up to month t",
            "caveat": "time-safe cumulative formation signal built from fct_formation_clean",
            "include_in_model": 1,
            "model_usage": "predictor",
        },
        {
            "feature_name": "formation_complete_events_to_month",
            "feature_level": "teacher_month",
            "feature_role": "teacher_formation",
            "definition": "cumulative count of completed formation events up to month t",
            "caveat": "time-safe completion signal; weaker if completion tracking is noisy",
            "include_in_model": 1,
            "model_usage": "predictor",
        },
        {
            "feature_name": "had_formation_to_month",
            "feature_level": "teacher_month",
            "feature_role": "teacher_formation",
            "definition": "whether the teacher had any formation event up to month t",
            "caveat": "binary formation proxy; less informative than cumulative counts",
            "include_in_model": 1,
            "model_usage": "predictor",
        },
        {
            "feature_name": "downloads_per_view_month",
            "feature_level": "teacher_month",
            "feature_role": "feature_engineering",
            "definition": "downloads divided by content views in month t",
            "caveat": "interpretable only when views exist; high values may reflect low denominators",
            "include_in_model": 1,
            "model_usage": "predictor",
        },
        {
            "feature_name": "strict_downloads_per_view_month",
            "feature_level": "teacher_month",
            "feature_role": "feature_engineering",
            "definition": "strict downloads divided by content views in month t",
            "caveat": "interpretable only when views exist; can be unstable with very low view counts",
            "include_in_model": 1,
            "model_usage": "predictor",
        },
        {
            "feature_name": "content_views_per_session_month",
            "feature_level": "teacher_month",
            "feature_role": "feature_engineering",
            "definition": "content views divided by clean sessions in month t",
            "caveat": "sensitive to low session counts; useful as intensity-per-session proxy",
            "include_in_model": 1,
            "model_usage": "predictor",
        },
        {
            "feature_name": "session_minutes_per_active_day_month",
            "feature_level": "teacher_month",
            "feature_role": "feature_engineering",
            "definition": "clean session minutes divided by active days in month t",
            "caveat": "captures depth per active day, but can spike when active_days_month is very small",
            "include_in_model": 1,
            "model_usage": "predictor",
        },
        {
            "feature_name": "lifetime_minutes_per_active_month",
            "feature_level": "teacher_month",
            "feature_role": "feature_engineering",
            "definition": "cumulative lifetime minutes divided by cumulative active months up to month t",
            "caveat": "summarizes historical intensity; partially overlaps with tenure and cumulative minutes",
            "include_in_model": 1,
            "model_usage": "predictor",
        },
        {
            "feature_name": "formation_completion_rate_to_month",
            "feature_level": "teacher_month",
            "feature_role": "feature_engineering",
            "definition": "completed formation events divided by total formation events up to month t",
            "caveat": "requires formation tracking coverage; undefined zero denominators are set to zero",
            "include_in_model": 1,
            "model_usage": "predictor",
        },
        {
            "feature_name": "raw_interaction_rows_month",
            "feature_level": "teacher_month",
            "feature_role": "behavior_proxy",
            "definition": "raw interaction rows observed in month t from fct_interaction_clean",
            "caveat": "mostly overlaps with interaction_rows_month in the mart; included here only because the behavior-share features are derived from the same raw aggregate",
            "include_in_model": 0,
            "model_usage": "context_only",
        },
        {
            "feature_name": "aula_share_month",
            "feature_level": "teacher_month",
            "feature_role": "behavior_proxy",
            "definition": "share of raw interactions in month t whose family is aula",
            "caveat": "share features are relative, not causal; they depend on observed interaction coverage",
            "include_in_model": 1,
            "model_usage": "predictor",
        },
        {
            "feature_name": "plano_share_month",
            "feature_level": "teacher_month",
            "feature_role": "behavior_proxy",
            "definition": "share of raw interactions in month t whose family is plano",
            "caveat": "share features are relative, not causal; they depend on observed interaction coverage",
            "include_in_model": 1,
            "model_usage": "predictor",
        },
        {
            "feature_name": "ia_share_month",
            "feature_level": "teacher_month",
            "feature_role": "behavior_proxy",
            "definition": "share of raw interactions in month t whose family is ia",
            "caveat": "share features are relative, not causal; they depend on observed interaction coverage",
            "include_in_model": 1,
            "model_usage": "predictor",
        },
        {
            "feature_name": "prova_share_month",
            "feature_level": "teacher_month",
            "feature_role": "behavior_proxy",
            "definition": "share of raw interactions in month t whose family is prova",
            "caveat": "share features are relative, not causal; they depend on observed interaction coverage",
            "include_in_model": 1,
            "model_usage": "predictor",
        },
        {
            "feature_name": "view_share_month",
            "feature_level": "teacher_month",
            "feature_role": "behavior_proxy",
            "definition": "share of raw interactions in month t whose action is view",
            "caveat": "share features are relative, not causal; they depend on observed interaction coverage",
            "include_in_model": 1,
            "model_usage": "predictor",
        },
        {
            "feature_name": "download_share_month",
            "feature_level": "teacher_month",
            "feature_role": "behavior_proxy",
            "definition": "share of raw interactions in month t whose action is download",
            "caveat": "share features are relative, not causal; they depend on observed interaction coverage",
            "include_in_model": 1,
            "model_usage": "predictor",
        },
        {
            "feature_name": "navigation_share_month",
            "feature_level": "teacher_month",
            "feature_role": "behavior_proxy",
            "definition": "share of raw interactions in month t whose action is navigation",
            "caveat": "high values may reflect exploration without value, but can also reflect missing action mapping",
            "include_in_model": 1,
            "model_usage": "predictor",
        },
        {
            "feature_name": "other_action_share_month",
            "feature_level": "teacher_month",
            "feature_role": "behavior_proxy",
            "definition": "share of raw interactions in month t mapped to other meaningful actions",
            "caveat": "depends on action taxonomy quality in fct_interaction_clean",
            "include_in_model": 1,
            "model_usage": "predictor",
        },
        {
            "feature_name": "zero_interaction_month_flag",
            "feature_level": "teacher_month",
            "feature_role": "behavior_proxy",
            "definition": "whether month t has no raw interactions even though the teacher-month is observed",
            "caveat": "captures both real passive behavior and possible tracking-thin months",
            "include_in_model": 1,
            "model_usage": "predictor",
        },
        {
            "feature_name": "navigation_without_activity_flag",
            "feature_level": "teacher_month",
            "feature_role": "behavior_proxy",
            "definition": "whether month t looks like navigation without value activity using raw action shares and clean-entry flags",
            "caveat": "best available proxy for passive navigation; not event-level dwell time",
            "include_in_model": 1,
            "model_usage": "predictor",
        },
    ]
    supplemental_df = pd.DataFrame(supplemental_specs)
    return enriched, supplemental_df


def extend_feature_candidates_with_supplemental_features(
    feature_candidates: pd.DataFrame,
    population: pd.DataFrame,
    supplemental_features: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    extended = strip_reference_cols(feature_candidates)
    existing = set(extended["feature_name"].tolist())
    supplemental_rows: List[Dict[str, Any]] = []

    for _, row in supplemental_features.iterrows():
        feature_name = row["feature_name"]
        if feature_name not in population.columns or feature_name in existing:
            continue
        series = pd.to_numeric(population[feature_name], errors="coerce")
        supplemental_rows.append(
            {
                "feature_name": feature_name,
                "feature_level": row["feature_level"],
                "feature_role": row["feature_role"],
                "definition": row["definition"],
                "caveat": row["caveat"],
                "missing_rate": float(series.isna().mean()),
                "zero_share": float((series.fillna(0) == 0).mean()),
                "std": float(series.std(ddof=0)) if series.notna().sum() else 0.0,
                "recommended_for_persona_analysis": 0,
                "recommended_for_persona_ranges": 0,
                "recommended_for_behavior_clustering": 0,
                "is_leakage_feature": 0,
                "is_context_only_feature": 0,
                "is_telemetry_support_feature": 0,
                "include_as_control": 0,
                "include_in_model": int(row["include_in_model"]),
                "model_usage": row["model_usage"],
                "p25": float(series.quantile(0.25)) if series.notna().sum() else float("nan"),
                "p50": float(series.quantile(0.50)) if series.notna().sum() else float("nan"),
                "p75": float(series.quantile(0.75)) if series.notna().sum() else float("nan"),
                "p90": float(series.quantile(0.90)) if series.notna().sum() else float("nan"),
                "note": "Supplemental time-safe predictor added for the prediction review.",
            }
        )

    if supplemental_rows:
        extended = pd.concat([extended, pd.DataFrame(supplemental_rows)], ignore_index=True, sort=False)
    extended = extended.sort_values(["include_in_model", "feature_name"], ascending=[False, True]).reset_index(drop=True)
    extended = attach_reference(
        extended,
        source_tables=["audit_persona_feature_readiness", "dim_persona_range_candidates", "mart_teacher_month_persona_ready", "fct_formation_clean"],
        build_summary="Candidate-feature table expanded with time-safe supplemental predictors for tenure, formation, and download subtype behavior.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, build the modeling population from mart_teacher_month_persona_ready, enrich it with cumulative formation features from fct_formation_clean, then combine the persona-readiness candidates with the supplemental time-safe predictors.",
    )

    numeric_features = sorted(
        set(
            strip_reference_cols(extended)
            .loc[
                (strip_reference_cols(extended)["include_in_model"] == 1)
                & (strip_reference_cols(extended)["model_usage"] != "control_only"),
                "feature_name",
            ]
            .tolist()
        )
    )
    numeric_features = [feature for feature in numeric_features if pd.api.types.is_numeric_dtype(population[feature])]
    categorical_features = [PROFILE_CONTROL_VAR, "month_signal_class"]
    categorical_features = [feature for feature in categorical_features if feature in population.columns]
    return extended, numeric_features, categorical_features


def build_feature_theme_review(
    feature_candidates: pd.DataFrame,
    feature_screening: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> pd.DataFrame:
    features_of_interest = {
        "clean_entry_avg_session_minutes_month": "time_spent",
        "session_minutes_per_active_day_month": "time_spent",
        "formation_events_to_month": "teacher_formation",
        "formation_complete_events_to_month": "teacher_formation",
        "had_formation_to_month": "teacher_formation",
        "lifetime_active_months": "platform_tenure",
        "lifetime_clean_entry_minutes_total": "platform_history",
        "active_streak_current_months": "platform_history",
        "strict_streak_current_months": "platform_history",
        "strict_download_count_month": "downloads",
        "download_count_month": "downloads",
        "download_aula_count_month": "downloads",
        "download_plano_count_month": "downloads",
        "content_views_month": "views_clicks",
        "aula_events_month": "views_clicks",
        "mapped_lessons_month": "views_clicks",
        "other_activity_non_download_events_month": "views_clicks",
        "view_share_month": "behavior_shares",
        "download_share_month": "behavior_shares",
        "navigation_share_month": "behavior_shares",
        "navigation_without_activity_flag": "behavior_shares",
        "interaction_rows_month": "behavior_shares",
    }
    candidates = strip_reference_cols(feature_candidates)
    screening = strip_reference_cols(feature_screening)
    importance = strip_reference_cols(feature_importance)

    rows: List[Dict[str, Any]] = []
    for feature_name, theme in features_of_interest.items():
        if feature_name not in candidates["feature_name"].values:
            rows.append(
                {
                    "theme": theme,
                    "feature_name": feature_name,
                    "included_in_model": 0,
                    "why_included_or_not": "feature not available or not admitted",
                    "best_univariate_target": "",
                    "best_univariate_auc": float("nan"),
                    "best_importance_target_window": "",
                    "best_importance_value": float("nan"),
                }
            )
            continue
        candidate_row = candidates.loc[candidates["feature_name"] == feature_name].iloc[0]
        screen_rows = screening.loc[screening["feature_name"] == feature_name].sort_values("separation_auc", ascending=False)
        importance_rows = importance.loc[importance["feature_name"] == feature_name].sort_values("permutation_importance_mean", ascending=False)
        rows.append(
            {
                "theme": theme,
                "feature_name": feature_name,
                "included_in_model": int(candidate_row["include_in_model"]),
                "why_included_or_not": str(candidate_row["model_usage"]),
                "best_univariate_target": str(screen_rows.iloc[0]["target"]) if not screen_rows.empty else "",
                "best_univariate_auc": float(screen_rows.iloc[0]["separation_auc"]) if not screen_rows.empty else float("nan"),
                "best_importance_target_window": (
                    f"{importance_rows.iloc[0]['target']} | {importance_rows.iloc[0]['window_label']}"
                    if not importance_rows.empty
                    else ""
                ),
                "best_importance_value": float(importance_rows.iloc[0]["permutation_importance_mean"]) if not importance_rows.empty else float("nan"),
            }
        )

    df = pd.DataFrame(rows).sort_values(["theme", "included_in_model", "feature_name"], ascending=[True, False, True]).reset_index(drop=True)
    return attach_reference(
        df,
        source_tables=["analytics_prediction_feature_candidates_v2", "analytics_prediction_feature_screening_v2", "analytics_prediction_feature_importance_explainable_v2"],
        build_summary="Review table showing whether formation, tenure, download, and click-related features were considered, admitted, and empirically useful in the ML workflow.",
        rebuild_from_raw="Rerun etapa_12b_prediction_explainable_v2.py after raw_para_base_modelada_v4.py; this table is derived from the feature-candidate, feature-screening, and feature-importance outputs generated in the same run.",
    )


def supported_control_windows(control_validity: pd.DataFrame) -> set[str]:
    if control_validity.empty:
        return set()
    grouped = (
        strip_reference_cols(control_validity)
        .groupby("window_label", as_index=False)["is_statistically_supported"]
        .min()
        .rename(columns={"is_statistically_supported": "all_targets_supported"})
    )
    return set(grouped.loc[grouped["all_targets_supported"] == 1, "window_label"].tolist())


def window_subset(population: pd.DataFrame, window_label: str) -> pd.DataFrame:
    months = sorted(population["month"].dropna().unique().tolist())
    month_count = ANALYSIS_WINDOWS.get(window_label)
    if month_count is None:
        return population.copy()
    if window_label in ANALYSIS_WINDOWS:
        return population[population["month"].isin(months[-month_count:])].copy()
    raise ValueError(f"Unsupported window_label: {window_label}")


def temporal_train_test_split(population: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.Timestamp], List[pd.Timestamp]]:
    months = sorted(population["month"].dropna().unique().tolist())
    if len(months) < 3:
        return pd.DataFrame(), pd.DataFrame(), [], []
    split_idx = max(1, min(len(months) - 1, int(np.floor(len(months) * 0.67))))
    train_months = months[:split_idx]
    test_months = months[split_idx:]
    train = population[population["month"].isin(train_months)].copy()
    test = population[population["month"].isin(test_months)].copy()
    return train, test, train_months, test_months


def eligible_numeric_features(feature_candidates: pd.DataFrame, population: pd.DataFrame) -> List[str]:
    candidates = strip_reference_cols(feature_candidates).copy()
    eligible = candidates[
        (candidates["include_in_model"] == 1)
        & (candidates["model_usage"] != "control_only")
        & (candidates["feature_name"].isin(population.columns))
    ].copy()
    eligible = eligible[
        (eligible["missing_rate"].fillna(1.0) <= 0.95)
        & (eligible["std"].fillna(0.0) > 0)
    ]
    feature_names = sorted(
        feature_name
        for feature_name in eligible["feature_name"].tolist()
        if pd.api.types.is_numeric_dtype(population[feature_name])
    )
    return feature_names


def build_data_sufficiency_review(
    population: pd.DataFrame,
    feature_candidates: pd.DataFrame,
) -> pd.DataFrame:
    feature_pool = eligible_numeric_features(feature_candidates, population)
    rows: List[Dict[str, Any]] = []
    for window_label in ANALYSIS_WINDOWS:
        subset = window_subset(population, window_label)
        train, test, train_months, test_months = temporal_train_test_split(subset)
        if train.empty or test.empty:
            continue
        for target_col, target_label in [
            ("target_churn_m1", "abandonar_m1"),
            ("target_return_active_m1", "retornar_ativo_m1"),
        ]:
            y_train = train[target_col].astype(int)
            positive_events = int(y_train.sum())
            negative_events = int((1 - y_train).sum())
            rows.append(
                {
                    "window_label": window_label,
                    "target": target_label,
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "train_months": int(len(train_months)),
                    "test_months": int(len(test_months)),
                    "positive_events_train": positive_events,
                    "negative_events_train": negative_events,
                    "candidate_numeric_features": int(len(feature_pool)),
                    "events_per_feature": float(positive_events / max(len(feature_pool), 1)),
                    "minority_events_per_feature": float(min(positive_events, negative_events) / max(len(feature_pool), 1)),
                    "enough_for_modeling": int(
                        len(train) >= 1000
                        and len(train_months) >= 4
                        and positive_events >= 200
                        and min(positive_events, negative_events) / max(len(feature_pool), 1) >= 10
                    ),
                    "plain_english_readout": (
                        "sufficient for regularized / tree-based modeling"
                        if len(train) >= 1000 and positive_events >= 200
                        else "thin data; results should be treated as exploratory"
                    ),
                }
            )
    df = pd.DataFrame(rows)
    return attach_reference(
        df,
        source_tables=["mart_teacher_month_persona_ready", "analytics_prediction_feature_candidates_v2"],
        build_summary="Data sufficiency review for temporal prediction, including event counts, rows, months, and events-per-feature diagnostics.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, prepare the prediction population from mart_teacher_month_persona_ready, derive the eligible numeric feature pool, split train/test by time, and compute rows plus events-per-feature diagnostics for each target and window.",
    )


def build_exclusion_bias_review(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    month = tables["mart_teacher_month_persona_ready"].copy()
    month["month"] = pd.to_datetime(month["month"], errors="coerce")
    for col in [
        "active_days_month",
        "strict_download_count_month",
        "content_views_month",
        "clean_entry_total_session_minutes_month",
        "lifetime_active_months",
    ]:
        month[col] = pd.to_numeric(month[col], errors="coerce")
    for col in ["teacher_utm_group", "teacher_currentsubject_group", "teacher_currentstage"]:
        month[col] = normalize_text(month[col])

    reason = np.select(
        [
            month["observed_month_flag"].fillna(0) != 1,
            month["persona_analysis_eligible_flag"].fillna(0) != 1,
            month["next_month_observed_flag"].fillna(0) != 1,
        ],
        [
            "excluded_not_observed",
            "excluded_not_persona_eligible",
            "excluded_no_next_month",
        ],
        default="included",
    )
    month["inclusion_status"] = reason

    numeric_rows: List[Dict[str, Any]] = []
    included = month[month["inclusion_status"] == "included"].copy()
    numeric_features = [
        "active_days_month",
        "strict_download_count_month",
        "content_views_month",
        "clean_entry_total_session_minutes_month",
        "lifetime_active_months",
    ]
    for status in ["excluded_not_observed", "excluded_not_persona_eligible", "excluded_no_next_month"]:
        subset = month[month["inclusion_status"] == status].copy()
        if subset.empty or included.empty:
            continue
        for feature_name in numeric_features:
            inc = pd.to_numeric(included[feature_name], errors="coerce").dropna()
            exc = pd.to_numeric(subset[feature_name], errors="coerce").dropna()
            if inc.empty or exc.empty:
                continue
            pooled_std = np.sqrt(((inc.var(ddof=0) + exc.var(ddof=0)) / 2.0))
            smd = float((exc.mean() - inc.mean()) / pooled_std) if pooled_std > 0 else float("nan")
            numeric_rows.append(
                {
                    "comparison_type": "numeric",
                    "inclusion_status": status,
                    "feature_name": feature_name,
                    "included_rows": int(len(included)),
                    "excluded_rows": int(len(subset)),
                    "included_mean": float(inc.mean()),
                    "excluded_mean": float(exc.mean()),
                    "standardized_mean_difference": smd,
                }
            )

    categorical_rows: List[Dict[str, Any]] = []
    for status in ["excluded_not_observed", "excluded_not_persona_eligible", "excluded_no_next_month"]:
        subset = month[month["inclusion_status"] == status].copy()
        if subset.empty or included.empty:
            continue
        for feature_name in ["teacher_utm_group", "teacher_currentsubject_group", "teacher_currentstage"]:
            inc_share = normalize_text(included[feature_name]).value_counts(normalize=True)
            exc_share = normalize_text(subset[feature_name]).value_counts(normalize=True)
            cats = sorted(set(inc_share.index).union(exc_share.index))
            diff_pp = float(max(abs(100 * (inc_share.get(cat, 0.0) - exc_share.get(cat, 0.0))) for cat in cats))
            categorical_rows.append(
                {
                    "comparison_type": "categorical",
                    "inclusion_status": status,
                    "feature_name": feature_name,
                    "included_rows": int(len(included)),
                    "excluded_rows": int(len(subset)),
                    "max_share_diff_pp": diff_pp,
                    "total_variation": float(0.5 * sum(abs(inc_share.get(cat, 0.0) - exc_share.get(cat, 0.0)) for cat in cats)),
                }
            )

    df = pd.concat([pd.DataFrame(numeric_rows), pd.DataFrame(categorical_rows)], ignore_index=True, sort=False)
    return attach_reference(
        df,
        source_tables=["mart_teacher_month_persona_ready"],
        build_summary="Bias review comparing included modeling rows with excluded rows across key numeric and categorical pre-target features.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, take mart_teacher_month_persona_ready, label rows by inclusion versus exclusion reason, and compare included versus excluded rows on key pre-target features and profile distributions.",
    )


def build_feature_relation_review(population: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("formation_events_to_month", "lifetime_active_months", "formation_vs_tenure"),
        ("formation_events_to_month", "lifetime_clean_entry_minutes_total", "formation_vs_platform_time"),
        ("formation_events_to_month", "download_count_month", "formation_vs_download"),
        ("formation_events_to_month", "content_views_month", "formation_vs_content_views"),
        ("formation_complete_events_to_month", "download_count_month", "formation_completion_vs_download"),
        ("lifetime_active_months", "download_count_month", "tenure_vs_download"),
        ("lifetime_active_months", "content_views_month", "tenure_vs_content_views"),
        ("lifetime_clean_entry_minutes_total", "download_count_month", "platform_time_vs_download"),
        ("download_count_month", "content_views_month", "download_vs_content_views"),
    ]
    rows: List[Dict[str, Any]] = []
    for x_feature, y_feature, relation_label in pairs:
        if x_feature not in population.columns or y_feature not in population.columns:
            continue
        x = pd.to_numeric(population[x_feature], errors="coerce")
        y = pd.to_numeric(population[y_feature], errors="coerce")
        valid = x.notna() & y.notna()
        if valid.sum() < 200:
            continue
        rho, p_value = spearmanr(x[valid], y[valid])
        by_month_rows: List[float] = []
        for _, month_group in population.loc[valid, ["month", x_feature, y_feature]].groupby("month"):
            if len(month_group) < 50:
                continue
            if pd.to_numeric(month_group[x_feature], errors="coerce").nunique(dropna=True) < 2:
                continue
            if pd.to_numeric(month_group[y_feature], errors="coerce").nunique(dropna=True) < 2:
                continue
            month_rho, _ = spearmanr(
                pd.to_numeric(month_group[x_feature], errors="coerce"),
                pd.to_numeric(month_group[y_feature], errors="coerce"),
            )
            if pd.notna(month_rho):
                by_month_rows.append(float(month_rho))
        within_month_rho = float(np.mean(by_month_rows)) if by_month_rows else float("nan")
        distortion_note = "possible accumulation/time-trend confounding" if "lifetime" in x_feature or "lifetime" in y_feature else "main risk is same-month behavioral confounding"
        rows.append(
            {
                "relation_label": relation_label,
                "x_feature": x_feature,
                "y_feature": y_feature,
                "rows_used": int(valid.sum()),
                "spearman_rho_overall": float(rho),
                "spearman_p_value": float(p_value),
                "mean_within_month_spearman_rho": within_month_rho,
                "plain_english_readout": (
                    "relationship persists after controlling for calendar month"
                    if pd.notna(within_month_rho) and abs(within_month_rho) >= 0.10
                    else "relationship is weak once same-month timing is respected"
                ),
                "distortion_risk": distortion_note,
            }
        )
    df = pd.DataFrame(rows).sort_values("spearman_rho_overall", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    return attach_reference(
        df,
        source_tables=["mart_teacher_month_persona_ready", "fct_formation_clean"],
        build_summary="Relationship review for the requested feature-engineering themes: formation, platform time, downloads, and content views, measured with overall and within-month Spearman correlations.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, enrich mart_teacher_month_persona_ready with cumulative formation features from fct_formation_clean, then compute overall and within-month Spearman correlations for the requested feature pairs.",
    )


def build_behavior_segment_review(
    tables: Dict[str, pd.DataFrame],
    population: pd.DataFrame,
) -> pd.DataFrame:
    interactions = tables["fct_interaction_clean"][
        [
            "teacher_unique_id",
            "interaction_month",
            "event_family",
            "event_action",
        ]
    ].copy()
    interactions["interaction_month"] = pd.to_datetime(interactions["interaction_month"], errors="coerce")
    interactions["event_family"] = normalize_text(interactions["event_family"])
    interactions["event_action"] = normalize_text(interactions["event_action"])
    monthly_counts = (
        interactions.groupby(["teacher_unique_id", "interaction_month"], as_index=False)
        .agg(
            interaction_rows=("teacher_unique_id", "size"),
            aula_rows=("event_family", lambda s: int((s == "aula").sum())),
            plano_rows=("event_family", lambda s: int((s == "plano").sum())),
            ia_rows=("event_family", lambda s: int((s == "ia").sum())),
            prova_rows=("event_family", lambda s: int((s == "prova").sum())),
            view_rows=("event_action", lambda s: int((s == "view").sum())),
            download_rows=("event_action", lambda s: int((s == "download").sum())),
            navigation_rows=("event_action", lambda s: int((s == "navigation").sum())),
            other_rows=("event_action", lambda s: int((~s.isin(["view", "download", "navigation", "missing"])).sum())),
        )
        .rename(columns={"interaction_month": "month"})
    )

    work = population.copy()
    work = work.merge(monthly_counts, on=["teacher_unique_id", "month"], how="left")
    fill_zero_cols = [
        "interaction_rows",
        "aula_rows",
        "plano_rows",
        "ia_rows",
        "prova_rows",
        "view_rows",
        "download_rows",
        "navigation_rows",
        "other_rows",
    ]
    for col in fill_zero_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
    base = work["interaction_rows"].where(work["interaction_rows"] > 0, 1.0)
    work["aula_share"] = work["aula_rows"] / base
    work["plano_share"] = work["plano_rows"] / base
    work["ia_share"] = work["ia_rows"] / base
    work["prova_share"] = work["prova_rows"] / base
    work["view_share"] = work["view_rows"] / base
    work["download_share"] = work["download_rows"] / base
    work["navigation_share"] = work["navigation_rows"] / base
    work["formation_segment"] = np.where(work["had_formation_to_month"].fillna(0) > 0, "formation_history", "no_formation_history")
    work["download_segment"] = np.where(work["download_count_month"].fillna(0) > 0, "download_month", "no_download_month")
    work["navigation_without_activity_flag"] = (
        (work["interaction_rows"] == 0)
        | (work["clean_entry_exposed_no_activity_no_download_flag"].fillna(0) == 1)
        | ((work["navigation_share"] >= 0.50) & (work["download_share"] == 0) & (work["other_rows"] == 0))
    ).astype(int)

    review = (
        work.groupby(["formation_segment", "download_segment"], as_index=False)
        .agg(
            rows=("teacher_unique_id", "size"),
            return_rate_m1=("target_return_active_m1", "mean"),
            avg_session_minutes=("clean_entry_total_session_minutes_month", "mean"),
            avg_active_days=("active_days_month", "mean"),
            avg_content_views=("content_views_month", "mean"),
            avg_downloads=("download_count_month", "mean"),
            avg_other_activity=("other_activity_non_download_events_month", "mean"),
            zero_interaction_share=("interaction_rows", lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0) == 0).mean())),
            navigation_without_activity_share=("navigation_without_activity_flag", "mean"),
            avg_aula_share=("aula_share", "mean"),
            avg_plano_share=("plano_share", "mean"),
            avg_ia_share=("ia_share", "mean"),
            avg_view_share=("view_share", "mean"),
            avg_download_share=("download_share", "mean"),
            avg_navigation_share=("navigation_share", "mean"),
        )
        .sort_values("rows", ascending=False)
        .reset_index(drop=True)
    )
    return attach_reference(
        review,
        source_tables=["mart_teacher_month_persona_ready", "fct_interaction_clean", "fct_session_clean", "fct_formation_clean"],
        build_summary="Behavior review showing what users appear to do in month t, using interaction shares and session proxies, split by formation-history and download behavior.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, enrich the modeling population with cumulative formation features, aggregate fct_interaction_clean to teacher-month event-family and event-action shares, then summarize session depth, navigation-without-activity, and return rates by formation and download segments.",
    )


def build_temporal_cv_splits(months: Sequence[pd.Timestamp], max_folds: int = 3) -> List[Tuple[List[pd.Timestamp], List[pd.Timestamp], int]]:
    ordered_months = sorted(pd.to_datetime(pd.Series(months).dropna().unique()).tolist())
    if len(ordered_months) < 3:
        return []
    split_points = list(range(2, len(ordered_months)))
    split_points = split_points[-max_folds:]
    folds: List[Tuple[List[pd.Timestamp], List[pd.Timestamp], int]] = []
    for fold_id, split_idx in enumerate(split_points, start=1):
        folds.append((ordered_months[:split_idx], [ordered_months[split_idx]], fold_id))
    return folds


def efron_pseudo_r2(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y = np.asarray(pd.Series(y_true).astype(float))
    p = np.asarray(pd.Series(y_score).astype(float))
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 0:
        return float("nan")
    return float(1 - (np.sum((y - p) ** 2) / denom))


def monthly_rate_mape(months: Sequence[pd.Timestamp], y_true: Sequence[int], y_score: Sequence[float]) -> float:
    frame = pd.DataFrame(
        {
            "month": pd.to_datetime(months),
            "y_true": pd.Series(y_true).astype(float),
            "y_score": pd.Series(y_score).astype(float),
        }
    )
    grouped = frame.groupby("month", as_index=False).agg(actual_rate=("y_true", "mean"), predicted_rate=("y_score", "mean"))
    valid = grouped["actual_rate"] > 0
    if not valid.any():
        return float("nan")
    return float((np.abs(grouped.loc[valid, "predicted_rate"] - grouped.loc[valid, "actual_rate"]) / grouped.loc[valid, "actual_rate"]).mean())


def choose_threshold(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y = pd.Series(y_true).astype(int)
    score = pd.Series(y_score).astype(float)
    candidate_thresholds = np.linspace(0.10, 0.90, 33)
    best_threshold = 0.50
    best_f1 = float("-inf")
    for threshold in candidate_thresholds:
        preds = (score >= threshold).astype(int)
        metric = f1_score(y, preds, zero_division=0)
        if metric > best_f1:
            best_f1 = metric
            best_threshold = float(threshold)
    return best_threshold


def classification_metrics_bundle(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float,
    months: Sequence[pd.Timestamp],
    prefix: str,
) -> Dict[str, Any]:
    y = pd.Series(y_true).astype(int)
    score = pd.Series(y_score).astype(float)
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    return {
        f"{prefix}roc_auc": safe_auc(y, score),
        f"{prefix}average_precision": safe_average_precision(y, score),
        f"{prefix}brier_score": safe_brier(y, score),
        f"{prefix}log_loss": safe_log_loss(y, score),
        f"{prefix}efron_pseudo_r2": efron_pseudo_r2(y, score),
        f"{prefix}monthly_rate_mape": monthly_rate_mape(months, y, score),
        f"{prefix}accuracy": float(accuracy_score(y, pred)),
        f"{prefix}balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        f"{prefix}precision": float(precision_score(y, pred, zero_division=0)),
        f"{prefix}recall": float(recall_score(y, pred, zero_division=0)),
        f"{prefix}specificity": specificity,
        f"{prefix}f1": float(f1_score(y, pred, zero_division=0)),
        f"{prefix}tp": int(tp),
        f"{prefix}fp": int(fp),
        f"{prefix}tn": int(tn),
        f"{prefix}fn": int(fn),
        f"{prefix}positive_rate": float(y.mean()),
        f"{prefix}top_decile_lift": top_decile_lift(y, score),
    }


def parameter_grid_for_model(model_name: str) -> List[Dict[str, Any]]:
    if model_name == "profile_only_logistic":
        return [{"C": c} for c in [0.5, 2.0]]
    if model_name == "behavior_plus_profile_logistic":
        return [{"C": c} for c in [0.5, 2.0]]
    if model_name == "behavior_plus_profile_random_forest":
        return [
            {"n_estimators": n_estimators, "min_samples_leaf": min_leaf, "max_depth": max_depth, "max_features": max_features}
            for n_estimators, min_leaf, max_depth, max_features in product(
                [200],
                [20, 40],
                [None],
                ["sqrt"],
            )
        ]
    if model_name == "behavior_plus_profile_hist_gradient_boosting":
        return [
            {"learning_rate": learning_rate, "max_leaf_nodes": max_leaf_nodes, "min_samples_leaf": min_leaf, "max_depth": max_depth}
            for learning_rate, max_leaf_nodes, min_leaf, max_depth in product(
                [0.05, 0.10],
                [31],
                [20],
                [None],
            )
        ]
    return [{}]


def fit_pipeline_with_params(
    estimator: Any,
    params: Dict[str, Any],
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> Pipeline:
    model = clone(estimator).set_params(**params)
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(list(numeric_cols), list(categorical_cols))),
            ("model", model),
        ]
    )


def tune_model_with_temporal_cv(
    train: pd.DataFrame,
    target_col: str,
    model_name: str,
    estimator: Any,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> Tuple[Pipeline, Dict[str, Any], float, pd.DataFrame]:
    y_train = train[target_col].astype(int)
    feature_cols = list(numeric_cols) + list(categorical_cols)
    if not feature_cols:
        raise ValueError(f"No feature columns available for {model_name}")

    x_train = train[feature_cols].copy()
    folds = build_temporal_cv_splits(train["month"].tolist())
    grid = parameter_grid_for_model(model_name)
    cv_rows: List[Dict[str, Any]] = []
    best_score = float("-inf")
    best_params = grid[0]

    if not folds:
        best_pipeline = fit_pipeline_with_params(estimator, best_params, numeric_cols, categorical_cols)
        best_pipeline.fit(x_train, y_train)
        train_score = best_pipeline.predict_proba(x_train)[:, 1]
        threshold = choose_threshold(y_train, train_score)
        cv_df = pd.DataFrame(
            [
                {
                    "model_name": model_name,
                    "fold_id": 0,
                    "params_json": json.dumps(best_params, sort_keys=True),
                    "validation_month_start": "",
                    "validation_month_end": "",
                    "fold_roc_auc": float("nan"),
                }
            ]
        )
        return best_pipeline, best_params, threshold, cv_df

    for params in grid:
        fold_scores: List[float] = []
        for train_months, valid_months, fold_id in folds:
            fold_train = train[train["month"].isin(train_months)].copy()
            fold_valid = train[train["month"].isin(valid_months)].copy()
            y_fold_train = fold_train[target_col].astype(int)
            y_fold_valid = fold_valid[target_col].astype(int)
            if y_fold_train.nunique() < 2 or y_fold_valid.nunique() < 2:
                continue
            pipeline = fit_pipeline_with_params(estimator, params, numeric_cols, categorical_cols)
            pipeline.fit(fold_train[feature_cols], y_fold_train)
            score = pipeline.predict_proba(fold_valid[feature_cols])[:, 1]
            fold_auc = safe_auc(y_fold_valid, score)
            fold_scores.append(fold_auc)
            cv_rows.append(
                {
                    "model_name": model_name,
                    "fold_id": fold_id,
                    "params_json": json.dumps(params, sort_keys=True),
                    "validation_month_start": str(min(valid_months)),
                    "validation_month_end": str(max(valid_months)),
                    "fold_roc_auc": fold_auc,
                }
            )
        mean_score = float(np.nanmean(fold_scores)) if fold_scores else float("nan")
        if pd.notna(mean_score) and mean_score > best_score:
            best_score = mean_score
            best_params = params

    best_pipeline = fit_pipeline_with_params(estimator, best_params, numeric_cols, categorical_cols)
    best_pipeline.fit(x_train, y_train)
    train_score = best_pipeline.predict_proba(x_train)[:, 1]
    threshold = choose_threshold(y_train, train_score)
    cv_df = pd.DataFrame(cv_rows)
    if not cv_df.empty:
        cv_df["mean_cv_roc_auc"] = cv_df.groupby("params_json")["fold_roc_auc"].transform("mean")
        cv_df["selected_params_flag"] = (cv_df["params_json"] == json.dumps(best_params, sort_keys=True)).astype(int)
    return best_pipeline, best_params, threshold, cv_df


def bootstrap_metric_ci(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float,
    months: Sequence[pd.Timestamp],
    metric_name: str,
    n_boot: int = 100,
    seed: int = 42,
) -> Tuple[float, float]:
    y = np.asarray(pd.Series(y_true).astype(int))
    score = np.asarray(pd.Series(y_score).astype(float))
    months_arr = np.asarray(pd.to_datetime(pd.Series(months)))
    rng = np.random.default_rng(seed)
    values: List[float] = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_sample = y[idx]
        score_sample = score[idx]
        month_sample = months_arr[idx]
        if len(np.unique(y_sample)) < 2:
            continue
        metrics = classification_metrics_bundle(y_sample, score_sample, threshold, month_sample, prefix="")
        value = metrics.get(metric_name)
        if pd.notna(value):
            values.append(float(value))
    if not values:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def fit_models(
    population: pd.DataFrame,
    feature_candidates: pd.DataFrame,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    control_validity: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparison_rows: List[Dict[str, Any]] = []
    selection_rows: List[Dict[str, Any]] = []
    importance_rows: List[Dict[str, Any]] = []
    cv_rows: List[Dict[str, Any]] = []
    roc_rows: List[Dict[str, Any]] = []
    recent_strategy_rows: List[Dict[str, Any]] = []
    bootstrap_rows: List[Dict[str, Any]] = []
    valid_control = supported_control_windows(control_validity)
    feature_pool = eligible_numeric_features(feature_candidates, population)

    model_specs = [
        {
            "model_name": "dummy_baseline",
            "feature_spec": "baseline",
            "numeric_features": [],
            "categorical_features": [],
            "estimator": DummyClassifier(strategy="prior"),
        },
        {
            "model_name": "profile_only_logistic",
            "feature_spec": "profile_only",
            "numeric_features": [],
            "categorical_features": [PROFILE_CONTROL_VAR],
            "estimator": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        },
        {
            "model_name": "behavior_plus_profile_logistic",
            "feature_spec": "behavior_plus_profile",
            "numeric_features": list(numeric_features),
            "categorical_features": list(categorical_features),
            "estimator": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        },
        {
            "model_name": "behavior_plus_profile_random_forest",
            "feature_spec": "behavior_plus_profile",
            "numeric_features": list(feature_pool),
            "categorical_features": list(categorical_features),
            "estimator": RandomForestClassifier(
                n_estimators=200,
                max_features="sqrt",
                min_samples_leaf=40,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=42,
            ),
        },
        {
            "model_name": "behavior_plus_profile_hist_gradient_boosting",
            "feature_spec": "behavior_plus_profile",
            "numeric_features": list(feature_pool),
            "categorical_features": list(categorical_features),
            "estimator": HistGradientBoostingClassifier(
                learning_rate=0.10,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                max_depth=None,
                random_state=42,
            ),
        },
    ]

    for target_col, target_label in [
        ("target_churn_m1", "abandonar_m1"),
        ("target_return_active_m1", "retornar_ativo_m1"),
    ]:
        for window_label in ANALYSIS_WINDOWS:
            window_data = window_subset(population, window_label)
            train, test, train_months, test_months = temporal_train_test_split(window_data)
            if train.empty or test.empty or train[target_col].nunique() < 2 or test[target_col].nunique() < 2:
                continue

            control_supported = window_label in valid_control
            best_auc = float("-inf")
            best_model_name = None
            best_pipeline = None
            best_x_test = None
            best_y_test = None
            best_test_score = None
            best_threshold = 0.50
            best_features: List[str] = []
            best_test_months: List[pd.Timestamp] = []

            for spec in model_specs:
                y_train = train[target_col].astype(int)
                y_test = test[target_col].astype(int)
                numeric_cols = list(spec["numeric_features"])
                categorical_cols = list(spec["categorical_features"])
                if not control_supported:
                    categorical_cols = [col for col in categorical_cols if col != PROFILE_CONTROL_VAR]
                if spec["model_name"] == "profile_only_logistic" and not categorical_cols:
                    continue

                if spec["model_name"] == "dummy_baseline":
                    estimator = spec["estimator"]
                    estimator.fit(np.zeros((len(train), 1)), y_train)
                    train_score = estimator.predict_proba(np.zeros((len(train), 1)))[:, 1]
                    test_score = estimator.predict_proba(np.zeros((len(test), 1)))[:, 1]
                    threshold = 0.50
                    params_json = "{}"
                    cv_df = pd.DataFrame()
                else:
                    feature_cols = numeric_cols + categorical_cols
                    x_train = train[feature_cols].copy()
                    x_test = test[feature_cols].copy()
                    pipeline, best_params, threshold, cv_df = tune_model_with_temporal_cv(
                        train=train,
                        target_col=target_col,
                        model_name=spec["model_name"],
                        estimator=spec["estimator"],
                        numeric_cols=numeric_cols,
                        categorical_cols=categorical_cols,
                    )
                    params_json = json.dumps(best_params, sort_keys=True)
                    train_score = pipeline.predict_proba(x_train)[:, 1]
                    test_score = pipeline.predict_proba(x_test)[:, 1]
                    roc_auc = safe_auc(y_test, test_score)
                    if not cv_df.empty:
                        cv_df["target"] = target_label
                        cv_df["window_label"] = window_label
                        cv_df["feature_spec"] = spec["feature_spec"]
                        cv_rows.extend(cv_df.to_dict(orient="records"))
                    if spec["feature_spec"] == "behavior_plus_profile" and pd.notna(roc_auc) and roc_auc > best_auc:
                        best_auc = roc_auc
                        best_model_name = spec["model_name"]
                        best_pipeline = pipeline
                        best_x_test = x_test
                        best_y_test = y_test
                        best_test_score = test_score
                        best_threshold = threshold
                        best_features = feature_cols
                        best_test_months = test["month"].tolist()

                train_metrics = classification_metrics_bundle(y_train, train_score, threshold, train["month"], prefix="train_")
                test_metrics = classification_metrics_bundle(y_test, test_score, threshold, test["month"], prefix="test_")
                comparison_rows.append(
                    {
                        "target": target_label,
                        "window_label": window_label,
                        "model_name": spec["model_name"],
                        "feature_spec": spec["feature_spec"],
                        "control_variable": PROFILE_CONTROL_VAR,
                        "control_variable_supported": int(control_supported),
                        "control_variable_applied": int(control_supported and PROFILE_CONTROL_VAR in categorical_cols),
                        "train_rows": int(len(train)),
                        "test_rows": int(len(test)),
                        "train_month_start": str(min(train_months)),
                        "train_month_end": str(max(train_months)),
                        "test_month_start": str(min(test_months)),
                        "test_month_end": str(max(test_months)),
                        "selected_threshold": threshold,
                        "params_json": params_json,
                        "feature_count": int((len(numeric_cols) + len(categorical_cols))),
                        **train_metrics,
                        **test_metrics,
                        "roc_auc_gap_train_minus_test": (
                            train_metrics["train_roc_auc"] - test_metrics["test_roc_auc"]
                            if pd.notna(train_metrics["train_roc_auc"]) and pd.notna(test_metrics["test_roc_auc"])
                            else float("nan")
                        ),
                        "f1_gap_train_minus_test": (
                            train_metrics["train_f1"] - test_metrics["test_f1"]
                            if pd.notna(train_metrics["train_f1"]) and pd.notna(test_metrics["test_f1"])
                            else float("nan")
                        ),
                        "accuracy_gap_train_minus_test": (
                            train_metrics["train_accuracy"] - test_metrics["test_accuracy"]
                            if pd.notna(train_metrics["train_accuracy"]) and pd.notna(test_metrics["test_accuracy"])
                            else float("nan")
                        ),
                    }
                )

            if best_pipeline is not None and best_x_test is not None and best_y_test is not None and best_test_score is not None:
                selection_rows.append(
                    {
                        "target": target_label,
                        "window_label": window_label,
                        "selected_model_name": best_model_name,
                        "selection_metric": "roc_auc",
                        "selected_model_roc_auc": best_auc,
                        "selected_threshold": best_threshold,
                    }
                )
                if len(best_x_test) > 10000:
                    sampled_index = pd.Series(best_x_test.index).sample(n=10000, random_state=42).tolist()
                    importance_x = best_x_test.loc[sampled_index, best_features]
                    importance_y = best_y_test.loc[sampled_index]
                else:
                    importance_x = best_x_test[best_features]
                    importance_y = best_y_test
                importance = permutation_importance(
                    best_pipeline,
                    importance_x,
                    importance_y,
                    n_repeats=2,
                    random_state=42,
                    n_jobs=1,
                    scoring="roc_auc",
                )
                order = np.argsort(importance.importances_mean)[::-1]
                for index in order[:15]:
                    importance_rows.append(
                        {
                            "target": target_label,
                            "window_label": window_label,
                            "model_name": best_model_name,
                            "feature_name": best_features[index],
                            "permutation_importance_mean": float(importance.importances_mean[index]),
                            "permutation_importance_std": float(importance.importances_std[index]),
                        }
                    )
                fpr, tpr, roc_thresholds = roc_curve(best_y_test, best_test_score)
                for curve_fpr, curve_tpr, curve_threshold in zip(fpr, tpr, roc_thresholds):
                    roc_rows.append(
                        {
                            "target": target_label,
                            "window_label": window_label,
                            "model_name": best_model_name,
                            "fpr": float(curve_fpr),
                            "tpr": float(curve_tpr),
                            "threshold": float(curve_threshold) if np.isfinite(curve_threshold) else float("nan"),
                        }
                    )
                for metric_name in ["roc_auc", "f1", "accuracy"]:
                    ci_low, ci_high = bootstrap_metric_ci(
                        best_y_test,
                        best_test_score,
                        best_threshold,
                        best_test_months,
                        metric_name=metric_name,
                    )
                    bootstrap_rows.append(
                        {
                            "target": target_label,
                            "window_label": window_label,
                            "model_name": best_model_name,
                            "metric_name": f"test_{metric_name}",
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                        }
                    )

    all_months = sorted(population["month"].dropna().unique().tolist())
    for strategy_window in [window for window in [PRIMARY_RECENT_WINDOW, "recent_6m"] if window in ANALYSIS_WINDOWS]:
        recent_months = sorted(window_subset(population, strategy_window)["month"].dropna().unique().tolist())
        if len(recent_months) < 6:
            continue
        split_idx = max(1, min(len(recent_months) - 1, int(np.floor(len(recent_months) * 0.67))))
        recent_train_months = recent_months[:split_idx]
        recent_test_months = recent_months[split_idx:]
        full_train_months = [month for month in all_months if month < min(recent_test_months)]
        recent_best_model_by_target = {
            row["target"]: row["selected_model_name"]
            for row in selection_rows
            if row["window_label"] == strategy_window
        }
        for target_col, target_label in [
            ("target_churn_m1", "abandonar_m1"),
            ("target_return_active_m1", "retornar_ativo_m1"),
        ]:
            same_test = population[population["month"].isin(recent_test_months)].copy()
            if same_test.empty or same_test[target_col].nunique() < 2:
                continue
            selected_model_name = recent_best_model_by_target.get(target_label)
            strategy_spec = next((spec for spec in model_specs if spec["model_name"] == selected_model_name), None)
            if strategy_spec is None:
                continue
            for strategy_label, train_months in [
                ("include_pre_drift_history", full_train_months),
                ("recent_only_post_drift", recent_train_months),
            ]:
                strategy_train = population[population["month"].isin(train_months)].copy()
                if strategy_train.empty or strategy_train[target_col].nunique() < 2:
                    continue
                control_supported = strategy_window in valid_control if strategy_label == "recent_only_post_drift" else "all_history" in valid_control
                numeric_cols = list(strategy_spec["numeric_features"])
                categorical_cols = list(strategy_spec["categorical_features"])
                if not control_supported:
                    categorical_cols = [col for col in categorical_cols if col != PROFILE_CONTROL_VAR]
                if strategy_spec["model_name"] == "profile_only_logistic" and not categorical_cols:
                    continue
                pipeline, best_params, threshold, _ = tune_model_with_temporal_cv(
                    train=strategy_train,
                    target_col=target_col,
                    model_name=strategy_spec["model_name"],
                    estimator=strategy_spec["estimator"],
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                )
                feature_cols = numeric_cols + categorical_cols
                y_test = same_test[target_col].astype(int)
                test_score = pipeline.predict_proba(same_test[feature_cols])[:, 1]
                test_metrics = classification_metrics_bundle(y_test, test_score, threshold, same_test["month"], prefix="")
                recent_strategy_rows.append(
                    {
                        "window_label": strategy_window,
                        "target": target_label,
                        "strategy_label": strategy_label,
                        "model_name": strategy_spec["model_name"],
                        "train_rows": int(len(strategy_train)),
                        "test_rows": int(len(same_test)),
                        "train_month_start": str(strategy_train["month"].min()),
                        "train_month_end": str(strategy_train["month"].max()),
                        "test_month_start": str(min(recent_test_months)),
                        "test_month_end": str(max(recent_test_months)),
                        "params_json": json.dumps(best_params, sort_keys=True),
                        "selected_threshold": threshold,
                        **test_metrics,
                    }
                )
                recent_strategy_rows.append(
                    {
                        "window_label": strategy_window,
                        "target": target_label,
                        "strategy_label": strategy_label,
                        "model_name": "__best_strategy__",
                        "best_strategy_model_name": strategy_spec["model_name"],
                        "best_strategy_roc_auc": float(test_metrics["roc_auc"]) if pd.notna(test_metrics["roc_auc"]) else float("nan"),
                    }
                )

    comparison_df = pd.DataFrame(comparison_rows).sort_values(["target", "window_label", "test_roc_auc"], ascending=[True, True, False]).reset_index(drop=True)
    comparison_df = attach_reference(
        comparison_df,
        source_tables=["mart_teacher_month_persona_ready", "dim_teacher", "audit_persona_feature_readiness"],
        build_summary="Temporal model comparison across all history and recent 6 months using baseline, logistic, random-forest, and histogram-gradient-boosting models with train/test diagnostics and richer metrics.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, build the modeling population from mart_teacher_month_persona_ready, enrich it with time-safe formation and engineered features, keep only eligible month-t predictors, split train/test by time, tune the models with temporal cross-validation, and compute train/test metrics on the held-out months.",
    )

    selection_df = pd.DataFrame(selection_rows).sort_values(["target", "window_label"]).reset_index(drop=True)
    selection_df = attach_reference(
        selection_df,
        source_tables=["analytics_prediction_model_comparison_explainable_v2"],
        build_summary="Selected best behavior-plus-profile model for each target and time window, based on held-out ROC AUC.",
        rebuild_from_raw="Rerun etapa_12b_prediction_explainable_v2.py after raw_para_base_modelada_v4.py; this table is derived from the model-comparison output generated in the same run.",
    )

    importance_df = pd.DataFrame(importance_rows).sort_values(["target", "window_label", "permutation_importance_mean"], ascending=[True, True, False]).reset_index(drop=True)
    importance_df = attach_reference(
        importance_df,
        source_tables=["mart_teacher_month_persona_ready", "dim_teacher", "audit_persona_feature_readiness"],
        build_summary="Permutation importance on the held-out test months for the best behavior-plus-profile model in each target/window pair.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, fit the selected best behavior-plus-profile model on the temporal train split, then compute permutation importance on the held-out test months.",
    )

    cv_df = pd.DataFrame(cv_rows).sort_values(["target", "window_label", "model_name", "selected_params_flag", "mean_cv_roc_auc"], ascending=[True, True, True, False, False]).reset_index(drop=True)
    cv_df = attach_reference(
        cv_df,
        source_tables=["mart_teacher_month_persona_ready", "analytics_prediction_feature_candidates_v2"],
        build_summary="Temporal cross-validation grid-search results used to tune the predictive models before the final train/test evaluation.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, prepare the month-t modeling population, define the eligible feature pool, create expanding temporal validation folds inside the train window, and evaluate the parameter grid for each model.",
    )

    roc_df = pd.DataFrame(roc_rows).sort_values(["target", "window_label", "model_name", "fpr"]).reset_index(drop=True)
    roc_df = attach_reference(
        roc_df,
        source_tables=["analytics_prediction_model_selection_v2"],
        build_summary="ROC curve points for the selected best model in each target/window pair.",
        rebuild_from_raw="Rerun etapa_12b_prediction_explainable_v2.py after raw_para_base_modelada_v4.py; this table is derived from the held-out scores of the selected best models.",
    )

    recent_strategy_df = pd.DataFrame(recent_strategy_rows)
    if not recent_strategy_df.empty:
        recent_strategy_df = recent_strategy_df.sort_values(["window_label", "target", "strategy_label", "model_name"], ascending=[True, True, True, True]).reset_index(drop=True)
    recent_strategy_df = attach_reference(
        recent_strategy_df,
        source_tables=["mart_teacher_month_persona_ready", "analytics_prediction_model_comparison_explainable_v2"],
        build_summary="Same-holdout comparison of two training strategies: using all pre-test history versus using only the recent post-drift months before the same recent holdout, for both the primary recent_12m window and the thinner recent_6m sensitivity window.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, prepare the month-t modeling population, define the recent window, hold out the last third of that window as a common test set, then compare models trained on all prior history versus only the recent pre-test months.",
    )

    bootstrap_df = pd.DataFrame(bootstrap_rows).sort_values(["target", "window_label", "metric_name"]).reset_index(drop=True)
    bootstrap_df = attach_reference(
        bootstrap_df,
        source_tables=["analytics_prediction_model_selection_v2"],
        build_summary="Bootstrap confidence intervals for the held-out performance metrics of the selected best models. Bootstrap is used for uncertainty, not as a predictive model.",
        rebuild_from_raw="Rerun etapa_12b_prediction_explainable_v2.py after raw_para_base_modelada_v4.py; this table is derived by bootstrap-resampling the held-out predictions of the selected best models.",
    )

    return comparison_df, selection_df, importance_df, cv_df, roc_df, recent_strategy_df, bootstrap_df


def prepare_population_variant(
    tables: Dict[str, pd.DataFrame],
    variant_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    month = tables["mart_teacher_month_persona_ready"].copy()
    month["month"] = pd.to_datetime(month["month"], errors="coerce")
    numeric_exceptions = {
        "teacher_unique_id",
        "month",
        "month_signal_class",
        "teacher_population_status",
        "teacher_estado",
        "teacher_currentsubject_group",
        "teacher_currentstage",
        "teacher_utm_group",
    }
    for col in [col for col in month.columns if col not in numeric_exceptions]:
        month[col] = pd.to_numeric(month[col], errors="coerce")
    for col in [
        "month_signal_class",
        "teacher_population_status",
        "teacher_estado",
        "teacher_currentsubject_group",
        "teacher_currentstage",
        "teacher_utm_group",
    ]:
        month[col] = normalize_text(month[col])

    if variant_name == "current_strict":
        population = month[
            (month["observed_month_flag"].fillna(0) == 1)
            & (month["persona_analysis_eligible_flag"].fillna(0) == 1)
            & (month["next_month_observed_flag"].fillna(0) == 1)
        ].copy()
    elif variant_name == "relaxed_observed_next":
        population = month[
            (month["observed_month_flag"].fillna(0) == 1)
            & (month["next_month_observed_flag"].fillna(0) == 1)
        ].copy()
    else:
        raise ValueError(f"Unsupported population variant: {variant_name}")

    population["target_return_active_m1"] = (population["returned_active_m1"].fillna(0) == 1).astype(int)
    population["target_churn_m1"] = (population["returned_active_m1"].fillna(0) == 0).astype(int)
    population, supplemental_features = enrich_prediction_population(tables, population)
    feature_candidates, _, _, _ = build_feature_candidates(tables, population)
    feature_candidates, _, _ = extend_feature_candidates_with_supplemental_features(
        feature_candidates=feature_candidates,
        population=population,
        supplemental_features=supplemental_features,
    )
    return population, feature_candidates


def review_model_specs() -> List[Dict[str, Any]]:
    return [
        {
            "model_name": "behavior_plus_profile_logistic",
            "numeric_features_mode": "all_eligible",
            "categorical_features": [PROFILE_CONTROL_VAR, "month_signal_class"],
            "estimator": LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced", random_state=42),
        },
        {
            "model_name": "behavior_plus_profile_random_forest",
            "numeric_features_mode": "all_eligible",
            "categorical_features": [PROFILE_CONTROL_VAR, "month_signal_class"],
            "estimator": RandomForestClassifier(
                n_estimators=200,
                max_features="sqrt",
                min_samples_leaf=40,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=42,
            ),
        },
        {
            "model_name": "behavior_plus_profile_hist_gradient_boosting",
            "numeric_features_mode": "all_eligible",
            "categorical_features": [PROFILE_CONTROL_VAR, "month_signal_class"],
            "estimator": HistGradientBoostingClassifier(
                learning_rate=0.10,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                max_depth=None,
                random_state=42,
            ),
        },
    ]


def evaluate_fixed_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    estimator: Any,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> Dict[str, Any]:
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(list(numeric_cols), list(categorical_cols))),
            ("model", clone(estimator)),
        ]
    )
    feature_cols = list(numeric_cols) + list(categorical_cols)
    x_train = train[feature_cols].copy()
    x_test = test[feature_cols].copy()
    y_train = train[target_col].astype(int)
    y_test = test[target_col].astype(int)
    pipeline.fit(x_train, y_train)
    train_score = pipeline.predict_proba(x_train)[:, 1]
    test_score = pipeline.predict_proba(x_test)[:, 1]
    threshold = choose_threshold(y_train, train_score)
    return {
        "selected_threshold": threshold,
        **classification_metrics_bundle(y_train, train_score, threshold, train["month"], prefix="train_"),
        **classification_metrics_bundle(y_test, test_score, threshold, test["month"], prefix="test_"),
    }


def build_population_strategy_review(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for population_variant in ["current_strict", "relaxed_observed_next"]:
        population, feature_candidates = prepare_population_variant(tables, population_variant)
        feature_pool = eligible_numeric_features(feature_candidates, population)
        zero_interaction_share = float((population["zero_interaction_month_flag"].fillna(0) == 1).mean())
        for window_label in [PRIMARY_RECENT_WINDOW, "recent_6m"]:
            subset = window_subset(population, window_label)
            train, test, train_months, test_months = temporal_train_test_split(subset)
            if train.empty or test.empty or train["target_return_active_m1"].nunique() < 2 or test["target_return_active_m1"].nunique() < 2:
                continue
            for spec in review_model_specs():
                metrics = evaluate_fixed_model(
                    train=train,
                    test=test,
                    target_col="target_return_active_m1",
                    estimator=spec["estimator"],
                    numeric_cols=feature_pool,
                    categorical_cols=[col for col in spec["categorical_features"] if col in population.columns],
                )
                rows.append(
                    {
                        "population_variant": population_variant,
                        "window_label": window_label,
                        "model_name": spec["model_name"],
                        "population_rows": int(len(population)),
                        "population_teachers": int(population["teacher_unique_id"].nunique()),
                        "population_return_rate_m1": float(population["target_return_active_m1"].mean()),
                        "population_zero_interaction_share": zero_interaction_share,
                        "train_rows": int(len(train)),
                        "test_rows": int(len(test)),
                        "train_month_start": str(min(train_months)),
                        "train_month_end": str(max(train_months)),
                        "test_month_start": str(min(test_months)),
                        "test_month_end": str(max(test_months)),
                        **metrics,
                    }
                )
    df = pd.DataFrame(rows).sort_values(["window_label", "population_variant", "test_roc_auc"], ascending=[True, True, False]).reset_index(drop=True)
    return attach_reference(
        df,
        source_tables=["mart_teacher_month_persona_ready", "fct_interaction_clean", "fct_formation_clean"],
        build_summary="Comparison of stricter versus more relaxed population filters, using the same recent windows and the same fixed model classes to show the sample-size versus noise tradeoff.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, rebuild the prediction-enriched population variants from mart_teacher_month_persona_ready plus fct_interaction_clean and fct_formation_clean, then evaluate the same fixed model classes on each recent window.",
    )


def build_feature_set_review(
    population: pd.DataFrame,
    model_comparison: pd.DataFrame,
) -> pd.DataFrame:
    comparison = strip_reference_cols(model_comparison).copy()
    full_feature_candidates = [
        "lifetime_clean_entry_minutes_total",
        "lifetime_active_months",
        "active_streak_current_months",
        "clean_entry_avg_session_minutes_month",
        "content_views_month",
        "download_count_month",
        "interaction_rows_month",
        "navigation_without_activity_flag",
        "had_formation_to_month",
        "view_share_month",
        "download_share_month",
        "navigation_share_month",
        "other_action_share_month",
        "session_minutes_per_active_day_month",
        "downloads_per_view_month",
        "content_views_per_session_month",
    ]
    compact_feature_candidates = [
        "lifetime_clean_entry_minutes_total",
        "lifetime_active_months",
        "active_streak_current_months",
        "clean_entry_avg_session_minutes_month",
        "content_views_month",
        "download_count_month",
        "interaction_rows_month",
        "navigation_without_activity_flag",
        "had_formation_to_month",
    ]
    rows: List[Dict[str, Any]] = []
    for window_label in [PRIMARY_RECENT_WINDOW, "recent_6m"]:
        subset = window_subset(population, window_label)
        train, test, train_months, test_months = temporal_train_test_split(subset)
        if train.empty or test.empty or train["target_return_active_m1"].nunique() < 2 or test["target_return_active_m1"].nunique() < 2:
            continue
        for spec in review_model_specs():
            for feature_set_label, feature_candidates in [
                ("compact_scorecard", compact_feature_candidates),
                ("full_behavior_plus_profile", full_feature_candidates),
            ]:
                numeric_cols = [feature for feature in feature_candidates if feature in population.columns]
                categorical_cols = [col for col in spec["categorical_features"] if col in population.columns]
                metrics = evaluate_fixed_model(
                    train=train,
                    test=test,
                    target_col="target_return_active_m1",
                    estimator=spec["estimator"],
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                )
                rows.append(
                    {
                        "window_label": window_label,
                        "model_name": spec["model_name"],
                        "feature_set_label": feature_set_label,
                        "feature_count": int(len(numeric_cols) + len(categorical_cols)),
                        "train_rows": int(len(train)),
                        "test_rows": int(len(test)),
                        "test_month_start": str(min(test_months)),
                        "test_month_end": str(max(test_months)),
                        **metrics,
                    }
                )

    df = pd.DataFrame(rows)
    if not df.empty:
        full = df[df["feature_set_label"] == "full_behavior_plus_profile"][
            ["window_label", "model_name", "test_roc_auc", "test_f1", "test_accuracy"]
        ].rename(
            columns={
                "test_roc_auc": "full_test_roc_auc",
                "test_f1": "full_test_f1",
                "test_accuracy": "full_test_accuracy",
            }
        )
        df = df.merge(full, on=["window_label", "model_name"], how="left")
        df["roc_auc_gap_vs_full"] = df["test_roc_auc"] - df["full_test_roc_auc"]
        df["f1_gap_vs_full"] = df["test_f1"] - df["full_test_f1"]
        df["accuracy_gap_vs_full"] = df["test_accuracy"] - df["full_test_accuracy"]
        df = df.sort_values(["window_label", "model_name", "feature_set_label"]).reset_index(drop=True)
    return attach_reference(
        df,
        source_tables=["mart_teacher_month_persona_ready", "fct_interaction_clean", "fct_formation_clean"],
        build_summary="Comparison between a compact interpretable scorecard and a larger behavior-plus-profile feature set, used to test whether the top predictors are mostly redundant family signals.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, enrich the primary modeling population with behavior-share and formation features, then evaluate compact versus full feature sets on the recent windows with the same fixed model classes.",
    )


def build_key_findings(
    control_validity: pd.DataFrame,
    model_selection: pd.DataFrame,
    feature_importance: pd.DataFrame,
    model_comparison: pd.DataFrame,
    recent_strategy: pd.DataFrame,
    feature_relations: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for _, row in strip_reference_cols(control_validity).iterrows():
        rows.append(
            {
                "finding_group": "control_variable",
                "finding_label": f"{row['window_label']} | {row['target']}",
                "evidence_value": float(row["cramers_v"]) if pd.notna(row["cramers_v"]) else float("nan"),
                "evidence_unit": "cramers_v",
                "why_it_matters": "The control variable must be empirically usable before we rely on it in modeling.",
                "interpretation": f"{PROFILE_CONTROL_VAR} is `{row['plain_english_readout']}` for {row['target']} in {row['window_label']}.",
            }
        )

    comparison = strip_reference_cols(model_comparison)
    for _, row in strip_reference_cols(model_selection).iterrows():
        comp = comparison[
            (comparison["target"] == row["target"])
            & (comparison["window_label"] == row["window_label"])
            & (comparison["model_name"] == row["selected_model_name"])
        ]
        comp_row = comp.iloc[0] if not comp.empty else None
        rows.append(
            {
                "finding_group": "best_model",
                "finding_label": f"{row['window_label']} | {row['target']}",
                "evidence_value": float(row["selected_model_roc_auc"]),
                "evidence_unit": "roc_auc",
                "why_it_matters": "This is the model we should treat as the strongest predictive baseline for this target and window.",
                "interpretation": (
                    f"The selected model for {row['target']} in {row['window_label']} is {row['selected_model_name']} "
                    f"with ROC AUC {row['selected_model_roc_auc']:.4f}"
                    + (
                        f", F1 {comp_row['test_f1']:.4f}, and accuracy {comp_row['test_accuracy']:.4f} on the held-out months."
                        if comp_row is not None
                        else "."
                    )
                ),
            }
        )

    for _, row in strip_reference_cols(feature_importance).groupby(["target", "window_label"], as_index=False).head(3).iterrows():
        rows.append(
            {
                "finding_group": "top_feature",
                "finding_label": f"{row['window_label']} | {row['target']} | {row['feature_name']}",
                "evidence_value": float(row["permutation_importance_mean"]),
                "evidence_unit": "permutation_importance",
                "why_it_matters": "Top held-out features show which month-t signals are most useful for prediction.",
                "interpretation": f"{row['feature_name']} is one of the strongest held-out predictors for {row['target']} in {row['window_label']}.",
            }
        )

    recent = strip_reference_cols(recent_strategy)
    recent_best = recent[recent["model_name"] == "__best_strategy__"].copy()
    for _, row in recent_best.iterrows():
        rows.append(
            {
                "finding_group": "training_strategy",
                "finding_label": f"{row['window_label']} | {row['target']} | {row['strategy_label']}",
                "evidence_value": float(row["best_strategy_roc_auc"]) if pd.notna(row["best_strategy_roc_auc"]) else float("nan"),
                "evidence_unit": "roc_auc_same_recent_holdout",
                "why_it_matters": "This answers whether the current regime is better predicted by using all historical data or only the recent post-drift slice.",
                "interpretation": f"On the same {row['window_label']} holdout, `{row['strategy_label']}` reaches ROC AUC {row['best_strategy_roc_auc']:.4f} for {row['target']}.",
            }
        )

    relations = strip_reference_cols(feature_relations)
    for _, row in relations.head(4).iterrows():
        rows.append(
            {
                "finding_group": "feature_relation",
                "finding_label": row["relation_label"],
                "evidence_value": float(row["mean_within_month_spearman_rho"]) if pd.notna(row["mean_within_month_spearman_rho"]) else float("nan"),
                "evidence_unit": "within_month_spearman_rho",
                "why_it_matters": "Requested feature-engineering ideas should be tested critically, not accepted just because they sound plausible.",
                "interpretation": f"{row['x_feature']} versus {row['y_feature']}: {row['plain_english_readout']}.",
            }
        )

    df = pd.DataFrame(rows)
    return attach_reference(
        df,
        source_tables=[
            "analytics_prediction_control_variable_validity_v2",
            "analytics_prediction_model_selection_v2",
            "analytics_prediction_model_comparison_explainable_v2",
            "analytics_prediction_feature_importance_explainable_v2",
            "analytics_prediction_recent_strategy_comparison_v2",
            "analytics_prediction_feature_relation_review_v2",
        ],
        build_summary="Plain-English prediction findings table derived from control validity, model comparison, recent-strategy comparison, feature importance, and feature-relation testing.",
        rebuild_from_raw="Rerun etapa_12b_prediction_explainable_v2.py after raw_para_base_modelada_v4.py; this table is derived from the prediction outputs generated in the same run.",
    )


def build_summary_payload(
    population_summary: pd.DataFrame,
    control_validity: pd.DataFrame,
    model_selection: pd.DataFrame,
    model_comparison: pd.DataFrame,
    feature_importance: pd.DataFrame,
    recent_strategy: pd.DataFrame,
    data_sufficiency: pd.DataFrame,
) -> Dict[str, Any]:
    best_models = strip_reference_cols(model_selection).to_dict(orient="records")
    return {
        "population_summary": strip_reference_cols(population_summary).to_dict(orient="records"),
        "control_validity": strip_reference_cols(control_validity).to_dict(orient="records"),
        "best_models": best_models,
        "model_comparison_top": strip_reference_cols(model_comparison).head(12).to_dict(orient="records"),
        "top_feature_importance": strip_reference_cols(feature_importance).groupby(["target", "window_label"], as_index=False).head(6).to_dict(orient="records"),
        "recent_strategy": strip_reference_cols(recent_strategy).to_dict(orient="records"),
        "data_sufficiency": strip_reference_cols(data_sufficiency).to_dict(orient="records"),
    }


def write_summary_markdown(path: Path, cfg: Config, payload: Dict[str, Any]) -> None:
    population = payload["population_summary"][0] if payload["population_summary"] else {}
    lines = [
        "# Prediction Review v2",
        "",
        "## Paths",
        "",
        f"- Source DuckDB: `{cfg.source_duckdb_path}`",
        f"- Output directory: `{cfg.output_dir}`",
        "",
        "## Step By Step",
        "",
        "1. Start from the relevant-table layer exported by `raw_para_base_modelada_v4.py`.",
        "2. Freeze the modeling population in `mart_teacher_month_persona_ready`.",
        f"3. Validate `{PROFILE_CONTROL_VAR}` before using it as a control variable.",
        "4. Exclude leakage and keep only month-t features.",
        "5. Train temporal models on all history and on the recent 6 months.",
        "6. Tune the models with temporal cross-validation and compare them with richer train/test metrics.",
        "7. Compare same-holdout training strategies: include all history vs recent-only post-drift.",
        "8. Review whether formation, platform time, downloads, and content views are relevant and how they relate.",
        "",
        "## Population",
        "",
        f"- Rows: {population.get('rows', 'n/a')}",
        f"- Teachers: {population.get('teachers', 'n/a')}",
        "",
        "## Data Sufficiency",
    ]
    for row in payload["data_sufficiency"]:
        lines.append(
            f"- `{row['window_label']} | {row['target']}` | events_per_feature={row['events_per_feature']:.2f} | {row['plain_english_readout']}"
        )
    lines.extend(
        [
            "",
            "## Selected Models",
        ]
    )
    for row in payload["best_models"]:
        lines.append(
            f"- `{row['target']}` | `{row['window_label']}` | `{row['selected_model_name']}` | auc={row['selected_model_roc_auc']:.4f}"
        )
    recent_best = [row for row in payload["recent_strategy"] if row.get("model_name") == "__best_strategy__"]
    if recent_best:
        lines.extend(["", "## Same Recent Holdout Strategy Comparison"])
        for row in recent_best:
            lines.append(
                f"- `{row['window_label']} | {row['target']}` | `{row['strategy_label']}` | auc={row['best_strategy_roc_auc']:.4f}"
            )
    lines.extend(["", "## Top Held-Out Features"])
    for row in payload["top_feature_importance"][:10]:
        lines.append(
            f"- `{row['window_label']} | {row['target']} | {row['feature_name']}` | importance={row['permutation_importance_mean']:.4f}"
        )
    lines.extend(
        [
            "",
            "## Rebuild",
            "",
            "1. Run `raw_para_base_modelada_v4.py`.",
            "2. Run `etapa_12b_prediction_explainable_v2.py`.",
        ]
    )
    write_markdown(path, lines)


def main() -> None:
    args = parse_args()
    cfg = build_config(base_dir=args.base_dir, source_dir=args.source_dir, output_dir=args.output_dir)
    source_conn = connect_source(cfg)
    output_conn = connect_output(cfg)

    try:
        tables = load_public_tables(source_conn)
        direct_tables = [
            "audit_base_modelada_validation",
            "audit_persona_feature_readiness",
            "dim_persona_range_candidates",
            "dim_teacher",
            "fct_formation_clean",
            "fct_interaction_clean",
            "fct_session_clean",
            "mart_teacher_month_persona_ready",
        ]
        input_map = build_input_map(
            tables,
            direct_tables=direct_tables,
            flag_column="used_directly_for_prediction",
            analysis_summary="Inventory of the declared relevant tables and whether each one is used directly in the prediction analysis.",
        )
        population, population_summary = prepare_model_population(tables)
        population, supplemental_features = enrich_prediction_population(tables, population)
        assumptions = build_assumptions_table()
        control_validity, control_group_summary = build_control_variable_validity(population)
        feature_candidates, numeric_features, categorical_features, _ = build_feature_candidates(tables, population)
        feature_candidates, numeric_features, categorical_features = extend_feature_candidates_with_supplemental_features(
            feature_candidates=feature_candidates,
            population=population,
            supplemental_features=supplemental_features,
        )
        feature_screening = build_feature_screening(population, feature_candidates)
        data_sufficiency = build_data_sufficiency_review(population, feature_candidates)
        exclusion_bias = build_exclusion_bias_review(tables)
        feature_relations = build_feature_relation_review(population)
        behavior_segment_review = build_behavior_segment_review(tables, population)
        model_comparison, model_selection, feature_importance, cv_results, roc_curve_points, recent_strategy, bootstrap_ci = fit_models(
            population,
            feature_candidates=feature_candidates,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            control_validity=control_validity,
        )
        population_strategy_review = build_population_strategy_review(tables)
        feature_set_review = build_feature_set_review(population, model_comparison)
        feature_theme_review = build_feature_theme_review(feature_candidates, feature_screening, feature_importance)
        key_findings = build_key_findings(
            control_validity,
            model_selection,
            feature_importance,
            model_comparison,
            recent_strategy,
            feature_relations,
        )

        outputs = {
            "analytics_prediction_input_map_v2": input_map,
            "analytics_prediction_assumptions_v2": assumptions,
            "analytics_prediction_population_summary_v2": population_summary,
            "analytics_prediction_control_variable_validity_v2": control_validity,
            "analytics_prediction_control_group_summary_v2": control_group_summary,
            "analytics_prediction_feature_candidates_v2": feature_candidates,
            "analytics_prediction_feature_screening_v2": feature_screening,
            "analytics_prediction_data_sufficiency_v2": data_sufficiency,
            "analytics_prediction_exclusion_bias_v2": exclusion_bias,
            "analytics_prediction_feature_relation_review_v2": feature_relations,
            "analytics_prediction_behavior_segment_review_v2": behavior_segment_review,
            "analytics_prediction_population_strategy_review_v2": population_strategy_review,
            "analytics_prediction_feature_set_review_v2": feature_set_review,
            "analytics_prediction_model_comparison_explainable_v2": model_comparison,
            "analytics_prediction_model_selection_v2": model_selection,
            "analytics_prediction_cv_results_v2": cv_results,
            "analytics_prediction_roc_curve_v2": roc_curve_points,
            "analytics_prediction_recent_strategy_comparison_v2": recent_strategy,
            "analytics_prediction_bootstrap_ci_v2": bootstrap_ci,
            "analytics_prediction_feature_importance_explainable_v2": feature_importance,
            "analytics_prediction_feature_theme_review_v2": feature_theme_review,
            "analytics_prediction_key_findings_v2": key_findings,
        }
        output_reference = build_output_reference(
            cfg,
            outputs.keys(),
            build_summary="Manifest of all prediction outputs generated by etapa_12b_prediction_explainable_v2.py.",
        )
        outputs["analytics_prediction_output_reference_v2"] = output_reference

        for table_name, df in outputs.items():
            persist_table(output_conn, cfg, table_name, df)

        payload = build_summary_payload(
            population_summary=population_summary,
            control_validity=control_validity,
            model_selection=model_selection,
            model_comparison=model_comparison,
            feature_importance=feature_importance,
            recent_strategy=recent_strategy,
            data_sufficiency=data_sufficiency,
        )
        write_json(cfg.output_dir / "json" / "prediction_summary_v2.json", payload)
        write_summary_markdown(cfg.output_dir / "audit" / "prediction_summary_v2.md", cfg, payload)
    finally:
        source_conn.close()
        output_conn.close()


if __name__ == "__main__":
    main()
