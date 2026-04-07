"""Busca e comparação das definições oficiais de label."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from targeted_ml.modeling.splitters import ExpandingMonthSplit

from . import analysis_setup as setup
from .modeling import bootstrap_ci_width, bootstrap_prevalence_ci_width_from_counts, pareto_front
from .selection import rank_primary_definition_candidates

WEIGHT_GRID: tuple[tuple[float, float], ...] = (
    (0.25, 0.75),
    (0.50, 0.50),
    (0.75, 0.25),
)


def definition_fold_support_valid(rows: int, positives: int, negatives: int) -> int:
    return int(
        int(rows) >= setup.MIN_OFFICIAL_TEST_ROWS
        and int(positives) >= setup.MIN_OFFICIAL_TEST_POSITIVES
        and int(negatives) >= setup.MIN_OFFICIAL_TEST_NEGATIVES
    )


def definition_candidate_validity_payload(rows: int, positives: int, negatives: int) -> dict[str, Any]:
    technical_valid_flag = int(int(positives) > 0 and int(negatives) > 0)
    support_valid_flag = definition_fold_support_valid(rows=int(rows), positives=int(positives), negatives=int(negatives))
    candidate_valid_flag = int(technical_valid_flag == 1 and support_valid_flag == 1)
    if technical_valid_flag == 0:
        invalid_reason = "single_class_fold"
    elif support_valid_flag == 0:
        invalid_reason = "insufficient_definition_support"
    else:
        invalid_reason = ""
    return {
        "technical_candidate_valid_flag": technical_valid_flag,
        "support_valid_flag": support_valid_flag,
        "candidate_valid_flag": candidate_valid_flag,
        "invalid_reason": invalid_reason,
    }


def compute_candidate_diagnostics(frame: pd.DataFrame, label: np.ndarray) -> dict[str, Any]:
    working = frame.copy()
    working["_label"] = label.astype(int)
    diagnostics: dict[str, Any] = {
        "rows": int(len(working)),
        "positives": int(working["_label"].sum()),
        "negatives": int((1 - working["_label"]).sum()),
        "prevalence": float(working["_label"].mean()) if len(working) else float("nan"),
    }
    prevalence = diagnostics["prevalence"]
    if prevalence in {0.0, 1.0} or pd.isna(prevalence):
        diagnostics["prevalence_entropy"] = 0.0
    else:
        diagnostics["prevalence_entropy"] = float(
            -(prevalence * math.log(prevalence) + (1.0 - prevalence) * math.log(1.0 - prevalence))
        )
    monthly = working.groupby("first_month", dropna=False)["_label"].mean()
    diagnostics["monthly_prevalence_mean"] = float(monthly.mean()) if not monthly.empty else float("nan")
    diagnostics["monthly_prevalence_std"] = float(monthly.std(ddof=0)) if len(monthly) > 1 else 0.0
    positives = int(working["_label"].sum())
    low, high, width = bootstrap_prevalence_ci_width_from_counts(len(working), positives)
    diagnostics["bootstrap_prevalence_ci_low"] = low
    diagnostics["bootstrap_prevalence_ci_high"] = high
    diagnostics["bootstrap_prevalence_ci_width"] = width
    negatives = int((1 - working["_label"]).sum())
    diagnostics.update(
        definition_candidate_validity_payload(
            rows=int(len(working)),
            positives=positives,
            negatives=negatives,
        )
    )
    for validator in setup.EXTERNAL_VALIDATORS:
        y_val = pd.to_numeric(working[validator], errors="coerce")
        valid_mask = y_val.notna()
        if valid_mask.sum() == 0 or working.loc[valid_mask, "_label"].nunique() < 2:
            diagnostics[f"gap_{validator}"] = float("nan")
            continue
        pos_mean = float(y_val[valid_mask & (working["_label"] == 1)].mean())
        neg_mean = float(y_val[valid_mask & (working["_label"] == 0)].mean())
        diagnostics[f"gap_{validator}"] = pos_mean - neg_mean
    return diagnostics


def make_atomic_rule(metric_name: str, threshold: float, operator: str = ">=") -> dict[str, Any]:
    return setup.make_atomic_rule(metric_name=metric_name, threshold=threshold, operator=operator)


def make_weighted_rule(
    components: list[dict[str, Any]],
    threshold: float,
    operator: str = ">=",
    normalization: str = "empirical_percentile",
    reference_payload: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    return setup.make_weighted_rule(
        components=components,
        threshold=threshold,
        operator=operator,
        normalization=normalization,
        reference_payload=reference_payload,
    )


def canonicalize_rule(rule: dict[str, Any]) -> dict[str, Any]:
    return setup.canonicalize_rule(rule)


def extract_rule_metric_names(rule: dict[str, Any]) -> list[str]:
    return setup.extract_rule_metric_names(rule)


def rule_size(rule: dict[str, Any]) -> int:
    return setup.rule_size(rule)


def rule_operator_label(rule: dict[str, Any]) -> str:
    return setup.rule_operator_label(rule)


def rule_metric_signature(rule: dict[str, Any]) -> str:
    return setup.rule_metric_signature(rule)


def build_rule_text(rule: dict[str, Any]) -> str:
    return setup.build_rule_text(rule)


def build_definition_a_label_name(rule: dict[str, Any]) -> str:
    return setup.build_definition_a_label_name(rule)


def freeze_rule(rule: dict[str, Any], reference_frame: pd.DataFrame | None = None) -> dict[str, Any]:
    return setup.freeze_rule(rule, reference_frame=reference_frame)


def compute_weighted_rule_score(
    frame: pd.DataFrame,
    rule: dict[str, Any],
    reference_frame: pd.DataFrame | None = None,
) -> pd.Series:
    return setup.compute_weighted_rule_score(frame, rule, reference_frame=reference_frame)


def apply_rule_to_frame(
    frame: pd.DataFrame,
    rule: dict[str, Any],
    reference_frame: pd.DataFrame | None = None,
) -> pd.Series:
    return setup.apply_rule_to_frame(frame, rule, reference_frame=reference_frame)


def candidate_group_key(rule: dict[str, Any], candidate_type: str) -> str:
    normalized = canonicalize_rule(rule)
    if normalized["kind"] == "weighted":
        weight_signature = "|".join(
            f'{component["metric_name"]}:{float(component["weight"]):.2f}'
            for component in normalized["components"]
        )
        return f"{candidate_type}::{weight_signature}"
    return f"{candidate_type}::{rule_metric_signature(normalized)}::{rule_operator_label(normalized)}"


def candidate_gain_score(row: dict[str, Any]) -> float:
    gaps = [float(row.get(f"gap_{validator}", float("nan"))) for validator in setup.EXTERNAL_VALIDATORS]
    finite = [value for value in gaps if np.isfinite(value)]
    if not finite:
        return float("-inf")
    return float(np.mean(finite))


def jaccard_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_bool = left.astype(bool)
    right_bool = right.astype(bool)
    union = np.logical_or(left_bool, right_bool).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(left_bool, right_bool).sum()
    return float(intersection / union)


def compute_label_hash(label: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(label.astype(np.uint8))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def make_rule_candidate_row(
    frame: pd.DataFrame,
    rule: dict[str, Any],
    candidate_type: str,
    definition_name: str = "definition_a",
    fitted_rule: dict[str, Any] | None = None,
    threshold_source: str = "observed_train_value",
    threshold_candidate_rank: int | None = None,
    threshold_candidate_count: int | None = None,
) -> dict[str, Any]:
    normalized = canonicalize_rule(rule)
    applied_rule = canonicalize_rule(fitted_rule) if fitted_rule is not None else normalized
    label = apply_rule_to_frame(frame, applied_rule).to_numpy(dtype=int)
    diagnostics = compute_candidate_diagnostics(frame, label)
    metric_names = extract_rule_metric_names(normalized)
    return {
        "definition_name": definition_name,
        "candidate_type": candidate_type,
        "candidate_group_key": candidate_group_key(normalized, candidate_type),
        "metric_name": rule_metric_signature(normalized),
        "threshold": float(normalized["threshold"]) if normalized["kind"] in {"atomic", "weighted"} else float("nan"),
        "rule_json": setup.stable_json(normalized),
        "rule_fit_json": setup.stable_json(applied_rule),
        "rule_text": build_rule_text(normalized),
        "rule_size": rule_size(normalized),
        "rule_operator": rule_operator_label(normalized),
        "metric_count": len(metric_names),
        "threshold_source": threshold_source,
        "threshold_candidate_rank": int(threshold_candidate_rank) if threshold_candidate_rank is not None else np.nan,
        "threshold_candidate_count": int(threshold_candidate_count) if threshold_candidate_count is not None else np.nan,
        "gain_score": candidate_gain_score(diagnostics),
        **diagnostics,
    }


def enumerate_threshold_candidates_from_series(
    train: pd.DataFrame,
    score_series: pd.Series,
    rule_builder,
    candidate_type: str,
    metric_name: str,
    threshold_source: str,
    fitted_rule_builder=None,
) -> pd.DataFrame:
    rows: List[dict[str, Any]] = []
    if train.empty:
        return pd.DataFrame(rows)
    month_totals = train.groupby("first_month").size().sort_index()
    work = train[["first_month"] + setup.EXTERNAL_VALIDATORS].copy()
    work["_score"] = pd.to_numeric(score_series, errors="coerce").fillna(0.0).astype(float).to_numpy()
    grouped = work.groupby("_score", dropna=False, sort=True)
    counts = grouped.size().sort_index(ascending=False)
    thresholds = counts.index.to_numpy(dtype=float)
    cum_pos = counts.cumsum().to_numpy(dtype=float)
    total_n = float(len(work))
    neg = total_n - cum_pos
    monthly_by_value = (
        work.groupby(["_score", "first_month"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=thresholds, columns=month_totals.index, fill_value=0)
    )
    monthly_by_value = monthly_by_value.sort_index(ascending=False)
    monthly_cum = monthly_by_value.cumsum(axis=0)
    monthly_prev = monthly_cum.div(month_totals, axis=1)
    monthly_mean = monthly_prev.mean(axis=1).to_numpy(dtype=float)
    monthly_std = monthly_prev.std(axis=1, ddof=0).to_numpy(dtype=float)

    validator_gaps: dict[str, np.ndarray] = {}
    for validator in setup.EXTERNAL_VALIDATORS:
        sum_by_value = grouped[validator].sum().reindex(thresholds).fillna(0).sort_index(ascending=False)
        pos_sum = sum_by_value.cumsum().to_numpy(dtype=float)
        total_sum = float(pd.to_numeric(work[validator], errors="coerce").fillna(0).sum())
        pos_mean = np.divide(pos_sum, cum_pos, out=np.full_like(pos_sum, np.nan), where=cum_pos > 0)
        neg_mean = np.divide(total_sum - pos_sum, neg, out=np.full_like(pos_sum, np.nan), where=neg > 0)
        validator_gaps[validator] = pos_mean - neg_mean

    for idx, threshold in enumerate(thresholds):
        positives = int(cum_pos[idx])
        negatives = int(neg[idx])
        low, high, width = bootstrap_prevalence_ci_width_from_counts(int(total_n), positives)
        conceptual_rule = canonicalize_rule(rule_builder(float(threshold)))
        fitted_rule = canonicalize_rule(fitted_rule_builder(float(threshold))) if fitted_rule_builder else conceptual_rule
        validity_payload = definition_candidate_validity_payload(
            rows=int(total_n),
            positives=positives,
            negatives=negatives,
        )
        rows.append(
            {
                "definition_name": "definition_a",
                "candidate_type": candidate_type,
                "candidate_group_key": candidate_group_key(conceptual_rule, candidate_type),
                "metric_name": metric_name,
                "threshold": float(threshold),
                "rule_json": setup.stable_json(conceptual_rule),
                "rule_fit_json": setup.stable_json(fitted_rule),
                "rule_text": build_rule_text(conceptual_rule),
                "rule_size": rule_size(conceptual_rule),
                "rule_operator": rule_operator_label(conceptual_rule),
                "metric_count": len(extract_rule_metric_names(conceptual_rule)),
                "rows": int(total_n),
                "positives": positives,
                "negatives": negatives,
                "prevalence": float(cum_pos[idx] / total_n) if total_n > 0 else float("nan"),
                "prevalence_entropy": float(
                    0.0
                    if total_n <= 0 or cum_pos[idx] <= 0 or cum_pos[idx] >= total_n
                    else -(
                        (cum_pos[idx] / total_n) * math.log(cum_pos[idx] / total_n)
                        + (1.0 - (cum_pos[idx] / total_n)) * math.log(1.0 - (cum_pos[idx] / total_n))
                    )
                ),
                "monthly_prevalence_mean": float(monthly_mean[idx]),
                "monthly_prevalence_std": float(monthly_std[idx]),
                "bootstrap_prevalence_ci_low": low,
                "bootstrap_prevalence_ci_high": high,
                "bootstrap_prevalence_ci_width": width,
                "threshold_source": threshold_source,
                "threshold_candidate_rank": int(idx + 1),
                "threshold_candidate_count": int(len(thresholds)),
                **validity_payload,
                **{f"gap_{validator}": float(validator_gaps[validator][idx]) for validator in setup.EXTERNAL_VALIDATORS},
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["gain_score"] = df.apply(lambda row: candidate_gain_score(row.to_dict()), axis=1)
    return df


def enumerate_univariate_candidates(train: pd.DataFrame, candidate_metric_registry: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    if train.empty:
        return pd.DataFrame()
    for metric_row in candidate_metric_registry.to_dict(orient="records"):
        if int(metric_row.get("definition_a_candidate_flag", 0)) != 1:
            continue
        metric_name = metric_row["metric_name"]
        score_series = pd.to_numeric(train[metric_name], errors="coerce").fillna(0)
        rows.append(
            enumerate_threshold_candidates_from_series(
                train=train,
                score_series=score_series,
                rule_builder=lambda threshold, metric_name=metric_name: make_atomic_rule(metric_name, threshold),
                candidate_type="univariate_exact_threshold",
                metric_name=metric_name,
                threshold_source="observed_train_metric_value",
            )
        )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def evaluate_threshold_grid_on_frame(
    frame: pd.DataFrame,
    metric_name: str,
    thresholds: Sequence[float] | pd.DataFrame,
    definition_name: str = "definition_a",
    candidate_type: str = "univariate_exact_threshold",
) -> pd.DataFrame:
    rows = []
    if isinstance(thresholds, pd.DataFrame):
        threshold_rows = thresholds.copy()
    else:
        threshold_rows = pd.DataFrame({"threshold": sorted({float(value) for value in thresholds})})
    if threshold_rows.empty:
        return pd.DataFrame(rows)
    if "threshold" not in threshold_rows.columns:
        raise KeyError("threshold candidates must include a 'threshold' column")
    threshold_rows["threshold"] = pd.to_numeric(threshold_rows["threshold"], errors="coerce")
    threshold_rows = threshold_rows.dropna(subset=["threshold"]).drop_duplicates(subset=["threshold"]).sort_values("threshold")
    threshold_rows["threshold_candidate_rank"] = pd.to_numeric(
        threshold_rows.get("threshold_candidate_rank"),
        errors="coerce",
    )
    threshold_rows["threshold_candidate_count"] = pd.to_numeric(
        threshold_rows.get("threshold_candidate_count"),
        errors="coerce",
    )
    if threshold_rows["threshold_candidate_count"].isna().all():
        threshold_rows["threshold_candidate_count"] = int(len(threshold_rows))
    for row in threshold_rows.to_dict(orient="records"):
        threshold = float(row["threshold"])
        rule = make_atomic_rule(metric_name, float(threshold))
        candidate_row = make_rule_candidate_row(
            frame,
            rule,
            candidate_type=candidate_type,
            definition_name=definition_name,
            threshold_source=str(row.get("threshold_source", "observed_train_value")),
            threshold_candidate_rank=int(row["threshold_candidate_rank"]) if pd.notna(row.get("threshold_candidate_rank")) else None,
            threshold_candidate_count=int(row["threshold_candidate_count"]) if pd.notna(row.get("threshold_candidate_count")) else None,
        )
        rows.append(candidate_row)
    return pd.DataFrame(rows)


def aggregate_definition_test_eval(test_eval: pd.DataFrame) -> pd.DataFrame:
    if test_eval.empty:
        return pd.DataFrame()
    group_cols = [
        "definition_name",
        "candidate_type",
        "candidate_group_key",
        "metric_name",
        "threshold",
        "rule_json",
        "rule_text",
        "rule_size",
        "rule_operator",
        "threshold_source",
    ]
    summary = (
        test_eval.groupby(
            group_cols,
            dropna=False,
            as_index=False,
        )
        .agg(
            folds=("fold_id", "nunique"),
            threshold_candidate_rank=("threshold_candidate_rank", "mean"),
            threshold_candidate_count=("threshold_candidate_count", "max"),
            test_gap_returned_active_post_label_m1=("gap_returned_active_post_label_m1", "mean"),
            test_gap_returned_active_post_label_m2=("gap_returned_active_post_label_m2", "mean"),
            test_gap_returned_active_post_label_m3=("gap_returned_active_post_label_m3", "mean"),
            test_gap_active_days_post_label_3m=("gap_active_days_post_label_3m", "mean"),
            test_gap_sustained_active_2of3_post_label=("gap_sustained_active_2of3_post_label", "mean"),
            test_prevalence_entropy=("prevalence_entropy", "mean"),
            test_monthly_prevalence_std=("monthly_prevalence_std", "mean"),
            test_bootstrap_prevalence_ci_width=("bootstrap_prevalence_ci_width", "mean"),
        )
    )
    return attach_fold_gap_bootstrap_summary(
        summary=summary,
        source=test_eval,
        group_cols=group_cols,
        source_gap_col="gap_sustained_active_2of3_post_label",
        target_prefix="test_gap_sustained_active_2of3_post_label",
    )


def attach_fold_gap_bootstrap_summary(
    summary: pd.DataFrame,
    source: pd.DataFrame,
    group_cols: list[str],
    source_gap_col: str,
    target_prefix: str,
) -> pd.DataFrame:
    low_col = f"{target_prefix}_ci_low"
    high_col = f"{target_prefix}_ci_high"
    width_col = f"{target_prefix}_ci_width"
    if summary.empty:
        summary = summary.copy()
        summary[low_col] = np.nan
        summary[high_col] = np.nan
        summary[width_col] = np.nan
        return summary
    rows: list[dict[str, Any]] = []
    for group_key, group in source.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        values = pd.to_numeric(group[source_gap_col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) >= 2:
            ci_low, ci_high, ci_width = bootstrap_ci_width(values, np.mean)
        else:
            ci_low = ci_high = ci_width = float("nan")
        payload = {col: value for col, value in zip(group_cols, group_key)}
        payload[low_col] = ci_low
        payload[high_col] = ci_high
        payload[width_col] = ci_width
        rows.append(payload)
    ci_df = pd.DataFrame(rows)
    return summary.merge(ci_df, on=group_cols, how="left")


def evaluate_numeric_gate(value: Any, operator: str, threshold: float) -> int:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return 0
    numeric = float(numeric)
    threshold = float(threshold)
    if operator == ">":
        return int(numeric > threshold)
    if operator == ">=":
        return int(numeric >= threshold)
    if operator == "<":
        return int(numeric < threshold)
    if operator == "<=":
        return int(numeric <= threshold)
    if operator == "==":
        return int(numeric == threshold)
    if operator == "!=":
        return int(numeric != threshold)
    raise ValueError(f"Unsupported numeric gate operator: {operator}")


def format_numeric_gate_reject_reason(column_name: str, operator: str, threshold: float) -> str:
    return f"numeric_gate_failed:{column_name} {operator} {float(threshold):g}"


def select_fold_pareto_candidates(train_candidates: pd.DataFrame) -> pd.DataFrame:
    valid = train_candidates[train_candidates["candidate_valid_flag"] == 1].copy()
    if valid.empty:
        return valid
    frontier = pareto_front(valid, setup.TRAIN_DEFINITION_OBJECTIVES)
    return frontier[frontier["pareto_frontier_flag"] == 1].copy()


def choose_metric_representatives(candidate_summary: pd.DataFrame, objectives: Dict[str, str]) -> pd.DataFrame:
    if candidate_summary.empty:
        return pd.DataFrame(columns=list(candidate_summary.columns) + ["worst_objective_rank", "mean_objective_rank"])
    selected_rows: list[dict[str, Any]] = []
    group_key = "candidate_group_key" if "candidate_group_key" in candidate_summary.columns else "metric_name"
    for _, group in candidate_summary.groupby(group_key, dropna=False):
        work = group.copy().reset_index(drop=True)
        if "rule_size" not in work.columns:
            work["rule_size"] = 1
        if "gain_score" not in work.columns:
            work["gain_score"] = np.nan
        rank_cols: list[str] = []
        for objective_name, direction in objectives.items():
            ascending = direction == "min"
            rank_col = f"rank__{objective_name}"
            work[rank_col] = work[objective_name].rank(method="min", ascending=ascending)
            rank_cols.append(rank_col)
        work["worst_objective_rank"] = work[rank_cols].max(axis=1)
        work["mean_objective_rank"] = work[rank_cols].mean(axis=1)
        sort_cols = ["worst_objective_rank", "mean_objective_rank", "rule_size"]
        ascending = [True, True, True]
        if work["gain_score"].notna().any():
            sort_cols.append("gain_score")
            ascending.append(False)
        sort_cols.extend(list(objectives.keys()))
        ascending.extend([direction == "min" for direction in objectives.values()])
        selected_rows.append(work.sort_values(sort_cols, ascending=ascending).iloc[0].to_dict())
    return pd.DataFrame(selected_rows)


def attach_rule_label_signatures(
    frame: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    reference_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if candidate_summary.empty:
        return candidate_summary.copy()
    ordered = frame.sort_values(["teacher_unique_id", "first_month"]).reset_index(drop=True)
    work = candidate_summary.copy().reset_index(drop=True)
    cache: dict[str, tuple[str, int, float]] = {}
    label_hashes: list[str] = []
    label_positives: list[int] = []
    label_share_pct: list[float] = []
    for rule_json in work["rule_json"].tolist():
        if rule_json not in cache:
            rule = canonicalize_rule(json.loads(rule_json))
            label = apply_rule_to_frame(ordered, rule, reference_frame=reference_frame).to_numpy(dtype=int)
            positives = int(label.sum())
            cache[rule_json] = (
                compute_label_hash(label),
                positives,
                float(100.0 * positives / len(ordered)) if len(ordered) else float("nan"),
            )
        label_hash, positives, share_pct = cache[rule_json]
        label_hashes.append(label_hash)
        label_positives.append(positives)
        label_share_pct.append(share_pct)
    work["label_hash"] = label_hashes
    work["label_positives"] = label_positives
    work["label_share_pct"] = label_share_pct
    work["label_vector_group_size"] = work.groupby("label_hash")["label_hash"].transform("size")
    return work


def choose_label_vector_representatives(candidate_summary: pd.DataFrame, objectives: Dict[str, str]) -> pd.DataFrame:
    if candidate_summary.empty:
        return candidate_summary.copy()
    selected_rows: list[dict[str, Any]] = []
    for _, group in candidate_summary.groupby("label_hash", dropna=False):
        work = group.copy().reset_index(drop=True)
        rank_cols: list[str] = []
        for objective_name, direction in objectives.items():
            ascending = direction == "min"
            rank_col = f"rank__{objective_name}"
            work[rank_col] = work[objective_name].rank(method="min", ascending=ascending)
            rank_cols.append(rank_col)
        work["worst_objective_rank"] = work[rank_cols].max(axis=1)
        work["mean_objective_rank"] = work[rank_cols].mean(axis=1)
        sort_cols = ["worst_objective_rank", "mean_objective_rank", "rule_size", "label_vector_group_size"]
        ascending = [True, True, True, False]
        if "gain_score" in work.columns and work["gain_score"].notna().any():
            sort_cols.append("gain_score")
            ascending.append(False)
        sort_cols.extend(list(objectives.keys()))
        ascending.extend([direction == "min" for direction in objectives.values()])
        best_row = work.sort_values(sort_cols, ascending=ascending).iloc[0].to_dict()
        best_row["label_vector_representative_flag"] = 1
        selected_rows.append(best_row)
    return pd.DataFrame(selected_rows)


def expand_promoted_atomic_candidates(train: pd.DataFrame, promoted_atomic: pd.DataFrame) -> pd.DataFrame:
    if train.empty or promoted_atomic.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    promoted_unique = (
        promoted_atomic.sort_values(["primary_selection_rank", "rule_size"], kind="mergesort")
        .drop_duplicates(subset=["metric_name"], keep="first")
        .reset_index(drop=True)
    )
    # Keep promoted atomics in the expanded competition.
    for row in promoted_unique.to_dict(orient="records"):
        rule = canonicalize_rule(json.loads(str(row["rule_json"])))
        rows.append(
            pd.DataFrame(
                [
                    make_rule_candidate_row(
                        train,
                        rule,
                        candidate_type=str(row.get("candidate_type", "univariate_exact_threshold")),
                        definition_name="definition_a",
                        fitted_rule=rule,
                        threshold_source=str(row.get("threshold_source", "observed_train_metric_value")),
                        threshold_candidate_rank=int(pd.to_numeric(row.get("threshold_candidate_rank"), errors="coerce"))
                        if pd.notna(pd.to_numeric(row.get("threshold_candidate_rank"), errors="coerce"))
                        else None,
                        threshold_candidate_count=int(pd.to_numeric(row.get("threshold_candidate_count"), errors="coerce"))
                        if pd.notna(pd.to_numeric(row.get("threshold_candidate_count"), errors="coerce"))
                        else None,
                    )
                ]
            )
        )
    promoted_records = promoted_unique.to_dict(orient="records")
    for left_idx in range(len(promoted_records)):
        left_rule = canonicalize_rule(json.loads(str(promoted_records[left_idx]["rule_json"])))
        left_metric = str(promoted_records[left_idx]["metric_name"])
        for right_idx in range(left_idx + 1, len(promoted_records)):
            right_rule = canonicalize_rule(json.loads(str(promoted_records[right_idx]["rule_json"])))
            right_metric = str(promoted_records[right_idx]["metric_name"])
            if left_metric == right_metric:
                continue
            for combiner, candidate_type in (("AND", "compound_pairwise_and"), ("OR", "compound_pairwise_or")):
                rule = canonicalize_rule({"kind": "compound", "combiner": combiner, "rules": [left_rule, right_rule]})
                rows.append(
                    pd.DataFrame(
                        [
                            make_rule_candidate_row(
                                train,
                                rule,
                                candidate_type=candidate_type,
                                definition_name="definition_a",
                                fitted_rule=rule,
                                threshold_source="composed_from_promoted_atomics",
                            )
                        ]
                    )
                )
            for left_weight, right_weight in WEIGHT_GRID:
                weighted_base = make_weighted_rule(
                    components=[
                        {"metric_name": left_metric, "weight": float(left_weight)},
                        {"metric_name": right_metric, "weight": float(right_weight)},
                    ],
                    threshold=0.0,
                )
                score_series = compute_weighted_rule_score(train, weighted_base, reference_frame=train)
                rows.append(
                    enumerate_threshold_candidates_from_series(
                        train=train,
                        score_series=score_series,
                        rule_builder=lambda threshold, left_metric=left_metric, right_metric=right_metric, left_weight=left_weight, right_weight=right_weight: make_weighted_rule(
                            components=[
                                {"metric_name": left_metric, "weight": float(left_weight)},
                                {"metric_name": right_metric, "weight": float(right_weight)},
                            ],
                            threshold=threshold,
                        ),
                        candidate_type="weighted_pairwise_percentile_threshold",
                        metric_name=f"{left_metric} + {right_metric}",
                        threshold_source="observed_train_weighted_percentile_score",
                        fitted_rule_builder=lambda threshold, left_metric=left_metric, right_metric=right_metric, left_weight=left_weight, right_weight=right_weight: freeze_rule(
                            make_weighted_rule(
                                components=[
                                    {"metric_name": left_metric, "weight": float(left_weight)},
                                    {"metric_name": right_metric, "weight": float(right_weight)},
                                ],
                                threshold=threshold,
                            ),
                            reference_frame=train,
                        ),
                    )
                )
    if not rows:
        return pd.DataFrame()
    expanded = pd.concat(rows, ignore_index=True)
    expanded["search_stage"] = "expanded_from_promoted_atomic_candidates"
    return expanded


def enumerate_definition_a_candidates(train: pd.DataFrame, candidate_metric_registry: pd.DataFrame) -> pd.DataFrame:
    strategy = str(setup.DEFINITION_A_STRATEGY).lower()
    if strategy not in {"univariate_exact", "screened_pairwise_compound_weighted"}:
        raise ValueError(f"Unsupported Definition A strategy: {setup.DEFINITION_A_STRATEGY}")
    screened = enumerate_univariate_candidates(train, candidate_metric_registry)
    if not screened.empty:
        screened["search_stage"] = "atomic_screening"
    return screened


def evaluate_candidate_rows_on_test(test: pd.DataFrame, train_candidates: pd.DataFrame) -> pd.DataFrame:
    if train_candidates.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for row in train_candidates.drop_duplicates(subset=["rule_json"]).to_dict(orient="records"):
        rule = canonicalize_rule(json.loads(row["rule_json"]))
        fitted_rule_json = row.get("rule_fit_json")
        fitted_rule = canonicalize_rule(json.loads(str(fitted_rule_json))) if setup.normalize_text(fitted_rule_json, "") else rule
        test_row = make_rule_candidate_row(
            test,
            rule,
            candidate_type=str(row.get("candidate_type", "definition_a_candidate")),
            definition_name="definition_a",
            fitted_rule=fitted_rule,
            threshold_source=str(row.get("threshold_source", "observed_train_value")),
            threshold_candidate_rank=int(pd.to_numeric(row.get("threshold_candidate_rank"), errors="coerce"))
            if pd.notna(pd.to_numeric(row.get("threshold_candidate_rank"), errors="coerce"))
            else None,
            threshold_candidate_count=int(pd.to_numeric(row.get("threshold_candidate_count"), errors="coerce"))
            if pd.notna(pd.to_numeric(row.get("threshold_candidate_count"), errors="coerce"))
            else None,
        )
        test_row["search_stage"] = row.get("search_stage")
        rows.append(test_row)
    return pd.DataFrame(rows)


def evaluate_candidates_on_lock_months(
    lock_frame: pd.DataFrame,
    candidate_rows: pd.DataFrame,
    reference_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if lock_frame.empty or candidate_rows.empty:
        return pd.DataFrame()
    ordered = lock_frame.sort_values(["first_month", "teacher_unique_id"]).reset_index(drop=True)
    lock_rows: list[dict[str, Any]] = []
    unique_candidates = candidate_rows.drop_duplicates(subset=["rule_json"]).copy()
    for month in setup.ordered_unique_months(ordered):
        month_subset = ordered[
            pd.to_datetime(ordered["first_month"], errors="coerce").dt.to_period("M").dt.to_timestamp() == pd.Timestamp(month)
        ].copy()
        if month_subset.empty:
            continue
        for row in unique_candidates.to_dict(orient="records"):
            rule = canonicalize_rule(json.loads(row["rule_json"]))
            fitted_rule = freeze_rule(rule, reference_frame=reference_frame) if rule.get("kind") == "weighted" else rule
            diagnostics = make_rule_candidate_row(
                month_subset,
                rule,
                candidate_type=str(row.get("candidate_type", "definition_a_candidate")),
                definition_name="definition_a",
                fitted_rule=fitted_rule,
                threshold_source=str(row.get("threshold_source", "observed_train_value")),
                threshold_candidate_rank=int(pd.to_numeric(row.get("threshold_candidate_rank"), errors="coerce"))
                if pd.notna(pd.to_numeric(row.get("threshold_candidate_rank"), errors="coerce"))
                else None,
                threshold_candidate_count=int(pd.to_numeric(row.get("threshold_candidate_count"), errors="coerce"))
                if pd.notna(pd.to_numeric(row.get("threshold_candidate_count"), errors="coerce"))
                else None,
            )
            diagnostics["lock_month"] = pd.Timestamp(month)
            diagnostics["development_rank"] = int(row.get("primary_selection_rank", 0))
            lock_rows.append(diagnostics)
    return pd.DataFrame(lock_rows)


def summarize_lock_neighbor_sensitivity(
    lock_frame: pd.DataFrame,
    promoted_candidates: pd.DataFrame,
    threshold_pool: pd.DataFrame,
    reference_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if lock_frame.empty or promoted_candidates.empty or threshold_pool.empty:
        return pd.DataFrame(
            columns=[
                "rule_json",
                "lock_neighbor_count",
                "lock_threshold_neighbor_count",
                "lock_structural_neighbor_count",
                "lock_weight_neighbor_count",
                "lock_min_label_jaccard",
                "lock_mean_label_jaccard",
                "lock_max_neighbor_gap_delta",
                "lock_mean_neighbor_gap_delta",
                "lock_max_neighbor_prevalence_delta",
            ]
        )
    ordered = lock_frame.sort_values(["first_month", "teacher_unique_id"]).reset_index(drop=True)
    pool = threshold_pool.copy()
    pool["threshold"] = pd.to_numeric(pool.get("threshold"), errors="coerce")
    pool = pool.dropna(subset=["metric_name", "threshold"]).drop_duplicates()
    thresholds_by_metric = {
        str(metric_name): sorted(group["threshold"].tolist())
        for metric_name, group in pool.groupby("metric_name", dropna=False)
    }
    weighted_thresholds_by_group = {
        str(group_key): sorted(group["threshold"].dropna().tolist())
        for group_key, group in pool.groupby("candidate_group_key", dropna=False)
    }

    def _nearest_thresholds(metric_name: str, candidate_threshold: float) -> list[float]:
        ordered_thresholds = thresholds_by_metric.get(metric_name, [])
        lower = [value for value in ordered_thresholds if value < float(candidate_threshold)]
        upper = [value for value in ordered_thresholds if value > float(candidate_threshold)]
        neighbor_thresholds: list[float] = []
        if lower:
            neighbor_thresholds.append(float(lower[-1]))
        if upper:
            neighbor_thresholds.append(float(upper[0]))
        return neighbor_thresholds

    def _build_variants(candidate_rule: dict[str, Any], candidate_row: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        normalized = canonicalize_rule(candidate_rule)
        variants: list[tuple[str, dict[str, Any]]] = []
        if normalized["kind"] == "atomic":
            candidate_threshold = pd.to_numeric(candidate_row.get("threshold"), errors="coerce")
            if pd.notna(candidate_threshold):
                for neighbor_threshold in _nearest_thresholds(str(normalized["metric_name"]), float(candidate_threshold)):
                    variants.append(
                        (
                            "threshold",
                            make_atomic_rule(
                                str(normalized["metric_name"]),
                                float(neighbor_threshold),
                                operator=str(normalized.get("operator", ">=")),
                            ),
                        )
                    )
            return variants
        if normalized["kind"] == "compound":
            children = [canonicalize_rule(child) for child in normalized["rules"]]
            if len(children) == 2:
                swapped = canonicalize_rule(
                    {
                        "kind": "compound",
                        "combiner": "OR" if normalized["combiner"] == "AND" else "AND",
                        "rules": children,
                    }
                )
                variants.append(("combiner_swap", swapped))
                for child_idx in range(2):
                    variants.append(("drop_one_literal", children[child_idx]))
                    if children[child_idx]["kind"] == "atomic":
                        candidate_threshold = pd.to_numeric(children[child_idx].get("threshold"), errors="coerce")
                        if pd.notna(candidate_threshold):
                            metric_name = str(children[child_idx]["metric_name"])
                            for neighbor_threshold in _nearest_thresholds(metric_name, float(candidate_threshold)):
                                replacement = make_atomic_rule(
                                    metric_name,
                                    float(neighbor_threshold),
                                    operator=str(children[child_idx].get("operator", ">=")),
                                )
                                rebuilt_children = children.copy()
                                rebuilt_children[child_idx] = replacement
                                variants.append(
                                    (
                                        "threshold",
                                        canonicalize_rule(
                                            {
                                                "kind": "compound",
                                                "combiner": normalized["combiner"],
                                                "rules": rebuilt_children,
                                            }
                                        ),
                                    )
                                )
            return variants
        if normalized["kind"] == "weighted":
            group_key = candidate_group_key(normalized, str(candidate_row.get("candidate_type", "weighted_pairwise_percentile_threshold")))
            thresholds = weighted_thresholds_by_group.get(group_key, [])
            candidate_threshold = float(normalized["threshold"])
            lower = [value for value in thresholds if value < candidate_threshold]
            upper = [value for value in thresholds if value > candidate_threshold]
            if lower:
                variants.append(
                    (
                        "threshold",
                        make_weighted_rule(
                            components=list(normalized["components"]),
                            threshold=float(lower[-1]),
                            operator=str(normalized.get("operator", ">=")),
                        ),
                    )
                )
            if upper:
                variants.append(
                    (
                        "threshold",
                        make_weighted_rule(
                            components=list(normalized["components"]),
                            threshold=float(upper[0]),
                            operator=str(normalized.get("operator", ">=")),
                        ),
                    )
                )
            metric_names = [str(component["metric_name"]) for component in normalized["components"]]
            if len(metric_names) == 2:
                for left_weight, right_weight in WEIGHT_GRID:
                    current = (round(float(normalized["components"][0]["weight"]), 2), round(float(normalized["components"][1]["weight"]), 2))
                    proposed = (round(float(left_weight), 2), round(float(right_weight), 2))
                    if proposed == current:
                        continue
                    variants.append(
                        (
                            "weight",
                            make_weighted_rule(
                                components=[
                                    {"metric_name": metric_names[0], "weight": float(left_weight)},
                                    {"metric_name": metric_names[1], "weight": float(right_weight)},
                                ],
                                threshold=float(normalized["threshold"]),
                                operator=str(normalized.get("operator", ">=")),
                            ),
                        )
                    )
            return variants
        return variants

    rows: list[dict[str, Any]] = []
    for candidate in promoted_candidates.drop_duplicates(subset=["rule_json"]).to_dict(orient="records"):
        candidate_rule = canonicalize_rule(json.loads(str(candidate["rule_json"])))
        fitted_candidate_rule = freeze_rule(candidate_rule, reference_frame=reference_frame) if candidate_rule.get("kind") == "weighted" else candidate_rule
        candidate_label = apply_rule_to_frame(ordered, fitted_candidate_rule, reference_frame=reference_frame).to_numpy(dtype=int)
        candidate_diag = make_rule_candidate_row(
            ordered,
            candidate_rule,
            candidate_type=str(candidate.get("candidate_type", "definition_a_candidate")),
            definition_name="definition_a",
            fitted_rule=fitted_candidate_rule,
        )
        gap_deltas: list[float] = []
        prevalence_deltas: list[float] = []
        jaccards: list[float] = []
        threshold_neighbor_count = 0
        structural_neighbor_count = 0
        weight_neighbor_count = 0
        variants = _build_variants(candidate_rule, candidate)
        if not variants:
            continue
        for variant_type, neighbor_rule in variants:
            fitted_neighbor_rule = freeze_rule(neighbor_rule, reference_frame=reference_frame) if neighbor_rule.get("kind") == "weighted" else neighbor_rule
            neighbor_label = apply_rule_to_frame(ordered, fitted_neighbor_rule, reference_frame=reference_frame).to_numpy(dtype=int)
            neighbor_diag = make_rule_candidate_row(
                ordered,
                neighbor_rule,
                candidate_type=f"definition_a_local_{variant_type}_neighbor",
                definition_name="definition_a",
                fitted_rule=fitted_neighbor_rule,
            )
            if variant_type == "threshold":
                threshold_neighbor_count += 1
            elif variant_type == "weight":
                weight_neighbor_count += 1
            else:
                structural_neighbor_count += 1
            jaccards.append(jaccard_similarity(candidate_label, neighbor_label))
            gap_deltas.append(
                float(
                    max(
                        abs(
                            float(candidate_diag.get(f"gap_{validator}", float("nan")))
                            - float(neighbor_diag.get(f"gap_{validator}", float("nan")))
                        )
                        for validator in setup.EXTERNAL_VALIDATORS
                    )
                )
            )
            prevalence_deltas.append(
                abs(float(candidate_diag.get("prevalence", float("nan"))) - float(neighbor_diag.get("prevalence", float("nan"))))
            )
        rows.append(
            {
                "rule_json": str(candidate["rule_json"]),
                "lock_neighbor_count": int(len(variants)),
                "lock_threshold_neighbor_count": int(threshold_neighbor_count),
                "lock_structural_neighbor_count": int(structural_neighbor_count),
                "lock_weight_neighbor_count": int(weight_neighbor_count),
                "lock_min_label_jaccard": float(min(jaccards)) if jaccards else float("nan"),
                "lock_mean_label_jaccard": float(np.mean(jaccards)) if jaccards else float("nan"),
                "lock_max_neighbor_gap_delta": float(max(gap_deltas)) if gap_deltas else float("nan"),
                "lock_mean_neighbor_gap_delta": float(np.mean(gap_deltas)) if gap_deltas else float("nan"),
                "lock_max_neighbor_prevalence_delta": float(max(prevalence_deltas)) if prevalence_deltas else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def summarize_lock_candidates(lock_eval: pd.DataFrame) -> pd.DataFrame:
    if lock_eval.empty:
        return pd.DataFrame()
    group_cols = [
        "definition_name",
        "candidate_type",
        "candidate_group_key",
        "metric_name",
        "threshold",
        "rule_json",
        "rule_text",
        "rule_size",
        "rule_operator",
        "threshold_source",
    ]
    summary = (
        lock_eval.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            lock_months=("lock_month", "nunique"),
            threshold_candidate_rank=("threshold_candidate_rank", "mean"),
            threshold_candidate_count=("threshold_candidate_count", "max"),
            lock_gap_returned_active_post_label_m1=("gap_returned_active_post_label_m1", "mean"),
            lock_gap_returned_active_post_label_m2=("gap_returned_active_post_label_m2", "mean"),
            lock_gap_returned_active_post_label_m3=("gap_returned_active_post_label_m3", "mean"),
            lock_gap_active_days_post_label_3m=("gap_active_days_post_label_3m", "mean"),
            lock_gap_sustained_active_2of3_post_label=("gap_sustained_active_2of3_post_label", "mean"),
            lock_prevalence_entropy=("prevalence_entropy", "mean"),
            lock_bootstrap_prevalence_ci_width=("bootstrap_prevalence_ci_width", "mean"),
            lock_prevalence_mean=("prevalence", "mean"),
            lock_prevalence_std=("prevalence", lambda values: float(pd.Series(values).std(ddof=0)) if len(values) > 1 else 0.0),
            development_rank=("development_rank", "min"),
        )
    )
    summary = attach_fold_gap_bootstrap_summary(
        summary=summary,
        source=lock_eval,
        group_cols=group_cols,
        source_gap_col="gap_sustained_active_2of3_post_label",
        target_prefix="lock_gap_sustained_active_2of3_post_label",
    )
    gate_spec = setup.get_definition_lock_bootstrap_gate_spec()
    gate_col = str(gate_spec["column_name"])
    gate_operator = str(gate_spec["operator"])
    gate_threshold = float(gate_spec["threshold"])
    if gate_col not in summary.columns:
        summary[gate_col] = np.nan
    summary["lock_primary_gate_column_name"] = gate_col
    summary["lock_primary_gate_operator"] = gate_operator
    summary["lock_primary_gate_threshold"] = gate_threshold
    summary["lock_primary_gate_pass_flag"] = (
        pd.to_numeric(summary.get(gate_col), errors="coerce")
        .map(lambda value: evaluate_numeric_gate(value, gate_operator, gate_threshold))
        .astype(int)
    )
    summary["lock_primary_gate_reject_reason"] = np.where(
        summary["lock_primary_gate_pass_flag"] == 1,
        "",
        format_numeric_gate_reject_reason(gate_col, gate_operator, gate_threshold),
    )
    summary["lock_primary_gap_ci_positive_flag"] = summary["lock_primary_gate_pass_flag"].astype(int)
    summary["lock_primary_gap_ci_reject_reason"] = summary["lock_primary_gate_reject_reason"].astype(str)
    gap_cols = [
        "gap_returned_active_post_label_m1",
        "gap_returned_active_post_label_m2",
        "gap_returned_active_post_label_m3",
        "gap_active_days_post_label_3m",
        "gap_sustained_active_2of3_post_label",
    ]
    variability_rows: list[dict[str, Any]] = []
    ordered = lock_eval.sort_values(["rule_json", "lock_month"]).copy()
    for rule_json, group in ordered.groupby("rule_json", dropna=False):
        payload: dict[str, Any] = {"rule_json": rule_json}
        std_values: list[float] = []
        jump_values: list[float] = []
        for gap_col in gap_cols:
            values = pd.to_numeric(group[gap_col], errors="coerce").dropna().to_numpy(dtype=float)
            std_val = float(np.std(values, ddof=0)) if len(values) > 1 else 0.0
            jump_val = float(np.max(np.abs(np.diff(values)))) if len(values) > 1 else 0.0
            payload[f"{gap_col}_std"] = std_val
            payload[f"{gap_col}_max_jump"] = jump_val
            std_values.append(std_val)
            jump_values.append(jump_val)
        payload["lock_max_gap_std"] = max(std_values) if std_values else float("inf")
        payload["lock_max_gap_jump"] = max(jump_values) if jump_values else float("inf")
        variability_rows.append(payload)
    variability = pd.DataFrame(variability_rows)
    if not variability.empty:
        summary = summary.merge(variability, on="rule_json", how="left")
    else:
        summary["lock_max_gap_std"] = float("inf")
        summary["lock_max_gap_jump"] = float("inf")
    return summary


def choose_final_definition_a_from_lock(lock_summary: pd.DataFrame) -> pd.DataFrame:
    if lock_summary.empty:
        return lock_summary.copy()
    ranked = lock_summary.copy()
    if "lock_primary_gate_pass_flag" not in ranked.columns:
        gate_spec = setup.get_definition_lock_bootstrap_gate_spec()
        gate_col = str(gate_spec["column_name"])
        gate_operator = str(gate_spec["operator"])
        gate_threshold = float(gate_spec["threshold"])
        if gate_col not in ranked.columns:
            ranked[gate_col] = np.nan
        ranked[gate_col] = pd.to_numeric(ranked[gate_col], errors="coerce")
        ranked["lock_primary_gate_column_name"] = gate_col
        ranked["lock_primary_gate_operator"] = gate_operator
        ranked["lock_primary_gate_threshold"] = gate_threshold
        ranked["lock_primary_gate_pass_flag"] = (
            ranked[gate_col]
            .map(lambda value: evaluate_numeric_gate(value, gate_operator, gate_threshold))
            .astype(int)
        )
        ranked["lock_primary_gate_reject_reason"] = np.where(
            ranked["lock_primary_gate_pass_flag"] == 1,
            "",
            format_numeric_gate_reject_reason(gate_col, gate_operator, gate_threshold),
        )
    ranked["lock_primary_gap_ci_positive_flag"] = ranked["lock_primary_gate_pass_flag"].astype(int)
    ranked["lock_primary_gap_ci_reject_reason"] = ranked["lock_primary_gate_reject_reason"].astype(str)
    ranked = ranked[ranked["lock_primary_gate_pass_flag"] == 1].copy()
    if ranked.empty:
        return ranked
    lock_frontier = pareto_front(ranked, setup.LOCK_DEFINITION_OBJECTIVES)
    if "pareto_frontier_flag" in lock_frontier.columns:
        ranked = lock_frontier[lock_frontier["pareto_frontier_flag"] == 1].copy()
        if ranked.empty:
            ranked = lock_summary.copy()
        else:
            ranked["lock_pareto_frontier_flag"] = 1
    if "lock_pareto_frontier_flag" not in ranked.columns:
        ranked["lock_pareto_frontier_flag"] = 0
    sort_defaults = {
        "lock_months": (False, float("-inf")),
        "lock_gap_returned_active_post_label_m1": (False, float("-inf")),
        "lock_gap_returned_active_post_label_m2": (False, float("-inf")),
        "lock_gap_returned_active_post_label_m3": (False, float("-inf")),
        "lock_gap_active_days_post_label_3m": (False, float("-inf")),
        "lock_gap_sustained_active_2of3_post_label": (False, float("-inf")),
        "lock_gap_sustained_active_2of3_post_label_ci_width": (True, float("inf")),
        "lock_max_gap_std": (True, float("inf")),
        "lock_max_gap_jump": (True, float("inf")),
        "lock_min_label_jaccard": (False, float("-inf")),
        "lock_max_neighbor_gap_delta": (True, float("inf")),
        "lock_max_neighbor_prevalence_delta": (True, float("inf")),
        "lock_prevalence_entropy": (False, float("-inf")),
        "lock_bootstrap_prevalence_ci_width": (True, float("inf")),
        "lock_prevalence_std": (True, float("inf")),
        "rule_size": (True, float("inf")),
        "development_rank": (True, float("inf")),
        "threshold": (True, float("inf")),
    }
    sort_cols: list[str] = []
    ascending: list[bool] = []
    for col, (asc, default) in sort_defaults.items():
        if col in ranked.columns:
            ranked[col] = pd.to_numeric(ranked[col], errors="coerce").fillna(default)
            sort_cols.append(col)
            ascending.append(asc)
    for col in [c for c in ["metric_name", "rule_operator", "rule_text", "rule_json"] if c in ranked.columns]:
        ranked[col] = ranked[col].astype(str)
        sort_cols.append(col)
        ascending.append(True)
    ranked = ranked.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)
    ranked["lock_selection_rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def build_definition_search(metrics: pd.DataFrame, candidate_metric_registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Search and freeze Definition A without using model metrics.

    Row-level logic:
    - candidate rules label teachers as active/inactive in the 30-day label window
    - fixed post-label validators then measure whether those groups continue to
      engage differently over the next 90 days

    Calendar-month workflow:
    - development months search and screen candidates
    - definition-lock months re-test the promoted candidates and freeze one winner
    - the final untouched months are reserved for the later model stage
    """
    frame = metrics.loc[metrics["full_followup_observed_flag"] == 1].copy()
    frame = setup.apply_official_population_filter(frame)
    frame = frame.sort_values(["first_month", "teacher_unique_id"]).reset_index(drop=True)
    development_frame, definition_lock_frame, _, development_months, lock_months, final_eval_months = setup.split_definition_workflow_frame(frame)
    search_frame = development_frame.sort_values(["first_month", "teacher_unique_id"]).reset_index(drop=True)
    splitter = ExpandingMonthSplit(month_col="first_month", min_train_periods=1, test_periods=1, max_splits=None)
    candidate_rows: List[dict[str, Any]] = []
    test_rows: List[dict[str, Any]] = []
    if lock_months or final_eval_months:
        print(
            "[definition_search] development months="
            f"{len(development_months)} | definition-lock months={len(lock_months)} | untouched model-eval months={len(final_eval_months)}",
            flush=True,
        )
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(search_frame), start=1):
        print(f"[definition_search] fold {fold_id} / {splitter.get_n_splits(search_frame)}", flush=True)
        train = search_frame.iloc[train_idx].copy()
        test = search_frame.iloc[test_idx].copy()
        train_candidates = enumerate_definition_a_candidates(train, candidate_metric_registry)
        if train_candidates.empty:
            continue
        train_candidates["fold_id"] = fold_id
        train_candidates["split_role"] = "train"
        candidate_rows.extend(train_candidates.to_dict(orient="records"))
        test_candidates = []
        for metric_row in candidate_metric_registry.to_dict(orient="records"):
            if int(metric_row.get("definition_a_candidate_flag", 0)) != 1:
                continue
            metric_name = metric_row["metric_name"]
            metric_train = train_candidates[train_candidates["candidate_type"] == "univariate_exact_threshold"]
            threshold_candidates = metric_train.loc[
                metric_train["metric_name"] == metric_name,
                [
                    "threshold",
                    "threshold_candidate_rank",
                    "threshold_candidate_count",
                    "threshold_source",
                ],
            ].dropna(subset=["threshold"])
            if threshold_candidates.empty:
                continue
            metric_test = evaluate_threshold_grid_on_frame(test, metric_name, threshold_candidates)
            if metric_test.empty:
                continue
            metric_test["fold_id"] = fold_id
            metric_test["split_role"] = "test"
            metric_test["search_stage"] = "atomic_screening"
            test_candidates.append(metric_test)
        if test_candidates:
            test_rows.extend(pd.concat(test_candidates, ignore_index=True).to_dict(orient="records"))
    candidate_df = pd.DataFrame(candidate_rows)
    test_df = pd.DataFrame(test_rows)
    expanded_candidate_df = pd.DataFrame()
    expanded_test_df = pd.DataFrame()
    lock_summary = pd.DataFrame()
    selection_rows: List[dict[str, Any]] = []
    winner_rule_json: str | None = None
    ranked_frontier = pd.DataFrame()
    lock_ranked = pd.DataFrame()
    if not test_df.empty:
        valid_test = test_df[test_df["candidate_valid_flag"] == 1].copy()
        if not valid_test.empty:
            aggregated = (
                valid_test.groupby(
                    [
                        "definition_name",
                        "candidate_type",
                        "candidate_group_key",
                        "metric_name",
                        "threshold",
                        "rule_json",
                        "rule_text",
                        "rule_size",
                        "rule_operator",
                        "threshold_source",
                    ],
                    dropna=False,
                    as_index=False,
                )
                .agg(
                    folds=("fold_id", "nunique"),
                    threshold_candidate_rank=("threshold_candidate_rank", "mean"),
                    threshold_candidate_count=("threshold_candidate_count", "max"),
                    test_gap_returned_active_post_label_m1=("gap_returned_active_post_label_m1", "mean"),
                    test_gap_returned_active_post_label_m2=("gap_returned_active_post_label_m2", "mean"),
                    test_gap_returned_active_post_label_m3=("gap_returned_active_post_label_m3", "mean"),
                    test_gap_active_days_post_label_3m=("gap_active_days_post_label_3m", "mean"),
                    test_gap_sustained_active_2of3_post_label=("gap_sustained_active_2of3_post_label", "mean"),
                    test_prevalence_entropy=("prevalence_entropy", "mean"),
                    test_monthly_prevalence_std=("monthly_prevalence_std", "mean"),
                    test_bootstrap_prevalence_ci_width=("bootstrap_prevalence_ci_width", "mean"),
                )
            )
            aggregated = attach_fold_gap_bootstrap_summary(
                summary=aggregated,
                source=valid_test,
                group_cols=[
                    "definition_name",
                    "candidate_type",
                    "candidate_group_key",
                    "metric_name",
                    "threshold",
                    "rule_json",
                    "rule_text",
                    "rule_size",
                    "rule_operator",
                    "threshold_source",
                ],
                source_gap_col="gap_sustained_active_2of3_post_label",
                target_prefix="test_gap_sustained_active_2of3_post_label",
            )
            aggregated = aggregated[aggregated["folds"] >= setup.MIN_OFFICIAL_VALID_OUTER_FOLDS].copy()
            if not aggregated.empty:
                metric_representatives = choose_metric_representatives(aggregated, setup.TEST_DEFINITION_OBJECTIVES)
                metric_representatives = attach_rule_label_signatures(search_frame, metric_representatives, reference_frame=search_frame)
                metric_representatives = choose_label_vector_representatives(metric_representatives, setup.TEST_DEFINITION_OBJECTIVES)
                frontier = pareto_front(metric_representatives, setup.TEST_DEFINITION_OBJECTIVES)
                ranked_frontier = rank_primary_definition_candidates(frontier)
                promoted_limit = max(1, int(setup.DEFINITION_A_PROMOTED_CANDIDATE_LIMIT))
                expansion_ranked = ranked_frontier.copy()
                atomic_topk = ranked_frontier.head(promoted_limit).copy()
                if str(setup.DEFINITION_A_STRATEGY).lower() == "screened_pairwise_compound_weighted" and not atomic_topk.empty:
                    expanded_train_rows: list[dict[str, Any]] = []
                    expanded_test_rows: list[dict[str, Any]] = []
                    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(search_frame), start=1):
                        train = search_frame.iloc[train_idx].copy()
                        test = search_frame.iloc[test_idx].copy()
                        expanded_train = expand_promoted_atomic_candidates(train, atomic_topk)
                        if expanded_train.empty:
                            continue
                        expanded_train["fold_id"] = fold_id
                        expanded_train["split_role"] = "train"
                        expanded_train_rows.extend(expanded_train.to_dict(orient="records"))
                        expanded_test = evaluate_candidate_rows_on_test(test, expanded_train)
                        if expanded_test.empty:
                            continue
                        expanded_test["fold_id"] = fold_id
                        expanded_test["split_role"] = "test"
                        expanded_test_rows.extend(expanded_test.to_dict(orient="records"))
                    expanded_candidate_df = pd.DataFrame(expanded_train_rows)
                    expanded_test_df = pd.DataFrame(expanded_test_rows)
                    if not expanded_test_df.empty:
                        valid_expanded = expanded_test_df[expanded_test_df["candidate_valid_flag"] == 1].copy()
                        if not valid_expanded.empty:
                            expanded_aggregated = aggregate_definition_test_eval(valid_expanded)
                            expanded_aggregated = expanded_aggregated[
                                expanded_aggregated["folds"] >= setup.MIN_OFFICIAL_VALID_OUTER_FOLDS
                            ].copy()
                            if not expanded_aggregated.empty:
                                expanded_metric_reps = choose_metric_representatives(
                                    expanded_aggregated,
                                    setup.TEST_DEFINITION_OBJECTIVES,
                                )
                                expanded_metric_reps = attach_rule_label_signatures(
                                    search_frame,
                                    expanded_metric_reps,
                                    reference_frame=search_frame,
                                )
                                expanded_metric_reps = choose_label_vector_representatives(
                                    expanded_metric_reps,
                                    setup.TEST_DEFINITION_OBJECTIVES,
                                )
                                expanded_frontier = pareto_front(expanded_metric_reps, setup.TEST_DEFINITION_OBJECTIVES)
                                expansion_ranked = rank_primary_definition_candidates(expanded_frontier)
                lock_topk = expansion_ranked.head(promoted_limit).copy()
                if not lock_topk.empty and not definition_lock_frame.empty:
                    lock_month_eval = evaluate_candidates_on_lock_months(
                        definition_lock_frame,
                        lock_topk,
                        reference_frame=search_frame,
                    )
                    lock_summary = summarize_lock_candidates(lock_month_eval)
                    if not lock_summary.empty:
                        lock_neighbor_summary = summarize_lock_neighbor_sensitivity(
                            definition_lock_frame,
                            lock_topk,
                            pd.concat([candidate_df, expanded_candidate_df], ignore_index=True) if not expanded_candidate_df.empty else candidate_df,
                            reference_frame=search_frame,
                        )
                        lock_ranked = lock_summary.merge(
                            lock_topk[
                                [
                                    "rule_json",
                                    "label_hash",
                                    "label_positives",
                                    "label_share_pct",
                                    "label_vector_group_size",
                                    "folds",
                                    "primary_selection_rank",
                                ]
                            ].drop_duplicates(),
                            on="rule_json",
                            how="left",
                        )
                        if not lock_neighbor_summary.empty:
                            lock_ranked = lock_ranked.merge(lock_neighbor_summary, on="rule_json", how="left")
                            lock_summary = lock_summary.merge(lock_neighbor_summary, on="rule_json", how="left")
                        lock_ranked = choose_final_definition_a_from_lock(lock_ranked)
                        if not lock_ranked.empty:
                            lock_summary = lock_summary.merge(
                                lock_ranked[
                                    [
                                        "rule_json",
                                        "lock_selection_rank",
                                        "lock_pareto_frontier_flag",
                                    ]
                                ].drop_duplicates(),
                                on="rule_json",
                                how="left",
                            )
                if lock_ranked.empty and definition_lock_frame.empty:
                    lock_ranked = lock_topk.copy()
                    if not lock_ranked.empty:
                        gate_spec = setup.get_definition_lock_bootstrap_gate_spec()
                        lock_ranked["lock_primary_gate_column_name"] = str(gate_spec["column_name"])
                        lock_ranked["lock_primary_gate_operator"] = str(gate_spec["operator"])
                        lock_ranked["lock_primary_gate_threshold"] = float(gate_spec["threshold"])
                        lock_ranked["lock_primary_gate_pass_flag"] = 1
                        lock_ranked["lock_primary_gate_reject_reason"] = ""
                        lock_ranked["lock_primary_gap_ci_positive_flag"] = 1
                        lock_ranked["lock_primary_gap_ci_reject_reason"] = ""
                        lock_ranked["lock_selection_rank"] = np.arange(1, len(lock_ranked) + 1)
                if not lock_ranked.empty:
                    winner_rule_json = str(lock_ranked.iloc[0]["rule_json"])
                    ranked_frontier = expansion_ranked.copy()
    if not expanded_candidate_df.empty:
        candidate_df = pd.concat([candidate_df, expanded_candidate_df], ignore_index=True)
    if not expanded_test_df.empty:
        test_df = pd.concat([test_df, expanded_test_df], ignore_index=True)
    if winner_rule_json and not ranked_frontier.empty:
        if str(setup.DEFINITION_A_STRATEGY).lower() == "screened_pairwise_compound_weighted":
            selection_basis = (
                "atomic_screening_on_development_outer_tests_then_pairwise_and_or_and_weighted_percentile_expansion_"
                "then_definition_lock_with_threshold_structural_and_weight_sensitivity_before_final_model_evaluation"
            )
        else:
            selection_basis = (
                "univariate_exact_development_outer_test_rank_aggregation_then_metric_pareto_front_"
                "then_definition_lock_pareto_with_local_threshold_sensitivity_before_final_model_evaluation"
            )
        promoted_rule_jsons = {str(value) for value in lock_ranked.get("rule_json", pd.Series(dtype=str)).tolist()}
        lock_meta_by_rule = {
            str(row["rule_json"]): row
            for row in lock_ranked.to_dict(orient="records")
        }
        freeze_reference_frame = pd.concat([search_frame, definition_lock_frame], ignore_index=True)
        for row in ranked_frontier.to_dict(orient="records"):
            rule_json = str(row.get("rule_json"))
            lock_meta = lock_meta_by_rule.get(rule_json, {})
            is_primary = rule_json == winner_rule_json
            is_promoted = rule_json in promoted_rule_jsons
            stored_rule_json = row["rule_json"]
            stored_rule_text = row["rule_text"]
            if is_primary:
                winner_rule = canonicalize_rule(json.loads(stored_rule_json))
                if winner_rule.get("kind") == "weighted":
                    frozen_winner_rule = freeze_rule(
                        winner_rule,
                        reference_frame=freeze_reference_frame if not freeze_reference_frame.empty else search_frame,
                    )
                    stored_rule_json = setup.stable_json(frozen_winner_rule)
                    stored_rule_text = build_rule_text(frozen_winner_rule)
            selection_rows.append(
                {
                    "definition_name": row["definition_name"],
                    "definition_group": "definition_a",
                    "official_status": (
                        "official_winner"
                        if is_primary
                        else ("sensitivity_lock_topk" if is_promoted else "sensitivity_development_frontier")
                    ),
                    "winner_flag": 1 if is_primary else 0,
                    "frontier_admissible_flag": 1,
                    "primary_selection_rank": int(row.get("primary_selection_rank", 0)),
                    "lock_selection_rank": lock_meta.get("lock_selection_rank", np.nan),
                    "promoted_candidate_limit": int(setup.DEFINITION_A_PROMOTED_CANDIDATE_LIMIT),
                    "candidate_type": row["candidate_type"],
                    "candidate_group_key": row.get("candidate_group_key"),
                    "metric_name": row["metric_name"],
                    "threshold": row["threshold"],
                    "rule_json": stored_rule_json,
                    "rule_text": stored_rule_text,
                    "rule_size": row["rule_size"],
                    "rule_operator": row["rule_operator"],
                    "threshold_source": row.get("threshold_source"),
                    "threshold_candidate_rank": row.get("threshold_candidate_rank"),
                    "threshold_candidate_count": row.get("threshold_candidate_count"),
                    "selection_basis": selection_basis,
                    "folds": row.get("folds"),
                    "label_hash": row.get("label_hash"),
                    "label_positives": row.get("label_positives"),
                    "label_share_pct": row.get("label_share_pct"),
                    "label_vector_group_size": row.get("label_vector_group_size"),
                    "lock_months": lock_meta.get("lock_months"),
                    "lock_max_gap_std": lock_meta.get("lock_max_gap_std"),
                    "lock_max_gap_jump": lock_meta.get("lock_max_gap_jump"),
                    "lock_prevalence_std": lock_meta.get("lock_prevalence_std"),
                    "lock_neighbor_count": lock_meta.get("lock_neighbor_count"),
                    "lock_min_label_jaccard": lock_meta.get("lock_min_label_jaccard"),
                    "lock_mean_label_jaccard": lock_meta.get("lock_mean_label_jaccard"),
                    "lock_max_neighbor_gap_delta": lock_meta.get("lock_max_neighbor_gap_delta"),
                    "lock_mean_neighbor_gap_delta": lock_meta.get("lock_mean_neighbor_gap_delta"),
                    "lock_max_neighbor_prevalence_delta": lock_meta.get("lock_max_neighbor_prevalence_delta"),
                    "lock_pareto_frontier_flag": lock_meta.get("lock_pareto_frontier_flag"),
                    "lock_primary_gate_column_name": lock_meta.get("lock_primary_gate_column_name"),
                    "lock_primary_gate_operator": lock_meta.get("lock_primary_gate_operator"),
                    "lock_primary_gate_threshold": lock_meta.get("lock_primary_gate_threshold"),
                    "lock_primary_gate_pass_flag": lock_meta.get("lock_primary_gate_pass_flag"),
                    "lock_primary_gate_reject_reason": lock_meta.get("lock_primary_gate_reject_reason"),
                    "lock_primary_gap_ci_positive_flag": lock_meta.get("lock_primary_gap_ci_positive_flag"),
                    "lock_primary_gap_ci_reject_reason": lock_meta.get("lock_primary_gap_ci_reject_reason"),
                    **{k: lock_meta.get(k) for k in [col for col in lock_meta.keys() if str(col).startswith("lock_gap_") or str(col).startswith("lock_prevalence_")]},
                    **{k: row.get(k) for k in setup.TEST_DEFINITION_OBJECTIVES},
                }
            )
    else:
        selection_rows.append(
            {
                "definition_name": "definition_a",
                "definition_group": "definition_a",
                "official_status": "not_official",
                "winner_flag": 0,
                "frontier_admissible_flag": 0,
                "primary_selection_rank": np.nan,
                "lock_selection_rank": np.nan,
                "promoted_candidate_limit": int(setup.DEFINITION_A_PROMOTED_CANDIDATE_LIMIT),
                "candidate_type": "univariate_exact",
                "candidate_group_key": None,
                "metric_name": None,
                "threshold": np.nan,
                "rule_json": setup.stable_json({}),
                "rule_text": "no_definition_a_candidate_survived_outer_test_selection",
                "rule_size": np.nan,
                "rule_operator": None,
                "threshold_source": None,
                "threshold_candidate_rank": np.nan,
                "threshold_candidate_count": np.nan,
                "selection_basis": "no_definition_a_candidate_survived_outer_test_selection",
            }
        )

    definition_b_spec = setup.get_definition_b_spec()
    selection_rows.append(
        {
            "definition_name": definition_b_spec["definition_name"],
            "definition_group": "definition_b",
            "official_status": "official_fixed_literal",
            "winner_flag": 0,
            "frontier_admissible_flag": 0,
            "primary_selection_rank": np.nan,
            "lock_selection_rank": np.nan,
            "promoted_candidate_limit": int(setup.DEFINITION_A_PROMOTED_CANDIDATE_LIMIT),
            "candidate_type": "fixed_literal",
            "candidate_group_key": "fixed_literal::definition_b",
            "metric_name": definition_b_spec["metric_name"],
            "threshold": float(definition_b_spec["threshold"]),
            "rule_json": setup.stable_json(
                {
                    "kind": "atomic",
                    "metric_name": definition_b_spec["metric_name"],
                    "operator": definition_b_spec["operator"],
                    "threshold": float(definition_b_spec["threshold"]),
                }
            ),
            "rule_text": definition_b_spec["rule_text"],
            "rule_size": 1,
            "rule_operator": definition_b_spec["operator"],
            "threshold_source": "fixed_literal",
            "threshold_candidate_rank": np.nan,
            "threshold_candidate_count": np.nan,
            "selection_basis": "literal_comparator_fixed_a_priori",
            "label_hash": None,
            "label_positives": np.nan,
            "label_share_pct": np.nan,
            "label_vector_group_size": np.nan,
        }
    )
    return candidate_df, test_df, lock_summary, pd.DataFrame(selection_rows)


def build_definition_search_stage_audit(
    candidate_df: pd.DataFrame,
    candidate_test_df: pd.DataFrame,
    definition_lock_df: pd.DataFrame,
    selection_df: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "stage_name",
        "search_stage",
        "candidate_type",
        "rows",
        "distinct_rules",
        "distinct_candidate_groups",
        "winner_rows",
        "official_rows",
    ]
    rows: list[dict[str, Any]] = []

    def _append_stage(stage_name: str, frame: pd.DataFrame, include_selection_flags: bool = False) -> None:
        if frame.empty or "candidate_type" not in frame.columns:
            return
        work = frame.copy()
        if "search_stage" not in work.columns:
            work["search_stage"] = np.nan
        group_cols = ["candidate_type", "search_stage"]
        grouped = work.groupby(group_cols, dropna=False)
        for (candidate_type, search_stage), group in grouped:
            rows.append(
                {
                    "stage_name": stage_name,
                    "search_stage": search_stage,
                    "candidate_type": str(candidate_type),
                    "rows": int(len(group)),
                    "distinct_rules": int(group["rule_json"].nunique()) if "rule_json" in group.columns else 0,
                    "distinct_candidate_groups": int(group["candidate_group_key"].nunique())
                    if "candidate_group_key" in group.columns
                    else 0,
                    "winner_rows": int(
                        pd.to_numeric(group.get("winner_flag"), errors="coerce").fillna(0).astype(int).sum()
                    )
                    if include_selection_flags
                    else 0,
                    "official_rows": int(group.get("official_status", pd.Series(dtype=object)).notna().sum())
                    if include_selection_flags
                    else 0,
                }
            )

    _append_stage("train_candidates", candidate_df)
    _append_stage("test_candidates", candidate_test_df)
    _append_stage("lock_summary", definition_lock_df)
    _append_stage("selection_rows", selection_df, include_selection_flags=True)

    audit = pd.DataFrame(rows, columns=columns)
    if audit.empty:
        return audit
    audit["search_stage"] = audit["search_stage"].astype(object)
    return audit.sort_values(["stage_name", "candidate_type", "search_stage"], kind="mergesort").reset_index(drop=True)


def evaluate_fixed_rule(frame: pd.DataFrame, rule: dict[str, Any], definition_name: str) -> pd.DataFrame:
    splitter = ExpandingMonthSplit(month_col="first_month", min_train_periods=1, test_periods=1, max_splits=setup.MAX_OUTER_TEST_MONTHS)
    rows: List[dict[str, Any]] = []
    ordered = frame.sort_values(["first_month", "teacher_unique_id"]).reset_index(drop=True)
    normalized = canonicalize_rule(rule)
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(ordered), start=1):
        for split_role, idx in (("train", train_idx), ("test", test_idx)):
            subset = ordered.iloc[idx].copy()
            label = apply_rule_to_frame(subset, normalized).to_numpy(dtype=int)
            diagnostics = compute_candidate_diagnostics(subset, label)
            rows.append(
            {
                "definition_name": definition_name,
                "candidate_type": "fixed_definition_evaluation",
                "candidate_group_key": candidate_group_key(normalized, "fixed_definition_evaluation"),
                "metric_name": rule_metric_signature(normalized),
                "threshold": float(normalized["threshold"]) if normalized["kind"] in {"atomic", "weighted"} else float("nan"),
                "rule_json": setup.stable_json(normalized),
                "rule_text": build_rule_text(normalized),
                "rule_size": rule_size(normalized),
                "rule_operator": rule_operator_label(normalized),
                "threshold_source": "fixed_definition",
                "threshold_candidate_rank": np.nan,
                "threshold_candidate_count": np.nan,
                "fold_id": fold_id,
                "split_role": split_role,
                **diagnostics,
                }
            )
    return pd.DataFrame(rows)


def compare_official_definitions(
    metrics: pd.DataFrame,
    selection_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare frozen official definitions on the common post-label validators.

    This is an A-vs-B definition comparison only. It does not use model outputs
    and it does not decide the model target.
    """
    frontier_columns = [
        "definition_name",
        "candidate_type",
        "candidate_group_key",
        "metric_name",
        "threshold",
        "rule_json",
        "rule_text",
        "rule_size",
        "rule_operator",
        "threshold_source",
        "folds",
        "threshold_candidate_rank",
        "threshold_candidate_count",
        "test_gap_returned_active_post_label_m1",
        "test_gap_returned_active_post_label_m2",
        "test_gap_returned_active_post_label_m3",
        "test_gap_active_days_post_label_3m",
        "test_gap_sustained_active_2of3_post_label",
        "test_prevalence_entropy",
        "test_monthly_prevalence_std",
        "test_bootstrap_prevalence_ci_width",
        "test_prevalence_std",
        "label_hash",
        "label_positives",
        "label_share_pct",
        "label_vector_group_size",
        "pareto_frontier_flag",
    ]
    frame = metrics.loc[metrics["full_followup_observed_flag"] == 1].copy()
    frame = setup.apply_official_population_filter(frame)
    definition_rows: List[pd.DataFrame] = []
    official_a = selection_df[
        (selection_df["definition_group"] == "definition_a")
        & selection_df["official_status"].str.startswith("official")
    ]
    for row in official_a.to_dict(orient="records"):
        rule = json.loads(row["rule_json"])
        definition_rows.append(evaluate_fixed_rule(frame, rule, build_definition_a_label_name(rule)))
    official_b = selection_df[
        (selection_df["definition_group"] == "definition_b")
        & selection_df["official_status"].str.startswith("official")
    ]
    if official_b.empty:
        definition_b_spec = setup.get_definition_b_spec()
        definition_b_rule = make_atomic_rule(
            metric_name=definition_b_spec["metric_name"],
            threshold=float(definition_b_spec["threshold"]),
            operator=definition_b_spec["operator"],
        )
        definition_rows.append(evaluate_fixed_rule(frame, definition_b_rule, "definition_b_label"))
    else:
        for row in official_b.to_dict(orient="records"):
            definition_b_rule = canonicalize_rule(json.loads(row["rule_json"]))
            definition_rows.append(evaluate_fixed_rule(frame, definition_b_rule, "definition_b_label"))
    fold_eval = pd.concat(definition_rows, ignore_index=True) if definition_rows else pd.DataFrame()
    if fold_eval.empty:
        return fold_eval, pd.DataFrame(columns=frontier_columns)
    test_eval = fold_eval[(fold_eval["split_role"] == "test") & (fold_eval["candidate_valid_flag"] == 1)].copy()
    if test_eval.empty:
        return fold_eval, pd.DataFrame(columns=frontier_columns)
    agg = (
        test_eval.groupby(
            [
                "definition_name",
                "candidate_type",
                "candidate_group_key",
                "metric_name",
                "threshold",
                "rule_json",
                "rule_text",
                "rule_size",
                "rule_operator",
                "threshold_source",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            folds=("fold_id", "nunique"),
            threshold_candidate_rank=("threshold_candidate_rank", "mean"),
            threshold_candidate_count=("threshold_candidate_count", "max"),
            test_gap_returned_active_post_label_m1=("gap_returned_active_post_label_m1", "mean"),
            test_gap_returned_active_post_label_m2=("gap_returned_active_post_label_m2", "mean"),
            test_gap_returned_active_post_label_m3=("gap_returned_active_post_label_m3", "mean"),
            test_gap_active_days_post_label_3m=("gap_active_days_post_label_3m", "mean"),
            test_gap_sustained_active_2of3_post_label=("gap_sustained_active_2of3_post_label", "mean"),
            test_prevalence_entropy=("prevalence_entropy", "mean"),
            test_monthly_prevalence_std=("monthly_prevalence_std", "mean"),
            test_bootstrap_prevalence_ci_width=("bootstrap_prevalence_ci_width", "mean"),
            test_prevalence_std=("prevalence", "std"),
        )
    )
    agg = attach_fold_gap_bootstrap_summary(
        summary=agg,
        source=test_eval,
        group_cols=[
            "definition_name",
            "candidate_type",
            "candidate_group_key",
            "metric_name",
            "threshold",
            "rule_json",
            "rule_text",
            "rule_size",
            "rule_operator",
            "threshold_source",
        ],
        source_gap_col="gap_sustained_active_2of3_post_label",
        target_prefix="test_gap_sustained_active_2of3_post_label",
    )
    agg = agg[agg["folds"] >= setup.MIN_OFFICIAL_VALID_OUTER_FOLDS].copy()
    if agg.empty:
        return fold_eval, pd.DataFrame(columns=frontier_columns)
    agg = attach_rule_label_signatures(frame, agg, reference_frame=frame)
    agg = choose_label_vector_representatives(agg, setup.TEST_DEFINITION_OBJECTIVES)
    frontier = pareto_front(agg, setup.TEST_DEFINITION_OBJECTIVES)
    return fold_eval, frontier


def build_definition_evaluability_audit(
    metrics: pd.DataFrame,
    selection_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = metrics.loc[metrics["full_followup_observed_flag"] == 1].copy()
    frame = setup.apply_official_population_filter(frame)
    frame = frame.sort_values(["first_month", "teacher_unique_id"]).reset_index(drop=True)
    _, lock_frame, final_eval_frame, _, lock_months, final_eval_months = setup.split_definition_workflow_frame(frame)
    rows: list[dict[str, Any]] = []
    official_a = selection_df[
        (selection_df["definition_group"] == "definition_a")
        & selection_df["official_status"].astype(str).str.startswith("official")
    ].copy()
    definition_rows: list[tuple[str, dict[str, Any]]] = []
    for row in official_a.to_dict(orient="records"):
        rule = canonicalize_rule(json.loads(str(row["rule_json"])))
        definition_rows.append((build_definition_a_label_name(rule), rule))
    definition_b = selection_df[
        (selection_df["definition_group"] == "definition_b")
        & selection_df["official_status"].astype(str).str.startswith("official")
    ].copy()
    if definition_b.empty:
        definition_b_spec = setup.get_definition_b_spec()
        definition_rows.append(
            (
                "definition_b_label",
                make_atomic_rule(
                    metric_name=definition_b_spec["metric_name"],
                    threshold=float(definition_b_spec["threshold"]),
                    operator=definition_b_spec["operator"],
                ),
            )
        )
    else:
        for row in definition_b.to_dict(orient="records"):
            definition_rows.append(("definition_b_label", canonicalize_rule(json.loads(str(row["rule_json"])))))
    for period_role, period_frame, period_months in (
        ("definition_lock_holdout", lock_frame, lock_months),
        ("official_model_evaluation_holdout", final_eval_frame, final_eval_months),
    ):
        if period_frame.empty:
            continue
        for month in period_months:
            subset = period_frame[
                pd.to_datetime(period_frame["first_month"], errors="coerce").dt.to_period("M").dt.to_timestamp() == pd.Timestamp(month)
            ].copy()
            if subset.empty:
                continue
            for definition_name, rule in definition_rows:
                label = apply_rule_to_frame(subset, rule).to_numpy(dtype=int)
                positives = int(label.sum())
                rows.append(
                    {
                        "definition_name": definition_name,
                        "period_role": period_role,
                        "month": pd.Timestamp(month),
                        "rows": int(len(subset)),
                        "positives": positives,
                        "negatives": int(len(subset) - positives),
                        "prevalence": float(positives / len(subset)) if len(subset) else float("nan"),
                        "two_class_flag": int(np.unique(label).size >= 2),
                        "meets_current_official_support_flag": int(
                            len(subset) >= setup.MIN_OFFICIAL_TEST_ROWS
                            and positives >= setup.MIN_OFFICIAL_TEST_POSITIVES
                            and (len(subset) - positives) >= setup.MIN_OFFICIAL_TEST_NEGATIVES
                        ),
                    }
                )
    audit = pd.DataFrame(
        rows,
        columns=[
            "definition_name",
            "period_role",
            "month",
            "rows",
            "positives",
            "negatives",
            "prevalence",
            "two_class_flag",
            "meets_current_official_support_flag",
        ],
    )
    if audit.empty:
        return audit, pd.DataFrame(
            columns=[
                "definition_name",
                "period_role",
                "months",
                "rows_total",
                "positives_total",
                "negatives_total",
                "mean_prevalence",
                "min_monthly_positives",
                "max_monthly_positives",
                "months_with_two_classes",
                "months_meeting_current_official_support",
            ]
        )
    summary = (
        audit.groupby(["definition_name", "period_role"], as_index=False)
        .agg(
            months=("month", "nunique"),
            rows_total=("rows", "sum"),
            positives_total=("positives", "sum"),
            negatives_total=("negatives", "sum"),
            mean_prevalence=("prevalence", "mean"),
            min_monthly_positives=("positives", "min"),
            max_monthly_positives=("positives", "max"),
            months_with_two_classes=("two_class_flag", "sum"),
            months_meeting_current_official_support=("meets_current_official_support_flag", "sum"),
        )
    )
    return audit, summary
