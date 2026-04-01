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
from .modeling import bootstrap_prevalence_ci_width_from_counts, pareto_front


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
    diagnostics["candidate_valid_flag"] = int(working["_label"].nunique() == 2)
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


def apply_rule_to_frame(frame: pd.DataFrame, rule: dict[str, Any]) -> pd.Series:
    return setup.apply_rule_to_frame(frame, rule)


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
) -> dict[str, Any]:
    normalized = canonicalize_rule(rule)
    label = apply_rule_to_frame(frame, normalized).to_numpy(dtype=int)
    diagnostics = compute_candidate_diagnostics(frame, label)
    metric_names = extract_rule_metric_names(normalized)
    return {
        "definition_name": definition_name,
        "candidate_type": candidate_type,
        "metric_name": rule_metric_signature(normalized),
        "threshold": float(normalized["threshold"]) if normalized["kind"] == "atomic" else float("nan"),
        "rule_json": setup.stable_json(normalized),
        "rule_text": build_rule_text(normalized),
        "rule_size": rule_size(normalized),
        "rule_operator": rule_operator_label(normalized),
        "metric_count": len(metric_names),
        "gain_score": candidate_gain_score(diagnostics),
        **diagnostics,
    }


def enumerate_univariate_candidates(train: pd.DataFrame, candidate_metric_registry: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict[str, Any]] = []
    if train.empty:
        return pd.DataFrame(rows)
    month_totals = train.groupby("first_month").size().sort_index()
    for metric_row in candidate_metric_registry.to_dict(orient="records"):
        if int(metric_row.get("definition_a_candidate_flag", 0)) != 1:
            continue
        metric_name = metric_row["metric_name"]
        work = train[["first_month"] + setup.EXTERNAL_VALIDATORS + [metric_name]].copy()
        work[metric_name] = pd.to_numeric(work[metric_name], errors="coerce").fillna(0)
        grouped = work.groupby(metric_name, dropna=False, sort=True)
        counts = grouped.size().sort_index(ascending=False)
        thresholds = counts.index.to_numpy(dtype=float)
        cum_pos = counts.cumsum().to_numpy(dtype=float)
        total_n = float(len(work))
        neg = total_n - cum_pos
        monthly_by_value = (
            work.groupby([metric_name, "first_month"], dropna=False)
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
            rule = make_atomic_rule(metric_name, float(threshold))
            rows.append(
                {
                    "definition_name": "definition_a",
                    "candidate_type": "univariate_exact_threshold",
                    "metric_name": metric_name,
                    "threshold": float(threshold),
                    "rule_json": setup.stable_json(rule),
                    "rule_text": build_rule_text(rule),
                    "rule_size": 1,
                    "rule_operator": ">=",
                    "metric_count": 1,
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
                    "candidate_valid_flag": int(positives > 0 and negatives > 0),
                    **{f"gap_{validator}": float(validator_gaps[validator][idx]) for validator in setup.EXTERNAL_VALIDATORS},
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["gain_score"] = df.apply(lambda row: candidate_gain_score(row.to_dict()), axis=1)
    return df


def evaluate_threshold_grid_on_frame(
    frame: pd.DataFrame,
    metric_name: str,
    thresholds: Sequence[float],
    definition_name: str = "definition_a",
    candidate_type: str = "univariate_exact_threshold",
) -> pd.DataFrame:
    rows = []
    for threshold in sorted({float(value) for value in thresholds}):
        rule = make_atomic_rule(metric_name, float(threshold))
        row = make_rule_candidate_row(frame, rule, candidate_type=candidate_type, definition_name=definition_name)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_definition_test_eval(test_eval: pd.DataFrame) -> pd.DataFrame:
    if test_eval.empty:
        return pd.DataFrame()
    return (
        test_eval.groupby(
            ["definition_name", "candidate_type", "metric_name", "threshold", "rule_json", "rule_text", "rule_size", "rule_operator"],
            dropna=False,
            as_index=False,
        )
        .agg(
            folds=("fold_id", "nunique"),
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
    for metric_name, group in candidate_summary.groupby("metric_name", dropna=False):
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


def attach_rule_label_signatures(frame: pd.DataFrame, candidate_summary: pd.DataFrame) -> pd.DataFrame:
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
            label = apply_rule_to_frame(ordered, rule).to_numpy(dtype=int)
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


def enumerate_definition_a_candidates(train: pd.DataFrame, candidate_metric_registry: pd.DataFrame) -> pd.DataFrame:
    strategy = str(setup.DEFINITION_A_STRATEGY).lower()
    if strategy != "univariate_exact":
        raise ValueError(f"Unsupported Definition A strategy: {setup.DEFINITION_A_STRATEGY}")
    return enumerate_univariate_candidates(train, candidate_metric_registry)


def evaluate_candidate_rows_on_test(test: pd.DataFrame, train_candidates: pd.DataFrame) -> pd.DataFrame:
    if train_candidates.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for row in train_candidates.drop_duplicates(subset=["rule_json"]).to_dict(orient="records"):
        rule = canonicalize_rule(json.loads(row["rule_json"]))
        test_row = make_rule_candidate_row(
            test,
            rule,
            candidate_type=str(row.get("candidate_type", "definition_a_candidate")),
            definition_name="definition_a",
        )
        rows.append(test_row)
    return pd.DataFrame(rows)


def build_definition_search(metrics: pd.DataFrame, candidate_metric_registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = metrics.loc[metrics["full_followup_observed_flag"] == 1].copy().sort_values(["first_month", "teacher_unique_id"]).reset_index(drop=True)
    splitter = ExpandingMonthSplit(month_col="first_month", min_train_periods=1, test_periods=1, max_splits=setup.MAX_OUTER_TEST_MONTHS)
    candidate_rows: List[dict[str, Any]] = []
    test_rows: List[dict[str, Any]] = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(frame), start=1):
        print(f"[definition_search] fold {fold_id} / {splitter.get_n_splits(frame)}", flush=True)
        train = frame.iloc[train_idx].copy()
        test = frame.iloc[test_idx].copy()
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
            thresholds = metric_train.loc[metric_train["metric_name"] == metric_name, "threshold"].dropna().tolist()
            if not thresholds:
                continue
            metric_test = evaluate_threshold_grid_on_frame(test, metric_name, thresholds)
            if metric_test.empty:
                continue
            metric_test["fold_id"] = fold_id
            metric_test["split_role"] = "test"
            test_candidates.append(metric_test)
        if test_candidates:
            test_rows.extend(pd.concat(test_candidates, ignore_index=True).to_dict(orient="records"))
    candidate_df = pd.DataFrame(candidate_rows)
    test_df = pd.DataFrame(test_rows)
    selection_rows: List[dict[str, Any]] = []
    official_a_rows: List[dict[str, Any]] = []
    if not test_df.empty:
        valid_test = test_df[test_df["candidate_valid_flag"] == 1].copy()
        if not valid_test.empty:
            aggregated = (
                valid_test.groupby(
                    ["definition_name", "candidate_type", "metric_name", "threshold", "rule_json", "rule_text", "rule_size", "rule_operator"],
                    dropna=False,
                    as_index=False,
                )
                .agg(
                    folds=("fold_id", "nunique"),
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
            aggregated = aggregated[aggregated["folds"] >= setup.MIN_OFFICIAL_VALID_OUTER_FOLDS].copy()
            if not aggregated.empty:
                metric_representatives = choose_metric_representatives(aggregated, setup.TEST_DEFINITION_OBJECTIVES)
                metric_representatives = attach_rule_label_signatures(frame, metric_representatives)
                metric_representatives = choose_label_vector_representatives(metric_representatives, setup.TEST_DEFINITION_OBJECTIVES)
                frontier = pareto_front(metric_representatives, setup.TEST_DEFINITION_OBJECTIVES)
                official_a_rows = frontier[frontier["pareto_frontier_flag"] == 1].to_dict(orient="records")
    if official_a_rows:
        selection_basis = "univariate_exact_outer_test_rank_aggregation_then_metric_pareto_front"
        for row in official_a_rows:
            selection_rows.append(
                {
                    "definition_name": row["definition_name"],
                    "definition_group": "definition_a",
                    "official_status": "official_admissible" if len(official_a_rows) > 1 else "official_unique",
                    "winner_flag": 1 if len(official_a_rows) == 1 else 0,
                    "candidate_type": row["candidate_type"],
                    "metric_name": row["metric_name"],
                    "threshold": row["threshold"],
                    "rule_json": row["rule_json"],
                    "rule_text": row["rule_text"],
                    "rule_size": row["rule_size"],
                    "rule_operator": row["rule_operator"],
                    "selection_basis": selection_basis,
                    "folds": row.get("folds"),
                    "label_hash": row.get("label_hash"),
                    "label_positives": row.get("label_positives"),
                    "label_share_pct": row.get("label_share_pct"),
                    "label_vector_group_size": row.get("label_vector_group_size"),
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
                "candidate_type": "univariate_exact",
                "metric_name": None,
                "threshold": np.nan,
                "rule_json": setup.stable_json({}),
                "rule_text": "no_definition_a_candidate_survived_outer_test_selection",
                "rule_size": np.nan,
                "rule_operator": None,
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
            "candidate_type": "fixed_literal",
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
            "selection_basis": "literal_comparator_fixed_a_priori",
            "label_hash": None,
            "label_positives": np.nan,
            "label_share_pct": np.nan,
            "label_vector_group_size": np.nan,
        }
    )
    return candidate_df, test_df, pd.DataFrame(selection_rows)


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
                    "metric_name": rule_metric_signature(normalized),
                    "threshold": float(normalized["threshold"]) if normalized["kind"] == "atomic" else float("nan"),
                    "rule_json": setup.stable_json(normalized),
                    "rule_text": build_rule_text(normalized),
                    "rule_size": rule_size(normalized),
                    "rule_operator": rule_operator_label(normalized),
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
    frontier_columns = [
        "definition_name",
        "candidate_type",
        "metric_name",
        "threshold",
        "rule_json",
        "rule_text",
        "rule_size",
        "rule_operator",
        "folds",
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
            ["definition_name", "candidate_type", "metric_name", "threshold", "rule_json", "rule_text", "rule_size", "rule_operator"],
            dropna=False,
            as_index=False,
        )
        .agg(
            folds=("fold_id", "nunique"),
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
    agg = agg[agg["folds"] >= setup.MIN_OFFICIAL_VALID_OUTER_FOLDS].copy()
    if agg.empty:
        return fold_eval, pd.DataFrame(columns=frontier_columns)
    agg = attach_rule_label_signatures(frame, agg)
    agg = choose_label_vector_representatives(agg, setup.TEST_DEFINITION_OBJECTIVES)
    frontier = pareto_front(agg, setup.TEST_DEFINITION_OBJECTIVES)
    return fold_eval, frontier
