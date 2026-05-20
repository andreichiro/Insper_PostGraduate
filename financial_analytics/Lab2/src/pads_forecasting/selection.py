"""Model and old-data selection rules."""

from __future__ import annotations

import numpy as np
import pandas as pd

PRIMARY_COVID_SELECTION_MODES = {"none", "adjusted_target"}


def old_data_gate_decision(cv_summary: pd.DataFrame, selection_params: dict) -> pd.DataFrame:
    """Decide whether old pre-merger data is admissible."""

    threshold = float(selection_params.get("old_data_min_improvement_pct", 0.0)) / 100.0
    residual_ratio_limit = float(selection_params.get("residual_diagnostics_max_ratio", 1.25))
    primary_score = (
        "normal_mean_common_mase"
        if "normal_mean_common_mase" in cv_summary.columns
        else "normal_mean_mase"
    )
    stability_score = "cv_common_mase" if "cv_common_mase" in cv_summary.columns else "cv_mase"

    if cv_summary.empty or "target_strategy" not in cv_summary:
        return pd.DataFrame(
            [
                {
                    "target_strategy": "post_only",
                    "passed": True,
                    "normal_mean_mase": np.nan,
                    "normal_mean_common_mase": np.nan,
                    "improvement_vs_post_only_pct": 0.0,
                    "beats_raw_full": False,
                    "beats_seasonal_naive": False,
                    "cv_mase": np.nan,
                    "cv_common_mase": np.nan,
                    "train_valid_ratio": np.nan,
                    "residual_abs_mean": np.nan,
                    "residual_ratio_vs_post_only": np.nan,
                    "residual_diagnostics_no_worse_than_post_only": False,
                    "decision": "fallback: no successful old-data gate model runs",
                }
            ]
        )

    agg_spec = {
        "normal_mean_mase": ("normal_mean_mase", "mean"),
        "normal_mean_common_mase": (primary_score, "mean"),
        "mean_mase": ("mean_mase", "mean"),
        "cv_mase": ("cv_mase", "mean"),
        "cv_common_mase": (stability_score, "mean"),
        "rel_mae": ("mean_relative_mae_vs_seasonal_naive", "mean"),
        "train_valid_ratio": ("mean_train_valid_ratio", "mean"),
    }
    if "mean_train_residual_abs_mean" in cv_summary:
        agg_spec["residual_abs_mean"] = ("mean_train_residual_abs_mean", "mean")
    if "mean_train_residual_std" in cv_summary:
        agg_spec["residual_std"] = ("mean_train_residual_std", "mean")

    rows = []
    by_strategy = cv_summary.groupby("target_strategy", dropna=False).agg(**agg_spec)
    post = (
        by_strategy.loc["post_only", "normal_mean_common_mase"]
        if "post_only" in by_strategy.index
        else np.nan
    )
    raw = (
        by_strategy.loc["raw_full", "normal_mean_common_mase"]
        if "raw_full" in by_strategy.index
        else np.nan
    )
    post_residual_abs = (
        by_strategy.loc["post_only", "residual_abs_mean"]
        if "post_only" in by_strategy.index and "residual_abs_mean" in by_strategy
        else np.nan
    )

    for strategy in ["proforma_sum", "calibrated_alpha"]:
        if strategy not in by_strategy.index:
            continue
        row = by_strategy.loc[strategy]
        improvement_vs_post = (
            (post - row["normal_mean_common_mase"]) / post
            if np.isfinite(post) and post
            else -np.inf
        )
        beats_raw = bool(np.isfinite(raw) and row["normal_mean_common_mase"] < raw)
        beats_snaive = row["rel_mae"] < 1.0
        stable = np.isfinite(row["cv_common_mase"])
        overfit_ok = row["train_valid_ratio"] < selection_params["train_valid_ratio_reject"]
        residual_ratio = (
            row["residual_abs_mean"] / post_residual_abs
            if "residual_abs_mean" in by_strategy
            and np.isfinite(post_residual_abs)
            and post_residual_abs > 0
            else np.nan
        )
        residual_ok = bool(
            not np.isfinite(residual_ratio) or residual_ratio <= residual_ratio_limit
        )
        passed = bool(
            improvement_vs_post > threshold
            and beats_raw
            and beats_snaive
            and stable
            and overfit_ok
            and residual_ok
        )
        rows.append(
            {
                "target_strategy": strategy,
                "passed": passed,
                "normal_mean_mase": row["normal_mean_mase"],
                "normal_mean_common_mase": row["normal_mean_common_mase"],
                "improvement_vs_post_only_pct": improvement_vs_post * 100,
                "beats_raw_full": bool(beats_raw),
                "beats_seasonal_naive": bool(beats_snaive),
                "cv_mase": row["cv_mase"],
                "cv_common_mase": row["cv_common_mase"],
                "train_valid_ratio": row["train_valid_ratio"],
                "residual_abs_mean": row.get("residual_abs_mean", np.nan),
                "residual_ratio_vs_post_only": residual_ratio,
                "residual_diagnostics_no_worse_than_post_only": residual_ok,
                "decision": "old data admissible"
                if passed
                else "old data not proven for this strategy",
            }
        )

    if not rows:
        rows.append(
            {
                "target_strategy": "post_only",
                "passed": True,
                "normal_mean_mase": post,
                "normal_mean_common_mase": post,
                "improvement_vs_post_only_pct": 0.0,
                "beats_raw_full": False,
                "beats_seasonal_naive": False,
                "cv_mase": np.nan,
                "cv_common_mase": np.nan,
                "train_valid_ratio": np.nan,
                "residual_abs_mean": np.nan,
                "residual_ratio_vs_post_only": np.nan,
                "residual_diagnostics_no_worse_than_post_only": False,
                "decision": "fallback: no old-data strategy passed",
            }
        )
    return pd.DataFrame(rows)


def admissible_strategies(old_data_gate: pd.DataFrame) -> list[str]:
    """Return target strategies allowed into final model competition."""

    if "record_type" in old_data_gate:
        decisions = old_data_gate[old_data_gate["record_type"].eq("decision")].copy()
    else:
        decisions = old_data_gate.copy()
    passed_mask = decisions["passed"].astype(str).str.lower().isin(["true", "1"])
    passed = decisions.loc[passed_mask, "target_strategy"].tolist()
    strategies = ["post_only"]
    for strategy in ["proforma_sum", "calibrated_alpha"]:
        if strategy in passed:
            strategies.append(strategy)
        elif _has_model_level_old_data_evidence(old_data_gate, strategy):
            strategies.append(strategy)
    return list(dict.fromkeys(strategies))


def _has_model_level_old_data_evidence(old_data_gate: pd.DataFrame, strategy: str) -> bool:
    """Allow an old-data strategy when a model-specific pair has predictive evidence.

    The Stage A strategy-level average is still reported, but it should not veto a
    pro-forma or calibrated-alpha challenger that is clearly useful for a specific
    model family. The comparison uses fixed-target normal-fold MASE where available.
    """

    if strategy not in {"proforma_sum", "calibrated_alpha"}:
        return False
    if old_data_gate.empty or "target_strategy" not in old_data_gate:
        return False
    if "record_type" in old_data_gate:
        summary = old_data_gate[old_data_gate["record_type"].eq("summary")].copy()
    else:
        summary = old_data_gate.copy()
    needed = {"target_strategy", "model_id"}
    if summary.empty or not needed.issubset(summary.columns):
        return False
    metric = (
        "normal_mean_common_mase"
        if "normal_mean_common_mase" in summary.columns
        else "normal_mean_mase"
    )
    if metric not in summary.columns:
        return False
    candidate_keys = [col for col in ["model_family", "model_id"] if col in summary.columns]
    if not candidate_keys:
        return False
    optional_cols = [
        col
        for col in ["mean_relative_mae_vs_seasonal_naive", "mean_train_valid_ratio"]
        if col in summary.columns
    ]
    scores = summary[summary["target_strategy"].isin(["post_only", "raw_full", strategy])][
        [*candidate_keys, "target_strategy", metric, *optional_cols]
    ].copy()
    scores[metric] = pd.to_numeric(scores[metric], errors="coerce")
    scores = scores.dropna(subset=[metric])
    if scores.empty or strategy not in set(scores["target_strategy"]):
        return False
    wide = scores.pivot_table(
        index=candidate_keys,
        columns="target_strategy",
        values=metric,
        aggfunc="min",
    )
    if strategy not in wide or "post_only" not in wide:
        return False
    comparators = ["post_only"]
    if "raw_full" in wide:
        comparators.append("raw_full")
    best_comparator = wide[comparators].min(axis=1)
    predictive = wide[strategy] < best_comparator

    if "mean_relative_mae_vs_seasonal_naive" in scores:
        rel = scores.pivot_table(
            index=candidate_keys,
            columns="target_strategy",
            values="mean_relative_mae_vs_seasonal_naive",
            aggfunc="min",
        )
        if strategy in rel:
            predictive = predictive & rel[strategy].le(1.0)

    if "mean_train_valid_ratio" in scores:
        ratio = scores.pivot_table(
            index=candidate_keys,
            columns="target_strategy",
            values="mean_train_valid_ratio",
            aggfunc="min",
        )
        if strategy in ratio:
            predictive = predictive & ratio[strategy].replace([np.inf, -np.inf], np.nan).notna()

    return bool(predictive.any())


def select_final_model(cv_summary: pd.DataFrame, selection_params: dict) -> pd.DataFrame:
    """Rank final candidates by the single 2024-hidden-target MASE objective."""

    if cv_summary.empty:
        return pd.DataFrame()
    ranked = cv_summary.copy()
    if "model_family" not in ranked:
        ranked["model_family"] = ranked["model_id"].map(_infer_model_family)
    if "complexity" not in ranked:
        ranked["complexity"] = ranked["model_family"].map(_infer_complexity)
    fallback_metric_map = {
        "normal_mean_common_mase": "normal_mean_mase",
        "mean_common_mase": "mean_mase",
        "cv_common_mase": "cv_mase",
        "max_common_mase": "max_mase",
    }
    for target, source in fallback_metric_map.items():
        if target not in ranked and source in ranked:
            ranked[target] = ranked[source]

    required_metrics = [
        "normal_mean_common_mase",
        "mean_common_mase",
        "cv_mae",
        "cv_rmse",
        "cv_common_mase",
        "max_common_mase",
        "mean_relative_mae_vs_seasonal_naive",
        "mean_train_valid_ratio",
        "folds",
    ]
    full_fold_count = int(ranked["folds"].max()) if "folds" in ranked else 0
    complete_metrics = ranked[required_metrics].notna().all(axis=1)
    complete_folds = ranked["folds"].eq(full_fold_count)
    ranked = _apply_data_driven_stability_filter(ranked, selection_params)
    if "covid_mode" in ranked:
        primary_covid_mode = ranked["covid_mode"].astype(str).isin(PRIMARY_COVID_SELECTION_MODES)
    else:
        primary_covid_mode = pd.Series(True, index=ranked.index)
    ranked["baseline_passed"] = ranked["mean_relative_mae_vs_seasonal_naive"].le(1.0)
    train_valid_ratio_reject = float(selection_params.get("train_valid_ratio_reject", 3.0))
    ranked["overfit_passed"] = ranked["mean_train_valid_ratio"].lt(train_valid_ratio_reject)
    ranked["eligible_for_selection"] = (
        complete_metrics
        & complete_folds
        & primary_covid_mode
        & ranked["baseline_passed"]
        & ranked["overfit_passed"]
    )
    ranked["robustness_eligible_for_selection"] = ranked["eligible_for_selection"] & ranked[
        "stability_passed"
    ].astype(bool)
    ranked["selected"] = False
    ranked["selected_with_robustness"] = False
    ranked["selection_reason"] = "not selected"
    ranked.loc[~complete_metrics | ~complete_folds, "selection_reason"] = (
        "ineligible: incomplete CV metrics or fold coverage"
    )
    ranked.loc[
        complete_metrics & complete_folds & ~primary_covid_mode,
        "selection_reason",
    ] = "ineligible: native COVID dummy mode is sensitivity-only"
    ranked.loc[
        complete_metrics & complete_folds & primary_covid_mode & ~ranked["baseline_passed"],
        "selection_reason",
    ] = "ineligible: does not beat SeasonalNaive baseline"
    ranked.loc[
        complete_metrics
        & complete_folds
        & primary_covid_mode
        & ranked["baseline_passed"]
        & ~ranked["overfit_passed"],
        "selection_reason",
    ] = "ineligible: train-validation gap indicates overfit risk"

    eligible = (
        ranked[ranked["eligible_for_selection"]]
        .sort_values(
            [
                "normal_mean_common_mase",
                "normal_mean_mase",
                "mean_common_mase",
                "cv_common_mase",
            ]
        )
        .copy()
    )
    ranked["rank"] = np.nan
    ranked.loc[eligible.index, "rank"] = np.arange(1, len(eligible) + 1)
    if eligible.empty:
        return ranked.sort_values(
            ["eligible_for_selection", "normal_mean_common_mase"],
            ascending=[False, True],
        )

    selected_idx = eligible.index[0]

    ranked.loc[selected_idx, "selected"] = True
    ranked.loc[selected_idx, "selection_reason"] = (
        "selected by the single production objective: lowest fixed-target MASE on "
        "2024-like post-merger validation folds after complete-CV, primary-COVID, "
        "SeasonalNaive, and overfit gates"
    )
    return ranked.sort_values(
        ["eligible_for_selection", "rank", "normal_mean_common_mase"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def _apply_data_driven_stability_filter(
    ranked: pd.DataFrame, selection_params: dict
) -> pd.DataFrame:
    """Flag fold-stability outliers with robust candidate-pool fences."""

    out = ranked.copy()
    stability_cols = [
        col
        for col in [
            "cv_mae",
            "cv_rmse",
            "cv_common_mase",
            "cv_mase",
            "cv_relative_mae_vs_seasonal_naive",
        ]
        if col in out.columns
    ]
    out["stability_passed"] = True
    out["stability_reason"] = "stability within robust candidate-pool fences"
    out["stability_rule"] = "tukey_iqr_candidate_pool"
    out["stability_upper_fence"] = np.nan
    out["stability_lower_fence"] = np.nan

    if not stability_cols:
        out["stability_reason"] = "no stability metrics available"
        return out

    multiplier = float(selection_params.get("stability_iqr_multiplier", 1.5))
    epsilon = float(selection_params.get("stability_min_fold_variation_epsilon", 1e-12))
    for column in stability_cols:
        values = out[column].replace([np.inf, -np.inf], np.nan).dropna().astype(float)
        if len(values) < 4:
            continue
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        iqr = q3 - q1
        lower = max(0.0, q1 - multiplier * iqr)
        upper = q3 + multiplier * iqr
        high = out[column] > upper
        low = out[column] < lower
        out.loc[high | low, "stability_passed"] = False
        out.loc[high, "stability_reason"] = (
            f"ineligible: {column} is a high fold-variation outlier "
            f"under Tukey IQR candidate-pool screening"
        )
        out.loc[low, "stability_reason"] = (
            f"ineligible: {column} is a low fold-variation outlier "
            f"under Tukey IQR candidate-pool screening"
        )
        out.loc[high | low, "stability_upper_fence"] = upper
        out.loc[high | low, "stability_lower_fence"] = lower

    flat = out[stability_cols].abs().le(epsilon).all(axis=1)
    out.loc[flat, "stability_passed"] = False
    out.loc[flat, "stability_reason"] = (
        "ineligible: MAE/RMSE/MASE fold variation is exactly flat, "
        "which is suspicious in temporal CV"
    )
    return out


def _infer_model_family(model_id: str) -> str:
    """Infer model family for legacy summaries that do not carry model_family."""

    for family in [
        "seasonal_naive",
        "sarimax",
        "lightgbm",
        "prophet",
        "elasticnet",
        "ridge",
        "ets",
        "bvar",
    ]:
        if str(model_id).startswith(family) or family in str(model_id):
            return family
    return "unknown"


def _infer_complexity(model_family: str) -> str:
    """Map model family to selection complexity bucket."""

    if model_family in {"seasonal_naive", "ets", "sarimax"}:
        return "simple"
    if model_family in {"ridge", "elasticnet"}:
        return "moderate"
    return "complex"


def _is_complex(row: pd.Series) -> bool:
    """Return whether a row should clear the complex-model improvement hurdle."""

    return str(row.get("complexity", _infer_complexity(row.get("model_family", "")))) == "complex"
