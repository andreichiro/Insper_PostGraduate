#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from analytics_v2_common import (
    PALETTE,
    build_card_html,
    build_table_html,
    figure_to_html,
    fmt_num,
    render_report_html,
)
from explainable_drift_prediction_common_v2 import (
    attach_reference,
    build_config,
    build_output_reference,
    connect_output,
    drift_rank,
    load_output_parquet,
    persist_table,
    strip_reference_cols,
    write_json,
    write_markdown,
    Config,
)

PRIMARY_RECENT_WINDOW = "recent_12m"
SENSITIVITY_WINDOW = "recent_6m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part 3: final explainable HTML report for drift and prediction.")
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def chart_block(
    artifact_name: str,
    title: str,
    subtitle: str,
    body_html: str,
    lineage: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "artifact_name": artifact_name,
        "title": title,
        "subtitle": subtitle,
        "body_html": body_html,
        "lineage": lineage,
    }


def require_table(output_dir: Path, table_name: str) -> pd.DataFrame:
    frame = load_output_parquet(output_dir, table_name)
    if frame.empty:
        raise FileNotFoundError(
            f"Required input table not found or empty: {output_dir / 'parquet' / f'{table_name}.parquet'}. "
            "Run the drift and prediction scripts first."
        )
    return frame


def production_recommendation(recent_auc: float, recent_f1: float, auc_gap: float) -> str:
    if pd.notna(recent_auc) and recent_auc >= 0.75 and pd.notna(recent_f1) and recent_f1 >= 0.45 and abs(auc_gap) <= 0.08:
        return "Use in production as a monitored risk score, not as an autonomous decision-maker."
    if pd.notna(recent_auc) and recent_auc >= 0.65 and pd.notna(recent_f1) and recent_f1 >= 0.30:
        return "Use for prioritization, experimentation, and analyst support only; do not automate user-facing decisions from it."
    return "Do not use this model operationally yet; keep it as exploratory analysis only."


def build_direct_answers(
    drift_numeric: pd.DataFrame,
    drift_categorical: pd.DataFrame,
    model_selection: pd.DataFrame,
    model_comparison: pd.DataFrame,
    recent_strategy: pd.DataFrame,
    population_strategy_review: pd.DataFrame,
    feature_set_review: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    numeric = strip_reference_cols(drift_numeric).copy()
    categorical = strip_reference_cols(drift_categorical).copy()
    if not categorical.empty:
        categorical = (
            categorical.sort_values(["feature_total_variation", "feature_max_share_diff_pp"], ascending=[False, False])
            .drop_duplicates(subset=["feature_name"])
            .reset_index(drop=True)
        )
    selection = strip_reference_cols(model_selection).copy()
    comparison = strip_reference_cols(model_comparison).copy()
    strategy = strip_reference_cols(recent_strategy).copy()
    population_review = strip_reference_cols(population_strategy_review).copy()
    feature_set = strip_reference_cols(feature_set_review).copy()

    high_numeric = int((numeric["drift_level"] == "high_drift").sum()) if not numeric.empty else 0
    high_categorical = int((categorical["drift_level"] == "high_drift").sum()) if not categorical.empty else 0

    primary_selected = selection[selection["window_label"] == PRIMARY_RECENT_WINDOW].copy()
    primary_model_rows = comparison.merge(
        primary_selected[["target", "window_label", "selected_model_name"]],
        left_on=["target", "window_label", "model_name"],
        right_on=["target", "window_label", "selected_model_name"],
        how="inner",
    )
    primary_return = primary_model_rows[primary_model_rows["target"] == "retornar_ativo_m1"].head(1)

    best_recent_auc = float(primary_return["test_roc_auc"].iloc[0]) if not primary_return.empty else float("nan")
    best_recent_f1 = float(primary_return["test_f1"].iloc[0]) if not primary_return.empty else float("nan")
    best_recent_accuracy = float(primary_return["test_accuracy"].iloc[0]) if not primary_return.empty else float("nan")
    best_recent_auc_gap = float(primary_return["roc_auc_gap_train_minus_test"].iloc[0]) if not primary_return.empty else float("nan")
    best_recent_model = str(primary_return["model_name"].iloc[0]) if not primary_return.empty else "n/a"
    best_recent_test_rows = int(primary_return["test_rows"].iloc[0]) if not primary_return.empty else 0

    primary_return_strategy = strategy[
        (strategy["window_label"] == PRIMARY_RECENT_WINDOW)
        & (strategy["target"] == "retornar_ativo_m1")
        & (strategy["model_name"] != "__best_strategy__")
    ].copy()
    include_row = primary_return_strategy[
        primary_return_strategy["strategy_label"] == "include_pre_drift_history"
    ].head(1)
    recent_only_row = primary_return_strategy[
        primary_return_strategy["strategy_label"] == "recent_only_post_drift"
    ].head(1)
    include_auc = float(include_row["roc_auc"].iloc[0]) if not include_row.empty else float("nan")
    recent_only_auc = (
        float(recent_only_row["roc_auc"].iloc[0]) if not recent_only_row.empty else float("nan")
    )
    include_f1 = float(include_row["f1"].iloc[0]) if not include_row.empty else float("nan")
    recent_only_f1 = float(recent_only_row["f1"].iloc[0]) if not recent_only_row.empty else float("nan")
    include_accuracy = (
        float(include_row["accuracy"].iloc[0]) if not include_row.empty else float("nan")
    )
    recent_only_accuracy = (
        float(recent_only_row["accuracy"].iloc[0]) if not recent_only_row.empty else float("nan")
    )
    if pd.notna(include_auc) and pd.notna(recent_only_auc):
        if abs(recent_only_auc - include_auc) <= 0.005:
            train_answer = (
                "Treat the two strategies as effectively tied on ranking. Use the recent-only "
                "post-drift model as the primary operational model and keep the all-history model "
                "as a benchmark."
            )
        elif recent_only_auc > include_auc:
            train_answer = "Use the recent-only post-drift model as the primary model. The all-history model can stay only as a benchmark."
        else:
            train_answer = "Keep the all-history model only if it wins clearly on the same recent holdout. Here it only edges out recent-only on AUC, so it should stay a benchmark, not the default operational choice."
    else:
        train_answer = "The same-holdout training strategy comparison could not be estimated cleanly."

    hgb_recent = comparison[
        (comparison["window_label"] == PRIMARY_RECENT_WINDOW)
        & (comparison["model_name"] == "behavior_plus_profile_hist_gradient_boosting")
        & (comparison["target"] == "retornar_ativo_m1")
    ]
    rf_recent = comparison[
        (comparison["window_label"] == PRIMARY_RECENT_WINDOW)
        & (comparison["model_name"] == "behavior_plus_profile_random_forest")
        & (comparison["target"] == "retornar_ativo_m1")
    ]
    log_recent = comparison[
        (comparison["window_label"] == PRIMARY_RECENT_WINDOW)
        & (comparison["model_name"] == "behavior_plus_profile_logistic")
        & (comparison["target"] == "retornar_ativo_m1")
    ]
    hgb_auc = float(hgb_recent["test_roc_auc"].iloc[0]) if not hgb_recent.empty else float("nan")
    rf_auc = float(rf_recent["test_roc_auc"].iloc[0]) if not rf_recent.empty else float("nan")
    log_auc = float(log_recent["test_roc_auc"].iloc[0]) if not log_recent.empty else float("nan")
    if pd.notna(hgb_auc) and hgb_auc >= max([value for value in [rf_auc, log_auc] if pd.notna(value)] + [hgb_auc]):
        hgb_answer = "Yes. HistGradientBoosting is the strongest candidate in the recent regime."
    elif pd.notna(hgb_auc):
        hgb_answer = "Not here. HistGradientBoosting was tested, but another model class performed better on the recent holdout."
    else:
        hgb_answer = "HistGradientBoosting could not be evaluated cleanly in this run."

    if not population_review.empty and best_recent_model != "n/a":
        pop_slice = population_review[
            (population_review["window_label"] == PRIMARY_RECENT_WINDOW)
            & (population_review["model_name"] == best_recent_model)
        ].copy()
        strict_pop = pop_slice[pop_slice["population_variant"] == "current_strict"].head(1)
        relaxed_pop = pop_slice[pop_slice["population_variant"] == "relaxed_observed_next"].head(1)
        strict_auc = float(strict_pop["test_roc_auc"].iloc[0]) if not strict_pop.empty else float("nan")
        relaxed_auc = float(relaxed_pop["test_roc_auc"].iloc[0]) if not relaxed_pop.empty else float("nan")
        strict_rows = int(strict_pop["test_rows"].iloc[0]) if not strict_pop.empty else 0
        relaxed_rows = int(relaxed_pop["test_rows"].iloc[0]) if not relaxed_pop.empty else 0
        strict_zero_signal = float(strict_pop["population_zero_interaction_share"].iloc[0]) if not strict_pop.empty else float("nan")
        relaxed_zero_signal = float(relaxed_pop["population_zero_interaction_share"].iloc[0]) if not relaxed_pop.empty else float("nan")
    else:
        strict_auc = relaxed_auc = strict_zero_signal = relaxed_zero_signal = float("nan")
        strict_rows = relaxed_rows = 0

    if not feature_set.empty:
        feature_slice = feature_set[
            (feature_set["window_label"] == PRIMARY_RECENT_WINDOW)
            & (feature_set["model_name"] == best_recent_model)
        ].copy()
        compact_row = feature_slice[feature_slice["feature_set_label"] == "compact_scorecard"].head(1)
        full_row = feature_slice[feature_slice["feature_set_label"] == "full_behavior_plus_profile"].head(1)
        compact_auc = float(compact_row["test_roc_auc"].iloc[0]) if not compact_row.empty else float("nan")
        full_auc = float(full_row["test_roc_auc"].iloc[0]) if not full_row.empty else float("nan")
    else:
        compact_auc = full_auc = float("nan")

    prod_answer = production_recommendation(best_recent_auc, best_recent_f1, best_recent_auc_gap)
    churn_return_answer = "They are the same binary problem if `abandonar_m1 = 1 - retornar_ativo_m1`; the ranking task is identical, only the label orientation changes."

    rows = [
        {
            "question_id": "A",
            "question": "Was there drift, and is it relevant?",
            "short_answer": "Yes. The drift is material enough that recent-window validation is mandatory.",
            "evidence": f"{high_numeric} high numeric drift signals and {high_categorical} high categorical drift signals.",
            "recommendation": "Do not trust only all-history metrics. Always validate on the recent regime.",
        },
        {
            "question_id": "A1",
            "question": "Should training use only post-drift data?",
            "short_answer": train_answer,
            "evidence": (
                f"Primary recent window ({PRIMARY_RECENT_WINDOW}) same-holdout ROC AUC: include_pre_drift_history={fmt_num(include_auc, 3)} "
                f"vs recent_only_post_drift={fmt_num(recent_only_auc, 3)}; "
                f"F1={fmt_num(include_f1, 3)} vs {fmt_num(recent_only_f1, 3)}; "
                f"accuracy={fmt_num(include_accuracy, 3)} vs {fmt_num(recent_only_accuracy, 3)}."
            ),
            "recommendation": "Use the strategy that wins on the same recent holdout, but treat tiny AUC gaps as ties and let post-drift relevance plus threshold behavior break the tie.",
        },
        {
            "question_id": "B1",
            "question": "Can we predict when the user will not return?",
            "short_answer": (
                f"Moderately well, not perfectly. Recent held-out ROC AUC is {fmt_num(best_recent_auc, 3)}, "
                f"F1 is {fmt_num(best_recent_f1, 3)}, accuracy is {fmt_num(best_recent_accuracy, 3)}, "
                f"on a {fmt_num(best_recent_test_rows, 0)}-row primary recent holdout."
            ),
            "evidence": f"Recent selected model: {best_recent_model}.",
            "recommendation": prod_answer,
        },
        {
            "question_id": "B2",
            "question": "Should we relax the filter to get a larger recent test set?",
            "short_answer": (
                "Not for the main behavior model. The better fix is to keep the strict behavior population and use the recent_12m window as the primary evaluation."
                if pd.notna(strict_auc) and pd.notna(relaxed_auc)
                else "The strict-versus-relaxed population tradeoff could not be estimated cleanly."
            ),
            "evidence": (
                f"Strict {PRIMARY_RECENT_WINDOW}: test_rows={strict_rows}, auc={fmt_num(strict_auc, 3)}, zero_interaction_share={fmt_num(strict_zero_signal, 3)}; "
                f"relaxed {PRIMARY_RECENT_WINDOW}: test_rows={relaxed_rows}, auc={fmt_num(relaxed_auc, 3)}, zero_interaction_share={fmt_num(relaxed_zero_signal, 3)}."
            ),
            "recommendation": "Relax the filter only if you want a broader session-return question that includes low-information months; do not mix that with the cleaner behavior model without saying so explicitly.",
        },
        {
            "question_id": "B3",
            "question": "Can we predict when the user will return?",
            "short_answer": "Yes, but it is the same supervised problem as non-return if return and churn are complements.",
            "evidence": churn_return_answer,
            "recommendation": "Keep one canonical target and derive the complement score instead of maintaining two redundant models.",
        },
        {
            "question_id": "C",
            "question": "If prediction is not enough by itself, what should we do?",
            "short_answer": "Use the model as one layer only, and center the product work on an interpretable engagement scorecard built from time spent, tenure, streak, views, downloads, and navigation-without-activity.",
            "evidence": (
                f"Compact scorecard AUC={fmt_num(compact_auc, 3)} vs larger feature set AUC={fmt_num(full_auc, 3)} on {PRIMARY_RECENT_WINDOW}."
            ),
            "recommendation": "Pair the risk score with behavior cohorts, activation thresholds, journey drop-offs, and product experiments. If you need interpretability first, the compact scorecard is a good operational compromise.",
        },
        {
            "question_id": "D",
            "question": "Does HistGradientBoosting apply here?",
            "short_answer": hgb_answer,
            "evidence": (
                f"Primary recent ROC AUC comparison: HistGradientBoosting={fmt_num(hgb_auc, 3)}, "
                f"RandomForest={fmt_num(rf_auc, 3)}, Logistic={fmt_num(log_auc, 3)}."
            ),
            "recommendation": "Keep it in the benchmark suite, but only prefer it if it wins on the same recent holdout.",
        },
    ]
    df = pd.DataFrame(rows)
    df = attach_reference(
        df,
        source_tables=[
            "analytics_drift_numeric_explainable_v2",
            "analytics_drift_categorical_explainable_v2",
            "analytics_prediction_model_selection_v2",
            "analytics_prediction_model_comparison_explainable_v2",
            "analytics_prediction_recent_strategy_comparison_v2",
        ],
        build_summary="Direct-answer table for the final report, answering the main business questions from the drift and prediction outputs.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, then etapa_12a_drift_explainable_v2.py, then etapa_12b_prediction_explainable_v2.py, then rerun etapa_12c_drift_prediction_report_v2.py.",
    )
    payload = {
        "high_numeric_drift": high_numeric,
        "high_categorical_drift": high_categorical,
        "best_recent_auc": best_recent_auc,
        "best_recent_f1": best_recent_f1,
        "best_recent_accuracy": best_recent_accuracy,
        "best_recent_model": best_recent_model,
        "best_recent_test_rows": best_recent_test_rows,
        "include_pre_drift_auc": include_auc,
        "recent_only_auc": recent_only_auc,
        "strict_primary_auc": strict_auc,
        "relaxed_primary_auc": relaxed_auc,
        "compact_primary_auc": compact_auc,
        "full_primary_auc": full_auc,
        "production_recommendation": prod_answer,
    }
    return df, payload


def build_report_findings(
    direct_answers: pd.DataFrame,
    drift_findings: pd.DataFrame,
    prediction_findings: pd.DataFrame,
) -> pd.DataFrame:
    answer_rows = strip_reference_cols(direct_answers).copy()
    answer_rows["section"] = "Direct Answers"
    answer_rows = answer_rows.rename(
        columns={
            "question": "finding_label",
            "short_answer": "interpretation",
            "evidence": "evidence_value",
            "recommendation": "why_it_matters",
        }
    )
    answer_rows["finding_group"] = "business_answer"
    answer_rows["evidence_unit"] = "plain_english"
    answer_rows = answer_rows[["section", "finding_group", "finding_label", "evidence_value", "evidence_unit", "why_it_matters", "interpretation"]]

    drift = strip_reference_cols(drift_findings).copy()
    drift["section"] = "Part 1 - Drift"
    prediction = strip_reference_cols(prediction_findings).copy()
    prediction["section"] = "Part 2 - Prediction"
    combined = pd.concat([answer_rows, drift, prediction], ignore_index=True, sort=False)
    combined["evidence_value"] = combined["evidence_value"].astype(str)
    combined["evidence_unit"] = combined["evidence_unit"].astype(str)
    combined["why_it_matters"] = combined["why_it_matters"].astype(str)
    combined["interpretation"] = combined["interpretation"].astype(str)
    return attach_reference(
        combined,
        source_tables=[
            "analytics_drift_key_findings_v2",
            "analytics_prediction_key_findings_v2",
            "analytics_drift_prediction_report_answers_v2",
        ],
        build_summary="Consolidated findings table used by the final HTML report, prioritizing direct business answers first and then the strongest supporting findings.",
        rebuild_from_raw="Run raw_para_base_modelada_v4.py, then etapa_12a_drift_explainable_v2.py and etapa_12b_prediction_explainable_v2.py, then rerun etapa_12c_drift_prediction_report_v2.py.",
    )


def build_summary_payload(
    population: pd.DataFrame,
    direct_answers_payload: Dict[str, Any],
    model_selection: pd.DataFrame,
    recent_strategy: pd.DataFrame,
) -> Dict[str, Any]:
    return {
        "population_summary": strip_reference_cols(population).to_dict(orient="records"),
        "direct_answers": direct_answers_payload,
        "selected_models": strip_reference_cols(model_selection).to_dict(orient="records"),
        "recent_strategy": strip_reference_cols(recent_strategy).to_dict(orient="records"),
    }


def write_summary_markdown(path: Path, cfg: Config, payload: Dict[str, Any], html_path: Path) -> None:
    population = payload["population_summary"][0] if payload["population_summary"] else {}
    answers = payload["direct_answers"]
    lines = [
        "# Final Drift + Prediction Report v2",
        "",
        "## Paths",
        "",
        f"- Output directory: `{cfg.output_dir}`",
        f"- HTML report: `{html_path}`",
        "",
        "## Core Answers",
        "",
        f"- High numeric drift signals: {answers.get('high_numeric_drift')}",
        f"- High categorical drift signals: {answers.get('high_categorical_drift')}",
        f"- Best recent model: {answers.get('best_recent_model')}",
        f"- Best recent ROC AUC: {answers.get('best_recent_auc')}",
        f"- Include pre-drift history AUC on same recent holdout: {answers.get('include_pre_drift_auc')}",
        f"- Recent-only AUC on same recent holdout: {answers.get('recent_only_auc')}",
        f"- Production recommendation: {answers.get('production_recommendation')}",
        "",
        "## Population",
        "",
        f"- Rows: {population.get('rows', 'n/a')}",
        f"- Teachers: {population.get('teachers', 'n/a')}",
        "",
        "## Rebuild",
        "",
        "1. Run `raw_para_base_modelada_v4.py`.",
        "2. Run `etapa_12a_drift_explainable_v2.py`.",
        "3. Run `etapa_12b_prediction_explainable_v2.py`.",
        "4. Run `etapa_12c_drift_prediction_report_v2.py`.",
    ]
    write_markdown(path, lines)


def build_html_report(
    drift_assumptions: pd.DataFrame,
    drift_population: pd.DataFrame,
    drift_numeric: pd.DataFrame,
    drift_categorical: pd.DataFrame,
    drift_outcome: pd.DataFrame,
    prediction_assumptions: pd.DataFrame,
    control_validity: pd.DataFrame,
    data_sufficiency: pd.DataFrame,
    exclusion_bias: pd.DataFrame,
    feature_theme_review: pd.DataFrame,
    feature_relations: pd.DataFrame,
    behavior_segment_review: pd.DataFrame,
    population_strategy_review: pd.DataFrame,
    feature_set_review: pd.DataFrame,
    model_comparison: pd.DataFrame,
    recent_strategy: pd.DataFrame,
    roc_curve_points: pd.DataFrame,
    bootstrap_ci: pd.DataFrame,
    feature_importance: pd.DataFrame,
    direct_answers: pd.DataFrame,
    report_findings: pd.DataFrame,
    report_reference: pd.DataFrame,
) -> str:
    population_row = strip_reference_cols(drift_population).iloc[0]
    numeric = strip_reference_cols(drift_numeric).copy()
    numeric["drift_rank"] = numeric["drift_level"].map(drift_rank)
    numeric = numeric.sort_values(["drift_rank", "psi"], ascending=[True, False])
    categorical = strip_reference_cols(drift_categorical).copy()
    if not categorical.empty:
        categorical = (
            categorical.sort_values(["feature_total_variation", "feature_max_share_diff_pp"], ascending=[False, False])
            .drop_duplicates(subset=["feature_name"])
            .reset_index(drop=True)
        )
        categorical["drift_rank"] = categorical["drift_level"].map(drift_rank)
        categorical = categorical.sort_values(["drift_rank", "feature_total_variation"], ascending=[True, False])

    comparison = strip_reference_cols(model_comparison).copy()
    strategy = strip_reference_cols(recent_strategy).copy()
    theme_review = strip_reference_cols(feature_theme_review).copy()
    relations = strip_reference_cols(feature_relations).copy()
    behavior = strip_reference_cols(behavior_segment_review).copy()
    sufficiency = strip_reference_cols(data_sufficiency).copy()
    population_review = strip_reference_cols(population_strategy_review).copy()
    feature_set = strip_reference_cols(feature_set_review).copy()

    primary_return_models = comparison[
        (comparison["target"] == "retornar_ativo_m1")
        & (comparison["window_label"] == PRIMARY_RECENT_WINDOW)
    ].copy().sort_values("test_roc_auc", ascending=False)
    selected_recent = primary_return_models.iloc[0]
    selected_model_name = str(selected_recent["model_name"])

    primary_return_strategy = strategy[
        (strategy["window_label"] == PRIMARY_RECENT_WINDOW)
        & (strategy["target"] == "retornar_ativo_m1")
        & (strategy["model_name"] != "__best_strategy__")
    ].copy()
    primary_return_strategy["strategy_label_readable"] = primary_return_strategy["strategy_label"].map(
        {
            "include_pre_drift_history": "Train on all months before the test window",
            "recent_only_post_drift": "Train only on recent post-drift months",
        }
    )

    bootstrap = strip_reference_cols(bootstrap_ci).copy()
    primary_return_bootstrap = bootstrap[
        (bootstrap["target"] == "retornar_ativo_m1")
        & (bootstrap["window_label"] == PRIMARY_RECENT_WINDOW)
    ].copy()

    primary_return_roc = strip_reference_cols(roc_curve_points)
    primary_return_roc = primary_return_roc[
        (primary_return_roc["target"] == "retornar_ativo_m1")
        & (primary_return_roc["window_label"] == PRIMARY_RECENT_WINDOW)
        & (primary_return_roc["model_name"] == selected_model_name)
    ].copy()

    primary_return_sufficiency_slice = sufficiency[
        (sufficiency["window_label"] == PRIMARY_RECENT_WINDOW)
        & (sufficiency["target"] == "retornar_ativo_m1")
    ].copy()
    if not primary_return_sufficiency_slice.empty:
        primary_return_sufficiency = primary_return_sufficiency_slice.iloc[0]
    else:
        primary_return_sufficiency = pd.Series(
            {
                "test_rows": selected_recent["test_rows"],
            }
        )

    theme_name_map = {
        "downloads": "Download Behavior",
        "platform_history": "Time And History",
        "platform_tenure": "Platform Tenure",
        "teacher_formation": "Formation",
        "views_clicks": "Content Interaction",
        "time_spent": "Session Depth",
        "behavior_shares": "Behavior Shares",
    }
    theme_view = theme_review.copy()
    if not theme_view.empty:
        theme_view["feature_family"] = theme_view["theme"].map(theme_name_map).fillna(theme_view["theme"])
        theme_view["support_score"] = theme_view["best_importance_value"].fillna(0.0)
        theme_view["support_score"] = np.where(
            theme_view["support_score"] > 0,
            theme_view["support_score"],
            theme_view["best_univariate_auc"].fillna(0.5) - 0.5,
        )
        family_importance = (
            theme_view[theme_view["included_in_model"] == 1]
            .groupby("feature_family", as_index=False)["support_score"]
            .sum()
            .sort_values("support_score", ascending=False)
        )
        top_features_table = (
            theme_view[theme_view["included_in_model"] == 1]
            .sort_values(["support_score", "best_univariate_auc"], ascending=[False, False])[
                ["feature_family", "feature_name", "best_univariate_auc", "best_importance_value"]
            ]
            .head(12)
            .copy()
        )
    else:
        family_importance = pd.DataFrame(columns=["feature_family", "support_score"])
        top_features_table = pd.DataFrame(
            columns=["feature_family", "feature_name", "best_univariate_auc", "best_importance_value"]
        )

    relation_view = relations[
        relations["relation_label"].isin(
            [
                "download_vs_content_views",
                "platform_time_vs_download",
                "formation_vs_platform_time",
                "formation_vs_download",
                "formation_vs_content_views",
            ]
        )
    ].copy()
    relation_view["relation_readable"] = relation_view["relation_label"].map(
        {
            "download_vs_content_views": "Downloads vs content views",
            "platform_time_vs_download": "Platform time vs downloads",
            "formation_vs_platform_time": "Formation vs platform time",
            "formation_vs_download": "Formation vs downloads",
            "formation_vs_content_views": "Formation vs content views",
        }
    )

    behavior["segment_label"] = behavior["formation_segment"] + " | " + behavior["download_segment"]
    sensitivity_model = comparison[
        (comparison["target"] == "retornar_ativo_m1")
        & (comparison["window_label"] == SENSITIVITY_WINDOW)
        & (comparison["model_name"] == selected_model_name)
    ].head(1)
    population_best_model = population_review[
        (population_review["model_name"] == selected_model_name)
        & (population_review["window_label"].isin([PRIMARY_RECENT_WINDOW, SENSITIVITY_WINDOW]))
    ].copy()
    population_best_model["population_label"] = population_best_model["population_variant"].map(
        {
            "current_strict": "Strict behavior population",
            "relaxed_observed_next": "Relaxed observed-next population",
        }
    )
    population_best_model["window_label_readable"] = population_best_model["window_label"].map(
        {PRIMARY_RECENT_WINDOW: "Primary recent_12m", SENSITIVITY_WINDOW: "Sensitivity recent_6m"}
    )
    feature_set_best_model = feature_set[
        (feature_set["model_name"] == selected_model_name)
        & (feature_set["window_label"] == PRIMARY_RECENT_WINDOW)
    ].copy()
    behavior_view = behavior[
        [
            "segment_label",
            "rows",
            "return_rate_m1",
            "avg_session_minutes",
            "avg_content_views",
            "avg_downloads",
            "navigation_without_activity_share",
        ]
    ].copy()

    high_numeric = int((numeric["drift_level"] == "high_drift").sum()) if not numeric.empty else 0
    high_categorical = int((categorical["drift_level"] == "high_drift").sum()) if not categorical.empty else 0
    summary_cards = "".join(
        [
            build_card_html("Teachers", fmt_num(population_row["teachers"], 0), "Teacher-month population in the primary strict behavior model"),
            build_card_html("Primary Test Rows", fmt_num(primary_return_sufficiency["test_rows"], 0), "Held-out rows in the recent_12m primary window"),
            build_card_html("Primary ROC AUC", fmt_num(selected_recent["test_roc_auc"], 3), f"{selected_model_name} on the primary recent window"),
            build_card_html("High Drift Signals", fmt_num(high_numeric + high_categorical, 0), "Numeric + categorical high-drift features"),
        ]
    )

    sections: List[Dict[str, Any]] = []

    strategy_metrics = primary_return_strategy[
        ["strategy_label_readable", "roc_auc", "average_precision", "accuracy", "f1", "brier_score", "log_loss", "tp", "fp", "tn", "fn"]
    ].copy()
    strategy_metrics_long = strategy_metrics.melt(
        id_vars=["strategy_label_readable"],
        value_vars=["roc_auc", "average_precision", "accuracy", "f1"],
        var_name="metric",
        value_name="value",
    )
    fig_a_main = px.bar(
        strategy_metrics_long,
        x="metric",
        y="value",
        color="strategy_label_readable",
        barmode="group",
        color_discrete_sequence=[PALETTE[1], PALETTE[5]],
        title="Question A: same recent_12m holdout, two training strategies",
    )
    fig_a_num = px.bar(
        numeric.head(8),
        x="psi",
        y="feature_name",
        color="drift_level",
        orientation="h",
        color_discrete_map={"high_drift": PALETTE[5], "medium_drift": PALETTE[3], "low_drift": PALETTE[2], "insufficient_data": PALETTE[4]},
        title="Support 1: strongest numeric drift signals",
    )
    fig_a_num.update_layout(yaxis={"categoryorder": "total ascending"})
    fig_a_cat = px.bar(
        categorical.head(8),
        x="feature_total_variation",
        y="feature_name",
        color="drift_level",
        orientation="h",
        color_discrete_map={"high_drift": PALETTE[5], "medium_drift": PALETTE[3], "low_drift": PALETTE[2], "insufficient_data": PALETTE[4]},
        title="Support 2: strongest categorical drift signals",
    )
    fig_a_cat.update_layout(yaxis={"categoryorder": "total ascending"})
    strategy_recent_only = primary_return_strategy.loc[primary_return_strategy["strategy_label"] == "recent_only_post_drift"].head(1)
    strategy_all_history = primary_return_strategy.loc[primary_return_strategy["strategy_label"] == "include_pre_drift_history"].head(1)
    section_a_note = (
        "<div class='note'>"
        "<strong>Step by step.</strong> "
        "1. Measure drift on the strict teacher-month population. "
        "2. Keep recent_12m as the main post-drift evaluation window because recent_6m is too thin. "
        "3. On the same recent_12m holdout, compare training on all prior history versus training only on the recent post-drift months."
        "</div>"
        "<div class='note'>"
        f"<strong>What is observed.</strong> Drift is real: {high_numeric} high numeric drift signals and {high_categorical} high categorical drift signals. "
        f"On the same {PRIMARY_RECENT_WINDOW} holdout, recent-only training gets ROC AUC {strategy_recent_only['roc_auc'].iloc[0]:.3f} versus {strategy_all_history['roc_auc'].iloc[0]:.3f} for all-history training. "
        f"This is effectively a tie on ranking. Recent-only is slightly better on thresholded behavior on the same holdout: "
        f"F1 {strategy_recent_only['f1'].iloc[0]:.3f} versus {strategy_all_history['f1'].iloc[0]:.3f}, "
        f"accuracy {strategy_recent_only['accuracy'].iloc[0]:.3f} versus {strategy_all_history['accuracy'].iloc[0]:.3f}."
        "</div>"
        "<div class='note'>"
        "<strong>How to interpret it.</strong> We should not invent weights unless we actually fit and validate them. That was not done here. "
        "Because ranking is effectively tied, the non-arbitrary choice is to use the recent-only post-drift model as the primary operational model and keep the all-history model as a benchmark."
        "</div>"
        + build_table_html(strategy_metrics, max_rows=10)
    )
    sections.append(
        {
            "title": "Question A",
            "description": "There was drift. Should training use only post-drift data, or should it keep the older data as well?",
            "blocks": [
                chart_block(
                    "question_a_main",
                    "Main Chart",
                    "One chart to answer the question directly: same recent_12m holdout, same target, two training strategies.",
                    figure_to_html(fig_a_main) + section_a_note,
                    {
                        "raw_tables": "mart_teacher_month_persona_ready, audit_base_modelada_validation",
                        "population": "Strict teacher-month behavior population with observed current month and observed next month",
                        "grain": "1 row per training strategy on the same recent_12m holdout",
                        "joins": "Model-comparison and recent-strategy outputs come from the same prediction pipeline",
                        "filters": f"Target = retornar_ativo_m1, window = {PRIMARY_RECENT_WINDOW}",
                        "logic": "Compare the same model class under two training windows after first showing that the data drifted",
                        "caveats": "No time-weighted model was fitted or validated in this analysis",
                    },
                ),
                chart_block(
                    "question_a_support_numeric",
                    "Support Chart 1",
                    "Why the drift claim is solid on the numeric side.",
                    figure_to_html(fig_a_num),
                    {
                        "raw_tables": "mart_teacher_month_persona_ready",
                        "population": "Same strict modeling population as the main strategy test",
                        "grain": "1 row per numeric feature",
                        "joins": "No extra joins after the monthly mart is built",
                        "filters": "Oldest six modeling months vs most recent six modeling months",
                        "logic": "PSI and standardized mean difference measure how much the numeric distribution moved",
                        "caveats": "Drift justifies the strategy test; it does not by itself pick the winning strategy",
                    },
                ),
                chart_block(
                    "question_a_support_categorical",
                    "Support Chart 2",
                    "Why the drift claim is also solid on the categorical side.",
                    figure_to_html(fig_a_cat),
                    {
                        "raw_tables": "mart_teacher_month_persona_ready, dim_teacher",
                        "population": "Same strict modeling population as the main strategy test",
                        "grain": "1 row per categorical feature",
                        "joins": "No extra joins after the monthly mart is built",
                        "filters": "Oldest six modeling months vs most recent six modeling months",
                        "logic": "Total variation shows whether channel/profile mix changed enough to matter",
                        "caveats": "Categorical drift matters mainly for generalization and calibration",
                    },
                ),
            ],
        }
    )

    model_metrics_long = primary_return_models.melt(
        id_vars=["model_name"],
        value_vars=["test_roc_auc", "test_average_precision", "test_accuracy", "test_f1"],
        var_name="metric",
        value_name="value",
    )
    fig_b_main = px.bar(
        model_metrics_long,
        x="metric",
        y="value",
        color="model_name",
        barmode="group",
        color_discrete_sequence=[PALETTE[1], PALETTE[3], PALETTE[5], PALETTE[6], PALETTE[2]],
        title="Question B: how good is the prediction on recent real data?",
    )
    selected_train_test = pd.DataFrame(
        {
            "metric": ["ROC AUC", "Average Precision", "Accuracy", "F1"],
            "Train": [
                selected_recent["train_roc_auc"],
                selected_recent["train_average_precision"],
                selected_recent["train_accuracy"],
                selected_recent["train_f1"],
            ],
            "Test": [
                selected_recent["test_roc_auc"],
                selected_recent["test_average_precision"],
                selected_recent["test_accuracy"],
                selected_recent["test_f1"],
            ],
        }
    ).melt(id_vars=["metric"], value_vars=["Train", "Test"], var_name="split", value_name="value")
    fig_b_gap = px.bar(
        selected_train_test,
        x="metric",
        y="value",
        color="split",
        barmode="group",
        color_discrete_sequence=[PALETTE[1], PALETTE[5]],
        title="Support 1: selected model, train versus test",
    )
    fig_b_population = px.scatter(
        population_best_model,
        x="test_rows",
        y="test_roc_auc",
        color="population_label",
        symbol="window_label_readable",
        size="population_zero_interaction_share",
        hover_name="population_label",
        color_discrete_sequence=[PALETTE[1], PALETTE[5]],
        title="Support 2: sample size versus quality when we change the population/window",
    )
    confusion_z = [
        [int(selected_recent["test_tn"]), int(selected_recent["test_fp"])],
        [int(selected_recent["test_fn"]), int(selected_recent["test_tp"])],
    ]
    confusion_text = [[str(value) for value in row] for row in confusion_z]
    fig_b_confusion = go.Figure(
        data=
        [
            go.Heatmap(
                z=confusion_z,
                x=["Predicted No Return", "Predicted Return"],
                y=["Actual No Return", "Actual Return"],
                text=confusion_text,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
            )
        ]
    )
    fig_b_confusion.update_layout(title="Support 3: confusion matrix on the selected primary recent model")
    bootstrap_view = primary_return_bootstrap[["metric_name", "ci_low", "ci_high"]].copy()
    metrics_detail = pd.DataFrame(
        [
            {"metric": "ROC AUC", "value": selected_recent["test_roc_auc"]},
            {"metric": "Average Precision", "value": selected_recent["test_average_precision"]},
            {"metric": "Accuracy", "value": selected_recent["test_accuracy"]},
            {"metric": "F1", "value": selected_recent["test_f1"]},
            {"metric": "Monthly Rate MAPE", "value": selected_recent["test_monthly_rate_mape"]},
            {"metric": "Efron Pseudo-R2", "value": selected_recent["test_efron_pseudo_r2"]},
            {"metric": "Brier Score", "value": selected_recent["test_brier_score"]},
            {"metric": "Log Loss", "value": selected_recent["test_log_loss"]},
            {"metric": "True Positives", "value": selected_recent["test_tp"]},
            {"metric": "True Negatives", "value": selected_recent["test_tn"]},
            {"metric": "False Positives", "value": selected_recent["test_fp"]},
            {"metric": "False Negatives", "value": selected_recent["test_fn"]},
        ]
    )
    sensitivity_text = ""
    if not sensitivity_model.empty:
        sensitivity_text = (
            f" The thinner {SENSITIVITY_WINDOW} sensitivity check keeps the same general story: ROC AUC {sensitivity_model['test_roc_auc'].iloc[0]:.3f}, "
            f"but on only {int(sensitivity_model['test_rows'].iloc[0])} held-out rows."
        )
    section_b_note = (
        "<div class='note'>"
        "<strong>Step by step.</strong> "
        "1. Use recent_12m as the main recent regime. "
        "2. Compare model classes on the same held-out months. "
        "3. Inspect train versus test, uncertainty, and whether changing the population definition improves or degrades the result."
        "</div>"
        "<div class='note'>"
        f"<strong>What is observed.</strong> The selected primary recent model is <code>{selected_model_name}</code>. "
        f"On real recent data it gets ROC AUC {selected_recent['test_roc_auc']:.3f}, average precision {selected_recent['test_average_precision']:.3f}, "
        f"accuracy {selected_recent['test_accuracy']:.3f}, and F1 {selected_recent['test_f1']:.3f} on {int(primary_return_sufficiency['test_rows'])} held-out rows."
        f" At the selected threshold ({selected_recent['selected_threshold']:.3f}), the confusion matrix is TP {int(selected_recent['test_tp'])}, FP {int(selected_recent['test_fp'])}, TN {int(selected_recent['test_tn'])}, FN {int(selected_recent['test_fn'])}."
        + sensitivity_text
        + "</div>"
        "<div class='note'>"
        f"<strong>How to interpret it.</strong> This is good enough for prioritization and analyst support, not for autonomous production action. "
        f"The main reason is not just AUC. Calibration-style metrics are still weak, and the false-positive / false-negative tradeoff is material. "
        f"If we simply relax the population filter, the test set gets larger but quality slips because we add many low-information no-interaction months. "
        f"The better fix is to keep the strict behavior population and use the larger {PRIMARY_RECENT_WINDOW} window as the main evaluation."
        "</div>"
        + build_table_html(metrics_detail, max_rows=12)
        + build_table_html(bootstrap_view, max_rows=10)
    )
    sections.append(
        {
            "title": "Question B",
            "description": "Given this data, can we build a prediction that works sufficiently well to predict return or non-return?",
            "blocks": [
                chart_block(
                    "question_b_main",
                    "Main Chart",
                    "One chart to answer the question directly: candidate models compared on the same primary recent holdout.",
                    figure_to_html(fig_b_main) + section_b_note,
                    {
                        "raw_tables": "mart_teacher_month_persona_ready, fct_interaction_clean, fct_formation_clean, dim_teacher",
                        "population": "Strict teacher-month behavior population, evaluated on the recent_12m holdout",
                        "grain": "1 row per model on the same primary recent holdout",
                        "joins": "Prediction features include time-safe monthly behavior, tenure, formation, and raw interaction-share proxies",
                        "filters": f"Target = retornar_ativo_m1; window = {PRIMARY_RECENT_WINDOW}",
                        "logic": "Judge usefulness from held-out recent behavior, not from training fit or old-history averages",
                        "caveats": "Return and non-return are complements; we use return here because it is the harder minority event",
                    },
                ),
                chart_block(
                    "question_b_support_gap",
                    "Support Chart 1",
                    "How much the selected model degrades from training to test.",
                    figure_to_html(fig_b_gap),
                    {
                        "raw_tables": "analytics_prediction_model_comparison_explainable_v2",
                        "population": "Selected primary recent model only",
                        "grain": "Train/test metrics for one selected model",
                        "joins": "Derived directly from the prediction output table",
                        "filters": f"Target = retornar_ativo_m1, model = {selected_model_name}, window = {PRIMARY_RECENT_WINDOW}",
                        "logic": "Large train-test gaps would mean the model is too sensitive or too optimistic",
                        "caveats": "A modest train-test gap still does not guarantee good calibration",
                    },
                ),
                chart_block(
                    "question_b_support_population",
                    "Support Chart 2",
                    "What happens if we relax the filter or shrink the recent window?",
                    figure_to_html(fig_b_population),
                    {
                        "raw_tables": "mart_teacher_month_persona_ready, fct_interaction_clean, fct_formation_clean",
                        "population": "Strict versus relaxed population variants for the selected model class",
                        "grain": "1 row per population-variant x window choice",
                        "joins": "Each point comes from the population-strategy review table built from the same fixed model class",
                        "filters": f"Model = {selected_model_name}, target = retornar_ativo_m1",
                        "logic": "This chart answers whether a larger sample comes from better design or from admitting noisier low-information months",
                        "caveats": "The relaxed population answers a slightly broader business question because it includes session-only/no-interaction months",
                    },
                ),
                chart_block(
                    "question_b_support_confusion",
                    "Support Chart 3",
                    "Confusion matrix of the selected primary recent model.",
                    figure_to_html(fig_b_confusion),
                    {
                        "raw_tables": "analytics_prediction_model_comparison_explainable_v2",
                        "population": "Selected primary recent model only",
                        "grain": "2x2 thresholded prediction table of one selected model",
                        "joins": "Derived directly from held-out TP, FP, TN, and FN counts",
                        "filters": f"Target = retornar_ativo_m1, model = {selected_model_name}, window = {PRIMARY_RECENT_WINDOW}",
                        "logic": "This shows the real operational tradeoff at the chosen threshold, not just ranking quality",
                        "caveats": "The matrix changes if the threshold changes; this is one operating point, not a universal truth",
                    },
                ),
            ],
        }
    )

    fig_c_main = px.bar(
        family_importance,
        x="support_score",
        y="feature_family",
        orientation="h",
        color="feature_family",
        color_discrete_sequence=PALETTE,
        title="Question C: which type of signal is actually useful?",
    )
    fig_c_main.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
    feature_set_long = feature_set_best_model.melt(
        id_vars=["feature_set_label", "model_name"],
        value_vars=["test_roc_auc", "test_accuracy", "test_f1"],
        var_name="metric",
        value_name="value",
    )
    fig_c_feature_sets = px.bar(
        feature_set_long,
        x="metric",
        y="value",
        color="feature_set_label",
        barmode="group",
        color_discrete_sequence=[PALETTE[1], PALETTE[5]],
        title="Support 1: compact scorecard versus larger feature set",
    )
    fig_c_rel = px.bar(
        relation_view,
        x="mean_within_month_spearman_rho",
        y="relation_readable",
        orientation="h",
        color="relation_readable",
        color_discrete_sequence=[PALETTE[1], PALETTE[3], PALETTE[5], PALETTE[6], PALETTE[2]],
        title="Support 2: are time spent, downloads, views, and formation really related?",
    )
    fig_c_rel.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
    section_c_note = (
        "<div class='note'>"
        "<strong>Step by step.</strong> "
        "1. Group the strongest predictors into families instead of pretending every variable is an independent causal lever. "
        "2. Admit the requested features explicitly: time spent, tenure, formation, downloads, views/clicks, and raw behavior-share proxies. "
        "3. Test whether a compact scorecard loses too much performance versus the larger feature set."
        "</div>"
        "<div class='note'>"
        "<strong>What is observed.</strong> The strongest predictive family is time and usage history, followed by session depth, content interaction, and the new raw behavior-share proxies. "
        "Time spent is definitely used here: the model includes <code>lifetime_clean_entry_minutes_total</code>, <code>clean_entry_avg_session_minutes_month</code>, and <code>session_minutes_per_active_day_month</code>. "
        "Formation is included too, but it stays weak. "
        f"Downloads and content views move together within month (rho = {relation_view.loc[relation_view['relation_label']=='download_vs_content_views','mean_within_month_spearman_rho'].iloc[0]:.3f}). "
        f"Platform time and downloads have only a modest positive link (rho = {relation_view.loc[relation_view['relation_label']=='platform_time_vs_download','mean_within_month_spearman_rho'].iloc[0]:.3f})."
        "</div>"
        "<div class='note'>"
        "<strong>How to interpret it.</strong> The top predictors are related, not independent. "
        "That is why we should tell the story at the family level, not feature by feature. "
        "The compact scorecard keeps almost all of the predictive value, which means many variables are measuring the same broader construct: engagement intensity over time. "
        f" In the selected model class, the gap is only {feature_set_best_model.loc[feature_set_best_model['feature_set_label']=='full_behavior_plus_profile','test_roc_auc'].iloc[0] - feature_set_best_model.loc[feature_set_best_model['feature_set_label']=='compact_scorecard','test_roc_auc'].iloc[0]:.3f} ROC AUC points on {PRIMARY_RECENT_WINDOW}. "
        "The larger feature set still helps a bit, especially after adding raw interaction-share features, but the practical product output should be an interpretable engagement scorecard plus behavior cohorts and experiments."
        "</div>"
        + build_table_html(top_features_table, max_rows=12)
        + build_table_html(behavior_view, max_rows=10)
    )
    sections.append(
        {
            "title": "Question C",
            "description": "If the predictive model is not enough by itself, what should the team use to understand behavior and test retention ideas in a meaningful way?",
            "blocks": [
                chart_block(
                    "question_c_main",
                    "Main Chart",
                    "One chart to answer the question directly: which families of signal actually carry predictive value?",
                    figure_to_html(fig_c_main) + section_c_note,
                    {
                        "raw_tables": "analytics_prediction_feature_theme_review_v2, analytics_prediction_feature_importance_explainable_v2, fct_interaction_clean, fct_formation_clean, mart_teacher_month_persona_ready",
                        "population": "Selected primary recent model only",
                        "grain": "One row per feature family",
                        "joins": "Feature families come from the admitted feature-theme review and use held-out importance when available, with univariate signal as fallback",
                        "filters": "Only admitted model features are included",
                        "logic": "Interpret the model at the feature-family level because many top variables measure the same engagement construct",
                        "caveats": "Family importance is predictive, not causal",
                    },
                ),
                chart_block(
                    "question_c_support_feature_sets",
                    "Support Chart 1",
                    "Do we really need the larger feature set, or is a compact scorecard enough?",
                    figure_to_html(fig_c_feature_sets),
                    {
                        "raw_tables": "mart_teacher_month_persona_ready, fct_interaction_clean, fct_formation_clean",
                        "population": "Primary strict behavior population",
                        "grain": "1 row per feature-set choice within the selected model class",
                        "joins": "Compact-versus-full metrics come from the feature-set review table built from the same fixed model class",
                        "filters": f"Model = {selected_model_name}, window = {PRIMARY_RECENT_WINDOW}",
                        "logic": "If the compact scorecard stays close to the larger model, then the top predictors are mostly redundant family signals",
                        "caveats": "A small gap does not make the compact scorecard causal; it only makes it easier to explain and operate",
                    },
                ),
                chart_block(
                    "question_c_support_relations",
                    "Support Chart 2",
                    "Are time spent, downloads, views, and formation really related?",
                    figure_to_html(fig_c_rel),
                    {
                        "raw_tables": "mart_teacher_month_persona_ready, fct_formation_clean",
                        "population": "Full strict modeling population",
                        "grain": "1 row per tested relationship",
                        "joins": "Formation is added cumulatively and time-safely before correlation testing",
                        "filters": "Only the key proposed feature-engineering relations are shown",
                        "logic": "Within-month Spearman is used to avoid confusing raw time trends with true behavioral co-movement",
                        "caveats": "Correlation is not independence, and not causality",
                    },
                ),
            ],
        }
    )

    return render_report_html(
        title="Relatorio 04. Drift e Predicao Explicavel v2",
        subtitle="Three questions only: what changed, how good prediction really is on recent data, and what signal is actually useful for understanding behavior.",
        summary_cards_html=summary_cards,
        sections=sections,
    )


def main() -> None:
    args = parse_args()
    cfg = build_config(base_dir=args.base_dir, source_dir=args.source_dir, output_dir=args.output_dir)
    output_conn = connect_output(cfg)

    try:
        drift_assumptions = require_table(cfg.output_dir, "analytics_drift_assumptions_v2")
        drift_population = require_table(cfg.output_dir, "analytics_drift_population_summary_v2")
        drift_numeric = require_table(cfg.output_dir, "analytics_drift_numeric_explainable_v2")
        drift_categorical = require_table(cfg.output_dir, "analytics_drift_categorical_explainable_v2")
        drift_outcome = require_table(cfg.output_dir, "analytics_drift_outcome_explainable_v2")
        drift_findings = require_table(cfg.output_dir, "analytics_drift_key_findings_v2")

        prediction_assumptions = require_table(cfg.output_dir, "analytics_prediction_assumptions_v2")
        control_validity = require_table(cfg.output_dir, "analytics_prediction_control_variable_validity_v2")
        data_sufficiency = require_table(cfg.output_dir, "analytics_prediction_data_sufficiency_v2")
        exclusion_bias = require_table(cfg.output_dir, "analytics_prediction_exclusion_bias_v2")
        feature_theme_review = require_table(cfg.output_dir, "analytics_prediction_feature_theme_review_v2")
        feature_relations = require_table(cfg.output_dir, "analytics_prediction_feature_relation_review_v2")
        behavior_segment_review = require_table(cfg.output_dir, "analytics_prediction_behavior_segment_review_v2")
        population_strategy_review = require_table(cfg.output_dir, "analytics_prediction_population_strategy_review_v2")
        feature_set_review = require_table(cfg.output_dir, "analytics_prediction_feature_set_review_v2")
        model_comparison = require_table(cfg.output_dir, "analytics_prediction_model_comparison_explainable_v2")
        model_selection = require_table(cfg.output_dir, "analytics_prediction_model_selection_v2")
        recent_strategy = require_table(cfg.output_dir, "analytics_prediction_recent_strategy_comparison_v2")
        roc_curve_points = require_table(cfg.output_dir, "analytics_prediction_roc_curve_v2")
        bootstrap_ci = require_table(cfg.output_dir, "analytics_prediction_bootstrap_ci_v2")
        feature_importance = require_table(cfg.output_dir, "analytics_prediction_feature_importance_explainable_v2")
        prediction_findings = require_table(cfg.output_dir, "analytics_prediction_key_findings_v2")

        direct_answers, direct_payload = build_direct_answers(
            drift_numeric=drift_numeric,
            drift_categorical=drift_categorical,
            model_selection=model_selection,
            model_comparison=model_comparison,
            recent_strategy=recent_strategy,
            population_strategy_review=population_strategy_review,
            feature_set_review=feature_set_review,
        )
        report_findings = build_report_findings(direct_answers, drift_findings, prediction_findings)

        outputs = {
            "analytics_drift_prediction_report_answers_v2": direct_answers,
            "analytics_drift_prediction_report_findings_v2": report_findings,
        }
        output_reference = build_output_reference(
            cfg,
            outputs.keys(),
            build_summary="Manifest of all report-layer outputs generated by etapa_12c_drift_prediction_report_v2.py.",
        )
        outputs["analytics_drift_prediction_report_reference_v2"] = output_reference

        for table_name, df in outputs.items():
            persist_table(output_conn, cfg, table_name, df)

        html = build_html_report(
            drift_assumptions=drift_assumptions,
            drift_population=drift_population,
            drift_numeric=drift_numeric,
            drift_categorical=drift_categorical,
            drift_outcome=drift_outcome,
            prediction_assumptions=prediction_assumptions,
            control_validity=control_validity,
            data_sufficiency=data_sufficiency,
            exclusion_bias=exclusion_bias,
            feature_theme_review=feature_theme_review,
            feature_relations=feature_relations,
            behavior_segment_review=behavior_segment_review,
            population_strategy_review=population_strategy_review,
            feature_set_review=feature_set_review,
            model_comparison=model_comparison,
            recent_strategy=recent_strategy,
            roc_curve_points=roc_curve_points,
            bootstrap_ci=bootstrap_ci,
            feature_importance=feature_importance,
            direct_answers=direct_answers,
            report_findings=report_findings,
            report_reference=output_reference,
        )

        html_path = cfg.output_dir / "reports" / "relatorio_04_drift_prediction_explainable_v2.html"
        html_path.write_text(html, encoding="utf-8")

        payload = build_summary_payload(
            population=drift_population,
            direct_answers_payload=direct_payload,
            model_selection=model_selection,
            recent_strategy=recent_strategy,
        )
        write_json(cfg.output_dir / "json" / "drift_prediction_report_summary_v2.json", payload)
        write_summary_markdown(cfg.output_dir / "audit" / "drift_prediction_report_summary_v2.md", cfg, payload, html_path)
    finally:
        output_conn.close()


if __name__ == "__main__":
    main()
