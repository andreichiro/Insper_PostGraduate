"""Runner principal do pipeline modelled -> ml.

Este arquivo deve ser o mais fácil de ler: ele só encadeia as etapas do build.
"""

from __future__ import annotations

import argparse
import duckdb
import pandas as pd
from pathlib import Path
from typing import Any

from targeted_ml.config.models import AnalysisSpec
from targeted_ml.definitions.custom_metric import apply_custom_metric_overrides

from . import analysis_setup as setup
from .analysis_setup import RuntimeBuildConfig
from .analysis_setup import (
    build_arbitrariness_registry,
    build_candidate_metric_registry,
    build_feature_registry,
    build_label_registry,
    build_policy_registry,
    build_track_registry,
)
from .dataset_builder import (
    build_feature_eligibility_log,
    build_first_session_journey_mart,
    build_future_metrics,
    build_leakage_audit,
    summarize_leakage_audit,
    build_official_frame,
    build_onboarding_mart,
)
from .definitions import build_definition_search, compare_official_definitions
from .modeling import (
    bootstrap_prediction_metrics,
    build_definition_b_feature_block_gain_diagnostics,
    build_scoring_scenarios,
    evaluate_model_problems,
    filter_official_predictions,
)
from .progress import BuildProgressTracker, ProgressStageSpec
from .post_model_outputs import (
    build_cluster_outputs,
    build_cv_metric_robustness_outputs,
    build_cv_score_robustness_outputs,
    build_cv_threshold_robustness_outputs,
    build_definition_b_excessive_separation_outputs,
    build_heavy_user_outputs,
    build_navigation_outputs,
    build_threshold_post_model_outputs,
    select_reference_scope_for_post_model_outputs,
)
from .storage import EnginePaths, TaskArtifactStore, attach_modelled_views, clear_previous_outputs, ensure_output_dirs, persist_table, write_json

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Aula_9 targeted ML rebuild as a single publishable pipeline.")
    parser.add_argument(
        "--modelled-duckdb",
        type=Path,
        default=setup.DEFAULT_MODELLED_DUCKDB,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--skip-post-model-refit",
        action="store_true",
        help="Skip the second filtered refit used only to enrich the post-model interpretation tables.",
    )
    return parser.parse_args()

def build_summary_payload(
    cfg: EnginePaths,
    official_problem_frontier: pd.DataFrame,
    definition_frontier: pd.DataFrame,
    arbitrariness_registry: pd.DataFrame,
) -> dict[str, Any]:
    frontier_count = int(official_problem_frontier["pareto_frontier_flag"].sum()) if "pareto_frontier_flag" in official_problem_frontier.columns else 0
    return {
        "pipeline_scope": "single_publishable_build",
        "modelled_duckdb": str(cfg.modelled_duckdb),
        "output_dir": str(cfg.output_dir),
        "official_definition_frontier_count": int(definition_frontier["pareto_frontier_flag"].sum()) if "pareto_frontier_flag" in definition_frontier.columns else 0,
        "official_problem_frontier_count": frontier_count,
        "official_score_status": "unique_official_score" if frontier_count == 1 else ("admissible_set_only" if frontier_count > 1 else "no_official_score"),
        "arbitrary_items_in_official_report": int(arbitrariness_registry["in_official_report_flag"].sum()),
    }

def run_build(cfg: EnginePaths, runtime_config: RuntimeBuildConfig | None = None) -> dict[str, Any]:
    if runtime_config is not None:
        setup.apply_runtime_config(runtime_config)

    ensure_output_dirs(cfg.output_dir)
    clear_previous_outputs(cfg.output_dir)
    task_store = TaskArtifactStore(cfg.output_dir / "staging")
    progress = BuildProgressTracker(
        [ProgressStageSpec(**row) for row in setup.BUILD_PROGRESS_STAGE_SPECS],
        reference_total_minutes=setup.BUILD_PROGRESS_REFERENCE_MINUTES,
    )
    out_conn = duckdb.connect(str(cfg.output_duckdb))
    try:
        attach_modelled_views(out_conn, cfg.modelled_duckdb, setup.MODELLED_TABLES)

        progress.start_stage("registries", detail="montando registries")
        track_registry = build_track_registry()
        arbitrariness_registry = build_arbitrariness_registry()
        policy_registry = build_policy_registry()
        candidate_metric_registry = build_candidate_metric_registry()
        progress.complete_stage("registries", detail="registries ready")
        print("[build] registries ready", flush=True)
        persist_table(out_conn, cfg.output_dir, "governance_track_registry_v1", track_registry)
        persist_table(out_conn, cfg.output_dir, "governance_arbitrariness_registry_v1", arbitrariness_registry)
        persist_table(out_conn, cfg.output_dir, "governance_policy_registry_v1", policy_registry)
        persist_table(out_conn, cfg.output_dir, "governance_definition_candidate_metric_registry_v1", candidate_metric_registry)

        progress.start_stage("onboarding_mart", detail="reconstruindo onboarding")
        onboarding = build_onboarding_mart(out_conn)
        progress.complete_stage("onboarding_mart", detail="onboarding mart ready")
        print("[build] onboarding mart ready", flush=True)
        persist_table(out_conn, cfg.output_dir, "mart_onboarding_population_v1", onboarding)
        progress.start_stage("first_session_journey", detail="resumindo 1ª sessão e 1º evento")
        journey = build_first_session_journey_mart(out_conn)
        progress.complete_stage("first_session_journey", detail="first session journey ready")
        print("[build] first session journey ready", flush=True)
        persist_table(out_conn, cfg.output_dir, "mart_first_session_journey_v1", journey)

        progress.start_stage("future_metrics", detail="calculando métricas futuras")
        future_metrics = build_future_metrics(out_conn)
        future_metrics = apply_custom_metric_overrides(future_metrics, setup.RUNTIME_CONFIG)
        progress.complete_stage("future_metrics", detail="future metrics ready")
        print("[build] future metrics ready", flush=True)
        persist_table(out_conn, cfg.output_dir, "mart_future_metrics_v1", future_metrics)

        feature_registry = build_feature_registry()
        persist_table(out_conn, cfg.output_dir, "governance_feature_registry_v1", feature_registry)
        feature_eligibility_log = build_feature_eligibility_log(feature_registry, track_registry)
        persist_table(out_conn, cfg.output_dir, "governance_feature_eligibility_v1", feature_eligibility_log)

        progress.start_stage("definition_search", detail="buscando candidatos de definição")
        candidate_df, candidate_test_df, selection_df = build_definition_search(future_metrics, candidate_metric_registry)
        progress.complete_stage("definition_search", detail="definition search ready")
        print("[build] definition search ready", flush=True)
        persist_table(out_conn, cfg.output_dir, "core_definition_candidates_train_v1", candidate_df)
        persist_table(out_conn, cfg.output_dir, "core_definition_candidates_test_frontier_v1", candidate_test_df)
        persist_table(out_conn, cfg.output_dir, "core_definition_selection_v1", selection_df)

        definition_b_row = selection_df.loc[selection_df["definition_group"] == "definition_b"].iloc[0].to_dict()
        official_a_rows = selection_df[(selection_df["definition_name"] == "definition_a") & selection_df["official_status"].str.startswith("official")].copy()
        label_registry = build_label_registry(official_a_rows, definition_b_row)
        persist_table(out_conn, cfg.output_dir, "governance_label_registry_v1", label_registry)

        progress.start_stage("definition_comparison", detail="comparando definições oficiais")
        definition_fold_eval, definition_frontier = compare_official_definitions(future_metrics, selection_df)
        progress.complete_stage("definition_comparison", detail="definition comparison ready")
        print("[build] definition comparison ready", flush=True)
        persist_table(out_conn, cfg.output_dir, "core_definition_external_validation_v1", definition_fold_eval)
        persist_table(out_conn, cfg.output_dir, "core_definition_frontier_v1", definition_frontier)

        frame = build_official_frame(journey, future_metrics, selection_df)
        official_definition_names = sorted([name for name in definition_frontier.get("definition_name", pd.Series(dtype=str)).dropna().unique().tolist() if name in frame.columns])
        if "definition_b_label" in frame.columns and "definition_b_label" not in official_definition_names:
            official_definition_names.append("definition_b_label")
        if not official_definition_names and "definition_b_label" in frame.columns:
            official_definition_names = ["definition_b_label"]
        scoring_scenarios = build_scoring_scenarios(frame, feature_registry, track_registry, definition_frontier)
        persist_table(out_conn, cfg.output_dir, "core_scoring_scenarios_v1", scoring_scenarios)

        progress.start_stage("leakage_audit", detail="auditando leakage")
        leakage_audit = build_leakage_audit(feature_registry, label_registry, scoring_scenarios)
        leakage_summary = summarize_leakage_audit(leakage_audit)
        progress.complete_stage("leakage_audit", detail="leakage audit ready")
        print("[build] leakage audit ready", flush=True)
        persist_table(out_conn, cfg.output_dir, "governance_leakage_audit_v1", leakage_audit)
        persist_table(out_conn, cfg.output_dir, "governance_leakage_summary_v1", leakage_summary)

        progress.start_stage("model_evaluation", detail="avaliando cenários oficiais")
        model_fold_metrics, model_predictions, model_frontier, inner_split_audit, feature_importance, post_model_output_status = evaluate_model_problems(
            frame,
            feature_registry,
            scoring_scenarios,
            compute_feature_importance=False,
            task_store=task_store,
            progress_stage_key="model_evaluation",
            progress_callback=progress.update_stage,
        )
        progress.complete_stage("model_evaluation", detail="model evaluation ready")
        print("[build] model evaluation ready", flush=True)
        persist_table(out_conn, cfg.output_dir, "core_model_fold_metrics_v1", model_fold_metrics)
        persist_table(out_conn, cfg.output_dir, "core_model_predictions_v1", model_predictions)
        persist_table(out_conn, cfg.output_dir, "core_model_frontier_v1", model_frontier)
        persist_table(out_conn, cfg.output_dir, "core_model_calibration_audit_v1", inner_split_audit)
        all_post_model_output_status = post_model_output_status.copy()
        official_predictions = filter_official_predictions(model_predictions)

        progress.start_stage("cv_score_robustness", detail="resumindo robustez do score")
        cv_score_folds, cv_score_summary = build_cv_score_robustness_outputs(official_predictions)
        progress.complete_stage("cv_score_robustness", detail="cv score robustness ready")
        progress.start_stage("cv_metric_robustness", detail="resumindo robustez das métricas")
        cv_metric_folds, cv_metric_summary = build_cv_metric_robustness_outputs(model_fold_metrics)
        progress.complete_stage("cv_metric_robustness", detail="cv metric robustness ready")
        persist_table(out_conn, cfg.output_dir, "core_cv_score_folds_v1", cv_score_folds)
        persist_table(out_conn, cfg.output_dir, "core_cv_score_summary_v1", cv_score_summary)
        persist_table(out_conn, cfg.output_dir, "core_cv_metric_folds_v1", cv_metric_folds)
        persist_table(out_conn, cfg.output_dir, "core_cv_metric_summary_v1", cv_metric_summary)

        progress.start_stage("prediction_bootstrap", detail="calculando intervalos bootstrap")
        bootstrap_df = bootstrap_prediction_metrics(official_predictions)
        progress.complete_stage("prediction_bootstrap", detail="prediction bootstrap ready")
        persist_table(out_conn, cfg.output_dir, "core_prediction_bootstrap_v1", bootstrap_df)

        progress.start_stage("definition_b_feature_block_gain", detail="medindo ganho incremental por bloco")
        definition_b_block_gain_folds, definition_b_block_gain_summary = build_definition_b_feature_block_gain_diagnostics(
            frame,
            feature_registry,
            scoring_scenarios,
            task_store=task_store,
            progress_callback=progress.update_stage,
        )
        progress.complete_stage("definition_b_feature_block_gain", detail="definition B feature blocks ready")
        persist_table(out_conn, cfg.output_dir, "core_definition_b_feature_block_gain_folds_v1", definition_b_block_gain_folds)
        persist_table(out_conn, cfg.output_dir, "core_definition_b_feature_block_gain_summary_v1", definition_b_block_gain_summary)

        progress.start_stage("definition_b_excessive_separation", detail="auditando separação excessiva")
        definition_b_excessive_separation = build_definition_b_excessive_separation_outputs(model_frontier)
        progress.complete_stage("definition_b_excessive_separation", detail="definition B excessive separation ready")
        persist_table(out_conn, cfg.output_dir, "core_definition_b_excessive_separation_v1", definition_b_excessive_separation)

        progress.start_stage("reference_scope", detail="selecionando escopo de referência")
        reference_scope = select_reference_scope_for_post_model_outputs(
            model_frontier,
            predictions=official_predictions,
            definition_selection=selection_df,
            definition_frontier=definition_frontier,
            scoring_scenarios=scoring_scenarios,
        )
        progress.complete_stage("reference_scope", detail="reference scope ready")
        refit_pairs = {
            (str(row["problem_key"]), str(row["model_name"]))
            for row in reference_scope[["problem_key", "model_name"]].dropna().to_dict(orient="records")
        } if not reference_scope.empty else set()

        inspection_predictions = official_predictions.copy()
        progress.start_stage("post_model_refit", detail="refit pós-modelo")
        if cfg.compute_post_model_refit and refit_pairs:
            _, inspection_predictions_raw, _, _, feature_importance, post_model_output_status = evaluate_model_problems(
                frame,
                feature_registry,
                scoring_scenarios,
                allowed_problem_model_pairs=refit_pairs,
                compute_feature_importance=True,
                task_store=task_store,
                progress_stage_key="post_model_refit",
                progress_callback=progress.update_stage,
            )
            inspection_predictions = filter_official_predictions(inspection_predictions_raw)
            all_post_model_output_status = pd.concat([all_post_model_output_status, post_model_output_status], ignore_index=True)
            progress.complete_stage("post_model_refit", detail=f"post-model refit ready | pares={len(refit_pairs)}")
        else:
            progress.complete_stage("post_model_refit", detail="post-model refit pulado")
        persist_table(out_conn, cfg.output_dir, "post_model_feature_importance_v1", feature_importance)
        persist_table(out_conn, cfg.output_dir, "post_model_reference_selection_v1", reference_scope)
        persist_table(out_conn, cfg.output_dir, "governance_post_model_output_status_v1", all_post_model_output_status)

        post_model_predictions = inspection_predictions if not inspection_predictions.empty else official_predictions
        progress.start_stage("threshold_outputs", detail="materializando thresholds, confusão e bandas")
        threshold_metrics, confusion_matrix_df, band_summary, monthly_fit = build_threshold_post_model_outputs(post_model_predictions)
        progress.complete_stage("threshold_outputs", detail="threshold outputs ready")
        progress.start_stage("cv_threshold_robustness", detail="resumindo robustez dos cutoffs")
        cv_threshold_folds, cv_confusion_folds, cv_threshold_summary, cv_confusion_summary = build_cv_threshold_robustness_outputs(post_model_predictions)
        progress.complete_stage("cv_threshold_robustness", detail="cv threshold robustness ready")
        persist_table(out_conn, cfg.output_dir, "post_model_threshold_metrics_v1", threshold_metrics)
        persist_table(out_conn, cfg.output_dir, "post_model_confusion_matrix_v1", confusion_matrix_df)
        persist_table(out_conn, cfg.output_dir, "post_model_band_summary_v1", band_summary)
        persist_table(out_conn, cfg.output_dir, "post_model_monthly_fit_v1", monthly_fit)
        persist_table(out_conn, cfg.output_dir, "post_model_cv_threshold_folds_v1", cv_threshold_folds)
        persist_table(out_conn, cfg.output_dir, "post_model_cv_confusion_folds_v1", cv_confusion_folds)
        persist_table(out_conn, cfg.output_dir, "post_model_cv_threshold_summary_v1", cv_threshold_summary)
        persist_table(out_conn, cfg.output_dir, "post_model_cv_confusion_summary_v1", cv_confusion_summary)

        progress.start_stage("cluster_outputs", detail="materializando saídas de cluster")
        cluster_assignment, cluster_profile, cluster_summary, cluster_validation = build_cluster_outputs(
            out_conn,
            post_model_predictions,
        )
        progress.complete_stage("cluster_outputs", detail="cluster outputs ready")
        persist_table(out_conn, cfg.output_dir, "post_model_cluster_assignment_v1", cluster_assignment)
        persist_table(out_conn, cfg.output_dir, "post_model_cluster_profile_v1", cluster_profile)
        persist_table(out_conn, cfg.output_dir, "post_model_cluster_summary_v1", cluster_summary)
        persist_table(out_conn, cfg.output_dir, "post_model_cluster_validation_v1", cluster_validation)

        progress.start_stage("heavy_user_outputs", detail="materializando saídas de heavy-user")
        heavy_scores, heavy_profile, heavy_summary = build_heavy_user_outputs(
            frame,
            post_model_predictions,
        )
        progress.complete_stage("heavy_user_outputs", detail="heavy-user outputs ready")
        persist_table(out_conn, cfg.output_dir, "post_model_heavy_user_scores_v1", heavy_scores)
        persist_table(out_conn, cfg.output_dir, "post_model_heavy_user_profile_v1", heavy_profile)
        persist_table(out_conn, cfg.output_dir, "post_model_heavy_user_summary_v1", heavy_summary)

        progress.start_stage("navigation_outputs", detail="materializando navegação")
        nav_sequences, nav_transitions = build_navigation_outputs(journey, frame, official_definition_names)
        progress.complete_stage("navigation_outputs", detail="navigation outputs ready")
        print("[build] navigation outputs ready", flush=True)
        persist_table(out_conn, cfg.output_dir, "core_navigation_sequences_v1", nav_sequences)
        persist_table(out_conn, cfg.output_dir, "core_navigation_transitions_v1", nav_transitions)

        progress.start_stage("summary_write", detail="escrevendo resumo final")
        summary_payload = build_summary_payload(cfg, model_frontier, definition_frontier, arbitrariness_registry)
        write_json(cfg.output_dir / "metadata" / "build_summary_v1.json", summary_payload)
        progress.complete_stage("summary_write", detail="summary write ready")
        return summary_payload
    finally:
        out_conn.close()

def run_build_for_spec(spec: AnalysisSpec, modelled_duckdb: Path, output_dir: Path, compute_post_model_refit: bool) -> dict[str, Any]:
    cfg = EnginePaths(
        project_root=setup.PROJECT_ROOT,
        modelled_duckdb=modelled_duckdb.resolve(),
        output_dir=output_dir.resolve(),
        output_duckdb=(output_dir.resolve() / "duckdb" / "build.duckdb"),
        compute_post_model_refit=compute_post_model_refit,
    )
    return run_build(cfg, runtime_config=RuntimeBuildConfig.from_analysis_spec(spec))

def main() -> None:
    args = parse_args()
    output_dir = (args.output_dir or (setup.PROJECT_ROOT / "build")).resolve()
    cfg = EnginePaths(
        project_root=setup.PROJECT_ROOT,
        modelled_duckdb=args.modelled_duckdb.resolve(),
        output_dir=output_dir,
        output_duckdb=output_dir / "duckdb" / "build.duckdb",
        compute_post_model_refit=not args.skip_post_model_refit,
    )
    run_build(cfg, runtime_config=RuntimeBuildConfig.from_payload(setup.RUNTIME_OVERRIDES))
