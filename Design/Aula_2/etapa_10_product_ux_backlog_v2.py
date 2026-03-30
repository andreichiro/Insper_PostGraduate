#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from analytics_v2_common import (
    DEFAULT_BASE_DIR,
    V2Config,
    build_config,
    connect_duckdb,
    persist_df_to_duckdb,
    setup_logging,
    utc_now_iso,
    write_df_bundle,
    write_json,
    write_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Etapa 10 v2: backlog priorizado de produto/UX.")
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
            "Tabelas obrigatorias ausentes para etapa_10_product_ux_backlog_v2.py: "
            + ", ".join(sorted(missing))
        )


def load_table(conn: Any, name: str) -> pd.DataFrame:
    return conn.execute(f"SELECT * FROM {name}").fetchdf()


def activation_target_string(thresholds: pd.DataFrame) -> str:
    if thresholds.empty:
        return ">=2 sessoes; >=12 eventos em 7d; >=6 views; >=16.684 min"
    preferred = thresholds[
        (thresholds["cohort_variant"] == "near_entry_0_1m")
        & (thresholds["outcome_variant"] == "returned_active_m1")
        & (thresholds["recommended_threshold_kind"] == "frozen_default")
    ].copy()
    if preferred.empty:
        preferred = thresholds.copy()
    values = (
        preferred.groupby("metric_name", as_index=False)["recommended_threshold_value"]
        .max()
        .sort_values("metric_name")
    )
    parts: List[str] = []
    label_map = {
        "session_count_month": "sessoes",
        "first7d_events": "eventos em 7d",
        "content_views_month": "views",
        "total_session_minutes_month": "min",
    }
    for row in values.to_dict(orient="records"):
        metric = row["metric_name"]
        label = label_map.get(metric, metric)
        value = row["recommended_threshold_value"]
        parts.append(f">={value:g} {label}")
    return "; ".join(parts) if parts else ">=2 sessoes; >=12 eventos em 7d; >=6 views; >=16.684 min"


def top_predictive_support(features: pd.DataFrame, target: str, model_name: str) -> str:
    subset = features[
        (features["target"] == target)
        & (features["window_label"] == "all_history")
        & (features["model_name"] == model_name)
    ].copy()
    if subset.empty:
        return "session_count_month, active_days_month, total_session_minutes_month, content_views_month"
    if model_name.endswith("random_forest"):
        subset = subset.sort_values("importance_value", ascending=False).head(6)
    else:
        subset = subset.sort_values("importance_value", key=lambda s: s.abs(), ascending=False).head(6)
    clean_names = (
        subset["feature_name"]
        .astype(str)
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
        .str.replace("currentsubject_group_", "subject=", regex=False)
        .tolist()
    )
    return ", ".join(clean_names)


def parent_issue_from_flow(journey_label: str) -> str:
    mapping = {
        "one_session_download_no_repeat": "one_session_download_no_repeat",
        "one_session_activity_no_repeat": "one_session_activity_no_repeat",
        "session_without_interaction": "first_session_no_interaction",
    }
    return mapping.get(journey_label, journey_label)


def flow_hypothesis(row: pd.Series) -> str:
    surface = row["first_session_entry_surface_top"]
    transition = row["break_transition"]
    if row["journey_pattern_label"] == "one_session_download_no_repeat":
        return (
            f"Se a superficie `{surface}` oferecer continuidade explicita logo apos `{transition}`, "
            "mais usuarios construirao segunda sessao no mesmo mes e retorno ativo em m+1."
        )
    if row["journey_pattern_label"] == "one_session_activity_no_repeat":
        return (
            f"Se a superficie `{surface}` transformar `{transition}` em proximo passo claro, "
            "mais usuarios repetirao uso no mesmo mes em vez de encerrar a jornada na primeira sessao."
        )
    return (
        f"Se reduzirmos friccao e melhorarmos instrumentacao em `{surface}`, "
        "mais usuarios farao a primeira acao observavel e terao maior chance de retorno."
    )


def flow_primary_kpi(row: pd.Series) -> str:
    if row["journey_pattern_label"] == "session_without_interaction":
        return "first_session_has_interaction_rate"
    return "second_session_same_month_rate"


def flow_secondary_kpis(row: pd.Series, activation_targets: str) -> str:
    if row["journey_pattern_label"] == "session_without_interaction":
        return "returned_active_m1, returned_any_session_m1, share_unknown_device"
    return f"returned_active_m1, returned_any_session_m1, activation_targets=({activation_targets})"


def flow_user_group(row: pd.Series) -> str:
    surface = row["first_session_entry_surface_top"]
    exit_state = row["first_session_exit_state"]
    return (
        f"Professores near_entry_0_1m que entram por `{surface}` e encerram a primeira sessao em `{exit_state}` "
        f"no fluxo `{row['break_transition']}`."
    )


def build_flow_backlog(
    flow_priority: pd.DataFrame,
    decision_table: pd.DataFrame,
    total_cohort: int,
    activation_targets: str,
    predictive_support: str,
) -> pd.DataFrame:
    evidence_map = (
        decision_table.set_index("issue_id")["evidence_class"].to_dict()
        if not decision_table.empty and "issue_id" in decision_table.columns
        else {}
    )
    rows: List[Dict[str, Any]] = []
    for row in flow_priority.to_dict(orient="records"):
        parent_issue = parent_issue_from_flow(str(row["journey_pattern_label"]))
        rows.append(
            {
                "backlog_lane": "flow",
                "priority_rank_in_lane": int(row["priority_rank"]),
                "issue_id": row["issue_id"],
                "parent_issue_id": parent_issue,
                "issue_stage": "onboarding",
                "evidence_class": evidence_map.get(parent_issue, "supported_correlational"),
                "decision_direction": "do",
                "affected_teachers": int(row["affected_teachers"]),
                "affected_share_cohort": float(row["affected_teachers"]) / max(1, total_cohort),
                "priority_score": float(row["priority_score"]),
                "problem_statement": (
                    f"Usuarios entram por `{row['first_session_entry_surface_top']}` e quebram o fluxo em "
                    f"`{row['break_transition']}`, encerrando em `{row['first_session_exit_state']}`."
                ),
                "hypothesis": flow_hypothesis(pd.Series(row)),
                "proposed_intervention": row["recommended_action"],
                "expected_primary_kpi": flow_primary_kpi(pd.Series(row)),
                "expected_secondary_kpis": flow_secondary_kpis(pd.Series(row), activation_targets),
                "measurement_window": "same_month + m1",
                "target_user_group": flow_user_group(pd.Series(row)),
                "entry_surface": row["first_session_entry_surface_top"],
                "exit_state": row["first_session_exit_state"],
                "break_transition": row["break_transition"],
                "top_journey_examples": row.get("top_step_sequence_examples"),
                "observed_delta_return_active_pp": float(row["delta_return_active_pp"]),
                "predictive_support": predictive_support,
                "recommended_action": row["recommended_action"],
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("priority_rank_in_lane").reset_index(drop=True)
    return out


def program_hypothesis(issue_id: str, activation_targets: str) -> str:
    if issue_id == "low_activation_first_week":
        return (
            "Se o onboarding for guiado por metas comportamentais claras, "
            f"`{activation_targets}`, o retorno ativo em m+1 deve subir."
        )
    if issue_id == "heavy_user_abandonment_after_peak":
        return "Se houver protecao de churn logo apos o primeiro/segundo mes heavy, a perda pos-pico deve cair."
    if issue_id == "first_session_no_interaction":
        return "Se a primeira tela capturar acao observavel com menos friccao, menos usuarios morrerao na sessao inicial."
    if issue_id == "do_not_force_initial_download":
        return "Se o produto parar de otimizar para download como sucesso final, o onboarding ficará mais alinhado a retorno real."
    return "Se atacarmos este problema com a metrica correta, o retorno do produto deve melhorar."


def program_primary_kpi(issue_id: str) -> str:
    mapping = {
        "low_activation_first_week": "activation_core_3of4_rate",
        "heavy_user_abandonment_after_peak": "returned_after_last_heavy_m1_rate",
        "first_session_no_interaction": "first_session_has_interaction_rate",
        "do_not_force_initial_download": "activation_core_3of4_rate",
        "one_session_download_no_repeat": "second_session_same_month_rate",
        "one_session_activity_no_repeat": "second_session_same_month_rate",
    }
    return mapping.get(issue_id, "returned_active_m1")


def program_secondary_kpis(issue_id: str, activation_targets: str) -> str:
    if issue_id == "low_activation_first_week":
        return f"returned_active_m1, returned_any_session_m1, activation_targets=({activation_targets})"
    if issue_id == "heavy_user_abandonment_after_peak":
        return "abandoned_after_last_heavy_rate, heavy_months_returned_active_rate"
    if issue_id == "first_session_no_interaction":
        return "returned_active_m1, share_unknown_device, missing_surface_share"
    if issue_id == "do_not_force_initial_download":
        return "second_session_same_month_rate, returned_active_m1"
    return "returned_active_m1, returned_any_session_m1"


def build_program_backlog(
    decision_table: pd.DataFrame,
    predictive_support_return: str,
    predictive_support_churn: str,
    activation_targets: str,
) -> pd.DataFrame:
    if decision_table.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for row in decision_table.to_dict(orient="records"):
        support = predictive_support_return
        if row["issue_id"] == "heavy_user_abandonment_after_peak":
            support = predictive_support_churn
        rows.append(
            {
                "backlog_lane": "program",
                "priority_rank_in_lane": int(row["priority_rank"]),
                "issue_id": row["issue_id"],
                "parent_issue_id": row["issue_id"],
                "issue_stage": row["issue_stage"],
                "evidence_class": row["evidence_class"],
                "decision_direction": row["decision_direction"],
                "affected_teachers": int(row["affected_teachers"]),
                "affected_share_cohort": float(row["affected_share_cohort"]),
                "priority_score": float(row["priority_score"]),
                "problem_statement": row["note"],
                "hypothesis": program_hypothesis(str(row["issue_id"]), activation_targets),
                "proposed_intervention": row["decision"],
                "expected_primary_kpi": program_primary_kpi(str(row["issue_id"])),
                "expected_secondary_kpis": program_secondary_kpis(str(row["issue_id"]), activation_targets),
                "measurement_window": "same_month + m1",
                "target_user_group": f"Usuarios no slice `{row['target_slice']}` afetados por `{row['issue_id']}`.",
                "entry_surface": row["target_slice"],
                "exit_state": None,
                "break_transition": None,
                "top_journey_examples": None,
                "observed_delta_return_active_pp": float(row["delta_return_active_pp"]),
                "predictive_support": support,
                "recommended_action": row["decision"],
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("priority_rank_in_lane").reset_index(drop=True)
    return out


def build_backlog_table(flow_backlog: pd.DataFrame, program_backlog: pd.DataFrame) -> pd.DataFrame:
    frames = [df for df in [program_backlog, flow_backlog] if not df.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    lane_order = {"program": 1, "flow": 2}
    out["backlog_priority_rank"] = (
        out.assign(_lane_order=out["backlog_lane"].map(lane_order).fillna(99))
        .sort_values(["_lane_order", "priority_rank_in_lane", "priority_score"], ascending=[True, True, False])
        .reset_index(drop=True)
        .index
        + 1
    )
    out["backlog_id"] = out["backlog_lane"].str.upper() + "_" + out["backlog_priority_rank"].astype(str).str.zfill(3)
    cols = [
        "backlog_id",
        "backlog_priority_rank",
        "backlog_lane",
        "priority_rank_in_lane",
        "issue_id",
        "parent_issue_id",
        "issue_stage",
        "evidence_class",
        "decision_direction",
        "affected_teachers",
        "affected_share_cohort",
        "priority_score",
        "problem_statement",
        "hypothesis",
        "proposed_intervention",
        "expected_primary_kpi",
        "expected_secondary_kpis",
        "measurement_window",
        "target_user_group",
        "entry_surface",
        "exit_state",
        "break_transition",
        "top_journey_examples",
        "observed_delta_return_active_pp",
        "predictive_support",
        "recommended_action",
    ]
    return out[cols].sort_values("backlog_priority_rank").reset_index(drop=True)


def build_backlog_users(
    backlog: pd.DataFrame,
    teacher_flow_priority: pd.DataFrame,
    risk_cohorts: pd.DataFrame,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if not backlog.empty and not teacher_flow_priority.empty:
        flow_meta = backlog[backlog["backlog_lane"] == "flow"][
            ["backlog_id", "issue_id", "backlog_priority_rank", "backlog_lane", "recommended_action", "target_user_group"]
        ].copy()
        flow_meta = flow_meta.rename(columns={"recommended_action": "backlog_recommended_action"})
        flow_users = teacher_flow_priority.merge(flow_meta, on="issue_id", how="inner")
        flow_users = flow_users.rename(columns={"first_month": "reference_month"})
        flow_users["user_group_label"] = flow_users["target_user_group"]
        frames.append(
            flow_users[
                [
                    "teacher_unique_id",
                    "reference_month",
                    "backlog_id",
                    "backlog_priority_rank",
                    "backlog_lane",
                    "issue_id",
                    "journey_pattern_label",
                    "user_group_label",
                    "backlog_recommended_action",
                ]
            ].rename(columns={"backlog_recommended_action": "recommended_action"})
        )

    if not backlog.empty and not risk_cohorts.empty:
        risk_mapping = {
            "onboarding_low_activation_no_repeat": "low_activation_first_week",
            "heavy_user_abandoned_after_last_heavy": "heavy_user_abandonment_after_peak",
            "onboarding_session_without_interaction": "first_session_no_interaction",
            "onboarding_one_session_download_no_repeat": "one_session_download_no_repeat",
            "onboarding_one_session_activity_no_repeat": "one_session_activity_no_repeat",
        }
        program_meta = backlog[backlog["backlog_lane"] == "program"][
            ["backlog_id", "issue_id", "backlog_priority_rank", "backlog_lane", "recommended_action", "target_user_group"]
        ].copy()
        program_meta = program_meta.rename(columns={"recommended_action": "backlog_recommended_action"})
        risk_users = risk_cohorts.copy()
        risk_users["issue_id"] = risk_users["risk_cohort_label"].map(risk_mapping)
        risk_users = risk_users.dropna(subset=["issue_id"]).merge(program_meta, on="issue_id", how="inner")
        risk_users = risk_users.rename(columns={"reference_month": "reference_month"})
        risk_users["journey_pattern_label"] = risk_users["journey_pattern_label"].fillna("not_applicable")
        risk_users["user_group_label"] = risk_users["target_user_group"]
        frames.append(
            risk_users[
                [
                    "teacher_unique_id",
                    "reference_month",
                    "backlog_id",
                    "backlog_priority_rank",
                    "backlog_lane",
                    "issue_id",
                    "journey_pattern_label",
                    "user_group_label",
                    "backlog_recommended_action",
                ]
            ].rename(columns={"backlog_recommended_action": "recommended_action"})
        )

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    out["reference_month"] = pd.to_datetime(out["reference_month"], errors="coerce")
    return out.sort_values(["backlog_priority_rank", "teacher_unique_id", "reference_month"]).reset_index(drop=True)


def build_summary_payload(backlog: pd.DataFrame, backlog_users: pd.DataFrame) -> Dict[str, Any]:
    top_items = backlog.head(10).copy() if not backlog.empty else pd.DataFrame()
    return {
        "generated_at_utc": utc_now_iso(),
        "backlog_items": int(len(backlog)),
        "program_items": int((backlog["backlog_lane"] == "program").sum()) if not backlog.empty else 0,
        "flow_items": int((backlog["backlog_lane"] == "flow").sum()) if not backlog.empty else 0,
        "users_mapped": int(backlog_users["teacher_unique_id"].nunique()) if not backlog_users.empty else 0,
        "top_items": top_items[
            ["backlog_id", "backlog_lane", "issue_id", "affected_teachers", "expected_primary_kpi"]
        ].to_dict(orient="records")
        if not top_items.empty
        else [],
    }


def write_summary_markdown(path: Path, backlog: pd.DataFrame, summary: Dict[str, Any]) -> None:
    lines = [
        "# Product UX backlog v2",
        "",
        f"- Gerado em UTC: {summary['generated_at_utc']}",
        f"- Itens de backlog: {summary['backlog_items']}",
        f"- Itens programaticos: {summary['program_items']}",
        f"- Itens de fluxo: {summary['flow_items']}",
        f"- Usuarios mapeados: {summary['users_mapped']}",
        "",
        "## Top Program Backlog",
    ]
    program = backlog[backlog["backlog_lane"] == "program"].sort_values("priority_rank_in_lane").head(6)
    if program.empty:
        lines.append("- none")
    else:
        for _, row in program.iterrows():
            lines.append(
                f"- `{row['backlog_id']}` `{row['issue_id']}` | kpi=`{row['expected_primary_kpi']}` | afetados={int(row['affected_teachers'])} | {row['proposed_intervention']}"
            )
    lines.append("")
    lines.append("## Top Flow Backlog")
    flow = backlog[backlog["backlog_lane"] == "flow"].sort_values("priority_rank_in_lane").head(8)
    if flow.empty:
        lines.append("- none")
    else:
        for _, row in flow.iterrows():
            lines.append(
                f"- `{row['backlog_id']}` `{row['issue_id']}` | kpi=`{row['expected_primary_kpi']}` | afetados={int(row['affected_teachers'])} | {row['proposed_intervention']}"
            )
    write_markdown(path, lines)


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg: V2Config = build_config(args.base_dir, args.data_dir, args.output_dir)
    conn = connect_duckdb(cfg)
    try:
        require_tables(
            conn,
            [
                "analytics_ux_flow_priority_v2",
                "analytics_ux_teacher_flow_priority_v2",
                "analytics_ux_decision_table_v2",
                "analytics_ux_risk_cohorts_v2",
                "analytics_prediction_model_top_features_v2",
                "analytics_onboarding_thresholds_v2",
                "mart_teacher_first_session_journey_v2",
            ],
        )
        flow_priority = load_table(conn, "analytics_ux_flow_priority_v2")
        teacher_flow_priority = load_table(conn, "analytics_ux_teacher_flow_priority_v2")
        decision_table = load_table(conn, "analytics_ux_decision_table_v2")
        risk_cohorts = load_table(conn, "analytics_ux_risk_cohorts_v2")
        prediction_features = load_table(conn, "analytics_prediction_model_top_features_v2")
        thresholds = load_table(conn, "analytics_onboarding_thresholds_v2")
        journey = load_table(conn, "mart_teacher_first_session_journey_v2")

        total_cohort = int(
            pd.to_numeric(journey["cohort_variant_near_entry_0_1m"], errors="coerce").fillna(0).sum()
        ) if not journey.empty else 0
        activation_targets = activation_target_string(thresholds)
        predictive_support_return = top_predictive_support(
            prediction_features, "retornar_ativo_m1", "behavior_plus_profile_random_forest"
        )
        predictive_support_churn = top_predictive_support(
            prediction_features, "abandonar_m1", "behavior_plus_profile_random_forest"
        )

        flow_backlog = build_flow_backlog(
            flow_priority,
            decision_table,
            total_cohort,
            activation_targets,
            predictive_support_return,
        )
        program_backlog = build_program_backlog(
            decision_table,
            predictive_support_return,
            predictive_support_churn,
            activation_targets,
        )
        backlog = build_backlog_table(flow_backlog, program_backlog)
        backlog_users = build_backlog_users(backlog, teacher_flow_priority, risk_cohorts)

        outputs: Dict[str, pd.DataFrame] = {
            "analytics_product_ux_backlog_v2": backlog,
            "analytics_product_ux_backlog_users_v2": backlog_users,
        }
        for name, df in outputs.items():
            persist_output(conn, cfg, name, df)

        summary = build_summary_payload(backlog, backlog_users)
        write_json(cfg.output_dir / "json" / "product_ux_backlog_summary_v2.json", summary)
        write_summary_markdown(cfg.output_dir / "audit" / "product_ux_backlog_summary_v2.md", backlog, summary)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
