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


PRIMARY_COHORT = "near_entry_0_1m"
PRIMARY_COHORT_FLAG = "cohort_variant_near_entry_0_1m"
RISKY_JOURNEYS = {
    "one_session_download_no_repeat": "download_then_repeat",
    "one_session_activity_no_repeat": "activity_then_repeat",
    "session_without_interaction": "__surface_interaction_baseline__",
}
MIN_PRIORITY_TEACHERS = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Etapa 08 v2: priorizacao de UX por fluxo, tela e transicao.")
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
            "Tabelas obrigatorias ausentes para etapa_08_ux_flow_prioritization_v2.py: "
            + ", ".join(sorted(missing))
        )


def normalize_text(series: pd.Series, default: str = "missing") -> pd.Series:
    out = series.fillna(default).astype(str)
    out = out.replace({"<missing>": default, "None": default, "nan": default}).str.strip()
    return out.replace("", default)


def select_primary_cohort(journey: pd.DataFrame) -> pd.DataFrame:
    return journey[pd.to_numeric(journey[PRIMARY_COHORT_FLAG], errors="coerce").fillna(0) == 1].copy()


def load_journey(conn: Any) -> pd.DataFrame:
    journey = conn.execute("SELECT * FROM mart_teacher_first_session_journey_v2").fetchdf()
    if journey.empty:
        return journey
    for col in [
        "first_month",
        "data_entrada_month",
        "first_session_start_ts",
        "first_session_end_ts",
    ]:
        if col in journey.columns:
            journey[col] = pd.to_datetime(journey[col], errors="coerce")
    numeric_cols = [
        "returned_active_m1",
        "returned_any_session_m1",
        "session_count_month",
        "first_session_has_interaction_flag",
        "first_session_has_meaningful_action_flag",
        "second_session_same_month_flag",
        "activation_threshold_hits_count",
        "activation_core_3of4_flag",
        "first_session_missing_flag",
    ]
    for col in numeric_cols:
        if col in journey.columns:
            journey[col] = pd.to_numeric(journey[col], errors="coerce")
    journey["first_session_entry_surface_top"] = normalize_text(journey["first_session_entry_surface_top"])
    journey["first_session_exit_state"] = normalize_text(journey["first_session_exit_state"])
    journey["first_session_device_bucket"] = normalize_text(journey["first_session_device_bucket"], default="unknown")
    for idx in range(1, 6):
        col = f"step_{idx}_token"
        if col in journey.columns:
            journey[col] = normalize_text(journey[col])
    journey["step_sequence_first5"] = normalize_text(journey["step_sequence_first5"])
    journey["journey_pattern_label"] = normalize_text(journey["journey_pattern_label"])
    return journey


def observed_step_tokens(row: pd.Series) -> List[str]:
    tokens: List[str] = []
    for idx in range(1, 6):
        value = str(row.get(f"step_{idx}_token", "missing")).strip().lower()
        if value in {"", "missing", "<missing>", "none", "nan"}:
            break
        tokens.append(value)
    return tokens


def build_transition_mart(journey: pd.DataFrame) -> pd.DataFrame:
    cohort = select_primary_cohort(journey)
    rows: List[Dict[str, Any]] = []
    for row in cohort.to_dict(orient="records"):
        teacher_id = row["teacher_unique_id"]
        entry_surface = str(row.get("first_session_entry_surface_top", "missing"))
        exit_state = str(row.get("first_session_exit_state", "missing"))
        journey_label = str(row.get("journey_pattern_label", "missing"))
        observed_tokens = observed_step_tokens(pd.Series(row))
        previous_node = f"entry::{entry_surface}"
        transition_order = 1

        if not observed_tokens:
            rows.append(
                {
                    "teacher_unique_id": teacher_id,
                    "first_month": row.get("first_month"),
                    "journey_pattern_label": journey_label,
                    "first_session_entry_surface_top": entry_surface,
                    "first_session_exit_state": exit_state,
                    "first_session_device_bucket": row.get("first_session_device_bucket", "unknown"),
                    "transition_order": transition_order,
                    "from_node": previous_node,
                    "to_node": f"exit::{exit_state}",
                    "from_node_kind": "entry_surface",
                    "to_node_kind": "exit_state",
                    "transition_signature": f"{previous_node}->{f'exit::{exit_state}'}",
                    "returned_active_m1": row.get("returned_active_m1"),
                    "returned_any_session_m1": row.get("returned_any_session_m1"),
                    "second_session_same_month_flag": row.get("second_session_same_month_flag"),
                    "activation_core_3of4_flag": row.get("activation_core_3of4_flag"),
                }
            )
            continue

        for token in observed_tokens:
            current_node = f"step::{token}"
            rows.append(
                {
                    "teacher_unique_id": teacher_id,
                    "first_month": row.get("first_month"),
                    "journey_pattern_label": journey_label,
                    "first_session_entry_surface_top": entry_surface,
                    "first_session_exit_state": exit_state,
                    "first_session_device_bucket": row.get("first_session_device_bucket", "unknown"),
                    "transition_order": transition_order,
                    "from_node": previous_node,
                    "to_node": current_node,
                    "from_node_kind": "entry_surface" if transition_order == 1 else "step",
                    "to_node_kind": "step",
                    "transition_signature": f"{previous_node}->{current_node}",
                    "returned_active_m1": row.get("returned_active_m1"),
                    "returned_any_session_m1": row.get("returned_any_session_m1"),
                    "second_session_same_month_flag": row.get("second_session_same_month_flag"),
                    "activation_core_3of4_flag": row.get("activation_core_3of4_flag"),
                }
            )
            previous_node = current_node
            transition_order += 1

        rows.append(
            {
                "teacher_unique_id": teacher_id,
                "first_month": row.get("first_month"),
                "journey_pattern_label": journey_label,
                "first_session_entry_surface_top": entry_surface,
                "first_session_exit_state": exit_state,
                "first_session_device_bucket": row.get("first_session_device_bucket", "unknown"),
                "transition_order": transition_order,
                "from_node": previous_node,
                "to_node": f"exit::{exit_state}",
                "from_node_kind": "step",
                "to_node_kind": "exit_state",
                "transition_signature": f"{previous_node}->{f'exit::{exit_state}'}",
                "returned_active_m1": row.get("returned_active_m1"),
                "returned_any_session_m1": row.get("returned_any_session_m1"),
                "second_session_same_month_flag": row.get("second_session_same_month_flag"),
                "activation_core_3of4_flag": row.get("activation_core_3of4_flag"),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["first_month"] = pd.to_datetime(out["first_month"], errors="coerce")
    return out


def build_transition_summary(transitions: pd.DataFrame, journey: pd.DataFrame) -> pd.DataFrame:
    if transitions.empty:
        return pd.DataFrame()
    total_by_journey = (
        select_primary_cohort(journey)
        .groupby("journey_pattern_label", dropna=False)["teacher_unique_id"]
        .nunique()
        .rename("journey_teachers")
        .reset_index()
    )
    out = (
        transitions.groupby(
            [
                "journey_pattern_label",
                "first_session_entry_surface_top",
                "first_session_exit_state",
                "transition_order",
                "from_node",
                "to_node",
                "transition_signature",
            ],
            dropna=False,
        )
        .agg(
            teachers=("teacher_unique_id", "nunique"),
            returned_active_rate=("returned_active_m1", "mean"),
            returned_any_session_rate=("returned_any_session_m1", "mean"),
            share_second_session_same_month=("second_session_same_month_flag", "mean"),
            share_activation_core_3of4=("activation_core_3of4_flag", "mean"),
        )
        .reset_index()
    )
    out = out.merge(total_by_journey, on="journey_pattern_label", how="left")
    out["share_within_journey"] = out["teachers"] / out["journey_teachers"].replace(0, np.nan)
    return out.sort_values(["journey_pattern_label", "teachers"], ascending=[True, False]).reset_index(drop=True)


def build_surface_exit_summary(journey: pd.DataFrame) -> pd.DataFrame:
    cohort = select_primary_cohort(journey)
    if cohort.empty:
        return pd.DataFrame()
    out = (
        cohort.groupby(
            ["journey_pattern_label", "first_session_entry_surface_top", "first_session_exit_state"],
            dropna=False,
        )
        .agg(
            teachers=("teacher_unique_id", "nunique"),
            returned_active_rate=("returned_active_m1", "mean"),
            returned_any_session_rate=("returned_any_session_m1", "mean"),
            avg_activation_threshold_hits=("activation_threshold_hits_count", "mean"),
            share_mobile=("first_session_device_bucket", lambda s: (s.astype(str) == "mobile").mean()),
            share_unknown_device=("first_session_device_bucket", lambda s: (s.astype(str) == "unknown").mean()),
        )
        .reset_index()
        .sort_values("teachers", ascending=False)
        .reset_index(drop=True)
    )
    return out


def derive_break_transition(row: pd.Series) -> str:
    observed_tokens = []
    for idx in range(1, 6):
        token = str(row.get(f"step_{idx}_token", "missing")).strip().lower()
        if token in {"", "missing", "<missing>", "none", "nan"}:
            break
        observed_tokens.append(token)
    exit_state = str(row.get("first_session_exit_state", "missing"))
    if not observed_tokens:
        return f"entry::{row.get('first_session_entry_surface_top', 'missing')}->exit::{exit_state}"
    return f"step::{observed_tokens[-1]}->exit::{exit_state}"


def comparator_subset(cohort: pd.DataFrame, journey_label: str, entry_surface: str) -> tuple[pd.DataFrame, str]:
    comparator_label = RISKY_JOURNEYS[journey_label]
    if comparator_label == "__surface_interaction_baseline__":
        same_surface = cohort[
            (cohort["first_session_entry_surface_top"] == entry_surface)
            & (cohort["first_session_has_interaction_flag"].fillna(0) == 1)
        ].copy()
        if len(same_surface) >= MIN_PRIORITY_TEACHERS:
            return same_surface, "same_surface_interactive_baseline"
        global_interactive = cohort[cohort["first_session_has_interaction_flag"].fillna(0) == 1].copy()
        return global_interactive, "global_interactive_baseline"

    same_surface = cohort[
        (cohort["journey_pattern_label"] == comparator_label)
        & (cohort["first_session_entry_surface_top"] == entry_surface)
    ].copy()
    if len(same_surface) >= MIN_PRIORITY_TEACHERS:
        return same_surface, "same_surface_repeat_baseline"
    global_journey = cohort[cohort["journey_pattern_label"] == comparator_label].copy()
    return global_journey, "global_repeat_baseline"


def recommended_action(row: pd.Series) -> str:
    journey_label = str(row["journey_pattern_label"])
    exit_state = str(row["first_session_exit_state"])
    entry_surface = str(row["first_session_entry_surface_top"])
    break_transition = str(row["break_transition"])
    if journey_label == "session_without_interaction":
        return f"Instrumentar e simplificar a superficie `{entry_surface}` para capturar a primeira acao antes do encerramento."
    if journey_label == "one_session_download_no_repeat":
        return f"Na superficie `{entry_surface}`, adicionar continuidade apos `{break_transition}` para puxar segunda sessao em vez de tratar download como fim."
    if exit_state == "ended_after_view_only":
        return f"Na superficie `{entry_surface}`, transformar consumo de view em proximo passo explicito antes de sair da sessao."
    return f"Na superficie `{entry_surface}`, reduzir o encerramento apos `{break_transition}` com CTA de continuidade e retorno."


def build_flow_priority(journey: pd.DataFrame) -> pd.DataFrame:
    cohort = select_primary_cohort(journey)
    if cohort.empty:
        return pd.DataFrame()

    risky = cohort[cohort["journey_pattern_label"].isin(RISKY_JOURNEYS.keys())].copy()
    if risky.empty:
        return pd.DataFrame()
    risky["break_transition"] = risky.apply(derive_break_transition, axis=1)
    group_cols = [
        "journey_pattern_label",
        "first_session_entry_surface_top",
        "first_session_exit_state",
        "break_transition",
    ]
    grouped = (
        risky.groupby(group_cols, dropna=False)
        .agg(
            affected_teachers=("teacher_unique_id", "nunique"),
            subset_return_active_rate=("returned_active_m1", "mean"),
            subset_return_any_session_rate=("returned_any_session_m1", "mean"),
            share_unknown_device=("first_session_device_bucket", lambda s: (s.astype(str) == "unknown").mean()),
            share_mobile=("first_session_device_bucket", lambda s: (s.astype(str) == "mobile").mean()),
        )
        .reset_index()
    )
    step_examples = (
        risky.groupby(group_cols, dropna=False)["step_sequence_first5"]
        .apply(lambda s: " || ".join(s.astype(str).value_counts().head(3).index.tolist()))
        .rename("top_step_sequence_examples")
        .reset_index()
    )
    grouped = grouped.merge(step_examples, on=group_cols, how="left")
    grouped = grouped[grouped["affected_teachers"] >= MIN_PRIORITY_TEACHERS].copy()
    if grouped.empty:
        return grouped

    rows: List[Dict[str, Any]] = []
    for row in grouped.to_dict(orient="records"):
        comparator, comparator_scope = comparator_subset(
            cohort,
            row["journey_pattern_label"],
            row["first_session_entry_surface_top"],
        )
        if comparator.empty:
            continue
        comparator_rate = float(pd.to_numeric(comparator["returned_active_m1"], errors="coerce").mean())
        comparator_any_session_rate = float(pd.to_numeric(comparator["returned_any_session_m1"], errors="coerce").mean())
        delta_pp = (comparator_rate - float(row["subset_return_active_rate"])) * 100
        priority_score = float(row["affected_teachers"]) * max(0.0, comparator_rate - float(row["subset_return_active_rate"]))
        issue_id = (
            f"{row['journey_pattern_label']}__"
            f"{row['first_session_entry_surface_top']}__"
            f"{row['first_session_exit_state']}__"
            f"{row['break_transition']}"
        )
        flow_row = {
            **row,
            "issue_id": issue_id.replace(" ", "_"),
            "comparator_scope": comparator_scope,
            "comparator_teachers": int(comparator["teacher_unique_id"].nunique()),
            "comparator_return_active_rate": comparator_rate,
            "comparator_return_any_session_rate": comparator_any_session_rate,
            "delta_return_active_pp": delta_pp,
            "priority_score": priority_score,
        }
        flow_row["recommended_action"] = recommended_action(pd.Series(flow_row))
        rows.append(flow_row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["priority_score", "affected_teachers"], ascending=[False, False]).reset_index(drop=True)
    out["priority_rank"] = np.arange(1, len(out) + 1)
    return out


def build_teacher_priority(journey: pd.DataFrame, flow_priority: pd.DataFrame) -> pd.DataFrame:
    cohort = select_primary_cohort(journey)
    if cohort.empty or flow_priority.empty:
        return pd.DataFrame()
    work = cohort.copy()
    work["break_transition"] = work.apply(derive_break_transition, axis=1)
    join_cols = [
        "journey_pattern_label",
        "first_session_entry_surface_top",
        "first_session_exit_state",
        "break_transition",
    ]
    priority_cols = join_cols + ["issue_id", "priority_rank", "recommended_action", "delta_return_active_pp"]
    out = work.merge(flow_priority[priority_cols], on=join_cols, how="inner")
    out = out[
        [
            "teacher_unique_id",
            "first_month",
            "issue_id",
            "priority_rank",
            "journey_pattern_label",
            "first_session_entry_surface_top",
            "first_session_device_bucket",
            "first_session_exit_state",
            "step_sequence_first5",
            "break_transition",
            "returned_active_m1",
            "returned_any_session_m1",
            "delta_return_active_pp",
            "recommended_action",
        ]
    ].copy()
    out = out.sort_values(["priority_rank", "teacher_unique_id", "first_month"]).reset_index(drop=True)
    out["is_primary_issue_for_teacher"] = (out.groupby("teacher_unique_id").cumcount() == 0).astype(int)
    return out


def build_summary_payload(
    transitions: pd.DataFrame,
    flow_priority: pd.DataFrame,
    teacher_priority: pd.DataFrame,
) -> Dict[str, Any]:
    top_issues = flow_priority.head(8).copy() if not flow_priority.empty else pd.DataFrame()
    return {
        "generated_at_utc": utc_now_iso(),
        "transition_rows": int(len(transitions)),
        "teachers_in_priority_flows": int(teacher_priority["teacher_unique_id"].nunique()) if not teacher_priority.empty else 0,
        "primary_issues_teachers": int(
            teacher_priority[teacher_priority["is_primary_issue_for_teacher"] == 1]["teacher_unique_id"].nunique()
        )
        if not teacher_priority.empty
        else 0,
        "top_issues": top_issues[
            [
                "priority_rank",
                "issue_id",
                "affected_teachers",
                "delta_return_active_pp",
                "first_session_entry_surface_top",
                "first_session_exit_state",
            ]
        ].to_dict(orient="records")
        if not top_issues.empty
        else [],
    }


def write_summary_markdown(path: Path, summary: Dict[str, Any], flow_priority: pd.DataFrame) -> None:
    lines = [
        "# UX flow prioritization v2",
        "",
        f"- Gerado em UTC: {summary['generated_at_utc']}",
        f"- Linhas na mart de transicoes: {summary['transition_rows']}",
        f"- Professores em flows priorizados: {summary['teachers_in_priority_flows']}",
        f"- Professores com issue primaria identificada: {summary['primary_issues_teachers']}",
        "",
        "## Top Flow Issues",
    ]
    if flow_priority.empty:
        lines.append("- none")
    else:
        for _, row in flow_priority.head(8).iterrows():
            lines.append(
                f"- `#{int(row['priority_rank'])}` `{row['issue_id']}` | afetados={int(row['affected_teachers'])} | delta_pp={float(row['delta_return_active_pp']):.2f} | {row['recommended_action']}"
            )
    write_markdown(path, lines)


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg: V2Config = build_config(args.base_dir, args.data_dir, args.output_dir)
    conn = connect_duckdb(cfg)
    try:
        require_tables(conn, ["mart_teacher_first_session_journey_v2"])
        journey = load_journey(conn)
        transitions = build_transition_mart(journey)
        surface_exit_summary = build_surface_exit_summary(journey)
        transition_summary = build_transition_summary(transitions, journey)
        flow_priority = build_flow_priority(journey)
        teacher_priority = build_teacher_priority(journey, flow_priority)

        outputs: Dict[str, pd.DataFrame] = {
            "mart_teacher_first_session_transitions_v2": transitions,
            "analytics_ux_surface_exit_summary_v2": surface_exit_summary,
            "analytics_ux_transition_summary_v2": transition_summary,
            "analytics_ux_flow_priority_v2": flow_priority,
            "analytics_ux_teacher_flow_priority_v2": teacher_priority,
        }
        for name, df in outputs.items():
            persist_output(conn, cfg, name, df)

        summary = build_summary_payload(transitions, flow_priority, teacher_priority)
        write_json(cfg.output_dir / "json" / "ux_flow_prioritization_summary_v2.json", summary)
        write_summary_markdown(cfg.output_dir / "audit" / "ux_flow_prioritization_summary_v2.md", summary, flow_priority)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
