#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

from analytics_v2_common import (
    DEFAULT_BASE_DIR,
    V2Config,
    build_config,
    connect_duckdb,
    ensure_output_dirs,
    make_quantile_band_labels,
    setup_logging,
    utc_now_iso,
    write_df_bundle,
    write_json,
    write_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Etapa 03 v2: EDA baseada na camada modelada.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg: V2Config = build_config(args.base_dir, args.data_dir, args.output_dir)
    ensure_output_dirs(cfg.output_dir)
    conn = connect_duckdb(cfg, read_only=True)
    try:
        outputs: Dict[str, pd.DataFrame] = {}

        outputs["eda_population_monthly_sessions"] = conn.execute(
            """
            SELECT
              session_month AS month,
              population_bucket,
              COUNT(*) AS sessions,
              COUNT(DISTINCT source_unique_id) AS source_ids,
              COUNT(DISTINCT teacher_unique_id) AS teachers
            FROM fct_session_raw
            WHERE session_month IS NOT NULL
            GROUP BY 1, 2
            ORDER BY 1, 2
            """
        ).fetchdf()

        outputs["eda_activity_vs_session_monthly"] = conn.execute(
            """
            SELECT
              month,
              COUNT(DISTINCT teacher_unique_id) AS teacher_month_rows,
              COUNT(DISTINCT teacher_unique_id) FILTER (WHERE active_user_flag=1) AS active_users,
              COUNT(DISTINCT teacher_unique_id) FILTER (WHERE strict_value_flag=1) AS strict_value_users,
              COUNT(DISTINCT teacher_unique_id) FILTER (WHERE session_exposed_no_download_flag=1) AS session_exposed_no_download_users,
              AVG(total_session_minutes_month) AS avg_total_session_minutes,
              AVG(strict_download_count_month) AS avg_strict_downloads
            FROM fct_teacher_month
            GROUP BY 1
            ORDER BY 1
            """
        ).fetchdf()

        outputs["eda_event_family_monthly"] = conn.execute(
            """
            WITH base AS (
              SELECT
                interaction_month AS month,
                event_family,
                COUNT(*) AS rows_total
              FROM fct_interaction_clean
              GROUP BY 1, 2
            ),
            totals AS (
              SELECT month, SUM(rows_total) AS month_total
              FROM base
              GROUP BY 1
            )
            SELECT
              b.month,
              b.event_family,
              b.rows_total,
              b.rows_total * 1.0 / NULLIF(t.month_total, 0) AS share_month
            FROM base b
            INNER JOIN totals t USING(month)
            ORDER BY b.month, b.rows_total DESC
            """
        ).fetchdf()

        outputs["eda_lesson_join_quality_monthly"] = conn.execute(
            """
            SELECT
              interaction_month AS month,
              COUNT(*) AS rows_total,
              AVG(CASE WHEN id_aula_semantic='lesson_like_22char' THEN 1.0 ELSE 0.0 END) AS valid_lesson_id_rate,
              AVG(CASE WHEN lesson_join_allowed=1 THEN 1.0 ELSE 0.0 END) AS lesson_join_allowed_rate,
              AVG(CASE WHEN lesson_mapped_flag=1 THEN 1.0 ELSE 0.0 END) AS lesson_mapped_rate,
              AVG(CASE WHEN is_strict_value_event=1 AND lesson_mapped_flag=1 THEN 1.0 ELSE 0.0 END) AS strict_download_with_lesson_rate
            FROM fct_interaction_clean
            GROUP BY 1
            ORDER BY 1
            """
        ).fetchdf()

        outputs["eda_session_duration_profile"] = conn.execute(
            """
            SELECT
              date_trunc('month', session_month) AS month,
              COUNT(*) AS sessions,
              quantile_cont(duration_sec, 0.5) AS median_duration_sec,
              quantile_cont(duration_sec, 0.75) AS p75_duration_sec,
              quantile_cont(duration_sec, 0.9) AS p90_duration_sec,
              AVG(duration_sec) AS avg_duration_sec
            FROM fct_session_clean
            GROUP BY 1
            ORDER BY 1
            """
        ).fetchdf()

        outputs["eda_teacher_missing_profile"] = conn.execute(
            """
            SELECT
              AVG(is_estado_missing) AS estado_missing_rate,
              AVG(is_utm_missing) AS utm_missing_rate,
              AVG(is_total_alunos_missing) AS total_alunos_missing_rate,
              AVG(is_total_alunos_negative) AS total_alunos_negative_rate,
              AVG(is_login_google_invalid) AS login_google_invalid_rate
            FROM dim_teacher
            """
        ).fetchdf()

        outputs["eda_state_distribution_core"] = conn.execute(
            """
            WITH active_teachers AS (
              SELECT DISTINCT teacher_unique_id
              FROM fct_teacher_month
              WHERE active_user_flag=1
            )
            SELECT
              coalesce(d.estado, 'missing') AS estado,
              COUNT(*) AS teachers
            FROM active_teachers a
            INNER JOIN dim_teacher d ON a.teacher_unique_id=d.teacher_unique_id
            GROUP BY 1
            ORDER BY teachers DESC
            LIMIT 20
            """
        ).fetchdf()

        outputs["eda_subject_distribution_core"] = conn.execute(
            """
            WITH active_teachers AS (
              SELECT DISTINCT teacher_unique_id
              FROM fct_teacher_month
              WHERE active_user_flag=1
            )
            SELECT
              coalesce(d.currentsubject_group, 'missing') AS currentsubject_group,
              COUNT(*) AS teachers
            FROM active_teachers a
            INNER JOIN dim_teacher d ON a.teacher_unique_id=d.teacher_unique_id
            GROUP BY 1
            ORDER BY teachers DESC
            """
        ).fetchdf()

        range_candidates = conn.execute(
            """
            SELECT
              strict_download_count_month,
              session_count_month,
              total_session_minutes_month,
              active_days_month
            FROM fct_teacher_month
            WHERE active_user_flag=1
            """
        ).fetchdf()
        if not range_candidates.empty:
            range_candidates["downloads_band_candidate"] = make_quantile_band_labels(range_candidates["strict_download_count_month"])
            range_candidates["sessions_band_candidate"] = make_quantile_band_labels(range_candidates["session_count_month"])
            range_candidates["minutes_band_candidate"] = make_quantile_band_labels(range_candidates["total_session_minutes_month"])
            outputs["eda_range_candidates_profile"] = (
                range_candidates.groupby(
                    ["downloads_band_candidate", "sessions_band_candidate", "minutes_band_candidate"], dropna=False
                )
                .agg(
                    teacher_month_rows=("strict_download_count_month", "size"),
                    avg_downloads=("strict_download_count_month", "mean"),
                    avg_sessions=("session_count_month", "mean"),
                    avg_minutes=("total_session_minutes_month", "mean"),
                    avg_active_days=("active_days_month", "mean"),
                )
                .reset_index()
                .sort_values("teacher_month_rows", ascending=False)
            )
        else:
            outputs["eda_range_candidates_profile"] = pd.DataFrame()

        for name, df in outputs.items():
            write_df_bundle(cfg.output_dir, name, df)

        summary = {
            "generated_at_utc": utc_now_iso(),
            "core_months": int(outputs["eda_activity_vs_session_monthly"]["month"].nunique()),
            "event_families": sorted(outputs["eda_event_family_monthly"]["event_family"].dropna().unique().tolist()),
            "state_top_5": outputs["eda_state_distribution_core"]["estado"].head(5).tolist(),
            "subject_groups": outputs["eda_subject_distribution_core"]["currentsubject_group"].tolist(),
        }
        write_json(cfg.output_dir / "json" / "eda_summary_v2.json", summary)

        md_lines = [
            "# EDA v2",
            "",
            f"- Gerado em UTC: {summary['generated_at_utc']}",
            f"- Meses core com `fct_teacher_month`: {summary['core_months']}",
            f"- Famílias de evento observadas: {', '.join(summary['event_families'])}",
            f"- Top 5 estados no core ativo: {', '.join(summary['state_top_5'])}",
            "",
            "## Artefatos",
        ]
        for key in outputs:
            md_lines.append(f"- `{key}`")
        write_markdown(cfg.output_dir / "audit" / "eda_summary_v2.md", md_lines)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
