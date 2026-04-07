from __future__ import annotations

import duckdb
import pandas as pd

from targeted_ml.pipelines.modelled_to_ml.dataset_builder import build_future_metrics


def _create_table(conn: duckdb.DuckDBPyConnection, table_name: str, frame: pd.DataFrame) -> None:
    conn.register("_frame", frame)
    conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _frame")
    conn.unregister("_frame")


def test_future_business_active_weeks_uses_literal_activity_definition() -> None:
    conn = duckdb.connect(":memory:")

    teachers = [
        "t_aula",
        "t_formacao",
        "t_mari",
        "t_ruido",
        "t_mesma_semana",
        "t_duas_semanas",
        "t_so_sessao",
        "t_ia_view",
        "t_conquista_share",
        "t_comunidade",
        "t_rascunho_plano",
    ]
    anchor = pd.Timestamp("2024-01-01 00:00:00")
    anchors = pd.DataFrame(
        {
            "teacher_unique_id": teachers,
            "first_month": [pd.Timestamp("2024-01-01")] * len(teachers),
            "months_after_entry": [0] * len(teachers),
            "onboarding_anchor_ts": [anchor] * len(teachers),
        }
    )
    _create_table(conn, "mart_first_session_journey_v1", anchors)

    session_rows = pd.DataFrame(
        {
            "teacher_unique_id": ["t_so_sessao"],
            "session_start_ts": [pd.Timestamp("2024-01-10 08:00:00")],
            "duration_min": [5.0],
        }
    )
    _create_table(conn, "fct_session_clean", session_rows)

    interaction_rows = pd.DataFrame(
        [
            {
                "teacher_unique_id": "t_aula",
                "interaction_ts": pd.Timestamp("2024-01-09 09:00:00"),
                "event_type": "visualizacao_aula",
                "event_type_lower": "visualizacao_aula",
                "event_action": "view",
                "event_family": "aula",
                "is_activity_event": 1,
                "is_download_event": 0,
                "is_content_view_event": 1,
                "lesson_mapped_flag": 0,
            },
            {
                "teacher_unique_id": "t_ruido",
                "interaction_ts": pd.Timestamp("2024-01-10 10:00:00"),
                "event_type": "acesso_aba_conquistas",
                "event_type_lower": "acesso_aba_conquistas",
                "event_action": "navigation",
                "event_family": "conquista",
                "is_activity_event": 0,
                "is_download_event": 0,
                "is_content_view_event": 0,
                "lesson_mapped_flag": 0,
            },
            {
                "teacher_unique_id": "t_mesma_semana",
                "interaction_ts": pd.Timestamp("2024-01-10 11:00:00"),
                "event_type": "visualizacao_aula",
                "event_type_lower": "visualizacao_aula",
                "event_action": "view",
                "event_family": "aula",
                "is_activity_event": 1,
                "is_download_event": 0,
                "is_content_view_event": 1,
                "lesson_mapped_flag": 0,
            },
            {
                "teacher_unique_id": "t_mesma_semana",
                "interaction_ts": pd.Timestamp("2024-01-12 11:00:00"),
                "event_type": "download_aula",
                "event_type_lower": "download_aula",
                "event_action": "download",
                "event_family": "aula",
                "is_activity_event": 1,
                "is_download_event": 1,
                "is_content_view_event": 0,
                "lesson_mapped_flag": 0,
            },
            {
                "teacher_unique_id": "t_duas_semanas",
                "interaction_ts": pd.Timestamp("2024-01-09 12:00:00"),
                "event_type": "visualizacao_aula",
                "event_type_lower": "visualizacao_aula",
                "event_action": "view",
                "event_family": "aula",
                "is_activity_event": 1,
                "is_download_event": 0,
                "is_content_view_event": 1,
                "lesson_mapped_flag": 0,
            },
            {
                "teacher_unique_id": "t_duas_semanas",
                "interaction_ts": pd.Timestamp("2024-01-20 12:00:00"),
                "event_type": "download_aula",
                "event_type_lower": "download_aula",
                "event_action": "download",
                "event_family": "aula",
                "is_activity_event": 1,
                "is_download_event": 1,
                "is_content_view_event": 0,
                "lesson_mapped_flag": 0,
            },
            {
                "teacher_unique_id": "t_ia_view",
                "interaction_ts": pd.Timestamp("2024-01-11 13:00:00"),
                "event_type": "visualizacao_conteudo_ia",
                "event_type_lower": "visualizacao_conteudo_ia",
                "event_action": "view",
                "event_family": "ia",
                "is_activity_event": 1,
                "is_download_event": 0,
                "is_content_view_event": 0,
                "lesson_mapped_flag": 0,
            },
            {
                "teacher_unique_id": "t_conquista_share",
                "interaction_ts": pd.Timestamp("2024-01-11 14:00:00"),
                "event_type": "botao_compartilhar_conquista_modal",
                "event_type_lower": "botao_compartilhar_conquista_modal",
                "event_action": "share",
                "event_family": "conquista",
                "is_activity_event": 1,
                "is_download_event": 0,
                "is_content_view_event": 0,
                "lesson_mapped_flag": 0,
            },
            {
                "teacher_unique_id": "t_comunidade",
                "interaction_ts": pd.Timestamp("2024-01-11 15:00:00"),
                "event_type": "acesso_comunidade",
                "event_type_lower": "acesso_comunidade",
                "event_action": "navigation",
                "event_family": "other",
                "is_activity_event": 1,
                "is_download_event": 0,
                "is_content_view_event": 0,
                "lesson_mapped_flag": 0,
            },
            {
                "teacher_unique_id": "t_rascunho_plano",
                "interaction_ts": pd.Timestamp("2024-01-11 16:00:00"),
                "event_type": "rascunho_plano_aula",
                "event_type_lower": "rascunho_plano_aula",
                "event_action": "other",
                "event_family": "plano",
                "is_activity_event": 1,
                "is_download_event": 0,
                "is_content_view_event": 0,
                "lesson_mapped_flag": 0,
            },
        ]
    )
    _create_table(conn, "fct_interaction_clean", interaction_rows)

    formation_rows = pd.DataFrame(
        {
            "teacher_unique_id": ["t_formacao"],
            "formation_ts": [pd.Timestamp("2024-01-20 10:00:00")],
        }
    )
    _create_table(conn, "fct_formation_clean", formation_rows)

    mari_rows = pd.DataFrame(
        {
            "teacher_unique_id": ["t_mari"],
            "mari_created_ts": [pd.Timestamp("2024-01-22 09:00:00")],
            "has_user_message": [1],
        }
    )
    _create_table(conn, "fct_mari_conversation_resolved", mari_rows)

    mari_help_rows = pd.DataFrame({"teacher_unique_id": pd.Series(dtype="object"), "help_ts": pd.Series(dtype="datetime64[ns]")})
    _create_table(conn, "fct_mari_help_resolved", mari_help_rows)

    metrics = build_future_metrics(conn).set_index("teacher_unique_id")

    assert int(metrics.loc["t_aula", "future_business_active_weeks"]) == 1
    assert int(metrics.loc["t_formacao", "future_business_active_weeks"]) == 1
    assert int(metrics.loc["t_mari", "future_business_active_weeks"]) == 1
    assert int(metrics.loc["t_ruido", "future_business_active_weeks"]) == 0
    assert int(metrics.loc["t_mesma_semana", "future_business_active_weeks"]) == 1
    assert int(metrics.loc["t_duas_semanas", "future_business_active_weeks"]) == 2
    assert int(metrics.loc["t_so_sessao", "future_business_active_weeks"]) == 0
    assert int(metrics.loc["t_ia_view", "future_business_active_weeks"]) == 0
    assert int(metrics.loc["t_conquista_share", "future_business_active_weeks"]) == 1
    assert int(metrics.loc["t_comunidade", "future_business_active_weeks"]) == 1
    assert int(metrics.loc["t_rascunho_plano", "future_business_active_weeks"]) == 0


def test_post_label_validators_use_fixed_meaningful_continuation_events() -> None:
    conn = duckdb.connect(":memory:")

    teachers = [
        "t_view_block1",
        "t_download_block2",
        "t_create_and_share",
        "t_navigation_only",
        "t_session_only",
        "t_view_ia_only",
    ]
    anchor = pd.Timestamp("2024-01-01 00:00:00")
    anchors = pd.DataFrame(
        {
            "teacher_unique_id": teachers,
            "first_month": [pd.Timestamp("2024-01-01")] * len(teachers),
            "months_after_entry": [0] * len(teachers),
            "onboarding_anchor_ts": [anchor] * len(teachers),
        }
    )
    _create_table(conn, "mart_first_session_journey_v1", anchors)

    session_rows = pd.DataFrame(
        {
            "teacher_unique_id": ["t_session_only"],
            "session_start_ts": [pd.Timestamp("2024-02-15 08:00:00")],
            "duration_min": [12.0],
        }
    )
    _create_table(conn, "fct_session_clean", session_rows)

    interaction_rows = pd.DataFrame(
        [
            {
                "teacher_unique_id": "t_view_block1",
                "interaction_ts": pd.Timestamp("2024-02-10 09:00:00"),
                "event_type": "visualizacao_aula",
                "event_type_lower": "visualizacao_aula",
                "event_action": "view",
                "event_family": "aula",
                "is_activity_event": 1,
                "is_download_event": 0,
                "is_content_view_event": 1,
                "lesson_mapped_flag": 0,
            },
            {
                "teacher_unique_id": "t_download_block2",
                "interaction_ts": pd.Timestamp("2024-03-12 10:00:00"),
                "event_type": "download_aula",
                "event_type_lower": "download_aula",
                "event_action": "download",
                "event_family": "aula",
                "is_activity_event": 1,
                "is_download_event": 1,
                "is_content_view_event": 0,
                "lesson_mapped_flag": 0,
            },
            {
                "teacher_unique_id": "t_create_and_share",
                "interaction_ts": pd.Timestamp("2024-02-12 11:00:00"),
                "event_type": "criacao_plano_aula",
                "event_type_lower": "criacao_plano_aula",
                "event_action": "create",
                "event_family": "plano",
                "is_activity_event": 1,
                "is_download_event": 0,
                "is_content_view_event": 0,
                "lesson_mapped_flag": 0,
            },
            {
                "teacher_unique_id": "t_create_and_share",
                "interaction_ts": pd.Timestamp("2024-04-10 11:00:00"),
                "event_type": "envio_email_ou_baixou_prova",
                "event_type_lower": "envio_email_ou_baixou_prova",
                "event_action": "share",
                "event_family": "prova",
                "is_activity_event": 1,
                "is_download_event": 0,
                "is_content_view_event": 0,
                "lesson_mapped_flag": 0,
            },
            {
                "teacher_unique_id": "t_navigation_only",
                "interaction_ts": pd.Timestamp("2024-02-13 12:00:00"),
                "event_type": "acesso_aba_conquistas",
                "event_type_lower": "acesso_aba_conquistas",
                "event_action": "navigation",
                "event_family": "conquista",
                "is_activity_event": 0,
                "is_download_event": 0,
                "is_content_view_event": 0,
                "lesson_mapped_flag": 0,
            },
            {
                "teacher_unique_id": "t_view_ia_only",
                "interaction_ts": pd.Timestamp("2024-02-14 13:00:00"),
                "event_type": "visualizacao_conteudo_ia",
                "event_type_lower": "visualizacao_conteudo_ia",
                "event_action": "view",
                "event_family": "ia",
                "is_activity_event": 1,
                "is_download_event": 0,
                "is_content_view_event": 0,
                "lesson_mapped_flag": 0,
            },
        ]
    )
    _create_table(conn, "fct_interaction_clean", interaction_rows)

    formation_rows = pd.DataFrame({"teacher_unique_id": pd.Series(dtype="object"), "formation_ts": pd.Series(dtype="datetime64[ns]")})
    _create_table(conn, "fct_formation_clean", formation_rows)

    mari_rows = pd.DataFrame(
        {
            "teacher_unique_id": pd.Series(dtype="object"),
            "mari_created_ts": pd.Series(dtype="datetime64[ns]"),
            "has_user_message": pd.Series(dtype="int64"),
        }
    )
    _create_table(conn, "fct_mari_conversation_resolved", mari_rows)

    mari_help_rows = pd.DataFrame({"teacher_unique_id": pd.Series(dtype="object"), "help_ts": pd.Series(dtype="datetime64[ns]")})
    _create_table(conn, "fct_mari_help_resolved", mari_help_rows)

    metrics = build_future_metrics(conn).set_index("teacher_unique_id")

    assert int(metrics.loc["t_view_block1", "returned_active_post_label_m1"]) == 1
    assert int(metrics.loc["t_view_block1", "returned_active_post_label_m2"]) == 0
    assert int(metrics.loc["t_download_block2", "returned_active_post_label_m1"]) == 0
    assert int(metrics.loc["t_download_block2", "returned_active_post_label_m2"]) == 1
    assert int(metrics.loc["t_create_and_share", "returned_active_post_label_m1"]) == 1
    assert int(metrics.loc["t_create_and_share", "returned_active_post_label_m3"]) == 1
    assert int(metrics.loc["t_create_and_share", "sustained_active_2of3_post_label"]) == 1
    assert int(metrics.loc["t_create_and_share", "active_days_post_label_3m"]) == 2
    assert int(metrics.loc["t_navigation_only", "returned_active_post_label_m1"]) == 0
    assert int(metrics.loc["t_session_only", "returned_active_post_label_m1"]) == 0
    assert int(metrics.loc["t_view_ia_only", "returned_active_post_label_m1"]) == 0
