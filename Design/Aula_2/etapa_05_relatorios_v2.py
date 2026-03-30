#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.express as px

from analytics_v2_common import (
    DEFAULT_BASE_DIR,
    PALETTE,
    V2Config,
    build_card_html,
    build_config,
    build_metric_lineage_rows,
    build_table_html,
    ensure_output_dirs,
    figure_to_html,
    fmt_num,
    fmt_pct,
    render_report_html,
    setup_logging,
    write_df_bundle,
    write_json,
    write_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Etapa 05 v2: relatórios HTML com linhagem explícita.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_csv(output_dir: Path, name: str) -> pd.DataFrame:
    path = output_dir / "csv" / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def chart_block(
    report_name: str,
    artifact_name: str,
    title: str,
    subtitle: str,
    body_html: str,
    lineage: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "artifact_name": artifact_name,
        "report_name": report_name,
        "artifact_type": "chart_or_table",
        "title": title,
        "subtitle": subtitle,
        "body_html": body_html,
        "lineage": lineage,
    }


def write_report(output_dir: Path, file_name: str, html: str) -> str:
    report_path = output_dir / "reports" / file_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    return str(report_path)


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg: V2Config = build_config(args.base_dir, args.data_dir, args.output_dir)
    ensure_output_dirs(cfg.output_dir)

    audit_summary = load_csv(cfg.output_dir, "audit_table_inventory")
    join_contracts = load_csv(cfg.output_dir, "audit_join_contracts")
    join_match_by_month = load_csv(cfg.output_dir, "audit_join_match_by_month")
    business_rules = load_csv(cfg.output_dir, "audit_business_rules")
    ping_sensitivity = load_csv(cfg.output_dir, "audit_ping_sensitivity")
    id_aula_semantics = load_csv(cfg.output_dir, "audit_id_aula_semantics")
    anomalies = load_csv(cfg.output_dir, "audit_content_type_anomalies")

    eda_population = load_csv(cfg.output_dir, "eda_population_monthly_sessions")
    eda_activity = load_csv(cfg.output_dir, "eda_activity_vs_session_monthly")
    eda_event_family = load_csv(cfg.output_dir, "eda_event_family_monthly")
    eda_lesson_quality = load_csv(cfg.output_dir, "eda_lesson_join_quality_monthly")
    eda_session_profile = load_csv(cfg.output_dir, "eda_session_duration_profile")
    eda_missing = load_csv(cfg.output_dir, "eda_teacher_missing_profile")
    eda_states = load_csv(cfg.output_dir, "eda_state_distribution_core")
    eda_subjects = load_csv(cfg.output_dir, "eda_subject_distribution_core")
    eda_ranges = load_csv(cfg.output_dir, "eda_range_candidates_profile")

    analytics_monthly = load_csv(cfg.output_dir, "analytics_monthly_core_metrics")
    analytics_download = load_csv(cfg.output_dir, "analytics_download_return_comparison")
    analytics_paths = load_csv(cfg.output_dir, "analytics_no_download_path_outcomes")
    analytics_session = load_csv(cfg.output_dir, "analytics_session_exposure_outcomes")
    analytics_heavy = load_csv(cfg.output_dir, "analytics_heavy_usage_outcomes")
    analytics_gap = load_csv(cfg.output_dir, "analytics_abandonment_gap_curve")
    analytics_clusters = load_csv(cfg.output_dir, "analytics_cluster_profiles")
    analytics_cluster_diag = load_csv(cfg.output_dir, "analytics_cluster_diagnostics")
    analytics_cluster_feature_quality = load_csv(cfg.output_dir, "analytics_cluster_feature_quality")
    analytics_models = load_csv(cfg.output_dir, "analytics_model_performance")
    analytics_model_features = load_csv(cfg.output_dir, "analytics_model_top_features")
    analytics_hypotheses = load_csv(cfg.output_dir, "analytics_hypotheses")
    analytics_feature_admission = load_csv(cfg.output_dir, "analytics_feature_admission")

    lineage_items: List[Dict[str, Any]] = []

    quality_cards = ""
    if not audit_summary.empty:
        dim_row = audit_summary[audit_summary["table_name"] == "raw_dim_teachers"].head(1)
        interactions_row = audit_summary[audit_summary["table_name"] == "raw_interactions"].head(1)
        quality_cards += build_card_html("Professores no cadastro", fmt_num(dim_row["row_count"].iloc[0], 0), "raw_dim_teachers")
        quality_cards += build_card_html("Interações raw", fmt_num(interactions_row["row_count"].iloc[0], 0), "raw_interactions")
    if not ping_sensitivity.empty:
        core_ping = ping_sensitivity[
            (ping_sensitivity["threshold_sec"] == 5)
            & (ping_sensitivity["user_type"] == "registered")
            & (ping_sensitivity["matched_teacher"] == 1)
        ].head(1)
        if not core_ping.empty:
            quality_cards += build_card_html("Ping <=5s no core", fmt_pct(core_ping["ping_rate"].iloc[0], 2), "Sessões registradas e casadas")
    if not join_contracts.empty:
        mari_row = join_contracts[
            join_contracts["contract_name"] == "mari_help.user_id -> mari_reports.id_mari -> teacher"
        ].head(1)
        if not mari_row.empty:
            quality_cards += build_card_html("Ponte mari_help->teacher", fmt_pct(mari_row["coverage_rate"].iloc[0], 2), "Somente resolução unívoca")

    quality_sections: List[Dict[str, Any]] = []
    if not join_contracts.empty:
        fig = px.bar(
            join_contracts,
            x="coverage_rate",
            y="contract_name",
            orientation="h",
            color="contract_semantics",
            color_discrete_sequence=PALETTE,
            title="Cobertura dos contratos de join",
        )
        block = chart_block(
            "relatorio_01_qualidade_e_joins_v2.html",
            "qualidade_join_contracts",
            "Cobertura dos contratos de join",
            "Cada join é tratado como contrato semântico, não só como SQL executável.",
            figure_to_html(fig) + build_table_html(join_contracts, max_rows=20),
            {
                "raw_tables": "dim_teachers, entries, interactions, formation, mari_conv, mari_reports, mari_help, lessons",
                "population": "IDs distintos por fonte",
                "grain": "1 linha por contrato",
                "joins": "joins exatos ou ponte semântica explicitada",
                "filters": "nenhum; cobertura em domínio completo da fonte",
                "logic": "source_ids, matched_ids, ambiguidades e classificação do contrato",
                "caveats": "mari_help direto em dim_teachers é inválido semanticamente",
            },
        )
        lineage_items.append(block)
        quality_sections.append({"title": "Contratos de Join", "description": "Validação semântica dos joins principais.", "blocks": [block]})

    if not join_match_by_month.empty:
        reg = join_match_by_month[join_match_by_month["user_type"] == "registered"].copy()
        reg["month"] = pd.to_datetime(reg["month"], errors="coerce")
        fig = px.line(
            reg,
            x="month",
            y="matched_id_rate",
            color="source_table",
            markers=True,
            color_discrete_sequence=[PALETTE[0], PALETTE[3]],
            title="Taxa de match mensal dos IDs registered",
        )
        block = chart_block(
            "relatorio_01_qualidade_e_joins_v2.html",
            "qualidade_match_mensal_registered",
            "Match mensal da população registered",
            "A queda após 2025-10 é explicitamente medida e não escondida no core.",
            figure_to_html(fig),
            {
                "raw_tables": "entries, interactions, dim_teachers",
                "population": "rows registered por mês",
                "grain": "mês x source_table x user_type",
                "joins": "unique_id -> dim_teachers.unique_id",
                "filters": "user_type='registered'",
                "logic": "matched_id_rate = IDs distintos casados / IDs distintos observados",
                "caveats": "queda de cobertura pode refletir frescor do cadastro e não comportamento do usuário",
            },
        )
        lineage_items.append(block)
        quality_sections.append({"title": "Cobertura Temporal", "description": "A cobertura é mostrada mês a mês para evitar mistura de janelas confiáveis e truncadas.", "blocks": [block]})

    ping_blocks: List[Dict[str, Any]] = []
    if not ping_sensitivity.empty:
        fig = px.line(
            ping_sensitivity,
            x="threshold_sec",
            y="ping_rate",
            color="user_type",
            line_dash="matched_teacher",
            markers=True,
            color_discrete_sequence=PALETTE,
            title="Sensibilidade do corte de ping",
        )
        block = chart_block(
            "relatorio_01_qualidade_e_joins_v2.html",
            "qualidade_ping_sensitivity",
            "Sensibilidade do corte de ping",
            "Mostra por que `<=5s` foi escolhido para a base clean.",
            figure_to_html(fig) + build_table_html(ping_sensitivity, max_rows=18),
            {
                "raw_tables": "entries, dim_teachers",
                "population": "sessões com timestamps válidos",
                "grain": "threshold x user_type x matched_teacher",
                "joins": "entries.unique_id -> dim_teachers.unique_id",
                "filters": "data_inicio e data_fim não nulos",
                "logic": "ping_rate por thresholds 1s/5s/10s",
                "caveats": "ping técnico não deve ser tratado como sessão de uso",
            },
        )
        lineage_items.append(block)
        ping_blocks.append(block)
    if not business_rules.empty:
        block = chart_block(
            "relatorio_01_qualidade_e_joins_v2.html",
            "qualidade_business_rules",
            "Regras de negócio e inconsistências",
            "Checklist de invalidades estruturais nas tabelas raw.",
            build_table_html(business_rules, max_rows=20),
            {
                "raw_tables": "entries, interactions, dim_teachers",
                "population": "linhas completas das tabelas raw",
                "grain": "1 linha por regra",
                "joins": "não aplicável ou join direto por unique_id",
                "filters": "nenhum",
                "logic": "métricas de invalidez e missing por regra",
                "caveats": "algumas regras são monitoramento, não reprovação automática do dataset",
            },
        )
        lineage_items.append(block)
        ping_blocks.append(block)
    if ping_blocks:
        quality_sections.append({"title": "Qualidade de Sessões e Regras", "description": "Sessões ping e regras de negócio são tratados como qualidade estrutural.", "blocks": ping_blocks})

    semantic_blocks: List[Dict[str, Any]] = []
    if not id_aula_semantics.empty:
        semantic_agg = (
            id_aula_semantics.groupby("id_aula_semantic", dropna=False)["rows_total"].sum().reset_index().sort_values("rows_total", ascending=False)
        )
        fig = px.bar(
            semantic_agg,
            x="id_aula_semantic",
            y="rows_total",
            color="id_aula_semantic",
            color_discrete_sequence=PALETTE,
            title="Classes semânticas de id_aula",
        )
        block = chart_block(
            "relatorio_01_qualidade_e_joins_v2.html",
            "qualidade_id_aula_semantic",
            "Classes semânticas de id_aula",
            "Nem todo `id_aula` representa aula válida; placeholders ficam fora do join de lesson.",
            figure_to_html(fig) + build_table_html(id_aula_semantics, max_rows=20),
            {
                "raw_tables": "interactions, lessons",
                "population": "interações raw",
                "grain": "id_aula_semantic x event_type",
                "joins": "id_aula -> lessons.id_aula apenas quando semanticamente válido",
                "filters": "nenhum",
                "logic": "classificação semântica do id e taxa de mapeamento",
                "caveats": "tokens como `0`, `30`, `s` e `abaConquistas` não podem virar aula",
            },
        )
        lineage_items.append(block)
        semantic_blocks.append(block)
    if not anomalies.empty:
        block = chart_block(
            "relatorio_01_qualidade_e_joins_v2.html",
            "qualidade_content_type_anomalies",
            "Anomalias de content_type",
            "Valores sujos e URLs vazadas são explicitamente auditados.",
            build_table_html(anomalies, max_rows=20),
            {
                "raw_tables": "interactions",
                "population": "interações raw",
                "grain": "1 linha por content_type anômalo",
                "joins": "não aplicável",
                "filters": "content_types suspeitos por regex/lista de anomalias",
                "logic": "listar valores atípicos que não devem entrar em leitura causal de produto",
                "caveats": "anomalias de content_type afetam interpretação, não identidade",
            },
        )
        lineage_items.append(block)
        semantic_blocks.append(block)
    if semantic_blocks:
        quality_sections.append({"title": "Semântica de Conteúdo", "description": "A qualidade semântica do conteúdo é validada antes de qualquer join com aula.", "blocks": semantic_blocks})

    quality_html = render_report_html(
        title="Relatório 01 - Qualidade e Joins v2",
        subtitle="Auditoria raw, validação semântica dos joins e qualidade estrutural da trilha v2.",
        summary_cards_html=quality_cards,
        sections=quality_sections,
    )
    quality_path = write_report(cfg.output_dir, "relatorio_01_qualidade_e_joins_v2.html", quality_html)

    eda_cards = ""
    if not eda_activity.empty:
        latest = eda_activity.sort_values("month").tail(1).iloc[0]
        eda_cards += build_card_html("Usuários ativos (último mês)", fmt_num(latest["active_users"], 0), "fct_teacher_month")
        eda_cards += build_card_html("Strict value users", fmt_num(latest["strict_value_users"], 0), "último mês")
        eda_cards += build_card_html("Média downloads strict", fmt_num(latest["avg_strict_downloads"], 2), "último mês")
    if not eda_missing.empty:
        row = eda_missing.iloc[0]
        eda_cards += build_card_html("Missing de estado", fmt_pct(row["estado_missing_rate"], 2), "dim_teacher")
        eda_cards += build_card_html("Missing de UTM", fmt_pct(row["utm_missing_rate"], 2), "dim_teacher")

    eda_sections: List[Dict[str, Any]] = []
    if not eda_population.empty:
        eda_population["month"] = pd.to_datetime(eda_population["month"], errors="coerce")
        fig = px.line(
            eda_population,
            x="month",
            y="sessions",
            color="population_bucket",
            markers=True,
            color_discrete_sequence=PALETTE,
            title="Sessões por população modelada",
        )
        block = chart_block(
            "relatorio_02_eda_v2.html",
            "eda_population_sessions",
            "Sessões por população modelada",
            "Separa core e sombras sem reabrir a raw diretamente.",
            figure_to_html(fig),
            {
                "raw_tables": "entries, dim_teachers",
                "population": "sessões modeladas em fct_session_raw",
                "grain": "mês x population_bucket",
                "joins": "unique_id -> teacher_unique_id quando houver match",
                "filters": "nenhum",
                "logic": "contagem de sessões por bucket populacional",
                "caveats": "sessões não equivalem a usuários ativos",
            },
        )
        lineage_items.append(block)
        eda_sections.append({"title": "População", "description": "Exposição modelada por população e relação com atividade.", "blocks": [block]})

    eda_blocks: List[Dict[str, Any]] = []
    if not eda_activity.empty:
        eda_activity["month"] = pd.to_datetime(eda_activity["month"], errors="coerce")
        long = eda_activity.melt(
            id_vars=["month"],
            value_vars=["active_users", "strict_value_users", "session_exposed_no_download_users"],
            var_name="metric",
            value_name="users",
        )
        fig = px.line(
            long,
            x="month",
            y="users",
            color="metric",
            markers=True,
            color_discrete_sequence=[PALETTE[0], PALETTE[3], PALETTE[5]],
            title="Ativos, strict value e expostos sem download",
        )
        block = chart_block(
            "relatorio_02_eda_v2.html",
            "eda_activity_vs_strict",
            "Ativos, strict value e expostos sem download",
            "Mostra a base comportamental do painel mensal que vai sustentar a análise de retenção.",
            figure_to_html(fig),
            {
                "raw_tables": "entries, interactions, dim_teachers",
                "population": "teacher-month do core",
                "grain": "mês",
                "joins": "fct_teacher_month derivada de fct_session_clean + fct_interaction_clean",
                "filters": "teacher-month modelado",
                "logic": "contagens distintas de ativos, strict value e expostos sem download",
                "caveats": "usuário ativo = interação limpa significativa; entry sozinho não basta",
            },
        )
        lineage_items.append(block)
        eda_blocks.append(block)
    if not eda_event_family.empty:
        eda_event_family["month"] = pd.to_datetime(eda_event_family["month"], errors="coerce")
        fig = px.area(
            eda_event_family,
            x="month",
            y="share_month",
            color="event_family",
            color_discrete_sequence=PALETTE,
            title="Share mensal das famílias de evento no core",
        )
        block = chart_block(
            "relatorio_02_eda_v2.html",
            "eda_event_family_share",
            "Share mensal das famílias de evento",
            "A taxonomia limpa separa aula, plano, prova, IA e outros blocos de uso.",
            figure_to_html(fig),
            {
                "raw_tables": "interactions, lessons, dim_teachers",
                "population": "interações limpas do core",
                "grain": "mês x família de evento",
                "joins": "fct_interaction_clean com classificação semântica do evento",
                "filters": "user_type registered e teacher match exato",
                "logic": "share_month = linhas da família / total de interações limpas do mês",
                "caveats": "share de eventos não é share de usuários",
            },
        )
        lineage_items.append(block)
        eda_blocks.append(block)
    if eda_blocks:
        eda_sections.append({"title": "Uso e Taxonomia", "description": "EDA de uso mensal e composição semântica dos eventos.", "blocks": eda_blocks})

    lesson_blocks: List[Dict[str, Any]] = []
    if not eda_lesson_quality.empty:
        eda_lesson_quality["month"] = pd.to_datetime(eda_lesson_quality["month"], errors="coerce")
        long = eda_lesson_quality.melt(
            id_vars=["month"],
            value_vars=["valid_lesson_id_rate", "lesson_mapped_rate", "strict_download_with_lesson_rate"],
            var_name="metric",
            value_name="value",
        )
        fig = px.line(
            long,
            x="month",
            y="value",
            color="metric",
            markers=True,
            color_discrete_sequence=[PALETTE[0], PALETTE[3], PALETTE[5]],
            title="Qualidade do join de aula ao longo do tempo",
        )
        block = chart_block(
            "relatorio_02_eda_v2.html",
            "eda_lesson_join_quality",
            "Qualidade do join de aula",
            "Separa validade semântica do ID, permissão de join e mapeamento efetivo.",
            figure_to_html(fig),
            {
                "raw_tables": "interactions, lessons",
                "population": "interações limpas do core",
                "grain": "mês",
                "joins": "id_aula -> lessons.id_aula apenas quando semanticamente válido",
                "filters": "fct_interaction_clean",
                "logic": "taxas de id válido, join permitido e join mapeado",
                "caveats": "strict download com aula mapeada não cobre 100% dos downloads strict",
            },
        )
        lineage_items.append(block)
        lesson_blocks.append(block)
    if not eda_session_profile.empty:
        eda_session_profile["month"] = pd.to_datetime(eda_session_profile["month"], errors="coerce")
        fig = px.line(
            eda_session_profile,
            x="month",
            y="median_duration_sec",
            markers=True,
            color_discrete_sequence=[PALETTE[2]],
            title="Mediana de duração das sessões limpas",
        )
        block = chart_block(
            "relatorio_02_eda_v2.html",
            "eda_session_duration_profile",
            "Mediana de duração das sessões limpas",
            "Mostra a intensidade temporal depois de remover pings.",
            figure_to_html(fig),
            {
                "raw_tables": "entries, dim_teachers",
                "population": "sessões limpas do core",
                "grain": "mês",
                "joins": "fct_session_clean",
                "filters": "registered + match exato + duração > 5s",
                "logic": "medianas e quantis de duração",
                "caveats": "sessão longa não implica necessariamente conteúdo de valor",
            },
        )
        lineage_items.append(block)
        lesson_blocks.append(block)
    if lesson_blocks:
        eda_sections.append({"title": "Qualidade Analítica", "description": "Qualidade de aula mapeada e duração das sessões já limpas.", "blocks": lesson_blocks})

    tables_blocks: List[Dict[str, Any]] = []
    if not eda_missing.empty:
        block = chart_block(
            "relatorio_02_eda_v2.html",
            "eda_teacher_missing_profile",
            "Missing de metadados do cadastro",
            "Perfil de missing que afeta segmentação por estado, UTM e base de alunos.",
            build_table_html(eda_missing, max_rows=10),
            {
                "raw_tables": "dim_teachers",
                "population": "professores do cadastro",
                "grain": "1 linha agregada",
                "joins": "não aplicável",
                "filters": "nenhum",
                "logic": "taxas agregadas de missing e invalidez no cadastro modelado",
                "caveats": "estado e UTM têm missing alto e entram com caveat nos modelos",
            },
        )
        lineage_items.append(block)
        tables_blocks.append(block)
    if not eda_states.empty:
        block = chart_block(
            "relatorio_02_eda_v2.html",
            "eda_state_distribution_core",
            "Estados mais frequentes no core ativo",
            "Distribuição geográfica da base ativa confiável.",
            build_table_html(eda_states, max_rows=20),
            {
                "raw_tables": "dim_teachers, interactions",
                "population": "teachers ativos no core",
                "grain": "estado",
                "joins": "fct_teacher_month -> dim_teacher",
                "filters": "active_user_flag=1",
                "logic": "contagem de professores ativos por estado",
                "caveats": "estado missing permanece fora do top quando aplicável",
            },
        )
        lineage_items.append(block)
        tables_blocks.append(block)
    if not eda_subjects.empty:
        block = chart_block(
            "relatorio_02_eda_v2.html",
            "eda_subject_distribution_core",
            "Grupos de disciplina no core ativo",
            "Mostra a distribuição dos grupos curriculares do cadastro.",
            build_table_html(eda_subjects, max_rows=20),
            {
                "raw_tables": "dim_teachers, interactions",
                "population": "teachers ativos no core",
                "grain": "grupo de disciplina",
                "joins": "fct_teacher_month -> dim_teacher",
                "filters": "active_user_flag=1",
                "logic": "contagem de professores ativos por grupo curricular",
                "caveats": "currentsubject_group depende da qualidade do cadastro",
            },
        )
        lineage_items.append(block)
        tables_blocks.append(block)
    if not eda_ranges.empty:
        block = chart_block(
            "relatorio_02_eda_v2.html",
            "eda_range_candidates",
            "Candidatos de faixas empíricas",
            "Faixas são tratadas como hipótese de apresentação, não como regra arbitrária fixa.",
            build_table_html(eda_ranges, max_rows=20),
            {
                "raw_tables": "entries, interactions, dim_teachers",
                "population": "teacher-month ativo",
                "grain": "combinação de bandas candidatas",
                "joins": "fct_teacher_month",
                "filters": "active_user_flag=1",
                "logic": "bandas por quantis empíricos para downloads, sessões e minutos",
                "caveats": "bandas só serão usadas na apresentação se forem estáveis e interpretáveis",
            },
        )
        lineage_items.append(block)
        tables_blocks.append(block)
    if tables_blocks:
        eda_sections.append({"title": "Perfis e Faixas", "description": "Metadados do core e candidatos de faixas não arbitrárias.", "blocks": tables_blocks})

    eda_html = render_report_html(
        title="Relatório 02 - EDA v2",
        subtitle="EDA da camada modelada: uso, taxonomia, qualidade analítica e perfis do core.",
        summary_cards_html=eda_cards,
        sections=eda_sections,
    )
    eda_path = write_report(cfg.output_dir, "relatorio_02_eda_v2.html", eda_html)

    analytics_cards = ""
    if not analytics_monthly.empty:
        latest = analytics_monthly.sort_values("month").tail(1).iloc[0]
        analytics_cards += build_card_html("Strict value rate", fmt_pct(latest["strict_value_rate"], 2), "último mês")
        analytics_cards += build_card_html("Strict user rate", fmt_pct(latest["strict_user_rate"], 2), "último mês observado")
        analytics_cards += build_card_html("Strict return value rate", fmt_pct(latest["strict_return_value_rate"], 2), "último mês observado")
    if not analytics_gap.empty:
        low_hazard = analytics_gap[analytics_gap["hazard"] < 0.05].head(1)
        if not low_hazard.empty:
            analytics_cards += build_card_html("Gap sugerido abandono", fmt_num(low_hazard["horizon_month"].iloc[0], 0), "primeiro hazard < 5%")
    if not analytics_models.empty:
        best_auc = analytics_models["roc_auc"].max()
        analytics_cards += build_card_html("Melhor ROC AUC", fmt_num(best_auc, 3), "modelos out-of-time")

    analytics_sections: List[Dict[str, Any]] = []
    metric_blocks: List[Dict[str, Any]] = []
    if not analytics_monthly.empty:
        analytics_monthly["month"] = pd.to_datetime(analytics_monthly["month"], errors="coerce")
        long = analytics_monthly.melt(
            id_vars=["month"],
            value_vars=["strict_value_rate", "strict_user_rate", "strict_return_value_rate"],
            var_name="metric",
            value_name="value",
        )
        fig = px.line(
            long,
            x="month",
            y="value",
            color="metric",
            markers=True,
            color_discrete_sequence=[PALETTE[0], PALETTE[3], PALETTE[5]],
            title="Métricas mensais do core",
        )
        block = chart_block(
            "relatorio_03_usuarios_metricas_v2.html",
            "analytics_monthly_metrics",
            "Métricas mensais do core",
            "Strict value, strict user e retorno com novo download são mostrados separadamente.",
            figure_to_html(fig) + build_table_html(analytics_monthly, max_rows=18),
            {
                "raw_tables": "entries, interactions, dim_teachers",
                "population": "teacher-month modelado",
                "grain": "mês",
                "joins": "fct_teacher_month",
                "filters": "core teacher-month",
                "logic": "rates derivadas de strict_value_flag, strict_user_flag e strict_return_value_m1",
                "caveats": "último mês pode sofrer censura para retornos m+1",
            },
        )
        lineage_items.append(block)
        metric_blocks.append(block)
    if not analytics_gap.empty:
        fig = px.line(
            analytics_gap,
            x="horizon_month",
            y="hazard",
            markers=True,
            color_discrete_sequence=[PALETTE[6]],
            title="Hazard de retorno por gap de inatividade",
        )
        block = chart_block(
            "relatorio_03_usuarios_metricas_v2.html",
            "analytics_abandonment_gap",
            "Hazard de retorno por gap",
            "Usado para investigar um threshold de abandono não arbitrário.",
            figure_to_html(fig) + build_table_html(analytics_gap, max_rows=20),
            {
                "raw_tables": "interactions, dim_teachers",
                "population": "teacher-month ativo",
                "grain": "horizon_month",
                "joins": "fct_teacher_month ordenado por professor",
                "filters": "active_user_flag=1 e next_month_observed_flag=1",
                "logic": "hazard da primeira volta após gap de meses",
                "caveats": "threshold sugerido é diagnóstico, não regra absoluta",
            },
        )
        lineage_items.append(block)
        metric_blocks.append(block)
    if metric_blocks:
        analytics_sections.append({"title": "Retenção e Abandono", "description": "Métricas centrais de retorno, valor e gap de abandono.", "blocks": metric_blocks})

    journey_blocks: List[Dict[str, Any]] = []
    if not analytics_download.empty:
        fig = px.bar(
            analytics_download,
            x="segment",
            y="return_active_rate",
            color="segment",
            color_discrete_sequence=[PALETTE[0], PALETTE[3]],
            title="Retorno ativo após strict value vs sem strict value",
        )
        block = chart_block(
            "relatorio_03_usuarios_metricas_v2.html",
            "analytics_download_return",
            "Retorno após strict value",
            "Compara quem fez strict value com quem não fez no mês t.",
            figure_to_html(fig) + build_table_html(analytics_download, max_rows=10),
            {
                "raw_tables": "entries, interactions, dim_teachers",
                "population": "active teacher-month com mês seguinte observado",
                "grain": "segmento",
                "joins": "fct_teacher_month",
                "filters": "active_user_flag=1 e next_month_observed_flag=1",
                "logic": "retorno ativo e retorno com download em m+1 por strict_value_flag",
                "caveats": "associacional; diferenças podem refletir baseline de uso",
            },
        )
        lineage_items.append(block)
        journey_blocks.append(block)
    if not analytics_paths.empty:
        fig = px.bar(
            analytics_paths,
            x="path_category",
            y="return_active_rate",
            color="path_category",
            color_discrete_sequence=PALETTE,
            title="Retorno por jornada sem download",
        )
        block = chart_block(
            "relatorio_03_usuarios_metricas_v2.html",
            "analytics_no_download_paths",
            "Retorno por jornada sem download",
            "Distingue visualização sem download, visualização com outras ações e ação sem visualização.",
            figure_to_html(fig) + build_table_html(analytics_paths, max_rows=10),
            {
                "raw_tables": "entries, interactions, dim_teachers",
                "population": "active teacher-month sem download e com mês seguinte observado",
                "grain": "categoria de jornada",
                "joins": "fct_teacher_month",
                "filters": "no_download_flag=1, active_user_flag=1, next_month_observed_flag=1",
                "logic": "taxas de retorno por categoria de jornada sem download",
                "caveats": "não mede causalidade de cada microevento",
            },
        )
        lineage_items.append(block)
        journey_blocks.append(block)
    if not analytics_session.empty:
        block = chart_block(
            "relatorio_03_usuarios_metricas_v2.html",
            "analytics_session_exposure_outcomes",
            "Resultados após exposição por sessão",
            "Compara acesso com e sem atividade, além de acesso com strict value.",
            build_table_html(analytics_session, max_rows=10),
            {
                "raw_tables": "entries, interactions, dim_teachers",
                "population": "teacher-month com sessões e mês seguinte observado",
                "grain": "categoria de exposição",
                "joins": "fct_teacher_month",
                "filters": "session_count_month>0 e next_month_observed_flag=1",
                "logic": "resultado em m+1 por tipo de exposição no mês t",
                "caveats": "sessão sem atividade não entra em active_user, mas entra na análise de exposição",
            },
        )
        lineage_items.append(block)
        journey_blocks.append(block)
    if journey_blocks:
        analytics_sections.append({"title": "Jornadas e Exposição", "description": "Retorno por strict value, jornadas sem download e exposição por sessão.", "blocks": journey_blocks})

    segment_blocks: List[Dict[str, Any]] = []
    if not analytics_heavy.empty:
        fig = px.bar(
            analytics_heavy,
            x="segment",
            y="return_active_rate",
            color="segment",
            color_discrete_sequence=[PALETTE[6], PALETTE[2]],
            title="Retorno heavy vs base ativa",
        )
        block = chart_block(
            "relatorio_03_usuarios_metricas_v2.html",
            "analytics_heavy_outcomes",
            "Retorno heavy vs base ativa",
            "Heavy month é definido empiricamente por intensidade, não herdado do pipeline legacy.",
            figure_to_html(fig) + build_table_html(analytics_heavy, max_rows=10),
            {
                "raw_tables": "entries, interactions, dim_teachers",
                "population": "active teacher-month com mês seguinte observado",
                "grain": "segmento heavy/base",
                "joins": "fct_teacher_month + heavy_intensity_score",
                "filters": "active_user_flag=1 e next_month_observed_flag=1",
                "logic": "comparação de retorno e intensidade entre heavy_month e base ativa",
                "caveats": "heavy_month é definição operacional v2 para análise, não verdade ontológica",
            },
        )
        lineage_items.append(block)
        segment_blocks.append(block)
    if not analytics_clusters.empty:
        block = chart_block(
            "relatorio_03_usuarios_metricas_v2.html",
            "analytics_cluster_profiles",
            "Perfis dos clusters comportamentais",
            "Clusters em teacher-month, sem leakage e com features contínuas admitidas.",
            build_table_html(analytics_clusters, max_rows=20)
            + build_table_html(analytics_cluster_diag, max_rows=20)
            + build_table_html(analytics_cluster_feature_quality, max_rows=20),
            {
                "raw_tables": "entries, interactions, dim_teachers",
                "population": "teacher-month ativo",
                "grain": "cluster",
                "joins": "fct_teacher_month com KMeans em features contínuas",
                "filters": "active_user_flag=1",
                "logic": "seleção de features, escolha de k por silhouette+estabilidade e perfil médio por cluster",
                "caveats": "clusters são descritivos; interpretação depende de estabilidade e cobertura",
            },
        )
        lineage_items.append(block)
        segment_blocks.append(block)
    if segment_blocks:
        analytics_sections.append({"title": "Segmentação", "description": "Heavy usage e clusters comportamentais reconstruídos do zero.", "blocks": segment_blocks})

    model_blocks: List[Dict[str, Any]] = []
    if not analytics_models.empty:
        fig = px.bar(
            analytics_models,
            x="target",
            y="roc_auc",
            color="model_name",
            barmode="group",
            color_discrete_sequence=[PALETTE[0], PALETTE[3]],
            title="Performance out-of-time dos modelos",
        )
        block = chart_block(
            "relatorio_03_usuarios_metricas_v2.html",
            "analytics_model_performance",
            "Performance out-of-time dos modelos",
            "Modelo primário interpretável e benchmark secundário por alvo.",
            figure_to_html(fig) + build_table_html(analytics_models, max_rows=20),
            {
                "raw_tables": "entries, interactions, dim_teachers",
                "population": "teacher-month ativo com mês seguinte observado",
                "grain": "target x modelo",
                "joins": "fct_teacher_month -> dim_teacher",
                "filters": "split temporal train/test",
                "logic": "ROC AUC, average precision e lift de decil superior",
                "caveats": "performance depende da estabilidade do período e do refresh do cadastro",
            },
        )
        lineage_items.append(block)
        model_blocks.append(block)
    if not analytics_model_features.empty:
        block = chart_block(
            "relatorio_03_usuarios_metricas_v2.html",
            "analytics_model_top_features",
            "Principais sinais do modelo interpretável",
            "Coeficientes do logit ajudam a entender sinais de parar de usar e de heavy user.",
            build_table_html(analytics_model_features, max_rows=20) + build_table_html(analytics_feature_admission, max_rows=20),
            {
                "raw_tables": "entries, interactions, dim_teachers",
                "population": "teacher-month ativo com mês seguinte observado",
                "grain": "feature",
                "joins": "fct_teacher_month -> dim_teacher",
                "filters": "targets de churn e heavy_next_m1",
                "logic": "coeficientes do modelo primário e matriz de admissão de features",
                "caveats": "coeficiente não equivale a causalidade; sinais podem refletir perfil subjacente",
            },
        )
        lineage_items.append(block)
        model_blocks.append(block)
    if not analytics_hypotheses.empty:
        block = chart_block(
            "relatorio_03_usuarios_metricas_v2.html",
            "analytics_hypotheses",
            "Hipóteses falseáveis testadas",
            "Todas reportadas como associacionais e com estratificação explícita.",
            build_table_html(analytics_hypotheses, max_rows=20),
            {
                "raw_tables": "entries, interactions, dim_teachers",
                "population": "subconjuntos elegíveis com mês seguinte observado",
                "grain": "hipótese",
                "joins": "fct_teacher_month -> dim_teacher",
                "filters": "estratos válidos com expostos e não expostos",
                "logic": "efeito estratificado por mês, tenure e baseline de uso",
                "caveats": "sem desenho causal, usar como hipótese disciplinada e não como prova de mecanismo",
            },
        )
        lineage_items.append(block)
        model_blocks.append(block)
    if model_blocks:
        analytics_sections.append({"title": "Modelos e Hipóteses", "description": "Predição out-of-time e hipóteses explicitamente associacionais.", "blocks": model_blocks})

    analytics_html = render_report_html(
        title="Relatório 03 - Usuários e Métricas v2",
        subtitle="Retenção, jornadas, abandono, segmentação, predição e hipóteses falseáveis sobre o core confiável.",
        summary_cards_html=analytics_cards,
        sections=analytics_sections,
    )
    analytics_path = write_report(cfg.output_dir, "relatorio_03_usuarios_metricas_v2.html", analytics_html)

    lineage_df = build_metric_lineage_rows(lineage_items)
    write_df_bundle(cfg.output_dir, "audit_metric_lineage", lineage_df)

    summary = {
        "reports": {
            "relatorio_01_qualidade_e_joins_v2": quality_path,
            "relatorio_02_eda_v2": eda_path,
            "relatorio_03_usuarios_metricas_v2": analytics_path,
        },
        "lineage_rows": int(len(lineage_df)),
    }
    write_json(cfg.output_dir / "json" / "reports_v2_summary.json", summary)
    write_markdown(
        cfg.output_dir / "reports" / "reports_v2_summary.md",
        [
            "# Relatórios v2",
            "",
            f"- `relatorio_01_qualidade_e_joins_v2.html`: `{quality_path}`",
            f"- `relatorio_02_eda_v2.html`: `{eda_path}`",
            f"- `relatorio_03_usuarios_metricas_v2.html`: `{analytics_path}`",
            f"- Linhas de lineage gravadas: {summary['lineage_rows']}",
        ],
    )


if __name__ == "__main__":
    main()
