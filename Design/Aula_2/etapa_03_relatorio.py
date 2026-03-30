from __future__ import annotations

"""
Etapa 03 - Relatório html consolidado 

Gera:
- reports/analise_inicial_dos_dados_interativa.html
- reports/analise_inicial_dos_dados_summary.json
- reports/analise_inicial_dos_dados.md
- excel/analise_inicial_dos_dados_bundle.xlsx
- parquet/*.parquet e csv/*.csv (bundle técnico)
- executive_quadro_por_item.csv
"""

import argparse
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs


LOGGER = logging.getLogger("etapa_03_relatorio")
DEFAULT_BASE_DIR = Path("/Users/akatsurada/Documents/INSPER/Design/Aula_2")

PALETTE = ["#0B132B", "#1C2541", "#3A506B", "#5BC0BE", "#CDEFF0"]

CHECK_CATEGORY_MAP = {
    "entries": "Sessões (uso)",
    "interactions": "Interações de conteúdo",
    "dim": "Cadastro de professores",
    "interaction_id_aula": "Mapeamento de aulas",
}

CHECK_LABEL_MAP = {
    "entries_negative_duration": "Sessões com duração negativa",
    "entries_zero_or_short_seconds": "Sessões <=5s (ping técnico)",
    "interactions_before_teacher_entry": "Interações antes do cadastro",
    "interactions_missing_timestamp_rate": "Interações sem timestamp",
    "interactions_missing_event_type_rate": "Interações sem tipo de evento",
    "dim_negative_total_alunos_rate": "Cadastro com total_alunos negativo",
    "interaction_id_aula_unmapped_rate": "id_aula sem mapeamento em aulas",
}

CHART_LINEAGE: Dict[str, Dict[str, str]] = {
    "journey_solution_volume": {
        "como_foi_gerado": "Soma mensal de eventos de Aula e Prova e visualização em série temporal.",
        "tabelas_usadas": "fct_teachers_contents_interactions.csv -> eda_monthly_solution_usage.csv",
        "colunas_chave": "data_inicio, event_type, unique_id, month, aula_events, prova_events",
        "transformacoes_joins": "Truncamento por mês; classificação de event_type com LIKE '%aula%'/'%prova%'; agregação mensal; sem join nesta etapa.",
    },
    "users_activity_curve": {
        "como_foi_gerado": "Curva de novos usuários e usuários ativos por mês.",
        "tabelas_usadas": "dim_teachers.csv + fct_teachers_contents_interactions.csv -> users_monthly_panel.csv",
        "colunas_chave": "data_entrada, data_inicio, user_type, unique_id, month, new_users, mau_registered_interactions",
        "transformacoes_joins": "Join por unique_id com dim_teachers; filtro user_type='registered'; contagem distinta mensal; reshape (melt) para desenhar as curvas.",
    },
    "retention_curve": {
        "como_foi_gerado": "Taxa de retenção m->m+1 e perda m->m+1 ao longo do tempo.",
        "tabelas_usadas": "fct_teachers_contents_interactions.csv + dim_teachers.csv -> retention_monthly_entries.csv",
        "colunas_chave": "unique_id, data_inicio, user_type, month, active_users, retained_next_month, retention_rate, drop_rate",
        "transformacoes_joins": "Join por unique_id com dim_teachers; filtro user_type='registered'; marca usuário ativo por mês (interações); autojoin no mês seguinte para medir retorno; cálculo de retention_rate/drop_rate.",
    },
    "journey_path_mix": {
        "como_foi_gerado": "Distribuição do caminho de adoção (aula->prova, prova->aula, mesmo dia).",
        "tabelas_usadas": "fct_teachers_contents_interactions.csv -> journey_path_counts.csv",
        "colunas_chave": "unique_id, data_inicio, event_type, first_aula_ts, first_prova_ts, path",
        "transformacoes_joins": "Primeiro timestamp de aula/prova por professor; cálculo do lag em dias; classificação do caminho; sem join nesta etapa.",
    },
    "activity_weekday_share": {
        "como_foi_gerado": "Share de eventos por dia da semana (todos vs heavy users).",
        "tabelas_usadas": "fct_teachers_contents_interactions.csv + teacher_dataset.csv",
        "colunas_chave": "unique_id, data_inicio, heavy_user_flag, weekday_label, events, share",
        "transformacoes_joins": "Filtra timestamps válidos e IDs presentes no teacher_dataset; agrega por dia da semana por segmento; calcula share no segmento.",
    },
    "activity_hour_share": {
        "como_foi_gerado": "Share de eventos por hora do dia (todos vs heavy users).",
        "tabelas_usadas": "fct_teachers_contents_interactions.csv + teacher_dataset.csv",
        "colunas_chave": "unique_id, data_inicio, heavy_user_flag, hour, events, share",
        "transformacoes_joins": "Filtra timestamps válidos e IDs presentes no teacher_dataset; agrega por hora por segmento; calcula share no segmento.",
    },
    "identity_coverage": {
        "como_foi_gerado": "Cobertura de IDs de cada fonte que casa com professores da dimensão.",
        "tabelas_usadas": "dim_teachers.csv + entries + interactions + mari_conv + mari_help -> identity_coverage.csv",
        "colunas_chave": "unique_id, unique_id_aprendizap, user_id, coverage_within_source, coverage_within_teachers",
        "transformacoes_joins": "Contagens distintas por fonte e contagens com match via joins de identidade; cálculo de percentuais de cobertura.",
    },
    "join_coverage": {
        "como_foi_gerado": "Percentual de cobertura dos joins críticos do pipeline.",
        "tabelas_usadas": "dim_teachers.csv + entries + interactions + lessons + formation + mari_conv + mari_help -> data_quality_join_coverage.csv",
        "colunas_chave": "join_name, source_distinct, matched_distinct, coverage",
        "transformacoes_joins": "Para cada join, compara IDs distintos na fonte vs IDs distintos com match; calcula coverage=matched/source.",
    },
    "state_conversion": {
        "como_foi_gerado": "Conversão por estado para UFs com base amostral mínima.",
        "tabelas_usadas": "teacher_dataset (derivado da etapa 01) -> state_stats.csv",
        "colunas_chave": "estado_group, teachers, conversion_rate",
        "transformacoes_joins": "Agrupamento por estado; cálculo de taxa de conversão média; filtro teachers>=500; ordenação.",
    },
    "utm_conversion": {
        "como_foi_gerado": "Relação entre volume e conversão por grupo de aquisição (UTM).",
        "tabelas_usadas": "teacher_dataset (derivado da etapa 01) -> utm_stats.csv",
        "colunas_chave": "utm_group, teachers, conversion_rate, median_interactions",
        "transformacoes_joins": "Normalização de utm_origin em grupos; agregação por grupo; dispersão com tamanho proporcional ao volume.",
    },
    "cluster_profiles": {
        "como_foi_gerado": "Subtipos comportamentais dentro de heavy users com tamanho e conversão.",
        "tabelas_usadas": "teacher_dataset -> deep_dive_cluster_profiles_detailed.csv",
        "colunas_chave": "features de uso, cluster, teachers, conversion_rate",
        "transformacoes_joins": "Filtro heavy_user_flag=1; features de mix comportamental; log1p + padronização; KMeans (k=2..6 por silhouette); perfil agregado por cluster.",
    },
    "heavy_summary": {
        "como_foi_gerado": "Comparação entre heavy users e base regular.",
        "tabelas_usadas": "teacher_dataset -> deep_dive_heavy_summary.csv",
        "colunas_chave": "heavy_user_flag, interaction_count, converted_within_window, teachers",
        "transformacoes_joins": "Heavy herdado da etapa 01 (heavy_score_fast_v1: PCA-1 + threshold holdout); agregação por segmento com mediana e conversão.",
    },
    "heavy_score_distribution": {
        "como_foi_gerado": "Distribuição do heavy_score_pca1 com marcação do threshold escolhido.",
        "tabelas_usadas": "teacher_dataset.csv + heavy_definition.json",
        "colunas_chave": "heavy_score_pca1, active_user_heavy_window_flag, heavy_threshold_value",
        "transformacoes_joins": "Filtro active_user_heavy_window_flag=1; histograma do score; linha vertical no threshold selecionado pelo grid holdout.",
    },
    "heavy_prevalence_stability": {
        "como_foi_gerado": "Série mensal de prevalência heavy entre usuários ativos.",
        "tabelas_usadas": "heavy_prevalence_monthly.csv",
        "colunas_chave": "month, active_users, heavy_users, heavy_prevalence",
        "transformacoes_joins": "Agregação por mês na base de interações registered; prevalência=heavy_users/active_users.",
    },
    "heavy_score_oot_validation": {
        "como_foi_gerado": "Validação out-of-time por decil do heavy score e lift heavy vs base.",
        "tabelas_usadas": "heavy_score_decile_diagnostics.csv + heavy_out_of_time_lift.csv",
        "colunas_chave": "score_decile, mean_future_interactions, future_value_event_rate, rsva_m1_lift_ratio_heavy_vs_base",
        "transformacoes_joins": "Decis por heavy_score no baseline e métricas no holdout; comparação heavy/base no holdout.",
    },
    "heavy_device_primary": {
        "como_foi_gerado": "Distribuição de dispositivo principal no segmento heavy.",
        "tabelas_usadas": "teacher_dataset.csv",
        "colunas_chave": "heavy_user_flag, primary_device, unique_id",
        "transformacoes_joins": "Filtro heavy_user_flag=1; contagem por primary_device; cálculo de participação no segmento.",
    },
    "heavy_device_reach": {
        "como_foi_gerado": "Cobertura de heavy users com ao menos um evento por dispositivo e uso desktop+mobile.",
        "tabelas_usadas": "teacher_dataset.csv",
        "colunas_chave": "heavy_user_flag, desktop_events, mobile_events, tablet_events, unknown_device_events",
        "transformacoes_joins": "Filtro heavy_user_flag=1; binarização de presença por dispositivo (>0 eventos) e contagem por métrica.",
    },
    "heavy_device_event_share": {
        "como_foi_gerado": "Participação dos eventos de heavy users por tipo de dispositivo.",
        "tabelas_usadas": "teacher_dataset.csv",
        "colunas_chave": "heavy_user_flag, desktop_events, mobile_events, tablet_events, unknown_device_events",
        "transformacoes_joins": "Filtro heavy_user_flag=1; soma de eventos por device; cálculo do share sobre o total de eventos de device.",
    },
    "heavy_profile_state_lift": {
        "como_foi_gerado": "Ranking de estados enriquecidos em heavy users por lift.",
        "tabelas_usadas": "dim_teachers.csv + teacher_dataset.csv",
        "colunas_chave": "estado, heavy_user_flag, teachers, heavy_share_within_cat, lift_vs_overall_heavy_rate",
        "transformacoes_joins": "Join de cadastro com heavy_user_flag da etapa 01; filtro de base ativa (active_user_heavy_window_flag=1); agregação por estado; cálculo de lift.",
    },
    "heavy_profile_utm_lift": {
        "como_foi_gerado": "Ranking de UTM enriched em heavy users por lift.",
        "tabelas_usadas": "dim_teachers.csv + teacher_dataset.csv",
        "colunas_chave": "utm_origin, heavy_user_flag, teachers, heavy_share_within_cat, lift_vs_overall_heavy_rate",
        "transformacoes_joins": "Join de cadastro com heavy_user_flag da etapa 01; filtro de base ativa (active_user_heavy_window_flag=1); agregação por utm_origin; cálculo de lift.",
    },
    "heavy_profile_tela_lift": {
        "como_foi_gerado": "Ranking de tela_origem enriched em heavy users por lift.",
        "tabelas_usadas": "dim_teachers.csv + teacher_dataset.csv",
        "colunas_chave": "tela_origem, heavy_user_flag, teachers, heavy_share_within_cat, lift_vs_overall_heavy_rate",
        "transformacoes_joins": "Join de cadastro com heavy_user_flag da etapa 01; filtro de base ativa (active_user_heavy_window_flag=1); agregação por tela_origem; cálculo de lift.",
    },
    "session_quality": {
        "como_foi_gerado": "Proxy de sessões muito curtas por perfil de atividade.",
        "tabelas_usadas": "teacher_dataset -> deep_dive_session_quality_by_profile.csv",
        "colunas_chave": "activity_tier, avg_session_min, short_session_proxy_rate, teachers",
        "transformacoes_joins": "Cria proxy short_session (avg_session_min<=5s); agrega por activity_tier; sem join adicional.",
    },
    "hypothesis_status": {
        "como_foi_gerado": "Contagem de hipóteses por status final.",
        "tabelas_usadas": "hypothesis_results.csv",
        "colunas_chave": "hypothesis_id, status",
        "transformacoes_joins": "Value_counts por status (validated/inconclusive/not_testable/rejected); sem join.",
    },
    "numeric_correlations": {
        "como_foi_gerado": "Top pares numéricos por correlação de Spearman.",
        "tabelas_usadas": "teacher_dataset -> top_corr_pairs.csv",
        "colunas_chave": "var1, var2, spearman, abs_spearman",
        "transformacoes_joins": "Matriz de correlação Spearman entre variáveis numéricas; ordenação por |rho|; top 20.",
    },
    "categorical_associations": {
        "como_foi_gerado": "Top associações categóricas medidas por Cramér's V.",
        "tabelas_usadas": "teacher_dataset -> cat_corr_pairs.csv",
        "colunas_chave": "var1, var2, cramers_v",
        "transformacoes_joins": "Crosstab para cada par categórico; cálculo de Cramér's V; ordenação decrescente.",
    },
    "activity_trend_windows": {
        "como_foi_gerado": "Inclinação da atividade mensal em janelas de 6/9/12 meses por regressão linear.",
        "tabelas_usadas": "users_monthly_panel.csv -> deep_dive_usage_trend_windows.csv",
        "colunas_chave": "month, mau_registered_interactions, window, slope_users_per_month, r_value, p_value",
        "transformacoes_joins": "Regressão linear por janela temporal; sem join adicional nesta etapa.",
    },
}


@dataclass(frozen=True)
class ReportConfig:
    base_dir: Path
    data_dir: Path
    output_dir: Path


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera relatório interativo final.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ReportConfig:
    base_dir = args.base_dir.resolve()
    data_dir = (args.data_dir if args.data_dir is not None else base_dir / "base_aprendizap").resolve()
    output_dir = (args.output_dir if args.output_dir is not None else base_dir / "analysis_output").resolve()
    return ReportConfig(base_dir=base_dir, data_dir=data_dir, output_dir=output_dir)


def ensure_dirs(output_dir: Path) -> Dict[str, Path]:
    paths = {
        "root": output_dir,
        "reports": output_dir / "reports",
        "excel": output_dir / "excel",
        "parquet": output_dir / "parquet",
        "csv": output_dir / "csv",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_teacher_dataset(output_dir: Path) -> pd.DataFrame:
    p_parquet = output_dir / "parquet" / "teacher_dataset.parquet"
    p_csv_full = output_dir / "teacher_dataset.csv"
    p_csv_sample = output_dir / "teacher_analytical_dataset_sample.csv"
    if p_parquet.exists():
        conn = duckdb.connect(database=":memory:")
        try:
            escaped = str(p_parquet).replace("'", "''")
            return conn.execute(f"SELECT * FROM read_parquet('{escaped}')").fetchdf()
        finally:
            conn.close()
    if p_csv_full.exists():
        return pd.read_csv(p_csv_full)
    if p_csv_sample.exists():
        return pd.read_csv(p_csv_sample)
    return pd.DataFrame()


def cleanup_legacy_artifacts(output_dir: Path) -> List[str]:
    removed: List[str] = []
    legacy_dirs = [
        output_dir / "survival_benchmark",
        output_dir / "archive",
        output_dir / "plots",
    ]
    legacy_globs = [
        ".DS_Store",
        "*churn*",
        "ml_*",
        "temporal_backtest*",
        "survival_*",
        "eda_churn_relationship_audit.csv",
        "deep_dive_churn_*",
    ]

    for d in legacy_dirs:
        if d.exists() and d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            removed.append(str(d))

    for pat in legacy_globs:
        for p in output_dir.glob(pat):
            if p.is_file():
                p.unlink(missing_ok=True)
                removed.append(str(p))

    for sub in ["csv", "parquet", "reports"]:
        subdir = output_dir / sub
        if not subdir.exists():
            continue
        for pat in legacy_globs:
            for p in subdir.glob(pat):
                if p.is_file():
                    p.unlink(missing_ok=True)
                    removed.append(str(p))

    return removed


def fmt_pct(x: float, digits: int = 1) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{100.0 * float(x):.{digits}f}%"


def fmt_num(x: float, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{float(x):,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def format_check_metric(check_name: str, value: float) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    v = float(value)
    if check_name.endswith("_rate"):
        return fmt_pct(v, 3)
    if abs(v - round(v)) < 1e-9:
        return fmt_num(v, 0)
    return fmt_num(v, 4)


def check_category(check_name: str) -> str:
    if check_name.startswith("entries_"):
        return CHECK_CATEGORY_MAP["entries"]
    if check_name.startswith("interactions_"):
        return CHECK_CATEGORY_MAP["interactions"]
    if check_name.startswith("dim_"):
        return CHECK_CATEGORY_MAP["dim"]
    if check_name.startswith("interaction_id_aula_"):
        return CHECK_CATEGORY_MAP["interaction_id_aula"]
    return "Outros checks"


def enrich_consistency_checks(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in ["check_name", "status", "expected", "details"]:
        if col not in out.columns:
            out[col] = ""
    out["metric_value"] = pd.to_numeric(out.get("metric_value"), errors="coerce")
    out["status"] = out["status"].astype(str).str.lower().fillna("info")
    out["category"] = out["check_name"].map(check_category)
    out["check_label"] = out["check_name"].map(CHECK_LABEL_MAP).fillna(out["check_name"])
    out["metric_label"] = out.apply(lambda row: format_check_metric(str(row["check_name"]), row["metric_value"]), axis=1)
    out["status"] = pd.Categorical(out["status"], categories=["fail", "warning", "info", "pass"], ordered=True)
    out = out.sort_values(["status", "category", "check_label"], ascending=[True, True, True]).reset_index(drop=True)
    return out


def build_consistency_table_html(consistency_df: pd.DataFrame) -> str:
    enriched = enrich_consistency_checks(consistency_df)
    if enriched.empty:
        return "<p class='small'>Sem checks de consistência disponíveis.</p>"
    view = enriched[["category", "check_label", "status", "metric_label", "expected", "details"]].copy()
    view.columns = ["Categoria", "Check", "Status", "Valor observado", "Regra esperada", "Motivo/descrição"]
    return view.to_html(index=False, classes="table", border=0, escape=True)


def to_month(df: pd.DataFrame, col: str = "month") -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = pd.to_datetime(out[col], errors="coerce")
        out = out.sort_values(col)
    return out


def compute_activity_time_panels(data_dir: Path, teacher_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    empty = pd.DataFrame()
    out: Dict[str, pd.DataFrame] = {
        "activity_weekday_segment_share": empty,
        "activity_hour_segment_share": empty,
        "activity_time_summary": empty,
    }
    if teacher_df is None or teacher_df.empty or "unique_id" not in teacher_df.columns:
        return out

    interactions_path = data_dir / "fct_teachers_contents_interactions.csv"
    if not interactions_path.exists():
        return out

    try:
        interactions = pd.read_csv(interactions_path, usecols=["unique_id", "data_inicio", "user_type"], low_memory=False)
    except ValueError:
        interactions = pd.read_csv(interactions_path, low_memory=False)
        needed = [c for c in ["unique_id", "data_inicio", "user_type"] if c in interactions.columns]
        if len(needed) < 2:
            return out
        interactions = interactions[needed].copy()

    interactions["data_inicio"] = pd.to_datetime(interactions["data_inicio"], errors="coerce")
    interactions = interactions[interactions["data_inicio"].notna()].copy()
    if "user_type" in interactions.columns:
        interactions["user_type"] = interactions["user_type"].astype(str).str.lower()
        interactions = interactions[interactions["user_type"] == "registered"].copy()
    if interactions.empty:
        return out

    teacher_cols = ["unique_id"]
    if "heavy_user_flag" in teacher_df.columns:
        teacher_cols.append("heavy_user_flag")
    teacher_key = teacher_df[teacher_cols].dropna(subset=["unique_id"]).drop_duplicates(subset=["unique_id"]).copy()
    if "heavy_user_flag" not in teacher_key.columns:
        teacher_key["heavy_user_flag"] = 0
    teacher_key["heavy_user_flag"] = pd.to_numeric(teacher_key["heavy_user_flag"], errors="coerce").fillna(0).astype(int)

    events = interactions.merge(teacher_key, on="unique_id", how="inner")
    if events.empty:
        return out

    weekday_labels = {0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui", 4: "Sex", 5: "Sab", 6: "Dom"}
    events["weekday_num"] = events["data_inicio"].dt.dayofweek.astype(int)
    events["weekday_label"] = events["weekday_num"].map(weekday_labels)
    events["hour"] = events["data_inicio"].dt.hour.astype(int)

    def segment_tables(df: pd.DataFrame, segment: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        weekday = (
            df.groupby(["weekday_num", "weekday_label"], as_index=False)
            .size()
            .rename(columns={"size": "events"})
            .sort_values("weekday_num")
            .reset_index(drop=True)
        )
        tot_weekday = float(weekday["events"].sum())
        weekday["share"] = weekday["events"] / tot_weekday if tot_weekday > 0 else np.nan
        weekday["segment"] = segment

        hour = (
            df.groupby("hour", as_index=False)
            .size()
            .rename(columns={"size": "events"})
            .sort_values("hour")
            .reset_index(drop=True)
        )
        tot_hour = float(hour["events"].sum())
        hour["share"] = hour["events"] / tot_hour if tot_hour > 0 else np.nan
        hour["segment"] = segment

        return weekday, hour

    w_all, h_all = segment_tables(events, "all_users")
    heavy_events = events[events["heavy_user_flag"] == 1].copy()
    if heavy_events.empty:
        w_heavy = pd.DataFrame(columns=w_all.columns)
        h_heavy = pd.DataFrame(columns=h_all.columns)
    else:
        w_heavy, h_heavy = segment_tables(heavy_events, "heavy_users")

    weekday_seg = pd.concat([w_all, w_heavy], ignore_index=True, sort=False)
    hour_seg = pd.concat([h_all, h_heavy], ignore_index=True, sort=False)

    def peak(df: pd.DataFrame, key_col: str) -> Tuple[float, float]:
        if df.empty or key_col not in df.columns:
            return (np.nan, np.nan)
        x = df.sort_values("events", ascending=False).head(1)
        if x.empty:
            return (np.nan, np.nan)
        return (x.iloc[0][key_col], float(x.iloc[0]["share"]) if "share" in x.columns else np.nan)

    all_peak_weekday, all_peak_weekday_share = peak(w_all, "weekday_label")
    all_peak_hour, all_peak_hour_share = peak(h_all, "hour")
    heavy_peak_weekday, heavy_peak_weekday_share = peak(w_heavy, "weekday_label")
    heavy_peak_hour, heavy_peak_hour_share = peak(h_heavy, "hour")

    summary_df = pd.DataFrame(
        [
            {
                "events_matched_teacher_dataset": int(len(events)),
                "events_heavy": int(len(heavy_events)),
                "all_peak_weekday": all_peak_weekday,
                "all_peak_weekday_share": all_peak_weekday_share,
                "all_peak_hour": all_peak_hour,
                "all_peak_hour_share": all_peak_hour_share,
                "heavy_peak_weekday": heavy_peak_weekday,
                "heavy_peak_weekday_share": heavy_peak_weekday_share,
                "heavy_peak_hour": heavy_peak_hour,
                "heavy_peak_hour_share": heavy_peak_hour_share,
            }
        ]
    )

    out["activity_weekday_segment_share"] = weekday_seg
    out["activity_hour_segment_share"] = hour_seg
    out["activity_time_summary"] = summary_df
    return out


def compute_heavy_user_device_panels(teacher_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    empty = pd.DataFrame()
    out: Dict[str, pd.DataFrame] = {
        "heavy_device_primary": empty,
        "heavy_interaction_device_reach": empty,
        "heavy_interaction_event_share": empty,
    }
    if teacher_df is None or teacher_df.empty or "heavy_user_flag" not in teacher_df.columns:
        return out

    heavy = teacher_df[teacher_df["heavy_user_flag"] == 1].copy()
    if heavy.empty:
        return out

    n_heavy = float(len(heavy))

    primary = heavy["primary_device"] if "primary_device" in heavy.columns else pd.Series(index=heavy.index, dtype="object")
    primary = primary.fillna("missing").astype(str).str.strip().str.lower().replace("", "missing")
    device_order = ["desktop", "mobile", "tablet", "unknown", "missing"]
    counts = primary.value_counts().reindex(device_order, fill_value=0)
    primary_df = pd.DataFrame({"device": counts.index, "teachers": counts.values})
    primary_df = primary_df[primary_df["teachers"] > 0].copy()
    primary_df["share"] = primary_df["teachers"] / n_heavy
    primary_df["device_label"] = primary_df["device"].str.title()
    out["heavy_device_primary"] = primary_df

    def num_col(col: str) -> pd.Series:
        if col not in heavy.columns:
            return pd.Series(0.0, index=heavy.index)
        return pd.to_numeric(heavy[col], errors="coerce").fillna(0.0)

    desktop_events = num_col("desktop_events")
    mobile_events = num_col("mobile_events")
    tablet_events = num_col("tablet_events")
    unknown_events = num_col("unknown_device_events")

    reach_rows = [
        {"metric": "users_with_desktop_event", "metric_label": ">=1 evento desktop", "users": int((desktop_events > 0).sum())},
        {"metric": "users_with_mobile_event", "metric_label": ">=1 evento mobile", "users": int((mobile_events > 0).sum())},
        {"metric": "users_with_both_desktop_mobile", "metric_label": ">=1 evento desktop e mobile", "users": int(((desktop_events > 0) & (mobile_events > 0)).sum())},
        {"metric": "users_with_tablet_event", "metric_label": ">=1 evento tablet", "users": int((tablet_events > 0).sum())},
        {"metric": "users_with_unknown_device_event", "metric_label": ">=1 evento unknown", "users": int((unknown_events > 0).sum())},
    ]
    reach_df = pd.DataFrame(reach_rows)
    reach_df["share"] = reach_df["users"] / n_heavy
    out["heavy_interaction_device_reach"] = reach_df

    event_rows = pd.DataFrame(
        [
            {"device": "desktop", "device_label": "Desktop", "events": float(desktop_events.sum())},
            {"device": "mobile", "device_label": "Mobile", "events": float(mobile_events.sum())},
            {"device": "unknown", "device_label": "Unknown", "events": float(unknown_events.sum())},
            {"device": "tablet", "device_label": "Tablet", "events": float(tablet_events.sum())},
        ]
    )
    total_events = float(event_rows["events"].sum())
    event_rows["event_share"] = event_rows["events"] / total_events if total_events > 0 else np.nan
    out["heavy_interaction_event_share"] = event_rows

    return out


def compute_heavy_non_activity_profile_sql(data_dir: Path, teacher_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    empty = pd.DataFrame()
    out: Dict[str, pd.DataFrame] = {
        "heavy_profile_definition": empty,
        "heavy_profile_available_characteristics": empty,
        "heavy_profile_state_stats": empty,
        "heavy_profile_state_enriched": empty,
        "heavy_profile_utm_stats": empty,
        "heavy_profile_tela_stats": empty,
        "heavy_profile_currentstage_stats": empty,
        "heavy_profile_currentsubject_stats": empty,
        "heavy_profile_tipo_total_alunos_stats": empty,
        "heavy_profile_numeric_compare": empty,
        "heavy_profile_login_google_stats": empty,
        "heavy_profile_tenure_compare": empty,
    }

    dim_path = data_dir / "dim_teachers.csv"
    interactions_path = data_dir / "fct_teachers_contents_interactions.csv"
    if not dim_path.exists() or teacher_df is None or teacher_df.empty:
        return out

    dim = pd.read_csv(dim_path, delimiter=";", low_memory=False)
    if dim.empty or "unique_id" not in dim.columns:
        return out

    candidate_features = [
        "estado",
        "currentstage",
        "currentsubject",
        "utm_origin",
        "tela_origem",
        "tipo_total_alunos",
        "total_alunos",
        "alunos_diretos",
        "alunos_indiretos",
        "login_google",
        "selectedstages",
        "visualizou_metodologia_ativa",
        "data_entrada",
    ]
    present_cols = set(dim.columns.tolist())
    out["heavy_profile_available_characteristics"] = pd.DataFrame(
        {
            "feature": candidate_features,
            "available": [int(col in present_cols) for col in candidate_features],
        }
    )

    if "unique_id" not in teacher_df.columns or "heavy_user_flag" not in teacher_df.columns:
        return out

    flags = teacher_df[["unique_id", "heavy_user_flag"]].copy()
    flags["unique_id"] = flags["unique_id"].astype(str).str.strip()
    flags["heavy_user_flag"] = pd.to_numeric(flags["heavy_user_flag"], errors="coerce").fillna(0).astype(int)
    if "active_user_heavy_window_flag" in teacher_df.columns:
        flags["active_user_heavy_window_flag"] = pd.to_numeric(
            teacher_df["active_user_heavy_window_flag"], errors="coerce"
        ).fillna(0).astype(int)
        population_rule = "active_user_heavy_window_flag=1"
    elif "interaction_count" in teacher_df.columns:
        flags["active_user_heavy_window_flag"] = (
            pd.to_numeric(teacher_df["interaction_count"], errors="coerce").fillna(0.0) > 0
        ).astype(int)
        population_rule = "interaction_count>0"
    else:
        flags["active_user_heavy_window_flag"] = 1
        population_rule = "all_teacher_rows"
    flags = flags.drop_duplicates(subset=["unique_id"], keep="last")

    base = dim.copy()
    base["unique_id"] = base["unique_id"].astype(str).str.strip()
    base = base.merge(flags, on="unique_id", how="left")
    base["heavy_user_flag"] = pd.to_numeric(base["heavy_user_flag"], errors="coerce").fillna(0).astype(int)
    base["active_user_heavy_window_flag"] = pd.to_numeric(
        base["active_user_heavy_window_flag"], errors="coerce"
    ).fillna(0).astype(int)
    base = base[base["active_user_heavy_window_flag"] == 1].copy()
    if base.empty:
        return out

    def norm_cat(series: pd.Series, mode: str = "lower") -> pd.Series:
        s = series.fillna("missing").astype(str).str.strip()
        s = s.replace("", "missing")
        if mode == "upper":
            return s.str.upper()
        if mode == "none":
            return s
        return s.str.lower()

    base["estado_norm"] = norm_cat(base["estado"] if "estado" in base.columns else pd.Series(index=base.index, dtype="object"), mode="upper")
    base["currentstage_norm"] = norm_cat(base["currentstage"] if "currentstage" in base.columns else pd.Series(index=base.index, dtype="object"), mode="lower")
    base["currentsubject_norm"] = norm_cat(base["currentsubject"] if "currentsubject" in base.columns else pd.Series(index=base.index, dtype="object"), mode="lower")
    base["utm_origin_norm"] = norm_cat(base["utm_origin"] if "utm_origin" in base.columns else pd.Series(index=base.index, dtype="object"), mode="none")
    base["tela_origem_norm"] = norm_cat(base["tela_origem"] if "tela_origem" in base.columns else pd.Series(index=base.index, dtype="object"), mode="none")
    base["tipo_total_alunos_norm"] = norm_cat(base["tipo_total_alunos"] if "tipo_total_alunos" in base.columns else pd.Series(index=base.index, dtype="object"), mode="none")

    for col in ["total_alunos", "alunos_diretos", "alunos_indiretos"]:
        base[f"{col}_num"] = pd.to_numeric(base[col], errors="coerce") if col in base.columns else np.nan

    login_raw = base["login_google"] if "login_google" in base.columns else pd.Series(index=base.index, dtype="object")
    login_num = pd.to_numeric(login_raw, errors="coerce")
    login_txt = login_raw.fillna("").astype(str).str.strip().str.lower()
    base["login_google_flag"] = ((login_num == 1) | (login_txt.isin({"true", "t", "yes", "sim"}))).astype(int)
    base["data_entrada_date"] = pd.to_datetime(base["data_entrada"], errors="coerce") if "data_entrada" in base.columns else pd.NaT

    total_teachers = int(len(base))
    heavy_users = int(base["heavy_user_flag"].sum())
    heavy_share = float(base["heavy_user_flag"].mean()) if total_teachers > 0 else np.nan
    out["heavy_profile_definition"] = pd.DataFrame(
        [
            {
                "total_teachers": total_teachers,
                "heavy_users": heavy_users,
                "heavy_share": heavy_share,
                "heavy_definition_method": "heavy_user_flag herdado da etapa 01 (heavy_score_fast_v1: PCA-1 em intensidade/consistência + threshold holdout).",
                "population_rule": population_rule,
            }
        ]
    )

    def category_profile(norm_col: str, out_col: str) -> pd.DataFrame:
        t = (
            base.groupby(norm_col, dropna=False, as_index=False)
            .agg(
                teachers=("unique_id", "count"),
                heavy_users=("heavy_user_flag", "sum"),
                heavy_share_within_cat=("heavy_user_flag", "mean"),
            )
            .rename(columns={norm_col: out_col})
        )
        t["lift_vs_overall_heavy_rate"] = t["heavy_share_within_cat"] / (heavy_share if (pd.notna(heavy_share) and heavy_share > 0) else np.nan)
        t["heavy_distribution_share"] = t["heavy_users"] / (heavy_users if heavy_users > 0 else np.nan)
        return t.sort_values(["heavy_share_within_cat", "teachers"], ascending=[False, False]).reset_index(drop=True)

    state_stats = category_profile("estado_norm", "estado")
    out["heavy_profile_state_stats"] = state_stats
    if not state_stats.empty:
        enriched = state_stats[(state_stats["estado"].astype(str).str.lower() != "missing") & (pd.to_numeric(state_stats["teachers"], errors="coerce") >= 500)]
        out["heavy_profile_state_enriched"] = enriched.sort_values(["lift_vs_overall_heavy_rate", "teachers"], ascending=[False, False]).reset_index(drop=True)

    out["heavy_profile_utm_stats"] = category_profile("utm_origin_norm", "utm_origin")
    out["heavy_profile_tela_stats"] = category_profile("tela_origem_norm", "tela_origem")
    out["heavy_profile_currentstage_stats"] = category_profile("currentstage_norm", "currentstage")
    out["heavy_profile_currentsubject_stats"] = category_profile("currentsubject_norm", "currentsubject")
    out["heavy_profile_tipo_total_alunos_stats"] = category_profile("tipo_total_alunos_norm", "tipo_total_alunos")

    metrics_rows: List[Dict[str, Any]] = []
    for metric, col in [
        ("total_alunos", "total_alunos_num"),
        ("alunos_diretos", "alunos_diretos_num"),
        ("alunos_indiretos", "alunos_indiretos_num"),
    ]:
        s_heavy = pd.to_numeric(base.loc[base["heavy_user_flag"] == 1, col], errors="coerce")
        s_base = pd.to_numeric(base.loc[base["heavy_user_flag"] == 0, col], errors="coerce")
        metrics_rows.append(
            {
                "metric": metric,
                "heavy_median": float(s_heavy.median()) if s_heavy.notna().any() else np.nan,
                "base_median": float(s_base.median()) if s_base.notna().any() else np.nan,
            }
        )
    out["heavy_profile_numeric_compare"] = pd.DataFrame(metrics_rows)

    login_stats = (
        base.groupby("login_google_flag", dropna=False, as_index=False)
        .agg(
            teachers=("unique_id", "count"),
            heavy_users=("heavy_user_flag", "sum"),
            heavy_share_within_cat=("heavy_user_flag", "mean"),
        )
        .sort_values("login_google_flag", ascending=False)
        .reset_index(drop=True)
    )
    login_stats["lift_vs_overall_heavy_rate"] = login_stats["heavy_share_within_cat"] / (heavy_share if (pd.notna(heavy_share) and heavy_share > 0) else np.nan)
    out["heavy_profile_login_google_stats"] = login_stats

    reference_date = pd.NaT
    if interactions_path.exists():
        try:
            dts = pd.read_csv(interactions_path, usecols=["data_inicio"], low_memory=False)
            reference_date = pd.to_datetime(dts["data_inicio"], errors="coerce").max()
        except Exception:
            reference_date = pd.NaT
    if pd.isna(reference_date):
        reference_date = pd.to_datetime(base["data_entrada_date"], errors="coerce").max()

    tenure_rows = []
    if pd.notna(reference_date):
        ages = (pd.Timestamp(reference_date).normalize() - pd.to_datetime(base["data_entrada_date"], errors="coerce")).dt.days
        heavy_age = pd.to_numeric(ages[base["heavy_user_flag"] == 1], errors="coerce")
        base_age = pd.to_numeric(ages[base["heavy_user_flag"] == 0], errors="coerce")
        tenure_rows.append(
            {
                "reference_date": pd.Timestamp(reference_date).date(),
                "heavy_median_account_age_days": float(heavy_age.median()) if heavy_age.notna().any() else np.nan,
                "base_median_account_age_days": float(base_age.median()) if base_age.notna().any() else np.nan,
            }
        )
    out["heavy_profile_tenure_compare"] = pd.DataFrame(tenure_rows)
    return out


def compute_heavy_cluster_diagnostics(teacher_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {"heavy_cluster_diagnostics": pd.DataFrame()}
    if teacher_df is None or teacher_df.empty or "heavy_user_flag" not in teacher_df.columns:
        return out

    heavy = teacher_df[teacher_df["heavy_user_flag"] == 1].copy()
    if heavy.empty or len(heavy) < 200:
        return out

    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return out

    rng = np.random.RandomState(42)

    def best_kmeans_silhouette(x: pd.DataFrame, scope: str) -> Dict[str, Any]:
        if x.empty:
            return {}
        data = x.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        if data.shape[1] == 0 or data.shape[0] < 200:
            return {}

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(data.values)

        sample_size = min(5000, x_scaled.shape[0])
        if sample_size < x_scaled.shape[0]:
            idx = rng.choice(x_scaled.shape[0], size=sample_size, replace=False)
            x_eval = x_scaled[idx]
        else:
            x_eval = x_scaled

        best_k = np.nan
        best_score = np.nan
        max_k = min(6, x_eval.shape[0] - 1)
        for k in range(2, max_k + 1):
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(x_eval)
            if len(np.unique(labels)) < 2:
                continue
            score = float(silhouette_score(x_eval, labels))
            if pd.isna(best_score) or score > best_score:
                best_k = int(k)
                best_score = score

        if pd.isna(best_score):
            quality = "indefinido"
        elif best_score >= 0.25:
            quality = "separacao_estrutura_relevante"
        elif best_score >= 0.10:
            quality = "separacao_moderada"
        else:
            quality = "separacao_fraca"

        return {
            "scope": scope,
            "n_heavy_users": int(len(heavy)),
            "n_rows_used": int(data.shape[0]),
            "n_features": int(data.shape[1]),
            "best_k": best_k,
            "best_silhouette": best_score,
            "quality_label": quality,
        }

    activity_cols = [
        "interaction_count",
        "session_count",
        "total_session_min",
        "avg_session_min",
        "aula_event_count",
        "prova_event_count",
        "plano_event_count",
        "download_event_count",
        "visualizacao_event_count",
        "metodologia_ativa_event_count",
        "ia_event_count",
        "unique_lessons_count",
        "desktop_events",
        "mobile_events",
        "tablet_events",
        "unknown_device_events",
    ]
    activity_cols = [c for c in activity_cols if c in heavy.columns]
    activity_df = heavy[activity_cols].apply(pd.to_numeric, errors="coerce") if activity_cols else pd.DataFrame(index=heavy.index)

    metadata_num_cols = [c for c in ["total_alunos", "alunos_diretos", "alunos_indiretos", "account_age_days", "login_google"] if c in heavy.columns]
    metadata_cat_cols = [c for c in ["estado", "currentstage", "currentsubject", "utm_origin", "tela_origem", "tipo_total_alunos"] if c in heavy.columns]

    metadata_num = heavy[metadata_num_cols].apply(pd.to_numeric, errors="coerce") if metadata_num_cols else pd.DataFrame(index=heavy.index)
    if metadata_cat_cols:
        metadata_cat = heavy[metadata_cat_cols].copy()
        for c in metadata_cat_cols:
            metadata_cat[c] = metadata_cat[c].fillna("missing").astype(str).str.strip().replace("", "missing").str.lower()
        metadata_cat = pd.get_dummies(metadata_cat, columns=metadata_cat_cols, prefix=metadata_cat_cols, drop_first=False)
    else:
        metadata_cat = pd.DataFrame(index=heavy.index)
    metadata_df = pd.concat([metadata_num, metadata_cat], axis=1)

    rows = []
    act_res = best_kmeans_silhouette(activity_df, "heavy_activity_features")
    if act_res:
        rows.append(act_res)
    meta_res = best_kmeans_silhouette(metadata_df, "heavy_metadata_features")
    if meta_res:
        rows.append(meta_res)

    if rows:
        out["heavy_cluster_diagnostics"] = pd.DataFrame(rows)
    return out


def fig_to_div(fig: go.Figure) -> str:
    return pio.to_html(fig, include_plotlyjs=False, full_html=False, config={"displaylogo": False})


def chart_lineage_html(chart_id: str) -> str:
    spec = CHART_LINEAGE.get(chart_id)
    if spec is None:
        return "<p class='small lineage'>Origem e transformação não documentadas para este gráfico.</p>"
    return (
        "<p class='small lineage'>"
        f"<b>Como foi gerado:</b> {spec['como_foi_gerado']}<br/>"
        f"<b>Tabelas usadas:</b> {spec['tabelas_usadas']}<br/>"
        f"<b>Colunas-chave:</b> {spec['colunas_chave']}<br/>"
        f"<b>Transformações/joins:</b> {spec['transformacoes_joins']}"
        "</p>"
    )


def chart_block(chart_id: str, title: str, subtitle: str, div: str) -> str:
    return f"""
    <div class="chart-card">
      <h3>{title}</h3>
      <p class="subtitle">{subtitle}</p>
      {div}
      {chart_lineage_html(chart_id)}
    </div>
    """


def build_figures(raw: Dict[str, pd.DataFrame], summary: Dict[str, float]) -> List[Tuple[str, str, str]]:
    figs: List[Tuple[str, str, str]] = []

    monthly = to_month(raw.get("monthly_solution_usage", pd.DataFrame()))
    if not monthly.empty and {"month", "aula_events", "prova_events"}.issubset(monthly.columns):
        m = monthly.copy()
        long = m.melt(id_vars="month", value_vars=["aula_events", "prova_events"], var_name="solution", value_name="events")
        long["solution"] = long["solution"].replace({"aula_events": "Aula", "prova_events": "Prova"})
        fig = px.line(long, x="month", y="events", color="solution", markers=True, color_discrete_sequence=[PALETTE[3], PALETTE[1]])
        fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), legend_title_text="Solução")
        figs.append(("journey_solution_volume", "Série mensal de eventos por solução", fig_to_div(fig)))

    users = to_month(raw.get("users_monthly_panel", pd.DataFrame()))
    if not users.empty and "month" in users.columns:
        for col in ["new_users", "mau_registered_entries", "mau_registered_interactions"]:
            if col in users.columns:
                users[col] = pd.to_numeric(users[col], errors="coerce")

        plot_cols = [c for c in ["new_users", "mau_registered_interactions"] if c in users.columns]
        if plot_cols:
            long = users.melt(id_vars="month", value_vars=plot_cols, var_name="metric", value_name="users")
            long["metric"] = long["metric"].replace(
                {
                    "new_users": "Novos usuários",
                    "mau_registered_interactions": "Usuários ativos (interactions)",
                }
            )
            fig = px.line(long, x="month", y="users", color="metric", markers=True, color_discrete_sequence=[PALETTE[2], PALETTE[0]])
            fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), legend_title_text="Métrica")
            figs.append(("users_activity_curve", "Aquisição vs atividade recorrente", fig_to_div(fig)))

    retention = to_month(raw.get("retention_monthly_entries", pd.DataFrame()))
    if not retention.empty and {"month", "retention_rate", "drop_rate"}.issubset(retention.columns):
        r = retention.copy()
        long = r.melt(id_vars="month", value_vars=["retention_rate", "drop_rate"], var_name="metric", value_name="rate")
        long["metric"] = long["metric"].replace({"retention_rate": "Retenção m->m+1", "drop_rate": "Perda m->m+1"})
        fig = px.line(long, x="month", y="rate", color="metric", markers=True, color_discrete_sequence=[PALETTE[3], "#B23A48"])
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), legend_title_text="Métrica")
        figs.append(("retention_curve", "Retenção mensal de usuários registered", fig_to_div(fig)))

    path = raw.get("journey_path_counts", pd.DataFrame())
    if not path.empty and {"path", "teachers"}.issubset(path.columns):
        fig = px.bar(path, x="path", y="teachers", color="path", color_discrete_sequence=PALETTE)
        fig.update_layout(height=320, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        figs.append(("journey_path_mix", "Ordem relativa de adoção entre aula e prova", fig_to_div(fig)))

    weekday_seg = raw.get("activity_weekday_segment_share", pd.DataFrame())
    if not weekday_seg.empty and {"weekday_label", "share", "segment"}.issubset(weekday_seg.columns):
        w = weekday_seg.copy()
        w["segment_label"] = w["segment"].map({"all_users": "Todos (teacher_dataset)", "heavy_users": "Heavy users"}).fillna(w["segment"])
        fig = px.bar(
            w,
            x="weekday_label",
            y="share",
            color="segment_label",
            barmode="group",
            text=w["share"].map(lambda v: fmt_pct(v, 2)),
            category_orders={"weekday_label": ["Seg", "Ter", "Qua", "Qui", "Sex", "Sab", "Dom"]},
            color_discrete_sequence=[PALETTE[2], PALETTE[0]],
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20), legend_title_text="Segmento", xaxis_title="Dia da semana")
        figs.append(("activity_weekday_share", "Atividade por dia da semana (share de eventos)", fig_to_div(fig)))

    hour_seg = raw.get("activity_hour_segment_share", pd.DataFrame())
    if not hour_seg.empty and {"hour", "share", "segment"}.issubset(hour_seg.columns):
        h = hour_seg.copy()
        h["segment_label"] = h["segment"].map({"all_users": "Todos (teacher_dataset)", "heavy_users": "Heavy users"}).fillna(h["segment"])
        fig = px.line(
            h,
            x="hour",
            y="share",
            color="segment_label",
            markers=True,
            color_discrete_sequence=[PALETTE[2], PALETTE[0]],
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_xaxes(dtick=1)
        fig.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20), legend_title_text="Segmento", xaxis_title="Hora do dia")
        figs.append(("activity_hour_share", "Atividade por hora do dia (share de eventos)", fig_to_div(fig)))

    id_cov = raw.get("identity_coverage", pd.DataFrame())
    if not id_cov.empty and {"source", "coverage_within_source"}.issubset(id_cov.columns):
        x = id_cov.copy()
        fig = px.bar(
            x,
            x="source",
            y="coverage_within_source",
            color="source",
            color_discrete_sequence=PALETTE,
            text=x["coverage_within_source"].map(lambda v: fmt_pct(v, 1)),
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(height=320, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        figs.append(("identity_coverage", "Cobertura de identidade por fonte", fig_to_div(fig)))

    join_cov = raw.get("join_coverage", pd.DataFrame())
    if not join_cov.empty and {"join_name", "coverage"}.issubset(join_cov.columns):
        j = join_cov.copy().sort_values("coverage", ascending=False)
        fig = px.bar(j, x="join_name", y="coverage", color="coverage", color_continuous_scale="Teal")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), coloraxis_showscale=False)
        figs.append(("join_coverage", "Cobertura de joins críticos", fig_to_div(fig)))

    state_stats = raw.get("state_stats", pd.DataFrame())
    if not state_stats.empty and {"estado_group", "teachers", "conversion_rate"}.issubset(state_stats.columns):
        s = state_stats.copy()
        s = s[s["teachers"] >= 500].sort_values("conversion_rate", ascending=False).head(20)
        fig = px.bar(s, x="estado_group", y="conversion_rate", color="teachers", color_continuous_scale="Tealgrn")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), coloraxis_colorbar_title="Professores")
        figs.append(("state_conversion", "Taxa de conversão por estado (n>=500)", fig_to_div(fig)))

    utm_stats = raw.get("utm_stats", pd.DataFrame())
    if not utm_stats.empty and {"utm_group", "teachers", "conversion_rate"}.issubset(utm_stats.columns):
        u = utm_stats.copy().sort_values("teachers", ascending=False)
        fig = px.scatter(u, x="teachers", y="conversion_rate", color="utm_group", size="teachers", hover_data=["median_interactions"] if "median_interactions" in u.columns else None)
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), legend_title_text="UTM")
        figs.append(("utm_conversion", "Conversão por grupo UTM", fig_to_div(fig)))

    cluster = raw.get("deep_dive_cluster_profiles_detailed", pd.DataFrame())
    if not cluster.empty and {"cluster", "teachers", "conversion_rate"}.issubset(cluster.columns):
        c = cluster.copy().sort_values("teachers", ascending=False)
        fig = px.bar(c, x="cluster", y="teachers", color="conversion_rate", color_continuous_scale="Teal")
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), coloraxis_colorbar_title="Conversão")
        figs.append(("cluster_profiles", "Subtipos de heavy users (clusters): tamanho e conversão", fig_to_div(fig)))

    teacher_df = raw.get("teacher", pd.DataFrame())
    heavy_def_json = raw.get("heavy_definition_json", {})
    if isinstance(teacher_df, pd.DataFrame) and not teacher_df.empty and {"heavy_score_pca1", "active_user_heavy_window_flag"}.issubset(teacher_df.columns):
        hs = teacher_df.copy()
        hs["active_user_heavy_window_flag"] = pd.to_numeric(hs["active_user_heavy_window_flag"], errors="coerce").fillna(0).astype(int)
        hs = hs[hs["active_user_heavy_window_flag"] == 1].copy()
        hs["heavy_score_pca1"] = pd.to_numeric(hs["heavy_score_pca1"], errors="coerce")
        hs = hs.dropna(subset=["heavy_score_pca1"])
        if not hs.empty:
            fig = px.histogram(
                hs,
                x="heavy_score_pca1",
                nbins=50,
                color_discrete_sequence=[PALETTE[2]],
                opacity=0.85,
            )
            thr = np.nan
            if isinstance(heavy_def_json, dict):
                thr = pd.to_numeric(heavy_def_json.get("selected_threshold_value"), errors="coerce")
            if pd.notna(thr):
                fig.add_vline(
                    x=float(thr),
                    line_width=2,
                    line_dash="dash",
                    line_color="#C92A2A",
                    annotation_text=f"threshold={float(thr):.3f}",
                    annotation_position="top right",
                )
            fig.update_layout(
                height=320,
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="heavy_score_pca1",
                yaxis_title="Usuários ativos (janela heavy)",
            )
            figs.append(("heavy_score_distribution", "Distribuição do heavy score e threshold selecionado", fig_to_div(fig)))

    prev = raw.get("heavy_prevalence_monthly", pd.DataFrame())
    if not prev.empty and {"month", "heavy_prevalence", "active_users"}.issubset(prev.columns):
        p = prev.copy()
        p["month"] = pd.to_datetime(p["month"], errors="coerce")
        p = p.dropna(subset=["month"]).sort_values("month")
        fig = px.line(p, x="month", y="heavy_prevalence", markers=True, color_discrete_sequence=[PALETTE[0]])
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Mês", yaxis_title="Prevalência heavy")
        figs.append(("heavy_prevalence_stability", "Estabilidade mensal da prevalência heavy", fig_to_div(fig)))

    dec = raw.get("heavy_score_decile_diagnostics", pd.DataFrame())
    oot = raw.get("heavy_out_of_time_lift", pd.DataFrame())
    if not dec.empty and {"score_decile", "mean_future_interactions"}.issubset(dec.columns):
        d = dec.copy().sort_values("score_decile")
        title_suffix = ""
        if not oot.empty and "rsva_m1_lift_ratio_heavy_vs_base" in oot.columns:
            lift = pd.to_numeric(oot["rsva_m1_lift_ratio_heavy_vs_base"], errors="coerce").dropna()
            if not lift.empty:
                title_suffix = f" | lift RSVA heavy/base={float(lift.iloc[0]):.2f}x"
        fig = px.bar(
            d,
            x="score_decile",
            y="mean_future_interactions",
            color="future_value_event_rate" if "future_value_event_rate" in d.columns else None,
            color_continuous_scale="Teal",
        )
        fig.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis_title="Decil do heavy_score_pca1",
            yaxis_title="Média de interações futuras (holdout)",
            coloraxis_colorbar_title="Taxa de valor futuro" if "future_value_event_rate" in d.columns else "",
        )
        figs.append(("heavy_score_oot_validation", f"Validação out-of-time por decil do heavy score{title_suffix}", fig_to_div(fig)))

    heavy = raw.get("deep_dive_heavy_summary", pd.DataFrame())
    if not heavy.empty and {"segment", "teachers", "conversion_rate"}.issubset(heavy.columns):
        fig = px.bar(heavy, x="segment", y="teachers", color="conversion_rate", color_continuous_scale="Teal")
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), coloraxis_colorbar_title="Conversão")
        figs.append(("heavy_summary", "Heavy users vs base regular", fig_to_div(fig)))

    heavy_primary = raw.get("heavy_device_primary", pd.DataFrame())
    if not heavy_primary.empty and {"device_label", "teachers", "share"}.issubset(heavy_primary.columns):
        d = heavy_primary.copy()
        fig = px.bar(d, x="device_label", y="teachers", color="share", text=d["share"].map(lambda v: fmt_pct(v, 2)), color_continuous_scale="Teal")
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), coloraxis_colorbar_title="Share")
        figs.append(("heavy_device_primary", "Dispositivo principal no segmento heavy", fig_to_div(fig)))

    # heavy_device_reach e heavy_device_event_share removidos da apresentação por redundância.

    heavy_state = raw.get("heavy_profile_state_enriched", pd.DataFrame())
    if not heavy_state.empty and {"estado", "teachers", "heavy_share_within_cat", "lift_vs_overall_heavy_rate"}.issubset(heavy_state.columns):
        s = heavy_state.copy().sort_values(["lift_vs_overall_heavy_rate", "teachers"], ascending=[False, False]).head(12)
        fig = px.bar(
            s,
            x="estado",
            y="lift_vs_overall_heavy_rate",
            color="heavy_share_within_cat",
            text=s["heavy_share_within_cat"].map(lambda v: fmt_pct(v, 1)),
            hover_data=["teachers"],
            color_continuous_scale="Teal",
        )
        fig.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20), yaxis_title="Lift vs taxa heavy geral", xaxis_title="Estado")
        figs.append(("heavy_profile_state_lift", "Estados enriquecidos em heavy users (lift)", fig_to_div(fig)))

    heavy_utm = raw.get("heavy_profile_utm_stats", pd.DataFrame())
    if not heavy_utm.empty and {"utm_origin", "teachers", "heavy_share_within_cat", "lift_vs_overall_heavy_rate"}.issubset(heavy_utm.columns):
        u = heavy_utm.copy()
        u = u[u["teachers"] >= 200].sort_values(["lift_vs_overall_heavy_rate", "teachers"], ascending=[False, False]).head(12)
        if not u.empty:
            fig = px.bar(
                u,
                x="utm_origin",
                y="lift_vs_overall_heavy_rate",
                color="heavy_share_within_cat",
                text=u["heavy_share_within_cat"].map(lambda v: fmt_pct(v, 1)),
                hover_data=["teachers"],
                color_continuous_scale="Teal",
            )
            fig.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20), yaxis_title="Lift vs taxa heavy geral", xaxis_title="UTM origin")
            figs.append(("heavy_profile_utm_lift", "UTM origins enriquecidas em heavy users (lift)", fig_to_div(fig)))

    heavy_tela = raw.get("heavy_profile_tela_stats", pd.DataFrame())
    if not heavy_tela.empty and {"tela_origem", "teachers", "heavy_share_within_cat", "lift_vs_overall_heavy_rate"}.issubset(heavy_tela.columns):
        t = heavy_tela.copy()
        t = t[t["teachers"] >= 200].sort_values(["lift_vs_overall_heavy_rate", "teachers"], ascending=[False, False]).head(12)
        if not t.empty:
            fig = px.bar(
                t,
                x="tela_origem",
                y="lift_vs_overall_heavy_rate",
                color="heavy_share_within_cat",
                text=t["heavy_share_within_cat"].map(lambda v: fmt_pct(v, 1)),
                hover_data=["teachers"],
                color_continuous_scale="Teal",
            )
            fig.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20), yaxis_title="Lift vs taxa heavy geral", xaxis_title="Tela de origem")
            figs.append(("heavy_profile_tela_lift", "Tela de origem enriquecida em heavy users (lift)", fig_to_div(fig)))

    session_quality = raw.get("deep_dive_session_quality_by_profile", pd.DataFrame())
    if not session_quality.empty and {"activity_tier", "short_session_proxy_rate"}.issubset(session_quality.columns):
        sq = session_quality.copy().sort_values("teachers", ascending=False)
        fig = px.bar(sq, x="activity_tier", y="short_session_proxy_rate", color="teachers", color_continuous_scale="Teal")
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=20), coloraxis_colorbar_title="Professores")
        figs.append(("session_quality", "Proxy de sessões <=5s por perfil de atividade", fig_to_div(fig)))

    hyp = raw.get("hypothesis_results", pd.DataFrame())
    if not hyp.empty and {"hypothesis_id", "status"}.issubset(hyp.columns):
        status = hyp["status"].value_counts().reset_index()
        status.columns = ["status", "count"]
        fig = px.bar(status, x="status", y="count", color="status", color_discrete_sequence=PALETTE)
        fig.update_layout(height=300, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        figs.append(("hypothesis_status", "Status das hipóteses testadas", fig_to_div(fig)))

    top_corr = raw.get("top_corr_pairs", pd.DataFrame())
    if not top_corr.empty and {"var1", "var2", "spearman"}.issubset(top_corr.columns):
        t = top_corr.head(20).copy()
        t["pair"] = t["var1"].astype(str) + " x " + t["var2"].astype(str)
        fig = px.bar(t.sort_values("abs_spearman", ascending=False), x="pair", y="spearman", color="spearman", color_continuous_scale="RdBu", range_color=[-1, 1])
        fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20), xaxis_tickangle=-35, coloraxis_colorbar_title="Spearman")
        figs.append(("numeric_correlations", "Top correlações numéricas (Spearman)", fig_to_div(fig)))

    cat_corr = raw.get("cat_corr_pairs", pd.DataFrame())
    if not cat_corr.empty and {"var1", "var2", "cramers_v"}.issubset(cat_corr.columns):
        c = cat_corr.head(15).copy()
        c["pair"] = c["var1"].astype(str) + " x " + c["var2"].astype(str)
        fig = px.bar(c.sort_values("cramers_v", ascending=False), x="pair", y="cramers_v", color="cramers_v", color_continuous_scale="Teal")
        fig.update_layout(height=340, margin=dict(l=20, r=20, t=40, b=20), xaxis_tickangle=-35, coloraxis_colorbar_title="Cramér's V")
        figs.append(("categorical_associations", "Associações categóricas (Cramér's V)", fig_to_div(fig)))

    trend = raw.get("deep_dive_usage_trend_windows", pd.DataFrame())
    if not trend.empty and {"window", "slope_users_per_month"}.issubset(trend.columns):
        fig = px.bar(trend, x="window", y="slope_users_per_month", color="slope_users_per_month", color_continuous_scale="RdBu")
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), xaxis_title="Janela (meses)", yaxis_title="Inclinação")
        figs.append(
            (
                "activity_trend_windows",
                "Inclinação (regressão linear) da atividade mensal em janelas de 6, 9 e 12 meses; valores negativos indicam queda.",
                fig_to_div(fig),
            )
        )

    return figs


def build_exec_quadro(summary: Dict[str, float], consolidated: Dict) -> pd.DataFrame:
    trust = consolidated.get("data_quality", {}).get("trust_assessment", {})

    rows = [
        {
            "item": "Qualidade",
            "metrica": "state_missing_pct",
            "valor": summary.get("state_missing_pct"),
            "unidade": "proporcao",
            "fonte": "consolidated_status.json -> eda",
            "nota": "Missing de estado na base de professores.",
        },
        {
            "item": "Qualidade",
            "metrica": "utm_missing_pct",
            "valor": summary.get("utm_missing_pct"),
            "unidade": "proporcao",
            "fonte": "consolidated_status.json -> eda",
            "nota": "Missing de utm_origin na base de professores.",
        },
        {
            "item": "Sessões",
            "metrica": "short_sessions_rate_le_5s",
            "valor": summary.get("short_sessions_rate_le_5s"),
            "unidade": "proporcao",
            "fonte": "entries",
            "nota": "Percentual de sessões com duração <= 5s.",
        },
        {
            "item": "Jornada",
            "metrica": "return_gap_median_days",
            "valor": summary.get("return_gap_median_days"),
            "unidade": "dias",
            "fonte": "entries",
            "nota": "Mediana do intervalo entre sessões consecutivas.",
        },
        {
            "item": "Atividade",
            "metrica": "recent_6m_mau_interactions_slope_users_per_month",
            "valor": summary.get("recent_6m_mau_interactions_slope_users_per_month"),
            "unidade": "usuarios_por_mes",
            "fonte": "users_monthly_panel",
            "nota": "Inclinação da atividade recorrente em 6 meses.",
        },
        {
            "item": "Retenção",
            "metrica": "retention_recent_avg_6m",
            "valor": summary.get("retention_recent_avg_6m"),
            "unidade": "proporcao",
            "fonte": "retention_monthly_entries",
            "nota": "Média de retenção m->m+1 nos 6 meses mais recentes (interações registered com match em professores).",
        },
        {
            "item": "Confiabilidade",
            "metrica": "trust_score_0_100",
            "valor": trust.get("trust_score_0_100"),
            "unidade": "score",
            "fonte": "consolidated_status.json -> data_quality.trust_assessment",
            "nota": "Score agregado de confiabilidade de dados (quanto maior, melhor).",
        },
    ]

    return pd.DataFrame(rows)


def write_bundles(paths: Dict[str, Path], tables: Dict[str, pd.DataFrame]) -> None:
    def write_parquet_duckdb(df: pd.DataFrame, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(database=":memory:")
        try:
            conn.register("tmp_df", df)
            escaped = str(out_path).replace("'", "''")
            conn.execute(f"COPY tmp_df TO '{escaped}' (FORMAT PARQUET)")
        finally:
            conn.close()

    excel_path = paths["excel"] / "analise_inicial_dos_dados_bundle.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for name, df in tables.items():
            if df is None or df.empty:
                continue
            safe_sheet = name[:31]
            df.to_excel(writer, sheet_name=safe_sheet, index=False)

    for name, df in tables.items():
        if df is None or df.empty:
            continue
        df.to_csv(paths["csv"] / f"{name}.csv", index=False)
        try:
            write_parquet_duckdb(df, paths["parquet"] / f"{name}.parquet")
        except Exception:
            # alguns tipos (listas/objetos complexos) podem não serializar em parquet; csv já cobre bundle técnico
            pass


def build_heavy_subtypes_summary(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def _add_enrichment_rows(df: pd.DataFrame, dim: str, value_col: str, min_teachers: int) -> None:
        if df is None or df.empty:
            return
        req = {value_col, "teachers", "heavy_share_within_cat", "lift_vs_overall_heavy_rate"}
        if not req.issubset(set(df.columns)):
            return
        x = df.copy()
        x["teachers"] = pd.to_numeric(x["teachers"], errors="coerce")
        x["heavy_share_within_cat"] = pd.to_numeric(x["heavy_share_within_cat"], errors="coerce")
        x["lift_vs_overall_heavy_rate"] = pd.to_numeric(x["lift_vs_overall_heavy_rate"], errors="coerce")
        x = x.dropna(subset=["teachers", "heavy_share_within_cat", "lift_vs_overall_heavy_rate"])
        x = x[(x["teachers"] >= float(min_teachers)) & (x["lift_vs_overall_heavy_rate"] >= 1.20)]
        if x.empty:
            return
        x = x.sort_values(["lift_vs_overall_heavy_rate", "teachers"], ascending=[False, False]).head(8)
        for _, r in x.iterrows():
            rows.append(
                {
                    "dimension": dim,
                    "subtype": str(r[value_col]),
                    "support_n": int(r["teachers"]),
                    "heavy_share_within_group": float(r["heavy_share_within_cat"]),
                    "lift_vs_overall_heavy_rate": float(r["lift_vs_overall_heavy_rate"]),
                    "delta_share_heavy_vs_all": np.nan,
                    "reliability_rule": f"teachers>={min_teachers} e lift>=1.20x",
                }
            )

    _add_enrichment_rows(raw.get("heavy_profile_state_enriched", pd.DataFrame()), "localizacao_estado", "estado", 500)
    _add_enrichment_rows(raw.get("heavy_profile_currentsubject_stats", pd.DataFrame()), "materia_currentsubject", "currentsubject", 200)

    heavy_primary = raw.get("heavy_device_primary", pd.DataFrame())
    if heavy_primary is not None and not heavy_primary.empty and {"device_label", "teachers", "share"}.issubset(heavy_primary.columns):
        hp = heavy_primary.copy()
        hp["teachers"] = pd.to_numeric(hp["teachers"], errors="coerce")
        hp["share"] = pd.to_numeric(hp["share"], errors="coerce")
        hp = hp.dropna(subset=["teachers", "share"])
        hp = hp[(hp["teachers"] >= 100) & (hp["share"] >= 0.10)].sort_values("share", ascending=False).head(6)
        for _, r in hp.iterrows():
            rows.append(
                {
                    "dimension": "device_primary",
                    "subtype": str(r["device_label"]),
                    "support_n": int(r["teachers"]),
                    "heavy_share_within_group": float(r["share"]),
                    "lift_vs_overall_heavy_rate": np.nan,
                    "delta_share_heavy_vs_all": np.nan,
                    "reliability_rule": "teachers>=100 e share>=10% no segmento heavy",
                }
            )

    def _add_time_delta(df: pd.DataFrame, key_col: str, dim: str) -> None:
        if df is None or df.empty:
            return
        req = {key_col, "segment", "share"}
        if not req.issubset(set(df.columns)):
            return
        t = df.copy()
        t["share"] = pd.to_numeric(t["share"], errors="coerce")
        t = t.dropna(subset=["share"])
        pvt = t.pivot_table(index=key_col, columns="segment", values="share", aggfunc="mean")
        if not {"all_users", "heavy_users"}.issubset(set(pvt.columns)):
            return
        pvt["delta"] = pd.to_numeric(pvt["heavy_users"], errors="coerce") - pd.to_numeric(pvt["all_users"], errors="coerce")
        pvt = pvt.dropna(subset=["delta"])
        pvt = pvt[pvt["delta"].abs() >= 0.015]
        if pvt.empty:
            return
        pvt = pvt.reindex(pvt["delta"].abs().sort_values(ascending=False).index).head(8)
        for idx, r in pvt.iterrows():
            rows.append(
                {
                    "dimension": dim,
                    "subtype": str(idx),
                    "support_n": np.nan,
                    "heavy_share_within_group": np.nan,
                    "lift_vs_overall_heavy_rate": np.nan,
                    "delta_share_heavy_vs_all": float(r["delta"]),
                    "reliability_rule": "abs(delta_share)>=1.5pp entre heavy_users e all_users",
                }
            )

    _add_time_delta(raw.get("activity_weekday_segment_share", pd.DataFrame()), "weekday_label", "dia_semana")
    _add_time_delta(raw.get("activity_hour_segment_share", pd.DataFrame()), "hour", "hora_dia")

    if not rows:
        return pd.DataFrame(
            columns=[
                "dimension",
                "subtype",
                "support_n",
                "heavy_share_within_group",
                "lift_vs_overall_heavy_rate",
                "delta_share_heavy_vs_all",
                "reliability_rule",
            ]
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["dimension", "lift_vs_overall_heavy_rate", "delta_share_heavy_vs_all", "support_n"], ascending=[True, False, False, False])
    out = out.reset_index(drop=True)
    return out


def build_html_report(
    figs: List[Tuple[str, str, str]],
    raw: Dict[str, pd.DataFrame],
    summary: Dict[str, float],
    consolidated: Dict,
) -> str:
    slope = summary.get("recent_6m_mau_interactions_slope_users_per_month")
    slope_txt = fmt_num(slope, 1)

    recent_users = summary.get("latest_new_users_count")
    latest_month = summary.get("latest_new_users_month")

    retention_avg = summary.get("retention_recent_avg_6m")
    short_rate = summary.get("short_sessions_rate_le_5s")
    gap_days = summary.get("return_gap_median_days")

    gap_hm_txt = "N/A"
    if gap_days is not None and not pd.isna(gap_days):
        total_minutes = int(round(float(gap_days) * 24 * 60))
        hh = total_minutes // 60
        mm = total_minutes % 60
        gap_hm_txt = f"~{hh}h{mm:02d}min"

    users_panel = to_month(raw.get("users_monthly_panel", pd.DataFrame()))
    activity_window_txt = "N/A"
    if not users_panel.empty and "month" in users_panel.columns:
        months = pd.to_datetime(users_panel["month"], errors="coerce").dropna().sort_values().unique()
        if len(months) >= 1:
            recent = months[-6:] if len(months) >= 6 else months
            activity_window_txt = f"{pd.Timestamp(recent[0]).date()} a {pd.Timestamp(recent[-1]).date()}"

    retention_panel = to_month(raw.get("retention_monthly_entries", pd.DataFrame()))
    retention_window_txt = "N/A"
    if not retention_panel.empty and "month" in retention_panel.columns:
        months = pd.to_datetime(retention_panel["month"], errors="coerce").dropna().sort_values().unique()
        if len(months) >= 1:
            recent = months[-6:] if len(months) >= 6 else months
            retention_window_txt = f"{pd.Timestamp(recent[0]).date()} a {pd.Timestamp(recent[-1]).date()}"

    hyp = raw.get("hypothesis_results", pd.DataFrame())
    if not hyp.empty:
        hyp_view = hyp.copy()
        for col in ["p_value", "effect_size"]:
            if col in hyp_view.columns:
                hyp_view[col] = pd.to_numeric(hyp_view[col], errors="coerce")
        hyp_html = hyp_view.to_html(index=False, classes="table", border=0)
    else:
        hyp_html = "<p class='small'>Sem resultados de hipótese disponíveis.</p>"

    table_sources = pd.DataFrame(
        [
            {"tabela": "dim_teachers.csv", "papel": "Cadastro base dos professores."},
            {"tabela": "fct_teachers_entries.csv", "papel": "Sessões e duração de uso."},
            {"tabela": "fct_teachers_contents_interactions.csv", "papel": "Eventos de interação por solução/canal/device."},
            {"tabela": "stg_lessons.csv", "papel": "Metadados de aula e disciplina."},
            {"tabela": "stg_formation.csv", "papel": "Progressão em formação."},
            {"tabela": "stg_mari_ia_conversation.csv", "papel": "Interações com IA assistente."},
            {"tabela": "fct_mari_ia_eventos_isso_ajudou.csv", "papel": "Feedback de utilidade (sem bridge de identidade)."},
        ]
    ).to_html(index=False, classes="table", border=0)

    fig_blocks = {k: chart_block(k, k, subtitle, div) for k, subtitle, div in figs}
    consistency_table_html = build_consistency_table_html(raw.get("consistency_checks", pd.DataFrame()))
    activity_time_summary = raw.get("activity_time_summary", pd.DataFrame())

    activity_time_note_html = ""
    if activity_time_summary is not None and not activity_time_summary.empty:
        row = activity_time_summary.iloc[0]
        events_all = pd.to_numeric(row.get("events_matched_teacher_dataset"), errors="coerce")
        events_heavy = pd.to_numeric(row.get("events_heavy"), errors="coerce")
        all_day = row.get("all_peak_weekday")
        all_day_share = pd.to_numeric(row.get("all_peak_weekday_share"), errors="coerce")
        all_hour = pd.to_numeric(row.get("all_peak_hour"), errors="coerce")
        all_hour_share = pd.to_numeric(row.get("all_peak_hour_share"), errors="coerce")
        hv_day = row.get("heavy_peak_weekday")
        hv_day_share = pd.to_numeric(row.get("heavy_peak_weekday_share"), errors="coerce")
        hv_hour = pd.to_numeric(row.get("heavy_peak_hour"), errors="coerce")
        hv_hour_share = pd.to_numeric(row.get("heavy_peak_hour_share"), errors="coerce")

        def _fmt_int(v: float) -> str:
            if v is None or pd.isna(v):
                return "N/A"
            return f"{int(round(float(v))):,}"

        def _fmt_hour(v: float) -> str:
            if v is None or pd.isna(v):
                return "N/A"
            return f"{int(round(float(v))):02d}h"

        activity_time_note_html = (
            "<div class='note'>"
            "<b>Base reprodutível de dia/hora:</b> interações com <code>data_inicio</code> válido e "
            "<code>unique_id</code> presente no <code>teacher_dataset</code>.<br/>"
            f"Eventos usados: {_fmt_int(events_all)} (todos) e {_fmt_int(events_heavy)} (heavy). "
            f"Pico semanal (todos): {all_day} ({fmt_pct(all_day_share,2)}). Pico horário (todos): {_fmt_hour(all_hour)} ({fmt_pct(all_hour_share,2)}). "
            f"Pico semanal (heavy): {hv_day} ({fmt_pct(hv_day_share,2)}). Pico horário (heavy): {_fmt_hour(hv_hour)} ({fmt_pct(hv_hour_share,2)})."
            "</div>"
        )

    heavy_definition = raw.get("heavy_profile_definition", pd.DataFrame())
    heavy_definition_json = raw.get("heavy_definition_json", {}) if isinstance(raw.get("heavy_definition_json", {}), dict) else {}
    heavy_available = raw.get("heavy_profile_available_characteristics", pd.DataFrame())
    heavy_state_enriched = raw.get("heavy_profile_state_enriched", pd.DataFrame())
    heavy_utm = raw.get("heavy_profile_utm_stats", pd.DataFrame())
    heavy_tela = raw.get("heavy_profile_tela_stats", pd.DataFrame())
    heavy_stage = raw.get("heavy_profile_currentstage_stats", pd.DataFrame())
    heavy_subject = raw.get("heavy_profile_currentsubject_stats", pd.DataFrame())
    heavy_tipo = raw.get("heavy_profile_tipo_total_alunos_stats", pd.DataFrame())
    heavy_numeric = raw.get("heavy_profile_numeric_compare", pd.DataFrame())
    heavy_login = raw.get("heavy_profile_login_google_stats", pd.DataFrame())
    heavy_tenure = raw.get("heavy_profile_tenure_compare", pd.DataFrame())

    def _fmt_int(v: float) -> str:
        if v is None or pd.isna(v):
            return "N/A"
        return f"{int(round(float(v))):,}"

    def _fmt_ratio(v: float, digits: int = 2) -> str:
        if v is None or pd.isna(v):
            return "N/A"
        return f"{float(v):.{digits}f}x"

    def _find_row(df: pd.DataFrame, col: str, candidates: List[str]) -> pd.Series:
        if df is None or df.empty or col not in df.columns:
            return pd.Series(dtype="object")
        x = df.copy()
        x[col] = x[col].astype(str)
        x["_key"] = x[col].str.strip().str.lower()
        for cand in candidates:
            m = x[x["_key"] == cand.strip().lower()]
            if not m.empty:
                return m.iloc[0]
        return pd.Series(dtype="object")

    def _profile_table_html(df: pd.DataFrame, cat_col: str, top_n: int = 10, min_teachers: int = 0) -> str:
        if df is None or df.empty or cat_col not in df.columns:
            return "<p class='small'>Tabela indisponível.</p>"
        t = df.copy()
        if "teachers" in t.columns and min_teachers > 0:
            t = t[pd.to_numeric(t["teachers"], errors="coerce") >= min_teachers]
        if "lift_vs_overall_heavy_rate" in t.columns:
            t = t.sort_values(["lift_vs_overall_heavy_rate", "teachers"], ascending=[False, False])
        t = t.head(top_n)
        if t.empty:
            return "<p class='small'>Sem linhas para os filtros atuais.</p>"
        if "heavy_share_within_cat" in t.columns:
            t["heavy_share_within_cat"] = pd.to_numeric(t["heavy_share_within_cat"], errors="coerce").map(lambda v: fmt_pct(v, 2))
        if "lift_vs_overall_heavy_rate" in t.columns:
            t["lift_vs_overall_heavy_rate"] = pd.to_numeric(t["lift_vs_overall_heavy_rate"], errors="coerce").map(lambda v: _fmt_ratio(v, 2))
        keep = [c for c in [cat_col, "teachers", "heavy_users", "heavy_share_within_cat", "lift_vs_overall_heavy_rate"] if c in t.columns]
        t = t[keep]
        rename = {
            cat_col: "categoria",
            "teachers": "professores",
            "heavy_users": "heavy_users",
            "heavy_share_within_cat": "heavy_share",
            "lift_vs_overall_heavy_rate": "lift_vs_heavy_rate_geral",
        }
        t = t.rename(columns=rename)
        return t.to_html(index=False, classes="table", border=0, escape=True)

    available_features_txt = "N/A"
    if heavy_available is not None and not heavy_available.empty and {"feature", "available"}.issubset(heavy_available.columns):
        feats = heavy_available[heavy_available["available"] == 1]["feature"].astype(str).tolist()
        if feats:
            available_features_txt = ", ".join(feats)

    heavy_profile_note_html = ""
    if heavy_definition is not None and not heavy_definition.empty:
        drow = heavy_definition.iloc[0]
        total_teachers = pd.to_numeric(drow.get("total_teachers"), errors="coerce")
        heavy_users = pd.to_numeric(drow.get("heavy_users"), errors="coerce")
        heavy_share = pd.to_numeric(drow.get("heavy_share"), errors="coerce")
        heavy_method = str(drow.get("heavy_definition_method", "")).strip()
        heavy_population_rule = str(drow.get("population_rule", "")).strip()
        heavy_q = pd.to_numeric(heavy_definition_json.get("selected_threshold_quantile"), errors="coerce")
        heavy_thr = pd.to_numeric(heavy_definition_json.get("selected_threshold_value"), errors="coerce")

        top_states = ""
        if heavy_state_enriched is not None and not heavy_state_enriched.empty and "estado" in heavy_state_enriched.columns:
            top_states = ", ".join(heavy_state_enriched.head(5)["estado"].astype(str).tolist())

        utm_bot = _find_row(heavy_utm, "utm_origin", ["Bot"])
        utm_seo_org = _find_row(heavy_utm, "utm_origin", ["SEO Orgânico", "SEO Organico"])
        utm_seo_ads = _find_row(heavy_utm, "utm_origin", ["SEO Ads"])
        utm_missing = _find_row(heavy_utm, "utm_origin", ["missing"])
        utm_google_ads = _find_row(heavy_utm, "utm_origin", ["Google Ads"])

        tela_aula_seo = _find_row(heavy_tela, "tela_origem", ["Aula SEO"])
        tela_bot = _find_row(heavy_tela, "tela_origem", ["Bot"])
        tela_inicial = _find_row(heavy_tela, "tela_origem", ["Tela inicial"])

        stage_all = _find_row(heavy_stage, "currentstage", ["all"])
        stage_fundii = _find_row(heavy_stage, "currentstage", ["fundii"])
        stage_em = _find_row(heavy_stage, "currentstage", ["em"])

        tipo_direto = _find_row(heavy_tipo, "tipo_total_alunos", ["direto", "Direto"])
        login_google_1 = pd.Series(dtype="object")
        if heavy_login is not None and not heavy_login.empty and "login_google_flag" in heavy_login.columns:
            x = heavy_login[pd.to_numeric(heavy_login["login_google_flag"], errors="coerce") == 1]
            if not x.empty:
                login_google_1 = x.iloc[0]

        metric_map = {}
        if heavy_numeric is not None and not heavy_numeric.empty and "metric" in heavy_numeric.columns:
            metric_map = {str(r["metric"]): r for _, r in heavy_numeric.iterrows()}

        tenure_row = heavy_tenure.iloc[0] if heavy_tenure is not None and not heavy_tenure.empty else pd.Series(dtype="object")
        ref_date = tenure_row.get("reference_date")
        heavy_tenure_days = pd.to_numeric(tenure_row.get("heavy_median_account_age_days"), errors="coerce")
        base_tenure_days = pd.to_numeric(tenure_row.get("base_median_account_age_days"), errors="coerce")

        missing_state_row = _find_row(raw.get("heavy_profile_state_stats", pd.DataFrame()), "estado", ["missing"])

        heavy_profile_note_html = (
            "<div class='note'>"
            "<b>Heavy definition used (same logic as etapa_01):</b><br/>"
            f"{heavy_method if heavy_method else 'heavy_user_flag herdado da etapa 01 (heavy_score_fast_v1: PCA-1 + threshold holdout).'}<br/>"
            f"Base de comparação de heavy: {heavy_population_rule if heavy_population_rule else 'active_user_heavy_window_flag=1'}.<br/>"
            f"Threshold selecionado: quantil {fmt_num(heavy_q,2) if pd.notna(heavy_q) else 'N/A'} "
            f"(score={fmt_num(heavy_thr,3) if pd.notna(heavy_thr) else 'N/A'}).<br/>"
            f"Resultado: {_fmt_int(heavy_users)} / {_fmt_int(total_teachers)} heavy ({fmt_pct(heavy_share,2)}).<br/>"
            f"<b>Características não-atividade disponíveis:</b> {available_features_txt}.<br/>"
            f"<b>estado:</b> estados enriquecidos (lift&gt;2x, n&gt;=500): {top_states if top_states else 'N/A'}. "
            f"Grupo estado=missing: heavy share {fmt_pct(pd.to_numeric(missing_state_row.get('heavy_share_within_cat'), errors='coerce'),2)} "
            f"(lift {_fmt_ratio(pd.to_numeric(missing_state_row.get('lift_vs_overall_heavy_rate'), errors='coerce'),2)}).<br/>"
            f"<b>utm_origin:</b> Bot {fmt_pct(pd.to_numeric(utm_bot.get('heavy_share_within_cat'), errors='coerce'),1)} "
            f"({_fmt_ratio(pd.to_numeric(utm_bot.get('lift_vs_overall_heavy_rate'), errors='coerce'),2)}), "
            f"SEO Orgânico {fmt_pct(pd.to_numeric(utm_seo_org.get('heavy_share_within_cat'), errors='coerce'),1)} "
            f"({_fmt_ratio(pd.to_numeric(utm_seo_org.get('lift_vs_overall_heavy_rate'), errors='coerce'),2)}), "
            f"SEO Ads {fmt_pct(pd.to_numeric(utm_seo_ads.get('heavy_share_within_cat'), errors='coerce'),1)} "
            f"({_fmt_ratio(pd.to_numeric(utm_seo_ads.get('lift_vs_overall_heavy_rate'), errors='coerce'),2)}); "
            f"depletados: missing {fmt_pct(pd.to_numeric(utm_missing.get('heavy_share_within_cat'), errors='coerce'),2)} "
            f"({_fmt_ratio(pd.to_numeric(utm_missing.get('lift_vs_overall_heavy_rate'), errors='coerce'),2)}) e Google Ads "
            f"{fmt_pct(pd.to_numeric(utm_google_ads.get('heavy_share_within_cat'), errors='coerce'),2)} "
            f"({_fmt_ratio(pd.to_numeric(utm_google_ads.get('lift_vs_overall_heavy_rate'), errors='coerce'),2)}).<br/>"
            f"<b>tela_origem:</b> Aula SEO {fmt_pct(pd.to_numeric(tela_aula_seo.get('heavy_share_within_cat'), errors='coerce'),1)} "
            f"({_fmt_ratio(pd.to_numeric(tela_aula_seo.get('lift_vs_overall_heavy_rate'), errors='coerce'),2)}), "
            f"Bot {fmt_pct(pd.to_numeric(tela_bot.get('heavy_share_within_cat'), errors='coerce'),1)} "
            f"({_fmt_ratio(pd.to_numeric(tela_bot.get('lift_vs_overall_heavy_rate'), errors='coerce'),2)}), "
            f"Tela inicial {fmt_pct(pd.to_numeric(tela_inicial.get('heavy_share_within_cat'), errors='coerce'),2)} "
            f"({_fmt_ratio(pd.to_numeric(tela_inicial.get('lift_vs_overall_heavy_rate'), errors='coerce'),2)}).<br/>"
            f"<b>currentstage:</b> all {fmt_pct(pd.to_numeric(stage_all.get('heavy_share_within_cat'), errors='coerce'),2)} "
            f"({_fmt_ratio(pd.to_numeric(stage_all.get('lift_vs_overall_heavy_rate'), errors='coerce'),2)}), "
            f"fundii {fmt_pct(pd.to_numeric(stage_fundii.get('heavy_share_within_cat'), errors='coerce'),2)}, "
            f"em {fmt_pct(pd.to_numeric(stage_em.get('heavy_share_within_cat'), errors='coerce'),2)}.<br/>"
            f"<b>tipo_total_alunos:</b> Direto {fmt_pct(pd.to_numeric(tipo_direto.get('heavy_share_within_cat'), errors='coerce'),1)} "
            f"({_fmt_ratio(pd.to_numeric(tipo_direto.get('lift_vs_overall_heavy_rate'), errors='coerce'),2)}; n={_fmt_int(pd.to_numeric(tipo_direto.get('teachers'), errors='coerce'))}).<br/>"
            f"<b>Base de alunos (medianas heavy vs base):</b> total_alunos "
            f"{fmt_num(pd.to_numeric(metric_map.get('total_alunos', pd.Series(dtype='object')).get('heavy_median'), errors='coerce'),0)} vs "
            f"{fmt_num(pd.to_numeric(metric_map.get('total_alunos', pd.Series(dtype='object')).get('base_median'), errors='coerce'),0)}; "
            f"alunos_diretos {fmt_num(pd.to_numeric(metric_map.get('alunos_diretos', pd.Series(dtype='object')).get('heavy_median'), errors='coerce'),0)} vs "
            f"{fmt_num(pd.to_numeric(metric_map.get('alunos_diretos', pd.Series(dtype='object')).get('base_median'), errors='coerce'),0)}; "
            f"alunos_indiretos {fmt_num(pd.to_numeric(metric_map.get('alunos_indiretos', pd.Series(dtype='object')).get('heavy_median'), errors='coerce'),0)} vs "
            f"{fmt_num(pd.to_numeric(metric_map.get('alunos_indiretos', pd.Series(dtype='object')).get('base_median'), errors='coerce'),0)}.<br/>"
            f"<b>login_google=1:</b> heavy share {fmt_pct(pd.to_numeric(login_google_1.get('heavy_share_within_cat'), errors='coerce'),2)} "
            f"({_fmt_ratio(pd.to_numeric(login_google_1.get('lift_vs_overall_heavy_rate'), errors='coerce'),2)}).<br/>"
            f"<b>Tenure (data_entrada):</b> referência {ref_date if ref_date is not None else 'N/A'}, "
            f"idade mediana da conta heavy {_fmt_int(heavy_tenure_days)} dias vs base {_fmt_int(base_tenure_days)} dias."
            "</div>"
        )

    heavy_numeric_html = "<p class='small'>Sem comparação numérica heavy vs base.</p>"
    if heavy_numeric is not None and not heavy_numeric.empty:
        n = heavy_numeric.copy()
        n["heavy_median"] = pd.to_numeric(n["heavy_median"], errors="coerce").map(lambda v: fmt_num(v, 0))
        n["base_median"] = pd.to_numeric(n["base_median"], errors="coerce").map(lambda v: fmt_num(v, 0))
        n = n.rename(columns={"metric": "métrica", "heavy_median": "mediana_heavy", "base_median": "mediana_base"})
        heavy_numeric_html = n.to_html(index=False, classes="table", border=0, escape=True)

    heavy_subtypes = raw.get("heavy_user_subtypes_summary", pd.DataFrame())
    heavy_subtypes_html = "<p class='small'>Sem evidência robusta de subtipos heavy nos critérios definidos.</p>"
    if heavy_subtypes is not None and not heavy_subtypes.empty:
        hs = heavy_subtypes.copy()
        if "support_n" in hs.columns:
            hs["support_n"] = pd.to_numeric(hs["support_n"], errors="coerce").round(0)
        for c in ["heavy_share_within_group", "lift_vs_overall_heavy_rate", "delta_share_heavy_vs_all"]:
            if c in hs.columns:
                hs[c] = pd.to_numeric(hs[c], errors="coerce").round(4)
        hs = hs.rename(
            columns={
                "dimension": "dimensão",
                "subtype": "subtipo",
                "support_n": "suporte_n",
                "heavy_share_within_group": "heavy_share_no_grupo",
                "lift_vs_overall_heavy_rate": "lift_vs_taxa_heavy_geral",
                "delta_share_heavy_vs_all": "delta_share_heavy_vs_all",
                "reliability_rule": "critério_de_confiabilidade",
            }
        )
        heavy_subtypes_html = hs.to_html(index=False, classes="table", border=0, escape=True)

    heavy_state_table_html = _profile_table_html(heavy_state_enriched, "estado", top_n=10, min_teachers=500)
    heavy_utm_table_html = _profile_table_html(heavy_utm, "utm_origin", top_n=12, min_teachers=200)
    heavy_tela_table_html = _profile_table_html(heavy_tela, "tela_origem", top_n=12, min_teachers=200)
    heavy_stage_table_html = _profile_table_html(heavy_stage, "currentstage", top_n=10, min_teachers=0)
    heavy_subject_table_html = _profile_table_html(heavy_subject, "currentsubject", top_n=10, min_teachers=200)
    heavy_tipo_table_html = _profile_table_html(heavy_tipo, "tipo_total_alunos", top_n=10, min_teachers=0)

    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #F7FAFC; color: #1A202C; }
    .container { max-width: 1200px; margin: 0 auto; padding: 24px; }
    h1 { margin: 0 0 8px 0; font-size: 30px; color: #102A43; }
    h2 { margin-top: 34px; margin-bottom: 8px; font-size: 22px; color: #102A43; }
    h3 { margin: 0 0 6px 0; font-size: 18px; color: #102A43; }
    p { margin: 8px 0; line-height: 1.45; }
    .small { color: #4A5568; font-size: 13px; }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 16px 0 8px 0; }
    .card { background: white; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px 14px; }
    .card h4 { margin: 0 0 6px 0; font-size: 13px; color: #486581; }
    .card .value { font-size: 24px; font-weight: 700; color: #102A43; }
    .chart-card { background: white; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px 14px; margin: 12px 0; }
    .subtitle { margin: 0 0 8px 0; color: #627D98; font-size: 13px; }
    .lineage { border-top: 1px dashed #CBD5E0; padding-top: 8px; margin-top: 8px; }
    .table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; background: white; }
    .table th, .table td { border: 1px solid #E2E8F0; padding: 8px; vertical-align: top; text-align: left; }
    .table th { background: #F0F4F8; color: #243B53; }
    .note { background: #E6FFFA; border: 1px solid #81E6D9; border-radius: 8px; padding: 10px 12px; margin: 12px 0; }
    """

    html = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Análise inicial dos dados (versão limpa)</title>
  <style>{css}</style>
  <script>{get_plotlyjs()}</script>
</head>
<body>
  <div class="container">
    <h1>Análise inicial dos dados</h1>
    <p class="small">Versão focada em transformações verificáveis da base.</p>

    <div class="cards">
      <div class="card"><h4>Missing de estado</h4><div class="value">{fmt_pct(summary.get('state_missing_pct'),1)}</div></div>
      <div class="card"><h4>Missing de UTM</h4><div class="value">{fmt_pct(summary.get('utm_missing_pct'),1)}</div></div>
      <div class="card"><h4>Sessões <=5s</h4><div class="value">{fmt_pct(short_rate,1)}</div></div>
      <div class="card"><h4>Gap mediano entre sessões</h4><div class="value">{gap_hm_txt}</div></div>
      <div class="card"><h4>Inclinação atividade (6m)</h4><div class="value">{slope_txt} usuários/mês</div></div>
      <div class="card"><h4>Retenção média m->m+1 (6m)</h4><div class="value">{fmt_pct(retention_avg,1)} dos ativos</div></div>
    </div>

    <div class="note">
      Leitura executiva: atividade mensal de interações apresentou inclinação de {slope_txt} usuários/mês nos 6 meses mais recentes. Último mês com novos usuários: {latest_month} (volume {recent_users}).
    </div>

    <div class="note">
      <b>Inclinação atividade (6m):</b> inclinação da regressão linear dos usuários ativos mensais de interações (MAU) na janela dos 6 meses mais recentes ({activity_window_txt}). Valor negativo ({slope_txt}) significa queda média de {fmt_num(abs(float(slope)),1) if slope is not None and not pd.isna(slope) else 'N/A'} usuários ativos por mês.<br/>
      <b>Retenção média m->m+1 (6m):</b> média de <code>retained_next_month / active_users</code> para os 6 meses mais recentes ({retention_window_txt}) na base de interações registered com match em professores. Interpretação: percentual dos ativos de um mês que volta no mês seguinte.
    </div>

    <h2>0) Tabelas usadas</h2>
    <p>Mede: escopo de dados usados na análise.</p>
    {table_sources}

    <h2>1) Jornada de uso: aula, prova e sequência</h2>
    <p>Mede: leitura integrada da jornada no tempo (uso por solução, atividade mensal e retenção).</p>
    {fig_blocks.get('journey_solution_volume', '')}
    {fig_blocks.get('users_activity_curve', '')}
    {fig_blocks.get('retention_curve', '')}
    {fig_blocks.get('journey_path_mix', '')}
    {activity_time_note_html}
    {fig_blocks.get('activity_weekday_share', '')}
    {fig_blocks.get('activity_hour_share', '')}

    <h2>2) Hipóteses de adoção e uso</h2>
    <p>Hipótese, teste e resultado com premissas explícitas e regra de decisão auditável.</p>
    {fig_blocks.get('hypothesis_status', '')}
    {hyp_html}

    <h2>3) Integridade de identidade, sessões e qualidade dos dados</h2>
    <p>Mede: cobertura de joins e identidade, sinais de sessões curtas e consistência dos dados brutos antes de inferências.</p>
    {fig_blocks.get('join_coverage', '')}
    <p class="small"><b>consistency_status</b>: Checks de integridade dos dados (pass/info/warning/fail), com regras fixas de validade.</p>
    <p class="small">Categorias dos checks: <b>Sessões (uso)</b> valida duração e pings; <b>Interações de conteúdo</b> verifica timestamp, tipo de evento e ordem temporal; <b>Cadastro de professores</b> valida valores de base; <b>Mapeamento de aulas</b> monitora match de id_aula.</p>
    {fig_blocks.get('identity_coverage', '')}
    {fig_blocks.get('session_quality', '')}
    {consistency_table_html}

    <h2>4) Estado e ativação regional</h2>
    <p>Mede: diferenças de conversão entre UFs com suporte amostral mínimo.</p>
    {fig_blocks.get('state_conversion', '')}

    <h2>5) UTM origin: o que significa e como usar</h2>
    <p>Mede: comportamento de conversão e escala por grupo de aquisição.</p>
    {fig_blocks.get('utm_conversion', '')}

    <h2>6) Perfis comportamentais (clusters + heavy users)</h2>
    <p>Mede: perfis de comportamento e diferença de intensidade/conversão entre segmentos.</p>
    {fig_blocks.get('cluster_profiles', '')}
    {fig_blocks.get('heavy_score_distribution', '')}
    {fig_blocks.get('heavy_prevalence_stability', '')}
    {fig_blocks.get('heavy_score_oot_validation', '')}
    {fig_blocks.get('heavy_summary', '')}
    {heavy_profile_note_html}
    {fig_blocks.get('heavy_device_primary', '')}
    {fig_blocks.get('heavy_profile_state_lift', '')}
    {fig_blocks.get('heavy_profile_utm_lift', '')}
    {fig_blocks.get('heavy_profile_tela_lift', '')}
    <p class="small"><b>Nota:</b> tabelas SQL detalhadas de heavy foram removidas desta versão para evitar redundância com os gráficos de perfil.</p>

    <h2>7) Correlações e tendência recente</h2>
    <p>Mede: associações numéricas (Spearman), categóricas (Cramér's V) e tendência da atividade recente.</p>
    <p class="small"><b>activity_trend_windows</b> aplica regressão linear da atividade mensal em janelas de 6, 9 e 12 meses; inclinação negativa indica queda de usuários ativos, positiva indica crescimento.</p>
    {fig_blocks.get('numeric_correlations', '')}
    {fig_blocks.get('categorical_associations', '')}
    {fig_blocks.get('activity_trend_windows', '')}
  </div>
</body>
</html>
"""
    return html


def main() -> None:
    setup_logging()
    cfg = build_config(parse_args())
    paths = ensure_dirs(cfg.output_dir)
    removed_legacy = cleanup_legacy_artifacts(cfg.output_dir)
    if removed_legacy:
        LOGGER.info("Legacy artifacts removed: %s", len(removed_legacy))

    LOGGER.info("Running report build | output_dir=%s", cfg.output_dir)

    consolidated = load_json(cfg.output_dir / "consolidated_status.json")

    raw: Dict[str, pd.DataFrame] = {
        "teacher": load_teacher_dataset(cfg.output_dir),
        "monthly_solution_usage": load_csv(cfg.output_dir / "eda_monthly_solution_usage.csv"),
        "users_monthly_panel": load_csv(cfg.output_dir / "users_monthly_panel.csv"),
        "retention_monthly_entries": load_csv(cfg.output_dir / "retention_monthly_entries.csv"),
        "hypothesis_results": load_csv(cfg.output_dir / "hypothesis_results.csv"),
        "state_stats": load_csv(cfg.output_dir / "state_stats.csv"),
        "utm_stats": load_csv(cfg.output_dir / "utm_stats.csv"),
        "geo_associations": load_csv(cfg.output_dir / "geo_associations.csv"),
        "top_corr_pairs": load_csv(cfg.output_dir / "top_corr_pairs.csv"),
        "cat_corr_pairs": load_csv(cfg.output_dir / "cat_corr_pairs.csv"),
        "journey_path_counts": load_csv(cfg.output_dir / "journey_path_counts.csv"),
        "heavy_summary": load_csv(cfg.output_dir / "heavy_summary.csv"),
        "identity_coverage": load_csv(cfg.output_dir / "identity_coverage.csv"),
        "join_coverage": load_csv(cfg.output_dir / "data_quality_join_coverage.csv"),
        "consistency_checks": load_csv(cfg.output_dir / "data_quality_consistency_checks.csv"),
        "deep_dive_session_quality_by_profile": load_csv(cfg.output_dir / "deep_dive_session_quality_by_profile.csv"),
        "deep_dive_cluster_profiles_detailed": load_csv(cfg.output_dir / "deep_dive_cluster_profiles_detailed.csv"),
        "deep_dive_heavy_summary": load_csv(cfg.output_dir / "deep_dive_heavy_summary.csv"),
        "deep_dive_usage_trend_windows": load_csv(cfg.output_dir / "deep_dive_usage_trend_windows.csv"),
        "heavy_threshold_grid_search": load_csv(cfg.output_dir / "heavy_threshold_grid_search.csv"),
        "heavy_prevalence_monthly": load_csv(cfg.output_dir / "heavy_prevalence_monthly.csv"),
        "heavy_out_of_time_lift": load_csv(cfg.output_dir / "heavy_out_of_time_lift.csv"),
        "heavy_score_decile_diagnostics": load_csv(cfg.output_dir / "heavy_score_decile_diagnostics.csv"),
    }
    raw["heavy_definition_json"] = load_json(cfg.output_dir / "heavy_definition.json")
    raw.update(compute_activity_time_panels(cfg.data_dir, raw.get("teacher", pd.DataFrame())))
    raw.update(compute_heavy_user_device_panels(raw.get("teacher", pd.DataFrame())))
    raw.update(compute_heavy_non_activity_profile_sql(cfg.data_dir, raw.get("teacher", pd.DataFrame())))
    raw["heavy_user_subtypes_summary"] = build_heavy_subtypes_summary(raw)

    summary = consolidated.get("eda", {}).copy()
    if not summary:
        summary = {}

    figs = build_figures(raw, summary)
    html = build_html_report(figs=figs, raw=raw, summary=summary, consolidated=consolidated)

    out_html = paths["reports"] / "analise_inicial_dos_dados_interativa.html"
    out_json = paths["reports"] / "analise_inicial_dos_dados_summary.json"
    out_md = paths["reports"] / "analise_inicial_dos_dados.md"

    out_html.write_text(html, encoding="utf-8")

    summary_payload = {
        "interactive_html": str(out_html),
        "data_bundle_xlsx": str(paths["excel"] / "analise_inicial_dos_dados_bundle.xlsx"),
        "parquet_dir": str(paths["parquet"]),
        "reports_dir": str(paths["reports"]),
        "total_plotly_figures": int(len(figs)),
        "state_missing_pct": summary.get("state_missing_pct"),
        "utm_missing_pct": summary.get("utm_missing_pct"),
        "short_sessions_le_5s": summary.get("short_sessions_le_5s"),
        "short_sessions_rate_le_5s": summary.get("short_sessions_rate_le_5s"),
        "return_gap_median_days": summary.get("return_gap_median_days"),
        "return_gap_heavy_median_days": summary.get("return_gap_heavy_median_days"),
        "return_gap_base_median_days": summary.get("return_gap_base_median_days"),
        "latest_new_users_month": summary.get("latest_new_users_month"),
        "latest_new_users_count": summary.get("latest_new_users_count"),
        "recent_6m_mau_interactions_slope_users_per_month": summary.get("recent_6m_mau_interactions_slope_users_per_month"),
        "retention_recent_avg_6m": summary.get("retention_recent_avg_6m"),
        "causal_claim_allowed": consolidated.get("causal_diagnostic_assessment", {}).get("causal_claim_allowed", False),
        "heavy_definition_json": str(cfg.output_dir / "heavy_definition.json"),
        "heavy_prevalence_monthly_csv": str(cfg.output_dir / "heavy_prevalence_monthly.csv"),
        "heavy_out_of_time_lift_csv": str(cfg.output_dir / "heavy_out_of_time_lift.csv"),
        "heavy_score_decile_diagnostics_csv": str(cfg.output_dir / "heavy_score_decile_diagnostics.csv"),
    }
    out_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out_md.write_text(
        "\n".join(
            [
                "# Análise inicial dos dados",
                "",
                f"- Relatório interativo: `{out_html}`",
                f"- Bundle Excel: `{paths['excel'] / 'analise_inicial_dos_dados_bundle.xlsx'}`",
                f"- Bundle Parquet (pasta): `{paths['parquet']}`",
                f"- Sumário técnico: `{out_json}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    executive_quadro = build_exec_quadro(summary_payload, consolidated)
    executive_quadro.to_csv(cfg.output_dir / "executive_quadro_por_item.csv", index=False)

    # Bundle tables (excel/parquet/csv)
    bundle_tables = {
        "executive_quadro_por_item": executive_quadro,
        "hypothesis_results": raw["hypothesis_results"],
        "state_stats": raw["state_stats"],
        "utm_stats": raw["utm_stats"],
        "geo_associations": raw["geo_associations"],
        "top_corr_pairs": raw["top_corr_pairs"],
        "cat_corr_pairs": raw["cat_corr_pairs"],
        "journey_path_counts": raw["journey_path_counts"],
        "identity_coverage": raw["identity_coverage"],
        "join_coverage": raw["join_coverage"],
        "consistency_checks": raw["consistency_checks"],
        "monthly_solution_usage": raw["monthly_solution_usage"],
        "users_monthly_panel": raw["users_monthly_panel"],
        "retention_monthly_entries": raw["retention_monthly_entries"],
        "deep_dive_session_quality_by_profile": raw["deep_dive_session_quality_by_profile"],
        "deep_dive_cluster_profiles_detailed": raw["deep_dive_cluster_profiles_detailed"],
        "deep_dive_heavy_summary": raw["deep_dive_heavy_summary"],
        "deep_dive_usage_trend_windows": raw["deep_dive_usage_trend_windows"],
        "activity_weekday_segment_share": raw["activity_weekday_segment_share"],
        "activity_hour_segment_share": raw["activity_hour_segment_share"],
        "activity_time_summary": raw["activity_time_summary"],
        "heavy_device_primary": raw["heavy_device_primary"],
        "heavy_interaction_device_reach": raw["heavy_interaction_device_reach"],
        "heavy_interaction_event_share": raw["heavy_interaction_event_share"],
        "heavy_profile_definition": raw["heavy_profile_definition"],
        "heavy_profile_available_characteristics": raw["heavy_profile_available_characteristics"],
        "heavy_profile_state_stats": raw["heavy_profile_state_stats"],
        "heavy_profile_state_enriched": raw["heavy_profile_state_enriched"],
        "heavy_profile_utm_stats": raw["heavy_profile_utm_stats"],
        "heavy_profile_tela_stats": raw["heavy_profile_tela_stats"],
        "heavy_profile_currentstage_stats": raw["heavy_profile_currentstage_stats"],
        "heavy_profile_currentsubject_stats": raw["heavy_profile_currentsubject_stats"],
        "heavy_profile_tipo_total_alunos_stats": raw["heavy_profile_tipo_total_alunos_stats"],
        "heavy_profile_numeric_compare": raw["heavy_profile_numeric_compare"],
        "heavy_profile_login_google_stats": raw["heavy_profile_login_google_stats"],
        "heavy_profile_tenure_compare": raw["heavy_profile_tenure_compare"],
        "heavy_user_subtypes_summary": raw["heavy_user_subtypes_summary"],
        "heavy_threshold_grid_search": raw["heavy_threshold_grid_search"],
        "heavy_prevalence_monthly": raw["heavy_prevalence_monthly"],
        "heavy_out_of_time_lift": raw["heavy_out_of_time_lift"],
        "heavy_score_decile_diagnostics": raw["heavy_score_decile_diagnostics"],
    }
    write_bundles(paths, bundle_tables)

    LOGGER.info("Report build finished successfully. HTML=%s", out_html)


if __name__ == "__main__":
    main()
