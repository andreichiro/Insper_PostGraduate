from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, r2_score

try:
    import plotly.express as px
except Exception:
    px = None


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the HTML report for the Aula_9 targeted ML rebuild.")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def read_table(build_dir: Path, table_name: str) -> pd.DataFrame:
    path = build_dir / "tables" / f"{table_name}.parquet"
    if not path.exists():
        path = build_dir / "parquet" / f"{table_name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def read_summary(build_dir: Path) -> dict[str, Any]:
    path = build_dir / "metadata" / "build_summary_v1.json"
    if not path.exists():
        path = build_dir / "json" / "build_summary_v1.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_serving_manifest(build_dir: Path) -> dict[str, Any]:
    path = build_dir / "serving" / "serving_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_reference_scope(build_dir: Path) -> pd.DataFrame:
    # The analytical report should stay anchored to the heavy-build
    # post-model reference table. Serving manifests may later narrow the
    # scope to a single deployable primary model, which is useful for
    # inference but would incorrectly hide analytical comparisons here.
    reference_scope = read_table(build_dir, "post_model_reference_selection_v1")
    if not reference_scope.empty:
        return reference_scope
    serving_manifest = read_serving_manifest(build_dir)
    rows = serving_manifest.get("reference_scope_rows", [])
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame()


def format_number(value: Any, digits: int = 3) -> Any:
    if pd.isna(value):
        return ""
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if value != 0 and abs(value) < 10 ** (-digits):
            return f"{value:.2e}"
        return round(value, digits)
    return value


def format_percent(value: Any, digits: int = 1) -> str:
    if pd.isna(value):
        return ""
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    return f"{round(numeric * 100, digits)}%"


def format_model_name(value: Any) -> str:
    mapping = {
        "logistic_regression": "Regressão logística",
        "random_forest": "Random Forest",
        "catboost": "CatBoost",
    }
    return mapping.get(str(value), str(value))


def format_track_name(value: Any) -> str:
    mapping = {
        "S1": "S1",
        "S7": "S7",
        "S1_PLUS_S7": "S1+S7",
        "STRICT_CONTEXT": "STRICT_CONTEXT",
    }
    return mapping.get(str(value), str(value))


def format_definition_name(value: Any) -> str:
    text = str(value)
    if text.startswith("definition_a::"):
        return "Definição A"
    if text.startswith("definition_b"):
        return "Definição B"
    mapping = {
        "definition_a": "Definição A",
        "definition_b": "Definição B",
        "definition_b_label": "Definição B",
    }
    return mapping.get(text, text)


def format_metric_name(value: Any) -> str:
    mapping = {
        "future_active_days": "dias ativos futuros",
        "future_activity_events": "eventos de atividade futuros",
        "future_business_active_weeks": "semanas ativas de negócio futuras",
        "future_content_views": "views futuras de conteúdo",
        "future_distinct_actions": "diversidade futura de ações",
        "future_downloads": "downloads futuros",
        "future_formation_events": "eventos futuros de formação",
        "future_interactions": "interações futuras",
        "future_mapped_lessons": "aulas mapeadas futuras",
        "future_session_minutes": "minutos futuros em sessão",
        "future_sessions": "sessões futuras",
    }
    text = str(value)
    return mapping.get(text, text)


def format_rule_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    for raw_name in sorted(
        [
            "future_business_active_weeks",
            "future_distinct_actions",
            "future_session_minutes",
            "future_activity_events",
            "future_content_views",
            "future_mapped_lessons",
            "future_formation_events",
            "future_active_days",
            "future_interactions",
            "future_downloads",
            "future_sessions",
        ],
        key=len,
        reverse=True,
    ):
        text = re.sub(rf"\b{re.escape(raw_name)}\b", format_metric_name(raw_name), text)
    text = text.replace(" AND ", " e ")
    text = text.replace(" OR ", " ou ")
    return text


def format_problem_key(value: Any) -> str:
    text = str(value)
    if text.startswith("definition_a::"):
        definition_part, _, track_part = text.partition("__")
        rule_text = format_rule_text(definition_part.replace("definition_a::", ""))
        track_label = format_track_name(track_part) if track_part else ""
        return f"Definição A ({rule_text}) | {track_label}" if track_label else f"Definição A ({rule_text})"
    if text.startswith("definition_b_label__"):
        track_label = format_track_name(text.replace("definition_b_label__", ""))
        return f"Definição B | {track_label}" if track_label else "Definição B"
    if text.startswith("definition_b__"):
        track_label = format_track_name(text.replace("definition_b__", ""))
        return f"Definição B | {track_label}" if track_label else "Definição B"
    if text == "definition_b_label":
        return "Definição B"
    if text == "definition_b":
        return "Definição B"
    text = text.replace("__S1_PLUS_S7", " | S1+S7")
    text = text.replace("__STRICT_CONTEXT", " | STRICT_CONTEXT")
    text = text.replace("__S7", " | S7")
    text = text.replace("__S1", " | S1")
    return text


def format_official_status(value: Any) -> str:
    mapping = {
        "official_admissible": "admissível na fronteira oficial",
        "official_fixed_literal": "comparador literal fixo",
        "official_unique": "definição oficial única",
        "official_winner": "vencedor oficial",
    }
    return mapping.get(str(value), str(value))


def format_selection_basis(value: Any) -> str:
    mapping = {
        "per_metric_out_of_sample_rank_aggregation_then_metric_pareto_front": "agregação por desempenho fora da amostra + fronteira de Pareto por métrica",
        "literal_comparator_fixed_a_priori": "regra literal fixada a priori",
    }
    return mapping.get(str(value), str(value))


def format_policy_name(value: Any) -> str:
    mapping = {
        "top_10_percent": "top 10% do risk_score",
        "tercis": "tercis do risk_score",
        "score_ge_0_70": "risk_score >= 0,70",
        "heavy_top_10_percent": "top 10% do heavy_intensity_score",
        "heavy_top_20_percent": "top 20% do heavy_intensity_score",
        "precision_recall_f1_by_cutoff": "métricas por cutoff",
        "r2_and_mape_on_monthly_realized_risk": "ajuste mensal do risco realizado",
        "permutation_importance_neg_brier": "importância por permutação (Brier)",
        "kmeans_cluster_ready_grid": "KMeans com grade de k",
        "definition_b_structural_leakage_audit": "auditoria estrutural da Definition B",
        "definition_b_feature_block_gain_test": "teste de ganho por bloco da Definition B",
        "definition_b_excessive_separation_red_flag": "red flag de separação excessiva da Definition B",
    }
    return mapping.get(str(value), str(value))


def format_policy_group(value: Any) -> str:
    mapping = {
        "feature_selection": "seleção de variáveis",
        "risk_band_policy": "bandas de risco",
        "heavy_user_policy": "heavy-user",
        "threshold_metrics": "cutoff operacional",
        "monthly_rate_metrics": "métricas mensais agregadas",
        "feature_inspection": "inspeção de sinais",
        "cluster_policy": "cluster",
        "leakage_audit": "diagnóstico de leakage",
    }
    return mapping.get(str(value), str(value))


def format_selection_reason(value: Any) -> str:
    text = str(value)
    if text.startswith("serving_primary::"):
        parts = text.split("::")
        if len(parts) >= 4:
            return (
                f"modelo servível primário da {format_definition_name(parts[2])} "
                "após desempate por erro probabilístico, variabilidade e informação disponível"
            )
        return "modelo servível primário após regra formal de desempate"
    if text.startswith("best_probability_first_within_definition::"):
        definition_name = text.split("::", 1)[1]
        return f"combinação oficial publicada da {format_definition_name(definition_name)}"
    mapping = {
        "best_probability_first_within_track::S1": "melhor erro probabilístico dentro da trilha S1",
        "best_probability_first_within_track::S7": "melhor erro probabilístico dentro da trilha S7",
        "best_probability_first_within_track::S1_PLUS_S7": "melhor erro probabilístico dentro da trilha S1+S7",
        "best_probability_first_within_track::STRICT_CONTEXT": "melhor erro probabilístico dentro da trilha STRICT_CONTEXT",
        "best_probability_first_overall": "melhor erro probabilístico no conjunto todo",
        "best_ap_first_overall": "melhor AP no conjunto todo",
    }
    return mapping.get(text, text)


def format_cluster_name(value: Any) -> str:
    text = str(value)
    if text.startswith("cluster_"):
        suffix = text.split("_", 1)[1]
        return f"Grupo {suffix}"
    return text


def format_event_token(value: Any) -> str:
    mapping = {
        "download": "download",
        "view": "visualização",
        "share": "compartilhamento",
        "navigation": "navegação",
        "access": "acesso",
        "create": "criação",
        "submit": "envio",
        "progress": "progresso",
        "message": "mensagem",
        "missing": "sem ação observável",
    }
    text = str(value)
    return mapping.get(text, text.replace("_", " "))


def format_navigation_sequence(value: Any) -> str:
    text = str(value)
    if text == "missing":
        return "sem ação observável"
    return " → ".join(format_event_token(token) for token in text.split(">"))


def format_feature_name(value: Any) -> str:
    mapping = {
        "months_after_entry": "meses desde a entrada observada",
        "teacher_population_status": "status da população do professor",
        "utm_group": "grupo de origem/UTM",
        "first_session_entry_surface": "superfície de entrada da 1ª sessão",
        "first_session_device_bucket": "bucket de device da 1ª sessão",
        "first_event_missing_flag": "falta do 1º evento observado",
        "first_device_missing_flag": "falta do device observado",
        "first_utm_missing_flag": "falta da origem/UTM observada",
        "session_without_interaction_flag": "sessão sem interação",
        "first_session_duration_min": "minutos da 1ª sessão",
        "first_session_interactions": "interações na 1ª sessão",
        "first_session_downloads": "downloads na 1ª sessão",
        "first_session_views": "visualizações na 1ª sessão",
        "first_session_other_actions": "outras ações na 1ª sessão",
        "first_session_navigation_events": "eventos de navegação na 1ª sessão",
        "first_session_meaningful_events": "eventos úteis na 1ª sessão",
        "first_session_has_interaction_flag": "houve interação na 1ª sessão",
        "first_session_has_meaningful_action_flag": "houve ação útil na 1ª sessão",
        "secs_to_first_interaction": "segundos até a 1ª interação",
        "secs_to_first_meaningful_action": "segundos até a 1ª ação útil",
        "first_session_first_event_action_group": "tipo do 1º evento da 1ª sessão",
        "first_session_first_meaningful_action_group": "tipo da 1ª ação útil da 1ª sessão",
        "first_session_exit_state": "estado de saída da 1ª sessão",
        "first7d_events": "eventos nos 7 primeiros dias",
        "first7d_active_days": "dias ativos nos 7 primeiros dias",
        "first7d_sessions": "sessões nos 7 primeiros dias",
        "first7d_session_minutes": "minutos de sessão nos 7 primeiros dias",
        "first3_interaction_downloads": "downloads nas 3 primeiras interações observáveis",
        "first3_interaction_views": "visualizações nas 3 primeiras interações observáveis",
        "first3_interaction_other_actions": "outras ações nas 3 primeiras interações observáveis",
        "heavy_intensity_score": "score contínuo de intensidade",
        "teacher_active_months_total": "meses ativos históricos do professor",
        "teacher_strict_months_total": "meses estritos históricos do professor",
        "teacher_active_month_share": "share histórico de meses ativos",
        "teacher_strict_month_share": "share histórico de meses estritos",
        "avg_activity_events_active_month": "média histórica de eventos de atividade por mês ativo",
        "std_activity_events_active_month": "desvio histórico de eventos de atividade por mês ativo",
        "avg_active_days_active_month": "média histórica de dias ativos por mês ativo",
        "std_active_days_active_month": "desvio histórico de dias ativos por mês ativo",
        "avg_strict_downloads_active_month": "média histórica de downloads estritos por mês ativo",
        "avg_downloads_active_month": "média histórica de downloads por mês ativo",
        "avg_content_views_active_month": "média histórica de visualizações por mês ativo",
        "avg_other_actions_active_month": "média histórica de outras ações por mês ativo",
        "avg_aula_events_active_month": "média histórica de eventos de aula por mês ativo",
        "avg_plano_events_active_month": "média histórica de eventos de plano por mês ativo",
        "avg_prova_events_active_month": "média histórica de eventos de prova por mês ativo",
        "avg_ia_events_active_month": "média histórica de eventos de IA por mês ativo",
        "avg_mapped_lessons_active_month": "média histórica de aulas mapeadas por mês ativo",
        "avg_clean_entry_sessions_active_month": "média histórica de sessões limpas por mês ativo",
        "avg_clean_entry_minutes_active_month": "média histórica de minutos limpos por mês ativo",
        "teacher_mobile_month_share": "share histórico de meses com uso mobile",
        "teacher_desktop_month_share": "share histórico de meses com uso desktop",
    }
    text = str(value)
    return mapping.get(text, text.replace("_", " "))


def describe_strict_context_feature(feature_name: str) -> str:
    mapping = {
        "months_after_entry": "Quanto tempo se passou entre a entrada do professor e o mês observado na base.",
        "teacher_population_status": "Situação básica do professor na população analisada, já conhecida no momento de entrada.",
        "utm_group": "Canal de aquisição agrupado do cadastro ou da primeira origem identificada.",
        "first_session_entry_surface": "Superfície ou origem da primeira entrada observada na plataforma.",
        "first_session_device_bucket": "Tipo de dispositivo da primeira entrada observada.",
        "first_event_missing_flag": "Flag que indica ausência do primeiro evento registrado.",
        "first_device_missing_flag": "Flag que indica ausência de informação de dispositivo no registro inicial.",
        "first_utm_missing_flag": "Flag que indica ausência de origem UTM no registro inicial.",
        "session_without_interaction_flag": "Flag que indica que a 1ª sessão observada não teve interação capturada.",
    }
    return mapping.get(feature_name, "")


def format_external_validator_name(value: Any) -> str:
    mapping = {
        "returned_active_post_label_m1": "retorno ativo no 1º bloco pós-label",
        "returned_active_post_label_m2": "retorno ativo no 2º bloco pós-label",
        "returned_active_post_label_m3": "retorno ativo no 3º bloco pós-label",
        "active_days_post_label_3m": "dias ativos acumulados nos 3 blocos",
        "sustained_active_2of3_post_label": "sustentação em 2 de 3 blocos",
    }
    text = str(value)
    return mapping.get(text, text.replace("_", " "))


def confusion_cell_code(actual_group: Any, predicted_group: Any) -> str:
    mapping = {
        ("nao_realiza", "nao_realiza"): "TP",
        ("realiza", "nao_realiza"): "FP",
        ("realiza", "realiza"): "TN",
        ("nao_realiza", "realiza"): "FN",
    }
    return mapping.get((str(actual_group), str(predicted_group)), f"{actual_group}->{predicted_group}")


def render_details(title: str, inner_html: str, open_by_default: bool = False) -> str:
    open_attr = " open" if open_by_default else ""
    return f"<details class='detail-card'{open_attr}><summary>{title}</summary>{inner_html}</details>"


def format_arbitrary_type(value: Any) -> str:
    mapping = {
        "arbitrary_required": "arbitrário explicitado",
        "mechanical": "mecânico",
        "business_input": "insumo de negócio",
        "to_remove": "a remover",
    }
    return mapping.get(str(value), str(value))


def format_arbitrary_status(value: Any) -> str:
    mapping = {
        "kept": "mantido",
        "removed": "removido",
        "pending": "pendente",
    }
    return mapping.get(str(value), str(value))


def format_where_used(value: Any) -> str:
    mapping = {
        "Definition A, Definition B, external validators": "Definições A e B e validadores externos",
        "external validators": "validadores externos",
        "definition diagnostics and prediction confidence intervals": "diagnóstico das definições e intervalos de confiança das predições",
        "ExpandingMonthSplit": "splitter temporal ExpandingMonthSplit",
        "definition search and outer model backtest": "busca de definição e backtest externo dos modelos",
        "definition comparison and official model frontier": "comparação entre definições e fronteira oficial de modelos",
        "CalibratedClassifierCV official path": "caminho oficial de calibração com CalibratedClassifierCV",
        "official model comparison": "comparação oficial entre famílias de modelo",
        "random_forest and catboost official path": "caminho oficial de Random Forest e CatBoost",
        "published descriptive cluster layer": "camada descritiva de cluster publicada",
        "published descriptive cluster validation": "validação descritiva de cluster publicada",
        "published heavy-user layer": "camada publicada de heavy-user",
        "published operational overlay": "camada operacional publicada",
    }
    return mapping.get(str(value), str(value))


def format_arbitrary_name(value: Any) -> str:
    mapping = {
        "label_window_days": "janela principal do label (dias)",
        "post_label_block_days": "blocos pós-label (dias)",
        "bootstrap_iterations": "iterações de bootstrap",
        "temporal_splitter_test_periods": "meses por fold de teste",
        "temporal_splitter_max_outer_test_months": "máximo de meses no outer backtest",
        "minimum_valid_outer_folds_for_official_summary": "mínimo de outer folds válidos para resumo oficial",
        "minimum_test_rows_for_official_fold": "mínimo de linhas de teste por fold oficial",
        "minimum_test_positives_for_official_fold": "mínimo de positivos por fold oficial",
        "minimum_test_negatives_for_official_fold": "mínimo de negativos por fold oficial",
        "light_temporal_tuning_enabled": "tuning temporal leve ativado",
        "light_temporal_tuning_iterations": "iterações da busca temporal leve",
        "light_temporal_tuning_max_inner_splits": "máximo de inner splits do tuning leve",
        "light_temporal_tuning_scoring": "métrica-guia do tuning leve",
        "calibration_method": "método de calibração",
        "model_family_scope": "escopo de famílias de modelo comparadas",
        "estimator_random_seed": "seed dos estimadores",
        "estimator_parallel_workers": "paralelismo dos estimadores",
        "cluster_k_candidate_grid": "grade candidata de k para cluster",
        "cluster_bootstrap_iterations": "iterações bootstrap de cluster",
        "cluster_sample_size": "amostra máxima da validação de cluster",
        "heavy_user_pca_proxy": "proxy de heavy-user via PCA",
        "registered_cutoff_and_band_policies": "políticas registradas de cutoff e bandas",
    }
    return mapping.get(str(value), str(value))


def format_arbitrary_why(value: Any) -> str:
    mapping = {
        "A binary future label needs a finite horizon. The official build uses a fixed 30-day label window and surfaces it explicitly as arbitrary.": "Um label binário futuro precisa de horizonte finito. O build oficial usa janela fixa de 30 dias e deixa isso explícito como arbitrário.",
        "Post-label validators are measured in three consecutive 30-day blocks after the label window.": "Os validadores pós-label são medidos em três blocos consecutivos de 30 dias depois da janela principal do label.",
        "Bootstrap needs a finite number of resamples. The value is surfaced explicitly instead of being hidden in the code.": "Bootstrap precisa de número finito de reamostragens. O valor aparece explicitamente no relatório em vez de ficar escondido no código.",
        "The splitter evaluates one unique month at a time to preserve month boundaries in the panel.": "O splitter avalia um mês por vez para preservar os limites mensais do painel.",
        "The published build limits the expanding outer backtest to the last 5 test months to keep the exact-threshold search and calibrated model comparison computationally feasible. This is surfaced explicitly instead of being hidden.": "O build publicado limita o outer backtest expansivo aos últimos 5 meses de teste para manter viável a busca exata de thresholds e a comparação calibrada de modelos. Isso aparece explicitamente no relatório.",
        "The official build requires at least two valid outer folds before publishing a mean and dispersion across folds. Single-fold results remain diagnostic only.": "O build oficial exige pelo menos dois outer folds válidos antes de publicar média e dispersão entre folds. Resultado de um único fold fica só como diagnóstico.",
        "The official path follows the library-guided default for smaller calibration samples and preserves monotonic ranking.": "O caminho oficial segue a escolha guiada pela biblioteca para amostras menores de calibração e preserva o ranking monotônico.",
        "The official comparison is intentionally limited to the three requested model families and does not claim that this scope is exhaustive.": "A comparação oficial fica intencionalmente limitada às três famílias de modelo pedidas e não afirma que esse escopo é exaustivo.",
        "The official path fixes the estimator seed for reproducibility across reruns.": "O caminho oficial fixa a seed dos estimadores para garantir reprodutibilidade entre reruns.",
        "The official path compares the requested model families in parallel threads while constraining estimator-level parallelism to avoid nested oversubscription during calibrated temporal backtests.": "O caminho oficial compara as famílias pedidas em threads paralelas e limita o paralelismo interno dos estimadores para evitar oversubscription nos backtests calibrados.",
    }
    return mapping.get(str(value), str(value))


def render_clean_table(df: pd.DataFrame, limit: int | None = None) -> str:
    if df.empty:
        return "<p class='section-text'>Sem linhas materializadas.</p>"
    show = df.head(limit).copy() if limit is not None else df.copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: format_number(x, 3))
    return show.to_html(index=False, classes="clean-table", border=0, escape=False)


def render_intro_table(rows: list[tuple[str, str]]) -> str:
    intro = pd.DataFrame(rows, columns=["Bloco", "Definição"])
    return render_clean_table(intro)


def render_plotly(df: pd.DataFrame, kind: str) -> str:
    if px is None or df.empty:
        return ""
    if kind == "definition":
        plot_df = df.copy()
        value_cols = [col for col in plot_df.columns if col.startswith("test_gap_")]
        if not value_cols:
            return ""
        plot_df = plot_df.melt(id_vars=["definition_name"], value_vars=value_cols, var_name="validator", value_name="gap")
        plot_df["definition_name"] = plot_df["definition_name"].map(format_definition_name)
        plot_df["validator"] = plot_df["validator"].str.replace("test_gap_", "", regex=False).map(format_external_validator_name)
        fig = px.bar(
            plot_df,
            x="validator",
            y="gap",
            color="definition_name",
            barmode="group",
            title="Validadores externos por definição",
        )
    elif kind == "models":
        plot_df = df.copy()
        if "pareto_frontier_flag" in plot_df.columns:
            plot_df = plot_df[plot_df["pareto_frontier_flag"] == 1].copy()
        if plot_df.empty:
            return ""
        plot_df["track_name"] = plot_df["track_name"].map(format_track_name)
        plot_df["model_name"] = plot_df["model_name"].map(format_model_name)
        plot_df["problem_key"] = plot_df["problem_key"].map(format_problem_key)
        fig = px.scatter(
            plot_df,
            x="mean_brier",
            y="mean_ap",
            color="track_name",
            symbol="model_name",
            hover_name="problem_key",
            title="Fronteira oficial de score no teste futuro concatenado",
        )
    elif kind == "cv_score_drift":
        plot_df = df.copy()
        if plot_df.empty:
            return ""
        plot_df["problem_key"] = plot_df["problem_key"].map(format_problem_key)
        plot_df["model_name"] = plot_df["model_name"].map(format_model_name)
        plot_df["serie"] = plot_df["problem_key"] + " | " + plot_df["model_name"]
        top_series = plot_df["serie"].value_counts().head(8).index.tolist()
        plot_df = plot_df[plot_df["serie"].isin(top_series)].copy()
        fig = px.line(
            plot_df,
            x="fold_id",
            y="mean_risk_score",
            color="serie",
            markers=True,
            title="Como o risk score médio variou entre outer folds",
        )
    elif kind == "cv_metric_drift":
        plot_df = df.copy()
        if plot_df.empty:
            return ""
        plot_df = plot_df[plot_df["metric_name"].isin(["ap", "brier"])].copy()
        plot_df["problem_key"] = plot_df["problem_key"].map(format_problem_key)
        plot_df["model_name"] = plot_df["model_name"].map(format_model_name)
        plot_df["metric_name"] = plot_df["metric_name"].map({"ap": "AP", "brier": "Brier"})
        plot_df["serie"] = plot_df["problem_key"] + " | " + plot_df["model_name"]
        top_series = plot_df["serie"].value_counts().head(6).index.tolist()
        plot_df = plot_df[plot_df["serie"].isin(top_series)].copy()
        fig = px.line(
            plot_df,
            x="fold_id",
            y="metric_value",
            color="serie",
            facet_row="metric_name",
            markers=True,
            title="Como AP e Brier variaram entre outer folds",
        )
    elif kind == "cv_threshold_drift":
        plot_df = df.copy()
        if plot_df.empty:
            return ""
        plot_df = plot_df[plot_df["policy_name"].isin(["top_10_percent", "score_ge_0_70"])].copy()
        plot_df["problem_key"] = plot_df["problem_key"].map(format_problem_key)
        plot_df["model_name"] = plot_df["model_name"].map(format_model_name)
        plot_df["policy_name"] = plot_df["policy_name"].map(format_policy_name)
        plot_df["serie"] = plot_df["problem_key"] + " | " + plot_df["model_name"] + " | " + plot_df["policy_name"]
        top_series = plot_df["serie"].value_counts().head(6).index.tolist()
        plot_df = plot_df[plot_df["serie"].isin(top_series)].copy()
        fig = px.line(
            plot_df,
            x="fold_id",
            y="f1",
            color="serie",
            markers=True,
            title="Como o F1 variou entre folds sob políticas de cutoff",
        )
    elif kind == "cv_confusion_drift":
        plot_df = df.copy()
        if plot_df.empty:
            return ""
        plot_df = plot_df[plot_df["policy_name"].isin(["top_10_percent", "score_ge_0_70"])].copy()
        plot_df["problem_key"] = plot_df["problem_key"].map(format_problem_key)
        plot_df["model_name"] = plot_df["model_name"].map(format_model_name)
        plot_df["policy_name"] = plot_df["policy_name"].map(format_policy_name)
        plot_df["cell_name"] = plot_df.apply(lambda row: confusion_cell_code(row["actual_group"], row["predicted_group"]), axis=1)
        plot_df["serie"] = plot_df["problem_key"] + " | " + plot_df["model_name"] + " | " + plot_df["policy_name"] + " | " + plot_df["cell_name"]
        top_series = (
            plot_df.groupby("serie", as_index=False)["rows"]
            .sum()
            .sort_values("rows", ascending=False)
            .head(8)["serie"]
            .tolist()
        )
        plot_df = plot_df[plot_df["serie"].isin(top_series)].copy()
        fig = px.line(
            plot_df,
            x="fold_id",
            y="rows",
            color="serie",
            markers=True,
            title="Como TP, FP, TN e FN variaram entre folds",
        )
    elif kind == "feature_importance":
        plot_df = (
            df.groupby(["problem_key", "model_name", "feature_name"], as_index=False)
            .agg(importance_mean=("importance_mean", "mean"))
            .copy()
        )
        if plot_df.empty:
            return ""
        plot_df["importance_abs"] = plot_df["importance_mean"].abs()
        plot_df["serie"] = plot_df["problem_key"].map(format_problem_key) + " | " + plot_df["model_name"].map(format_model_name)
        plot_df = (
            plot_df.sort_values(["serie", "importance_abs"], ascending=[True, False])
            .groupby("serie", group_keys=False)
            .head(8)
            .copy()
        )
        plot_df["feature_name"] = plot_df["feature_name"].map(format_feature_name)
        fig = px.bar(
            plot_df.sort_values(["serie", "importance_abs"], ascending=[True, True]),
            x="importance_mean",
            y="feature_name",
            color="serie",
            orientation="h",
            facet_col="serie",
            title="Sinais com maior impacto por permutação",
        )
    elif kind == "band_summary":
        plot_df = df.copy()
        if plot_df.empty:
            return ""
        plot_df["serie"] = plot_df["problem_key"].map(format_problem_key) + " | " + plot_df["model_name"].map(format_model_name)
        fig = px.bar(
            plot_df,
            x="band_name",
            y="share",
            color="band_name",
            facet_col="serie",
            barmode="group",
            title="Share das faixas de risco por política registrada",
        )
    elif kind == "monthly_fit":
        plot_df = df.copy()
        if plot_df.empty:
            return ""
        plot_df["serie"] = plot_df["problem_key"].map(format_problem_key) + " | " + plot_df["model_name"].map(format_model_name)
        plot_df = plot_df.melt(
            id_vars=["serie"],
            value_vars=["monthly_r2", "monthly_mape_positive_months"],
            var_name="metric_name",
            value_name="metric_value",
        )
        plot_df["metric_name"] = plot_df["metric_name"].map(
            {
                "monthly_r2": "R2 mensal",
                "monthly_mape_positive_months": "MAPE mensal",
            }
        )
        fig = px.bar(
            plot_df,
            x="serie",
            y="metric_value",
            color="metric_name",
            barmode="group",
            title="Ajuste mensal agregado do risk_score",
        )
    else:
        return ""
    default_height = 520
    if kind in {"feature_importance", "cv_metric_drift", "cv_threshold_drift", "cv_confusion_drift"}:
        default_height = 620
    if kind in {"band_summary", "monthly_fit"}:
        default_height = 560
    fig.update_layout(height=default_height, margin=dict(l=40, r=30, t=70, b=40))
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _best_model_row_for_definition(
    model_frontier: pd.DataFrame,
    definition_prefix: str,
    *,
    ranking_priority: bool,
) -> pd.Series | None:
    frontier_only = _frontier_only(model_frontier)
    if frontier_only.empty or "definition_name" not in frontier_only.columns:
        return None
    subset = frontier_only[
        frontier_only["definition_name"].astype(str).str.startswith(definition_prefix)
    ].copy()
    if subset.empty:
        return None
    if ranking_priority:
        subset = subset.sort_values(
            ["mean_ap", "mean_roc_auc", "mean_brier", "mean_log_loss"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
    else:
        subset = subset.sort_values(
            ["mean_brier", "mean_log_loss", "mean_ap", "mean_roc_auc"],
            ascending=[True, True, False, False],
            kind="mergesort",
        )
    return subset.iloc[0]


def build_problem_level_model_text(
    selected_problem_model_comparison: pd.DataFrame,
    selected_problem_operational_comparison: pd.DataFrame,
    reference_scope: pd.DataFrame,
) -> str:
    if (
        selected_problem_model_comparison.empty
        or selected_problem_operational_comparison.empty
        or reference_scope.empty
    ):
        return ""
    paragraphs: list[str] = []
    for _, scope_row in reference_scope.iterrows():
        problem_label = format_problem_key(scope_row.get("problem_key", ""))
        selected_model_label = format_model_name(scope_row.get("model_name", ""))
        metric_rows = selected_problem_model_comparison[
            selected_problem_model_comparison["Problema"] == problem_label
        ].copy()
        operational_rows = selected_problem_operational_comparison[
            selected_problem_operational_comparison["Problema"] == problem_label
        ].copy()
        if metric_rows.empty:
            continue
        selected_metric_row = metric_rows[metric_rows["Modelo"] == selected_model_label]
        if selected_metric_row.empty:
            continue
        selected_metric_row = selected_metric_row.iloc[0]
        ap_winner = metric_rows.sort_values(["AP", "ROC AUC"], ascending=[False, False]).iloc[0]
        prob_winner = metric_rows.sort_values(["Brier", "Log loss"], ascending=[True, True]).iloc[0]
        best_metric_names: list[str] = []
        if ap_winner["Modelo"] == selected_model_label:
            best_metric_names.extend(["AP", "ROC AUC"])
        if prob_winner["Modelo"] == selected_model_label:
            best_metric_names.extend(["Brier", "log loss"])
        best_metric_names = list(dict.fromkeys(best_metric_names))
        if operational_rows.empty:
            paragraphs.append(
                f"Em <code>{problem_label}</code>, <b>{selected_model_label}</b> foi o modelo publicado. "
                f"Na comparação direta com os outros modelos da mesma combinação, ele liderou em "
                f"{', '.join(best_metric_names) if best_metric_names else 'parte do núcleo probabilístico'}."
            )
            continue
        selected_operational_row = operational_rows[
            operational_rows["Modelo"] == selected_model_label
        ]
        if selected_operational_row.empty:
            paragraphs.append(
                f"Em <code>{problem_label}</code>, <b>{selected_model_label}</b> foi o modelo publicado. "
                f"Na comparação direta com os outros modelos da mesma combinação, ele liderou em "
                f"{', '.join(best_metric_names) if best_metric_names else 'parte do núcleo probabilístico'}."
            )
            continue
        selected_operational_row = selected_operational_row.iloc[0]
        monthly_r2_winner = operational_rows.sort_values(
            ["R2 mensal do risk_score", "MAPE mensal do risk_score"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[0]
        if monthly_r2_winner["Modelo"] == selected_model_label:
            monthly_text = (
                "No agregado mensal, ele também ficou na frente em <code>R2</code>/<code>MAPE</code> "
                "do <code>risk_score</code>."
            )
        else:
            monthly_text = (
                f"No agregado mensal, quem ficou melhor em <code>R2</code>/<code>MAPE</code> "
                f"foi <b>{monthly_r2_winner['Modelo']}</b>."
            )
        metric_text = ", ".join(best_metric_names) if best_metric_names else "parte do núcleo probabilístico"
        paragraphs.append(
            f"Em <code>{problem_label}</code>, <b>{selected_model_label}</b> foi o modelo publicado porque "
            f"liderou no núcleo probabilístico da própria combinação, ficando à frente em {metric_text}. "
            f"{monthly_text}"
        )
    return " ".join(paragraphs)


def build_intro_rows(
    summary: dict[str, Any],
    track_registry: pd.DataFrame,
    definition_frontier: pd.DataFrame,
    model_frontier: pd.DataFrame,
) -> list[tuple[str, str]]:
    official_definition = "Definição A oficial + Definição B comparadora"
    if not definition_frontier.empty:
        bits: list[str] = []
        a_rows = definition_frontier[definition_frontier["definition_name"].astype(str).str.startswith("definition_a")].copy()
        b_rows = definition_frontier[definition_frontier["definition_name"].astype(str).str.startswith("definition_b")].copy()
        if not a_rows.empty:
            bits.append(f"Definição A oficial: <code>{format_rule_text(a_rows.iloc[0].get('rule_text', ''))}</code>")
        if not b_rows.empty:
            bits.append(f"Definição B comparadora: <code>{format_rule_text(b_rows.iloc[0].get('rule_text', ''))}</code>")
        if bits:
            official_definition = " | ".join(bits)

    frontier_only = _frontier_only(model_frontier)
    score_text = "O build oficial ainda não materializou score publicável."
    if not frontier_only.empty:
        best_a_row = _best_model_row_for_definition(
            model_frontier, "definition_a", ranking_priority=False
        )
        best_b_row = _best_model_row_for_definition(
            model_frontier, "definition_b", ranking_priority=True
        )
        text_bits = [
            "O output oficial é probabilidade calibrada contínua. "
            "A leitura principal não compara famílias de modelo em problemas diferentes; "
            "ela fixa uma combinação final para a Definição A oficial e outra para a Definição B comparadora."
        ]
        if best_a_row is not None:
            text_bits.append(
                f"Na Definição A oficial, a combinação probabilística publicada ficou em "
                f"<code>{format_problem_key(best_a_row['problem_key'])}</code> com "
                f"<code>{format_model_name(best_a_row['model_name'])}</code> "
                f"(AP {format_number(best_a_row['mean_ap'])}, ROC AUC {format_number(best_a_row['mean_roc_auc'])}, "
                f"Brier {format_number(best_a_row['mean_brier'])}, log loss {format_number(best_a_row['mean_log_loss'])})."
            )
        if best_b_row is not None:
            text_bits.append(
                f"Na Definição B comparadora, a combinação mais forte em ranking ficou em "
                f"<code>{format_problem_key(best_b_row['problem_key'])}</code> com "
                f"<code>{format_model_name(best_b_row['model_name'])}</code> "
                f"(AP {format_number(best_b_row['mean_ap'])}, ROC AUC {format_number(best_b_row['mean_roc_auc'])})."
            )
        score_text = " ".join(text_bits)

    track_rows = []
    if not track_registry.empty:
        for row in track_registry.to_dict(orient="records"):
            track_rows.append(f"<code>{format_track_name(row['track_name'])}</code>: {row['score_moment_text']}")
    track_text = " ".join(track_rows) if track_rows else "Sem trilhas materializadas."

    rows = [
        (
            "Pergunta de negócio",
            "Dado o que o professor fez no começo da jornada, qual é a chance de ele voltar e mostrar atividade futura observável depois disso?",
        ),
        (
            "Por que isso importa",
            "Essa resposta ajuda a priorizar onboarding, acompanhamento comercial, suporte e leitura de risco. A utilidade não está só em prever; está em saber <b>quem merece atenção primeiro</b> quando o time não consegue olhar todos os casos ao mesmo tempo.",
        ),
        (
            "Por que confiar",
            "O build treina e testa sempre em meses diferentes, faz tuning temporal leve dentro do treino, calibra a probabilidade em um bloco temporal separado, invalida folds com suporte fraco e publica auditoria de leakage. Ou seja: a resposta não vem de reusar o mesmo mês para tudo, nem de misturar entrada com resultado futuro, nem de deixar um fold minúsculo mandar no resumo principal.",
        ),
        (
            "O que foi feito",
            "A base modelada oficial foi usada para reconstruir o começo da jornada do professor, separar a 1ª sessão, o 1º evento e a janela inicial de 7 dias. A partir daí, o build compara definições futuras de atividade, roda uma busca leve de hiperparâmetros por família em validação temporal, calibra a probabilidade de realização futura e publica, na mesma trilha oficial, camadas complementares de cutoff, bandas, heavy-user, cluster, navegação e robustez temporal por fold com política registrada.",
        ),
        (
            "Base raw e base modelada",
            "<b>Base raw</b> = os logs e tabelas originais de origem. <b>Base modelada</b> = a versão limpa e organizada em fatos e dimensões, usada como fonte oficial deste relatório. Aqui, o ML não lê direto da raw; ele lê a modelada oficial e constrói marts específicos para onboarding, jornada inicial e métricas futuras.",
        ),
        (
            "O que está sendo previsto aqui",
            "Previsão de atividade futura de professores. A predição acontece no fim da 1ª sessão ou no fim dos primeiros 7 dias, dependendo da trilha. O resultado futuro começa no <code>day_8</code>, isto é, no primeiro instante depois de completar 7 dias desde a âncora de onboarding.",
        ),
        (
            "Onboarding",
            "Onboarding é o primeiro mês com uso observável na base modelada. Aqui, isso significa pelo menos uma sessão em <code>fct_session_clean</code> ou pelo menos uma interação em <code>fct_interaction_clean</code> no <code>first_month</code>.",
        ),
        (
            "Sessão e interação",
            "<b>Sessão</b> = um bloco contínuo de uso já consolidado em <code>fct_session_clean</code>, com início e fim. <b>Interação</b> = uma ação individual com timestamp em <code>fct_interaction_clean</code>, como visualizar, baixar, navegar ou executar outra atividade.",
        ),
        (
            "Primeira sessão, primeiro evento e 7 dias",
            "<b>Primeira sessão</b> = a primeira sessão observada no primeiro mês com uso. <b>Primeiro evento</b> = a primeira interação observada nesse mesmo começo de jornada. <b>Primeiros 7 dias</b> = janela móvel de <code>7 x 24h</code> a partir da âncora de onboarding. Depois disso começa a janela do resultado.",
        ),
        (
            "Janela do resultado",
            "Depois do <code>day_8</code>, o resultado futuro é medido por <code>30 dias</code>. Os validadores externos usam três blocos consecutivos de <code>30 dias</code> após essa janela principal. Essa convenção de horizonte continua explícita no relatório como arbitrária.",
        ),
        (
            "Definição oficial",
            f"No build atual, a escolha final do alvo ficou em <b>{official_definition}</b>. A métrica nativa <code>future_business_active_weeks</code> conta quantas semanas futuras tiveram, na mesma semana, pelo menos uma sessão em <code>fct_session_clean</code> e pelo menos um evento de atividade em <code>fct_interaction_clean</code>. A Definição A oficial acrescenta recorrência, diversidade de uso e um piso mínimo de profundidade em minutos futuros.",
        ),
        (
            "Score oficial",
            score_text,
        ),
        (
            "Como ler o score",
            "O <code>score</code> é a probabilidade calibrada de realizar a atividade futura. Ele pode ser lido como probabilidade porque o modelo cru é recalibrado com <code>sigmoid</code> em um bloco temporal separado do treino e depois checado em meses futuros nunca vistos. O <code>risk_score = 1 - score</code> é a mesma informação vista do lado do risco de não realizar.",
        ),
        (
            "Trilhas oficiais",
            track_text,
        ),
    ]
    return rows


def build_compatibility_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Assunto": "Cluster",
                "Como entra no oficial": "Camada descritiva publicada",
                "Base": "mart_teacher_cluster_ready",
                "Leitura correta": "O algoritmo e a grade de k continuam registrados como escolha de modelagem descritiva. O oficial publica os grupos, seus perfis e sua relação com score e resultado, sem tratar cluster como verdade causal.",
            },
            {
                "Assunto": "Heavy-user",
                "Como entra no oficial": "Proxy contínuo + políticas registradas",
                "Base": "métricas futuras nativas",
                "Leitura correta": "O oficial publica o <code>heavy_intensity_score</code> e os cortes configuráveis usados para chamar alguém de heavy user. A fórmula deixa de ser escondida e passa a ser explicitada como proxy.",
            },
            {
                "Assunto": "Bandas de risco",
                "Como entra no oficial": "Overlay operacional registrado",
                "Base": "risk_score e políticas configuráveis",
                "Leitura correta": "As bandas entram no oficial como overlay operacional e nunca substituem o score contínuo calibrado. A política usada fica explícita na tabela de políticas.",
            },
            {
                "Assunto": "Threshold",
                "Como entra no oficial": "Política configurável publicada",
                "Base": "risk_score, matriz de confusão e métricas por cutoff",
                "Leitura correta": "O score contínuo continua sendo o núcleo do produto. O threshold aparece no oficial como decisão operacional registrada, não como descoberta do modelo.",
            },
            {
                "Assunto": "Robustez por fold",
                "Como entra no oficial": "Auditoria temporal publicada",
                "Base": "outer folds, bootstrap e políticas por fold",
                "Leitura correta": "A estabilidade é lida diretamente na variação do score, das métricas e dos cutoffs ao longo dos outer folds.",
            },
        ]
    )


def display_track_registry(track_registry: pd.DataFrame) -> pd.DataFrame:
    if track_registry.empty:
        return pd.DataFrame()
    show = track_registry.copy()
    show["track_name"] = show["track_name"].map(format_track_name)
    show["allowed_feature_classes_json"] = show["allowed_feature_classes_json"].map(
        lambda x: ", ".join(json.loads(x)) if pd.notna(x) else ""
    )
    show = show.rename(
        columns={
            "track_name": "Trilha",
            "score_window_end_day": "Fim da janela de entrada (dias)",
            "score_moment_text": "Momento do score",
            "allowed_feature_classes_json": "Classes de variáveis permitidas",
        }
    )
    return show[["Trilha", "Fim da janela de entrada (dias)", "Momento do score", "Classes de variáveis permitidas"]]


def display_feature_summary(feature_registry: pd.DataFrame) -> pd.DataFrame:
    if feature_registry.empty:
        return pd.DataFrame()
    show = feature_registry.copy()
    block_map = {
        "context": ("Contexto inicial", "já conhecido no momento de entrada"),
        "s1": ("1ª sessão", "até o fim da 1ª sessão"),
        "s7": ("Primeiros 7 dias", "até o fim do 7º dia"),
    }
    show["block_name"] = show["feature_class"].map(lambda x: block_map.get(str(x), (str(x), ""))[0])
    show["when_available"] = show["feature_class"].map(lambda x: block_map.get(str(x), ("", str(x)))[1])
    show["allowed_tracks"] = show.apply(
        lambda row: ", ".join(
            [
                track
                for track, col in [
                    ("S1", "allowed_in_S1"),
                    ("S7", "allowed_in_S7"),
                    ("S1+S7", "allowed_in_S1_PLUS_S7"),
                    ("STRICT_CONTEXT", "allowed_in_STRICT_CONTEXT"),
                ]
                if int(row.get(col, 0) or 0) == 1
            ]
        ),
        axis=1,
    )
    show["feature_name"] = show["feature_name"].map(format_feature_name)
    show = (
        show.groupby(["block_name", "when_available", "allowed_tracks"], as_index=False)
        .agg(
            variaveis=(
                "feature_name",
                lambda values: ", ".join(sorted(dict.fromkeys(str(value) for value in values))),
            )
        )
        .rename(
            columns={
                "block_name": "Bloco",
                "when_available": "Quando entra",
                "allowed_tracks": "Trilhas em que pode entrar",
                "variaveis": "Variáveis",
            }
        )
    )
    return show[["Bloco", "Quando entra", "Trilhas em que pode entrar", "Variáveis"]]


def display_strict_context_features(feature_registry: pd.DataFrame) -> pd.DataFrame:
    if feature_registry.empty:
        return pd.DataFrame()
    show = feature_registry.copy()
    show = show[(show["allowed_in_STRICT_CONTEXT"] == 1) & (show["feature_class"] == "context")].copy()
    if show.empty:
        return pd.DataFrame()
    show["feature_label"] = show["feature_name"].map(format_feature_name)
    show["feature_meaning"] = show["feature_name"].map(describe_strict_context_feature)
    show = show.rename(
        columns={
            "feature_label": "Variável",
            "behavior_family": "Família",
            "feature_meaning": "O que mede",
        }
    )
    show["Família"] = show["Família"].map(lambda value: str(value).replace("_", " "))
    show = show.sort_values(["Família", "Variável"]).reset_index(drop=True)
    return show[["Variável", "Família", "O que mede"]]


def display_label_registry(label_registry: pd.DataFrame) -> pd.DataFrame:
    if label_registry.empty:
        return pd.DataFrame()
    show = label_registry.copy()
    show["label_name"] = show["label_name"].map(format_definition_name)
    if "rule_json" in show.columns:
        def _rule_from_json(raw: Any) -> str:
            if pd.isna(raw):
                return ""
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                return str(raw)
            if isinstance(parsed, dict):
                metric_name = format_metric_name(parsed.get("metric_name", ""))
                operator = parsed.get("operator", "")
                threshold = parsed.get("threshold", "")
                if metric_name and operator:
                    return f"{metric_name} {operator} {threshold}"
            return str(parsed)

        show["rule_json"] = show["rule_json"].map(_rule_from_json)
    show = show.rename(
        columns={
            "label_name": "Rótulo",
            "label_group": "Papel",
            "source_table": "Tabela fonte",
            "window_start_day": "Início da medição (dia)",
            "window_end_day": "Fim da medição (dia)",
            "rule_json": "Regra do rótulo",
        }
    )
    return show[["Rótulo", "Papel", "Tabela fonte", "Início da medição (dia)", "Fim da medição (dia)", "Regra do rótulo"]]


def display_definition_selection(definition_selection: pd.DataFrame) -> pd.DataFrame:
    if definition_selection.empty:
        return pd.DataFrame()
    show = definition_selection.copy()
    show["definition_name"] = show["definition_name"].map(format_definition_name)
    show["official_status"] = show["official_status"].map(format_official_status)
    show["selection_basis"] = show["selection_basis"].map(format_selection_basis)
    show["metric_name"] = show["metric_name"].map(format_metric_name)
    if "rule_text" in show.columns:
        show["rule_text"] = show["rule_text"].map(format_rule_text)
    show = show.rename(
        columns={
            "definition_name": "Definição",
            "official_status": "Situação no build",
            "metric_name": "Métrica futura",
            "threshold": "Corte",
            "selection_basis": "Como entrou",
            "rule_text": "Regra do label",
        }
    )
    return show[["Definição", "Situação no build", "Métrica futura", "Corte", "Regra do label", "Como entrou"]]


def display_definition_frontier(definition_frontier: pd.DataFrame) -> pd.DataFrame:
    if definition_frontier.empty:
        return pd.DataFrame()
    show = definition_frontier.copy()
    show["definition_name"] = show["definition_name"].map(format_definition_name)
    if "rule_text" in show.columns:
        show["rule_text"] = show["rule_text"].map(format_rule_text)
    show["label_share_pct"] = pd.to_numeric(show.get("label_share_pct"), errors="coerce") / 100.0
    show = show.rename(
        columns={
            "definition_name": "Definição",
            "rule_text": "Regra",
            "label_positives": "Usuários marcados como ativos",
            "label_share_pct": "Share na base",
            "folds": "Outer folds válidos",
            "test_gap_returned_active_post_label_m1": "Gap de retorno no bloco 1",
            "test_gap_returned_active_post_label_m2": "Gap de retorno no bloco 2",
            "test_gap_returned_active_post_label_m3": "Gap de retorno no bloco 3",
            "test_gap_active_days_post_label_3m": "Gap de dias ativos nos 3 blocos",
            "test_gap_sustained_active_2of3_post_label": "Gap de sustentação em 2 de 3 blocos",
        }
    )
    cols = [
        "Definição",
        "Regra",
        "Usuários marcados como ativos",
        "Share na base",
        "Outer folds válidos",
        "Gap de retorno no bloco 1",
        "Gap de retorno no bloco 2",
        "Gap de retorno no bloco 3",
        "Gap de dias ativos nos 3 blocos",
        "Gap de sustentação em 2 de 3 blocos",
    ]
    return show[cols]


def display_scoring_scenarios(scoring_scenarios: pd.DataFrame) -> pd.DataFrame:
    if scoring_scenarios.empty:
        return pd.DataFrame()
    show = scoring_scenarios.copy()
    show["problem_key"] = show["problem_key"].map(format_problem_key)
    show["track_name"] = show["track_name"].map(format_track_name)
    show = show.rename(
        columns={
            "problem_key": "Problema",
            "track_name": "Trilha",
            "feature_count": "Variáveis",
            "rows": "Linhas",
            "positives": "Positivos",
            "negatives": "Negativos",
            "months": "Meses",
        }
    )
    return show[["Problema", "Trilha", "Variáveis", "Linhas", "Positivos", "Negativos", "Meses"]]


def display_model_fold_validity(model_fold_metrics: pd.DataFrame) -> pd.DataFrame:
    if model_fold_metrics.empty:
        return pd.DataFrame()
    show = model_fold_metrics.copy()
    if "invalid_reason" not in show.columns:
        show["invalid_reason"] = ""
    if "fold_valid_flag" not in show.columns:
        show["fold_valid_flag"] = 1
    show = (
        show.groupby(["problem_key", "model_name", "fold_valid_flag", "invalid_reason"], dropna=False, as_index=False)
        .size()
        .rename(columns={"size": "folds"})
    )
    show["problem_key"] = show["problem_key"].map(format_problem_key)
    show["model_name"] = show["model_name"].map(format_model_name)
    show["fold_valid_flag"] = show["fold_valid_flag"].map({1: "válido", 0: "inválido"})
    show["invalid_reason"] = show["invalid_reason"].replace(
        {
            "": "",
            "single_class_fold": "outer fold mono-classe",
            "no_valid_inner_split": "sem inner split válido",
        }
    )
    show = show.rename(
        columns={
            "problem_key": "Problema",
            "model_name": "Modelo",
            "fold_valid_flag": "Status do outer fold",
            "invalid_reason": "Motivo quando inválido",
            "folds": "Quantidade",
        }
    )
    return show[["Problema", "Modelo", "Status do outer fold", "Motivo quando inválido", "Quantidade"]]


def display_inner_split_summary(model_inner_split_audit: pd.DataFrame) -> pd.DataFrame:
    if model_inner_split_audit.empty:
        return pd.DataFrame()
    show = (
        model_inner_split_audit.groupby(["problem_key", "model_name", "split_strategy", "valid_inner_split_flag", "invalid_reason"], dropna=False, as_index=False)
        .size()
        .rename(columns={"size": "inner_splits"})
    )
    show["problem_key"] = show["problem_key"].map(format_problem_key)
    show["model_name"] = show["model_name"].map(format_model_name)
    show["split_strategy"] = show["split_strategy"].replace(
        {
            "temporal_calibration_holdout": "calibração temporal",
            "temporal_tuning_validation": "tuning temporal leve",
        }
    )
    show["valid_inner_split_flag"] = show["valid_inner_split_flag"].map({1: "válido", 0: "inválido"})
    show["invalid_reason"] = show["invalid_reason"].replace(
        {
            "": "",
            "inner_train_single_class": "inner treino mono-classe",
            "inner_test_single_class": "inner teste mono-classe",
            "not_enough_months_for_calibration_holdout": "meses insuficientes para bloco de calibração",
            "calibration_fit_single_class": "treino da calibração mono-classe",
            "calibration_holdout_single_class": "holdout da calibração mono-classe",
            "no_valid_temporal_calibration_holdout": "sem bloco temporal válido para calibração",
            "not_enough_months_for_tuning_validation": "meses insuficientes para tuning temporal",
            "tuning_train_single_class": "treino do tuning mono-classe",
            "tuning_validation_single_class": "validação temporal do tuning mono-classe",
        }
    )
    show = show.rename(
        columns={
            "problem_key": "Problema",
            "model_name": "Modelo",
            "split_strategy": "Etapa interna",
            "valid_inner_split_flag": "Status da etapa interna",
            "invalid_reason": "Motivo quando inválido",
            "inner_splits": "Quantidade",
        }
    )
    return show[["Problema", "Modelo", "Etapa interna", "Status da etapa interna", "Motivo quando inválido", "Quantidade"]]


def display_model_frontier(model_frontier: pd.DataFrame) -> pd.DataFrame:
    if model_frontier.empty:
        return pd.DataFrame()
    show = model_frontier.copy()
    show["problem_key"] = show["problem_key"].map(format_problem_key)
    show["track_name"] = show["track_name"].map(format_track_name)
    show["model_name"] = show["model_name"].map(format_model_name)
    show["pareto_frontier_flag"] = show["pareto_frontier_flag"].map({1: "na fronteira", 0: "fora da fronteira"})
    show = show.rename(
        columns={
            "problem_key": "Problema",
            "track_name": "Trilha",
            "model_name": "Modelo",
            "valid_folds": "Outer folds válidos",
            "pooled_rows": "Linhas no teste concatenado",
            "pooled_positive_rate": "Share ativo no teste concatenado",
            "mean_ap": "AP no teste concatenado",
            "mean_roc_auc": "ROC AUC no teste concatenado",
            "mean_brier": "Brier no teste concatenado",
            "mean_log_loss": "Log loss no teste concatenado",
            "mean_calibration_slope_error": "Erro pooled da slope",
            "mean_calibration_intercept_abs": "Intercepto absoluto pooled",
            "std_ap": "Desvio de AP entre folds",
            "std_brier": "Desvio de Brier entre folds",
            "pareto_frontier_flag": "Status",
        }
    )
    cols = [
        "Problema",
        "Trilha",
        "Modelo",
        "Outer folds válidos",
        "Linhas no teste concatenado",
        "Share ativo no teste concatenado",
        "AP no teste concatenado",
        "ROC AUC no teste concatenado",
        "Brier no teste concatenado",
        "Log loss no teste concatenado",
        "Erro pooled da slope",
        "Intercepto absoluto pooled",
        "Desvio de AP entre folds",
        "Desvio de Brier entre folds",
        "Status",
    ]
    return show[cols]


def _frontier_only(model_frontier: pd.DataFrame) -> pd.DataFrame:
    if model_frontier.empty:
        return pd.DataFrame()
    if "pareto_frontier_flag" not in model_frontier.columns:
        return model_frontier.copy()
    frontier_only = model_frontier[model_frontier["pareto_frontier_flag"] == 1].copy()
    if "definition_name" not in model_frontier.columns:
        return frontier_only if not frontier_only.empty else model_frontier.copy()
    missing_definitions = sorted(
        set(model_frontier["definition_name"].dropna().astype(str).unique().tolist())
        - set(frontier_only["definition_name"].dropna().astype(str).unique().tolist())
    )
    if not missing_definitions:
        return frontier_only
    extras: list[pd.DataFrame] = []
    sort_cols = [col for col in ["mean_brier", "mean_log_loss", "mean_ap", "mean_roc_auc"] if col in model_frontier.columns]
    ascending = [True, True, False, False][: len(sort_cols)]
    for definition_name in missing_definitions:
        group = model_frontier[model_frontier["definition_name"].astype(str) == definition_name].copy()
        if group.empty:
            continue
        if sort_cols:
            group = group.sort_values(sort_cols, ascending=ascending, kind="mergesort")
        extras.append(group.head(1))
    if not extras:
        return frontier_only if not frontier_only.empty else model_frontier.copy()
    return pd.concat([frontier_only, *extras], ignore_index=True).drop_duplicates()


def _lookup_population(scoring_scenarios: pd.DataFrame, problem_key: Any) -> tuple[Any, Any, Any]:
    if scoring_scenarios.empty or "problem_key" not in scoring_scenarios.columns:
        return "", "", ""
    match = scoring_scenarios[scoring_scenarios["problem_key"] == problem_key]
    if match.empty:
        return "", "", ""
    row = match.iloc[0]
    rows = row.get("rows", np.nan)
    positives = row.get("positives", np.nan)
    if pd.isna(rows) or rows == 0 or pd.isna(positives):
        return rows, positives, ""
    return rows, positives, positives / rows


def filter_to_reference_scope(df: pd.DataFrame, reference_scope: pd.DataFrame) -> pd.DataFrame:
    if df.empty or reference_scope.empty:
        return df.copy()
    join_cols = [col for col in ["problem_key", "model_name"] if col in df.columns and col in reference_scope.columns]
    if not join_cols:
        return df.copy()
    keys = reference_scope[join_cols].drop_duplicates()
    return df.merge(keys, on=join_cols, how="inner")


def select_presentation_scope(reference_scope: pd.DataFrame) -> pd.DataFrame:
    if reference_scope.empty:
        return reference_scope.copy()
    show = reference_scope.copy()
    if "selection_reason" not in show.columns:
        return show.head(2).copy()
    by_definition = show[show["selection_reason"].astype(str).str.startswith("best_probability_first_within_definition::")].copy()
    if not by_definition.empty:
        return by_definition.sort_values(["problem_key", "model_name"]).drop_duplicates(subset=["problem_key", "model_name"]).copy()
    overall = show[show["selection_reason"].astype(str).str.endswith("_overall")].copy()
    if overall.empty:
        return show.head(2).copy()
    priority = {
        "best_probability_first_overall": 0,
        "best_ap_first_overall": 1,
    }
    overall["priority"] = overall["selection_reason"].map(lambda x: priority.get(str(x), 99))
    overall = overall.sort_values(["priority", "problem_key", "model_name"]).drop(columns=["priority"])
    return overall.drop_duplicates(subset=["problem_key", "model_name"]).copy()


def filter_to_reference_definitions(df: pd.DataFrame, reference_scope: pd.DataFrame) -> pd.DataFrame:
    if df.empty or reference_scope.empty:
        return df.copy()
    if "problem_key" not in reference_scope.columns:
        return df.copy()
    definition_names: list[str] = []
    for raw_problem_key in reference_scope["problem_key"].dropna().astype(str).unique().tolist():
        if "__" in raw_problem_key:
            definition_names.append(raw_problem_key.rsplit("__", 1)[0])
        else:
            definition_names.append(raw_problem_key)
    definition_names = list(dict.fromkeys(definition_names))
    if "definition_name" in df.columns:
        return df[df["definition_name"].astype(str).isin(definition_names)].copy()
    return df.copy()


def display_definition_family_summary(model_frontier: pd.DataFrame, scoring_scenarios: pd.DataFrame) -> pd.DataFrame:
    frontier_only = _frontier_only(model_frontier)
    if frontier_only.empty or "definition_name" not in frontier_only.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for definition_name, group in frontier_only.groupby("definition_name", dropna=False):
        best_row = group.sort_values(
            ["mean_ap", "mean_roc_auc", "mean_brier", "mean_log_loss"],
            ascending=[False, False, True, True],
        ).iloc[0]
        total_rows, positives, positive_rate = _lookup_population(scoring_scenarios, best_row["problem_key"])
        rows.append(
            {
                "Definição": format_definition_name(definition_name),
                "Melhor cenário da definição": format_problem_key(best_row["problem_key"]),
                "Melhor modelo nesse cenário": format_model_name(best_row["model_name"]),
                "Outer folds válidos": best_row.get("valid_folds", ""),
                "AP no teste concatenado": best_row.get("mean_ap", np.nan),
                "ROC AUC no teste concatenado": best_row.get("mean_roc_auc", np.nan),
                "Brier no teste concatenado": best_row.get("mean_brier", np.nan),
                "Log loss no teste concatenado": best_row.get("mean_log_loss", np.nan),
                "Linhas": total_rows,
                "Ativos": positives,
                "Share ativo": positive_rate,
            }
        )
    show = pd.DataFrame(rows).sort_values(["AP no teste concatenado", "ROC AUC no teste concatenado"], ascending=[False, False])
    show["Share ativo"] = show["Share ativo"].map(format_percent)
    return show[
        [
            "Definição",
            "Melhor cenário da definição",
            "Melhor modelo nesse cenário",
            "Outer folds válidos",
            "AP no teste concatenado",
            "ROC AUC no teste concatenado",
            "Brier no teste concatenado",
            "Log loss no teste concatenado",
            "Linhas",
            "Ativos",
            "Share ativo",
        ]
    ]


def display_model_family_summary(model_frontier: pd.DataFrame, scoring_scenarios: pd.DataFrame) -> pd.DataFrame:
    frontier_only = _frontier_only(model_frontier)
    if frontier_only.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for model_name, group in frontier_only.groupby("model_name", dropna=False):
        best_row = group.sort_values(
            ["mean_ap", "mean_roc_auc", "mean_brier", "mean_log_loss"],
            ascending=[False, False, True, True],
        ).iloc[0]
        total_rows, positives, positive_rate = _lookup_population(scoring_scenarios, best_row["problem_key"])
        rows.append(
            {
                "Família de modelo": format_model_name(model_name),
                "Melhor cenário do modelo": format_problem_key(best_row["problem_key"]),
                "Definição": format_definition_name(best_row.get("definition_name", "")),
                "Trilha": format_track_name(best_row.get("track_name", "")),
                "Outer folds válidos": best_row.get("valid_folds", ""),
                "AP no teste concatenado": best_row.get("mean_ap", np.nan),
                "ROC AUC no teste concatenado": best_row.get("mean_roc_auc", np.nan),
                "Brier no teste concatenado": best_row.get("mean_brier", np.nan),
                "Log loss no teste concatenado": best_row.get("mean_log_loss", np.nan),
                "Share ativo": positive_rate,
            }
        )
    show = pd.DataFrame(rows).sort_values(["AP no teste concatenado", "ROC AUC no teste concatenado"], ascending=[False, False])
    show["Share ativo"] = show["Share ativo"].map(format_percent)
    return show[
        [
            "Família de modelo",
            "Melhor cenário do modelo",
            "Definição",
            "Trilha",
            "Outer folds válidos",
            "AP no teste concatenado",
            "ROC AUC no teste concatenado",
            "Brier no teste concatenado",
            "Log loss no teste concatenado",
            "Share ativo",
        ]
    ]


def build_external_validator_guide() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Validador externo": "Gap de retorno no bloco 1",
                "O que mede": "diferença de retorno ativo no 1º bloco de 30 dias depois da janela principal do label",
                "Por que ajuda": "mostra se a definição escolhida continua sinalizando retorno logo depois do período que definiu o próprio rótulo",
            },
            {
                "Validador externo": "Gap de retorno no bloco 2",
                "O que mede": "diferença de retorno ativo no 2º bloco de 30 dias depois da janela principal",
                "Por que ajuda": "testa se o sinal continua vivo um pouco mais à frente, e não só imediatamente após a janela do label",
            },
            {
                "Validador externo": "Gap de retorno no bloco 3",
                "O que mede": "diferença de retorno ativo no 3º bloco de 30 dias depois da janela principal",
                "Por que ajuda": "testa se a definição ainda separa comportamento quando o horizonte já está bem mais distante",
            },
            {
                "Validador externo": "Gap de dias ativos nos 3 blocos",
                "O que mede": "diferença no total de dias ativos somados ao longo dos três blocos pós-label",
                "Por que ajuda": "resume sustentação e recorrência futura em uma única leitura acumulada",
            },
            {
                "Validador externo": "Gap de sustentação em 2 de 3 blocos",
                "O que mede": "diferença na fração que permaneceu ativa em pelo menos 2 dos 3 blocos pós-label",
                "Por que ajuda": "evita chamar de ativo alguém que só voltou uma vez e depois sumiu",
            },
        ]
    )


def display_selected_problem_model_comparison(model_frontier: pd.DataFrame, reference_scope: pd.DataFrame) -> pd.DataFrame:
    if model_frontier.empty or reference_scope.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, ref_row in reference_scope.iterrows():
        problem_key = ref_row.get("problem_key")
        selected_model = str(ref_row.get("model_name", ""))
        subset = model_frontier[model_frontier["problem_key"] == problem_key].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(["mean_brier", "mean_log_loss", "mean_ap"], ascending=[True, True, False])
        for _, row in subset.iterrows():
            rows.append(
                {
                    "Problema": format_problem_key(problem_key),
                    "Modelo": format_model_name(row.get("model_name", "")),
                    "Situação na comparação": "selecionado" if str(row.get("model_name", "")) == selected_model else "comparado",
                    "Outer folds válidos": row.get("valid_folds", np.nan),
                    "AP": row.get("mean_ap", np.nan),
                    "ROC AUC": row.get("mean_roc_auc", np.nan),
                    "Brier": row.get("mean_brier", np.nan),
                    "Log loss": row.get("mean_log_loss", np.nan),
                    "Erro pooled da slope": row.get("mean_calibration_slope_error", np.nan),
                    "Intercepto absoluto pooled": row.get("mean_calibration_intercept_abs", np.nan),
                }
            )
    return pd.DataFrame(rows)


def display_selected_problem_operational_comparison(predictions: pd.DataFrame, reference_scope: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or reference_scope.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    valid_predictions = predictions[predictions["fold_valid_flag"] == 1].copy() if "fold_valid_flag" in predictions.columns else predictions.copy()
    for _, ref_row in reference_scope.iterrows():
        problem_key = ref_row.get("problem_key")
        subset = valid_predictions[valid_predictions["problem_key"] == problem_key].copy()
        if subset.empty:
            continue
        for model_name, group in subset.groupby("model_name", dropna=False):
            y_true = pd.to_numeric(group["y_true"], errors="coerce").fillna(0).astype(int).to_numpy()
            score = pd.to_numeric(group["score"], errors="coerce").fillna(0).to_numpy()
            risk_score = 1 - score
            y_risk = (y_true == 0).astype(int)

            top10_cutoff = float(np.quantile(risk_score, 0.9))
            tercis_cutoff = float(np.quantile(risk_score, 2 / 3))
            policy_map = {
                "top10": top10_cutoff,
                "tercis": tercis_cutoff,
                "risk70": 0.7,
            }
            metrics: dict[str, Any] = {}
            for policy_key, cutoff in policy_map.items():
                predicted = (risk_score >= cutoff).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_risk, predicted, labels=[0, 1]).ravel()
                metrics[f"{policy_key}_f1"] = f1_score(y_risk, predicted, zero_division=0)
                metrics[f"{policy_key}_tp"] = int(tp)
                metrics[f"{policy_key}_fp"] = int(fp)
                metrics[f"{policy_key}_tn"] = int(tn)
                metrics[f"{policy_key}_fn"] = int(fn)

            monthly = (
                group.assign(
                    risk_score=risk_score,
                    month=pd.to_datetime(group["first_month"]).dt.to_period("M").astype(str),
                )
                .groupby("month", as_index=False)
                .agg(
                    realized_non_realization=(
                        "y_true",
                        lambda values: float((pd.to_numeric(values, errors="coerce").fillna(0).astype(int) == 0).mean()),
                    ),
                    mean_risk_score=("risk_score", "mean"),
                )
            )
            if len(monthly) >= 2:
                monthly_r2 = r2_score(monthly["realized_non_realization"], monthly["mean_risk_score"])
            else:
                monthly_r2 = float("nan")
            positive_months = monthly[monthly["realized_non_realization"] > 0].copy()
            if positive_months.empty:
                monthly_mape = float("nan")
            else:
                monthly_mape = float(
                    np.mean(
                        np.abs(
                            (positive_months["realized_non_realization"] - positive_months["mean_risk_score"])
                            / positive_months["realized_non_realization"]
                        )
                    )
                )
            rows.append(
                {
                    "Problema": format_problem_key(problem_key),
                    "Modelo": format_model_name(model_name),
                    "F1 no top 10% do risk_score": metrics["top10_f1"],
                    "TP/FP/TN/FN no top 10%": f"{metrics['top10_tp']} / {metrics['top10_fp']} / {metrics['top10_tn']} / {metrics['top10_fn']}",
                    "F1 nos tercis": metrics["tercis_f1"],
                    "TP/FP/TN/FN nos tercis": f"{metrics['tercis_tp']} / {metrics['tercis_fp']} / {metrics['tercis_tn']} / {metrics['tercis_fn']}",
                    "F1 no risk_score >= 0,70": metrics["risk70_f1"],
                    "TP/FP/TN/FN no risk_score >= 0,70": f"{metrics['risk70_tp']} / {metrics['risk70_fp']} / {metrics['risk70_tn']} / {metrics['risk70_fn']}",
                    "R2 mensal do risk_score": monthly_r2,
                    "MAPE mensal do risk_score": monthly_mape,
                }
            )
    return pd.DataFrame(rows)


def build_definition_answer(definition_family_summary: pd.DataFrame) -> str:
    if definition_family_summary.empty:
        return (
            "Resposta curta. Nesta execução, nenhuma definição conseguiu deixar uma resposta simples e publicável "
            "depois do protocolo temporal completo."
        )
    best_rank = definition_family_summary.sort_values(["AP no teste concatenado", "ROC AUC no teste concatenado"], ascending=[False, False]).iloc[0]
    best_prob = definition_family_summary.sort_values(["Brier no teste concatenado", "Log loss no teste concatenado"], ascending=[True, True]).iloc[0]
    if str(best_rank["Definição"]) == str(best_prob["Definição"]):
        return (
            f"Resposta curta. A definição que ficou mais forte foi <b>{best_rank['Definição']}</b>, "
            f"com melhor leitura de ranking e também melhor erro probabilístico no seu melhor cenário "
            f"(<code>{best_rank['Melhor cenário da definição']}</code>)."
        )
    return (
        f"Resposta curta. Em ranking, a definição que ficou mais forte foi <b>{best_rank['Definição']}</b>. "
        f"Em erro probabilístico, quem ficou melhor foi <b>{best_prob['Definição']}</b>. "
        "Isso quer dizer que a resposta final ainda precisa ser lida como compromisso entre separar bem e calibrar bem."
    )


def build_model_answer(model_family_summary: pd.DataFrame) -> str:
    if model_family_summary.empty:
        return (
            "Resposta curta. Nesta execução, nenhuma família de modelo conseguiu deixar uma resposta simples e publicável "
            "depois do protocolo temporal completo."
        )
    best_rank = model_family_summary.sort_values(["AP no teste concatenado", "ROC AUC no teste concatenado"], ascending=[False, False]).iloc[0]
    best_prob = model_family_summary.sort_values(["Brier no teste concatenado", "Log loss no teste concatenado"], ascending=[True, True]).iloc[0]
    if str(best_rank["Família de modelo"]) == str(best_prob["Família de modelo"]):
        return (
            f"Resposta curta. A família de modelo que ficou mais forte foi <b>{best_rank['Família de modelo']}</b>, "
            f"no cenário <code>{best_rank['Melhor cenário do modelo']}</code>."
        )
    return (
        f"Resposta curta. Em ranking, a família de modelo que ficou mais forte foi <b>{best_rank['Família de modelo']}</b>. "
        f"Em erro probabilístico, quem ficou melhor foi <b>{best_prob['Família de modelo']}</b>. "
        "Isso mostra que o melhor modelo depende do que a operação quer priorizar: ordenar bem ou calibrar melhor a probabilidade."
    )


def display_scenario_balance(scoring_scenarios: pd.DataFrame) -> pd.DataFrame:
    if scoring_scenarios.empty:
        return pd.DataFrame()
    show = scoring_scenarios.copy()
    show["problem_key"] = show["problem_key"].map(format_problem_key)
    positives = pd.to_numeric(show["positives"], errors="coerce").fillna(0)
    rows = pd.to_numeric(show["rows"], errors="coerce").replace(0, np.nan)
    show["positive_rate"] = positives / rows
    show["imbalance_note"] = np.where(
        show["positive_rate"] <= 0.10,
        "atividade futura rara",
        np.where(show["positive_rate"] >= 0.90, "Negativo raro", "Sem extremo severo"),
    )
    show = show.rename(
        columns={
            "problem_key": "Problema",
            "rows": "Linhas",
            "positives": "Ativos futuros",
            "negatives": "Não ativos futuros",
            "positive_rate": "Taxa de atividade futura",
            "months": "Meses",
            "imbalance_note": "Leitura de balanceamento",
        }
    )
    cols = ["Problema", "Linhas", "Ativos futuros", "Não ativos futuros", "Taxa de atividade futura", "Meses", "Leitura de balanceamento"]
    return show[cols]


def display_arbitrariness(arbitrariness: pd.DataFrame) -> pd.DataFrame:
    if arbitrariness.empty:
        return pd.DataFrame()
    show = arbitrariness[arbitrariness["in_official_report_flag"] == 1].copy()
    show["choice_name"] = show["choice_name"].map(format_arbitrary_name)
    show["choice_type"] = show["choice_type"].map(format_arbitrary_type)
    show["status"] = show["status"].map(format_arbitrary_status)
    show["where_used"] = show["where_used"].map(format_where_used)
    show["why"] = show["why"].map(format_arbitrary_why)
    show = show.rename(
        columns={
            "choice_name": "Escolha",
            "choice_value": "Valor",
            "choice_type": "Tipo",
            "where_used": "Onde é usada",
            "status": "Status",
            "why": "Motivo",
        }
    )
    return show[["Escolha", "Valor", "Tipo", "Onde é usada", "Status", "Motivo"]]


def display_policy_registry(policy_registry: pd.DataFrame) -> pd.DataFrame:
    if policy_registry.empty:
        return pd.DataFrame()
    show = policy_registry.copy()
    show["policy_group"] = show["policy_group"].map(format_policy_group)
    show["policy_name"] = show["policy_name"].map(format_policy_name)
    show = show.rename(
        columns={
            "policy_group": "Grupo",
            "policy_name": "Política",
            "policy_value_json": "Parâmetros",
            "active_in_build_flag": "Ativa no build",
            "official_flag": "Oficial",
            "why": "Motivo",
        }
    )
    show["Ativa no build"] = show["Ativa no build"].map({1: "sim", 0: "não"})
    show["Oficial"] = show["Oficial"].map({1: "sim", 0: "não"})
    return show[["Grupo", "Política", "Parâmetros", "Ativa no build", "Oficial", "Motivo"]]


def display_reference_scope(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Problema", "Modelo", "Motivo da seleção"])
    show = df.copy()
    if "problem_key" in show.columns:
        show["problem_key"] = show["problem_key"].map(format_problem_key)
    if "model_name" in show.columns:
        show["model_name"] = show["model_name"].map(format_model_name)
    if "selection_reason" in show.columns:
        show["selection_reason"] = show["selection_reason"].map(format_selection_reason)
    show = show.rename(
        columns={
            "problem_key": "Problema",
            "model_name": "Modelo",
            "selection_reason": "Motivo da seleção",
        }
    )
    return show[[c for c in ["Problema", "Modelo", "Motivo da seleção"] if c in show.columns]]


def display_threshold_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    show = df.copy()
    show["problem_key"] = show["problem_key"].map(format_problem_key)
    show["model_name"] = show["model_name"].map(format_model_name)
    show["policy_name"] = show["policy_name"].map(format_policy_name)
    show = show.rename(
        columns={
            "problem_key": "Problema",
            "model_name": "Modelo",
            "policy_name": "Política de cutoff",
            "risk_threshold": "Cutoff no risk_score",
            "tp": "TP",
            "fp": "FP",
            "tn": "TN",
            "fn": "FN",
            "precision": "Precisão",
            "recall": "Recall",
            "f1": "F1",
            "accuracy": "Acurácia",
            "predicted_positive_rate": "Share previsto como alto risco",
        }
    )
    return show[
        [
            "Problema",
            "Modelo",
            "Política de cutoff",
            "Cutoff no risk_score",
            "TP",
            "FP",
            "TN",
            "FN",
            "Precisão",
            "Recall",
            "F1",
            "Acurácia",
            "Share previsto como alto risco",
        ]
    ]


def display_confusion_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    show = df.copy()
    show["cell_name"] = show.apply(lambda row: confusion_cell_code(row["actual_group"], row["predicted_group"]), axis=1)
    show = (
        show.pivot_table(
            index=["problem_key", "model_name", "policy_name"],
            columns="cell_name",
            values="rows",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    show.columns.name = None
    for col in ["TP", "FP", "TN", "FN"]:
        if col not in show.columns:
            show[col] = 0
    show["problem_key"] = show["problem_key"].map(format_problem_key)
    show["model_name"] = show["model_name"].map(format_model_name)
    show["policy_name"] = show["policy_name"].map(format_policy_name)
    show = show.rename(
        columns={
            "problem_key": "Problema",
            "model_name": "Modelo",
            "policy_name": "Política de cutoff",
        }
    )
    return show[["Problema", "Modelo", "Política de cutoff", "TP", "FP", "TN", "FN"]]


def display_band_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    show = df.copy()
    show["problem_key"] = show["problem_key"].map(format_problem_key)
    show["model_name"] = show["model_name"].map(format_model_name)
    show["policy_name"] = show["policy_name"].map(format_policy_name)
    show = show.rename(
        columns={
            "problem_key": "Problema",
            "model_name": "Modelo",
            "policy_name": "Política de bandas",
            "band_name": "Faixa",
            "rows": "Linhas",
            "share": "Share",
        }
    )
    return show[["Problema", "Modelo", "Política de bandas", "Faixa", "Linhas", "Share"]]


def display_monthly_fit(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    show = df.copy()
    show["problem_key"] = show["problem_key"].map(format_problem_key)
    show["model_name"] = show["model_name"].map(format_model_name)
    show = show.rename(
        columns={
            "problem_key": "Problema",
            "model_name": "Modelo",
            "monthly_r2": "R2 mensal do risco realizado",
            "monthly_mape_positive_months": "MAPE mensal do risco realizado",
            "months_used": "Meses usados",
        }
    )
    return show[["Problema", "Modelo", "R2 mensal do risco realizado", "MAPE mensal do risco realizado", "Meses usados"]]


def display_bootstrap(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    show = df.copy()
    if "problem_key" in show.columns:
        show["problem_key"] = show["problem_key"].map(format_problem_key)
    if "model_name" in show.columns:
        show["model_name"] = show["model_name"].map(format_model_name)
    metric_map = {
        "ap": "AP",
        "roc_auc": "ROC AUC",
        "brier": "Brier",
        "log_loss": "Log loss",
    }
    if "metric_name" in show.columns:
        show["metric_name"] = show["metric_name"].map(lambda x: metric_map.get(str(x), str(x)))
    show = show.rename(
        columns={
            "problem_key": "Problema",
            "model_name": "Modelo",
            "metric_name": "Métrica",
            "ci_low": "IC baixo",
            "ci_high": "IC alto",
            "ci_width": "Largura do IC",
        }
    )
    return show[["Problema", "Modelo", "Métrica", "IC baixo", "IC alto", "Largura do IC"]]


def display_feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    show = (
        df.groupby(["problem_key", "model_name", "feature_name"], as_index=False)
        .agg(importance_media=("importance_mean", "mean"), importance_std=("importance_std", "mean"), folds=("fold_id", "nunique"))
        .sort_values(["problem_key", "model_name", "importance_media"], ascending=[True, True, False])
    )
    show["problem_key"] = show["problem_key"].map(format_problem_key)
    show["model_name"] = show["model_name"].map(format_model_name)
    show["feature_name"] = show["feature_name"].map(format_feature_name)
    show = show.rename(
        columns={
            "problem_key": "Problema",
            "model_name": "Modelo",
            "feature_name": "Sinal",
            "importance_media": "Importância média por permutação",
            "importance_std": "Desvio médio",
            "folds": "Folds",
        }
    )
    return show[["Problema", "Modelo", "Sinal", "Importância média por permutação", "Desvio médio", "Folds"]]


def display_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    show = df.copy()
    show["problem_key"] = show["problem_key"].map(format_problem_key)
    show["model_name"] = show["model_name"].map(format_model_name)
    show["cluster_name"] = show["cluster_name"].map(format_cluster_name)
    show = show.rename(
        columns={
            "problem_key": "Problema",
            "model_name": "Modelo",
            "cluster_name": "Cluster",
            "rows": "Linhas",
            "share": "Share",
            "mean_score": "Score médio",
            "mean_risk_score": "Risk score médio",
            "realized_inactivity_rate": "Taxa realizada de não atividade",
        }
    )
    return show[["Problema", "Modelo", "Cluster", "Linhas", "Share", "Score médio", "Risk score médio", "Taxa realizada de não atividade"]]


def display_cluster_profile(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    show = df.copy().sort_values(["cluster_name", "cluster_rows", "feature_name"])
    show["cluster_name"] = show["cluster_name"].map(format_cluster_name)
    show["feature_name"] = show["feature_name"].map(format_feature_name)
    show = show.rename(
        columns={
            "cluster_name": "Cluster",
            "feature_name": "Sinal do mart de cluster",
            "feature_mean": "Média no cluster",
            "cluster_rows": "Linhas do cluster",
        }
    )
    return show[["Cluster", "Sinal do mart de cluster", "Média no cluster", "Linhas do cluster"]]


def display_cluster_validation(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    show = df.copy().rename(
        columns={
            "iteration_id": "Iteração bootstrap",
            "cluster_k": "k do bootstrap",
            "silhouette": "Silhouette do bootstrap",
            "stability_ari_vs_full": "ARI vs ajuste cheio",
            "selected_cluster_k": "k selecionado",
            "selected_cluster_silhouette": "Silhouette selecionado",
        }
    )
    return show[["Iteração bootstrap", "k do bootstrap", "Silhouette do bootstrap", "ARI vs ajuste cheio", "k selecionado", "Silhouette selecionado"]]


def display_heavy_user_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    show = df.copy()
    show["problem_key"] = show["problem_key"].map(format_problem_key)
    show["model_name"] = show["model_name"].map(format_model_name)
    show["policy_name"] = show["policy_name"].map(format_policy_name)
    show["heavy_user_flag"] = show["heavy_user_flag"].map({1: "heavy user", 0: "demais"})
    show = show.rename(
        columns={
            "problem_key": "Problema",
            "model_name": "Modelo",
            "policy_name": "Política de heavy user",
            "heavy_user_flag": "Grupo",
            "rows": "Linhas",
            "share": "Share",
            "mean_score": "Score médio",
            "mean_risk_score": "Risk score médio",
            "realized_inactivity_rate": "Taxa realizada de não atividade",
        }
    )
    return show[["Problema", "Modelo", "Política de heavy user", "Grupo", "Linhas", "Share", "Score médio", "Risk score médio", "Taxa realizada de não atividade"]]


def display_heavy_user_profile(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    show = df.copy()
    show["policy_name"] = show["policy_name"].map(format_policy_name)
    show["heavy_user_flag"] = show["heavy_user_flag"].map({1: "heavy user", 0: "demais"})
    show["metric_name"] = show["metric_name"].map(format_metric_name)
    show = show.rename(
        columns={
            "policy_name": "Política de heavy user",
            "heavy_user_flag": "Grupo",
            "metric_name": "Métrica futura nativa",
            "metric_mean": "Média da métrica",
            "heavy_cutoff": "Cutoff do heavy_intensity_score",
        }
    )
    return show[["Política de heavy user", "Grupo", "Métrica futura nativa", "Média da métrica", "Cutoff do heavy_intensity_score"]]


def display_leak_summary(leakage_audit: pd.DataFrame) -> pd.DataFrame:
    if leakage_audit.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "Linhas auditadas": int(len(leakage_audit)),
                "Linhas com leakage_flag": int(leakage_audit["leakage_flag"].sum()) if "leakage_flag" in leakage_audit.columns else 0,
                "Linhas com coluna-fonte compartilhada": int(leakage_audit["same_source_column_flag"].sum()) if "same_source_column_flag" in leakage_audit.columns else 0,
                "Linhas com violação temporal": int((leakage_audit["temporal_window_ok_flag"] == 0).sum()) if "temporal_window_ok_flag" in leakage_audit.columns else 0,
                "Linhas tocando tabela ou coluna futura": int(leakage_audit["source_touches_future_window_flag"].sum()) if "source_touches_future_window_flag" in leakage_audit.columns else 0,
            }
        ]
    )


def display_definition_b_leakage_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    show = df[df["definition_name"] == "definition_b_label"].copy() if "definition_name" in df.columns else df.copy()
    if show.empty:
        return pd.DataFrame()
    show["problem_key"] = show["problem_key"].map(format_problem_key)
    show["track_name"] = show["track_name"].map(format_track_name)
    show = show.rename(
        columns={
            "problem_key": "Problema",
            "track_name": "Trilha",
            "audited_features": "Variáveis auditadas",
            "features_with_leakage_flag": "Variáveis com leakage_flag",
            "features_with_source_overlap": "Variáveis com coluna-fonte sobreposta",
            "features_with_future_named_source": "Variáveis com nome de coluna futura",
            "features_touching_label_source_table": "Variáveis tocando a tabela do label",
            "features_with_temporal_violation": "Variáveis com violação temporal",
            "features_with_high_risk_future_touch": "Variáveis com toque de alto risco",
            "all_features_available_at_score_time_flag": "Todas disponíveis no momento do score",
            "all_features_pit_safe_flag": "Todas PIT-safe",
            "any_leakage_flag": "Algum leakage_flag",
        }
    )
    cols = [
        "Problema",
        "Trilha",
        "Variáveis auditadas",
        "Variáveis com leakage_flag",
        "Variáveis com coluna-fonte sobreposta",
        "Variáveis com nome de coluna futura",
        "Variáveis tocando a tabela do label",
        "Variáveis com violação temporal",
        "Variáveis com toque de alto risco",
        "Todas disponíveis no momento do score",
        "Todas PIT-safe",
        "Algum leakage_flag",
    ]
    return show[[col for col in cols if col in show.columns]]


def display_definition_b_feature_block_gain(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    show = df.copy()
    show["reference_problem_key"] = show["reference_problem_key"].map(format_problem_key)
    show["model_name"] = show["model_name"].map(format_model_name)
    show["block_name"] = (
        show["block_name"]
        .astype(str)
        .str.replace("baseline_context_only", "baseline de contexto", regex=False)
        .str.replace("context_plus_feature_class::", "contexto + classe ", regex=False)
        .str.replace("context_plus_behavior_family::", "contexto + família ", regex=False)
        .str.replace("full_allowed_features", "todas as variáveis permitidas", regex=False)
        .str.replace("_", " ", regex=False)
    )
    show = show.rename(
        columns={
            "reference_problem_key": "Problema de referência",
            "model_name": "Modelo",
            "block_name": "Bloco testado",
            "block_type": "Tipo de bloco",
            "selected_feature_count": "Variáveis no bloco",
            "added_feature_count": "Variáveis adicionadas",
            "valid_folds": "Folds válidos",
            "mean_ap": "AP no teste concatenado",
            "mean_roc_auc": "ROC AUC no teste concatenado",
            "mean_brier": "Brier no teste concatenado",
            "mean_log_loss": "Log loss no teste concatenado",
            "delta_ap_vs_context": "Ganho de AP vs contexto",
            "delta_roc_auc_vs_context": "Ganho de ROC AUC vs contexto",
            "brier_improvement_vs_context": "Melhora de Brier vs contexto",
            "log_loss_improvement_vs_context": "Melhora de log loss vs contexto",
            "mean_uplift_percentile": "Percentil médio de uplift",
            "abnormal_uplift_flag": "Salto anormal vs baseline",
        }
    )
    cols = [
        "Problema de referência",
        "Modelo",
        "Bloco testado",
        "Tipo de bloco",
        "Variáveis no bloco",
        "Variáveis adicionadas",
        "Folds válidos",
        "AP no teste concatenado",
        "ROC AUC no teste concatenado",
        "Brier no teste concatenado",
        "Log loss no teste concatenado",
        "Ganho de AP vs contexto",
        "Ganho de ROC AUC vs contexto",
        "Melhora de Brier vs contexto",
        "Melhora de log loss vs contexto",
        "Percentil médio de uplift",
        "Salto anormal vs baseline",
    ]
    return show[[col for col in cols if col in show.columns]]


def filter_definition_b_feature_block_gain_for_report(
    df: pd.DataFrame,
    presentation_scope: pd.DataFrame,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    show = df.copy()
    if "definition_name" in show.columns:
        show = show[show["definition_name"].astype(str) == "definition_b_label"].copy()
    return show


def _block_gain_row(df: pd.DataFrame, block_name: str) -> pd.Series | None:
    if df.empty or "block_name" not in df.columns:
        return None
    match = df[df["block_name"].astype(str) == block_name].copy()
    if match.empty:
        return None
    sort_cols = [col for col in ["mean_uplift_percentile", "delta_ap_vs_context", "delta_roc_auc_vs_context"] if col in match.columns]
    if sort_cols:
        match = match.sort_values(sort_cols, ascending=[False] * len(sort_cols), kind="mergesort")
    return match.iloc[0]


def build_definition_b_feature_block_gain_text(df: pd.DataFrame) -> tuple[str, str]:
    if df.empty:
        return (
            "O teste incremental por blocos não ficou disponível para a trilha publicada da Definição B nesta materialização.",
            "Sem essa materialização, o relatório fica só com a auditoria estrutural de leakage e a comparação principal dos modelos completos.",
        )
    s1_row = _block_gain_row(df, "context_plus_feature_class::s1")
    s7_row = _block_gain_row(df, "context_plus_feature_class::s7")
    early_views = _block_gain_row(df, "context_plus_behavior_family::early_views")
    early_downloads = _block_gain_row(df, "context_plus_behavior_family::early_downloads")
    week_views = _block_gain_row(df, "context_plus_behavior_family::week_views")
    week_downloads = _block_gain_row(df, "context_plus_behavior_family::week_downloads")

    text_1 = (
        "Este teste é complementar e usa regressão logística como modelo linear de referência. "
        "Ele compara <b>contexto apenas</b> contra <b>contexto + um bloco de sinais</b>, então não escolhe o modelo oficial. "
        "Serve só para medir ganho incremental de sinal."
    )
    parts: list[str] = []
    if s1_row is not None:
        parts.append(
            "Na 1ª sessão, o bloco agregado de sessão inicial melhorou o baseline de contexto em "
            f"<b>+{format_number(s1_row.get('delta_ap_vs_context'), 3)}</b> de AP e "
            f"<b>+{format_number(s1_row.get('delta_roc_auc_vs_context'), 3)}</b> de ROC AUC."
        )
    if s7_row is not None:
        parts.append(
            "Na trilha de 7 dias, o bloco agregado de semana inicial melhorou o baseline de contexto em "
            f"<b>+{format_number(s7_row.get('delta_ap_vs_context'), 3)}</b> de AP e "
            f"<b>+{format_number(s7_row.get('delta_roc_auc_vs_context'), 3)}</b> de ROC AUC, "
            "o que mostra ganho adicional claro ao esperar essa janela."
        )
    if early_views is not None and early_downloads is not None:
        parts.append(
            "Nos sinais da 1ª sessão, <b>views</b> adicionaram mais do que <b>downloads</b>: "
            f"AP {format_number(early_views.get('delta_ap_vs_context'), 3)} vs {format_number(early_downloads.get('delta_ap_vs_context'), 3)}."
        )
    if week_views is not None and week_downloads is not None:
        parts.append(
            "Nos sinais de semana, o padrão se repete: "
            f"<b>views</b> ficam acima de <b>downloads</b> em AP "
            f"({format_number(week_views.get('delta_ap_vs_context'), 3)} vs {format_number(week_downloads.get('delta_ap_vs_context'), 3)})."
        )
    if not parts:
        parts.append(
            "A leitura principal aqui é apenas qualitativa: alguns blocos de comportamento acrescentam sinal sobre o contexto puro, mas esse diagnóstico não substitui a avaliação principal do modelo completo."
        )
    return text_1, " ".join(parts)


def display_definition_b_excessive_separation(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    show = df[df["definition_name"] == "definition_b_label"].copy() if "definition_name" in df.columns else df.copy()
    if show.empty:
        return pd.DataFrame()
    show["problem_key"] = show["problem_key"].map(format_problem_key)
    show["track_name"] = show["track_name"].map(format_track_name)
    show["model_name"] = show["model_name"].map(format_model_name)
    show = show.rename(
        columns={
            "problem_key": "Problema",
            "track_name": "Trilha",
            "model_name": "Modelo",
            "mean_ap": "AP no teste concatenado",
            "mean_roc_auc": "ROC AUC no teste concatenado",
            "mean_brier": "Brier no teste concatenado",
            "mean_log_loss": "Log loss no teste concatenado",
            "combined_separation_score": "Score combinado de separação",
            "combined_separation_percentile_within_track": "Percentil dentro da trilha",
            "comparator_rows_in_track": "Comparadores na trilha",
            "red_flag_eligible_flag": "Red flag elegível",
            "excessive_separation_red_flag": "Red flag de separação excessiva",
        }
    )
    if "Red flag elegível" in show.columns:
        show["Red flag elegível"] = show["Red flag elegível"].map({1: "sim", 0: "não"})
    if "Red flag de separação excessiva" in show.columns:
        show["Red flag de separação excessiva"] = show["Red flag de separação excessiva"].map({1: "sim", 0: "não"})
    cols = [
        "Problema",
        "Trilha",
        "Modelo",
        "AP no teste concatenado",
        "ROC AUC no teste concatenado",
        "Brier no teste concatenado",
        "Log loss no teste concatenado",
        "Score combinado de separação",
        "Percentil dentro da trilha",
        "Comparadores na trilha",
        "Red flag elegível",
        "Red flag de separação excessiva",
    ]
    return show[[col for col in cols if col in show.columns]]


def build_operational_snapshot_text(
    reference_scope: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    confusion_df: pd.DataFrame,
    band_summary: pd.DataFrame,
) -> tuple[str, str]:
    if threshold_metrics.empty:
        return (
            "As leituras operacionais abaixo dependem de uma política registrada de cutoff. Quando essa camada ainda não foi materializada, o score contínuo continua sendo a única leitura oficial disponível.",
            "Sem política operacional materializada, também não há como resumir matriz de confusão ou faixas de risco.",
        )
    selected = None
    if not reference_scope.empty:
        for _, ref_row in reference_scope.iterrows():
            subset = threshold_metrics[
                (threshold_metrics["problem_key"] == ref_row["problem_key"])
                & (threshold_metrics["model_name"] == ref_row["model_name"])
            ].copy()
            if not subset.empty:
                selected = subset
                break
    if selected is None:
        selected = threshold_metrics.copy()
    policy_priority = {"top_10_percent": 0, "tercis": 1, "score_ge_0_70": 2}
    selected = selected.copy()
    selected["policy_priority"] = selected["policy_name"].map(lambda x: policy_priority.get(str(x), 99))
    selected = selected.sort_values(["policy_priority", "f1", "precision", "recall"], ascending=[True, False, False, False])
    best_row = selected.iloc[0]
    problem_key = best_row["problem_key"]
    model_name = best_row["model_name"]
    policy_name = best_row["policy_name"]
    confusion_subset = confusion_df[
        (confusion_df["problem_key"] == problem_key)
        & (confusion_df["model_name"] == model_name)
        & (confusion_df["policy_name"] == policy_name)
    ].copy()
    confusion_counts = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
    if not confusion_subset.empty:
        confusion_subset["cell_name"] = confusion_subset.apply(
            lambda row: confusion_cell_code(row["actual_group"], row["predicted_group"]),
            axis=1,
        )
        for cell_name, rows in confusion_subset.groupby("cell_name")["rows"].sum().items():
            confusion_counts[str(cell_name)] = int(rows)
    band_subset = band_summary[
        (band_summary["problem_key"] == problem_key)
        & (band_summary["model_name"] == model_name)
        & (band_summary["policy_name"] == policy_name)
    ].copy()
    if not band_subset.empty and "band_name" in band_subset.columns:
        band_subset = band_subset.sort_values("share", ascending=False)
        band_bits = [f"{str(row['band_name'])}: {format_percent(row['share'])}" for _, row in band_subset.head(4).iterrows()]
        band_text = ", ".join(band_bits)
    else:
        band_text = "Sem faixas materializadas para a política de referência."
    text_1 = (
        f"Como exemplo prático, a combinação de referência <code>{format_problem_key(problem_key)}</code> com "
        f"<code>{format_model_name(model_name)}</code> e política <code>{format_policy_name(policy_name)}</code> "
        f"usou cutoff {format_number(best_row['risk_threshold'])}. Nesse recorte, a leitura operacional ficou em "
        f"precisão {format_number(best_row['precision'])}, recall {format_number(best_row['recall'])}, "
        f"F1 {format_number(best_row['f1'])} e acurácia {format_number(best_row['accuracy'])}."
    )
    text_2 = (
        f"Nessa mesma política, a matriz de confusão resumida ficou em TP {confusion_counts['TP']}, FP {confusion_counts['FP']}, "
        f"TN {confusion_counts['TN']} e FN {confusion_counts['FN']}. As faixas de risco foram montadas pela política registrada "
        f"<code>{format_policy_name(policy_name)}</code>; a distribuição observada foi {band_text}. A confiabilidade dessas leituras "
        "deve ser lida junto com a variação por outer fold logo abaixo."
    )
    return text_1, text_2


def build_calibration_text(model_inner_split_audit: pd.DataFrame, model_fold_metrics: pd.DataFrame) -> tuple[str, str]:
    if model_inner_split_audit.empty:
        return (
            "A calibração oficial não foi materializada neste build.",
            "Sem auditoria de calibração materializada, o score deve ser lido só como ranking, não como probabilidade publicada.",
        )
    calibration_rows = model_inner_split_audit[model_inner_split_audit["split_strategy"] == "temporal_calibration_holdout"].copy()
    tuning_rows = model_inner_split_audit[model_inner_split_audit["split_strategy"] == "temporal_tuning_validation"].copy()
    calibration_months = sorted(calibration_rows["calibration_month_count"].dropna().astype(float).unique().tolist()) if "calibration_month_count" in calibration_rows.columns else []
    valid_calibration = int((calibration_rows.get("valid_inner_split_flag", 0) == 1).sum()) if not calibration_rows.empty else 0
    tuning_count = int((tuning_rows.get("valid_inner_split_flag", 0) == 1).sum()) if not tuning_rows.empty else 0
    invalid_outer_folds = 0
    if not model_fold_metrics.empty and "fold_valid_flag" in model_fold_metrics.columns:
        invalid_outer_folds = int((model_fold_metrics["fold_valid_flag"] == 0).sum())
    calibration_bits = ", ".join(str(int(month)) for month in calibration_months) if calibration_months else "bloco não materializado"
    text_1 = (
        "A probabilidade publicada não sai direto do estimador cru. Em cada outer fold, o treino foi quebrado em ordem temporal: "
        "primeiro veio uma validação temporal leve para escolher hiperparâmetros, e depois um bloco temporal mais recente, separado do ajuste do modelo, "
        "foi usado só para calibrar o score com <code>sigmoid</code>. É isso que permite ler o resultado como probabilidade e não só como ranking."
    )
    text_2 = (
        f"Nesta execução, a auditoria materializou {valid_calibration} blocos internos válidos de calibração e {tuning_count} blocos internos válidos de tuning temporal. "
        f"O bloco de calibração ficou em janelas de {calibration_bits} mês(es) dentro do treino. Outer folds com suporte insuficiente foram invalidados antes do resumo final "
        f"({invalid_outer_folds} fold(s) inválido(s) no escopo publicado)."
    )
    return text_1, text_2


def build_cluster_text(cluster_summary: pd.DataFrame, cluster_profile: pd.DataFrame) -> str:
    if cluster_summary.empty:
        return (
            "Clusters entram como camada descritiva sobre o mart oficial <code>mart_teacher_cluster_ready</code>. "
            "Quando essa camada não foi materializada, o relatório simplesmente não força uma leitura segmentada."
        )
    group_count = cluster_summary["cluster_name"].nunique() if "cluster_name" in cluster_summary.columns else 0
    highest_risk = None
    if "realized_inactivity_rate" in cluster_summary.columns:
        ordered = cluster_summary.sort_values("realized_inactivity_rate", ascending=False)
        if not ordered.empty:
            highest_risk = ordered.iloc[0]
    feature_names = []
    if not cluster_profile.empty and "feature_name" in cluster_profile.columns:
        feature_names = (
            cluster_profile["feature_name"].dropna().astype(str).map(format_feature_name).drop_duplicates().head(6).tolist()
        )
    if highest_risk is not None:
        highest_risk_text = (
            f"O grupo com maior taxa realizada de não atividade foi <code>{format_cluster_name(highest_risk['cluster_name'])}</code>, "
            f"com taxa {format_percent(highest_risk['realized_inactivity_rate'])} e share {format_percent(highest_risk['share'])}."
        )
    else:
        highest_risk_text = "O relatório compara score médio e taxa realizada de não atividade entre os grupos."
    if feature_names:
        feature_text = "Os perfis foram descritos a partir de sinais como " + ", ".join(feature_names) + "."
    else:
        feature_text = "Os perfis detalhados dos grupos aparecem na tabela de perfil logo abaixo."
    return (
        f"Clusters entram só como camada descritiva, construída sobre o mart <code>mart_teacher_cluster_ready</code>, e não como parte do score operacional. "
        f"Nesta materialização, apareceram {group_count} grupos. {highest_risk_text} {feature_text}"
    )


def build_heavy_user_text(heavy_user_summary: pd.DataFrame) -> str:
    if heavy_user_summary.empty:
        return (
            "A camada de heavy-user não foi materializada neste build. Quando aparece, ela é apenas exploratória e descritiva."
        )
    top10 = heavy_user_summary[heavy_user_summary["policy_name"] == "heavy_top_10_percent"].copy()
    if top10.empty:
        top10 = heavy_user_summary.copy()
    active_group = top10[top10["heavy_user_flag"] == 1].sort_values("share", ascending=False).head(1)
    baseline_group = top10[top10["heavy_user_flag"] == 0].sort_values("share", ascending=False).head(1)
    if active_group.empty or baseline_group.empty:
        return (
            "Heavy-user entra aqui só como proxy descritivo de intensidade futura. Ele não participa da escolha do score nem da decisão operacional principal."
        )
    heavy_row = active_group.iloc[0]
    base_row = baseline_group.iloc[0]
    return (
        f"Heavy-user entra aqui só como proxy descritivo de intensidade futura, não como parte do score operacional. "
        f"No corte top 10%, o grupo heavy ficou com share {format_percent(heavy_row['share'])} e taxa realizada de não atividade {format_percent(heavy_row['realized_inactivity_rate'])}, "
        f"enquanto o restante da base ficou com {format_percent(base_row['realized_inactivity_rate'])}. Em outras palavras: neste build, heavy-user ajuda a descrever quem teve uso futuro muito intenso, "
        "não a redefinir quem é ativo."
    )


def build_definition_conclusion(model_frontier: pd.DataFrame) -> str:
    frontier_only = _frontier_only(model_frontier)
    if frontier_only.empty or "definition_name" not in frontier_only.columns:
        return "Conclusão. Nesta execução, a fronteira oficial ainda não foi suficiente para sustentar uma leitura comparativa das definições."
    grouped = []
    for definition_name, group in frontier_only.groupby("definition_name", dropna=False):
        best_row = group.sort_values(
            ["mean_ap", "mean_roc_auc", "mean_brier", "mean_log_loss"],
            ascending=[False, False, True, True],
        ).iloc[0]
        grouped.append(
            {
                "definition_name": str(definition_name),
                "mean_ap": best_row.get("mean_ap", np.nan),
                "mean_roc_auc": best_row.get("mean_roc_auc", np.nan),
                "mean_brier": best_row.get("mean_brier", np.nan),
                "mean_log_loss": best_row.get("mean_log_loss", np.nan),
            }
        )
    summary = pd.DataFrame(grouped)
    best_rank = summary.sort_values(["mean_ap", "mean_roc_auc"], ascending=[False, False]).iloc[0]
    best_prob = summary.sort_values(["mean_brier", "mean_log_loss"], ascending=[True, True]).iloc[0]
    if best_rank["definition_name"] == best_prob["definition_name"]:
        return (
            f"Conclusão. Na leitura agregada da fronteira admissível, <b>{format_definition_name(best_rank['definition_name'])}</b> "
            f"ficou à frente ao mesmo tempo em ranking e em erro probabilístico."
        )
    return (
        f"Conclusão. Na fronteira admissível, <b>{format_definition_name(best_rank['definition_name'])}</b> "
        f"ficou melhor em ranking (<code>AP</code>/<code>ROC AUC</code>), enquanto "
        f"<b>{format_definition_name(best_prob['definition_name'])}</b> ficou melhor em erro probabilístico "
        f"(<code>Brier</code>/<code>log loss</code>)."
    )


def build_model_conclusion(model_frontier: pd.DataFrame) -> str:
    frontier_only = _frontier_only(model_frontier)
    if frontier_only.empty:
        return "Conclusão. Nesta execução, a fronteira oficial ainda não foi suficiente para sustentar uma leitura comparativa das famílias de modelo."
    grouped = []
    for model_name, group in frontier_only.groupby("model_name", dropna=False):
        best_row = group.sort_values(
            ["mean_ap", "mean_roc_auc", "mean_brier", "mean_log_loss"],
            ascending=[False, False, True, True],
        ).iloc[0]
        grouped.append(
            {
                "model_name": str(model_name),
                "mean_ap": best_row.get("mean_ap", np.nan),
                "mean_roc_auc": best_row.get("mean_roc_auc", np.nan),
                "mean_brier": best_row.get("mean_brier", np.nan),
                "mean_log_loss": best_row.get("mean_log_loss", np.nan),
            }
        )
    summary = pd.DataFrame(grouped)
    best_rank = summary.sort_values(["mean_ap", "mean_roc_auc"], ascending=[False, False]).iloc[0]
    best_prob = summary.sort_values(["mean_brier", "mean_log_loss"], ascending=[True, True]).iloc[0]
    if best_rank["model_name"] == best_prob["model_name"]:
        return (
            f"Conclusão. Entre as famílias de modelo, <b>{format_model_name(best_rank['model_name'])}</b> "
            f"entregou o melhor equilíbrio entre ranking e erro probabilístico dentro da fronteira admissível."
        )
    return (
        f"Conclusão. Entre as famílias de modelo, <b>{format_model_name(best_rank['model_name'])}</b> "
        f"ficou melhor em ranking, enquanto <b>{format_model_name(best_prob['model_name'])}</b> ficou melhor "
        "em erro probabilístico."
    )


def display_cv_score_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    show = df.copy()
    show["problem_key"] = show["problem_key"].map(format_problem_key)
    show["model_name"] = show["model_name"].map(format_model_name)
    metric_map = {
        "mean_score": "score médio por fold",
        "mean_risk_score": "risk score médio por fold",
        "realized_risk_rate": "taxa realizada de não atividade",
        "score_std": "dispersão interna do score",
        "risk_score_std": "dispersão interna do risk score",
    }
    show["metric_name"] = show["metric_name"].map(lambda x: metric_map.get(str(x), str(x)))
    show = show.rename(
        columns={
            "problem_key": "Problema",
            "model_name": "Modelo",
            "metric_name": "Métrica",
            "valid_folds": "Folds válidos",
            "mean_value": "Média",
            "std_value": "Desvio padrão",
            "value_range": "Amplitude",
            "max_fold_to_fold_jump": "Maior salto entre folds",
            "fold_order_slope": "Inclinação no tempo",
            "fold_order_pvalue": "p-valor da tendência",
        }
    )
    return show[["Problema", "Modelo", "Métrica", "Folds válidos", "Média", "Desvio padrão", "Amplitude", "Maior salto entre folds", "Inclinação no tempo", "p-valor da tendência"]]


def display_cv_metric_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    show = df.copy()
    show["problem_key"] = show["problem_key"].map(format_problem_key)
    show["model_name"] = show["model_name"].map(format_model_name)
    metric_map = {
        "ap": "AP",
        "roc_auc": "ROC AUC",
        "brier": "Brier",
        "log_loss": "Log loss",
        "calibration_slope": "Calibration slope",
        "calibration_intercept": "Calibration intercept",
        "calibration_slope_error": "Erro absoluto do slope",
        "calibration_intercept_abs": "Intercepto absoluto",
    }
    show["metric_name"] = show["metric_name"].map(lambda x: metric_map.get(str(x), str(x)))
    show = show.rename(
        columns={
            "problem_key": "Problema",
            "track_name": "Trilha",
            "model_name": "Modelo",
            "metric_name": "Métrica",
            "valid_folds": "Folds válidos",
            "mean_value": "Média",
            "std_value": "Desvio padrão",
            "value_range": "Amplitude",
            "max_fold_to_fold_jump": "Maior salto entre folds",
            "fold_order_slope": "Inclinação no tempo",
            "fold_order_pvalue": "p-valor da tendência",
        }
    )
    show["Trilha"] = show["Trilha"].map(format_track_name)
    return show[["Problema", "Trilha", "Modelo", "Métrica", "Folds válidos", "Média", "Desvio padrão", "Amplitude", "Maior salto entre folds", "Inclinação no tempo", "p-valor da tendência"]]


def display_cv_threshold_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    show = df.copy()
    show["problem_key"] = show["problem_key"].map(format_problem_key)
    show["model_name"] = show["model_name"].map(format_model_name)
    show["policy_name"] = show["policy_name"].map(format_policy_name)
    metric_map = {
        "risk_threshold": "Cutoff do risk score",
        "precision": "Precisão",
        "recall": "Recall",
        "f1": "F1",
        "accuracy": "Acurácia",
        "predicted_positive_rate": "Share previsto como alto risco",
    }
    show["metric_name"] = show["metric_name"].map(lambda x: metric_map.get(str(x), str(x)))
    show = show.rename(
        columns={
            "problem_key": "Problema",
            "model_name": "Modelo",
            "policy_name": "Política",
            "metric_name": "Métrica",
            "valid_folds": "Folds válidos",
            "mean_value": "Média",
            "std_value": "Desvio padrão",
            "value_range": "Amplitude",
            "max_fold_to_fold_jump": "Maior salto entre folds",
            "fold_order_slope": "Inclinação no tempo",
            "fold_order_pvalue": "p-valor da tendência",
        }
    )
    return show[["Problema", "Modelo", "Política", "Métrica", "Folds válidos", "Média", "Desvio padrão", "Amplitude", "Maior salto entre folds", "Inclinação no tempo", "p-valor da tendência"]]


def display_cv_confusion_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    show = df.copy()
    show["problem_key"] = show["problem_key"].map(format_problem_key)
    show["model_name"] = show["model_name"].map(format_model_name)
    show["policy_name"] = show["policy_name"].map(format_policy_name)
    show["cell_name"] = show.apply(lambda row: confusion_cell_code(row["actual_group"], row["predicted_group"]), axis=1)
    show = show.rename(
        columns={
            "problem_key": "Problema",
            "model_name": "Modelo",
            "policy_name": "Política",
            "cell_name": "Célula da matriz",
            "valid_folds": "Folds válidos",
            "mean_rows": "Média de linhas",
            "std_rows": "Desvio padrão",
            "max_fold_to_fold_jump": "Maior salto entre folds",
            "fold_order_slope": "Inclinação no tempo",
            "fold_order_pvalue": "p-valor da tendência",
        }
    )
    return show[["Problema", "Modelo", "Política", "Célula da matriz", "Folds válidos", "Média de linhas", "Desvio padrão", "Maior salto entre folds", "Inclinação no tempo", "p-valor da tendência"]]


def display_navigation(navigation_sequences: pd.DataFrame) -> pd.DataFrame:
    if navigation_sequences.empty:
        return pd.DataFrame()
    show = (
        navigation_sequences.sort_values(["definition_name", "label_value", "teachers"], ascending=[True, True, False])
        .groupby(["definition_name", "label_value"], group_keys=False)
        .head(5)
        .copy()
    )
    if "definition_name" in show.columns:
        show["definition_name"] = show["definition_name"].map(format_definition_name)
    if "label_value" in show.columns:
        show["label_value"] = show["label_value"].map({1: "ativo", 0: "não ativo"})
    if "step_sequence_first5" in show.columns:
        show["step_sequence_first5"] = show["step_sequence_first5"].map(format_navigation_sequence)
    show = show.rename(
        columns={
            "definition_name": "Definição",
            "label_value": "Grupo",
            "step_sequence_first5": "Sequência inicial",
            "teachers": "Professores",
        }
    )
    cols = [col for col in ["Definição", "Grupo", "Sequência inicial", "Professores"] if col in show.columns]
    return show[cols]


def display_navigation_transitions(navigation_transitions: pd.DataFrame) -> pd.DataFrame:
    if navigation_transitions.empty:
        return pd.DataFrame()
    show = (
        navigation_transitions.sort_values(["definition_name", "label_value", "teachers"], ascending=[True, True, False])
        .groupby(["definition_name", "label_value"], group_keys=False)
        .head(5)
        .copy()
    )
    if "definition_name" in show.columns:
        show["definition_name"] = show["definition_name"].map(format_definition_name)
    if "label_value" in show.columns:
        show["label_value"] = show["label_value"].map({1: "ativo", 0: "não ativo"})
    if "from_token" in show.columns:
        show["from_token"] = show["from_token"].map(format_navigation_sequence)
    if "to_token" in show.columns:
        show["to_token"] = show["to_token"].map(format_navigation_sequence)
    show = show.rename(
        columns={
            "definition_name": "Definição",
            "label_value": "Grupo",
            "from_token": "Origem",
            "to_token": "Destino",
            "teachers": "Professores",
        }
    )
    return show[["Definição", "Grupo", "Origem", "Destino", "Professores"]]


def build_method_text(model_frontier: pd.DataFrame, summary: dict[str, Any]) -> tuple[str, str]:
    frontier_only = model_frontier[model_frontier["pareto_frontier_flag"] == 1].copy() if not model_frontier.empty and "pareto_frontier_flag" in model_frontier.columns else pd.DataFrame()
    if frontier_only.empty:
        return (
            "O build oficial não encontrou score publicável depois de aplicar a validação temporal e a calibração oficial.",
            "Sem score oficial, a leitura correta é que o método oficial ainda não sustentou um problema/modelo publicável.",
        )
    text_1 = (
        "A comparação oficial foi feita sempre fora do tempo de treino. Em cada outer fold, o modelo viu apenas meses anteriores, "
        "fez tuning temporal leve dentro do treino e foi avaliado no mês seguinte, nunca visto antes. O resumo principal publicado aqui "
        "usa teste futuro concatenado pooled, e não média simples de fold."
    )
    text_2 = (
        "A hierarquia de decisão é esta: primeiro entram AP, ROC AUC, Brier e log loss no teste futuro concatenado; depois entram calibração e robustez; "
        "só depois vêm cutoff, matriz de confusão, faixas e ajuste mensal. Isso evita escolher modelo por um detalhe operacional isolado e perder a qualidade da probabilidade contínua."
    )
    return text_1, text_2


def main() -> None:
    args = parse_args()
    build_dir = (args.build_dir or (PROJECT_DIR / "build")).resolve()
    summary = read_summary(build_dir)
    track_registry = read_table(build_dir, "governance_track_registry_v1")
    arbitrariness = read_table(build_dir, "governance_arbitrariness_registry_v1")
    policy_registry = read_table(build_dir, "governance_policy_registry_v1")
    feature_registry = read_table(build_dir, "governance_feature_registry_v1")
    label_registry = read_table(build_dir, "governance_label_registry_v1")
    definition_selection = read_table(build_dir, "core_definition_selection_v1")
    definition_frontier = read_table(build_dir, "core_definition_frontier_v1")
    definition_external_validation = read_table(build_dir, "core_definition_external_validation_v1")
    scoring_scenarios = read_table(build_dir, "core_scoring_scenarios_v1")
    model_fold_metrics = read_table(build_dir, "core_model_fold_metrics_v1")
    model_inner_split_audit = read_table(build_dir, "core_model_calibration_audit_v1")
    model_frontier = read_table(build_dir, "core_model_frontier_v1")
    predictions = read_table(build_dir, "core_model_predictions_v1")
    cv_score_folds = read_table(build_dir, "core_cv_score_folds_v1")
    cv_score_summary = read_table(build_dir, "core_cv_score_summary_v1")
    cv_metric_folds = read_table(build_dir, "core_cv_metric_folds_v1")
    cv_metric_summary = read_table(build_dir, "core_cv_metric_summary_v1")
    leakage_audit = read_table(build_dir, "governance_leakage_audit_v1")
    leakage_summary = read_table(build_dir, "governance_leakage_summary_v1")
    definition_b_feature_block_gain_summary = read_table(build_dir, "core_definition_b_feature_block_gain_summary_v1")
    definition_b_feature_block_gain_summary_full = definition_b_feature_block_gain_summary.copy()
    definition_b_excessive_separation = read_table(build_dir, "core_definition_b_excessive_separation_v1")
    navigation_sequences = read_table(build_dir, "core_navigation_sequences_v1")
    navigation_transitions = read_table(build_dir, "core_navigation_transitions_v1")
    bootstrap = read_table(build_dir, "core_prediction_bootstrap_v1")
    threshold_metrics = read_table(build_dir, "post_model_threshold_metrics_v1")
    confusion_df = read_table(build_dir, "post_model_confusion_matrix_v1")
    band_summary = read_table(build_dir, "post_model_band_summary_v1")
    monthly_fit = read_table(build_dir, "post_model_monthly_fit_v1")
    cv_threshold_folds = read_table(build_dir, "post_model_cv_threshold_folds_v1")
    cv_threshold_summary = read_table(build_dir, "post_model_cv_threshold_summary_v1")
    cv_confusion_folds = read_table(build_dir, "post_model_cv_confusion_folds_v1")
    cv_confusion_summary = read_table(build_dir, "post_model_cv_confusion_summary_v1")
    feature_importance = read_table(build_dir, "post_model_feature_importance_v1")
    reference_scope = read_reference_scope(build_dir)
    cluster_summary = read_table(build_dir, "post_model_cluster_summary_v1")
    cluster_profile = read_table(build_dir, "post_model_cluster_profile_v1")
    cluster_validation = read_table(build_dir, "post_model_cluster_validation_v1")
    heavy_user_summary = read_table(build_dir, "post_model_heavy_user_summary_v1")
    heavy_user_profile = read_table(build_dir, "post_model_heavy_user_profile_v1")

    definition_frontier = filter_to_reference_definitions(definition_frontier, reference_scope)
    definition_selection = filter_to_reference_definitions(definition_selection, reference_scope)
    definition_external_validation = filter_to_reference_definitions(definition_external_validation, reference_scope)
    scoring_scenarios = filter_to_reference_scope(scoring_scenarios, reference_scope)
    model_fold_metrics = filter_to_reference_scope(model_fold_metrics, reference_scope)
    model_inner_split_audit = filter_to_reference_scope(model_inner_split_audit, reference_scope)
    model_frontier = filter_to_reference_scope(model_frontier, reference_scope)
    predictions = filter_to_reference_scope(predictions, reference_scope)
    cv_score_folds = filter_to_reference_scope(cv_score_folds, reference_scope)
    cv_score_summary = filter_to_reference_scope(cv_score_summary, reference_scope)
    cv_metric_folds = filter_to_reference_scope(cv_metric_folds, reference_scope)
    cv_metric_summary = filter_to_reference_scope(cv_metric_summary, reference_scope)
    leakage_audit = filter_to_reference_scope(leakage_audit, reference_scope)
    leakage_summary = filter_to_reference_scope(leakage_summary, reference_scope)
    definition_b_feature_block_gain_summary = filter_to_reference_scope(definition_b_feature_block_gain_summary, reference_scope)
    definition_b_excessive_separation = filter_to_reference_scope(definition_b_excessive_separation, reference_scope)
    navigation_sequences = filter_to_reference_definitions(navigation_sequences, reference_scope)
    navigation_transitions = filter_to_reference_definitions(navigation_transitions, reference_scope)
    bootstrap = filter_to_reference_scope(bootstrap, reference_scope)
    threshold_metrics = filter_to_reference_scope(threshold_metrics, reference_scope)
    confusion_df = filter_to_reference_scope(confusion_df, reference_scope)
    band_summary = filter_to_reference_scope(band_summary, reference_scope)
    monthly_fit = filter_to_reference_scope(monthly_fit, reference_scope)
    cv_threshold_folds = filter_to_reference_scope(cv_threshold_folds, reference_scope)
    cv_threshold_summary = filter_to_reference_scope(cv_threshold_summary, reference_scope)
    cv_confusion_folds = filter_to_reference_scope(cv_confusion_folds, reference_scope)
    cv_confusion_summary = filter_to_reference_scope(cv_confusion_summary, reference_scope)
    feature_importance = filter_to_reference_scope(feature_importance, reference_scope)
    cluster_summary = filter_to_reference_scope(cluster_summary, reference_scope)
    cluster_profile = filter_to_reference_scope(cluster_profile, reference_scope)
    cluster_validation = filter_to_reference_scope(cluster_validation, reference_scope)
    heavy_user_summary = filter_to_reference_scope(heavy_user_summary, reference_scope)
    heavy_user_profile = filter_to_reference_scope(heavy_user_profile, reference_scope)

    presentation_scope = select_presentation_scope(reference_scope)
    presentation_definition_frontier = filter_to_reference_definitions(definition_frontier, presentation_scope)
    presentation_scoring_scenarios = filter_to_reference_scope(scoring_scenarios, presentation_scope)
    presentation_model_frontier = filter_to_reference_scope(model_frontier, presentation_scope)
    presentation_predictions = filter_to_reference_scope(predictions, presentation_scope)
    presentation_cv_score_folds = filter_to_reference_scope(cv_score_folds, presentation_scope)
    presentation_cv_score_summary = filter_to_reference_scope(cv_score_summary, presentation_scope)
    presentation_cv_metric_folds = filter_to_reference_scope(cv_metric_folds, presentation_scope)
    presentation_cv_metric_summary = filter_to_reference_scope(cv_metric_summary, presentation_scope)
    presentation_leakage_audit = filter_to_reference_scope(leakage_audit, presentation_scope)
    presentation_leakage_summary = filter_to_reference_scope(leakage_summary, presentation_scope)
    presentation_definition_b_feature_block_gain_summary = filter_definition_b_feature_block_gain_for_report(
        definition_b_feature_block_gain_summary_full,
        presentation_scope,
    )
    presentation_definition_b_excessive_separation = filter_to_reference_scope(definition_b_excessive_separation, presentation_scope)
    presentation_navigation_sequences = filter_to_reference_definitions(navigation_sequences, presentation_scope)
    presentation_navigation_transitions = filter_to_reference_definitions(navigation_transitions, presentation_scope)
    presentation_bootstrap = filter_to_reference_scope(bootstrap, presentation_scope)
    presentation_threshold_metrics = filter_to_reference_scope(threshold_metrics, presentation_scope)
    presentation_confusion_df = filter_to_reference_scope(confusion_df, presentation_scope)
    presentation_band_summary = filter_to_reference_scope(band_summary, presentation_scope)
    presentation_monthly_fit = filter_to_reference_scope(monthly_fit, presentation_scope)
    presentation_cv_threshold_folds = filter_to_reference_scope(cv_threshold_folds, presentation_scope)
    presentation_cv_threshold_summary = filter_to_reference_scope(cv_threshold_summary, presentation_scope)
    presentation_cv_confusion_folds = filter_to_reference_scope(cv_confusion_folds, presentation_scope)
    presentation_cv_confusion_summary = filter_to_reference_scope(cv_confusion_summary, presentation_scope)
    presentation_feature_importance = filter_to_reference_scope(feature_importance, presentation_scope)
    presentation_cluster_summary = filter_to_reference_scope(cluster_summary, presentation_scope)
    presentation_cluster_profile = filter_to_reference_scope(cluster_profile, presentation_scope)
    presentation_cluster_validation = filter_to_reference_scope(cluster_validation, presentation_scope)
    presentation_heavy_user_summary = filter_to_reference_scope(heavy_user_summary, presentation_scope)
    presentation_heavy_user_profile = filter_to_reference_scope(heavy_user_profile, presentation_scope)

    intro_rows = build_intro_rows(summary, track_registry, presentation_definition_frontier, presentation_model_frontier)
    method_text_1, method_text_2 = build_method_text(model_frontier, summary)
    operational_text_1, operational_text_2 = build_operational_snapshot_text(presentation_scope, presentation_threshold_metrics, presentation_confusion_df, presentation_band_summary)
    cluster_text = build_cluster_text(presentation_cluster_summary, presentation_cluster_profile)
    heavy_user_text = build_heavy_user_text(presentation_heavy_user_summary)
    definition_b_feature_block_text_1, definition_b_feature_block_text_2 = build_definition_b_feature_block_gain_text(
        presentation_definition_b_feature_block_gain_summary,
    )
    calibration_text_1, calibration_text_2 = build_calibration_text(model_inner_split_audit, model_fold_metrics)
    validator_guide = build_external_validator_guide()
    selected_problem_model_comparison = display_selected_problem_model_comparison(presentation_model_frontier, presentation_scope)
    selected_problem_operational_comparison = display_selected_problem_operational_comparison(presentation_predictions, presentation_scope)
    selected_problem_story = build_problem_level_model_text(
        selected_problem_model_comparison,
        selected_problem_operational_comparison,
        presentation_scope,
    )
    definition_conclusion = build_definition_conclusion(presentation_model_frontier)
    model_conclusion = build_model_conclusion(presentation_model_frontier)
    definition_family_summary = display_definition_family_summary(presentation_model_frontier, presentation_scoring_scenarios)
    model_family_summary = display_model_family_summary(presentation_model_frontier, presentation_scoring_scenarios)
    definition_answer = build_definition_answer(definition_family_summary)
    model_answer = build_model_answer(model_family_summary)
    definition_a_rule = ""
    definition_b_rule = ""
    if not presentation_definition_frontier.empty:
        a_rows = presentation_definition_frontier[presentation_definition_frontier["definition_name"].astype(str).str.startswith("definition_a")].copy()
        b_rows = presentation_definition_frontier[presentation_definition_frontier["definition_name"].astype(str).str.startswith("definition_b")].copy()
        if not a_rows.empty:
            definition_a_rule = format_rule_text(a_rows.iloc[0].get("rule_text", ""))
        if not b_rows.empty:
            definition_b_rule = format_rule_text(b_rows.iloc[0].get("rule_text", ""))

    definition_external_display = definition_external_validation.copy()
    if not definition_external_display.empty:
        if "definition_name" in definition_external_display.columns:
            definition_external_display["definition_name"] = definition_external_display["definition_name"].map(format_definition_name)
        if "split_role" in definition_external_display.columns:
            definition_external_display["split_role"] = definition_external_display["split_role"].replace({"train": "treino", "test": "teste"})
        if "metric_name" in definition_external_display.columns:
            definition_external_display["metric_name"] = definition_external_display["metric_name"].map(format_metric_name)
        definition_external_display = definition_external_display.rename(
            columns={
                "definition_name": "Definição",
                "split_role": "Papel do fold",
                "fold_id": "Fold",
                "metric_name": "Métrica futura",
                "threshold": "Corte",
                "gap_returned_active_post_label_m1": "Gap retorno M+1",
                "gap_returned_active_post_label_m2": "Gap retorno M+2",
                "gap_returned_active_post_label_m3": "Gap retorno M+3",
                "gap_active_days_post_label_3m": "Gap dias ativos 3m",
                "gap_sustained_active_2of3_post_label": "Gap sustentação 2 de 3",
                "candidate_valid_flag": "Label válido no fold",
            }
        )
        wanted = [
            "Definição",
            "Papel do fold",
            "Fold",
            "Métrica futura",
            "Corte",
            "Gap retorno M+1",
            "Gap retorno M+2",
            "Gap retorno M+3",
            "Gap dias ativos 3m",
            "Gap sustentação 2 de 3",
            "Label válido no fold",
        ]
        definition_external_display = definition_external_display[[c for c in wanted if c in definition_external_display.columns]]

    report_title = "Previsão de atividade futura"
    subtitle = "Build único com base modelada oficial, comparação temporal entre Definição A e Definição B, score contínuo calibrado e relatório final em português."
    label_registry_detail = render_details(
        "Ver tabela técnica dos rótulos",
        render_clean_table(display_label_registry(label_registry)),
    )
    definition_validation_detail = render_details(
        "Ver tabela técnica da validação externa por fold",
        render_clean_table(definition_external_display, limit=24),
    )
    validator_guide_detail = render_details(
        "Ver definição de cada validador externo",
        render_clean_table(validator_guide),
        open_by_default=True,
    )
    calibration_audit_detail = render_details(
        "Ver auditoria técnica do bloco temporal de calibração",
        render_clean_table(display_inner_split_summary(model_inner_split_audit), limit=20),
    )
    arbitrariness_detail = render_details(
        "Ver tabela técnica das convenções",
        render_clean_table(display_arbitrariness(arbitrariness), limit=20),
    )
    track_detail = render_details(
        "Ver trilhas oficiais",
        render_clean_table(display_track_registry(track_registry)),
    )
    feature_detail = render_details(
        "Ver variáveis elegíveis por bloco e trilha",
        render_clean_table(display_feature_summary(feature_registry)),
    )
    policy_detail = render_details(
        "Ver políticas registradas do build",
        render_clean_table(display_policy_registry(policy_registry), limit=20),
    )
    strict_context_detail = render_details(
        "Ver exatamente o que entra no STRICT_CONTEXT",
        render_clean_table(display_strict_context_features(feature_registry)),
        open_by_default=True,
    )
    model_fold_validity_detail = render_details(
        "Ver validade dos outer folds",
        render_clean_table(display_model_fold_validity(model_fold_metrics), limit=20),
    )

    operational_sections = f"""
    <section>
      <h2>Uso do score</h2>
      <div class="chart-card">
        <p class="section-text">O produto central continua sendo o <b>score contínuo calibrado</b>. Aqui, <code>score</code> é a probabilidade calibrada de realizar a atividade futura e <code>risk_score = 1 - score</code> é a probabilidade calibrada de não realizar. Essa leitura como probabilidade só é aceitável porque o score bruto foi recalibrado em um bloco temporal separado do treino e depois checado em meses futuros nunca vistos.</p>
        <h3>Como o score é composto e para que ele serve</h3>
        <h4>1. Entrada do modelo</h4>
        <p class="section-text">Em <code>S1</code>, entram apenas sinais disponíveis até o fim da 1ª sessão. Em <code>S7</code>, entram apenas sinais disponíveis até o fim dos primeiros 7 dias. Em <code>S1+S7</code>, os dois blocos entram juntos. Em <code>STRICT_CONTEXT</code>, entram só variáveis já conhecidas no começo da jornada ou flags de <b>completude do contexto inicial</b>, como origem, dispositivo e ausência de metadados básicos. Ele funciona como o cenário mais conservador do build, porque exclui sinais de comportamento do produto.</p>
        {strict_context_detail}
        <h4>2. Score contínuo</h4>
        <p class="section-text">O uso central do score é ordenar a base e comparar professores com mais ou menos chance de voltar a usar produto. A interpretação correta é sempre esta: quanto maior o <code>score</code>, maior a chance de realizar a atividade futura; quanto maior o <code>risk_score</code>, maior o risco de não realizar.</p>
        <h4>3. Uso prático</h4>
        <p class="section-text">Cutoffs, faixas e outras leituras operacionais entram depois, como políticas registradas por cima desse núcleo. Por isso, primeiro o relatório mostra a comparação probabilística principal; só depois entra em cutoff, matriz de confusão, bandas e ajuste mensal.</p>
        {render_clean_table(display_reference_scope(presentation_scope), limit=4)}
        <p class="section-text">{operational_text_1}</p>
        <p class="section-text">{operational_text_2}</p>
        <h3>Como os modelos finais se comportaram sob as mesmas políticas</h3>
        <p class="section-text">A tabela abaixo compara, nas combinações finais, como cada modelo se comporta quando entra uma política operacional comparável. É aqui que aparecem juntos <code>F1</code>, matriz de confusão resumida e ajuste mensal por <code>R2</code>/<code>MAPE</code>.</p>
        {render_clean_table(selected_problem_operational_comparison, limit=12)}
        <h3>Onde o modelo de ativação acerta e erra no teste futuro</h3>
        <p class="section-text">As tabelas abaixo mostram o que acontece quando a probabilidade contínua vira decisão por cutoff. Aqui, a <b>classe positiva é “não realiza”</b>. Então: <b>TP</b> = marcou alto risco e depois realmente não realizou; <b>FP</b> = marcou alto risco e depois realizou; <b>TN</b> = não marcou alto risco e depois realizou; <b>FN</b> = não marcou alto risco e depois não realizou.</p>
        <p class="section-text">Importante: <b>matriz de confusão e métricas por cutoff não escolhem a definição de atividade</b>. Elas entram depois, como leitura operacional de um score contínuo já escolhido, porque dependem da política de corte.</p>
        <h3>Precision, recall, F1 e accuracy por cutoff</h3>
        {render_clean_table(display_threshold_metrics(presentation_threshold_metrics), limit=4)}
        <h3>Matriz de confusão por cutoff</h3>
        <p class="section-text">A matriz abaixo já está resumida em <code>TP</code>, <code>FP</code>, <code>TN</code> e <code>FN</code> para facilitar a leitura de quem não quer navegar pela matriz aberta linha a linha.</p>
        {render_clean_table(display_confusion_matrix(presentation_confusion_df), limit=4)}
        <h3>Como cutoff e métricas operacionais variaram entre folds</h3>
        <p class="section-text">Os outer folds também funcionam aqui como pequenas variações do contexto de uso. A pergunta é: quando o mês muda, a política operacional continua parecida ou muda demais?</p>
        <div class="embedded-chart-wrap">
          {render_plotly(presentation_cv_threshold_folds, "cv_threshold_drift")}
        </div>
        {render_details("Ver tabela do drift operacional por cutoff", render_clean_table(display_cv_threshold_summary(presentation_cv_threshold_summary), limit=12))}
        <h3>Como a matriz de confusão variou entre folds</h3>
        <p class="section-text">Os outer folds funcionam aqui como pequenas variações temporais do problema: muda o mês de teste, muda um pouco a composição da base e vemos se a leitura operacional continua parecida. Quando a matriz de confusão salta demais entre folds, a política fica menos confiável para uso recorrente.</p>
        <div class="embedded-chart-wrap">
          {render_plotly(presentation_cv_confusion_folds, "cv_confusion_drift")}
        </div>
        {render_details("Ver tabela do drift da matriz de confusão", render_clean_table(display_cv_confusion_summary(presentation_cv_confusion_summary), limit=12))}
        <h3>Como o score foi separado em faixas e o que aconteceu em cada faixa</h3>
        <p class="section-text">As faixas não substituem o score; elas só resumem a distribuição operacional da base. A leitura útil é: qual share da base caiu em cada faixa e como a taxa realizada muda quando saímos das faixas de menor risco para as de maior risco.</p>
        <div class="embedded-chart-wrap">
          {render_plotly(presentation_band_summary, "band_summary")}
        </div>
        {render_details("Ver tabela completa das faixas", render_clean_table(display_band_summary(presentation_band_summary), limit=12))}
        <h3>R2 e MAPE em risco mensal realizado</h3>
        <p class="section-text"><code>R2</code> e <code>MAPE</code> não são métricas linha a linha do alvo binário. Aqui eles medem uma coisa diferente: o quanto a média mensal do <code>risk_score</code> acompanha a taxa mensal observada de não realização. Isso ajuda a ver se o score continua útil para leitura agregada de risco ao longo do tempo.</p>
        <div class="embedded-chart-wrap">
          {render_plotly(presentation_monthly_fit, "monthly_fit")}
        </div>
        {render_details("Ver tabela do ajuste mensal", render_clean_table(display_monthly_fit(presentation_monthly_fit), limit=12))}
      </div>
    </section>

    <section>
        <h2>Leitura do comportamento</h2>
      <div class="chart-card">
        <p class="section-text">Esta seção ajuda a interpretar o score sem redefinir o núcleo oficial. Tudo que aparece aqui é leitura complementar: ajuda a entender quais sinais parecem puxar mais o risco, quem concentra intensidade futura e quais caminhos iniciais aparecem com mais frequência, mas não muda a definição oficial nem o modelo escolhido.</p>
        <h3>Quais sinais mais puxam o score de ativação</h3>
        <p class="section-text">A importância por permutação mostra quanto o erro probabilístico piora quando um sinal é embaralhado no conjunto de teste externo. Ela serve para interpretação e robustez, não para decidir se a variável entra ou sai do caminho oficial.</p>
        <div class="embedded-chart-wrap">
          {render_plotly(presentation_feature_importance, "feature_importance")}
        </div>
        {render_details("Ver tabela da importância por permutação", render_clean_table(display_feature_importance(presentation_feature_importance), limit=12))}
        <h3>O que a 1ª sessão e os 7 dias acrescentam além do contexto</h3>
        <p class="section-text">{definition_b_feature_block_text_1}</p>
        <p class="section-text">{definition_b_feature_block_text_2}</p>
        {render_details("Ver teste por blocos de features da Definição B", render_clean_table(display_definition_b_feature_block_gain(presentation_definition_b_feature_block_gain_summary), limit=12))}
        <h3>Heavy-user descritivo</h3>
        <p class="section-text">{heavy_user_text}</p>
        {render_clean_table(display_heavy_user_summary(presentation_heavy_user_summary), limit=6)}
        {render_details("Ver perfil técnico do heavy-user", render_clean_table(display_heavy_user_profile(presentation_heavy_user_profile), limit=12))}
        <h3>Clusters descritivos complementares</h3>
        <p class="section-text">{cluster_text}</p>
        {render_clean_table(display_cluster_summary(presentation_cluster_summary), limit=4)}
        {render_details("Ver perfil detalhado dos clusters", render_clean_table(display_cluster_profile(presentation_cluster_profile), limit=16))}
        {render_details("Ver validação técnica dos clusters", render_clean_table(display_cluster_validation(presentation_cluster_validation), limit=12))}
        <h3>Sequências iniciais mais frequentes</h3>
        <p class="section-text">Aqui a comparação é entre <b>ativo</b> e <b>não ativo</b> realizados no futuro, não entre grupos artificiais de risco. A ideia é ver quais caminhos iniciais aparecem mais em quem depois ficou ativo e em quem depois não ficou.</p>
        {render_clean_table(display_navigation(presentation_navigation_sequences), limit=10)}
        {render_details("Ver transições iniciais mais frequentes", render_clean_table(display_navigation_transitions(presentation_navigation_transitions), limit=12))}
      </div>
    </section>
"""

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{report_title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #F7FAFC; color: #14213D; }}
    .container {{ max-width: 1360px; margin: 0 auto; padding: 28px; }}
    h1 {{ margin: 0 0 18px 0; font-size: 32px; color: #102A43; }}
    h2 {{ margin: 36px 0 10px 0; font-size: 25px; color: #102A43; }}
    h3 {{ margin: 0 0 16px 0; font-size: 20px; color: #102A43; }}
    h4 {{ margin: 14px 0 8px 0; font-size: 16px; color: #102A43; }}
    .subtitle {{ margin: 0 0 14px 0; color: #486581; font-size: 14px; }}
    .section-text {{ margin: 0 0 16px 0; color: #486581; font-size: 16px; line-height: 1.55; }}
    .intro-card {{ background: white; border: 1px solid #D9E2EC; border-radius: 14px; padding: 20px 22px; margin-bottom: 28px; }}
    .intro-card h2 {{ margin: 0 0 12px 0; font-size: 27px; }}
    .intro-card p {{ margin: 10px 0; font-size: 16px; line-height: 1.65; color: #243B53; }}
    .clean-table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    .clean-table th {{ text-align: left; background: #102A43; color: white; padding: 12px 14px; font-size: 14px; }}
    .clean-table td {{ vertical-align: top; padding: 12px 14px; border-bottom: 1px solid #D9E2EC; font-size: 14px; line-height: 1.6; color: #243B53; }}
    .clean-table td:first-child {{ width: 220px; font-weight: 700; color: #102A43; }}
    .chart-card {{ background: white; border: 1px solid #D9E2EC; border-radius: 14px; padding: 18px 20px; margin: 18px 0 34px 0; }}
    .note {{ background: #E6FFFA; border: 1px solid #81E6D9; border-radius: 10px; padding: 12px 14px; margin: 18px 0 16px 0; font-size: 15px; line-height: 1.6; color: #234E52; }}
    .lineage {{ background: #F8FAFC; border: 1px solid #D9E2EC; border-radius: 10px; padding: 12px 14px; margin-top: 18px; }}
    .lineage p {{ margin: 8px 0; font-size: 13.5px; line-height: 1.55; color: #334E68; }}
    .two-col {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 28px; align-items: start; }}
    .definition-card {{ background: #F8FAFC; border: 1px solid #D9E2EC; border-radius: 12px; padding: 14px 16px; }}
    .definition-card h4 {{ margin: 0 0 8px 0; font-size: 16px; color: #102A43; }}
    .definition-card p {{ margin: 0; font-size: 14px; line-height: 1.55; color: #334E68; }}
    .embedded-chart-wrap {{ margin-top: 10px; }}
    .chart-card .plotly-graph-div {{ margin-top: 10px; }}
    .detail-card {{ margin-top: 14px; border: 1px solid #D9E2EC; border-radius: 10px; background: #FAFCFF; padding: 10px 12px; }}
    .detail-card summary {{ cursor: pointer; font-weight: 700; color: #102A43; }}
    .detail-card[open] summary {{ margin-bottom: 10px; }}
    code {{ background: #F0F4F8; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>{report_title}</h1>
    <p class="subtitle">{subtitle}</p>

    <div class="intro-card">
      {render_intro_table(intro_rows)}
    </div>

    <section>
      <h2>Leitura executiva</h2>
      <div class="chart-card">
        <p class="section-text">Este relatório responde a uma pergunta simples: <b>dado o que o professor fez no começo da jornada, qual é a probabilidade de voltar e usar produto de forma observável depois disso?</b></p>
        <p class="section-text">A leitura correta para alguém totalmente de fora é esta: <b>1)</b> entender qual atividade futura está sendo chamada de sucesso, <b>2)</b> ver como essa probabilidade se comportou em teste futuro de verdade, <b>3)</b> só então olhar cutoff, faixas, heavy-user, cluster e navegação como <b>camadas complementares</b>.</p>
        <p class="section-text">Nesta versão final, o relatório publica uma <b>Definição A oficial</b>, mais estrita que a Definição B, e mantém a <b>Definição B</b> como comparador. No lado do modelo, a leitura principal mostra o melhor par <b>definição + trilha + modelo</b> para cada uma delas; o restante entra como leitura complementar, não como critério central.</p>
        <div class="note"><b>Como ler.</b> Primeiro leia <b>Escolha do alvo</b>. Depois veja <b>Desempenho do modelo</b> para entender por que a probabilidade publicada é confiável. Só então entre em <b>Uso do score</b> e <b>Leitura do comportamento</b>.</div>
      </div>
    </section>

    <section>
        <h2>Escolha do alvo</h2>
      <div class="chart-card">
        <h3>O que chamamos de atividade futura aqui</h3>
        <p class="section-text">A pergunta central do build foi: <b>qual definição de “ativo” faz mais sentido para o negócio, sem leakage, e continua separando melhor o comportamento futuro?</b> Para responder isso, o pipeline comparou a Definição B literal contra uma família de candidatas para a Definição A, geradas a partir de métricas futuras nativas como semanas ativas, dias ativos, minutos de sessão, diversidade de ações e outros sinais de uso observável na janela futura.</p>
        <p class="section-text">O protocolo foi este: gerar candidatas com cortes observados na própria base, deduplicar regras que marcavam exatamente o mesmo vetor de usuários, descartar candidatas que escorregavam para <b>superatividade</b> estreita demais e, no conjunto sobrevivente, comparar prevalência, leakage estrutural, validadores externos pós-label e sensibilidade a pequenas variações dos cortes. A Definição A oficial foi escolhida porque ficou <b>mais estrita que a B</b>, continuou com escala de negócio plausível e foi a que melhor sustentou a combinação de <b>recorrência + profundidade mínima</b> sem virar uma regra extrema de elite de uso.</p>
        <div class="two-col">
          <div class="definition-card">
            <h4>Definição A</h4>
            <p><b>Regra vencedora:</b> <code>{definition_a_rule}</code>. Em português: o professor precisa mostrar recorrência futura e também um mínimo de profundidade futura. Essa foi a regra que ficou mais robusta a pequenas variações sem ficar trivial nem instável demais.</p>
          </div>
          <div class="definition-card">
            <h4>Definição B</h4>
            <p><b>Regra comparadora:</b> <code>{definition_b_rule}</code>. Em português: o professor é considerado ativo se tiver pelo menos uma semana futura com sessão e atividade observável na mesma semana. Ela entra como baseline literal e comparador fixo.</p>
          </div>
        </div>
        <h3>Quando cada previsão pode ser feita e quando cada resultado é medido</h3>
        <p class="section-text">O caminho oficial usa uma única cadeia: <code>base_modelada_v2</code> para ML e ML para HTML. Primeiro, o build reconstrói o começo da jornada do professor. Depois, define até onde a entrada pode ir em cada trilha. Só então mede o resultado futuro depois dessa janela inicial. Isso garante que a previsão seja feita antes do resultado, e não misturada com ele.</p>
        <p class="section-text">As variáveis de entrada do score não entram por “teste mágico”. Elas entram porque fazem parte do contrato oficial do problema, estão disponíveis no momento do score e passam na checagem de <code>PIT-safe</code>, isto é, não usam informação do futuro. A tabela abaixo mostra exatamente quais variáveis entram em cada bloco e em quais trilhas elas podem ser usadas.</p>
        <div class="lineage">
          <p><b>Base modelada usada:</b> <code>base_modelada_v2</code>, <code>dim_teacher</code>, <code>fct_session_clean</code>, <code>fct_interaction_clean</code>, <code>fct_formation_clean</code>, <code>fct_mari_conversation_resolved</code>, <code>fct_mari_help_resolved</code>, <code>mart_teacher_cluster_ready</code> e <code>mart_teacher_persona_ready</code>.</p>
          <p><b>Marts desta trilha:</b> <code>mart_onboarding_population_v1</code>, <code>mart_first_session_journey_v1</code>, <code>mart_future_metrics_v1</code>.</p>
        </div>
        {track_detail}
        {feature_detail}
        {policy_detail}
        <p class="section-text">A tabela abaixo mostra <b>o que é sucesso futuro</b>, em que janela ele é medido e em qual base ele foi calculado. Isso é importante porque o score é treinado para prever exatamente esse resultado.</p>
        {label_registry_detail}
        <h3>Validação externa das definições</h3>
        <p class="section-text">Depois que a janela principal do label termina, o build ainda olha <b>três blocos adicionais de 30 dias</b>. A ideia é simples: se uma definição realmente faz sentido, quem foi marcado como ativo por ela deveria continuar mostrando mais retorno e mais sustentação também depois da janela que criou o próprio label. Isso impede escolher uma definição que “parece boa” só porque está colada no próprio rótulo.</p>
        {validator_guide_detail}
        <p class="section-text">Na prática, a comparação final entre A e B ficou assim: a Definição A oficial marca menos usuários como ativos, mas entrega gaps maiores nos validadores externos, o que sustenta a leitura de que ela é uma versão mais exigente e mais forte de atividade futura.</p>
        {render_clean_table(display_definition_frontier(presentation_definition_frontier), limit=4)}
        {definition_validation_detail}
        <div class="note"><b>{definition_answer}</b></div>
        <p class="section-text">Aqui, “melhor definição” quer dizer a melhor leitura <b>na família da definição</b> depois do protocolo temporal completo. A escolha do alvo olha principalmente validadores externos, modelabilidade e robustez. Matriz de confusão, bandas e cutoff entram depois, como uso do score.</p>
        <div class="note"><b>{definition_conclusion}</b></div>
      </div>
    </section>

    <section>
      <h2>Desempenho do modelo</h2>
      <div class="chart-card">
        <p class="section-text">{method_text_1}</p>
        <p class="section-text">{method_text_2}</p>
        <h3>Qualidade dos modelos</h3>
        <p class="section-text">Em cada outer fold, o problema é separado em quatro partes, sempre respeitando o tempo: <b>treino para ajuste</b>, <b>validação temporal interna para tuning</b>, <b>holdout temporal de calibração</b> e <b>teste futuro final</b>. O mês de teste nunca entra nem no tuning nem na calibração.</p>
        <p class="section-text">Pré-processamento também respeita essa separação. Imputação, padronização e one-hot encoding ficam dentro de <code>Pipeline</code> e <code>ColumnTransformer</code>, então são ajustados apenas com dados de treino antes de serem aplicados ao teste. A calibração usa <code>sigmoid</code> em um bloco temporal mais recente dentro do treino, separado do ajuste do estimador cru.</p>
        <p class="section-text">{calibration_text_1}</p>
        <p class="section-text">{calibration_text_2}</p>
        <p class="section-text">O problema é desbalanceado, então o relatório publica quantos casos ficaram como <b>ativos futuros</b> e <b>não ativos futuros</b> em cada cenário. No núcleo do modelo, a classe positiva é <b>realiza a atividade futura</b>. Já na camada operacional de risco, a classe positiva vira <b>não realiza</b>, porque o uso de cutoff e matriz de confusão está olhando risco de não retorno.</p>
        {render_clean_table(display_scenario_balance(presentation_scoring_scenarios), limit=8)}
        <p class="section-text">O build não usa reamostragem nem SMOTE. <code>class_weight</code> pode entrar quando o tuning temporal escolhe isso para uma família de modelo, mas o resumo principal continua priorizando métricas compatíveis com probabilidade e evento raro, como <code>AP</code>, <code>Brier</code> e <code>log loss</code>.</p>
        <h3>Drift do score dentro do CV</h3>
        <p class="section-text">Aqui, cada outer fold funciona como uma pequena mudança temporal na amostra. O relatório mostra a média do <code>risk_score</code> por fold, o maior salto de um fold para o seguinte e um p-valor por permutação para tendência temporal. A pergunta é simples: quando o mês muda, o score muda pouco e de forma gradual, ou pula demais?</p>
        <div class="embedded-chart-wrap">
          {render_plotly(presentation_cv_score_folds, "cv_score_drift")}
        </div>
        {render_details("Ver tabela do drift do score por fold", render_clean_table(display_cv_score_summary(presentation_cv_score_summary), limit=12))}
        <h3>Drift das métricas dentro do CV</h3>
        <p class="section-text">A mesma lógica vale para <code>AP</code>, <code>ROC AUC</code>, <code>Brier</code>, <code>log loss</code> e calibração. O que importa aqui não é só a média; é o quanto cada métrica varia entre folds e se essa variação sugere fragilidade temporal.</p>
        <div class="embedded-chart-wrap">
          {render_plotly(presentation_cv_metric_folds, "cv_metric_drift")}
        </div>
        {render_details("Ver tabela do drift das métricas por fold", render_clean_table(display_cv_metric_summary(presentation_cv_metric_summary), limit=12))}
        <h3>Como os três modelos se saíram nas combinações finais</h3>
        <p class="section-text">A tabela abaixo compara <b>Regressão logística</b>, <b>Random Forest</b> e <b>CatBoost</b> dentro das combinações finais publicadas. É aqui que se vê, sem misturar problemas diferentes, como cada família se comportou em AP, ROC AUC, Brier, log loss e calibração pooled.</p>
        <div class="note"><b>{model_answer}</b></div>
        {render_clean_table(selected_problem_model_comparison, limit=12)}
        <div class="note"><b>{model_conclusion}</b></div>
        <p class="section-text">{selected_problem_story}</p>
        <h3>Quais combinações finais ficaram publicáveis</h3>
        <p class="section-text">A tabela abaixo mostra só o subconjunto final usado para leitura principal do relatório.</p>
        {render_clean_table(display_reference_scope(presentation_scope), limit=4)}
        {model_fold_validity_detail}
        {calibration_audit_detail}
        {render_details("Ver intervalos bootstrap das métricas de probabilidade", render_clean_table(display_bootstrap(presentation_bootstrap), limit=12))}
        <h3>Auditoria de leakage</h3>
        <p class="section-text">A auditoria oficial checa compartilhamento de colunas-fonte entre entrada e rótulo, violação de janela temporal e qualquer toque estrutural da origem da feature com a mesma tabela ou nomenclatura futura usada no label. O ponto central é este: a entrada termina na 1ª sessão ou no fim dos primeiros 7 dias, e o resultado começa apenas depois disso.</p>
        {render_clean_table(display_leak_summary(presentation_leakage_audit))}
        <h3>Definição B: auditoria estrutural expandida</h3>
        <p class="section-text">Para a Definição B, o build cruza cada feature elegível com <code>source_table</code>, <code>source_columns</code>, <code>pit_class</code> e o fim da janela de score. A leitura correta é: se alguma variável tocasse a mesma origem futura do label, isso apareceria aqui explicitamente.</p>
        {render_clean_table(display_definition_b_leakage_summary(presentation_leakage_summary), limit=6)}
        <p class="section-text">O teste incremental por blocos da Definição B não entra aqui como auditoria de leakage. Ele aparece depois, em <b>Leitura do comportamento</b>, porque serve para interpretar ganho incremental de sinal e hoje foi rodado só em <b>regressão logística</b> como modelo linear de referência.</p>
        {render_details("Ver diagnóstico de separação excessiva da Definição B", render_clean_table(display_definition_b_excessive_separation(presentation_definition_b_excessive_separation), limit=6))}
      </div>
    </section>

    <section>
      <h2>O que ainda é convenção</h2>
      <div class="chart-card">
        <p class="section-text">O build oficial não esconde convenções. Se alguma escolha ainda não foi derivada dos dados nem de uma regra de negócio formal, ela aparece aqui com nome, valor, tipo e motivo. Isso evita vender convenção como se fosse descoberta do modelo.</p>
        {render_clean_table(display_arbitrariness(arbitrariness), limit=8)}
        {arbitrariness_detail}
      </div>
    </section>
    {operational_sections}
  </div>
</body>
</html>
"""

    output_html = args.output_html or (build_dir / "reports" / "targeted_ml_report_v1.html")
    output_html.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
