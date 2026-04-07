from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, r2_score

from targeted_ml.pipelines.modelled_to_ml import analysis_setup as setup

try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None


PROJECT_DIR = Path(__file__).resolve().parent

COLOR_INFO = "#3B82F6"
COLOR_NEUTRAL = "#94A3B8"
COLOR_POSITIVE = "#0F766E"
COLOR_POSITIVE_LIGHT = "#2DD4BF"
COLOR_NEGATIVE = "#D97706"
COLOR_NEGATIVE_LIGHT = "#F59E0B"
COLOR_ERROR = "#EF4444"


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


def format_problem_short_label(value: Any) -> str:
    text = str(value)
    if text.startswith("definition_a::"):
        _, _, track_part = text.partition("__")
        track_label = format_track_name(track_part) if track_part else ""
        return f"Definição A | {track_label}" if track_label else "Definição A"
    if text.startswith("definition_b_label__"):
        track_label = format_track_name(text.replace("definition_b_label__", ""))
        return f"Definição B | {track_label}" if track_label else "Definição B"
    if text.startswith("definition_b__"):
        track_label = format_track_name(text.replace("definition_b__", ""))
        return f"Definição B | {track_label}" if track_label else "Definição B"
    return format_problem_key(text)


def format_official_status(value: Any) -> str:
    mapping = {
        "official_admissible": "admissível na fronteira oficial",
        "official_fixed_literal": "comparador literal fixo",
        "official_unique": "definição oficial única",
        "official_winner": "vencedor oficial",
        "sensitivity_admissible": "admissível na sensibilidade",
        "sensitivity_lock_topk": "sensibilidade no lock temporal",
        "sensitivity_development_frontier": "sensibilidade na fronteira do development",
    }
    return mapping.get(str(value), str(value))


def format_selection_basis(value: Any) -> str:
    mapping = {
        "per_metric_out_of_sample_rank_aggregation_then_metric_pareto_front": "agregação por desempenho fora da amostra + fronteira de Pareto por métrica",
        "literal_comparator_fixed_a_priori": "regra literal fixada a priori",
        "univariate_exact_development_outer_test_rank_aggregation_then_metric_pareto_front_then_deterministic_ranked_promotion_with_temporal_holdout_reserved": "busca no development + fronteira admissível + promoção ranqueada antes da avaliação temporal intocada",
        "univariate_exact_development_outer_test_rank_aggregation_then_metric_pareto_front_then_definition_lock_pareto_with_local_threshold_sensitivity_before_final_model_evaluation": "busca no development + top-K admissível + lock temporal com sensibilidade local de threshold antes da avaliação final do modelo",
        "atomic_screening_on_development_outer_tests_then_pairwise_and_or_and_weighted_percentile_expansion_then_definition_lock_with_threshold_structural_and_weight_sensitivity_before_final_model_evaluation": "screening atômico no development + expansão pairwise AND/OR e ponderada + lock temporal com sensibilidade de threshold, estrutura e peso antes da avaliação final do modelo",
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
        if len(parts) >= 3:
            scope = str(parts[1])
            if scope == "definition_group_matched_frontier_candidates":
                return "modelo servível primário dentro do definition_group congelado após desempate por erro probabilístico, variabilidade e informação disponível"
            if scope == "all_pareto_frontier_candidates":
                return "modelo servível primário entre todos os candidatos da fronteira após desempate por erro probabilístico, variabilidade e informação disponível"
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


def build_strict_context_detail(feature_registry: pd.DataFrame, *, multiline: bool = False) -> str:
    if feature_registry.empty:
        return ""
    show = feature_registry.copy()
    if "allowed_in_STRICT_CONTEXT" in show.columns:
        show = show[show["allowed_in_STRICT_CONTEXT"] == 1].copy()
    if show.empty:
        return ""
    ordered_features = [
        "months_after_entry",
        "teacher_population_status",
        "utm_group",
        "first_session_entry_surface",
        "first_session_device_bucket",
        "first_event_missing_flag",
        "first_device_missing_flag",
        "first_utm_missing_flag",
        "session_without_interaction_flag",
    ]
    parts: list[str] = []
    available = set(show["feature_name"].astype(str)) if "feature_name" in show.columns else set()
    for feature_name in ordered_features:
        if feature_name in available:
            description = describe_strict_context_feature(feature_name)
            parts.append(f"<code>{feature_name}</code>: {description}")
    if not parts:
        return ""
    separator = "<br/>" if multiline else " "
    return separator.join(parts)


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
        "The published build limits the expanding outer backtest to the last 5 test months to keep the exact-threshold search and calibrated model comparison computationally feasible. This is surfaced explicitly instead of being hidden.": "O build publicado limita o outer backtest expansivo aos últimos meses de teste configurados para manter viável a busca exata de thresholds e a comparação calibrada de modelos. Isso aparece explicitamente no relatório.",
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


def render_lineage_box(items: list[tuple[str, str]]) -> str:
    if not items:
        return ""
    body = "".join(f"<p><b>{label}:</b> {value}</p>" for label, value in items if value)
    if not body:
        return ""
    return f"<div class='lineage'>{body}</div>"


def render_teaching_block(
    *,
    title: str,
    quick_definition: str,
    how_text: str,
    why_text: str,
    chart_html: str,
    detail_html: str = "",
    what_text: str,
    conclusion_text: str,
    lineage_items: list[tuple[str, str]],
) -> str:
    chart_wrap = f"<div class='embedded-chart-wrap'>{chart_html}</div>" if chart_html else ""
    lineage_html = render_lineage_box(lineage_items)
    what_html = f'<p class="section-text">{what_text}</p>' if str(what_text).strip() else ""
    conclusion_html = f'<div class="note">{conclusion_text}</div>' if str(conclusion_text).strip() else ""
    return f"""
    <div class="guide-card">
      <h3>{title}</h3>
      <p class="section-text lead-text">{quick_definition}</p>
      <p class="section-text"><b>Como foi feito.</b> {how_text}</p>
      <p class="section-text"><b>Por que isso importa.</b> {why_text}</p>
      {chart_wrap}
      {detail_html}
      {what_html}
      {conclusion_html}
      {lineage_html}
    </div>
    """


def render_protocol_sections(sections: list[tuple[str, list[str]]]) -> str:
    if not sections:
        return ""
    chunks: list[str] = []
    for title, bullets in sections:
        if not bullets:
            continue
        bullet_html = "".join(f"<li>{bullet}</li>" for bullet in bullets)
        chunks.append(
            f"""
            <div class="protocol-section">
              <h4>{title}</h4>
              <ul class="protocol-list">
                {bullet_html}
              </ul>
            </div>
            """
        )
    if not chunks:
        return ""
    return f"<div class='protocol-wrap'>{''.join(chunks)}</div>"


def render_checklist_block(
    *,
    title: str,
    quick_definition: str,
    how_text: str,
    why_text: str,
    sections: list[tuple[str, list[str]]],
    conclusion_text: str,
    lineage_items: list[tuple[str, str]],
) -> str:
    section_html = render_protocol_sections(sections)
    lineage_html = render_lineage_box(lineage_items)
    conclusion_html = f'<div class="note">{conclusion_text}</div>' if str(conclusion_text).strip() else ""
    return f"""
    <div class="guide-card">
      <h3>{title}</h3>
      <p class="section-text lead-text">{quick_definition}</p>
      <p class="section-text"><b>Como deve ser lido.</b> {how_text}</p>
      <p class="section-text"><b>Por que isso importa.</b> {why_text}</p>
      {section_html}
      {conclusion_html}
      {lineage_html}
    </div>
    """


def render_strict_context_note(feature_registry: pd.DataFrame) -> str:
    return (
        "<div class='definition-note'>"
        "<strong>STRICT_CONTEXT</strong>"
        "<p>Usa só 9 variáveis de contexto inicial e completude: tempo desde a entrada, status básico do professor, canal de aquisição, origem e dispositivo da 1ª entrada, e flags de ausência/1ª sessão sem interação.</p>"
        "</div>"
    )


def render_compact_block(
    *,
    title: str,
    description: str,
    chart_html: str,
    finding_text: str,
    lineage_items: list[tuple[str, str]],
) -> str:
    chart_wrap = f"<div class='embedded-chart-wrap'>{chart_html}</div>" if chart_html else ""
    lineage_html = render_lineage_box(lineage_items)
    return f"""
    <div class="guide-card compact-card">
      <h3>{title}</h3>
      <p class="section-text lead-text">{description}</p>
      {chart_wrap}
      <p class="section-text">{finding_text}</p>
      {lineage_html}
    </div>
    """


def render_assumptions_timeline(
    assumption_points: pd.DataFrame,
    feature_registry: pd.DataFrame,
) -> str:
    if assumption_points.empty:
        return ""
    ordered_tracks = ["S1", "S7", "S1+S7", "STRICT_CONTEXT"]
    point_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in assumption_points.to_dict(orient="records"):
        point_map[(str(row.get("track_name", "")), str(row.get("event_name", "")))] = row
    rows: list[str] = []
    for track in ordered_tracks:
        score = point_map.get((track, "momento do score"), {})
        raw_score_day = float(pd.to_numeric(score.get("day_number"), errors="coerce") or 0.0)
        score_slot = "session" if track == "S1" and raw_score_day <= 1.0 else "week"
        slot_icons = {
            "start": "<i class='legend-dot legend-start'></i>",
            "session": "<i class='legend-dot legend-predict'></i>" if score_slot == "session" else "",
            "week": "<i class='legend-dot legend-predict'></i>" if score_slot == "week" else "",
            "result": "<i class='legend-dot legend-result-shape'></i>",
        }
        rows.append(
            f"""
            <div class="assumption-matrix-row">
              <div class="assumption-track">{track}</div>
              <div class="assumption-trackline">
                <div class="assumption-rail-line"></div>
                <div class="assumption-cell">{slot_icons["start"]}</div>
                <div class="assumption-cell">{slot_icons["session"]}</div>
                <div class="assumption-cell">{slot_icons["week"]}</div>
                <div class="assumption-cell">{slot_icons["result"]}</div>
              </div>
            </div>
            """
        )
    axis = """
    <div class="assumption-header">
      <div></div>
      <div class="assumption-header-cell">Início</div>
      <div class="assumption-header-cell">Fim da 1ª sessão</div>
      <div class="assumption-header-cell">Fim dos 7 dias</div>
      <div class="assumption-header-cell">Fim da janela futura</div>
    </div>
    """
    legend = """
    <div class="assumption-legend">
      <span><i class="legend-dot legend-start"></i>Início observado</span>
      <span><i class="legend-dot legend-predict"></i>Já dá para prever</span>
      <span><i class="legend-dot legend-result-shape"></i>Janela futura já pode ser medida</span>
    </div>
    """
    return f"<div class='assumption-panel'>{legend}{axis}<div class='assumption-matrix'>{''.join(rows)}</div><div class='assumption-axis-title'>Momento do processo</div></div>"


def render_model_selection_board(model_frontier: pd.DataFrame) -> str:
    if model_frontier.empty:
        return ""
    sections: list[str] = []
    for problem_key, group in model_frontier.groupby("problem_key", dropna=False):
        group = group.sort_values(
            ["mean_brier", "mean_log_loss", "mean_ap", "mean_roc_auc"],
            ascending=[True, True, False, False],
            kind="mergesort",
        ).copy()
        body_rows: list[str] = []
        for idx, row in enumerate(group.to_dict(orient="records"), start=1):
            model_label = format_model_name(row.get("model_name"))
            is_winner = idx == 1
            is_primary = int(row.get("selected_flag", 0)) == 1
            row_class = "comparison-table-row-selected" if (is_winner or is_primary) else ""
            badges: list[str] = []
            if is_winner:
                badges.append("<span class='status-pill winner-pill'>vencedor</span>")
            if is_primary:
                badges.append("<span class='status-pill selected-pill'>primário</span>")
            badge_html = (" " + " ".join(badges)) if badges else ""
            body_rows.append(
                f"""
                <tr class="{row_class}">
                  <td><span class="rank-pill">{idx}º</span></td>
                  <td><strong>{model_label}</strong>{badge_html}</td>
                  <td>{format_number(row.get('mean_ap'), 3)}</td>
                  <td>{format_number(row.get('mean_roc_auc'), 3)}</td>
                  <td>{format_number(row.get('mean_brier'), 3)}</td>
                  <td>{format_number(row.get('mean_log_loss'), 3)}</td>
                </tr>
                """
            )
        problem_class = "comparison-section-a" if str(problem_key).startswith("definition_a") else "comparison-section-b"
        sections.append(
            f"""
            <div class="comparison-section {problem_class}">
              <div class="comparison-section-title">{format_problem_short_label(problem_key)}</div>
              <table class="comparison-table">
                <thead>
                  <tr>
                    <th>Posição</th>
                    <th>Modelo</th>
                    <th>AP</th>
                    <th>ROC AUC</th>
                    <th>Brier</th>
                    <th>Log loss</th>
                  </tr>
                </thead>
                <tbody>
                  {''.join(body_rows)}
                </tbody>
              </table>
            </div>
            """
        )
    return f"<div class='comparison-sections'>{''.join(sections)}</div>"


def render_trust_panel(
    cv_metric_folds: pd.DataFrame,
    cv_threshold_summary: pd.DataFrame,
) -> str:
    if cv_metric_folds.empty:
        return ""
    ap_rows = cv_metric_folds[cv_metric_folds["metric_name"].astype(str) == "ap"].copy()
    brier_rows = cv_metric_folds[cv_metric_folds["metric_name"].astype(str) == "brier"].copy()
    ap_values = pd.to_numeric(ap_rows["metric_value"], errors="coerce").dropna()
    brier_values = pd.to_numeric(brier_rows["metric_value"], errors="coerce").dropna()
    folds = sorted(set(pd.to_numeric(ap_rows["fold_id"], errors="coerce").dropna().astype(int).tolist()))
    trust_cards: list[str] = []
    if folds:
        trust_cards.append(
            f"""
            <div class="trust-card">
              <span>Outer folds válidos</span>
              <strong>{len(folds)}</strong>
              <small>Cada fold é um mês futuro nunca visto no treino.</small>
            </div>
            """
        )
    if not ap_values.empty:
        trust_cards.append(
            f"""
            <div class="trust-card">
              <span>AP</span>
              <strong>{format_number(ap_values.mean(), 3)}</strong>
              <small>Faixa {format_number(ap_values.min(), 3)} a {format_number(ap_values.max(), 3)}</small>
            </div>
            """
        )
    if not brier_values.empty:
        trust_cards.append(
            f"""
            <div class="trust-card">
              <span>Brier</span>
              <strong>{format_number(brier_values.mean(), 3)}</strong>
              <small>Faixa {format_number(brier_values.min(), 3)} a {format_number(brier_values.max(), 3)}</small>
            </div>
            """
        )
    if not cv_threshold_summary.empty:
        subset = cv_threshold_summary[cv_threshold_summary["policy_name"].astype(str) == "tercis"].copy()
        for metric_name, label in [("precision", "Precisão em tercis"), ("recall", "Recall em tercis")]:
            metric_row = subset[subset["metric_name"].astype(str) == metric_name]
            if not metric_row.empty:
                row = metric_row.iloc[0]
                trust_cards.append(
                    f"""
                    <div class="trust-card">
                      <span>{label}</span>
                      <strong>{format_percent(row.get('mean_value'), 1)}</strong>
                      <small>Faixa {format_percent(row.get('min_value'), 1)} a {format_percent(row.get('max_value'), 1)}</small>
                    </div>
                    """
                )
    fold_lines: list[str] = []
    for fold_id in folds:
        ap_match = ap_rows[pd.to_numeric(ap_rows["fold_id"], errors="coerce").astype("Int64") == fold_id]
        brier_match = brier_rows[pd.to_numeric(brier_rows["fold_id"], errors="coerce").astype("Int64") == fold_id]
        ap_value = ap_match.iloc[0]["metric_value"] if not ap_match.empty else np.nan
        brier_value = brier_match.iloc[0]["metric_value"] if not brier_match.empty else np.nan
        fold_lines.append(
            f"""
            <div class="fold-line">
              <b>Fold {fold_id}</b>
              <span>AP {format_number(ap_value, 3)}</span>
              <span>Brier {format_number(brier_value, 3)}</span>
            </div>
            """
        )
    return (
        f"<div class='trust-kpi-grid'>{''.join(trust_cards)}</div>"
        f"<div class='fold-strip'>{''.join(fold_lines)}</div>"
    )


def render_confusion_matrix_panel(confusion_df: pd.DataFrame) -> str:
    if confusion_df.empty:
        return ""
    preferred = confusion_df[confusion_df["policy_name"].astype(str) == "tercis"].copy()
    if not preferred.empty:
        confusion_df = preferred
    tp = _get_confusion_value(confusion_df, "nao_realiza", "nao_realiza")
    fp = _get_confusion_value(confusion_df, "realiza", "nao_realiza")
    fn = _get_confusion_value(confusion_df, "nao_realiza", "realiza")
    tn = _get_confusion_value(confusion_df, "realiza", "realiza")
    return f"""
    <div class="confusion-grid">
      <div class="confusion-head empty-head"></div>
      <div class="confusion-head top-head-risk">Modelo marcou como inativo</div>
      <div class="confusion-head top-head-active">Modelo marcou como ativo</div>
      <div class="confusion-head side-head side-head-risk">Na prática, ficou inativo</div>
      <div class="confusion-cell tp-cell"><span class="conf-label">Acerto de risco</span><strong>{tp}</strong><small>Ficou inativo e entrou na fila de risco</small></div>
      <div class="confusion-cell fn-cell"><span class="conf-label">Risco perdido</span><strong>{fn}</strong><small>Ficou inativo, mas ficou fora da fila</small></div>
      <div class="confusion-head side-head side-head-active">Na prática, continuou ativo</div>
      <div class="confusion-cell fp-cell"><span class="conf-label">Alarme falso</span><strong>{fp}</strong><small>Entrou na fila de risco, mas continuou ativo</small></div>
      <div class="confusion-cell tn-cell"><span class="conf-label">Acerto de atividade</span><strong>{tn}</strong><small>Continuou ativo e ficou fora da fila</small></div>
    </div>
    """


def render_final_model_panel(
    model_frontier: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    confusion_df: pd.DataFrame,
) -> str:
    if model_frontier.empty:
        return ""
    row = model_frontier.iloc[0]
    problem_key = str(row.get("problem_key", ""))
    threshold_row = _get_threshold_row(threshold_metrics, problem_key, "tercis")
    cards = [
        f"<div class='trust-card'><span>AP</span><strong>{format_number(row.get('mean_ap'), 3)}</strong><small>Teste futuro concatenado</small></div>",
        f"<div class='trust-card'><span>ROC AUC</span><strong>{format_number(row.get('mean_roc_auc'), 3)}</strong><small>Separação entre realizou e não realizou</small></div>",
        f"<div class='trust-card'><span>Brier</span><strong>{format_number(row.get('mean_brier'), 3)}</strong><small>Erro médio da probabilidade</small></div>",
        f"<div class='trust-card'><span>Log loss</span><strong>{format_number(row.get('mean_log_loss'), 3)}</strong><small>Pune probabilidade muito errada com confiança alta</small></div>",
    ]
    if threshold_row is not None:
        cards.append(
            f"<div class='trust-card'><span>Precisão em tercis</span><strong>{format_percent(threshold_row.get('precision'))}</strong><small>Entre os marcados como alto risco, quantos realmente não realizaram</small></div>"
        )
        cards.append(
            f"<div class='trust-card'><span>Recall em tercis</span><strong>{format_percent(threshold_row.get('recall'))}</strong><small>Entre quem não realizou, quantos foram capturados como alto risco</small></div>"
        )
    return f"<div class='trust-kpi-grid'>{''.join(cards)}</div>{render_confusion_matrix_panel(confusion_df)}"


def render_score_panel(
    score_deciles: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    band_summary: pd.DataFrame,
) -> str:
    parts: list[str] = []
    if not score_deciles.empty:
        ordered = score_deciles.sort_values("score_decile", kind="mergesort").copy()
        decile_rows: list[str] = []
        for row in ordered.to_dict(orient="records"):
            score_value = float(pd.to_numeric(row.get("mean_score"), errors="coerce") or 0.0)
            realized_value = float(pd.to_numeric(row.get("realized_rate"), errors="coerce") or 0.0)
            decile_rows.append(
                f"""
                <div class="decile-row">
                  <div class="decile-label">D{int(pd.to_numeric(row.get('score_decile'), errors='coerce') or 0)}</div>
                  <div class="decile-metric">
                    <span>Score previsto {format_percent(score_value)}</span>
                    <div class="decile-track"><div class="decile-fill decile-pred-fill" style="width:{score_value * 100:.1f}%"></div></div>
                  </div>
                  <div class="decile-metric">
                    <span>Taxa observada {format_percent(realized_value)}</span>
                    <div class="decile-track"><div class="decile-fill decile-real-fill" style="width:{realized_value * 100:.1f}%"></div></div>
                  </div>
                </div>
                """
            )
        parts.append(
            "<div class='decile-panel'>"
            "<div class='decile-legend'><span><i class='legend-box decile-pred-fill'></i>Score previsto</span><span><i class='legend-box decile-real-fill'></i>Taxa observada</span></div>"
            f"{''.join(decile_rows)}</div>"
        )
    if threshold_metrics.empty:
        return "".join(parts)
    problem_key = str(score_deciles.iloc[0].get("problem_key", "")) if not score_deciles.empty else ""
    policy_cards: list[str] = []
    share_lookup: dict[tuple[str, str], Any] = {}
    if not band_summary.empty:
        match = band_summary[band_summary["problem_key"].astype(str) == problem_key].copy()
        for row in match.to_dict(orient="records"):
            share_lookup[(str(row.get("policy_name", "")), str(row.get("band_name", "")))] = row.get("share")
    for policy_name in ["top_10_percent", "tercis", "score_ge_0_70"]:
        row = _get_threshold_row(threshold_metrics, problem_key, policy_name)
        if row is None:
            continue
        policy_cards.append(
            f"""
            <div class="policy-card">
              <h4>{format_policy_name(policy_name)}</h4>
              <div class="metric-grid">
                <div class="metric-mini"><span>Precisão</span><strong>{format_percent(row.get('precision'))}</strong></div>
                <div class="metric-mini"><span>Recall</span><strong>{format_percent(row.get('recall'))}</strong></div>
                <div class="metric-mini"><span>Faixa alta</span><strong>{format_percent(share_lookup.get((policy_name, 'alto')))}</strong></div>
              </div>
            </div>
            """
        )
    if policy_cards:
        parts.append(f"<div class='policy-grid'>{''.join(policy_cards)}</div>")
    return "".join(parts)


def render_driver_panel(
    feature_importance: pd.DataFrame,
    definition_b_feature_block_gain_summary: pd.DataFrame,
) -> str:
    bars: list[str] = []
    if not feature_importance.empty:
        grouped = (
            feature_importance.groupby(["problem_key", "model_name", "feature_name"], as_index=False)
            .agg(importance_mean=("importance_mean", "mean"))
        )
        grouped["importance_abs"] = grouped["importance_mean"].abs()
        grouped = grouped.sort_values(["importance_abs", "problem_key", "model_name"], ascending=[False, True, True], kind="mergesort")
        primary = grouped[grouped["problem_key"].astype(str).str.startswith("definition_a::")].copy()
        if primary.empty:
            primary = grouped.copy()
        if not primary.empty:
            problem_key = str(primary.iloc[0]["problem_key"])
            model_name = str(primary.iloc[0]["model_name"])
            plot_df = primary[
                (primary["problem_key"].astype(str) == problem_key)
                & (primary["model_name"].astype(str) == model_name)
            ].head(5).copy()
            max_abs = float(plot_df["importance_abs"].max() or 0.0)
            for row in plot_df.to_dict(orient="records"):
                width = 0.0 if max_abs == 0 else float(row["importance_abs"]) / max_abs * 100.0
                bars.append(
                    f"""
                    <div class="feature-bar-row">
                      <div class="feature-bar-label">{format_feature_name(row.get('feature_name'))}</div>
                      <div class="feature-bar-track"><div class="feature-bar-fill" style="width:{width:.1f}%"></div></div>
                      <div class="feature-bar-value">{float(pd.to_numeric(row.get('importance_mean'), errors='coerce') or 0.0):.5f}</div>
                    </div>
                    """
                )
    chips: list[str] = []
    s1_row = _block_gain_row(definition_b_feature_block_gain_summary, "context_plus_feature_class::s1")
    s7_row = _block_gain_row(definition_b_feature_block_gain_summary, "context_plus_feature_class::s7")
    if s1_row is not None and s7_row is not None:
        chips.append(
            f"<div class='signal-chip'><strong>1ª sessão vs 7 dias</strong><span>+{format_number(s1_row.get('delta_ap_vs_context'), 3)} de AP com S1 e +{format_number(s7_row.get('delta_ap_vs_context'), 3)} com S7.</span></div>"
        )
    early_views = _block_gain_row(definition_b_feature_block_gain_summary, "context_plus_behavior_family::early_views")
    early_downloads = _block_gain_row(definition_b_feature_block_gain_summary, "context_plus_behavior_family::early_downloads")
    week_views = _block_gain_row(definition_b_feature_block_gain_summary, "context_plus_behavior_family::week_views")
    week_downloads = _block_gain_row(definition_b_feature_block_gain_summary, "context_plus_behavior_family::week_downloads")
    if early_views is not None and early_downloads is not None and week_views is not None and week_downloads is not None:
        chips.append(
            f"<div class='signal-chip'><strong>Views vs downloads</strong><span>Na 1ª sessão, views {format_number(early_views.get('delta_ap_vs_context'), 3)} vs downloads {format_number(early_downloads.get('delta_ap_vs_context'), 3)}. Na semana, views {format_number(week_views.get('delta_ap_vs_context'), 3)} vs downloads {format_number(week_downloads.get('delta_ap_vs_context'), 3)}.</span></div>"
        )
    feature_html = f"<div class='feature-bar-list'>{''.join(bars)}</div>" if bars else ""
    chips_html = f"<div class='signal-chip-grid'>{''.join(chips)}</div>" if chips else ""
    return feature_html + chips_html


def render_cluster_panel(cluster_summary: pd.DataFrame, cluster_profile: pd.DataFrame) -> str:
    if cluster_summary.empty:
        return ""
    summary = cluster_summary.sort_values("mean_risk_score", ascending=False, kind="mergesort").copy()
    profile_map: dict[str, dict[str, float]] = {}
    if not cluster_profile.empty:
        subset = cluster_profile[
            cluster_profile["feature_name"].astype(str).isin(
                [
                    "teacher_active_months_total",
                    "avg_activity_events_active_month",
                    "avg_active_days_active_month",
                    "avg_strict_downloads_active_month",
                    "avg_content_views_active_month",
                ]
            )
        ].copy()
        for cluster_name, group in subset.groupby("cluster_name", dropna=False):
            profile_map[str(cluster_name)] = {
                str(row["feature_name"]): float(row["feature_mean"])
                for row in group.to_dict(orient="records")
            }
    cards: list[str] = []
    for row in summary.to_dict(orient="records"):
        cluster_name = str(row.get("cluster_name", ""))
        profile = profile_map.get(cluster_name, {})
        cards.append(
            f"""
            <div class="profile-card">
              <h4>{format_cluster_name(cluster_name)}</h4>
              <div class="metric-grid">
                <div class="metric-mini"><span>Share</span><strong>{format_percent(row.get('share'))}</strong></div>
                <div class="metric-mini"><span>risk_score médio</span><strong>{format_percent(row.get('mean_risk_score'))}</strong></div>
                <div class="metric-mini"><span>Não realização</span><strong>{format_percent(row.get('realized_inactivity_rate'))}</strong></div>
              </div>
              <div class="profile-list">
                <div><b>Meses ativos totais</b>: {format_number(profile.get('teacher_active_months_total'), 1)}</div>
                <div><b>Eventos por mês ativo</b>: {format_number(profile.get('avg_activity_events_active_month'), 1)}</div>
                <div><b>Dias ativos por mês</b>: {format_number(profile.get('avg_active_days_active_month'), 1)}</div>
                <div><b>Downloads por mês ativo</b>: {format_number(profile.get('avg_strict_downloads_active_month'), 1)}</div>
                <div><b>Visualizações por mês ativo</b>: {format_number(profile.get('avg_content_views_active_month'), 1)}</div>
              </div>
            </div>
            """
        )
    return f"<div class='profile-grid'>{''.join(cards)}</div>"


def render_heavy_user_panel(
    heavy_user_summary: pd.DataFrame,
    heavy_user_profile: pd.DataFrame,
) -> str:
    if px is None or heavy_user_summary.empty:
        return ""
    summary = heavy_user_summary[heavy_user_summary["policy_name"].astype(str) == "heavy_top_10_percent"].copy()
    if summary.empty:
        summary = heavy_user_summary.copy()
    profile_map: dict[int, dict[str, float]] = {}
    if not heavy_user_profile.empty:
        subset = heavy_user_profile[heavy_user_profile["policy_name"].astype(str) == "heavy_top_10_percent"].copy()
        if subset.empty:
            subset = heavy_user_profile.copy()
        subset = subset[
            subset["metric_name"].astype(str).isin(
                [
                    "future_sessions",
                    "future_session_minutes",
                    "future_active_days",
                    "future_activity_events",
                ]
            )
        ].copy()
        for flag, group in subset.groupby("heavy_user_flag", dropna=False):
            profile_map[int(flag)] = {
                str(row["metric_name"]): float(row["metric_mean"])
                for row in group.to_dict(orient="records")
            }
    rows: list[dict[str, Any]] = []
    base_profile = profile_map.get(0, {})
    heavy_profile = profile_map.get(1, {})
    summary_by_flag = {
        int(row.get("heavy_user_flag", 0)): row
        for row in summary.to_dict(orient="records")
    }
    heavy_row = summary_by_flag.get(1, {})
    base_row = summary_by_flag.get(0, {})
    comparisons = [
        ("Taxa de inatividade", float(pd.to_numeric(heavy_row.get("realized_inactivity_rate"), errors="coerce") or 0.0), float(pd.to_numeric(base_row.get("realized_inactivity_rate"), errors="coerce") or 0.0), "negative"),
        ("Sessões futuras", float(heavy_profile.get("future_sessions", 0.0)), float(base_profile.get("future_sessions", 0.0)), "positive"),
        ("Dias ativos futuros", float(heavy_profile.get("future_active_days", 0.0)), float(base_profile.get("future_active_days", 0.0)), "positive"),
        ("Minutos futuros", float(heavy_profile.get("future_session_minutes", 0.0)), float(base_profile.get("future_session_minutes", 0.0)), "positive"),
        ("Eventos futuros", float(heavy_profile.get("future_activity_events", 0.0)), float(base_profile.get("future_activity_events", 0.0)), "positive"),
    ]
    for metric_name, heavy_value, base_value, semantic in comparisons:
        if base_value == 0:
            ratio = np.nan
        else:
            ratio = heavy_value / base_value
        rows.append(
            {
                "metric_name": metric_name,
                "ratio_vs_rest": ratio,
                "semantic": semantic,
            }
        )
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return ""
    order = ["Taxa de inatividade", "Sessões futuras", "Dias ativos futuros", "Minutos futuros", "Eventos futuros"]
    plot_df["metric_name"] = pd.Categorical(plot_df["metric_name"], categories=order, ordered=True)
    fig = px.bar(
        plot_df.sort_values(["metric_name"], kind="mergesort"),
        x="metric_name",
        y="ratio_vs_rest",
        color="semantic",
        barmode="group",
        text="ratio_vs_rest",
        title="Quanto o grupo de uso forte difere do restante da base",
        color_discrete_map={"positive": COLOR_POSITIVE, "negative": COLOR_NEGATIVE},
    )
    fig.update_traces(texttemplate="%{text:.1f}x", textposition="outside", cliponaxis=False)
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Vezes o restante da base")
    fig.update_layout(height=560, margin=dict(l=40, r=30, t=70, b=70), legend_title_text="", showlegend=False)
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def render_product_block() -> str:
    return """
    <div class="guide-card">
      <h3>Produto</h3>
      <div class="product-list">
        <div><b>YAML controla</b>: caminhos de dados, janela do label, trilhas oficiais, famílias de modelo, tuning e políticas de cutoff/faixas.</div>
        <div><b>Definições aceitas</b>: A definição A oficial por busca univariada exata e a definição B literal declarada no YAML.</div>
        <div><b>Pipelines separados</b>: treino em <code>build-modelled</code>, <code>build-ml</code>, <code>build-report</code> e <code>build</code>; inferência em <code>export-serving</code>, <code>score-modelled</code>, <code>score-frame</code> e <code>score-raw</code>.</div>
        <div><b>Lista de score</b>: teacher id, mês, modelo, definição, trilha, <code>score</code>, <code>risk_score</code>, elegibilidade e rank.</div>
      </div>
      <p class="section-text"><a class="button-link" href="http://localhost:8501" target="_blank" rel="noopener noreferrer">Abrir Streamlit</a> <a class="button-link secondary-link" href="http://localhost:8081" target="_blank" rel="noopener noreferrer">Abrir dbt docs</a></p>
    </div>
    """


def _choice_numeric(arbitrariness: pd.DataFrame, choice_name: str, default: int | None = None) -> int | None:
    if arbitrariness.empty or "choice_name" not in arbitrariness.columns:
        return default
    match = arbitrariness[arbitrariness["choice_name"].astype(str) == choice_name]
    if match.empty:
        return default
    value = match.iloc[0].get("choice_value", default)
    try:
        return int(float(value))
    except Exception:
        return default


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
            color_discrete_map={"Definição A": COLOR_INFO, "Definição B": COLOR_NEGATIVE},
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
    elif kind == "assumptions":
        plot_df = df.copy()
        if plot_df.empty or go is None:
            return ""
        track_order = ["S1", "S7", "S1+S7", "STRICT_CONTEXT"]
        point_map = {
            (str(row["track_name"]), str(row["event_name"])): row
            for row in plot_df.to_dict(orient="records")
        }
        y_positions = {track: len(track_order) - idx for idx, track in enumerate(track_order)}
        fig = go.Figure()
        for track in track_order:
            score = point_map.get((track, "momento do score"), {})
            end = point_map.get((track, "fim da janela do resultado"), {})
            y = y_positions[track]
            score_day = float(pd.to_numeric(score.get("day_number"), errors="coerce") or 0.0)
            end_day = float(pd.to_numeric(end.get("day_number"), errors="coerce") or 37.0)
            fig.add_trace(
                go.Scatter(
                    x=[0, end_day],
                    y=[y, y],
                    mode="lines",
                    line=dict(color="#64748B", width=8),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        start_x = [0] * len(track_order)
        pred_x: list[float] = []
        for track in track_order:
            raw_score_day = float(pd.to_numeric(point_map.get((track, "momento do score"), {}).get("day_number"), errors="coerce") or 0.0)
            pred_x.append(1.0 if track == "S1" and raw_score_day == 0.0 else raw_score_day)
        result_x = [float(pd.to_numeric(point_map.get((track, "fim da janela do resultado"), {}).get("day_number"), errors="coerce") or 37.0) for track in track_order]
        ys = [y_positions[track] for track in track_order]
        fig.add_trace(go.Scatter(x=start_x, y=ys, mode="markers", name="Início observado", marker=dict(color="#E2E8F0", line=dict(color="#475569", width=2), size=16, symbol="circle")))
        fig.add_trace(go.Scatter(x=pred_x, y=ys, mode="markers", name="Já dá para prever", marker=dict(color=COLOR_INFO, size=16, symbol="circle")))
        fig.add_trace(go.Scatter(x=result_x, y=ys, mode="markers", name="Resultado pode ser medido", marker=dict(color=COLOR_POSITIVE, size=18, symbol="triangle-up")))
        fig.update_yaxes(
            tickmode="array",
            tickvals=ys,
            ticktext=track_order,
            title="",
            range=[0.5, len(track_order) + 0.5],
        )
        fig.update_xaxes(
            title="Momento do processo",
            tickmode="array",
            tickvals=[0, 1, 7, 8, 37],
            ticktext=["Início", "1ª sessão", "7 dias", "Início<br>do resultado", "Fim<br>do resultado"],
            tickangle=0,
        )
        fig.update_layout(
            title=dict(
                text="Quando cada previsão pode ser feita e quando o resultado passa a poder ser medido",
                x=0.0,
                xanchor="left",
                pad=dict(b=32),
            ),
            height=470,
            margin=dict(l=40, r=30, t=90, b=78),
            legend_title_text="",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
    elif kind == "scenarios_tested":
        plot_df = df.copy()
        if plot_df.empty:
            return ""
        plot_df["definition_name"] = plot_df["definition_name"].map(format_definition_name)
        plot_df["track_name"] = plot_df["track_name"].map(format_track_name)
        fig = px.bar(
            plot_df,
            x="track_name",
            y="feature_count",
            color="definition_name",
            barmode="group",
            text="feature_count",
            title="Definições e trilhas testadas",
            color_discrete_map={"Definição A": COLOR_INFO, "Definição B": COLOR_NEGATIVE},
        )
        fig.update_traces(texttemplate="%{text}", textposition="outside", cliponaxis=False)
        fig.update_xaxes(title="Trilha")
        fig.update_yaxes(title="Quantidade de variáveis elegíveis")
    elif kind == "model_selection":
        plot_df = df.copy()
        if plot_df.empty:
            return ""
        plot_df["problem_label"] = plot_df["problem_key"].map(format_problem_short_label)
        plot_df["model_label"] = plot_df["model_name"].map(format_model_name)
        metric_rows: list[dict[str, Any]] = []
        for row in plot_df.to_dict(orient="records"):
            metric_rows.extend(
                [
                    {
                        "problem_label": row["problem_label"],
                        "model_label": row["model_label"],
                        "metric_label": "AP",
                        "metric_value": float(pd.to_numeric(row.get("mean_ap"), errors="coerce") or 0.0),
                        "raw_value": float(pd.to_numeric(row.get("mean_ap"), errors="coerce") or 0.0),
                    },
                    {
                        "problem_label": row["problem_label"],
                        "model_label": row["model_label"],
                        "metric_label": "ROC AUC",
                        "metric_value": float(pd.to_numeric(row.get("mean_roc_auc"), errors="coerce") or 0.0),
                        "raw_value": float(pd.to_numeric(row.get("mean_roc_auc"), errors="coerce") or 0.0),
                    },
                    {
                        "problem_label": row["problem_label"],
                        "model_label": row["model_label"],
                        "metric_label": "1 - Brier",
                        "metric_value": 1.0 - float(pd.to_numeric(row.get("mean_brier"), errors="coerce") or 0.0),
                        "raw_value": float(pd.to_numeric(row.get("mean_brier"), errors="coerce") or 0.0),
                    },
                    {
                        "problem_label": row["problem_label"],
                        "model_label": row["model_label"],
                        "metric_label": "1 - Log loss",
                        "metric_value": 1.0 - float(pd.to_numeric(row.get("mean_log_loss"), errors="coerce") or 0.0),
                        "raw_value": float(pd.to_numeric(row.get("mean_log_loss"), errors="coerce") or 0.0),
                    },
                ]
            )
        plot_df = pd.DataFrame(metric_rows)
        plot_df["comparison_label"] = plot_df["problem_label"] + "<br>" + plot_df["metric_label"]
        fig = px.bar(
            plot_df,
            x="comparison_label",
            y="metric_value",
            color="model_label",
            barmode="group",
            color_discrete_map={
                "CatBoost": COLOR_INFO,
                "Regressão logística": COLOR_POSITIVE,
                "Random Forest": COLOR_NEGATIVE,
            },
            category_orders={
                "comparison_label": [
                    "Definição A | S1+S7<br>AP",
                    "Definição A | S1+S7<br>ROC AUC",
                    "Definição A | S1+S7<br>1 - Brier",
                    "Definição A | S1+S7<br>1 - Log loss",
                    "Definição B | S7<br>AP",
                    "Definição B | S7<br>ROC AUC",
                    "Definição B | S7<br>1 - Brier",
                    "Definição B | S7<br>1 - Log loss",
                ],
            },
            hover_data={
                "metric_value": ":.3f",
                "raw_value": ":.3f",
                "problem_label": False,
                "model_label": False,
                "metric_label": False,
                "comparison_label": False,
            },
            title="Comparação entre famílias nas duas combinações publicadas",
        )
        fig.update_xaxes(title="")
        fig.update_yaxes(title="", range=[0, 1.05])
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
        plot_df["problem_key"] = plot_df["problem_key"].map(format_problem_short_label)
        plot_df["model_name"] = plot_df["model_name"].map(format_model_name)
        plot_df["metric_name"] = plot_df["metric_name"].map({"ap": "AP", "brier": "Brier"})
        plot_df["serie"] = plot_df["problem_key"] + " | " + plot_df["model_name"]
        fig = px.line(
            plot_df,
            x="fold_id",
            y="metric_value",
            color="metric_name",
            facet_row="metric_name",
            markers=True,
            title="AP e Brier do modelo final em cada mês de teste",
            color_discrete_map={"AP": COLOR_INFO, "Brier": COLOR_NEGATIVE},
        )
        fig.update_traces(line=dict(width=3), marker=dict(size=10))
        fig.for_each_xaxis(lambda axis: axis.update(title_text=""))
        fig.for_each_annotation(lambda ann: ann.update(text=str(ann.text).split("=")[-1]))
        fig.update_yaxes(title="", matches=None)
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
        plot_df["serie"] = plot_df["problem_key"].map(format_problem_short_label) + " | " + plot_df["model_name"].map(format_model_name)
        preferred_series = None
        if any(plot_df["problem_key"].astype(str).str.startswith("definition_a::")):
            preferred = plot_df[plot_df["problem_key"].astype(str).str.startswith("definition_a::")].copy()
            if not preferred.empty:
                preferred = preferred.sort_values(
                    ["importance_abs", "problem_key", "model_name"],
                    ascending=[False, True, True],
                    kind="mergesort",
                )
                preferred_series = preferred.iloc[0]["serie"]
        if preferred_series is None:
            preferred_series = plot_df.sort_values(
                ["importance_abs", "serie"],
                ascending=[False, True],
                kind="mergesort",
            ).iloc[0]["serie"]
        plot_df = plot_df[plot_df["serie"] == preferred_series].copy()
        plot_df = plot_df.sort_values("importance_abs", ascending=False, kind="mergesort").head(8).copy()
        plot_df["feature_name"] = plot_df["feature_name"].map(format_feature_name)
        fig = px.bar(
            plot_df.sort_values("importance_abs", ascending=True, kind="mergesort"),
            x="importance_mean",
            y="feature_name",
            orientation="h",
            text="importance_mean",
            title=f"Sinais com maior impacto no modelo principal: {preferred_series}",
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
        fig.update_xaxes(title="Importância média por permutação")
        fig.update_yaxes(title="")
    elif kind == "final_confusion":
        plot_df = df.copy()
        if plot_df.empty:
            return ""
        if "policy_name" in plot_df.columns:
            preferred = plot_df[plot_df["policy_name"].astype(str) == "tercis"].copy()
            if not preferred.empty:
                plot_df = preferred
        matrix = (
            plot_df.groupby(["actual_group", "predicted_group"], as_index=False)["rows"]
            .sum()
            .pivot(index="actual_group", columns="predicted_group", values="rows")
            .reindex(index=["nao_realiza", "realiza"], columns=["nao_realiza", "realiza"])
            .fillna(0)
        )
        matrix.index = ["Observado: não realizou", "Observado: realizou"]
        matrix.columns = ["Previsto: não realizou", "Previsto: realizou"]
        fig = px.imshow(
            matrix,
            text_auto=True,
            color_continuous_scale="Blues",
            labels={"x": "", "y": "", "color": "Professores"},
            title="Matriz de confusão do modelo final em tercis",
        )
        fig.update_xaxes(side="top")
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
    elif kind == "score_deciles":
        plot_df = df.copy()
        if plot_df.empty:
            return ""
        plot_df = plot_df.melt(
            id_vars=["serie", "score_decile"],
            value_vars=["mean_score", "realized_rate"],
            var_name="metric_name",
            value_name="metric_value",
        )
        plot_df["metric_name"] = plot_df["metric_name"].map(
            {
                "mean_score": "score médio previsto",
                "realized_rate": "taxa realmente observada",
            }
        )
        fig = px.line(
            plot_df,
            x="score_decile",
            y="metric_value",
            color="metric_name",
            markers=True,
            title="Score previsto e taxa observada por decil",
            color_discrete_map={
                "score médio previsto": COLOR_INFO,
                "taxa realmente observada": COLOR_POSITIVE,
            },
        )
        fig.update_xaxes(title="Decil do score", dtick=1)
        fig.update_yaxes(title="Proporção", tickformat=".0%")
    elif kind == "cluster_summary":
        plot_df = df.copy()
        if plot_df.empty:
            return ""
        plot_df["cluster_name"] = plot_df["cluster_name"].map(format_cluster_name)
        fig = px.bar(
            plot_df,
            x="cluster_name",
            y="realized_inactivity_rate",
            color="cluster_name",
            text="share",
            title="Clusters descritivos no modelo final",
        )
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside", cliponaxis=False)
        fig.update_xaxes(title="")
        fig.update_yaxes(title="Taxa observada de não realização", tickformat=".0%")
    elif kind == "heavy_user_summary":
        plot_df = df.copy()
        if plot_df.empty:
            return ""
        if "policy_name" in plot_df.columns:
            preferred = plot_df[plot_df["policy_name"].astype(str) == "heavy_top_10_percent"].copy()
            if not preferred.empty:
                plot_df = preferred
        plot_df["grupo"] = plot_df["heavy_user_flag"].map({1: "heavy-user", 0: "restante"})
        fig = px.bar(
            plot_df,
            x="grupo",
            y="realized_inactivity_rate",
            color="grupo",
            text="share",
            title="Heavy-user como camada descritiva",
        )
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside", cliponaxis=False)
        fig.update_xaxes(title="")
        fig.update_yaxes(title="Taxa observada de não realização", tickformat=".0%")
    elif kind == "navigation_sequences":
        plot_df = df.copy()
        if plot_df.empty:
            return ""
        plot_df["step_sequence_first5"] = plot_df["step_sequence_first5"].map(format_navigation_sequence)
        plot_df["grupo"] = plot_df["label_value"].map({0: "não realizou", 1: "realizou"})
        fig = px.bar(
            plot_df.sort_values("teachers", ascending=True, kind="mergesort"),
            x="teachers",
            y="step_sequence_first5",
            color="grupo",
            orientation="h",
            barmode="group",
            title="Caminhos iniciais mais frequentes",
        )
        fig.update_xaxes(title="Professores")
        fig.update_yaxes(title="")
    else:
        return ""
    default_height = 520
    if kind in {"feature_importance", "cv_metric_drift", "cv_threshold_drift", "cv_confusion_drift", "score_deciles", "navigation_sequences"}:
        default_height = 620
    if kind in {"band_summary", "monthly_fit", "assumptions", "scenarios_tested", "model_selection", "cluster_summary", "heavy_user_summary"}:
        default_height = 560
    if kind == "cv_metric_drift":
        default_height = 680
    if kind == "score_deciles":
        default_height = 560
    if kind == "feature_importance":
        default_height = 680
    if kind == "final_confusion":
        default_height = 520
    if kind == "assumptions":
        fig.update_layout(height=470, margin=dict(l=40, r=30, t=90, b=78))
    else:
        fig.update_layout(height=default_height, margin=dict(l=40, r=30, t=70, b=60))
    if kind in {"assumptions", "feature_importance", "score_deciles", "cv_metric_drift", "model_selection"}:
        fig.update_layout(legend_title_text="")
    for annotation in getattr(fig.layout, "annotations", []) or []:
        if isinstance(annotation.text, str):
            annotation.text = (
                annotation.text.replace("serie=", "")
                .replace("metric_name=", "")
                .replace("validator=", "")
                .replace("problem_label=", "")
            )
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
            f"No build atual, o alvo oficial do modelo ficou em <b>{official_definition}</b>. Separadamente, o estudo de definição continua publicando uma <code>Definition A</code> oficial apenas como comparador substantivo; ela não vira alvo do modelo. A métrica nativa <code>future_business_active_weeks</code> agora conta quantas semanas futuras tiveram pelo menos uma atividade literal da lista fixa da Definição B, ou progresso em formação, ou mensagem enviada pelo usuário na Mari IA.",
        ),
        (
            "Score oficial",
            score_text,
        ),
        (
            "Como ler o score",
            "O <code>score</code> é a probabilidade calibrada de o professor entrar no grupo definido como ativo no período futuro deste relatório. Ele pode ser lido como probabilidade porque o modelo cru é recalibrado com <code>sigmoid</code> em um bloco temporal separado do treino e depois checado em meses futuros nunca vistos. O <code>risk_score = 1 - score</code> é a mesma informação vista do lado do risco de ficar fora desse grupo.",
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


def filter_to_problem_keys(df: pd.DataFrame, scope: pd.DataFrame) -> pd.DataFrame:
    if df.empty or scope.empty or "problem_key" not in df.columns or "problem_key" not in scope.columns:
        return df.copy()
    keys = scope["problem_key"].dropna().astype(str).unique().tolist()
    return df[df["problem_key"].astype(str).isin(keys)].copy()


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


def build_primary_scope(reference_scope: pd.DataFrame, serving_manifest: dict[str, Any]) -> pd.DataFrame:
    rows = serving_manifest.get("reference_scope_rows", []) if serving_manifest else []
    if rows:
        return pd.DataFrame([rows[0]])
    if reference_scope.empty:
        return pd.DataFrame()
    return reference_scope.head(1).copy()


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


def build_assumption_event_points(track_registry: pd.DataFrame, arbitrariness: pd.DataFrame) -> pd.DataFrame:
    if track_registry.empty:
        return pd.DataFrame()
    label_window_days = _choice_numeric(arbitrariness, "label_window_days", 30) or 30
    future_start_day = 8
    future_end_day = future_start_day + label_window_days - 1
    show = track_registry.copy()
    if "official_flag" in show.columns:
        show = show[show["official_flag"] == 1].copy()
    rows: list[dict[str, Any]] = []
    for row in show.to_dict(orient="records"):
        track_label = format_track_name(row.get("track_name", ""))
        score_day = int(pd.to_numeric(row.get("score_window_end_day"), errors="coerce") or 0)
        score_detail = str(row.get("score_moment_text", "")).strip()
        rows.extend(
            [
                {
                    "track_name": track_label,
                    "event_name": "momento do score",
                    "event_detail": score_detail or "Momento em que o modelo para de olhar a entrada.",
                    "day_number": score_day,
                },
                {
                    "track_name": track_label,
                    "event_name": "início do resultado futuro",
                    "event_detail": "A janela do resultado começa depois de completar 7 dias desde a âncora.",
                    "day_number": future_start_day,
                },
                {
                    "track_name": track_label,
                    "event_name": "fim da janela do resultado",
                    "event_detail": f"O resultado futuro é medido por {label_window_days} dias corridos.",
                    "day_number": future_end_day,
                },
            ]
        )
    return pd.DataFrame(rows)


def build_score_decile_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or "score" not in predictions.columns or "y_true" not in predictions.columns:
        return pd.DataFrame()
    show = predictions.copy()
    if "fold_valid_flag" in show.columns:
        show = show[show["fold_valid_flag"] == 1].copy()
    if "technical_fold_valid_flag" in show.columns:
        show = show[show["technical_fold_valid_flag"] == 1].copy()
    if show.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for (problem_key, model_name), group in show.groupby(["problem_key", "model_name"], dropna=False):
        group = group.sort_values("score", kind="mergesort").reset_index(drop=True)
        if group.empty:
            continue
        bins = min(10, len(group))
        if bins <= 1:
            group["score_decile"] = 1
        else:
            group["score_decile"] = pd.qcut(
                np.arange(len(group)),
                q=bins,
                labels=False,
                duplicates="drop",
            ) + 1
        summary = (
            group.groupby("score_decile", as_index=False)
            .agg(
                mean_score=("score", "mean"),
                realized_rate=("y_true", "mean"),
                rows=("y_true", "size"),
            )
            .sort_values("score_decile", kind="mergesort")
        )
        summary["problem_key"] = problem_key
        summary["model_name"] = model_name
        summary["serie"] = format_problem_short_label(problem_key) + " | " + format_model_name(model_name)
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_navigation_sequence_summary(navigation_sequences: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    if navigation_sequences.empty:
        return pd.DataFrame()
    show = navigation_sequences.copy()
    totals = (
        show.groupby("step_sequence_first5", as_index=False)["teachers"]
        .sum()
        .sort_values("teachers", ascending=False, kind="mergesort")
        .head(top_n)
    )
    return show[show["step_sequence_first5"].isin(totals["step_sequence_first5"])].copy()


def _get_reference_model_row(model_frontier: pd.DataFrame, problem_key: str) -> pd.Series | None:
    if model_frontier.empty:
        return None
    match = model_frontier[model_frontier["problem_key"].astype(str) == str(problem_key)].copy()
    if match.empty:
        return None
    return match.iloc[0]


def _get_threshold_row(threshold_metrics: pd.DataFrame, problem_key: str, policy_name: str) -> pd.Series | None:
    if threshold_metrics.empty:
        return None
    match = threshold_metrics[
        (threshold_metrics["problem_key"].astype(str) == str(problem_key))
        & (threshold_metrics["policy_name"].astype(str) == str(policy_name))
    ].copy()
    if match.empty:
        return None
    return match.iloc[0]


def _get_confusion_rows(confusion_df: pd.DataFrame, problem_key: str, policy_name: str) -> pd.DataFrame:
    if confusion_df.empty:
        return pd.DataFrame()
    return confusion_df[
        (confusion_df["problem_key"].astype(str) == str(problem_key))
        & (confusion_df["policy_name"].astype(str) == str(policy_name))
    ].copy()


def _get_confusion_value(confusion_rows: pd.DataFrame, actual_group: str, predicted_group: str) -> int:
    if confusion_rows.empty:
        return 0
    match = confusion_rows[
        (confusion_rows["actual_group"].astype(str) == actual_group)
        & (confusion_rows["predicted_group"].astype(str) == predicted_group)
    ]
    if match.empty:
        return 0
    return int(match.iloc[0].get("rows", 0))


def _resolve_primary_problem_model(
    model_frontier: pd.DataFrame,
    serving_manifest: dict[str, Any],
) -> tuple[str, str]:
    if serving_manifest:
        rows = serving_manifest.get("reference_scope_rows", [])
        if rows:
            return str(rows[0].get("problem_key", "")), str(rows[0].get("model_name", ""))
    if not model_frontier.empty:
        row = model_frontier.iloc[0]
        return str(row.get("problem_key", "")), str(row.get("model_name", ""))
    return "", ""


def build_assumptions_block_text(
    track_registry: pd.DataFrame,
    arbitrariness: pd.DataFrame,
    feature_registry: pd.DataFrame,
) -> tuple[str, str, str, str, str]:
    label_window_days = _choice_numeric(arbitrariness, "label_window_days", 30) or 30
    quick_definition = "O modelo só lê o que já existia no momento da previsão."
    how_text = (
        "<code>S1</code> para no fim da 1ª sessão. <code>S7</code> e <code>S1+S7</code> param no fim dos primeiros 7 dias corridos. "
        "<code>STRICT_CONTEXT</code> para no mesmo ponto, mas só com contexto inicial. Depois disso começa a janela futura do resultado: "
        f"do <code>day_8</code> ao <code>day_{8 + label_window_days - 1}</code>. Esses números aparecem porque os primeiros 7 dias ficam reservados para ler os sinais iniciais, "
        f"e os 30 dias seguintes são usados para medir se o professor continuou ativo."
    )
    why_text = "A previsão sempre termina antes do período que depois será medido."
    what_text = ""
    conclusion_text = (
        "Esperar 7 dias abre mais informação do que parar na 1ª sessão; <code>STRICT_CONTEXT</code> é o cenário mais conservador."
    )
    return quick_definition, how_text, why_text, what_text, conclusion_text


def build_methodology_protocol_block_text(
    build_dir: Path,
    summary: dict[str, Any],
    future_metrics: pd.DataFrame,
    definition_selection: pd.DataFrame,
    candidate_metric_registry: pd.DataFrame,
) -> tuple[str, str, str, list[tuple[str, list[str]]], str]:
    a_candidates = (
        definition_selection[definition_selection["definition_group"].astype(str) == "definition_a"].head(1)
        if not definition_selection.empty and "definition_group" in definition_selection.columns
        else pd.DataFrame()
    )
    a_row = a_candidates
    current_rule = format_rule_text(a_row.iloc[0]["rule_text"]) if not a_row.empty and "rule_text" in a_row.columns else "n/d"
    candidate_metrics: list[str] = []
    if not candidate_metric_registry.empty and "definition_a_candidate_flag" in candidate_metric_registry.columns:
        candidate_work = candidate_metric_registry[
            pd.to_numeric(candidate_metric_registry["definition_a_candidate_flag"], errors="coerce").fillna(0).astype(int) == 1
        ].copy()
        if "metric_name" in candidate_work.columns:
            candidate_metrics = [format_metric_name(metric) for metric in candidate_work["metric_name"].dropna().astype(str).tolist()]
    periods_materialized = (build_dir / "tables" / "governance_definition_selection_periods_v1.parquet").exists()
    development_months = int(pd.to_numeric(summary.get("definition_selection_development_months"), errors="coerce")) if pd.notna(pd.to_numeric(summary.get("definition_selection_development_months"), errors="coerce")) else 0
    lock_months = int(pd.to_numeric(summary.get("definition_lock_realized_months"), errors="coerce")) if pd.notna(pd.to_numeric(summary.get("definition_lock_realized_months"), errors="coerce")) else 0
    final_eval_months = int(pd.to_numeric(summary.get("final_model_evaluation_months"), errors="coerce")) if pd.notna(pd.to_numeric(summary.get("final_model_evaluation_months"), errors="coerce")) else 0
    split_text = (
        "O protocolo fechado do estudo separa <code>development</code>, <code>definition lock</code> e "
        "<code>final untouched model evaluation</code>."
    )
    if development_months > 0 or lock_months > 0 or final_eval_months > 0:
        split_text += (
            f" Nesta execução, isso corresponde a <code>{development_months}</code> meses de development, "
            f"<code>{lock_months}</code> meses de definition lock e "
            f"<code>{final_eval_months}</code> meses de avaliação final do modelo."
        )
    if not periods_materialized:
        split_text += " O build materializado atual ainda não traz a tabela dessa separação."
    removed_population_text = (
        "A decomposição exata dos casos excluídos por <code>same_month_entry_only</code> ainda não está materializada neste build."
    )
    if not future_metrics.empty and {"full_followup_observed_flag", "months_after_entry"}.issubset(future_metrics.columns):
        population_frame = future_metrics.copy()
        population_frame = population_frame[
            pd.to_numeric(population_frame["full_followup_observed_flag"], errors="coerce").fillna(0).astype(int) == 1
        ].copy()
        months_after_entry = pd.to_numeric(population_frame.get("months_after_entry"), errors="coerce")
        removed_mask = months_after_entry.ne(0)
        removed_total = int(removed_mask.sum())
        delayed_total = int((months_after_entry > 0).sum())
        inconsistent_total = int((months_after_entry < 0).sum())
        removed_population_text = (
            "Na execução materializada atual, esse filtro excluiu "
            f"<code>{removed_total:,}</code> professores distintos: "
            f"<code>{delayed_total:,}</code> casos de entrada observada com atraso "
            "(\u200b<code>months_after_entry &gt; 0</code>) e "
            f"<code>{inconsistent_total:,}</code> inconsistências temporais "
            "(\u200b<code>months_after_entry &lt; 0</code>)."
        ).replace(",", ".")
    gate_spec = setup.get_definition_lock_bootstrap_gate_spec()
    support_gate_text = (
        "Para um fold mensal da definição ou do modelo entrar no resumo oficial, ele precisa passar no gate mínimo de suporte: "
        f"pelo menos <code>{int(setup.MIN_OFFICIAL_TEST_ROWS)}</code> linhas, "
        f"<code>{int(setup.MIN_OFFICIAL_TEST_POSITIVES)}</code> positivos e "
        f"<code>{int(setup.MIN_OFFICIAL_TEST_NEGATIVES)}</code> negativos."
    )
    lock_gate_text = (
        "O output publicado desse bootstrap é <code>ci_low</code>, <code>ci_high</code> e <code>ci_width</code>. "
        "No lock oficial, a candidata só pode sobreviver se "
        f"<code>{gate_spec['column_name']} {gate_spec['operator']} {float(gate_spec['threshold']):g}</code>; "
        "isso faz a regra dura do lock ficar explícita e configurável na spec."
    )
    population_text = (
        "A população oficial principal é <code>same_month_entry_only</code>: "
        "ficam apenas professores com <code>months_after_entry == 0</code>, isto é, "
        "cujo <code>first_month</code> observado coincide com o mês de entrada cadastral."
    )
    if str(summary.get("official_population_filter", "")) != "same_month_entry_only":
        population_text = "A população oficial atual não está marcada como <code>same_month_entry_only</code> neste summary."
    sections = [
        (
            "Guardrails fechados",
            [
                "<code>Definition A</code> e <code>Definition B</code> devem ser decididas sem AP, Brier, confusion matrix, feature importance ou qualquer outra métrica de modelo.",
                split_text,
                (
                    f"Com <code>{development_months}</code> meses em development, a busca da definição gera "
                    f"<code>{max(0, development_months - 1)}</code> outer folds mensais: em cada fold, treina em meses acumulados anteriores e testa no mês seguinte."
                ) if development_months > 0 else "A busca da definição usa outer folds mensais em expanding window: treino em meses acumulados anteriores e teste no mês seguinte.",
                population_text,
                "Na prática, esse filtro remove professores cujo primeiro uso observável aconteceu meses depois da entrada cadastral, além de poucos casos com inconsistência temporal. O objetivo é comparar a coorte oficial sempre no mesmo estágio de ciclo de vida.",
                removed_population_text,
                "Esses casos excluídos não são duplicatas da base analítica oficial; são professores distintos que não estão no mês zero da jornada observada.",
                "O estudo de definição continua separado e usa os validadores pós-label de 90 dias apenas como checagem externa de continuidade; a modelagem oficial desta análise usa apenas o alvo fixo de 30 dias <code>definition_b_label</code>.",
                "Os 90 dias não redefinem o label e não repetem nem a regra da <code>Definition A</code> nem a <code>Definition B</code>; eles medem continuidade futura com um construto fixo comum a todas as candidatas.",
                support_gate_text,
                f"No modelo, o outer test oficial é limitado aos <code>{int(setup.MAX_OUTER_TEST_MONTHS)}</code> meses mais recentes do protocolo; isso não significa 'usar só esse número de meses no total', porque cada fold do modelo treina em todo o histórico oficial anterior acumulado.",
                "Tuning, calibração e outer test continuam separados temporalmente dentro da etapa de modelagem.",
            ],
        ),
        (
            "Como ler gap e grupos",
            [
                "Em cada outer fold da definição, a candidata divide o mês de teste em dois grupos: positivos = professores que a regra marcou como ativos na janela inicial; negativos = professores que a regra não marcou como ativos nessa mesma janela.",
                "O grupo negativo não significa 'permaneceu inativo depois'. Ele pode voltar a aparecer como ativo nos 90 dias seguintes; só significa que a regra não o marcou como ativo no primeiro window.",
                "Para cada validador pós-label, <b>gap</b> significa: média do validador no grupo positivo menos média do mesmo validador no grupo negativo.",
                "Exemplo: se 46% dos positivos sustentaram atividade futura e 19% dos negativos também sustentaram atividade futura, o gap do fold é <code>0.46 - 0.19 = 0.27</code>.",
            ],
        ),
        (
            "Busca oficial da Definition A",
            [
                "O search space oficial precisa incluir regras atômicas <code>m &gt;= t</code>, compostas booleanas <code>AND/OR</code> e combinações ponderadas <code>w1*z1 + w2*z2 &gt;= τ</code>.",
                "Combinações entre métricas são requisito explícito; o motor suportar <code>AND/OR</code> não basta, a busca oficial precisa gerá-las.",
                "Pesos entre métricas exigem escala comparável; a recomendação atual é percentil empírico no treino do fold, mapeando cada métrica para <code>[0,1]</code>.",
                (
                    "O universo atual de métricas candidatas da <code>Definition A</code> é uma decisão de projeto, não algo descoberto automaticamente: "
                    + ", ".join(f"<code>{metric}</code>" for metric in candidate_metrics)
                ) if candidate_metrics else "O universo de métricas candidatas da <code>Definition A</code> precisa ser explicitado como política do estudo.",
            ],
        ),
        (
            "Sensibilidades obrigatórias",
            [
                "<code>Threshold testing</code> é parte oficial do protocolo: thresholds testados e sensibilidade local precisam ficar registrados.",
                "Para regras compostas, o lock precisa medir <code>threshold sensitivity</code>, troca <code>AND -&gt; OR</code>, <code>drop-one-literal</code> e <code>weight perturbation</code>.",
                "Thresholds pseudo-exatos como <code>8.3138</code> devem ser tratados como cutpoints empíricos; sua leitura substantiva vem da estabilidade local, não do número bruto.",
                "Os validadores de 90 dias usam um construto fixo de continuação comportamental: downloads, criações, compartilhamentos e views de conteúdo pedagógico central. Eles não usam a regra da candidata sendo testada.",
                "O bootstrap da definição não reamostra professores crus. Ele pega, para uma mesma candidata, a lista dos gaps observados ao longo dos folds válidos e reamostra essa lista com reposição para estimar a estabilidade da média do gap entre meses.",
                lock_gate_text,
                "Depois desse corte duro, <code>ci_width</code> vira critério de estabilidade: largura pequena quer dizer que a média do gap muda pouco quando reamostramos os folds; largura grande quer dizer instabilidade maior entre meses.",
            ],
        ),
        (
            "Overfitting e underfitting",
            [
                "O protocolo agora exige uma auditoria formal de generalização comparando <code>apparent_train</code> e <code>calibration_holdout</code> contra <code>outer test</code>, com gap positivo significando desempenho melhor dentro do treino do que fora dele.",
                "Para reduzir overfitting, o ranking oficial precisa privilegiar candidatos com menor gap de generalização, menos complexidade de regra e melhor estabilidade temporal antes de preferir ganhos marginais de métrica.",
                "Para reduzir underfitting, a busca oficial pode ampliar interações controladas via <code>AND/OR</code> e regras ponderadas, mas sem abrir árvores profundas ou search spaces pouco auditáveis na etapa de definição.",
            ],
        ),
        (
            "Serving e auditoria",
            [
                "Nesta arquitetura, o estudo de definição não escolhe mais o alvo do modelo; a modelagem oficial roda apenas sobre <code>definition_b_label</code>.",
                "<code>serving</code> e <code>reference scope</code> seguem o grupo único presente no <code>model_frontier</code> materializado, em vez de herdar o vencedor do estudo de definição.",
                "A trilha de auditoria continua precisando refletir o escopo real da seleção, sem fallback silencioso e sem metadata enganosa.",
            ],
        ),
    ]
    quick_definition = "O relatório agora declara explicitamente o protocolo metodológico e separa o que é regra fechada do que ainda é gap de implementação."
    how_text = (
        "As regras de governança do estudo ficam visíveis no próprio relatório: como o alvo deve ser escolhido, "
        "que tipos de regra a <code>Definition A</code> deve explorar, como thresholds/pesos precisam ser auditados e qual é a política correta de serving."
    )
    why_text = "Sem isso, o leitor fica forçado a inferir metodologia a partir de tabelas finais e pode confundir artefato materializado com protocolo oficial."
    conclusion_text = (
        f"No material vigente, a <code>Definition A</code> oficial materializada é <code>{current_rule}</code>; "
        "o protocolo agora exige que a explicação dessa escolha apareça no relatório, não só no código ou em conversas paralelas."
    )
    return quick_definition, how_text, why_text, sections, conclusion_text


def build_methodology_gaps_block_text(
    build_dir: Path,
    definition_selection: pd.DataFrame,
    serving_manifest: dict[str, Any],
) -> tuple[str, str, str, list[tuple[str, list[str]]], str]:
    a_candidates = (
        definition_selection[definition_selection["definition_group"].astype(str) == "definition_a"].head(1)
        if not definition_selection.empty and "definition_group" in definition_selection.columns
        else pd.DataFrame()
    )
    a_row = a_candidates
    current_rule = format_rule_text(a_row.iloc[0]["rule_text"]) if not a_row.empty and "rule_text" in a_row.columns else "n/d"
    serving_status = str(serving_manifest.get("serving_status", "n/d")) if serving_manifest else "n/d"
    selection_meta = serving_manifest.get("selection_meta", {}) if serving_manifest else {}
    current_scope = str(selection_meta.get("selection_scope", "n/d"))
    sections = [
        (
            "Fatos do materializado atual",
            [
                "O estado materializado vigente é o de <code>build/tables</code>; o rerun rigoroso deletado não é mais a base para leitura do projeto.",
                f"A <code>Definition A</code> oficial hoje materializada é <code>{current_rule}</code>.",
                "Isso prova que o projeto materializado já usou regra composta; portanto, não é correto dizer genericamente que o projeto não usa <code>AND/OR</code>.",
            ],
        ),
        (
            "Inconsistências e gaps que o relatório precisa admitir",
            [
                "O artefato vigente tem <code>Definition A</code> composta, enquanto o build materializado atual ainda é anterior ao search oficial composto/ponderado agora implementado no código; portanto, código e materializado continuam defasados até o próximo rerun oficial.",
                "O repositório atual não sustenta sozinho a narrativa auditável <code>5, 6, 7, 8, 9, 10 -&gt; 8 -&gt; 8.3138</code>.",
                "Ainda falta uma etapa explícita de <code>definition evaluability/modelability audit</code> antes da modelagem oficial.",
                "O código agora prevê uma auditoria formal <code>apparent_train/calibration_holdout vs outer test</code>, mas o build materializado atual só vai exibir essa tabela depois de um novo rerun de <code>build-ml</code>.",
                "Evento raro + outer test mensal + poucos positivos aumenta variância e fragilidade interpretativa; isso precisa ser lido como limitação estatística, não como prova automática de overfitting.",
            ],
        ),
        (
            "Estado atual do serving",
            [
                f"O <code>serving_status</code> materializado hoje é <code>{serving_status}</code>.",
                f"A metadata atual do serving reporta <code>selection_scope = {current_scope}</code>; o relatório precisa deixar claro quando esse valor pertence a um artefato antigo e não à política corrigida do código.",
                "Na arquitetura atual, o modelo oficial usa apenas <code>definition_b_label</code>; portanto, o serving não deve reaproveitar o vencedor do estudo de definição para redefinir o alvo do modelo.",
            ],
        ),
    ]
    quick_definition = "Além do protocolo, o relatório também precisa admitir explicitamente onde o materializado atual ainda diverge do código ou do desenho metodológico desejado."
    how_text = (
        "Este bloco separa fato materializado, inconsistência conhecida e pendência metodológica. Ele evita que o relatório pareça mais fechado do que o projeto realmente está."
    )
    why_text = "Sem essa camada, o HTML passa a impressão errada de que artefato, search atual, serving e auditorias adicionais já estão completamente alinhados."
    conclusion_text = (
        "O relatório passa a distinguir explicitamente o que já é fato do build atual, o que é política corrigida no código e o que ainda precisa ser fechado antes do próximo rerun oficial."
    )
    return quick_definition, how_text, why_text, sections, conclusion_text


def build_activity_definition_block_text(definition_frontier: pd.DataFrame) -> tuple[str, str, str, str, str]:
    if definition_frontier.empty:
        return (
            "Definição de atividade é a regra que decide quem foi considerado ativo no futuro.",
            "Nesta materialização não há definições oficiais publicadas para comparar.",
            "Sem definição clara, não existe alvo confiável para o score.",
            "O bloco de definições não foi materializado nesta execução.",
            "Sem definição publicada, não existe score interpretável.",
        )
    a_row = definition_frontier[definition_frontier["definition_name"].astype(str).str.startswith("definition_a")].head(1)
    b_row = definition_frontier[definition_frontier["definition_name"].astype(str).str.startswith("definition_b")].head(1)
    a_row = a_row.iloc[0] if not a_row.empty else None
    b_row = b_row.iloc[0] if not b_row.empty else None
    quick_definition = "Definição de atividade é a regra que separa quem realizou e quem não realizou a atividade futura."
    how_text = (
        "A <b>Definição B</b> é o comparador fixo e literal do negócio na janela de 30 dias. "
        "A <b>Definição A</b> é a definição pesquisada, também aplicada nessa janela de 30 dias. "
        "Depois disso, as duas são julgadas pelos mesmos validadores pós-label de 90 dias, que medem continuidade futura e não repetem nem a regra da A nem a regra da B."
    )
    why_text = "Antes de olhar o score, o próprio alvo precisa ser congelado e separar melhor o que acontece depois, sem AP, Brier ou confusion matrix entrarem na decisão."
    if a_row is not None and b_row is not None:
        a_share = pd.to_numeric(a_row.get("label_share_pct"), errors="coerce") / 100.0
        b_share = pd.to_numeric(b_row.get("label_share_pct"), errors="coerce") / 100.0
        what_text = (
            f"Nesta execução, a <b>Definição A</b> marcou <b>{int(a_row.get('label_positives', 0))}</b> professores "
            f"(<b>{format_percent(a_share)}</b> da base), enquanto a <b>Definição B</b> marcou <b>{int(b_row.get('label_positives', 0))}</b> "
            f"(<b>{format_percent(b_share)}</b>). A regra oficial materializada da <b>Definição A</b> é <code>{format_rule_text(a_row.get('rule_text'))}</code>. "
            f"Ela ficou mais exigente e também entregou gaps externos maiores. Aqui, <b>gap</b> significa: média do validador futuro entre os marcados como ativos menos média do mesmo validador entre os marcados como não ativos. No 1º bloco pós-label, "
            f"o gap de retorno foi <b>{format_number(a_row.get('test_gap_returned_active_post_label_m1'), 3)}</b> na A contra "
            f"<b>{format_number(b_row.get('test_gap_returned_active_post_label_m1'), 3)}</b> na B; em dias ativos acumulados dos 3 blocos, "
            f"foi <b>{format_number(a_row.get('test_gap_active_days_post_label_3m'), 3)}</b> na A contra "
            f"<b>{format_number(b_row.get('test_gap_active_days_post_label_3m'), 3)}</b> na B."
        )
    else:
        what_text = "O gráfico compara as definições oficiais nos validadores externos, mas uma das linhas esperadas não ficou materializada."
    conclusion_text = "A leitura correta aqui é: a <b>Definição A</b> é a definição pesquisada, a <b>Definição B</b> é o comparador fixo, e os 90 dias entram apenas como validação externa comum para ambas."
    return quick_definition, how_text, why_text, what_text, conclusion_text


def build_scenarios_block_text(scoring_scenarios: pd.DataFrame) -> tuple[str, str, str, str, str]:
    if scoring_scenarios.empty:
        return (
            "Cada combinação junta uma definição, uma trilha temporal e um conjunto permitido de variáveis.",
            "As combinações testadas não ficaram materializadas nesta execução.",
            "Sem esse mapa, não dá para saber o que realmente foi comparado.",
            "O bloco de cenários não foi materializado.",
            "Sem cenários publicados, a comparação principal fica incompleta.",
        )
    rows = int(pd.to_numeric(scoring_scenarios["rows"], errors="coerce").dropna().max() or 0)
    months = int(pd.to_numeric(scoring_scenarios["months"], errors="coerce").dropna().max() or 0)
    definitions = sorted(scoring_scenarios["definition_name"].dropna().astype(str).unique().tolist()) if "definition_name" in scoring_scenarios.columns else []
    problem_count = int(len(scoring_scenarios))
    track_count = int(scoring_scenarios["track_name"].dropna().astype(str).nunique()) if "track_name" in scoring_scenarios.columns else 0
    feature_counts = {
        format_track_name(track): int(pd.to_numeric(group["feature_count"], errors="coerce").dropna().max() or 0)
        for track, group in scoring_scenarios.groupby("track_name", dropna=False)
    }
    quick_definition = "Cada combinação junta uma definição, uma trilha temporal e um conjunto permitido de variáveis."
    how_text = (
        f"Foram testadas <b>{max(1, len(definitions))} definição(ões) oficiais de modelagem</b> em <b>{track_count}</b> trilhas "
        "(<code>S1</code>, <code>S7</code>, <code>S1+S7</code> e <code>STRICT_CONTEXT</code>, quando disponíveis). "
        f"Isso gera <b>{problem_count}</b> combinações principais. Em cada combinação, o pipeline compara <b>3 famílias de modelo</b>: regressão logística, random forest e CatBoost."
    )
    why_text = (
        "Isso importa porque o modelo final não saiu de um teste único. Primeiro foi comparado o mesmo problema em trilhas com quantidades diferentes de informação observável antes do score."
    )
    what_text = (
        f"As {problem_count} combinações usaram a mesma base analítica, com <b>{rows}</b> linhas e <b>{months}</b> meses. "
        f"O número de variáveis elegíveis foi <b>{feature_counts.get('S1', 0)}</b> no <code>S1</code>, "
        f"<b>{feature_counts.get('S7', 0)}</b> no <code>S7</code>, <b>{feature_counts.get('S1+S7', 0)}</b> no <code>S1+S7</code> "
        f"e <b>{feature_counts.get('STRICT_CONTEXT', 0)}</b> no <code>STRICT_CONTEXT</code>."
    )
    conclusion_text = "Na arquitetura atual, a modelagem oficial roda apenas sobre o alvo fixo de 30 dias <code>definition_b_label</code>; o estudo de definição fica fora desta etapa."
    return quick_definition, how_text, why_text, what_text, conclusion_text


def build_model_comparison_block_text(
    model_frontier: pd.DataFrame,
    definition_frontier: pd.DataFrame,
    serving_manifest: dict[str, Any],
) -> tuple[str, str, str, str, str]:
    if model_frontier.empty:
        return (
            "Os modelos comparados são as famílias candidatas para o mesmo problema.",
            "A comparação entre famílias não ficou materializada nesta execução.",
            "Sem essa comparação, não dá para justificar a escolha do modelo.",
            "O bloco de comparação entre modelos não foi materializado.",
            "Sem comparação publicada, a escolha do modelo fica opaca.",
        )
    show = model_frontier.copy()
    show["selected_flag"] = 0
    primary_problem_key = ""
    primary_model_name = ""
    if serving_manifest:
        rows = serving_manifest.get("reference_scope_rows", [])
        if rows:
            primary_problem_key = str(rows[0].get("problem_key", ""))
            primary_model_name = str(rows[0].get("model_name", ""))
            mask = (
                show["problem_key"].astype(str).eq(primary_problem_key)
                & show["model_name"].astype(str).eq(primary_model_name)
            )
            show.loc[mask, "selected_flag"] = 1
    quick_definition = "Foram comparadas três famílias: regressão logística, random forest e CatBoost."
    how_text = (
        f"Nas <b>{show['problem_key'].nunique()}</b> combinações publicadas, as três famílias foram comparadas nas mesmas quatro métricas: <b>AP</b>, <b>ROC AUC</b>, <b>Brier</b> e <b>log loss</b>."
    )
    why_text = "O melhor modelo precisa ranquear bem e também errar pouco na probabilidade."
    problem_texts: list[str] = []
    for problem_key, group in show.groupby("problem_key", dropna=False):
        group = group.sort_values(["mean_brier", "mean_log_loss", "mean_ap", "mean_roc_auc"], ascending=[True, True, False, False], kind="mergesort")
        best = group.iloc[0]
        second = group.iloc[1] if len(group) > 1 else None
        part = (
            f"Em <b>{format_problem_short_label(problem_key)}</b>, <b>{format_model_name(best.get('model_name'))}</b> ficou em 1º lugar. "
        )
        if second is not None:
            part += f"O 2º lugar ficou com <b>{format_model_name(second.get('model_name'))}</b>."
        problem_texts.append(part)
    primary_text = ""
    if primary_problem_key and primary_model_name:
        primary_text = f" O modelo servível primário ficou em <b>{format_problem_short_label(primary_problem_key)}</b> com <b>{format_model_name(primary_model_name)}</b>."
    a_row = definition_frontier[definition_frontier["definition_name"].astype(str).str.startswith("definition_a")].head(1)
    b_row = definition_frontier[definition_frontier["definition_name"].astype(str).str.startswith("definition_b")].head(1)
    prevalence_text = ""
    if not a_row.empty and not b_row.empty:
        a_share = float(pd.to_numeric(a_row.iloc[0].get("label_share_pct"), errors="coerce") or 0.0)
        b_share = float(pd.to_numeric(b_row.iloc[0].get("label_share_pct"), errors="coerce") or 0.0)
        prevalence_text = (
            f" A <b>Definição B</b> ficou com AP maior em parte porque é um alvo mais amplo: ela marca <b>{format_number(b_share, 1)}%</b> da base, "
            f"contra <b>{format_number(a_share, 1)}%</b> na <b>Definição A</b>. Isso tende a facilitar o ranking e, por si só, não torna a definição melhor."
        )
    definition_b_text = (
        " A tabela mostra só a <b>melhor combinação publicada por definição</b>. "
        "Na <b>Definição B</b>, <code>S1</code>, <code>S7</code> e <code>S1+S7</code> foram comparadas. "
        "A publicada ficou em <b>S7</b> porque foi ligeiramente melhor que <code>S1+S7</code> ao mesmo tempo em <b>AP</b> "
        "(0.468 contra 0.465) e em <b>Brier</b> (0.085 contra 0.085, com pequena vantagem para <code>S7</code>)."
    )
    what_text = " ".join(problem_texts) + definition_b_text + prevalence_text + primary_text
    conclusion_text = ""
    return quick_definition, how_text, why_text, what_text, conclusion_text


def build_trust_block_text(
    model_frontier: pd.DataFrame,
    cv_metric_folds: pd.DataFrame,
    cv_threshold_summary: pd.DataFrame,
    cv_confusion_summary: pd.DataFrame,
) -> tuple[str, str, str, str, str]:
    quick_definition = "O modelo foi testado em meses futuros que não entraram no treino."
    how_text = (
        "Em cada mês futuro fora do treino, o pipeline recalculou duas leituras do mesmo modelo. A primeira foi o <b>score contínuo</b>: "
        "<b>AP</b>, que resume se os casos de maior risco aparecem mais no topo do ranking, e <b>Brier</b>, que mede o erro médio da probabilidade publicada naquele mês. "
        "A segunda foi a <b>fila operacional</b>: com o corte em <code>tercis</code>, o pipeline mediu <b>precisão</b>, <b>recall</b>, <b>F1</b>, <b>acurácia</b> e a <b>matriz de confusão</b> completa daquele próprio mês."
    )
    why_text = (
        "Isso importa porque robustez, aqui, não é um único número. É mostrar que o mesmo modelo continua funcionando de dois jeitos quando o mês muda: "
        "como score contínuo, que ordena a base do menor para o maior risco, e como fila operacional, que transforma esse score em decisão prática."
    )
    if model_frontier.empty:
        return quick_definition, how_text, why_text, "As métricas de confiança não ficaram materializadas nesta execução.", "Sem folds válidos, não há confiança publicável."
    row = model_frontier.iloc[0]
    problem_key = str(row.get("problem_key", ""))
    fold_metrics = cv_metric_folds[
        (cv_metric_folds["problem_key"].astype(str) == problem_key)
        & (cv_metric_folds["model_name"].astype(str) == str(row.get("model_name", "")))
    ].copy()
    ap_values = pd.to_numeric(fold_metrics.loc[fold_metrics["metric_name"] == "ap", "metric_value"], errors="coerce").dropna()
    brier_values = pd.to_numeric(fold_metrics.loc[fold_metrics["metric_name"] == "brier", "metric_value"], errors="coerce").dropna()
    if ap_values.empty or brier_values.empty:
        return quick_definition, how_text, why_text, "Os folds válidos do modelo final não ficaram materializados nesta execução.", "Sem outer folds válidos, não há confiança publicável."
    tercis_precision = tercis_recall = None
    confusion_tp = confusion_fn = confusion_fp = confusion_tn = None
    if not cv_threshold_summary.empty:
        subset = cv_threshold_summary[
            (cv_threshold_summary["problem_key"].astype(str) == problem_key)
            & (cv_threshold_summary["model_name"].astype(str) == str(row.get("model_name", "")))
            & (cv_threshold_summary["policy_name"].astype(str) == "tercis")
        ].copy()
        precision_row = subset[subset["metric_name"].astype(str) == "precision"]
        recall_row = subset[subset["metric_name"].astype(str) == "recall"]
        if not precision_row.empty:
            tercis_precision = precision_row.iloc[0]
        if not recall_row.empty:
            tercis_recall = recall_row.iloc[0]
    if not cv_confusion_summary.empty:
        confusion_subset = cv_confusion_summary[
            (cv_confusion_summary["problem_key"].astype(str) == problem_key)
            & (cv_confusion_summary["model_name"].astype(str) == str(row.get("model_name", "")))
            & (cv_confusion_summary["policy_name"].astype(str) == "tercis")
        ].copy()
        if not confusion_subset.empty:
            def _pick(actual: str, predicted: str) -> Any:
                match = confusion_subset[
                    (confusion_subset["actual_group"].astype(str) == actual)
                    & (confusion_subset["predicted_group"].astype(str) == predicted)
                ]
                return match.iloc[0] if not match.empty else None

            confusion_tp = _pick("nao_realiza", "nao_realiza")
            confusion_fn = _pick("nao_realiza", "realiza")
            confusion_fp = _pick("realiza", "nao_realiza")
            confusion_tn = _pick("realiza", "realiza")
    what_text = (
        f"Nos <b>{int(row.get('valid_folds', 0))}</b> meses futuros válidos, o pipeline checou o score contínuo com <b>AP</b>, <b>ROC AUC</b>, <b>Brier</b> e <b>log loss</b>; "
        f"na leitura operacional, recalculou a fila alta com <b>precisão</b>, <b>recall</b>, <b>F1</b>, <b>acurácia</b> e a <b>matriz de confusão</b> do próprio mês. "
        f"No modelo final, a <b>AP</b> ficou entre <b>{format_number(ap_values.min(), 3)}</b> e <b>{format_number(ap_values.max(), 3)}</b>; "
        f"o <b>Brier</b> ficou entre <b>{format_number(brier_values.min(), 3)}</b> e <b>{format_number(brier_values.max(), 3)}</b>. "
    )
    if tercis_precision is not None and tercis_recall is not None:
        what_text += (
            f"Na fila alta em <code>tercis</code>, a <b>precisão média</b> ficou em <b>{format_percent(float(tercis_precision.get('mean_value', np.nan)))}</b> "
            f"e o <b>recall médio</b> em <b>{format_percent(float(tercis_recall.get('mean_value', np.nan)))}</b>. "
            f"Isso quer dizer o seguinte: quando o modelo colocava alguém na fila alta, quase não errava; o custo era deixar uma parte do risco fora da fila nesse primeiro corte. "
        )
    if all(item is not None for item in [confusion_tp, confusion_fn, confusion_fp, confusion_tn]):
        what_text += (
            f" Na média dos folds mensais nessa fila, o modelo capturou <b>{format_number(float(confusion_tp.get('mean_rows', np.nan)), 1)}</b> casos de inatividade, "
            f"deixou <b>{format_number(float(confusion_fn.get('mean_rows', np.nan)), 1)}</b> casos de inatividade fora da fila, "
            f"gerou <b>{format_number(float(confusion_fp.get('mean_rows', np.nan)), 1)}</b> alarmes falsos "
            f"e preservou <b>{format_number(float(confusion_tn.get('mean_rows', np.nan)), 1)}</b> casos ativos fora da fila."
        )
    conclusion_text = ""
    return quick_definition, how_text, why_text, what_text, conclusion_text


def build_final_model_block_text(
    model_frontier: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    confusion_df: pd.DataFrame,
) -> tuple[str, str, str, str, str]:
    if model_frontier.empty:
        return (
            "Modelo final é o par definição + trilha + família de modelo que ficou selecionado para uso.",
            "O modelo final não ficou materializado nesta execução.",
            "Sem modelo final, não existe leitura operacional consolidada.",
            "O bloco do modelo final não foi materializado.",
            "Sem modelo final, não existe decisão publicável.",
        )
    row = model_frontier.iloc[0]
    problem_key = str(row.get("problem_key", ""))
    model_name = str(row.get("model_name", ""))
    threshold_row = _get_threshold_row(threshold_metrics, problem_key, "tercis")
    confusion_rows = _get_confusion_rows(confusion_df, problem_key, "tercis")
    tp = _get_confusion_value(confusion_rows, "nao_realiza", "nao_realiza")
    fp = _get_confusion_value(confusion_rows, "realiza", "nao_realiza")
    tn = _get_confusion_value(confusion_rows, "realiza", "realiza")
    fn = _get_confusion_value(confusion_rows, "nao_realiza", "realiza")
    quick_definition = f"O modelo final é <b>{format_model_name(model_name)}</b> em <b>{format_problem_short_label(problem_key)}</b>."
    how_text = (
        "A matriz cruza duas coisas: o que o modelo marcou como risco e o que de fato aconteceu depois."
    )
    why_text = "Ela mostra quantos casos de risco entraram na fila, quantos ficaram de fora e quantos alarmes falsos apareceram."
    total_rows = tp + fp + tn + fn
    what_text = (
        f"A matriz usa só o teste futuro concatenado da faixa alta; por isso o <b>N = {total_rows}</b> não é a base inteira. "
        f"Nesse recorte, <b>{tp + fn}</b> professores ficaram inativos e <b>{fp + tn}</b> continuaram ativos. "
        f"O modelo marcou risco para <b>{tp}</b> casos corretos, gerou <b>{fp}</b> alarmes falsos, deixou <b>{fn}</b> casos de risco fora da fila e preservou <b>{tn}</b> casos ativos fora da fila."
    )
    conclusion_text = ""
    return quick_definition, how_text, why_text, what_text, conclusion_text


def build_score_block_text(
    score_deciles: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    band_summary: pd.DataFrame,
) -> tuple[str, str, str, str, str]:
    quick_definition = "<b>Score</b> é a chance de continuar ativo; <b>risk_score</b> é a chance complementar de ficar inativo."
    how_text = (
        "Cada ponto representa 10% da base, ordenada do menor para o maior score. A linha azul mostra o <b>score médio previsto</b>, isto é, a média das probabilidades publicadas pelo modelo naquele grupo. A linha verde mostra a <b>taxa observada</b>, isto é, a fração de professores daquele mesmo grupo que realmente continuou ativa depois."
    )
    why_text = "Comparar as duas linhas é útil porque mostra, grupo por grupo, se o score está acompanhando o que de fato aconteceu. Se o score funciona bem, grupos com score mais alto precisam terminar com taxa observada mais alta."
    if score_deciles.empty:
        return quick_definition, how_text, why_text, "O score por decil não ficou materializado nesta execução.", "Sem score materializado, não há fila de priorização."
    ordered = score_deciles.sort_values("score_decile", kind="mergesort").copy()
    low = ordered.iloc[0]
    high = ordered.iloc[-1]
    top10 = _get_threshold_row(threshold_metrics, str(low.get("problem_key", "")), "top_10_percent")
    tercis = _get_threshold_row(threshold_metrics, str(low.get("problem_key", "")), "tercis")
    ge70 = _get_threshold_row(threshold_metrics, str(low.get("problem_key", "")), "score_ge_0_70")
    band_rows = band_summary.copy()
    top10_share = tercis_high_share = ge70_high_share = None
    if not band_rows.empty:
        key = str(low.get("problem_key", ""))
        model_name = str(low.get("model_name", ""))
        match = band_rows[
            (band_rows["problem_key"].astype(str) == key)
            & (band_rows["model_name"].astype(str) == model_name)
        ].copy()
        top10_match = match[(match["policy_name"].astype(str) == "top_10_percent") & (match["band_name"].astype(str) == "alto")]
        tercis_match = match[(match["policy_name"].astype(str) == "tercis") & (match["band_name"].astype(str) == "alto")]
        ge70_match = match[(match["policy_name"].astype(str) == "score_ge_0_70") & (match["band_name"].astype(str) == "alto")]
        if not top10_match.empty:
            top10_share = format_percent(top10_match.iloc[0].get("share"))
        if not tercis_match.empty:
            tercis_high_share = format_percent(tercis_match.iloc[0].get("share"))
        if not ge70_match.empty:
            ge70_high_share = format_percent(ge70_match.iloc[0].get("share"))
    what_text = (
        f"Nos grupos do começo, quase ninguém volta: no decil mais baixo, o score médio ficou em <b>{format_percent(low.get('mean_score'))}</b> "
        f"e a taxa observada em <b>{format_percent(low.get('realized_rate'))}</b>. No fim da fila, isso sobe para <b>{format_percent(high.get('mean_score'))}</b> "
        f"e <b>{format_percent(high.get('realized_rate'))}</b>. No meio do gráfico há oscilações pequenas; isso é normal. O ponto principal é que a curva observada cresce quando o score entra nas faixas mais altas. Em operação, o <code>top 10%</code> do risco monta uma fila pequena "
        f"(<b>{top10_share or ''}</b> da base) e muito precisa; os <code>tercis</code> montam uma fila maior "
        f"(<b>{tercis_high_share or ''}</b> da base), ainda com precisão de <b>{format_percent(tercis.get('precision')) if tercis is not None else ''}</b>."
    )
    conclusion_text = "O score ordena a base; cutoff e faixas só decidem quantos casos entram na fila."
    return quick_definition, how_text, why_text, what_text, conclusion_text


def build_driver_block_text(
    feature_importance: pd.DataFrame,
    definition_b_feature_block_gain_summary: pd.DataFrame,
) -> tuple[str, str, str, str, str]:
    quick_definition = "<b>Driver</b>, aqui, não significa causa. Significa o sinal que mais muda a previsão quando é embaralhado no teste."
    how_text = (
        "A leitura principal usa <b>importância por permutação</b>: um sinal é embaralhado no conjunto de teste e medimos o quanto a previsão piora."
    )
    why_text = "Isso ajuda a explicar o score sem tratar o modelo como caixa-preta."
    pieces: list[str] = []
    if not feature_importance.empty:
        grouped = (
            feature_importance.groupby(["problem_key", "model_name", "feature_name"], as_index=False)
            .agg(importance_mean=("importance_mean", "mean"))
        )
        grouped["importance_abs"] = grouped["importance_mean"].abs()
        grouped = grouped.sort_values(["importance_abs", "problem_key", "model_name"], ascending=[False, True, True], kind="mergesort")
        primary = grouped[grouped["problem_key"].astype(str).str.startswith("definition_a::")].copy()
        if primary.empty:
            primary = grouped.copy()
        if not primary.empty:
            problem_key = str(primary.iloc[0]["problem_key"])
            model_name = str(primary.iloc[0]["model_name"])
            top_features = [
                format_feature_name(name)
                for name in primary[
                    (primary["problem_key"].astype(str) == problem_key)
                    & (primary["model_name"].astype(str) == model_name)
                ].head(4)["feature_name"].tolist()
            ]
            if top_features:
                pieces.append(
                    f"No modelo principal <b>{format_problem_short_label(problem_key)}</b> com <b>{format_model_name(model_name)}</b>, os sinais mais fortes foram "
                    f"<b>{'</b>, <b>'.join(top_features)}</b>."
                )
                pieces.append(
                    "<b>Sessões nos 7 primeiros dias</b> contam quantas vezes o professor entrou. "
                    "<b>Dias ativos nos 7 primeiros dias</b> contam em quantos dias distintos houve uso. "
                    "<b>Minutos de sessão nos 7 primeiros dias</b> somam o tempo total de uso. "
                    "Os três parecem próximos, mas medem frequência, cadência e intensidade."
                )
    what_text = " ".join(pieces) if pieces else "A importância dos sinais não ficou materializada nesta execução."
    conclusion_text = ""
    return quick_definition, how_text, why_text, what_text, conclusion_text


def build_cluster_block_text(cluster_summary: pd.DataFrame, cluster_profile: pd.DataFrame) -> tuple[str, str]:
    if cluster_summary.empty:
        return (
            "Cluster é uma camada descritiva de grupos parecidos de score.",
            "A camada de cluster não ficou materializada nesta execução.",
        )
    show = cluster_summary.sort_values("mean_risk_score", ascending=False, kind="mergesort").copy()
    high = show.iloc[0]
    low = show.iloc[-1]
    description = "Cluster é uma leitura descritiva de perfis de professores depois que o score já existe."
    detail = ""
    if not cluster_profile.empty:
        profile = cluster_profile.copy()
        profile = profile[profile["feature_name"].astype(str).isin(["teacher_active_months_total", "avg_activity_events_active_month", "avg_active_days_active_month", "avg_strict_downloads_active_month", "avg_content_views_active_month"])].copy()
        if not profile.empty:
            pivot = profile.pivot(index="cluster_name", columns="feature_name", values="feature_mean")
            if str(high.get("cluster_name")) in pivot.index and str(low.get("cluster_name")) in pivot.index:
                high_profile = pivot.loc[str(high.get("cluster_name"))]
                low_profile = pivot.loc[str(low.get("cluster_name"))]
                detail = (
                    f" O grupo menos arriscado tem mais histórico e intensidade: {format_number(low_profile.get('teacher_active_months_total'), 1)} meses ativos totais, "
                    f"{format_number(low_profile.get('avg_activity_events_active_month'), 1)} eventos por mês ativo, {format_number(low_profile.get('avg_active_days_active_month'), 1)} dias ativos por mês "
                    f"e {format_number(low_profile.get('avg_content_views_active_month'), 1)} views por mês, contra "
                    f"{format_number(high_profile.get('teacher_active_months_total'), 1)}, {format_number(high_profile.get('avg_activity_events_active_month'), 1)}, "
                    f"{format_number(high_profile.get('avg_active_days_active_month'), 1)} e {format_number(high_profile.get('avg_content_views_active_month'), 1)} no grupo mais arriscado."
                )
    finding = (
        f"O grupo mais arriscado concentrou <b>{format_percent(high.get('share'))}</b> da base e ficou com <b>{format_percent(high.get('realized_inactivity_rate'))}</b> de não realização; "
        f"o menos arriscado ficou em <b>{format_percent(low.get('realized_inactivity_rate'))}</b>.{detail}"
    )
    return description, finding


def build_heavy_user_block_text(heavy_user_summary: pd.DataFrame, heavy_user_profile: pd.DataFrame) -> tuple[str, str, str, str, str]:
    if heavy_user_summary.empty:
        return (
            "Heavy-user é um marcador retrospectivo de uso futuro intenso.",
            "A camada de heavy-user não ficou materializada nesta execução.",
            "Sem isso, não dá para separar quem parecia risco, mas ainda usou bastante, de quem de fato abandonou.",
            "A camada de heavy-user não ficou materializada nesta execução.",
            "Sem essa camada, a leitura retrospectiva de intensidade fica incompleta.",
        )
    show = heavy_user_summary[heavy_user_summary["policy_name"].astype(str) == "heavy_top_10_percent"].copy()
    if show.empty:
        show = heavy_user_summary.copy()
    heavy = show[show["heavy_user_flag"] == 1].head(1)
    non_heavy = show[show["heavy_user_flag"] == 0].head(1)
    quick_definition = "Heavy-user é uma leitura retrospectiva de quem ainda usou bastante depois de parecer risco."
    how_text = (
        "Depois que o score já ficou pronto, o pipeline olha só o comportamento futuro realizado e separa o grupo com uso mais intenso. Essa separação usa semanas ativas, sessões, minutos, dias ativos e eventos futuros, sempre medidos depois do momento da previsão."
    )
    why_text = (
        "Isso ajuda porque nem todo caso que parecia risco termina do mesmo jeito. Uma parte realmente some; outra parte continua usando bastante. O heavy-user marca esse segundo caso ao comparar o grupo de intensidade futura mais alta com o restante da base."
    )
    if heavy.empty or non_heavy.empty:
        return quick_definition, how_text, why_text, "A camada de heavy-user ficou incompleta nesta execução.", "Use essa camada só como leitura retrospectiva."
    heavy_row = heavy.iloc[0]
    non_heavy_row = non_heavy.iloc[0]
    profile_text = ""
    if not heavy_user_profile.empty:
        profile = heavy_user_profile[heavy_user_profile["policy_name"].astype(str) == "heavy_top_10_percent"].copy()
        if not profile.empty:
            pivot = profile.pivot(index="heavy_user_flag", columns="metric_name", values="metric_mean")
            if 1 in pivot.index and 0 in pivot.index:
                heavy_profile = pivot.loc[1]
                base_profile = pivot.loc[0]
                profile_text = (
                    f" Entre os heavy-users, a média foi de {format_number(heavy_profile.get('future_sessions'), 1)} sessões futuras, "
                    f"{format_number(heavy_profile.get('future_session_minutes'), 1)} minutos e {format_number(heavy_profile.get('future_active_days'), 1)} dias ativos, "
                    f"contra {format_number(base_profile.get('future_sessions'), 1)}, {format_number(base_profile.get('future_session_minutes'), 1)} e {format_number(base_profile.get('future_active_days'), 1)} no restante."
                )
    what_text = (
        f"No grupo de uso forte, a taxa observada de inatividade caiu para <b>{format_percent(heavy_row.get('realized_inactivity_rate'))}</b>; "
        f"no restante da base, ficou em <b>{format_percent(non_heavy_row.get('realized_inactivity_rate'))}</b>.{profile_text}"
    )
    conclusion_text = "Use essa camada como bandeira retrospectiva na análise salva, não como regra do score."
    return quick_definition, how_text, why_text, what_text, conclusion_text


def build_navigation_block_text(navigation_sequences: pd.DataFrame) -> tuple[str, str]:
    if navigation_sequences.empty:
        return (
            "Navegação resume os caminhos iniciais mais frequentes.",
            "A camada de navegação não ficou materializada nesta execução.",
        )
    show = navigation_sequences.copy()
    inactive = show[show["label_value"] == 0].sort_values("teachers", ascending=False, kind="mergesort").head(3)
    active = show[show["label_value"] == 1].sort_values("teachers", ascending=False, kind="mergesort").head(3)
    description = "Navegação não substitui o score. Ela ajuda a ler os caminhos mais comuns do começo da jornada."
    inactive_text = ", ".join(format_navigation_sequence(value) for value in inactive["step_sequence_first5"].tolist())
    active_text = ", ".join(format_navigation_sequence(value) for value in active["step_sequence_first5"].tolist())
    finding = (
        f"Entre quem não realizou a atividade futura, os caminhos mais frequentes foram <b>{inactive_text}</b>. "
        f"Entre quem realizou, apareceram mais <b>{active_text}</b>. O começo da jornada é parecido nos dois grupos, mas a mistura entre download, view e ausência de ação ajuda a qualificar o risco."
    )
    return description, finding


def build_beginner_guide_section(
    *,
    build_dir: Path,
    summary: dict[str, Any],
    track_registry: pd.DataFrame,
    arbitrariness: pd.DataFrame,
    feature_registry: pd.DataFrame,
    candidate_metric_registry: pd.DataFrame,
    definition_selection: pd.DataFrame,
    definition_frontier: pd.DataFrame,
    scoring_scenarios: pd.DataFrame,
    model_frontier: pd.DataFrame,
    cv_metric_folds: pd.DataFrame,
    cv_threshold_summary: pd.DataFrame,
    cv_confusion_summary: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    confusion_df: pd.DataFrame,
    band_summary: pd.DataFrame,
    predictions: pd.DataFrame,
    feature_importance: pd.DataFrame,
    definition_b_feature_block_gain_summary: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    cluster_profile: pd.DataFrame,
    heavy_user_summary: pd.DataFrame,
    heavy_user_profile: pd.DataFrame,
    serving_manifest: dict[str, Any],
) -> str:
    primary_problem_key, primary_model_name = _resolve_primary_problem_model(model_frontier, serving_manifest)
    primary_model_frontier = model_frontier[
        (model_frontier["problem_key"].astype(str) == primary_problem_key)
        & (model_frontier["model_name"].astype(str) == primary_model_name)
    ].copy() if not model_frontier.empty else pd.DataFrame()
    primary_cv_metric_folds = cv_metric_folds[
        (cv_metric_folds["problem_key"].astype(str) == primary_problem_key)
        & (cv_metric_folds["model_name"].astype(str) == primary_model_name)
    ].copy() if not cv_metric_folds.empty else pd.DataFrame()
    primary_cv_threshold_summary = cv_threshold_summary[
        (cv_threshold_summary["problem_key"].astype(str) == primary_problem_key)
        & (cv_threshold_summary["model_name"].astype(str) == primary_model_name)
    ].copy() if not cv_threshold_summary.empty else pd.DataFrame()
    primary_threshold_metrics = threshold_metrics[
        (threshold_metrics["problem_key"].astype(str) == primary_problem_key)
        & (threshold_metrics["model_name"].astype(str) == primary_model_name)
    ].copy() if not threshold_metrics.empty else pd.DataFrame()
    primary_confusion_df = confusion_df[
        (confusion_df["problem_key"].astype(str) == primary_problem_key)
        & (confusion_df["model_name"].astype(str) == primary_model_name)
    ].copy() if not confusion_df.empty else pd.DataFrame()
    primary_predictions = predictions[
        (predictions["problem_key"].astype(str) == primary_problem_key)
        & (predictions["model_name"].astype(str) == primary_model_name)
    ].copy() if not predictions.empty else pd.DataFrame()
    primary_feature_importance = feature_importance[
        (feature_importance["problem_key"].astype(str) == primary_problem_key)
        & (feature_importance["model_name"].astype(str) == primary_model_name)
    ].copy() if not feature_importance.empty else pd.DataFrame()
    primary_band_summary = band_summary[
        (band_summary["problem_key"].astype(str) == primary_problem_key)
        & (band_summary["model_name"].astype(str) == primary_model_name)
    ].copy() if not band_summary.empty else pd.DataFrame()
    primary_heavy_user_summary = heavy_user_summary[
        (heavy_user_summary["problem_key"].astype(str) == primary_problem_key)
        & (heavy_user_summary["model_name"].astype(str) == primary_model_name)
    ].copy() if not heavy_user_summary.empty else pd.DataFrame()

    assumption_points = build_assumption_event_points(track_registry, arbitrariness)
    score_deciles = build_score_decile_summary(primary_predictions)
    comparison_problem_keys: list[str] = []
    for prefix in ["definition_a", "definition_b"]:
        best = _best_model_row_for_definition(model_frontier, prefix, ranking_priority=False)
        if best is not None:
            comparison_problem_keys.append(str(best.get("problem_key", "")))
    model_selection_df = model_frontier.copy()
    if comparison_problem_keys:
        model_selection_df = model_selection_df[
            model_selection_df["problem_key"].astype(str).isin(comparison_problem_keys)
        ].copy()
    if not model_selection_df.empty:
        model_selection_df["selected_flag"] = 0
        if serving_manifest:
            rows = serving_manifest.get("reference_scope_rows", [])
            if rows:
                primary_problem_key = str(rows[0].get("problem_key", ""))
                primary_model_name = str(rows[0].get("model_name", ""))
                mask = (
                    model_selection_df["problem_key"].astype(str).eq(primary_problem_key)
                    & model_selection_df["model_name"].astype(str).eq(primary_model_name)
                )
                model_selection_df.loc[mask, "selected_flag"] = 1

    methodology_protocol_block = build_methodology_protocol_block_text(
        build_dir=build_dir,
        summary=summary,
        future_metrics=future_metrics,
        definition_selection=definition_selection,
        candidate_metric_registry=candidate_metric_registry,
    )
    methodology_gaps_block = build_methodology_gaps_block_text(
        build_dir=build_dir,
        definition_selection=definition_selection,
        serving_manifest=serving_manifest,
    )
    assumptions_block = build_assumptions_block_text(track_registry, arbitrariness, feature_registry)
    definition_block = build_activity_definition_block_text(definition_frontier)
    scenarios_block = build_scenarios_block_text(scoring_scenarios)
    model_comparison_block = build_model_comparison_block_text(model_selection_df, definition_frontier, serving_manifest)
    trust_block = build_trust_block_text(primary_model_frontier, primary_cv_metric_folds, primary_cv_threshold_summary, cv_confusion_summary)
    final_model_block = build_final_model_block_text(primary_model_frontier, primary_threshold_metrics, primary_confusion_df)
    score_block = build_score_block_text(score_deciles, primary_threshold_metrics, primary_band_summary)
    driver_block = build_driver_block_text(primary_feature_importance, definition_b_feature_block_gain_summary)
    heavy_user_block = build_heavy_user_block_text(primary_heavy_user_summary, heavy_user_profile)

    return f"""
    <section>
      <div class="guide-grid">
        {render_checklist_block(
            title="0. Qual é o protocolo metodológico declarado do estudo?",
            quick_definition=methodology_protocol_block[0],
            how_text=methodology_protocol_block[1],
            why_text=methodology_protocol_block[2],
            sections=methodology_protocol_block[3],
            conclusion_text=methodology_protocol_block[4],
            lineage_items=[
                ("Base usada", "<code>build/tables</code> + política metodológica registrada no repositório"),
                ("Escopo", "Guardrails do alvo, da modelagem, do serving e da trilha de auditoria."),
            ],
        )}
        {render_checklist_block(
            title="0B. O que o material atual ainda não resolve sozinho?",
            quick_definition=methodology_gaps_block[0],
            how_text=methodology_gaps_block[1],
            why_text=methodology_gaps_block[2],
            sections=methodology_gaps_block[3],
            conclusion_text=methodology_gaps_block[4],
            lineage_items=[
                ("Base usada", "<code>build/tables</code>, <code>build/serving/serving_manifest.json</code> e estado atual do código"),
                ("Leitura correta", "Este bloco separa fato materializado, política corrigida e gap ainda pendente."),
            ],
        )}
        {render_teaching_block(
            title="1. O que o modelo pode ver antes de prever?",
            quick_definition=assumptions_block[0],
            how_text=assumptions_block[1],
            why_text=assumptions_block[2],
            chart_html=render_assumptions_timeline(assumption_points, feature_registry),
            detail_html=render_strict_context_note(feature_registry),
            what_text=assumptions_block[3],
            conclusion_text=assumptions_block[4],
            lineage_items=[
                ("Tabelas usadas", "<code>governance_track_registry_v1</code>, <code>governance_arbitrariness_registry_v1</code> e <code>governance_feature_registry_v1</code>"),
                ("Unidade do gráfico", "Cada linha mostra quando cada trilha já tem dado suficiente para prever e quando o resultado passa a poder ser medido."),
            ],
        )}
        {render_teaching_block(
            title="2. O que é atividade futura aqui?",
            quick_definition=definition_block[0],
            how_text=definition_block[1],
            why_text=definition_block[2],
            chart_html=render_plotly(definition_frontier, "definition"),
            what_text=definition_block[3],
            conclusion_text=definition_block[4],
            lineage_items=[
                ("Tabelas usadas", "<code>core_definition_frontier_v1</code>, <code>core_definition_external_validation_v1</code> e <code>governance_label_registry_v1</code>"),
                ("Unidade do gráfico", "Cada barra é o gap observado entre grupos ativos e não ativos nos validadores externos pós-label."),
            ],
        )}
        {render_teaching_block(
            title="3. Quais definições, trilhas e combinações foram testadas?",
            quick_definition=scenarios_block[0],
            how_text=scenarios_block[1],
            why_text=scenarios_block[2],
            chart_html=render_plotly(scoring_scenarios, "scenarios_tested"),
            what_text=scenarios_block[3],
            conclusion_text=scenarios_block[4],
            lineage_items=[
                ("Tabelas usadas", "<code>core_scoring_scenarios_v1</code> e <code>governance_feature_registry_v1</code>"),
                ("Unidade do gráfico", "Cada barra mostra quantas variáveis elegíveis entraram em cada definição + trilha."),
            ],
        )}
        {render_teaching_block(
            title="4. Quais modelos foram comparados e como o melhor foi escolhido?",
            quick_definition=model_comparison_block[0],
            how_text=model_comparison_block[1],
            why_text=model_comparison_block[2],
            chart_html=render_model_selection_board(model_selection_df),
            what_text=model_comparison_block[3],
            conclusion_text=model_comparison_block[4],
            lineage_items=[
                ("Tabelas usadas", "<code>core_model_frontier_v1</code> e <code>build/serving/serving_manifest.json</code>"),
                ("Unidade da matriz", "Cada linha é uma combinação definição + modelo; a posição mostra o ranking dentro de cada combinação publicada."),
            ],
        )}
        {render_teaching_block(
            title="5. Por que dá para confiar no modelo?",
            quick_definition=trust_block[0],
            how_text=trust_block[1],
            why_text=trust_block[2],
            chart_html=render_plotly(primary_cv_metric_folds, "cv_metric_drift"),
            what_text=trust_block[3],
            conclusion_text=trust_block[4],
            lineage_items=[
                ("Tabelas usadas", "<code>core_cv_metric_folds_v1</code> e <code>post_model_cv_threshold_summary_v1</code>"),
                ("Unidade do gráfico", "Cada ponto é um mês futuro nunca visto no treino."),
            ],
        )}
        {render_teaching_block(
            title="6. Qual é o modelo final e onde ele acerta e erra?",
            quick_definition=final_model_block[0],
            how_text=final_model_block[1],
            why_text=final_model_block[2],
            chart_html=render_confusion_matrix_panel(primary_confusion_df),
            what_text=final_model_block[3],
            conclusion_text=final_model_block[4],
            lineage_items=[
                ("Tabelas usadas", "<code>core_model_frontier_v1</code>, <code>post_model_threshold_metrics_v1</code> e <code>post_model_confusion_matrix_v1</code>"),
                ("Unidade da matriz", "A matriz usa só o teste futuro concatenado do modelo final com a política <code>tercis</code>."),
            ],
        )}
        {render_teaching_block(
            title="7. O que é o score e como usar cutoff e faixas?",
            quick_definition=score_block[0],
            how_text=score_block[1],
            why_text=score_block[2],
            chart_html=render_plotly(score_deciles, "score_deciles"),
            what_text=score_block[3],
            conclusion_text=score_block[4],
            lineage_items=[
                ("Tabelas usadas", "<code>core_model_predictions_v1</code>, <code>post_model_threshold_metrics_v1</code> e <code>post_model_band_summary_v1</code>"),
                ("Unidade do gráfico", "A curva mostra score previsto e taxa observada por decil; cutoff e faixas ficam resumidos no texto do bloco."),
            ],
        )}
        {render_teaching_block(
            title="8. Quais sinais mais puxam a previsão de atividade?",
            quick_definition=driver_block[0],
            how_text=driver_block[1],
            why_text=driver_block[2],
            chart_html=render_plotly(primary_feature_importance, "feature_importance"),
            what_text=driver_block[3],
            conclusion_text=driver_block[4],
            lineage_items=[
                ("Tabelas usadas", "<code>post_model_feature_importance_v1</code> e <code>core_definition_b_feature_block_gain_summary_v1</code>"),
                ("Unidade do gráfico", "As barras mostram o impacto médio de embaralhar um sinal no modelo final."),
            ],
        )}
      </div>
    </section>
    <section>
      <h2>Camadas complementares</h2>
      <div class="guide-grid">
        {render_teaching_block(
            title="Heavy-user",
            quick_definition=heavy_user_block[0],
            how_text=heavy_user_block[1],
            why_text=heavy_user_block[2],
            chart_html=render_heavy_user_panel(primary_heavy_user_summary, heavy_user_profile),
            what_text=heavy_user_block[3],
            conclusion_text=heavy_user_block[4],
            lineage_items=[
                ("Tabelas usadas", "<code>post_model_heavy_user_summary_v1</code> e <code>post_model_heavy_user_profile_v1</code>"),
                ("Unidade do gráfico", "Cada barra mostra quantas vezes o grupo heavy-user difere do restante da base em inatividade e intensidade futura."),
            ],
        )}
      </div>
    </section>
    <section>
      {render_product_block()}
    </section>
    """


def main() -> None:
    args = parse_args()
    build_dir = (args.build_dir or (PROJECT_DIR / "build")).resolve()
    summary = read_summary(build_dir)
    serving_manifest = read_serving_manifest(build_dir)
    track_registry = read_table(build_dir, "governance_track_registry_v1")
    arbitrariness = read_table(build_dir, "governance_arbitrariness_registry_v1")
    policy_registry = read_table(build_dir, "governance_policy_registry_v1")
    candidate_metric_registry = read_table(build_dir, "governance_definition_candidate_metric_registry_v1")
    feature_registry = read_table(build_dir, "governance_feature_registry_v1")
    label_registry = read_table(build_dir, "governance_label_registry_v1")
    future_metrics = read_table(build_dir, "mart_future_metrics_v1")
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

    definition_frontier_all = definition_frontier.copy()
    scoring_scenarios_all = scoring_scenarios.copy()
    model_frontier_all = model_frontier.copy()
    predictions_all = predictions.copy()
    cv_metric_folds_all = cv_metric_folds.copy()
    cv_threshold_summary_all = cv_threshold_summary.copy()
    threshold_metrics_all = threshold_metrics.copy()
    confusion_df_all = confusion_df.copy()
    band_summary_all = band_summary.copy()
    feature_importance_all = feature_importance.copy()
    cluster_summary_all = cluster_summary.copy()
    heavy_user_summary_all = heavy_user_summary.copy()

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
    primary_scope = build_primary_scope(reference_scope, serving_manifest)
    presentation_definition_frontier = filter_to_reference_definitions(definition_frontier_all, presentation_scope)
    presentation_scoring_scenarios = filter_to_reference_definitions(scoring_scenarios_all, presentation_scope)
    comparison_model_frontier = filter_to_problem_keys(model_frontier_all, presentation_scope)
    primary_model_frontier = filter_to_reference_scope(model_frontier_all, primary_scope)
    primary_predictions = filter_to_reference_scope(predictions_all, primary_scope)
    primary_cv_metric_folds = filter_to_reference_scope(cv_metric_folds_all, primary_scope)
    primary_cv_threshold_summary = filter_to_reference_scope(cv_threshold_summary_all, primary_scope)
    primary_cv_confusion_summary = filter_to_reference_scope(cv_confusion_summary, primary_scope)
    primary_threshold_metrics = filter_to_reference_scope(threshold_metrics_all, primary_scope)
    primary_confusion_df = filter_to_reference_scope(confusion_df_all, primary_scope)
    primary_band_summary = filter_to_reference_scope(band_summary_all, primary_scope)
    primary_feature_importance = filter_to_reference_scope(feature_importance_all, primary_scope)
    primary_cluster_summary = filter_to_reference_scope(cluster_summary_all, primary_scope)
    primary_heavy_user_summary = filter_to_reference_scope(heavy_user_summary_all, primary_scope)
    presentation_definition_b_feature_block_gain_summary = filter_definition_b_feature_block_gain_for_report(
        definition_b_feature_block_gain_summary_full,
        presentation_scope,
    )

    report_title = "Previsão de atividade futura"
    intro_rows = int(pd.to_numeric(presentation_scoring_scenarios["rows"], errors="coerce").dropna().max() or 0) if not presentation_scoring_scenarios.empty else 0
    intro_months = int(pd.to_numeric(presentation_scoring_scenarios["months"], errors="coerce").dropna().max() or 0) if not presentation_scoring_scenarios.empty else 0
    eligible_population = future_metrics.copy()
    if not eligible_population.empty and "full_followup_observed_flag" in eligible_population.columns:
        eligible_population = eligible_population[eligible_population["full_followup_observed_flag"] == 1].copy()
    if (
        not eligible_population.empty
        and str(summary.get("official_population_filter", "")) == "same_month_entry_only"
        and "same_month_entry_flag" in eligible_population.columns
    ):
        eligible_population = eligible_population[eligible_population["same_month_entry_flag"] == 1].copy()
    eligible_month_start = "n/d"
    eligible_month_end = "n/d"
    if not eligible_population.empty and "first_month" in eligible_population.columns:
        eligible_months = pd.to_datetime(eligible_population["first_month"], errors="coerce").dropna()
        if not eligible_months.empty:
            eligible_month_start = eligible_months.min().strftime("%Y-%m")
            eligible_month_end = eligible_months.max().strftime("%Y-%m")
    population_note = ""
    if str(summary.get("official_population_filter", "")) == "same_month_entry_only":
        population_note = " O estudo principal foi restringido a professores com primeiro uso observado no mesmo mês do cadastro."
    beginner_guide_section = build_beginner_guide_section(
        build_dir=build_dir,
        summary=summary,
        track_registry=track_registry,
        arbitrariness=arbitrariness,
        feature_registry=feature_registry,
        candidate_metric_registry=candidate_metric_registry,
        definition_selection=definition_selection,
        definition_frontier=presentation_definition_frontier,
        scoring_scenarios=presentation_scoring_scenarios,
        model_frontier=comparison_model_frontier,
        cv_metric_folds=primary_cv_metric_folds,
        cv_threshold_summary=primary_cv_threshold_summary,
        cv_confusion_summary=primary_cv_confusion_summary,
        threshold_metrics=primary_threshold_metrics,
        confusion_df=primary_confusion_df,
        band_summary=primary_band_summary,
        predictions=primary_predictions,
        feature_importance=primary_feature_importance,
        definition_b_feature_block_gain_summary=presentation_definition_b_feature_block_gain_summary,
        cluster_summary=primary_cluster_summary,
        cluster_profile=cluster_profile,
        heavy_user_summary=primary_heavy_user_summary,
        heavy_user_profile=heavy_user_profile,
        serving_manifest=serving_manifest,
    )

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
    .chart-card {{ background: white; border: 1px solid #D9E2EC; border-radius: 14px; padding: 18px 20px; margin: 18px 0 34px 0; }}
    .note {{ background: #E6FFFA; border: 1px solid #81E6D9; border-radius: 10px; padding: 12px 14px; margin: 18px 0 16px 0; font-size: 15px; line-height: 1.6; color: #234E52; }}
    .lineage {{ background: #F8FAFC; border: 1px solid #D9E2EC; border-radius: 10px; padding: 12px 14px; margin-top: 18px; }}
    .lineage p {{ margin: 8px 0; font-size: 13.5px; line-height: 1.55; color: #334E68; }}
    .embedded-chart-wrap {{ margin: 18px 0 22px 0; }}
    .definition-note {{ background: #F8FAFC; border: 1px solid #D9E2EC; border-radius: 12px; padding: 14px 16px; margin: 0 0 18px 0; }}
    .definition-note strong {{ display: block; font-size: 15px; color: #102A43; margin-bottom: 6px; }}
    .definition-note p {{ margin: 0 0 10px 0; font-size: 14px; color: #486581; line-height: 1.5; }}
    .protocol-wrap {{ display: grid; gap: 14px; margin: 18px 0 20px 0; }}
    .protocol-section {{ background: #F8FAFC; border: 1px solid #D9E2EC; border-radius: 12px; padding: 14px 16px; }}
    .protocol-section h4 {{ margin: 0 0 10px 0; }}
    .protocol-list {{ margin: 0; padding-left: 20px; color: #334E68; }}
    .protocol-list li {{ margin: 0 0 8px 0; line-height: 1.55; }}
    .definition-note ul {{ margin: 0; padding-left: 18px; display: grid; gap: 6px; }}
    .definition-note li {{ color: #334E68; font-size: 14px; line-height: 1.5; }}
    .definition-note-compact {{ padding: 8px 10px; margin: 8px 0 14px 0; }}
    .definition-note-compact strong {{ font-size: 12.5px; margin-bottom: 3px; }}
    .definition-note-compact p {{ margin: 0; font-size: 11.75px; line-height: 1.45; }}
    .definition-chip-list {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .definition-chip-list span {{ display: inline-flex; align-items: center; gap: 4px; background: white; border: 1px solid #D9E2EC; border-radius: 999px; padding: 4px 8px; font-size: 12px; line-height: 1.35; color: #486581; }}
    .chart-card .plotly-graph-div {{ margin-top: 10px; }}
    .guide-grid {{ display: grid; grid-template-columns: 1fr; gap: 24px; align-items: start; margin: 20px 0 34px 0; }}
    .guide-card {{ background: white; border: 1px solid #D9E2EC; border-radius: 14px; padding: 18px 20px; }}
    .compact-card {{ padding-bottom: 14px; }}
    .guide-card h3 {{ margin-bottom: 12px; }}
    .lead-text {{ font-size: 17px; color: #243B53; }}
    .compact-grid {{ grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
    .button-link {{ display: inline-block; background: #0F766E; color: white; text-decoration: none; padding: 10px 14px; border-radius: 10px; font-weight: 600; }}
    .button-link:hover {{ background: #115E59; }}
    .secondary-link {{ background: #0F172A; margin-left: 8px; }}
    .secondary-link:hover {{ background: #111827; }}
    code {{ background: #F0F4F8; padding: 1px 4px; border-radius: 4px; }}
    .assumption-panel {{ background: #FFFFFF; border: 1px solid #D4E1EE; border-radius: 12px; padding: 18px 20px 22px 20px; }}
    .assumption-legend {{ display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 32px; color: #385170; font-size: 13px; }}
    .legend-dot {{ display: inline-block; width: 14px; height: 14px; border-radius: 999px; margin-right: 8px; vertical-align: -2px; }}
    .legend-start {{ background: #FFFFFF; border: 2px solid #36537A; box-sizing: border-box; }}
    .legend-predict {{ background: {COLOR_INFO}; }}
    .legend-result-shape {{ width: 0; height: 0; border-left: 7px solid transparent; border-right: 7px solid transparent; border-bottom: 12px solid {COLOR_POSITIVE}; border-radius: 0; margin-right: 8px; vertical-align: -2px; }}
    .assumption-header {{ display: grid; grid-template-columns: 160px repeat(4, minmax(0, 1fr)); gap: 14px; margin-bottom: 8px; }}
    .assumption-header-cell {{ text-align: center; font-size: 12px; line-height: 1.35; color: #36506B; font-weight: 700; }}
    .assumption-matrix {{ display: grid; gap: 14px; }}
    .assumption-matrix-row {{ display: grid; grid-template-columns: 160px 1fr; gap: 14px; align-items: center; }}
    .assumption-track {{ font-weight: 700; color: #102A43; font-size: 16px; }}
    .assumption-trackline {{ position: relative; display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); align-items: center; min-height: 38px; }}
    .assumption-rail-line {{ position: absolute; left: 12.5%; right: 12.5%; top: 50%; transform: translateY(-50%); height: 4px; background: #6C88B4; border-radius: 999px; }}
    .assumption-cell {{ position: relative; z-index: 1; display: flex; align-items: center; justify-content: center; min-height: 38px; }}
    .assumption-axis-title {{ text-align: center; color: #36506B; font-size: 13px; margin-top: 12px; }}
    .status-pill {{ display: inline-flex; align-items: center; padding: 3px 8px; border-radius: 999px; font-size: 12px; background: #E2E8F0; color: #334155; }}
    .selected-pill {{ background: #CCFBF1; color: #115E59; }}
    .winner-pill {{ background: #DBEAFE; color: #1D4ED8; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-bottom: 10px; }}
    .metric-mini {{ background: white; border: 1px solid #E2E8F0; border-radius: 10px; padding: 8px 10px; display: grid; gap: 3px; }}
    .metric-mini span {{ font-size: 12px; color: #64748B; }}
    .metric-mini strong {{ font-size: 15px; color: #102A43; }}
    .bar-label {{ font-size: 12px; color: #64748B; margin: 6px 0 4px 0; }}
    .metric-bar {{ background: #E2E8F0; border-radius: 999px; height: 10px; overflow: hidden; }}
    .metric-bar-fill {{ height: 100%; border-radius: 999px; }}
    .ap-fill {{ background: {COLOR_INFO}; }}
    .brier-fill {{ background: {COLOR_POSITIVE}; }}
    .comparison-sections {{ display: grid; gap: 16px; }}
    .comparison-section {{ border: 1px solid #D9E2EC; border-radius: 12px; overflow: hidden; background: #F8FAFC; }}
    .comparison-section-a {{ border-left: 4px solid {COLOR_INFO}; }}
    .comparison-section-b {{ border-left: 4px solid {COLOR_NEGATIVE}; }}
    .comparison-section-title {{ padding: 12px 14px; font-weight: 700; color: #102A43; background: white; border-bottom: 1px solid #E2E8F0; }}
    .comparison-table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    .comparison-table th {{ background: #102A43; color: white; padding: 10px 12px; text-align: center; font-size: 12px; }}
    .comparison-table td {{ padding: 12px 14px; border-top: 1px solid #E2E8F0; color: #243B53; text-align: center; }}
    .comparison-table td:nth-child(2) {{ text-align: left; }}
    .comparison-table-row-selected td {{ background: #ECFDF5; }}
    .rank-pill {{ display: inline-flex; align-items: center; justify-content: center; min-width: 36px; padding: 3px 8px; border-radius: 999px; background: #E0F2FE; color: #1D4ED8; font-weight: 700; font-size: 12px; }}
    .trust-kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 14px; }}
    .trust-card {{ border: 1px solid #D9E2EC; border-radius: 12px; padding: 12px 14px; background: #F8FAFC; display: grid; gap: 4px; }}
    .trust-card span {{ font-size: 12px; color: #64748B; }}
    .trust-card strong {{ font-size: 18px; color: #102A43; }}
    .trust-card small {{ font-size: 12px; color: #486581; line-height: 1.45; }}
    .fold-strip {{ display: grid; gap: 10px; }}
    .fold-line {{ display: grid; grid-template-columns: 84px 1fr 1fr; gap: 12px; border: 1px solid #D9E2EC; border-radius: 12px; padding: 10px 12px; background: #F8FAFC; font-size: 14px; color: #334E68; }}
    .confusion-grid {{ display: grid; grid-template-columns: 160px repeat(2, minmax(220px, 1fr)); gap: 12px; align-items: stretch; }}
    .confusion-head {{ background: #102A43; color: white; border-radius: 12px; padding: 14px; font-weight: 700; font-size: 14px; display: flex; align-items: center; justify-content: center; text-align: center; }}
    .empty-head {{ background: transparent; padding: 0; }}
    .side-head {{ background: #334155; }}
    .top-head-risk {{ background: #334155; }}
    .top-head-active {{ background: #334155; }}
    .side-head-risk {{ background: #334155; }}
    .side-head-active {{ background: #334155; }}
    .confusion-cell {{ border-radius: 12px; padding: 16px; display: grid; gap: 6px; border: 1px solid #D9E2EC; }}
    .conf-label {{ font-size: 12px; font-weight: 700; color: #0F172A; letter-spacing: 0.02em; text-transform: uppercase; }}
    .confusion-cell strong {{ font-size: 28px; color: #102A43; }}
    .confusion-cell small {{ font-size: 13px; line-height: 1.45; color: #486581; }}
    .tp-cell {{ background: #DBEAFE; border-color: {COLOR_INFO}; }}
    .fp-cell {{ background: #FEE2E2; border-color: {COLOR_ERROR}; }}
    .tn-cell {{ background: #DBEAFE; border-color: {COLOR_INFO}; }}
    .fn-cell {{ background: #FEE2E2; border-color: {COLOR_ERROR}; }}
    .policy-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-top: 16px; }}
    .policy-card {{ border: 1px solid #D9E2EC; border-radius: 12px; padding: 14px; background: #F8FAFC; }}
    .policy-card h4 {{ margin: 0 0 10px 0; }}
    .decile-panel {{ display: grid; gap: 10px; }}
    .decile-legend {{ display: flex; gap: 16px; flex-wrap: wrap; color: #486581; font-size: 13px; }}
    .legend-box {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -2px; }}
    .decile-row {{ display: grid; grid-template-columns: 48px 1fr 1fr; gap: 12px; align-items: center; }}
    .decile-label {{ font-weight: 700; color: #102A43; }}
    .decile-metric {{ display: grid; gap: 6px; font-size: 13px; color: #486581; }}
    .decile-track {{ background: #E2E8F0; border-radius: 999px; height: 12px; overflow: hidden; }}
    .decile-fill {{ height: 100%; border-radius: 999px; }}
    .decile-pred-fill {{ background: {COLOR_INFO}; }}
    .decile-real-fill {{ background: {COLOR_POSITIVE}; }}
    .feature-bar-list {{ display: grid; gap: 12px; }}
    .feature-bar-row {{ display: grid; grid-template-columns: minmax(180px, 1.2fr) 2fr 72px; gap: 12px; align-items: center; }}
    .feature-bar-label {{ font-size: 14px; color: #243B53; }}
    .feature-bar-track {{ background: #E2E8F0; border-radius: 999px; height: 12px; overflow: hidden; }}
    .feature-bar-fill {{ background: linear-gradient(90deg, {COLOR_INFO}, {COLOR_POSITIVE_LIGHT}); height: 100%; border-radius: 999px; }}
    .feature-bar-value {{ font-size: 13px; color: #102A43; font-weight: 700; text-align: right; }}
    .signal-chip-grid {{ display: grid; gap: 12px; margin-top: 16px; }}
    .signal-chip {{ border: 1px solid #D9E2EC; border-radius: 12px; padding: 12px 14px; background: #F8FAFC; display: grid; gap: 4px; }}
    .signal-chip strong {{ font-size: 14px; color: #102A43; }}
    .signal-chip span {{ font-size: 14px; line-height: 1.5; color: #486581; }}
    .profile-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }}
    .profile-card {{ border: 1px solid #D9E2EC; border-radius: 12px; padding: 14px; background: #F8FAFC; }}
    .profile-card h4 {{ margin: 0 0 10px 0; }}
    .profile-list {{ display: grid; gap: 6px; font-size: 14px; color: #486581; }}
    .product-list {{ display: grid; gap: 10px; margin-bottom: 18px; font-size: 15px; line-height: 1.5; color: #486581; }}
    @media (max-width: 980px) {{
      .assumption-row,
      .confusion-grid,
      .feature-bar-row,
      .decile-row,
      .fold-line {{ grid-template-columns: 1fr; }}
      .assumption-axis {{ margin-left: 0; }}
      .empty-head {{ display: none; }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>{report_title}</h1>

    <section>
      <h2>Introdução</h2>
      <div class="chart-card">
        <p class="section-text">Probabilidade de atividade futura a partir do começo da jornada.</p>
        <p class="section-text"><b>Base modelada</b> é a versão limpa em fatos e dimensões usada pelo ML. Nesta execução, a comparação principal usou <b>{intro_rows}</b> linhas distribuídas entre <b>{eligible_month_start}</b> e <b>{eligible_month_end}</b>. Esses <b>{intro_months}</b> meses são os meses em que a base já permite observar, do começo ao fim, a janela futura de 30 dias usada para medir o resultado.{population_note}</p>
        <p class="section-text"><b>Score</b> é a probabilidade de o professor entrar no grupo definido como ativo no período futuro deste relatório. <b>risk_score</b> é a probabilidade complementar de ficar fora desse grupo.</p>
      </div>
    </section>

    {beginner_guide_section}
  </div>
</body>
</html>
"""

    output_html = args.output_html or (build_dir / "reports" / "targeted_ml_report_v1.html")
    output_html.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
