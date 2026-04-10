from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import duckdb
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "dbt_lineage" / "models"


def _resolve_default_modelled_duckdb() -> Path:
    candidates = [
        PROJECT_ROOT / "build" / "modelled" / "duckdb" / "base_modelada_v2.duckdb",
        PROJECT_ROOT / "data" / "modelled" / "duckdb" / "base_modelada_v2.duckdb",
    ]
    return next((candidate.resolve() for candidate in candidates if candidate.exists()), candidates[0].resolve())


def _resolve_default_build_duckdb() -> Path:
    candidates = [
        PROJECT_ROOT / "build" / "duckdb" / "build.duckdb",
    ]
    return next((candidate.resolve() for candidate in candidates if candidate.exists()), candidates[0].resolve())


MODELED_DUCKDB = Path(os.getenv("TARGETED_ML_MODELLED_DUCKDB", str(_resolve_default_modelled_duckdb()))).resolve()
BUILD_DUCKDB = Path(os.getenv("TARGETED_ML_BUILD_DUCKDB", str(_resolve_default_build_duckdb()))).resolve()


RAW_SOURCE_DESCRIPTION = (
    "Camada documental das tabelas raw que originam a base modelada. "
    "Essas fontes existem para explicar a linhagem e o significado dos dados brutos, "
    "mesmo quando a transformação raw -> modeled acontece fora deste projeto dbt."
)
MODELED_SOURCE_DESCRIPTION = (
    "Base modelada oficial usada pelo pipeline de ML. Aqui entram fatos, dimensões e marts "
    "limpos, prontos para reconstruir a jornada inicial e medir atividade futura."
)
ML_OUTPUTS_SOURCE_DESCRIPTION = (
    "Saídas materializadas do pipeline Python de ML. O dbt documenta, descreve e conecta "
    "essas tabelas à base modelada oficial."
)


RAW_TABLES: dict[str, dict[str, Any]] = {
    "dim_teachers": {
        "description": "Tabela dimensão que armazena informações dos professores cadastrados na plataforma AprendiZAP.",
        "columns": {
            "unique_id": "Identificador único do professor.",
            "utm_origin": "Origem UTM do cadastro.",
            "tela_origem": "Tela de origem do cadastro.",
            "estado": "Estado ou UF do professor.",
            "total_alunos": "Total de alunos do professor.",
            "tipo_total_alunos": "Tipo de alunos diretos ou indiretos imputados pelo professor.",
            "alunos_diretos": "Quantidade de alunos diretos.",
            "alunos_indiretos": "Quantidade de alunos indiretos.",
            "login_google": "Indicador se usa login via Google.",
            "currentstage": "Etapa ou série atual em que leciona.",
            "currentsubject": "Disciplina atual em que leciona.",
            "selectedstages": "Etapas selecionadas pelo professor.",
            "selectedsubjectsem": "Disciplinas selecionadas para Ensino Médio.",
            "selectedsubjectsfundii": "Disciplinas selecionadas para Fundamental II.",
            "visualizou_metodologia_ativa": "Flag se visualizou metodologia ativa.",
            "data_entrada": "Data de entrada na plataforma.",
        },
    },
    "fct_teachers_contents_interactions": {
        "description": "Tabela de fatos que registra as interações dos professores com conteúdos da plataforma.",
        "columns": {
            "unique_id": "Identificador único do professor.",
            "user_type": "Tipo do usuário.",
            "user_agent_device_type": "Tipo de dispositivo usado.",
            "data_inicio": "Data e hora de início da interação.",
            "event_type": "Tipo do evento ou interação.",
            "content_type": "Tipo do conteúdo acessado.",
            "id_aula": "Identificador da aula.",
            "utm_source": "Fonte UTM da sessão.",
        },
    },
    "fct_teachers_entries": {
        "description": "Tabela de fatos que registra as entradas dos professores na plataforma.",
        "columns": {
            "unique_id": "Identificador único do professor.",
            "user_type": "Tipo do usuário.",
            "data_inicio": "Data e hora de início da sessão.",
            "data_fim": "Data e hora de fim da sessão.",
        },
    },
    "fct_mari_ia_eventos_isso_ajudou": {
        "description": "Tabela de fatos que registra interações com a Mari IA e avalia as respostas recebidas.",
        "columns": {
            "user_id": "Identificador do usuário associado ao evento.",
            "date": "Data da interação.",
            "turno": "Turno da interação.",
            "key": "Chave do evento ou atributo reportado.",
            "isso_ajudou": "Resposta textual sobre se a interação ajudou.",
            "isso_ajudou_num": "Resposta numérica sobre se a interação ajudou.",
        },
    },
    "stg_formation": {
        "description": "Tabela de staging que registra interações dos usuários com cursos de formação.",
        "columns": {
            "unique_id_aprendizap": "Identificador do usuário no AprendiZAP.",
            "itemid": "ID do item do curso.",
            "createdat": "Data de criação do registro.",
            "updatedat": "Data da última atualização.",
            "type": "Tipo do item do curso.",
            "completionstatus": "Status de conclusão.",
            "progress": "Progresso percentual.",
            "questionstatus": "Status das questões.",
            "coursemodulecount": "Quantidade de módulos do curso.",
            "moduleblockcount": "Quantidade de blocos do módulo.",
            "quizquestioncount": "Quantidade de questões do quiz.",
        },
    },
    "stg_mari_ia_conversation": {
        "description": "Tabela de staging que armazena conversas com a IA Mari em dados anteriores a 21/02/2025.",
        "columns": {
            "id_mari": "Identificador da conversa ou thread na Mari.",
            "createdat": "Data de criação da conversa.",
            "updatedat": "Data da última atualização.",
            "userlastmessage": "Última mensagem enviada pelo usuário.",
            "ailastmessage": "Última mensagem enviada pela IA.",
            "originsource": "Fonte de origem da conversa.",
            "userreaction": "Reação do usuário à resposta.",
            "unique_id_aprendizap": "Identificador reconciliado do usuário no AprendiZAP.",
        },
    },
    "stg_mari_ia_reports": {
        "description": "Tabela de staging com relatórios estruturados das interações com a IA Mari em dados posteriores a 20/02/2025.",
        "columns": {
            "id_mari": "Identificador da conversa ou thread na Mari.",
            "updatedat": "Data da última atualização.",
            "key": "Chave do atributo reportado.",
            "value": "Valor do atributo reportado.",
            "metadata": "Metadados em JSON.",
            "unique_id_aprendizap": "Identificador reconciliado do usuário no AprendiZAP.",
        },
    },
    "stg_lessons": {
        "description": "Tabela de staging que armazena metadados educacionais das aulas disponíveis na plataforma.",
        "columns": {
            "id_aula": "Identificação da aula.",
            "titulo": "Título da aula.",
            "nivel": "Nível de ensino.",
            "ano": "Ano de ensino.",
            "ano_em": "Ano no recorte de Ensino Médio quando aplicável.",
            "disciplina": "Disciplina de ensino.",
            "unidade": "Unidade de ensino.",
            "bncc": "BNCC atrelada à aula.",
            "possui_metodologia_ativa": "Flag que identifica se existe metodologia ativa.",
            "total_metodologias_ativa": "Contagem de metodologias ativas.",
        },
    },
    "hotjar_pesquisa_mobile": {
        "description": "Tabela contendo informações sobre pesquisas quanto ao uso do AprendiZAP em contexto mobile.",
        "columns": {
            "User": "Identificação do usuário.",
            "Date Submitted": "Data de envio da resposta.",
            "Country": "País do usuário.",
            "Source URL": "Fonte da pesquisa.",
            "Device": "Tipo de dispositivo.",
            "Browser": "Tipo de navegador.",
            "OS": "Sistema operacional.",
            "Hotjar User ID": "Identificação do usuário no CRM.",
            "Como você avalia a sua experiência?": "Avaliação da experiência.",
            "Conte-nos sobre a sua experiência...": "Descrição da experiência.",
            "Você topa conversar com a gente para melhorar sua experiência?": "Disponibilidade para conversar com o time.",
            "Sentiment for: Conte-nos sobre a sua experiência...": "Sentimento do comentário aberto.",
        },
    },
    "hotjar_pesquisa_desktop": {
        "description": "Tabela contendo informações sobre pesquisas quanto ao uso do AprendiZAP em contexto desktop.",
        "columns": {
            "User": "Identificação do usuário.",
            "Date Submitted": "Data de envio da resposta.",
            "Country": "País do usuário.",
            "Source URL": "Fonte da pesquisa.",
            "Device": "Tipo de dispositivo.",
            "Browser": "Tipo de navegador.",
            "OS": "Sistema operacional.",
            "Hotjar User ID": "Identificação do usuário no CRM.",
            "Como você avalia a sua experiência?": "Avaliação da experiência.",
            "Conte-nos sobre a sua experiência...": "Descrição da experiência.",
            "Você topa conversar com a gente para melhorar sua experiência?": "Disponibilidade para conversar com o time.",
            "Sentiment for: Conte-nos sobre a sua experiência...": "Sentimento do comentário aberto.",
        },
    },
    "hotjar_teste_interesse": {
        "description": "Tabela contendo informações sobre pesquisas quanto ao uso do AprendiZAP voltadas a interesse e engajamento.",
        "columns": {
            "User": "Identificação do usuário.",
            "Date Submitted": "Data de envio da resposta.",
            "Country": "País do usuário.",
            "Source URL": "Fonte da pesquisa.",
            "Device": "Tipo de dispositivo.",
            "Browser": "Tipo de navegador.",
            "OS": "Sistema operacional.",
            "Hotjar User ID": "Identificação do usuário no CRM.",
            "Quais estratégias você utiliza para engajar os alunos desinteressados?": "Estratégias usadas para engajar alunos desinteressados.",
            "Quão útil seria para você discutir desafios educacionais com outros educadores através da nossa plataforma?": "Utilidade percebida de discutir desafios com outros educadores.",
            "Sentiment for: Quais estratégias você utiliza para engajar os alunos desinteressados?": "Sentimento do comentário sobre estratégias.",
        },
    },
    "school_calendar": {
        "description": "Calendário escolar raw usado para enriquecer a dimensão de calendário da base modelada.",
        "columns": {
            "year": "Ano de referência.",
            "month": "Mês numérico de referência.",
            "month_start": "Início do mês de referência.",
            "uf": "UF do calendário.",
            "rede": "Rede de ensino.",
            "business_days": "Quantidade de dias úteis estimados no mês.",
            "official_holiday_weekdays": "Quantidade de feriados oficiais em dias úteis.",
            "school_days_estimate": "Estimativa de dias letivos no mês.",
            "calendar_source": "Fonte do calendário consolidado.",
        },
    },
}


MODELED_TABLE_DESCRIPTIONS = {
    "audit_base_modelada_validation": "Auditoria final de validação estrutural da base modelada.",
    "audit_persona_feature_readiness": "Auditoria de prontidão das variáveis usadas em personas e clustering.",
    "base_modelada_v2": "Painel mensal principal da base modelada com contexto e uso por professor.",
    "bridge_mari_conversation_teacher": "Bridge de resolução entre identificadores da Mari e teacher_unique_id.",
    "bridge_teacher_identity_audit": "Auditoria de reconciliação de identidade entre fontes raw e teacher_unique_id.",
    "dim_calendar": "Dimensão de calendário escolar usada como contexto temporal.",
    "dim_device": "Dimensão de dispositivo padronizado.",
    "dim_event": "Dimensão semântica dos eventos observados nas interações.",
    "dim_lesson": "Dimensão de aulas reconciliada com o catálogo educacional.",
    "dim_persona_range_candidates": "Candidatos de faixas para leitura de personas.",
    "dim_teacher": "Dimensão modelada de professores com cadastro, histórico e flags de qualidade.",
    "fct_formation_clean": "Fato de formação limpo por evento de curso.",
    "fct_interaction_clean": "Fato de interações limpas com classificação semântica de evento.",
    "fct_mari_conversation_resolved": "Fato resolvido das conversas da Mari IA.",
    "fct_mari_help_resolved": "Fato resolvido dos eventos de ajuda da Mari IA.",
    "fct_mari_reports_resolved": "Fato resolvido dos reports estruturados da Mari IA.",
    "fct_session_clean": "Fato de sessões limpas por sessão.",
    "fct_session_raw": "Fato bruto de sessões antes dos filtros de limpeza.",
    "fct_teacher_month": "Fato mensal por professor que sintetiza sessões e interações.",
    "mart_teacher_cluster_ready": "Mart por professor pronto para segmentação exploratória.",
    "mart_teacher_month_cluster_ready": "Mart mensal por professor pronto para segmentação exploratória.",
    "mart_teacher_month_panel": "Painel mensal expandido com meses observados e não observados por professor.",
    "mart_teacher_month_persona_ready": "Mart mensal por professor pronto para leitura de personas.",
    "mart_teacher_persona_ready": "Mart por professor pronto para leitura de personas.",
}


ML_TABLE_DESCRIPTIONS = {
    "core_cv_metric_folds_v1": "Robustez por fold das métricas probabilísticas do modelo.",
    "core_cv_metric_summary_v1": "Resumo de robustez por fold das métricas probabilísticas.",
    "core_cv_score_folds_v1": "Robustez por fold da distribuição de score e risco.",
    "core_cv_score_summary_v1": "Resumo de robustez por fold da distribuição de score e risco.",
    "core_definition_b_excessive_separation_v1": "Diagnóstico de separação excessiva na Definição B.",
    "core_definition_b_feature_block_gain_folds_v1": "Ganho incremental por bloco de sinais da Definição B em nível de fold.",
    "core_definition_b_feature_block_gain_summary_v1": "Resumo do ganho incremental por bloco de sinais da Definição B.",
    "core_definition_candidates_test_frontier_v1": "Diagnóstico em teste dos candidatos de definição futura.",
    "core_definition_candidates_train_v1": "Diagnóstico em treino dos candidatos de definição futura.",
    "core_definition_external_validation_v1": "Validação externa temporal das definições futuras.",
    "core_definition_frontier_v1": "Fronteira admissível das definições futuras.",
    "core_definition_selection_v1": "Seleção oficial das definições futuras testadas.",
    "core_model_calibration_audit_v1": "Auditoria dos splits internos de tuning e calibração.",
    "core_model_fold_metrics_v1": "Métricas por fold externo dos modelos avaliados.",
    "core_model_frontier_v1": "Fronteira final de definição, trilha e família de modelo.",
    "core_model_predictions_v1": "Predições externas dos modelos avaliados.",
    "core_navigation_sequences_v1": "Sequências iniciais de navegação observadas na jornada.",
    "core_navigation_transitions_v1": "Transições entre passos de navegação na jornada inicial.",
    "core_prediction_bootstrap_v1": "Intervalos bootstrap das métricas probabilísticas das predições oficiais.",
    "core_scoring_scenarios_v1": "Cenários oficiais de score com definição, trilha e conjunto elegível de variáveis.",
    "governance_arbitrariness_registry_v1": "Registro das escolhas arbitrárias ou mecânicas expostas no estudo.",
    "governance_definition_candidate_metric_registry_v1": "Registro das métricas permitidas na busca de candidatos de definição.",
    "governance_feature_eligibility_v1": "Elegibilidade de cada variável por trilha de score.",
    "governance_feature_registry_v1": "Registro das variáveis candidatas e suas classes analíticas.",
    "governance_label_registry_v1": "Registro das definições futuras materializadas e suas regras.",
    "governance_leakage_audit_v1": "Auditoria detalhada de leakage por variável e cenário.",
    "governance_leakage_summary_v1": "Resumo expandido da auditoria de leakage.",
    "governance_policy_registry_v1": "Registro das políticas operacionais e analíticas publicadas.",
    "governance_post_model_output_status_v1": "Status das saídas pós-modelo por problema, modelo e fold.",
    "governance_track_registry_v1": "Registro das trilhas de score e sua janela de leitura.",
    "mart_first_session_journey_v1": "Resumo da 1ª sessão, do 1º evento e da janela inicial da jornada.",
    "mart_future_metrics_v1": "Métricas futuras nativas usadas para definição de atividade e validadores.",
    "mart_onboarding_population_v1": "População inicial de onboarding reconstruída para o modelo.",
    "post_model_band_summary_v1": "Resumo das faixas de risco por política registrada.",
    "post_model_cluster_assignment_v1": "Atribuição de cluster descritivo por caso publicado.",
    "post_model_cluster_profile_v1": "Perfil agregado dos clusters descritivos publicados.",
    "post_model_cluster_summary_v1": "Resumo dos clusters descritivos publicados.",
    "post_model_cluster_validation_v1": "Métricas de validação dos clusters descritivos publicados.",
    "post_model_confusion_matrix_v1": "Matriz de confusão operacional por política de cutoff.",
    "post_model_cv_confusion_folds_v1": "Matriz de confusão por fold e política de cutoff.",
    "post_model_cv_confusion_summary_v1": "Resumo da robustez da matriz de confusão por fold.",
    "post_model_cv_threshold_folds_v1": "Métricas por fold das políticas de cutoff.",
    "post_model_cv_threshold_summary_v1": "Resumo da robustez das políticas de cutoff por fold.",
    "post_model_feature_importance_v1": "Importância por permutação dos modelos de referência.",
    "post_model_heavy_user_profile_v1": "Perfil agregado do proxy de heavy user.",
    "post_model_heavy_user_scores_v1": "Scores do proxy de heavy user por caso.",
    "post_model_heavy_user_summary_v1": "Resumo do proxy de heavy user publicado.",
    "post_model_monthly_fit_v1": "Ajuste mensal entre risco previsto e risco realizado.",
    "post_model_reference_selection_v1": "Escopo final de referência usado no relatório.",
    "post_model_threshold_metrics_v1": "Métricas operacionais por política de cutoff.",
}


COMMON_DESCRIPTIONS = {
    "teacher_unique_id": "Identificador único do professor.",
    "first_month": "Primeiro mês observado do professor.",
    "month": "Mês de referência.",
    "problem_key": "Chave do problema de score.",
    "definition_name": "Nome da definição futura.",
    "definition_group": "Grupo da definição futura.",
    "track_name": "Nome da trilha de score.",
    "model_name": "Nome da família de modelo.",
    "policy_name": "Nome da política registrada.",
    "metric_name": "Nome da métrica.",
    "selection_reason": "Motivo da seleção no escopo de referência.",
    "rule_json": "Regra estruturada em JSON.",
    "rule_text": "Regra em texto para exibição.",
    "rule_size": "Quantidade de componentes da regra.",
    "rule_operator": "Operador usado na regra.",
    "selection_basis": "Base usada para decidir o status oficial.",
    "feature_names_json": "Lista JSON das variáveis elegíveis.",
    "selected_feature_names_json": "Lista JSON das variáveis selecionadas no bloco.",
    "added_feature_names_json": "Lista JSON das variáveis adicionadas sobre o baseline.",
    "parameter_json": "Parâmetros da política em JSON.",
    "best_params_json": "Melhores hiperparâmetros em JSON.",
    "label_col": "Coluna binária usada como alvo.",
    "candidate_type": "Tipo do candidato de definição.",
    "metric_value": "Valor medido na auditoria.",
    "status": "Status do check.",
    "note": "Observação complementar do check.",
    "description": "Descrição textual.",
    "value": "Valor observado.",
    "definition": "Definição textual da variável.",
    "caveat": "Ressalva metodológica.",
    "source_table": "Tabela de origem.",
    "source_key_name": "Nome da chave observada na origem.",
    "source_key": "Valor bruto da chave observada.",
    "source_user_types": "Tipos de usuário observados na origem.",
    "content_type": "Tipo de conteúdo.",
    "utm_source": "Origem UTM da sessão.",
    "user_type": "Tipo de usuário.",
    "item_type": "Tipo do item.",
    "origin_source": "Fonte de origem.",
    "feature_type": "Tipo da variável.",
    "metric_type": "Tipo da métrica.",
    "label_hash": "Hash do vetor de label.",
    "features_with_future_named_source": "Quantidade de variáveis com source nomeado como futuro.",
    "row_hash": "Hash técnico da linha.",
    "session_row_hash": "Hash técnico da sessão.",
    "interaction_row_hash": "Hash técnico da interação.",
    "formation_row_hash": "Hash técnico do evento de formação.",
    "resolution_path": "Caminho usado na reconciliação.",
    "resolved_teacher_unique_id": "Professor reconciliado.",
    "resolved_teacher_count": "Quantidade de professores candidatos.",
    "is_unambiguous": "Flag indicando resolução sem ambiguidade.",
    "source_key_domain": "Domínio inferido da chave de origem.",
    "is_resolved_flag": "Flag indicando se a chave foi resolvida.",
    "id_mari": "Identificador bruto da Mari.",
    "teacher_resolution_count": "Quantidade de resoluções candidatas encontradas.",
    "resolution_source": "Fonte usada na reconciliação.",
    "teacher_candidates": "Lista de professores candidatos.",
    "report_rows": "Quantidade de linhas de report associadas.",
    "conv_rows": "Quantidade de linhas de conversa associadas.",
    "device_group": "Agrupamento padronizado do dispositivo.",
    "event_type": "Tipo bruto do evento.",
    "event_family": "Família semântica do evento.",
    "event_action": "Ação semântica do evento.",
    "lesson_id": "Identificador reconciliado da aula.",
    "month_start": "Início do mês de referência.",
    "uf": "UF de referência.",
    "rede": "Rede de ensino.",
    "calendar_source": "Fonte do calendário.",
    "school_phase": "Fase do período escolar.",
    "rows": "Quantidade de linhas.",
    "positives": "Quantidade de casos positivos.",
    "negatives": "Quantidade de casos negativos.",
    "valid_folds": "Quantidade de folds externos válidos.",
    "fold_id": "Identificador do fold externo.",
    "inner_fold_id": "Identificador do fold interno.",
    "outer_fold_id": "Identificador do fold externo.",
    "y_true": "Valor observado do alvo binário.",
    "score": "Probabilidade prevista de atividade futura.",
    "risk_score": "Probabilidade prevista de não realizar o alvo.",
    "y_risk_true": "Valor observado do alvo de risco.",
    "realized_risk_rate": "Taxa realizada de risco.",
    "realized_inactivity_rate": "Taxa realizada de inatividade.",
    "score_std": "Desvio-padrão do score.",
    "risk_score_std": "Desvio-padrão do risk_score.",
    "invalid_reason": "Motivo de invalidação do fold ou da saída.",
    "fold_valid_flag": "Flag indicando se o fold conta no resumo oficial.",
    "technical_fold_valid_flag": "Flag indicando se o fold era tecnicamente calculável.",
    "tuning_applied_flag": "Flag indicando se houve tuning.",
    "tuning_status": "Status do tuning.",
    "tuning_valid_splits": "Quantidade de splits internos válidos no tuning.",
    "tuning_best_score": "Melhor score encontrado no tuning.",
    "ap": "Average Precision do fold.",
    "roc_auc": "ROC AUC do fold.",
    "brier": "Brier score do fold.",
    "log_loss": "Log loss do fold.",
    "calibration_slope": "Inclinação de calibração.",
    "calibration_intercept": "Intercepto de calibração.",
    "calibration_slope_error": "Distância da inclinação de calibração até 1.",
    "calibration_intercept_abs": "Valor absoluto do intercepto de calibração.",
    "mean_ap": "Average Precision na predição concatenada.",
    "mean_roc_auc": "ROC AUC na predição concatenada.",
    "mean_brier": "Brier score na predição concatenada.",
    "mean_log_loss": "Log loss na predição concatenada.",
    "mean_calibration_slope": "Inclinação de calibração na predição concatenada.",
    "mean_calibration_intercept": "Intercepto de calibração na predição concatenada.",
    "mean_calibration_slope_error": "Distância da inclinação de calibração até 1 na predição concatenada.",
    "mean_calibration_intercept_abs": "Valor absoluto do intercepto de calibração na predição concatenada.",
    "pooled_rows": "Quantidade de linhas na predição concatenada.",
    "pooled_positives": "Quantidade de positivos na predição concatenada.",
    "pooled_negatives": "Quantidade de negativos na predição concatenada.",
    "pooled_positive_rate": "Taxa positiva na predição concatenada.",
    "fold_mean_ap": "Média de AP entre folds válidos.",
    "fold_mean_roc_auc": "Média de ROC AUC entre folds válidos.",
    "fold_mean_brier": "Média de Brier entre folds válidos.",
    "fold_mean_log_loss": "Média de log loss entre folds válidos.",
    "fold_mean_calibration_slope": "Média da inclinação de calibração entre folds válidos.",
    "fold_mean_calibration_intercept": "Média do intercepto de calibração entre folds válidos.",
    "fold_mean_calibration_slope_error": "Média da distância da inclinação até 1.",
    "fold_mean_calibration_intercept_abs": "Média do valor absoluto do intercepto.",
    "std_ap": "Desvio-padrão de AP entre folds válidos.",
    "std_roc_auc": "Desvio-padrão de ROC AUC entre folds válidos.",
    "std_brier": "Desvio-padrão de Brier entre folds válidos.",
    "std_log_loss": "Desvio-padrão de log loss entre folds válidos.",
    "pareto_frontier_flag": "Flag indicando presença na fronteira de Pareto.",
    "ci_low": "Limite inferior do intervalo.",
    "ci_high": "Limite superior do intervalo.",
    "ci_width": "Largura do intervalo.",
    "risk_threshold": "Cutoff de risco usado na política.",
    "tp": "Verdadeiros positivos.",
    "fp": "Falsos positivos.",
    "tn": "Verdadeiros negativos.",
    "fn": "Falsos negativos.",
    "precision": "Precisão no cutoff.",
    "recall": "Recall no cutoff.",
    "f1": "F1 no cutoff.",
    "accuracy": "Acurácia no cutoff.",
    "predicted_positive_rate": "Taxa prevista como positiva no cutoff.",
    "actual_group": "Grupo observado na matriz de confusão.",
    "predicted_group": "Grupo previsto na matriz de confusão.",
    "band_name": "Nome da faixa de risco.",
    "share": "Participação do grupo na base.",
    "monthly_r2": "R² entre risco previsto e realizado por mês.",
    "monthly_mape_positive_months": "MAPE mensal considerando só meses com risco realizado positivo.",
    "months_used": "Quantidade de meses usados no ajuste mensal.",
    "feature_name": "Nome da variável.",
    "importance_mean": "Importância média por permutação.",
    "importance_std": "Desvio-padrão da importância por permutação.",
    "reference_problem_key": "Problema de referência usado no diagnóstico.",
    "diagnostic_problem_key": "Problema diagnóstico derivado do bloco de variáveis.",
    "block_name": "Nome do bloco de variáveis testado.",
    "block_type": "Tipo do bloco de variáveis.",
    "selected_feature_count": "Quantidade de variáveis no bloco testado.",
    "added_feature_count": "Quantidade de variáveis adicionadas sobre o baseline.",
    "baseline_mean_ap": "AP do baseline de contexto.",
    "baseline_mean_roc_auc": "ROC AUC do baseline de contexto.",
    "baseline_mean_brier": "Brier do baseline de contexto.",
    "baseline_mean_log_loss": "Log loss do baseline de contexto.",
    "delta_ap_vs_context": "Ganho de AP sobre o baseline de contexto.",
    "delta_roc_auc_vs_context": "Ganho de ROC AUC sobre o baseline de contexto.",
    "brier_improvement_vs_context": "Melhora de Brier sobre o baseline de contexto.",
    "log_loss_improvement_vs_context": "Melhora de log loss sobre o baseline de contexto.",
    "uplift_metric_positive_count": "Quantidade de métricas de ganho positivas.",
    "mean_uplift_percentile": "Percentil médio de ganho do bloco.",
    "abnormal_uplift_flag": "Flag de ganho anormal sobre o baseline.",
    "mean_value": "Valor médio entre folds válidos.",
    "std_value": "Desvio-padrão entre folds válidos.",
    "min_value": "Valor mínimo entre folds válidos.",
    "max_value": "Valor máximo entre folds válidos.",
    "value_range": "Amplitude entre o menor e o maior valor.",
    "max_fold_to_fold_jump": "Maior salto entre folds consecutivos.",
    "fold_order_slope": "Inclinação da métrica ao longo da ordem dos folds.",
    "fold_order_pvalue": "P-valor da inclinação por permutação.",
    "prevalence_entropy": "Entropia da prevalência do label.",
    "monthly_prevalence_std": "Desvio-padrão da prevalência mensal.",
    "bootstrap_prevalence_ci_width": "Largura do intervalo bootstrap da prevalência.",
    "test_prevalence_entropy": "Entropia da prevalência do label em teste.",
    "test_monthly_prevalence_std": "Desvio-padrão da prevalência mensal em teste.",
    "test_bootstrap_prevalence_ci_width": "Largura do intervalo bootstrap da prevalência em teste.",
    "rank_test_monthly_prevalence_std": "Rank do desvio-padrão da prevalência mensal em teste.",
    "std_rows": "Desvio-padrão da quantidade de linhas entre folds.",
    "check_name": "Nome do check de auditoria.",
    "feature_level": "Nível analítico da variável.",
    "feature_role": "Papel analítico da variável.",
    "missing_rate": "Proporção de valores ausentes.",
    "zero_share": "Proporção de valores zero.",
    "std": "Desvio-padrão da variável.",
    "recommended_for_persona_analysis": "Flag indicando recomendação para análise de personas.",
    "recommended_for_persona_ranges": "Flag indicando recomendação para faixas de personas.",
    "recommended_for_behavior_clustering": "Flag indicando recomendação para clustering comportamental.",
    "n_rows": "Quantidade de linhas usadas no cálculo.",
    "min_value": "Menor valor observado.",
    "p10": "Percentil 10.",
    "p25": "Percentil 25.",
    "p50": "Percentil 50.",
    "p75": "Percentil 75.",
    "p90": "Percentil 90.",
    "p95": "Percentil 95.",
    "max_value": "Maior valor observado.",
    "raw_rows_total": "Quantidade total de linhas raw.",
    "core_rows_total": "Quantidade total de linhas mantidas na camada core.",
    "core_activity_rows_total": "Quantidade total de linhas de atividade na camada core.",
    "core_download_rows_total": "Quantidade total de linhas de download na camada core.",
    "interaction_rows_total": "Quantidade total de interações.",
    "distinct_teachers_total": "Quantidade total de professores distintos.",
    "first_observed_ts": "Primeiro timestamp observado.",
    "last_observed_ts": "Último timestamp observado.",
    "download_events_total": "Quantidade total de eventos de download.",
    "strict_download_events_total": "Quantidade total de downloads estritos.",
    "content_view_events_total": "Quantidade total de visualizações de conteúdo.",
    "lesson_metadata_matched_flag": "Flag indicando se a aula foi conciliada ao catálogo.",
    "lesson_title": "Título da aula.",
    "lesson_level": "Nível de ensino da aula.",
    "lesson_year": "Ano ou série da aula.",
    "lesson_year_em": "Ano ou série conciliado para Ensino Médio.",
    "lesson_discipline": "Disciplina da aula.",
    "lesson_discipline_group": "Grupo analítico da disciplina da aula.",
    "lesson_unit": "Unidade didática da aula.",
    "lesson_bncc": "Lista de referências BNCC da aula.",
    "lesson_has_active_methodology": "Flag indicando se a aula tem metodologia ativa.",
    "lesson_total_active_methodologies": "Quantidade de metodologias ativas da aula.",
    "lesson_id_semantic": "Identificador semântico da aula.",
    "raw_lesson_id_valid_flag": "Flag indicando se o identificador bruto da aula é válido.",
    "is_active_methodology_missing": "Flag de ausência da informação de metodologia ativa.",
    "is_metadata_missing": "Flag de ausência de metadados da aula.",
    "calendar_year": "Ano do calendário.",
    "calendar_month": "Mês do calendário.",
    "business_days": "Quantidade de dias úteis no mês.",
    "official_holiday_weekdays": "Quantidade de feriados em dias úteis.",
    "school_days_estimate": "Estimativa de dias letivos no mês.",
    "label_start_ts": "Início da janela de label.",
    "label_end_ts": "Fim da janela de label.",
    "validator_1_end_ts": "Fim do 1º bloco validador pós-label.",
    "validator_2_end_ts": "Fim do 2º bloco validador pós-label.",
    "validator_3_end_ts": "Fim do 3º bloco validador pós-label.",
    "full_followup_observed_flag": "Flag indicando seguimento futuro completo observado.",
    "official_status": "Status oficial do candidato.",
    "winner_flag": "Flag indicando vencedor único.",
    "threshold": "Threshold aplicado na regra.",
    "folds": "Quantidade de folds válidos.",
    "label_hash": "Hash do vetor de label.",
    "label_positives": "Quantidade de positivos do label.",
    "label_share_pct": "Participação percentual dos positivos do label.",
    "label_vector_group_size": "Quantidade de candidatos com o mesmo vetor de label.",
    "months": "Quantidade de meses distintos.",
    "feature_count": "Quantidade de variáveis elegíveis.",
    "score_window_end_day": "Último dia disponível para leitura do score.",
    "raw_entry_session_count_month": "Quantidade mensal de sessões de entrada brutas.",
    "ping_entry_session_count_month": "Quantidade mensal de sessões de entrada do tipo ping.",
    "clean_entry_session_count_month": "Quantidade mensal de sessões de entrada limpas.",
    "clean_entry_total_session_minutes_month": "Quantidade mensal de minutos em sessões de entrada limpas.",
    "clean_entry_avg_session_minutes_month": "Média mensal de minutos por sessão de entrada limpa.",
    "interaction_rows_month": "Quantidade mensal de linhas de interação.",
    "activity_events_month": "Quantidade mensal de eventos de atividade.",
    "active_days_month": "Quantidade mensal de dias ativos.",
    "aula_events_month": "Quantidade mensal de eventos de aula.",
    "plano_events_month": "Quantidade mensal de eventos de plano.",
    "prova_events_month": "Quantidade mensal de eventos de prova.",
    "ia_events_month": "Quantidade mensal de eventos de IA.",
    "download_count_month": "Quantidade mensal de downloads.",
    "download_aula_count_month": "Quantidade mensal de downloads de aula.",
    "download_plano_count_month": "Quantidade mensal de downloads de plano.",
    "strict_download_count_month": "Quantidade mensal de downloads estritos.",
    "content_views_month": "Quantidade mensal de visualizações de conteúdo.",
    "other_activity_non_download_events_month": "Quantidade mensal de eventos de atividade não ligados a download.",
    "mapped_lessons_month": "Quantidade mensal de aulas mapeadas.",
    "interaction_signal_flag": "Flag indicando presença de sinal de interação.",
    "entry_signal_flag": "Flag indicando presença de sinal de sessão de entrada.",
    "clean_entry_signal_flag": "Flag indicando presença de sinal de sessão limpa de entrada.",
    "month_signal_class": "Classe mensal de sinal observado.",
    "active_user_flag": "Flag indicando usuário ativo no mês.",
    "viewed_aula_flag": "Flag indicando visualização de aula.",
    "viewed_plano_flag": "Flag indicando visualização de plano.",
    "viewed_prova_flag": "Flag indicando visualização de prova.",
    "no_download_flag": "Flag indicando ausência de download.",
    "no_download_view_only_flag": "Flag indicando apenas visualização sem download.",
    "no_download_view_plus_action_flag": "Flag indicando visualização e ação sem download.",
    "no_download_action_only_flag": "Flag indicando ação sem download.",
    "clean_entry_exposed_no_download_flag": "Flag indicando exposição em sessão limpa de entrada sem download.",
    "clean_entry_exposed_no_activity_no_download_flag": "Flag indicando exposição em sessão limpa de entrada sem atividade e sem download.",
    "clean_entry_exposed_activity_no_download_flag": "Flag indicando exposição em sessão limpa de entrada com atividade, mas sem download.",
    "month_num": "Número sequencial do mês observado.",
    "next_month": "Mês seguinte observado.",
    "returned_strict_value_m1": "Flag indicando retorno de valor estrito no 1º mês.",
    "strict_return_value_m1": "Valor estrito realizado no 1º mês seguinte.",
    "lifetime_active_months": "Quantidade histórica de meses ativos.",
    "lifetime_clean_entry_minutes_total": "Total histórico de minutos em sessões de entrada limpas.",
    "active_streak_current_months": "Quantidade de meses na sequência corrente de atividade.",
    "active_streak_max_months": "Quantidade de meses na maior sequência de atividade.",
    "strict_streak_current_months": "Quantidade de meses na sequência corrente de atividade estrita.",
    "strict_streak_max_months": "Quantidade de meses na maior sequência de atividade estrita.",
    "ap_good_percentile": "Percentil favorável de AP dentro da trilha.",
    "roc_auc_good_percentile": "Percentil favorável de ROC AUC dentro da trilha.",
    "brier_good_percentile": "Percentil favorável de Brier dentro da trilha.",
    "log_loss_good_percentile": "Percentil favorável de log loss dentro da trilha.",
    "calibration_good_percentile": "Percentil favorável de calibração dentro da trilha.",
    "stability_good_percentile": "Percentil favorável de estabilidade dentro da trilha.",
    "combined_separation_score": "Score combinado de separação.",
    "combined_separation_percentile_within_track": "Percentil do score combinado de separação dentro da trilha.",
    "comparator_rows_in_track": "Quantidade de comparadores disponíveis na trilha.",
    "brier_improvement_vs_context_percentile": "Percentil da melhora de Brier sobre o baseline de contexto.",
    "log_loss_improvement_vs_context_percentile": "Percentil da melhora de log loss sobre o baseline de contexto.",
    "choice_name": "Nome da escolha registrada.",
    "choice_type": "Tipo da escolha registrada.",
    "choice_status": "Status da escolha registrada.",
    "justification": "Justificativa da escolha.",
    "in_official_report_flag": "Flag indicando presença no relatório oficial.",
    "entity_name": "Nome da entidade registrada.",
    "entity_type": "Tipo da entidade registrada.",
    "allowed_flag": "Flag indicando elegibilidade.",
    "post_model_output_name": "Nome da saída pós-modelo.",
    "post_model_output_status": "Status da saída pós-modelo.",
    "error_type": "Tipo do erro ocorrido.",
    "error_message": "Mensagem resumida do erro.",
    "traceback_snippet": "Trecho resumido do traceback.",
}


TEACHER_FIELD_LABELS = {
    "population_status": "status do professor na população observada",
    "utm_origin": "origem UTM do cadastro",
    "utm_group": "grupo analítico da origem UTM",
    "tela_origem": "tela de origem do cadastro",
    "estado": "estado informado",
    "total_alunos": "total de alunos informado",
    "tipo_total_alunos": "tipo do total de alunos informado",
    "alunos_diretos": "quantidade de alunos diretos",
    "alunos_indiretos": "quantidade de alunos indiretos",
    "login_google": "uso de login via Google",
    "currentstage": "etapa atual do professor",
    "currentsubject": "disciplina atual do professor",
    "currentsubject_group": "grupo analítico da disciplina atual",
    "selectedstages": "etapas selecionadas no cadastro",
    "selectedsubjectsem": "disciplinas selecionadas no Ensino Médio",
    "selectedsubjectsfundii": "disciplinas selecionadas no Fundamental II",
    "visualizou_metodologia_ativa": "visualização de metodologia ativa",
    "data_entrada": "data de entrada do professor",
    "first_observed_month": "primeiro mês observado do professor",
    "last_observed_month": "último mês observado do professor",
    "months_since_last_observed_month_dataset_end": "meses entre a última observação e o fim do dataset",
    "observed_months_total": "quantidade total de meses observados",
    "active_months_total": "quantidade total de meses ativos",
    "strict_months_total": "quantidade total de meses ativos na definição estrita",
    "total_strict_downloads": "total histórico de downloads estritos",
    "total_downloads": "total histórico de downloads",
    "total_clean_entry_sessions": "total histórico de sessões limpas",
    "total_clean_entry_minutes": "total histórico de minutos em sessões limpas",
    "active_streak_max_months": "maior sequência mensal de atividade",
    "strict_streak_max_months": "maior sequência mensal de atividade estrita",
}

FIELD_PHRASES = {
    "raw_entry_session_count": "sessões de entrada brutas",
    "ping_entry_session_count": "sessões de entrada do tipo ping",
    "clean_entry_session_count": "sessões de entrada limpas",
    "clean_entry_total_session_minutes": "minutos totais em sessões de entrada limpas",
    "clean_entry_avg_session_minutes": "minutos médios por sessão de entrada limpa",
    "active": "dias ativos",
    "interaction_rows": "linhas de interação",
    "activity_events": "eventos de atividade",
    "active_days": "dias ativos",
    "aula_events": "eventos de aula",
    "plano_events": "eventos de plano",
    "prova_events": "eventos de prova",
    "ia_events": "eventos de IA",
    "download_count": "downloads",
    "download_aula_count": "downloads de aula",
    "download_plano_count": "downloads de plano",
    "strict_download_count": "downloads estritos",
    "content_views": "visualizações de conteúdo",
    "other_activity_non_download_events": "eventos de atividade não ligados a download",
    "mapped_lessons": "aulas mapeadas",
    "any_signal": "qualquer sinal observado",
    "month_signal_class": "classe mensal de sinal",
    "strict_value": "valor estrito",
    "strict_user": "usuário estrito",
    "next_month": "mês seguinte",
    "returned_strict_value": "valor estrito",
    "used_ia": "uso de IA",
    "used_desktop": "uso de desktop",
    "used_mobile": "uso de mobile",
    "registered_entry": "sessão registrada",
    "registered_interaction": "interação registrada",
    "utm": "origem UTM",
    "tela_origem": "tela de origem",
    "total_alunos": "total de alunos",
    "tipo_total_alunos": "tipo do total de alunos",
    "login_google": "login via Google",
    "currentstage": "etapa atual",
    "currentsubject": "disciplina atual",
    "selectedstages": "etapas selecionadas",
    "selectedsubjectsem": "disciplinas selecionadas do Ensino Médio",
    "selectedsubjectsfundii": "disciplinas selecionadas do Fundamental II",
    "content_type": "tipo de conteúdo",
    "utm_source": "origem UTM da sessão",
    "user_type": "tipo de usuário",
    "item_type": "tipo do item",
    "origin_source": "fonte de origem",
    "feature_type": "tipo da variável",
    "metric_type": "tipo da métrica",
    "label_hash": "hash do vetor de label",
    "features_with_future_named_source": "variáveis com source nomeado como futuro",
    "row_hash": "hash técnico da linha",
    "session_row_hash": "hash técnico da sessão",
    "interaction_row_hash": "hash técnico da interação",
    "formation_row_hash": "hash técnico do evento de formação",
}


TOKEN_LABELS = {
    "entry": "sessão de entrada",
    "entries": "sessões de entrada",
    "session": "sessão",
    "sessions": "sessões",
    "interaction": "interação",
    "interactions": "interações",
    "activity": "atividade",
    "events": "eventos",
    "event": "evento",
    "active": "atividade",
    "days": "dias",
    "day": "dia",
    "download": "download",
    "downloads": "downloads",
    "content": "conteúdo",
    "views": "visualizações",
    "view": "visualização",
    "other": "outras",
    "actions": "ações",
    "action": "ação",
    "meaningful": "relevante",
    "mapped": "mapeadas",
    "lessons": "aulas",
    "lesson": "aula",
    "plano": "planos",
    "prova": "provas",
    "aula": "aulas",
    "ia": "IA",
    "desktop": "desktop",
    "mobile": "mobile",
    "clean": "limpas",
    "ping": "ping",
    "raw": "brutas",
    "strict": "estrita",
    "value": "valor",
    "signal": "sinal",
    "class": "classe",
    "current": "corrente",
    "max": "máxima",
    "lifetime": "histórica",
    "minutes": "minutos",
    "minute": "minuto",
    "avg": "média",
    "total": "total",
    "first": "primeiro",
    "last": "último",
    "next": "próximo",
    "observed": "observado",
    "returned": "retorno",
    "post": "pós",
    "label": "label",
    "business": "de negócio",
    "weeks": "semanas",
    "week": "semana",
    "distinct": "distintas",
    "formation": "formação",
    "conversation": "conversa",
    "help": "ajuda",
    "risk": "risco",
    "any": "qualquer",
    "only": "apenas",
    "plus": "mais",
    "non": "não",
}


EXACT_TABLE_ALIASES = {
    "mart_first_session_journey_v1": ["mart_onboarding_population_v1", "mart_first_session_journey_v1"],
}


EXACT_BY_TABLE = {
    "audit_base_modelada_validation": {
        "check_name": "Nome do check de validação.",
        "metric_value": "Valor medido no check.",
        "status": "Status do check.",
        "note": "Observação complementar do check.",
    },
    "audit_persona_feature_readiness": {
        "feature_name": "Nome da variável auditada.",
        "feature_level": "Nível analítico da variável.",
        "feature_role": "Papel analítico da variável.",
        "definition": "Definição curta da variável.",
        "missing_rate": "Proporção de valores ausentes.",
        "zero_share": "Proporção de valores zero.",
        "std": "Desvio-padrão da variável.",
        "recommended_for_persona_analysis": "Flag indicando recomendação para personas.",
        "recommended_for_persona_ranges": "Flag indicando recomendação para faixas.",
        "recommended_for_behavior_clustering": "Flag indicando recomendação para clustering.",
        "caveat": "Ressalva metodológica da variável.",
    },
    "mart_onboarding_population_v1": {
        "onboarding_anchor_ts": "Timestamp âncora do onboarding.",
        "data_entrada_month": "Mês de entrada do professor.",
        "months_after_entry": "Meses entre a entrada e o primeiro mês observado.",
        "utm_group": "Grupo analítico da origem UTM.",
        "first_event_type": "Tipo do 1º evento observado.",
        "first_event_family": "Família do 1º evento observado.",
        "first_event_action": "Ação do 1º evento observado.",
        "first_device": "Dispositivo do 1º evento observado.",
        "first_utm_source": "UTM source do 1º evento observado.",
        "first3_interaction_downloads": "Downloads nas 3 primeiras interações observadas.",
        "first3_interaction_views": "Visualizações nas 3 primeiras interações observadas.",
        "first3_interaction_other_actions": "Outras ações nas 3 primeiras interações observadas.",
        "first7d_events": "Eventos nos 7 primeiros dias.",
        "first7d_active_days": "Dias ativos nos 7 primeiros dias.",
        "first_session_minutes": "Minutos da 1ª sessão.",
        "first7d_sessions": "Sessões nos 7 primeiros dias.",
        "first7d_session_minutes": "Minutos em sessão nos 7 primeiros dias.",
        "first_event_missing_flag": "Flag de ausência do 1º evento.",
        "first_event_action_missing_flag": "Flag de ausência da ação do 1º evento.",
        "first_utm_missing_flag": "Flag de ausência da UTM inicial.",
        "first_device_missing_flag": "Flag de ausência do dispositivo inicial.",
    },
    "mart_first_session_journey_v1": {
        "first_session_row_hash": "Hash técnico da 1ª sessão.",
        "first_session_start_ts": "Início da 1ª sessão.",
        "first_session_end_ts": "Fim da 1ª sessão.",
        "first_session_duration_sec": "Duração da 1ª sessão em segundos.",
        "first_session_duration_min": "Duração da 1ª sessão em minutos.",
        "first_session_interactions": "Interações dentro da 1ª sessão.",
        "first_session_downloads": "Downloads dentro da 1ª sessão.",
        "first_session_views": "Visualizações dentro da 1ª sessão.",
        "first_session_other_actions": "Outras ações dentro da 1ª sessão.",
        "first_session_navigation_events": "Eventos de navegação na 1ª sessão.",
        "first_session_meaningful_events": "Eventos relevantes na 1ª sessão.",
        "first_session_first_event_ts": "Timestamp do 1º evento da 1ª sessão.",
        "first_session_first_event_type": "Tipo do 1º evento da 1ª sessão.",
        "first_session_first_event_family": "Família do 1º evento da 1ª sessão.",
        "first_session_first_event_action": "Ação do 1º evento da 1ª sessão.",
        "first_session_first_event_utm_source": "UTM source do 1º evento da 1ª sessão.",
        "first_session_first_event_device": "Dispositivo do 1º evento da 1ª sessão.",
        "first_session_first_meaningful_ts": "Timestamp da 1ª ação relevante da 1ª sessão.",
        "first_session_first_meaningful_type": "Tipo da 1ª ação relevante da 1ª sessão.",
        "first_session_first_meaningful_family": "Família da 1ª ação relevante da 1ª sessão.",
        "first_session_first_meaningful_action": "Ação da 1ª ação relevante da 1ª sessão.",
        "first_session_last_event_ts": "Timestamp do último evento da 1ª sessão.",
        "first_session_last_event_type": "Tipo do último evento da 1ª sessão.",
        "first_session_last_event_family": "Família do último evento da 1ª sessão.",
        "first_session_last_event_action": "Ação do último evento da 1ª sessão.",
        "first_session_missing_flag": "Flag indicando ausência da 1ª sessão.",
        "first_session_has_interaction_flag": "Flag indicando interação dentro da 1ª sessão.",
        "first_session_has_meaningful_action_flag": "Flag indicando ação relevante na 1ª sessão.",
        "session_without_interaction_flag": "Flag de sessão sem interação associada.",
        "first_session_entry_surface": "Superfície de entrada da 1ª sessão.",
        "first_session_device_raw": "Dispositivo bruto da 1ª sessão.",
        "first_session_device_bucket": "Bucket do dispositivo da 1ª sessão.",
        "secs_to_first_interaction": "Segundos até a 1ª interação.",
        "secs_to_first_meaningful_action": "Segundos até a 1ª ação relevante.",
        "first_session_first_meaningful_action_group": "Grupo da 1ª ação relevante da 1ª sessão.",
        "first_session_first_event_action_group": "Grupo da ação do 1º evento da 1ª sessão.",
        "first_session_exit_state": "Estado de saída da 1ª sessão.",
        "observed_step_count_first5": "Quantidade de passos observados entre os 5 primeiros.",
        "step_sequence_first5": "Sequência dos 5 primeiros passos.",
        "step_sequence_observed_first5": "Sequência observada dos 5 primeiros passos.",
    },
    "mart_future_metrics_v1": {
        "anchor_ts": "Timestamp âncora da jornada inicial.",
        "label_start_ts": "Início da medição futura após a janela inicial.",
        "label_end_ts": "Fim da janela principal de label.",
        "validator_1_end_ts": "Fim do 1º bloco validador pós-label.",
        "validator_2_end_ts": "Fim do 2º bloco validador pós-label.",
        "validator_3_end_ts": "Fim do 3º bloco validador pós-label.",
        "future_sessions": "Sessões futuras após o início do label.",
        "future_session_minutes": "Minutos futuros em sessão após o início do label.",
        "future_interactions": "Interações futuras após o início do label.",
        "future_activity_events": "Eventos futuros de atividade após o início do label.",
        "future_active_days": "Dias ativos futuros após o início do label.",
        "future_distinct_actions": "Ações distintas futuras após o início do label.",
        "future_downloads": "Downloads futuros após o início do label.",
        "future_content_views": "Visualizações futuras de conteúdo após o início do label.",
        "future_mapped_lessons": "Aulas futuras mapeadas após o início do label.",
        "future_formation_events": "Eventos futuros de formação após o início do label.",
        "future_mari_help_events": "Eventos futuros de ajuda da Mari após o início do label.",
        "future_mari_conversation_events": "Conversas futuras com a Mari após o início do label.",
        "future_business_active_weeks": "Semanas futuras com atividade mínima de negócio.",
        "returned_active_post_label_m1": "Flag indicando retorno ativo no 1º bloco pós-label.",
        "returned_active_post_label_m2": "Flag indicando retorno ativo no 2º bloco pós-label.",
        "returned_active_post_label_m3": "Flag indicando retorno ativo no 3º bloco pós-label.",
        "active_days_post_label_3m": "Dias ativos somados nos 3 blocos pós-label.",
        "sustained_active_2of3_post_label": "Flag indicando atividade em pelo menos 2 dos 3 blocos pós-label.",
    },
}


def render_yaml(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            data,
            allow_unicode=True,
            sort_keys=False,
            width=120,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )


def table_columns(conn: duckdb.DuckDBPyConnection, table_name: str) -> list[str]:
    return [row[0] for row in conn.execute(f'describe "{table_name}"').fetchall()]


def exact_or_none(table_name: str, column_name: str) -> str | None:
    table_names = EXACT_TABLE_ALIASES.get(table_name, [table_name])
    for name in table_names:
        by_table = EXACT_BY_TABLE.get(name, {})
        if column_name in by_table:
            return by_table[column_name]
    if column_name in COMMON_DESCRIPTIONS:
        return COMMON_DESCRIPTIONS[column_name]
    return None


def humanize_tokens(text: str) -> str:
    tokens = [t for t in text.split("_") if t]
    rendered: list[str] = []
    for token in tokens:
        rendered.append(TOKEN_LABELS.get(token, token))
    return " ".join(rendered)


def phrase_for_field(text: str) -> str:
    if text in FIELD_PHRASES:
        return FIELD_PHRASES[text]
    if text in TEACHER_FIELD_LABELS:
        return TEACHER_FIELD_LABELS[text]
    return humanize_tokens(text)


def describe_teacher_column(column_name: str) -> str | None:
    if not column_name.startswith("teacher_"):
        return None
    suffix = column_name[len("teacher_") :]
    label = TEACHER_FIELD_LABELS.get(suffix)
    if label:
        return label[:1].upper() + label[1:] + "."
    return None


def describe_boolean_pattern(column_name: str) -> str | None:
    patterns = [
        (r"^is_(.+)_missing$", "Flag de ausência de {x}."),
        (r"^is_(.+)_invalid$", "Flag de valor inválido em {x}."),
        (r"^is_(.+)_negative$", "Flag de valor negativo em {x}."),
        (r"^has_(.+)_flag$", "Flag indicando presença de {x}."),
        (r"^has_(.+)$", "Flag indicando presença de {x}."),
        (r"^only_(.+)_flag$", "Flag indicando presença apenas de {x}."),
        (r"^(.+)_eligible_flag$", "Flag indicando elegibilidade para {x}."),
        (r"^(.+)_mapped_flag$", "Flag indicando mapeamento de {x}."),
        (r"^(.+)_observed_flag$", "Flag indicando observação de {x}."),
        (r"^(.+)_analysis_eligible_flag$", "Flag indicando elegibilidade para análise de {x}."),
        (r"^(.+)_flag$", "Flag indicando {x}."),
    ]
    for pattern, template in patterns:
        match = re.match(pattern, column_name)
        if match:
            label = phrase_for_field(match.group(1))
            return template.format(x=label)
    return None


def describe_temporal_pattern(column_name: str) -> str | None:
    if column_name.endswith("_ts"):
        return f"Timestamp de {humanize_tokens(column_name[:-3])}."
    if column_name.endswith("_month"):
        return f"Mês de {humanize_tokens(column_name[:-6])}."
    if column_name.endswith("_sec"):
        return f"{humanize_tokens(column_name[:-4]).capitalize()} em segundos."
    if column_name.endswith("_min"):
        return f"{humanize_tokens(column_name[:-4]).capitalize()} em minutos."
    return None


def describe_quantity_pattern(column_name: str) -> str | None:
    patterns = [
        (r"^(.+)_avg_session_minutes_month$", "Média mensal de minutos por sessão em {x}."),
        (r"^(.+)_minutes_month$", "Quantidade mensal de minutos em {x}."),
        (r"^(.+)_count_month$", "Quantidade mensal de {x}."),
        (r"^(.+)_count$", "Quantidade de {x}."),
        (r"^(.+)_events_month$", "Quantidade mensal de {x}."),
        (r"^(.+)_events$", "Quantidade de {x}."),
        (r"^(.+)_days_month$", "Quantidade mensal de {x}."),
        (r"^(.+)_days$", "Quantidade de {x}."),
        (r"^(.+)_sessions$", "Quantidade de {x}."),
        (r"^(.+)_minutes$", "Quantidade de {x}."),
        (r"^(.+)_rows_total$", "Quantidade total de linhas de {x}."),
        (r"^(.+)_rows$", "Quantidade de linhas de {x}."),
        (r"^(.+)_months$", "Quantidade de meses de {x}."),
    ]
    for pattern, template in patterns:
        match = re.match(pattern, column_name)
        if match:
            label = phrase_for_field(match.group(1))
            return template.format(x=label)
    return None


def describe_metric_pattern(column_name: str) -> str | None:
    patterns = [
        (r"^test_gap_(.+)$", "Gap em teste de {x}."),
        (r"^gap_(.+)$", "Gap de {x}."),
        (r"^mean_(.+)$", "Média de {x}."),
        (r"^std_(.+)$", "Desvio-padrão de {x}."),
        (r"^min_(.+)$", "Valor mínimo de {x}."),
        (r"^max_(.+)$", "Valor máximo de {x}."),
        (r"^delta_(.+)$", "Diferença de {x}."),
        (r"^baseline_(.+)$", "Valor de baseline de {x}."),
        (r"^fold_mean_(.+)$", "Média entre folds de {x}."),
        (r"^(.+)_percentile$", "Percentil de {x}."),
    ]
    for pattern, template in patterns:
        match = re.match(pattern, column_name)
        if match:
            label = humanize_tokens(match.group(1))
            return template.format(x=label)
    return None


def describe_misc_pattern(column_name: str) -> str | None:
    if column_name.startswith("step_") and column_name.endswith("_token"):
        step = column_name.split("_")[1]
        return f"Passo observado na posição {step}."
    if column_name.startswith("first3_interaction_"):
        label = humanize_tokens(column_name[len("first3_interaction_") :])
        return f"{label.capitalize()} nas 3 primeiras interações observadas."
    if column_name.startswith("first7d_"):
        label = humanize_tokens(column_name[len("first7d_") :])
        return f"{label.capitalize()} nos 7 primeiros dias."
    if column_name.startswith("first_session_"):
        label = humanize_tokens(column_name[len("first_session_") :])
        return f"{label.capitalize()} da 1ª sessão."
    if column_name.startswith("future_"):
        label = humanize_tokens(column_name[len("future_") :])
        return f"{label.capitalize()} futuros após o início do label."
    if column_name.startswith("returned_") and re.search(r"_m[123]$", column_name):
        match = re.match(r"^returned_(.+)_m([123])$", column_name)
        if match:
            return f"Flag indicando retorno de {humanize_tokens(match.group(1))} no {match.group(2)}º mês."
    if column_name.startswith("used_"):
        return f"Flag indicando uso de {humanize_tokens(column_name[len('used_') :])}."
    if column_name.startswith("no_"):
        return f"Flag indicando ausência de {humanize_tokens(column_name[len('no_') :])}."
    if column_name.startswith("clean_entry_exposed_"):
        return f"Flag indicando exposição em sessão limpa de entrada com {phrase_for_field(column_name[len('clean_entry_exposed_') :])}."
    return None


def fallback_description(column_name: str) -> str:
    return f"{humanize_tokens(column_name).capitalize()}."


def describe_column(table_name: str, column_name: str) -> str:
    for resolver in (
        lambda: exact_or_none(table_name, column_name),
        lambda: describe_teacher_column(column_name),
        lambda: describe_boolean_pattern(column_name),
        lambda: describe_quantity_pattern(column_name),
        lambda: describe_temporal_pattern(column_name),
        lambda: describe_metric_pattern(column_name),
        lambda: describe_misc_pattern(column_name),
        lambda: exact_or_none("", column_name),
    ):
        value = resolver()
        if value:
            return value
    return fallback_description(column_name)


def source_table_entry(name: str, description: str, columns: list[str], table_name_for_desc: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "columns": [
            {"name": column_name, "description": describe_column(table_name_for_desc, column_name)}
            for column_name in columns
        ],
    }


def model_entry(name: str, description: str, columns: list[str], table_name_for_desc: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "columns": [
            {"name": column_name, "description": describe_column(table_name_for_desc, column_name)}
            for column_name in columns
        ],
    }


def build_sources_yaml() -> dict[str, Any]:
    modeled_con = duckdb.connect(str(MODELED_DUCKDB), read_only=True)
    build_con = duckdb.connect(str(BUILD_DUCKDB), read_only=True)
    try:
        raw_tables = []
        for table_name, spec in RAW_TABLES.items():
            raw_tables.append(
                {
                    "name": table_name,
                    "description": spec["description"],
                    "columns": [
                        {"name": col_name, "description": col_desc}
                        for col_name, col_desc in spec["columns"].items()
                    ],
                }
            )

        modeled_tables = []
        for table_name in sorted(MODELED_TABLE_DESCRIPTIONS):
            modeled_tables.append(
                source_table_entry(
                    table_name,
                    MODELED_TABLE_DESCRIPTIONS[table_name],
                    table_columns(modeled_con, table_name),
                    table_name,
                )
            )

        ml_tables = []
        for table_name in sorted(ML_TABLE_DESCRIPTIONS):
            ml_tables.append(
                source_table_entry(
                    table_name,
                    ML_TABLE_DESCRIPTIONS[table_name],
                    table_columns(build_con, table_name),
                    table_name,
                )
            )
    finally:
        modeled_con.close()
        build_con.close()

    return {
        "version": 2,
        "sources": [
            {
                "name": "raw_conceptual",
                "description": RAW_SOURCE_DESCRIPTION,
                "schema": "main",
                "tables": raw_tables,
            },
            {
                "name": "modeled_base",
                "description": MODELED_SOURCE_DESCRIPTION,
                "database": "modelled_base",
                "schema": "main",
                "tables": modeled_tables,
            },
            {
                "name": "ml_outputs",
                "description": ML_OUTPUTS_SOURCE_DESCRIPTION,
                "schema": "main",
                "tables": ml_tables,
            },
        ],
    }


def build_modeled_schema_yaml() -> dict[str, Any]:
    con = duckdb.connect(str(MODELED_DUCKDB), read_only=True)
    try:
        models = []
        for table_name, description in sorted(MODELED_TABLE_DESCRIPTIONS.items()):
            models.append(
                model_entry(
                    f"modeled_{table_name}",
                    description,
                    table_columns(con, table_name),
                    table_name,
                )
            )
        return {"version": 2, "models": models}
    finally:
        con.close()


def build_ml_inputs_schema_yaml() -> dict[str, Any]:
    con = duckdb.connect(str(BUILD_DUCKDB), read_only=True)
    try:
        input_tables = {
            "mart_onboarding_population_v1": ("ml_mart_onboarding_population", ML_TABLE_DESCRIPTIONS["mart_onboarding_population_v1"]),
            "mart_first_session_journey_v1": ("ml_mart_first_session_journey", ML_TABLE_DESCRIPTIONS["mart_first_session_journey_v1"]),
            "mart_future_metrics_v1": ("ml_mart_future_metrics", ML_TABLE_DESCRIPTIONS["mart_future_metrics_v1"]),
        }
        models = []
        for source_name, (model_name, description) in input_tables.items():
            models.append(model_entry(model_name, description, table_columns(con, source_name), source_name))
        return {"version": 2, "models": models}
    finally:
        con.close()


def build_ml_outputs_schema_yaml() -> dict[str, Any]:
    con = duckdb.connect(str(BUILD_DUCKDB), read_only=True)
    try:
        models = []
        for table_name, description in sorted(ML_TABLE_DESCRIPTIONS.items()):
            if table_name.startswith("mart_"):
                continue
            model_name = f"ml_{table_name[:-3]}" if table_name.endswith("_v1") else f"ml_{table_name}"
            models.append(model_entry(model_name, description, table_columns(con, table_name), table_name))
        return {"version": 2, "models": models}
    finally:
        con.close()


def main() -> None:
    render_yaml(build_sources_yaml(), MODELS_DIR / "sources.yml")
    render_yaml(build_modeled_schema_yaml(), MODELS_DIR / "modeled" / "schema.yml")
    render_yaml(build_ml_inputs_schema_yaml(), MODELS_DIR / "ml_inputs" / "schema.yml")
    render_yaml(build_ml_outputs_schema_yaml(), MODELS_DIR / "ml_outputs" / "schema.yml")


if __name__ == "__main__":
    main()
