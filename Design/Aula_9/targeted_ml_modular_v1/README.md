# Predição de usuários ativos

## Objetivo

- Testar diferentes definições de atividade
- Testar o quão informativa são as informações disponíveis olhando apenas a atividade da 1a sessão, apenas a atividade de 7 dias após a 1a sessão, e combinando 1a sessão com 1a semana de atividade
- Criar um score de risco baseado nos usuários com maior risco de não se tornarem ativos
- Identificar e interpretar possíveis riscos
- Permitir modularidade: rerun com novos dados, override de definições centrais, e mudanças de parâmetros, tudo sem editar o core, e separando bem o que é 'data-driven' e o que são decisões de negócio que tomamos

## Comece aqui

Este README agora é o ponto de entrada principal do projeto.

Se alguém pegar este projeto para:

- mexer no HTML
- achar os dados rapidamente
- criar gráficos novos
- entender onde está o código principal

comece por esta seção.

Atalho rápido:

- mudar o HTML:
  - [targeted_ml/runtime/html_report_engine.py](targeted_ml/runtime/html_report_engine.py)
- pegar tabelas para gráfico:
  - [build/tables](build/tables)
- abrir a base modelada do run:
  - [build/modelled/duckdb/base_modelada_v2.duckdb](build/modelled/duckdb/base_modelada_v2.duckdb)
- ver o relatório final:
  - [build/reports/targeted_ml_report_v1.html](build/reports/targeted_ml_report_v1.html)
- abrir a app:
  - [targeted_ml/apps/streamlit_app.py](targeted_ml/apps/streamlit_app.py)
- abrir a linhagem dbt:
  - [dbt_lineage/target/index.html](dbt_lineage/target/index.html)
- regenerar e subir app + dbt docs:
  - `make refresh-ui-stack`
  - portas fixas:
    - Streamlit em `http://localhost:8501`
    - dbt docs em `http://localhost:8081`

Se preferir um atalho curtíssimo, [QUICK_REFERENCE.md](QUICK_REFERENCE.md) agora só redireciona para este README.

Convenção da app:

- a aba inicial padrão é `Relatórios`
- o estudo default é `Atividade (principal)`
- o editor mostra a configuração efetiva resolvida do spec, não o YAML cru que só herda de `base.yaml`

## Estrutura

- `targeted_ml/`
  Pacote Python com config, CLI, orquestração, pipeline que carrega os dados modelados e aplica todos os objetivos, `modelled -> ml` com renderização de um report em html.
- `specs/`
  Specs configuráveis em YAML para atividade, churn e retorno.
- `data/`
  Dados locais do projeto. No pacote mínimo de entrega, a base modelada histórica em `data/modelled/` pode ser removida; o caminho canônico dos runs é `output_root/modelled/duckdb/base_modelada_v2.duckdb`.
  Dados brutos `raw` não devem acompanhar a entrega final.
  Se entrar uma nova massa de dados, o `dataset_root` pode ser sobrescrito via spec ou CLI.
- `build/`
  Saídas prontas para entrega:
  - `build/modelled/`: base modelada materializada pelo próprio pipeline quando `build-modelled` roda
  - `build/tables/`: tabelas materializadas
  - `build/metadata/`: metadata do run, manifests e checagens de compatibilidade
  - `build/reports/`: relatório HTML final
  - `build/serving/`: artefatos servíveis do modelo escolhido
  - `build/inference_runs/`: exemplos de inferência materializados
- `dbt_lineage/`
  Projeto `dbt Core` usado para documentar a linhagem `raw -> modeled -> ml` e descrever as tabelas novas mais importantes para a entrega.

## Estrutura mínima para entender o projeto

Se alguém quiser uma visão simples, pense só nestes blocos:

1. `targeted_ml/`
   código principal
2. `specs/`
   configuração do estudo
3. `build/modelled/`
   base modelada do run
4. `build/tables/`
   tabelas para análise, HTML e gráficos
5. `build/reports/`
   relatório final
6. `build/serving/`
   modelo salvo e contrato de inferência
7. `build/inference_runs/delivery_modelled_inference/`
   run de entrega com base inteira rankeada, filas filtradas por cutoff e relatório de validação
8. `build/inference_runs/`
   runs de inferência salvos; no pacote mínimo, enviar só se quiser incluir um exemplo pronto
9. `dbt_lineage/models/`
   source da linhagem
10. `dbt_lineage/target/index.html`
   docs da linhagem prontas para abrir

### Mapa rápido de código

- `targeted_ml/runtime/html_report_engine.py`
  arquivo principal para narrativa, tabelas e gráficos do HTML
- `targeted_ml/reporting/render.py`
  wrapper que chama o engine e grava o relatório final
- `targeted_ml/data/raw_to_modelled.py`
  transformação `raw -> modelled`
- `targeted_ml/pipelines/modelled_to_ml/runner.py`
  orquestração principal do `modelled -> ml`
- `targeted_ml/inference/service.py`
  exportação do modelo salvo, contrato de inferência e scoring
- `targeted_ml/apps/streamlit_app.py`
  app para treino, inferência, relatórios e saídas salvas

### Mapa rápido de dados

- `data/raw/base_aprendizap/`
  raw local do projeto
- `build/modelled/duckdb/base_modelada_v2.duckdb`
  base modelada do run
- `build/tables/`
  tabelas prontas para análise, gráficos e HTML
- `build/serving/`
  artefatos do modelo salvo
- `build/inference_runs/`
  exemplos locais de inferência materializados; enviar só se a entrega pedir um batch scoreado ou um exemplo pronto
  quando incluído na entrega, os artefatos principais são:
  - `all_scored_clients.parquet`: base inteira elegível, rankeada por `risk_score`, com flags por cutoff
  - `high_risk_clients_top10.parquet`: fila filtrada pela política top 10%
  - `high_risk_clients_tercis.parquet`: fila filtrada pela política de tercis
  - `high_risk_clients_score_ge_0_70.parquet`: fila filtrada pela política `risk_score >= 0,70`

## Gráficos: caminho mais curto

As tabelas mais úteis costumam ser estas:

- `build/tables/core_model_frontier_v1.parquet`
  comparação entre modelos
  use `x = model_name` e `y = mean_ap`, `mean_roc_auc`, `mean_brier` ou `mean_log_loss`
- `build/tables/core_cv_metric_folds_v1.parquet`
  estabilidade das métricas por fold
  use `x = fold_id` e `y = metric_value`
- `build/tables/post_model_feature_importance_v1.parquet`
  importância das variáveis
  use `x = importance_mean` e `y = feature_name`
- `build/tables/post_model_threshold_metrics_v1.parquet`
  precision / recall / F1 por política de cutoff
  use `x = policy_name` e `y = precision`, `recall` ou `f1`
- `build/tables/post_model_confusion_matrix_v1.parquet`
  matriz de confusão
  use `x = predicted_group`, `y = actual_group` e `value = rows`
- `build/tables/core_definition_b_feature_block_gain_summary_v1.parquet`
  ganho incremental dos blocos da Definition B
  use `x = block_name` e `y = delta_ap_vs_context` ou `delta_roc_auc_vs_context`

Para gráficos de comportamento e label, as tabelas-base são:

- `build/tables/mart_first_session_journey_v1.parquet`
- `build/tables/mart_onboarding_population_v1.parquet`
- `build/tables/mart_future_metrics_v1.parquet`

Se a pessoa quiser só abrir uma tabela rápido em Python:

```python
import pandas as pd

df = pd.read_parquet("build/tables/core_model_frontier_v1.parquet")
print(df.columns.tolist())
print(df.head())
```

## Regra de dados e execução

- a entrega final inclui a base modelada oficial
- a entrega final não inclui dados `raw`
- o caminho oficial continua separado em:
  - `raw -> modelled`
  - `modelled -> ml`
  - `ml -> html`
- não existe um script oficial direto de `raw -> ml`
- a linhagem entre etapas deve permanecer explícita no código, no `dbt` e na documentação

## O que entra na entrega final

Na entrega final, o `dbt` deve participar explicitamente.

### Enviar na entrega

Enviar:

- `targeted_ml/`
  código-fonte do produto
- `specs/`
  configuração oficial dos estudos
- `README.md`
  documentação de uso e estrutura
- `pyproject.toml`, `Dockerfile`, `Makefile`
  empacotamento e reprodução do ambiente
- `build/tables/`
  tabelas finais materializadas do pipeline
- `build/metadata/`
  manifests, resumo do build e metadata de governança
- `build/modelled/duckdb/base_modelada_v2.duckdb`
  base modelada oficial do run quando o pacote reconstrói `raw -> modelled`
- `build/reports/`
  relatório HTML final
- `build/serving/`
  artefatos de produção do score:
  - `serving_manifest.json`
  - `serving_scope.json`
  - `serving_selection_candidates.parquet`
  - `models/*.joblib`
  - `models/*.schema.json`
  - `models/*.feature_list.json`
  - `models/*.manifest.json`
- `dbt_lineage/models/`
  projeto dbt com a modelagem declarativa da linhagem
- `dbt_lineage/dbt_project.yml` e `dbt_lineage/profiles.yml`
  configuração do projeto dbt
- `dbt_lineage/target/index.html`
  dbt docs navegável
- `dbt_lineage/target/manifest.json`
  linhagem completa em JSON
- `dbt_lineage/target/catalog.json`
  catálogo e descrições das tabelas
Em outras palavras:
- `dbt_lineage/` faz parte da entrega
- principalmente `models/` e `target/` com os docs finais

## O que é local e não deve ir para a entrega

### Não enviar

Esses itens podem existir localmente durante execução e desenvolvimento, mas não devem compor o pacote final de entrega:

- `build/staging/`
  arquivos temporários de execução e retomada
- `build/duckdb/build.duckdb`
  banco DuckDB local de trabalho do pipeline `modelled -> ml`
- `build/duckdb/build.duckdb.wal`
  write-ahead log transitório do DuckDB
- `build/logs/`
  logs locais de execução manual do build
- `build/inference_runs/`
  exemplos locais de inferência; só enviar se a entrega pedir explicitamente um batch já scoreado ou um exemplo pronto
- `dbt_lineage/logs/`
  logs locais do dbt
- `dbt_lineage/target/compiled/`
  SQL compilado auxiliar do dbt
- `dbt_lineage/target/partial_parse.msgpack`
  cache interno do dbt
- `dbt_lineage/target/perf_info.json`
  telemetria/performance de execução
- `data/modelled/`
  base modelada histórica/local; no pacote mínimo de entrega ela pode ser removida porque `build/modelled/` já concentra a base modelada oficial do run
- `data/source_v2/`
  área local de insumo bruto; não deve ir para a entrega final

Recomendação:
- esses itens podem permanecer localmente no ambiente de trabalho
- mas devem ser removidos do pacote final de entrega

## Manutenção da estrutura

Para manter a árvore limpa:

- `make clean-local`
  remove caches, `__pycache__`, logs e artefatos auxiliares do dbt
- `make trim-delivery-view`
  mantém só os exemplos de inferência principais, o export de serving mais recente e os arquivos essenciais do dbt docs

## UI local rápida

- `make refresh-ui-stack`
  comando oficial para uso local
  regenera o `dbt docs`, sobe o servidor do `dbt docs` em `http://localhost:8081` e sobe a app Streamlit em `http://localhost:8501`
  o HTML final já aponta para esses mesmos links, então as portas não devem ser trocadas

## Nomes mais claros para entrega vs local

Hoje a estrutura final do projeto está assim:

- `build/` = pasta principal do run final
- `build/tables/`
  tabelas finais entregáveis
- `build/metadata/`
  metadata final entregável
- `build/reports/`
  HTML final entregável
- `build/serving/`
  artefatos servíveis e contrato formal de inferência
- `build/staging/`
  staging local de execução e retomada, nunca parte da entrega
- `build/duckdb/`
  banco DuckDB local de trabalho do pipeline, nunca parte da entrega
- `build/logs/`
  logs locais de execução manual
- `data/`
  manter apenas o que for necessário como insumo bruto; a base modelada oficial do run fica em `build/modelled/`
- `dbt_lineage/`
  manter na entrega quando a linhagem fizer parte do pacote final

Regra prática:
- tudo que é produto final deve ter nome de entrega
- tudo que é cache, staging, log ou insumo local deve ter nome de trabalho local

## Caminho principal do `modelled -> ml`

Esses são os arquivos do pipeline:

- `targeted_ml/pipelines/modelled_to_ml/runner.py`
  orquestração do build, em ordem de leitura
- `targeted_ml/pipelines/modelled_to_ml/analysis_setup.py`
  spec resolvida, registries, políticas e definições centrais
- `targeted_ml/pipelines/modelled_to_ml/dataset_builder.py`
  construção das bases analíticas, elegibilidade de features e leakage audit
- `targeted_ml/pipelines/modelled_to_ml/modeling.py`
  folds temporais, pré-processamento, calibração, treino e avaliação
- `targeted_ml/pipelines/modelled_to_ml/post_model_outputs.py`
  threshold, confusion matrix, bands, feature importance, navegação, heavy-user, cluster e robustez
- `targeted_ml/pipelines/modelled_to_ml/storage.py`
  persistência, staging incremental e manifests por tarefa
- `targeted_ml/inference/service.py`
  exportação do modelo final, contrato de inferência e scoring oficial de novos dados modelados

## Arquivos `.py` principais

Os arquivos `.py` da raiz são wrappers finos de entrada:

- `raw_to_modelled_build.py`
  entrada para a etapa `raw -> modelled`
- `modelled_to_ml_build.py`
  entrada para a etapa `modelled -> ml`
- `ml_to_html_build.py`
  entrada para a etapa `ml -> html`

Os arquivos `.py` principais do pacote são:

- `targeted_ml/__main__.py`
  permite rodar `python -m targeted_ml`
- `targeted_ml/cli.py`
  CLI oficial com os subcomandos `build`, `build-ml`, `build-report` etc.
- `targeted_ml/entrypoints.py`
  auxiliares para expor os entrypoints instaláveis do pacote

Em termos práticos:
- os `.py` são o código-fonte e a forma de execução
- os artefatos finais saem em `build/`

## Saídas oficiais do `modelled -> ml`

As saídas oficiais dessa etapa são materializadas em `build/tables/`.

### 1. Definição oficial do alvo

- `core_definition_selection_v1.parquet`
  seleção oficial da `Definition A` e manutenção da `Definition B` como comparadora
- `core_definition_frontier_v1.parquet`
  comparação final entre as definições oficiais
- `core_definition_external_validation_v1.parquet`
  folds e validações externas das definições
- `core_scoring_scenarios_v1.parquet`
  cenários de score que vão para a modelagem

### 2. Resultado oficial dos modelos

- `core_model_fold_metrics_v1.parquet`
- `core_model_predictions_v1.parquet`
- `core_model_frontier_v1.parquet`
- `core_model_calibration_audit_v1.parquet`

### 3. Robustez do score

- `core_cv_score_folds_v1.parquet`
- `core_cv_score_summary_v1.parquet`
- `core_cv_metric_folds_v1.parquet`
- `core_cv_metric_summary_v1.parquet`
- `core_prediction_bootstrap_v1.parquet`

### 4. Comparadores e checagens da `Definition B`

- `core_definition_b_feature_block_gain_folds_v1.parquet`
- `core_definition_b_feature_block_gain_summary_v1.parquet`
- `core_definition_b_excessive_separation_v1.parquet`

### 5. Uso do score no produto final

- `post_model_reference_selection_v1.parquet`
- `post_model_threshold_metrics_v1.parquet`
- `post_model_confusion_matrix_v1.parquet`
- `post_model_band_summary_v1.parquet`
- `post_model_monthly_fit_v1.parquet`

### 6. Robustez operacional

- `post_model_cv_threshold_folds_v1.parquet`
- `post_model_cv_threshold_summary_v1.parquet`
- `post_model_cv_confusion_folds_v1.parquet`
- `post_model_cv_confusion_summary_v1.parquet`

### 7. Leitura complementar

- `post_model_feature_importance_v1.parquet`
- `post_model_cluster_assignment_v1.parquet`
- `post_model_cluster_profile_v1.parquet`
- `post_model_cluster_summary_v1.parquet`
- `post_model_cluster_validation_v1.parquet`
- `post_model_heavy_user_scores_v1.parquet`
- `post_model_heavy_user_profile_v1.parquet`
- `post_model_heavy_user_summary_v1.parquet`
- `core_navigation_sequences_v1.parquet`
- `core_navigation_transitions_v1.parquet`

### 8. Governança

- `governance_label_registry_v1.parquet`
- `governance_leakage_audit_v1.parquet`
- `governance_leakage_summary_v1.parquet`
- `governance_post_model_output_status_v1.parquet`

### 9. Resumo do run

- `build/metadata/build_summary_v1.json`

Essas tabelas alimentam o relatório HTML final em `build/reports/targeted_ml_report_v1.html`.

## Artefatos de serving e inferência

- `build/serving/serving_manifest.json`
  manifesto principal do serving mais recente, com `spec_hash`, `git_revision`, `export_id`, regra de seleção e artefatos exportados
- `build/serving/latest.json`
  ponteiro para o último export materializado
- `build/serving/inference_contract.json`
  contrato consolidado de inferência, com input aceito, colunas obrigatórias e modelos exportados
- `build/serving/scoring_frame_template.csv`
  template mínimo para inferência por `scoring_frame_file`
- `build/serving/serving_scope.json`
  escopo final exportado para serving
- `build/serving/serving_selection_candidates.parquet`
  todos os candidatos considerados na escolha do modelo primário
- `build/serving/exports/<export_id>/`
  export versionado do serving, preservando artefatos anteriores
- `build/serving/exports/<export_id>/models/*.joblib`
  pipeline calibrado pronto para `predict_proba`
- `build/serving/exports/<export_id>/models/*.schema.json`
  contrato explícito de entrada para inferência
- `build/serving/exports/<export_id>/models/*.feature_list.json`
  lista final de features usadas
- `build/serving/exports/<export_id>/models/*.manifest.json`
  metadata detalhada do modelo exportado
- `build/inference_runs/latest.json`
  ponteiro para o último batch inferido
- `build/inference_runs/<timestamp>/scores_all_models.parquet`
  score completo do batch inferido
- `build/inference_runs/<timestamp>/high_risk_users.parquet`
  usuários ordenados por maior risco
- `build/inference_runs/<timestamp>/validation_report.parquet`
  validação do contrato de inferência
- `build/inference_runs/<timestamp>/run_manifest.json`
  manifesto do run de inferência

Status atual importante:
- o raw oficial pode permanecer apenas no ambiente local em `data/raw/base_aprendizap`; não deve acompanhar a entrega final
- com `data.modeled_source=auto`, o comportamento padrão agora é preferir `raw -> modelled` quando o raw estiver disponível no `dataset_root`
- o fallback para seed modelada histórica fica restrito aos casos em que o raw não estiver presente
- para forçar explicitamente rebuild a partir do raw em qualquer contexto, use `data.modeled_source=raw` ou o comando `score-raw`
- o serving fecha inferência para `raw dataset_root`, `modelled_duckdb` compatível e `scoring_frame_file` (`csv` ou `parquet`) com schema compatível
- o build pesado antigo continua histórico; se o serving for exportado sem rerodar o ML pesado, os artefatos autoritativos passam a ser os de `build/serving/`
- a seleção do modelo primário de serving é automática e usa métricas probabilísticas, calibração, variabilidade operacional, variabilidade da confusion matrix e, por último, informação disponível (`score_window_end_day`)

Regra oficial atual da `Definition A`:
- a spec oficial usa `univariate_exact`
- não usa `preferred_rule_text`
- não usa busca combinatória como caminho oficial

## Comandos

- `python -m targeted_ml validate-spec --analysis-spec specs/activity.yaml`
- `python -m targeted_ml build-modelled --analysis-spec specs/activity.yaml`
- `python -m targeted_ml build-ml --analysis-spec specs/activity.yaml`
- `python -m targeted_ml build-ml --analysis-spec specs/activity.yaml --skip-post-model-refit`
- `python -m targeted_ml build-report --analysis-spec specs/activity.yaml`
- `python -m targeted_ml build --analysis-spec specs/activity.yaml`
- `python -m targeted_ml export-serving --analysis-spec specs/activity.yaml --output-root build`
- `python -m targeted_ml score-modelled --analysis-spec specs/activity.yaml --output-root build --modelled-duckdb build/modelled/duckdb/base_modelada_v2.duckdb`
- `python -m targeted_ml score-frame --analysis-spec specs/activity.yaml --output-root build --scoring-frame build/tables/mart_first_session_journey_v1.parquet --latest-observed-ts 2025-02-28`
- `python -m targeted_ml score-raw --analysis-spec specs/activity.yaml --dataset-root /caminho/para/dataset_root_raw --output-root build`
- `python -m targeted_ml check-compatibility --output-root build`
- `streamlit run targeted_ml/apps/streamlit_app.py`

## Streamlit local

Instalação:

- `pip install -e .[app]`

Subir a app sozinha:

- `streamlit run targeted_ml/apps/streamlit_app.py`

Subir app + dbt docs juntos:

- `make refresh-ui-stack`
- esse é o caminho recomendado para uso normal, porque mantém os links do HTML consistentes com a app e com o docs
- portas fixas:
  - Streamlit: `http://localhost:8501`
  - dbt docs: `http://localhost:8081`

O que a app já suporta:

- aba `Training`
  valida spec, mostra YAML editável, diff e dispara `build-modelled`, `build-ml`, `build-report`, `build`, `export-serving`
- aba `Inference`
  usa o serving exportado e permite inferência por `raw dataset_root`, `modelled_duckdb` ou `scoring_frame_file`
- aba `Saved outputs`
  mostra runs já materializados e a tabela de `high risk`
- aba `Reports`
  aponta para os HTMLs gerados sem rerodar o build pesado

## dbt docs e lineage

Abrir o docs já gerado:

- `open dbt_lineage/target/index.html`

Regenerar os docs a partir do build atual:

- `cd dbt_lineage`
- `python generate_lineage_docs.py`
- `dbt docs generate --project-dir . --profiles-dir .`

Servir localmente:

- `cd dbt_lineage`
- `dbt docs serve --project-dir . --profiles-dir . --port 8081`

Recomendação prática:

- para uso integrado com o HTML e com a app, prefira `make refresh-ui-stack`
- esse comando regenera o catálogo dbt e serve a pasta `dbt_lineage/target/` na porta fixa `8081`

Observação:
- o `profiles.yml` deste projeto aponta para `../build/duckdb/build.duckdb`, que é o DuckDB materializado pelo build atual
- o `profiles.yml` aceita override via `TARGETED_ML_BUILD_DUCKDB` e `TARGETED_ML_MODELLED_DUCKDB` quando a linhagem precisar apontar para outro build/modelled
- `generate_lineage_docs.py` regenera as descrições completas de raw, modeled e ML antes do `dbt docs generate`
- `make dbt-parse`
- `make dbt-docs`

Exemplos de override:
- `python -m targeted_ml build --analysis-spec specs/activity.yaml --override label.window_days=60`
- `python -m targeted_ml build --analysis-spec specs/churn_m1.yaml --override modeling.calibration_method=isotonic`
- `python -m targeted_ml build --analysis-spec specs/activity.yaml --override modeling.tuning_n_iter=4`
- `python -m targeted_ml build --analysis-spec specs/activity.yaml --dataset-root /caminho/novo_drop`

Overrides centrais por estratégia:
- `label.definition_b.sql_file`
- `label.definition_b.python_strategy`
- `label.definition_a.sql_file`
- `label.definition_a.python_strategy`

Esses hooks são controlados:
- SQL deve produzir `teacher_unique_id`, `first_month` e uma ou mais colunas métricas.
- Python deve apontar para `arquivo.py:callable` e retornar um `DataFrame` com o mesmo contrato.

Se o pacote estiver instalado via `pip`, o console script `targeted-ml` também funciona com os mesmos subcomandos.
