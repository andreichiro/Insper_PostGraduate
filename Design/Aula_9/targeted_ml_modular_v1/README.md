# Predição de usuários ativos

## Objetivo

Este projeto faz quatro coisas:

- testa diferentes definições de atividade
- estima um score de risco de não se tornar ativo
- compara cenários de informação disponíveis no começo da jornada
- materializa tabelas, relatório HTML, app e artefatos de serving

O desenho foi feito para permitir reruns com novos dados e mudanças de parâmetros via YAML, sem editar o core do pipeline.

## Comece aqui

Se alguém pegar o projeto agora e quiser se localizar rápido:

- relatório HTML:
  - [build/reports/targeted_ml_report_v1.html](/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/build/reports/targeted_ml_report_v1.html)
- motor do HTML:
  - [targeted_ml/runtime/html_report_engine.py](/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/targeted_ml/runtime/html_report_engine.py)
- pipeline principal:
  - [targeted_ml/pipelines/modelled_to_ml/runner.py](/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/targeted_ml/pipelines/modelled_to_ml/runner.py)
- configuração oficial:
  - [specs/base.yaml](/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/specs/base.yaml)
- tabelas materializadas:
  - [build/tables](/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/build/tables)
- base modelada do run:
  - [build/modelled/duckdb/base_modelada_v2.duckdb](/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/build/modelled/duckdb/base_modelada_v2.duckdb)
- app:
  - [targeted_ml/apps/streamlit_app.py](/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/targeted_ml/apps/streamlit_app.py)
- dbt docs:
  - [dbt_lineage/target/index.html](/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/dbt_lineage/target/index.html)

Atalho local para subir app + dbt docs:

- `make refresh-ui-stack`
- Streamlit: `http://localhost:8501`
- dbt docs: `http://localhost:8081`

## Estrutura do projeto

- `targeted_ml/`
  - pacote Python com config, CLI, pipelines, inferência e relatório
- `specs/`
  - specs YAML dos estudos
- `data/`
  - insumos locais do projeto
- `build/`
  - saídas materializadas do run
- `dbt_lineage/`
  - projeto dbt da linhagem `raw -> modelled -> ml`
- `tests/`
  - suíte de testes
- `scripts/`
  - atalhos operacionais locais

## Fluxo do pipeline

O caminho oficial é:

1. `raw -> modelled`
2. `modelled -> ml`
3. `ml -> html`
4. `ml -> serving`
5. `serving -> inference`

Não existe um caminho oficial direto de `raw -> ml` ignorando a separação das etapas.

## Arquivos principais

### Código

- [targeted_ml/pipelines/modelled_to_ml/runner.py](/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/targeted_ml/pipelines/modelled_to_ml/runner.py)
  - orquestra o `build`
- [targeted_ml/pipelines/modelled_to_ml/analysis_setup.py](/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/targeted_ml/pipelines/modelled_to_ml/analysis_setup.py)
  - registries, políticas, objetivos e runtime config
- [targeted_ml/pipelines/modelled_to_ml/dataset_builder.py](/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/targeted_ml/pipelines/modelled_to_ml/dataset_builder.py)
  - base analítica, label window, validadores de 90 dias e leakage audit
- [targeted_ml/pipelines/modelled_to_ml/definitions.py](/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/targeted_ml/pipelines/modelled_to_ml/definitions.py)
  - busca, lock e comparação das definições
- [targeted_ml/pipelines/modelled_to_ml/modeling.py](/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/targeted_ml/pipelines/modelled_to_ml/modeling.py)
  - folds temporais, treino, calibração e métricas
- [targeted_ml/inference/service.py](/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/targeted_ml/inference/service.py)
  - export de serving e scoring
- [targeted_ml/runtime/html_report_engine.py](/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/targeted_ml/runtime/html_report_engine.py)
  - texto, tabelas e gráficos do HTML

### Entradas

- [targeted_ml/cli.py](/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/targeted_ml/cli.py)
  - CLI principal
- [targeted_ml/__main__.py](/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/targeted_ml/__main__.py)
  - permite `python -m targeted_ml`

## Artefatos principais

### Base do run

- `build/modelled/duckdb/base_modelada_v2.duckdb`

### Tabelas principais

- `build/tables/core_definition_selection_v1.parquet`
- `build/tables/core_definition_frontier_v1.parquet`
- `build/tables/core_definition_external_validation_v1.parquet`
- `build/tables/core_scoring_scenarios_v1.parquet`
- `build/tables/core_model_frontier_v1.parquet`
- `build/tables/core_model_fold_metrics_v1.parquet`
- `build/tables/core_model_predictions_v1.parquet`
- `build/tables/post_model_threshold_metrics_v1.parquet`
- `build/tables/post_model_confusion_matrix_v1.parquet`
- `build/tables/post_model_feature_importance_v1.parquet`
- `build/tables/governance_label_registry_v1.parquet`
- `build/tables/governance_leakage_audit_v1.parquet`
- `build/tables/governance_leakage_summary_v1.parquet`

### Relatório

- `build/reports/targeted_ml_report_v1.html`

### Serving

- `build/serving/serving_manifest.json`
- `build/serving/inference_contract.json`
- `build/serving/scoring_frame_template.csv`
- `build/serving/serving_scope.json`
- `build/serving/serving_selection_candidates.parquet`
- `build/serving/exports/<export_id>/models/*.joblib`
- `build/serving/exports/<export_id>/models/*.schema.json`
- `build/serving/exports/<export_id>/models/*.feature_list.json`
- `build/serving/exports/<export_id>/models/*.manifest.json`

### Inferência

- `build/inference_runs/<timestamp>/all_scored_clients.parquet`
- `build/inference_runs/<timestamp>/high_risk_clients_top10.parquet`
- `build/inference_runs/<timestamp>/high_risk_clients_tercis.parquet`
- `build/inference_runs/<timestamp>/high_risk_clients_score_ge_0_70.parquet`
- `build/inference_runs/<timestamp>/validation_report.parquet`
- `build/inference_runs/<timestamp>/run_manifest.json`

## Tabelas mais úteis para gráfico

- `build/tables/core_model_frontier_v1.parquet`
  - comparação entre modelos
- `build/tables/core_cv_metric_folds_v1.parquet`
  - estabilidade por fold
- `build/tables/post_model_feature_importance_v1.parquet`
  - importância das variáveis
- `build/tables/post_model_threshold_metrics_v1.parquet`
  - precision, recall e F1 por cutoff
- `build/tables/post_model_confusion_matrix_v1.parquet`
  - matriz de confusão
- `build/tables/core_definition_b_feature_block_gain_summary_v1.parquet`
  - ganho incremental dos blocos da Definition B
- `build/tables/mart_first_session_journey_v1.parquet`
- `build/tables/mart_onboarding_population_v1.parquet`
- `build/tables/mart_future_metrics_v1.parquet`

Exemplo rápido:

```python
import pandas as pd

df = pd.read_parquet("build/tables/core_model_frontier_v1.parquet")
print(df.columns.tolist())
print(df.head())
```

## Comandos principais

- `python -m targeted_ml validate-spec --analysis-spec specs/base.yaml`
- `python -m targeted_ml build-modelled --analysis-spec specs/base.yaml`
- `python -m targeted_ml build-ml --analysis-spec specs/base.yaml`
- `python -m targeted_ml build-report --analysis-spec specs/base.yaml`
- `python -m targeted_ml build --analysis-spec specs/base.yaml`
- `python -m targeted_ml export-serving --analysis-spec specs/base.yaml --output-root build`
- `python -m targeted_ml score-modelled --analysis-spec specs/base.yaml --output-root build --modelled-duckdb build/modelled/duckdb/base_modelada_v2.duckdb`
- `python -m targeted_ml score-frame --analysis-spec specs/base.yaml --output-root build --scoring-frame build/tables/mart_first_session_journey_v1.parquet --latest-observed-ts 2025-02-28`
- `python -m targeted_ml score-raw --analysis-spec specs/base.yaml --dataset-root /caminho/para/dataset_root_raw --output-root build`
- `python -m targeted_ml check-compatibility --output-root build`

## Streamlit local

Instalação:

- `pip install -e .[app]`

Subir só a app:

- `streamlit run targeted_ml/apps/streamlit_app.py`

Subir app + dbt docs:

- `make refresh-ui-stack`

O que a app suporta hoje:

- `Training`
  - valida spec, mostra YAML resolvido e dispara os builds
- `Inference`
  - usa o serving exportado e permite score por raw, modelled ou scoring frame
- `Saved outputs`
  - mostra runs e artefatos salvos
- `Reports`
  - abre os HTMLs gerados

## dbt docs e lineage

Abrir o docs já gerado:

- `open dbt_lineage/target/index.html`

Regenerar:

- `cd dbt_lineage`
- `python generate_lineage_docs.py`
- `dbt docs generate --project-dir . --profiles-dir .`

Servir localmente:

- `cd dbt_lineage`
- `dbt docs serve --project-dir . --profiles-dir . --port 8081`

Atalho recomendado:

- `make refresh-ui-stack`

## Overrides importantes

Exemplos:

- `python -m targeted_ml build --analysis-spec specs/base.yaml --override label.window_days=60`
- `python -m targeted_ml build --analysis-spec specs/base.yaml --override modeling.calibration_method=isotonic`
- `python -m targeted_ml build --analysis-spec specs/base.yaml --override modeling.tuning_n_iter=4`
- `python -m targeted_ml build --analysis-spec specs/base.yaml --dataset-root /caminho/novo_drop`

Hooks centrais:

- `label.definition_b.sql_file`
- `label.definition_b.python_strategy`
- `label.definition_a.sql_file`
- `label.definition_a.python_strategy`

Contrato:

- SQL deve produzir `teacher_unique_id`, `first_month` e colunas métricas
- Python deve apontar para `arquivo.py:callable` e retornar um `DataFrame` com o mesmo contrato

Se o pacote estiver instalado via `pip`, o console script `targeted-ml` também funciona com os mesmos subcomandos.
