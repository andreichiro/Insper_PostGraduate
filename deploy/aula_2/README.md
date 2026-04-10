# Predição de Diabetes: Pipeline ML

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

Projeto ML que prevê diabetes usando um pipeline Kedro
e serve predições via FastAPI + dashboard Streamlit

## Início 

```bash
# 1 Instalar dependências (Python 3.13+ e uv)
uv sync

# 2 Treinar o pipeline data engineering -> modelagem -> refit
uv run kedro run

# 3 Dashboard Streamlit (métricas + inferência)
uv run streamlit run src/insper_deploy_kedro/dashboard.py

# 4 Subir o server da API
uv run uvicorn insper_deploy_kedro.api:app --host 0.0.0.0 --port 8000

# 5 Abrir doc swagger
open http://localhost:8000/docs
```

### Docker (1 comando, tudo sobe)

```bash
docker compose up --build
```

Sobe dois serviços:
- **api** (porta 8000) — treina o modelo se necessário + FastAPI
- **dashboard** (porta 8501) — Streamlit com métricas e inferência ao vivo

### Docker reprodutível

O caminho de container agora foi pensado para dois usos diferentes:

- `docker compose up --build`
  imagem runtime enxuta e autocontida, com código, `pyproject.toml`,
  `uv.lock`, configs Kedro, dados raw e app Streamlit já copiados para dentro
  da imagem; no primeiro boot, o container faz seed automático dos CSVs raw para
  o volume persistente
- `docker compose --profile dev up --build workspace`
  workspace de desenvolvimento com o código montado por volume em
  `/workspace`, `uv` disponível dentro do container, dependências de
  desenvolvimento instaladas no target `dev` e o mesmo volume `app-data`
  compartilhado com API e dashboard, incluindo seed automático de `data/01_raw`
  quando o volume estiver vazio

Detalhes importantes:

- a imagem usa versões fixas de Python e `uv`
- o runtime usa `uv sync --frozen`, então as dependências seguem exatamente o
  `uv.lock`
- API e dashboard só consideram o ambiente pronto quando os 3 artefatos de
  serving existem: `production_encoders.pkl`, `production_scalers.pkl` e
  `production_model.pkl`
- a stack define `KEDRO_LOGGING_CONFIG`, então logs/tracebacks do FastAPI,
  Streamlit e jobs Kedro usam a configuração do projeto em `conf/logging.yml`
- o volume `app-data` persiste `data/` entre reinícios da stack, mas o
  bootstrap dos CSVs raw continua reprodutível mesmo com volume vazio porque a
  imagem mantém uma cópia imutável para seed
- o volume `uv-cache` acelera reinstalações no serviço `workspace`
- endpoints de status de jobs seguem a mesma proteção por `API_KEY` quando ela
  estiver configurada, e a API pública devolve só o resumo do erro; traceback
  completo continua nos logs e no store operacional

Great Expectations 1.15.x ainda importa os nomes legados `sre_constants` e
`sre_parse`. O projeto inclui um `sitecustomize.py` pequeno e documentado para
mapear esses imports para `re._constants` e `re._parser`, evitando warning spam
sem silenciar warnings globalmente nem editar dependências em `site-packages`.

O container reproduz o stack atual do repositório: Kedro, FastAPI, Streamlit,
pipelines, testes e utilitários Python. Não existe um projeto `dbt` neste
repositório hoje.

## CI

O repositório inclui um workflow de CI com:

1. `uv sync --frozen --extra dev`
2. `docker compose config`
3. `ruff check .`
4. `ruff format --check .`
5. `pytest`
6. `pip-audit`
7. `Trivy` filesystem scan
8. `docker build .`
9. `docker compose up --build` com smoke test dos endpoints da API e do Streamlit

Também existe um workflow de release para publicar a imagem no GHCR quando
houver tag `v*`. Deploy em cloud continua fora do escopo deste repositório.

## O que versionar

Estes itens devem ir para o GitHub:

- código-fonte em `src/`
- testes em `tests/`
- configurações compartilhadas em `conf/base/`
- `pyproject.toml`, `uv.lock`, `Dockerfile`, `docker-compose.yml`
- workflows de CI/release em `.github/workflows/`
- dados raw pequenos e não sensíveis necessários para reproduzir o projeto

## O que fica local

Estes itens devem permanecer locais, fora do GitHub:

- `conf/local/` e qualquer arquivo de credenciais
- `.env` e chaves como `API_KEY`
- logs, caches e ambientes virtuais
- artefatos gerados em `data/` além dos raw inputs versionados
- banco operacional local, manifests temporários e modelos produzidos durante runs

## O que vai para produção

No ambiente de produção, vale promover:

- imagem Docker construída a partir do repositório
- variáveis e segredos via secret manager ou configuração do ambiente
- volume persistente para `data/`
- artefatos de serving gerados pelo pipeline: `production_encoders.pkl`,
  `production_scalers.pkl` e `production_model.pkl`
- logs centralizados e monitoramento dos endpoints `/health` e `/_stcore/health`

## Validação de Dados (Great Expectations)

O pipeline DE roda validações automáticas com Great Expectations em 2 pontos críticos:

1. **Pós-limpeza** — verifica schema, ausência de NaN, ranges médicos (ex: Glicose 0-300, IMC 0-80)
2. **Pós-split** — verifica que cada split tem amostras suficientes e balanço de classes aceitável

Se uma validação crítica falha, o pipeline para. Warnings são logados mas não bloqueiam.

## Config declarativa (sklearn, XGBoost, CatBoost, Optuna)

Quase tudo que é “classe + hiperparâmetro” vem do YAML — mesmo padrão do `class_path` / `init_args` da modelagem:

| Arquivo | O que controla |
|---------|----------------|
| `parameters/data_engineering.yml` → `preprocessing` | `train_test_split`, `OrdinalEncoder`, `StandardScaler`, limiar de estratificação |
| `parameters/modelling.yml` → `ml_runtime` | `LabelEncoder`, `StratifiedKFold`, sampler/study do Optuna |
| `parameters/modelling.yml` → `evaluation` | funções de métrica (`sklearn.metrics.*`), matriz de confusão, derivados (r2, mape) |
| `parameters/modelling.yml` → `baseline` / `optimization` / `xgboost` | modelos e grids Optuna (já existia) |
| `parameters/refit.yml` → `calibration` | `CalibratedClassifierCV` + `init_args` |
| `parameters/data_quality.yml` | classes GX + ranges (já documentado acima) |

O código só instancia via `insper_deploy_kedro.class_loading` (`load_class` / `load_callable`).

## Estrutura do Projeto

```
├── conf/
│   ├── base/               # Config compartilhada 
│   │   ├── catalog.yml      # Data Catalog 
│   │   └── parameters/      # Parâmetros dos pipelines
│   └── local/               # Config local (gitignored)
│       └── credentials.yml  # Secrets
├── data/
│   └── 01_raw/              # CSVs de entrada
├── src/insper_deploy_kedro/
│   ├── api.py               # Camada serving FastAPI
│   ├── class_loading.py     # load_class / load_callable (YAML → objeto)
│   ├── dashboard.py         # Dashboard Streamlit (métricas + inferência)
│   ├── constants.py         # Constantes e tipos compartilhados
│   └── pipelines/
│       ├── data_engineering/ # limpar -> validar(GE) -> features -> split -> validar(GE) -> encode -> scale
│       ├── modelling/        # treinar -> avaliar -> otimizar
│       ├── inference/        # só transform -> predição
│       └── refit/            # retreinar c/ todos os dados pra produção
├── tests/                   # Testes unitários, integração e e2e
├── Dockerfile
├── docker-compose.yml       # API (8000) + Dashboard (8501)
└── pyproject.toml           # Dependências + config
```

## Dashboard Streamlit

O dashboard foi expandido para cobrir:

- comparação entre CV e validação externa
- robustez por fold
- políticas clínicas de falso positivo vs falso negativo
- risk / score report a partir do output batch
- visibilidade do último experimento e do último registro de produção
- ações operacionais para `kedro run`, `refit` e inferência batch
- predição ao vivo com score, faixa de risco e threshold de produção

```bash
# local
uv run streamlit run src/insper_deploy_kedro/dashboard.py

# via Docker (sobe junto com a API)
docker compose up --build
# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

## Novos artefatos de modelagem

Além das métricas tradicionais, o pipeline agora materializa:

- `data/07_model_output/model_frontier.parquet`
  CV score vs métricas de validação do modelo
- `data/07_model_output/cv_fold_metrics.parquet`
  métricas por fold para leitura de estabilidade
- `data/07_model_output/cv_metric_summary.parquet`
  resumo de dispersão relativa entre folds
- `data/07_model_output/threshold_metrics.parquet`
  comparação entre políticas clínicas de cutoff
- `data/07_model_output/selected_deployment_policy.pkl`
  política/threshold efetivo usado em produção
- `data/09_ops/data_contract_report.parquet`
  contrato simples de schema do dataset limpo
- `data/09_ops/data_freshness_report.parquet`
  leitura de SLA de frescor da fonte raw
- `data/09_ops/data_drift_report.parquet`
  comparação train vs validation/test com PSI por feature
- `data/09_ops/latest_experiment_run.pkl`
  resumo estruturado do último experimento selecionado
- `data/09_ops/latest_model_registry_entry.pkl`
  registro estruturado do artefato de produção atual

O artefato `production_model.pkl` também passa a carregar:

- `decision_threshold`
- `decision_policy_name`
- `policy_catalog`
- `risk_bands`

Com isso, o output da inferência pode ser tratado como relatório de risco,
não apenas como label binária.
