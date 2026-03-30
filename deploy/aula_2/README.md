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

## CI/CD — Um Comando e Pronto

O workflow do GitHub Actions roda tudo de ponta a ponta: lint, testes, treino,
quality gate, deploy da API **e** do dashboard.

```bash
# rodar tudo c/ 1 comando (API + dashboard deployados)
gh workflow run aula-2-ci.yml

# só treinar sem deployar
gh workflow run aula-2-ci.yml -f deploy=false

# rodar só o refit (sem retreinar tudo do zero)
gh workflow run aula-2-ci.yml -f pipeline=refit

# ex c/ thresholds customizados
gh workflow run aula-2-ci.yml -f min_roc_auc=0.80 -f min_f1=0.60
```

O que acontece por baixo:
1. `ruff check` + `pytest` (lint e testes)
2. Pipeline Kedro completo + quality gate (roc_auc ≥ 0.75, f1 ≥ 0.50, mape ≤ 150%)
3. Upload artefatos pro GCS
4. Deploy da **API** + **Dashboard** pro Cloud Run

Secrets necessários no GitHub:
`GCP_SA_KEY`, `GCP_PROJECT`, `GCS_BUCKET`, `API_KEY`

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

O dashboard mostra métricas comparativas dos 3 modelos, confusion matrix 
e tem uma aba de predição:

```bash
# local
uv run streamlit run src/insper_deploy_kedro/dashboard.py

# via Docker (sobe junto com a API)
docker compose up --build
# → API: http://localhost:8000
# → Dashboard: http://localhost:8501
```

Precisa ter rodado `uv run kedro run` antes (ou usar Docker, que treina automaticamente).
