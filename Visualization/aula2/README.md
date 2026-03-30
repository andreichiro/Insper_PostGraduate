# Previsão de Demanda - Engenharia de Features (Reprodutível)

## Execução
```bash
cd /Users/akatsurada/Documents/INSPER/Visualization/aula2
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python pipeline_engenharia_features.py
```

## Entregáveis oficiais
- `/Users/akatsurada/Documents/INSPER/Visualization/aula2/entregaveis/entregaveis_oficiais.md`

## Documento didático (sem pré-requisito)
- `/Users/akatsurada/Documents/INSPER/Visualization/aula2/entregaveis/revisao_critica_final.md`

## Verificações técnicas
- Métricas e parâmetros finais: `/Users/akatsurada/Documents/INSPER/Visualization/aula2/entregaveis/resumo_execucao.json`
- QA de transformação: `/Users/akatsurada/Documents/INSPER/Visualization/aula2/entregaveis/tabelas/validacao_transformacoes.csv`
- Before/after das novas features: `/Users/akatsurada/Documents/INSPER/Visualization/aula2/entregaveis/tabelas/comparacao_feature_engineering_upgrade.csv`
- Overfit/underfit: `/Users/akatsurada/Documents/INSPER/Visualization/aula2/entregaveis/tabelas/diagnostico_overfit_underfit.csv`
- Ablação por grupo: `/Users/akatsurada/Documents/INSPER/Visualization/aula2/entregaveis/tabelas/ablation_impacto_grupos_features.csv`
