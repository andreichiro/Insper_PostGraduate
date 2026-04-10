# Implementation Checklist

Este checklist separa o que foi implementado agora do que continua sendo item metodológico maior a fechar antes do próximo rerun oficial.

## Implementado nesta mudança

- [x] Separar estudo de definição e modelagem oficial:
  - o estudo de definição continua rodando com `Definition A` vs `Definition B` e validadores pós-label
  - a modelagem oficial agora usa apenas `definition_b_label`

- [x] Ajustar `build_scoring_scenarios()` para não consumir mais o vencedor de `Definition A`:
  - `definition_frontier` não entra mais na escolha do `y_true` do modelo
  - os cenários de modelagem oficiais agora são só `definition_b_label__*`

- [x] Ajustar `serving` e `reference scope` para a arquitetura separada:
  - quando o `model_frontier` tem um único `definition_group`, ele passa a ser o contexto oficial do serving
  - `definition_selection` deixa de poder reintroduzir o alvo do estudo de definição na etapa de serving/modelo

- [x] Corrigir a metadata de `serving`:
  - `selection_scope`
  - `serving_status`
  - `serving_candidate_found`
  - `available_model_groups`

- [x] Corrigir a `selection_reason` do candidato selecionado:
  - ela agora reflete o escopo real da seleção (`definition_group_matched_frontier_candidates` vs `all_pareto_frontier_candidates`)

- [x] Fazer `export_reference_models()` lidar com run não servable sem explodir o export:
  - manifest explícito
  - `reference_scope_rows = []`
  - contrato de inferência vazio, mas válido

- [x] Atualizar o HTML para incluir explicitamente:
  - protocolo metodológico declarado
  - guardrails de seleção do alvo
  - espaço de busca oficial da `Definition A`
  - threshold sensitivity
  - sensibilidade estrutural de regra composta
  - política correta de serving
  - gaps e inconsistências entre artefato vigente e search atual

- [x] Corrigir a narrativa do bloco de definição no HTML para não simplificar demais a `Definition A`

- [x] Atualizar testes de `serving` para refletir a política correta

- [x] Implementar oficialmente a busca composta da `Definition A` no pipeline:
  - screening atômico no development
  - pairwise `AND`
  - pairwise `OR`
  - combinações ponderadas com percentil empírico no treino do fold

- [x] Implementar sensibilidade estrutural da regra composta:
  - threshold sensitivity
  - `AND -> OR`
  - drop-one-literal
  - weight perturbation

- [x] Implementar auditoria formal `train vs test`:
  - comparação entre `apparent_train` e `outer test`
  - comparação entre `calibration_holdout` e `outer test`
  - gap com CI bootstrap e `statistical_gap_flag`

- [x] Explicitar e corrigir a camada de validação pós-label de 90 dias:
  - os 90 dias continuam sendo apenas validação externa da definição
  - eles não redefinem o label e não entram como feature
  - o retorno ativo deixou de depender do proxy amplo anterior `session + broad interaction event`
  - o retorno ativo agora usa um construto fixo de continuação comportamental:
    - `download`
    - `create` em famílias centrais
    - `share` em artefatos pedagógicos
    - `view` apenas de conteúdo pedagógico central

- [x] Registrar explicitamente na documentação:
  - o que os 90 dias fazem
  - o que `gap` significa
  - por que os validadores de 90 dias não são nem `Definition A` repetida nem `Definition B`
  - e como isso fica separado do alvo de 30 dias usado pelo modelo

## Ainda pendente antes do próximo rerun oficial

- [ ] Implementar auditoria explícita de `definition evaluability/modelability`

- [ ] Fechar tiers oficiais do universo de métricas da `Definition A`

- [ ] Fechar política auditável de thresholds contínuos:
  - cutoff bruto
  - posição empírica no treino
  - apresentação arredondada/faixa

- [ ] Resolver a inconsistência entre:
  - build materializado atual
  - e o search composto/ponderado agora implementado no código

- [ ] Rerodar `build-ml` para materializar:
  - `core_model_generalization_folds_v1`
  - `core_model_generalization_summary_v1`
  - nova `core_definition_selection_v1`
  - novo `core_definition_lock_summary_v1`
