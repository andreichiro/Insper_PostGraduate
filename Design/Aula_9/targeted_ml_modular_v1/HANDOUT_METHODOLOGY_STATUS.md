# Targeted ML Handout

Este documento registra, em um só lugar, o que o código faz hoje, o que já foi corrigido, quais são as restrições metodológicas não negociáveis, quais achados empíricos já existem nos artefatos, e o que ainda está em aberto.

Ele deve ser lido como documento de trabalho técnico. Quando houver conflito entre conversa e código, vale o que está implementado e materializado nos artefatos.

Importante: depois da remoção do diretório `build_same_month_entry_rigorous`, o conjunto de artefatos materializados vigente voltou a ser o de [build/tables](/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/build/tables). Então, quando este documento falar em "estado materializado atual", ele se refere a `build/tables`, não ao rerun rigoroso deletado.

## 1. Objetivo do projeto

O pipeline constrói um problema de predição binária a partir de dados longitudinais de uso do produto, mas a matriz final de modelagem é por professor:

- `1 linha por teacher_unique_id`
- âncora temporal no onboarding observado
- features construídas antes do momento do score
- label medido em janela futura fixa
- validação temporal por mês

O objetivo final do build é:

- definir um alvo oficial de atividade/inatividade sem leakage de modelo
- comparar trilhas (`S1`, `S7`, `S1+S7`, `STRICT_CONTEXT`)
- comparar famílias de modelo
- calibrar score
- produzir `score`, `risk_score`, cutoffs e artefatos de entrega

## 2. Restrições metodológicas não negociáveis

Essas restrições foram assumidas como obrigatórias e precisam continuar valendo:

1. A escolha da `Definition A` não pode usar AP, ROC AUC, Brier, confusion matrix, feature importance nem qualquer outra métrica de modelo.
2. O modelo não pode influenciar a definição do que é "ativo".
3. Não pode haver leakage temporal entre features e label.
4. Não pode haver vazamento entre `train`, `inner validation`, `calibration holdout` e `outer test`.
5. A seleção da definição e a avaliação oficial do modelo não podem usar o mesmo período temporal.
6. Sensibilidade da definição e sensibilidade do modelo são coisas diferentes e não podem ser misturadas.
7. Se uma definição for semanticamente interessante, mas não tiver suporte suficiente para avaliação robusta, isso precisa aparecer explicitamente.

## 3. População oficial atual

O código hoje usa como população oficial:

- `official_population: same_month_entry_only`

Isso significa:

- entram apenas casos com `months_after_entry == 0`
- os casos `months_after_entry > 0` não entram na população oficial do loop principal
- eles aparecem na camada de sensibilidade populacional

Arquivos relevantes:

- [/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/specs/base.yaml](/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/specs/base.yaml)
- [/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/targeted_ml/pipelines/modelled_to_ml/analysis_setup.py](/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/targeted_ml/pipelines/modelled_to_ml/analysis_setup.py)

## 4. Janelas temporais do estudo

No código atual, os papéis temporais estão separados assim:

- janela de feature do modelo:
  - `S1`: até o fim da 1ª sessão
  - `S7`, `S1+S7`, `STRICT_CONTEXT`: até o fim dos 7 primeiros dias
- janela de definição/benchmark de 30 dias:
  - `day 8 -> day 37`
- janela de validação pós-label de 90 dias:
  - três blocos consecutivos de 30 dias depois do `day 37`

Em termos simples, hoje o build trabalha com a seguinte lógica por professor:

1. usar a janela `day 8 -> day 37` para dizer quem foi marcado como ativo ou não ativo por uma definição
2. congelar esse rótulo
3. olhar os 90 dias seguintes só para perguntar se esse grupo continuou mais ativo depois

Isso é importante:

- o modelo oficial tenta prever o alvo fixo de 30 dias `definition_b_label`
- o estudo de definição usa os 90 dias apenas como validação externa de continuidade
- os validadores pós-label não entram como feature e não são, por si só, leakage de modelo

### 4.1 O que os 90 dias fazem

Os 90 dias não redefinem o label. Eles não "viram o positivo" depois.

Eles servem apenas para responder:

- "essa definição de 30 dias separou um grupo que continuou engajando mais depois?"

O pipeline mede isso em 3 blocos:

- bloco 1: `day 38 -> day 67`
- bloco 2: `day 68 -> day 97`
- bloco 3: `day 98 -> day 127`

E constrói 5 validadores fixos:

- `returned_active_post_label_m1`
- `returned_active_post_label_m2`
- `returned_active_post_label_m3`
- `active_days_post_label_3m`
- `sustained_active_2of3_post_label`

Interpretação:

- `m1`, `m2`, `m3`: houve pelo menos um evento de continuação relevante no bloco?
- `active_days_post_label_3m`: em quantos dias distintos houve continuação relevante ao longo dos 90 dias?
- `sustained_active_2of3_post_label`: o professor apareceu como ativo em pelo menos 2 dos 3 blocos?

### 4.2 Qual é o construto fixo de continuação nos 90 dias

O build agora não usa mais o proxy amplo anterior de “sessão + evento genérico de interação”
como proxy de retorno ativo.

Em vez disso, ele usa um construto fixo de continuação comportamental, comum a todas as definições:

- `download`
- `create` para famílias como `plano`, `prova`, `relatorio`
- `share` para famílias como `aula`, `plano`, `prova`
- `view` apenas para conteúdo pedagógico central:
  - `aula`
  - `plano`
  - `prova`
  - `metodologia`

Isso é deliberadamente:

- diferente da `Definition A`
- diferente da `Definition B`
- comum a todas as candidatas

Ou seja, os 90 dias não perguntam:

- "o professor satisfez a mesma regra A de novo?"
- nem:
- "o professor satisfez B de novo?"

Eles perguntam:

- "o professor continuou mostrando engajamento relevante depois?"

### 4.3 O que significa "gap" nos 90 dias

Para cada candidata, o código primeiro separa a base em 2 grupos na janela `day 8 -> day 37`:

- `label = 1`
- `label = 0`

Aqui, isso quer dizer exatamente:

- grupo positivo: professores que a regra candidata marcou como ativos na janela inicial de 30 dias
- grupo negativo: professores que a regra candidata marcou como não ativos nessa mesma janela inicial

Importante:

- "grupo negativo" não significa "professor que continuou inativo depois"
- significa apenas "professor que não satisfez a regra da candidata no primeiro window de 30 dias"
- depois disso, esse professor ainda pode voltar a aparecer como ativo nos 90 dias seguintes

Depois, para cada validador pós-label, ele calcula:

- média do validador entre os marcados como `1`
- menos
- média do validador entre os marcados como `0`

Exemplos:

- se `returned_active_post_label_m1` for binário, a média é a taxa de retorno no bloco 1
- se `active_days_post_label_3m` for contagem, a média é o número médio de dias ativos nos 90 dias

Então o `gap` mede separação futura:

- gap grande e positivo: a definição separou um grupo que continuou mais
- gap perto de zero: a definição quase não separou continuidade futura
- gap negativo: sinal ruim; o grupo marcado como ativo não continuou mais do que o outro

Exemplo concreto:

- suponha que, em um mês de teste, uma candidata marque 100 professores como positivos e 400 como negativos
- depois olhamos o validador `sustained_active_2of3_post_label`
- se 46% do grupo positivo sustentou atividade em pelo menos 2 dos 3 blocos
- e 19% do grupo negativo sustentou atividade em pelo menos 2 dos 3 blocos

então:

- o grupo negativo não "permaneceu negativo"
- ele simplesmente era o grupo que a regra não marcou como ativo na janela inicial
- e, mesmo assim, 19% dele mostrou continuidade posterior
- o `gap` desse fold fica `0.46 - 0.19 = 0.27`

Arquivos relevantes:

- [/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/targeted_ml/pipelines/modelled_to_ml/dataset_builder.py](/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/targeted_ml/pipelines/modelled_to_ml/dataset_builder.py)
- [/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/targeted_ml/pipelines/modelled_to_ml/analysis_setup.py](/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/targeted_ml/pipelines/modelled_to_ml/analysis_setup.py)

### 4.4 Como o bootstrap do gap é usado hoje

No código atual, o bootstrap da definição não reamostra professores crus.

Ele reamostra uma lista de valores já agregados por fold.

O fluxo é:

1. escolher uma única candidata
2. olhar todos os outer folds válidos dessa mesma candidata
3. em cada fold, calcular um único número de gap
4. empilhar esses números em um vetor
5. aplicar bootstrap sobre esse vetor para estimar quão estável é a média do gap entre folds

Exemplo:

- fold 1: gap de sustentação = `0.18`
- fold 2: gap de sustentação = `0.22`
- fold 3: gap de sustentação = `0.11`
- fold 4: gap de sustentação = `0.27`
- fold 5: gap de sustentação = `0.20`

O vetor bootstrapado é:

- `[0.18, 0.22, 0.11, 0.27, 0.20]`

Esse vetor não é:

- um valor por professor
- nem um valor por evento

Ele é:

- um valor por fold válido da mesma candidata

Depois, o código faz:

1. sortear com reposição um novo vetor do mesmo tamanho
2. calcular a média desse novo vetor
3. repetir isso `200` vezes
4. pegar os percentis `2.5%` e `97.5%`
5. definir:
   - `ci_low`
   - `ci_high`
   - `ci_width = ci_high - ci_low`

Leitura:

- no padrão atual, `ci_low > 0` = mesmo depois de incorporar incerteza, a candidata ainda separa os grupos na direção correta no validador principal
- `ci_width` pequena = a candidata tem separação mais estável entre folds
- `ci_width` grande = a candidata depende mais de quais meses entraram no cálculo

Uso oficial no código atual:

- primeiro, a candidata precisa passar no suporte mínimo por fold (`rows`, `positives`, `negatives`)
- depois, no `definition lock`, ela só pode seguir viva se passar pela regra configurada em `modeling.definition_lock_bootstrap_gate`
- no padrão atual dessa spec, a regra é `lock_gap_sustained_active_2of3_post_label_ci_low > 0`
- entre as candidatas que continuam vivas, `ci_width` entra como critério de estabilidade: menor largura é melhor

## 5. Split temporal atual do workflow

O pipeline foi reorganizado para separar:

1. `definition_selection_development`
2. `definition_lock_holdout`
3. `official_model_evaluation_holdout`

Nos artefatos do rerun rigoroso, isso está assim:

| Papel | Início | Fim | Meses |
|---|---|---:|---:|
| development | 2022-01 | 2024-10 | 34 |
| definition lock | 2024-11 | 2025-04 | 6 |
| final model evaluation | 2025-05 | 2025-10 | 6 |

Observação:

- esse particionamento foi materializado no rerun rigoroso já deletado
- portanto, hoje ele está documentado neste handout, mas não há mais artefato local correspondente para abrir

### 5.1 O que é bloco de meses e o que é fold

Esses dois conceitos não são a mesma coisa.

Blocos de meses:

- `development`
- `definition lock`
- `final model evaluation`

servem para separar papéis cronológicos do estudo.

Folds:

- são os cortes mensais de treino/teste feitos dentro de um bloco

Então:

- `34 meses de development` não significa `34 folds`
- significa que existe um bloco cronológico de 34 meses dentro do qual a busca da definição roda

### 5.2 Como os folds da definição são formados

Na busca da definição:

- treino = meses acumulados até aqui
- teste = mês seguinte

Com `34` meses no `development`, isso gera `33` outer folds:

- fold 1: treina no 1º mês e testa no 2º
- fold 2: treina nos 2 primeiros meses e testa no 3º
- ...
- fold 33: treina nos 33 primeiros meses do development e testa no 34º

Ou seja:

- a definição não é escolhida com base em um único mês
- ela é testada repetidamente mês a mês dentro do bloco de development

### 5.3 Como os folds do modelo são formados

No modelo, a lógica do fold também é:

- treino = meses anteriores acumulados
- teste = próximo mês

Mas o código limita o número de outer test months a `6`.

Isso quer dizer:

- o modelo oficial publica apenas `6` outer folds de teste
- mas, em cada um deles, o treino continua podendo usar todo o histórico oficial anterior disponível

Então:

- o modelo não "usa só 6 meses"
- ele usa `6 meses como outer test`
- e meses anteriores como histórico de treino

## 6. O que `Definition A` significa no código

`Definition A` não é uma regra única fixa. Ela é uma família de candidatas geradas agora em duas etapas:

1. screening atômico `m >= t`
2. expansão controlada a partir dos atômicos promovidos:
   - pairwise `AND`
   - pairwise `OR`
   - combinações ponderadas `w1*z1 + w2*z2 >= τ`, com `z` em escala de percentil empírico no treino do fold

Fluxo atual da `Definition A`:

1. gerar candidatas atômicas no `development`
2. avaliar cada candidata sem modelo usando:
   - gaps em validadores pós-label
   - prevalência
   - entropia da prevalência
   - estabilidade mensal
   - largura do bootstrap CI da prevalência
3. agregar em outer folds temporais do development
4. escolher representantes por métrica
5. escolher representantes por vetor de label
6. construir fronteira de Pareto
7. ranquear candidatas primárias
8. promover só o `top-K` atômico para expansão
9. gerar candidatas compostas e ponderadas apenas a partir desse conjunto promovido
10. reranquear sem modelo
11. promover só o `top-K` final para o `definition lock`
12. no lock, medir:
   - gaps médios
   - estabilidade dos gaps mês a mês
   - sensibilidade de threshold
   - troca `AND -> OR`
   - `drop-one-literal`
   - `weight perturbation`
13. escolher 1 vencedora final

Arquivos relevantes:

- [/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/targeted_ml/pipelines/modelled_to_ml/definitions.py](/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/targeted_ml/pipelines/modelled_to_ml/definitions.py)

## 7. O que está materializado hoje para `Definition A`

Nos artefatos vigentes em [build/tables](/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/build/tables), o `core_definition_selection_v1.parquet` tem só duas linhas:

- `definition_a`
- `definition_b`

E a `Definition A` oficial materializada hoje é:

- `((future_business_active_weeks >= 3 OR future_distinct_actions >= 4) AND future_session_minutes >= 8.3138)`

com:

- `official_status = official_unique`
- `selection_basis = preferred_definition_a_rule_text_fixed_after_robustness_review`

Isso é um fato do artefato atual em:

- [core_definition_selection_v1.parquet](/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/build/tables/core_definition_selection_v1.parquet)

Conclusão importante:

- o projeto materializado atual **já contém** `AND/OR`
- o código atual também voltou a gerar compostas e ponderadas
- ainda existe divergência entre:
  - o build materializado vigente, que é anterior a essa implementação
  - e o search oficial agora implementado no código

## 8. O que isso implica sobre o estado real do projeto

Hoje coexistem duas verdades diferentes:

1. **Artefato materializado vigente**
- mostra `Definition A` composta
- com seleção baseada em `preferred_definition_a_rule_text_fixed_after_robustness_review`

2. **Código de search atualmente inspecionado**
- está configurado para `screened_pairwise_compound_weighted`
- gera screening atômico, pairwise booleano e pairwise ponderado

Então, neste momento, o projeto está metodologicamente inconsistente entre:

- o que já foi materializado
- e o que o search atual geraria se fosse rerodado

Esse ponto é crítico e precisa ser reconhecido explicitamente.

## 9. O que foi medido sem modelo para escolher a definição

As métricas de validade da definição hoje são:

- `gap_returned_active_post_label_m1`
- `gap_returned_active_post_label_m2`
- `gap_returned_active_post_label_m3`
- `gap_active_days_post_label_3m`
- `gap_sustained_active_2of3_post_label`
- `prevalence_entropy`
- `monthly_prevalence_std`
- `bootstrap_prevalence_ci_width`

Interpretação correta:

- os `gaps` medem separação comportamental futura depois do label
- as métricas de prevalência medem estabilidade/uso operacional do alvo
- nenhuma delas é métrica de modelo

## 10. O que o código já protege contra leakage

No loop de modelagem, o desenho atual está separado assim:

- `outer test`: mês futuro fora do treino
- `outer train`: meses anteriores acumulados
- dentro do `outer train`:
  - tuning em splits temporais internos
  - calibração em holdout temporal interno
- o `outer test` não entra em tuning nem calibração

Além disso:

- o dataset oficial só entra se `full_followup_observed_flag == 1`
- existe auditoria estrutural de leakage por feature/cenário

Importante:

- isso é forte como proteção temporal/estrutural
- mas não prova ausência absoluta de leakage semântico

## 11. O que o código faz hoje para auditoria formal de overfitting

O código agora **materializa** uma comparação formal `train vs test` para dizer:

- "o modelo está muito melhor no treino e muito pior no teste"
- "isso é um sinal forte de overfitting/generalization gap"

O que entra agora no loop principal é:

- métricas de `apparent_train`
- métricas de `calibration_holdout`
- métricas de `outer test`
- comparação pareada por fold
- `generalization_gap` por métrica
- CI bootstrap do gap
- `statistical_gap_flag`

Essa auditoria fica pensada como:

1. nível de outer fold
2. comparação `apparent_train vs outer test`
3. comparação `calibration_holdout vs outer test`
4. sinal formal quando o CI do gap fica inteiramente acima de zero

Observação importante:

- o código está pronto para isso
- o build materializado vigente ainda não traz essas tabelas, porque ele é anterior à implementação atual

Importante:

- isso é **separado** do problema da definição do alvo
- e também é **separado** do leakage

## 12. O que o código faz hoje para desbalanceamento

O treino já tenta lidar com classe desbalanceada:

- regressão logística:
  - `class_weight = None / balanced`
- random forest:
  - `class_weight = None / balanced / balanced_subsample`
- CatBoost:
  - `auto_class_weights = Balanced / SqrtBalanced`

O código não usa, até onde foi verificado:

- SMOTE
- undersampling
- oversampling

Isso ajuda o ajuste do modelo, mas não resolve falta de positivos no fold de teste.

## 13. Fato importante: eu não posso concluir que "toda a família A é rara"

Esse foi um erro de raciocínio anterior e precisa ficar explícito.

O que os dados mostram é:

- a vencedora atual `future_active_days >= 9` é rara
- isso não autoriza dizer que todas as candidatas `A` são raras

Foi feita uma auditoria anterior das `10` candidatas `A` do rerun rigoroso já deletado, medindo suporte no `lock` e no `final`.

Resultado resumido:

| rank | regra `A` | status | positivos no lock | positivos no final |
|---:|---|---|---:|---:|
| 1 | `future_business_active_weeks >= 3` | sensitivity_lock_topk | 105 | 72 |
| 2 | `future_distinct_actions >= 4` | sensitivity_lock_topk | 172 | 74 |
| 3 | `future_active_days >= 9` | official_winner | 17 | 13 |
| 4 | `future_downloads >= 31` | dev frontier | 9 | 4 |
| 5 | `future_sessions >= 21` | dev frontier | 20 | 14 |
| 6 | `future_interactions >= 93` | dev frontier | 27 | 11 |
| 7 | `future_activity_events >= 80` | dev frontier | 35 | 14 |
| 8 | `future_mapped_lessons >= 83` | dev frontier | 16 | 3 |
| 9 | `future_content_views >= 49` | dev frontier | 25 | 8 |
| 10 | `future_session_minutes >= 7.61188` | dev frontier | 588 | 352 |

Conclusão correta daquela auditoria:

- a família `A` tem candidatas com suportes muito diferentes
- a regra `A` que venceu naquele rerun não era representativa da família inteira em termos de raridade

Importante:

- essa auditoria dizia respeito ao rerun rigoroso deletado
- ela **não** descreve a regra composta atualmente materializada em `build/tables`

## 14. Problema real hoje

O problema real hoje tem duas partes:

1. **Inconsistência entre artefato e código**
- o artefato materializado vigente usa uma regra composta escolhida por revisão/robustness review
- o search atual do código está em `univariate_exact`

2. **Falta de uma etapa explícita de avaliabilidade/modelabilidade do alvo**
- o pipeline tem boa camada de validade sem modelo
- mas ainda não tem uma etapa formal separada para dizer:
  - "essa candidata é semanticamente forte"
  - "essa candidata também é avaliável de forma robusta no desenho oficial"

Em outras palavras:

- existe uma camada de seleção sem modelo
- mas ainda não existe um protocolo fechado e explícito que una:
  - espaço de busca
  - composição de regras
  - tratamento de thresholds
  - sensibilidade estrutural da regra
  - avaliabilidade final do alvo

## 14. O que o código faz hoje com suporte de fold

No estágio de modelagem, o código marca fold como oficial só se tiver:

- `min_official_test_rows = 50`
- `min_official_test_positives = 5`
- `min_official_test_negatives = 20`
- `min_official_valid_outer_folds = 2`

Isso é um fato do código atual.

Importante:

- esses thresholds existem no estágio de modelagem
- eles **não equivalem** a uma auditoria formal de avaliabilidade da definição no estágio de seleção do alvo
- então hoje ainda existe um desencaixe:
  - a definição pode ganhar semanticamente
  - e só depois descobrir que não sustenta evidência oficial no modelo

## 15. Regra composta e thresholds "milimétricos"

No artefato atual, a `Definition A` oficial é:

- `((future_business_active_weeks >= 3 OR future_distinct_actions >= 4) AND future_session_minutes >= 8.3138)`

Isso expõe um problema importante:

- thresholds contínuos exatos como `8.3138` podem parecer arbitrários ou espúrios
- isso gera a pergunta certa:
  - "testamos 5, 6, 7, 8, 9, 10 e 8 foi melhor?"
  - "ou 8.3138 apareceu como um valor muito específico e talvez pouco interpretável?"

O jeito rigoroso de tratar isso não é fingir que o valor exato é semanticamente importante.

Também é importante deixar explícito o que **não** dá para afirmar com o que existe hoje no repositório:

- não dá para sustentar a narrativa:
  - "testamos 5, 6, 7, 8, 9, 10 minutos, 8 foi o melhor, e depois refinamos para 8.3138"

O que dá para afirmar hoje é:

- existe um artefato final materializado com `future_session_minutes >= 8.3138`
- mas o repositório atual não preserva, de forma auditável e reconstituível a partir dos artefatos finais vigentes, a trilha completa que provaria essa história exata de refinamento

Outro detalhe importante:

- `8.3138` não é "milissegundo"
- `future_session_minutes` vem de `SUM(COALESCE(s.duration_min, 0))`
- então é **minuto fracionário**, não unidade temporal ultrafina

Arquivo relevante:

- [/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/targeted_ml/pipelines/modelled_to_ml/dataset_builder.py](/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/targeted_ml/pipelines/modelled_to_ml/dataset_builder.py)

O framing correto é:

- o threshold bruto é um **cutpoint empírico candidato**
- ele não deve ser interpretado como número "mágico" por si só
- ele precisa passar por checagem de estabilidade local

Para métricas contínuas, o tratamento metodologicamente mais sólido é:

1. gerar thresholds a partir do treino, nunca do holdout
2. representar o threshold também por sua posição empírica, por exemplo:
   - percentil no fold de treino
3. exigir estabilidade local:
   - pequenas mudanças no threshold não podem inverter completamente o comportamento da definição
4. de preferência, apresentar thresholds contínuos em forma arredondada ou por faixa/quantil, e não como valor pseudo-exato

Então, para minutos de sessão, a leitura correta não é:

- "`8.3138` minutos é um número teoricamente especial"

E sim:

- "há um ponto de corte empírico nessa vizinhança, cuja robustez precisa ser verificada por sensibilidade local"

## 16. Fragilidade estatística já observada

A fragilidade estatística central que já apareceu nas análises anteriores é:

- evento raro
- teste de um único mês
- poucos positivos por mês

Essa combinação gera:

- alta variância nas métricas
- alta variância na calibração
- maior chance de leituras instáveis fold a fold

Isso não é, por si só, prova de overfitting.

Mas é um cenário em que:

- qualquer conclusão forte sobre generalização precisa ser mais cautelosa
- e uma auditoria formal `train vs test` ficaria ainda mais importante

## 17. Busca composta, pesos e sensibilidade estrutural

Se a meta é que `Definition A` represente:

- continuidade geral de uso
- mas também combinações interessantes descobertas nos dados

então o espaço de busca precisa incluir explicitamente:

1. regras atômicas
- `m >= t`

2. regras booleanas compostas
- `(m1 >= t1) AND (m2 >= t2)`
- `(m1 >= t1) OR (m2 >= t2)`

3. regras com pesos
- `w1*z1 + w2*z2 >= τ`

onde:

- `z1`, `z2` são versões em escala comparável
- o jeito mais seguro é usar percentil empírico no fold de treino, em `[0,1]`

Esse ponto é central:

- sim, isso é uma forma de normalização/escalonamento
- e ela precisa ser feita só com o treino do fold, nunca com holdout

Além disso, a sensibilidade do lock precisa ir além do caso univariado atual.

Para regra composta, a sensibilidade deve medir também:

- trocar `AND` por `OR`
- remover uma perna da regra
- variar um threshold mantendo o outro fixo
- variar pesos em uma grade pequena e auditável

Pergunta formal correta:

- "se eu tenho `A AND B`, ou um score ponderado de `A` e `B`, pequenas mudanças de threshold/combiner/peso mudam pouco, mas não irrelevante, o label e os validadores?"

Hoje o código **não faz isso** de forma oficial na busca atual.

## 18. O que ainda não existe no código

Até este momento, **não existe** uma tabela ou etapa explícita do tipo:

- `core_definition_modelability_audit_v1`
- ou `core_definition_evaluability_audit_v1`

Também não existe ainda uma regra metodológica separada que diga:

- quais candidatas `A` são elegíveis para virar alvo oficial final
- com base não em métricas de modelo, mas em suporte/avaliabilidade do próprio label no desenho oficial

Esse é o principal gap metodológico aberto.

## 19. Situação do serving

Foi identificado um problema real:

- o código de serving conseguia escolher um candidato de outro `definition_group` se ele fosse melhor no `model_frontier`

Foi adicionada proteção em:

- [/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/targeted_ml/pipelines/modelled_to_ml/selection.py](/Users/akatsurada/Documents/INSPER/Design/Aula_9/targeted_ml_modular_v1/targeted_ml/pipelines/modelled_to_ml/selection.py)

Hoje:

- o serving restringe candidatos ao `definition_group` escolhido no contexto da definição
- se não houver candidato do grupo escolhido, ele falha explicitamente

Importante:

- isso é só um guardrail tardio
- não resolve o problema metodológico principal
- o lugar certo para resolver a consistência entre alvo e modelo é antes, na seleção do alvo

## 20. Estado do rerun rigoroso

O rerun rigoroso atual foi interrompido propositalmente depois de materializar artefatos centrais, para evitar continuar rodando com política ainda em debate.

Status prático:

- não há build `build_same_month_entry_rigorous` rodando agora
- existem artefatos parciais suficientes para auditoria metodológica
- ainda não existem, nesse rerun:
  - `post_model_threshold_metrics_v1.parquet`
  - `post_model_feature_importance_v1.parquet`
  - `build_summary_v1.json`
  - HTML final novo

## 21. O que já está sólido

O que já está sólido no código:

- separação `development -> lock -> final eval`
- filtro populacional oficial `same_month_entry_only`
- target-definition sem modelo
- tuning, calibração e outer test temporalmente separados
- paralelização global de comparação de modelo com `workers = 6`
- artefatos de sensibilidade populacional por `months_after_entry`
- guardrail de serving por `definition_group`

## 22. O que ainda está em aberto

As decisões metodológicas ainda em aberto são:

1. Deve existir uma etapa formal de `definition evaluability/modelability audit` antes da modelagem oficial?
2. Como essa etapa deve ser definida de forma rigorosa, sem usar métricas de modelo e sem introduzir números arbitrários mal justificados?
3. O espaço oficial de busca de `Definition A` deve voltar a incluir compostas e pesos?
4. Como tratar thresholds contínuos pseudo-exatos de forma interpretável e robusta?
5. Como auditar formalmente `train vs test` para sinal forte de overfitting/generalization gap?
6. O que exatamente torna uma candidata `A` elegível para virar o alvo oficial final?
7. Como tratar o fato de que algumas candidatas `A` têm validade comportamental forte, mas suportes muito diferentes?

## 23. Leituras corretas e incorretas

Leituras corretas:

- `Definition A` é uma família de candidatas, não uma regra única.
- O artefato materializado vigente tem `Definition A` composta.
- O motor de regras suporta `AND/OR`.
- O search atual do código foi reorganizado para `univariate_exact`.
- Threshold contínuo exato não deve ser tratado como número "mágico".
- O repositório atual não sustenta sozinho a narrativa exata de refinamento "5,6,7,8,9,10 -> 8 -> 8.3138".
- O código atual já faz parte relevante da seleção do alvo sem modelo.
- Ainda falta uma etapa explícita separada para avaliabilidade/modelabilidade do alvo.
- Ainda falta fechar o espaço oficial de busca composto e sua sensibilidade.
- Ainda falta uma auditoria formal `train vs test` para sinal forte de overfitting.

Leituras incorretas:

- "Toda a família A é rara."
- "O projeto atual nunca usou regra composta."
- "O valor 8.3138 tem significado semântico próprio só porque apareceu no artefato."
- "Brier muito baixo por si só prova overfitting."
- "Se a A venceu, ela automaticamente deveria ser o alvo oficial do modelo."
- "O problema pode ser resolvido no `serving`."
- "Balanceamento de classe no treino resolve falta de positivos no teste."

## 24. Próximo passo recomendado

O próximo passo recomendado não é mexer no modelo ainda.

O próximo passo recomendado é:

1. especificar formalmente a etapa de `definition evaluability/modelability audit`
2. decidir, sem usar modelo:
   - quais critérios de avaliabilidade fazem parte do protocolo oficial
   - e como eles entram no filtro final da família `A`
3. fechar explicitamente o espaço de busca da `Definition A`:
   - atômicas
   - compostas `AND/OR`
   - combinações com pesos sobre métricas em escala comparável
4. fechar explicitamente a sensibilidade estrutural da regra composta:
   - threshold
   - combiner
   - drop-one-literal
   - peso
5. especificar a auditoria formal `train vs test` com regra de decisão pré-fixada para sinal forte de sobreajuste
6. só depois rerodar `modelled -> ml` end-to-end

Enquanto isso não estiver fechado, qualquer rerun completo continua correndo o risco de materializar uma definição vencedora semanticamente interessante, mas metodologicamente frágil como alvo oficial de modelagem.

## 25. Itens que devem entrar explicitamente no protocolo oficial

Esta seção consolida **tudo o que já concluímos que deve entrar** no protocolo oficial do projeto.

### A. Seleção da definição do alvo separada do modelo

Deve entrar explicitamente:

- `Definition A` é escolhida **sem métricas de modelo**
- `Definition B` continua sendo comparador literal fixo
- a seleção do alvo acontece antes da modelagem oficial
- o modelo não pode escolher o alvo
- `definition_selection` congela o `definition_group`

### B. Split temporal em três etapas

Deve entrar explicitamente:

1. `development`
2. `definition lock`
3. `final untouched model evaluation`

E deve ficar claro que:

- o período usado para escolher a definição não é o mesmo usado para publicar a performance oficial do modelo

### C. Espaço oficial de busca da `Definition A`

Deve entrar explicitamente:

1. **Regras atômicas**
- `m >= t`

2. **Regras compostas booleanas**
- `(m1 >= t1) AND (m2 >= t2)`
- `(m1 >= t1) OR (m2 >= t2)`

3. **Regras com pesos**
- `w1*z1 + w2*z2 >= τ`

onde:

- `z` é a métrica futura em escala comparável
- a escala comparável deve ser obtida por transformação no **treino do fold**, não no holdout

### D. Normalização para combinações com peso

Deve entrar explicitamente:

- métricas em escalas diferentes não podem ser somadas brutas
- a recomendação atual é usar **percentil empírico no fold de treino**
- isso gera valores em `[0,1]`
- a transformação deve ser calculada apenas com o treino do fold

### E. Threshold testing como sensibilidade oficial

Deve entrar explicitamente:

- `threshold testing` é uma forma de sensibilidade
- não basta dizer que um threshold foi "escolhido"
- é preciso mostrar como o comportamento muda quando o threshold varia

Isso vale para:

1. **Regras atômicas**
- variar `t`

2. **Regras compostas**
- variar `t1`
- variar `t2`
- mantendo o restante fixo

3. **Regras com pesos**
- variar `τ`

O nome correto a registrar é:

- **threshold sensitivity**

### F. Sensibilidade estrutural da regra composta

Deve entrar explicitamente:

- o que acontece se trocar `AND` por `OR`
- o que acontece se remover uma perna da regra
- o que acontece se variar pesos

Isso deve ser medido por:

- mudança no label
- mudança nos gaps pós-label
- mudança na prevalência

Métricas já coerentes com o código atual:

- `label_jaccard`
- `gap delta`
- `prevalence delta`

### G. Universo de métricas candidatas da `Definition A`

Deve entrar explicitamente:

- o universo de métricas candidatas não é "descoberto automaticamente"
- ele é uma decisão de projeto
- precisa ser registrado como política do estudo

Hoje, no código, ele inclui:

- `future_business_active_weeks`
- `future_sessions`
- `future_session_minutes`
- `future_interactions`
- `future_activity_events`
- `future_active_days`
- `future_distinct_actions`
- `future_downloads`
- `future_content_views`
- `future_mapped_lessons`
- `future_formation_events`

E exclui:

- `future_mari_help_events`
- `future_mari_conversation_events`

### H. Tratamento de thresholds contínuos pseudo-exatos

Deve entrar explicitamente:

- thresholds contínuos como `8.3138` são **cutpoints empíricos candidatos**
- eles não devem ser tratados como números teoricamente especiais
- sua robustez precisa ser demonstrada por `threshold sensitivity`
- idealmente, também devem ser descritos por posição empírica, faixa ou valor arredondado

Também deve ficar explícito:

- o repositório atual **não sustenta sozinho** a narrativa detalhada de refinamento do tipo:
  - `5, 6, 7, 8, 9, 10 -> 8 -> 8.3138`

### I. Auditoria formal de overfitting / generalization gap

Deve entrar explicitamente:

- comparação formal `train vs test`
- com regra de decisão pré-fixada
- para detectar sinal forte de:
  - muito bom no treino
  - claramente pior no teste

Isso deve ser separado de:

- leakage
- problema de definição do alvo
- problema de suporte do fold

### J. Problema de evento raro + teste mensal + poucos positivos

Deve entrar explicitamente:

- evento raro
- outer test mensal
- poucos positivos por mês

gera:

- alta variância nas métricas
- alta variância na calibração
- maior fragilidade interpretativa

Isso não é automaticamente overfitting, mas precisa entrar como fragilidade estatística oficial.

### K. Audit de avaliabilidade/modelabilidade do alvo

Deve entrar explicitamente uma etapa separada para responder:

- esta candidata é semanticamente forte?
- esta candidata também é avaliável de forma robusta no desenho oficial?

Essa etapa:

- não pode usar métrica de modelo
- não pode deixar o modelo escolher o alvo
- deve vir antes da modelagem oficial

### L. Política correta de serving

Deve entrar explicitamente:

- o estudo de definição não decide mais o alvo da modelagem
- a modelagem oficial desta análise usa apenas `definition_b_label`
- `serving` e `reference scope` seguem o grupo único presente no `model_frontier`
- `definition_selection` e `definition_frontier` permanecem como artefatos do estudo de definição, não como autoridade para trocar o alvo do modelo

Se não houver candidato válido naquele grupo:

- não trocar silenciosamente para outro grupo
- não usar fallback implícito para outro alvo
- registrar explicitamente que o alvo oficial não ficou servable naquele run

### M. Metadata e trilha de auditoria coerentes

Deve entrar explicitamente:

- metadata precisa refletir o escopo real da seleção
- se o pool foi filtrado por `definition_group`, a reason string precisa dizer isso
- logs e tabelas não podem sugerir `all_frontier_candidates` se houve filtro de grupo

### N. Divergência entre artefato vigente e search atual do código

Deve entrar explicitamente:

- o artefato vigente tem `Definition A` composta
- o search atualmente inspecionado no código está em `univariate_exact`
- isso é uma inconsistência real entre estado materializado e estado atual do pipeline
- esse ponto precisa ser resolvido antes de qualquer rerun oficial

## 26. Especificação operacional proposta

Esta seção transforma as conclusões acima em desenho operacional mais concreto.

### 26.1 Universo de métricas da `Definition A` por tiers

Para manter o significado de "atividade" como **continuidade de uso**, mas ainda permitir traços interessantes descobertos pelos dados, o universo de busca deve ser particionado em tiers.

#### Tier 1. Métricas de continuidade básica

Estas métricas representam continuidade geral de uso e podem definir atividade sozinhas:

- `future_business_active_weeks`
- `future_sessions`
- `future_session_minutes`
- `future_active_days`
- `future_distinct_actions`
- `future_activity_events`

#### Tier 2. Métricas de intensidade/valor de uso

Estas métricas podem complementar atividade, mas não deveriam, sozinhas, monopolizar o conceito de "ativo":

- `future_interactions`
- `future_downloads`
- `future_content_views`
- `future_mapped_lessons`

#### Tier 3. Métricas específicas de domínio

Estas métricas capturam comportamento relevante, mas mais específico que continuidade geral:

- `future_formation_events`

#### Regra de elegibilidade do search space

Proposta:

- regra atômica pode usar:
  - Tier 1
  - Tier 2
  - Tier 3
- regra composta final deve conter **pelo menos 1 métrica Tier 1**

Consequência:

- `future_formation_events` pode entrar
- mas não pode, sozinha, sequestrar o conceito final de atividade se a regra composta não tiver também uma dimensão clara de continuidade

### 26.2 Gramática oficial de busca da `Definition A`

O search space oficial deve conter:

#### A. Regras atômicas

- `m >= t`

#### B. Regras compostas booleanas

- `(m1 >= t1) AND (m2 >= t2)`
- `(m1 >= t1) OR (m2 >= t2)`

#### C. Regras compostas ponderadas

- `w1*z1 + w2*z2 >= τ`

com:

- `w1, w2 >= 0`
- `w1 + w2 = 1`
- `z1, z2` em escala comparável

#### Escopo recomendado

Para controlar complexidade:

- começar com regras de tamanho 2
- não usar árvores de regras neste estágio
- permitir apenas combinações pareadas

### 26.3 Geração de thresholds

Thresholds devem ser testados explicitamente. Isso é parte da sensibilidade e precisa aparecer como protocolo formal.

#### Para métricas de contagem/discretas

Proposta:

- gerar candidatos a partir dos valores observados no treino
- deduplicar thresholds que produzam o mesmo vetor de label

#### Para métricas contínuas

Proposta:

- não tratar o valor bruto exato como número "mágico"
- gerar thresholds a partir de uma grade empírica definida no treino
- registrar simultaneamente:
  - raw cutoff
  - posição empírica do cutoff no treino

Leitura correta:

- o threshold bruto é só a materialização de um cutpoint empírico
- a interpretação substantiva deve vir da sua estabilidade, não da pseudo-precisão do número

### 26.4 Threshold sensitivity

Threshold testing precisa entrar explicitamente como parte da seleção da definição.

Para cada regra candidata:

- medir o que acontece quando o threshold sobe um passo
- medir o que acontece quando o threshold desce um passo

Saídas mínimas:

- mudança do label
- mudança dos gaps pós-label
- mudança da prevalência

Métricas já coerentes com o projeto:

- `label_jaccard`
- `gap delta`
- `prevalence delta`

### 26.5 Pipeline recomendado da busca da `Definition A`

#### Etapa 1. Screening atômico no development

Objetivo:

- descobrir regras atômicas promissoras

Procedimento:

1. gerar candidatas atômicas no treino dos folds de development
2. avaliar sem modelo
3. agregar no test dos folds de development
4. escolher representantes por métrica
5. deduplicar por label hash
6. construir fronteira e ranking

#### Etapa 2. Promoção

Objetivo:

- escolher as peças que podem ser combinadas

Procedimento:

- promover um conjunto limitado de regras atômicas
- o limite deve ser explicitado e registrado

#### Etapa 3. Expansão composta

Objetivo:

- gerar noções mais ricas de atividade

Procedimento:

- a partir das atômicas promovidas, gerar:
  - pares `AND`
  - pares `OR`
  - pares ponderados

Restrições:

- toda regra composta final deve conter pelo menos um literal Tier 1
- regras logicamente redundantes devem ser deduplicadas por label hash

#### Etapa 4. Re-ranking sem modelo

Objetivo:

- comparar atômicas e compostas no mesmo plano metodológico

Procedimento:

- usar os mesmos validadores pós-label
- usar estabilidade temporal
- usar penalidade de complexidade

#### Etapa 5. Definition lock

Objetivo:

- congelar a definição final sem usar modelo

Procedimento:

- levar as candidatas sobreviventes ao `definition lock`
- aplicar sensibilidade:
  - de threshold
  - estrutural
  - de peso
- escolher 1 vencedora final

### 26.6 Sensibilidade estrutural da regra composta

Para regras compostas, o lock precisa medir explicitamente:

#### A. Threshold sensitivity

- variar `t1`, mantendo `t2`
- variar `t2`, mantendo `t1`

#### B. Combiner sensitivity

- trocar `AND` por `OR`
- quando a regra base for booleana

#### C. Literal ablation

- remover o literal `A`
- remover o literal `B`

#### D. Weight sensitivity

Para regras ponderadas:

- variar pesos em uma grade pequena e auditável

Exemplo de grade inicial:

- `(0.25, 0.75)`
- `(0.50, 0.50)`
- `(0.75, 0.25)`

### 26.7 Escala comparável para regras com peso

Quando métricas entram em combinação ponderada, é necessário escalonamento.

Proposta atual:

- usar **percentil empírico no fold de treino**

Vantagens:

- coloca tudo em `[0,1]`
- preserva ordem
- reduz dependência da unidade bruta
- impede que uma métrica domine apenas por escala

Restrições:

- a transformação deve ser calculada só no treino
- aplicada ao lock/test sem vazamento

### 26.8 Avaliabilidade/modelabilidade do alvo

Antes da modelagem oficial, deve existir uma etapa explícita de auditoria da definição final.

Objetivo:

- responder se a candidata vencedora é avaliável de forma robusta no desenho oficial

Essa etapa deve usar:

- distribuição temporal do label
- suporte por mês
- suporte pooled no holdout final
- incerteza de prevalência
- estabilidade dos gaps pós-label

Essa etapa **não** pode usar:

- AP
- Brier
- confusion matrix do modelo
- score do modelo

### 26.9 Auditoria formal `train vs test` para overfitting

Também deve existir uma auditoria formal separada para generalization gap.

Importante:

- comparar treino in-sample com teste não é o melhor desenho
- o mais rigoroso é comparar:
  - estimativa no outer-train obtida de forma out-of-fold
  - versus outer-test

Saídas desejadas:

- `train_oof_ap`
- `train_oof_roc_auc`
- `train_oof_brier`
- `train_oof_log_loss`
- `test_ap`
- `test_roc_auc`
- `test_brier`
- `test_log_loss`
- deltas `train_oof - test`

Regra de decisão:

- precisa ser pré-fixada
- deve gerar flag explícita de sinal forte de sobreajuste

### 26.10 Política de serving

O protocolo oficial deve declarar:

1. o estudo de definição fica separado da modelagem oficial
2. a modelagem oficial desta análise usa apenas `definition_b_label`
3. `serving` e `reference scope` seguem o grupo único presente no `model_frontier` materializado
4. `definition_selection` e `definition_frontier` não podem redefinir o alvo do modelo nessa etapa

Se não houver candidato válido dentro do grupo congelado:

- não trocar silenciosamente para outro grupo
- não usar fallback oculto
- registrar explicitamente:
  - grupo do `model_frontier`
  - grupos disponíveis no `model_frontier`
  - status de não-serving naquele run

### 26.11 Metadata obrigatória

O protocolo deve exigir que os artefatos registrem:

- universo de métricas candidatas
- tier de cada métrica
- estratégia de search
- thresholds testados
- transformações de escala
- regras compostas geradas
- sensibilidade de threshold
- sensibilidade estrutural
- decisão final do alvo
- razão da decisão
- status de serving

### 26.12 O que precisa ser resolvido antes do próximo rerun oficial

Antes de rerodar `modelled -> ml`, precisamos fechar:

1. tiers oficiais do universo de métricas
2. gramática oficial da busca composta
3. política de thresholds para contínuas
4. regra de promoção do screening atômico
5. sensibilidade estrutural da regra composta
6. auditoria de avaliabilidade/modelabilidade do alvo
7. auditoria formal `train vs test`
8. política de `serving` sem fallback implícito entre grupos
