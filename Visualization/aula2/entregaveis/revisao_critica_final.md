# Revisão Crítica Final (Guia Didático Completo)

## 1) Objetivo em linguagem simples
Você tem uma série diária de passageiros (`s_40380`) da estação Clark/Lake (Chicago). O objetivo é prever os próximos dias e, principalmente, mostrar **como e por que** as features ajudam o modelo a prever melhor.

Em termos simples:
- Queremos reduzir erro de previsão.
- Para isso, transformamos dados brutos em variáveis mais informativas (feature engineering).
- Depois validamos com método temporal (sem vazamento de futuro).

## 2) O que cada tipo de variável representa

### 2.1 Variável alvo
- `s_40380`: número diário de passageiros na estação Clark/Lake.

### 2.2 Clima
- Temperatura, umidade, vento, precipitação e probabilidades de chuva/neve/nuvem/tempestade.
- Ideia: clima pode afetar o quanto as pessoas usam transporte público.

### 2.3 Calendário
- Dia da semana, mês, fim de semana, posição no mês/ano.
- Ideia: rotina de segunda a sexta é diferente de sábado/domingo.

### 2.4 Feriados
- Indicadores de feriado e distância para feriado.
- Ideia: feriado altera deslocamento urbano no próprio dia e em dias próximos.

### 2.5 Esportes
- Flags de jogos (home/away), agregações e interações.
- Ideia: eventos esportivos mudam fluxo de mobilidade em certos dias.

### 2.6 Economia
- `l14_gas_price`, `l30_unemployment_rate`.
- Ideia: podem capturar contexto econômico com efeito indireto na demanda.

## 3) Qualidade de dados (EDA de consistência)
Antes de modelar, os checks críticos foram feitos:
- Duplicidade de data: sem duplicatas (treino e teste).
- Continuidade temporal: sem buracos nas datas.
- Nulos: identificados e tratados (incluindo `l30_unemployment_rate`).
- Faixas válidas: sem inconsistências de probabilidade e sem alvo negativo.
- Esportes binários: sem valores fora de 0/1.
- Consistência lógica: detectado conflito de baseball (colunas redundantes).

### 3.1 Sobre Cubs/WhiteSox (ponto crítico)
Foi identificado que colunas de baseball estavam redundantes e com conflito lógico. O tratamento aplicado foi:
- Consolidar sinal útil em `baseball_game_flag`.
- Remover colunas redundantes (`WhiteSox_Away`, `WhiteSox_Home`, `Cubs_Away`, `Cubs_Home`) antes do treino.

Isso evita colinearidade desnecessária e evita que o modelo “aprenda ruído duplicado”.

## 4) Engenharia de features (o que foi criado e por quê)

### 4.1 Calendário e sazonalidade
- Ex.: `day_of_week`, `is_weekend`, `week_of_year`, `month_boundary_weekday`.
- Captura padrão estrutural da rotina urbana.

### 4.2 Feriados e janelas
- Ex.: `is_holiday`, `is_holiday_fixed`, `is_holiday_movable`, `is_holiday_observed`, `days_to_nearest_holiday`, `pre_holiday_1d`, `post_holiday_1d`.
- Captura efeito no dia do feriado e “halo” antes/depois.

### 4.3 Memória temporal (lags e rolling)
- `lag` significa “valor passado da própria série”.
- Ex.: `lag_1`, `lag_7`, `lag_14`, `lag_28`.
- Rolling (média/desvio em janela) resume comportamento recente (ex.: 7 dias).

Por que isso ajuda?
- Demanda diária tem recorrência semanal forte; lags semanais ajudam muito previsão de curto prazo.

### 4.4 Clima transformado
- Ex.: `temp_mean`, `temp_range`, `weather_severity`, `temp_mean_delta_1d`, `precip_delta_1d`.
- Captura tanto nível do clima quanto mudança brusca de um dia para o outro.

### 4.5 Interações
- Ex.: `is_holiday_dow_0..6`, `sports_home_dow_0..6`, `weather_x_weekend`.
- Mesma condição pode ter efeito diferente conforme o dia da semana.

### 4.6 Regras de segurança
- Features criadas só com dados disponíveis até o dia de previsão.
- Sem vazamento de informação futura (essencial em séries temporais).

## 5) Como ler cada gráfico do entregável

## 5.1 Calendário semanal (demanda por dia da semana)
O que observar:
- Altura das categorias de segunda a domingo.

Como interpretar:
- Se dias úteis ficam consistentemente acima de sábado/domingo, calendário é driver forte.
- Neste caso, o gap é grande (queda forte em fim de semana).

## 5.2 Feriado vs não feriado
O que observar:
- Diferença média entre grupos “feriado” e “não feriado”.

Como interpretar:
- Queda relevante em feriado indica efeito estrutural (não ruído).

## 5.3 Distância para feriado
O que observar:
- Demanda em buckets de distância (0, 1-2, 3-7, >7 dias etc.).

Como interpretar:
- Mostra se existe “efeito vizinhança” de feriado além do dia exato.

## 5.4 Lag 7 vs demanda atual
O que observar:
- Relação entre valor de 7 dias atrás e valor atual.

Como interpretar:
- Tendência positiva indica recorrência semanal.
- Quanto mais alinhados os pontos, maior utilidade do lag.

## 5.5 Clima e demanda
O que observar:
- Tendência média e dispersão.

Como interpretar:
- Clima geralmente ajuda, mas neste projeto o efeito médio foi menor que calendário + feriados + lags.

## 5.6 Comparação de modelos
Modelos testados sob o mesmo protocolo temporal:
- Ridge
- RandomForest
- HistGradientBoosting

Como interpretar:
- Menor RMSE vence (MAE e MAPE como apoio).
- Com esse protocolo, HistGradientBoosting foi o melhor.

## 5.7 Efeito da engenharia de features (gráfico 5.3 atualizado)
O que observar:
- Para cada métrica (RMSE, MAE, MAPE), comparar baseline vs enhanced.

Como interpretar:
- Barras menores no enhanced = erro menor.
- Aqui houve melhoria nas três métricas ao mesmo tempo (sem trade-off negativo).

## 5.8 Diagnóstico treino vs validação (gráfico 5.4 atualizado)
O que observar:
- Erro no treino versus erro na validação temporal.

Como interpretar:
- Treino menor que validação é normal.
- Se a diferença for exagerada, há overfit forte.
- Resultado ficou em faixa de ajuste equilibrado (sem overfit severo).

## 6) Validação temporal (ponto-chave de ciência de dados)
Não foi usada validação aleatória. Foi usado backtesting rolling com horizonte real de 13 dias.

Por que isso é importante:
- Em série temporal, embaralhar dados destrói causalidade temporal.
- Rolling valida cenários parecidos com produção (prever futuro com passado).

## 7) Métricas (o que significam sem jargão)
- RMSE: erro médio com penalização maior para erros grandes.
- MAE: erro médio absoluto (mais direto de interpretar).
- MAPE: erro percentual médio (comparável em termos relativos).

Regra prática:
- Não olhar só uma métrica. Ver as três juntas evita decisão enviesada.

## 8) Conclusões finais (interpretadas)
1. Driver principal: calendário semanal (dias úteis muito acima de fim de semana).
2. Feriado: efeito estrutural relevante no dia e no entorno.
3. Lags: memória semanal melhora previsão de curto prazo.
4. Clima: agrega sinal, mas menor que calendário/feriado/lags.
5. Melhor modelo: HistGradientBoosting com tuning, mantendo generalização aceitável.

## 9) Trade-offs e limites (visão crítica)
- Não existe “ótimo global garantido” sem busca exaustiva muito maior.
- Ganhos adicionais podem exigir tuning mais caro e mais complexo.
- Algumas variáveis podem ter efeito contextual (mudam ao longo do tempo), exigindo monitoramento.

## 10) Checklist para você repetir sozinho
1. Ler e validar dados (duplicatas, nulos, faixa, consistência lógica).
2. Separar temporalmente treino/validação/teste (sem embaralhar).
3. Criar features por hipótese de negócio (calendário, feriado, lags, clima, interações).
4. Avaliar com rolling backtesting.
5. Comparar modelos com mesmo protocolo.
6. Diagnosticar overfit/underfit com treino vs validação.
7. Gerar previsões e relatório com interpretação, não só números.

## 11) Como usar este guia junto do HTML final
- O HTML final mostra os resultados visuais e entregáveis oficiais.
- Este arquivo é o “manual didático”: explica conceitos, leitura de gráficos e lógica de decisão.

Em resumo: o HTML responde “o que foi entregue”; este guia responde “como entender e reproduzir mentalmente a solução”.
