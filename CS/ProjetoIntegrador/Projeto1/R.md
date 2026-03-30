# Análise de Sobrevivência

## 1) O que entra como dado

### Arquivos de entrada
- **Treino** e **Teste**, localizados automaticamente na pasta do projeto  
  (nomes contendo `__SURVIVAL__TRAIN.csv` e `__SURVIVAL__TEST.csv`).

### Coluna de tempo
- Detectada automaticamente entre as numéricas (ex.: `time`, `duration`).  
- Se não houver nome padrão, escolhe-se a numérica com **mediana mais alta** (regra simples para capturar a “cara de tempo”).

### Coluna de evento
- Detectada automaticamente (ex.: `status`, `event`).  
- Padronizada para **0/1** (0 = não aconteceu até o fim; 1 = aconteceu).

### Demais variáveis (preditoras)
- Usam-se **todas as colunas comuns** aos dois arquivos, exceto **IDs** e as próprias colunas de **tempo/evento**.  
- **Numéricas**: vazios recebem a **mediana do treino**.  
- **Textos/Categorias**: viram **fatores** com os **mesmos níveis** em treino e teste; vazios recebem a **categoria mais frequente** do treino.

### Fundamentação
- **Aproveitamento máximo da informação**: mantêm-se todas as colunas úteis e consistentes entre treino e teste.  
- **Preenchimento simples e reprodutível**: mediana/moda do **treino** evita “inventar” informação e garante repetibilidade.  
- **Remoção de IDs**: previne que o modelo “decore” casos específicos que **não generalizam**.

---

## 2) Termos (sem jargão)

- **Tempo**: quantos períodos cada empresa ficou sob observação até o evento ou até o fim do acompanhamento.  
- **Evento**: o acontecimento de interesse (1 = aconteceu; 0 = não aconteceu até o fim).  
- **Pontuação de risco**: número que resume o **quão cedo** o modelo espera que o evento ocorra (maior = tende a acontecer antes).  
- **Probabilidade ao longo do tempo**: chance prevista de **seguir sem evento** ou de **ter o evento** à medida que o relógio avança.  
- **Horizonte**: um ponto no tempo (curto, médio, longo) em que avaliamos as previsões.

---

## 3) Como garantimos uma comparação justa

### Estratégia de avaliação em camadas
- **5 partes (dobras) externas** no treino: em cada rodada, treina-se com 4 partes e testa-se na 5ª.  
- **3 partes internas** dentro do treino: usadas **só** para escolher os “ajustes finos” de cada modelo (ex.: número de árvores, força da penalização).  
- **Proporções preservadas**: cada divisão mantém proporção semelhante de linhas com e sem evento.  
- **Semente fixa**: resultados reprodutíveis.  
- **3 horizontes de tempo (curto/médio/longo)**: retirados dos próprios dados do treino de cada rodada (quartis **25%**, **50%** e **75%**).

### Fundamentação
- **Separação entre treinar e avaliar**: evita superestimar desempenho.  
- **Ajustes finos sem olhar o teste externo**: escolha imparcial dos parâmetros.  
- **Vários prazos de avaliação**: alguns métodos acertam mais cedo, outros mais tarde — mede-se **todo o arco temporal**.

---

## 4) Modelos comparados (como pensar neles e por que incluí-los)

> Todos recebem as mesmas colunas e geram **pontuações de risco** comparáveis.

### 4.1 Cox “simples” (CoxPH)
- **Como pensa**: efeito **aditivo e estável** das variáveis sobre o risco ao longo do tempo.  
- **Por que incluir**: **referência clássica** e transparente; ótimo **marco de comparação**.

### 4.2 Cox com “freio e seleção” (penalizado)
- **Como pensa**: igual ao Cox simples, mas com um **freio** que evita exageros e **seleciona** o essencial quando há muitas variáveis.  
- **Por que incluir**: mais **prudente** em bases amplas; reduz **ruído** e **sobreajuste**.

### 4.3 Floresta de Sobrevivência (muitas árvores)
- **Como pensa**: várias **árvores de decisão** aprendem regras diferentes e **votam**.  
- **Por que incluir**: captura **padrões tortos** e **interações** sem precisar especificá-las; robusto a **relações não lineares**.

### 4.4 Modelo por etapas (GBM com Cox)
- **Como pensa**: uma série de **pequenas correções sucessivas** que refinam o erro anterior.  
- **Por que incluir**: flexível para **não linearidades** e **efeitos complexos**; forte quando há **muitos sinais fracos**.

### 4.5 Modelo que prevê o tempo diretamente (AFT)
- **Como pensa**: prevê **o próprio tempo** até o evento; as variáveis **aceleram/desaceleram** esse relógio.  
- **Por que incluir**: visão **complementar** (foca no **tempo**, não só no risco); útil quando a “forma do tempo” segue **padrões simples**.

### Ajustes finos (como escolhemos)
- **Cox penalizado**: varrem-se a **força** e o **estilo** do freio e escolhe-se o melhor nas divisões internas.  
- **Floresta**: varrem-se **nº de árvores**, **nº de variáveis por divisão** e **tamanho mínimo de ramo**.  
- **GBM**: varrem-se **nº de etapas**, **profundidade** e **passo** das correções.  
- **AFT**: testam-se **três formatos plausíveis de tempo** e escolhe-se o melhor no treino interno.

### Fundamentação do portfólio
- **Equilíbrio entre clareza e flexibilidade**: **Cox** (claro) vs. **Floresta/GBM** (flexíveis).  
- **Redução de risco de viés de modelo único**: se um falhar num padrão, outro cobre.  
- **Cobertura de hipóteses diferentes**: **risco estável** (Cox), **relações tortas** (Floresta/GBM), **foco direto no tempo** (AFT).

---

## 5) Métricas (o que medem e como interpretar)

> Perguntas-chave: **(1) Quem vai antes?** (ordenação)   **(2) As probabilidades ao longo do tempo fazem sentido?**

### 5.1 C de Harrell — ordenação
- **O que mede**: porcentagem de pares onde quem recebeu **pontuação mais alta** realmente **teve o evento antes**.  
- **Interpretação**: **maior = melhor** em ordenar “quem vai primeiro”.

### 5.2 C de Uno — ordenação justa quando muitos não tiveram evento
- **O que mede**: variação do C que **permanece justa** mesmo quando **muitas linhas não tiveram evento** até o fim.  
- **Interpretação**: **maior = melhor**.

### 5.3 Brier Integrado (IBS) — qualidade média no tempo
- **O que mede**: **erro médio** entre as **probabilidades previstas** e o **que ocorreu** ao longo de todo o período.  
- **Interpretação**: **menor = melhor** (probabilidades mais próximas da realidade).

### 5.4 AUC dependente do tempo (IAUC) — separação ao longo dos prazos
- **O que mede**: quão bem o modelo **distingue** quem **já teve** o evento até certo tempo de quem **ainda não teve**, **integrado** por vários prazos.  
- **Interpretação**: **maior = melhor**.

### 5.5 Brier em um tempo representativo (`BS@t*`) — checagem pontual
- **O que mede**: o **erro** em um **tempo de referência** (o **mediano do treino** daquela rodada).  
- **Interpretação**: **menor = melhor**.

### Fundamentação do conjunto de métricas
- **C de Harrell/Uno** respondem à **ordenação** (prioridade de quem tende a ir antes).  
- **IBS/IAUC/`BS@t*`** avaliam a **qualidade das probabilidades ao longo do tempo**.  
- Em conjunto, temos **ranking + calibração temporal** — uma visão **completa e comparável**.

---

## 6) Gráficos gerados (o que mostram e como ler)

**Arquivo:** `survival_model_benchmark.pdf`  
**Conteúdo:** 5 gráficos de barras, com **média** e **barras de erro** ao longo das 5 rodadas externas.

- **C de Harrell** — “Quem ordena melhor quem vai antes.” (**maior = melhor**)  
- **C de Uno** — “Ordenação justa quando muitos não tiveram evento.” (**maior = melhor**)  
- **Brier Integrado (IBS)** — “Probabilidades mais próximas do observado ao longo do tempo.” (**menor = melhor**)  
- **AUC Integrada no Tempo (IAUC)** — “Separação consistente ao longo dos prazos.” (**maior = melhor**)  
- **Brier em `t*`** — “Erro num prazo representativo.” (**menor = melhor**)

### Como ler
- **Altura da barra** = desempenho **médio** do modelo.  
- **Tracinhos verticais** = **variação** entre as 5 rodadas (desvio-padrão).  
- **Ordenação das barras** facilita ver o **vencedor por métrica**.

---

## 7) Passo a passo do fluxo (execução)

1. **Localiza** os arquivos de treino e teste.  
2. **Detecta** e padroniza **tempo** e **evento (0/1)**.  
3. **Seleciona** apenas colunas **comuns** e remove **IDs**.  
4. **Preenche vazios** (mediana/moda do treino) e **alinha categorias**.  
5. **Cria** 5 rodadas **externas** e 3 **internas** (somente para ajustes finos).  
6. **Treina** os 5 modelos e gera **pontuações de risco**.  
7. **Avalia** pelos 5 indicadores em **3 horizontes** (curto/médio/longo).  
8. **Agrega** os resultados (médias e variações).  
9. **Salva** o **PDF** com os 5 gráficos e **CSVs** de resumo.

---

## 8) Saídas gravadas

- **PDF:** `survival_model_benchmark.pdf`  
- **CSVs:**  
  - `summary_HarrellC.csv`  
  - `summary_UnoC.csv`  
  - `summary_IBS.csv`  
  - `summary_IAUC.csv`  
  - `summary_Brier_t.csv`

---

## 9) Fundamentação — síntese final

- **Dados**: máximo aproveitamento de colunas, limpeza simples e coerente entre treino e teste.  
- **Justa comparação**: divisões externas para avaliação, internas para ajustes, horizontes curtos/médios/longos extraídos do próprio treino.  
- **Modelos**: portfólio que equilibra **clareza** (Cox), **prudência** (Cox penalizado), **flexibilidade** (Floresta/GBM) e **foco direto no tempo** (AFT).  
- **Métricas + Gráficos**: conjunto que cobre **ordenação** e **qualidade de probabilidade no tempo**, com leitura direta por barras e variações entre rodadas.
"""