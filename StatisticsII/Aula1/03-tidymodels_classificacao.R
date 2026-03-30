# Aprendizagem Estatistica de Maquina II 
# Aula 1 
# Modelos de classificacao com tidymodels

# Pacotes que serao utilizados --------------------------------------------

library(tidymodels)
library(ISLR)
library(tidyverse)
library(doParallel)

# vamos usar os dados Default
dados <- Default %>% 
  as_tibble()

glimpse(dados)

# tabela com a proporcao de cada uma das classes
dados %>% 
  count(default) %>% 
  mutate(prop = n/sum(n))

# treinamento x teste -----------------------------------------------------
set.seed(321)

split1 <- initial_split(dados, prop = 0.7, strata = "default")
split2 <- initial_split(dados, prop = 0.8, strata = "default")

treinamento <- training(split2) # treinamento
validacao <- testing(split2) # validacao
teste <- testing(split1) # teste

# processamento -----------------------------------------------------------

receita <- recipe(default ~ ., treinamento) %>% # define a receita, com a variavel resposta e os dados de treinamento
  step_normalize(all_numeric()) %>% #normaliza todas variaveis numerias 
  step_dummy(student) # define variavel dummy a variavel student

receita_prep <- prep(receita) # prepara a receita definida acima

treinamento_proc <- bake(receita_prep, new_data = NULL) # obtem os dados de treinamento processados
validacao_proc <- bake(receita_prep, new_data = validacao) # obtem os dados de validacao processados
teste_proc <- bake(receita_prep, new_data = teste) # obtem os dados de teste processados

# logistica -------------------------------------------------------------------
fit_glm <- logistic_reg() %>% # define um modelo de regressao logistica
  set_engine("glm") %>% # define a engine do modelo
  set_mode("classification") %>% # define que e'  problema de classificacao
  fit(default ~ ., treinamento_proc) # executa o modelo e estima os parametros


fit_glm # estimativas do modelo ajustado

tidy(fit_glm) # estimativas do modelo ajustado em formato tidy

fitted <- fit_glm %>% 
  predict(new_data = validacao_proc, type = "prob") %>% # realiza predicao para os dados de teste
  mutate(observado = validacao_proc$default, # cria uma coluna com o valor observado de default
         modelo = "logistica") # cria uma coluna para indicar qual o modelo ajustado

head(fitted) # mostra as 6 primeiras linhas do tibble criado


# LASSO -------------------------------------------------------------------

lasso <- logistic_reg(penalty = tune(), mixture = 1) %>% # define o modelo lasso e o parametro a ser tunado (o lambda)
  set_engine("glmnet") %>% # define a engine do modelo
  set_mode("classification") # define que e'  problema de classificacao

lasso # o modelo foi definido mas ainda nao foi ajustado

translate(lasso) # traduz o codigo tidymodels para o correspondente a engine utilizada

set.seed(321)

# validação cruzada para ajuste do hiperparametro em 10 lotes
cv_split <- vfold_cv(treinamento, v = 10, strata = "default")

lambda_tune <- tune_grid(lasso, # especificacao do modelo
                         receita,# a receita a ser aplicada a cada lote
                         resamples = cv_split, # os lotes da validacao cruzada
                         grid = 30,# quantas combinacoes de parametros vamos considerar
                         metrics = metric_set(roc_auc, accuracy)) # metricas consideradas

autoplot(lambda_tune) # plota os resultados

lambda_tune %>% 
  collect_metrics() # obtem as metricas calculadas

best <- lambda_tune %>% 
  select_best(metric = "roc_auc") # seleciona a melhor combinacao de hiperparametros

fit_lasso <- finalize_model(lasso, parameters = best) %>% # informa os valores de hiperparametros a serem considerados
  fit(default ~ .,data = treinamento_proc) # executa o modelo com os valores de hiperparametros definidos acima

fitted <- fitted %>% #      empilha as previsoes do lasso
  bind_rows(fit_lasso %>% # os valores preditos pelo lasso
              predict(new_data = validacao_proc, type = "prob") %>% 
              mutate(observado = validacao_proc$default, 
                     modelo = "lasso"))

head(fitted)
tail(fitted)

# arvore ----------------------------------------------------------------

arvore <- decision_tree(tree_depth = tune(), # define o modelo arvore de decicao e o parametros a serem tunados
                        cost_complexity = tune()) %>% 
  set_engine("rpart") %>% # define qual função sera usada
  set_mode("classification") # define que e'  problema de classificacao

arvore_tune <- tune_grid(arvore, # especificacao do modelo
                         receita, # a receita a ser aplicada a cada lote
                         resamples = cv_split, # os lotes da validacao cruzada
                         grid = 30, # quantas combinacoes de parametros vamos considerar
                         metrics = metric_set(roc_auc, accuracy)) # metricas consideradas

arvore_tune %>% 
  collect_metrics() # obtem as metricas calculadas

(best <- arvore_tune %>% 
    select_best(metric="roc_auc")) # seleciona a melhor combinacao de hiperparametros

fit_arvore <- finalize_model(arvore, parameters = best) %>% # informa os valores de hiperparametros a serem considerados
  fit(default ~ ., data = treinamento_proc) # executa o modelo com os valores de hiperparametros definidos acima

fitted <- fitted %>% #       empilha as previsoes da arvore de decisao
  bind_rows(fit_arvore %>% # os valores preditos pela arvore de decisao
              predict(new_data = validacao_proc, type = "prob") %>%
              mutate(observado = validacao_proc$default, 
                     modelo = "arvore"))

head(fitted)
tail(fitted)

# avaliação ---------------------------------------------------------------

# calcula a auc de cada modelo ajustado
fitted %>% 
  group_by(modelo) %>% 
  roc_auc(observado, .pred_No)

# Tarefa:
# escolher melhor modelo de acordo com o conjunto de validacao e 
# calcular o erro de generalizacao com o conjunto de testes
