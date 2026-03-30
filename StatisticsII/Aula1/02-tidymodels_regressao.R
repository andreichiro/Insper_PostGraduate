# Aprendizagem Estatistica de Maquina II
# Aula 1 
# Modelos de regressao com tidymodels

# Pacotes que serao utilizados --------------------------------------------

# install.packages("tidymodels")
library(tidymodels)
library(tidyverse)
library(ISLR) # conjuntos de dados e funcoes do livro
library(vip) #visualizacao da importancia das variaveis
library(doParallel) #Facilita o uso de paralelismo para acelerar calculos
library(skimr) #resumos rapidos e estatisticas descritivas

# vamos usar os dados Credit
class(Credit)
view(Credit)

dados <- Credit %>% 
  as_tibble() 

dados %>% glimpse()


# rsample -----------------------------------------------------------------
set.seed(1) # definir semente aleatoria

split1 <- initial_split(dados, prop = 0.9) # definir particao dos dados
split1

split2 <- initial_split(training(split1), 0.8)
split2

treinamento <- training(split2) # treinamento
validacao <- testing(split2) # validacao
teste <- testing(split1) # teste

treinamento %>% 
  slice(1:2) # obtem uma fatia do dataframe, no caso, contendo as 2 primeiras linhas

erro <- function(split){
  tr <- training(split) # obtem o conjunto de treino 
  tst <- testing(split) # obtem o conjunto de teste
  
  fit_1 <- lm(Balance ~ Income + Limit, data = tr) # ajusta modelo linear com Income e Limit
  fit_2 <- lm(Balance ~ Rating + Age, data = tr) # ajusta modelo linear com Rating e Age
  
  tibble(eqm1 = Metrics::mse(tst$Balance, # calcula o eqm do modelo 1
                             predict(fit_1, tst)),
         eqm2 = Metrics::mse(tst$Balance, # calcula o eqm do modelo 2
                             predict(fit_2, tst)))
}

erro(split2)

# avaliacao do erro de generalizacao
fit_1 <- lm(Balance ~ Income + Limit, data = training(split1))
Metrics::mse(predict(fit_1, teste), teste$Balance)

# Valicadao cruzada com rsample -------------------------------------------

# definição de 5 lotes de validação cruzada 
# para os dados Credit
splits <- vfold_cv(treinamento, v = 5)

set.seed(123)

resultado_vc <- vfold_cv(treinamento, v = 5) %>% 
  mutate(erros = purrr::map(splits, erro)) %>% 
  unnest(erros)

resultado_vc

# TAREFA: fazer um grafico de barras
#         com os resultados acima

resultado_vc %>% 
  summarise(mean_eqm1 = mean(eqm1),
            mean_eqm2 = mean(eqm2))

resultado_vc %>% 
  summarise(across(where(is.numeric), mean))


# recipes - processamento -------------------------------------------------

treinamento %>% 
  group_by(Ethnicity) %>% 
  summarise(prop = n()/nrow(treinamento))

treinamento$Ethnicity %>% unique()

receita <- recipe(Balance ~ ., data = treinamento) %>% # define a receita, com a variavel resposta e os dados de treinamento
  step_rm(ID) %>% #remove o ID
  step_normalize(all_numeric(), -all_outcomes()) %>% # normaliza todas numericas exceto a variavel resposta
  step_other(Ethnicity, threshold = .30, other = "Outros") %>%  # define a categoria 'outros' para etinias com menos de 30% de frequencia
  step_dummy(all_nominal(), -all_outcomes()) # define variavel dummy para todas variaveis qualitativas

receita

class(receita)

(receita_prep <- prep(receita)) # prepara a receita definida acima

receita_prep$steps

tidy(receita_prep, number = 2)



treinamento_proc <- bake(receita_prep, new_data = NULL) # obtem os dados de treinamento processados
validacao_proc <- bake(receita_prep, new_data = validacao) # obtem os dados de validacao processados
teste_proc <- bake(receita_prep, new_data = teste) # obtem os dados de teste processados


# parsnip - modelos -------------------------------------------------------

# regressão linear

lm <- linear_reg() %>% # define um modelo de regressao linear
  set_engine("lm") # define a engine do modelo
lm

lm_fit <- linear_reg() %>% # define um modelo de regressao linear
  set_engine("lm") %>%  # define a engine do modelo
  fit(Balance ~ ., treinamento_proc) # executa o modelo e estima os parametros

lm_fit # estimativas do modelo ajustado

tidy(lm_fit) # estimativas do modelo ajustado em formato tidy

fitted_lm <- lm_fit %>% 
  predict(new_data = validacao_proc) %>% # realiza predicao para os dados de teste
  mutate(observado = validacao_proc$Balance, # cria uma coluna com o valor observado de Balance
         modelo = "lm") # cria uma coluna para indicar qual o modelo ajustado

head(fitted_lm) # mostra as 6 primeiras linhas do tibble criado

# grafico de dispersao entre valor observado e valor predito
fitted_lm %>% 
  ggplot(aes(observado, .pred)) + #eixo x observado, eixo y predito 
  geom_point(size = 2, col = "blue") + 
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(y = "Predito", x = "Observado") +
  xlim(0, 1600) +
  ylim(0, 1600)


# floresta aleatoria 

rf <- rand_forest() %>% # define o modelo floresta aleatoria
  set_engine("ranger", # define o pacote que vai fazer o ajuste do modelo
             importance = "permutation") %>%  #
  set_mode("regression") # define que é um modelo de regressao

rf

rf_fit <- rf %>% 
  fit(Balance ~ ., treinamento_proc) # ajuste do modelo definido acima
rf_fit


# importância das variaveis
vip(rf_fit)

fitted_rf <- rf_fit %>% 
  predict(new_data = validacao_proc) %>% # realiza predicao para os dados de teste
  mutate(observado = validacao_proc$Balance, # mesma estrutura do fitted_lm
         modelo = "random forest")

# grafico de dispersao entre valor observado e valor predito
fitted_rf %>% 
  ggplot(aes(observado, .pred)) + #eixo x observado, eixo y predito 
  geom_point(size = 2, col = "blue") + 
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(y = "Predito", x = "Observado") +
  xlim(0, 1600) +
  ylim(0, 1600)

fitted <- fitted_lm %>% 
  bind_rows(fitted_rf) # empilha o tibble fitted_rf abaixo do fitted_lm

head(fitted)
tail(fitted)
view(fitted)

# yardstick - avaliar desempenho ------------------------------------------

fitted %>% 
  group_by(modelo) %>% # agrupa pelo modelo ajustado
  metrics(truth = observado, estimate = .pred) # obtem as metricas de avaliacao dos modelos



# tune - ajuste de hiperparametros ----------------------------------------

rf2 <- rand_forest(mtry = tune(), # definicao da floresta aleatoria 
                   trees = tune(), # todos argumentos com tune() serao tunados a seguir  
                   min_n = tune()) %>% 
  set_engine("ranger") %>% # define qual função sera usada
  set_mode("regression") # define que e'  problema de regressao
rf2


# validação cruzada para ajuste de hiperparametros
set.seed(123)
cv_split <- vfold_cv(treinamento, v = 5)

# para tunar os parametros
rf_grid <- tune_grid(rf2, # especificacao do modelo
                     receita, # a receita a ser aplicada a cada lote
                     resamples = cv_split, # os lotes da validacao cruzada
                     grid = 5, # quantas combinacoes de parametros vamos considerar
                     metrics = metric_set(rmse, mae)) 

autoplot(rf_grid) # plota os resultados

rf_grid %>% 
  collect_metrics() 

rf_grid %>% 
  select_best(metric = "rmse") # seleciona a melhor combinacao de hiperparametros

best <- rf_grid %>% 
  select_best(metric = "rmse") # salva o melhor modelo na variavel best


# finaliza modelo
rf_fit2 <- finalize_model(rf2, parameters = best) %>% # informa os valores de hiperparametros a serem considerados
  fit(Balance ~ ., treinamento_proc) # executa o modelo com os valores de hiperparametros definidos acima

fitted_rf2 <- rf_fit2 %>% # faz previsao para os dados de teste
  predict(new_data = validacao_proc) %>% 
  mutate(observado = validacao_proc$Balance, 
         modelo = "random forest - tuned")

fitted <- fitted %>% # empilha as previsoes da floresta tunada
  bind_rows(fitted_rf2)

fitted %>% # obtem as metricas de todos os modelos ajustados
  group_by(modelo) %>% 
  metrics(truth = observado, estimate = .pred) %>% 
  arrange(.metric, .estimate)


# fit do melhor modelo ----------------------------------------------------
treinamento_validacao <- training(split1)
treinamento_validacao_proc <- bake(receita_prep, new_data = treinamento_validacao)
  
lm_fit_final <- linear_reg() %>% # define um modelo de regressao linear
  set_engine("lm") %>%  # define a engine do modelo
  fit(Balance ~ ., treinamento_validacao_proc) # executa o modelo e estima os parametros

fitted_teste <- lm_fit_final %>% 
  predict(new_data = teste_proc) %>% # realiza predicao para os dados de teste
  mutate(observado = teste_proc$Balance, # cria uma coluna com o valor observado de Balance
         modelo = "lm")

fitted_teste %>% 
  metrics(truth = observado, estimate = .pred)

# Tarefa
# implemente o boosting usando tidymodels e compare com os demais modelos
# para ajudar, veja https://parsnip.tidymodels.org/reference/boost_tree.html
