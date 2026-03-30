# Aprendizagem Estatistica de Maquina II
# Aula 3
# PCA em modelagem

# Pacotes que serao utilizados --------------------------------------------

library(MASS)
library(tidymodels)
library(ggcorrplot)
library(vip)
library(factoextra)

conflicted::conflict_prefer("select", "dplyr")


head(Boston) # cabecalho dos dados Boston

# grafico de correlacao entre as preditoras
Boston |> 
  select(-medv) |> # remove a variavel resposta
  cor() |>  # calcula as correlacoes
  ggcorrplot(lab = TRUE, lab_size = 4, tl.cex = 9,
             colors = c("#FC4E07", "#fdfdfd", "#00AFBB"))

# grafico de correlacao entre as componentes principais
prcomp(Boston, scale = TRUE)$x |> # calcula PCA e obtem os scores
  cor() |> # calcula as correlacoes
  ggcorrplot(lab = TRUE, lab_size = 4, tl.cex = 9,
             colors = c("#FC4E07", "#fdfdfd", "#00AFBB"))

# treinamento e teste -----------------------------------------------------

set.seed(321)

split <- initial_split(Boston, prop = 0.8)

treinamento <- training(split)
teste <- testing(split)


# processamento -----------------------------------------------------------

receita <- recipe(medv ~ ., treinamento) |> 
  step_normalize(all_predictors()) |> # normaliza todas preditoras
  step_pca(all_predictors(), # aplica o PCA em todas as preditoras
           num_comp = 6) # obtem 6 primeiras componentes

receita_prep <- prep(receita) # prepara a receita

treinamento_proc <- bake(receita_prep, new_data = NULL) # aplica a receita no treino

teste_proc <- bake(receita_prep, new_data = teste) # aplica a receita no teste

# para obter as informacoes do PCA no tidymodels:
receita_prep$steps[[2]]$res
fviz_eig(receita_prep$steps[[2]]$res, addlabels = TRUE) #grafico PVE

# regressão linear --------------------------------------------------------

lm_fit <- linear_reg() |> 
  set_engine("lm") |> 
  fit(medv ~ ., treinamento_proc)

lm_fit # modelo estimado

tidy(lm_fit) # organiza as estimativas dos coeficientes do modelo

lm_fit |> 
  predict(new_data = teste_proc) # predicao no teste

# Tarefa: ajustar o modelo linear sem o PCA e comparar com o resultado
# acima usando uma metrica adequada
