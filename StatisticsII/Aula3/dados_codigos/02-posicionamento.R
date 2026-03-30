# Aprendizagem Estatistica de Maquina II
# Aula 3
# PCA - Posicionamento de marcas

# Pacotes que serao utilizados --------------------------------------------

library(tidyverse)
library(factoextra)
library(ggrepel)

# avaliações --------------------------------------------------------------

avaliacoes <- read_csv("avaliacoes.csv") # carregar avaliacoes

avaliacoes # cada respondente respondeu todas as perguntas para cada marca que ele/ela conhece

# o grafico abaixo mostra em percentual quais as marcas foram mais avaliadas
avaliacoes |>
  count(marca) |> # obtem o numero de respondentes para cada marca
  mutate(percentual = 100 * n / sum(n)) |> # calcula o percentual equivalente
  ggplot(aes(x = reorder(marca, n), 
             y = percentual)) +
  geom_col(fill = "steelblue", color = "black") + 
  labs(x="Marcas", y="Percentual das avaliacoes",
         title = "Marcas de alvejante avaliadas") +
  coord_flip() + 
  theme_classic()

# o grafico abaixo mostra em percentual o numero de marcas que cada respondente avaliou
avaliacoes |>
  group_by(respondente) |>
  summarise(num_marcas_avaliadas = n()) |> # conta quantas marcas cada respondente avaliou
  count(num_marcas_avaliadas) |> # conta o numero de marcas avaliadas por respondente
  mutate(percentual = 100 * n / sum(n)) |>  # calcula o percentual equivalente
  ggplot(aes(x = num_marcas_avaliadas, y = percentual)) +
  geom_col(fill = "steelblue", color = "black") + 
  labs(x = "Numero de marcas avaliadas pelos respondentes",
       y = "Percentual dos respondentes",
       title = "Distribuicao do numero de marcas avaliadas") +
  theme_classic()


# questões ----------------------------------------------------------------

questoes <- read_csv2("questoes.csv", locale = locale(encoding = "UTF-8"))
questoes$pergunta[1]

# PCA ---------------------------------------------------------------------

pca <- avaliacoes |>
  select(starts_with("Q")) |> # somente as colunas que tem as notas das questoes
  prcomp(scale = TRUE) # aplica PCA


# o grafico abaixo mostra o percentual explicado da variancia de cada componente
fviz_eig(pca, addlabels = TRUE, 
         ncp = nrow(questoes)) + # ncp - numero de componentes mostrados
  labs(x = "Componente Principal",
       y = "Percentual explicado da variancia")

# abaixo obtemos a soma acumulada do percentual explicado da variancia
(cumsum(pca$sdev^2) / sum(pca$sdev^2))[1:10]

Phi <- pca$rotation # matriz de cargas

Z <- pca$x

# interpretando os drivers ------------------------------------------------

# contribuicoes das questoes para a primeira componente
pca |> 
  fviz_contrib(choice = "var", axes = 1, sort.val = "asc",
               fill = "steelblue", color = "black") +
  labs(x = "", title = "Contribuicoes das questoes para o primeiro driver") +
  coord_flip()

pca |> 
  fviz_contrib(choice = "var", axes = 2, sort.val = "asc",
               fill = "steelblue", color = "black") +
  labs(x = "", title = "Contribuicoes das questoes para o segundo driver") +
  coord_flip()

# a funcao abaixo obtem o valor das cargas em uma componente principal (driver)
# especifica, faz um merge com as questoes e devolve um tibble ordenado
# segundo a contribuicao de cada questao
get_driver <- function(Phi, questoes, drv, top) {
  tibble(numero = rownames(Phi), 
         carga = Phi[, drv]) |>
    left_join(questoes) |>
    mutate(contribuicao = 100 * carga^2 / sum(carga^2)) |>
    arrange(desc(contribuicao)) |>
    top_n(top)
} 

colnames(Z) <- paste0("driver_", 1:ncol(Z))

# obtem as 6 perguntas que mais contribuiram para a primeira componente principal
driver_1 <- get_driver(Phi, questoes, drv = 1, top = 8)
driver_1

# como as cargas estao negativas, vamos trocar o sinal
# temos que trocar o sinal das cargas
Phi[, 1] <- -Phi[, 1]

# e tambem o sinal dos scores
Z[, 1] <- -Z[, 1]

(driver_1 <- get_driver(Phi, questoes, drv = 1, top = 6)) # qual nome podemos dar a este driver?

(driver_2 <- get_driver(Phi, questoes, drv = 2, top = 10)) # qual nome podemos dar a este driver?

(driver_3 <- get_driver(Phi, questoes, drv = 3, top = 10)) # qual nome podemos dar a este driver?

(driver_4 <- get_driver(Phi, questoes, drv = 4, top = 10)) # qual nome podemos dar a este driver?


View(Z)

# Posicionamento de marcas ------------------------------------------------

# tibble com todas as avaliacoes: a respectiva marca avaliada e as cargas 
# correspondentes nas componentes principais (apenas as 3 primeiras)
tb <- tibble(marca = avaliacoes$marca) |>
  bind_cols(as_tibble(Z[,1:3]))

tb |>
  arrange(marca)

# o grafico abaixo mostra o score medio das avaliacoes de cada marca
# para cada driver. Assim, temos um comparativo do posicionamento 
# das marcas segundo esta pesquisa
tb |>
  group_by(marca) |> # agrupa pela marca
  summarise_all(mean) |> # obtem a media do score de todas avaliacoes
  pivot_longer(-marca, 
               names_to = "driver", 
               values_to = "score_medio") |> 
  ggplot(aes(x = driver, y = score_medio, 
             group = marca, color = marca,
             label = ifelse(driver == "driver_1", 
                            marca, ""))) +
  geom_line(size = 1, alpha = 0.55) +
  geom_point(size = 2) +
  labs(x = "", y = "Score Medio", 
       title = "Posicionamento das marcas de alvejante") +
  geom_label_repel(direction = "both") +
  scale_x_discrete(labels = c("PC1 - Limpeza", 
                              "PC2 - Suavidade/Aroma", 
                              "PC3 - Intensidade")) +
  theme_classic() + 
  theme(legend.position = "none") 
