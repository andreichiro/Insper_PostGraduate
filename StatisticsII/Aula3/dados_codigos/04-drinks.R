# Aprendizagem Estatistica de Maquina II
# Aula 4
# k-means - consumo de alcool

# Fonte
# https://github.com/fivethirtyeight/data/tree/master/alcohol-consumption
# https://fivethirtyeight.com/features/dear-mona-followup-where-do-people-drink-the-most-beer-wine-and-spirits/


# Pacotes que serao utilizados --------------------------------------------
library(tidyverse)
library(GGally)
library(factoextra)
library(ggrepel)
library(plotly)
library(cluster)

# leitura e mudanca de tipo da variavel total_litres_of_pure_alcohol
dados <- read_csv2("drinks.csv") |> 
  mutate(total_litres_of_pure_alcohol = as.numeric(total_litres_of_pure_alcohol))

dados

# renomeando colunas
dados <- dados |> 
  rename(beer = beer_servings, 
         spirit = spirit_servings, 
         wine = wine_servings, 
         total_litres_alcohol = total_litres_of_pure_alcohol)

# primeiras linhas dos dados
dados |> 
  slice_head(n = 5)

# plot por pares de variaveis
dados |> 
  select(-country) |> 
  ggpairs()

# o que esse codigo faz?
dados |> 
  select(country, wine) |> 
  arrange(desc(wine)) |> 
  slice_head(n = 10)

# o que esse codigo faz?
dados |> 
  select(country, beer) |> 
  arrange(desc(beer)) |> 
  slice_head(n = 10)

# o que esse codigo faz?
dados |> 
  select(country, spirit) |> 
  arrange(desc(spirit)) |> 
  slice_head(n = 10)

# padronizacao dos dados
dados_pad <- dados |> 
  select(-country) |> 
  scale()

# kmeans ------------------------------------------------------------------
set.seed(1)
k_medias <- kmeans(dados_pad, 
                   centers = 2, # k
                   nstart = 10) # obtem o kmeans considerando 2 clusters

# armazena em um mesmo tibble os dados e o cluster estimado
(auxiliar <- tibble(cluster = k_medias$cluster) |> 
  bind_cols(as_tibble(dados_pad)))
  
# total SS
k_medias$totss # W geral

k_medias$centers

# o codigo abaixo faz a conta da soma de quadrados total (mesmo que o numero acima)
auxiliar |> 
  transmute(beer = (beer - mean(beer))^2, 
            spirit = (spirit - mean(spirit))^2, 
            wine = (wine - mean(wine))^2,
            total_litres_alcohol = (total_litres_alcohol - mean(total_litres_alcohol))^2, 
            totss = beer + spirit + wine + total_litres_alcohol) |> 
  summarise(totss = sum(totss))


# within SS
k_medias$withinss |> 
  sum()

# o codigo abaixo faz a conta da soma de quadrados total intracluster (mesmo que o numero acima)
auxiliar |> 
  group_by(cluster) |> 
  transmute(beer = (beer - mean(beer))^2, 
            spirit = (spirit - mean(spirit))^2, 
            wine = (wine - mean(wine))^2,
            total_litres_alcohol = (total_litres_alcohol - mean(total_litres_alcohol))^2, 
            withinss = beer + spirit + wine + total_litres_alcohol) |> 
  summarise(withinss = sum(withinss))

# tot within SS
k_medias$withinss
k_medias$tot.withinss


# Analise do numero de clusters -------------------------------------------

set.seed(123)

k <- 2:20
tibble(k = k) |> 
  mutate(w = map_dbl(k, ~ kmeans(dados_pad, centers = .x,
                                 nstart = 10)$tot.withinss)) |> 
  ggplot(aes(k, w)) + 
  geom_point() + 
  scale_x_continuous(breaks = k) +
  geom_line()


# especificar k com base no grafico de cotovelo acima
set.seed(123)
(descricao <- dados |> 
  mutate(cluster = factor(kmeans(dados_pad, centers = 5, nstart = 10)$cluster)))

# visualizacao dos clusters na escala original dos dados
# total de litros de alcool por cluster
descricao |> 
  mutate(cluster = fct_reorder(cluster,  
                               total_litres_alcohol,
                               .fun = median)) |> 
  ggplot(aes(total_litres_alcohol,
             cluster,
             fill = cluster, group = cluster)) + 
  geom_boxplot(show.legend = F) +
  labs(x = "Total litres of alcohol", y = "Cluster")

# consumo de cerveja e vinho por cluster
descricao |> 
  ggplot(aes(wine, beer, color = cluster)) + 
  geom_point(size = 5, alpha = .3) +
  theme_bw() 

# consumo de vinho e spirit
g1 <- descricao |> 
  ggplot(aes(spirit, wine, color = cluster, label = country)) + 
  geom_point() 

descricao |> 
  filter(country == "Brazil")

ggplotly(g1)

# calculo de medias de cada grupo
descricao |>
  group_by(cluster) |> 
  summarise(across(where(is.numeric), mean))

# visualizando caracteristicas de cada cluster
descricao |>
  group_by(cluster) |> 
  summarise(across(where(is.numeric), mean)) |> 
  pivot_longer(-cluster) |> 
  ggplot(aes(name, value, group = cluster, color = cluster)) + 
  geom_line() + 
  geom_point() +
  labs(x = "", y = "Valor medio", color = "Grupo")

# visualizando caracteristicas de cada cluster (dados padronizados)
descricao |> 
  mutate(across(where(is.numeric), scale)) |> 
  group_by(cluster) |> 
  summarise(across(where(is.numeric), mean)) |> 
  pivot_longer(-cluster) |> 
  ggplot(aes(name, value, group = cluster, color = cluster)) + 
  geom_line(size = 2) + 
  geom_point(size = 5) +
  labs(x = "", y = "Valor medio (padronizado)", color = "Grupo")

# silhouette --------------------------------------------------------------

# obtem os coeficientes
res_sil <- silhouette(as.numeric(descricao$cluster), 
                      dist(dados_pad)^2)

# obtem a media do coeficiente por cluster
fviz_silhouette(res_sil)

# grafico silhouette
fviz_nbclust(dados_pad, kmeans, method = "silhouette")

# grupos ------------------------------------------------------------------

set.seed(1)
avaliacao <- descricao |> 
  mutate(cluster = factor(kmeans(dados_pad, centers = 3)$cluster))

avaliacao |> 
  ggplot(aes(total_litres_alcohol, 
             fct_reorder(cluster, total_litres_alcohol, .fun = median), 
             fill = cluster, group = cluster)) + 
  geom_boxplot(show.legend = FALSE) +
  labs(x = "Total litres of alcohol", y = NULL)

# nomeando os clusters encontrados
avaliacao <- avaliacao |> 
  mutate(cluster = case_when(cluster == 2 ~ "alto", 
                             cluster == 1 ~ "intermediario", 
                             TRUE ~ "baixo"))

# visualizando caracteristicas de cada cluster
avaliacao |> 
  group_by(cluster) |> 
  summarise(across(where(is.numeric), mean)) |> 
  pivot_longer(-cluster) |> 
  ggplot(aes(name, value, group = cluster, color = cluster)) + 
  geom_line() + 
  geom_point()

# visualizando caracteristicas de cada cluster  (dados padronizados)
avaliacao |> 
  mutate(across(where(is.numeric), scale)) |> 
  group_by(cluster) |> 
  summarise(across(where(is.numeric), mean)) |> 
  pivot_longer(-cluster) |> 
  ggplot(aes(name, value, group = cluster, color = cluster)) + 
  geom_line(size = 2) + 
  geom_point(size = 5)

