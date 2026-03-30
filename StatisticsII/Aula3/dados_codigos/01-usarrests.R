# Aprendizagem Estatistica de Maquina II
# Aula 3
# PCA - USArrests

# Pacotes que serao utilizados --------------------------------------------

library(tidyverse)
library(GGally)
library(ggrepel)
library(factoextra)

data("USArrests") # carregar dados
?USArrests

View(USArrests)

ggpairs(USArrests) #corrplot

USArrests |> 
  rownames_to_column(var="State") |> 
  View()


USArrests |> 
  rownames_to_column(var="State") |> 
  ggplot(aes(Murder, Assault, label=State)) +
  geom_point(size = 2, alpha = 0.3)+
  geom_text_repel()


# PCA considerando apenas 2 colunas ---------------------------------------

# o codigo abaixo centraliza os dados e salva no dataframe X
X <- USArrests |> 
  select(Murder, Assault) |> 
  scale(center=TRUE, scale=FALSE) |>  
  as.data.frame()

View(X)

apply(X, 2, mean) # obtem a media das colunas de X
apply(X, 2, sd) # obtem o desvio padrao das colunas de X

# forma alternativa de obter media e desvio das colunas de X
X |> 
  summarise(across(where(is.numeric), list(media = mean, desvio = sd)))

head(X) # mostra as 6 primeiras linhas de X

summary(X)

pca <- prcomp(X) # aplica o PCA

Phi <- pca$rotation # matriz de cargas/loadings
Phi

# cada phi e' um vetor unitario
apply(Phi, 2, function(col) sum(col^2)) # calcula soma de phi^2

Z <- pca$x # projecoes/scores

head(Z) # cabecalho das projecoes/scores

# o grafico de dispersao abaixo mostra os dados na escala original
USArrests |> 
  rownames_to_column() |> # adiciona uma coluna com o none das linhas (nome dos estados)
  ggplot(aes(Murder, Assault, label = rowname)) +
  geom_point() +
  geom_text_repel() + # adiciona o nome do estado no grafico de dispersao
  labs(title = "Dados na escala original")

# o grafico de dispersao abaixo mostra os dados centralizados
X |> 
  rownames_to_column() |> 
  ggplot(aes(Murder, Assault, label = rowname)) +
  geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) + # linha horizontal na origem
  geom_vline(xintercept = 0, linetype = "dashed", alpha = 0.5) + # linha vertical na origem
  geom_point() +
  # geom_text_repel() +
  labs(title = "Dados centralizados")

# o grafico de dispersao abaixo dados obtidos com o PCA
Z |> 
  as.data.frame() |> 
  rownames_to_column() |> 
  ggplot(aes(PC1, PC2, label=rowname)) +
  geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
  geom_vline(xintercept = 0, linetype = "dashed", alpha = 0.5) +
  geom_point() + 
  # geom_text_repel() +
  labs(title = "Projecao PCA")

# PCA considerando todas colunas ------------------------------------------

X <- scale(USArrests, 
           center = TRUE, # centraliza os dados
           scale = TRUE) # escalona os dados (pois estao em medidas diferentes)

pca <- prcomp(X) # aplica o PCA

pca

pca$rotation <- -pca$rotation # troca o sinal das cargas
pca$x <- -pca$x # troca o sinal dos scores

Phi <- pca$rotation # matriz de cargas
Phi

Z <- pca$x # matriz de scores
head(Z)

# o grafico abaixo mostra as cargas e os scoroes da PCA em um unico plot
# os valores nos eixos superior e direito sao as cargas
# os valores nos eixos esquerdo e inferior sao os scores
biplot(pca, scale = 0, cex = 0.75,
       xlab = "PC1",
       ylab = "PC2")

# biplot do factoextra
fviz_pca_biplot(pca, repel = TRUE, 
                xlab = "PC1 - Criminalidade", 
                ylab = "PC2 - Urbanizacao")

# para visualizar as cargas graficamente
fviz_pca_var(pca, repel = TRUE, geom = c("arrow", "text"))

# o grafico abaixo mostra o percentual explicado da variancia de cada componente
fviz_eig(pca, addlabels = TRUE) + 
  labs(x = "Componente Principal",
       y = "Percentual explicado da variancia")

# abaixo obtemos a soma acumulada do percentual explicado da variancia
pca$sdev
round(cumsum(pca$sdev^2) / sum(pca$sdev^2), 4)

# Tarefa: rode novamente o codigo acima considerando como X a 
# base de dados original (sem escalonar os dados)

