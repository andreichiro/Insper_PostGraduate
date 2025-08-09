#' @title Exercicio 1 aula 2
#' @description Valores preditos para x = 0.5, 2.1 e 4 usando KNN
#' @author André Katsurada
#' @date 06/ago/25
#' @course APRENDIZAGEM ESTATÍSTICA DE MÁQUINA I

suppressPackageStartupMessages({
  pkgs <- c("R6", "tibble", "dplyr")
  for (p in pkgs) {
    if (!requireNamespace(p, quietly = TRUE)) {
      install.packages(p, repos = "https://cloud.r-project.org", type = "binary")
    }
  }
})

#Classe abstrata inicial
DistanceStrategy <- R6::R6Class(
  "DistanceStrategy",
  public = list(
    distance = function(a, b) {
      stop("Método abstrato – implemente em subclasses.")
    }
  )
)

#Distância Euclidiana em 1‑D 
Euclidean1D <- R6::R6Class(
  "Euclidean1D",
  inherit = DistanceStrategy,
  public = list(
    distance = function(a, b) abs(a - b)         
  )
)


#KNN 
KNNRegressor <- R6::R6Class(
  "KNNRegressor",

  private = list(
    x = NULL,
    y = NULL,
    dist_strategy = NULL
  ),

  public = list(

    #' @description Construtor
    #' @param x Vetor de features
    #' @param y Vetor de targets 
    #' @param dist_strategy Objeto que implementa método da distância
    initialize = function(x, y, dist_strategy = Euclidean1D$new()) {
      stopifnot(length(x) == length(y))
      private$x <- x
      private$y <- y
      private$dist_strategy <- dist_strategy
    },

    #' @description Predição k‑NN
    #' @param query Vetor de pontos x onde se deseja prever y
    #' @param k Número de vizinhos
    #' @return Vetor de predições
    predict = function(query, k = 1L) {
      vapply(query, function(q) {
        d <- private$dist_strategy$distance(private$x, q)        #Distância
        idx <- order(d)[seq_len(k)]                              #k + próximos
        mean(private$y[idx])                                     #Média y
      }, numeric(1))
    }
  )
)

#Resolução
ListaSolver <- R6::R6Class(
  "ListaSolver",

  private = list(
    #Infos iniciais  
    dados_treino = tibble::tibble(
      x = c(1, 2, 3),
      y = c(1, 4, 9)
    ),

    consultas = c(0.5, 2.1, 4.0)
  ),

  public = list(

    #' @title Exercício 1 – KNN em 1‑D
    #' @description Calcula predições para k = 1, 2, 3
    ex1 = function() {

      #Instancia injetando dependência
      knn <- KNNRegressor$new(
        x = private$dados_treino$x,
        y = private$dados_treino$y,
        dist_strategy = Euclidean1D$new()
      )

      ks <- c(1L, 2L, 3L)

      #Matriz 3×3
      preds <- sapply(ks, function(k) knn$predict(private$consultas, k))
      colnames(preds) <- paste0("k =", ks)
      rownames(preds) <- paste0("x = ", private$consultas)

      #Saída arredondada 
      preds <- round(preds, 3)

      message("\nPredições KNN")
      print(preds)

      invisible(preds)       
    }
  )
)

# Execução automática quando o script for rodado
solver <- ListaSolver$new()
solver$ex1()

#P/ executar
# source("ListaKNN_Ex1.R")
# solver <- ListaSolver$new()
# solver$ex1()
