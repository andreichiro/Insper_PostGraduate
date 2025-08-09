#' @title lista_exercicios_regressao.R
#' @description Lista de exercícios 15
#' @author André Katsurada
#' @date 30/jul/25
#' @course APRENDIZAGEM ESTATÍSTICA DE MÁQUINA I

suppressPackageStartupMessages({
  pkgs <- c("R6", "tibble", "dplyr", "ggplot2", "scales")
  for (p in pkgs) {
    if (!requireNamespace(p, quietly = TRUE)) {
      stop(sprintf("O pacote %s é necessário para execução.", p))
    }
  }
})

# Classe p/ prever vendas (R6)        #
RegressaoSolver <- R6::R6Class(
  "RegressaoSolver",

  private = list(
    #Reprodutibilidade 
    set_rng = function(seed = NULL) {
      if (!is.null(seed)) set.seed(seed)
      RNGkind("Mersenne-Twister", "Inversion")
    },

    #Exporta gráficos em PNG 
    save_png = function(filename, plot, width = 4, height = 3) {
      if (isTRUE(self$save_plots))
        ggplot2::ggsave(filename, plot, width = width, height = height)
    }
  ),

  public = list(
    #Flag global p/ exportar gráficos                           
    save_plots = TRUE,
    initialize = function(save_plots = TRUE)
      self$save_plots <- isTRUE(save_plots),

    #' @title Exercício 15 – previsão de venda
    #' @description
    #' Predições de vendas seguindo a fórmula: 10 + 1.5*Mkt + 2*Satisfacao
    ex15 = function(seed = NULL) {
      private$set_rng(seed)

      #intercepto: vendas esperadas (10 mil) qdo marketing = 0 e
      #satisfação = 0 (baseline)
      interpret_intercepto <- paste(
        "Intercepto = 10: se Marketing = 0 e Satisfação = 0,",
        "espera‑se vender 10 mil reais.")
      interpret_marketing  <- paste(
        "Coef. Marketing = 1,5: cada 1 mil investido eleva as vendas em 1,5 mil,",
        "mantendo Satisfação constante.")
      interpret_satisfacao <- paste(
        "Coef. Satisfação = 2: cada ponto extra no índice",
        "aumenta as vendas em 2 mil, mantendo Marketing fixo.")

      #2Clientes
      clientes <- tibble::tibble(
        cliente      = c("A",  "B",  "C"),
        marketing_k  = c(12,    7,   10),  
        satisfacao   = c( 8,    5,    9)   
      )

      #Vendas
      clientes_pred <- clientes |>
        dplyr::mutate(
          venda_pred_k = 10 + 1.5 * marketing_k + 2 * satisfacao
        )

      plt <- ggplot2::ggplot(
        clientes_pred,
        ggplot2::aes(x = cliente, y = venda_pred_k, fill = cliente)) +
        ggplot2::geom_col() +
        ggplot2::scale_y_continuous(labels = scales::comma) +
        ggplot2::labs(title = "Vendas preditas por cliente",
                      x = "Cliente", y = "Vendas (mil R$)") +
        ggplot2::theme(legend.position = "none")

      private$save_png("ex12_vendas_predit.png", plt, 4.5, 3.5)

      #Resultados
      list(
        interpretacoes = list(
          intercepto = interpret_intercepto,
          marketing  = interpret_marketing,
          satisfacao = interpret_satisfacao
        ),
        tabela_predicoes = clientes_pred,
        grafico_predicoes = plt
      )
    }
  ) 
)

# Demo
print_sep <- function(n)
  cat("\n", strrep("─", 20), " Exercício", n, "\n")

solver <- RegressaoSolver$new()
print_sep(15)
res15 <- solver$ex15()
cat(res15$interpretacoes$intercepto, "\n")
cat(res15$interpretacoes$marketing,  "\n")
cat(res15$interpretacoes$satisfacao, "\n\n")
print(res15$tabela_predicoes)

#P/ rodar:
#source("Lista1_12.R") ou Rscript Lista1_12.R
