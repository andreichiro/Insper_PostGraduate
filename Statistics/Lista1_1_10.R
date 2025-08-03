#' @title Lista1_1_10.R
#' @description Lista de exercícios 1 a 10 da lista 1
#' @author André Katsurada
#' @date 30/jul/25
#' @course APRENDIZAGEM ESTATÍSTICA DE MÁQUINA I

suppressPackageStartupMessages({
  pkgs <- c("R6", "ggplot2", "dplyr", "tibble", "scales", "bit64")
  for (p in pkgs) {
    if (!requireNamespace(p, quietly = TRUE)) {
      install.packages(p, repos = "https://cloud.r-project.org", type = "binary")
    }
  }
})

# Classe p/ a lista de exercícios (R6)     
ListaSolver <- R6::R6Class(
  "ListaSolver",

  private = list(
    #Reprodutibilidade
    set_rng = function(seed = NULL) {
      if (!is.null(seed)) set.seed(seed)
      RNGkind("Mersenne-Twister", "Inversion")
    },

    #P/ salvar gráficos em png
    save_png = function(filename, plot, width = 4, height = 3) {
      if (isTRUE(self$save_plots))
        ggplot2::ggsave(filename, plot, width = width, height = height)
    }

  ),

  public = list(
    #Flag p/ salvar gráficos em png como default
    save_plots = TRUE,
    initialize = function(save_plots = TRUE) self$save_plots <- isTRUE(save_plots),

    #' @title Exercício 1 – vetor 1:20 e filtro > 10
    #' @description
    #' Cria um vetor inteiro de 1 a 20 e retorna um outro vetor apenas com os números maiores que 10
    ex1 = function(seed = NULL) {
      private$set_rng(seed)

      v <- 1:20                    # cria o vetor de 1 a 20
      v_gt10 <- v[v > 10]          # indexação 

      df_plot <- data.frame(x = v, gt10 = v > 10)
      plt <- ggplot2::ggplot(df_plot, ggplot2::aes(x = x, y = 0, colour = gt10)) +
        ggplot2::geom_point() +
        ggplot2::labs(title = "Ex1 – Vetor 1:20 (filtro >10)", x = NULL, y = NULL, colour = "Maior que 10?")
      private$save_png("ex1_vetor.png", plt)
      

      list(original = v, filtrado = v_gt10)
    },

    #' @title Exercício 2 – amostra uniforme inteira + filtro 
    #' @description
    #' Gera n inteiros de 1 a 100 e devolve maiores que 50 e menores que 75
    ex2 = function(n = 100, seed = NULL) {
      private$set_rng(seed)
      v <- sample.int(100L, size = n, replace = TRUE)
      v_filtro <- v[v > 50 & v < 75]

      plt <- ggplot2::ggplot(data.frame(x = v_filtro),
             ggplot2::aes(x = x)) +
        ggplot2::geom_histogram(bins = 15) +
        ggplot2::labs(title = "Ex2 – Valores entre 51 e 74",
                      x = "Valor", y = "Frequência")
      private$save_png("ex2_hist.png", plt)
    

      list(vetor = v, filtrado = v_filtro)
    },

    #' @title Exercício 3 – pares 1:20
    #' @description
    #' Cria um vetor de 1 a 20 e retorna outro vetor com os números pares
    ex3 = function() {
      v <- 1:20
      pares <- v[v %% 2L == 0L]     # resto zero -> par

      plt <- ggplot2::ggplot(data.frame(x = pares),
             ggplot2::aes(x = x, y = 0)) +
        ggplot2::geom_point() +
        ggplot2::labs(title = "Ex3 – Números pares de 1:20",
                      x = "Valor", y = NULL)
      private$save_png("ex3_pares.png", plt)


      list(original = v, pares = pares)
    },

    #' @title Exercício 4 – runif, média e desvio‑padrão
    #' @description
    #' Gera um vetor de n numeros e calcula a média e desvio‑padrão
    ex4 = function(n = 50, seed = NULL) {
      private$set_rng(seed)
      v <- runif(n, 0, 1)

      #sd() usa n‑1 no denominador
      list(vetor = v, media = mean(v), desvio_padrao = sd(v))
    },

    #' @title Exercício 5 – rnorm, proporção > lim
    #' @description
    #' Gera n números e calcula a proporção de quais são maiores que lim

    ex5 = function(n = 1000, mu = 50, sigma = 10, lim = 60, seed = NULL) {
      private$set_rng(seed)

      v <- rnorm(n, mu, sigma)
      prop <- mean(v > lim)         # TRUE=1, FALSE=0
      list(vetor = v, proporcao_maior_que_limite = prop)
    },

    #' @title Exercício 6 – métricas sobre vetor 
    #' @description
    #' Calcula média, desvio‑padrão, comprimento do vetor, e retorna o seu terceiro elemento

    ex6 = function() {
      v <- c(15, 10, 21, 30, 52, 60)
      list(
        media              = mean(v),
        #alternative: sd(v) * sqrt((length(v)-1)/length(v))
        desvio_padrao      = sd(v),
        terceiro_elemento  = v[3],       
        comprimento        = length(v)
      )
    },

    #' @title Exercício 7 – matriz 3×3 e soma por linha
    #' @description
    #' Cria uma matriz 3×3 e retorna somatórias das linhas
    ex7 = function() {
      m <- matrix(1:9, nrow = 3, byrow = TRUE)
      list(matriz = m, soma_linhas = rowSums(m))
    },

    #' @title Exercício 8 – sequência de Fibonacci até n
    #' @description
    #' Calcula a sequência de Fibonacci até n, faz casting para numeric evitando um overflow
    ex8 = function(n) {
      stopifnot(n >= 1L)

      #Retorno para n = 1 ou 2
      if (n <= 2L) return(c(0L, 1L)[1:n])

      #Retorno dependendo do vetor
      alloc_fib <- function(n) {
        if (n <= 47L)                return(integer(n))           # 32‑bit 
        if (n <= 92L)                return(bit64::integer64(n))  # 64‑bit 
        warning("n > 92: usando numeric para evitar overflow 64‑bit")
        numeric(n)                
      }

      fib <- alloc_fib(n)

      #Inicializa os dois primeiros termos
      fib[1:2] <- if (inherits(fib, "integer64"))
                     bit64::as.integer64(c(0, 1))
                   else
                     c(0, 1)

      for (i in 3:n) fib[i] <- fib[i - 1] + fib[i - 2]
      fib
    },

    #' @title Exercício 9 – tibble e gráficos          
    #' @description
    #' Cria um tibble com diversas métricas e plota os gráficos correspondentes
    ex9 = function() {

      df <- tibble::tibble(
        id        = 1:10,
        idade     = c(35,24,31,29,20,19,42,54,49,60),
        categoria = factor(
          c("tecnologia","romance","ficção","tecnologia","tecnologia",
            "tecnologia","ficção","romance","tecnologia","ficção")),
        satisfacao = c(70,80,93,79,95,88,85,91,100,79)
      )

      n_linhas <- nrow(df); n_colunas <- ncol(df)
      vetor_idades <- df$idade
      linhas_1_2 <- df[1:2, ]
      idade_media <- mean(df$idade)
      satisfacao_mediana <- median(df$satisfacao)
      contagem_categoria <- dplyr::count(df, categoria)
      idade_media_categoria <- df |>
        dplyr::group_by(categoria) |>
        dplyr::summarise(idade_media = mean(idade), .groups = "drop")

      grafico_base <- ggplot2::ggplot(
        df, ggplot2::aes(x = idade, y = satisfacao)) +
        ggplot2::geom_point() +
        ggplot2::labs(title = "Idade vs. Satisfação",
                      x = "Idade", y = "Satisfação")

      grafico_color <- ggplot2::ggplot(
        df, ggplot2::aes(x = idade, y = satisfacao, colour = categoria)) +
        ggplot2::geom_point(size = 3) +
        ggplot2::labs(title = "Idade vs. Satisfação por Categoria",
                      x = "Idade", y = "Satisfação",
                      colour = "Categoria")


      grafico_trend <- grafico_color +
        ggplot2::geom_smooth(method = "lm", se = FALSE)

      private$save_png("ex9_color.png",  grafico_color,  4.5, 3.5)
      private$save_png("ex9_trend.png",  grafico_trend,  4.5, 3.5)

      list(
        df                       = df,
        n_linhas                 = n_linhas,
        n_colunas                = n_colunas,
        vetor_idades             = vetor_idades,
        linhas_1_2               = linhas_1_2,
        idade_media              = idade_media,
        satisfacao_mediana       = satisfacao_mediana,
        contagem_categoria       = contagem_categoria,
        idade_media_categoria    = idade_media_categoria,
        grafico_base             = grafico_base,
        grafico_color            = grafico_color,
        grafico_trend            = grafico_trend
      )
    },

    #' @title Exercício 10 – dataset diamonds                          #
    #' @description
    #' Calcula diversas métricas sobre o dataset diamonds e plota os gráficos correspondentes

    ex10 = function() {

      data("diamonds", package = "ggplot2")   # 53 940 observações

      #a) Frequência por cor
      freq_color <- dplyr::count(diamonds, color, sort = TRUE)

      #b) Combinação categoria × cor 
      count_cut_color <- dplyr::count(diamonds, cut, color, sort = TRUE)

      #c) Razão entre preço e   
      diamonds2 <- dplyr::mutate(diamonds, ratio = price / table)

      #d) Média e sd de preço por categoria de corte
      stats_cut <- diamonds2 |>
        dplyr::group_by(cut) |>
        dplyr::summarise(
          media_preco = mean(price),
          sd_preco    = sd(price),
          .groups     = "drop")

      #e) Mesmo cálculo do item "d", mas apenas para table > 50 
      stats_cut_table <- diamonds2 |>
        dplyr::filter(table > 50) |>
        dplyr::group_by(cut) |>
        dplyr::summarise(
          media_preco = mean(price),
          sd_preco    = sd(price),
          .groups     = "drop")

      #f) Gráfico de preço médio por categoria 
      grafico_colunas_cut <- ggplot2::ggplot(
        stats_cut,
        ggplot2::aes(x = cut, y = media_preco, fill = cut)) +
        ggplot2::geom_col() +
        ggplot2::scale_y_continuous(labels = scales::comma) +
        ggplot2::labs(title = "Preço médio por Cut", y = "Preço (R$)") +
        ggplot2::theme(legend.position = "none")

      #g) Scatter plot preço × profundidade 
      grafico_price_depth <- ggplot2::ggplot(
        diamonds2, ggplot2::aes(x = depth, y = price)) +
        ggplot2::geom_point(alpha = .3) +
        ggplot2::labs(title = "Preço vs. Profundidade",
                      x = "Profundidade", y = "Preço")

      #h) Dispersão por categoria 
      grafico_price_depth_facete <- grafico_price_depth +
        ggplot2::facet_wrap(~ cut)

      private$save_png("ex10_colunas.png", grafico_colunas_cut, 5, 4)

      #Lista de resultados
      list(
        freq_color                  = freq_color,
        count_cut_color             = count_cut_color,
        diamonds_com_ratio          = diamonds2,
        preco_stats_por_cut         = stats_cut,
        preco_stats_cut_table       = stats_cut_table,
        grafico_colunas_cut         = grafico_colunas_cut,
        grafico_price_depth         = grafico_price_depth,
        grafico_price_depth_facete  = grafico_price_depth_facete
      )
    }

  )
)

#Demo
print_sep <- function(n){
  cat("\n", strrep("─", 20), " Exercício", n, "\n")
}

solver <- ListaSolver$new() 

print_sep(1);  print(solver$ex1())
print_sep(2);  print(solver$ex2(seed = 42))
print_sep(3);  print(solver$ex3())
print_sep(4);  print(solver$ex4(seed = 42))
print_sep(5);  print(solver$ex5(seed = 42))
print_sep(6);  print(solver$ex6())
print_sep(7);  print(solver$ex7())
print_sep(8);  print(solver$ex8(15))

print_sep(9)
res9 <- solver$ex9()
str(res9, max.level = 1)       
print(res9$contagem_categoria)
print(res9$idade_media_categoria)

print_sep(10)
res10 <- solver$ex10()
str(res10, max.level = 1)
print(head(res10$freq_color))
print(head(res10$count_cut_color))

#Para rodar:
#source("Lista1_1_10.R")
