#' @title Lista1_1_11_gapminder.R
#' @description Lista de exercícios 11 da lista 1
#' @author André Katsurada
#' @date 30/jul/25
#' @course APRENDIZAGEM ESTATÍSTICA DE MÁQUINA I

suppressPackageStartupMessages({
  pkgs <- c("R6", "ggplot2", "dplyr", "gapminder")

for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p, repos = "https://cloud.r-project.org", type = "binary")
  }
  }
})

#Classe p/ a database do Hans Rosling
GapminderSolver <- R6::R6Class(
  "GapminderSolver",


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

    #' @title Exercício 11 – dataset gapminder
    #' @description
    #' Analisa o dataset gapminder, calcula métricas e gera gráficos
    ex11 = function(seed = NULL) {
      private$set_rng(seed)

      data("gapminder", package = "gapminder") 

      #a) Número total de observações 
      total_obs <- nrow(gapminder)          

      #b) Anos existentes
      anos <- sort(unique(gapminder$year))  

      #c) Observações por continente
      obs_por_cont <- dplyr::count(gapminder, continent, sort = TRUE)

      #d) Observações por continente e ano 
      obs_cont_ano <- dplyr::count(gapminder, continent, year, sort = TRUE)

      #e) Similar ao item d mas a partir de 2000 
      obs_cont_ano_2000 <- gapminder |>
        dplyr::filter(year >= 2000) |>      # filtra anos ≥ 2000
        dplyr::count(continent, year, sort = TRUE)

      #f) Média e sd do PIB per capita desde 2000
      gdp_stats_2000 <- gapminder |>
        dplyr::filter(year >= 2000) |>
        dplyr::group_by(continent, year) |>
        dplyr::summarise(
          media_gdp  = mean(gdpPercap),
          sd_gdp     = sd(gdpPercap),
          .groups    = "drop")

      #g) Gráfico de expectativa de vida por país
      grafico_life_country <- ggplot2::ggplot(
        gapminder,
        ggplot2::aes(year, lifeExp, group = country, colour = continent)) +
        ggplot2::geom_line(alpha = .5) +
        ggplot2::labs(title = "Expectativa de vida por País",
                      x = "Ano", y = "Expectativa de Vida",
                      colour = "Continente")
      private$save_png("ex11_life_country.png", grafico_life_country)

      #h)Expectativa média de vida por continente e ano 
      life_mean <- gapminder |>
        dplyr::group_by(continent, year) |>
        dplyr::summarise(
          lifeExp_media = mean(lifeExp),
          .groups       = "drop")

      #i) Gráfico média de vida por continente 
      grafico_life_cont <- ggplot2::ggplot(
        life_mean,
        ggplot2::aes(year, lifeExp_media, colour = continent)) +
        ggplot2::geom_line(size = 1.2) +
        ggplot2::labs(title = "Expectativa média de vida por continente",
                      x = "Ano", y = "Expectativa de vida média",
                      colour = "Continente")
      private$save_png("ex11_life_continent.png", grafico_life_cont)

      #Resultados
    list(
        total_observacoes           = total_obs,
        anos_registrados            = anos,
        obs_por_continente          = obs_por_cont,
        obs_por_continente_ano      = obs_cont_ano,
        obs_por_continente_ano_2000 = obs_cont_ano_2000,
        gdp_stats_2000              = gdp_stats_2000,
        grafico_life_country        = grafico_life_country,
        life_mean_continent         = life_mean,
        grafico_life_continent      = grafico_life_cont
      )
    }
  )
)

#Demo
print_sep <- function(n) {
  cat("\n", strrep("─", 20), " Exercício", n, "\n")
}

solver <- GapminderSolver$new()

print_sep(11)
res11 <- solver$ex11()

print(res11$total_observacoes)
print(res11$obs_por_continente)
print(res11$gdp_stats_2000)
print(res11$grafico_life_continent)

#P/ rodar:
#source("Lista1_11_gapminder.R") ou Rscript Lista1_11_gapminder.R