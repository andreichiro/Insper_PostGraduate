library(tidyverse)
library(viridis)

set.seed(123)

df <- tibble(x = runif(100, 0, 20), 
             y = 2 * x + rnorm(100))

df


# perda quadrática --------------------------------------------------------

quadratica <- function(beta, x, y) {
  
  sum((y - (beta * x))^2)
  
}

quadratica(beta = 3, x = df$x, y = df$y)

quadratica(beta = 4, x = df$x, y = df$y)


# perda absoluta ----------------------------------------------------------


absoluta <- function(beta, x, y) {
  
  sum(abs(y - (beta * x)))
  
}

absoluta(beta = 3, x = df$x, y = df$y)

absoluta(beta = 4, x = df$x, y = df$y)




# função ------------------------------------------------------------------


tibble(beta = seq(1, 3, 0.005)) %>% 
  mutate(perda = map_dbl(beta, ~quadratica(.x, df$x, df$y)), 
         tipo = "quadrática") %>% 

  bind_rows(tibble(beta = seq(1, 3, 0.005)) %>% 
              mutate(perda = map_dbl(beta, ~absoluta(.x, df$x, df$y)), 
                     tipo = "absoluta")) %>% 
  ggplot(aes(beta, perda, group = tipo)) + 
    geom_line() +
    facet_wrap(tipo~., scales = "free") +
    theme_bw()






# otimização --------------------------------------------------------------

optimize(quadratica, c(0, 4), x = df$x, y = df$y)$minimum

optimize(absoluta, c(0, 4), x = df$x, y = df$y)$minimum









# 2 parâmetros ------------------------------------------------------------

set.seed(123)

df <- tibble(x = runif(100, 0, 20), 
             y = 2 * x + 3 + rnorm(100))

df




# perda quadrática --------------------------------------------------------

quadratica <- function(beta0, beta1, x, y) {
  
  sum((y - (beta0 + beta1 * x))^2)
  
}

quadratica(beta0 = 3, beta1 = 1, x = df$x, y = df$y)

quadratica(beta0 = 4, beta1 = 1, x = df$x, y = df$y)


# perda absoluta ----------------------------------------------------------


absoluta <- function(beta0, beta1, x, y) {
  
  sum(abs(y - (beta0 + beta1 * x)))
  
}

absoluta(beta0 = 3, beta1 = 1, x = df$x, y = df$y)

absoluta(beta0 = 4, beta1 = 1, x = df$x, y = df$y)




# otimização --------------------------------------------------------------

quadratica_vetor <- function(beta, x, y) {
  
  sum((y - (beta[1] + beta[2] * x))^2)
  
}


(otimizado <- optim(c(0, 4), quadratica_vetor, x = df$x, y = df$y))



crossing(beta0 = seq(0, 6, 0.01), 
         beta1 = seq(0, 4, 0.01)) %>% 
  mutate(perda = map2_dbl(beta0, beta1, ~quadratica(.x, .y, df$x, df$y)), 
         tipo = "quadrática") %>% 
  ggplot(aes(beta0, beta1)) + 
   geom_raster(aes(fill = perda)) +
    geom_point(data = tibble(beta0 = otimizado$par[1], beta1 = otimizado$par[2]), 
               color = "red", size = 2) + 
    scale_fill_viridis() +
    theme_bw()
