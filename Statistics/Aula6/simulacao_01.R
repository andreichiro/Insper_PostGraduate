library(tidyverse)

set.seed(123)

B <- 1000

banco_unico  <- vector("numeric", B)
banco_50_aux <- vector("numeric", 50)
banco_50     <- vector("numeric", B)


for (b in 1:B) {

  # amostra de tamanho 100 de uma distribuição normal com 
  # média 45 e desvio padrão = 3

  banco_unico[b] <- mean(rnorm(100, 45, 3))


  # 50 amostras de tamanho 100 de uma dist. normal com 
  # média 45 e desvio padrão 3

  for (i in 1:50) banco_50_aux[i] <- mean(rnorm(100, 45, 3))

  banco_50[b] <- mean(banco_50_aux)

}


tibble(previsoes = c(banco_unico, banco_50), 
       metodo = rep(c("banco único", "50 amostras"), each = B)) %>% 
  ggplot(aes(previsoes, fill = metodo)) +
    geom_density(alpha = .5)






# programação funcional ----------------------------------------------------

pop <- 1:500

tibble(B = 1:5000) %>% 
  mutate(out = map_lgl(B, ~!(15 %in% sample(pop, 500, replace = TRUE)))) %>% 
  mutate(prop_out = cumsum(out) / B) %>% 
  ggplot(aes(B, prop_out)) + 
    geom_hline(yintercept = exp(-1), color = "red") + 
    geom_line() + 
    scale_y_continuous(limits = c(0, 1)) +
    labs(y = "Proporção Out of Bag")


