library(tidyverse)
library(FNN)

# psi(x) - modelo base ----------------------------------------------------

ggplot(data = data.frame(x = 0), aes(x = x)) +
  stat_function(fun = function(x) 45*tanh(x/1.9 - 7) + 57) +
  xlim(8, 18)

# amostra gerada ----------------------------------------------------------

n_obs <- 100

set.seed(123)

dados <- tibble(x = sort(runif(n = n_obs, min = 8, max = 18)), 
                y = 45*tanh(x/1.9 - 7) + 57 + rnorm(n = n_obs, mean = 0, sd = 4))

head(dados)

dados %>% 
  ggplot(aes(x, y)) +
  geom_point()


# spline ------------------------------------------------------------------

fit <- smooth.spline(dados$x, dados$y, df = 10, all.knots = TRUE)

fit  

predict(fit, dados$x)$y



# avaliar o EQM considerando de 2 a 100 graus de liberdade ----------------
