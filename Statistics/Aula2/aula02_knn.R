library(tidyverse)
library(FNN)

# amostra gerada ----------------------------------------------------------

n_obs <- 100

set.seed(123)

dados <- tibble(x = sort(runif(n = n_obs, min = 8, max = 18)), 
                y = 45*tanh(x/1.9 - 7) + 57 + rnorm(n = n_obs, mean = 0, sd = 4))

dados %>% 
  ggplot(aes(x, y)) + 
  geom_point() + 
  theme_bw()


# KNN ---------------------------------------------------------------------

knn.reg(train = dados$x, 
        test = matrix(dados$x[1:10]), 
        y =  dados$y, 
        k = 10)$pred


n_obs <- 100

set.seed(123)

dados <- tibble(x = sort(runif(n = n_obs, min = 8, max = 18)), 
                y = 45*tanh(x/1.9 - 7) + 57 + rnorm(n = n_obs, mean = 0, sd = 4))

x_pred <- seq(8, 18, 0.1)

tibble(ajuste = x_pred,
       y_fit = knn.reg(train = dados$x, 
                       test = matrix(x_pred), 
                       y =  dados$y, 
                       k = 25)$pred) %>% 
  ggplot(aes(ajuste, y_fit)) + 
    geom_point(data = dados, aes(x, y)) +
    geom_step(color = "red")



