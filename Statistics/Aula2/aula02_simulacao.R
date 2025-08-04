library(tidyverse)

# geração de números aleatórios
# ?rnorm

rnorm(10, mean = 0, sd = 1)

# criar uma sequência 

seq(10, 13, .5)

# criar um data frame

df <- tibble(var1 = c(1, 3, 2), 
             var2 = c(11, 7, 18))

df

df$var1



# modelo utilizando apenas as médias das respostas ------------------------
# para previsão

# número de observações geradas

n_obs <- 30

# banco de dados simulado

dados <- tibble(x = sort(runif(n = n_obs, min = 8, max = 18)), 
                y = 45*tanh(x/1.9 - 7) + 57 + rnorm(n = n_obs, mean = 0, sd = 4))

# valores para os quais faremos as previsões

x_prev <- seq(8, 18, 0.5)

# banco de dados com os valores e previsões

dados_prev <- tibble(x = x_prev, 
                     y = mean(dados$y))

# gráfico dos valores simulados e valores preditos

ggplot() + 
  geom_point(data = dados, aes(x, y)) + 
  geom_line(data = dados_prev, aes(x, y), color = "red") + 
  xlim(8, 18) + ylim(0, 130)


# mais flexível -----------------------------------------------------------

set.seed(3322)

# utilize diferentes graus de liberdade entre 2 e 30 e 
# verifique a variabilidade das curvas ajustadas

gl <- 29

n_obs <- 30

dados <- tibble(x = sort(runif(n = n_obs, min = 8, max = 18)), 
                y = 45*tanh(x/1.9 - 7) + 57 + rnorm(n = n_obs, mean = 0, sd = 4))

x_prev <- seq(8, 18, 0.05)

fit <- smooth.spline(dados$x, dados$y, df = gl)

dados_prev <- tibble(x = x_prev, 
                     y = predict(fit, x_prev)$y)

ggplot() + 
  geom_point(data = dados, aes(x, y)) + 
  geom_line(data = dados_prev, aes(x, y), color = "red") + 
  xlim(8, 18) + ylim(0, 130)



# simulação de modelos ----------------------------------------------------

set.seed(312)

modelo_medio <- vector("numeric", 5000)


n_obs <- 30

for (i in 1:5000) {
  
  dados <- tibble(x = sort(runif(n = n_obs, min = 8, max = 18)), 
                  y = 45*tanh(x/1.9 - 7) + 57 + rnorm(n = n_obs, mean = 0, sd = 4))
  
  modelo_medio[i] <- mean(dados$y)

}


ggplot(data = data.frame(x = 0), mapping = aes(x = x)) + 
  stat_function(fun = function(x) 45*tanh(x/1.9 - 7) + 57, linewidth = .8) +  
  geom_hline(yintercept = mean(modelo_medio), color = "blue", linewidth = 1.2) + 
  geom_point(data = tibble(x = 11, y = 45*tanh(11/1.9 - 7) + 57), 
             aes(x, y), color = "red", size = 5) + 
  xlim(8, 18)


(vies <- mean(modelo_medio) - (45*tanh(11/1.9 - 7) + 57))
(variancia <- var(modelo_medio))

vies^2 + variancia


# mais flexível - calcula o valor médio para x = 11 -----------------------

set.seed(1234)

modelo_medio <- vector("numeric", 5000)

gl <- 4

n_obs <- 30


for (i in 1:5000) {
  
  dados <- tibble(x = sort(runif(n = n_obs, min = 8, max = 18)), 
                  y = 45*tanh(x/1.9 - 7) + 57 + rnorm(n = n_obs, mean = 0, sd = 4))

  fit <- smooth.spline(dados$x, dados$y, df = gl)

  modelo_medio[i] <- predict(fit, 11)$y
  
}


ggplot(data = data.frame(x = 0), mapping = aes(x = x)) + 
  stat_function(fun = function(x) 45*tanh(x/1.9 - 7) + 57, linewidth = .8) +  
  geom_point(data = tibble(x = 11, y = mean(modelo_medio)), aes(x, y), color = "blue", size = 5) + 
  geom_point(data = tibble(x = 11, y = 45*tanh(11/1.9 - 7) + 57), 
             aes(x, y), color = "red", size = 5) + 
  xlim(8, 18) +
  labs(title = paste0("graus de liberdade: ", gl))


(vies <- mean(modelo_medio) - (45*tanh(11/1.9 - 7) + 57))
(variancia <- var(modelo_medio))

vies^2 + variancia

