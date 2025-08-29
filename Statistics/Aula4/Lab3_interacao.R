# Instala e carrega os pacotes necessários
library(dados)
library(dplyr)
library(ggplot2)
library(skimr)

pinguins %>% skim()

# remove as linhas com valores ausentes (NA)
penguins_clean <- pinguins %>%
  na.omit()

# Modelo de regressão sem interação
modelo1 <- lm(massa_corporal ~ comprimento_nadadeira + especie, data = penguins_clean)
summary(modelo1)

# Modelo de regressão com interação
# A sintaxe comprimento_nadadeira * especie já inclui os termos principais e a interação
modelo2 <- lm(massa_corporal ~ comprimento_nadadeira * especie, data = penguins_clean)
summary(modelo2)


# Visualização
ggplot(penguins_clean, aes(x = comprimento_nadadeira, y = massa_corporal, color = especie)) +
  geom_point(alpha = 0.7) +
  #geom_smooth(method = "lm", se = FALSE, formula = y ~ x) +
  labs(title = "Relação entre Comprimento da Nadadeira e Massa Corporal",
       x = "Comprimento da Nadadeira (mm)",
       y = "Massa Corporal (g)") +
  theme_minimal()


