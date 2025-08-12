library(ISLR)
library(tidyverse)
library(skimr)
library(rsample)
library(pROC)
library(yardstick)
library(patchwork)
# install.packages("rsample")


head(Default)
str(Default)

Default %>% 
  group_by(default) %>% 
  skim()


(fig_01 <- Default %>% 
  ggplot(aes(default, balance, fill = default)) + 
    geom_boxplot(show.legend = FALSE))


(fig_02 <- Default %>% 
    ggplot(aes(default, income, fill = default)) + 
    geom_boxplot(show.legend = FALSE))

fig_01 + fig_02


(fig_03 <- Default %>% 
  ggplot(aes(balance, income, color = default)) + 
    geom_point(alpha = .4) +
    facet_grid(~student))

fit <- glm(default ~ ., family = "binomial", data = Default)

summary(fit)



# treinamento x teste -----------------------------------------------------

set.seed(123)

splits <- initial_split(Default, prop = .7, strata = default)

tr <- training(splits)  
test <- testing(splits)

tr %>% 
  count(default)

test %>% 
  count(default)

splits <- initial_split(Default, prop = .8, strata = default)

fit <- glm(default ~ ., data = tr, family = "binomial")

prob <- predict(fit, test, type = "response")


# Medidas e curva ROC -----------------------------------------------------

(roc_logistica <- roc(test$default, prob))

coords(roc_logistica, seq(.1, .7, 0.025), 
       ret = c("threshold", "accuracy", "specificity", "sensitivity", "npv", "ppv"))

# Forma 1

coords(roc_log, seq(0, 1, 0.0005), ret = c("threshold", "specificity", "sensitivity")) %>% 
  as_tibble() %>% 
  ggplot(aes(1 - specificity, sensitivity)) +
  geom_step(direction = "vh")

# Forma 2

plot(roc_logistica, 
     legacy.axes = TRUE, 
     print.auc = TRUE, lwd = 4, col = "blue")


# Forma 3 (preferível)

tibble(classe = test$default, marcador = prob) %>% 
  roc_curve(classe, marcador, event_level = "second") %>% 
  autoplot()


tibble(classe = test$default, marcador = prob) %>% 
  roc_auc(classe, marcador, event_level = "second") 
