library(tidyverse)
library(rsample)
library(yardstick)
library(ISLR)
library(skimr)
library(modeldata)


# dados -------------------------------------------------------------------

data(mlc_churn)

# Customer churn data
# Description
# A data set from the MLC++ machine learning software for modeling customer churn. There 
# are 19 predictors, mostly numeric: state (categorical), account_length area_code 
# international_plan (yes/no), voice_mail_plan (yes/no), number_vmail_messages 
# total_day_minutes total_day_calls total_day_charge total_eve_minutes total_eve_calls 
# total_eve_charge total_night_minutes total_night_calls total_night_charge 
# total_intl_minutes total_intl_calls total_intl_charge, and number_customer_service_calls.
# 
# Details
# The outcome is contained in a column called churn (also yes/no). A note in one of the 
# source files states that the data are "artificial based on claims similar to real world".

mlc_churn %>% 
  group_by(churn) %>% 
  skim()

# mlc_churn$state <- NULL

mlc_churn <- mlc_churn %>% 
  mutate(churn = factor(churn, levels = c("no", "yes")))


# treinamento x teste -----------------------------------------------------

set.seed(313)

splits <- initial_split(mlc_churn, prop = .8, strata = "churn")

treinamento <- training(splits)
teste       <- testing(splits)



# mod 1: total_day_minutes e total_night_calls ---------------------------------

fit1 <- glm(churn ~ total_day_minutes + total_night_calls, 
            data = treinamento, family = "binomial")

# probabilidade de churn
predict(fit, teste, type = "response")

# mod 2: international_plan, number_customer_service_calls e total_day_charge --

fit2 <- glm()

# mod 3: com todas as variáveis do banco ---------------------------------------

fit3 <- glm()

# curva ROC comparando os modelos -----------------------------------------



