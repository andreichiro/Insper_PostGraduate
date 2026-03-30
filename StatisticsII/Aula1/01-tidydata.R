# Aprendizagem Estatistica de Maquina II
# Aula 1
# Formato de dados tidy

library(tidyverse)

# a seguir, o mesmo conjunto de dados organizado de 4 formas diferentes
table1

table2

table3

table4a  # cases

table4b  # population

 
table1 %>% 
  mutate(rate = cases / population * 10000) # calcula a taxa

table4a %>% 
  pivot_longer(cols = -country, # pivota a tabela para deixa-la mais longa
               names_to = "year", # o nome das colunas estarao na coluna year
               values_to = "cases") # os valores estarao na coluna cases

table4a %>% 
  pivot_longer(cols = -country)

table2 %>%
  pivot_wider(# pivota a tabela para deixa-la mais larga
              names_from = type, # os nomes das colunas sao valores de type
              values_from = count) # os valores das novas colunas sao os da coluna count





