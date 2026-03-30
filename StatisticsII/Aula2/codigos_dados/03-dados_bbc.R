# Aprendizagem Estatistica de Maquina II
# Aula 2
# Analise de textos - topicos BBC

# bibliotecas necessárias na analise
library(tm)
library(Matrix)
library(tidyverse)
library(tidytext)

# carrega os dados, que estao armazenados em formato de matriz esparsa
mtx <- readMM("bbc/bbc.mtx")

# matriz termo documento
tdm <- as.TermDocumentMatrix(mtx, weighting = weightTf)

# obtem a transposta: matriz documento termo
dtm <- t(tdm)
dim(dtm)

# cada linha da matriz dtm corresponde a um  documento do corpus 
# e cada coluna corresponde a um termo; e esta matriz armazena as
# frequências de ocorrências dos termos nos documentos
dtm

# carrega o nome dos termos
dtm$dimnames$Terms <- scan("bbc/bbc.terms", what = "character")

# carrega o nome dos documentos
dtm$dimnames$Docs <- scan("bbc/bbc.docs", what = "character")

# o os primeiros 5 documentos e as frequências de 6 de seus termos
inspect(dtm[1:5, 1:6])

# listar todos os termos do documento business.001
tidy(dtm) %>% 
  filter(document == "business.001") %>% 
  arrange(desc(count))%>% 
  print(n = Inf)

tidy(dtm) %>% 
  filter(term == "sale") %>% 
  print(n = Inf)

# qual a frequência de tópicos no corpus? 
tidy(dtm) %>% 
  count(document) %>% 
  mutate(topic = map_chr(document, ~str_extract(.x, "[a-z]+"))) %>%
  count(topic)

# qual a frequência dos termos? 
tidy(dtm) %>% 
  group_by(term) %>% 
  summarise(count = sum(count)) 

# 10 termos mais frequentes
tidy(dtm) %>% 
  group_by(term) %>% 
  summarise(count = sum(count)) %>% 
  slice_max(count, n = 10)

# top 10 palavras mais frequentes de cada tópico 
tidy(dtm) %>% 
  mutate(topic = map_chr(document, ~str_extract(.x, "[a-z]+"))) %>% 
  group_by(topic, term) %>% 
  summarise(count = sum(count)) %>% 
  slice_max(count, n = 10) %>% 
  print(n = Inf)

# grafico top 10 palavras mais frequentes de cada tópico 
tidy(dtm) %>% 
  mutate(topic = map_chr(document, ~str_extract(.x, "[a-z]+"))) %>% 
  group_by(topic, term) %>% 
  summarise(count = sum(count)) %>% 
  slice_max(count, n = 10) %>% 
  mutate(term = reorder_within(term, count, topic)) %>% 
  ggplot(aes(term, count, fill = topic)) + 
  geom_col(show.legend = FALSE) + 
  labs(x = "", y = "") + 
  facet_wrap(~ topic, scales = "free", ncol = 3) +
  coord_flip() +
  scale_x_reordered() + 
  theme_classic() +
  theme(text = element_text(size = 12))

# palavras mais importantes (tf-idf) de acordo com documento 
tidy(dtm) %>% 
  mutate(topic = map_chr(document, ~str_extract(.x, "[a-z]+"))) %>% 
  group_by(topic, term) %>% 
  summarise(n = sum(count)) %>% 
  bind_tf_idf(term, topic, n) %>% 
  group_by(topic) %>% 
  slice_max(tf_idf, n = 10) %>% 
  ungroup() %>% 
  mutate(term = reorder_within(term, tf_idf, topic)) %>% 
  ggplot(aes(term, tf_idf, fill = topic)) + 
  geom_col(show.legend = FALSE) + 
  labs(x = "") + 
  facet_wrap(~ topic, scales = "free", ncol = 3) + 
  coord_flip() + 
  scale_x_reordered() + 
  scale_y_continuous(labels = scales::label_number_auto()) + 
  theme_classic() +
  theme(text = element_text(size = 12))



# LDA ---------------------------------------------------------------------

library(topicmodels)

model <- LDA(dtm, k = 5, method = "Gibbs", 
             control = list(seed = 4321))


# word-topic probabilities ------------------------------------------------

# beta é a matriz de per-topic-per-word probabilities
bbc_topics <- tidy(model, matrix = "beta")

bbc_topics %>% 
  print(n = 20)


# 10 termos com maior probabilidade de ocorrência em cada topico
bbc_top_terms <- bbc_topics %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  ungroup() %>%
  arrange(topic, desc(beta))

bbc_top_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  scale_x_reordered() + 
  theme_classic()



# Document-topic probabilities --------------------------------------------

# gamma é a matriz de per-document-per-topic probabilities
bbc_documents <- tidy(model, matrix = "gamma")

# mostra as probabilidades de pertencimento a cada topico
# para os documentos business.001, business.002, business.003
bbc_documents %>% 
  filter(document %in% paste0("business.00", 1:3)) %>% 
  arrange(document, desc(gamma))

bbc_documents %>% 
  filter(document %in% paste0("business.00", 1:3)) %>% 
  mutate(topic = factor(topic)) %>% 
  ggplot(aes(document, gamma, fill = topic)) + 
  geom_col(position = "dodge") + 
  theme_classic()

# obtem a contagem de documentos por topico segundo o topico
# com maior probabilidade obtido pelo LDA
bbc_documents %>% 
  group_by(document) %>% 
  slice_max(gamma, n = 1) %>% 
  ungroup() %>%
  count(topic)

# para nomear os 5 topicos de acordo com os topicos observados
bbc_documents %>% 
  group_by(document) %>% 
  slice_max(gamma, n = 1) %>% 
  mutate(topic_obs = map_chr(document, ~str_extract(.x, "[a-z]+"))) %>% 
  ungroup() %>% 
  count(topic_obs, topic) %>% 
  mutate(topic = reorder_within(factor(topic), n, topic_obs)) %>% 
  ggplot(aes(topic, n, fill = topic_obs)) + 
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic_obs, scales = "free") + 
  coord_flip() + 
  scale_x_reordered() + 
  theme_bw()


# proporcao classificada corretamente
bbc_documents %>% 
  group_by(document) %>% 
  slice_max(gamma, n = 1) %>% 
  mutate(topic_obs = map_chr(document, ~str_extract(.x, "[a-z]+")), 
         topic_pred = recode(topic, 
                             '1' = "business", 
                             '2' = "entertainment", 
                             '3' = "tech",
                             '4' = "sport", 
                             '5' = "politics")) %>%  
  ungroup() %>% 
  summarise(mean(topic_pred == topic_obs))


# estudo do número de tópicos 
avaliacao <- tibble(k = 2:14,
                    loglike = map_dbl(k, ~LDA(dtm, k = .x, method = "Gibbs",
                                              control = list(seed = 4321))@loglikelihood))

# gráfico do cotovelo
avaliacao %>%
  ggplot(aes(k, loglike)) +
  geom_point(color = "steelblue", size = 5) +
  geom_line(color = "steelblue", size = 1, alpha = .6) +
  labs(x = "Número de tópicos", y = "log-verossimilhança") +
  scale_y_continuous(labels = scales::label_number_auto()) +
  scale_x_continuous(breaks = 2:14) +
  theme_bw()
