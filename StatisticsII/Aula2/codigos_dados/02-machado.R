# Aprendizagem Estatistica de Maquina II
# Aula 2
# Analise de textos - Machado de Assis

# bibliotecas necessarias na analise
library(tm)
library(tidyverse)
library(tidytext)
library(wordcloud)

# leitura dos dados
corpus <- VCorpus(DirSource("machado", 
                            encoding = "UTF-8"))

corpus

corpus <- corpus %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(stripWhitespace) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, c("editora", "linha", "«", "não")) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeWords, iconv(stopwords("portuguese"),
                            "UTF-8")) %>% 
  tm_map(stemDocument, language = "pt")

# matriz documento termo
dtm <- DocumentTermMatrix(corpus)

dtm

dim(dtm)

# mostra linhas e algumas colunas da dtm
inspect(dtm)

# as.matrix(dtm)

machado <- tidy(dtm)

machado$document[1]
str_split(machado$document[1], "\\.")[[1]][1]

# remove a extensao .txt que aparece nas linhas da coluna document
machado <- machado %>% 
  mutate(document = map_chr(document,
                            ~str_split(.x, "\\.")[[1]][1]))

machado
# para voltar para o formato dtm
dtm <- cast_dtm(machado, document, term, count)

# quais termos são os que mais aparecem em dom casmurro
machado %>% 
  filter(document == "dom casmurro") %>% 
  slice_max(count, n = 100) %>% 
  with(wordcloud(term, count, random.order = FALSE, 
                 colors = brewer.pal(8, "Dark2")))


# quais termos são os mais aparecem em memórias póstumas de braz cubas?
machado %>% 
  filter(document == "memorias postumas de braz cubas") %>% 
  slice_max(count, n = 100) %>% 
  with(wordcloud(term, count, random.order = FALSE, 
                 colors = brewer.pal(8, "Dark2")))

# qual o top 15 de termos em cada livro?
machado %>% 
  group_by(document) %>%
  slice_max(count, n = 15) %>%
  arrange(document, desc(count)) %>% 
  print(n = Inf) 


# grafico com o top 15 de termos em cada livro
machado %>% 
  group_by(document) %>%
  slice_max(count, n = 15) %>% 
  mutate(term = reorder_within(term, count, document)) %>% 
  ggplot(aes(term, count, fill = document)) + 
  geom_col(show.legend = FALSE) + 
  facet_wrap(~ document, ncol = 2, scales = "free") + 
  coord_flip() +
  scale_x_reordered() +
  labs(x = "", y = "Frequência") + 
  theme(text = element_text(size = 16)) +
  theme_classic()


# quais os 15 termos mais importantes para cada documento? 
# usando tf_idf
machado %>% 
  bind_tf_idf(term, document, count) %>% 
  group_by(document) %>% 
  slice_max(tf_idf, n = 15) %>% 
  mutate(term = reorder_within(term, tf_idf, document)) %>% 
  ggplot(aes(term, tf_idf, fill = document)) + 
  geom_col(show.legend = FALSE) + 
  labs(x = "") + 
  facet_wrap(~ document, scales = "free", ncol = 3) + 
  coord_flip() + 
  scale_x_reordered() + 
  theme(text = element_text(size = 16)) +
  theme_classic()


# trabalhando com n-grams -----------------------------------------------------------------

# leitura do arquivo mpbc
doc <- readLines("machado/memorias postumas de braz cubas.txt", 
                 encoding = "UTF-8")

# cria um tibble com o documento
mpbc <- tibble(line = 1:length(doc), 
               text = doc)

# faz o unnest tokens considerando bi-grams
mpbc_bigrams <- mpbc %>% 
  unnest_tokens(bigram, text, token = "ngrams", n = 2) %>% 
  na.omit()

# conta os bi-grams mais frequentes
mpbc_bigrams %>% 
  count(bigram) %>% 
  arrange(desc(n))

# o que o codigo abaixo faz?
mpbc_bigrams %>% 
  separate(bigram, c("palavra1", "palavra2"), sep = " ") %>% 
  count(palavra1, palavra2) %>% 
  arrange(desc(n)) %>% 
  filter(!(palavra1 %in% c("de", "que", "e", "do", "da", 
                           "é", "o", "a", "me", "d")),
         !(palavra2 %in% c("de", "que", "e", "do", "da",
                           "é", "o", "a", "me", "d"))) %>% 
  print(n = 15)


# tri-grams
mpbc_trigrams <- mpbc %>% 
  unnest_tokens(trigram, text, token = "ngrams", n = 3)

# o que o codigo abaixo faz?
mpbc_trigrams %>% 
  na.omit() %>% 
  count(trigram) %>% 
  arrange(desc(n))

