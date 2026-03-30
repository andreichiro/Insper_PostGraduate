# Aprendizagem Estatistica de Maquina II
# Aula 2
# Analise de textos - trava-linguas

# bibliotecas necessarias para analise de texto
library(tm) # install.packages("tm")
library(tidyverse)

doc1 <- "Magno foi à casa de Yasmin"
doc2 <- "Yasmin foi à casa de Magno"

class(doc1)

(corpus <- VCorpus(VectorSource(list(doc1, doc2))))

dtm1 <- corpus %>% 
  DocumentTermMatrix()

dtm1 %>% 
  as.matrix()

# documentos que vamos considerar
frases <- c("Não     sei se é fato ou se é fita. Não sei se é fita ou fato. O fato é que você me fita e fita mesmo de fato.",
            "Se cada um vai a casa de cada um é porque cada um quer que cada um vá lá. Porque se cada um não fosse a casa de cada um é porque cada um não queria que cada um fosse lá.",
            "O doce perguntou pro doce qual é o doce mais doce que o doce de batata-doce. O doce respondeu pro doce que o doce mais doce que o doce de batata-doce é o doce de doce de batata-doce.")

length(frases)

# cria o corpus a partir dos documentos na variavel frases
(corpus <- VCorpus(VectorSource(frases)))

# inspeciona o corpus
inspect(corpus)

meta(corpus[[1]])

getTransformations()

"Aula" == "aula"

stopwords("portuguese")

stopwords("portuguese")[stopwords("portuguese") != "mesmo"]


# processamento do texto
corpus <- corpus %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(stripWhitespace) %>%
  tm_map(removeNumbers) %>%
  # tm_map(removeWords, c("palavra1", "palavra2")) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeWords, iconv(stopwords("portuguese"), "UTF-8")) %>% 
  tm_map(stemDocument, language = "portuguese")

# conteudo dos documentos apos processamento
corpus[[1]]$content
corpus[[2]]$content
corpus[[3]]$content

# cria matriz documento termo
dtm <- DocumentTermMatrix(corpus)
dtm

# inspeciona matriz documento termo
inspect(dtm)

as.matrix(dtm) %>% 
  View()

# bibioteca para trabalhar com dados em formato tidy
library(tidytext)

# transforma o dtm em formato tidy
tidy(dtm)

# forma alternativa de construcao do dataframe tidy
corpus2 <- tibble(id = 1:3, 
                  text = frases)

corpus2

# com a funcao unnest_tokens, obtemos os tokens
corpus2 %>% 
  unnest_tokens(term, text) %>% 
  count(id, term) %>% 
  print(n = Inf)

# bind_tf_idf calula as estatisticas tf e idf
corpus2 %>% 
  unnest_tokens(term, text) %>% 
  count(id, term) %>% 
  bind_tf_idf(term, id, n)

# bigramas
corpus2 %>% 
  unnest_tokens(bigram, text, 
                token = "ngrams", n = 2) %>% 
  count(id, bigram) %>% 
  arrange(id, desc(n))

# tf-idf
corpus2 %>% 
  unnest_tokens(bigram, text, 
                token = "ngrams", n = 2) %>% 
  count(id, bigram) %>% 
  bind_tf_idf(bigram, id, n) %>% 
  arrange(desc(tf_idf))
