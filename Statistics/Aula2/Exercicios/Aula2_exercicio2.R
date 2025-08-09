#' @title Exercicio 2 aula 2
#' @description Ver se o cliente compra ou não um produto com base em índices de satisfação e engajamento
#' @author André Katsurada
#' @date 30/ago/25
#' @course APRENDIZAGEM ESTATÍSTICA DE MÁQUINA I

#Pacotes 
pkg_list <- c("tidyverse", "class", "janitor", "glue", "png", "grid")

install_and_load <- function(pkg) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
    library(pkg, character.only = TRUE)
  }
}

suppressPackageStartupMessages(
  invisible(lapply(pkg_list, install_and_load))
)

#Dados iniciais
df <- tribble(
  ~ID, ~satisf, ~engage, ~comprou,
  "a", 0.71, 0.89, "sim",
  "b", 0.25, 0.57, "nao",
  "c", 0.39, 0.59, "sim",
  "d", 0.09, 0.37, "nao",
  "e", 0.96, 0.36, "sim",
  "f", 0.01, 0.59, "sim",
  "g", 0.57, 0.87, "nao",
  "h", 0.76, 0.68, "sim",
  "i", 0.87, 0.14, "nao",
  "j", 0.04, 0.55, "nao",
  "k", 0.66, 0.68, "nao",
  "l", 0.88, 0.53, "sim"
) %>% 
  mutate(comprou = factor(comprou, levels = c("nao", "sim"))) |>
  arrange(ID)                                     # deterministic order

#Imagem do gráfico de dispersão
img_path <- "/mnt/data/Screenshot 2025-08-06 at 15.43.59.png"
if (file.exists(img_path)) {
  try({
    img <- png::readPNG(img_path)
    grid::grid.raster(img)
    message(glue("Illustrative scatter‑plot loaded from: {img_path}"))
  }, silent = TRUE)
}

#Reprodutibilidade
set.seed(42)                                    
df <- df |>
  group_by(comprou) |>                           
  mutate(fold = sample(rep(1:4, length.out = n()))) |>
  ungroup()

#KNN com 4 lotes
cv_knn <- function(data, k, folds = 4) {
  stopifnot(all(c("satisf", "engage", "comprou", "fold") %in% names(data)))
  
  preds <- map_dfr(seq_len(folds), function(f) {
    train <- filter(data, fold != f)
    test  <- filter(data, fold == f)
    
    pred <- knn(
      train = train |> select(satisf, engage) |> as.matrix(),
      test  = test  |> select(satisf, engage) |> as.matrix(),
      cl    = train$comprou,
      k     = k
    )
    
    tibble(ID = test$ID,
           fold = f,
           atual = test$comprou,
           pred   = pred)
  })
  
  list(
    error      = mean(preds$pred != preds$atual),
    confusion  = preds |> janitor::tabyl(atual, pred),
    predictions = preds
  )
}

#CV para k = 1 e k = 3
metrics <- map_dfr(c(1, 3), \(K) {
  res <- cv_knn(df, k = K)
  tibble(k = K,
         error    = res$error,
         accuracy = 1 - res$error)
})

print(metrics)

#Salvar tabela de predições
cv_preds <- bind_rows(
  cv_knn(df, 1)$predictions |> mutate(k = 1),
  cv_knn(df, 3)$predictions |> mutate(k = 3)
)

# Caminho de saída (arquivo ficará na pasta atual)
out_csv <- file.path(getwd(), "aula2_exercicio2.csv")
write_csv(cv_preds, out_csv, na = "")
message(glue("↳ Predictions written to {out_csv}"))

#Confusion matrix para k = 1 e k = 3
conf_k1 <- cv_knn(df, 1)$confusion
conf_k3 <- cv_knn(df, 3)$confusion

message("\nConfusion matrix : k = 1")
print(conf_k1)

message("\nConfusion matrix : k = 3")
print(conf_k3)

#Baseline
majority_class <- names(sort(table(df$comprou), decreasing = TRUE))[1]
baseline_err   <- mean(df$comprou != majority_class)

message("\nBaseline (majority‑class ‘", majority_class, "’) error = ",
        sprintf("%.4f", baseline_err))

#Verifica se as duas dimensões estão corretas
d_ok <- TRUE
try({
  dm <- as.matrix(dist(df |> select(satisf, engage)))
  stopifnot(all(diag(dm) == 0), nrow(dm) == 12)
}, silent = TRUE)

if (d_ok) message("sanity check ok")

if (interactive()) print(sessionInfo())