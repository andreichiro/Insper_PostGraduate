#' @title quiz3.R — Rede Neural com Keras p/ classificação de imagens
#' @author André e Danillo
#' @date 19/nov/2025

library(reticulate)

venv_name <- "r-mnist"

if (!virtualenv_exists(venv_name)) {
  stop(
    "Virtualenv '", venv_name,
    "' não existe. Crie antes com: keras3::install_keras(envname = '", venv_name, "')."
  )
}

Sys.setenv(RETICULATE_PYTHON = virtualenv_python(venv_name))
use_virtualenv(venv_name, required = TRUE)


library(tensorflow)
library(keras3)

library(future)
library(future.apply)

plan(multisession, workers = 4L)

# Seeds 
set.seed(42)
tf$random$set_seed(42L)

# 1) mnist
mnist <- dataset_mnist()

x_train <- as.array(mnist$train$x)
y_train <- as.array(mnist$train$y)

x_test  <- as.array(mnist$test$x)
y_test  <- as.array(mnist$test$y)

img_rows    <- 28L
img_cols    <- 28L
num_classes <- 10L

# Converter p/ arrays 4D (batch, rows, cols, channels)
x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1L))
x_test  <- array_reshape(x_test,  c(nrow(x_test),  img_rows, img_cols, 1L))

# Normalizar para [0,1]
x_train <- x_train / 255
x_test  <- x_test  / 255

# One-hot encoding p/ labels
y_train_cat <- to_categorical(y_train, num_classes)
y_test_cat  <- to_categorical(y_test,  num_classes)

# 2) Split treino / validação
set.seed(42)
n_train  <- dim(x_train)[1]
val_size <- 5000L

val_indices <- sample(seq_len(n_train), size = val_size)

x_val <- x_train[val_indices, , , , drop = FALSE]
y_val <- y_train_cat[val_indices, , drop = FALSE]

x_train_final <- x_train[-val_indices, , , , drop = FALSE]
y_train_final <- y_train_cat[-val_indices, , drop = FALSE]

hpo_train_size <- 15000L  

set.seed(42)
hpo_idx <- sample(seq_len(dim(x_train_final)[1]), size = hpo_train_size)

x_train_hpo <- x_train_final[hpo_idx, , , , drop = FALSE]
y_train_hpo <- y_train_final[hpo_idx, , drop = FALSE]

cat("Train samples:", dim(x_train_final)[1], "\n")
cat("Val samples  :", dim(x_val)[1], "\n")
cat("Test samples :", dim(x_test)[1], "\n")

# 3) modelo 
# testar adam, adamw e lion

build_mnist_cnn_model <- function(
  input_shape      = c(28L, 28L, 1L),
  num_classes      = 10L,
  l2_reg           = 1e-4,
  dropout_conv     = 0.25,
  dropout_dense    = 0.5,
  rotation_deg     = 10,
  translation_frac = 0.1,
  zoom_frac        = 0.1,
  dense_units      = 256L,
  optimizer_name   = c("adam", "adamw", "lion"),
  learning_rate    = 1e-3,
  weight_decay     = 0
) {
  regularizer    <- regularizer_l2(l = l2_reg)
  he_init        <- "he_normal"
  optimizer_name <- match.arg(optimizer_name)

  optimizer <- switch(
    optimizer_name,
    "adam"  = optimizer_adam(learning_rate = learning_rate),
    "adamw" = optimizer_adam_w(
      learning_rate = learning_rate,
      weight_decay  = weight_decay
    ),
    "lion"  = optimizer_lion(
      learning_rate = learning_rate,
      weight_decay  = weight_decay
    )
  )

  model <- keras_model_sequential(input_shape = input_shape) %>%
    # Data augmentation
    layer_random_rotation(factor = rotation_deg / 360) %>%
    layer_random_translation(
      height_factor = translation_frac,
      width_factor  = translation_frac
    ) %>%
    layer_random_zoom(
      height_factor = zoom_frac,
      width_factor  = zoom_frac
    ) %>%
    
    # Bloco 1
    layer_conv_2d(
      filters = 32, kernel_size = c(3, 3), padding = "same",
      kernel_regularizer = regularizer,
      kernel_initializer = he_init
    ) %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(
      filters = 32, kernel_size = c(3, 3), padding = "same",
      kernel_regularizer = regularizer,
      kernel_initializer = he_init
    ) %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout(rate = dropout_conv) %>%
    
    # Bloco 2
    layer_conv_2d(
      filters = 64, kernel_size = c(3, 3), padding = "same",
      kernel_regularizer = regularizer,
      kernel_initializer = he_init
    ) %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_conv_2d(
      filters = 64, kernel_size = c(3, 3), padding = "same",
      kernel_regularizer = regularizer,
      kernel_initializer = he_init
    ) %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout(rate = dropout_conv) %>%
    
    # Bloco 3
    layer_conv_2d(
      filters = 128, kernel_size = c(3, 3), padding = "same",
      kernel_regularizer = regularizer,
      kernel_initializer = he_init
    ) %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout(rate = 0.4) %>%
    
    # Classificador denso c/ softmax
    layer_global_average_pooling_2d() %>%
    layer_dense(
      units              = dense_units,
      kernel_regularizer = regularizer,
      kernel_initializer = he_init
    ) %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_dropout(rate = dropout_dense) %>%
    layer_dense(
      units      = num_classes,
      activation = "softmax"
    )

  model %>% compile(
    loss      = loss_categorical_crossentropy(label_smoothing = 0.1),
    optimizer = optimizer,
    metrics   = list(
      metric_categorical_accuracy(name = "acc"),
      metric_top_k_categorical_accuracy(k = 3L, name = "top3_acc")
    )
  )
  model
}

# 4) Callbacks: early stopping e reduce learning rate
cb_early <- callback_early_stopping(
  monitor = "val_loss",
  patience = 5,
  restore_best_weights = TRUE
)

cb_reduce <- callback_reduce_lr_on_plateau(
  monitor = "val_loss",
  factor  = 0.5,
  patience = 2,
  min_lr  = 1e-6
)

cb_checkpoint <- callback_model_checkpoint(
  filepath          = "best_mnist_cnn.keras",
  monitor           = "val_loss",
  save_best_only    = TRUE,
  mode              = "min",
  save_weights_only = FALSE
)

# callbacks_search <- list(cb_early, cb_reduce)

# treino final
callbacks_train <- list(cb_early, cb_reduce, cb_checkpoint)

batch_size  <- 256L

epochs_hpo  <- 10L         
epochs_full <- 30L           

set.seed(42)

# 5) Função de uma rodada de treino p/ HPO (run_config)
run_config <- function(
  l2_reg,
  dropout_conv,
  dropout_dense,
  rotation_deg,
  translation_frac,
  zoom_frac,
  dense_units,
  optimizer_name,
  learning_rate,
  weight_decay = 0,
  epochs      = 40L,
  batch_size  = 128L
) {
  cat(
    sprintf(
      "[HPO] Starting config: opt=%s, lr=%.1e, dense=%d, l2=%g, wd=%g\n",
      optimizer_name, learning_rate, dense_units, l2_reg, weight_decay
    )
  )
  flush.console()

  model <- build_mnist_cnn_model(
    input_shape      = c(img_rows, img_cols, 1L),
    num_classes      = num_classes,
    l2_reg           = l2_reg,
    dropout_conv     = dropout_conv,
    dropout_dense    = dropout_dense,
    rotation_deg     = rotation_deg,
    translation_frac = translation_frac,
    zoom_frac        = zoom_frac,
    dense_units      = dense_units,
    optimizer_name   = optimizer_name,
    learning_rate    = learning_rate,
    weight_decay     = weight_decay
  )

  # callbacks no worker
  callbacks_search_local <- list(
    callback_early_stopping(
      monitor = "val_loss",
      patience = 2,
      restore_best_weights = TRUE
    ),
    callback_reduce_lr_on_plateau(
      monitor = "val_loss",
      factor  = 0.5,
      patience = 2,
      min_lr  = 1e-6
    )
  )

  t0 <- Sys.time()

  history <- model %>% fit(
    x = x_train_hpo,
    y = y_train_hpo,
    batch_size      = batch_size,
    epochs          = epochs,
    validation_data = list(x_val, y_val),
    callbacks       = callbacks_search_local,
    verbose         = 2   # logar 1 linha por epoch
  )

  t1 <- Sys.time()
  cat(sprintf("[HPO] Finished config in %.1f sec\n\n",
              as.numeric(difftime(t1, t0, units = "secs"))))
  flush.console()
  
  # Métricas de validação (resto igual)
  history_df <- as.data.frame(history)

  val_loss_df  <- subset(history_df, metric == "loss"     & data == "validation")
  val_acc_df   <- subset(history_df, metric == "acc"      & data == "validation")
  val_top3_df  <- subset(history_df, metric == "top3_acc" & data == "validation")

  if (nrow(val_loss_df) > 0) {
    # epoch com menor val_loss
    best_epoch_idx <- which.min(val_loss_df$value)
    best_epoch     <- val_loss_df$epoch[best_epoch_idx]

    best_val_loss <- val_loss_df$value[best_epoch_idx]

    # pegar acc e top3 no mesmo epoch (se existirem)
    best_val_acc <- if (nrow(val_acc_df) > 0) {
      vals <- val_acc_df$value[val_acc_df$epoch == best_epoch]
      if (length(vals) > 0) vals[1] else NA_real_
    } else {
      NA_real_
    }

    best_val_top3 <- if (nrow(val_top3_df) > 0) {
      vals <- val_top3_df$value[val_top3_df$epoch == best_epoch]
      if (length(vals) > 0) vals[1] else NA_real_
    } else {
      NA_real_
    }

  } else {
    # fallback defensivo, caso algo estranho aconteça
    best_val_loss  <- NA_real_
    best_val_acc   <- if (nrow(val_acc_df)  > 0) max(val_acc_df$value,  na.rm = TRUE) else NA_real_
    best_val_top3  <- if (nrow(val_top3_df) > 0) max(val_top3_df$value, na.rm = TRUE) else NA_real_
  }

  data.frame(
    l2_reg        = l2_reg,
    dropout_conv  = dropout_conv,
    dropout_dense = dropout_dense,
    rotation_deg  = rotation_deg,
    translation   = translation_frac,
    zoom          = zoom_frac,
    dense_units   = dense_units,
    optimizer     = optimizer_name,
    learning_rate = learning_rate,
    weight_decay  = weight_decay,
    batch_size    = batch_size,
    best_val_loss = best_val_loss,
    best_val_acc  = best_val_acc,
    best_val_top3 = best_val_top3
  )
}

# 6) Espaço de busca de hiperparâmetros
grid <- expand.grid(
  l2_reg        = c(5e-5, 1e-4, 3e-4),
  dropout_conv  = c(0.15, 0.25, 0.35),
  dropout_dense = c(0.3, 0.5, 0.6),
  rotation_deg  = c(10, 20),
  translation   = c(0.05, 0.10, 0.15),
  zoom          = c(0.0, 0.10, 0.20),
  dense_units   = c(128L, 256L, 512L),
  optimizer     = c("adam", "adamw", "lion"),
  learning_rate = c(1e-2, 3e-3, 1e-3, 3e-4),
  weight_decay  = c(0, 1e-5, 1e-4),
  batch_size    = c(128L, 256L),      
  stringsAsFactors = FALSE
)

# Random search em subconjunto das combinações 
set.seed(42)
n_configs  <- min(10L, nrow(grid))  # ajustar p/ 5, 20, etc.
config_ids <- sample(seq_len(nrow(grid)), size = n_configs)

# subconjunto de configs que vamos rodar
grid_sub <- grid[config_ids, , drop = FALSE]

cat(sprintf("\n[HPO] Rodando %d configurações em paralelo...\n", nrow(grid_sub)))
flush.console()

results_list <- future_lapply(
  X = seq_len(nrow(grid_sub)),
  FUN = function(j) {
    cfg <- grid_sub[j, ]

    cat(sprintf("[HPO] (worker) Config %d/%d — row %d\n",
                j, nrow(grid_sub), config_ids[j]))
    flush.console()

    with(cfg, run_config(
      l2_reg           = l2_reg,
      dropout_conv     = dropout_conv,
      dropout_dense    = dropout_dense,
      rotation_deg     = rotation_deg,
      translation_frac = translation,
      zoom_frac        = zoom,
      dense_units      = dense_units,
      optimizer_name   = optimizer,
      learning_rate    = learning_rate,
      weight_decay     = weight_decay,
      epochs           = epochs_hpo,
      batch_size       = batch_size  
    ))

  },
  future.seed     = TRUE,                           
  future.packages = c("keras3", "tensorflow", "reticulate")
)

results <- do.call(rbind, results_list)

cat("\nTop 10 configs (por best_val_acc):\n")
print(head(results[order(-results$best_val_acc), ], 10))

# 7) Melhor config 
best_cfg_idx <- which.max(results$best_val_acc)
best_cfg     <- results[best_cfg_idx, ]
cat("\nMelhor configuração encontrada:\n")
print(best_cfg)

model_cnn <- build_mnist_cnn_model(
  input_shape      = c(img_rows, img_cols, 1L),
  num_classes      = num_classes,
  l2_reg           = best_cfg$l2_reg,
  dropout_conv     = best_cfg$dropout_conv,
  dropout_dense    = best_cfg$dropout_dense,
  rotation_deg     = best_cfg$rotation_deg,
  translation_frac = best_cfg$translation,
  zoom_frac        = best_cfg$zoom,
  dense_units      = best_cfg$dense_units,
  optimizer_name   = best_cfg$optimizer,
  learning_rate    = best_cfg$learning_rate,
  weight_decay     = best_cfg$weight_decay
)

cat("\nResumo do modelo (melhor config):\n")
print(summary(model_cnn))

history_final <- model_cnn %>% fit(
  x = x_train_final,
  y = y_train_final,
  batch_size      = best_cfg$batch_size,
  epochs          = epochs_full,  
  validation_data = list(x_val, y_val),
  callbacks       = callbacks_train,
  verbose         = 2
)

# Melhor modelo salvo (pelo menos - val_loss)
best_model <- model_cnn
if (file.exists("best_mnist_cnn.keras")) {
  best_model <- load_model("best_mnist_cnn.keras")
}

# 8) Eval no test
eval_test <- best_model %>% evaluate(x_test, y_test_cat, verbose = 0)

cat("\n== Test metrics ==\n")
cat("Test loss       :", eval_test[["loss"]],     "\n")
cat("Test accuracy   :", eval_test[["acc"]],      "\n")
cat("Test top-3 acc. :", eval_test[["top3_acc"]], "\n")

# Probs (n_test x 10)
pred_prob <- best_model %>% predict(x_test)

# Classe predita (0..9)
pred_class <- max.col(pred_prob) - 1L

# Accuracy manual
test_accuracy <- mean(pred_class == y_test)
cat("Test accuracy (manual):", test_accuracy, "\n")

# Confusion matrix
conf_mat <- table(Predicted = pred_class, Actual = y_test)
cat("\nConfusion matrix:\n")
print(conf_mat)

# Data frame true x preds
results_df <- data.frame(
  y_true = y_test,
  y_pred = pred_class,
  stringsAsFactors = FALSE
)

head(results_df)

# Entregável: eval usando base_avaliacao.rds + html
load_base_avaliacao <- function(path, img_rows, img_cols) {
  x_raw <- readRDS(path)

  if (!is.array(x_raw) || length(dim(x_raw)) != 3L) {
    stop("base_avaliacao.rds deve ser um array 3D (n, rows, cols).")
  }

  if (dim(x_raw)[2] != img_rows || dim(x_raw)[3] != img_cols) {
    stop(
      "Dimensões diferentes das esperadas: ",
      paste(dim(x_raw), collapse = " x "),
      " (esperado: n x ", img_rows, " x ", img_cols, ")"
    )
  }

  n <- dim(x_raw)[1]

  # (n, rows, cols, channels)
  x <- array_reshape(x_raw, c(n, img_rows, img_cols, 1L))

  # Normalizar para [0,1]
  x <- x / 255

  # Neste .rds não há rótulos -> y = NULL
  list(
    x = x,
    y = NULL
  )
}

# Eval COM rótulos (caso algum dia exista y)
evaluate_with_labels <- function(model, x, y, num_classes, dataset_name) {
  if (is.matrix(y) || is.array(y)) {
    if (ncol(y) == num_classes) {
      y_vec <- max.col(y) - 1L
      y_cat <- y
    } else {
      y_vec <- as.integer(y)
      y_cat <- to_categorical(y_vec, num_classes)
    }
  } else {
    y_vec <- as.integer(y)
    y_cat <- to_categorical(y_vec, num_classes)
  }

  eval <- model %>% evaluate(x, y_cat, verbose = 0)

  prob <- model %>% predict(x)
  pred_class <- max.col(prob) - 1L

  acc_manual <- mean(pred_class == y_vec)
  conf_mat   <- table(Predicted = pred_class, Actual = y_vec)

  list(
    dataset_name = dataset_name,
    n_samples    = length(y_vec),
    has_labels   = TRUE,
    eval         = eval,
    acc_manual   = acc_manual,
    conf_mat     = conf_mat
  )
}

# Eval p/ base_avaliacao.rds SEM rótulos
evaluate_unlabeled <- function(model, x, dataset_name) {
  prob <- model %>% predict(x)
  pred_class <- max.col(prob) - 1L
  class_counts <- table(pred_class)

  list(
    dataset_name = dataset_name,
    n_samples    = length(pred_class),
    has_labels   = FALSE,
    class_counts = class_counts
  )
}

# Código html
generate_mnist_html <- function(html_path, new_eval) {
  con <- file(html_path, open = "wt", encoding = "UTF-8")
  on.exit(close(con), add = TRUE)

  # Cabeçalho + CSS
  cat(
    "<!DOCTYPE html><html lang='pt-br'><head><meta charset='utf-8'>",
    "<title>Relatório CNN — MNIST</title>",
    "<style>
      :root{--fg:#222;--muted:#555;--line:#e6e6e6;--bg:#fff}
      *{box-sizing:border-box}
      body{font-family:-apple-system,system-ui,Segoe UI,Roboto,Arial,sans-serif;margin:24px;color:var(--fg);background:var(--bg)}
      h1,h2,h3{margin:1.0em 0 .5em}
      p{margin:.6em 0;line-height:1.45}
      table{border-collapse:collapse;width:100%;margin:8px 0 16px}
      th,td{border:1px solid var(--line);padding:6px 8px}
      th{text-align:left;background:#f7f7f7;position:sticky;top:0}
      td.num{text-align:right;font-variant-numeric:tabular-nums}
      tbody tr:nth-child(even){background:#fbfbfb}
      ul{margin:.2em 0 .8em 1.2em}
      .small{color:var(--muted);font-size:.9em}
    </style></head><body>\n",
    file = con, sep = ""
  )

  cat("<h1>Relatório CNN — MNIST</h1>\n", file = con)

  # top-10 configs
  if (exists("results", inherits = TRUE) && is.data.frame(results) && nrow(results) > 0) {
    hpo_results <- head(results[order(-results$best_val_acc), ], 10)

    cat("<h2>Busca de hiperparâmetros — top 10 configs (por val_acc)</h2>\n", file = con)
    cat("<table><thead><tr>", file = con)
    cols <- colnames(hpo_results)
    for (nm in cols) cat(sprintf("<th>%s</th>", nm), file = con)
    cat("</tr></thead><tbody>\n", file = con)
    for (i in seq_len(nrow(hpo_results))) {
      cat("<tr>", file = con)
      for (nm in cols) {
        val <- hpo_results[i, nm]
        is_num <- is.numeric(val)
        cat(
          sprintf(
            "<td%s>%s</td>",
            if (is_num) " class='num'" else "",
            as.character(val)
          ),
          file = con
        )
      }
      cat("</tr>\n", file = con)
    }
    cat("</tbody></table>\n", file = con)
  }

  # Best config
  if (exists("best_cfg", inherits = TRUE) && is.data.frame(best_cfg) && nrow(best_cfg) == 1) {
    cat("<h2>Melhor configuração encontrada</h2>\n", file = con)
    cat("<table><thead><tr>", file = con)
    cols <- colnames(best_cfg)
    for (nm in cols) cat(sprintf("<th>%s</th>", nm), file = con)
    cat("</tr></thead><tbody><tr>", file = con)
    for (nm in cols) {
      cat(sprintf("<td>%s</td>", as.character(best_cfg[[nm]])), file = con)
    }
    cat("</tr></tbody></table>\n", file = con)
  }

  # Metrics 
  if (exists("eval_test", inherits = TRUE)) {
    cat("<h2>Desempenho no conjunto de teste (MNIST)</h2>\n<ul>\n", file = con)
    if (!is.null(eval_test[["loss"]])) {
      cat(sprintf("<li>Loss: %.4f</li>\n", eval_test[["loss"]]), file = con)
    }
    if (!is.null(eval_test[["acc"]])) {
      cat(sprintf("<li>Acurácia: %.4f</li>\n", eval_test[["acc"]]), file = con)
    }
    if (!is.null(eval_test[["top3_acc"]])) {
      cat(sprintf("<li>Top-3 acc: %.4f</li>\n", eval_test[["top3_acc"]]), file = con)
    }
    cat("</ul>\n", file = con)
  }

  # Matriz de confusão, teste
  if (exists("conf_mat", inherits = TRUE)) {
    cm <- as.matrix(conf_mat)
    cat("<h3>Matriz de confusão (teste)</h3>\n<table><thead><tr><th>Pred \\ Real</th>", file = con)
    for (j in colnames(cm)) cat(sprintf("<th>%s</th>", j), file = con)
    cat("</tr></thead><tbody>\n", file = con)
    for (i in rownames(cm)) {
      cat(sprintf("<tr><th>%s</th>", i), file = con)
      for (j in colnames(cm)) {
        cat(sprintf("<td class='num'>%d</td>", cm[i, j]), file = con)
      }
      cat("</tr>\n", file = con)
    }
    cat("</tbody></table>\n", file = con)
  }

  # Resultados base_avaliacao.rds
  if (!is.null(new_eval)) {
    cat("<h2>Avaliação em base_avaliacao.rds</h2>\n", file = con)
    if (!is.null(new_eval$dataset_name)) {
      cat(sprintf("<p>Dataset: %s</p>\n", new_eval$dataset_name), file = con)
    }
    if (!is.null(new_eval$n_samples)) {
      cat(sprintf("<p>N = %d amostras</p>\n", as.integer(new_eval$n_samples)), file = con)
    }

    if (isTRUE(new_eval$has_labels)) {
      cat("<h3>Métricas (com rótulos)</h3>\n<ul>\n", file = con)
      if (!is.null(new_eval$eval[["loss"]])) {
        cat(sprintf("<li>Loss: %.4f</li>\n", new_eval$eval[["loss"]]), file = con)
      }
      if (!is.null(new_eval$eval[["acc"]])) {
        cat(sprintf("<li>Acurácia (Keras): %.4f</li>\n", new_eval$eval[["acc"]]), file = con)
      }
      if (!is.null(new_eval$eval[["top3_acc"]])) {
        cat(sprintf("<li>Top-3 acc (Keras): %.4f</li>\n", new_eval$eval[["top3_acc"]]), file = con)
      }
      if (!is.null(new_eval$acc_manual)) {
        cat(sprintf("<li>Acurácia manual: %.4f</li>\n", new_eval$acc_manual), file = con)
      }
      cat("</ul>\n", file = con)

      if (!is.null(new_eval$conf_mat)) {
        cm_new <- as.matrix(new_eval$conf_mat)
        cat("<h3>Matriz de confusão (base_avaliacao.rds)</h3>\n<table><thead><tr><th>Pred \\ Real</th>", file = con)
        for (j in colnames(cm_new)) cat(sprintf("<th>%s</th>", j), file = con)
        cat("</tr></thead><tbody>\n", file = con)
        for (i in rownames(cm_new)) {
          cat(sprintf("<tr><th>%s</th>", i), file = con)
          for (j in colnames(cm_new)) {
            cat(sprintf("<td class='num'>%d</td>", cm_new[i, j]), file = con)
          }
          cat("</tr>\n", file = con)
        }
        cat("</tbody></table>\n", file = con)
      }
    } else {
      if (!is.null(new_eval$class_counts)) {
        cc <- new_eval$class_counts
        cat("<h3>Base sem rótulos — distribuição de classes preditas</h3>\n", file = con)
        cat("<table><thead><tr><th>Classe</th><th>Contagem</th></tr></thead><tbody>\n", file = con)
        for (k in names(cc)) {
          cat(sprintf("<tr><td>%s</td><td class='num'>%d</td></tr>\n", k, cc[[k]]), file = con)
        }
        cat("</tbody></table>\n", file = con)
      }
    }
  }

  # Modelo salvo
  if (exists("best_model", inherits = TRUE)) {
    cat("<h2>Resumo do modelo (best_model)</h2>\n<pre>\n", file = con)
    cat(
      paste(capture.output(summary(best_model)), collapse = "\n"),
      "\n</pre>\n",
      file = con
    )
  }

  cat("</body></html>\n", file = con)
  invisible(html_path)
}

output_dir <- "outputs_mnist_cnn"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
}
html_path <- file.path(output_dir, "relatorio_mnist_cnn.html")

base_eval_path <- "/Users/akatsurada/Documents/INSPER/StatisticsII/Aula7_lab/base_avaliacao.rds"

if (!file.exists(base_eval_path)) {
  warning("Arquivo da base de avaliação não encontrado: ", base_eval_path)
} else {
  # Load base avaliacao
  base_eval <- load_base_avaliacao(
    path     = base_eval_path,
    img_rows = img_rows,
    img_cols = img_cols
  )

  # Eval p/ base avaliacao
  if (!is.null(base_eval$y)) {
    new_eval <- evaluate_with_labels(
      model        = best_model,
      x            = base_eval$x,
      y            = base_eval$y,
      num_classes  = num_classes,
      dataset_name = "base_avaliacao.rds"
    )
  } else {
    new_eval <- evaluate_unlabeled(
      model        = best_model,
      x            = base_eval$x,
      dataset_name = "base_avaliacao.rds (sem rótulos)"
    )
  }

  # Gerar html
  generate_mnist_html(
    html_path = html_path,
    new_eval  = new_eval
  )

  cat("\nRelatório HTML gerado em:\n  ", html_path, "\n", sep = "")
}
