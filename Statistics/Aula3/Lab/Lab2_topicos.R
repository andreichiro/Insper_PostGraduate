# instalar bibliotecas
pkgs <- c("modeldata","tidyverse","skimr","pROC","yardstick","ggplot2","httpgd")
to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_install)) install.packages(to_install, dependencies = TRUE)

if (requireNamespace("httpgd", quietly = TRUE)) {
  if (!httpgd::hgd_running()) httpgd::hgd()
  options(device = function(...) httpgd::hgd())
}

align_to_train_levels <- function(train, test) {
  fcols <- names(train)[sapply(train, is.factor)]
  for (cl in fcols) {
    test[[cl]] <- factor(as.character(test[[cl]]), levels = levels(train[[cl]]))
  }
  test
}


# Carregando as bibliotecas
library(modeldata) 
library(tidyverse)
library(skimr)
library(pROC)
library(yardstick)  
# Abrimos as "ferramentas" que vamos usar ao longo do script

# base de dados
# Trazemos a base de exemplo para a memória e damos uma olhada rápida
credit_data %>% head()
data("credit_data") #salvar como objeto

credit_data %>% skim()

# Analise Descritiva
# Fazemos desenhos simples para conhecer os dados
ggplot(credit_data, aes(x=Income, y=Debt, color=Status))+
  geom_point(alpha=0.6)

ggplot(credit_data, aes(x=Marital))+
  geom_bar()

# analisando a distribuicao da renda de casados
credit_data %>% filter(Marital=='married') %>% 
  ggplot(aes(x=Income, y=Debt, color=Status))+
  geom_point(alpha=0.6)

# comparacao da renda x marital 
ggplot(credit_data,aes( x=Marital,fill=Marital,y=Income))+
  geom_violin()

ggplot(credit_data,aes( x=Marital,fill=Marital,y=Income))+
  geom_boxplot()

# Preparação da base

# remoção dos missings
# Tiramos linhas com valores faltando
df <- na.omit(credit_data)


##### Treinamento do Modelo

# 70% Treinamento e 30% Teste
# Separamos dados para aprender (treino) e para testar (teste)
set.seed(61)

idx <- sample(nrow(df),size = 0.7*nrow(df), replace = FALSE)

base_treino <- df[idx,]
base_teste <- df[-idx,]

# Ajuste do modelo 
# Variável Resposta -> Status
# Problema de classificação

base_treino %>% head()

# Reg. Logística
# Criamos uma "regra" que dá a chance de a pessoa ser "good"
# Status ~ .
fit <- glm(Status ~ ., base_treino, family="binomial") # ~. significa que estou usando todas as demais vars como explicativas
#  family="binomial" para ajustar uma reg logistica

summary(fit)

# Predição na teste
# Calculamos a pontuação (0 a 1) para cada pessoa do teste
(pred_logistica <- predict(fit, base_teste, type="response"))
base_teste['pred']<- predict(fit, base_teste, type="response")
base_teste %>% head()

# Densidade 
# Vemos como essas pontuações se distribuem
base_teste %>% ggplot(aes(x=pred))+geom_histogram()

# Avaliação do Modelo

# Métrica de acuracia
# Usamos um corte fixo (0.65) para transformar pontuação em rótulo e medir acertos
corte= 0.65 # entao acima desse ponto, todos sao bons pagadores
mean(base_teste$Status == ifelse(base_teste$pred>=corte, "good", "bad"))

# Curva ROC
# Comparamos vários cortes para ver o melhor equilíbrio entre acertos e erros
roc <- roc(base_teste$Status,base_teste$pred)
roc$auc

plot(roc)
# A "nota geral" dessa curva aparece em roc$auc

# coordenadas
coords(roc, seq(0.5,0.8,0.05),ret=c("accuracy","specificity","npv","ppv"))
# Mostramos medidas para vários cortes

yardstick::roc_curve(
  tibble(truth = base_teste$Status, .pred_good = base_teste$pred),
  truth, .pred_good, event_level = "second"
) %>% autoplot()

tibble(classe = base_teste$Status, marcador = pred_logistica) %>% 
  roc_curve(classe, marcador, event_level = "second") %>% 
  autoplot()

# Atividade
# Modelo 2 -> modelo com pessoas sem divida ativa
# Coluna Debt = 0
base_treino %>% head()

# Avaliação em cima da base filtrada - Debt = 0
# Base filtrada: Debt == 0
# Só quem não tem dívida ativa s/ a coluna de dívida 
df0 <- df %>% filter(Debt == 0) %>% select(-Debt)

# Segurança: remover preditores constantes 
# Se uma coluna não muda p/ ninguém, ela não ajuda a diferenciar e é removida
is_constant <- function(x) {
  if (is.numeric(x)) return(sd(x, na.rm = TRUE) == 0)
  if (is.factor(x) || is.character(x)) return(dplyr::n_distinct(x) <= 1)
  FALSE
}
const_cols <- names(df0)[sapply(df0, is_constant)]
if (length(const_cols)) {
  message("Dropping constant columns: ", paste(const_cols, collapse = ", "))
  df0 <- df0 %>% dplyr::select(-dplyr::all_of(const_cols))
}

# Novo particionamento na população filtrada
# Separamos de novo: teste e treino
set.seed(61)
idx0 <- sample(nrow(df0), size = floor(0.7 * nrow(df0)), replace = FALSE)
base_treino_0 <- df0[idx0, ]
base_teste_0  <- df0[-idx0, ]

# Alinhar níveis fatoriais no teste aos vistos no treino (evita previsões NA por níveis não vistos)
# garantir que os nomes das categorias do teste combinam com os do treino
base_teste_0 <- align_to_train_levels(base_treino_0, base_teste_0)

# Ajustar regressão logística (Status ~ .) no segmento
# Nova regra gml só para esse grupo (Debt == 0)
fit_0 <- glm(Status ~ ., data = base_treino_0, family = "binomial")
summary(fit_0)

# Prever e avaliar
# Fazemos as pontuações (0 a 1) nesse grupo e medir
base_teste_0$pred <- predict(fit_0, base_teste_0, type = "response")

# Escolher um limiar orientado por dados (índice de Youden) para este segmento
# Em vez de um corte fixo, escolhemos automaticamente o melhor corte
# Aqui tb poderia ser o corte fixo de 0.65
roc_0_refit <- pROC::roc(response = base_teste_0$Status,
                         predictor = base_teste_0$pred,
                         levels = c("bad","good"),
                         direction = "<")
best_thresh <- pROC::coords(roc_0_refit, x = "best",
                            best.method = "youden", ret = "threshold")

pred_lab_0 <- ifelse(base_teste_0$pred >= best_thresh, "good", "bad") %>% factor(levels = c("bad","good"))
acc_0_refit <- mean(base_teste_0$Status == pred_lab_0)
auc_0_refit <- as.numeric(roc_0_refit$auc)

cat(sprintf("Debt==0 segment model — AUC: %.3f | Accuracy @Youden(%.3f): %.3f\n",
            auc_0_refit, best_thresh, acc_0_refit))

# Matriz de confusão e grade de limiares
# Tabela simples de acertos e erros e medidas para vários cortes
yardstick::conf_mat(tibble(truth = base_teste_0$Status, estimate = pred_lab_0), truth, estimate)

pROC::coords(roc_0_refit, seq(0.5, 0.8, 0.05),
             ret = c("accuracy","specificity","npv","ppv"))

# Plots
# Pontuações por classe e a curva de cortes
p_hist_debt0 <- ggplot(base_teste_0, aes(x = pred, fill = Status)) +
  geom_histogram(bins = 30, alpha = 0.6, position = "identity") +
  labs(title = "Predicted probability by class — Debt==0 (segment model)",
       x = "P(Status == 'good')", y = "Count")
print(p_hist_debt0)
ggsave("hist_debt0.jpg", p_hist_debt0, width = 9, height = 6, dpi = 300)

# Curva que mostra a troca entre acertos e erros em vários cortes
p_roc_debt0 <- yardstick::roc_curve(
  tibble(truth = base_teste_0$Status, .pred_good = base_teste_0$pred),
  truth, .pred_good, event_level = "second"
) %>% autoplot() + ggplot2::ggtitle("ROC — Debt==0 (segment model)")
print(p_roc_debt0)
ggsave("roc_debt0_yardstick.jpg", p_roc_debt0, width = 9, height = 6, dpi = 300)

# Função que repete o mesmo passo a passo para qualquer filtro
fit_eval_segment <- function(data,
                             filter_expr,
                             response = "Status",
                             seed = 61,
                             threshold = NULL,  # se NULL, escolher Youden
                             positive = "good",
                             plot_title_suffix = "") {
  # 1) Filtrar
  d <- data %>% filter({{ filter_expr }})
  if (!nrow(d)) stop("No rows after filtering.")

  # 2) Remover preditores constantes (inclui a coluna usada no filtro, ex.: Debt)
  is_constant <- function(x) {
    if (is.numeric(x)) return(sd(x, na.rm = TRUE) == 0)
    if (is.factor(x) || is.character(x)) return(dplyr::n_distinct(x) <= 1)
    FALSE
  }
  const_cols <- names(d)[names(d) != response & sapply(d, is_constant)]
  if (length(const_cols)) d <- d %>% select(-all_of(const_cols))

  # 3) Separar treino/teste
  set.seed(seed)
  idx <- sample(nrow(d), size = floor(0.7 * nrow(d)), replace = FALSE)
  train <- d[idx, ]
  test  <- d[-idx, ]

  # 4) Alinhar níveis fatoriais
  fcols <- names(train)[sapply(train, is.factor)]
  for (cl in fcols) test[[cl]] <- factor(as.character(test[[cl]]), levels = levels(train[[cl]]))
  test <- test %>% drop_na()

  # 5) Ajustar GLM
  predictors <- setdiff(names(train), response)
  form <- as.formula(paste(response, "~", paste(predictors, collapse = " + ")))
  mdl <- glm(form, data = train, family = "binomial")

  # 6) Prever
  test$.pred_good <- predict(mdl, test, type = "response")

  # 7) Limiar: fixo ou Youden
  roc_obj <- pROC::roc(response = test[[response]],
                       predictor = test$.pred_good,
                       levels = c(setdiff(levels(test[[response]]), positive), positive),
                       direction = "<")
  if (is.null(threshold)) {
    threshold <- pROC::coords(roc_obj, x = "best", best.method = "youden", ret = "threshold")
  }

  neg <- setdiff(levels(test[[response]]), positive)
  test$.pred_class <- ifelse(test$.pred_good >= threshold, positive, neg) %>%
    factor(levels = c(neg, positive))

  # 8) Métricas
  acc <- mean(test[[response]] == test$.pred_class)
  auc <- as.numeric(roc_obj$auc)
  cm  <- yardstick::conf_mat(test, truth = !!sym(response), estimate = .pred_class)

  # 9) Gráficos
  p_seg_roc <- yardstick::roc_curve(test, truth = !!sym(response), .pred_good, event_level = "second") %>%
    autoplot() + ggplot2::ggtitle(paste0("ROC — ", plot_title_suffix))
  print(p_seg_roc)
  ggsave(paste0("roc_", gsub("[^A-Za-z0-9_-]+", "_", plot_title_suffix), ".jpg"),
         p_seg_roc, width = 9, height = 6, dpi = 300)

  list(
    model      = mdl,
    train_n    = nrow(train),
    test_n     = nrow(test),
    auc        = auc,
    accuracy   = acc,
    threshold  = threshold,
    conf_mat   = cm,
    roc_object = roc_obj,
    test_data  = test
  )
}

# Exemplo: executar a função para Debt==0 
res_debt0 <- fit_eval_segment(
  data = df,
  filter_expr = Debt == 0,
  plot_title_suffix = "Debt==0"
)

res_debt0$auc
res_debt0$accuracy
res_debt0$threshold
res_debt0$conf_mat

# Comparação entre modelos (Modelo 1 vs Modelo 2), no msm teste (Debt==0) 

# Conjunto comum de avaliação: subset do base_teste com Debt == 0
common_debt0 <- base_teste %>% dplyr::filter(Debt == 0)

# Predições do Modelo 1 (geral) em base_teste$pred
common_debt0$.pred_m1 <- common_debt0$pred

# Predições do Modelo 2 (segmento) no mesmo conjunto (removendo Debt e alinhando níveis)
m2_input <- common_debt0 %>% dplyr::select(-Debt)
m2_input <- align_to_train_levels(base_treino_0, m2_input)
common_debt0$.pred_m2 <- predict(fit_0, m2_input, type = "response")

# Tabela de comparação
cmp <- common_debt0 %>% 
  dplyr::select(truth = Status, .pred_m1, .pred_m2)

# AUC (yardstick)
auc_m1 <- yardstick::roc_auc(cmp, truth = truth, .pred_m1, event_level = "second")$.estimate
auc_m2 <- yardstick::roc_auc(cmp, truth = truth, .pred_m2, event_level = "second")$.estimate

# Acurácia com corte fixo 0.65 (mesmo da parte 1)
corte <- 0.65
acc_m1_065 <- mean(cmp$truth == ifelse(cmp$.pred_m1 >= corte, "good", "bad"))
acc_m2_065 <- mean(cmp$truth == ifelse(cmp$.pred_m2 >= corte, "good", "bad"))

# Acurácia no melhor corte (Youden) de cada modelo
roc1 <- pROC::roc(response = cmp$truth, predictor = cmp$.pred_m1,
                  levels = c("bad","good"), direction = "<")
roc2 <- pROC::roc(response = cmp$truth, predictor = cmp$.pred_m2,
                  levels = c("bad","good"), direction = "<")

thr1 <- pROC::coords(roc1, x = "best", best.method = "youden", ret = "threshold")
thr2 <- pROC::coords(roc2, x = "best", best.method = "youden", ret = "threshold")

pred_lab_m1 <- ifelse(cmp$.pred_m1 >= thr1, "good", "bad") %>% factor(levels = c("bad","good"))
pred_lab_m2 <- ifelse(cmp$.pred_m2 >= thr2, "good", "bad") %>% factor(levels = c("bad","good"))

acc_m1_youden <- mean(cmp$truth == pred_lab_m1)
acc_m2_youden <- mean(cmp$truth == pred_lab_m2)

# Matriz de confusão em Youden
cm_m1 <- yardstick::conf_mat(tibble::tibble(truth = cmp$truth, estimate = pred_lab_m1), truth, estimate)
cm_m2 <- yardstick::conf_mat(tibble::tibble(truth = cmp$truth, estimate = pred_lab_m2), truth, estimate)
print(cm_m1); print(cm_m2)

# Curvas ROC sobrepostas 
roc_df_m1 <- yardstick::roc_curve(cmp, truth, .pred_m1, event_level = "second") %>% dplyr::mutate(model = "Modelo 1 (geral)")
roc_df_m2 <- yardstick::roc_curve(cmp, truth, .pred_m2, event_level = "second") %>% dplyr::mutate(model = "Modelo 2 (Debt==0)")
roc_df <- dplyr::bind_rows(roc_df_m1, roc_df_m2)

p_roc_cmp <- ggplot2::ggplot(roc_df, ggplot2::aes(x = 1 - specificity, y = sensitivity, color = model)) +
  ggplot2::geom_path(linewidth = 1) +
  ggplot2::geom_abline(linetype = 3) +
  ggplot2::coord_equal() +
  ggplot2::labs(title = "ROC — Comparação Modelo 1 vs Modelo 2 (teste comum: Debt==0)",
                x = "1 - Especificidade (FPR)", y = "Sensibilidade (TPR)", color = "Modelo")
print(p_roc_cmp)
ggplot2::ggsave("roc_comparacao_m1_m2.jpg", p_roc_cmp, width = 9, height = 6, dpi = 300)

# Tabela de métricas e gráfico comparativo 
metrics_tbl <- tibble::tibble(
  modelo      = c("Modelo 1 (geral)","Modelo 2 (Debt==0)"),
  AUC         = c(as.numeric(auc_m1), as.numeric(auc_m2)),
  Acc_065     = c(acc_m1_065, acc_m2_065),
  Acc_Youden  = c(acc_m1_youden, acc_m2_youden),
  Thr_Youden  = c(as.numeric(thr1), as.numeric(thr2))
)
print(metrics_tbl)

metrics_long <- metrics_tbl %>%
  dplyr::select(-Thr_Youden) %>%
  tidyr::pivot_longer(cols = -modelo, names_to = "metrica", values_to = "valor")

p_metrics <- ggplot2::ggplot(metrics_long, ggplot2::aes(x = metrica, y = valor, fill = modelo)) +
  ggplot2::geom_col(position = ggplot2::position_dodge()) +
  ggplot2::scale_y_continuous(limits = c(0, 1)) +
  ggplot2::labs(title = "Métricas — Comparação Modelo 1 vs Modelo 2 (teste comum: Debt==0)",
                x = NULL, y = NULL, fill = "Modelo")
print(p_metrics)
ggplot2::ggsave("metricas_comparacao_m1_m2.jpg", p_metrics, width = 9, height = 6, dpi = 300)

