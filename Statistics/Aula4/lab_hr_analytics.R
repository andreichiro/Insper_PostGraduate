# 0) Setup: pacotes & opções #

# Installs
pkgs <- c(
  "tidyverse","tidymodels","glmnet","vip","broom","MASS",
  "ggcorrplot","naniar","skimr","patchwork","scales","pROC"
)
new_pkgs <- setdiff(pkgs, rownames(installed.packages()))
if (length(new_pkgs) > 0) install.packages(new_pkgs, dependencies = TRUE)

# Carregar libraries
library(tidyverse)
library(tidymodels)
library(glmnet)
library(vip)
library(broom)
library(MASS)
library(ggcorrplot)
library(naniar)
library(skimr)
library(patchwork)
library(scales)
library(pROC)

theme_set(theme_minimal(base_size = 12))
set.seed(1234)

# salvar
dir.create("plots", showWarnings = FALSE)
dir.create("outputs", showWarnings = FALSE)

save_plot <- function(p, filename, w = 9, h = 6, dpi = 300){
  out_path <- file.path("plots", filename)
  ext <- tools::file_ext(out_path)
  if (tolower(ext) %in% c("jpg","jpeg")) {
    ggplot2::ggsave(out_path, plot = p, width = w, height = h, dpi = dpi,
                    device = function(...){ grDevices::jpeg(..., type = "cairo") })
  } else if (tolower(ext) == "png") {
    ggplot2::ggsave(out_path, plot = p, width = w, height = h, dpi = dpi,
                    device = function(...){ grDevices::png(..., type = "cairo") })
  } else {
    ggplot2::ggsave(out_path, plot = p, width = w, height = h, dpi = dpi)
  }
}
                      

# 1) Leitura e preparação dos dados
csv_path <- "/Users/akatsurada/Documents/INSPER/Statistics/Aula4/employee_attrition.csv"

# Carregar planilha
hr <- readr::read_csv(csv_path, show_col_types = FALSE)

# Casting e limpeza  
# Texto para categorias
hr <- hr %>%
  mutate(across(where(is.character), as.factor))

# Remover id
hr <- hr %>% dplyr::select(-EmployeeNumber)

# Variável alvo, garantindo que "Yes" seja o primeiro nível 
# P/ AUC, definir a classe "positiva"
hr <- hr %>%
  mutate(Attrition = fct_relevel(as.factor(Attrition), "Yes", "No"))

# Ver NA
miss_summary <- naniar::miss_var_summary(hr)
print(miss_summary)
readr::write_csv(miss_summary, "outputs/missing_summary.csv")
cat(">> Diagnóstico: 'Attrition' é binária (Yes/No) -> PROBLEMA DE CLASSIFICAÇÃO.\n")

# 2) EDA (Análise Exploratória) com gráficos

# Skimr p/ ver dados gerais
skimr::skim(hr)

# Percentual de attrition 
p_class <- hr %>%
  count(Attrition) %>%
  mutate(perc = n / sum(n)) %>%
  ggplot(aes(x = Attrition, y = perc, fill = Attrition)) +
  geom_col() +
  geom_text(aes(label = scales::percent(perc, accuracy = 0.1)), vjust = -0.4) +
  scale_y_continuous(labels = percent_format()) +
  labs(title = "Distribuição de Attrition",
       x = NULL, y = "Proporção") +
  theme(legend.position = "none")

# Distribuições de variáveis numéricas por Attrition 
num_cols <- hr %>% dplyr::select(where(is.numeric)) %>% names()

num_focus <- intersect(num_cols, c("Age","MonthlyIncome","DistanceFromHome","TotalWorkingYears"))

plot_num <- function(var){
  ggplot(hr, aes(x = .data[[var]], fill = Attrition)) +
    geom_density(alpha = 0.3) +
    labs(title = paste("Densidade de", var, "por Attrition"),
         x = var, y = "Densidade") +
    scale_fill_brewer(palette = "Dark2")
}
p_age   <- plot_num("Age")
p_inc   <- if ("MonthlyIncome" %in% num_cols) plot_num("MonthlyIncome") else NULL
p_dist  <- if ("DistanceFromHome" %in% num_cols) plot_num("DistanceFromHome") else NULL
p_twy   <- if ("TotalWorkingYears" %in% num_cols) plot_num("TotalWorkingYears") else NULL

# Proporção de Attrition por categorias 
cat_focus <- intersect(names(hr), c("BusinessTravel","JobRole","OverTime","EducationField","Department"))
props_cat_plots <- lapply(cat_focus, function(v){
  hr %>%
    group_by(.data[[v]], Attrition) %>%
    tally() %>%
    group_by(.data[[v]]) %>%
    mutate(prop = n/sum(n)) %>%
    ggplot(aes(x = .data[[v]], y = prop, fill = Attrition)) +
    geom_col(position = "fill") +
    scale_y_continuous(labels = percent_format()) +
    coord_flip() +
    labs(title = paste("Attrition por", v), x = v, y = "Proporção") +
    scale_fill_brewer(palette = "Dark2")
})

# Correlação entre vars numéricas 
if (length(num_cols) > 1) {
  num_df <- hr %>% dplyr::select(all_of(num_cols))
  sds <- vapply(num_df, function(x) stats::sd(x, na.rm = TRUE), numeric(1))
  non_const_cols <- names(sds)[sds > 0]
  if (length(non_const_cols) > 1) {
    corr <- cor(num_df %>% dplyr::select(all_of(non_const_cols)), use = "pairwise.complete.obs")
    p_corr <- ggcorrplot::ggcorrplot(corr, type = "lower", lab = FALSE) +
      labs(title = "Correlação entre variáveis numéricas")
  } else {
    p_corr <- NULL
  }
} else {
  p_corr <- NULL
}


p_dist_by_role <- hr %>%
  ggplot(aes(x = JobRole, y = DistanceFromHome, fill = Attrition)) +
  geom_boxplot(outlier.alpha = 0.2) +
  coord_flip() +
  labs(title = "DistanceFromHome por JobRole e Attrition", x = "JobRole", y = "Distância")

p_income_by_edu <- hr %>%
  mutate(Education = as.factor(Education)) %>%
  ggplot(aes(x = Education, y = MonthlyIncome, fill = Attrition)) +
  geom_boxplot(outlier.alpha = 0.2) +
  labs(title = "MonthlyIncome por Education e Attrition", x = "Education (1-5)", y = "Renda Mensal")

# EDA rapido
eda_top    <- p_class + (p_age | p_dist)
eda_bottom <- (p_inc | p_twy)
if (!is.null(p_inc) & !is.null(p_twy)) {
  p_eda_panel <- eda_top / eda_bottom
  print(p_eda_panel)
  save_plot(p_eda_panel, "eda_painel_principal.jpg", w = 12, h = 8)
} else {
  print(eda_top)
  save_plot(eda_top, "eda_painel_top.jpg", w = 12, h = 6)
}

if (!is.null(p_corr)) { print(p_corr); save_plot(p_corr, "eda_correlacao_numericas.jpg", w = 9, h = 8) }

# Salvar gráficos individuais úteis
save_plot(p_class, "eda_distribuicao_classe.jpg")
if (!is.null(p_age))  save_plot(p_age,  "eda_densidade_age.jpg")
if (!is.null(p_dist)) save_plot(p_dist, "eda_densidade_distancefromhome.jpg")
if (!is.null(p_inc))  save_plot(p_inc,  "eda_densidade_monthlyincome.jpg")
if (!is.null(p_twy))  save_plot(p_twy,  "eda_densidade_totalworkingyears.jpg")

# Salvar proporções por categorias
purrr::walk2(props_cat_plots, cat_focus, ~save_plot(.x, paste0("eda_attrition_por_", .y, ".jpg")))

# Perguntas do dataset (boxplots) + salvar
print(p_dist_by_role);  save_plot(p_dist_by_role,  "eda_distance_by_jobrole.jpg", w = 11, h = 7)
print(p_income_by_edu); save_plot(p_income_by_edu, "eda_income_by_education.jpg", w = 9, h = 6)

# 3) Split treino/teste p/ estratificado

# Separar dados evita "vazar" informação do teste no treino.
set.seed(1234)
split <- initial_split(hr, prop = 0.80, strata = Attrition)
train <- training(split)
test  <- testing(split)

# Folds para CV (estratificados)
folds <- vfold_cv(train, v = 10, strata = Attrition)

# 4) Baseline: Regressão Logística + Stepwise

# Modelo basico p/ referencia + seleção de variáveis
#  Stepwise AIC 

# Remover fatores/colunas que não estão em 'train' pós limpeza
glm_null <- glm(Attrition ~ 1, data = train, family = binomial)

glm_step <- stats::step(
  glm_null,
  scope = list(lower = ~ 1, upper = Attrition ~ .),
  direction = "both",
  trace = 0
)

pred_step_prob <- predict(glm_step, newdata = test, type = "response")
pred_step_cls  <- factor(ifelse(pred_step_prob >= 0.5, "Yes", "No"),
                         levels = c("Yes","No"))

df_pred_step <- test %>%
  dplyr::select(Attrition) %>%
  mutate(.pred_Yes = pred_step_prob,
         .pred_No  = 1 - pred_step_prob,
         .pred_class = pred_step_cls,
         model = "GLM Stepwise")

# 5) Modelos
# Lasso, Ridge e Elastic Net (glmnet) e hiper via CV

# Recipe
rec_glmnet <- recipe(Attrition ~ ., data = train) %>%
  # Tirar NA
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  # Remover variáveis sem variância
  step_zv(all_predictors()) %>%
  # One-hot encoding para categorias (pq glmnet requer numérico)
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

# Métricas AUC, PR-AUC, F1, etc
class_metrics <- metric_set(roc_auc, pr_auc, brier_class)

# Especificações
lasso_spec <- logistic_reg(
  mode = "classification",
  penalty = tune(),   # lambda
  mixture = 1         # alpha = 1 (ou seja, Lasso)
) %>% set_engine("glmnet")

ridge_spec <- logistic_reg(
  mode = "classification",
  penalty = tune(),   # lambda
  mixture = 0         # alpha = 0 (ou seja, Ridge)
) %>% set_engine("glmnet")

enet_spec <- logistic_reg(
  mode = "classification",
  penalty = tune(),   
  mixture = tune()  
) %>% set_engine("glmnet")

# Workflows
wf_lasso <- workflow() %>% add_model(lasso_spec) %>% add_recipe(rec_glmnet)
wf_ridge <- workflow() %>% add_model(ridge_spec) %>% add_recipe(rec_glmnet)
wf_enet  <- workflow() %>% add_model(enet_spec)  %>% add_recipe(rec_glmnet)

#Grades de hiperparâmetros
lambda_grid <- grid_regular(penalty(range = c(-4, 1)), levels = 50)  # 1e-4 .. 10
alpha_grid  <- grid_regular(mixture(range = c(0, 1)), levels = 11)   # 0.0 .. 1.0
enet_grid   <- crossing(lambda_grid, alpha_grid)

# Tuning com CV
ctrl <- control_grid(save_pred = TRUE, save_workflow = TRUE, verbose = TRUE)

res_lasso <- tune_grid(
  wf_lasso, resamples = folds, grid = lambda_grid,
  metrics = class_metrics, control = ctrl
)
res_ridge <- tune_grid(
  wf_ridge, resamples = folds, grid = lambda_grid,
  metrics = class_metrics, control = ctrl
)
res_enet <- tune_grid(
  wf_enet, resamples = folds, grid = enet_grid,
  metrics = class_metrics, control = ctrl
)

# Visualização das curvas de tuning p/ ENet
p_tune_lasso <- res_lasso %>% collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = penalty, y = mean)) +
  geom_line(linewidth = 0.7) + geom_point(size = 0.9) +
  scale_x_log10() +
  labs(title = "Lasso: AUC vs lambda (CV)", x = "lambda (log10)", y = "AUC (média CV)")

# idem para ridge/enet (substituir λ -> lambda; α -> alpha; usar linewidth)
p_tune_ridge <- res_ridge %>% collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = penalty, y = mean)) +
  geom_line(linewidth = 0.7) + geom_point(size = 0.9) +
  scale_x_log10() +
  labs(title = "Ridge: AUC vs lambda (CV)", x = "lambda (log10)", y = "AUC (média CV)")

p_tune_enet <- res_enet %>% collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = penalty, y = mean, color = mixture, group = mixture)) +
  geom_line(linewidth = 0.6) + geom_point(size = 0.7) +
  scale_x_log10() + scale_color_viridis_c(end = 0.9) +
  labs(title = "Elastic Net: AUC vs lambda por alpha (CV)", x = "lambda (log10)", y = "AUC (média CV)", color = "alpha")

# salvar combinado
p_tune_all <- (p_tune_lasso | p_tune_ridge) / p_tune_enet
print(p_tune_all)
save_plot(p_tune_all, "tuning_auc_plots.jpg", w = 12, h = 10)

p_tune_ridge <- res_ridge %>% collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = penalty, y = mean)) +
  geom_line() + geom_point(size = 0.8) +
  scale_x_log10() +
  labs(title = "Ridge: AUC vs λ (CV)", x = "λ (log10)", y = "AUC (média CV)")

p_tune_enet <- res_enet %>% collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = penalty, y = mean, color = mixture, group = mixture)) +
  geom_line() + geom_point(size = 0.6) +
  scale_x_log10() + scale_color_viridis_c(end = 0.9) +
  labs(title = "Elastic Net: AUC vs λ por α (CV)", x = "λ (log10)", y = "AUC (média CV)", color = "α")

print((p_tune_lasso | p_tune_ridge) / p_tune_enet)

# Melhores hiperparâmetros por AUC
metric_to_select <- "roc_auc"   # altere para "brier_class" se quiser otimizar MSE (Brier)
best_lasso <- select_best(res_lasso, metric = metric_to_select)
best_ridge <- select_best(res_ridge, metric = metric_to_select)
best_enet  <- select_best(res_enet,  metric = metric_to_select)


# Selecao dos hiperparâmetros e avaliar no TESTE (hold-out)
last_lasso <- last_fit(finalize_workflow(wf_lasso, best_lasso), split)
last_ridge <- last_fit(finalize_workflow(wf_ridge, best_ridge), split)
last_enet  <- last_fit(finalize_workflow(wf_enet,  best_enet),  split)

best_threshold_from_oof <- function(res_object, grid = seq(0.05, 0.95, by = 0.01)){
  preds <- collect_predictions(res_object) %>% dplyr::select(.pred_Yes, Attrition)
  f1_tbl <- purrr::map_dfr(grid, function(t){
    est <- factor(ifelse(preds$.pred_Yes >= t, "Yes", "No"), levels = c("Yes","No"))
    f1  <- yardstick::f_meas_vec(truth = preds$Attrition, estimate = est, event_level = "first")
    tibble(threshold = t, f1 = f1)
  })
  f1_tbl %>% arrange(desc(f1)) %>% slice(1) %>% pull(threshold)
}

best_threshold_from_train <- function(model, data_train, grid = seq(0.05, 0.95, by = 0.01)){
  prob  <- predict(model, newdata = data_train, type = "response")
  truth <- data_train$Attrition
  f1_tbl <- purrr::map_dfr(grid, function(t){
    est <- factor(ifelse(prob >= t, "Yes", "No"), levels = c("Yes","No"))
    f1  <- yardstick::f_meas_vec(truth = truth, estimate = est, event_level = "first")
    tibble(threshold = t, f1 = f1)
  })
  f1_tbl %>% arrange(desc(f1)) %>% slice(1) %>% pull(threshold)
}

thr_lasso <- best_threshold_from_oof(res_lasso)
thr_ridge <- best_threshold_from_oof(res_ridge)
thr_enet  <- best_threshold_from_oof(res_enet)
thr_step  <- best_threshold_from_train(glm_step, train)

# salvar thresholds 
thr_tbl <- tibble(
  model = c("GLM Stepwise","Lasso","Ridge","Elastic Net"),
  threshold = c(thr_step, thr_lasso, thr_ridge, thr_enet)
)
readr::write_csv(thr_tbl, "outputs/thresholds_escolhidos.csv")

pred_lasso <- collect_predictions(last_lasso) %>% mutate(model = "Lasso")
pred_ridge <- collect_predictions(last_ridge) %>% mutate(model = "Ridge")
pred_enet  <- collect_predictions(last_enet)  %>% mutate(model = "Elastic Net")

# Nomes p/ comparação
df_pred_lasso <- pred_lasso %>%
  transmute(Attrition = Attrition, .pred_Yes = .pred_Yes, .pred_No = .pred_No,
            .pred_class = factor(ifelse(.pred_Yes >= thr_lasso, "Yes", "No"), levels = c("Yes","No")),
            model)

df_pred_ridge <- pred_ridge %>%
  transmute(Attrition = Attrition, .pred_Yes = .pred_Yes, .pred_No = .pred_No,
            .pred_class = factor(ifelse(.pred_Yes >= thr_ridge, "Yes", "No"), levels = c("Yes","No")),
            model)

df_pred_enet  <- pred_enet %>%
  transmute(Attrition = Attrition, .pred_Yes = .pred_Yes, .pred_No = .pred_No,
            .pred_class = factor(ifelse(.pred_Yes >= thr_enet, "Yes", "No"), levels = c("Yes","No")),
            model)

# Ajustar o baseline para usar o threshold tunado do treino:
df_pred_step <- df_pred_step %>%
  mutate(.pred_class = factor(ifelse(.pred_Yes >= thr_step, "Yes", "No"), levels = c("Yes","No")))

# 6) Avaliação dos modelos no teste


# Helper p/ métricas
compute_metrics <- function(df_pred, model_name){
  tibble(
    model    = model_name,
    roc_auc  = yardstick::roc_auc(df_pred, truth = Attrition, .pred_Yes, event_level = "first") %>% dplyr::pull(.estimate),
    pr_auc   = yardstick::pr_auc(df_pred,  truth = Attrition, .pred_Yes, event_level = "first") %>% dplyr::pull(.estimate),
    f1       = yardstick::f_meas(df_pred,  truth = Attrition, .pred_class, event_level = "first") %>% dplyr::pull(.estimate),
    accuracy = yardstick::accuracy(df_pred, truth = Attrition, .pred_class) %>% dplyr::pull(.estimate),
    kappa    = yardstick::kap(df_pred,      truth = Attrition, .pred_class) %>% dplyr::pull(.estimate),
    sens     = yardstick::sens(df_pred,     truth = Attrition, .pred_class, event_level = "first") %>% dplyr::pull(.estimate),
    spec     = yardstick::spec(df_pred,     truth = Attrition, .pred_class) %>% dplyr::pull(.estimate),
    brier    = yardstick::brier_class(df_pred, truth = Attrition, .pred_Yes) %>% dplyr::pull(.estimate)
  )
}

# Após montar metrics_tbl:
metrics_tbl <- bind_rows(
  compute_metrics(df_pred_step,  "GLM Stepwise"),
  compute_metrics(df_pred_lasso, "Lasso"),
  compute_metrics(df_pred_ridge, "Ridge"),
  compute_metrics(df_pred_enet,  "Elastic Net")
) %>%
  arrange(desc(roc_auc))

print(metrics_tbl)
readr::write_csv(metrics_tbl, "outputs/metrics_teste.csv")

# Grafico ROC
roc_df <- bind_rows(
  yardstick::roc_curve(df_pred_step,  truth = Attrition, .pred_Yes) %>% mutate(model = "GLM Stepwise"),
  yardstick::roc_curve(df_pred_lasso, truth = Attrition, .pred_Yes) %>% mutate(model = "Lasso"),
  yardstick::roc_curve(df_pred_ridge, truth = Attrition, .pred_Yes) %>% mutate(model = "Ridge"),
  yardstick::roc_curve(df_pred_enet,  truth = Attrition, .pred_Yes) %>% mutate(model = "Elastic Net")
)

p_roc <- ggplot(roc_df, aes(x = 1 - specificity, y = sensitivity, color = model)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = "dashed") + coord_equal() +
  labs(title = "Curvas ROC - Teste", x = "1 - Especificidade (Falso Positivo)", y = "Sensibilidade", color = "Modelo")
print(p_roc); save_plot(p_roc, "eval_roc_teste.jpg", w = 9, h = 7)

 

# 6.3 Curvas de Precisão-Recall
pr_df <- bind_rows(
  yardstick::pr_curve(df_pred_step,  truth = Attrition, .pred_Yes) %>% mutate(model = "GLM Stepwise"),
  yardstick::pr_curve(df_pred_lasso, truth = Attrition, .pred_Yes) %>% mutate(model = "Lasso"),
  yardstick::pr_curve(df_pred_ridge, truth = Attrition, .pred_Yes) %>% mutate(model = "Ridge"),
  yardstick::pr_curve(df_pred_enet,  truth = Attrition, .pred_Yes) %>% mutate(model = "Elastic Net")
)
p_pr <- ggplot(pr_df, aes(x = recall, y = precision, color = model)) +
  geom_path(size = 1) +
  labs(title = "Curvas Precisão-Recall - Teste",
       x = "Recall", y = "Precisão", color = "Modelo")
print(p_pr)
save_plot(p_pr, "eval_pr_teste.jpg", w = 9, h = 7)

# 6.4 Curvas de Ganho (Cumulative Gain) e Lift
gain_df <- bind_rows(
  yardstick::gain_curve(df_pred_step,  truth = Attrition, .pred_Yes) %>% mutate(model = "GLM Stepwise"),
  yardstick::gain_curve(df_pred_lasso, truth = Attrition, .pred_Yes) %>% mutate(model = "Lasso"),
  yardstick::gain_curve(df_pred_ridge, truth = Attrition, .pred_Yes) %>% mutate(model = "Ridge"),
  yardstick::gain_curve(df_pred_enet,  truth = Attrition, .pred_Yes) %>% mutate(model = "Elastic Net")
)
p_gain <- ggplot(gain_df, aes(x = .percent_tested, y = .percent_found, color = model)) +
  geom_line(size = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dotted") +
  scale_x_continuous(labels = percent_format()) +
  scale_y_continuous(labels = percent_format()) +
  labs(title = "Curvas de Ganho (Cumulative Gain) - Teste",
       x = "% de funcionários abordados",
       y = "% de desligamentos capturados (Recall)",
       color = "Modelo")
print(p_gain)
save_plot(p_gain, "eval_gain_teste.jpg", w = 9, h = 7)

lift_df <- bind_rows(
  yardstick::lift_curve(df_pred_step,  truth = Attrition, .pred_Yes) %>% mutate(model = "GLM Stepwise"),
  yardstick::lift_curve(df_pred_lasso, truth = Attrition, .pred_Yes) %>% mutate(model = "Lasso"),
  yardstick::lift_curve(df_pred_ridge, truth = Attrition, .pred_Yes) %>% mutate(model = "Ridge"),
  yardstick::lift_curve(df_pred_enet,  truth = Attrition, .pred_Yes) %>% mutate(model = "Elastic Net")
)
p_lift <- ggplot(lift_df, aes(x = .percent_tested, y = .lift, color = model)) +
  geom_line(size = 1) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  scale_x_continuous(labels = percent_format()) +
  labs(title = "Curvas de Lift - Teste",
       x = "% de funcionários abordados",
       y = "Lift (ganho relativo vs aleatório)",
       color = "Modelo")
print(p_lift)
save_plot(p_lift, "eval_lift_teste.jpg", w = 9, h = 7)

# 6.5 Comparações rápidas em barras (AUC e F1)
p_auc_bar <- metrics_tbl %>%
  ggplot(aes(x = reorder(model, roc_auc), y = roc_auc, fill = model)) +
  geom_col() +
  coord_flip() +
  labs(title = "Comparação de AUC (Teste)", x = "Modelo", y = "AUC") +
  theme(legend.position = "none")

p_f1_bar <- metrics_tbl %>%
  ggplot(aes(x = reorder(model, f1), y = f1, fill = model)) +
  geom_col() +
  coord_flip() +
  labs(title = "Comparação de F1 (Teste)", x = "Modelo", y = "F1") +
  theme(legend.position = "none")

print(p_auc_bar | p_f1_bar)
save_plot(p_auc_bar, "eval_auc_bar.jpg", w = 8, h = 6)
save_plot(p_f1_bar,  "eval_f1_bar.jpg",  w = 8, h = 6)

# 7) Coeficientes 
# Lasso zera coeficientes de variáveis pouco úteis,
# Ridge encolhe mas n zera. ENet mescla ambos.

# Extrair modelos treinados 
fit_lasso_engine <- last_lasso$.workflow[[1]] %>% extract_fit_parsnip() %>% pluck("fit")
fit_ridge_engine <- last_ridge$.workflow[[1]] %>% extract_fit_parsnip() %>% pluck("fit")
fit_enet_engine  <- last_enet$.workflow[[1]]  %>% extract_fit_parsnip() %>% pluck("fit")

coef_lasso <- broom::tidy(fit_lasso_engine, s = best_lasso$penalty, return_zeros = TRUE) %>%
  filter(term != "(Intercept)") %>%
  arrange(desc(abs(estimate))) %>%
  mutate(zeroed = estimate == 0)

coef_ridge <- broom::tidy(fit_ridge_engine, s = best_ridge$penalty, return_zeros = TRUE) %>%
  filter(term != "(Intercept)") %>%
  arrange(desc(abs(estimate)))

coef_enet <- broom::tidy(fit_enet_engine, s = best_enet$penalty, return_zeros = TRUE) %>%
  filter(term != "(Intercept)") %>%
  arrange(desc(abs(estimate)))

p_coef_lasso <- coef_lasso %>%
  slice_max(order_by = abs(estimate), n = 20) %>%
  ggplot(aes(x = reorder(term, estimate), y = estimate, fill = zeroed)) +
  geom_col() + coord_flip() +
  scale_fill_manual(values = c("TRUE"="#bbbbbb","FALSE"="#1b9e77")) +
  labs(title = "Lasso (lambda ótimo): Top 20 coeficientes",
       x = "Variáveis dummy/numéricas", y = "Coeficiente", fill = "Zerado?")
print(p_coef_lasso); save_plot(p_coef_lasso, "coef_top20_lasso.jpg", w = 10, h = 7)

p_coef_ridge <- coef_ridge %>%
  slice_max(order_by = abs(estimate), n = 20) %>%
  ggplot(aes(x = reorder(term, estimate), y = estimate)) +
  geom_col(fill = "#7570b3") + coord_flip() +
  labs(title = "Ridge (lambda ótimo): Top 20 coeficientes",
       x = "Variáveis dummy/numéricas", y = "Coeficiente")
print(p_coef_ridge); save_plot(p_coef_ridge, "coef_top20_ridge.jpg", w = 10, h = 7)

p_coef_enet <- coef_enet %>%
  slice_max(order_by = abs(estimate), n = 20) %>%
  ggplot(aes(x = reorder(term, estimate), y = estimate)) +
  geom_col(fill = "#d95f02") + coord_flip() +
  labs(title = sprintf("Elastic Net (lambda=%.4g, alpha=%.2f): Top 20 coeficientes", best_enet$penalty, best_enet$mixture),
       x = "Variáveis dummy/numéricas", y = "Coeficiente")
print(p_coef_enet); save_plot(p_coef_enet, "coef_top20_enet.jpg", w = 10, h = 7)

# exportar tabelas de coeficientes
readr::write_csv(coef_lasso, "outputs/coef_lasso_completos.csv")
readr::write_csv(coef_ridge, "outputs/coef_ridge_completos.csv")
readr::write_csv(coef_enet,  "outputs/coef_enet_completos.csv")

# Plot: top 20 absolutos do Ridge
p_coef_ridge <- coef_ridge %>%
  slice_max(order_by = abs(estimate), n = 20) %>%
  ggplot(aes(x = reorder(term, estimate), y = estimate)) +
  geom_col(fill = "#7570b3") +
  coord_flip() +
  labs(title = "Ridge (λ ótimo): Top 20 coeficientes",
       x = "Variáveis dummy/numéricas", y = "Coeficiente")
print(p_coef_ridge)

# Plot: top 20 absolutos do Elastic Net
p_coef_enet <- coef_enet %>%
  slice_max(order_by = abs(estimate), n = 20) %>%
  ggplot(aes(x = reorder(term, estimate), y = estimate)) +
  geom_col(fill = "#d95f02") +
  coord_flip() +
  labs(title = sprintf("Elastic Net (λ=%.4g, α=%.2f): Top 20 coeficientes",
                       best_enet$penalty, best_enet$mixture),
       x = "Variáveis dummy/numéricas", y = "Coeficiente")
print(p_coef_enet)

# 8) Top 10% de maior risco: qual a taxa de sucesso?

# Pergunta: Se abordarmos apenas os 10% com maior risco previsto, qual a
# taxa de acerto (precisão), o recall e o lift vs aleatório?

# Descobrir melhor modelo por AUC
best_model_name <- metrics_tbl$model[which.max(metrics_tbl$roc_auc)]
best_pred_df <- list(
  "GLM Stepwise" = df_pred_step,
  "Lasso" = df_pred_lasso,
  "Ridge" = df_pred_ridge,
  "Elastic Net" = df_pred_enet
)[[best_model_name]]

# Função: estatísticas no top k%
top_k_stats <- function(df_pred, k = 0.10){
  n <- nrow(df_pred)
  k_n <- max(1, round(k * n))
  base_rate <- mean(df_pred$Attrition == "Yes")
  df_top <- df_pred %>%
    arrange(desc(.pred_Yes)) %>%
    slice_head(n = k_n) %>%
    mutate(hit = Attrition == "Yes")
  precision_at_k <- mean(df_top$hit)              # taxa de sucesso entre os abordados
  recall_at_k    <- sum(df_top$hit) / sum(df_pred$Attrition == "Yes")
  lift_at_k      <- precision_at_k / base_rate
  tibble(
    n_total = n,
    k_prop = k,
    k_n = k_n,
    base_rate = base_rate,
    precision_at_k = precision_at_k,
    recall_at_k = recall_at_k,
    lift_at_k = lift_at_k
  )
}

top10_tbl <- top_k_stats(best_pred_df, k = 0.10)
print(top10_tbl)

# Marcação do ponto de 10% na curva de ganho do melhor modelo
best_gain <- yardstick::gain_curve(best_pred_df, truth = Attrition, .pred_Yes) %>%
  mutate(model = best_model_name)

p_gain_best <- ggplot(best_gain, aes(x = .percent_tested, y = .percent_found)) +
  geom_line(color = "#1b9e77", size = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dotted") +
  geom_vline(xintercept = 0.10, linetype = "dashed", color = "grey40") +
  annotate("text", x = 0.11, y = best_gain$.percent_found[which.min(abs(best_gain$.percent_tested - 0.10))],
           label = "Top 10% alvo", hjust = 0, vjust = -0.5, size = 3.5) +
  scale_x_continuous(labels = percent_format()) +
  scale_y_continuous(labels = percent_format()) +
  labs(title = paste0("Curva de Ganho - Melhor Modelo (", best_model_name, ")"),
       x = "% de funcionários abordados", y = "% de desligamentos capturados")

print(p_gain_best)
save_plot(p_gain_best, "eval_gain_top10_melhor_modelo.jpg", w = 9, h = 7)

# salvar tabela top-10% e predições do melhor modelo <<<
readr::write_csv(top10_tbl,     "outputs/top10_stats_best_model.csv")
readr::write_csv(best_pred_df,  "outputs/predicoes_teste_best_model.csv")

# confusion matrix do melhor modelo no TESTE (threshold tunado) <<<
best_pred_df2 <- best_pred_df %>% mutate(pred_class = .pred_class)
conf_best <- yardstick::conf_mat(best_pred_df2, truth = Attrition, estimate = pred_class)
readr::write_csv(broom::tidy(conf_best), "outputs/confusion_matrix_best_model.csv")

# 9) Responder às perguntas do enunciado (com base nos resultados)

cat("\n================== RESPOSTAS (baseadas nos resultados rodados) ==================\n")

# 1) Qual modelo confiar mais?
cat(sprintf("1) Modelo recomendado: %s (AUC teste = %.3f; PR-AUC = %.3f; F1 = %.3f; Brier = %.3f).\n",
            best_model_name,
            metrics_tbl$roc_auc[metrics_tbl$model == best_model_name],
            metrics_tbl$pr_auc[metrics_tbl$model == best_model_name],
            metrics_tbl$f1[metrics_tbl$model == best_model_name],
            metrics_tbl$brier[metrics_tbl$model == best_model_name]))
cat("Motivo: melhor desempenho global nas métricas próprias de classificação, com bom equilíbrio entre\n",
    "discriminação (AUC/PR-AUC), decisão (F1) e calibração (Brier). Em termos de custo-benefício para RH,\n",
    "o modelo mostra maior ganho/lift nos percentis-alvo, maximizando impactos sob orçamento limitado.\n\n", sep="")

# 2) O que o Lasso fez com variáveis menos importantes?
n_zero <- sum(coef_lasso$zeroed, na.rm = TRUE)
cat(sprintf("2) Lasso zerou %d coeficientes (excluindo intercepto), reduzindo o modelo a um subconjunto parcimonioso.\n",
            n_zero))
cat("Significado: variáveis pouco informativas são 'desligadas' (coeficiente = 0), tornando o modelo mais interpretável\n",
    "e robusto (menor risco de overfitting) sem sacrificar desempenho. Já o Ridge tende a encolher, mas não zera.\n\n", sep="")

# 3) Com orçamento para apenas 10% de maior risco
cat(sprintf("3) Top 10%% de risco (melhor modelo = %s):\n", best_model_name))
cat(sprintf("   - Taxa de sucesso (Precisão@10%%): %.1f%%\n", 100 * top10_tbl$precision_at_k))
cat(sprintf("   - Cobertura (Recall@10%%): %.1f%% dos desligamentos totais\n", 100 * top10_tbl$recall_at_k))
cat(sprintf("   - Lift@10%%: %.2fx a taxa base (%.1f%%)\n",
            top10_tbl$lift_at_k, 100 * top10_tbl$base_rate))
cat("Interpretação: Se abordarmos somente os 10% com maior risco previsto, esperamos acertar essa fração de desligamentos\n",
    "com ganho relativo significativo frente a uma seleção aleatória.\n", sep="")

cat("=================================================================================\n")

# 10) Observações finais
