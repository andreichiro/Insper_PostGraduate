
# Pacotes
suppressPackageStartupMessages({
  library(dados)      
  library(dplyr)        
  library(tidyr)     
  library(ggplot2)    
  library(skimr)        
  library(GGally)       # p/ scatterplot matrix.
  library(tidymodels)   # recipes, parsnip, workflows, tune, rsample, yardstick
  library(glmnet)       # engine dos modelos penalizados
  library(patchwork)    # combina gráficos
  library(broom)        # arrumar saídas (tidy)
  library(scales)       # escalas log10 etc.
})

set.seed(123)
theme_set(theme_minimal(base_size = 12))

# 1) Carregar e limpar dados
dados_raw <- dados::pinguins

penguins_clean <- dados_raw %>%
  #Remove NAs 
  drop_na(massa_corporal, comprimento_nadadeira, especie,
          comprimento_bico, profundidade_bico)

# Skim
cat("\n===== SKIM =====\n")
print(skimr::skim(penguins_clean))

# ---------------------------------------
# 2) EDA (Exploratory Data Analysis) objetiva
# ---------------------------------------
# Foco: entender distribuição da resposta, relação com preditores e correlações.

# 2.1 Distribuição da variável resposta
g_dist_target <- ggplot(penguins_clean, aes(x = massa_corporal)) +
  geom_histogram(bins = 30, alpha = 0.9) +
  geom_density(aes(y = after_stat(density))) +
  labs(title = "Distribuição da Massa Corporal (g)",
       x = "Massa corporal (g)", y = "Frequência / Densidade") +
  theme_minimal()

# 2.2 Boxplot da resposta por espécie (mostra diferenças de nível entre grupos)
g_box_species <- ggplot(penguins_clean, aes(x = especie, y = massa_corporal, fill = especie)) +
  geom_boxplot(alpha = 0.8, outlier.alpha = 0.5) +
  guides(fill = "none") +
  labs(title = "Massa corporal por espécie",
       x = "Espécie", y = "Massa corporal (g)") +
  theme_minimal()

# 2.3 Relação massa vs comprimento da nadadeira, por espécie (relação linear clara)
g_scatter_flipper <- ggplot(penguins_clean,
                            aes(x = comprimento_nadadeira, y = massa_corporal, color = especie)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE) +
  labs(title = "Massa vs Comprimento da Nadadeira (por espécie)",
       x = "Comprimento da nadadeira (mm)",
       y = "Massa corporal (g)",
       color = "Espécie") +
  theme_minimal()

# 2.4 Correlação entre variáveis numéricas (para ver colinearidades e sinal esperado)
num_vars <- penguins_clean %>%
  select(where(is.numeric))

cor_mat <- cor(num_vars, use = "pairwise.complete.obs")

# Converter a matriz de correlação para formato longo e fazer um heatmap
cor_df <- as.data.frame(as.table(cor_mat))
names(cor_df) <- c("var1", "var2", "correlacao")

g_heat_corr <- ggplot(cor_df, aes(x = var1, y = var2, fill = correlacao)) +
  geom_tile() +
  scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B",
                       midpoint = 0, limits = c(-1, 1)) +
  coord_equal() +
  labs(title = "Correlação entre variáveis numéricas", x = NULL, y = NULL, fill = "ρ") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Mostrar EDA principal (opcionalmente, lado a lado)
# print(g_dist_target); print(g_box_species); print(g_scatter_flipper); print(g_heat_corr)
# Para apresentar de forma compacta:
(g_dist_target | g_box_species) / (g_scatter_flipper | g_heat_corr)

# ---------------------------------------
# 3) Partição treino/teste
# ---------------------------------------
# Estratégia: 80% treino e 20% teste. Estratificar por espécie para manter proporções.
set.seed(123)
data_split <- initial_split(penguins_clean, prop = 0.8, strata = especie)
train_data  <- training(data_split)
test_data   <- testing(data_split)

# ---------------------------------------
# 4) Receitas (pré-processamento)
# ---------------------------------------
# - SEM interação: massa ~ comprimento_nadadeira + especie
# - COM interação: massa ~ comprimento_nadadeira + especie + (comprimento_nadadeira:especie)
#   Em recipes: criamos dummies e depois incluímos termos de interação entre a variável contínua
#   e cada dummy de espécie.

# Receita SEM interação
rec_base <- recipe(massa_corporal ~ comprimento_nadadeira + especie, data = train_data) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>% # one-hot para cada espécie
  step_zv(all_predictors()) %>%                            # remove colunas com variância zero
  step_normalize(all_numeric_predictors())                 # padroniza numéricos (bom p/ glmnet)

# Receita COM interação
rec_inter <- recipe(massa_corporal ~ comprimento_nadadeira + especie, data = train_data) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  # Interações entre a variável contínua e cada dummy de espécie.
  step_interact(terms = ~ comprimento_nadadeira:starts_with("especie_")) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

# ---------------------------------------
# 5) Definição dos modelos (parsnip + glmnet)
# ---------------------------------------
# Notas:
# - penalty = lambda (força de regularização). Quanto maior, mais “encolhe” os coeficientes.
# - mixture = alpha. 0 = Ridge, 1 = Lasso, (0,1) = Elastic Net.
# - Vamos “tunar” (otimizar) lambda em todos os casos e alpha no Elastic Net.

spec_ridge <- linear_reg(penalty = tune(), mixture = 0) %>%
  set_engine("glmnet")

spec_lasso <- linear_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet")

spec_enet  <- linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")

# ---------------------------------------
# 6) Workflow(s)
# ---------------------------------------
wf_ridge_base <- workflow() %>% add_model(spec_ridge) %>% add_recipe(rec_base)
wf_lasso_base <- workflow() %>% add_model(spec_lasso) %>% add_recipe(rec_base)
wf_enet_base  <- workflow() %>% add_model(spec_enet)  %>% add_recipe(rec_base)

wf_ridge_inter <- workflow() %>% add_model(spec_ridge) %>% add_recipe(rec_inter)
wf_lasso_inter <- workflow() %>% add_model(spec_lasso) %>% add_recipe(rec_inter)
wf_enet_inter  <- workflow() %>% add_model(spec_enet)  %>% add_recipe(rec_inter)

# ---------------------------------------
# 7) Validação cruzada (k-fold)
# ---------------------------------------
# 10-fold CV com repetição (robustez). Estratificamos por espécie para manter representatividade.
set.seed(123)
folds <- vfold_cv(train_data, v = 10, repeats = 5, strata = especie)

# ---------------------------------------
# 8) Grids de hiperparâmetros
# ---------------------------------------
# - Para Ridge/Lasso: apenas lambda (penalty). Usamos grid na escala log (1e-6 a 1e+1).
# - Para EN: lambda + alpha. Usamos amostragem Max Entropy para cobrir bem o espaço.
lambda_grid <- grid_regular(penalty(range = c(-6, 1)), levels = 50) # 1e-6 ... 10

# Para Elastic Net: variar mixture (alpha) entre 0.05 e 0.95 para evitar extremos exatos
param_enet <- parameters(
  penalty(range = c(-6, 1)),
  mixture(range = c(0.05, 0.95))
)
set.seed(123)
grid_enet <- grid_max_entropy(param_enet, size = 60)  # bom compromisso entre cobertura e custo

# Métricas de interesse p/ tuning (RMSE, MAE, R²).
# Observação: RMSE é monotônica com MSE (RMSE^2 = MSE). Usaremos RMSE no tuning
# e reportaremos MSE na comparação final.
tune_metrics <- metric_set(rmse, mae, rsq)

ctrl_grid <- control_grid(save_pred = TRUE, save_workflow = TRUE, verbose = TRUE)

# ---------------------------------------
# 9) Tuning por grid + CV
# ---------------------------------------
# SEM interação
set.seed(123)
res_ridge_base <- tune_grid(wf_ridge_base, resamples = folds, grid = lambda_grid,
                            metrics = tune_metrics, control = ctrl_grid)

set.seed(123)
res_lasso_base <- tune_grid(wf_lasso_base, resamples = folds, grid = lambda_grid,
                            metrics = tune_metrics, control = ctrl_grid)

set.seed(123)
res_enet_base  <- tune_grid(wf_enet_base,  resamples = folds, grid = grid_enet,
                            metrics = tune_metrics, control = ctrl_grid)

# COM interação
set.seed(123)
res_ridge_inter <- tune_grid(wf_ridge_inter, resamples = folds, grid = lambda_grid,
                             metrics = tune_metrics, control = ctrl_grid)

set.seed(123)
res_lasso_inter <- tune_grid(wf_lasso_inter, resamples = folds, grid = lambda_grid,
                             metrics = tune_metrics, control = ctrl_grid)

set.seed(123)
res_enet_inter  <- tune_grid(wf_enet_inter,  resamples = folds, grid = grid_enet,
                             metrics = tune_metrics, control = ctrl_grid)

# ---------------------------------------
# 10) Visualizações das curvas de tuning (úteis para entender sensibilidade a λ e α)
# ---------------------------------------
plot_tuning_curve <- function(res, title_suffix) {
  # Função para plotar RMSE médio x lambda (escala log) para grids 1D (Ridge/Lasso)
  metrics <- collect_metrics(res) %>% filter(.metric == "rmse")
  ggplot(metrics, aes(x = penalty, y = mean)) +
    geom_line() +
    geom_point(alpha = 0.7) +
    scale_x_log10(labels = label_number()) +
    labs(title = paste0("Curva de tuning (RMSE) — ", title_suffix),
         x = "Lambda (penalidade, escala log10)",
         y = "RMSE médio (validação cruzada)") +
    theme_minimal()
}

plot_tuning_enet <- function(res, title_suffix) {
  # Função para plotar desempenho (RMSE) em função de alpha e lambda (scatter com escala log)
  metrics <- collect_metrics(res) %>% filter(.metric == "rmse")
  ggplot(metrics, aes(x = mixture, y = penalty, size = 1/mean, color = mean)) +
    geom_point(alpha = 0.8) +
    scale_y_log10(labels = label_number()) +
    scale_color_gradient(low = "#2166AC", high = "#B2182B", name = "RMSE") +
    labs(title = paste0("Elastic Net — RMSE vs α (mixture) e λ (penalty) — ", title_suffix),
         x = expression(alpha~"(mistura: 0=Ridge, 1=Lasso)"),
         y = "Lambda (penalidade, escala log10)",
         size = "1 / RMSE") +
    theme_minimal()
}

# Gráficos de tuning (descomente se quiser ver individualmente)
g_ridge_base  <- plot_tuning_curve(res_ridge_base,  "Ridge (sem interação)")
g_lasso_base  <- plot_tuning_curve(res_lasso_base,  "Lasso (sem interação)")
g_enet_base   <- plot_tuning_enet (res_enet_base,   "Elastic Net (sem interação)")

g_ridge_inter <- plot_tuning_curve(res_ridge_inter, "Ridge (com interação)")
g_lasso_inter <- plot_tuning_curve(res_lasso_inter, "Lasso (com interação)")
g_enet_inter  <- plot_tuning_enet (res_enet_inter,  "Elastic Net (com interação)")

# Visualização compacta:
(g_ridge_base | g_lasso_base) / g_enet_base
(g_ridge_inter | g_lasso_inter) / g_enet_inter

# ---------------------------------------
# 11) Seleção dos melhores hiperparâmetros (menor RMSE na CV)
# ---------------------------------------
best_ridge_base  <- select_best(res_ridge_base,  metric = "rmse")
best_lasso_base  <- select_best(res_lasso_base,  metric = "rmse")
best_enet_base   <- select_best(res_enet_base,   metric = "rmse")

best_ridge_inter <- select_best(res_ridge_inter, metric = "rmse")
best_lasso_inter <- select_best(res_lasso_inter, metric = "rmse")
best_enet_inter  <- select_best(res_enet_inter,  metric = "rmse")

# ---------------------------------------
# 12) Ajuste final no treino e avaliação no teste (last_fit)
# ---------------------------------------
# Função auxiliar para finalizar workflow, ajustar e avaliar
fit_and_eval <- function(wf, best_params, split, id_modelo) {
  final_wf <- finalize_workflow(wf, best_params)
  lf <- last_fit(final_wf, split = split, metrics = metric_set(rmse, mae, rsq))
  # Coletar métricas e previsões do teste
  metrics_test <- collect_metrics(lf) %>%
    mutate(modelo = id_modelo) %>%
    # Acrescentar MSE (RMSE^2) para facilitar comparação
    mutate(.metric = as.character(.metric)) %>%
    tidyr::pivot_wider(names_from = .metric, values_from = .estimate) %>%
    mutate(mse = rmse^2) %>%
    select(modelo, rmse, mae, rsq, mse)
  preds_test <- collect_predictions(lf) %>% mutate(modelo = id_modelo)
  list(last_fit = lf, metrics_test = metrics_test, preds_test = preds_test, final_wf = final_wf)
}

# Rodar para os 6 cenários
res_final <- list(
  ridge_base  = fit_and_eval(wf_ridge_base,  best_ridge_base,  data_split, "Ridge | sem interação"),
  lasso_base  = fit_and_eval(wf_lasso_base,  best_lasso_base,  data_split, "Lasso | sem interação"),
  enet_base   = fit_and_eval(wf_enet_base,   best_enet_base,   data_split, "Elastic Net | sem interação"),
  ridge_inter = fit_and_eval(wf_ridge_inter, best_ridge_inter, data_split, "Ridge | com interação"),
  lasso_inter = fit_and_eval(wf_lasso_inter, best_lasso_inter, data_split, "Lasso | com interação"),
  enet_inter  = fit_and_eval(wf_enet_inter,  best_enet_inter,  data_split, "Elastic Net | com interação")
)

# Tabela de métricas no conjunto de teste
metrics_test_tbl <- bind_rows(
  res_final$ridge_base$metrics_test,
  res_final$lasso_base$metrics_test,
  res_final$enet_base$metrics_test,
  res_final$ridge_inter$metrics_test,
  res_final$lasso_inter$metrics_test,
  res_final$enet_inter$metrics_test
) %>%
  arrange(mse)

cat("\n===== MÉTRICAS NO TESTE (ordenado por MSE) =====\n")
print(metrics_test_tbl)

# ---------------------------------------
# 13) Gráfico de comparação por MSE (teste)
# ---------------------------------------
g_cmp_mse <- metrics_test_tbl %>%
  mutate(modelo = factor(modelo, levels = metrics_test_tbl$modelo)) %>%
  ggplot(aes(x = modelo, y = mse)) +
  geom_col(alpha = 0.9) +
  geom_text(aes(label = round(mse, 1)), vjust = -0.4, size = 3.2) +
  labs(title = "Comparação dos modelos no conjunto de teste",
       subtitle = "Métrica principal: MSE (quanto menor, melhor)",
       x = NULL, y = "MSE (g²)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))
print(g_cmp_mse)

# ---------------------------------------
# 14) Diagnóstico do modelo vencedor (menor MSE no teste)
# ---------------------------------------
modelo_vencedor <- metrics_test_tbl$modelo[1]
cat("\nModelo vencedor (menor MSE no teste): ", modelo_vencedor, "\n")

# Obter objetos do vencedor
obj_vencedor <- switch(as.character(modelo_vencedor),
  "Ridge | sem interação"           = res_final$ridge_base,
  "Lasso | sem interação"           = res_final$lasso_base,
  "Elastic Net | sem interação"     = res_final$enet_base,
  "Ridge | com interação"           = res_final$ridge_inter,
  "Lasso | com interação"           = res_final$lasso_inter,
  "Elastic Net | com interação"     = res_final$enet_inter
)

preds_best <- obj_vencedor$preds_test
# Observado vs Predito
g_obs_pred <- ggplot(preds_best, aes(x = .pred, y = massa_corporal)) +
  geom_point(alpha = 0.75) +
  geom_abline(slope = 1, intercept = 0, linetype = 2) +
  labs(title = paste0("Observado vs Predito — ", modelo_vencedor),
       x = "Predito (g)", y = "Observado (g)") +
  theme_minimal()

# Resíduos vs Ajustados
preds_best <- preds_best %>%
  mutate(residuo = massa_corporal - .pred)

g_res_fitted <- ggplot(preds_best, aes(x = .pred, y = residuo)) +
  geom_point(alpha = 0.75) +
  geom_hline(yintercept = 0, linetype = 2) +
  labs(title = paste0("Resíduos vs Ajustados — ", modelo_vencedor),
       x = "Ajustado (predito, g)", y = "Resíduo (g)") +
  theme_minimal()

# QQ-plot dos resíduos (normalidade aproximada ajuda em inferência; aqui é diagnóstico visual)
g_qq <- ggplot(preds_best, aes(sample = residuo)) +
  stat_qq(alpha = 0.8) +
  stat_qq_line(linetype = 2) +
  labs(title = paste0("QQ-plot dos resíduos — ", modelo_vencedor),
       x = "Quantis teóricos", y = "Quantis amostrais") +
  theme_minimal()

(g_obs_pred | g_res_fitted) / g_qq

# ---------------------------------------
# 15) Importância / Coeficientes do modelo vencedor
# ---------------------------------------
# Explicação: em modelos penalizados, coeficientes podem ser “encolhidos” até zero (Lasso),
# ou apenas reduzidos (Ridge). Vamos extrair os coeficientes na lambda/alpha otimizados.

# Recuperar workflow final e parâmetros escolhidos
final_wf_vencedor <- obj_vencedor$final_wf

# Precisamos dos melhores hiperparâmetros (lambda/alpha) do vencedor:
best_params_vencedor <- switch(as.character(modelo_vencedor),
  "Ridge | sem interação"           = best_ridge_base,
  "Lasso | sem interação"           = best_lasso_base,
  "Elastic Net | sem interação"     = best_enet_base,
  "Ridge | com interação"           = best_ridge_inter,
  "Lasso | com interação"           = best_lasso_inter,
  "Elastic Net | com interação"     = best_enet_inter
)

# Extrair o objeto glmnet ajustado no last_fit (engine) e fazer o tidy no lambda escolhido
fit_engine <- obj_vencedor$last_fit$.workflow[[1]] %>% extract_fit_engine()

# Valor de lambda escolhido
best_lambda <- best_params_vencedor$penalty
# Valor de alpha (quando aplicável)
best_alpha  <- if ("mixture" %in% names(best_params_vencedor)) best_params_vencedor$mixture else NA_real_

# Tabela de coeficientes (inclui zeros se houver, útil para ver “seleção” do Lasso)
coefs_tbl <- broom::tidy(fit_engine, s = best_lambda, return_zeros = TRUE) %>%
  arrange(desc(abs(estimate))) %>%
  mutate(term = ifelse(term == "(Intercept)", "Intercepto", term))

cat("\n===== COEFICIENTES (ordenados por |coef|) — ", modelo_vencedor, " =====\n", sep = "")
print(coefs_tbl)

# Gráfico dos coeficientes não-nulos (excluindo o intercepto para foco)
coefs_plot_tbl <- coefs_tbl %>%
  filter(term != "Intercepto", estimate != 0)

g_coefs <- ggplot(coefs_plot_tbl, aes(x = reorder(term, abs(estimate)), y = estimate)) +
  geom_col(alpha = 0.9) +
  coord_flip() +
  labs(title = paste0("Coeficientes (não nulos) — ", modelo_vencedor),
       x = "Variável", y = "Coeficiente (escala normalizada)") +
  theme_minimal()

print(g_coefs)

# ---------------------------------------
# 16) Resumo executivo (mensagens amigáveis) — opcional
# ---------------------------------------
cat("\n===== RESUMO EXECUTIVO =====\n")
cat("- Problema de REGRESSÃO; portanto, AUC/ROC/F1 NÃO se aplicam aqui.\n")
cat("- Com base em k-fold CV e avaliação no conjunto de teste, comparamos modelos por MSE.\n")
cat("- Ranking por MSE (menor é melhor):\n")
print(metrics_test_tbl %>% select(modelo, mse, rmse, mae, rsq))
cat("- Modelo vencedor:", modelo_vencedor, "\n")
if (!is.na(best_alpha)) {
  cat(sprintf("- Hiperparâmetros vencedores: lambda = %.5f, alpha = %.3f\n",
              best_lambda, best_alpha))
} else {
  cat(sprintf("- Hiperparâmetro vencedor: lambda = %.5f\n", best_lambda))
}
cat("- Use os gráficos de tuning para entender a sensibilidade a λ/α.\n")
cat("- Use os diagnósticos (observado vs predito, resíduos, QQ-plot) para checar qualidade do ajuste.\n")
cat("- Coeficientes ajudam a interpretar a direção/intensidade dos efeitos.\n")
cat("Dica: se a interação trouxe ganho, é sinal de inclinações por espécie serem diferentes (linhas não paralelas).\n")

# Fim. Se precisar, podemos estender para incluir mais variáveis (ex.: bico) e comparar novamente.
