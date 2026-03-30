# ================================================================
# Fase 1 — Entradas e premissas (na moral, direto ao ponto)
# - Lê o CSV único deixado pelo pipeline Python
# - Valida T/D e imprime estatísticas essenciais
# - Gera um snapshot de schema para auditoria rápida
# ================================================================
# Alunos: André Ichiro Katsurada e Danilo Guimarães 

# ------------------ Setup básico (robusto a getwd()) -------------------
options(stringsAsFactors = FALSE)
set.seed(42)

# Resolve o caminho do próprio script (funciona com Rscript e source())
.script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  i <- grep("^--file=", args)
  if (length(i) > 0) return(normalizePath(sub("^--file=", "", args[i][1])))
  if (!is.null(sys.frames()[[1]]$ofile)) return(normalizePath(sys.frames()[[1]]$ofile))
  normalizePath(getwd())  # último recurso (console)
}

PROJECT_DIR <- dirname(.script_path())

# Primeiro osoutputs/processed no mesmo diretório do script
OUT_DIR <- file.path(PROJECT_DIR, "outputs", "processed")

# Fallback: se não existir, tentar na pasta (caso o script esteja em /scripts)
if (!dir.exists(OUT_DIR)) {
  OUT_DIR_UP <- file.path(dirname(PROJECT_DIR), "outputs", "processed")
  if (dir.exists(OUT_DIR_UP)) {
    OUT_DIR <- OUT_DIR_UP
  }
}

DATA_PATH  <- file.path(OUT_DIR, "bisnode_survival_firmlevel.csv")
SCHEMA_OUT <- file.path(OUT_DIR, "phase1_schema.csv")

# ------------------ Checagens de existência de arquivo -----------------------
if (!dir.exists(OUT_DIR)) {
  stop("[phase1] Pasta 'outputs/processed' não encontrada.\n",
       "Procurado em:\n - ", file.path(PROJECT_DIR, "outputs", "processed"), "\n - ",
       file.path(dirname(PROJECT_DIR), "outputs", "processed"))
}

if (!file.exists(DATA_PATH)) {
  stop("[phase1] Arquivo não encontrado em nenhum dos locais tentados:\n",
       " - ", file.path(PROJECT_DIR, "outputs", "processed", "bisnode_survival_firmlevel.csv"), "\n",
       " - ", file.path(dirname(PROJECT_DIR), "outputs", "processed", "bisnode_survival_firmlevel.csv"), "\n",
       "Caminho real informado por você:\n",
       " - /Users/akatsurada/Documents/INSPER/ProjetoIntegrador/Projeto1/outputs/processed/bisnode_survival_firmlevel.csv\n",
       "Solução: execute o script a partir de qualquer diretório — este ajuste já resolve — ",
       "ou mova o script para a mesma raiz do arquivo.")
}

message("[phase1] Lendo dados: ", DATA_PATH)
df <- tryCatch({ utils::read.csv(DATA_PATH, check.names = FALSE) },
               error = function(e) stop("[phase1] Falha ao ler CSV: ", conditionMessage(e)))

# ------------------ Checagens de existência de arquivo -----------------------
if (!dir.exists(OUT_DIR)) {
  stop("[phase1] Pasta 'outputs/processed' não encontrada. Gere os artefatos no Python primeiro.")
}
if (!file.exists(DATA_PATH)) {
  stop("[phase1] Arquivo não encontrado: ", DATA_PATH,
       "\nCertifique-se de que o pipeline Python gravou 'bisnode_survival_firmlevel.csv' em outputs/processed/.")
}

# ------------------ Leitura do CSV (preservando nomes) -----------------------
message("[phase1] Lendo dados: ", DATA_PATH)
df <- tryCatch({
  utils::read.csv(DATA_PATH, check.names = FALSE)
}, error = function(e) {
  stop("[phase1] Falha ao ler CSV: ", conditionMessage(e))
})

# ------------------ Validações essenciais de schema --------------------------
# Precisa das colunas T (tempo) e D (evento)
must_have <- c("T", "D")
missing_cols <- setdiff(must_have, names(df))
if (length(missing_cols) > 0) {
  stop("[phase1] Faltam colunas obrigatórias: ", paste(missing_cols, collapse = ", "),
       "\nColunas disponíveis: ", paste(names(df), collapse = ", "))
}

# Typinge validações de T/D
# - T: numérico, finito, > 0
# - D: binário {0,1}; aceita lógico TRUE/FALSE e converte para 1/0
# -----------------------------------------------------------------------------
# T
df$T <- suppressWarnings(as.numeric(df$T))
if (any(!is.finite(df$T))) {
  stop("[phase1] 'T' contém valores não finitos (NA/Inf/NaN).")
}
if (any(df$T <= 0, na.rm = TRUE)) {
  stop("[phase1] 'T' deve ser estritamente positivo (> 0). Há linhas com T <= 0.")
}

# D
if (is.logical(df$D)) df$D <- as.integer(df$D)                    # TRUE/FALSE -> 1/0
if (is.character(df$D)) {
  # Se veio como string, coerção direta p/ numérico;
  # se falhar, interrompemos (não tentamos mapear rótulos arbitrários aqui).
  df$D <- suppressWarnings(as.numeric(df$D))
}
if (any(is.na(df$D))) stop("[phase1] 'D' contém NA após coerção. Deve ser 0/1.")
uD <- sort(unique(df$D))
if (!all(uD %in% c(0, 1))) {
  stop("[phase1] 'D' deve ser binário em {0,1}. Valores distintos encontrados: ",
       paste(uD, collapse = ", "))
}

# ------------------ Estatísticas rápidas (sanity check) ----------------------
n  <- nrow(df)
p  <- length(setdiff(names(df), c("T","D")))
ev <- mean(df$D == 1)
message(sprintf("[phase1] n=%d | p=%d (excl. T/D) | eventos=%d (%.1f%%) | censura=%.1f%%",
                n, p, sum(df$D==1), 100*ev, 100*(1-ev)))

# Se houver algum indicativo de base problemática, avisar (sem travar):
if (n < 100)          message("[phase1][warn] n<100 — amostra pequena; métricas podem ficar instáveis.")
if (ev < 0.01 || ev > 0.99) message("[phase1][warn] taxa de evento muito extrema; cuidado com discriminação/calibração.")

# ------------------ Snapshot de schema (para auditoria) ----------------------
schema_df <- data.frame(
  column     = names(df),
  type       = vapply(df, function(x) paste(class(x), collapse = "+"), character(1)),
  n_missing  = vapply(df, function(x) sum(is.na(x)), integer(1)),
  unique_n   = vapply(df, function(x) length(unique(x)), integer(1)),
  example    = vapply(df, function(x) {
    v <- x[!is.na(x)]
    if (length(v) == 0) return(NA_character_)
    as.character(v[1])
  }, character(1)),
  stringsAsFactors = FALSE
)

# Grava schema completo e mostra o resumo no console
utils::write.csv(schema_df, SCHEMA_OUT, row.names = FALSE)
message("[phase1] Schema salvo em: ", SCHEMA_OUT)

# Preview curto no console (primeiras 10 linhas)
message("[phase1] Prévia do schema (10 primeiras colunas):")
print(utils::head(schema_df, 10), row.names = FALSE)

# ------------------ Mensagem final -------------------------------------------
message("[phase1] OK — entradas validadas. Próximos passos: Phase 2 (reprodutibilidade) e Phase 3 (política de features).")

# ================================================================
# Fase 2 — Reprodutibilidade e performance 
# - Seed e threads estáveis
# - Pacotes compactos (carrega/instala quando possível)
# - Snapshot de sessão 
# - CV 5-fold estratificada (por D; adiciona setor se existir)
# - Logger simples de runtime (per-step/per-fold)
# ================================================================

# ------------------ Seeds e threads ---------------------------
SEED <- 42L
set.seed(SEED)
# RNG estável e compatível com paralelismo
try(RNGkind(kind = "L'Ecuyer-CMRG"), silent = TRUE)

# Número de threads disponível (usado depois por XGBoost/CatBoost/BLAS)
N_THREADS <- tryCatch({
  max(1L, as.integer(Sys.getenv("OMP_NUM_THREADS",
                                unset = parallel::detectCores(logical = TRUE))))
}, error = function(e) 1L)
options(mc.cores = N_THREADS)
# Muitos libs respeitam essas variáveis:
Sys.setenv(OMP_NUM_THREADS = N_THREADS,
           MKL_NUM_THREADS = N_THREADS,
           OPENBLAS_NUM_THREADS = N_THREADS,
           NUMEXPR_NUM_THREADS = N_THREADS)

# Se data.table existir, alinhar threads (não obrigatório nesta fase)
if (requireNamespace("data.table", quietly = TRUE)) {
  try(data.table::setDTthreads(threads = N_THREADS), silent = TRUE)
}

# ------------------ Pacotes compactos -------------------------
pkgs_phase2 <- c(
  "survival","glmnet","flexsurv","xgboost","catboost",
  "riskRegression","pec","survAUC","timeROC","isotone",
  "iml","shapviz",         # usamos iml p/ ALE; dispensa ALEPlot
  "ggplot2","dplyr","tidyr","tibble","stringr","scales","viridisLite"
)

install.packages("remotes")
remotes::install_url(
    "https://github.com/catboost/catboost/releases/download/v1.2.8/catboost-R-darwin-universal2-1.2.8.tgz",
    INSTALL_opts = c("--no-multiarch", "--no-test-load", "--no-staged-install")
)

#Se nao funcionar, tente:
# conda install r-catboost

# Ou:
# Instalção do CatBoost no R
# > install.packages("remotes")
# > install.packages("https://github.com/catboost/catboost/releases/download/v1.2.8/catboost-R-darwin-universal2-1.2.8.tgz", 
#                   repos = NULL, type = "source")

# Pro MAC sillicon, isso aqui funcionou:
# brew install miniforge
# conda config --add channels conda-forge
# conda config --set channel_priority strict

#conda create -n catboost_env r-base r-essentials
#conda activate catboost_env
#conda install -c conda-forge r-catboost


# Não depender de ALEPlot
pkgs_phase2 <- setdiff(pkgs_phase2, "ALEPlot")

ensure_packages <- function(pkgs) {
  loaded <- character(0)
  missing <- character(0)
  for (p in pkgs) {
    if (!requireNamespace(p, quietly = TRUE)) {
      # Tenta instalar, mas NUNCA interrompe a execução se não conseguir
      try(utils::install.packages(p, quiet = TRUE), silent = TRUE)
    }
    if (requireNamespace(p, quietly = TRUE)) {
      suppressPackageStartupMessages(library(p, character.only = TRUE))
      loaded <- c(loaded, p)
    } else {
      missing <- c(missing, p)
    }
  }
  list(loaded = loaded, missing = missing)
}

pkg_status <- ensure_packages(pkgs_phase2)

# -------- CatBoost é obrigatório --------
REQUIRE_CATBOOST <- TRUE

if (!requireNamespace("catboost", quietly = TRUE)) {
  # Tenta a instalação robusta (CRAN -> GitHub, com toolchain/macOS Homebrew libomp)
  try(ensure_catboost_or_fail(), silent = FALSE)
}

if (!requireNamespace("catboost", quietly = TRUE)) {
  msg <- paste0(
    "[phase2][fatal] 'catboost' segue indisponível após tentativa automática.\n",
    "Siga os passos abaixo e execute novamente:\n",
    if (Sys.info()[['sysname']] == "Darwin")
      "  macOS:\n    1) xcode-select --install\n    2) brew install libomp\n    3) R: remotes::install_github('catboost/catboost', subdir='catboost/R-package')\n"
    else if (Sys.info()[['sysname']] == "Linux")
      "  Linux (Debian/Ubuntu):\n    sudo apt-get update && sudo apt-get install -y libomp-dev\n    R: remotes::install_github('catboost/catboost', subdir='catboost/R-package')\n"
    else
      "  Windows:\n    R: install.packages('catboost')  # ou remotes::install_github('catboost/catboost', subdir='catboost/R-package')\n"
  )
  if (REQUIRE_CATBOOST) stop(msg) else message("[phase2][warn] ", msg)
} else {
  suppressPackageStartupMessages(library(catboost))
  # Atualiza o snapshot p/ refletir o carregamento do catboost
  if (!("catboost" %in% pkg_status$loaded)) {
    pkg_status$loaded  <- c(pkg_status$loaded,  "catboost")
    pkg_status$missing <- setdiff(pkg_status$missing, "catboost")
  }
}

# ------------------ Logger de runtime -------------------------
RUNTIME_LOG <- file.path(OUT_DIR, "runtime_log.csv")
if (!file.exists(RUNTIME_LOG)) {
  utils::write.table(
    data.frame(phase=character(), fold=integer(), step=character(),
               seconds=numeric(), timestamp=character()),
    RUNTIME_LOG, sep=",", row.names=FALSE, col.names=TRUE, quote=TRUE
  )
}
log_runtime <- function(phase, fold, step, seconds) {
  line <- data.frame(
    phase = as.character(phase),
    fold  = as.integer(ifelse(is.null(fold), NA_integer_, fold)),
    step  = as.character(step),
    seconds = as.numeric(seconds),
    timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")
  )
  utils::write.table(line, RUNTIME_LOG, sep=",", row.names=FALSE,
                     col.names=FALSE, append=TRUE, quote=TRUE)
}

# ------------------ Snapshot de sessão ------------------------
t0 <- Sys.time()
session_txt <- capture.output({
  cat("===== Reproducibility snapshot =====\n")
  cat(sprintf("Timestamp: %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")))
  cat(sprintf("Seed: %s\n", SEED))
  cat(sprintf("Threads (OMP): %s\n", N_THREADS))
  cat(sprintf("Script dir: %s\n", PROJECT_DIR))
  cat(sprintf("Data path : %s\n", DATA_PATH))
  cat(sprintf("n=%d | p=%d (excl. T/D) | eventos=%d (%.2f%%) | censura=%.2f%%\n",
              n, p, sum(df$D==1), 100*mean(df$D==1), 100*(1-mean(df$D==1))))
  cat("\n-- Packages loaded --\n")
  cat(paste0(sort(pkg_status$loaded), collapse=", "), "\n")
  if (length(pkg_status$missing) > 0) {
    cat("\n-- Packages missing (not loaded) --\n")
    cat(paste0(sort(pkg_status$missing), collapse=", "), "\n")
  }
  cat("\n-- sessionInfo() --\n")
  print(utils::sessionInfo())
})
writeLines(session_txt, file.path(OUT_DIR, "session_info.txt"))
log_runtime("phase2", NA, "session_snapshot", as.numeric(difftime(Sys.time(), t0, units = "secs")))
message("[phase2] Snapshot salvo em: ", file.path(OUT_DIR, "session_info.txt"))

# ------------------ CV 5-fold estratificada -------------------
# Estratos: sempre por D; se existir 'ind2_2d' ou 'nace_main_2d' ou 'sector' ou 'region_m',
# adiciona esse campo para manter balanceamento por setor/região.
t1 <- Sys.time()
strat_cols <- "D"
cand_sector <- intersect(names(df), c("ind2_2d","nace_main_2d","sector","region_m"))
if (length(cand_sector) >= 1L) strat_cols <- c(strat_cols, cand_sector[1L])

# Converte estratos para fator de forma segura
strata_df <- df[, strat_cols, drop = FALSE]
for (nm in names(strata_df)) strata_df[[nm]] <- as.factor(strata_df[[nm]])
key <- interaction(strata_df, drop = TRUE, lex.order = TRUE)

K <- 5L
fold_id <- integer(nrow(df))
set.seed(SEED)
for (lvl in levels(key)) {
  idx <- which(key == lvl)
  if (length(idx) == 0) next
  idx <- sample(idx, length(idx))                  # embaralha dentro do estrato
  splits <- cut(seq_along(idx), breaks = K, labels = FALSE)
  fold_id[idx] <- splits
}
# Caso algum estrato muito pequeno tenha ficado com 0 (raríssimo), faz round-robin
if (any(fold_id == 0)) {
  rest <- which(fold_id == 0)
  fold_id[rest] <- ((seq_along(rest) - 1L) %% K) + 1L
}

# Persistência: lista de índices por dobra + mapa linha->dobra
row_id <- seq_len(nrow(df))
folds_list <- split(row_id, fold_id)
saveRDS(folds_list, file.path(OUT_DIR, "cv5_folds.rds"))
fold_map <- data.frame(
  row_id = row_id, fold = fold_id, T = df$T, D = df$D,
  strat  = as.character(key),
  stringsAsFactors = FALSE
)
utils::write.csv(fold_map, file.path(OUT_DIR, "cv5_fold_map.csv"), row.names = FALSE)

# Pequeno relatório no console
tab <- as.data.frame(table(fold_map$fold))
names(tab) <- c("fold","n")
balance_msg <- sprintf("[phase2] CV 5-fold pronta | balanceamento por dobra: %s",
                       paste(sprintf("%s:%d", tab$fold, tab$n), collapse="  "))
message(balance_msg)
log_runtime("phase2", NA, "build_cv5", as.numeric(difftime(Sys.time(), t1, units = "secs")))
message("[phase2] Artefatos salvos: cv5_folds.rds, cv5_fold_map.csv, runtime_log.csv")

# ================================================================
# Fase 3 — Política de features & pré-processamento (sem leakage)
# - Usa todos os numéricos + pequeno conjunto de categóricas de negócio
# - Imputa (train-only): numérico=mediana | fator=moda
# - Remove degenerescências (train): variância zero / fator 1 nível
# - Alinha test às decisões de train (níveis de fator, imputa, dummies)
# - Salva artefatos por dobra pras próximas fases
# ================================================================

t_phase3 <- Sys.time()

PHASE3_DIR <- file.path(OUT_DIR, "prep_phase3")
if (!dir.exists(PHASE3_DIR)) dir.create(PHASE3_DIR, recursive = TRUE, showWarnings = FALSE)

FOLDS_RDS <- file.path(OUT_DIR, "cv5_folds.rds")
if (!file.exists(FOLDS_RDS)) {
  stop("[phase3] Arquivo de dobras não encontrado: ", FOLDS_RDS,
       "\nExecute a Phase 2 antes (gera cv5_folds.rds).")
}
folds_list <- readRDS(FOLDS_RDS)

# Conjunto de categóricas de negócio
BIZ_CANDIDATES <- c("ind2_2d","region_m","urban_m_std","nace_main_2d")
biz_cats_present <- intersect(BIZ_CANDIDATES, names(df))
if (length(biz_cats_present) == 0) {
  message("[phase3] Nenhuma categórica de negócio encontrada nas colunas: ",
          paste(BIZ_CANDIDATES, collapse = ", "))
} else {
  message("[phase3] Categóricas de negócio habilitadas: ",
          paste(biz_cats_present, collapse = ", "))
}

# -------- utilitários específicos da fase --------
is_num <- function(x) is.numeric(x) || is.integer(x)

# ---- Utilitário: moda (nível/valor não‑NA mais frequente) ----
mode_level <- function(x) {
  # Para fatores, pega o nível mais frequente; se tudo for NA, cai pro primeiro nível
  if (is.factor(x)) {
    tab <- table(x, useNA = "no")
    if (length(tab) == 0) {
      levs <- levels(x)
      return(if (length(levs) > 0) levs[1] else NA_character_)
    }
    return(names(which.max(tab)))
  }
  # Para character/numeric/logical, pega o não‑NA mais frequente; se não tiver, retorna NA
  tab <- table(x, useNA = "no")
  if (length(tab) == 0) return(NA)
  names(which.max(tab))
}

# global utils
align_mm <- function(mm, mm_cols) {
  # n rows even when mm has 0 columns
  n <- if (!is.null(dim(mm))) nrow(mm) else length(mm)
  if (is.null(mm_cols) || length(mm_cols) == 0) {
    # nothing to align; return mm unchanged (or empty matrix with right n)
    if (!is.null(dim(mm))) return(mm)
    return(matrix(numeric(0), nrow = n, ncol = 0))
  }
  M <- matrix(0, nrow = n, ncol = length(mm_cols))
  colnames(M) <- mm_cols
  if (!is.null(dim(mm)) && ncol(mm) > 0) {
    common <- intersect(mm_cols, colnames(mm))
    if (length(common)) M[, common] <- mm[, common, drop = FALSE]
  }
  M
}

# Ensure time grid is strictly inside support; prefer max EVENT time on the test fold.
sanitize_times <- function(times, T_vec, D_vec = NULL) {
  T_vec <- as.numeric(T_vec)
  ok <- is.finite(T_vec)
  if (!any(ok)) return(numeric(0))
  if (!is.null(D_vec) && any(D_vec == 1 & ok)) {
    tmax <- max(T_vec[D_vec == 1 & ok])
  } else {
    tmax <- max(T_vec[ok])
  }
  times <- sort(unique(as.numeric(times)))
  times <- times[is.finite(times) & times > 0 & times < tmax]
  if (length(times) >= 2) return(times)

  base <- if (!is.null(D_vec) && any(D_vec == 1 & ok)) T_vec[D_vec == 1 & ok] else T_vec[ok]
  q <- as.numeric(stats::quantile(base, probs = c(0.25, 0.50, 0.75), na.rm = TRUE))
  q <- sort(unique(q[q > 0 & q < tmax]))
  if (length(q) >= 2) return(q)

  # Last resort: at least 2 interior points
  seq(from = tmax/4, to = 3*tmax/4, length.out = 2)
}

# Prepara (train,test) da dobra seguindo a política e sem vazamento
make_fold_preproc <- function(df_train, df_test, time_col = "T", status_col = "D", biz_cats = biz_cats_present) {
  # 1) Seleção de colunas pela política
  num_cols_all <- setdiff(names(df_train)[vapply(df_train, is_num, logical(1))], c(time_col, status_col))
  cat_cols_all <- intersect(biz_cats, names(df_train))

  # 2) Tipos: garante fatores nas categóricas de negócio
  for (nm in cat_cols_all) df_train[[nm]] <- as.factor(df_train[[nm]])
  for (nm in cat_cols_all) df_test [[nm]] <- factor(df_test[[nm]], levels = levels(df_train[[nm]]))

RARE_MIN_COUNT <- 50L  # ajuste seu limiar se quiser

collapse_levels_by_event <- function(df_train, df_test, nm, min_count = RARE_MIN_COUNT) {
  ftr <- factor(df_train[[nm]])
  # 1) raro por contagem
  tab <- table(ftr, useNA = "no")
  rare_levels <- names(tab)[tab < min_count]

  # 2) sem evento pelo desfecho (zero eventos OU zero não‑eventos)
  ev_tab <- tapply(df_train$D, ftr, function(v) sum(v == 1, na.rm = TRUE))
  ne_tab <- tapply(df_train$D, ftr, function(v) sum(v == 0, na.rm = TRUE))
  bad_levels <- union(names(ev_tab)[is.finite(ev_tab) & ev_tab == 0],
                      names(ne_tab)[is.finite(ne_tab) & ne_tab == 0])

  to_other <- union(rare_levels, bad_levels)

  # Mapeia train/test -> "Other" pra esses níveis (e qualquer nível inédito no test)
  tr_new <- ifelse(is.na(ftr), NA_character_,
                   ifelse(as.character(ftr) %in% to_other, "Other", as.character(ftr)))

  fte <- df_test[[nm]]
  unseen_in_train <- !is.na(fte) & !(as.character(fte) %in% names(tab))
  te_new <- ifelse(is.na(fte), NA_character_,
                   ifelse(unseen_in_train | (as.character(fte) %in% to_other),
                          "Other", as.character(fte)))

  lev_final <- unique(c(setdiff(sort(unique(tr_new)), "Other"), "Other"))
  df_train[[nm]] <- factor(tr_new, levels = lev_final)
  df_test [[nm]] <- factor(te_new, levels = lev_final)

  list(train = df_train, test = df_test, collapsed = to_other)
}

if (length(cat_cols_all)) {
  for (nm in cat_cols_all) {
    res <- collapse_levels_by_event(df_train, df_test, nm, min_count = RARE_MIN_COUNT)
    df_train <- res$train
    df_test  <- res$test
    # (opcional) exibe mensagem se algo foi colapsado:
    if (length(res$collapsed)) {
      message(sprintf("[phase3] %s: colapsadas %d categorias em 'Other' (raras/sem evento).",
                      nm, length(res$collapsed)))
    }
  }
}


  # 3) Imputação (train-only -> aplica em ambos)
  #    Numérico: mediana; Fator: moda (nível mais frequente)
  num_medians <- if (length(num_cols_all)) {
    vapply(num_cols_all, function(nm) {
      m <- stats::median(df_train[[nm]], na.rm = TRUE)
      if (!is.finite(m)) 0 else m
    }, numeric(1))
  } else numeric(0)

  if (length(num_cols_all)) {
    for (nm in num_cols_all) {
      df_train[[nm]][is.na(df_train[[nm]])] <- num_medians[[nm]]
      df_test [[nm]][is.na(df_test [[nm]])] <- num_medians[[nm]]
    }
  }

  fac_modes <- if (length(cat_cols_all)) {
    vapply(cat_cols_all, function(nm) mode_level(df_train[[nm]]), character(1))
  } else character(0)

  if (length(cat_cols_all)) {
    for (nm in cat_cols_all) {
      df_train[[nm]][is.na(df_train[[nm]])] <- fac_modes[[nm]]
      df_test [[nm]][is.na(df_test [[nm]])] <- fac_modes[[nm]]
    }
  }

  # 4) Remoção de degenerescências na dobra (train)
  drop_zero_var <- if (length(num_cols_all)) {
    num_cols_all[vapply(num_cols_all, function(nm) {
      v <- suppressWarnings(stats::var(df_train[[nm]], na.rm = TRUE))
      !is.finite(v) || v == 0
    }, logical(1))]
  } else character(0)

  keep_num <- setdiff(num_cols_all, drop_zero_var)

  drop_single_lev <- if (length(cat_cols_all)) {
    cat_cols_all[vapply(cat_cols_all, function(nm) {
      nlevels(droplevels(df_train[[nm]])) < 2
    }, logical(1))]
  } else character(0)

  keep_cat <- setdiff(cat_cols_all, drop_single_lev)

  # 5) Subconjunto final + realinhamento de níveis no test
  keep_cols <- c(time_col, status_col, keep_num, keep_cat)
  df_train <- df_train[, keep_cols, drop = FALSE]
  df_test  <- df_test [, keep_cols, drop = FALSE]

  if (length(keep_cat)) {
    for (nm in keep_cat) {
      df_train[[nm]] <- droplevels(df_train[[nm]])
      df_test [[nm]] <- factor(df_test[[nm]], levels = levels(df_train[[nm]]))
      # Se test recebeu NA por níveis inéditos, troca por moda do train
      if (anyNA(df_test[[nm]])) {
        repl <- mode_level(df_train[[nm]])
        df_test[[nm]][is.na(df_test[[nm]])] <- repl
      }
    }
  }

  # 6) Mapa de dummies para boosters (model.matrix, sem intercepto)
  #    Guardamos APENAS os nomes das colunas resultantes em train
  xcols_for_mm <- c(keep_num, keep_cat)
  mm_cols <- if (length(xcols_for_mm)) {
    mm <- stats::model.matrix(stats::reformulate(xcols_for_mm), data = df_train)
    colnames(mm)[colnames(mm) != "(Intercept)"]
  } else character(0)

  artifacts <- list(
    num_cols      = keep_num,
    cat_cols      = keep_cat,
    num_medians   = num_medians[names(num_medians) %in% keep_num],
    fac_modes     = fac_modes[names(fac_modes) %in% keep_cat],
    factor_levels = lapply(stats::setNames(keep_cat, keep_cat), function(nm) levels(df_train[[nm]])),
    dropped       = list(zero_var = drop_zero_var, single_level = drop_single_lev),
    mm_cols       = mm_cols
  )
  list(train = df_train, test = df_test, artifacts = artifacts)
}

# -------- executa por dobra e persiste --------
fold_summary <- list()

for (k in seq_along(folds_list)) {
  t_fold <- Sys.time()
  te_idx <- folds_list[[k]]
  tr_idx <- setdiff(seq_len(nrow(df)), te_idx)

  dtr <- df[tr_idx, , drop = FALSE]
  dte <- df[te_idx, , drop = FALSE]

  prep <- make_fold_preproc(dtr, dte, time_col = "T", status_col = "D", biz_cats = biz_cats_present)

  saveRDS(prep$train,     file.path(PHASE3_DIR, sprintf("fold_%d_train.rds",     k)))
  saveRDS(prep$test,      file.path(PHASE3_DIR, sprintf("fold_%d_test.rds",      k)))
  saveRDS(prep$artifacts, file.path(PHASE3_DIR, sprintf("fold_%d_artifacts.rds", k)))

  # Log e mensagens
  elapsed <- as.numeric(difftime(Sys.time(), t_fold, units = "secs"))
  log_runtime("phase3", k, "preprocess_fold", elapsed)

  kept_num <- length(prep$artifacts$num_cols)
  kept_cat <- length(prep$artifacts$cat_cols)
  drop_zv  <- length(prep$artifacts$dropped$zero_var)
  drop_1l  <- length(prep$artifacts$dropped$single_level)
  message(sprintf("[phase3] Dobra %d: train=%d, test=%d | kept num=%d, cat=%d | dropped zv=%d, 1lvl=%d",
                  k, nrow(prep$train), nrow(prep$test), kept_num, kept_cat, drop_zv, drop_1l))

  fold_summary[[k]] <- data.frame(
    fold = k,
    n_train = nrow(prep$train),
    n_test  = nrow(prep$test),
    kept_num = kept_num,
    kept_cat = kept_cat,
    dropped_zero_var = drop_zv,
    dropped_single_level = drop_1l,
    mm_cols = length(prep$artifacts$mm_cols)
  )
}

# Resumo da fase
fold_summary_df <- dplyr::bind_rows(fold_summary)
policy_txt <- capture.output({
  cat("===== Phase 3 — Feature policy summary =====\n")
  cat("Requested business categoricals: ", paste(BIZ_CANDIDATES, collapse = ", "), "\n")
  cat("Present in data:                 ", ifelse(length(biz_cats_present)==0, "(none)", paste(biz_cats_present, collapse = ", ")), "\n\n")
  print(fold_summary_df)
})
writeLines(policy_txt, file.path(OUT_DIR, "phase3_policy_summary.txt"))
message("[phase3] Artefatos salvos em: ", PHASE3_DIR)
message("[phase3] Resumo salvo em: ", file.path(OUT_DIR, "phase3_policy_summary.txt"))

log_runtime("phase3", NA, "complete", as.numeric(difftime(Sys.time(), t_phase3, units = "secs")))


# ================================================================
# Phase 4 — Resampling & evaluation times (grid + tau* por dobra)
# - Curvas: tempo global (Q25/Q50/Q75) dos TEMPOS DE EVENTO no conjunto completo
# - Decisão por dobra (tau*): mediana dos tempos de EVENTO no TREINO da dobra
# - Sem leak: tau* vem só do treino; grid global serve apenas para curvas comparáveis
# - Persistência: phase4_times.rds + CSVs auxiliares
# ================================================================

t_phase4 <- Sys.time()

PHASE4_RDS <- file.path(OUT_DIR, "phase4_times.rds")
PHASE4_CSV <- file.path(OUT_DIR, "phase4_times.csv")
PHASE4_GRID_CSV <- file.path(OUT_DIR, "phase4_times_global.csv")

# --- sanity: dobras ---
if (!exists("folds_list")) {
  FOLDS_RDS <- file.path(OUT_DIR, "cv5_folds.rds")
  if (!file.exists(FOLDS_RDS)) {
    stop("[phase4] cv5_folds.rds não encontrado. Rode Phase 2 antes.")
  }
  folds_list <- readRDS(FOLDS_RDS)
}

# --- 4.1 - Grid global (Q25/Q50/Q75) baseado nos TEMPOS DE EVENTO do corte ---
evt_all <- df$T[is.finite(df$T) & df$D == 1]
if (length(evt_all) >= 3) {
  base_q <- as.numeric(stats::quantile(evt_all, probs = c(0.25, 0.50, 0.75), na.rm = TRUE))
} else {
  # fallback: usa T global se #eventos for insuficiente (ainda mantém comparabilidade)
  base_q <- as.numeric(stats::quantile(df$T[is.finite(df$T)], probs = c(0.25, 0.50, 0.75), na.rm = TRUE))
}
times_global <- sort(unique(base_q[is.finite(base_q) & base_q > 0]))
if (length(times_global) == 0) {
  # último recurso: mediana positiva de T global
  tg <- stats::median(df$T[is.finite(df$T)], na.rm = TRUE)
  times_global <- tg[tg > 0]
}
# garante formato útil para curvas; dedup pode reduzir a <3; isso é OK (Score exige tempos estritamente crescentes)
times_global <- as.numeric(times_global)

# --- 4.2 - tau* por dobra (mediana do TEMPO DE EVENTO no TREINO) ---
PHASE3_DIR <- file.path(OUT_DIR, "prep_phase3")  # preferimos usar os splits já pré-processados
tau_star_by_fold <- numeric(length(folds_list))
fold_details <- vector("list", length(folds_list))

for (k in seq_along(folds_list)) {
  te_idx <- folds_list[[k]]
  tr_idx <- setdiff(seq_len(nrow(df)), te_idx)

  # Preferir o train da Phase 3 (garante alinhamento com futuros passos)
  tr_rds <- file.path(PHASE3_DIR, sprintf("fold_%d_train.rds", k))
  if (file.exists(tr_rds)) {
    dtr <- readRDS(tr_rds)
    # Segurança: se por alguma razão T/D foram alterados, força nomes
    if (!("T" %in% names(dtr)) || !("D" %in% names(dtr))) {
      dtr$T <- df$T[tr_idx]; dtr$D <- df$D[tr_idx]
    }
  } else {
    # robust fallback qdo Phase 3 ainda não rodou por completo
    dtr <- df[tr_idx, c("T","D"), drop = FALSE]
  }

  ev_tr <- dtr$T[is.finite(dtr$T) & dtr$D == 1]
  tau_k <- if (length(ev_tr) >= 1) {
    as.numeric(stats::median(ev_tr, na.rm = TRUE))
  } else {
    # se treino sem eventos (caso extremo), usa mediana de T do treino
    as.numeric(stats::median(dtr$T[is.finite(dtr$T)], na.rm = TRUE))
  }
  if (!is.finite(tau_k) || tau_k <= 0) {
    # fallback final: mediana global positiva
    tau_k <- max(1e-6, as.numeric(stats::median(df$T[is.finite(df$T)], na.rm = TRUE)))
  }

  tau_star_by_fold[k] <- tau_k

  fold_details[[k]] <- data.frame(
    fold = k,
    n_train = length(tr_idx),
    events_train = sum(dtr$D == 1, na.rm = TRUE),
    tau_star = tau_k
  )
}

fold_details_df <- dplyr::bind_rows(fold_details)

# --- 4.3 - Persistência e logs ---
saveRDS(
  list(
    times_global = times_global,
    tau_star_by_fold = tau_star_by_fold,
    fold_details = fold_details_df
  ),
  PHASE4_RDS
)

utils::write.csv(
  data.frame(name = paste0("Q", c(25,50,75)), value = times_global),
  PHASE4_GRID_CSV, row.names = FALSE
)
utils::write.csv(fold_details_df, PHASE4_CSV, row.names = FALSE)

message(sprintf("[phase4] Grid global (Q25/Q50/Q75 de eventos): %s",
                paste(signif(times_global, 5), collapse = ", ")))
message(sprintf("[phase4] tau* por dobra: %s",
                paste(sprintf("%d=%.5f", seq_along(tau_star_by_fold), tau_star_by_fold), collapse = "  ")))
message("[phase4] Artefatos salvos: phase4_times.rds, phase4_times_global.csv, phase4_times.csv")

log_runtime("phase4", NA, "compute_times",
            as.numeric(difftime(Sys.time(), t_phase4, units = "secs")))

# ================================================================
# Phase 5 — Cinco modelos (CoxPH, CoxNet, XGB(Cox), CatBoost(Cox), AFT)
#            c/ calibração unificada e métricas por dobra
# - Consome: prep_phase3/fold_* .rds  e phase4_times.rds
# - Gera:    per_fold_metrics.csv, metrics_summary.csv,
#            phase5_models/*.csv (curvas), predictions/fold_*.csv
# ================================================================

t_phase5 <- Sys.time()

PHASE3_DIR <- file.path(OUT_DIR, "prep_phase3")
PHASE4_RDS <- file.path(OUT_DIR, "phase4_times.rds")
PHASE5_DIR <- file.path(OUT_DIR, "phase5_models")
PRED_DIR   <- file.path(OUT_DIR, "predictions")
if (!dir.exists(PHASE5_DIR)) dir.create(PHASE5_DIR, recursive = TRUE, showWarnings = FALSE)
if (!dir.exists(PRED_DIR))   dir.create(PRED_DIR,   recursive = TRUE, showWarnings = FALSE)

if (!file.exists(PHASE4_RDS)) stop("[phase5] 'phase4_times.rds' não encontrado. Execute a Phase 4.")
if (!dir.exists(PHASE3_DIR)) stop("[phase5] Pasta da Phase 3 não encontrada: ", PHASE3_DIR)

# --- Carrega tempos globais e tau* por dobra (Phase 4) ---
times_obj     <- readRDS(PHASE4_RDS)
times_global  <- as.numeric(times_obj$times_global)
tau_by_fold   <- as.numeric(times_obj$tau_star_by_fold)

# --- Disponibilidade de pacotes de boosters ---
HAS_XGB <- requireNamespace("xgboost", quietly = TRUE)
HAS_CAT <- requireNamespace("catboost", quietly = TRUE)
if (!HAS_XGB) message("[phase5][note] Pacote 'xgboost' não carregado — pulando XGB(Cox).")
if (!HAS_CAT) message("[phase5][note] Pacote 'catboost' não carregado — pulando CatBoost(Cox).")

# --- Helpers locais ---
thresholds_ipcw_fallback <- function(T, D, r, tau, n_grid = 101) {
  C <- 1L - as.integer(D)
  fitG <- try(prodlim::prodlim(survival::Surv(T, C) ~ 1), silent = TRUE)
  if (inherits(fitG, "try-error")) return(NULL)

  tt   <- pmin(T, tau)
  Ghat <- try(predict(fitG, times = tt, type = "surv"), silent = TRUE)
  if (inherits(Ghat, "try-error")) return(NULL)
  Ghat <- as.numeric(Ghat)
  w    <- 1 / pmax(Ghat, 1e-6)
  N    <- length(T)
  event_tau <- as.integer(T <= tau & D == 1)
  prevalence <- sum(w * event_tau) / N

  thr <- unique(as.numeric(stats::quantile(r, probs = seq(0, 1, length.out = n_grid),
                                           na.rm = TRUE, type = 8)))
  thr <- thr[is.finite(thr)]
  if (!length(thr)) return(NULL)

  comp <- function(t) {
    sel <- r >= t
    WP  <- sum(w[sel]);   WN  <- sum(w[!sel])
    TPh <- sum(w[sel]  * event_tau[sel])
    TNh <- sum(w[!sel] * (1 - event_tau[!sel]))
    Se  <- if (prevalence > 0) TPh / (prevalence * N) else NA_real_
    Sp  <- if ((1 - prevalence) > 0) TNh / ((1 - prevalence) * N) else NA_real_
    PPV <- if (WP > 0) TPh / WP else NA_real_
    NPV <- if (WN > 0) TNh / WN else NA_real_
    NB  <- prevalence * Se - (1 - prevalence) * (1 - Sp) * t/(1 - t)
    c(Se = Se, Sp = Sp, PPV = PPV, NPV = NPV, NB = NB)
  }
  M <- t(sapply(thr, comp))
  data.frame(
    threshold = thr,
    Se = M[, "Se"], Sp = M[, "Sp"], PPV = M[, "PPV"], NPV = M[, "NPV"], NB = M[, "NB"],
    PredPosShare = vapply(thr, function(t) mean(r >= t, na.rm = TRUE), numeric(1)),
    stringsAsFactors = FALSE
  ) -> df

  list(df = df, prevalence = prevalence)
}

trapz <- function(x, y, denom=NULL) {
  o <- order(x); x <- x[o]; y <- y[o]
  if (length(x) < 2) return(0)
  A <- sum(diff(x) * ((y[-1] + y[-length(y)]) / 2))
  if (is.null(denom)) A else if (!is.finite(denom) || denom <= 0) A else A / denom
}

build_mm <- function(df, x_cols) {
  if (length(x_cols) == 0) return(matrix(numeric(0), nrow = nrow(df), ncol = 0))
  mm <- stats::model.matrix(stats::reformulate(x_cols), data = df)
  mm[, colnames(mm) != "(Intercept)", drop = FALSE]
}

sanitize_times <- function(times, T_vec, D_vec = NULL) {
  # times must be strictly inside support; prefer max EVENT time when available
  tmax <- max(T_vec[is.finite(T_vec)], na.rm = TRUE)
  if (!is.null(D_vec) && any(D_vec == 1 & is.finite(T_vec))) {
    tmax_evt <- max(T_vec[D_vec == 1 & is.finite(T_vec)], na.rm = TRUE)
    if (is.finite(tmax_evt) && tmax_evt > 0) tmax <- min(tmax, tmax_evt)
  }
  times <- sort(unique(as.numeric(times)))
  times <- times[is.finite(times) & times > 0 & times < tmax]
  if (length(times) >= 2) return(times)

  base <- if (!is.null(D_vec) && any(D_vec == 1 & is.finite(T_vec))) {
    T_vec[D_vec == 1 & is.finite(T_vec)]
  } else {
    T_vec[is.finite(T_vec)]
  }
  q <- as.numeric(stats::quantile(base, probs = c(0.25, 0.50, 0.75), na.rm = TRUE))
  q <- sort(unique(q[q > 0 & q < tmax]))
  if (length(q) >= 2) return(q)

  # último recurso: ao menos 2 pontos internos
  seq(from = tmax/4, to = 3*tmax/4, length.out = 2)
}

safe_fit_coxph <- function(df, time_col = "T", status_col = "D", max_iter = 3L) {
  form <- stats::as.formula(paste0("Surv(", time_col, ",", status_col, ") ~ ."))
  dropped <- character(0)
  d <- df

  for (it in 0:max_iter) {
    fit <- try(survival::coxph(form, data = d,
                               ties = "efron", x = TRUE, y = TRUE,
                               model = TRUE, na.action = na.omit),
               silent = TRUE)
    if (inherits(fit, "try-error")) {
      # última tentativa: remove o fator com mais níveis e tenta de novo
      facs <- names(d)[vapply(d, is.factor, logical(1))]
      if (!length(facs)) break
      card <- sapply(facs, function(v) nlevels(d[[v]]))
      victim <- names(sort(card, decreasing = TRUE))[1]
      d <- d[, setdiff(names(d), victim), drop = FALSE]
      dropped <- c(dropped, victim)
      next
    }

    cf <- try(stats::coef(fit), silent = TRUE)
    if (!inherits(cf, "try-error") && length(cf) && all(is.finite(cf))) {
      attr(fit, "dropped_terms") <- dropped
      return(list(fit = fit, data = d, dropped = dropped))
    }

    # Mapeia NA/Inf, estava gerando um erro
    mm <- try(stats::model.matrix(fit), silent = TRUE)
    if (inherits(mm, "try-error")) break
    assign <- attr(mm, "assign")
    term_labels <- attr(fit$terms, "term.labels")
    bad_idx <- which(!is.finite(as.numeric(cf)))
    bad_terms <- unique(assign[bad_idx])
    to_drop <- unique(term_labels[bad_terms])

    if (!length(to_drop)) break
    d <- d[, setdiff(names(d), to_drop), drop = FALSE]
    dropped <- c(dropped, to_drop)
  }

  # Retorno 
  list(fit = fit, data = d, dropped = dropped)
}

safe_unoC <- function(time_tr, status_tr, time_te, status_te, score_te, tau) {
  out <- try(
    survAUC::UnoC(
      survival::Surv(time_tr, status_tr),
      survival::Surv(time_te, status_te),
      score_te, tau = tau
    )$C, silent = TRUE
  )
  if (inherits(out, "try-error") || is.na(out) || !is.finite(out)) return(NA_real_)
  as.numeric(out)
}

risk_at_tau <- function(model, newdata, tau) {
  # flexsurv
  if (inherits(model, "flexsurvreg")) {
    res <- try(flexsurv::summary(model, newdata = newdata, t = tau, type = "survival"), silent = TRUE)
    if (!inherits(res, "try-error")) {
      s <- vapply(res, function(xx) if (is.data.frame(xx) && "est" %in% names(xx)) as.numeric(xx$est[1]) else NA_real_, numeric(1))
      return(as.numeric(1 - s))
    }
  }
  # riskRegression primeiro
  pr <- try(riskRegression::predictRisk(model, newdata = newdata, times = tau), silent = TRUE)
  if (!inherits(pr, "try-error")) return(as.numeric(pr))
  # pec como 2a opcao
  sp <- try(pec::predictSurvProb(model, newdata = newdata, times = tau), silent = TRUE)
  if (!inherits(sp, "try-error")) return(as.numeric(1 - sp))

  # --- CoxPH usando basehaz + LP ---
  if (inherits(model, "coxph")) {
    lp <- try(as.numeric(predict(model, newdata = newdata, type = "lp")), silent = TRUE)
    bh <- try(survival::basehaz(model, centered = FALSE), silent = TRUE)
    if (!inherits(lp, "try-error") && !inherits(bh, "try-error") && nrow(bh) > 0) {
      H0_tau <- stats::approx(x = bh$time, y = bh$hazard, xout = tau, method = "linear", rule = 2)$y
      risk <- 1 - exp(-H0_tau * exp(lp))
      # clamp p/ (0,1) para evitar NA/Inf 
      return(pmin(pmax(as.numeric(risk), 1e-12), 1 - 1e-12))
    }
  }
  # Se falhar, fallback:
  rep(NA_real_, nrow(newdata))
}

# --- Verifica dobras geradas na Phase 2 ---
FOLDS_RDS <- file.path(OUT_DIR, "cv5_folds.rds")
if (!file.exists(FOLDS_RDS)) stop("[phase5] cv5_folds.rds não encontrado (rode Phase 2).")
folds_list <- readRDS(FOLDS_RDS)

# --- Loop por dobra: treina, calibra, pontua e mede ---
res_rows <- list()

for (k in seq_along(folds_list)) {
  t_fold <- Sys.time()

  train_path <- file.path(PHASE3_DIR, sprintf("fold_%d_train.rds", k))
  test_path  <- file.path(PHASE3_DIR, sprintf("fold_%d_test.rds",  k))
  art_path   <- file.path(PHASE3_DIR, sprintf("fold_%d_artifacts.rds", k))
  if (!file.exists(train_path) || !file.exists(test_path) || !file.exists(art_path)) {
    stop("[phase5] Artefatos da Phase 3 ausentes para a dobra ", k, " em ", PHASE3_DIR)
  }

  dtr <- readRDS(train_path)
  dte <- readRDS(test_path)
  art <- readRDS(art_path)

  x_cols     <- c(art$num_cols, art$cat_cols)
  tau_star   <- as.numeric(tau_by_fold[k])

  times_eval <- sort(unique(c(times_global, tau_star)))
  times_eval <- sanitize_times(times_eval, dte$T, dte$D)

  # ------------------ Modelos ------------------
  form_surv <- stats::as.formula("Surv(T, D) ~ .")

  # 1) CoxPH (baseline)
  sf <- safe_fit_coxph(dtr, time_col = "T", status_col = "D")
  fit_cox <- sf$fit
  if (length(sf$dropped)) {
    message(sprintf("[phase5][note] CoxPH: removidos termos problemáticos: %s",
                    paste(sf$dropped, collapse = ", ")))
    }


  # 2) CoxNet (glmnet) + calibração por offset
  mm_tr <- build_mm(dtr, x_cols)
  mm_te <- build_mm(dte, x_cols)
  mm_tr <- align_mm(mm_tr, art$mm_cols)
  mm_te <- align_mm(mm_te, art$mm_cols)


  fit_glmnet <- NULL; cox_off_glm <- NULL; lam_use <- NA_real_
  if (ncol(mm_tr) > 0) {
    a_use <- 0.5
    lam_use <- tryCatch({
      cv <- glmnet::cv.glmnet(x = mm_tr, y = survival::Surv(dtr$T, dtr$D), family = "cox",
                               alpha = a_use, nfolds = 5, type.measure = "deviance")
      if (is.finite(cv$lambda.min)) cv$lambda.min else cv$lambda.1se
    }, error = function(e) 0.01)
    fit_glmnet <- try(glmnet::glmnet(mm_tr, y = survival::Surv(dtr$T, dtr$D),
                                     family = "cox", alpha = a_use, lambda = lam_use), silent = TRUE)
    if (!inherits(fit_glmnet, "try-error")) {
      dtr$lp_glm <- as.numeric(predict(fit_glmnet, newx = mm_tr, type = "link", s = lam_use))
      cox_off_glm <- survival::coxph(survival::Surv(T, D) ~ offset(lp_glm),
                                     data = dtr, ties = "efron", x = TRUE, y = TRUE, model = TRUE)
    }
  }

  # 3) XGBoost (Cox) + calibração por offset
  xgb_model <- NULL; cox_off_xgb <- NULL
  if (HAS_XGB && ncol(mm_tr) > 0) {
    set.seed(SEED + 100 * k)
    # split simples para early stopping
    val_size <- max(1L, min(1000L, floor(0.2 * nrow(dtr))))
    idx_val  <- sample(seq_len(nrow(dtr)), size = val_size)

    dtrain2 <- xgboost::xgb.DMatrix(data = mm_tr[-idx_val, , drop = FALSE],
                                    label = ifelse(dtr$D[-idx_val] == 1, dtr$T[-idx_val], -dtr$T[-idx_val]))
    dval    <- xgboost::xgb.DMatrix(data = mm_tr[idx_val, , drop = FALSE],
                                    label = ifelse(dtr$D[idx_val] == 1, dtr$T[idx_val], -dtr$T[idx_val]))
    watch <- list(train = dtrain2, val = dval)
    params <- list(
      objective = "survival:cox", eval_metric = "cox-nloglik",
      eta = 0.05, max_depth = 4,
      subsample = 0.8, colsample_bytree = 0.8,
      min_child_weight = 1, lambda = 1, alpha = 0
    )
    xgb_model <- xgboost::xgb.train(params = params, data = dtrain2,
                                    watchlist = watch, nrounds = 1500,
                                    verbose = 0, early_stopping_rounds = 50)
    dtr$lp_xgb <- as.numeric(predict(xgb_model, xgboost::xgb.DMatrix(mm_tr), outputmargin = TRUE))
    cox_off_xgb <- survival::coxph(survival::Surv(T, D) ~ offset(lp_xgb),
                                   data = dtr, ties = "efron", x = TRUE, y = TRUE, model = TRUE)
  }

  # 4) CatBoost (Cox) + calibração por offset
  cat_model <- NULL; cox_off_cat <- NULL
  if (HAS_CAT && length(x_cols) > 0) {
    label_cat <- ifelse(dtr$D == 1, dtr$T, -dtr$T)
    pool_tr   <- catboost::catboost.load_pool(dtr[, x_cols, drop = FALSE], label = label_cat)
    cat_model <- catboost::catboost.train(
      pool_tr,
      params = list(
        loss_function = "Cox", eval_metric = "Cox",
        iterations = 1500, learning_rate = 0.06, depth = 6,
        l2_leaf_reg = 3.0, od_type = "Iter", od_wait = 100,
        random_seed = SEED + 200 * k
      )
    )
    dtr$lp_cat   <- as.numeric(catboost::catboost.predict(cat_model, pool_tr, prediction_type = "RawFormulaVal"))
    cox_off_cat  <- survival::coxph(survival::Surv(T, D) ~ offset(lp_cat),
                                    data = dtr, ties = "efron", x = TRUE, y = TRUE, model = TRUE)
  }

  # 5) AFT (flexsurv): escolhe dist por AIC
  fit_aft <- NULL
  aic_best <- Inf
  for (dist in c("weibull", "lognormal", "loglogistic")) {
    fit_try <- try(flexsurv::flexsurvreg(form_surv, data = dtr, dist = dist), silent = TRUE)
    if (!inherits(fit_try, "try-error")) {
      aic <- try(stats::AIC(fit_try), silent = TRUE)
      if (!inherits(aic, "try-error") && is.finite(aic) && aic < aic_best) {
        aic_best <- aic; fit_aft <- fit_try
      }
    }
  }

  # ------------------ Scoring @ tau* e nas curvas ------------------
  dte_off <- dte
  sc_map <- list()

  # Risco tau (absoluto) por modelo
  sc_map[["CoxPH"]] <- risk_at_tau(fit_cox, dte, tau_star)

  if (!is.null(cox_off_glm)) {
    dte_off$lp_glm <- as.numeric(predict(fit_glmnet, newx = mm_te, type = "link", s = lam_use))
    sc_map[["CoxNet"]] <- risk_at_tau(cox_off_glm, dte_off, tau_star)
  }
  if (!is.null(cox_off_xgb)) {
    lp_xgb_te <- as.numeric(predict(xgb_model, xgboost::xgb.DMatrix(mm_te), outputmargin = TRUE))
    dte_off$lp_xgb <- lp_xgb_te
    sc_map[["XGB(Cox)"]] <- risk_at_tau(cox_off_xgb, dte_off, tau_star)
  }
  if (!is.null(cox_off_cat)) {
    pool_te <- catboost::catboost.load_pool(dte[, x_cols, drop = FALSE],
                                            label = ifelse(dte$D == 1, dte$T, -dte$T))
    lp_cat_te <- as.numeric(catboost::catboost.predict(cat_model, pool_te, prediction_type = "RawFormulaVal"))
    dte_off$lp_cat <- lp_cat_te
    sc_map[["CatBoost(Cox)"]] <- risk_at_tau(cox_off_cat, dte_off, tau_star)
  }
  if (!is.null(fit_aft)) {
    sc_map[["AFT"]] <- risk_at_tau(fit_aft, dte, tau_star)
  }

  # C-Harrell (com base no risco tau)
  harrell <- function(time, status, score) {
    tryCatch(as.numeric(survival::concordance(survival::Surv(time, status) ~ score)$concordance),
             error = function(e) NA_real_)
  }
  c_har <- vapply(names(sc_map), function(nm) harrell(dte$T, dte$D, sc_map[[nm]]), numeric(1))

  # Uno’s C (treino -> teste)
  c_uno <- vapply(names(sc_map), function(nm) {
    safe_unoC(dtr$T, dtr$D, dte$T, dte$D, sc_map[[nm]], tau_star)
  }, numeric(1))

  # AUC(t) e Brier(t) nas curvas (times_eval) via riskRegression::Score
  cand_models <- list(CoxPH = fit_cox)
  if (!is.null(cox_off_glm)) cand_models[["CoxNet"]]     <- cox_off_glm
  if (!is.null(cox_off_xgb)) cand_models[["XGB(Cox)"]]   <- cox_off_xgb
  if (!is.null(cox_off_cat)) cand_models[["CatBoost(Cox)"]] <- cox_off_cat
  if (!is.null(fit_aft))     cand_models[["AFT"]]        <- fit_aft

    # --- Score
    score_one_model <- function(mod, name, dte_off, times_eval) {
  # Try Score() first; if it errors OR returns 0 rows, hard‑fallback to timeROC + IPCW
  out <- try(
    riskRegression::Score(
      object   = list(m = mod),
      formula  = survival::Surv(T, D) ~ 1,
      data     = dte_off,
      metrics  = c("AUC", "Brier"),
      times    = times_eval,
      conf.int = FALSE,
      summary  = "none"
    ),
    silent = TRUE
  )

  if (!inherits(out, "try-error")) {
    brier_df <- try(as.data.frame(out$Brier$score), silent = TRUE)
    auc_df   <- try(as.data.frame(out$AUC$score),   silent = TRUE)
    if (!inherits(brier_df, "try-error") && nrow(brier_df) > 0 &&
        !inherits(auc_df,   "try-error") && nrow(auc_df)   > 0) {
      if ("model" %in% names(brier_df)) brier_df$model <- name
      if ("model" %in% names(auc_df))   auc_df$model   <- name
      return(list(brier = brier_df, auc = auc_df))
    }
  }

  # --- Robust fallback: AUC(t) via timeROC; Brier(t) via IPCW (prodlim) ---
  Brier_rows <- vector("list", length(times_eval))
  AUC_rows   <- vector("list", length(times_eval))

  # IPCW weights (censoring)
  C <- 1L - as.integer(dte_off$D)
  fitG <- try(prodlim::prodlim(survival::Surv(dte_off$T, C) ~ 1), silent = TRUE)
  haveG <- !inherits(fitG, "try-error")

  for (i in seq_along(times_eval)) {
    tt <- times_eval[i]
    pr <- suppressWarnings(try(risk_at_tau(mod, dte_off, tt), silent = TRUE))
    if (inherits(pr, "try-error")) pr <- rep(NA_real_, nrow(dte_off))
    pr <- pmin(pmax(as.numeric(pr), 1e-12), 1 - 1e-12)

    auc_val <- suppressWarnings(try({
      to <- timeROC::timeROC(
        T = dte_off$T, delta = dte_off$D, marker = pr,
        cause = 1, weighting = "marginal", times = tt, iid = FALSE
      )
      if (length(to$AUC)) as.numeric(to$AUC[1]) else NA_real_
    }, silent = TRUE))
    if (inherits(auc_val, "try-error")) auc_val <- NA_real_

    bs <- NA_real_
    if (haveG) {
      tt_clamped <- pmin(dte_off$T, tt)
      Ghat <- suppressWarnings(try(predict(fitG, times = tt_clamped, type = "surv"), silent = TRUE))
      if (!inherits(Ghat, "try-error") && length(Ghat) == nrow(dte_off)) {
        w <- 1 / pmax(as.numeric(Ghat), 1e-6)
        y <- as.integer(dte_off$T <= tt & dte_off$D == 1)
        bs <- mean(w * (y - pr)^2, na.rm = TRUE)
      }
    }

    AUC_rows[[i]]   <- data.frame(model = name, times = tt, AUC   = as.numeric(auc_val))
    Brier_rows[[i]] <- data.frame(model = name, times = tt, Brier = as.numeric(bs))
  }

  list(brier = do.call(rbind, Brier_rows),
       auc   = do.call(rbind, AUC_rows))
}

    # Avalia cada modelo de cand_models separadamente
    IBS <- setNames(rep(NA_real_, length(cand_models)), names(cand_models))
    BS_tau <- setNames(rep(NA_real_, length(cand_models)), names(cand_models))
    AUC_tau <- setNames(rep(NA_real_, length(cand_models)), names(cand_models))
    auc_df <- brier_df <- NULL
    tdenom <- max(times_eval) - min(times_eval); if (!is.finite(tdenom) || tdenom <= 0) tdenom <- 1

    for (nm in names(cand_models)) {
    res <- score_one_model(cand_models[[nm]], nm, dte_off, times_eval)
    if (is.null(res)) next

    # Empilha curvas
    if (is.null(brier_df)) brier_df <- res$brier else brier_df <- rbind(brier_df, res$brier)
    if (is.null(auc_df))   auc_df   <- res$auc   else auc_df   <- rbind(auc_df,   res$auc)

    # Agregados por modelo
    sb <- subset(res$brier, model == nm)
    sa <- subset(res$auc,   model == nm)
    if (nrow(sb)) {
        IBS[nm]    <- trapz(sb$times, sb$Brier, denom = tdenom)
        BS_tau[nm] <- sb$Brier[which.min(abs(sb$times - tau_star))]
    }
    if (nrow(sa)) {
        AUC_tau[nm] <- sa$AUC[which.min(abs(sa$times - tau_star))]
    }
    }

    # Salva curvas se houver
    if (!is.null(auc_df) && nrow(auc_df) > 0) {
    auc_df$fold <- k
    utils::write.csv(auc_df,   file.path(PHASE5_DIR, sprintf("auc_curve_fold_%d.csv", k)),   row.names = FALSE)
    }
    if (!is.null(brier_df) && nrow(brier_df) > 0) {
    brier_df$fold <- k
    utils::write.csv(brier_df, file.path(PHASE5_DIR, sprintf("brier_curve_fold_%d.csv", k)), row.names = FALSE)
    }

  # Predições linha a linha (tau) pra auditoria
  pred_df <- data.frame(fold = k, row_id = folds_list[[k]], T = dte$T, D = dte$D)
  for (nm in names(sc_map)) pred_df[[paste0("risk_", gsub("[^A-Za-z0-9]+", "_", nm))]] <- sc_map[[nm]]
  utils::write.csv(pred_df, file.path(PRED_DIR, sprintf("fold_%d_predictions_tau.csv", k)), row.names = FALSE)

  # Métricas por modelo nesta dobra
  models_k <- names(sc_map)
  fold_tbl <- data.frame(
    fold = k,
    model = models_k,
    C_Harrell = as.numeric(c_har[models_k]),
    C_Uno_tau = as.numeric(c_uno[models_k]),
    IBS = as.numeric(IBS[models_k]),
    Brier_at_tau = as.numeric(BS_tau[models_k]),
    AUC_at_tau   = as.numeric(AUC_tau[models_k]),
    tau_star = tau_star,
    stringsAsFactors = FALSE
  )
  utils::write.csv(fold_tbl, file.path(PHASE5_DIR, sprintf("metrics_fold_%d.csv", k)), row.names = FALSE)

  # Curvas por dobra (se rolar)
  if (!is.null(auc_df) && nrow(auc_df) > 0) {
    auc_df$fold <- k
    utils::write.csv(auc_df, file.path(PHASE5_DIR, sprintf("auc_curve_fold_%d.csv", k)), row.names = FALSE)
  }
  if (!is.null(brier_df) && nrow(brier_df) > 0) {
    brier_df$fold <- k
    utils::write.csv(brier_df, file.path(PHASE5_DIR, sprintf("brier_curve_fold_%d.csv", k)), row.names = FALSE)
  }

  res_rows[[k]] <- fold_tbl

  # Log de progresso
  elapsed <- as.numeric(difftime(Sys.time(), t_fold, units = "secs"))
  message(sprintf("[phase5] Dobra %d concluída | modelos: %s | tau*=%.5f | secs=%.1f",
                  k, paste(models_k, collapse = ", "), tau_star, elapsed))
  log_runtime("phase5", k, "fit_and_score_fold", elapsed)
}

# --- Consolida e sumariza ---
metrics_all <- dplyr::bind_rows(res_rows)
utils::write.csv(metrics_all, file.path(OUT_DIR, "per_fold_metrics.csv"), row.names = FALSE)

metrics_summary <- metrics_all |>
  dplyr::group_by(model) |>
  dplyr::summarize(
    C_Harrell_mean = mean(C_Harrell, na.rm = TRUE),
    C_Harrell_sd   = stats::sd(C_Harrell,   na.rm = TRUE),
    C_Uno_tau_mean = mean(C_Uno_tau, na.rm = TRUE),
    C_Uno_tau_sd   = stats::sd(C_Uno_tau,   na.rm = TRUE),
    IBS_mean       = mean(IBS, na.rm = TRUE),
    IBS_sd         = stats::sd(IBS, na.rm = TRUE),
    Brier_tau_mean = mean(Brier_at_tau, na.rm = TRUE),
    Brier_tau_sd   = stats::sd(Brier_at_tau,   na.rm = TRUE),
    AUC_tau_mean   = mean(AUC_at_tau, na.rm = TRUE),
    AUC_tau_sd     = stats::sd(AUC_at_tau,   na.rm = TRUE),
    .groups = "drop"
  )
utils::write.csv(metrics_summary, file.path(OUT_DIR, "metrics_summary.csv"), row.names = FALSE)

message("[phase5] Artefatos salvos: per_fold_metrics.csv, metrics_summary.csv, predictions/fold_*.csv, phase5_models/*.csv")
log_runtime("phase5", NA, "complete", as.numeric(difftime(Sys.time(), t_phase5, units = "secs")))

# ================================================================
# Phase 6 — Calibration tau (per fold)
# - Consome: predictions/fold_*_predictions_tau.csv (Phase 5) e prep_phase3/fold_*_test.rds
# - Gera: phase6_calibration/calibration_bins.csv, calibration_summary.csv, calibration_summary_by_model.csv
# ================================================================

t_phase6 <- Sys.time()

PHASE6_DIR <- file.path(OUT_DIR, "phase6_calibration")
if (!dir.exists(PHASE6_DIR)) dir.create(PHASE6_DIR, recursive = TRUE, showWarnings = FALSE)

# Garantir prodlim (usado para KM)
if (!requireNamespace("prodlim", quietly = TRUE)) {
  try(utils::install.packages("prodlim", quiet = TRUE), silent = TRUE)
}
HAS_PRODLIM <- requireNamespace("prodlim", quietly = TRUE)
if (!HAS_PRODLIM) stop("[phase6] Pacote 'prodlim' é necessário para a calibração (KM).")

# Utilitários locais
clamp01 <- function(x, eps = 1e-6) pmin(pmax(as.numeric(x), eps), 1 - eps)

km_observed_risk <- function(time, status, tau) {
  # 1 - KM(tau) dentro do subconjunto
  if (length(time) < 2L) return(NA_real_)
  fit <- try(prodlim::prodlim(survival::Surv(time, status) ~ 1), silent = TRUE)
  if (inherits(fit, "try-error")) return(NA_real_)
  # IMPORTANT: use o genérico predict(), não prodlim::predict()
  s_at_tau <- try(predict(fit, times = tau, type = "surv"), silent = TRUE)
  if (inherits(s_at_tau, "try-error") || length(s_at_tau) < 1L) return(NA_real_)
  as.numeric(1 - s_at_tau[1])
}

make_calibration_bins <- function(time, status, pred, tau, bins = 10L) {
  pred <- clamp01(pred)

  # Guard: tudo NA/Inf — retorna estrutura vazia
  if (!any(is.finite(pred))) {
    return(list(
      bins = data.frame(bin = integer(0), n = integer(0),
                        pred_mean = numeric(0), pred_median = numeric(0),
                        obs_km = numeric(0)),
      slope = NA_real_, intercept = NA_real_,
      E_avg = NA_real_, E_max = NA_real_, CIL = NA_real_
    ))
  }

  # Quebras por quantis; se houver empates fortes, tentamos sequência regular;
  # se ainda assim não houver estrita monotonicidade, caímos no bin único.
  v  <- pred[is.finite(pred)]
  qs <- stats::quantile(v, probs = seq(0, 1, length.out = bins + 1L), na.rm = TRUE, type = 8)
  br <- sort(unique(as.numeric(qs)))

  single_bin_fallback <- function() {
    bstats <- data.frame(
      bin         = 1L,
      n           = length(v),
      pred_mean   = mean(v, na.rm = TRUE),
      pred_median = stats::median(v, na.rm = TRUE),
      obs_km      = km_observed_risk(time, status, tau)
    )
    Eavg <- abs(bstats$obs_km - bstats$pred_mean)
    return(list(
      bins      = bstats,
      slope     = NA_real_,
      intercept = NA_real_,
      E_avg     = Eavg,
      E_max     = Eavg,
      CIL       = bstats$obs_km - bstats$pred_mean
    ))
  }

  if (length(br) < 2L) {
    rng <- range(v, na.rm = TRUE, finite = TRUE)
    if (!all(is.finite(rng)) || diff(rng) <= 0) return(single_bin_fallback())
    br <- sort(unique(seq(rng[1], rng[2], length.out = bins + 1L)))
  }

  # Garantia final: 'breaks' estritamente crescentes; se falhar, um único bin
  if (length(br) < 2L || any(diff(br) <= 0)) return(single_bin_fallback())

  # Binning seguro
  bin_id <- cut(pred, breaks = br, include.lowest = TRUE, labels = FALSE)
  B <- sort(unique(bin_id[is.finite(bin_id)]))

  # Se, por algum motivo, nada foi binned, cai no bin único
  if (!length(B)) return(single_bin_fallback())

  # Se por algum motivo nada foi binned, cai no bin único
  if (!length(B)) {
    bstats <- data.frame(
      bin         = 1L,
      n           = length(v),
      pred_mean   = mean(v, na.rm = TRUE),
      pred_median = stats::median(v, na.rm = TRUE),
      obs_km      = km_observed_risk(time, status, tau)
    )
    Eavg <- abs(bstats$obs_km - bstats$pred_mean)
    return(list(
      bins      = bstats,
      slope     = NA_real_,
      intercept = NA_real_,
      E_avg     = Eavg,
      E_max     = Eavg,
      CIL       = bstats$obs_km - bstats$pred_mean
    ))
  }

  rows <- vector("list", length(B))
  for (i in seq_along(B)) {
    idx <- which(bin_id == B[i])
    if (!length(idx)) next
    obs <- km_observed_risk(time[idx], status[idx], tau)
    rows[[i]] <- data.frame(
      bin = B[i],
      n   = length(idx),
      pred_mean   = mean(pred[idx]),
      pred_median = stats::median(pred[idx]),
      obs_km      = obs
    )
  }
  bins_df <- do.call(rbind, rows)

  # Regressão linear e isotônica
  lin <- try(stats::lm(obs_km ~ pred_mean, data = bins_df, weights = n), silent = TRUE)
  if (inherits(lin, "try-error")) {
    slope <- NA_real_; intercept <- NA_real_
  } else {
    cf <- stats::coef(lin)
    intercept <- as.numeric(cf[1])
    slope <- if (length(cf) >= 2) as.numeric(cf[2]) else NA_real_
  }

  iso_est <- rep(NA_real_, nrow(bins_df))
  o <- order(bins_df$pred_mean)
  iso <- try(stats::isoreg(bins_df$pred_mean[o], bins_df$obs_km[o]), silent = TRUE)
  if (!inherits(iso, "try-error") && !is.null(iso$yf)) iso_est[o] <- as.numeric(iso$yf)
  bins_df$obs_iso <- iso_est

  # Erros de calibração (ECE-like)
  dif <- abs(bins_df$obs_km - bins_df$pred_mean)
  fin <- is.finite(dif)
  E_avg <- if (any(fin)) sum(bins_df$n[fin] * dif[fin]) / sum(bins_df$n[fin]) else NA_real_
  E_max <- if (any(fin)) max(dif[fin]) else NA_real_

  obs_overall <- km_observed_risk(time, status, tau)
  CIL <- if (is.finite(obs_overall)) obs_overall - mean(pred, na.rm = TRUE) else NA_real_

  list(bins = bins_df, slope = slope, intercept = intercept, E_avg = E_avg, E_max = E_max, CIL = CIL)
}
# Folds e tau
if (!exists("folds_list")) {
  FOLDS_RDS <- file.path(OUT_DIR, "cv5_folds.rds")
  if (!file.exists(FOLDS_RDS)) stop("[phase6] cv5_folds.rds não encontrado.")
  folds_list <- readRDS(FOLDS_RDS)
}
if (!exists("tau_by_fold")) {
  PHASE4_RDS <- file.path(OUT_DIR, "phase4_times.rds")
  if (!file.exists(PHASE4_RDS)) stop("[phase6] 'phase4_times.rds' não encontrado. Execute as fases anteriores.")
  times_obj   <- readRDS(PHASE4_RDS)
  tau_by_fold <- as.numeric(times_obj$tau_star_by_fold)
}

sanitize_tau_for_timeROC <- function(Tvec, tau) {
  Tvec <- Tvec[is.finite(Tvec)]
  if (!length(Tvec)) return(NA_real_)
  tmax <- max(Tvec); tmin <- min(Tvec)
  tau_use <- min(tau, tmax - 1e-6)  # timeROC exige times < max(T)
  if (!is.finite(tau_use) || tau_use <= 0) tau_use <- (tmin + tmax) / 2
  tau_use
}

pretty_model <- function(risk_col) {
  nm <- sub("^risk_", "", risk_col)
  nm <- gsub("_", " ", nm)
  nm <- gsub("XGB Cox", "XGB(Cox)", nm)
  nm <- gsub("CatBoost Cox", "CatBoost(Cox)", nm)
  nm <- gsub("AFT", "AFT", nm)
  nm <- gsub("CoxPH", "CoxPH", nm)
  nm <- gsub("CoxNet", "CoxNet", nm)
  nm
}

calib_bins_all   <- list()
calib_summary_all<- list()

for (k in seq_along(folds_list)) {
  t_fold <- Sys.time()
  tau_k  <- as.numeric(tau_by_fold[k])

  test_path <- file.path(OUT_DIR, "prep_phase3", sprintf("fold_%d_test.rds",  k))
  pred_path <- file.path(OUT_DIR, "predictions", sprintf("fold_%d_predictions_tau.csv", k))
  if (!file.exists(test_path) || !file.exists(pred_path)) {
    message(sprintf("[phase6][warn] Artefatos ausentes para a dobra %d — pulando.", k))
    next
  }

  dte   <- readRDS(test_path)
  preds <- utils::read.csv(pred_path, check.names = FALSE)

  # Checagens mínimas
  if (!all(c("T","D") %in% names(preds))) {
    preds$T <- dte$T; preds$D <- dte$D
  }

  risk_cols <- grep("^risk_", names(preds), value = TRUE)
  if (length(risk_cols) == 0L) {
    message(sprintf("[phase6][warn] Sem colunas 'risk_*' na dobra %d — pulando.", k))
    next
  }

  # Por modelo: bins e resumo
  fold_bins <- list()
  fold_sum  <- list()
  for (rc in risk_cols) {
  p <- preds[[rc]]

  # Se o modelo retornou apenas NA/Inf, pula e registra aviso claro
  if (!any(is.finite(p))) {
    message(sprintf("[phase6][warn] Dobra %d | %s: todas as predições são NA/Inf — calibração pulada.",
                    k, pretty_model(rc)))
    next
  }

  cal <- make_calibration_bins(time = preds$T, status = preds$D, pred = p, tau = tau_k, bins = 10L)

  bins_df <- cal$bins
  bins_df$fold   <- k
  bins_df$model  <- pretty_model(rc)
  bins_df$tau    <- tau_k

  fold_bins[[rc]] <- bins_df

  fold_sum[[rc]] <- data.frame(
    fold      = k,
    model     = pretty_model(rc),
    tau       = tau_k,
    n_test    = nrow(preds),
    n_bins    = nrow(bins_df),
    bin_min_n = if (nrow(bins_df)) min(bins_df$n, na.rm = TRUE) else 0L,
    slope     = cal$slope,
    intercept = cal$intercept,
    E_avg     = cal$E_avg,
    E_max     = cal$E_max,
    CIL       = cal$CIL,
    stringsAsFactors = FALSE
  )
}

  calib_bins_all[[k]]    <- dplyr::bind_rows(fold_bins)
  calib_summary_all[[k]] <- dplyr::bind_rows(fold_sum)

  # Persistência por dobra (útil para debug)
  utils::write.csv(calib_bins_all[[k]],
                   file.path(PHASE6_DIR, sprintf("calibration_bins_fold_%d.csv", k)),
                   row.names = FALSE)

  log_runtime("phase6", k, "calibrate_fold", as.numeric(difftime(Sys.time(), t_fold, units = "secs")))
  message(sprintf("[phase6] Dobra %d calibrada @ tau*=%.5f | modelos: %s",
                  k, tau_k, paste(unique(calib_bins_all[[k]]$model), collapse = ", ")))
}

# Consolida e salva
calib_bins_df    <- dplyr::bind_rows(calib_bins_all)
calib_summary_df <- dplyr::bind_rows(calib_summary_all)

utils::write.csv(calib_bins_df,    file.path(PHASE6_DIR, "calibration_bins.csv"),    row.names = FALSE)
utils::write.csv(calib_summary_df, file.path(PHASE6_DIR, "calibration_summary.csv"), row.names = FALSE)

# Resumo por modelo (média±sd por dobra)
calib_by_model <- calib_summary_df %>%
  dplyr::group_by(model) %>%
  dplyr::summarize(
    slope_mean = mean(slope, na.rm = TRUE),  slope_sd = stats::sd(slope, na.rm = TRUE),
    int_mean   = mean(intercept, na.rm = TRUE), int_sd = stats::sd(intercept, na.rm = TRUE),
    Eavg_mean  = mean(E_avg, na.rm = TRUE),  Eavg_sd  = stats::sd(E_avg, na.rm = TRUE),
    Emax_mean  = mean(E_max, na.rm = TRUE),  Emax_sd  = stats::sd(E_max, na.rm = TRUE),
    CIL_mean   = mean(CIL,  na.rm = TRUE),   CIL_sd   = stats::sd(CIL,  na.rm = TRUE),
    .groups = "drop"
  )

utils::write.csv(calib_by_model, file.path(PHASE6_DIR, "calibration_summary_by_model.csv"), row.names = FALSE)

message("[phase6] Artefatos salvos em: ", PHASE6_DIR)
log_runtime("phase6", NA, "complete", as.numeric(difftime(Sys.time(), t_phase6, units = "secs")))

# ================================================================
# Phase 7 — Training & scoring orchestrator (per fold, canonical outputs)
# - Consolida predições tau, métricas (Phase 5) e calibração (Phase 6)
# - Adiciona relatório de cobertura de NA e resumo de runtime
# ================================================================

t_phase7 <- Sys.time()
PHASE7_DIR <- file.path(OUT_DIR, "phase7_pipeline")
if (!dir.exists(PHASE7_DIR)) dir.create(PHASE7_DIR, recursive = TRUE, showWarnings = FALSE)

# 1) Predições empilhadas tau (todas as dobras)
PRED_DIR <- file.path(OUT_DIR, "predictions")
pred_files <- list.files(PRED_DIR, pattern = "^fold_\\d+_predictions_tau\\.csv$", full.names = TRUE)
if (length(pred_files) == 0L) {
  stop("[phase7] Nenhum arquivo de predição encontrado em 'predictions/'. Rode Phase 5 antes.")
}

pred_stack <- lapply(pred_files, function(f) {
  df <- utils::read.csv(f, check.names = FALSE)
  k  <- as.integer(gsub(".*fold_(\\d+)_predictions_tau\\.csv$", "\\1", f))
  df$fold <- k
  df
})
pred_stack_df <- dplyr::bind_rows(pred_stack)

# Colunas canônicas (se 'row_id' não existir, cria NA)
if (!("row_id" %in% names(pred_stack_df))) pred_stack_df$row_id <- NA_integer_
pred_stack_df <- pred_stack_df |>
  dplyr::relocate(fold, row_id, T, D)

utils::write.csv(pred_stack_df,
                 file.path(PHASE7_DIR, "predictions_tau_stacked.csv"),
                 row.names = FALSE)

# 2) Métricas + Calibração (join por dobra e modelo)
PER_FOLD_METRICS <- file.path(OUT_DIR, "per_fold_metrics.csv")
CALIB_SUMMARY    <- file.path(OUT_DIR, "phase6_calibration", "calibration_summary.csv")
if (!file.exists(PER_FOLD_METRICS)) stop("[phase7] per_fold_metrics.csv não encontrado (Phase 5).")
if (!file.exists(CALIB_SUMMARY))    stop("[phase7] calibration_summary.csv não encontrado (Phase 6).")

met_all  <- utils::read.csv(PER_FOLD_METRICS)
calib_sm <- utils::read.csv(CALIB_SUMMARY)

calib_keep <- calib_sm[, c("fold","model","tau","n_test","n_bins","bin_min_n",
                           "slope","intercept","E_avg","E_max","CIL")]

phase7_join <- met_all |>
  dplyr::left_join(calib_keep, by = c("fold","model"))

utils::write.csv(phase7_join,
                 file.path(PHASE7_DIR, "metrics_plus_calibration.csv"),
                 row.names = FALSE)

# 3) Relatório de cobertura de NA nas predições @ tau*
na_cov <- pred_stack_df |>
  dplyr::select(fold, dplyr::starts_with("risk_")) |>
  tidyr::pivot_longer(-fold, names_to = "risk_col", values_to = "risk") |>
  dplyr::group_by(fold, risk_col) |>
  dplyr::summarize(n = dplyr::n(),
                   na_n = sum(!is.finite(risk) | is.na(risk)),
                   na_pct = na_n / n,
                   .groups = "drop") |>
  dplyr::arrange(fold, dplyr::desc(na_pct))

utils::write.csv(na_cov,
                 file.path(PHASE7_DIR, "na_coverage_report.csv"),
                 row.names = FALSE)

# 4) Resumo de runtime por fase/etapa (se disponível)
RUNTIME_LOG <- file.path(OUT_DIR, "runtime_log.csv")
if (file.exists(RUNTIME_LOG)) {
  rt <- utils::read.csv(RUNTIME_LOG)
  rt_sum <- rt |>
    dplyr::group_by(phase, step) |>
    dplyr::summarize(total_seconds = sum(seconds, na.rm = TRUE),
                     n_calls = dplyr::n(),
                     .groups = "drop") |>
    dplyr::arrange(phase, step)
  utils::write.csv(rt_sum,
                   file.path(PHASE7_DIR, "runtime_summary.csv"),
                   row.names = FALSE)
}

# 5) Mensagens pro console
n_preds <- nrow(pred_stack_df)
n_rows_mp <- nrow(phase7_join)
flag_na <- na_cov |>
  dplyr::filter(na_pct > 0) |>
  dplyr::distinct(risk_col) |>
  dplyr::pull(risk_col) |>
  unique()

message(sprintf("[phase7] Consolidados: preds=%d linhas | metrics+calib=%d linhas",
                n_preds, n_rows_mp))
if (length(flag_na)) {
  message("[phase7][warn] Modelos com NA>0% nas predições @ tau*: ", paste(flag_na, collapse = ", "))
}
message("[phase7] Artefatos salvos em: ", PHASE7_DIR)

log_runtime("phase7", NA, "consolidate",
            as.numeric(difftime(Sys.time(), t_phase7, units = "secs")))


# ================================================================
# Phase 8 — Interpretability (global, semi-global, group-level)
# - Reajusta a campeã no corte completo para diagnósticos estáveis
# - Gera HR table, Schoenfeld, volcano; SHAP (XGB); ALE/PDP/ICE; KM por decis
# ================================================================

t_phase8 <- Sys.time()
PHASE8_DIR <- file.path(OUT_DIR, "phase8_interpretability")
if (!dir.exists(PHASE8_DIR)) dir.create(PHASE8_DIR, recursive = TRUE, showWarnings = FALSE)

# --- Dependências auxiliares ---
pkgs_phase8 <- c("ggplot2","dplyr","tidyr","survival","prodlim","iml","shapviz","ggpubr","scales",
                 "Metrics","shades","ggfittext","gggenes")
for (p in pkgs_phase8) if (!requireNamespace(p, quietly = TRUE)) try(utils::install.packages(p, quiet = TRUE), silent = TRUE)
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(survival); library(scales)
  if (requireNamespace("prodlim", quietly = TRUE)) library(prodlim)
  if (requireNamespace("iml", quietly = TRUE))      library(iml)
  if (requireNamespace("shapviz", quietly = TRUE))  library(shapviz)
})

HAS_XGB  <- requireNamespace("xgboost", quietly = TRUE)
HAS_CAT  <- requireNamespace("catboost", quietly = TRUE)  # pode faltar; tratar c/ fallback
HAS_IML  <- requireNamespace("iml", quietly = TRUE)

# --- 8.1 Campeã pelo resumo de métricas (C_Uno) ---
champion <- "CoxPH"
crit_used <- "C_Uno@tau*"
ms_path <- file.path(OUT_DIR, "metrics_summary.csv")
if (file.exists(ms_path)) {
  ms <- try(utils::read.csv(ms_path), silent = TRUE)
  if (!inherits(ms, "try-error") && nrow(ms) > 0) {
    if ("C_Uno_tau_mean" %in% names(ms) && any(is.finite(ms$C_Uno_tau_mean))) {
      champion <- as.character(ms$model[which.max(ms$C_Uno_tau_mean)])
      crit_used <- "C_Uno@tau*"
    } else if ("C_Harrell_mean" %in% names(ms) && any(is.finite(ms$C_Harrell_mean))) {
      champion <- as.character(ms$model[which.max(ms$C_Harrell_mean)])
      crit_used <- "C_Harrell"
    }
  }
}
message(sprintf("[phase8] Campeã: %s (critério: %s).", champion, crit_used))

# --- 8.2 Prepara corte completo (reuso da política da Phase 3) ---
#    (usar make_fold_preproc com train=test=df para imputa/níveis consistentes)
prep8 <- make_fold_preproc(df, df, time_col = "T", status_col = "D", biz_cats = biz_cats_present)
d_all <- prep8$train
art8  <- prep8$artifacts
x_cols8 <- c(art8$num_cols, art8$cat_cols)

# tau* global para interpretabilidade (mediana dos tempos de evento)
ev_all <- d_all$T[d_all$D == 1 & is.finite(d_all$T)]
tau8 <- if (length(ev_all) >= 1) as.numeric(stats::median(ev_all, na.rm = TRUE)) else
        as.numeric(stats::median(d_all$T, na.rm = TRUE))
if (!is.finite(tau8) || tau8 <= 0) tau8 <- max(1e-6, as.numeric(stats::median(d_all$T, na.rm = TRUE)))
message(sprintf("[phase8] tau* (interpretabilidade) = %.5f.", tau8))

# Local Helpers
build_mm <- function(df, x_cols) {
  if (length(x_cols) == 0) return(matrix(numeric(0), nrow = nrow(df), ncol = 0))
  mm <- stats::model.matrix(stats::reformulate(x_cols), data = df)
  mm[, colnames(mm) != "(Intercept)", drop = FALSE]
}

risk_at_tau <- function(model, newdata, tau) {
  # flexsurv first
  if (inherits(model, "flexsurvreg")) {
    res <- try(flexsurv::summary(model, newdata = newdata, t = tau, type = "survival"), silent = TRUE)
    if (!inherits(res, "try-error")) {
      s <- vapply(res, function(xx) if (is.data.frame(xx) && "est" %in% names(xx)) as.numeric(xx$est[1]) else NA_real_, numeric(1))
      return(as.numeric(1 - s))
    }
  }
  # riskRegression / pec
  pr <- try(riskRegression::predictRisk(model, newdata = newdata, times = tau), silent = TRUE)
  if (!inherits(pr, "try-error")) return(pmin(pmax(as.numeric(pr), 1e-12), 1 - 1e-12))
  sp <- try(pec::predictSurvProb(model, newdata = newdata, times = tau), silent = TRUE)
  if (!inherits(sp, "try-error")) return(pmin(pmax(as.numeric(1 - sp), 1e-12), 1 - 1e-12))
  # --- CoxPH fallback: basehaz + LP (robust) ---
  if (inherits(model, "coxph")) {
    lp <- try(as.numeric(predict(model, newdata = newdata, type = "lp")), silent = TRUE)
    bh <- try(survival::basehaz(model, centered = FALSE), silent = TRUE)
    if (!inherits(lp, "try-error") && !inherits(bh, "try-error") && nrow(bh) > 0) {
      H0_tau <- stats::approx(x = bh$time, y = bh$hazard, xout = tau, method = "linear", rule = 2)$y
      risk <- 1 - exp(-H0_tau * exp(lp))
      return(pmin(pmax(as.numeric(risk), 1e-12), 1 - 1e-12))
    }
  }
  rep(NA_real_, nrow(newdata))
}

# --- 8.3 Ajustes no corte completo  ---
form_surv <- stats::as.formula("Surv(T, D) ~ .")

# CoxPH: sempre geramos para HR/Schoenfeld/Volcano
sf8 <- safe_fit_coxph(d_all, time_col = "T", status_col = "D")
fit_cox8 <- sf8$fit
if (length(sf8$dropped)) {
  message(sprintf("[phase8][note] CoxPH global: removidos termos problemáticos: %s",
                  paste(sf8$dropped, collapse = ", ")))
}

# CoxNet (se for campeã — para risco; HR detalhado fica no CoxPH)
glmnet8 <- NULL; cox_off_glm8 <- NULL; lam8 <- NA_real_
mm_all <- build_mm(d_all, x_cols8)
if (identical(champion, "CoxNet") && ncol(mm_all) > 0) {
  a_use <- 0.5
  lam8 <- tryCatch({
    cv <- glmnet::cv.glmnet(x = mm_all, y = survival::Surv(d_all$T, d_all$D),
                             family = "cox", alpha = a_use, nfolds = 5, type.measure = "deviance")
    if (is.finite(cv$lambda.min)) cv$lambda.min else cv$lambda.1se
  }, error = function(e) 0.01)
  glmnet8 <- try(glmnet::glmnet(mm_all, y = survival::Surv(d_all$T, d_all$D),
                                family = "cox", alpha = a_use, lambda = lam8), silent = TRUE)
  if (!inherits(glmnet8, "try-error")) {
    d_all$lp_glm8 <- as.numeric(predict(glmnet8, newx = mm_all, type = "link", s = lam8))
    cox_off_glm8 <- survival::coxph(Surv(T, D) ~ offset(lp_glm8),
                                    data = d_all, ties = "efron", x = TRUE, y = TRUE, model = TRUE)
  }
}

# XGB (se for campeã) + calibração via Cox offset
xgb8 <- NULL; cox_off_xgb8 <- NULL
if (identical(champion, "XGB(Cox)") && HAS_XGB && ncol(mm_all) > 0) {
  set.seed(SEED + 800)
  label <- ifelse(d_all$D == 1, d_all$T, -d_all$T)
  dmat  <- xgboost::xgb.DMatrix(data = mm_all, label = label)
  xgb8  <- xgboost::xgb.train(
    params = list(objective = "survival:cox", eval_metric = "cox-nloglik",
                  eta = 0.05, max_depth = 4, subsample = 0.8, colsample_bytree = 0.8,
                  min_child_weight = 1, lambda = 1, alpha = 0),
    data = dmat, nrounds = 1500, verbose = 0, early_stopping_rounds = 50
  )
  d_all$lp_xgb8 <- as.numeric(predict(xgb8, dmat, outputmargin = TRUE))
  cox_off_xgb8  <- survival::coxph(Surv(T, D) ~ offset(lp_xgb8),
                                   data = d_all, ties = "efron", x = TRUE, y = TRUE, model = TRUE)
} else if (identical(champion, "XGB(Cox)") && !HAS_XGB) {
  message("[phase8][warn] 'xgboost' indisponível — SHAP e risco da campeã XGB ficarão indisponíveis nesta fase.")
}

# CatBoost (se for campeã) + calibração via Cox offset
cat8 <- NULL; cox_off_cat8 <- NULL
if (identical(champion, "CatBoost(Cox)") && HAS_CAT && length(x_cols8) > 0) {
  label_cat <- ifelse(d_all$D == 1, d_all$T, -d_all$T)
  pool_all  <- catboost::catboost.load_pool(d_all[, x_cols8, drop = FALSE], label = label_cat)
  cat8 <- catboost::catboost.train(
    pool_all,
    params = list(
      loss_function = "Cox", eval_metric = "Cox",
      iterations = 1500, learning_rate = 0.06, depth = 6,
      l2_leaf_reg = 3.0, od_type = "Iter", od_wait = 100,
      random_seed = SEED + 800
    )
  )
  d_all$lp_cat8  <- as.numeric(catboost::catboost.predict(cat8, pool_all, prediction_type = "RawFormulaVal"))
  cox_off_cat8   <- survival::coxph(Surv(T, D) ~ offset(lp_cat8),
                                    data = d_all, ties = "efron", x = TRUE, y = TRUE, model = TRUE)
} else if (identical(champion, "CatBoost(Cox)") && !HAS_CAT) {
  message("[phase8][warn] 'catboost' indisponível — risco da campeã não será calculado.")
}

# AFT (se for campeã)
aft8 <- NULL
if (identical(champion, "AFT")) {
  aic_best <- Inf
  for (dist in c("weibull","lognormal","loglogistic")) {
    fit_try <- try(flexsurv::flexsurvreg(form_surv, data = d_all, dist = dist), silent = TRUE)
    if (!inherits(fit_try, "try-error")) {
      aic <- try(stats::AIC(fit_try), silent = TRUE)
      if (!inherits(aic, "try-error") && is.finite(aic) && aic < aic_best) { aic_best <- aic; aft8 <- fit_try }
    }
  }
}

# --- 8.4 Tabela de HR, Schoenfeld e Volcano (CoxPH) ---
cox_sum <- summary(fit_cox8)
hr_tbl <- data.frame(
  term = rownames(cox_sum$coef),
  coef = cox_sum$coef[, "coef"],
  HR   = exp(cox_sum$coef[, "coef"]),
  se   = cox_sum$coef[, "se(coef)"],
  z    = cox_sum$coef[, "z"],
  pval = cox_sum$coef[, "Pr(>|z|)"],
  stringsAsFactors = FALSE
)
ci <- try(stats::confint(fit_cox8), silent = TRUE)
if (!inherits(ci, "try-error") && nrow(ci) == nrow(hr_tbl)) {
  hr_tbl$HR_lo <- exp(ci[,1]); hr_tbl$HR_hi <- exp(ci[,2])
} else {
  hr_tbl$HR_lo <- NA_real_; hr_tbl$HR_hi <- NA_real_
}
utils::write.csv(hr_tbl, file.path(PHASE8_DIR, "coxph_hazard_ratios.csv"), row.names = FALSE)

# Teste de Schoenfeld (PH)
cz <- try(survival::cox.zph(fit_cox8), silent = TRUE)
if (!inherits(cz, "try-error")) {
  cz_tbl <- as.data.frame(cz$table)
  cz_tbl$term <- rownames(cz$table)
  utils::write.csv(cz_tbl, file.path(PHASE8_DIR, "coxph_schoenfeld_test.csv"), row.names = FALSE)
}

# Volcano: efeito (log(HR)) vs -log10(p)
vol_df <- hr_tbl %>% mutate(logHR = log(HR), neglog10p = -log10(pval))
p_vol <- ggplot(vol_df, aes(x = logHR, y = neglog10p)) +
  geom_point(alpha = 0.8) +
  geom_vline(xintercept = 0, linetype = 2) +
  labs(title = "Volcano (CoxPH)", x = "log(HR)", y = expression(-log[10](p))) +
  theme_minimal(base_size = 12)
try(ggsave(file.path(PHASE8_DIR, "coxph_volcano.png"), p_vol, width = 7, height = 5, dpi = 150), silent = TRUE)

# --- 8.5 SHAP (XGB), se disponível e relevante ---
if (identical(champion, "XGB(Cox)") && !is.null(xgb8) && HAS_XGB) {
  # SHAP via predcontrib
  dmat <- xgboost::xgb.DMatrix(data = mm_all)
  shap <- try(predict(xgb8, dmat, predcontrib = TRUE), silent = TRUE)
  if (!inherits(shap, "try-error")) {
    shap <- as.matrix(shap)
    # última coluna costuma ser BIAS; descartamos
    if (ncol(shap) > 1) {
      shap <- shap[, seq_len(ncol(shap) - 1), drop = FALSE]
    }
    colnames(shap) <- colnames(mm_all)
    mean_abs <- sort(colMeans(abs(shap), na.rm = TRUE), decreasing = TRUE)
    top20 <- head(data.frame(feature = names(mean_abs), mean_abs_shap = as.numeric(mean_abs)), 20)
    utils::write.csv(top20, file.path(PHASE8_DIR, "xgb_shap_top20.csv"), row.names = FALSE)

  # Plano B do beeswarm: se tiver shapviz, plota; se não, vai de barra simples
    if (requireNamespace("shapviz", quietly = TRUE)) {
      sv <- shapviz::shapviz(shap, X = as.data.frame(mm_all))
      png(file.path(PHASE8_DIR, "xgb_shap_beeswarm.png"), width = 1100, height = 800)
      print(shapviz::sv_importance(sv, kind = "bee", max_display = 20))
      dev.off()
    } else {
      p_bar <- ggplot(top20, aes(x = reorder(feature, mean_abs_shap), y = mean_abs_shap)) +
        geom_col() + coord_flip() + labs(title = "XGB — |SHAP| médio (top‑20)", x = NULL, y = "|SHAP| médio") +
        theme_minimal(base_size = 12)
      try(ggsave(file.path(PHASE8_DIR, "xgb_shap_top20_bar.png"), p_bar, width = 7, height = 6, dpi = 150), silent = TRUE)
      message("[phase8][note] 'shapviz' não encontrado — gerado gráfico de barras simples de |SHAP|.")
    }
  } else {
    message("[phase8][warn] SHAP via predcontrib falhou para XGB — sem gráficos.")
  }
}

# --- 8.6 ALE / PDP / ICE para os principais drivers (campeã) ---
# Escolha de features líderes
top_features <- character(0)
if (identical(champion, "CoxPH")) {
  # |coef| como proxy de importância
  ord <- order(abs(hr_tbl$coef), decreasing = TRUE, na.last = NA)
  top_features <- hr_tbl$term[ord]
} else if (identical(champion, "CoxNet") && !is.null(glmnet8)) {
  cf <- try(as.numeric(coef(glmnet8, s = lam8)), silent = TRUE)
  if (!inherits(cf, "try-error")) {
    nm <- rownames(coef(glmnet8, s = lam8))
    ord <- order(abs(cf), decreasing = TRUE, na.last = NA)
    top_features <- nm[ord]
  }
} else if (identical(champion, "XGB(Cox)") && HAS_XGB && !is.null(xgb8)) {
  # Se existir top20 de SHAP, usa-os; senão, ganho do xgb.importance
  shap_csv <- file.path(PHASE8_DIR, "xgb_shap_top20.csv")
  if (file.exists(shap_csv)) {
    top_features <- utils::read.csv(shap_csv)$feature
  } else {
    imp <- try(xgboost::xgb.importance(model = xgb8), silent = TRUE)
    if (!inherits(imp, "try-error") && nrow(imp)) top_features <- imp$Feature
  }
} else if (identical(champion, "AFT") && !is.null(aft8)) {
  # usa variáveis do modelo (todas, ordenação neutra)
  top_features <- x_cols8
}

# Limita a 6–8 variáveis úteis
top_features <- intersect(top_features, x_cols8)
if (length(top_features) > 8) top_features <- top_features[1:8]

# Preditor de risco@tau* conforme a campeã
predict_risk_champion <- function(newdata) {
  if (identical(champion, "CoxNet") && !is.null(glmnet8) && !is.null(cox_off_glm8)) {
    mm_new <- build_mm(newdata, x_cols8)
    mm_new <- align_mm(mm_new, art8$mm_cols)
    newdata$lp_glm8 <- as.numeric(predict(glmnet8, newx = mm_new, type = "link", s = lam8))
    return(risk_at_tau(cox_off_glm8, newdata, tau8))

  } else if (identical(champion, "XGB(Cox)") && !is.null(xgb8) && !is.null(cox_off_xgb8)) {
    mm_new <- build_mm(newdata, x_cols8)
    mm_new <- align_mm(mm_new, art8$mm_cols)
    lp <- as.numeric(predict(xgb8, xgboost::xgb.DMatrix(mm_new), outputmargin = TRUE))
    newdata$lp_xgb8 <- lp
    return(risk_at_tau(cox_off_xgb8, newdata, tau8))

  } else if (identical(champion, "CatBoost(Cox)") && !is.null(cat8) && !is.null(cox_off_cat8)) {
    pool_new <- catboost::catboost.load_pool(newdata[, x_cols8, drop = FALSE])
    newdata$lp_cat8 <- as.numeric(catboost::catboost.predict(cat8, pool_new, prediction_type = "RawFormulaVal"))
    return(risk_at_tau(cox_off_cat8, newdata, tau8))

  } else if (identical(champion, "AFT") && !is.null(aft8)) {
    return(risk_at_tau(aft8, newdata, tau8))
  } else {
    return(risk_at_tau(fit_cox8, newdata, tau8))
  }
}

if (HAS_IML && length(top_features) > 0) {
  X_only <- d_all[, x_cols8, drop = FALSE]
  pred_fun <- function(dat) as.numeric(predict_risk_champion(cbind(d_all[, c("T","D")][FALSE, , drop=FALSE], dat)))
  predictor <- iml::Predictor$new(model = pred_fun, data = X_only, y = d_all$D)  # y só para assinatura

  for (v in top_features) {
    # ALE
    ale <- try(iml::FeatureEffect$new(predictor, feature = v, method = "ale"), silent = TRUE)
    if (!inherits(ale, "try-error")) {
      p <- ale$plot() + ggtitle(sprintf("ALE — %s (campeã: %s)", v, champion))
      try(ggsave(file.path(PHASE8_DIR, sprintf("ale_%s.png", gsub("[^A-Za-z0-9_]+","_", v))), p, width = 7, height = 5, dpi = 150), silent = TRUE)
    }
    # PDP+ICE para 2–3 chaves
    if (which(top_features == v)[1] <= 3) {
      pdp <- try(iml::FeatureEffect$new(predictor, feature = v, method = "pdp+ice", grid.size = 25), silent = TRUE)
      if (!inherits(pdp, "try-error")) {
        p <- pdp$plot() + ggtitle(sprintf("PDP + ICE — %s (campeã: %s)", v, champion))
        try(ggsave(file.path(PHASE8_DIR, sprintf("pdp_ice_%s.png", gsub("[^A-Za-z0-9_]+","_", v))), p, width = 7, height = 5, dpi = 150), silent = TRUE)
      }
    }
  }
} else if (!HAS_IML) {
  message("[phase8][note] Pacote 'iml' indisponível — pulando ALE/PDP/ICE.")
}

# --- 8.7 KM por decis de risco (campeã) + log-rank ---
risk_all <- try(predict_risk_champion(d_all), silent = TRUE)
if (!inherits(risk_all, "try-error") && any(is.finite(risk_all))) {

  # Novo: decil ROBUSTO por ranking (nunca usa 'breaks' do cut)
  ok  <- is.finite(risk_all)
  dec <- rep(NA_integer_, length(risk_all))
  rnk <- rank(risk_all[ok], ties.method = "average")
  dec[ok] <- pmin(pmax(ceiling(10 * rnk / max(rnk)), 1L), 10L)

  d_all$decile <- factor(paste0("D", dec), levels = paste0("D", 1:10))

  sf <- survival::survfit(Surv(T, D) ~ decile, data = d_all)
ss <- summary(sf)

# Constrói o vetor de rótulos de mesmo comprimento de ss$time
dec_labels <- NULL
if (!is.null(ss$strata) && length(ss$strata)) {
  # Caso com múltiplos estratos: repetir cada nome pelo nº de linhas daquele estrato
  dec_labels <- rep(gsub("^decile=", "", names(ss$strata)), ss$strata)
} else {
  # Caso com 1 único estrato (summary não preenche $strata)
  lab <- as.character(stats::na.omit(d_all$decile)[1])
  if (!length(lab) || is.na(lab)) lab <- "D?"
  dec_labels <- rep(lab, length(ss$time))
}

km_df <- data.frame(
  time   = ss$time,
  surv   = ss$surv,
  decile = dec_labels,
  stringsAsFactors = FALSE
)
km_df$decile <- factor(km_df$decile, levels = paste0("D", 1:10))

utils::write.csv(km_df, file.path(PHASE8_DIR, "km_by_decile.csv"), row.names = FALSE)

  # Log-rank
  lr <- try(survival::survdiff(Surv(T, D) ~ decile, data = d_all), silent = TRUE)
  p_lr <- NA_real_
  if (!inherits(lr, "try-error")) {
    chisq <- unname(lr$chisq); df_lr <- max(1, length(lr$n) - 1)
    p_lr <- stats::pchisq(chisq, df = df_lr, lower.tail = FALSE)
    writeLines(sprintf("Log-rank p = %.3g (df=%d)", p_lr, df_lr),
               con = file.path(PHASE8_DIR, "km_logrank_p.txt"))
  }

  p_km <- ggplot(km_df, aes(x = time, y = surv, color = decile)) +
    geom_step() +
    labs(title = sprintf("KM por decis de risco (campeã: %s) — log-rank p=%.3g", champion, p_lr),
         x = "Tempo", y = "S(t)") +
    theme_minimal(base_size = 12)
  try(ggsave(file.path(PHASE8_DIR, "km_by_decile.png"), p_km, width = 8, height = 5, dpi = 150), silent = TRUE)
} else {
  message("[phase8][warn] Não foi possível obter risco@tau* da campeã — KM por decis não gerado.")
}

message("[phase8] Artefatos salvos em: ", PHASE8_DIR)
log_runtime("phase8", NA, "complete", as.numeric(difftime(Sys.time(), t_phase8, units = "secs")))

# ================================================================
# Phase 9 — Visualization system (contextualized & refined)
# - Gera figuras com barras (média±DP), curvas com faixas de SD,
#   calibração @ tau*, e decision curves
# - Salva tudo em outputs/processed/figs/
# ================================================================

t_phase9 <- Sys.time()
PHASE9_DIR <- file.path(OUT_DIR, "figs")
if (!dir.exists(PHASE9_DIR)) dir.create(PHASE9_DIR, recursive = TRUE, showWarnings = FALSE)

# Dependências leves (instala se faltar)
for (p in c("ggplot2","dplyr","tidyr","scales","viridisLite")) {
  if (!requireNamespace(p, quietly = TRUE)) try(utils::install.packages(p, quiet = TRUE), silent = TRUE)
}
suppressPackageStartupMessages({ library(ggplot2); library(dplyr); library(tidyr); library(scales) })

pal_for <- function(keys) {
  cols <- viridisLite::viridis(max(1L, length(keys)))
  names(cols) <- keys
  cols
}

footer_caption <- function() {
  n  <- nrow(df)
  ev <- sum(df$D == 1, na.rm = TRUE)
  rate <- if (n > 0) ev / n else NA_real_
  tau_mean <- tryCatch({
    obj <- readRDS(file.path(OUT_DIR, "phase4_times.rds"))
    mean(as.numeric(obj$tau_star_by_fold), na.rm = TRUE)
  }, error = function(e) NA_real_)
  pv <- function(p) ifelse(is.finite(p), percent(p, accuracy = 0.1), "NA")
  pkgv <- function(p) if (requireNamespace(p, quietly = TRUE)) as.character(utils::packageVersion(p)) else "–"
  paste0("n=", n, " | eventos=", ev, " (", pv(rate), ") | censura≈", pv(1-rate),
         " | τ* (média das dobras)=", ifelse(is.finite(tau_mean), signif(tau_mean, 5), "NA"),
         "\nPackages: survival ", pkgv("survival"), "; glmnet ", pkgv("glmnet"),
         "; flexsurv ", pkgv("flexsurv"), "; xgboost ", pkgv("xgboost"),
         "; catboost ", pkgv("catboost"), "; riskRegression ", pkgv("riskRegression"))
}

# ------------------ 9.1 Ranking bars (média±DP) ------------------
per_fold_csv <- file.path(OUT_DIR, "per_fold_metrics.csv")
if (file.exists(per_fold_csv)) {
  met <- utils::read.csv(per_fold_csv, check.names = FALSE)
  long <- met |>
    dplyr::select(fold, model, C_Harrell, C_Uno_tau, IBS, Brier_at_tau, AUC_at_tau) |>
    tidyr::pivot_longer(-c(fold, model), names_to = "metric", values_to = "value")

  # rótulos + ordem
  metric_lab <- c(
    C_Harrell    = "Harrell's C (↑)",
    C_Uno_tau    = "Uno's C @ τ* (↑)",
    IBS          = "IBS (↓)",
    Brier_at_tau = "Brier @ τ* (↓)",
    AUC_at_tau   = "AUC @ τ* (↑)"
  )
  long$metric <- factor(long$metric, levels = names(metric_lab), labels = unname(metric_lab))

  sum_df <- long |>
    dplyr::group_by(model, metric) |>
    dplyr::summarize(mean = mean(value, na.rm = TRUE),
                     sd   = stats::sd(value, na.rm = TRUE),
                     .groups = "drop")

  # ordenar dentro de cada métrica
  sum_df <- sum_df |>
    dplyr::filter(is.finite(mean)) |>  # <<--- pipe que faltava
    dplyr::group_by(metric) |>
    dplyr::arrange(dplyr::desc(mean), .by_group = TRUE) |>
    dplyr::mutate(model_ord = factor(model, levels = unique(model))) |>
    dplyr::ungroup()

  pal <- pal_for(unique(sum_df$model))
  p_bar <- ggplot(sum_df, aes(x = model_ord, y = mean, fill = model)) +
    geom_col(width = 0.7) +
    geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = 0.2) +
    coord_flip() +
    facet_wrap(~ metric, ncol = 1, scales = "free_x") +
    scale_fill_manual(values = pal, guide = "none") +
    labs(title = "Comparativo de modelos — média±DP (5-fold CV)",
         x = NULL, y = NULL, caption = footer_caption()) +
    theme_minimal(base_size = 12) +
    theme(strip.text = element_text(face = "bold"))

  ggsave(file.path(PHASE9_DIR, "phase9_metric_bars.png"), p_bar, width = 9, height = 10, dpi = 150)
} else {
  message("[phase9][warn] per_fold_metrics.csv não encontrado — barras não geradas.")
}

# Descobre campeã (C_Uno@τ* > C_Harrell)
champion <- "CoxPH"
ms_path <- file.path(OUT_DIR, "metrics_summary.csv")
if (file.exists(ms_path)) {
  ms <- try(utils::read.csv(ms_path), silent = TRUE)
  if (!inherits(ms, "try-error") && nrow(ms) > 0) {
    if ("C_Uno_tau_mean" %in% names(ms) && any(is.finite(ms$C_Uno_tau_mean))) {
      champion <- as.character(ms$model[which.max(ms$C_Uno_tau_mean)])
    } else if ("C_Harrell_mean" %in% names(ms) && any(is.finite(ms$C_Harrell_mean))) {
      champion <- as.character(ms$model[which.max(ms$C_Harrell_mean)])
    }
  }
}
tau_mean <- tryCatch({
  obj <- readRDS(file.path(OUT_DIR, "phase4_times.rds"))
  mean(as.numeric(obj$tau_star_by_fold), na.rm = TRUE)
}, error = function(e) NA_real_)

# ------------------ 9.2 AUC(t) & Brier(t) com faixas ------------------
PHASE5_DIR <- file.path(OUT_DIR, "phase5_models")
auc_files   <- list.files(PHASE5_DIR, pattern = "^auc_curve_fold_\\d+\\.csv$", full.names = TRUE)
brier_files <- list.files(PHASE5_DIR, pattern = "^brier_curve_fold_\\d+\\.csv$", full.names = TRUE)

mk_curve <- function(files, value_col) {
  if (!length(files)) return(NULL)
  x <- lapply(files, function(f) utils::read.csv(f, check.names = FALSE))
  df <- dplyr::bind_rows(x)
  nm <- if (value_col == "AUC") "AUC" else "Brier"
  stopifnot(nm %in% names(df), "times" %in% names(df), "model" %in% names(df))
  df |>
    dplyr::group_by(model, times) |>
    dplyr::summarize(
      n_obs = sum(is.finite(.data[[nm]])),
      mean  = mean(.data[[nm]], na.rm = TRUE),
      sd    = ifelse(n_obs > 1, stats::sd(.data[[nm]], na.rm = TRUE), 0),
      .groups = "drop"
    ) |>
    dplyr::filter(is.finite(mean))
}

auc_s   <- mk_curve(auc_files,   "AUC")
brier_s <- mk_curve(brier_files, "Brier")

if (!is.null(auc_s) && nrow(auc_s)) {
  pal <- pal_for(unique(auc_s$model))
  auc_s <- dplyr::filter(auc_s, is.finite(mean), is.finite(sd))

  p_auc <- ggplot(auc_s, aes(x = times, y = mean, color = model, fill = model, group = model)) +
    geom_ribbon(aes(ymin = pmax(0, mean - sd), ymax = pmin(1, mean + sd)), alpha = 0.14, color = NA) +
    geom_line(aes(linewidth = model == champion)) +
    scale_linewidth_manual(values = c("TRUE" = 1.3, "FALSE" = 0.9), guide = "none") +
    scale_color_manual(values = pal) + scale_fill_manual(values = pal) +
    geom_vline(xintercept = tau_mean, linetype = 2) +
    coord_cartesian(ylim = c(0.5, 1)) +
    labs(title = "AUC(t) com faixas de DP",
         x = "Tempo", y = "AUC(t)",
         subtitle = paste0("Campeã destacada: ", champion, " | τ* marcado"),
         caption = footer_caption()) +
    theme_minimal(base_size = 12)
  ggsave(file.path(PHASE9_DIR, "phase9_auc_curves.png"), p_auc, width = 8, height = 5, dpi = 150)
} else {
  message("[phase9][warn] Curvas AUC não disponíveis — pular.")
}

if (!is.null(brier_s) && nrow(brier_s)) {
  pal <- pal_for(unique(brier_s$model))
  brier_s <- dplyr::filter(brier_s, is.finite(mean), is.finite(sd))

  p_br <- ggplot(brier_s, aes(x = times, y = mean, color = model, fill = model, group = model)) +
    geom_ribbon(aes(ymin = pmax(0, mean - sd), ymax = pmin(1, mean + sd)), alpha = 0.14, color = NA) +
    geom_line(aes(linewidth = model == champion)) +
    scale_linewidth_manual(values = c("TRUE" = 1.3, "FALSE" = 0.9), guide = "none") +
    scale_color_manual(values = pal) + scale_fill_manual(values = pal) +
    geom_vline(xintercept = tau_mean, linetype = 2) +
    coord_cartesian(ylim = c(0, 0.5)) +
    labs(title = "Brier(t) com faixas de DP",
         x = "Tempo", y = "Brier(t)",
         subtitle = paste0("Campeã destacada: ", champion, " | τ* marcado"),
         caption = footer_caption()) +
    theme_minimal(base_size = 12)
  ggsave(file.path(PHASE9_DIR, "phase9_brier_curves.png"), p_br, width = 8, height = 5, dpi = 150)
} else {
  message("[phase9][warn] Curvas Brier não disponíveis — pular.")
}

# ------------------ 9.3 Calibração @ τ* (por modelo) ------------------
PHASE6_DIR <- file.path(OUT_DIR, "phase6_calibration")
calib_bins_csv    <- file.path(PHASE6_DIR, "calibration_bins.csv")
calib_summary_csv <- file.path(PHASE6_DIR, "calibration_summary_by_model.csv")

if (file.exists(calib_bins_csv)) {
  cb <- utils::read.csv(calib_bins_csv, check.names = FALSE)
  # cb: bin, n, pred_mean, pred_median, obs_km, obs_iso (opcional), fold, model, tau
  # Isotônica global por modelo com base nas médias de bin (todas as dobras)
  iso_by_model <- cb |>
    dplyr::filter(is.finite(pred_mean), is.finite(obs_km)) |>
    dplyr::arrange(model, pred_mean) |>
    dplyr::group_by(model) |>
    dplyr::group_modify(~{
      if (nrow(.x) < 2) return(data.frame(pred = numeric(0), iso = numeric(0)))
      o <- order(.x$pred_mean)
      iz <- try(stats::isoreg(.x$pred_mean[o], .x$obs_km[o]), silent = TRUE)
      if (inherits(iz, "try-error") || is.null(iz$yf)) {
        data.frame(pred = numeric(0), iso = numeric(0))
      } else {
        data.frame(pred = .x$pred_mean[o], iso = as.numeric(iz$yf))
      }
    }) |>
    dplyr::ungroup()

  # Anota slope/intercept médios
  cal_annot <- if (file.exists(calib_summary_csv)) try(utils::read.csv(calib_summary_csv), silent = TRUE) else NULL
  if (inherits(cal_annot, "try-error")) cal_annot <- NULL

  pal <- pal_for(unique(cb$model))
  p_cal <- ggplot(cb, aes(x = pred_mean, y = obs_km)) +
    geom_abline(slope = 1, intercept = 0, linetype = 2) +
    geom_point(aes(size = n, color = model), alpha = 0.55) +
    geom_step(data = iso_by_model, aes(x = pred, y = iso, color = model), linewidth = 1) +
    facet_wrap(~ model, scales = "fixed", ncol = 2) +     # <-- "free", arrumado
    scale_color_manual(values = pal, guide = "none") +
    scale_size(range = c(1.2, 4), guide = "none") +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +         # coord_equal agr funcionando 
    labs(title = "Calibração @ τ* (observado vs predito em 10 bins)",
        x = "Risco predito (média do bin)", y = "Risco observado (KM @ τ*)",
        caption = footer_caption()) +
    theme_minimal(base_size = 12)

  # Pequenas anotações de slope/intercept, se disponível
  if (!is.null(cal_annot) && nrow(cal_annot)) {
    cal_annot$model <- as.character(cal_annot$model)
    # Anexa a tabela como arquivo auxiliar
    utils::write.csv(cal_annot, file.path(PHASE9_DIR, "phase9_calibration_summary_by_model.csv"), row.names = FALSE)
  }

  ggsave(file.path(PHASE9_DIR, "phase9_calibration_tau.png"), p_cal, width = 9, height = 8, dpi = 150)
} else {
  message("[phase9][warn] calibration_bins.csv não encontrado — calibração não gerada.")
}

# ------------------ 9.4 Decision curve @ τ* (net benefit) ------------------
# Implementação com IPCW simples para censura; média dos folds; trata 'treat-all'/'treat-none'
if (!requireNamespace("prodlim", quietly = TRUE)) {
  try(utils::install.packages("prodlim", quiet = TRUE), silent = TRUE)
}
if (requireNamespace("prodlim", quietly = TRUE)) {
  PRED_DIR <- file.path(OUT_DIR, "predictions")
  pred_files <- list.files(PRED_DIR, pattern = "^fold_\\d+_predictions_tau\\.csv$", full.names = TRUE)
  times_obj  <- try(readRDS(file.path(OUT_DIR, "phase4_times.rds")), silent = TRUE)

  if (length(pred_files) && !inherits(times_obj, "try-error")) {
    tau_by_fold <- as.numeric(times_obj$tau_star_by_fold)

    compute_nb_fold <- function(df_pred, tau, model_cols, thr_grid) {
    # df_pred: precisa conter colunas T, D e risk_*
    # 1) Ajusta G(t) = P(C >= t) via prodlim 
    df_pred$C <- 1L - as.integer(df_pred$D)
    fitG <- prodlim::prodlim(survival::Surv(T, C) ~ 1, data = df_pred)

    # tempos truncados em tau
    tt   <- pmin(df_pred$T, tau)
    Ghat <- as.numeric(predict(fitG, times = tt, type = "surv"))

    w    <- 1 / pmax(Ghat, 1e-6)

    # indicador de evento até tau
    event_tau <- as.integer(df_pred$T <= tau & df_pred$D == 1)
    N <- nrow(df_pred)

    # prevalência (KM @ tau) estimada por IPCW
    prev_hat <- sum(w * event_tau) / N

    # Net benefit por limiar (p = threshold)
    calc_nb <- function(p) sapply(thr_grid, function(t) {
        sel <- p >= t
        TP <- sum(w[sel] * event_tau[sel]) / N
        FP <- sum(w[sel] * (1 - event_tau[sel])) / N
        TP - FP * t/(1 - t)
    })

    nb_list <- lapply(model_cols, function(mc) calc_nb(df_pred[[mc]]))
    names(nb_list) <- model_cols

    nb_df <- do.call(rbind, nb_list)
    nb_df <- as.data.frame(nb_df)
    nb_df$model <- rownames(nb_df)
    nb_df <- tidyr::pivot_longer(nb_df, -model, names_to = "k", values_to = "NB")
    nb_df$threshold <- thr_grid[as.integer(gsub("^V", "", nb_df$k))]
    nb_df$k <- NULL

    # Referências
    nb_all  <- data.frame(model = "treat-all",  threshold = thr_grid,
                            NB = prev_hat - (1 - prev_hat) * thr_grid/(1 - thr_grid))
    nb_none <- data.frame(model = "treat-none", threshold = thr_grid, NB = 0)

    rbind(nb_df, nb_all, nb_none)
    }

    # grid de limiares: 1% a 50% (ajuste se quiser)
    thr_grid <- seq(0.01, 0.50, by = 0.01)

    #saneamento do tau
    sanitize_tau_for_timeROC <- function(Tvec, tau) {
    Tvec <- Tvec[is.finite(Tvec)]
    if (!length(Tvec)) return(NA_real_)
    tmax <- max(Tvec); tmin <- min(Tvec)
    tau_use <- min(tau, tmax - 1e-6)   # timeROC exige times < max(T)
    if (!is.finite(tau_use) || tau_use <= 0) tau_use <- (tmin + tmax)/2
    tau_use
    }

    all_nb <- list()
    for (f in pred_files) {
      k <- as.integer(gsub(".*fold_(\\d+)_predictions_tau\\.csv$", "\\1", f))
      dfp <- utils::read.csv(f, check.names = FALSE)
      if (!all(c("T","D") %in% names(dfp))) next
      model_cols <- grep("^risk_", names(dfp), value = TRUE)
      if (!length(model_cols)) next
      nbk <- compute_nb_fold(dfp, tau = tau_by_fold[k], model_cols = model_cols, thr_grid = thr_grid)
      nbk$fold <- k
      # normaliza rótulos
      nbk$model <- gsub("^risk_", "", nbk$model)
      nbk$model <- gsub("_", " ", nbk$model, fixed = TRUE)
      nbk$model <- gsub("XGB Cox", "XGB(Cox)", nbk$model)
      nbk$model <- gsub("CatBoost Cox", "CatBoost(Cox)", nbk$model)
      all_nb[[length(all_nb)+1]] <- nbk
    }

    if (length(all_nb)) {
      nb_all <- dplyr::bind_rows(all_nb)
      nb_sum <- nb_all |>
        dplyr::group_by(model, threshold) |>
        dplyr::summarize(NB_mean = mean(NB, na.rm = TRUE),
                         NB_sd   = stats::sd(NB, na.rm = TRUE),
                         .groups = "drop")

      pal <- pal_for(setdiff(unique(nb_sum$model), c("treat-all","treat-none")))
      # escala manual incluindo linhas cinza para referências
      col_all  <- "#777777"; col_none <- "#999999"

      p_nb <- ggplot() +
        # ribbons só para modelos (não para referências)
        geom_ribbon(data = subset(nb_sum, !(model %in% c("treat-all","treat-none"))),
                    aes(x = threshold, ymin = NB_mean - NB_sd, ymax = NB_mean + NB_sd, fill = model),
                    alpha = 0.12, color = NA) +
        geom_line(data = subset(nb_sum, model == "treat-all"),
                  aes(x = threshold, y = NB_mean), color = col_all, linewidth = 0.8, linetype = 2) +
        geom_line(data = subset(nb_sum, model == "treat-none"),
                  aes(x = threshold, y = NB_mean), color = col_none, linewidth = 0.8, linetype = 3) +
        geom_line(data = subset(nb_sum, !(model %in% c("treat-all","treat-none"))),
                  aes(x = threshold, y = NB_mean, color = model, linewidth = model == champion)) +
        scale_color_manual(values = pal, guide = "none") +
        scale_fill_manual(values = pal, guide = "none") +
        scale_linewidth_manual(values = c("TRUE" = 1.4, "FALSE" = 0.9), guide = "none") +
        # Faixa de negócio (exemplo: 5%–20%)
        annotate("rect", xmin = 0.05, xmax = 0.20, ymin = -Inf, ymax = Inf, alpha = 0.05) +
        labs(title = "Decision curve @ τ* (net benefit)",
             subtitle = paste0("Campeã destacada: ", champion, " | referências: tratar todos / tratar nenhum"),
             x = "Limiar de risco (probabilidade de evento até τ*)",
             y = "Net benefit",
             caption = footer_caption()) +
        theme_minimal(base_size = 12)
      ggsave(file.path(PHASE9_DIR, "phase9_decision_curve_tau.png"), p_nb, width = 8, height = 5, dpi = 150)
    } else {
      message("[phase9][warn] Não foi possível calcular decision curves — sem predições/risco.")
    }
  } else {
    message("[phase9][warn] Artefatos de predição ou tempos da Phase 4 ausentes — decision curves não geradas.")
  }
} else {
  message("[phase9][warn] 'prodlim' indisponível — decision curves puladas.")
}

message("[phase9] Figuras salvas em: ", PHASE9_DIR)
log_runtime("phase9", NA, "complete",
            as.numeric(difftime(Sys.time(), t_phase9, units = "secs")))

# ================================================================
# Phase 10 — Threshold & policy selection (at τ*)
# - Lê predições @ τ* por dobra
# - Constrói grade de limiares via timeROC (Se, Sp, PPV, NPV)
# - Seleciona 3 políticas: Youden-J, Custo-sensível, Capacidade
# - Mede Lift e Net Benefit no limiar escolhido
# ================================================================

t_phase10 <- Sys.time()
PHASE10_DIR <- file.path(OUT_DIR, "phase10_thresholds")
if (!dir.exists(PHASE10_DIR)) dir.create(PHASE10_DIR, recursive = TRUE, showWarnings = FALSE)

# Parâmetros (podem ser sobrepostos via variáveis de ambiente)
COST_FP <- as.numeric(Sys.getenv("PHASE10_COST_FP", "1"))
COST_FN <- as.numeric(Sys.getenv("PHASE10_COST_FN", "5"))
CAP_PCT <- as.numeric(Sys.getenv("PHASE10_CAP_PCT", "0.10"))
if (!is.finite(CAP_PCT) || CAP_PCT <= 0 || CAP_PCT >= 1) CAP_PCT <- 0.10

# Funções utilitárias
pretty_model <- function(risk_col) {
  nm <- sub("^risk_", "", risk_col)
  nm <- gsub("_", " ", nm)
  nm <- gsub("XGB Cox", "XGB(Cox)", nm)
  nm <- gsub("CatBoost Cox", "CatBoost(Cox)", nm)
  nm
}

safe_prev_at_tau <- function(T, D, tau) {
  fitS <- try(prodlim::prodlim(survival::Surv(T, D) ~ 1), silent = TRUE)
  if (inherits(fitS, "try-error")) return(NA_real_)
  s <- try(predict(fitS, times = tau, type = "surv"), silent = TRUE)
  if (inherits(s, "try-error") || length(s) < 1 || !is.finite(s[1])) return(NA_real_)
  as.numeric(1 - s[1])
}


select_by_policies <- function(thr_df, prevalence, cap_pct, cost_fp, cost_fn) {
  # Youden-J
  yt <- thr_df[which.max(thr_df$YoudenJ), , drop = FALSE]

  # Custo esperado = FN*π*(1-Se) + FP*(1-π)*(1-Sp)
  cost <- cost_fn * prevalence * (1 - thr_df$Se) + cost_fp * (1 - prevalence) * (1 - thr_df$Sp)
  ct <- thr_df[which.min(cost), , drop = FALSE]
  ct$ExpectedCost <- min(cost, na.rm = TRUE)

  # Capacidade (top-k%): threshold ~ quantil 1-k da distribuição
  # Usar a coluna PredPosShare se existir, senão aproximar por fração de pontos acima do threshold
  # Aqui, o thr_df já deve ter "PredPosShare"; se não tiver, cria
  idx <- which.min(abs(thr_df$PredPosShare - cap_pct))
  kt <- thr_df[idx, , drop = FALSE]

  # Net Benefit dos escolhidos já está no thr_df$NB (se fornecido); caso contrário, re-calcular no upstream
  list(youden = yt, cost = ct, capacity = kt)
}

# Carrega tau* por dobra
times_obj <- try(readRDS(file.path(OUT_DIR, "phase4_times.rds")), silent = TRUE)
if (inherits(times_obj, "try-error")) stop("[phase10] 'phase4_times.rds' não encontrado.")
tau_by_fold <- as.numeric(times_obj$tau_star_by_fold)

# Lista de arquivos de predição
PRED_DIR <- file.path(OUT_DIR, "predictions")
pred_files <- list.files(PRED_DIR, pattern = "^fold_\\d+_predictions_tau\\.csv$", full.names = TRUE)
if (!length(pred_files)) stop("[phase10] Sem predições @ τ* (rode Phase 5).")

selections_all <- list()

for (f in pred_files) {
  k <- as.integer(gsub(".*fold_(\\d+)_predictions_tau\\.csv$", "\\1", f))
  tau_k <- tau_by_fold[k]
  preds <- utils::read.csv(f, check.names = FALSE)
  if (!all(c("T","D") %in% names(preds))) next

  prevalence <- safe_prev_at_tau(preds$T, preds$D, tau_k)

  if (!is.finite(prevalence) || prevalence <= 0) {
    # Bug consertado
    prevalence <- mean(preds$D == 1 & preds$T <= tau_k, na.rm = TRUE)
    if (!is.finite(prevalence) || prevalence <= 0) prevalence <- 0.1 
    message(sprintf("[phase10][warn] Fold %d: Using fallback prevalence %.3f", k, prevalence))
  }

  risk_cols <- grep("^risk_", names(preds), value = TRUE)
  if (!length(risk_cols)) next
    for (rc in risk_cols) {
      r <- preds[[rc]]
      ok <- is.finite(preds$T) & is.finite(preds$D) & is.finite(r)
      if (sum(ok) < 20) next

      tau_use <- sanitize_tau_for_timeROC(preds$T[ok], tau_k)
      if (!is.finite(tau_use)) next

      # Tenta timeROC + Se/Sp/PPV/NPV
      tr <- try(timeROC::timeROC(
        T = preds$T[ok], delta = preds$D[ok], marker = r[ok],
        cause = 1, weighting = "marginal", times = tau_use, iid = FALSE
      ), silent = TRUE)

      ss <- try(timeROC::SeSpPPVNPV(
        T = preds$T[ok], delta = preds$D[ok], marker = r[ok],
        cause = 1, weighting = "marginal", times = tau_use
      ), silent = TRUE)

      thr_df <- NULL
      if (!(inherits(tr, "try-error") || inherits(ss, "try-error")) && length(ss$thresholds)) {
        thr_df <- data.frame(
          threshold = as.numeric(ss$thresholds),
          Se = as.numeric(ss$Se),
          Sp = as.numeric(ss$Sp),
          PPV = as.numeric(ss$PPV),
          NPV = as.numeric(ss$NPV),
          stringsAsFactors = FALSE
        )
      }

      # Fallback IPCW se timeROC falhar ou retornar vazio
      if (is.null(thr_df) || !nrow(thr_df)) {
        fb <- thresholds_ipcw_fallback(preds$T[ok], preds$D[ok], r[ok], tau_use, n_grid = 101)
        if (is.null(fb)) next
        thr_df <- fb$df
        prevalence <- fb$prevalence
      } else {
        prevalence <- safe_prev_at_tau(preds$T[ok], preds$D[ok], tau_use)
      }
      if (!is.finite(prevalence) || prevalence <= 0) {
        prevalence <- mean(preds$D[ok] == 1 & preds$T[ok] <= tau_use, na.rm = TRUE)
      }

      thr_df <- thr_df[is.finite(thr_df$Se) & is.finite(thr_df$Sp), , drop = FALSE]
      if (!nrow(thr_df)) next

      thr_df$YoudenJ      <- thr_df$Se + thr_df$Sp - 1
      thr_df$PredPosShare <- vapply(thr_df$threshold, function(t) mean(r[ok] >= t, na.rm = TRUE), numeric(1))
            thr_df$NB           <- with(thr_df, prevalence * Se - (1 - prevalence) * (1 - Sp) * threshold/(1 - threshold))

      # annotate and persist per-fold threshold grid
      thr_df$model <- pretty_model(rc)
      thr_df$fold  <- k
      utils::write.csv(
        thr_df,
        file.path(PHASE10_DIR, sprintf("thresholds_grid__fold%d__%s.csv", k, gsub("[^A-Za-z0-9]+","_", thr_df$model[1]))),
        row.names = FALSE
      )

      # pick policies
      sel <- select_by_policies(thr_df, prevalence, CAP_PCT, COST_FP, COST_FN)
      sel_df <- dplyr::bind_rows(
        cbind(policy = "YoudenJ",          sel$youden),
        cbind(policy = "CostSensitive",    sel$cost),
        cbind(policy = sprintf("Capacity_%d%%", round(100*CAP_PCT)), sel$capacity)
      )
      sel_df$prevalence <- prevalence
      sel_df$fold       <- k
      sel_df$model      <- pretty_model(rc)

      selections_all[[length(selections_all) + 1]] <- sel_df
    } # end for rc
} # end for f

# Consolidate selections across folds
if (length(selections_all)) {
  sel_df <- dplyr::bind_rows(selections_all)
  utils::write.csv(sel_df, file.path(PHASE10_DIR, "selections_by_fold.csv"), row.names = FALSE)

  sel_sum <- sel_df %>%
    dplyr::group_by(model, policy) %>%
    dplyr::summarize(
      threshold_mean   = mean(threshold, na.rm = TRUE),
      threshold_sd     = stats::sd(threshold, na.rm = TRUE),
      Se_mean          = mean(Se, na.rm = TRUE),
      Sp_mean          = mean(Sp, na.rm = TRUE),
      PPV_mean         = mean(PPV, na.rm = TRUE),
      NPV_mean         = mean(NPV, na.rm = TRUE),
      NB_mean          = mean(NB, na.rm = TRUE),
      lift_mean        = mean(PPV / prevalence, na.rm = TRUE),
      prevalence_mean  = mean(prevalence, na.rm = TRUE),
      .groups = "drop"
    )
  utils::write.csv(sel_sum, file.path(PHASE10_DIR, "selections_summary.csv"), row.names = FALSE)

  message("[phase10] Seleções salvas em: ", PHASE10_DIR)
} else {
  message("[phase10][warn] Nenhuma seleção gerada — verifique se há risk_* e se timeROC executou nas dobras.")
}

log_runtime("phase10", NA, "complete", as.numeric(difftime(Sys.time(), t_phase10, units = "secs")))

  # Consolida seleções
if (length(selections_all)) {
  sel_df <- dplyr::bind_rows(selections_all)
  # Seleções por dobra
  utils::write.csv(sel_df, file.path(PHASE10_DIR, "selections_by_fold.csv"), row.names = FALSE)

  # Resumo por modelo/política
  sel_sum <- sel_df %>%
    dplyr::group_by(model, policy) %>%
    dplyr::summarize(
      threshold_mean = mean(threshold, na.rm = TRUE),
      threshold_sd   = stats::sd(threshold, na.rm = TRUE),
      Se_mean  = mean(Se, na.rm = TRUE),
      Sp_mean  = mean(Sp, na.rm = TRUE),
      PPV_mean = mean(PPV, na.rm = TRUE),
      NPV_mean = mean(NPV, na.rm = TRUE),
      NB_mean  = mean(NB, na.rm = TRUE),
      lift_mean = mean(PPV / prevalence, na.rm = TRUE),
      prevalence_mean = mean(prevalence, na.rm = TRUE),
      .groups = "drop"
    )
  utils::write.csv(sel_sum, file.path(PHASE10_DIR, "selections_summary.csv"), row.names = FALSE)

  message("[phase10] Seleções salvas em: ", PHASE10_DIR)
} else {
  message("[phase10][warn] Nenhuma seleção gerada — verifique se há risk_* e se timeROC executou nas dobras.")
}

log_runtime("phase10", NA, "complete",
            as.numeric(difftime(Sys.time(), t_phase10, units = "secs")))
