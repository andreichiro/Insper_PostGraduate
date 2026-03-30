# Projeto final André e Danillo
# Mercado de vagas IA/ML

suppressPackageStartupMessages({
  ensure_pkgs <- function(pkgs) {
    repos <- getOption("repos")
    if (is.null(repos) || length(repos) == 0L || is.na(repos["CRAN"]) || repos["CRAN"] == "@CRAN@") {
      options(repos = c(CRAN = "https://cloud.r-project.org"))
    }
    need <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
    if (length(need) > 0L) {
      install.packages(need, dependencies = c("Depends", "Imports", "LinkingTo"))
    }
    invisible(lapply(pkgs, function(p) library(p, character.only = TRUE)))
  }

  required_pkgs <- c(
    "tidyverse", "lubridate", "stringr", "tidymodels",
    "text2vec", "Matrix", "irlba", "cluster",
    "kernlab", "tidytext", "topicmodels",
    "glmnet", "ranger", "xgboost",
    "hardhat", "scales", "ggrepel",
    "tm" # <- (5) tidytext::cast_dtm() frequentemente depende de tm
  )
  ensure_pkgs(required_pkgs)
  tidymodels::tidymodels_prefer()
})

if (capabilities("cairo")) options(bitmapType = "cairo")

# Funções HTML

html_begin <- function(path, title = "Análise de Vagas em IA/ML") {
  if (file.exists(path)) file.remove(path)
  con <- file(path, open = "wt", encoding = "UTF-8")
  writeLines(paste0(
    "<!DOCTYPE html><html><head><meta charset='utf-8'><title>", title, "</title>",
    "<style>
      :root{--fg:#222;--muted:#555;--line:#e6e6e6;--bg:#fff;--note:#f9fbff;--note-b:#9bbcff;--warn:#fff9f0;--warn-b:#ffb155}
      *{box-sizing:border-box}
      body{font-family:-apple-system,system-ui,Segoe UI,Roboto,Arial,sans-serif;margin:24px;color:var(--fg);background:var(--bg)}
      h1,h2,h3{margin:1.0em 0 .5em}
      p{margin:.6em 0;line-height:1.45}
      table{border-collapse:collapse;width:100%;margin:8px 0 16px}
      th,td{border:1px solid var(--line);padding:6px 8px}
      th{text-align:left;background:#f7f7f7;position:sticky;top:0}
      td.num{text-align:right;font-variant-numeric:tabular-nums}
      tbody tr:nth-child(even){background:#fbfbfb}
      .small{color:var(--muted);font-size:.9em}
      .note{background:var(--note);border-left:4px solid var(--note-b);padding:.6em .8em;margin:.6em 0}
      .warn{background:var(--warn);border-left:4px solid var(--warn-b);padding:.6em .8em;margin:.6em 0}
      img{max-width:100%;height:auto;border:1px solid #eee;box-shadow:0 1px 2px rgba(0,0,0,.05)}
      ul{margin:.2em 0 .8em 1.2em}
      .pill{display:inline-block;background:#eef;border:1px solid #dde;padding:2px 8px;border-radius:999px;margin:2px}
      .kpi{display:inline-block;margin:.2em .6em .2em 0;padding:.2em .6em;background:#f5f7ff;border:1px solid #dfe6ff;border-radius:6px}
      .hr{height:1px;background:var(--line);margin:16px 0}
    </style></head><body>"
  ), con = con)
  close(con)
  sink(path, append = TRUE, split = FALSE)
}

html_end <- function(path) {
  cat("</body></html>")
  sink()
}

html_h1    <- function(txt) cat("<h1>", txt, "</h1>\n", sep = "")
html_h2    <- function(txt) cat("<h2>", txt, "</h2>\n", sep = "")
html_h3    <- function(txt) cat("<h3>", txt, "</h3>\n", sep = "")

# (1) html_p cola strings com paste0(collapse=""); então usar 1 string, ou incluir espaços,
# ou html_list/html_note. Aqui mantemos html_p como está, mas evitamos concatenar frases sem espaços.
html_p     <- function(...) { cat("<p>", paste0(..., collapse = ""), "</p>\n") }
html_note  <- function(...) { cat("<p class='note'>", paste0(..., collapse = ""), "</p>\n") }
html_warn  <- function(...) { cat("<p class='warn'>", paste0(..., collapse = ""), "</p>\n") }
html_list  <- function(v) { cat("<ul>", paste(sprintf("<li>%s</li>", v), collapse = ""), "</ul>\n") }

html_table <- function(x, caption = NULL, digits = 3) {
  df <- as.data.frame(x, stringsAsFactors = FALSE, check.names = FALSE)
  num <- vapply(df, is.numeric, TRUE)
  if (any(num)) {
    df[num] <- lapply(df[num], function(v) format(round(v, digits), trim = TRUE))
  }
  if (!is.null(caption)) cat("<h3>", caption, "</h3>\n", sep = "")
  cat("<table><thead><tr>",
      paste(sprintf("<th>%s</th>", colnames(df)), collapse = ""),
      "</tr></thead><tbody>\n", sep = "")
  apply(df, 1, function(row) {
    isnum <- suppressWarnings(!is.na(as.numeric(row)))
    cat("<tr>",
        paste(sprintf("<td%s>%s</td>",
                      ifelse(isnum, " class='num'", ""), row),
              collapse = ""),
        "</tr>\n")
  })
  cat("</tbody></table>\n")
}

save_gg <- function(plot, filename, width = 12, height = 7, dpi = 140, dir = getwd()) {
  ggplot2::ggsave(
    file.path(dir, filename),
    plot = plot,
    width = width,
    height = height,
    dpi = dpi,
    device = "png"
  )
}

# Funções auxiliares de dados/texto

# EDA: correlação/associação com salário 

cap_top_levels <- function(x, n = 12L, other_label = "Outros") {
  x <- as.character(x)
  x <- ifelse(is.na(x) | stringr::str_squish(x) == "", NA_character_, stringr::str_squish(x))
  tab <- sort(table(x, useNA = "no"), decreasing = TRUE)
  keep <- names(tab)[seq_len(min(n, length(tab)))]
  ifelse(is.na(x), NA_character_, ifelse(x %in% keep, x, other_label))
}

safe_cor_test <- function(x, y, method = "spearman") {
  ok <- !is.na(x) & !is.na(y)
  x <- x[ok]; y <- y[ok]

  if (length(x) < 10L) {
    return(tibble::tibble(estimate = NA_real_, p_value = NA_real_, n = length(x)))
  }
  if (sd(x) == 0 || sd(y) == 0) {
    return(tibble::tibble(estimate = NA_real_, p_value = NA_real_, n = length(x)))
  }

  ct <- suppressWarnings(
    cor.test(
      x, y,
      method = method,
      exact  = FALSE
    )
  )
  tibble::tibble(
    estimate = unname(ct$estimate[[1]]),
    p_value  = ct$p.value,
    n        = length(x)
  )
}

eta2_oneway <- function(y, g) {
  ok <- !is.na(y) & !is.na(g)
  y <- y[ok]; g <- as.factor(g[ok])

  if (length(y) < 20L || nlevels(g) < 2L) {
    return(tibble::tibble(eta2 = NA_real_, n = length(y), n_levels = nlevels(g)))
  }

  mu <- mean(y)
  ss_total <- sum((y - mu)^2)
  if (ss_total == 0) {
    return(tibble::tibble(eta2 = NA_real_, n = length(y), n_levels = nlevels(g)))
  }

  means <- tapply(y, g, mean)
  ns    <- tapply(y, g, length)
  ss_between <- sum(ns * (means - mu)^2)

  tibble::tibble(
    eta2     = ss_between / ss_total,
    n        = length(y),
    n_levels = nlevels(g)
  )
}

compute_salary_associations <- function(df_model_base, top_n_levels = 12L) {
  df_a <- df_model_base %>%
    dplyr::mutate(
      country = stringr::str_extract(location, "[A-Za-z ]+$"),
      posted_year     = lubridate::year(posted_date),
      posted_month    = lubridate::month(posted_date),
      posted_date_num = as.numeric(posted_date),
      n_skills = ifelse(skills_required == "" | is.na(skills_required), 0L,
                        stringr::str_count(skills_required, ",") + 1L),
      n_tools  = ifelse(tools_preferred == "" | is.na(tools_preferred), 0L,
                        stringr::str_count(tools_preferred, ",") + 1L),
      title_n_words = ifelse(job_title == "" | is.na(job_title), 0L,
                             stringr::str_count(stringr::str_squish(job_title), "\\S+")),
      text_chars = nchar(text_blob)
    )

  # (6) Evitar variáveis derivadas do próprio salário (leakage trivial)
  num_vars <- c(
    "posted_year", "posted_month", "posted_date_num",
    "n_skills", "n_tools", "title_n_words", "text_chars"
  )
  num_vars <- num_vars[num_vars %in% names(df_a)]

  num_tbl <- purrr::map_dfr(num_vars, function(v) {
    x <- df_a[[v]]
    pe <- safe_cor_test(x, df_a$salary_log, method = "pearson")
    sp <- safe_cor_test(x, df_a$salary_log, method = "spearman")
    tibble::tibble(
      predictor   = v,
      pearson     = pe$estimate,
      p_pearson   = pe$p_value,
      spearman    = sp$estimate,
      p_spearman  = sp$p_value,
      n           = sp$n
    )
  }) %>%
    dplyr::mutate(
      p_spearman_fdr = p.adjust(p_spearman, method = "BH"),
      abs_spearman   = abs(spearman)
    ) %>%
    dplyr::arrange(dplyr::desc(abs_spearman))

  df_a <- df_a %>%
    dplyr::mutate(
      industry_top   = cap_top_levels(industry,  n = top_n_levels),
      country_top    = cap_top_levels(country,   n = top_n_levels),
      job_title_top  = cap_top_levels(job_title, n = top_n_levels)
    )

  cat_defs <- list(
    experience_level = df_a$experience_level,
    employment_type  = df_a$employment_type,
    company_size     = df_a$company_size,
    industry_top     = df_a$industry_top,
    country_top      = df_a$country_top,
    job_title_top    = df_a$job_title_top
  )

  cat_tbl <- purrr::imap_dfr(cat_defs, function(g, nm) {
    eta <- eta2_oneway(df_a$salary_log, g)
    tibble::tibble(
      variable = nm,
      eta2     = eta$eta2,
      n        = eta$n,
      n_levels = eta$n_levels
    )
  }) %>%
    dplyr::arrange(dplyr::desc(eta2))

  exp_rank <- dplyr::case_when(
    stringr::str_detect(tolower(df_a$experience_level), "^entry|^jun") ~ 1,
    stringr::str_detect(tolower(df_a$experience_level), "^mid")        ~ 2,
    stringr::str_detect(tolower(df_a$experience_level), "^senior")     ~ 3,
    stringr::str_detect(tolower(df_a$experience_level), "^lead|^principal") ~ 4,
    stringr::str_detect(tolower(df_a$experience_level), "^exec|^director|^chief") ~ 5,
    TRUE ~ NA_real_
  )
  exp_rank_corr <- safe_cor_test(exp_rank, df_a$salary_log, method = "spearman") %>%
    dplyr::transmute(
      measure  = "experience_level (ordinal) × salary_log",
      spearman = estimate,
      p_value  = p_value,
      n        = n
    )

  list(
    numeric_corr   = num_tbl,
    cat_assoc_eta2 = cat_tbl,
    exp_rank_corr  = exp_rank_corr
  )
}

normalize_text_col <- function(x) {
  ifelse(
    is.na(x),
    NA_character_,
    stringr::str_squish(as.character(x))
  )
}

parse_salary_range_one <- function(s) {
  if (is.na(s) || s == "") return(c(NA_real_, NA_real_))
  s <- stringr::str_trim(as.character(s))
  s <- stringr::str_replace_all(s, "[,$]", "")
  s <- stringr::str_replace_all(s, "[^0-9\\-–]", "")

  m <- stringr::str_match(s, "^(\\d+)\\s*[-–]\\s*(\\d+)$")
  if (!is.na(m[1, 1])) {
    lo <- as.numeric(m[1, 2])
    hi <- as.numeric(m[1, 3])
    if (!is.na(lo) && !is.na(hi) && hi < lo) {
      tmp <- lo; lo <- hi; hi <- tmp
    }
    return(c(lo, hi))
  }
  m2 <- stringr::str_match(s, "^(\\d+)$")
  if (!is.na(m2[1, 1])) {
    v <- as.numeric(m2[1, 2])
    return(c(v, v))
  }
  c(NA_real_, NA_real_)
}

escape_regex <- function(x) {
  stringr::str_replace_all(
    x,
    "([\\^$.|?*+(){}\\[\\]])",
    "\\\\\\1"
  )
}

# Regras de salary_valid 

salary_rules_tbl <- tibble::tribble(
  ~rule_name,            ~type,        ~params,
  "none",                "none",       list(),
  "quantile_1_99",       "quantile",   list(lo = 0.01,  hi = 0.99),
  "quantile_0.5_99.5",   "quantile",   list(lo = 0.005, hi = 0.995),
  "iqr_1.5",             "iqr",        list(k = 1.5),
  "iqr_3",               "iqr",        list(k = 3.0)
)

apply_salary_rule_mask <- function(salary_mid, type, params) {
  if (type == "none") {
    return(!is.na(salary_mid))
  }
  if (type == "quantile") {
    qs <- quantile(salary_mid, c(params$lo, params$hi), na.rm = TRUE)
    return(!is.na(salary_mid) & salary_mid >= qs[1] & salary_mid <= qs[2])
  }
  if (type == "iqr") {
    q   <- quantile(salary_mid, c(0.25, 0.75), na.rm = TRUE)
    iqr <- q[2] - q[1]
    lo  <- q[1] - params$k * iqr
    hi  <- q[2] + params$k * iqr
    return(!is.na(salary_mid) & salary_mid >= lo & salary_mid <= hi)
  }
  stop("Tipo de regra desconhecido.")
}

tune_salary_rules <- function(df, rules_tbl, n_folds = 3, max_docs = 3000) {
  df_probe <- df %>%
    filter(!is.na(salary_mid)) %>%
    mutate(
      salary_log = log(salary_mid),
      text_blob  = stringr::str_to_lower(text_blob)
    )

  if (nrow(df_probe) > max_docs) {
    set.seed(123)
    df_probe <- df_probe %>%
      group_by(cut_number(salary_log, n = n_folds, labels = FALSE)) %>%
      sample_frac(size = max_docs / nrow(df_probe), replace = FALSE) %>%
      ungroup()
  }

  it_probe <- itoken(df_probe$text_blob, tokenizer = word_tokenizer, progressbar = FALSE)
  vocab_probe <- create_vocabulary(it_probe, ngram = c(1L, 1L))
  vocab_probe <- prune_vocabulary(vocab_probe, term_count_min = 5L)
  vectorizer_probe <- vocab_vectorizer(vocab_probe)

  it_probe <- itoken(df_probe$text_blob, tokenizer = word_tokenizer, progressbar = FALSE)
  dtm_probe <- create_dtm(it_probe, vectorizer_probe)
  tfidf_tr  <- TfIdf$new()
  dtm_probe_tfidf <- tfidf_tr$fit_transform(dtm_probe)

  probe_df_model <- df_probe %>%
    select(job_id, salary_log) %>%
    bind_cols(as.data.frame(as.matrix(dtm_probe_tfidf)))

  rec <- recipe(salary_log ~ ., data = probe_df_model) %>%
    update_role(job_id, new_role = "id") %>%
    step_zv(all_predictors())

  en_spec <- linear_reg(
    penalty = tune(),
    mixture = tune()
  ) %>%
    set_engine("glmnet")

  results <- purrr::map_dfr(
    split(rules_tbl, rules_tbl$rule_name),
    function(rule_row) {
      rule_row <- rule_row[1, ]
      mask <- apply_salary_rule_mask(
        df_probe$salary_mid,
        type   = rule_row$type,
        params = rule_row$params[[1]]
      )
      df_rule <- probe_df_model[mask, , drop = FALSE]
      if (nrow(df_rule) < 50) {
        return(tibble(
          rule_name = rule_row$rule_name,
          rmse      = NA_real_,
          n         = nrow(df_rule),
          penalty   = NA_real_,
          mixture   = NA_real_
        ))
      }

      set.seed(123)
      folds_rule <- vfold_cv(df_rule, v = n_folds, strata = salary_log)

      wf <- workflow() %>%
        add_model(en_spec) %>%
        add_recipe(rec)

      param_set <- hardhat::extract_parameter_set_dials(wf) %>%
        dials::finalize(df_rule)

      grid <- dials::grid_space_filling(param_set, size = 10)

      res <- tune_grid(
        wf,
        resamples = folds_rule,
        grid      = grid,
        metrics   = metric_set(rmse),
        control   = control_grid(save_pred = FALSE)
      )

      best_res <- tryCatch(
        tune::show_best(res, metric = "rmse", n = 1),
        error = function(e) tibble()
      )

      if (nrow(best_res) == 0) {
        tibble(
          rule_name = rule_row$rule_name,
          rmse      = NA_real_,
          n         = nrow(df_rule),
          penalty   = NA_real_,
          mixture   = NA_real_
        )
      } else {
        tibble(
          rule_name = rule_row$rule_name,
          rmse      = best_res$mean[1],
          n         = nrow(df_rule),
          penalty   = if ("penalty" %in% names(best_res)) best_res$penalty[1] else NA_real_,
          mixture   = if ("mixture" %in% names(best_res)) best_res$mixture[1] else NA_real_
        )
      }
    }
  )

  results
}

# TF-IDF + LSA 

build_tfidf_lsa_auto <- function(df_text,
                                 doc_prop_min   = NULL,
                                 doc_prop_max   = 0.8,
                                 vocab_term_max = 40000L,
                                 var_target     = 0.90,
                                 svd_rank_max   = 200L) {
  it_full <- itoken(
    df_text$text_blob,
    tokenizer   = word_tokenizer,
    progressbar = FALSE
  )
  vocab_full <- create_vocabulary(it_full, ngram = c(1L, 2L))
  n_docs     <- nrow(df_text)
  if (n_docs <= 0) stop("Sem documentos para TF-IDF.")

  if (is.null(doc_prop_min)) {
    doc_prop_min <- max(2 / n_docs, 0.001)
  }

  vocab <- prune_vocabulary(
    vocab_full,
    doc_proportion_min = doc_prop_min,
    doc_proportion_max = doc_prop_max,
    vocab_term_max     = min(vocab_term_max, nrow(vocab_full))
  )

  if (nrow(vocab) == 0L) stop("Vocabulário vazio após pruning.")

  vectorizer <- vocab_vectorizer(vocab)

  it_all <- itoken(
    df_text$text_blob,
    tokenizer   = word_tokenizer,
    progressbar = FALSE
  )
  dtm_counts <- create_dtm(it_all, vectorizer)

  tfidf_tr  <- TfIdf$new()
  dtm_tfidf <- tfidf_tr$fit_transform(dtm_counts)
  dtm_tfidf_norm <- text2vec::normalize(dtm_tfidf, "l2")

  max_rank <- min(
    svd_rank_max,
    nrow(dtm_tfidf_norm) - 1L,
    ncol(dtm_tfidf_norm) - 1L
  )
  if (max_rank < 2L) stop("Matriz TF-IDF muito pequena para LSA.")

  svd_res <- irlba(dtm_tfidf_norm, nv = max_rank)

  var_cum <- cumsum(svd_res$d^2) / sum(svd_res$d^2)
  lsa_k   <- which(var_cum >= var_target)[1]
  if (is.na(lsa_k)) lsa_k <- max_rank
  lsa_k <- max(2L, lsa_k)

  tfidf_lsa <- svd_res$u[, 1:lsa_k, drop = FALSE] %*% diag(svd_res$d[1:lsa_k])
  colnames(tfidf_lsa) <- paste0("tfidf_lsa_", seq_len(ncol(tfidf_lsa)))

  list(
    vocab          = vocab,
    vectorizer     = vectorizer,
    tfidf_tr       = tfidf_tr,
    dtm_tfidf_norm = dtm_tfidf_norm,
    tfidf_lsa      = tfidf_lsa,
    lsa_k          = lsa_k,
    variance_curve = tibble(
      k       = seq_along(var_cum),
      cum_var = var_cum
    )
  )
}

# Embeddings GloVe

adaptive_indices <- function(n,
                             n_min  = 2000L,
                             n_max  = 8000L,
                             power  = 0.7) {
  if (n <= n_min) return(seq_len(n))
  target <- min(n_max, ceiling(n^power))
  sort(sample.int(n, size = target))
}

build_glove_tcm <- function(tokens_list,
                            vocab,
                            fast = FALSE) {
  if (!fast) {
    idx <- seq_along(tokens_list)
  } else {
    idx <- adaptive_indices(length(tokens_list))
  }

  tokens_sub <- tokens_list[idx]
  avg_doc_len <- mean(vapply(tokens_sub, length, FUN.VALUE = integer(1)))
  skip_window <- if (avg_doc_len <= 20) 2L else if (avg_doc_len <= 100) 5L else 8L

  it_tokens <- itoken(tokens_sub, progressbar = FALSE)

  tcm <- create_tcm(
    it                = it_tokens,
    vectorizer        = vocab_vectorizer(vocab),
    skip_grams_window = skip_window
  )

  list(
    tcm         = tcm,
    skip_window = skip_window,
    indices     = idx
  )
}

doc_embedding_one <- function(tokens, word_vec) {
  tokens <- intersect(tokens, rownames(word_vec))
  if (length(tokens) == 0L) {
    return(rep(0, ncol(word_vec)))
  }
  colMeans(word_vec[tokens, , drop = FALSE])
}

# Receitas e tuning de modelos

build_recipe <- function(data, feature_pattern = c("tfidf", "emb", "hybrid")) {
  feature_pattern <- match.arg(feature_pattern)

  rec <- recipe(
    salary_log ~ .,
    data = data
  ) %>%
    step_rm(
      any_of(c(
        "job_id",
        "company_name",
        "skills_required",
        "tools_preferred",
        "salary_range_usd",
        "salary_min",
        "salary_max",
        "salary_mid",
        "salary_valid",
        "text_blob"
      ))
    ) %>%
    step_date(posted_date, features = c("year", "month"), keep_original_cols = FALSE)

  if (feature_pattern == "tfidf") {
    rec <- rec %>% step_rm(starts_with("emb_"))
  } else if (feature_pattern == "emb") {
    rec <- rec %>% step_rm(starts_with("tfidf_lsa_"))
  }

  rec %>%
    step_impute_median(all_numeric_predictors()) %>%
    step_nzv(all_predictors()) %>%
    step_novel(all_nominal_predictors()) %>%
    step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
    step_zv(all_numeric_predictors()) %>%
    step_normalize(all_numeric_predictors())
}

regression_metrics <- metric_set(rmse, mae)
# --- Metrics (robust) ------------------------------------------------------

regression_metrics <- yardstick::metric_set(yardstick::rmse, yardstick::mae)

assert_has_cols <- function(df, cols) {
  miss <- setdiff(cols, names(df))
  if (length(miss)) {
    stop(
      "Missing required columns: ", paste(miss, collapse = ", "),
      "\nAvailable columns: ", paste(names(df), collapse = ", "),
      call. = FALSE
    )
  }
  invisible(TRUE)
}

regression_metrics_safe <- function(data, truth = salary_log, estimate = .pred) {
  truth_q    <- rlang::ensym(truth)
  estimate_q <- rlang::ensym(estimate)

  needed <- c(rlang::as_string(truth_q), rlang::as_string(estimate_q))
  assert_has_cols(data, needed)

  regression_metrics(data, truth = !!truth_q, estimate = !!estimate_q)
}


tune_model <- function(rec,
                       spec,
                       folds,
                       metrics   = regression_metrics,
                       grid_size = 40L) {
  wf <- workflow() %>%
    add_model(spec) %>%
    add_recipe(rec)

  param_set <- hardhat::extract_parameter_set_dials(wf)
  first_train_data <- rsample::analysis(folds$splits[[1]])
  param_set <- dials::finalize(param_set, first_train_data)

  grid <- dials::grid_space_filling(
    param_set,
    size = grid_size
  )

  tune_grid(
    wf,
    resamples = folds,
    grid      = grid,
    metrics   = metrics,
    control   = control_grid(save_pred = FALSE)
  )
}

# Sistema de recomendação 

build_user_profile_text <- function(
  skills           = NULL,
  tools            = NULL,
  desired_title    = NULL,
  industry         = NULL,
  experience_level = NULL
) {
  parts <- c()
  if (!is.null(desired_title))    parts <- c(parts, tolower(desired_title))
  if (!is.null(industry))         parts <- c(parts, tolower(industry))
  if (!is.null(experience_level)) parts <- c(parts, tolower(experience_level))
  if (!is.null(skills) && length(skills) > 0) {
    parts <- c(parts, paste(tolower(skills), collapse = " "))
  }
  if (!is.null(tools) && length(tools) > 0) {
    parts <- c(parts, paste(tolower(tools), collapse = " "))
  }
  paste(parts, collapse = " | ")
}

vectorize_text_tfidf <- function(text, vectorizer, tfidf_transformer) {
  it_q <- itoken(
    text,
    tokenizer   = word_tokenizer,
    progressbar = FALSE
  )
  dtm_q <- create_dtm(it_q, vectorizer)
  tfidf_transformer$transform(dtm_q)
}

apply_filters <- function(
  df,
  experience_levels = NULL,
  employment_types  = NULL,
  locations         = NULL,
  min_salary        = NULL,
  max_salary        = NULL
) {
  n <- nrow(df)
  mask <- rep(TRUE, n)

  log_filter <- function(name, before, after) {
    if (getOption("ai_jobs_recs_debug", FALSE)) {
      message(sprintf("Filtro %s: %d -> %d vagas", name, before, after))
    }
  }

  if (!is.null(experience_levels) && length(experience_levels) > 0) {
    before <- sum(mask)
    lvls <- tolower(experience_levels)
    re_lvl <- paste0("\\b(", paste(escape_regex(lvls), collapse = "|"), ")\\b")
    exp_col <- tolower(df$experience_level)
    mask <- mask & stringr::str_detect(exp_col, re_lvl)
    after <- sum(mask)
    log_filter("experience_level", before, after)
  }

  if (!is.null(employment_types) && length(employment_types) > 0) {
    before <- sum(mask)
    tys <- tolower(employment_types)
    re_type <- paste0("\\b(", paste(escape_regex(tys), collapse = "|"), ")\\b")
    type_col <- tolower(df$employment_type)
    mask <- mask & stringr::str_detect(type_col, re_type)
    after <- sum(mask)
    log_filter("employment_type", before, after)
  }

  if (!is.null(locations) && length(locations) > 0) {
    before <- sum(mask)
    locs <- tolower(locations)
    locs_escaped <- escape_regex(locs)
    re_loc <- paste0(locs_escaped, collapse = "|")
    loc_col <- tolower(df$location)
    mask <- mask & stringr::str_detect(loc_col, re_loc)
    after <- sum(mask)
    log_filter("location", before, after)
  }

  if (!is.null(min_salary)) {
    before <- sum(mask)
    mask <- mask & coalesce(df$salary_mid, -Inf) >= min_salary
    after <- sum(mask)
    log_filter("min_salary", before, after)
  }
  if (!is.null(max_salary)) {
    before <- sum(mask)
    mask <- mask & coalesce(df$salary_mid,  Inf) <= max_salary
    after <- sum(mask)
    log_filter("max_salary", before, after)
  }

  mask & !is.na(mask)
}

mmr <- function(doc_scores, doc_embeddings, k = 10, lambda_mult = 0.7) {
  n <- length(doc_scores)
  if (n == 0) return(integer(0))
  k <- min(k, n)

  doc_scores <- as.numeric(doc_scores)
  doc_scores[is.na(doc_scores)] <- -Inf

  if (inherits(doc_embeddings, "dgCMatrix")) {
    norms <- sqrt(Matrix::rowSums(doc_embeddings ^ 2))
    norms[!is.finite(norms) | norms == 0] <- 1
    emb <- Matrix::Diagonal(x = 1 / norms) %*% doc_embeddings
  } else {
    emb <- as.matrix(doc_embeddings)
    norms <- sqrt(rowSums(emb^2))
    norms[!is.finite(norms) | norms == 0] <- 1
    emb <- emb / norms
  }

  selected   <- integer(0)
  candidates <- seq_len(n)

  first <- which.max(doc_scores)
  if (length(first) == 0L) first <- candidates[1]
  selected   <- c(selected, first)
  candidates <- setdiff(candidates, first)

  while (length(selected) < k && length(candidates) > 0) {
    best_idx <- NA_integer_
    best_val <- -Inf

    for (c in candidates) {
      sim_vec <- emb[c, , drop = FALSE] %*% Matrix::t(emb[selected, , drop = FALSE])
      sim_to_selected <- max(as.numeric(sim_vec))
      val <- lambda_mult * doc_scores[c] - (1 - lambda_mult) * sim_to_selected
      if (val > best_val) {
        best_val <- val
        best_idx <- c
      }
    }

    selected   <- c(selected, best_idx)
    candidates <- setdiff(candidates, best_idx)
  }

  selected
}

recommend_jobs_for_user <- function(
  df_model_base,
  dtm_tfidf_norm,
  vectorizer,
  tfidf_transformer,
  skills           = NULL,
  tools            = NULL,
  desired_title    = NULL,
  industry         = NULL,
  experience_level = NULL,
  top_k            = 15,
  filters          = list(),
  diversify        = TRUE,
  lambda_mmr       = 0.7,
  allow_fallback   = TRUE
) {
  profile_text <- build_user_profile_text(
    skills           = skills,
    tools            = tools,
    desired_title    = desired_title,
    industry         = industry,
    experience_level = experience_level
  )

  q_vec <- vectorize_text_tfidf(
    text              = profile_text,
    vectorizer        = vectorizer,
    tfidf_transformer = tfidf_transformer
  )

  sims <- as.numeric(q_vec %*% t(dtm_tfidf_norm))

  mask <- apply_filters(
    df_model_base,
    experience_levels = filters$experience_levels,
    employment_types  = filters$employment_types,
    locations         = filters$locations,
    min_salary        = filters$min_salary,
    max_salary        = filters$max_salary
  )

  idxs <- which(mask)
  if (length(idxs) == 0) {
    if (allow_fallback && !is.null(filters$locations) && length(filters$locations) > 0) {
      warning("Sem vagas após filtros; removendo filtro de localização e tentando novamente.")
      filters2 <- filters
      filters2$locations <- NULL
      return(recommend_jobs_for_user(
        df_model_base     = df_model_base,
        dtm_tfidf_norm    = dtm_tfidf_norm,
        vectorizer        = vectorizer,
        tfidf_transformer = tfidf_transformer,
        skills           = skills,
        tools            = tools,
        desired_title    = desired_title,
        industry         = industry,
        experience_level = experience_level,
        top_k            = top_k,
        filters          = filters2,
        diversify        = diversify,
        lambda_mmr       = lambda_mmr,
        allow_fallback   = FALSE
      ))
    }

    warning("Sem vagas após filtros.")
    empty_res <- df_model_base[0, , drop = FALSE] %>%
      mutate(score = numeric(0))
    return(empty_res)
  }

  cand_scores <- sims[idxs]
  cand_scores[is.na(cand_scores)] <- -Inf

  if (diversify && length(idxs) > top_k) {
    # (4) Prefiltragem para limitar pool e evitar densificar matrizes enormes no MMR
    pool_n <- min(length(idxs), max(300L, 20L * top_k))
    ord <- order(cand_scores, decreasing = TRUE)[seq_len(pool_n)]
    idxs_pool <- idxs[ord]
    scores_pool <- cand_scores[ord]

    cand_emb <- dtm_tfidf_norm[idxs_pool, , drop = FALSE]
    sel_local <- mmr(
      doc_scores     = scores_pool,
      doc_embeddings = cand_emb,
      k              = top_k,
      lambda_mult    = lambda_mmr
    )
    chosen <- idxs_pool[sel_local]
  } else {
    chosen <- idxs[order(cand_scores, decreasing = TRUE)][
      seq_len(min(top_k, length(idxs)))
    ]
  }

  df_out <- df_model_base[chosen, , drop = FALSE] %>%
    mutate(score = sims[chosen]) %>%
    arrange(desc(score))

  df_out
}

resolve_job_index <- function(job_identifier, df_model_base, JOB_INDEX) {
  if (is.numeric(job_identifier) && job_identifier %% 1 == 0 &&
      job_identifier >= 1 && job_identifier <= nrow(df_model_base)) {
    return(as.integer(job_identifier))
  }
  key <- as.character(job_identifier)
  idx <- JOB_INDEX[[key]]
  if (!is.null(idx)) return(as.integer(idx))
  stop("job_identifier não encontrado.")
}

recommend_similar_jobs <- function(
  df_model_base,
  dtm_tfidf_norm,
  JOB_INDEX,
  job_identifier,
  top_k      = 10,
  filters    = list(),
  diversify  = FALSE,
  lambda_mmr = 0.7
) {
  base_idx <- resolve_job_index(job_identifier, df_model_base, JOB_INDEX)

  v    <- dtm_tfidf_norm[base_idx, , drop = FALSE]
  sims <- as.numeric(v %*% t(dtm_tfidf_norm))
  sims[base_idx] <- -Inf

  mask <- apply_filters(
    df_model_base,
    experience_levels = filters$experience_levels,
    employment_types  = filters$employment_types,
    locations         = filters$locations,
    min_salary        = filters$min_salary,
    max_salary        = filters$max_salary
  )

  idxs <- which(mask)
  idxs <- setdiff(idxs, base_idx)

  if (length(idxs) == 0) {
    warning("Sem vagas similares após filtros.")
    empty_res <- df_model_base[0, , drop = FALSE] %>%
      mutate(score = numeric(0))
    return(empty_res)
  }

  cand_scores <- sims[idxs]
  cand_scores[is.na(cand_scores)] <- -Inf

  if (diversify && length(idxs) > top_k) {
    # (4) Prefiltragem para limitar pool antes do MMR
    pool_n <- min(length(idxs), max(300L, 20L * top_k))
    ord <- order(cand_scores, decreasing = TRUE)[seq_len(pool_n)]
    idxs_pool <- idxs[ord]
    scores_pool <- cand_scores[ord]

    cand_emb <- dtm_tfidf_norm[idxs_pool, , drop = FALSE]
    sel_local <- mmr(
      doc_scores     = scores_pool,
      doc_embeddings = cand_emb,
      k              = top_k,
      lambda_mult    = lambda_mmr
    )
    chosen <- idxs_pool[sel_local]
  } else {
    chosen <- idxs[order(cand_scores, decreasing = TRUE)][
      seq_len(min(top_k, length(idxs)))
    ]
  }

  df_out <- df_model_base[chosen, , drop = FALSE] %>%
    mutate(score = sims[chosen]) %>%
    arrange(desc(score))

  df_out
}

# Função de geração do HTML

generate_html_report <- function(html_path,
                                 out_dir,
                                 df,
                                 df_model_base,
                                 res) {

  html_begin(html_path, "Mercado de Vagas em IA/ML – Relatório Final")
  html_h1("Mercado de trabalho em IA/ML – Relatório final")

  # 1. Introdução 

  html_h2("1. Introdução e contexto")
  html_p(paste0(
    "Este trabalho utiliza um conjunto de dados de vagas para o mercado de Inteligência Artificial e Machine Learning, ",
    "obtido via Kaggle (",
    "<a href='https://www.kaggle.com/datasets/abhishekjaiswal4896/ai-job-market-trends/data' target='_blank' rel='noopener'>AI Job Market Trends</a>",
    "). Cada linha representa uma vaga identificada por <code>job_id</code>, com informações de cargo, empresa, localização, ",
    "nível de experiência, tipo de vínculo, setor, habilidades e ferramentas, além de uma faixa salarial anual em dólares."
  ))

  # (1) Bloco de objetivos em lista + nota (sem colagem de frases)
  html_h3("1.1 Objetivos do projeto")
  html_list(c(
    "Predizer o salário anual da vaga (regressão supervisionada) com variáveis estruturadas e texto.",
    "Construir perfis de vagas/profissionais (clusters e tópicos) para segmentar o mercado por stack, senioridade e faixa salarial.",
    "Recomendação content-based: perfil → vagas, e vaga → vagas similares, com filtros de negócio."
  ))
  html_note(
    "Como o salário é assimétrico, o alvo do supervisionado é <code>salary_log = log(salary_mid)</code>. ",
    "A EDA inclui correlação (numéricas) e η² (categóricas) com <code>salary_log</code>."
  )

  # (1) Visão geral dinâmica
  html_h3("1.2 Visão geral (resultados desta execução)")

  html_note(
    "Tamanho do dataset: <b>", nrow(df), "</b> vagas e <b>", ncol(df), "</b> colunas. ",
    "Base modelável (após <code>salary_valid</code> e remoção de salários ausentes): <b>", nrow(df_model_base), "</b> vagas."
  )

  if (!is.null(res$supervised_test_results) && nrow(res$supervised_test_results) > 0) {
    tst <- res$supervised_test_results
    best_rmse_row <- tst %>% filter(.metric == "rmse") %>% arrange(.estimate) %>% slice(1)
    best_model <- best_rmse_row$model[1]
    best_rmse  <- best_rmse_row$.estimate[1]
    best_mae_row <- tst %>% filter(model == best_model, .metric == "mae") %>% slice(1)
    best_mae <- if (nrow(best_mae_row) > 0) best_mae_row$.estimate[1] else NA_real_

    html_note(
      "Modelo com melhor RMSE em teste: <b>", best_model,
      "</b> (RMSE ≈ ", sprintf("%.3f", best_rmse),
      ifelse(is.finite(best_mae), paste0(", MAE ≈ ", sprintf("%.3f", best_mae)), ""),
      ", em <code>salary_log</code>)."
    )
  }

  if (!is.null(res$repr_test_results) && nrow(res$repr_test_results) > 0) {
    rr <- res$repr_test_results
    best_repr_row <- rr %>% filter(.metric == "rmse") %>% arrange(.estimate) %>% slice(1)
    if (nrow(best_repr_row) > 0) {
      html_note(
        "Na comparação das representações textuais (mantendo Random Forest fixa), a menor RMSE em teste foi obtida com <b>",
        best_repr_row$features[1], "</b>."
      )
    }
  }

  # 2. Descrição dos dados e EDA 

  html_h2("2. Descrição dos dados e EDA")

  ## 2.1 estrutura
  html_h3("2.1 Estrutura geral dos dados")

  dims_tbl <- tibble::tibble(
    linhas  = nrow(df),
    colunas = ncol(df)
  )
  html_table(dims_tbl, "Dimensão do dataset", digits = 0)

  sal_sum <- summary(df$salary_mid)
  sal_tbl <- tibble::tibble(
    estatistica = names(sal_sum),
    valor       = as.numeric(sal_sum)
  )
  html_table(sal_tbl, "Resumo da variável salary_mid", digits = 2)

  date_sum <- summary(df$posted_date)
  date_tbl <- tibble::tibble(
    estatistica = names(date_sum),
    valor       = as.character(date_sum)
  )
  html_table(date_tbl, "Resumo da variável posted_date", digits = 0)

  exp_tbl <- df %>%
    count(experience_level, sort = TRUE)
  html_table(exp_tbl, "Distribuição de experience_level", digits = 0)

  emp_tbl <- df %>%
    count(employment_type, sort = TRUE)
  html_table(emp_tbl, "Distribuição de employment_type", digits = 0)

  ind_tbl <- df %>%
    count(industry, sort = TRUE) %>%
    head(20)
  html_table(ind_tbl, "Top 20 indústrias por número de vagas", digits = 0)

  html_note(
    "Na limpeza inicial, garantimos <code>job_id</code> como identificador, parseamos a faixa salarial em ",
    "<code>salary_min</code>/<code>salary_max</code>, definimos <code>salary_mid</code> como alvo numérico e padronizamos datas e campos textuais."
  )

  ## 2.2 distribuição de salário
  html_h3("2.2 Distribuição de salários (salary_mid)")

  p_sal <- ggplot(
    df %>% filter(!is.na(salary_mid)),
    aes(x = salary_mid)
  ) +
    geom_histogram(bins = 50) +
    scale_x_continuous(labels = scales::dollar_format(prefix = "US$")) +
    labs(
      title = "Distribuição de salary_mid",
      x = "Salário anual (US$)",
      y = "Número de vagas"
    )

  fname_sal <- "eda_salary_mid_hist.png"
  save_gg(p_sal, fname_sal, dir = out_dir)
  cat("<p><b>Figura 1 – Distribuição de salary_mid</b><br/><img src='", fname_sal, "'></p>\n", sep = "")

  p_sal_log <- ggplot(
    df %>% filter(!is.na(salary_mid), salary_mid > 0),
    aes(x = log(salary_mid))
  ) +
    geom_histogram(bins = 50) +
    labs(
      title = "Distribuição de log(salary_mid)",
      x = "log(salário anual)",
      y = "Número de vagas"
    )

  fname_sal_log <- "eda_salary_log_hist.png"
  save_gg(p_sal_log, fname_sal_log, dir = out_dir)
  cat("<p><b>Figura 2 – Distribuição de log(salary_mid)</b><br/><img src='", fname_sal_log, "'></p>\n", sep = "")

  med_sal <- as.numeric(sal_sum["Median"])
  q3_sal  <- as.numeric(sal_sum["3rd Qu."])
  max_sal <- as.numeric(sal_sum["Max."])

  html_p(paste0(
    "Entre as vagas com <code>salary_mid</code> informado, a mediana é ",
    scales::dollar(med_sal, prefix = "US$"),
    ", o 3º quartil é ",
    scales::dollar(q3_sal, prefix = "US$"),
    " e o máximo alcança ",
    scales::dollar(max_sal, prefix = "US$"),
    ". Essa distância entre mediana e máximo indica uma distribuição assimétrica à direita. ",
    "A transformação logarítmica (Figura 2) produz uma distribuição mais regular, justificando o uso de ",
    "<code>salary_log = log(salary_mid)</code> como alvo da regressão."
  ))

  ## 2.3 Salário por nível de experiência e país
  html_h3("2.3 Salário por nível de experiência e país")

  p_box_exp <- ggplot(
    df %>% filter(!is.na(salary_mid)),
    aes(x = experience_level, y = salary_mid)
  ) +
    geom_boxplot() +
    scale_y_continuous(labels = scales::dollar_format(prefix = "US$")) +
    labs(
      title = "Salário (mid) por nível de experiência",
      x = "Nível",
      y = "Salário anual (US$)"
    )

  fname_box_exp <- "eda_salary_box_experience.png"
  save_gg(p_box_exp, fname_box_exp, dir = out_dir)
  cat("<p><b>Figura 3 – Salário por nível de experiência</b><br/><img src='", fname_box_exp, "'></p>\n", sep = "")

  df_country <- df %>%
    mutate(country = stringr::str_extract(location, "[A-Za-z ]+$")) %>%
    count(country, sort = TRUE)
  top_countries <- df_country %>%
    slice_head(n = 10) %>%
    pull(country)

  p_box_country <- ggplot(
    df %>%
      mutate(country = stringr::str_extract(location, "[A-Za-z ]+$")) %>%
      filter(country %in% top_countries, !is.na(salary_mid)),
    aes(
      x = reorder(country, salary_mid, median, na.rm = TRUE),
      y = salary_mid
    )
  ) +
    geom_boxplot() +
    coord_flip() +
    scale_y_continuous(labels = scales::dollar_format(prefix = "US$")) +
    labs(
      title = "Salário (mid) por país (top 10 em número de vagas)",
      x = "País",
      y = "Salário anual (US$)"
    )

  fname_box_country <- "eda_salary_box_country.png"
  save_gg(p_box_country, fname_box_country, dir = out_dir)
  cat("<p><b>Figura 4 – Salário por país (top 10)</b><br/><img src='", fname_box_country, "'></p>\n", sep = "")

  salary_by_exp <- df %>%
    filter(!is.na(salary_mid), !is.na(experience_level)) %>%
    group_by(experience_level) %>%
    summarise(
      n   = n(),
      med = median(salary_mid, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(med))

  salary_by_country <- df %>%
    mutate(country = stringr::str_extract(location, "[A-Za-z ]+$")) %>%
    filter(!is.na(salary_mid), country %in% top_countries) %>%
    group_by(country) %>%
    summarise(
      n   = n(),
      med = median(salary_mid, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(med))

  html_table(
    salary_by_exp,
    "Mediana de salary_mid por nível de experiência",
    digits = 0
  )

  html_table(
    salary_by_country,
    "Mediana de salary_mid por país (entre os 10 países com mais vagas)",
    digits = 0
  )

  if (nrow(salary_by_exp) > 0 && nrow(salary_by_country) > 0) {
    n_exp     <- min(2L, nrow(salary_by_exp))
    n_country <- min(2L, nrow(salary_by_country))

    top_exp <- salary_by_exp %>% slice_head(n = n_exp)
    top_country_med <- salary_by_country %>% slice_head(n = n_country)

    exp_nomes <- paste(sprintf("<code>%s</code>", top_exp$experience_level), collapse = " e ")
    exp_meds  <- paste(scales::dollar(top_exp$med, prefix = "US$"), collapse = " e ")

    country_nomes <- paste(top_country_med$country, collapse = " e ")
    country_meds  <- paste(scales::dollar(top_country_med$med, prefix = "US$"), collapse = " e ")

    html_p(paste0(
      "Os boxplots das Figuras 3 e 4 mostram heterogeneidade salarial. ",
      "Na base, os níveis com maior mediana salarial são ",
      exp_nomes, " (medianas em torno de ", exp_meds, "). ",
      "Entre os 10 países com mais vagas, as maiores medianas aparecem em ",
      country_nomes, " (medianas em torno de ", country_meds, "). ",
      "Isso reforça a importância de <code>experience_level</code> e localização como preditores."
    ))
  } else {
    html_p("Os boxplots das Figuras 3 e 4 evidenciam diferenças de distribuição salarial por nível de experiência e país.")
  }

  ## 2.4 missing
  html_h3("2.4 Valores faltantes")

  missing_df <- df %>%
    summarise(across(everything(), ~ mean(is.na(.x)), .names = "missing_{.col}")) %>%
    tidyr::pivot_longer(
      everything(),
      names_to  = "variavel",
      values_to = "taxa_missing"
    ) %>%
    mutate(variavel = stringr::str_remove(variavel, "^missing_")) %>%
    arrange(desc(taxa_missing))

  missing_df_nonzero <- missing_df %>% filter(taxa_missing > 0)

  if (nrow(missing_df_nonzero) > 0) {
    html_table(
      missing_df_nonzero %>%
        mutate(taxa_percent = round(100 * taxa_missing, 1)) %>%
        select(variavel, taxa_percent) %>%
        head(25),
      "Top 25 variáveis com maior taxa de missing (%)",
      digits = 1
    )

    p_missing <- ggplot(
      missing_df_nonzero,
      aes(x = reorder(variavel, taxa_missing), y = taxa_missing)
    ) +
      geom_col() +
      coord_flip() +
      scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
      labs(
        title = "Taxa de missing por variável",
        x = "Variável",
        y = "% de missing"
      )

    fname_missing <- "eda_missing_rates.png"
    save_gg(p_missing, fname_missing, dir = out_dir)
    cat("<p><b>Figura 5 – Taxa de missing por variável</b><br/><img src='", fname_missing, "'></p>\n", sep = "")
  } else {
    html_note("Após a etapa de preparação, não foram identificados valores faltantes residuais nas variáveis analisadas.")
  }

  html_p(paste0(
    "Em variáveis numéricas, os faltantes são imputados com mediana na receita (<code>step_impute_median</code>), ",
    "pois a mediana é robusta a outliers. Nas colunas textuais usadas em <code>text_blob</code> ",
    "(<code>job_title</code>, <code>skills_required</code>, <code>tools_preferred</code> e <code>industry</code>), ",
    "<code>NA</code> é substituído por string vazia antes da concatenação, evitando problemas na tokenização."
  ))

  ## 2.5 texto (skills)
  html_h3("2.5 EDA de texto – skills")

  skills_tokens <- df %>%
    select(job_id, skills_required) %>%
    mutate(skills_required = coalesce(skills_required, "")) %>%
    tidyr::separate_rows(skills_required, sep = ",") %>%
    mutate(
      skill = skills_required %>%
        stringr::str_to_lower() %>%
        stringr::str_squish()
    ) %>%
    filter(skill != "")

  skills_freq <- skills_tokens %>%
    count(skill, sort = TRUE) %>%
    slice_head(n = 20)

  p_skills <- ggplot(
    skills_freq,
    aes(x = reorder(skill, n), y = n)
  ) +
    geom_col() +
    coord_flip() +
    labs(
      title = "Top 20 skills mais mencionadas",
      x = "Skill",
      y = "Frequência"
    )

  fname_skills <- "eda_top_skills.png"
  save_gg(p_skills, fname_skills, dir = out_dir)
  cat("<p><b>Figura 6 – Skills mais frequentes</b><br/><img src='", fname_skills, "'></p>\n", sep = "")

  top_skills <- skills_freq %>%
    slice_head(n = min(5L, nrow(skills_freq))) %>%
    pull(skill)

  html_p(paste0(
    "A Figura 6 mostra que as skills mais mencionadas incluem ",
    paste(sprintf("<code>%s</code>", top_skills), collapse = ", "),
    ". Essa concentração em um conjunto relativamente pequeno de tecnologias motiva o uso de representações textuais ",
    "mais ricas (TF‑IDF, LSA e embeddings) na modelagem supervisionada."
  ))

  # (2) Agora a seção 2.6 vem depois da 2.5 (ordem correta)
  html_h3("2.6 Correlações e associações com o salário")

  sa <- res$salary_assoc
  html_p(paste0(
    "A seguir, medimos relações entre variáveis e salário usando <code>salary_log</code>. ",
    "Para variáveis numéricas, reportamos correlação de Pearson e Spearman. ",
    "Para variáveis categóricas, usamos <b>η²</b> (proporção da variância de <code>salary_log</code> explicada por grupos; não implica causalidade). ",
    "Para manter interpretabilidade, categorias raras em variáveis de alta cardinalidade são agrupadas em <code>Outros</code>. ",
    "Variáveis diretamente derivadas do próprio salário não são incluídas nesta EDA para evitar correlações triviais."
  ))

  if (!is.null(sa$numeric_corr) && nrow(sa$numeric_corr) > 0) {
    html_table(
      sa$numeric_corr %>%
        mutate(
          sig_spearman = case_when(
            is.na(p_spearman_fdr) ~ "",
            p_spearman_fdr < 0.001 ~ "***",
            p_spearman_fdr < 0.01  ~ "**",
            p_spearman_fdr < 0.05  ~ "*",
            TRUE ~ ""
          )
        ) %>%
        select(predictor, spearman, p_spearman, p_spearman_fdr, sig_spearman, pearson, p_pearson, n),
      "Correlação com salary_log (numéricas)",
      digits = 4
    )

    p_corr <- ggplot(
      sa$numeric_corr %>% slice_head(n = min(10L, nrow(sa$numeric_corr))),
      aes(x = reorder(predictor, spearman), y = spearman)
    ) +
      geom_col() +
      coord_flip() +
      labs(
        title = "Top 10 correlações (Spearman) com salary_log",
        x = "Variável",
        y = "ρ de Spearman"
      )

    fname_corr <- "eda_salary_spearman_top10.png"
    save_gg(p_corr, fname_corr, dir = out_dir)
    cat("<p><b>Gráfico – Top correlações numéricas com salary_log</b><br/><img src='", fname_corr, "'></p>\n", sep = "")
  } else {
    html_note("Não foi possível estimar correlações numéricas (dados insuficientes ou variância nula).")
  }

  if (!is.null(sa$cat_assoc_eta2) && nrow(sa$cat_assoc_eta2) > 0) {
    html_table(
      sa$cat_assoc_eta2,
      "Associação categórica com salary_log (η² – variância explicada)",
      digits = 4
    )

    p_eta <- ggplot(
      sa$cat_assoc_eta2,
      aes(x = reorder(variable, eta2), y = eta2)
    ) +
      geom_col() +
      coord_flip() +
      labs(
        title = "η² por variável categórica (salary_log)",
        x = "Variável",
        y = "η²"
      )

    fname_eta <- "eda_salary_eta2_categorical.png"
    save_gg(p_eta, fname_eta, dir = out_dir)
    cat("<p><b>Gráfico – Associação categórica (η²) com salary_log</b><br/><img src='", fname_eta, "'></p>\n", sep = "")
  }

  if (!is.null(sa$exp_rank_corr) && nrow(sa$exp_rank_corr) > 0) {
    html_note(
      "Tratando <code>experience_level</code> como ordinal (Entry < Mid < Senior < Lead/Principal < Executive), ",
      "a correlação de Spearman com <code>salary_log</code> é ≈ <b>",
      sprintf("%.3f", sa$exp_rank_corr$spearman[1]), "</b> (p ≈ ",
      format(sa$exp_rank_corr$p_value[1], scientific = TRUE, digits = 3), ")."
    )
  }

  html_p(paste0(
    "Essas relações descritivas ajudam a interpretar o problema e a priorizar variáveis no modelo, ",
    "mas não devem ser lidas como efeito causal (há confusão por setor, país, senioridade, empresa etc.)."
  ))

  # 3. Pré-processamento e decisões 

  html_h2("3. Pré-processamento e decisões")

  ## 3.1 parsing salario
  html_h3("3.1 Parsing de salário e definição do alvo")

  html_p(paste0(
    "A coluna original <code>salary_range_usd</code> traz faixas em texto (ex.: \"100000-150000\" ou \"100,000 - 150,000 USD\"). ",
    "Implementamos uma função que limpa símbolos de moeda e vírgulas, identifica padrões <code>min-max</code> e valores únicos, ",
    "corrige faixas invertidas e constrói <code>salary_min</code>, <code>salary_max</code> e <code>salary_mid</code>. ",
    "O alvo da regressão é <code>salary_log = log(salary_mid)</code>."
  ))

  ## 3.2 tuning salary_valid
  html_h3("3.2 Regras de trimming de salário (salary_valid)")

  sr <- res$salary_rule_results
  salary_rule_effects <- res$salary_rule_effects

  html_table(
    sr %>% head(10),
    "Regras candidatas de trimming de salário (modelo probe – RMSE em log) – até 10 linhas",
    digits = 4
  )

  # (3) Remoção deve ser contada apenas sobre salários não-NA
  html_table(
    salary_rule_effects %>%
      mutate(
        prop_removed = ifelse(n_non_na > 0, n_removed_non_na / n_non_na, NA_real_)
      ) %>%
      select(rule_name, type, n_non_na, n_valid_full, n_removed_non_na, prop_removed, min_kept, max_kept),
    "Impacto das regras de trimming no dataset completo (apenas sobre salários não ausentes)",
    digits = 4
  )

  p_rules <- ggplot(
    sr %>% filter(!is.na(rmse)),
    aes(x = reorder(rule_name, rmse), y = rmse)
  ) +
    geom_col() +
    coord_flip() +
    labs(
      title = "Comparação de regras de trimming por RMSE (CV)",
      x = "Regra",
      y = "RMSE em log(salário)"
    )

  fname_rules <- "prep_salary_rules_rmse.png"
  save_gg(p_rules, fname_rules, dir = out_dir)
  cat("<p><b>Figura 7 – Regras de trimming de salário</b><br/><img src='", fname_rules, "'></p>\n", sep = "")

  best_rule_name <- res$best_salary_rule$rule_name[1]
  n_total <- nrow(df)
  n_non_na_salary <- sum(!is.na(df$salary_mid))
  n_valid <- sum(df$salary_valid, na.rm = TRUE)

  html_note(
    "A melhor regra de trimming pelo menor RMSE em CV foi <b>", best_rule_name,
    "</b>, mantendo ", n_valid, " de ", n_non_na_salary,
    " vagas com salário informado (", scales::percent(n_valid / max(1, n_non_na_salary), accuracy = 0.1),
    "). Em relação ao dataset completo (incluindo salários ausentes), isso corresponde a ",
    scales::percent(n_valid / max(1, n_total), accuracy = 0.1), "."
  )

  iqr_zero <- salary_rule_effects %>%
    filter(rule_name %in% c("iqr_1.5", "iqr_3"), n_removed_non_na == 0)

  if (nrow(iqr_zero) > 0) {
    html_p(paste0(
      "As regras baseadas em IQR (",
      paste(sprintf("<code>%s</code>", iqr_zero$rule_name), collapse = ", "),
      ") resultaram no mesmo conjunto de vagas que a regra <code>none</code> (após exclusão de salários ausentes), ",
      "indicando ausência de outliers extremos segundo esse critério específico. ",
      "Na prática, somente as regras por quantis efetivamente removem observações."
    ))
  }

  ## 3.3 texto unificado
  html_h3("3.3 Construção de texto unificado (text_blob)")

  html_p(paste0(
    "As colunas <code>job_title</code>, <code>skills_required</code>, <code>tools_preferred</code> e <code>industry</code> ",
    "são concatenadas em <code>text_blob</code>. Isso permite aplicar um pipeline único de NLP e capturar tanto skills quanto contexto de setor e cargo."
  ))

  ## 3.4 representações de texto
  html_h3("3.4 Representações de texto: TF‑IDF+LSA, embeddings e híbrido")

  html_p(paste0(
    "Em todas as variantes, utilizamos as mesmas variáveis estruturadas (nível de experiência, tipo de vínculo, localização, indústria, etc.). ",
    "O que muda é apenas a forma como o texto unificado <code>text_blob</code> é projetado em espaço vetorial: ",
    "os componentes <code>tfidf_lsa_*</code> vêm de TF‑IDF seguido de SVD (LSA); ",
    "os vetores <code>emb_*</code> são embeddings GloVe médios por vaga; ",
    "e a representação <code>hybrid</code> concatena as duas."
  ))

  var_curve <- res$tfidf_variance
  p_lsa <- ggplot(
    var_curve,
    aes(x = k, y = cum_var)
  ) +
    geom_line() +
    geom_point() +
    geom_hline(yintercept = 0.9, linetype = "dashed") +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    labs(
      title = "TF‑IDF + LSA – variância acumulada explicada",
      x = "Dimensão latente (k)",
      y = "Variância acumulada"
    )

  fname_lsa <- "prep_lsa_variance.png"
  save_gg(p_lsa, fname_lsa, dir = out_dir)
  cat("<p><b>Figura 8 – Variância explicada por componentes LSA</b><br/><img src='", fname_lsa, "'></p>\n", sep = "")

  repr_test_results <- res$repr_test_results

  html_table(
    repr_test_results %>% head(10),
    "Desempenho de Random Forest por representação textual (teste) – até 10 linhas",
    digits = 6
  )

  repr_diff <- repr_test_results %>%
    group_by(.metric) %>%
    mutate(
      best      = min(.estimate),
      delta     = .estimate - best,
      rel_delta = .estimate / best - 1
    ) %>%
    ungroup()

  repr_rmse <- repr_diff %>%
    filter(.metric == "rmse")

  p_repr_rmse <- ggplot(
    repr_rmse,
    aes(x = features, y = rel_delta)
  ) +
    geom_col() +
    geom_text(aes(label = scales::percent(rel_delta, accuracy = 0.01)), vjust = -0.3) +
    scale_y_continuous(labels = scales::percent_format(accuracy = 0.01)) +
    labs(
      title = "Random Forest – diferença relativa de RMSE por representação textual (teste)",
      x = "Features de texto",
      y = "Δ RMSE (% vs melhor)"
    )

  fname_repr_rmse <- "prep_repr_rf_rmse.png"
  save_gg(p_repr_rmse, fname_repr_rmse, dir = out_dir)
  cat("<p><b>Figura 9 – Diferença relativa de RMSE por representação textual</b><br/><img src='", fname_repr_rmse, "'></p>\n", sep = "")

  repr_mae <- repr_diff %>%
    filter(.metric == "mae")

  if (nrow(repr_mae) > 0) {
    p_repr_mae <- ggplot(
      repr_mae,
      aes(x = features, y = rel_delta)
    ) +
      geom_col() +
      geom_text(aes(label = scales::percent(rel_delta, accuracy = 0.01)), vjust = -0.3) +
      scale_y_continuous(labels = scales::percent_format(accuracy = 0.01)) +
      labs(
        title = "Random Forest – diferença relativa de MAE por representação textual (teste)",
        x = "Features de texto",
        y = "Δ MAE (% vs melhor)"
      )

    fname_repr_mae <- "prep_repr_rf_mae.png"
    save_gg(p_repr_mae, fname_repr_mae, dir = out_dir)
    cat("<p><b>Gráfico – Diferença relativa de MAE por representação textual</b><br/><img src='", fname_repr_mae, "'></p>\n", sep = "")
  }

  best_repr_row <- repr_rmse %>% arrange(.estimate) %>% slice(1)
  best_repr <- best_repr_row$features[1]

  extra_obs <- if (identical(best_repr, "hybrid")) {
    " (representação híbrida de TF‑IDF+LSA e embeddings)"
  } else if (identical(best_repr, "tfidf_lsa")) {
    " (apenas TF‑IDF+LSA)"
  } else if (identical(best_repr, "embeddings")) {
    " (apenas embeddings GloVe)"
  } else {
    ""
  }

  html_note(
    "Mantendo a família Random Forest fixa, a representação com menor RMSE de teste foi <b>",
    best_repr, "</b>", extra_obs,
    ". Por isso, adotamos essa representação como padrão na comparação entre famílias de modelos."
  )

  repr_spread <- repr_rmse %>%
    summarise(max_rel = max(rel_delta, na.rm = TRUE))

  html_p(paste0(
    "As diferenças relativas de RMSE entre representações textuais ficam abaixo de ",
    scales::percent(repr_spread$max_rel, accuracy = 0.01),
    ", indicando que a escolha fina da representação tem impacto marginal no desempenho, frente às variáveis estruturadas e ao ruído do alvo."
  ))

  ## 3.5 receita de pré-processamento
  html_h3("3.5 Receita de pré-processamento (build_recipe)")

  html_p(paste0(
    "A função <code>build_recipe()</code> remove IDs e texto bruto, extrai ano/mês da data, imputa faltantes numéricos com mediana, ",
    "cria dummies one‑hot para variáveis categóricas, marca níveis novos em dados futuros (<code>step_novel</code>), ",
    "remove preditores de variância quase zero e normaliza preditores numéricos, garantindo entrada consistente para Elastic Net, Random Forest e XGBoost."
  ))

  # 4. Modelagem supervisionada 

  html_h2("4. Modelagem supervisionada – predição de salário")

  supervised_cv_results   <- res$supervised_cv_results
  supervised_test_results <- res$supervised_test_results
  split_info              <- res$split_info
  pred_bins_xgb           <- res$pred_xgb_hybrid_bins

  html_h3("4.1 Configuração de treino, validação e teste")

  n_train <- split_info$n[split_info$set == "train"]
  n_test  <- split_info$n[split_info$set == "test"]
  prop_train <- split_info$prop[split_info$set == "train"]
  prop_test  <- split_info$prop[split_info$set == "test"]

  html_p(paste0(
    "O alvo é <code>salary_log</code>. Usamos split treino/teste estratificado em <code>salary_log</code>, com ",
    n_train, " observações (", scales::percent(prop_train, accuracy = 0.1),
    ") no treino e ", n_test, " (", scales::percent(prop_test, accuracy = 0.1),
    ") no teste. No treino, aplicamos validação cruzada em 5 folds. ",
    "Foram comparados três modelos sobre as features híbridas: Elastic Net, Random Forest e XGBoost."
  ))

  cv_examples <- supervised_cv_results %>%
    group_by(model, .metric) %>%
    slice_min(mean, n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    mutate(
      params = purrr::pmap_chr(
        list(
          penalty, mixture, mtry, trees, min_n,
          tree_depth, learn_rate, loss_reduction, sample_size
        ),
        ~ {
          vals <- c(
            if (!is.na(..1)) sprintf("penalty=%.4g", ..1) else NULL,
            if (!is.na(..2)) sprintf("mixture=%.3f", ..2) else NULL,
            if (!is.na(..3)) sprintf("mtry=%d",    ..3) else NULL,
            if (!is.na(..4)) sprintf("trees=%d",   ..4) else NULL,
            if (!is.na(..5)) sprintf("min_n=%d",   ..5) else NULL,
            if (!is.na(..6)) sprintf("depth=%d",   ..6) else NULL,
            if (!is.na(..7)) sprintf("lr=%.4f",    ..7) else NULL,
            if (!is.na(..8)) sprintf("gamma=%.3f", ..8) else NULL,
            if (!is.na(..9)) sprintf("sample=%.3f",..9) else NULL
          )
          if (length(vals) == 0L) "" else paste(vals, collapse = ", ")
        }
      )
    ) %>%
    select(model, .metric, mean, std_err, n, params)

  html_table(
    cv_examples,
    "Melhor combinação de hiperparâmetros por modelo (validação cruzada, features híbridas)",
    digits = 4
  )

  cv_summary <- supervised_cv_results %>%
    group_by(model, .metric) %>%
    slice_min(mean, n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    transmute(
      model,
      .metric,
      mean_cv = mean,
      se_cv   = std_err
    )

  html_table(
    cv_summary,
    "Métricas de validação cruzada (melhor combinação de hiperparâmetros por modelo)",
    digits = 6
  )

  p_models_cv <- ggplot(
    cv_summary,
    aes(x = model, y = mean_cv, fill = .metric)
  ) +
    geom_col(position = "dodge") +
    labs(
      title = "Validação cruzada – RMSE e MAE por modelo (melhor configuração)",
      x = "Modelo",
      y = "Média da métrica em CV",
      fill = "Métrica"
    )

  fname_models_cv <- "sup_models_cv_metrics.png"
  save_gg(p_models_cv, fname_models_cv, dir = out_dir)
  cat("<p><b>Gráfico – Validação cruzada: métricas por modelo (melhor combinação)</b><br/><img src='", fname_models_cv, "'></p>\n", sep = "")

  cv_rmse_summary <- cv_summary %>%
    filter(.metric == "rmse")

  if (nrow(cv_rmse_summary) > 0) {
    html_p(
      "Na validação cruzada, os três modelos apresentam RMSE muito próximos quando avaliados na melhor combinação de hiperparâmetros. ",
      "As diferenças estão na segunda/terceira casa decimal e são da mesma ordem da incerteza (erro padrão) das métricas por fold."
    )
  }

  html_h3("4.2 Desempenho no conjunto de teste")

  html_table(
    supervised_test_results %>% head(10),
    "Métricas no conjunto de teste (features híbridas) – até 10 linhas",
    digits = 4
  )

  p_models_test <- ggplot(
    supervised_test_results,
    aes(x = model, y = .estimate, fill = .metric)
  ) +
    geom_col(position = "dodge") +
    labs(
      title = "Teste – RMSE e MAE por modelo",
      x = "Modelo",
      y = "Valor da métrica",
      fill = "Métrica"
    )

  fname_models_test <- "sup_models_test_metrics.png"
  save_gg(p_models_test, fname_models_test, dir = out_dir)
  cat("<p><b>Figura 10 – Desempenho em teste por modelo</b><br/><img src='", fname_models_test, "'></p>\n", sep = "")

  test_summary <- supervised_test_results %>%
    select(model, .metric, .estimate) %>%
    tidyr::pivot_wider(
      names_from = .metric,
      values_from = .estimate
    )

  if (nrow(test_summary) > 0) {
    html_p(paste0(
      "No conjunto de teste, os valores de RMSE/MAE (em log‑salário) por modelo são: ",
      paste(
        sprintf(
          "%s – RMSE ≈ %.3f, MAE ≈ %.3f",
          test_summary$model,
          test_summary$rmse,
          test_summary$mae
        ),
        collapse = "; "
      ),
      ". As diferenças absolutas são pequenas, sugerindo que a qualidade das features e o ruído do alvo dominam o desempenho."
    ))
  }

  test_rmse <- supervised_test_results %>%
    filter(.metric == "rmse") %>%
    arrange(.estimate)

  best_row   <- test_rmse %>% slice(1)
  best_model <- best_row$model[1]
  best_rmse  <- best_row$.estimate[1]
  rmse_mult  <- exp(best_rmse)
  rel_err    <- rmse_mult - 1

  html_note(
    "No teste, o menor RMSE em log foi do modelo <b>", best_model,
    "</b> (RMSE ≈ ", sprintf("%.3f", best_rmse),
    "). Isso implica um erro multiplicativo típico em torno de <b>",
    sprintf("%.2f×</b>", rmse_mult),
    " no salário previsto (erro relativo típico ≈ ",
    scales::percent(rel_err, accuracy = 1),
    ")."
  )

  # 4.3 "Matriz de confusão" em faixas de salário (regressão) --------------

  html_h3("4.3 Coerência entre salários reais e previstos")

  if (!is.null(pred_bins_xgb) && nrow(pred_bins_xgb) > 0) {
    p_cm_reg <- ggplot(
      pred_bins_xgb,
      aes(x = factor(y_pred_bin), y = factor(y_true_bin), fill = prop)
    ) +
      geom_tile() +
      geom_text(aes(label = scales::percent(prop, accuracy = 1))) +
      scale_fill_gradient(low = "white", high = "black") +
      labs(
        title = "Matriz de confusão em faixas de salário (modelo final – quintis de log-salário)",
        x = "Faixa prevista (quintil)",
        y = "Faixa real (quintil)",
        fill = "Proporção\nna faixa real"
      )

    fname_cm_reg <- "sup_regression_quintile_confusion.png"
    save_gg(p_cm_reg, fname_cm_reg, dir = out_dir)
    cat("<p><b>Gráfico – \"Matriz de confusão\" em faixas de salário (modelo final)</b><br/><img src='", fname_cm_reg, "'></p>\n", sep = "")

    html_p(paste0(
      "A matriz acima discretiza salários reais e previstos em quintis de <code>salary_log</code>. ",
      "Concentração na diagonal indica que o modelo tende a colocar a vaga na faixa correta ou adjacente."
    ))
  } else {
    html_p("Não foi possível construir a matriz de confusão em faixas de salário para o modelo final (objeto ausente ou vazio).")
  }

  # 5. Análise não supervisionada 

  html_h2("5. Análise não supervisionada – clusters e tópicos")

  pc_df                <- res$pc_df
  silhouette_k_df      <- res$silhouette_k_df
  best_k               <- res$best_k
  kmeans_global_stats  <- res$kmeans_global_stats
  pca_scores_df        <- res$pca_scores_df
  cluster_profiles     <- res$cluster_profiles
  top_ind              <- res$top_industries_by_cluster
  top_titles           <- res$top_titles_by_cluster
  spec_profiles        <- res$spec_profiles
  mean_sil_spec        <- res$mean_sil_spec
  lda_terms_short      <- res$lda_terms_short
  lda_topic_sizes      <- res$lda_topic_sizes
  lda_k                <- res$lda_k
  lda_search           <- res$lda_search
  mds_skills_df        <- res$mds_skills_df
  unsup_method_summary <- res$unsup_method_summary
  cluster_assignments  <- res$cluster_assignments
  topic_assignments    <- res$topic_assignments
  df_clusters          <- res$df_clusters
  pca_num_pcs_kmeans   <- res$pca_num_pcs_kmeans
  pca_var_expl_kmeans  <- res$pca_var_expl_kmeans
  cm_kmeans_spectral   <- res$cm_kmeans_spectral
  cm_cluster_topic     <- res$cm_cluster_topic

  topic_labels <- lda_terms_short %>%
    group_by(topic) %>%
    slice_max(beta, n = 5, with_ties = FALSE) %>%
    summarise(
      label = paste(term, collapse = ", "),
      .groups = "drop"
    ) %>%
    mutate(
      topic_label = paste0("T", topic, ": ", label)
    )

  ## 5.1 PCA + k-means
  html_h3("5.1 PCA + k‑means sobre features híbridas")

  pca_scree <- ggplot(
    pc_df,
    aes(x = pc, y = cum_var)
  ) +
    geom_line() +
    geom_point() +
    geom_hline(yintercept = 0.9, linetype = "dashed") +
    scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    labs(
      title = "PCA – variância acumulada explicada",
      x = "Componente principal",
      y = "Variância acumulada"
    )

  fname_scree <- "unsup_pca_scree.png"
  save_gg(pca_scree, fname_scree, dir = out_dir)
  cat("<p><b>Figura 11 – Scree plot da PCA</b><br/><img src='", fname_scree, "'></p>\n", sep = "")

  p_sil <- ggplot(
    silhouette_k_df,
    aes(x = k, y = mean_silhouette)
  ) +
    geom_line() +
    geom_point() +
    geom_vline(xintercept = best_k, linetype = "dashed") +
    labs(
      title = "k‑means – silhouette médio por k",
      x = "k",
      y = "Silhouette médio"
    )

  fname_sil <- "unsup_kmeans_silhouette.png"
  save_gg(p_sil, fname_sil, dir = out_dir)
  cat("<p><b>Figura 12 – Silhouette médio por número de clusters</b><br/><img src='", fname_sil, "'></p>\n", sep = "")

  html_table(
    silhouette_k_df %>% head(10),
    "Silhouette médio por k – até 10 linhas",
    digits = 3
  )

  bss_ratio <- kmeans_global_stats$betweenss_ratio[1]
  bss_pct   <- scales::percent(bss_ratio, accuracy = 0.1)
  var_pct_kmeans <- scales::percent(pca_var_expl_kmeans, accuracy = 0.1)
  sil_best <- silhouette_k_df %>%
    filter(k == best_k) %>%
    pull(mean_silhouette)
  sil_best <- sil_best[1]

  html_p(paste0(
    "Aplicamos PCA às features híbridas e rodamos k‑means nos scores das primeiras ",
    pca_num_pcs_kmeans,
    " componentes, que explicam aproximadamente ",
    var_pct_kmeans,
    " da variância total. ",
    "O k foi escolhido pelo maior silhouette médio: <b>k = ", best_k,
    "</b> (silhouette médio ≈ ", sprintf("%.3f", sil_best), "). ",
    "A razão <code>betweenss / totss</code> é ", bss_pct, "."
  ))

  html_table(
    kmeans_global_stats,
    "Estatísticas globais do k‑means para o k selecionado",
    digits = 3
  )

  p_pca_clusters <- ggplot(
    pca_scores_df,
    aes(x = PC1, y = PC2, color = cluster_kmeans)
  ) +
    geom_point(alpha = 0.6) +
    labs(
      title = "Clusters de vagas no espaço PCA (k‑means)",
      x = "PC1",
      y = "PC2",
      color = "Cluster"
    )

  fname_pca_clusters <- "unsup_pca_clusters.png"
  save_gg(p_pca_clusters, fname_pca_clusters, dir = out_dir)
  cat("<p><b>Figura 13 – Clusters no espaço PCA</b><br/><img src='", fname_pca_clusters, "'></p>\n", sep = "")

  html_table(
    cluster_profiles %>% head(10),
    "Perfil salarial por cluster (k‑means) – até 10 linhas",
    digits = 0
  )

  if (nrow(cluster_profiles) > 0) {
    ylim_min <- min(cluster_profiles$salary_p25, na.rm = TRUE) * 0.9
    ylim_max <- max(cluster_profiles$salary_p75, na.rm = TRUE) * 1.1

    p_cluster_salary <- ggplot(
      cluster_profiles,
      aes(x = cluster_kmeans, y = salary_med)
    ) +
      geom_col() +
      geom_errorbar(
        aes(ymin = salary_p25, ymax = salary_p75),
        width = 0.2
      ) +
      scale_y_continuous(
        labels = scales::dollar_format(prefix = "US$"),
        limits = c(ylim_min, ylim_max)
      ) +
      labs(
        title = "Perfil salarial por cluster (k‑means)",
        x = "Cluster (k‑means)",
        y = "Salário anual (mediana e intervalo interquartil)"
      )

    fname_cluster_salary <- "unsup_kmeans_cluster_salary.png"
    save_gg(p_cluster_salary, fname_cluster_salary, dir = out_dir)
    cat("<p><b>Gráfico – Perfil salarial por cluster (k‑means)</b><br/><img src='", fname_cluster_salary, "'></p>\n", sep = "")
  }

  html_table(
    top_ind %>% head(10),
    "Top indústrias por cluster (k‑means) – até 10 linhas",
    digits = 0
  )

  if (nrow(top_ind) > 0) {
    p_top_ind <- ggplot(
      top_ind,
      aes(
        x = tidytext::reorder_within(industry, n, cluster_kmeans),
        y = n
      )
    ) +
      geom_col() +
      coord_flip() +
      tidytext::scale_x_reordered() +
      facet_wrap(~ cluster_kmeans, scales = "free_y") +
      labs(
        title = "Top indústrias por cluster (k‑means)",
        x = "Indústria",
        y = "Número de vagas"
      )

    fname_top_ind <- "unsup_kmeans_top_industries.png"
    save_gg(p_top_ind, fname_top_ind, dir = out_dir)
    cat("<p><b>Gráfico – Top indústrias por cluster</b><br/><img src='", fname_top_ind, "'></p>\n", sep = "")
  }

  html_table(
    top_titles %>% head(10),
    "Top títulos por cluster (k‑means) – até 10 linhas",
    digits = 0
  )

  if (nrow(top_titles) > 0) {
    p_top_titles <- ggplot(
      top_titles,
      aes(
        x = tidytext::reorder_within(job_title, n, cluster_kmeans),
        y = n
      )
    ) +
      geom_col() +
      coord_flip() +
      tidytext::scale_x_reordered() +
      facet_wrap(~ cluster_kmeans, scales = "free_y") +
      labs(
        title = "Top títulos por cluster (k‑means)",
        x = "Título",
        y = "Número de vagas"
      )

    fname_top_titles <- "unsup_kmeans_top_titles.png"
    save_gg(p_top_titles, fname_top_titles, dir = out_dir)
    cat("<p><b>Gráfico – Top títulos por cluster</b><br/><img src='", fname_top_titles, "'></p>\n", sep = "")
  }

  html_table(
    cluster_assignments %>% head(10),
    "Exemplos de atribuições de vagas a clusters (k‑means e espectral) – até 10 linhas",
    digits = 0
  )

  cluster_counts <- cluster_assignments %>%
    count(cluster_kmeans)

  p_cluster_counts <- ggplot(
    cluster_counts,
    aes(x = cluster_kmeans, y = n)
  ) +
    geom_col() +
    labs(
      title = "Distribuição de vagas por cluster (k‑means)",
      x = "Cluster (k‑means)",
      y = "Número de vagas"
    )

  fname_cluster_counts <- "unsup_kmeans_cluster_sizes.png"
  save_gg(p_cluster_counts, fname_cluster_counts, dir = out_dir)
  cat("<p><b>Gráfico – Distribuição de vagas por cluster (k‑means)</b><br/><img src='", fname_cluster_counts, "'></p>\n", sep = "")

  if (nrow(cluster_profiles) > 0) {
    cluster_salary_order <- cluster_profiles %>%
      arrange(salary_med)
    cluster_low  <- cluster_salary_order$cluster_kmeans[1]
    cluster_high <- cluster_salary_order$cluster_kmeans[nrow(cluster_salary_order)]
    med_low      <- cluster_salary_order$salary_med[1]
    med_high     <- cluster_salary_order$salary_med[nrow(cluster_salary_order)]

    cluster_exp_profiles <- df_clusters %>%
      group_by(cluster_kmeans, experience_level) %>%
      summarise(n = n(), .groups = "drop") %>%
      group_by(cluster_kmeans) %>%
      mutate(prop = n / sum(n)) %>%
      arrange(cluster_kmeans, desc(prop))

    dominant_exp_high <- cluster_exp_profiles %>%
      filter(cluster_kmeans == cluster_high) %>%
      slice_max(prop, n = 1, with_ties = FALSE)

    dominant_exp_low <- cluster_exp_profiles %>%
      filter(cluster_kmeans == cluster_low) %>%
      slice_max(prop, n = 1, with_ties = FALSE)

    html_p(paste0(
      "Os clusters exibem perfis salariais distintos: a mediana de <code>salary_mid</code> varia de ",
      scales::dollar(med_low,  prefix = "US$"), " (cluster ", cluster_low,
      ") a ", scales::dollar(med_high, prefix = "US$"), " (cluster ", cluster_high, "). ",
      "No cluster de maior mediana, o nível de experiência mais frequente é <code>",
      dominant_exp_high$experience_level, "</code> (",
      scales::percent(dominant_exp_high$prop, accuracy = 1), "); no cluster de menor mediana, domina <code>",
      dominant_exp_low$experience_level, "</code> (",
      scales::percent(dominant_exp_low$prop, accuracy = 1), ")."
    ))

    cluster_topic_dist <- df_clusters %>%
      left_join(topic_assignments, by = "job_id") %>%
      filter(!is.na(dominant_topic)) %>%
      group_by(cluster_kmeans, dominant_topic) %>%
      summarise(n = n(), .groups = "drop") %>%
      group_by(cluster_kmeans) %>%
      mutate(prop = n / sum(n)) %>%
      slice_max(prop, n = 1, with_ties = FALSE) %>%
      ungroup() %>%
      left_join(topic_labels, by = c("dominant_topic" = "topic"))

    cluster_industry_main <- top_ind %>%
      group_by(cluster_kmeans) %>%
      slice_max(n, n = 1, with_ties = FALSE) %>%
      ungroup() %>%
      select(cluster_kmeans, main_industry = industry)

    cluster_title_main <- top_titles %>%
      group_by(cluster_kmeans) %>%
      slice_max(n, n = 1, with_ties = FALSE) %>%
      ungroup() %>%
      select(cluster_kmeans, main_title = job_title)

    cluster_narrative <- cluster_profiles %>%
      left_join(cluster_industry_main, by = "cluster_kmeans") %>%
      left_join(cluster_title_main,   by = "cluster_kmeans") %>%
      left_join(
        cluster_topic_dist %>%
          select(cluster_kmeans, dominant_topic, prop, topic_label),
        by = "cluster_kmeans"
      ) %>%
      arrange(cluster_kmeans)

    for (i in seq_len(nrow(cluster_narrative))) {
      row <- cluster_narrative[i, ]
      html_note(
        "Cluster ", as.character(row$cluster_kmeans), ": mediana salarial ",
        scales::dollar(row$salary_med, prefix = "US$"),
        " (P25–P75: ",
        scales::dollar(row$salary_p25, prefix = "US$"), " – ",
        scales::dollar(row$salary_p75, prefix = "US$"), "); ",
        "indústria mais comum: <code>", row$main_industry, "</code>; ",
        "título mais recorrente: <code>", row$main_title, "</code>; ",
        "tópico LDA dominante: <code>", row$topic_label,
        "</code> (≈ ", scales::percent(row$prop, accuracy = 1), ")."
      )
    }
  }

  ## 5.2 Clustering espectral em embeddings
  html_h3("5.2 Clustering espectral em embeddings")

  if (!is.null(spec_profiles)) {
    html_table(
      spec_profiles %>% head(10),
      "Perfil salarial por cluster (clustering espectral) – até 10 linhas",
      digits = 0
    )

    if (nrow(spec_profiles) > 0) {
      ylim_min_spec <- min(spec_profiles$salary_p25, na.rm = TRUE) * 0.9
      ylim_max_spec <- max(spec_profiles$salary_p75, na.rm = TRUE) * 1.1

      p_spec_salary <- ggplot(
        spec_profiles,
        aes(x = cluster_spectral, y = salary_med)
      ) +
        geom_col() +
        geom_errorbar(
          aes(ymin = salary_p25, ymax = salary_p75),
          width = 0.2
        ) +
        scale_y_continuous(
          labels = scales::dollar_format(prefix = "US$"),
          limits = c(ylim_min_spec, ylim_max_spec)
        ) +
        labs(
          title = "Perfil salarial por cluster (clustering espectral)",
          x = "Cluster espectral",
          y = "Salário anual (mediana e intervalo interquartil)"
        )

      fname_spec_salary <- "unsup_spectral_cluster_salary.png"
      save_gg(p_spec_salary, fname_spec_salary, dir = out_dir)
      cat("<p><b>Gráfico – Perfil salarial por cluster (clustering espectral)</b><br/><img src='", fname_spec_salary, "'></p>\n", sep = "")
    }

    if (is.finite(mean_sil_spec)) {
      html_note(
        "O clustering espectral nos embeddings apresentou silhouette médio ≈ ",
        sprintf("%.3f", mean_sil_spec),
        ", sugerindo que estruturas não esféricas podem ser capturadas de forma um pouco mais nítida do que pelo k‑means em PCA. ",
        "Ainda assim, os valores seguem baixos, indicando fronteiras difusas."
      )
    }
  } else {
    html_p("O clustering espectral não pôde ser estimado de forma estável neste conjunto.")
  }

  ## 5.3 Tópicos de skills (LDA)
  html_h3("5.3 Tópicos de skills (LDA)")

  html_table(
    lda_search %>% arrange(k) %>% head(10),
    "Busca de número de tópicos (k) por log-verossimilhança – até 10 linhas",
    digits = 0
  )

  p_lda_ll <- ggplot(
    lda_search,
    aes(x = k, y = loglik)
  ) +
    geom_line() +
    geom_point() +
    labs(
      title = "LDA em skills_required – log-verossimilhança por k",
      x = "Número de tópicos (k)",
      y = "Log-verossimilhança"
    )

  fname_lda_ll <- "unsup_lda_k_loglik.png"
  save_gg(p_lda_ll, fname_lda_ll, dir = out_dir)
  cat("<p><b>Gráfico – Escolha de k em LDA por log-verossimilhança</b><br/><img src='", fname_lda_ll, "'></p>\n", sep = "")

  best_lda_row <- lda_search %>% slice_max(loglik, n = 1, with_ties = FALSE)
  html_note(
    "Entre os k testados, o melhor valor foi <code>k = ", best_lda_row$k,
    "</code> (loglik ≈ ", sprintf("%.0f", best_lda_row$loglik), ")."
  )

  p_lda <- ggplot(
    lda_terms_short,
    aes(x = tidytext::reorder_within(term, beta, topic), y = beta)
  ) +
    geom_col() +
    coord_flip() +
    tidytext::scale_x_reordered() +
    facet_wrap(~ topic, scales = "free_y") +
    labs(
      title = paste("LDA em skills_required – tópicos (k =", lda_k, ")"),
      x = "Skill",
      y = "Probabilidade no tópico"
    )

  fname_lda <- "unsup_lda_top_terms.png"
  save_gg(p_lda, fname_lda, dir = out_dir)
  cat("<p><b>Figura 14 – Tópicos de skills (LDA)</b><br/><img src='", fname_lda, "'></p>\n", sep = "")

  html_table(
    lda_terms_short %>% head(10),
    "Exemplos de termos com maior peso em cada tópico (até 10 linhas)",
    digits = 3
  )

  lda_topic_sizes_labeled <- lda_topic_sizes %>%
    left_join(topic_labels, by = c("dominant_topic" = "topic"))

  html_table(
    lda_topic_sizes_labeled %>% head(10),
    "Número de vagas por tópico dominante (skills_required) – até 10 linhas",
    digits = 0
  )

  p_lda_sizes <- ggplot(
    lda_topic_sizes_labeled,
    aes(x = factor(dominant_topic), y = n_jobs)
  ) +
    geom_col() +
    labs(
      title = "Número de vagas por tópico dominante",
      x = "Tópico dominante",
      y = "Número de vagas"
    )

  fname_lda_sizes <- "unsup_lda_topic_sizes.png"
  save_gg(p_lda_sizes, fname_lda_sizes, dir = out_dir)
  cat("<p><b>Gráfico – Tamanho dos tópicos (número de vagas)</b><br/><img src='", fname_lda_sizes, "'></p>\n", sep = "")

  html_table(
    topic_assignments %>% head(10),
    "Exemplos de atribuição de vagas ao tópico dominante – até 10 linhas",
    digits = 0
  )

  topic_summaries <- topic_labels %>%
    transmute(topic, stack = label)

  topic_examples_text <- paste0(
    "Tópico ", topic_summaries$topic, ": ",
    topic_summaries$stack
  )
  topic_examples_str <- paste(topic_examples_text, collapse = " ; ")

  html_p(paste0(
    "O LDA foi ajustado sobre <code>skills_required</code>. Cada tópico corresponde a um stack recorrente; por exemplo: ",
    topic_examples_str,
    "."
  ))

  ## 5.4 mapa MDS de skills
  html_h3("5.4 Mapa de skills (MDS)")

  if (nrow(mds_skills_df) > 0) {
    p_mds <- ggplot(
      mds_skills_df,
      aes(x = x, y = y, label = skill)
    ) +
      geom_point(alpha = 0.4) +
      ggrepel::geom_text_repel(size = 3) +
      labs(
        title = "Mapa 2D de skills (MDS)",
        x = NULL,
        y = NULL
      )

    fname_mds <- "unsup_skills_mds.png"
    save_gg(p_mds, fname_mds, dir = out_dir)
    cat("<p><b>Figura 15 – Mapa de proximidade entre skills</b><br/><img src='", fname_mds, "'></p>\n", sep = "")

    html_p(paste0(
      "O MDS em cima da matriz de coocorrência gera um mapa em que habilidades próximas tendem a aparecer juntas em vagas semelhantes. ",
      "Agrupamentos visuais indicam stacks que coocorrem com frequência (BI, ML clássico, DL, MLOps etc.)."
    ))
  } else {
    html_p("Número insuficiente de skills frequentes para construir um mapa MDS informativo.")
  }

  ## 5.5 resumo geral
  html_h3("5.5 Comparação geral de métodos não supervisionados")

  html_table(
    unsup_method_summary %>% head(10),
    "Resumo de métodos não supervisionados – até 10 linhas",
    digits = 3
  )

  p_unsup_summary <- ggplot(
    unsup_method_summary %>% dplyr::filter(!is.na(mean_silhouette)),
    aes(x = method, y = mean_silhouette)
  ) +
    geom_col() +
    labs(
      title = "Comparação de métodos não supervisionados por silhouette médio",
      x = "Método",
      y = "Silhouette médio"
    )

  fname_unsup_summary <- "unsup_methods_silhouette_summary.png"
  save_gg(p_unsup_summary, fname_unsup_summary, dir = out_dir)
  cat("<p><b>Gráfico – Comparação entre métodos não supervisionados</b><br/><img src='", fname_unsup_summary, "'></p>\n", sep = "")

  html_p(
    "Os valores de silhouette médio são modestos, indicando estrutura apenas moderadamente definida: há tendência a formação de grupos, ",
    "mas com fronteiras difusas. Na prática, os clusters organizam o espaço de vagas em segmentos interpretáveis, não classes rígidas."
  )

  ## 5.6 Consistência entre métodos de clustering --------------------------

  html_h3("5.6 Consistência entre métodos de clustering e tópicos")

  if (!is.null(cm_kmeans_spectral) && nrow(cm_kmeans_spectral) > 0) {
    p_cm_ks <- ggplot(
      cm_kmeans_spectral,
      aes(x = cluster_spectral, y = cluster_kmeans, fill = prop)
    ) +
      geom_tile() +
      geom_text(aes(label = scales::percent(prop, accuracy = 1))) +
      scale_fill_gradient(low = "white", high = "black") +
      labs(
        title = "Matriz de \"confusão\" entre k-means e clustering espectral",
        x = "Cluster espectral",
        y = "Cluster k-means",
        fill = "Proporção\nno k-means"
      )

    fname_cm_ks <- "unsup_cm_kmeans_spectral.png"
    save_gg(p_cm_ks, fname_cm_ks, dir = out_dir)
    cat("<p><b>Gráfico – Heatmap k-means × clustering espectral</b><br/><img src='", fname_cm_ks, "'></p>\n", sep = "")

    html_p(
      "As proporções são normalizadas por linha (dentro de cada cluster do k‑means). Concentração em poucas células sugere alta consistência."
    )
  }

  if (!is.null(cm_cluster_topic) && nrow(cm_cluster_topic) > 0) {
    p_cm_ct <- ggplot(
      cm_cluster_topic,
      aes(x = factor(dominant_topic), y = cluster_kmeans, fill = prop)
    ) +
      geom_tile() +
      geom_text(aes(label = scales::percent(prop, accuracy = 1))) +
      scale_fill_gradient(low = "white", high = "black") +
      labs(
        title = "Distribuição de tópicos LDA dentro de cada cluster (k-means)",
        x = "Tópico dominante",
        y = "Cluster k-means",
        fill = "Proporção\nno cluster"
      )

    fname_cm_ct <- "unsup_cm_cluster_topic.png"
    save_gg(p_cm_ct, fname_cm_ct, dir = out_dir)
    cat("<p><b>Gráfico – Heatmap cluster × tópico LDA</b><br/><img src='", fname_cm_ct, "'></p>\n", sep = "")

    html_p(
      "O heatmap cluster × tópico LDA mostra quais stacks de skills são mais frequentes em cada cluster salarial."
    )
  }

  # 6. Sistema de recomendação baseado em conteúdo 

  html_h2("6. Sistema de recomendação baseado em conteúdo")

  html_p(
    "Implementamos um sistema de recomendação <i>content‑based</i> para recomendar vagas semelhantes a um perfil de usuário ",
    "ou a uma vaga específica."
  )
  html_p(
    "Fluxo: representar vagas por TF‑IDF normalizado; gerar vetor do usuário; calcular similaridade cosseno; aplicar filtros de negócio; ",
    "e, opcionalmente, usar MMR para balancear relevância e diversidade."
  )

  recs_similar_view <- res$recs_similar_view
  recs_user_view    <- res$recs_user_view
  user_profile_label <- res$user_profile_label

  html_h3("6.1 Vagas similares a uma vaga base")

  html_table(
    recs_similar_view %>%
      select(
        rank, job_id, company_name, job_title, industry,
        experience_level, employment_type, location,
        salary_range_usd, score
      ) %>%
      head(10),
    "Exemplo de recomendações similares a uma vaga real – até 10 linhas",
    digits = 3
  )

  if (nrow(recs_similar_view) > 0) {
    p_recs_similar <- ggplot(
      recs_similar_view,
      aes(x = rank, y = score)
    ) +
      geom_line() +
      geom_point() +
      labs(
        title = "Recomendações similares – score por posição no ranking",
        x = "Posição (rank)",
        y = "Score de similaridade"
      )

    fname_recs_similar <- "recs_similar_scores.png"
    save_gg(p_recs_similar, fname_recs_similar, dir = out_dir)
    cat("<p><b>Gráfico – Score das recomendações similares</b><br/><img src='", fname_recs_similar, "'></p>\n", sep = "")

    first_score <- recs_similar_view$score[1]
    last_score  <- recs_similar_view$score[nrow(recs_similar_view)]
    html_p(paste0(
      "O score é a similaridade cosseno entre descrições. No exemplo, a primeira recomendação tem score ≈ ",
      sprintf("%.3f", first_score),
      " e a última do top‑", nrow(recs_similar_view), " tem score ≈ ", sprintf("%.3f", last_score), "."
    ))
  }

  html_h3("6.2 Vagas recomendadas para um perfil de usuário")

  if (nrow(recs_user_view) > 0) {
    html_table(
      recs_user_view %>%
        select(
          rank, job_id, company_name, job_title, industry,
          experience_level, employment_type, location,
          salary_range_usd, score
        ) %>%
        head(10),
      paste0("Exemplo de recomendações para um perfil: ", user_profile_label, " – até 10 linhas"),
      digits = 3
    )

    p_recs_user <- ggplot(
      recs_user_view,
      aes(x = rank, y = score)
    ) +
      geom_line() +
      geom_point() +
      labs(
        title = "Recomendações para o perfil – score por posição",
        x = "Posição (rank)",
        y = "Score de similaridade"
      )

    fname_recs_user <- "recs_user_profile_scores.png"
    save_gg(p_recs_user, fname_recs_user, dir = out_dir)
    cat("<p><b>Gráfico – Score das recomendações para o perfil</b><br/><img src='", fname_recs_user, "'></p>\n", sep = "")

    first_score_u <- recs_user_view$score[1]
    last_score_u  <- recs_user_view$score[nrow(recs_user_view)]
    html_p(paste0(
      "Para o perfil configurado, o score da primeira vaga recomendada é ≈ ",
      sprintf("%.3f", first_score_u),
      " e o da última do top‑", nrow(recs_user_view), " é ≈ ", sprintf("%.3f", last_score_u), "."
    ))
  } else {
    html_note(
      "No exemplo de perfil configurado, nenhum resultado atendeu simultaneamente a todos os filtros de negócio. ",
      "O sistema trata esse caso retornando lista vazia (ou relaxando localização quando permitido)."
    )
  }

  html_note(
    "Para reduzir risco de consumo excessivo de memória, aplicamos uma prefiltragem (pool_top) antes do MMR: ",
    "o MMR roda apenas sobre os candidatos mais relevantes (ex.: top 300–1000 conforme top_k)."
  )

  # 7. Conclusões ----------------------------------------------------------

  html_h2("7. Conclusões, limitações e trabalhos futuros")

  html_h3("7.1 Conclusões sobre o modelo supervisionado")

  obs_sd_log <- sd(df_model_base$salary_log, na.rm = TRUE)

  test_summary_final <- supervised_test_results %>%
    select(model, .metric, .estimate) %>%
    tidyr::pivot_wider(
      names_from = .metric,
      values_from = .estimate
    )

  if (nrow(test_summary_final) > 0) {
    html_p(paste0(
      "No teste, os modelos apresentaram RMSE/MAE (em log‑salário): ",
      paste(
        sprintf(
          "%s: RMSE ≈ %.3f, MAE ≈ %.3f",
          test_summary_final$model,
          test_summary_final$rmse,
          test_summary_final$mae
        ),
        collapse = "; "
      ),
      ". O modelo final (menor RMSE) foi <b>", best_model,
      "</b>. O RMSE corresponde a erro multiplicativo típico ≈ ",
      sprintf("%.2f×", rmse_mult),
      " (erro relativo ≈ ", scales::percent(rel_err, accuracy = 1),
      "), comparável à dispersão observada de <code>salary_log</code> (DP ≈ ",
      sprintf("%.3f", obs_sd_log), ")."
    ))
  }

  html_h3("7.2 Insights não supervisionados e estrutura do mercado")

  if (nrow(cluster_profiles) > 0) {
    cluster_salary_order <- cluster_profiles %>%
      arrange(salary_med)
    cluster_low  <- cluster_salary_order$cluster_kmeans[1]
    cluster_high <- cluster_salary_order$cluster_kmeans[nrow(cluster_salary_order)]
    med_low      <- cluster_salary_order$salary_med[1]
    med_high     <- cluster_salary_order$salary_med[nrow(cluster_salary_order)]

    top_topics_for_summary <- topic_summaries %>%
      slice_head(n = min(3L, nrow(topic_summaries)))

    topic_examples_summary <- paste0(
      "Tópico ", top_topics_for_summary$topic, ": ",
      top_topics_for_summary$stack
    )
    topic_examples_summary_str <- paste(topic_examples_summary, collapse = " ; ")

    html_p(paste0(
      "A análise de clusters revela segmentos com perfis salariais relativamente distintos: a mediana de <code>salary_mid</code> varia de ",
      scales::dollar(med_low,  prefix = "US$"), " (cluster ", cluster_low,
      ") a ", scales::dollar(med_high, prefix = "US$"), " (cluster ", cluster_high, "). ",
      "Ainda assim, há ampla variação dentro de cada cluster, sugerindo um mercado mais contínuo que discretizado."
    ))

    html_p(paste0(
      "O LDA identificou stacks de skills recorrentes, como ",
      topic_examples_summary_str,
      ", úteis para sumarizar perfis técnicos e conectar tópicos às faixas salariais."
    ))
  }

  html_end(html_path)
}

# 
# Execução principal da análise
# 

DATA_PATH <- "ai_job_market.csv"

OUTPUT_DIR <- file.path(dirname(DATA_PATH), "outputs_ai_jobs")
if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

df_raw <- readr::read_csv(DATA_PATH, show_col_types = FALSE) %>%
  mutate(job_id = as.character(job_id))

if (!"company_size" %in% names(df_raw)) {
  df_raw$company_size <- NA_character_
}
if (!"tools_preferred" %in% names(df_raw)) {
  df_raw$tools_preferred <- NA_character_
}

df <- df_raw %>%
  mutate(
    job_id = as.character(job_id),
    across(
      where(is.character),
      ~ normalize_text_col(.x)
    )
  ) %>%
  mutate(
    posted_date = as.Date(posted_date)
  )

salary_mat <- t(vapply(df$salary_range_usd, parse_salary_range_one, numeric(2L)))
colnames(salary_mat) <- c("salary_min", "salary_max")

df <- df %>%
  bind_cols(as.data.frame(salary_mat)) %>%
  mutate(
    salary_mid   = rowMeans(cbind(salary_min, salary_max), na.rm = TRUE),
    salary_mid   = ifelse(is.nan(salary_mid), NA_real_, salary_mid)
  )

df <- df %>%
  mutate(
    job_title       = coalesce(job_title, ""),
    skills_required = coalesce(skills_required, ""),
    tools_preferred = coalesce(tools_preferred, ""),
    industry        = coalesce(industry, ""),
    text_blob = paste(
      job_title,
      skills_required,
      tools_preferred,
      industry,
      sep = " | "
    )
  )

# 
# (3) Diagnóstico das regras de salário no dataset completo (contagem correta)
# 

salary_rule_effects <- salary_rules_tbl %>%
  rowwise() %>%
  mutate(
    mask = list(apply_salary_rule_mask(df$salary_mid, type = type, params = params)),
    n_total    = nrow(df),
    n_non_na   = sum(!is.na(df$salary_mid)),
    n_valid_full = sum(mask[[1]]),
    n_removed_non_na = sum(!mask[[1]] & !is.na(df$salary_mid)),
    min_kept = {
      kept <- df$salary_mid[mask[[1]]]
      if (length(kept) == 0L) NA_real_ else suppressWarnings(min(kept, na.rm = TRUE))
    },
    max_kept = {
      kept <- df$salary_mid[mask[[1]]]
      if (length(kept) == 0L) NA_real_ else suppressWarnings(max(kept, na.rm = TRUE))
    }
  ) %>%
  ungroup() %>%
  select(-mask)

# 
# Tuning de salary_valid
# 

salary_rule_results <- tune_salary_rules(df, salary_rules_tbl)

ok <- !is.na(salary_rule_results$rmse)
if (!any(ok)) stop("Nenhuma regra válida de salário.")

sr_ok <- salary_rule_results[ok, , drop = FALSE]
best_ix <- which.min(sr_ok$rmse)
best_salary_rule <- dplyr::left_join(
  sr_ok[best_ix, , drop = FALSE],
  salary_rules_tbl,
  by = "rule_name"
)

best_mask <- apply_salary_rule_mask(
  df$salary_mid,
  type   = best_salary_rule$type[1],
  params = best_salary_rule$params[[1]]
)

df <- df %>%
  mutate(
    salary_valid = best_mask
  )

df_model_base <- df %>%
  filter(salary_valid, !is.na(salary_mid)) %>%
  mutate(
    salary_log = log(salary_mid),
    text_blob  = stringr::str_to_lower(text_blob)
  )

# EDA: correlação/associação com salário
salary_assoc <- compute_salary_associations(df_model_base, top_n_levels = 12L)

readr::write_csv(
  salary_assoc$numeric_corr,
  file = file.path(OUTPUT_DIR, "eda_salary_numeric_correlations.csv")
)
readr::write_csv(
  salary_assoc$cat_assoc_eta2,
  file = file.path(OUTPUT_DIR, "eda_salary_categorical_eta2.csv")
)
readr::write_csv(
  salary_assoc$exp_rank_corr,
  file = file.path(OUTPUT_DIR, "eda_salary_experience_ordinal_corr.csv")
)

# 
# TF-IDF + LSA, embeddings e híbrido
# 

tfidf_res <- build_tfidf_lsa_auto(df_model_base)

vocab             <- tfidf_res$vocab
vectorizer        <- tfidf_res$vectorizer
tfidf_transformer <- tfidf_res$tfidf_tr
dtm_tfidf_norm    <- tfidf_res$dtm_tfidf_norm
tfidf_lsa         <- tfidf_res$tfidf_lsa

df_tfidf <- df_model_base %>%
  bind_cols(as.data.frame(tfidf_lsa))

tokens_list_all <- word_tokenizer(df_model_base$text_blob)

glove_tcm_res <- build_glove_tcm(tokens_list_all, vocab, fast = FALSE)
tcm           <- glove_tcm_res$tcm
n_terms_train <- nrow(vocab)

emb_dim <- as.integer(
  max(50L, min(300L, round(log2(max(2, n_terms_train)) * 8)))
)
x_max <- as.numeric(
  max(10, min(100, round(sqrt(max(2, n_terms_train)))))
)

glove <- GlobalVectors$new(rank = emb_dim, x_max = x_max)

glove_w_main <- glove$fit_transform(
  tcm,
  n_iter          = if (n_terms_train < 5000) 50L else 30L,
  convergence_tol = 0.01
)

glove_w_ctx <- glove$components
word_vectors <- glove_w_main + t(glove_w_ctx)

doc_embeds_all <- t(vapply(
  tokens_list_all,
  FUN       = doc_embedding_one,
  FUN.VALUE = numeric(ncol(word_vectors)),
  word_vec  = word_vectors
))

colnames(doc_embeds_all) <- paste0("emb_", seq_len(ncol(doc_embeds_all)))

df_emb <- df_model_base %>%
  bind_cols(as.data.frame(doc_embeds_all))

df_hybrid <- df_model_base %>%
  bind_cols(
    as.data.frame(tfidf_lsa),
    as.data.frame(doc_embeds_all)
  )

# 
# Modelos supervisionados
# 

set.seed(123)
split_base <- initial_split(df_model_base, prop = 0.8, strata = salary_log)

train_ids <- training(split_base)$job_id
test_ids  <- testing(split_base)$job_id

train_tfidf  <- df_tfidf  %>% filter(job_id %in% train_ids)
test_tfidf   <- df_tfidf  %>% filter(job_id %in% test_ids)
train_emb    <- df_emb    %>% filter(job_id %in% train_ids)
test_emb     <- df_emb    %>% filter(job_id %in% test_ids)
train_hybrid <- df_hybrid %>% filter(job_id %in% train_ids)
test_hybrid  <- df_hybrid %>% filter(job_id %in% test_ids)

split_info <- tibble::tibble(
  set = c("train", "test"),
  n   = c(nrow(train_hybrid), nrow(test_hybrid))
) %>%
  mutate(prop = n / sum(n))

set.seed(123)
folds_tfidf  <- vfold_cv(train_tfidf,  v = 5, strata = salary_log)
set.seed(123)
folds_emb    <- vfold_cv(train_emb,    v = 5, strata = salary_log)
set.seed(123)
folds_hybrid <- vfold_cv(train_hybrid, v = 5, strata = salary_log)

elastic_spec <- linear_reg(
  penalty = tune(),
  mixture = tune()
) %>%
  set_engine("glmnet")

rf_spec <- rand_forest(
  mtry  = tune(),
  min_n = tune(),
  trees = tune()
) %>%
  set_engine("ranger") %>%
  set_mode("regression")

xgb_spec <- boost_tree(
  trees          = tune(),
  tree_depth     = tune(),
  learn_rate     = tune(),
  loss_reduction = tune(),
  sample_size    = tune(),
  mtry           = tune(),
  min_n          = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

recipe_hybrid <- build_recipe(train_hybrid, "hybrid")

set.seed(123)
res_elastic_hybrid <- tune_model(recipe_hybrid, elastic_spec, folds_hybrid)
set.seed(123)
res_rf_hybrid      <- tune_model(recipe_hybrid, rf_spec,     folds_hybrid)
set.seed(123)
res_xgb_hybrid     <- tune_model(recipe_hybrid, xgb_spec,    folds_hybrid)

metrics_elastic_hybrid <- collect_metrics(res_elastic_hybrid)
metrics_rf_hybrid      <- collect_metrics(res_rf_hybrid)
metrics_xgb_hybrid     <- collect_metrics(res_xgb_hybrid)

best_elastic_hybrid <- select_best(res_elastic_hybrid, metric = "rmse")
best_rf_hybrid      <- select_best(res_rf_hybrid,      metric = "rmse")
best_xgb_hybrid     <- select_best(res_xgb_hybrid,     metric = "rmse")

wf_elastic_hybrid <- workflow() %>%
  add_model(elastic_spec) %>%
  add_recipe(recipe_hybrid) %>%
  finalize_workflow(best_elastic_hybrid)

wf_rf_hybrid <- workflow() %>%
  add_model(rf_spec) %>%
  add_recipe(recipe_hybrid) %>%
  finalize_workflow(best_rf_hybrid)

wf_xgb_hybrid <- workflow() %>%
  add_model(xgb_spec) %>%
  add_recipe(recipe_hybrid) %>%
  finalize_workflow(best_xgb_hybrid)

fit_elastic_hybrid <- fit(wf_elastic_hybrid, data = train_hybrid)
fit_rf_hybrid      <- fit(wf_rf_hybrid,      data = train_hybrid)
fit_xgb_hybrid     <- fit(wf_xgb_hybrid,     data = train_hybrid)

pred_elastic_hybrid <- predict(fit_elastic_hybrid, test_hybrid) %>%
  bind_cols(test_hybrid %>% dplyr::select(salary_log, salary_mid)) %>%
  { assert_has_cols(., c(".pred","salary_log")); . }

pred_rf_hybrid <- predict(fit_rf_hybrid, test_hybrid) %>%
  bind_cols(test_hybrid %>% dplyr::select(salary_log, salary_mid)) %>%
  { assert_has_cols(., c(".pred","salary_log")); . }

pred_xgb_hybrid <- predict(fit_xgb_hybrid, test_hybrid) %>%
  bind_cols(test_hybrid %>% dplyr::select(salary_log, salary_mid)) %>%
  { assert_has_cols(., c(".pred","salary_log")); . }

metrics_elastic_test <- pred_elastic_hybrid %>%
  regression_metrics_safe(truth = salary_log, estimate = .pred)
metrics_rf_test <- pred_rf_hybrid %>%
  regression_metrics_safe(truth = salary_log, estimate = .pred)
metrics_xgb_test <- pred_xgb_hybrid %>%
  regression_metrics_safe(truth = salary_log, estimate = .pred)

pred_xgb_hybrid_bins <- pred_xgb_hybrid %>%
  mutate(
    y_true_bin = ntile(salary_log, 5),
    y_pred_bin = ntile(.pred,       5)
  ) %>%
  count(y_true_bin, y_pred_bin) %>%
  group_by(y_true_bin) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup()

# 
# Comparação TF-IDF vs Embeddings vs Híbrido (RF)
# 

recipe_tfidf <- build_recipe(train_tfidf, "tfidf")
recipe_emb   <- build_recipe(train_emb,   "emb")

set.seed(123)
res_rf_tfidf <- tune_model(recipe_tfidf, rf_spec, folds_tfidf)
set.seed(123)
res_rf_emb   <- tune_model(recipe_emb,   rf_spec, folds_emb)

best_rf_tfidf <- select_best(res_rf_tfidf, metric = "rmse")
best_rf_emb   <- select_best(res_rf_emb,   metric = "rmse")

wf_rf_tfidf <- workflow() %>%
  add_model(rf_spec) %>%
  add_recipe(recipe_tfidf) %>%
  finalize_workflow(best_rf_tfidf)

wf_rf_emb <- workflow() %>%
  add_model(rf_spec) %>%
  add_recipe(recipe_emb) %>%
  finalize_workflow(best_rf_emb)

fit_rf_tfidf <- fit(wf_rf_tfidf, data = train_tfidf)
fit_rf_emb   <- fit(wf_rf_emb,   data = train_emb)

pred_rf_tfidf <- predict(fit_rf_tfidf, test_tfidf) %>%
  bind_cols(test_tfidf %>% select(salary_log, salary_mid))

pred_rf_emb <- predict(fit_rf_emb, test_emb) %>%
  bind_cols(test_emb %>% select(salary_log, salary_mid))

metrics_rf_tfidf_test <- pred_rf_tfidf %>%
  regression_metrics(truth = salary_log, estimate = .pred)
metrics_rf_emb_test <- pred_rf_emb %>%
  regression_metrics(truth = salary_log, estimate = .pred)

# 
# Não supervisionado: PCA + k-means, espectral, LDA
# 

unsup_mat <- df_hybrid %>%
  select(starts_with("tfidf_lsa_"), starts_with("emb_")) %>%
  as.matrix()

unsup_sds <- apply(unsup_mat, 2, sd, na.rm = TRUE)
unsup_non_const <- which(unsup_sds > 0)
unsup_mat <- unsup_mat[, unsup_non_const, drop = FALSE]
unsup_mat_scaled <- scale(unsup_mat)

pca_full <- prcomp(unsup_mat_scaled, center = TRUE, scale. = FALSE)
pca_var <- pca_full$sdev^2
pca_cum <- cumsum(pca_var) / sum(pca_var)
pc_df <- tibble(
  pc = seq_along(pca_cum),
  cum_var = pca_cum
)

var_targets <- c(0.80, 0.90, 0.95)
num_pcs_candidates <- sapply(var_targets, function(vt) {
  idx <- which(pca_cum >= vt)[1]
  if (is.na(idx)) length(pca_cum) else idx
})
num_pcs_candidates <- unique(pmax(2L, pmin(num_pcs_candidates, ncol(pca_full$x))))
max_pcs <- min(num_pcs_candidates[num_pcs_candidates >= 2][1], ncol(pca_full$x))
if (is.na(max_pcs) || max_pcs < 2L) {
  max_pcs <- min(10L, ncol(pca_full$x))
}
pca_scores <- pca_full$x[, 1:max_pcs, drop = FALSE]

pca_num_pcs_kmeans   <- max_pcs
pca_var_expl_kmeans  <- pca_cum[max_pcs]

n_obs_unsup <- nrow(pca_scores)
k_max <- max(2L, min(20L, floor(sqrt(n_obs_unsup))))
ks <- 2:k_max
dist_pca <- dist(pca_scores)

set.seed(123)
sil_scores <- purrr::map_dbl(ks, function(k) {
  km <- kmeans(
    pca_scores,
    centers  = k,
    nstart   = 20,
    iter.max = 50
  )
  sil <- silhouette(km$cluster, dist_pca)
  mean(sil[, 3])
})

best_k <- ks[which.max(sil_scores)]

set.seed(123)
km_final <- kmeans(pca_scores, centers = best_k, nstart = 50, iter.max = 100)
df_clusters <- df_hybrid %>%
  mutate(cluster_kmeans = factor(km_final$cluster))

pca_scores_df <- as.data.frame(pca_scores[, 1:2, drop = FALSE])
colnames(pca_scores_df) <- c("PC1", "PC2")
pca_scores_df$cluster_kmeans <- df_clusters$cluster_kmeans

cluster_profiles <- df_clusters %>%
  group_by(cluster_kmeans) %>%
  summarise(
    n          = n(),
    salary_med = median(salary_mid, na.rm = TRUE),
    salary_p25 = quantile(salary_mid, 0.25, na.rm = TRUE),
    salary_p75 = quantile(salary_mid, 0.75, na.rm = TRUE),
    .groups    = "drop"
  )

top_industries_by_cluster <- df_clusters %>%
  group_by(cluster_kmeans, industry) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(cluster_kmeans) %>%
  slice_max(n, n = 5) %>%
  arrange(cluster_kmeans, desc(n))

top_titles_by_cluster <- df_clusters %>%
  group_by(cluster_kmeans, job_title) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(cluster_kmeans) %>%
  slice_max(n, n = 5) %>%
  arrange(cluster_kmeans, desc(n))

emb_mat <- df_hybrid %>%
  select(starts_with("emb_")) %>%
  as.matrix()

emb_sds       <- apply(emb_mat, 2, sd, na.rm = TRUE)
emb_non_const <- which(emb_sds > 0)

if (length(emb_non_const) < 2L) {
  warning("Poucos componentes não constantes para clustering espectral.")
  df_clusters$cluster_spectral <- NA_character_
} else {
  emb_mat_filtered <- emb_mat[, emb_non_const, drop = FALSE]

  set.seed(123)
  emb_mat_jitter <- emb_mat_filtered + matrix(
    rnorm(length(emb_mat_filtered), mean = 0, sd = 1e-6),
    nrow = nrow(emb_mat_filtered),
    ncol = ncol(emb_mat_filtered)
  )

  emb_mat_scaled <- scale(emb_mat_jitter)

  n_obs <- nrow(emb_mat_scaled)
  max_k_spec <- max(2L, floor(n_obs / 3L))
  k_spec     <- min(best_k, max_k_spec)

  set.seed(123)
  spec_assignments <- tryCatch(
    {
      specc(emb_mat_scaled, centers = k_spec)
    },
    error = function(e) {
      warning("specc falhou, usando k-means como fallback.")
      km <- kmeans(
        emb_mat_scaled,
        centers  = k_spec,
        nstart   = 50,
        iter.max = 100
      )
      factor(km$cluster)
    }
  )

  df_clusters$cluster_spectral <- factor(as.integer(spec_assignments))
}

# 
# LDA em skills_required
# 

skills_long <- df %>%
  select(job_id, skills_required) %>%
  mutate(
    skills_required = coalesce(skills_required, "")
  ) %>%
  tidyr::separate_rows(skills_required, sep = ",") %>%
  mutate(
    skill = skills_required %>%
      stringr::str_to_lower() %>%
      stringr::str_squish()
  ) %>%
  filter(skill != "")

skills_dtm <- skills_long %>%
  count(job_id, skill, name = "n") %>%
  cast_dtm(document = job_id, term = skill, value = n)

n_docs  <- nrow(skills_dtm)
n_terms <- ncol(skills_dtm)

lda_k_candidates <- c(5L, 10L, 20L, 40L)
max_topics_possible <- max(2L, min(100L, n_docs, n_terms))
lda_k_candidates <- lda_k_candidates[lda_k_candidates <= max_topics_possible]
if (length(lda_k_candidates) == 0L) {
  lda_k_candidates <- 2L:max(2L, min(10L, max_topics_possible))
}

lda_search <- purrr::map_df(lda_k_candidates, function(k) {
  set.seed(123)
  mdl <- LDA(skills_dtm, k = k, control = list(seed = 123))
  tibble(k = k, loglik = as.numeric(logLik(mdl)))
})

lda_k <- lda_search$k[which.max(lda_search$loglik)]

set.seed(123)
lda_model <- LDA(skills_dtm, k = lda_k, control = list(seed = 123))

n_top_terms <- max(5L, min(20L, round(log10(max(2, n_terms)) * 5)))

lda_terms <- tidy(lda_model, matrix = "beta") %>%
  group_by(topic) %>%
  slice_max(beta, n = n_top_terms) %>%
  ungroup() %>%
  arrange(topic, desc(beta))

lda_gamma <- tidy(lda_model, matrix = "gamma") %>%
  group_by(document) %>%
  slice_max(gamma, n = 1) %>%
  ungroup() %>%
  rename(job_id = document, dominant_topic = topic)

df_topics <- df %>%
  left_join(lda_gamma, by = "job_id")

topic_assignments <- df_topics %>%
  select(job_id, dominant_topic)

top_skills <- skills_long %>%
  count(skill, sort = TRUE) %>%
  slice_head(n = 50)

skills_top_dtm <- skills_long %>%
  filter(skill %in% top_skills$skill) %>%
  count(job_id, skill, name = "n") %>%
  cast_dtm(document = job_id, term = skill, value = n)

skill_mat <- as.matrix(t(skills_top_dtm))

if (nrow(skill_mat) >= 2L) {
  skill_dist  <- dist(skill_mat, method = "euclidean")
  hc_skills   <- hclust(skill_dist, method = "ward.D2")
  mds_skills  <- cmdscale(skill_dist, k = 2)
  mds_skills_df <- as.data.frame(mds_skills)
  mds_skills_df$skill <- rownames(mds_skills_df)
  colnames(mds_skills_df)[1:2] <- c("x", "y")
} else {
  hc_skills     <- NULL
  mds_skills_df <- tibble()
}

# 
# Resumos de clustering e LDA
# 

silhouette_k_df <- tibble(
  k = ks,
  mean_silhouette = sil_scores
)

kmeans_global_stats <- tibble(
  k                = best_k,
  totss            = km_final$totss,
  betweenss        = km_final$betweenss,
  betweenss_ratio  = km_final$betweenss / km_final$totss
)

cluster_assignments <- df_clusters %>%
  select(job_id, cluster_kmeans, cluster_spectral)

spec_profiles <- NULL
mean_sil_spec <- NA_real_

if (!all(is.na(df_clusters$cluster_spectral))) {
  spec_profiles <- df_clusters %>%
    group_by(cluster_spectral) %>%
    summarise(
      n          = n(),
      salary_med = median(salary_mid, na.rm = TRUE),
      salary_p25 = quantile(salary_mid, 0.25, na.rm = TRUE),
      salary_p75 = quantile(salary_mid, 0.75, na.rm = TRUE),
      .groups    = "drop"
    )

  emb_mat_spec <- df_hybrid %>%
    select(starts_with("emb_")) %>%
    as.matrix()
  emb_sds2 <- apply(emb_mat_spec, 2, sd, na.rm = TRUE)
  emb_non_const2 <- which(emb_sds2 > 0)
  emb_mat_spec <- scale(emb_mat_spec[, emb_non_const2, drop = FALSE])

  dist_emb_spec <- dist(emb_mat_spec)
  sil_spec <- cluster::silhouette(
    as.integer(df_clusters$cluster_spectral),
    dist_emb_spec
  )
  mean_sil_spec <- mean(sil_spec[, 3])
}

lda_topic_sizes <- df_topics %>%
  count(dominant_topic, name = "n_jobs") %>%
  arrange(dominant_topic)

lda_terms_short <- lda_terms %>%
  group_by(topic) %>%
  slice_max(beta, n = min(n_top_terms, 10L)) %>%
  ungroup()

unsup_method_summary <- tibble(
  method          = c("pca_kmeans", "spectral_clustering", "lda_topics"),
  n_clusters      = c(
    best_k,
    if (!all(is.na(df_clusters$cluster_spectral))) nlevels(df_clusters$cluster_spectral) else NA_integer_,
    lda_k
  ),
  mean_silhouette = c(
    max(sil_scores),
    mean_sil_spec,
    NA_real_
  )
)

cm_kmeans_spectral <- cluster_assignments %>%
  filter(!is.na(cluster_spectral)) %>%
  count(cluster_kmeans, cluster_spectral) %>%
  group_by(cluster_kmeans) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup()

cm_cluster_topic <- df_clusters %>%
  left_join(topic_assignments, by = "job_id") %>%
  filter(!is.na(dominant_topic)) %>%
  count(cluster_kmeans, dominant_topic) %>%
  group_by(cluster_kmeans) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup()

# 
# Recomendador: exemplos
# 

job_ids <- df_model_base$job_id
row.names(dtm_tfidf_norm) <- as.character(job_ids)
JOB_INDEX <- setNames(seq_len(nrow(df_model_base)), as.character(job_ids))

sample_job_id <- df_model_base$job_id[1]

recs_similar <- recommend_similar_jobs(
  df_model_base   = df_model_base,
  dtm_tfidf_norm  = dtm_tfidf_norm,
  JOB_INDEX       = JOB_INDEX,
  job_identifier  = sample_job_id,
  top_k           = 10,
  filters         = list(),
  diversify       = TRUE,
  lambda_mmr      = 0.7
)

recs_similar_view <- recs_similar %>%
  transmute(
    rank = row_number(),
    job_id,
    company_name,
    job_title,
    industry,
    experience_level,
    employment_type,
    location,
    salary_range_usd,
    score
  )

user_profile_skills   <- c("python", "pandas", "sql", "tensorflow", "nlp")
user_profile_tools    <- c("pytorch", "mlflow")
user_profile_title    <- "machine learning engineer"
user_profile_industry <- "technology"
user_profile_level    <- "Senior"

recs_user <- recommend_jobs_for_user(
  df_model_base     = df_model_base,
  dtm_tfidf_norm    = dtm_tfidf_norm,
  vectorizer        = vectorizer,
  tfidf_transformer = tfidf_transformer,
  skills           = user_profile_skills,
  tools            = user_profile_tools,
  desired_title    = user_profile_title,
  industry         = user_profile_industry,
  experience_level = user_profile_level,
  top_k            = 15,
  filters          = list(
    experience_levels = c("Senior", "Mid", "Lead"),
    employment_types  = c("Full-time"),
    locations         = c("remote", "new york", "san francisco"),
    min_salary        = 90000,
    max_salary        = 300000
  ),
  diversify        = TRUE,
  lambda_mmr       = 0.7
)

user_profile_label <- paste0(
  stringr::str_to_title(user_profile_title), " ", user_profile_level,
  " (skills: ", paste(user_profile_skills, collapse = ", "),
  "; tools: ", paste(user_profile_tools, collapse = ", "), ")"
)

recs_user_view <- recs_user %>%
  transmute(
    rank = row_number(),
    job_id,
    company_name,
    job_title,
    industry,
    experience_level,
    employment_type,
    location,
    salary_range_usd,
    score
  )

# Salvando modelos, objetos e CSVs


supervised_cv_results <- bind_rows(
  metrics_elastic_hybrid %>%
    mutate(model = "elastic_net",    features = "hybrid"),
  metrics_rf_hybrid %>%
    mutate(model = "random_forest", features = "hybrid"),
  metrics_xgb_hybrid %>%
    mutate(model = "xgboost",       features = "hybrid")
)

readr::write_csv(
  supervised_cv_results,
  file = file.path(OUTPUT_DIR, "supervised_salary_models_cv_metrics_hybrid.csv")
)

supervised_test_results <- bind_rows(
  metrics_elastic_test %>%
    mutate(model = "elastic_net",    features = "hybrid"),
  metrics_rf_test %>%
    mutate(model = "random_forest", features = "hybrid"),
  metrics_xgb_test %>%
    mutate(model = "xgboost",       features = "hybrid")
) %>%
  arrange(.metric, .estimate)

readr::write_csv(
  supervised_test_results,
  file = file.path(OUTPUT_DIR, "supervised_salary_models_test_metrics_hybrid.csv")
)

saveRDS(
  list(
    elastic_net_hybrid = fit_elastic_hybrid,
    rf_hybrid          = fit_rf_hybrid,
    xgb_hybrid         = fit_xgb_hybrid
  ),
  file = file.path(OUTPUT_DIR, "supervised_salary_models_fitted_hybrid.rds")
)

repr_test_results <- bind_rows(
  metrics_rf_tfidf_test %>%
    mutate(model = "random_forest", features = "tfidf_lsa"),
  metrics_rf_emb_test %>%
    mutate(model = "random_forest", features = "embeddings"),
  metrics_rf_test %>%              # RF on hybrid
    mutate(model = "random_forest", features = "hybrid")
) %>%
  arrange(.metric, .estimate)

readr::write_csv(
  repr_test_results,
  file = file.path(OUTPUT_DIR, "text_representations_rf_test_metrics.csv")
)

readr::write_csv(
  cluster_profiles,
  file = file.path(OUTPUT_DIR, "unsup_pca_kmeans_cluster_profiles.csv")
)

readr::write_csv(
  silhouette_k_df,
  file = file.path(OUTPUT_DIR, "unsup_pca_kmeans_silhouette_by_k.csv")
)

readr::write_csv(
  kmeans_global_stats,
  file = file.path(OUTPUT_DIR, "unsup_pca_kmeans_global_stats.csv")
)

readr::write_csv(
  top_industries_by_cluster,
  file = file.path(OUTPUT_DIR, "unsup_pca_kmeans_top_industries_by_cluster.csv")
)

readr::write_csv(
  top_titles_by_cluster,
  file = file.path(OUTPUT_DIR, "unsup_pca_kmeans_top_titles_by_cluster.csv")
)

readr::write_csv(
  cluster_assignments,
  file = file.path(OUTPUT_DIR, "unsup_cluster_assignments_jobs.csv")
)

if (!is.null(spec_profiles)) {
  readr::write_csv(
    spec_profiles,
    file = file.path(OUTPUT_DIR, "unsup_spectral_cluster_profiles.csv")
  )
}

readr::write_csv(
  lda_terms_short,
  file = file.path(OUTPUT_DIR, "unsup_lda_top_terms.csv")
)

readr::write_csv(
  lda_topic_sizes,
  file = file.path(OUTPUT_DIR, "unsup_lda_topic_sizes.csv")
)

readr::write_csv(
  lda_search,
  file = file.path(OUTPUT_DIR, "unsup_lda_k_loglik.csv")
)

readr::write_csv(
  topic_assignments,
  file = file.path(OUTPUT_DIR, "unsup_lda_job_topic_assignments.csv")
)

readr::write_csv(
  unsup_method_summary,
  file = file.path(OUTPUT_DIR, "unsup_methods_comparison_summary.csv")
)

readr::write_csv(
  cm_kmeans_spectral,
  file = file.path(OUTPUT_DIR, "unsup_cm_kmeans_spectral.csv")
)

readr::write_csv(
  cm_cluster_topic,
  file = file.path(OUTPUT_DIR, "unsup_cm_cluster_topic.csv")
)

readr::write_csv(
  recs_similar_view,
  file = file.path(OUTPUT_DIR, "recommender_similar_jobs_example.csv")
)

readr::write_csv(
  recs_user_view,
  file = file.path(OUTPUT_DIR, "recommender_user_profile_example.csv")
)

readr::write_csv(
  salary_rule_results,
  file = file.path(OUTPUT_DIR, "salary_rules_probe_results.csv")
)

readr::write_csv(
  salary_rule_effects,
  file = file.path(OUTPUT_DIR, "salary_rules_effects_full_dataset.csv")
)

# salvar objetos de texto e não supervisionados ----------------------------

text_objects <- list(
  vocab              = vocab,
  vectorizer         = vectorizer,
  tfidf_transformer  = tfidf_transformer,
  tfidf_lsa          = tfidf_lsa,
  lsa_k              = tfidf_res$lsa_k,
  variance_curve     = tfidf_res$variance_curve,
  word_vectors       = word_vectors
)
saveRDS(
  text_objects,
  file = file.path(OUTPUT_DIR, "text_representation_objects.rds")
)

unsup_objects <- list(
  pca_full             = pca_full,
  best_k               = best_k,
  km_final             = km_final,
  df_clusters          = df_clusters,
  kmeans_global_stats  = kmeans_global_stats,
  spec_profiles        = spec_profiles,
  mean_sil_spec        = mean_sil_spec,
  lda_model            = lda_model,
  lda_terms            = lda_terms,
  lda_gamma            = lda_gamma,
  lda_search           = lda_search,
  mds_skills_df        = mds_skills_df,
  hc_skills            = hc_skills,
  cluster_assignments  = cluster_assignments,
  topic_assignments    = topic_assignments,
  pca_num_pcs_kmeans   = pca_num_pcs_kmeans,
  pca_var_expl_kmeans  = pca_var_expl_kmeans,
  cm_kmeans_spectral   = cm_kmeans_spectral,
  cm_cluster_topic     = cm_cluster_topic
)
saveRDS(
  unsup_objects,
  file = file.path(OUTPUT_DIR, "unsupervised_objects.rds")
)

recommender_objects <- list(
  dtm_tfidf_norm    = dtm_tfidf_norm,
  vectorizer        = vectorizer,
  tfidf_transformer = tfidf_transformer,
  JOB_INDEX         = JOB_INDEX,
  job_ids           = job_ids
)
saveRDS(
  recommender_objects,
  file = file.path(OUTPUT_DIR, "recommender_objects.rds")
)


# Geração do relatório HTML final

results_list <- list(
  salary_assoc              = salary_assoc,
  salary_rule_results       = salary_rule_results,
  salary_rule_effects       = salary_rule_effects,
  best_salary_rule          = best_salary_rule,
  tfidf_variance            = tfidf_res$variance_curve,
  tfidf_lsa_k               = tfidf_res$lsa_k,
  supervised_cv_results     = supervised_cv_results,
  supervised_test_results   = supervised_test_results,
  repr_test_results         = repr_test_results,
  split_info                = split_info,
  pc_df                     = pc_df,
  silhouette_k_df           = silhouette_k_df,
  best_k                    = best_k,
  kmeans_global_stats       = kmeans_global_stats,
  df_clusters               = df_clusters,
  cluster_profiles          = cluster_profiles,
  top_industries_by_cluster = top_industries_by_cluster,
  top_titles_by_cluster     = top_titles_by_cluster,
  spec_profiles             = spec_profiles,
  mean_sil_spec             = mean_sil_spec,
  lda_terms_short           = lda_terms_short,
  lda_topic_sizes           = lda_topic_sizes,
  lda_k                     = lda_k,
  lda_search                = lda_search,
  mds_skills_df             = mds_skills_df,
  unsup_method_summary      = unsup_method_summary,
  cluster_assignments       = cluster_assignments,
  topic_assignments         = topic_assignments,
  recs_similar_view         = recs_similar_view,
  recs_user_view            = recs_user_view,
  user_profile_label        = user_profile_label,
  pca_scores_df             = pca_scores_df,
  pca_num_pcs_kmeans        = pca_num_pcs_kmeans,
  pca_var_expl_kmeans       = pca_var_expl_kmeans,
  cm_kmeans_spectral        = cm_kmeans_spectral,
  cm_cluster_topic          = cm_cluster_topic,
  pred_xgb_hybrid_bins      = pred_xgb_hybrid_bins
)

html_output <- file.path(OUTPUT_DIR, "relatorio_mercado_vagas_ia_ml.html")

generate_html_report(
  html_path     = html_output,
  out_dir       = OUTPUT_DIR,
  df            = df,
  df_model_base = df_model_base,
  res           = results_list
)

cat("Relatório gerado em:", html_output, "\n")
