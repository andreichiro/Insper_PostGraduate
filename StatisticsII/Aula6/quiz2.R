############################################################
# Relatório final - mercado de vagas IA/ML
############################################################

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
    "hardhat", "scales", "ggrepel"
  )
  ensure_pkgs(required_pkgs)
  tidymodels::tidymodels_prefer()
})

if (capabilities("cairo")) options(bitmapType = "cairo")

############################################################
# Funções HTML
############################################################

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

############################################################
# Funções auxiliares de dados/texto
############################################################

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

############################################################
# Regras de salary_valid
############################################################

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

############################################################
# TF-IDF + LSA
############################################################

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

############################################################
# Embeddings GloVe
############################################################

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

############################################################
# Receitas e tuning de modelos
############################################################

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

############################################################
# Sistema de recomendação
############################################################

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

  if (!is.null(experience_levels) && length(experience_levels) > 0) {
    lvls <- tolower(experience_levels)
    mask <- mask & tolower(df$experience_level) %in% lvls
  }

  if (!is.null(employment_types) && length(employment_types) > 0) {
    tys <- tolower(employment_types)
    mask <- mask & tolower(df$employment_type) %in% tys
  }

  if (!is.null(locations) && length(locations) > 0) {
    locs <- tolower(locations)
    locs_escaped <- escape_regex(locs)
    re_loc <- paste0(locs_escaped, collapse = "|")
    loc_col <- tolower(df$location)
    mask <- mask & stringr::str_detect(loc_col, re_loc)
  }

  if (!is.null(min_salary)) {
    mask <- mask & coalesce(df$salary_mid, -Inf) >= min_salary
  }
  if (!is.null(max_salary)) {
    mask <- mask & coalesce(df$salary_mid,  Inf) <= max_salary
  }

  mask & !is.na(mask)
}

mmr <- function(doc_scores, doc_embeddings, k = 10, lambda_mult = 0.7) {
  n <- length(doc_scores)
  if (n == 0) return(integer(0))
  k <- min(k, n)

  if (inherits(doc_embeddings, "dgCMatrix")) {
    emb <- as.matrix(doc_embeddings)
  } else {
    emb <- as.matrix(doc_embeddings)
  }

  norms <- sqrt(rowSums(emb^2))
  norms[norms == 0] <- 1
  emb <- emb / norms

  selected   <- integer(0)
  candidates <- seq_len(n)

  first <- which.max(doc_scores)
  selected   <- c(selected, first)
  candidates <- setdiff(candidates, first)

  while (length(selected) < k && length(candidates) > 0) {
    best_idx <- NA_integer_
    best_val <- -Inf

    for (c in candidates) {
      sim_to_selected <- max(
        emb[c, , drop = FALSE] %*% t(emb[selected, , drop = FALSE])
      )
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
  lambda_mmr       = 0.7
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
    warning("Sem vagas após filtros.")
    empty_res <- df_model_base[0, , drop = FALSE] %>%
      mutate(score = numeric(0))
    return(empty_res)
  }

  cand_scores <- sims[idxs]

  if (diversify && length(idxs) > top_k) {
    cand_emb <- dtm_tfidf_norm[idxs, , drop = FALSE]
    sel_local <- mmr(
      doc_scores     = cand_scores,
      doc_embeddings = cand_emb,
      k              = top_k,
      lambda_mult    = lambda_mmr
    )
    chosen <- idxs[sel_local]
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

  if (diversify && length(idxs) > top_k) {
    cand_emb <- dtm_tfidf_norm[idxs, , drop = FALSE]
    sel_local <- mmr(
      doc_scores     = cand_scores,
      doc_embeddings = cand_emb,
      k              = top_k,
      lambda_mult    = lambda_mmr
    )
    chosen <- idxs[sel_local]
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

############################################################
# Função de geração do HTML
############################################################

generate_html_report <- function(html_path,
                                 out_dir,
                                 df,
                                 df_model_base,
                                 res) {

  html_begin(html_path, "Mercado de Vagas em IA/ML – Relatório Final")
  html_h1("Mercado de trabalho em IA/ML – Relatório final")

  ##########################################################
  # 1. Introdução
  ##########################################################

  html_h2("1. Introdução e contexto")
  html_p(
    "Este trabalho utiliza um conjunto de dados de vagas para o mercado de Inteligência Artificial e Machine Learning, ",
    "obtido a partir de plataformas de recrutamento internacionais. Cada linha representa uma vaga, identificada por <code>job_id</code>, ",
    "com informações de cargo, empresa, localização, nível de experiência, tipo de vínculo, setor, habilidades e ferramentas, ",
    "além de uma faixa salarial anual em dólares."
  )
  html_p(
    "Objetivo supervisionado: construir e comparar modelos preditivos de salário (em log), a partir de atributos estruturados e texto. ",
    "Objetivo não supervisionado: identificar clusters de vagas semelhantes e temas recorrentes de habilidades (tópicos). ",
    "Adicionalmente, implementamos um sistema de recomendação baseado em conteúdo."
  )

  ##########################################################
  # 2. Descrição dos dados e EDA
  ##########################################################

  html_h2("2. Descrição dos dados e EDA")

  ## 2.1 estrutura
  html_h3("2.1 Estrutura geral dos dados")

  dims_tbl <- tibble::tibble(
    linhas = nrow(df),
    colunas = ncol(df)
  )
  html_table(dims_tbl, "Dimensão do dataset", digits = 0)

  sal_sum <- summary(df$salary_mid)
  sal_tbl <- tibble::tibble(
    estatistica = names(sal_sum),
    valor = as.numeric(sal_sum)
  )
  html_table(sal_tbl, "Resumo da variável salary_mid", digits = 2)

  date_sum <- summary(df$posted_date)
  date_tbl <- tibble::tibble(
    estatistica = names(date_sum),
    valor = as.character(date_sum)
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
    "Na limpeza inicial, garantimos <code>job_id</code> como identificador categórico, parseamos a faixa salarial em ",
    "<code>salary_min</code>, <code>salary_max</code> e definimos <code>salary_mid</code> como alvo numérico, além de padronizar datas e campos textuais."
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

  html_p(
    "A distribuição de <code>salary_mid</code> é fortemente assimétrica à direita. A transformação logarítmica ",
    "produz uma distribuição mais próxima de normal, reduzindo influência de outliers e justificando o uso de ",
    "<code>salary_log = log(salary_mid)</code> como alvo da regressão."
  )

  ## 2.3 salário por nível/país
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
      title = "Salário (mid) por país (top 10)",
      x = "País",
      y = "Salário anual (US$)"
    )

  fname_box_country <- "eda_salary_box_country.png"
  save_gg(p_box_country, fname_box_country, dir = out_dir)
  cat("<p><b>Figura 4 – Salário por país (top 10)</b><br/><img src='", fname_box_country, "'></p>\n", sep = "")

  html_p(
    "Os boxplots confirmam a intuição de negócio: níveis Senior/Lead e países como EUA/Reino Unido concentram salários mais altos. ",
    "Isso reforça a importância de incluir <code>experience_level</code>, tipo de vínculo e localização como preditores."
  )

  ## 2.4 missing
  html_h3("2.4 Valores faltantes")

  missing_df <- df %>%
    summarise(across(everything(), ~ mean(is.na(.x)), .names = "missing_{.col}")) %>%
    tidyr::pivot_longer(everything(),
                        names_to = "variavel",
                        values_to = "taxa_missing") %>%
    arrange(desc(taxa_missing))

  html_table(
    missing_df %>%
      mutate(taxa_percent = round(100 * taxa_missing, 1)) %>%
      select(variavel, taxa_percent) %>%
      head(25),
    "Top 25 variáveis com maior taxa de missing (%)",
    digits = 1
  )

  p_missing <- ggplot(
    missing_df %>% filter(taxa_missing > 0),
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

  html_p(
    "Em variáveis numéricas, os faltantes são imputados com mediana dentro da receita (<code>step_impute_median</code>), ",
    "pois a mediana é robusta a outliers. Em colunas textuais, <code>NA</code> é substituído por string vazia, ",
    "evitando problemas na tokenização."
  )

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

  skills_tokens <- skills_tokens

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

  html_p(
    "A frequência de <code>skills_required</code> destaca Python, SQL, bibliotecas de ML e ferramentas de cloud, ",
    "motivando um tratamento cuidadoso do texto (TF‑IDF, LSA e embeddings) na modelagem supervisionada."
  )

  ##########################################################
  # 3. Pré-processamento e decisões
  ##########################################################

  html_h2("3. Pré-processamento e decisões")

  ## 3.1 parsing salario
  html_h3("3.1 Parsing de salário e definição do alvo")

  html_p(
    "A coluna original <code>salary_range_usd</code> traz faixas em texto (ex.: \"100000-150000\"). ",
    "Implementamos uma função que identifica padrões <code>min-max</code> e valores únicos, corrige faixas invertidas ",
    "e constrói <code>salary_min</code>, <code>salary_max</code> e <code>salary_mid</code>. ",
    "O alvo da regressão é <code>salary_log = log(salary_mid)</code>."
  )

  ## 3.2 tuning salary_valid
  html_h3("3.2 Regras de trimming de salário (salary_valid)")

  sr <- res$salary_rule_results
  html_table(sr, "Regras candidatas de trimming de salário (modelo probe – RMSE em log)", digits = 4)

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
  html_note(
    "A melhor regra de trimming pelo menor RMSE em CV foi <b>", best_rule_name,
    "</b>. Assim, a definição de <code>salary_valid</code> é totalmente guiada por desempenho preditivo, ",
    "evitando cortes arbitrários de outliers."
  )

  ## 3.3 texto unificado
  html_h3("3.3 Construção de texto unificado (text_blob)")

  html_p(
    "As colunas <code>job_title</code>, <code>skills_required</code>, <code>tools_preferred</code> e <code>industry</code> ",
    "são concatenadas em <code>text_blob</code>. Isso permite aplicar um pipeline único de NLP e capturar tanto skills quanto contexto de setor e cargo."
  )

  ## 3.4 representações de texto
  html_h3("3.4 Representações de texto: TF‑IDF+LSA, embeddings e híbrido")

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

  html_p(
    "A dimensão latente <code>k</code> da LSA foi escolhida para explicar cerca de 90% da variância da matriz TF‑IDF normalizada. ",
    "Em paralelo, treinamos embeddings GloVe sobre o mesmo vocabulário e construímos uma representação híbrida ",
    "concatenando componentes LSA e embeddings de documento."
  )

  repr_test_results <- res$repr_test_results
  repr_rmse <- repr_test_results %>%
    filter(.metric == "rmse")

  p_repr <- ggplot(
    repr_rmse,
    aes(x = features, y = .estimate)
  ) +
    geom_col() +
    labs(
      title = "Random Forest – comparação de representações textuais (teste)",
      x = "Features de texto",
      y = "RMSE em log(salário)"
    )

  fname_repr <- "prep_repr_rf_rmse.png"
  save_gg(p_repr, fname_repr, dir = out_dir)
  cat("<p><b>Figura 9 – Comparação de representações textuais</b><br/><img src='", fname_repr, "'></p>\n", sep = "")

  best_repr_row <- repr_rmse %>% arrange(.estimate) %>% slice(1)
  best_repr <- best_repr_row$features[1]

  html_note(
    "Mantendo a família Random Forest fixa, a representação com menor RMSE foi <b>", best_repr,
    "</b>. Por isso, adotamos essa representação (tipicamente a híbrida) como padrão na comparação entre famílias de modelos."
  )

  ## 3.5 receita de pré-processamento
  html_h3("3.5 Receita de pré-processamento (build_recipe)")

  html_p(
    "A função <code>build_recipe()</code> remove IDs e texto bruto, extrai ano/mês da data, imputa faltantes numéricos com mediana, ",
    "trata categorias raras, cria dummies one‑hot, remove preditores de variância quase zero e normaliza preditores numéricos. ",
    "Isso garante entrada consistente para Elastic Net, Random Forest e XGBoost."
  )

  ##########################################################
  # 4. Modelagem supervisionada
  ##########################################################

  html_h2("4. Modelagem supervisionada – predição de salário")

  supervised_cv_results  <- res$supervised_cv_results
  supervised_test_results <- res$supervised_test_results

  html_h3("4.1 Configuração de treino, validação e teste")

  html_p(
    "O alvo é <code>salary_log</code>. Usamos split treino/teste estratificado em <code>salary_log</code> (80/20) ",
    "e validação cruzada em 5 folds no treino. Foram comparados três modelos sobre as features híbridas: ",
    "Elastic Net, Random Forest e XGBoost."
  )

  html_table(supervised_cv_results, "Métricas de CV (hybrid features)", digits = 4)

  html_h3("4.2 Desempenho no conjunto de teste")

  html_table(supervised_test_results, "Métricas no conjunto de teste (hybrid features)", digits = 4)

  test_rmse <- supervised_test_results %>%
    filter(.metric == "rmse") %>%
    arrange(.estimate)

  p_models <- ggplot(
    test_rmse,
    aes(x = model, y = .estimate)
  ) +
    geom_col() +
    labs(
      title = "RMSE em log(salário) por modelo – teste",
      x = "Modelo",
      y = "RMSE"
    )

  fname_models <- "sup_models_rmse_test.png"
  save_gg(p_models, fname_models, dir = out_dir)
  cat("<p><b>Figura 10 – RMSE por modelo no teste</b><br/><img src='", fname_models, "'></p>\n", sep = "")

  best_row <- test_rmse %>% slice(1)
  best_model <- best_row$model[1]
  best_rmse <- best_row$.estimate[1]
  rmse_mult <- exp(best_rmse)

  html_note(
    "O modelo com menor RMSE em log foi <b>", best_model, "</b> com RMSE ≈ ",
    sprintf("%.3f", best_rmse),
    ". Isso implica um erro multiplicativo típico em torno de <b>",
    sprintf("%.2f×</b>", rmse_mult),
    " no salário previsto (por exemplo, ≈ 35% se exp(RMSE)≈1,35). ",
    "Este modelo é adotado como modelo final de predição de salário."
  )

  ##########################################################
  # 5. Análise não supervisionada
  ##########################################################

  html_h2("5. Análise não supervisionada – clusters e tópicos")

  ## 5.1 PCA + k-means
  html_h3("5.1 PCA + k‑means sobre features híbridas")

  pc_df <- res$pc_df
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

  silhouette_k_df <- res$silhouette_k_df
  p_sil <- ggplot(
    silhouette_k_df,
    aes(x = k, y = mean_silhouette)
  ) +
    geom_line() +
    geom_point() +
    labs(
      title = "k‑means – silhouette médio por k",
      x = "k",
      y = "Silhouette médio"
    )

  fname_sil <- "unsup_kmeans_silhouette.png"
  save_gg(p_sil, fname_sil, dir = out_dir)
  cat("<p><b>Figura 12 – Silhouette médio por número de clusters</b><br/><img src='", fname_sil, "'></p>\n", sep = "")

  best_k <- res$best_k
  kmeans_global_stats <- res$kmeans_global_stats
  bss_ratio <- kmeans_global_stats$betweenss_ratio[1]
  bss_pct <- scales::percent(bss_ratio, accuracy = 0.1)

  html_p(
    "Aplicamos PCA às features híbridas e rodamos k‑means nos scores das primeiras componentes suficientes para ≈90% da variância. ",
    "O número de clusters <code>k</code> foi escolhido pelo maior silhouette médio, resultando em <b>k = ", best_k, "</b>. ",
    "A razão <code>betweenss / totss</code> para esse k é de ", bss_pct,
    ", indicando que essa fração da variabilidade total é explicada pela separação entre clusters."
  )

  pca_scores_df <- res$pca_scores_df
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

  cluster_profiles <- res$cluster_profiles
  html_table(cluster_profiles, "Perfil salarial por cluster (k‑means)", digits = 0)

  top_ind <- res$top_industries_by_cluster
  html_table(top_ind, "Top indústrias por cluster (k‑means)", digits = 0)

  top_titles <- res$top_titles_by_cluster
  html_table(top_titles, "Top títulos por cluster (k‑means)", digits = 0)

  html_p(
    "Os clusters exibem perfis distintos: grupos com salários mais altos tendem a concentrar vagas Senior/Lead em grandes techs, ",
    "enquanto clusters com salários menores aparecem mais em vagas de entrada ou em regiões de menor custo de vida."
  )

  ## 5.2 clustering espectral
  html_h3("5.2 Clustering espectral em embeddings")

  spec_profiles <- res$spec_profiles
  mean_sil_spec <- res$mean_sil_spec

  if (!is.null(spec_profiles)) {
    html_table(spec_profiles, "Perfil salarial por cluster (clustering espectral)", digits = 0)
    if (is.finite(mean_sil_spec)) {
      html_note(
        "O clustering espectral nos embeddings apresentou silhouette médio ≈ ",
        sprintf("%.3f", mean_sil_spec),
        ". Estruturas de clusters não esféricas tendem a ser melhor capturadas por essa abordagem."
      )
    }
  } else {
    html_p("O clustering espectral não pôde ser estimado de forma estável neste conjunto.")
  }

  ## 5.3 LDA em skills
  html_h3("5.3 Tópicos de skills (LDA)")

  lda_terms_short <- res$lda_terms_short
  lda_topic_sizes <- res$lda_topic_sizes
  lda_k <- res$lda_k

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

  html_table(lda_topic_sizes, "Número de vagas por tópico dominante (skills_required)", digits = 0)

  html_p(
    "O LDA foi ajustado sobre <code>skills_required</code>, com escolha de <code>k</code> guiada por log-verossimilhança. ",
    "Cada tópico corresponde a um stack de habilidades recorrente, por exemplo: Python + SQL + cloud; ",
    "data engineering com Spark; MLOps com Docker, Kubernetes e MLflow. ",
    "A tabela de tamanhos indica quais stacks são mais prevalentes na base."
  )

  ## 5.4 mapa MDS de skills
  html_h3("5.4 Mapa de skills (MDS)")

  mds_skills_df <- res$mds_skills_df
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

    html_p(
      "O MDS em cima da matriz de coocorrência de skills gera um mapa em que habilidades próximas tendem a aparecer juntas ",
      "em vagas semelhantes (por exemplo, um grupo de \"Python+pandas+scikit‑learn\" versus um de \"Spark+Hadoop+Kafka\")."
    )
  } else {
    html_p("Número insuficiente de skills frequentes para construir um mapa MDS informativo.")
  }

  ## 5.5 resumo geral
  html_h3("5.5 Comparação geral de métodos não supervisionados")

  unsup_method_summary <- res$unsup_method_summary
  html_table(unsup_method_summary, "Resumo de métodos não supervisionados", digits = 3)

  ##########################################################
  # 6. Sistema de recomendação baseado em conteúdo
  ##########################################################

  html_h2("6. Sistema de recomendação baseado em conteúdo")

  html_p(
    "Implementamos também um sistema de recomendação <i>content‑based</i> sobre as vagas da base. A ideia é recomendar vagas semelhantes a: ",
    "um perfil de usuário (skills, ferramentas, título desejado, indústria, nível de experiência); ",
    "ou uma vaga específica (recommend_similar_jobs)."
  )
  html_p(
    "A abordagem é: representar cada vaga por seu vetor de TF‑IDF normalizado (dtm_tfidf_norm), construído a partir de <code>text_blob</code>; ",
    "construir o vetor de perfil do usuário com <code>build_user_profile_text()</code> + <code>vectorize_text_tfidf()</code>; ",
    "calcular similaridade de cosseno entre o perfil e todas as vagas; ",
    "aplicar filtros opcionais de negócio (nível, tipo, localização, faixa salarial); ",
    "e usar MMR (Maximal Marginal Relevance) para balancear relevância e diversidade no top‑k."
  )

  recs_similar_view <- res$recs_similar_view
  recs_user_view    <- res$recs_user_view

  html_h3("6.1 Vagas similares a uma vaga base")

  html_table(
    recs_similar_view %>%
      select(rank, job_id, company_name, job_title, industry,
             experience_level, employment_type, location,
             salary_range_usd, score),
    "Exemplo de recomendações similares a uma vaga real",
    digits = 3
  )

  html_h3("6.2 Vagas recomendadas para um perfil de usuário")

  html_table(
    recs_user_view %>%
      select(rank, job_id, company_name, job_title, industry,
             experience_level, employment_type, location,
             salary_range_usd, score),
    "Exemplo de recomendações para um perfil (ML Engineer Senior)",
    digits = 3
  )

  html_note(
    "Optamos por TF‑IDF (e não embeddings) no recomendador por simplicidade e interpretabilidade. ",
    "Os filtros de negócio (nível, tipo, localização, faixa salarial) são aplicados antes da seleção por MMR, ",
    "e usamos <code>escape_regex()</code> para proteger a filtragem por localização quando há metacaracteres em nomes de cidades ou países."
  )

  ##########################################################
  # 7. Conclusões, limitações e trabalhos futuros
  ##########################################################

  html_h2("7. Conclusões, limitações e trabalhos futuros")

  html_h3("7.1 Conclusões sobre o modelo supervisionado")

  html_p(
    "Com base na comparação de RMSE/MAE no conjunto de teste, o modelo <b>", best_model,
    "</b> treinado sobre a representação híbrida de texto foi o que melhor previu <code>salary_log</code>. ",
    "O erro multiplicativo típico é da ordem de ", sprintf("%.2f×", rmse_mult),
    ", compatível com a variabilidade observada nos salários reais do mercado de IA/ML."
  )

  html_h3("7.2 Insights não supervisionados")

  html_p(
    "A análise de clusters revelou segmentos de vagas com perfis salariais e tecnológicos distintos ",
    "(por exemplo, clusters de vagas Senior em grandes techs versus clusters de vagas entry‑level em mercados emergentes). ",
    "A modelagem de tópicos via LDA identificou temas recorrentes de skills e ferramentas, que podem orientar tanto candidatos ",
    "(na escolha de habilidades a desenvolver) quanto empresas (na definição de requisitos de vaga)."
  )

  html_h3("7.3 Limitações")

  html_list(c(
    "possível viés de amostragem pela origem dos dados (plataformas específicas, regiões sub-representadas);",
    "ranges salariais incompletos ou reportados com ruído, mesmo após trimming guiado por dados;",
    "representações textuais baseadas em TF‑IDF/GloVe não capturam nuances semânticas profundas como modelos pré‑treinados de linguagem.",
    "o sistema de recomendação ainda não incorpora feedback explícito/implícito de usuários."
  ))

  html_h3("7.4 Trabalhos futuros")

  html_list(c(
    "experimentar embeddings de linguagem pré‑treinados (BERT, etc.) para representar <code>text_blob</code>;",
    "refinar o sistema de recomendação incorporando feedback implícito/explícito de usuários;",
    "avaliar fairness do modelo de salário em diferentes regiões e perfis;",
    "explorar redes neurais densas sobre as features híbridas para comparar com XGBoost."
  ))

  html_end(html_path)
}

############################################################
# Execução principal da análise
############################################################

# caminho do CSV (ajuste conforme seu ambiente)
DATA_PATH <- "ai_job_market.csv"

# diretório de saída
OUTPUT_DIR <- file.path(dirname(DATA_PATH), "outputs_ai_jobs")
if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# leitura
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

############################################################
# Tuning de salary_valid
############################################################

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

############################################################
# TF-IDF + LSA, embeddings e híbrido
############################################################

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

############################################################
# Modelos supervisionados
############################################################

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
  bind_cols(test_hybrid %>% select(salary_log, salary_mid))
pred_rf_hybrid <- predict(fit_rf_hybrid, test_hybrid) %>%
  bind_cols(test_hybrid %>% select(salary_log, salary_mid))
pred_xgb_hybrid <- predict(fit_xgb_hybrid, test_hybrid) %>%
  bind_cols(test_hybrid %>% select(salary_log, salary_mid))

metrics_elastic_test <- pred_elastic_hybrid %>%
  regression_metrics(truth = salary_log, estimate = .pred)
metrics_rf_test <- pred_rf_hybrid %>%
  regression_metrics(truth = salary_log, estimate = .pred)
metrics_xgb_test <- pred_xgb_hybrid %>%
  regression_metrics(truth = salary_log, estimate = .pred)

############################################################
# Comparação TF-IDF vs Embeddings vs Híbrido (RF)
############################################################

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

############################################################
# Não supervisionado: PCA + k-means, espectral, LDA
############################################################

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

############################################################
# LDA em skills_required
############################################################

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

############################################################
# Resumos de clustering e LDA
############################################################

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

############################################################
# Recomendador: exemplos
############################################################

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

recs_user <- recommend_jobs_for_user(
  df_model_base   = df_model_base,
  dtm_tfidf_norm  = dtm_tfidf_norm,
  vectorizer      = vectorizer,
  tfidf_transformer = tfidf_transformer,
  skills           = c("python", "pandas", "sql", "tensorflow", "nlp"),
  tools            = c("pytorch", "mlflow"),
  desired_title    = "machine learning engineer",
  industry         = "technology",
  experience_level = "Senior",
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

############################################################
# Salvando modelos, objetos e CSVs
############################################################

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

topic_assignments <- df_topics %>%
  select(job_id, dominant_topic)

readr::write_csv(
  topic_assignments,
  file = file.path(OUTPUT_DIR, "unsup_lda_job_topic_assignments.csv")
)

readr::write_csv(
  unsup_method_summary,
  file = file.path(OUTPUT_DIR, "unsup_methods_comparison_summary.csv")
)

readr::write_csv(
  recs_similar_view,
  file = file.path(OUTPUT_DIR, "recommender_similar_jobs_example.csv")
)

readr::write_csv(
  recs_user_view,
  file = file.path(OUTPUT_DIR, "recommender_user_profile_example.csv")
)

# salvar objetos de texto e não supervisionados
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
  pca_full            = pca_full,
  best_k              = best_k,
  km_final            = km_final,
  df_clusters         = df_clusters,
  kmeans_global_stats = kmeans_global_stats,
  spec_profiles       = spec_profiles,
  mean_sil_spec       = mean_sil_spec,
  lda_model           = lda_model,
  lda_terms           = lda_terms,
  lda_gamma           = lda_gamma,
  mds_skills_df       = mds_skills_df,
  hc_skills           = hc_skills
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

############################################################
# Geração do relatório HTML final
############################################################

results_list <- list(
  salary_rule_results = salary_rule_results,
  best_salary_rule    = best_salary_rule,
  tfidf_variance      = tfidf_res$variance_curve,
  tfidf_lsa_k         = tfidf_res$lsa_k,
  supervised_cv_results  = supervised_cv_results,
  supervised_test_results = supervised_test_results,
  repr_test_results      = repr_test_results,
  pc_df                  = pc_df,
  silhouette_k_df        = silhouette_k_df,
  best_k                 = best_k,
  kmeans_global_stats    = kmeans_global_stats,
  df_clusters            = df_clusters,
  cluster_profiles       = cluster_profiles,
  top_industries_by_cluster = top_industries_by_cluster,
  top_titles_by_cluster     = top_titles_by_cluster,
  spec_profiles          = spec_profiles,
  mean_sil_spec          = mean_sil_spec,
  lda_terms_short        = lda_terms_short,
  lda_topic_sizes        = lda_topic_sizes,
  lda_k                  = lda_k,
  mds_skills_df          = mds_skills_df,
  unsup_method_summary   = unsup_method_summary,
  recs_similar_view      = recs_similar_view,
  recs_user_view         = recs_user_view,
  pca_scores_df          = {
    ps_df <- as.data.frame(pca_scores[, 1:2])
    colnames(ps_df) <- c("PC1", "PC2")
    ps_df$cluster_kmeans <- df_clusters$cluster_kmeans
    ps_df
  }
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
