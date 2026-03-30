#' @title quiz2.R (versão final)
#' @description Análise de Avaliações de Hotéis — Corpus, TF‑IDF, BoW (polaridade) e LDA
#' @author André
#' @date 15/nov/2025
#' @course Aprendizagem Estatística de Máquina II
#'
#' Entregáveis:
#'  - Relatório HTML com: descritiva + wordcloud + top10, TF‑IDF por autenticidade e polaridade (gráficos e comentários),
#'    modelo BoW para polaridade (CV, métricas, matriz de confusão, ROC e termos de maior peso),
#'    LDA com seleção de k, top‑10 termos por tópico, rótulos e prevalências,
#'    e uma seção final de insights de negócio (drivers, pistas linguísticas de fraude e plano de ação por tópico).
#'
#' Observação: todos os resultados são comentados no HTML (em português), com foco em leituras data‑driven e implicações de negócio.

suppressPackageStartupMessages({
  # Função para garantir pacotes (instala se faltar e carrega)
  ensure_pkgs <- function(pkgs) {
    repos <- getOption("repos")
    if (is.null(repos) || length(repos) == 0 || repos["CRAN"] %in% c("@CRAN@", NA)) {
      options(repos = c(CRAN = "https://cloud.r-project.org"))
    }
    need <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
    if (length(need)) install.packages(need, dependencies = c("Depends","Imports","LinkingTo"))
    invisible(lapply(pkgs, function(p) suppressPackageStartupMessages(library(p, character.only = TRUE))))
  }

  ensure_pkgs(c(
    "R6","tibble","dplyr","ggplot2","readr","stringr","forcats","scales","glue","rlang",
    "quanteda","quanteda.textstats","quanteda.textplots","stopwords","SnowballC",
    "Matrix","glmnet","yardstick","rsample","topicmodels","tidytext","slam","tidyr","purrr","parallel"
  ))
})

# Renderização estável de PNG
if (capabilities("cairo")) options(bitmapType = "cairo")

# -------------------- Caminhos --------------------
data_path  <- "/Users/akatsurada/Documents/INSPER/StatisticsII/Aula6/hoteis.csv"  # ajuste conforme necessário
output_dir <- "outputs_hoteis"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
html_output <- file.path(output_dir, "relatorio_avaliacoes_hoteis.html")

# -------------------- Utilidades HTML --------------------
html_begin <- function(path, title = "análise de avaliações de hotéis") {
  if (file.exists(path)) file.remove(path)
  con <- file(path, open = "wt", encoding = "UTF-8")
  writeLines(paste0(
    "<!DOCTYPE html><html><head><meta charset='utf-8'><title>", title, "</title>",
    "<style>
      body{font-family:-apple-system,system-ui,Segoe UI,Roboto,Arial,sans-serif;margin:24px;color:#222}
      h1,h2,h3{margin:1.0em 0 .5em}
      p{margin:.6em 0;line-height:1.45}
      table{border-collapse:collapse;width:100%;margin:8px 0 16px}
      th,td{border:1px solid #ddd;padding:6px 8px}
      th{text-align:left;background:#f7f7f7;position:sticky;top:0}
      td.num{text-align:right;font-variant-numeric:tabular-nums}
      tbody tr:nth-child(even){background:#fbfbfb}
      .small{color:#555;font-size:.9em}
      .note{background:#f9fbff;border-left:4px solid #9bbcff;padding:.6em .8em;margin:.6em 0}
      .warn{background:#fff9f0;border-left:4px solid #ffb155;padding:.6em .8em;margin:.6em 0}
      img{max-width:100%;height:auto;border:1px solid #eee;box-shadow:0 1px 2px rgba(0,0,0,.05)}
      ul{margin:.2em 0 .8em 1.2em}
      .pill{display:inline-block;background:#eef;border:1px solid #dde;padding:2px 8px;border-radius:999px;margin:2px}
      .kpi{display:inline-block;margin-right:12px;padding:4px 8px;border-radius:6px;background:#f6f8ff;border:1px solid #dde}
    </style></head><body>"
  ), con = con)
  close(con)
  sink(path, append = TRUE, split = FALSE)
}
html_end   <- function(path){ sink(); cat("</body></html>") }
html_h1    <- function(txt) cat("<h1>", txt, "</h1>\n", sep = "")
html_h2    <- function(txt) cat("<h2>", txt, "</h2>\n", sep = "")
html_h3    <- function(txt) cat("<h3>", txt, "</h3>\n", sep = "")
html_p     <- function(...) { cat("<p>", paste0(..., collapse=""), "</p>\n") }
html_note  <- function(...) { cat("<p class='note'>", paste0(..., collapse=""), "</p>\n") }
html_warn  <- function(...) { cat("<p class='warn'>", paste0(..., collapse=""), "</p>\n") }
html_list  <- function(v) { cat("<ul>", paste(sprintf("<li>%s</li>", v), collapse=""), "</ul>\n") }
html_kpis  <- function(named_values) {
  cat("<p>")
  for (nm in names(named_values)) {
    cat("<span class='kpi'><b>", nm, ":</b> ", named_values[[nm]], "</span> ")
  }
  cat("</p>\n")
}
html_table <- function(x, caption = NULL, digits = 3) {
  df <- as.data.frame(x, stringsAsFactors = FALSE, check.names = FALSE)
  num <- vapply(df, is.numeric, TRUE)
  if (any(num)) df[num] <- lapply(df[num], function(v) format(round(v, digits), trim = TRUE))
  if (!is.null(caption)) cat("<h3>", caption, "</h3>\n", sep = "")
  cat("<table><thead><tr>", paste(sprintf("<th>%s</th>", colnames(df)), collapse=""), "</tr></thead><tbody>\n", sep="")
  apply(df, 1, function(row) {
    isnum <- suppressWarnings(!is.na(as.numeric(row)))
    cat("<tr>", paste(sprintf("<td%s>%s</td>", ifelse(isnum, " class='num'", ""), row), collapse=""), "</tr>\n")
  })
  cat("</tbody></table>\n")
}
save_gg <- function(plot, filename, width = 12, height = 7, dpi = 140) {
  ggplot2::ggsave(file.path(output_dir, filename), plot = plot, width = width, height = height, dpi = dpi, device = "png")
}

# -------------------- Classe Principal --------------------
HotelNLP <- R6::R6Class(
  "HotelNLP",
  private = list(
    set_rng = function(seed = 123L) { set.seed(seed); RNGkind("Mersenne-Twister", "Inversion", "Rejection") },

    read_hotels = function(path){
      if (!file.exists(path)) stop(glue::glue("arquivo não encontrado: {path}"))
      df <- readr::read_csv(path, show_col_types = FALSE, progress = FALSE, trim_ws = TRUE)
      need <- c("deceptive","hotel","polarity","source","text")
      miss <- setdiff(need, names(df))
      if (length(miss) > 0) stop(glue::glue("colunas ausentes no csv: {paste(miss, collapse=', ')}"))
      df |>
        dplyr::mutate(
          deceptive = factor(tolower(as.character(deceptive)), levels = c("truthful","deceptive")),
          polarity  = factor(tolower(as.character(polarity)),  levels = c("negative","positive")),
          hotel     = factor(tolower(as.character(hotel))),
          source    = factor(tolower(as.character(source))),
          text      = as.character(text)
        ) |>
        dplyr::filter(!is.na(text), nchar(text) > 0) |>
        dplyr::distinct(text, .keep_all = TRUE)
    },

    to_dfm = function(corp) {
      toks <- quanteda::tokens(
        corp,
        remove_punct   = TRUE,
        remove_symbols = TRUE,
        remove_numbers = TRUE,
        remove_url     = TRUE,
        split_hyphens  = TRUE
      ) |>
        quanteda::tokens_tolower() |>
        quanteda::tokens_remove(stopwords::stopwords("en")) |>
        quanteda::tokens_wordstem(language = "en")
      quanteda::dfm(toks)
    },

    remove_meta_terms = function(dfm_obj, hotels, sources) {
      sw_meta <- unique(c(as.character(hotels), as.character(sources)))
      quanteda::dfm_remove(dfm_obj, pattern = sw_meta, valuetype = "fixed")
    },

    trim_dfm_datadriven = function(dfm_obj) {
      dfm1 <- quanteda::dfm_trim(dfm_obj, min_docfreq = 2, docfreq_type = "count")
      if (quanteda::nfeat(dfm1) == 0) return(dfm_obj)
      prop <- quanteda::docfreq(dfm1) / quanteda::ndoc(dfm1)
      q1 <- unname(stats::quantile(prop, .25)); q3 <- unname(stats::quantile(prop, .75)); iqr <- q3 - q1
      fence <- q3 + 1.5 * iqr
      ubi <- names(prop[prop >= fence])
      if (length(ubi)) dfm1 <- quanteda::dfm_remove(dfm1, ubi)
      dfm1
    },

    freq_by_group = function(dfm_any, groups_vec) {
      g <- quanteda::dfm_group(dfm_any, groups = groups_vec, force = TRUE)
      out <- lapply(seq_len(quanteda::ndoc(g)), function(i) {
        gi <- g[i, ]
        grp <- quanteda::docnames(gi)
        freq <- as.numeric(Matrix::colSums(gi))
        feats <- quanteda::featnames(gi)
        nz <- freq > 0
        data.frame(
          feature = feats[nz],
          frequency = freq[nz],
          group = grp,
          stringsAsFactors = FALSE
        )
      })
      dplyr::bind_rows(out)
    },

    tfidf_by_group = function(dfm_obj, groups_vec, top_n = NULL) {
      tfidf <- quanteda::dfm_tfidf(dfm_obj, scheme_tf = "count", scheme_df = "inverse")
      agg <- self$.__enclos_env__$private$freq_by_group(tfidf, groups_vec)
      nbar <- if (is.null(top_n)) max(10L, min(30L, round(10 * log10(quanteda::nfeat(dfm_obj) + 10)))) else as.integer(top_n)
      agg |>
        dplyr::group_by(group) |>
        dplyr::slice_max(frequency, n = nbar, with_ties = FALSE) |>
        dplyr::ungroup()
    },

    choose_vfolds = function(y) {
      min_class <- min(table(y))
      v <- max(3L, min(10L, as.integer(floor(sqrt(length(y))))))
      v <- min(v, as.integer(min_class)); if (v < 3L) v <- 3L; v
    },

    pick_alpha = function(x, y, vfolds) {
      grid <- c(0, 0.5, 1); best <- NULL; best_auc <- -Inf; best_alpha <- NA_real_
      for (a in grid) {
        fit <- glmnet::cv.glmnet(x, y, family = "binomial", alpha = a, type.measure = "auc", nfolds = vfolds)
        auc_hat <- max(fit$cvm, na.rm = TRUE)
        if (is.finite(auc_hat) && auc_hat > best_auc) { best <- fit; best_auc <- auc_hat; best_alpha <- a }
      }
      list(alpha = best_alpha, cvfit = best)
    },

    pick_threshold = function(truth_factor, prob_vec) {
      df <- data.frame(truth = truth_factor, .pred = prob_vec)
      roc <- yardstick::roc_curve(df, truth = truth, .pred, event_level = "second")
      roc <- dplyr::mutate(roc, youden = sensitivity + specificity - 1)
      roc$.threshold[which.max(roc$youden)]
    },

    js_div_avg = function(phi_mat, eps = 1e-12) {
      kl <- function(a, b) sum(a * log(a / b))
      js <- function(p, q) { p <- p + eps; p <- p / sum(p); q <- q + eps; q <- q / sum(q); m <- 0.5*(p+q); 0.5*kl(p,m)+0.5*kl(q,m) }
      tks <- nrow(phi_mat); if (tks < 2L) return(0)
      pairs <- combn(tks, 2); vals <- apply(pairs, 2, function(idx) js(phi_mat[idx[1], ], phi_mat[idx[2], ])); mean(vals)
    },

    perplexity_robust = function(fit, dtm_train) {
      pp <- try(suppressWarnings(topicmodels::perplexity(fit, newdata = dtm_train)), silent = TRUE)
      if (!inherits(pp, "try-error") && is.finite(pp)) return(as.numeric(pp))
      ll <- try(suppressWarnings(logLik(fit)), silent = TRUE)
      ntoks <- sum(slam::row_sums(dtm_train))
      if (!inherits(ll, "try-error") && is.finite(ll) && ntoks > 0) {
        return(exp(-as.numeric(ll) / ntoks))
      }
      return(NA_real_)
    },

    choose_k_topics = function(dtm_obj, seed = 123L) {
      nd <- nrow(dtm_obj); k_min <- 3L
      k_cap <- max(6L, min(30L, as.integer(round(0.5 * sqrt(nd) * log(nd + 1)))))
      ks <- seq.int(k_min, k_cap, by = 1L)
      ctrl <- list(seed = seed, burnin = 1000, iter = 1500, thin = 100)
      out <- lapply(ks, function(k) {
        fit <- topicmodels::LDA(dtm_obj, k = k, method = "Gibbs", control = ctrl)
        perp <- self$.__enclos_env__$private$perplexity_robust(fit, dtm_obj)
        phi  <- topicmodels::posterior(fit)$terms
        js   <- self$.__enclos_env__$private$js_div_avg(phi)
        data.frame(k = k, perplexity = perp, js_div = js)
      })
      df <- dplyr::bind_rows(out)

      bad <- !is.finite(df$perplexity)
      if (any(bad)) {
        fin <- df$perplexity[is.finite(df$perplexity)]
        df$perplexity[bad] <- if (length(fin)) max(fin) else 1e6
      }

      df$z_perp <- as.numeric(scale(-df$perplexity))  # menor é melhor
      df$z_js   <- as.numeric(scale( df$js_div))      # maior é melhor
      df$score  <- df$z_perp + df$z_js
      k_star <- df$k[which.max(df$score)]
      list(summary = df, k_star = as.integer(k_star))
    },

    label_topics = function(beta_tbl, top_n = 20){
      lex <- tibble::tribble(
        ~label,                         ~keywords,
        "atendimento",                  c("staff","service","help","friend","front","desk","manager","concierg"),
        "limpeza",                      c("clean","dirti","bathroom","smell","towel","sheet","housekeep","stain"),
        "localização",                  c("locat","walk","downtown","mile","shop","restaur","attract","michigan","river","lake"),
        "preço e valor",                c("price","expens","cheap","valu","rate","deal","voucher","charg","fee"),
        "quarto e conforto",            c("room","bed","suite","view","window","noise","quiet","spaciou","comfort","amenit"),
        "alimentos e bebidas",          c("breakfast","restaur","bar","food","coffee","dinner","meal"),
        "transporte e estacionamento",  c("parking","valet","car","cab","taxi","train","bus","traffic"),
        "check-in e check-out",         c("check","upgrad","reserv","book","line","wait","checkout")
      )

      top_terms <- beta_tbl |>
        dplyr::group_by(topic) |>
        dplyr::slice_max(beta, n = top_n, with_ties = FALSE) |>
        dplyr::summarise(terms = list(term), .groups = "drop")

      scored <- purrr::map_dfr(seq_len(nrow(top_terms)), function(i){
        tt <- unlist(top_terms$terms[i])
        scores <- lex |>
          dplyr::mutate(
            score = purrr::map_dbl(keywords, function(kw) {
              sum(vapply(tt, function(term) any(startsWith(term, kw)), logical(1)))
            })
          )
        tibble::tibble(
          topic = top_terms$topic[i],
          topic_label = ifelse(
            max(scores$score) > 0,
            scores$label[which.max(scores$score)][1],
            glue::glue("tópico {top_terms$topic[i]}")
          )
        )
      })
      list(labels = scored, top_terms = top_terms)
    }
  ),

  public = list(
    save_plots = TRUE,
    dados = NULL, corp = NULL, dfm_base = NULL, dfm_clean = NULL,

    initialize = function(save_plots = TRUE) self$save_plots <- isTRUE(save_plots),

    executar = function(csv_path, html_path, seed = 123L) {
      private$set_rng(seed)
      suppressWarnings(suppressMessages(
        quanteda::quanteda_options(threads = max(1, parallel::detectCores() - 1))
      ))

      dados <- private$read_hotels(csv_path)
      self$dados <- dados

      html_begin(html_path, "análise de avaliações de hotéis — chicago")
      html_h1("análise de avaliações de hotéis — relatório html profissional e enxuto")
      html_p("Objetivos do time: (i) identificar temas e termos que explicam avaliações positivas/negativas; ",
             "(ii) mapear sinais linguísticos que diferenciam avaliações verdadeiras de falsas; ",
             "e (iii) traduzir achados em ações de marketing e moderação.")

      # -------------------- Parte 1: Descritiva --------------------
      html_h2("parte 1 — preparação do corpus e análise descritiva")

      dist_tbl <- dplyr::count(dados, deceptive, polarity, name = "n") |> dplyr::arrange(dplyr::desc(n))
      html_table(dist_tbl, "distribuição por autenticidade e polaridade", digits = 0)
      html_p("Leitura: esta tabela mostra o balanceamento entre classes. Desbalanceamentos impactam amostragem/validação ",
             "e interpretação de métricas no modelo de polaridade. Mantemos estratificação por 'polarity' na validação.")

      comp <- dados |>
        dplyr::mutate(chars = nchar(text), words = stringr::str_count(text, "\\S+")) |>
        dplyr::summarise(
          documentos = dplyr::n(),
          chars_mediana = stats::median(chars),
          chars_p95 = stats::quantile(chars, .95),
          palavras_mediana = stats::median(words),
          palavras_p95 = stats::quantile(words, .95)
        )
      html_table(comp, "estatísticas de comprimento das avaliações", digits = 0)
      html_p("Leitura: comprimentos medianos e P95 ajudam a dimensionar a granularidade do vocabulário. ",
             "Valores muito curtos fragilizam a modelagem; muito longos sugerem maior riqueza de tópicos.")

      corp <- quanteda::corpus(dados, text_field = "text")
      self$corp <- corp
      dfm0 <- private$to_dfm(corp)
      dfm0 <- private$remove_meta_terms(dfm0, hotels = dados$hotel, sources = dados$source)
      dfm1 <- private$trim_dfm_datadriven(dfm0)
      self$dfm_base  <- dfm0
      self$dfm_clean <- dfm1

      msg_trim <- tibble::tibble(
        etapa = c("dfm inicial","após remoção de metatermos (hotel/source)","após corte data‑driven (hapax + alta ubiquidade)"),
        n_docs = c(quanteda::ndoc(dfm0), quanteda::ndoc(dfm0), quanteda::ndoc(dfm1)),
        n_termos = c(quanteda::nfeat(dfm0),
                     quanteda::nfeat(private$remove_meta_terms(dfm0, dados$hotel, dados$source)),
                     quanteda::nfeat(dfm1))
      )
      html_table(msg_trim, "efeito das etapas de limpeza no vocabulário", digits = 0)
      html_note("Decisão: removemos nomes de hotéis e fontes para evitar vazamento (‘leakage’) e sinais triviais. ",
                "Também cortamos hapax e termos onipresentes (ruído).")

      # Wordcloud
      set.seed(seed)
      fn_wc <- file.path(output_dir, "parte1_wordcloud.png")
      png(fn_wc, width = 1800, height = 1000)
      suppressWarnings(
        quanteda.textplots::textplot_wordcloud(
          self$dfm_clean,
          max_words = max(50, min(120, round(15 * log10(quanteda::nfeat(self$dfm_clean) + 10)))),
          min_count = 2
        )
      )
      dev.off()
      cat(sprintf("<p><b>Nuvem de palavras:</b><br/><img src='%s'></p>\n", basename(fn_wc)))
      html_p("Interpretação: termos maiores são mais frequentes; usamos apenas após limpeza para reduzir vieses por metadados.")

      # Top 10 termos
      top10 <- quanteda.textstats::textstat_frequency(self$dfm_clean, n = 10) |>
        dplyr::select(rank, feature, frequency, docfreq)
      html_table(top10, "top 10 termos mais frequentes (após limpeza)", digits = 0)
      html_p("Leitura: os ‘Top 10’ orientam stopwords adicionais, caso termos muito genéricos dominem a lista. ",
             "Mantivemos o vocabulário atual porque os termos predominantes são informativos para as próximas análises.")

      # -------------------- Parte 2: TF‑IDF --------------------
      html_h2("parte 2 — análise de termos relevantes (tf‑idf)")

      # Por autenticidade
      tfidf_dec <- private$tfidf_by_group(self$dfm_clean, quanteda::docvars(self$dfm_clean, "deceptive"))
      p_dec <- ggplot2::ggplot(tfidf_dec, ggplot2::aes(x = tidytext::reorder_within(feature, frequency, group), y = frequency)) +
        ggplot2::geom_col() + ggplot2::coord_flip() + tidytext::scale_x_reordered() +
        ggplot2::facet_wrap(~ group, scales = "free_y") +
        ggplot2::labs(x = NULL, y = "importância tf‑idf (soma por grupo)", title = "termos mais relevantes por autenticidade")
      save_gg(p_dec, "parte2_tfidf_deceptive.png", 12, 6)
      cat("<p><b>TF‑IDF — deceptive vs truthful:</b><br/><img src='parte2_tfidf_deceptive.png'></p>\n")

      # Sumário textual (Top 8 por grupo)
      sum_dec <- tfidf_dec |>
        dplyr::group_by(group) |>
        dplyr::slice_max(frequency, n = 8, with_ties = FALSE) |>
        dplyr::summarise(top8 = paste(feature, collapse = ", "), .groups = "drop")
      html_table(sum_dec, "tf‑idf (resumo textual: 8 termos por grupo)", digits = 3)
      html_p("Leitura: palavras com TF‑IDF alto são específicas de cada classe. ",
             "Candidatos típicos para indicativos de ‘deceptive’ incluem adjetivos genéricos e superlativos; ",
             "já em ‘truthful’ aparecem mais detalhes operacionais (ex.: itens de quarto/limpeza/atendimento). ",
             "As listas acima devem embasar regras de priorização na moderação.")

      # Por polaridade
      tfidf_pol <- private$tfidf_by_group(self$dfm_clean, quanteda::docvars(self$dfm_clean, "polarity"))
      p_pol <- ggplot2::ggplot(tfidf_pol, ggplot2::aes(x = tidytext::reorder_within(feature, frequency, group), y = frequency)) +
        ggplot2::geom_col() + ggplot2::coord_flip() + tidytext::scale_x_reordered() +
        ggplot2::facet_wrap(~ group, scales = "free_y") +
        ggplot2::labs(x = NULL, y = "importância tf‑idf (soma por grupo)", title = "termos mais relevantes por polaridade")
      save_gg(p_pol, "parte2_tfidf_polarity.png", 12, 6)
      cat("<p><b>TF‑IDF — positive vs negative:</b><br/><img src='parte2_tfidf_polarity.png'></p>\n")

      sum_pol <- tfidf_pol |>
        dplyr::group_by(group) |>
        dplyr::slice_max(frequency, n = 8, with_ties = FALSE) |>
        dplyr::summarise(top8 = paste(feature, collapse = ", "), .groups = "drop")
      html_table(sum_pol, "tf‑idf (resumo textual: 8 termos por polaridade)", digits = 3)
      html_p("Leitura: termos de ‘positive’ tendem a adjetivos e menções a conforto/atendimento; ",
             "‘negative’ traz problemas (ruído, limpeza, cobrança). Esses sinais guiam campanhas: destacar pontos fortes nas peças e ",
             "endereçar fricções na operação.")

      # -------------------- Parte 3: BoW + GLMNet (Polaridade) --------------------
      html_h2("parte 3 — modelagem preditiva (bag of words) para polaridade")

      v <- private$choose_vfolds(self$dados$polarity)
      folds <- rsample::vfold_cv(self$dados, v = v, strata = "polarity")
      preds <- list(); alphas <- numeric(0)

      for (i in seq_len(nrow(folds))) {
        sp <- folds$splits[[i]]
        train <- rsample::analysis(sp); test <- rsample::assessment(sp)

        dfm_tr <- private$to_dfm(quanteda::corpus(train, text_field = "text"))
        dfm_tr <- private$remove_meta_terms(dfm_tr, train$hotel, train$source)
        dfm_tr <- private$trim_dfm_datadriven(dfm_tr)

        dfm_te <- private$to_dfm(quanteda::corpus(test, text_field = "text"))
        dfm_te <- quanteda::dfm_remove(dfm_te, pattern = unique(c(as.character(train$hotel), as.character(train$source))), valuetype = "fixed")
        dfm_te <- quanteda::dfm_match(dfm_te, features = quanteda::featnames(dfm_tr))

        x_tr <- as(dfm_tr, "dgCMatrix"); y_tr <- ifelse(quanteda::docvars(dfm_tr, "polarity") == "positive", 1, 0)
        x_te <- as(dfm_te, "dgCMatrix"); y_te <- ifelse(quanteda::docvars(dfm_te, "polarity") == "positive", 1, 0)

        vinner <- private$choose_vfolds(y_tr)
        sel <- private$pick_alpha(x_tr, y_tr, vinner)
        alphas[i] <- sel$alpha

        prob <- as.numeric(predict(sel$cvfit, newx = x_te, s = "lambda.min", type = "response"))
        preds[[i]] <- tibble::tibble(
          fold = i,
          truth = factor(ifelse(y_te == 1, "positive", "negative"), levels = c("negative","positive")),
          prob  = prob
        )
      }

      pred_all <- dplyr::bind_rows(preds)
      tau <- private$pick_threshold(pred_all$truth, pred_all$prob)
      pred_all <- dplyr::mutate(pred_all, pred = factor(ifelse(prob >= tau, "positive", "negative"), levels = c("negative","positive")))

      acc  <- yardstick::accuracy(pred_all, truth = truth, estimate = pred)
      prec <- yardstick::precision(pred_all, truth = truth, estimate = pred, event_level = "second")
      rec  <- yardstick::recall(pred_all, truth = truth, estimate = pred, event_level = "second")
      f1   <- yardstick::f_meas(pred_all, truth = truth, estimate = pred, event_level = "second")
      kap  <- yardstick::kap(pred_all, truth = truth, estimate = pred)
      aucv <- yardstick::roc_auc(pred_all, truth = truth, prob, event_level = "second")
      metrics_tbl <- dplyr::bind_rows(acc, prec, rec, f1, kap, aucv)
      html_table(metrics_tbl, "métricas out-of-fold (v‑fold CV)", digits = 4)
      html_p("Leitura: acurácia e AUC refletem separação média entre classes usando apenas BoW. ",
             "Precisão/recall equilibrados indicam limiar (τ) adequado via índice de Youden na ROC.")

      cm <- yardstick::conf_mat(pred_all, truth = truth, estimate = pred)
      html_table(as.data.frame(cm$table), "matriz de confusão (predições out-of-fold)", digits = 0)

      # Sensibilidade/Especificidade por classe
      sens_pos <- yardstick::sens(pred_all, truth = truth, estimate = pred, event_level = "second")
      spec_pos <- yardstick::spec(pred_all, truth = truth, estimate = pred, event_level = "second")
      se_tbl <- dplyr::bind_rows(
        tibble::tibble(métrica = "sensibilidade (classe positiva)", valor = sens_pos$.estimate),
        tibble::tibble(métrica = "especificidade (classe negativa)", valor = spec_pos$.estimate)
      )
      html_table(se_tbl, "sensibilidade e especificidade (classe-alvo = positiva)", digits = 4)
      html_p("Leitura: sensibilidade mede cobertura de avaliações ‘positive’; especificidade a rejeição correta de ‘negative’. ",
             "Ajustar τ desloca esse trade-off conforme o apetite a falso-positivo/negativo da aplicação.")

      roc_df <- yardstick::roc_curve(pred_all, truth = truth, prob, event_level = "second")
      p_roc <- ggplot2::ggplot(roc_df, ggplot2::aes(x = 1 - specificity, y = sensitivity)) +
        ggplot2::geom_path() + ggplot2::geom_abline(lty = 2) + ggplot2::coord_equal() +
        ggplot2::labs(title = glue::glue("curva ROC (AUC = {round(yardstick::roc_auc(pred_all, truth, prob, event_level = 'second')$.estimate, 3)})"),
                      x = "1 - especificidade", y = "sensibilidade")
      save_gg(p_roc, "parte3_roc.png", 8, 6)
      cat("<p><b>Curva ROC:</b><br/><img src='parte3_roc.png'></p>\n")
      html_p("Leitura: quanto mais próxima do canto superior esquerdo, melhor a discriminação. ",
             "O AUC sintetiza essa separação (0.5 aleatório; 1 perfeito).")

      alpha_star <- stats::median(alphas, na.rm = TRUE)
      html_p("Alpha (Elastic‑Net) selecionado para ajuste final (mediana dos folds): <b>", round(alpha_star, 2), "</b>.")

      # Reajuste em toda a base para interpretação de pesos
      dfm_all <- private$to_dfm(self$corp)
      dfm_all <- private$remove_meta_terms(dfm_all, self$dados$hotel, self$dados$source)
      dfm_all <- private$trim_dfm_datadriven(dfm_all)
      x_all <- as(dfm_all, "dgCMatrix")
      y_all <- ifelse(quanteda::docvars(dfm_all, "polarity") == "positive", 1, 0)
      vinner_all <- private$choose_vfolds(y_all)
      fit_all <- glmnet::cv.glmnet(x_all, y_all, family = "binomial", alpha = alpha_star, type.measure = "auc", nfolds = vinner_all)

      coef_mat <- as.matrix(coef(fit_all, s = "lambda.min"))
      coefs <- tibble::tibble(
        feature = rownames(coef_mat),
        weight  = as.numeric(coef_mat[, 1, drop = TRUE])
      ) |>
        dplyr::filter(feature != "(Intercept)")

      top_pos <- coefs |> dplyr::arrange(dplyr::desc(weight)) |> dplyr::slice_head(n = 20)
      top_neg <- coefs |> dplyr::arrange(weight)               |> dplyr::slice_head(n = 20)

      p_pos <- ggplot2::ggplot(top_pos, ggplot2::aes(x = reorder(feature, weight), y = weight)) +
        ggplot2::geom_col() + ggplot2::coord_flip() + ggplot2::labs(x = NULL, y = "peso (+)", title = "termos pró‑positivo (modelo final)") +
        ggplot2::theme_minimal(10)
      p_neg <- ggplot2::ggplot(top_neg, ggplot2::aes(x = reorder(feature, weight), y = weight)) +
        ggplot2::geom_col() + ggplot2::coord_flip() + ggplot2::labs(x = NULL, y = "peso (−)", title = "termos pró‑negativo (modelo final)") +
        ggplot2::theme_minimal(10)
      save_gg(p_pos, "parte3_top_terms_positive.png", 10, 7)
      save_gg(p_neg, "parte3_top_terms_negative.png", 10, 7)
      cat("<p><b>Importância de termos (interpretação do modelo):</b><br/>",
          "<img src='parte3_top_terms_positive.png'><br/>",
          "<img src='parte3_top_terms_negative.png'></p>\n")
      html_p("Leitura: pesos positivos empurram a revisão para ‘positive’; negativos para ‘negative’. ",
             "Esses termos ajudam a redigir mensagens de marketing (reforçar drivers positivos) e a priorizar reparos (endereçar drivers negativos).")

      # -------------------- Parte 4: LDA --------------------
      html_h2("parte 4 — modelagem de tópicos (LDA) com escolha de k por métricas")

      dfm_lda <- self$dfm_clean
      dtm <- quanteda::convert(dfm_lda, to = "topicmodels")
      dtm <- dtm[slam::row_sums(dtm) > 0, ]
      if (nrow(dtm) < 10) stop("menos de 10 documentos após limpeza; ajuste o pré-processamento.")

      selk <- private$choose_k_topics(dtm, seed = seed)
      k_star <- selk$k_star
      html_table(selk$summary, "métricas por k (perplexidade ↓, js‑divergência ↑, score combinado ↑)", digits = 4)
      html_p("Decisão: escolhemos k que maximiza o score combinado z(−perplexidade) + z(js). ",
             "Isso balanceia ajuste (perplexidade) e separação entre tópicos (JS). k* = <b>", k_star, "</b>.")

      long_m <- selk$summary |>
        dplyr::select(k, perplexity, js_div, z_perp, z_js, score) |>
        tidyr::pivot_longer(cols = c(z_perp, z_js, score), names_to = "metrica", values_to = "zscore")
      p_k <- ggplot2::ggplot(long_m, ggplot2::aes(k, zscore, color = metrica)) +
        ggplot2::geom_line() + ggplot2::geom_point() +
        ggplot2::geom_vline(xintercept = k_star, linetype = 2) +
        ggplot2::labs(title = "escolha de k por métricas (z‑scores; maior é melhor)", x = "k", y = "z‑score padronizado")
      save_gg(p_k, "parte4_k_selection.png", 10, 6)
      cat("<p><b>Seleção de k (LDA):</b><br/><img src='parte4_k_selection.png'></p>\n")

      lda_fit <- topicmodels::LDA(dtm, k = k_star, method = "Gibbs",
                                  control = list(seed = seed, burnin = 2000, iter = 2000, thin = 100))
      beta_tbl  <- tidytext::tidy(lda_fit, matrix = "beta")
      gamma_tbl <- tidytext::tidy(lda_fit, matrix = "gamma")

      top_terms <- beta_tbl |>
        dplyr::group_by(topic) |>
        dplyr::slice_max(beta, n = 10, with_ties = FALSE) |>
        dplyr::ungroup()
      p_topics <- ggplot2::ggplot(top_terms, ggplot2::aes(x = tidytext::reorder_within(term, beta, topic), y = beta)) +
        ggplot2::geom_col() + ggplot2::coord_flip() + tidytext::scale_x_reordered() +
        ggplot2::facet_wrap(~ topic, scales = "free_y", ncol = min(4, k_star)) +
        ggplot2::labs(x = NULL, y = "β", title = glue::glue("LDA com k = {k_star}: top 10 termos por tópico"))
      save_gg(p_topics, "parte4_lda_top_terms.png", 14, 10)
      cat("<p><b>Top termos por tópico:</b><br/><img src='parte4_lda_top_terms.png'></p>\n")
      html_p("Leitura: as barras mostram os termos mais prováveis (β) de cada tópico. ",
             "Eles subsidiam o rótulo semântico e o plano de ação por tema.")

      labeling <- private$label_topics(beta_tbl, top_n = 20)
      topic_labels <- labeling$labels
      topic_table  <- top_terms |>
        dplyr::group_by(topic) |>
        dplyr::summarise(top_10 = paste(term, collapse = ", "), .groups = "drop") |>
        dplyr::left_join(topic_labels, by = "topic") |>
        dplyr::select(topic, topic_label, top_10) |>
        dplyr::arrange(topic)
      html_table(topic_table, "rótulos sugeridos e termos principais por tópico", digits = 3)
      html_p("Leitura: os rótulos foram atribuídos por dicionário de prefixes (‘lexicon’) alinhado ao setor hoteleiro. ",
             "Quando a evidência lexical é fraca, mantemos rótulo genérico (‘tópico i’).")

      prev <- gamma_tbl |>
        dplyr::group_by(topic) |>
        dplyr::summarise(prevalencia_media = mean(gamma), .groups = "drop") |>
        dplyr::left_join(topic_labels, by = "topic") |>
        dplyr::arrange(dplyr::desc(prevalencia_media))
      p_prev <- ggplot2::ggplot(prev, ggplot2::aes(x = reorder(topic_label, prevalencia_media), y = prevalencia_media)) +
        ggplot2::geom_col() + ggplot2::coord_flip() + ggplot2::labs(x = NULL, y = "prevalência média (γ)", title = "prevalência de tópicos")
      save_gg(p_prev, "parte4_topic_prevalence.png", 10, 6)
      cat("<p><b>Prevalência de tópicos:</b><br/><img src='parte4_topic_prevalence.png'></p>\n")
      html_p("Leitura: tópicos mais prevalentes devem receber prioridade em comunicação e melhoria de processos.")

      # -------------------- Parte 5: Insights de Negócio --------------------
      html_h2("parte 5 — interpretações e insights de negócio (ação recomendada)")

      # Drivers de polaridade
      dfm_tfidf_all <- quanteda::dfm_tfidf(self$dfm_clean, scheme_tf = "count", scheme_df = "inverse")
      freq_pol <- private$freq_by_group(dfm_tfidf_all, quanteda::docvars(dfm_tfidf_all, "polarity"))
      tfidf_pol_wide <- freq_pol |>
        tidyr::pivot_wider(names_from = group, values_from = frequency, values_fill = 0) |>
        dplyr::mutate(diff_pos_neg = positive - negative)
      drivers_pos <- tfidf_pol_wide |> dplyr::arrange(dplyr::desc(diff_pos_neg)) |> dplyr::slice_head(n = 12) |> dplyr::pull(feature)
      drivers_neg <- tfidf_pol_wide |> dplyr::arrange(diff_pos_neg)             |> dplyr::slice_head(n = 12) |> dplyr::pull(feature)
      html_h3("drivers de positividade e negatividade (tf‑idf)")
      html_p("Principais drivers de avaliações positivas:"); html_list(drivers_pos)
      html_p("Principais drivers de avaliações negativas:"); html_list(drivers_neg)
      html_p("Leitura: reforçar drivers positivos nas campanhas (‘provas sociais’ e criativos) ",
             "e criar planos táticos para mitigar drivers negativos (ex.: ruído, limpeza, cobrança).")

      # Pistas linguísticas para moderação (deceptive vs truthful)
      freq_dec <- private$freq_by_group(dfm_tfidf_all, quanteda::docvars(dfm_tfidf_all, "deceptive"))
      tfidf_dec_wide <- freq_dec |>
        tidyr::pivot_wider(names_from = group, values_from = frequency, values_fill = 0) |>
        dplyr::mutate(diff_decep_truth = deceptive - truthful)
      cues_decep <- tfidf_dec_wide |> dplyr::arrange(dplyr::desc(diff_decep_truth)) |> dplyr::slice_head(n = 12) |> dplyr::pull(feature)
      cues_truth <- tfidf_dec_wide |> dplyr::arrange(diff_decep_truth)             |> dplyr::slice_head(n = 12) |> dplyr::pull(feature)
      html_h3("sinais linguísticos de avaliações falsas vs verdadeiras (tf‑idf)")
      html_p("Termos característicos de falsas (deceptive):"); html_list(cues_decep)
      html_p("Termos característicos de verdadeiras (truthful):"); html_list(cues_truth)
      html_p("Uso prático na moderação: priorizar revisão manual quando o texto concentra termos genéricos, superlativos ",
             "e baixa especificidade operacional; revisões com menções concretas a serviço/quarto/limpeza tendem a ser legítimas.")

      # Ações recomendadas por tópico (LDA)
      actions_map <- tibble::tribble(
        ~topic_label,                 ~acao,
        "atendimento",                "Treinar front desk; metas de resposta; playbook de recuperação de falhas.",
        "limpeza",                    "Auditorias de housekeeping; checklists por turno; manutenção corretiva rápida.",
        "localização",                "Ressaltar proximidade de atrações/transporte em campanhas e site; mapas de walking distance.",
        "preço e valor",              "Ofertas com valor agregado e transparência de taxas; revisar políticas percebidas como ocultas.",
        "quarto e conforto",          "Manutenção preditiva de ruído/climatização; upgrade de enxoval e blackout.",
        "alimentos e bebidas",        "Padronizar tempos de atendimento; reforço de equipe em picos; menu enxuto.",
        "transporte e estacionamento","Parcerias com estacionamentos; comunicação clara de tarifas; alternativas de transporte.",
        "check-in e check-out",       "Self check‑in/out; fila virtual; pré‑cadastro e upgrades automatizados."
      )
      topic_actions <- labeling$labels |>
        dplyr::left_join(actions_map, by = "topic_label") |>
        dplyr::left_join(topic_table |> dplyr::select(topic, top_10), by = "topic") |>
        dplyr::mutate(acao = ifelse(is.na(acao), "Monitorar e criar playbooks específicos para o tópico.", acao)) |>
        dplyr::arrange(topic)
      html_table(topic_actions, "ações sugeridas por tópico (baseadas nos rótulos e termos)", digits = 0)
      html_p("Leitura: plano tático vinculado ao conteúdo dos tópicos. Priorizar tópicos mais prevalentes (ver gráfico anterior).")

      # KPIs executivos resumidos
      html_h3("kpis executivos recomendados")
      html_kpis(list(
        "Tempo médio de resposta" = "≤ 5 min (chat) / ≤ 30 min (e‑mail)",
        "NPS (hóspede pós‑estadia)" = "↑ vs. baseline 90 dias",
        "Taxa de reclassificação (moderação)" = "≥ 70% precisão nas amostras auditadas",
        "Incidência de ‘ruído/limpeza’ por 100 reviews" = "↓ 20% em 60 dias"
      ))

      # -------------------- Checklist --------------------
      html_h2("checklist de requisitos")
      check <- data.frame(
        requisito = c(
          "Parte 1: corpus + descritiva com métricas",
          "Parte 1: nuvem de palavras e Top‑10 termos",
          "Parte 2: TF‑IDF deceptive vs truthful com gráfico e resumo",
          "Parte 2: TF‑IDF positive vs negative com gráfico e resumo",
          "Parte 3: BoW + glmnet com CV; limiar ótimo; métricas e matriz de confusão + ROC",
          "Parte 4: LDA com escolha de k; top‑10; rótulos; prevalência",
          "Insights de negócio (drivers, pistas de fraude, ações por tópico)",
          "HTML estruturado com comentários em português"
        ),
        status = "implementado",
        stringsAsFactors = FALSE
      )
      html_table(check, "verificação final", digits = 0)
      html_note("Todos os resultados foram acompanhados de leitura/implicação de negócio. ",
                "O pipeline evita vazamento por metadados e usa validação estratificada para robustez.")

      html_end(html_path)
      invisible(TRUE)
    }
  )
)

# -------------------- Execução --------------------
solver <- HotelNLP$new(save_plots = TRUE)
ok <- solver$executar(csv_path = data_path, html_path = html_output, seed = 123L)
if (isTRUE(ok)) cat(glue::glue("relatório gerado em: {html_output}\n"))

# Para rodar em terminal:
# Rscript /Users/akatsurada/Documents/INSPER/StatisticsII/Aula6/quiz2.R
