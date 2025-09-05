# Libraries
pkgs_core <- c("modeldata","tidyverse","skimr","ggplot2","R6","knitr",
               "ggpointdensity","patchwork","glmnet","rpart","ranger",
               "rsample","yardstick","doParallel")

to_install <- pkgs_core[!pkgs_core %in% rownames(installed.packages())]
if (length(to_install)) install.packages(to_install, dependencies = TRUE)

library(modeldata)
library(tidyverse)
library(skimr)
library(ggplot2)
library(R6)
library(knitr)
library(ggpointdensity)
library(patchwork)
library(glmnet)  
library(rpart)                          
library(ranger)                
library(rsample)    
library(yardstick) 
library(doParallel)

#Utils
#Níveis de test train
align_to_train_levels <- function(train, test) {
  fcols <- names(train)[sapply(train, is.factor)]
  for (cl in fcols) {
    tr_lvls <- levels(train[[cl]])
    tr_lvls <- unique(c(tr_lvls, "Other"))  # garante "Other" apenas na referência de níveis
    t_chr <- as.character(test[[cl]])
    t_chr_clean <- ifelse(t_chr %in% tr_lvls, t_chr, "Other")
    test[[cl]] <- factor(t_chr_clean, levels = tr_lvls)
  }
  test
}

# embaralha níveis e mapeia nível->fold (group k-fold)
make_group_foldid <- function(group, v = 5L, seed = 123L) {
  set.seed(seed)
  g  <- as.factor(group)
  lv <- sample(levels(g))
  fmap <- setNames(rep(seq_len(v), length.out = length(lv)), lv)
  unname(as.integer(fmap[as.character(g)]))
}

# Folders e path
safe_get_script_dir <- function() {
  ca <- commandArgs(trailingOnly = FALSE)
  m <- grepl("^--file=", ca)
  if (any(m)) return(dirname(normalizePath(sub("^--file=", "", ca[m][1]))))
  of <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(of)) return(dirname(normalizePath(of)))
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    p <- rstudioapi::getActiveDocumentContext()$path
    if (nzchar(p)) return(dirname(normalizePath(p)))
  }
  normalizePath(getwd())
}

# Formata números c/ R$ e ','
fmt_real <- function(...) scales::label_currency(prefix = "R$ ", big.mark = ".", decimal.mark = ",")(...)

# Dps de pesquisar, aparentemente p/ targets c/ log1p é preciso usar essa ideia de 'Duan smearing'. 
#Talvez haja uma alternativa melhor (pesquisar)
# Aparentemente é o padráo p retransformar a variável de volta do logaritimo
smear_backtransform <- function(pred_log, resid_log) {
  s <- mean(exp(resid_log), na.rm = TRUE) 
  yhat <- exp(pred_log) * s - 1         
  pmax(0, yhat)
}

# Teste de cores
COL_POINT <- "#2D3748"  # cinza escuro 
COL_LINE  <- "#0072B2"  # azul p/ linha 
COL_BAR   <- "#0072B2"  # azul p/ barras
COL_GRID  <- "#E5E7EB"  # grade clarinha
COL_TICK  <- "#9CA3AF"  # ticks
COL_TEXT  <- "#111827"  # quase preto p/ textos

# Tema visual 
theme_andre <- function(base_size = 15, base_family = "") {
  ggplot2::theme_minimal(base_size = base_size, base_family = base_family) +
    ggplot2::theme(
      text = ggplot2::element_text(color = COL_TEXT),
      plot.title.position = "plot",
      plot.title   = ggplot2::element_text(face = "bold"),
      plot.subtitle= ggplot2::element_text(margin = ggplot2::margin(b = 6)),
      axis.title   = ggplot2::element_text(),
      axis.text    = ggplot2::element_text(color = COL_TEXT),
      axis.ticks   = ggplot2::element_line(color = COL_TICK),
      panel.grid.major = ggplot2::element_line(color = COL_GRID, linewidth = 0.3),
      panel.grid.minor = ggplot2::element_blank(),
      strip.background = ggplot2::element_rect(fill = "#F8FAFC", color = NA),
      strip.text  = ggplot2::element_text(face = "bold"),
      legend.position = "none",
      plot.background = ggplot2::element_rect(fill = "white", color = NA),
      plot.caption = ggplot2::element_text(color = "#6B7280", size = ggplot2::rel(0.9)),
      plot.margin  = ggplot2::margin(10, 16, 10, 10)
    )
}

# Padroniza "Negotiation Type" em 2 labels simples (limpa, lower case, sinonimos)
norm_neg_type <- function(x) {
  x0 <- x %>% as.character() %>% stringr::str_squish() %>% stringr::str_to_lower()
  dplyr::case_when(
    x0 %in% c("rent","aluguel","for rent","aluga","aluguel mensal","alug.") ~ "rent",
    x0 %in% c("sale","venda","for sale","vende") ~ "sale",
    TRUE ~ NA_character_
  )
}

# Load do csv
script_dir <- safe_get_script_dir()
candidates <- c(
  file.path(script_dir, "sao-paulo-properties-april-2019.csv"),
  file.path(getwd(),   "sao-paulo-properties-april-2019.csv")
)
csv_path <- candidates[file.exists(candidates)][1]
if (is.na(csv_path)) {
  stop("CSV não encontrado. Coloque 'sao-paulo-properties-april-2019.csv' na mesma pasta do script.")
}
sp <- readr::read_csv(csv_path, show_col_types = FALSE)

# Check rápido 
sp %>% head()
sp %>% skim()

# Classe abstrata base pros exercicios
Exercicio <- R6::R6Class(
  "Exercicio",
  public = list(
    data = NULL,
    initialize = function(data) { self$data <- data },  # guarda os dados
    run = function() stop("método abstrato"),           # roda a análise
    as_md = function(...) stop("método abstrato")       # devolve o MD
  )
)

# Exercício 1a: Frequência de Negotiation Type 
ExA <- R6::R6Class(
  "ExA",
  inherit = Exercicio,
  public = list(
    neg_col = NULL,
    freq_raw = NULL, 
    freq_total = NULL,
    freq_valid = NULL,
    resumo = NULL,

    run = function() {
      # Procura o nome certo 
      cand <- c("Negotiation.Type", "Negotiation Type", "negotiation_type", "Negotiation_Type")
      nm <- intersect(cand, names(self$data))
      self$neg_col <- if (length(nm)) nm[1] else NA_character_
      if (is.na(self$neg_col)) stop("Coluna 'Negotiation Type' não encontrada.")


# Frequências como vieram no CSV e c/ missing 
self$freq_raw <- self$data %>%
  transmute(negotiation_raw = as.character(.data[[self$neg_col]])) %>%
  mutate(negotiation_raw = ifelse(is.na(negotiation_raw) | !nzchar(negotiation_raw),
                                  "Missing", negotiation_raw)) %>%
  count(negotiation_raw, name = "freq_abs") %>%
  arrange(desc(freq_abs)) %>%
  mutate(freq_rel = freq_abs / sum(freq_abs),
         pct = round(100 * freq_rel, 2))

      # Padroniza labels de rent, sale e conta
      d <- self$data %>%
        mutate(negotiation_std = norm_neg_type(.data[[self$neg_col]])) %>%
        transmute(negotiation_std)

      self$freq_total <- d %>%
        mutate(negotiation_std = tidyr::replace_na(negotiation_std, "Missing")) %>%
        count(negotiation_std, name = "freq_abs") %>%
        arrange(desc(freq_abs)) %>%
        mutate(freq_rel = freq_abs / sum(freq_abs),
              pct = round(100 * freq_rel, 2))

      self$freq_valid <- d %>%
        filter(!is.na(negotiation_std)) %>%
        count(negotiation_std, name = "freq_abs") %>%
        arrange(desc(freq_abs)) %>%
        mutate(freq_rel = freq_abs / sum(freq_abs),
               pct = round(100 * freq_rel, 2))

      # Resuminho útil
      tot_n   <- nrow(d)
      miss_n  <- sum(is.na(d$negotiation_std))
      miss_pc <- if (tot_n == 0) 0 else round(100 * miss_n / tot_n, 2)

      if (nrow(self$freq_valid) > 0) {
        top_v  <- self$freq_valid %>% slice_max(freq_abs, n = 1, with_ties = FALSE)
        top_nm <- top_v$negotiation_std
        top_pc <- top_v$pct
      } else {
        top_nm <- "Sem categoria válida"
        top_pc <- NA_real_
      }

      self$resumo <- list(
        total_n     = tot_n,
        missing_n   = miss_n,
        missing_pct = miss_pc,
        top_name    = top_nm,
        top_pct     = top_pc
      )

      invisible(self)
    },

    as_md = function() {
      tbl_total <- knitr::kable(self$freq_total, format = "pipe",
                                col.names = c("Categoria","Freq. Absoluta","Freq. Relativa","%"))
      tbl_valid <- knitr::kable(self$freq_valid, format = "pipe",
                                col.names = c("Categoria","Freq. Absoluta","Freq. Relativa","%"))

      c(
        "# Exercício (1a): Frequência de `Negotiation Type`",
        "",
        "### O que foi feito",
        "- Padronizei labels básicos (ex.: 'aluguel' → 'rent') e contei.",
        "- Mostra o **Total** (c/ missing e s/ missing).",
        "",

        if (!is.null(self$freq_raw) && nrow(self$freq_raw) > 0) {
            c(
              "### Frequências — Original (como no CSV, com *Missing*)",
              paste(
                knitr::kable(
                  self$freq_raw,
                  format = "pipe",
                  col.names = c("Categoria (raw)","Freq. Absoluta","Freq. Relativa","%")
                ),
                collapse = "\n"
              )
            )
          } else {
            NULL
          },
        "",
        "### Frequências — Normalizadas (com *Missing*)",
        paste(tbl_total, collapse = "\n"),
        "",
        "### Frequências — Normalizadas (sem *Missing*)",
        paste(tbl_valid, collapse = "\n"),

        "",
        "### Interpretação",
        paste0("- Registros: **", self$resumo$total_n, "**."),
        paste0("- Missing em `Negotiation Type`: **", self$resumo$missing_n, " (", self$resumo$missing_pct, "%)**."),
        paste0("- Categoria que mais aparece: **", self$resumo$top_name, "**",
               if (!is.na(self$resumo$top_pct)) paste0(" (", self$resumo$top_pct, "% dos válidos).") else ".")
      )
    }
  )
)

# Exercício 1b: Dispersão Condo vs Price
ExB <- R6::R6Class(
  "ExB",
  inherit = Exercicio,
  public = list(
    price_col = NULL,
    condo_col = NULL,
    df_clean = NULL,
    df_plot  = NULL,
    stats = NULL,
    plot = NULL,

    run = function() {
      # Acha as colunas de preço e condo
      pcand <- c("Price","price"); ccand <- c("Condo","condo","Condominium")
      nm_price <- intersect(pcand, names(self$data))
      nm_condo <- intersect(ccand, names(self$data))
      self$price_col <- if (length(nm_price)) nm_price[1] else NA_character_
      self$condo_col <- if (length(nm_condo)) nm_condo[1] else NA_character_
      if (is.na(self$price_col) || is.na(self$condo_col)) stop("Colunas 'Price' e/ou 'Condo' não encontradas.")

      # Limpa: valores válidos e > 0
      Price <- suppressWarnings(as.numeric(self$data[[self$price_col]]))
      Condo <- suppressWarnings(as.numeric(self$data[[self$condo_col]]))
      ok    <- is.finite(Price) & is.finite(Condo)
      df0   <- data.frame(Condo = Condo[ok], Price = Price[ok], check.names = FALSE)
      df    <- df0[df0$Condo > 0 & df0$Price > 0, , drop = FALSE]
      if (nrow(df) < 5) stop("Dados insuficientes pro (b) depois da limpeza (>= 5 pontos).")
      self$df_clean <- df

      # Métricas, td em log10 
      lx <- log10(df$Condo); ly <- log10(df$Price)
      pearson  <- suppressWarnings(stats::cor(lx, ly, method = "pearson"))
      spearman <- suppressWarnings(stats::cor(lx, ly, method = "spearman"))
      mdl_log  <- stats::lm(ly ~ lx)

      a        <- unname(stats::coef(mdl_log)[1])
      beta     <- unname(stats::coef(mdl_log)[2])
      r2_log   <- summary(mdl_log)$r.squared

      # Corta extremos só pra visual ficar menos "achatado"
      qx <- quantile(df$Condo, probs = c(0.01, 0.99), na.rm = TRUE)
      qy <- quantile(df$Price, probs = c(0.01, 0.99), na.rm = TRUE)
      dfp <- df %>% dplyr::filter(Condo >= qx[1], Condo <= qx[2], Price >= qy[1], Price <= qy[2])
      self$df_plot <- dfp

      # Posição da caixinha de métricas (canto superior esquerdo dentro do painel)
      x_min <- min(dfp$Condo, na.rm = TRUE); x_max <- max(dfp$Condo, na.rm = TRUE)
      y_min <- min(dfp$Price, na.rm = TRUE); y_max <- max(dfp$Price, na.rm = TRUE)

      # Dispersão em log–log, cor por densidade e linha
      self$plot <- ggplot2::ggplot(dfp, ggplot2::aes(x = Condo, y = Price)) +
        ggpointdensity::geom_pointdensity(adjust = 1.0, size = 0.9) +
        ggplot2::scale_color_viridis_c(option = "viridis", name = "Densidade") +
        ggplot2::guides(color = ggplot2::guide_colorbar(
          title.position = "top", title.hjust = 0.5,
          barheight = grid::unit(55, "pt"), barwidth = grid::unit(8, "pt")
        )) +
        ggplot2::scale_x_log10(breaks = scales::log_breaks(n = 6), labels = fmt_real) +
        ggplot2::scale_y_log10(breaks = scales::log_breaks(n = 6), labels = fmt_real) +
        ggplot2::annotation_logticks(
          sides = "bl",
          short = grid::unit(1.5, "mm"), mid = grid::unit(2, "mm"), long = grid::unit(3, "mm"),
          linewidth = 0.25, color = COL_TICK
        ) +
        ggplot2::stat_function(
          fun = function(x) (10^a) * x^beta,
          linewidth = 1.0, alpha = 1.0, color = COL_LINE
        ) +
        ggplot2::labs(
          title = "Condomínio vs Preço (dispersão em log–log)",
          # Substitui o texto antigo por métricas legíveis (essa parte não estava legível):
          subtitle = sprintf(
            "n=%d (plot=%d)\nPearson(log)=%.2f · Spearman(log)=%.2f · β=%.2f · R²=%.2f",
            nrow(df), nrow(dfp), pearson, spearman, beta, r2_log
          ),
          x = "Condomínio (R$)", y = "Preço anunciado (R$)"
        ) +
        theme_andre(15) +
        ggplot2::theme(
          legend.position = "right",
          plot.subtitle = ggplot2::element_text(margin = ggplot2::margin(t = 6)),
          plot.margin  = ggplot2::margin(10, 16, 16, 10)
        ) +
        ggplot2::coord_cartesian(clip = "off")


      self$stats <- list(
        n_total = nrow(df0),
        n_clean = nrow(df),
        n_plot  = nrow(dfp),
        pearson = pearson,
        spearman = spearman,
        elasticidade = beta,
        r2_log = r2_log
      )
      invisible(self)
    },

    as_md = function(image_dir) {
      dir.create(image_dir, showWarnings = FALSE, recursive = TRUE)
      plot_path <- file.path(image_dir, "b_scatter_condo_price_scatter_loglog.png")
      ggplot2::ggsave(filename = plot_path, plot = self$plot,
                      width = 10, height = 6, dpi = 320, bg = "white")

      fnum <- function(x, d = 3) if (is.na(x)) "NA" else format(round(x, d), nsmall = d, trim = TRUE)

      c(
        "# Exercício 1b: Dispersão do preço do condomínio (Condo) e do preço anunciado (Price)",
        "",
        "### O que foi feito",
        "- Dispersão com **cor pela densidade**",
        "- Escalas **log–log** e **linha do ajuste**",
        "- Cortes **visuais** 1%–99% pra não achatar.",
        "",
        "### Gráfico",
        paste0("![](", basename(plot_path), ")"),
        "",
        "### Métricas (sem cortes de percentil)",
        paste0("- Pearson (log): ", fnum(self$stats$pearson)),
        paste0("- Spearman (log): ", fnum(self$stats$spearman)),
        paste0("- Elasticidade (log–log): ", fnum(self$stats$elasticidade),
              " · efeito de +10% no condomínio ≈ ",
              fnum(((1.10^self$stats$elasticidade) - 1) * 100, 1), "%"),
        paste0("- R² (log–log): ", fnum(self$stats$r2_log, 3)),
        paste0("- Pontos limpos: ", self$stats$n_clean, " (de ", self$stats$n_total,
              "); no desenho: ", self$stats$n_plot, ")."),
        "",
        "### Interpretação",
        "- **Relação positiva, mas fraca**: condomínio maior tende a preço maior, c/ bastante variação."
      )
    }
  )
)

# Exercício 1c: Mesmo do (b), mas separado por tipo (rent vs sale)
ExC <- R6::R6Class(
  "ExC",
  inherit = Exercicio,
  public = list(
    neg_col = NULL,
    price_col = NULL,
    condo_col = NULL,
    df_clean = NULL,
    df_plot  = NULL,
    stats_by = NULL,
    plot = NULL,

  run = function() {
    # Encontra a coluna de tipo de negociação e padroniza (rent/sale)
    cand <- c("Negotiation.Type", "Negotiation Type", "negotiation_type", "Negotiation_Type")
    nm <- intersect(cand, names(self$data))
    self$neg_col <- if (length(nm)) nm[1] else NA_character_
    if (is.na(self$neg_col)) stop("Coluna 'Negotiation Type' não encontrada.")

    d_neg <- self$data %>%
      mutate(neg_std = norm_neg_type(.data[[self$neg_col]])) %>%
      transmute(neg_std)

    # Acha preço/condomínio
    pcand <- c("Price","price"); ccand <- c("Condo","condo","Condominium")
    nm_price <- intersect(pcand, names(self$data))
    nm_condo <- intersect(ccand, names(self$data))
    self$price_col <- if (length(nm_price)) nm_price[1] else NA_character_
    self$condo_col <- if (length(nm_condo)) nm_condo[1] else NA_character_
    if (is.na(self$price_col) || is.na(self$condo_col)) stop("Colunas 'Price' e/ou 'Condo' não encontradas.")

    Price <- suppressWarnings(as.numeric(self$data[[self$price_col]]))
    Condo <- suppressWarnings(as.numeric(self$data[[self$condo_col]]))
    ok    <- is.finite(Price) & is.finite(Condo)

    df0 <- tibble(Condo = Condo[ok], Price = Price[ok]) %>%
      bind_cols(d_neg[ok, , drop = FALSE]) %>%
      filter(!is.na(neg_std), neg_std %in% c("rent","sale"), Condo > 0, Price > 0)

    if (nrow(df0) < 10) stop("Poucos dados pro (c) depois da limpeza.")
    self$df_clean <- df0

    # (1) Métricas por regime (s/ cortes; tudo em log10) e aí cria self$stats_by
    split_list <- split(df0, df0$neg_std)
    stats_tbl <- lapply(names(split_list), function(k) {
      dd <- split_list[[k]]
      lx <- log10(dd$Condo); ly <- log10(dd$Price)

      pearson  <- suppressWarnings(stats::cor(lx, ly, method = "pearson"))
      spearman <- suppressWarnings(stats::cor(lx, ly, method = "spearman"))
      mdl_log  <- stats::lm(ly ~ lx)

      a        <- unname(stats::coef(mdl_log)[1])
      b        <- unname(stats::coef(mdl_log)[2])
      r2_log   <- summary(mdl_log)$r.squared
      tibble(tipo = k, n = nrow(dd),
            pearson = pearson, spearman = spearman,
            intercepto = a, elasticidade = b, r2_log = r2_log)
    }) %>% bind_rows()
    self$stats_by <- stats_tbl %>% mutate(Tipo = recode(tipo, rent = "Aluguel", sale = "Venda"))

    # (2) Cortes visuais 1–99% por 'faceta'
    qx_by <- df0 %>% group_by(neg_std) %>%
      summarise(x_lo = quantile(Condo, 0.01, na.rm = TRUE),
                x_hi = quantile(Condo, 0.99, na.rm = TRUE),
                .groups = "drop")
    qy_by <- df0 %>% group_by(neg_std) %>%
      summarise(y_lo = quantile(Price, 0.01, na.rm = TRUE),
                y_hi = quantile(Price, 0.99, na.rm = TRUE),
                .groups = "drop")

    dfp <- df0 %>%
      left_join(qx_by, by = "neg_std") %>%
      left_join(qy_by, by = "neg_std") %>%
      filter(Condo >= x_lo, Condo <= x_hi, Price >= y_lo, Price <= y_hi) %>%
      select(-x_lo, -x_hi, -y_lo, -y_hi)
    self$df_plot <- dfp

    # (3) Limites comuns e grid de x para as linhas
    dfp2 <- dfp %>% mutate(Tipo = recode(neg_std, rent = "Aluguel", sale = "Venda"))
    x_limits <- range(dfp2$Condo, na.rm = TRUE)
    y_limits <- range(dfp2$Price, na.rm = TRUE)

    x_grid <- 10^seq(log10(x_limits[1]), log10(x_limits[2]), length.out = 200)

    # (4) Linhas do ajuste por faceta (usa self$stats_by calculado antes)
    lines_df <- self$stats_by %>%
      transmute(Tipo, a = intercepto, b = elasticidade) %>%
      tidyr::crossing(x = x_grid) %>%
      mutate(y = (10^a) * x^b) %>%
      dplyr::filter(x >= x_limits[1], x <= x_limits[2],
                    y >= y_limits[1], y <= y_limits[2])

    # Dispersão + linhas
    self$plot <- ggplot2::ggplot(dfp2, ggplot2::aes(x = Condo, y = Price)) +
      ggpointdensity::geom_pointdensity(adjust = 1.0, size = 0.8) +
      ggplot2::scale_color_viridis_c(option = "viridis", name = "Densidade") +
      ggplot2::guides(color = ggplot2::guide_colorbar(
        direction    = "horizontal",
        title.position = "left", title.hjust = 1,
        barwidth     = grid::unit(140, "pt"),
        barheight    = grid::unit(8, "pt")
      )) +
      ggplot2::scale_x_log10(limits = x_limits,
                            breaks = scales::log_breaks(n = 6),
                            labels = fmt_real) +
      ggplot2::scale_y_log10(limits = y_limits,
                            breaks = scales::log_breaks(n = 6),
                            labels = fmt_real) +
      ggplot2::annotation_logticks(
        sides = "bl",
        short = grid::unit(1.5, "mm"), mid = grid::unit(2, "mm"), long = grid::unit(3, "mm"),
        linewidth = 0.25, color = COL_TICK
      ) +
      ggplot2::geom_line(
        data = lines_df,
        ggplot2::aes(x = x, y = y),
        inherit.aes = FALSE,
        linewidth = 1.0, alpha = 1.0, color = COL_LINE
      ) +
      ggplot2::facet_wrap(~ Tipo, ncol = 2) +
      ggplot2::labs(
        title = "Condomínio vs Preço por regime (dispersão em log–log)",
        subtitle = "Cores = densidade local; cortes visuais 1%–99% por faceta; linha = ajuste por faceta",
        x = "Condomínio (R$)", y = "Preço anunciado (R$)"
      ) +
      theme_andre(15) +
      ggplot2::theme(
        legend.position   = "bottom",
        legend.box.margin = ggplot2::margin(t = 6),
        legend.margin     = ggplot2::margin(t = 2),
        plot.margin       = ggplot2::margin(10, 16, 28, 10)
      ) +
      ggplot2::coord_cartesian(clip = "off")

    invisible(self)
  },

    as_md = function(image_dir) {
      dir.create(image_dir, showWarnings = FALSE, recursive = TRUE)
      plot_path <- file.path(image_dir, "c_scatter_condo_price_facets_scatter_loglog.png")
      ggplot2::ggsave(filename = plot_path, plot = self$plot, width = 10, height = 6,
                      dpi = 320, bg = "white")

      stats_fmt <- self$stats_by %>%
        mutate(across(c(pearson, spearman, elasticidade, r2_log), ~round(., 3))) %>%
        mutate(`+10% em condomínio → Δ% preço` = paste0(
         round(((1.10^elasticidade) - 1) * 100, 1), "%")) %>%
        select(Tipo, n, Pearson = pearson, Spearman = spearman,
               `Elasticidade (log–log)` = elasticidade, `R² (log–log)` = r2_log,
               `Variação no preço` = `+10% em condomínio → Δ% preço`)
      
      stats_tbl_md <- knitr::kable(stats_fmt, format = "pipe")
      
      c(
        "# Exercício 1c: Dispersão com facetas por `Negotiation Type` (rent vs sale)",
        "",
        "### O que foi feito",
        "- Dispersão com **cor pela densidade** **e** **linha do ajuste** em cada faceta.",
        "- Cortes **visuais** 1%–99% por faceta; **eixos iguais**",
        "- Escalas **log–log** com **limites iguais**",
        "",
        "### Gráfico (dispersão em log–log, com linha do ajuste)",
        paste0("![](", basename(plot_path), ")"),
        "",
        "### Números por regime (sem cortes nos cálculos)",
        paste(stats_tbl_md, collapse = "\n")

      )
    }
  )
)

# Exercício 1d: Top 10 distritos por frequência 
ExD <- R6::R6Class(
  "ExD",
  inherit = Exercicio,
  public = list(
    dist_col = NULL,
    tabela_top10 = NULL,
    plot = NULL,
    resumo = NULL,

    run = function(n_top = 10) {
      # Procura a coluna de bairro/distrito
      cand <- c("District","district","Neighborhood","neighborhood","Neighbourhood",
                "neighbourhood","Bairro","bairro")
      nm <- intersect(cand, names(self$data))
      self$dist_col <- if (length(nm)) nm[1] else NA_character_
      if (is.na(self$dist_col)) stop("Coluna de distrito/bairro não encontrada.")

      # Limpa sufixos e cria chave 'sem acento/caixa' pra juntar grafias parecidas. Daria p/ pensar em alguma alternativa melhor
      d <- self$data %>%
        mutate(
          district_raw = as.character(.data[[self$dist_col]]),
          district_left = district_raw %>%
            stringr::str_replace("(?i)\\s*/\\s*s[aã]o\\s*paulo.*$", "") %>%
            stringr::str_replace("(?i),\\s*s[aã]o\\s*paulo.*$", "") %>%
            stringr::str_replace("(?i)\\s*[-/]\\s*sp$", "") %>%
            stringr::str_squish(),
          district_left = na_if(district_left, ""),
          district_key = district_left %>%
            stringi::stri_trans_general("Latin-ASCII") %>%
            stringr::str_to_lower() %>%
            stringr::str_replace_all("[^a-z0-9 ]+", " ") %>%
            stringr::str_squish(),
          district_key = na_if(district_key, "")
        ) %>%
        select(district_left, district_key)

      tot_n   <- nrow(self$data)
      miss_n  <- sum(is.na(d$district_key))
      miss_pc <- if (tot_n == 0) 0 else round(100 * miss_n / tot_n, 2)

      d_valid <- d %>% filter(!is.na(district_key))
      if (nrow(d_valid) == 0) stop("Sem distritos válidos depois da limpeza.")

      counts <- d_valid %>% count(district_key, name = "freq_abs")

      # Usando o label + comum 
      label_map <- d_valid %>%
        count(district_key, district_left, name = "n") %>%
        arrange(district_key, desc(n), district_left) %>%
        group_by(district_key) %>%
        summarise(District = dplyr::first(district_left), .groups = "drop")

      counts2 <- counts %>%
        left_join(label_map, by = "district_key") %>%
        arrange(desc(freq_abs), District) %>%
        mutate(freq_rel = freq_abs / sum(freq_abs),
               pct = round(100 * freq_rel, 2)) %>%
        select(District, freq_abs, freq_rel, pct)

      self$tabela_top10 <- counts2 %>% slice_head(n = min(n_top, nrow(counts2)))

      # Barras ordenadas (maior no topo)
      dfp <- self$tabela_top10 %>%
        mutate(
          District_wrapped = stringr::str_wrap(District, width = 28),
          District_wrapped = forcats::fct_reorder(District_wrapped, freq_abs, .desc = TRUE)

        )

      self$plot <- ggplot2::ggplot(dfp, ggplot2::aes(x = District_wrapped, y = freq_abs)) +
        ggplot2::geom_col(width = 0.68, fill = COL_BAR, color = COL_TEXT) +
        ggplot2::coord_flip(clip = "off") +
        ggplot2::geom_text(
          ggplot2::aes(label = format(freq_abs, big.mark = ".", decimal.mark = ",")),
          hjust = -0.05, size = 3.9, color = COL_TEXT
        ) +
        ggplot2::scale_y_continuous(
          labels = function(x) format(x, big.mark = ".", decimal.mark = ","),
          expand = ggplot2::expansion(mult = c(0, 0.12))
        ) +
        ggplot2::labs(
          title = paste0("Top ", nrow(self$tabela_top10), " distritos por frequência"),
          subtitle = "Chaves normalizadas (sem acentos/caixa); rótulos mantêm a grafia mais comum",
          x = NULL, y = "N de anúncios"
        ) +
        theme_andre(15)

      self$resumo <- list(
        total_n   = tot_n,
        valid_n   = nrow(d_valid),
        missing_n = miss_n,
        missing_pct = miss_pc
      )
      invisible(self)
    },

    as_md = function(image_dir) {
      dir.create(image_dir, showWarnings = FALSE, recursive = TRUE)
      plot_path <- file.path(image_dir, "d_top10_distritos.png")
      ggplot2::ggsave(filename = plot_path, plot = self$plot, width = 9, height = 6,
                      dpi = 320, bg = "white")

      tbl_md <- knitr::kable(
        self$tabela_top10 %>%
          mutate(`Freq. Relativa` = round(freq_rel, 3)) %>%
          select(Distrito = District,
                 `Freq. Absoluta` = freq_abs,
                 `Freq. Relativa`,
                 `%` = pct),
        format = "pipe"
      )

      c(
        "# Exercício 1d: Top 10 distritos por frequência",
        "",
        "### O que foi feito",
        "- Limpei sufixos (ex.: '/São Paulo', '- SP') e juntei grafias parecidas.",
        "- Usei a grafia mais comum como label",
        "",
        paste0("- Registros com distrito válido: **", self$resumo$valid_n, "** (de ",
               self$resumo$total_n, "); Missing: **", self$resumo$missing_n,
               " (", self$resumo$missing_pct, "%)**."),
        "",
        "### Tabela — Top 10",
        paste(tbl_md, collapse = "\n"),
        "",
        "### Gráfico — Frequências",
        paste0("![](", basename(plot_path), ")")
      )
    }
  )
)

# Exercício e, OLS, Ridge, LASSO + Árvore + RF + ElasticNet
ExE <- R6::R6Class(
  "ExE",
  inherit = Exercicio,
  public = list(
    neg_col = NULL, price_col = NULL, pred_cols = NULL, form_lm = NULL,
    seed = 12345, split_ratio = 0.80, s_rule = "lambda.1se",
    train = NULL, test = NULL,

    fit_lm = NULL, fit_ridge = NULL, fit_lasso = NULL, fit_enet = NULL, # + ENet
    fit_tree0 = NULL, fit_tree = NULL, fit_rf = NULL,
    alpha_grid = NULL, alpha_enet = NA_real_,                            
    metrics = NULL, meta = NULL,

    plot_metrics = NULL, plot_metrics_log = NULL, plot_pred = NULL, plot_path = NULL,
    # heatmaps de CV (um por modelo)
    plot_cv_ridge = NULL, plot_cv_lasso = NULL, plot_cv_enet = NULL,
    # heatmap para Random Forest (OOB MSE por mtry × min.node.size)
    plot_cv_rf = NULL,


    run = function(s_rule = c("lambda.1se","lambda.min"), fast = TRUE) {
      message("ExE: start (fast=", fast, ")")
      set.seed(self$seed)
      self$s_rule <- match.arg(s_rule)

      # 1) Colunas
      neg_cand <- c("Negotiation.Type","Negotiation Type","negotiation_type","Negotiation_Type")
      pcand    <- c("Price","price")
      nm_neg   <- intersect(neg_cand, names(self$data))
      nm_price <- intersect(pcand,    names(self$data))
      self$neg_col   <- if (length(nm_neg))   nm_neg[1]   else NA_character_
      self$price_col <- if (length(nm_price)) nm_price[1] else NA_character_
      if (is.na(self$neg_col) || is.na(self$price_col)) stop("Faltam Negotiation Type e/ou Price.")

      # 2) Preditoras
      preds_num <- c("Condo","Size","Rooms","Toilets","Suites","Parking",
                     "Elevator","Furnished","Swimming Pool","New","Latitude","Longitude")
      preds_cat <- c("District")
      self$pred_cols <- intersect(c(preds_num, preds_cat), names(self$data))
      if (!length(self$pred_cols)) stop("Sem preditoras disponíveis.")

      # 3) Dataset: rent; Price > 0 + filtro de coordenadas
      message("ExE: preparando dados ...")
      d0 <- self$data %>%
        mutate(neg_std = norm_neg_type(.data[[self$neg_col]])) %>%
        filter(neg_std == "rent") %>%
        transmute(Price = suppressWarnings(as.numeric(.data[[self$price_col]])),
                  !!!rlang::syms(self$pred_cols)) %>%
        filter(is.finite(Price), Price > 0)

      if (all(c("Latitude","Longitude") %in% names(d0))) {
        d0 <- d0 %>%
          mutate(
            Latitude  = ifelse(Latitude  == 0, NA_real_, Latitude),
            Longitude = ifelse(Longitude == 0, NA_real_, Longitude)
          ) %>%
          filter(
            is.na(Latitude)  | (Latitude  >= -24.5 & Latitude  <= -23.2),
            is.na(Longitude) | (Longitude >= -47.4 & Longitude <= -46.0)
          )
      }

      if ("District" %in% names(d0)) d0$District <- as.factor(d0$District)
      d <- tidyr::drop_na(d0)
      if (!"District" %in% names(d)) self$pred_cols <- setdiff(self$pred_cols, "District")
      if (nrow(d) < 50) stop("Poucos dados após filtros (>=50).")

      # 4) Split 80/20
      message("ExE: split train/test ...")
      spl <- rsample::initial_split(d, prop = self$split_ratio)
      self$train <- rsample::training(spl)
      self$test  <- rsample::testing(spl)

      # Lumping/níveis do District
      if ("District" %in% names(self$train)) {
        self$train$District <- forcats::fct_lump_n(self$train$District, n = 50, other_level = "Other")
        self$test <- align_to_train_levels(self$train, self$test)
      }

      # Fórmula + matrizes (usa termos do treino para o teste)
      self$pred_cols <- setdiff(names(self$train), "Price")
      self$form_lm   <- as.formula("log1p(Price) ~ .")
      tr_terms <- stats::terms(self$form_lm, data = self$train)
      x_tr <- stats::model.matrix(tr_terms, data = self$train)[, -1, drop = FALSE]
      x_te <- stats::model.matrix(tr_terms, data = self$test)[,  -1, drop = FALSE]
      y_tr_log <- log1p(self$train$Price)
      y_te     <- self$test$Price

      keep_te <- is.finite(y_te) & rowSums(is.na(x_te)) == 0
      if (!any(keep_te)) stop("Nenhuma linha de teste válida após alinhamento de níveis.")
      x_te_ok <- x_te[keep_te, , drop = FALSE]
      y_te_ok <- y_te[keep_te]

      # Aumentar velocidade
      kfold         <- if (fast) 5L else 10L
      rf_importance <- if (fast) "impurity" else "permutation"
      n_threads     <- max(1L, parallel::detectCores() - 1L)

      # Alpha pro Elastic Net
      self$alpha_grid <- if (fast) c(0.1, 0.3, 0.5, 0.7, 0.9) else seq(0.05, 0.95, by = 0.05)

      # 6.1) OLS
      message("ExE: OLS ...")

      self$fit_lm <- stats::lm(self$form_lm, data = self$train)
      pred_lm_log   <- as.numeric(predict(self$fit_lm, newdata = self$test[keep_te, , drop = FALSE]))
      resid_lm_log  <- residuals(self$fit_lm)                    # train tb precisa dos residuals em log1p! por isso o resultado estava estranho
      pred_lm       <- smear_backtransform(pred_lm_log, resid_lm_log)

      # 6.2) Ridge (s/ cluster)
      message("ExE: Ridge (cv.glmnet, k=", kfold, ", single-thread) ...")

      foldid <- if ("District" %in% names(self$train)) {
        make_group_foldid(self$train$District, v = kfold, seed = self$seed)
      } else {
        sample(rep(seq_len(kfold), length.out = nrow(self$train)))
      }

      self$fit_ridge <- glmnet::cv.glmnet(
        x = x_tr, y = y_tr_log, family = "gaussian",
        alpha = 0, nfolds = kfold, foldid = foldid, standardize = TRUE,
        type.measure = "mse", parallel = FALSE
      )
      pred_ridge_log    <- as.numeric(predict(self$fit_ridge, newx = x_te_ok, s = self$s_rule))
      pred_tr_ridge_log <- as.numeric(predict(self$fit_ridge, newx = x_tr,    s = self$s_rule)) #train tb precisa dos residuals em log1p! por isso o resultado estava estranho
      resid_ridge_log   <- y_tr_log - pred_tr_ridge_log
      pred_ridge        <- smear_backtransform(pred_ridge_log, resid_ridge_log)

      # 6.3) LASSO (sem cluster)
      message("ExE: LASSO (cv.glmnet, k=", kfold, ", single-thread) ...")
      self$fit_lasso <- glmnet::cv.glmnet(
        x = x_tr, y = y_tr_log, family = "gaussian",
        alpha = 1, nfolds = kfold, foldid = foldid, standardize = TRUE,
        type.measure = "mse", parallel = FALSE
      )
      pred_lasso_log    <- as.numeric(predict(self$fit_lasso, newx = x_te_ok, s = self$s_rule))
      pred_tr_lasso_log <- as.numeric(predict(self$fit_lasso, newx = x_tr,    s = self$s_rule)) #train tb precisa dos residuals em log1p! por isso o resultado estava estranho
      resid_lasso_log   <- y_tr_log - pred_tr_lasso_log
      pred_lasso        <- smear_backtransform(pred_lasso_log, resid_lasso_log)

      # 6.4) Árvore (1-SE)
      message("ExE: Árvore (rpart + poda 1-SE) ...")
      self$fit_tree0 <- rpart::rpart(
        formula = self$form_lm, data = self$train, method = "anova", xval = 10,
        control = rpart::rpart.control(cp = 0.001, minbucket = 10, maxdepth = 30)
      )
      ct <- self$fit_tree0$cptable
      ix_min   <- which.min(ct[, "xerror"])
      xerr_1se <- ct[ix_min, "xerror"] + ct[ix_min, "xstd"]
      ix_1se   <- min(which(ct[, "xerror"] <= xerr_1se))
      cp_1se   <- ct[ix_1se, "CP"]
      self$fit_tree <- rpart::prune(self$fit_tree0, cp = cp_1se)

      pred_tree_log    <- as.numeric(predict(self$fit_tree, newdata = self$test[keep_te, , drop = FALSE], type = "vector"))
      pred_tr_tree_log <- as.numeric(predict(self$fit_tree, newdata = self$train, type = "vector")) #train tb precisa dos residuals em log1p! por isso o resultado estava estranho
      resid_tree_log   <- log1p(self$train$Price) - pred_tr_tree_log
      pred_tree        <- smear_backtransform(pred_tree_log, resid_tree_log)

      # 6.3b) Elastic Net (grid de alpha c/ o msm foldid)
      message("ExE: Elastic Net (cv.glmnet em grade de alpha) ...")
      cv_list <- lapply(self$alpha_grid, function(a) {
        glmnet::cv.glmnet(
          x = x_tr, y = y_tr_log, family = "gaussian",
          alpha = a, nfolds = kfold, foldid = foldid, standardize = TRUE,
          type.measure = "mse", parallel = FALSE
        )
      })

      # escolhe pela menor CV no s escolhido (lambda.min ou lambda.1se)
      get_cvm_at <- function(cvfit, s_rule) {
        lam <- cvfit[[s_rule]]
        ix  <- which.min(abs(cvfit$lambda - lam))
        cvfit$cvm[ix]
      }
      cvm_vec <- vapply(cv_list, get_cvm_at, numeric(1), s_rule = self$s_rule)
      best_ix <- which.min(cvm_vec)

      self$fit_enet   <- cv_list[[best_ix]]
      self$alpha_enet <- self$alpha_grid[best_ix]

      if (abs(self$alpha_enet - 1) < 1e-3) {
        message("ExE (aviso): α*≈1 — o Elastic Net coincide com o LASSO para o critério selecionado.")
      }

      # previsões + Duan smearing. Estudar melhor como funciona internamente
      pred_enet_log    <- as.numeric(predict(self$fit_enet, newx = x_te_ok, s = self$s_rule))
      pred_tr_enet_log <- as.numeric(predict(self$fit_enet, newx = x_tr,    s = self$s_rule))
      resid_enet_log   <- y_tr_log - pred_tr_enet_log
      pred_enet        <- smear_backtransform(pred_enet_log, resid_enet_log)

      # 6.5) Floresta (ranger), chunks c/ barra de progresso 
      # Target em log1p
      self$train$Price_log1p <- log1p(self$train$Price)

      p_vars       <- length(self$pred_cols)
      mtry_default <- max(1L, floor(p_vars / 3))
      rf_cols      <- c("Price_log1p", self$pred_cols)

      trees_total  <- if (fast) 300L else 500L
      minbucket_rf <- if (fast) 20L else 10L
      maxdepth_rf  <- if (fast) 20L else 0L
      samp_frac    <- if (fast) 0.70 else 0.80
      n_threads    <- max(1L, parallel::detectCores(logical = TRUE) - 2L)

      # Fit único do modelo
      self$fit_rf <- ranger::ranger(
        dependent.variable.name   = "Price_log1p",
        data                      = self$train[, rf_cols, drop = FALSE],
        num.trees                 = trees_total,
        mtry                      = mtry_default,
        min.node.size             = minbucket_rf,
        max.depth                 = maxdepth_rf,
        sample.fraction           = samp_frac,
        replace                   = TRUE,
        splitrule                 = "variance",
        respect.unordered.factors = "order",     # manter simples/rápido; mude p/ "partition" se quiser evitar target-ordering
        importance                = "none",
        num.threads               = n_threads,
        seed                      = self$seed,
        verbose                   = FALSE
      )

      # Previsões no log (teste e treino) para smearing
      test_block     <- self$test[keep_te, self$pred_cols, drop = FALSE]
      pred_rf_log    <- predict(self$fit_rf, data = test_block)$predictions
      pred_rf_log_tr <- self$fit_rf$predictions  # OOB 

      # Smearing para voltar ao nível
      resid_rf_log <- log1p(self$train$Price) - pred_rf_log_tr
      pred_rf      <- smear_backtransform(pred_rf_log, resid_rf_log)

      # Bootstrap 95% CIs pro RMSE/RMSLE no test (resample)
      boot_ci_metrics <- function(y, yhat, times = 200, seed = self$seed) {
        df <- tibble(truth = y, estimate = pmax(yhat, 0))
        set.seed(seed)
        boots <- rsample::bootstraps(df, times = times)
        res <- boots %>%
          dplyr::mutate(metrics = purrr::map(splits, ~{
            dat <- rsample::analysis(.x)
            tibble(
              RMSE  = yardstick::rmse_vec(dat$truth, dat$estimate),
              RMSLE = sqrt(mean((log1p(dat$truth) - log1p(dat$estimate))^2))
            )
          })) %>%
          tidyr::unnest(metrics)
        tibble(
          RMSE_lo  = stats::quantile(res$RMSE,   0.025, na.rm = TRUE),
          RMSE_hi  = stats::quantile(res$RMSE,   0.975, na.rm = TRUE),
          RMSLE_lo = stats::quantile(res$RMSLE,  0.025, na.rm = TRUE),
          RMSLE_hi = stats::quantile(res$RMSLE,  0.975, na.rm = TRUE)
        )
      }
      times_boot <- if (fast) 200L else 1000L

      # 7) Métricas
      message("ExE: métricas ...")
      met <- function(y, yhat) {
        dfm <- tibble(truth = y, estimate = yhat)
        tibble(
          RMSE  = yardstick::rmse_vec(dfm$truth, dfm$estimate),
          MAE   = yardstick::mae_vec(dfm$truth, dfm$estimate),
          R2    = yardstick::rsq_trad_vec(dfm$truth, dfm$estimate),
          RMSLE = sqrt(mean((log1p(dfm$truth) - log1p(pmax(dfm$estimate, 0)))^2))
        )
      }
      
      m_lm    <- met(y_te_ok, pred_lm)    %>% mutate(Modelo = "Linear (OLS)")
      m_ridge <- met(y_te_ok, pred_ridge) %>% mutate(Modelo = sprintf("Ridge (%s)", self$s_rule))
      m_lasso <- met(y_te_ok, pred_lasso) %>% mutate(Modelo = sprintf("LASSO (%s)", self$s_rule))
      m_enet  <- met(y_te_ok, pred_enet)  %>% mutate(Modelo = sprintf("Elastic Net (α=%.2f, %s)", self$alpha_enet, self$s_rule))
      m_tree  <- met(y_te_ok, pred_tree)  %>% mutate(Modelo = "Árvore (1-SE)")
      m_rf    <- met(y_te_ok, pred_rf)    %>% mutate(Modelo = "Floresta (ranger)")

      ci_lm    <- boot_ci_metrics(y_te_ok, pred_lm,    times = times_boot)
      ci_ridge <- boot_ci_metrics(y_te_ok, pred_ridge, times = times_boot)
      ci_lasso <- boot_ci_metrics(y_te_ok, pred_lasso, times = times_boot)
      ci_enet  <- boot_ci_metrics(y_te_ok, pred_enet,  times = times_boot)
      ci_tree  <- boot_ci_metrics(y_te_ok, pred_tree,  times = times_boot)
      ci_rf    <- boot_ci_metrics(y_te_ok, pred_rf,    times = times_boot)

      m_lm    <- dplyr::bind_cols(m_lm,    ci_lm)
      m_ridge <- dplyr::bind_cols(m_ridge, ci_ridge)
      m_lasso <- dplyr::bind_cols(m_lasso, ci_lasso)
      m_enet  <- dplyr::bind_cols(m_enet,  ci_enet)
      m_tree  <- dplyr::bind_cols(m_tree,  ci_tree)
      m_rf    <- dplyr::bind_cols(m_rf,    ci_rf)

      nz <- function(cvfit, s) sum(as.numeric(coef(cvfit, s = s)) != 0) - 1
      m_lm$lambda <- NA_real_; m_lm$n_coef_neq0 <- sum(coef(self$fit_lm)[-1] != 0)
      m_ridge <- m_ridge %>%
        mutate(lambda = unname(if (self$s_rule == "lambda.1se") self$fit_ridge$lambda.1se else self$fit_ridge$lambda.min),
               n_coef_neq0 = nz(self$fit_ridge, self$s_rule))
     
      m_lasso <- m_lasso %>%
        mutate(
          lambda = unname(if (self$s_rule == "lambda.1se") self$fit_lasso$lambda.1se else self$fit_lasso$lambda.min),
          n_coef_neq0 = nz(self$fit_lasso, self$s_rule)
        )

      m_enet  <- m_enet  %>% mutate(
        lambda = unname(if (self$s_rule == "lambda.1se") self$fit_enet$lambda.1se else self$fit_enet$lambda.min),
        n_coef_neq0 = nz(self$fit_enet, self$s_rule)
      )
      m_tree$lambda <- NA_real_; m_tree$n_coef_neq0 <- NA_real_
      m_rf$lambda   <- NA_real_; m_rf$n_coef_neq0   <- NA_real_

      self$metrics <- bind_rows(m_lm, m_ridge, m_lasso, m_enet, m_tree, m_rf) %>%
        relocate(Modelo) %>% arrange(RMSE)


      # 8) Barras RMSE / RMSLE
      dfm_rmse <- self$metrics %>% mutate(Modelo = forcats::fct_reorder(Modelo, RMSE))
      
      self$plot_metrics <- ggplot2::ggplot(dfm_rmse, aes(x = Modelo, y = RMSE)) +
      ggplot2::geom_col(width = 0.70, fill = COL_BAR, color = COL_TEXT) +
      ggplot2::geom_errorbar(ggplot2::aes(ymin = RMSE_lo, ymax = RMSE_hi),
                            width = 0.25, linewidth = 0.5, color = COL_TEXT) +
      ggplot2::coord_flip() +
      ggplot2::scale_y_continuous(labels = fmt_real, expand = ggplot2::expansion(mult = c(0, 0.08))) +
      ggplot2::labs(title = "Comparação de modelos (teste — mesma amostra)",
                    subtitle = "Menor RMSE (em R$) é melhor — IC 95% via bootstrap (sem refit)",
                    x = NULL, y = "RMSE (R$)") +
      theme_andre(15)


      dfm_rmsle <- self$metrics %>% mutate(Modelo = forcats::fct_reorder(Modelo, RMSLE))
      self$plot_metrics_log <- ggplot2::ggplot(dfm_rmsle, aes(x = Modelo, y = RMSLE)) +
      ggplot2::geom_col(width = 0.70, fill = COL_BAR, color = COL_TEXT) +
      ggplot2::geom_errorbar(ggplot2::aes(ymin = RMSLE_lo, ymax = RMSLE_hi),
                            width = 0.25, linewidth = 0.5, color = COL_TEXT) +
      ggplot2::coord_flip() +
      ggplot2::scale_y_continuous(expand = ggplot2::expansion(mult = c(0, 0.08))) +
      ggplot2::labs(title = "Comparação de modelos (teste — mesma amostra)",
                    subtitle = "Menor RMSLE (log) é melhor — IC 95% via bootstrap (sem refit)",
                    x = NULL, y = "RMSLE") +
      theme_andre(15)

      # 9) ŷ vs y (log–log)
      pred_df <- bind_rows(
        tibble(Modelo = "Linear (OLS)", y = y_te_ok, yhat = pred_lm),
        tibble(Modelo = sprintf("Ridge (%s)", self$s_rule), y = y_te_ok, yhat = pred_ridge),
        tibble(Modelo = sprintf("LASSO (%s)", self$s_rule), y = y_te_ok, yhat = pred_lasso),
        tibble(Modelo = sprintf("Elastic Net (α=%.2f, %s)", self$alpha_enet, self$s_rule), y = y_te_ok, yhat = pred_enet),
        tibble(Modelo = "Árvore (1-SE)", y = y_te_ok, yhat = pred_tree),
        tibble(Modelo = "Floresta (ranger)", y = y_te_ok, yhat = pred_rf)
      ) %>% dplyr::filter(y > 0, yhat > 0)

      self$plot_pred <- ggplot2::ggplot(pred_df, aes(x = y, y = yhat)) +
        ggpointdensity::geom_pointdensity(size = 0.7) +
        ggplot2::scale_color_viridis_c(option = "viridis", name = "Densidade") +

        ggplot2::scale_x_log10(breaks = scales::log_breaks(n = 4), labels = fmt_real) +
        ggplot2::scale_y_log10(breaks = scales::log_breaks(n = 4), labels = fmt_real) +
        ggplot2::annotation_logticks(
          sides = "bl",
          short = grid::unit(1.2, "mm"), mid = grid::unit(1.8, "mm"), long = grid::unit(2.4, "mm"),
          linewidth = 0.25, color = COL_TICK
        ) +
        ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = COL_LINE) +
        ggplot2::facet_wrap(~ Modelo, ncol = 3) +
        ggplot2::labs(
          title = "ŷ vs y (teste, log–log; mesma amostra)",
          subtitle = "Escalas em log10; considera apenas y,ŷ > 0 (log1p c/ Duan smearing)",
          x = "Preço real (R$)", y = "Preço previsto (R$)"
        ) +
        theme_andre(15) +
        ggplot2::theme(legend.position = "right",
                       panel.spacing.x = grid::unit(10, "pt"),
                       axis.text.x = ggplot2::element_text(margin = ggplot2::margin(t = 2))) +
        ggplot2::coord_cartesian(clip = "off")

      # 10) Caminho lambda (Ridge/LASSO/Elastic)
      get_path <- function(cvfit, nome) {
        g <- cvfit$glmnet.fit
        tibble(
          lambda = g$lambda,
          nnz    = colSums(as.matrix(g$beta) != 0),  # sem intercepto
          Modelo = nome
        )
      }

      nome_enet <- sprintf("Elastic Net (α=%.2f)", self$alpha_enet)

      path_df <- bind_rows(
        get_path(self$fit_ridge, "Ridge"),
        get_path(self$fit_lasso, "LASSO"),
        get_path(self$fit_enet,  nome_enet)
      )

      vl_df <- tibble(
        Modelo = c("Ridge","Ridge","LASSO","LASSO", nome_enet, nome_enet),
        tipo   = c("lambda.min","lambda.1se","lambda.min","lambda.1se","lambda.min","lambda.1se"),
        x      = c(
          log10(self$fit_ridge$lambda.min), log10(self$fit_ridge$lambda.1se),
          log10(self$fit_lasso$lambda.min), log10(self$fit_lasso$lambda.1se),
          log10(self$fit_enet$lambda.min),  log10(self$fit_enet$lambda.1se)
        )
      )

      self$plot_path <- ggplot(path_df, aes(x = log10(lambda), y = nnz, color = Modelo)) +
        geom_line(linewidth = 0.9) +
        geom_vline(data = vl_df, aes(xintercept = x, linetype = tipo, color = Modelo),
                  alpha = 0.7, linewidth = 0.6, show.legend = TRUE) +
        scale_linetype_manual(values = c(lambda.min = "dotted", lambda.1se = "dashed"),
                              name = "λ de referência") +
        labs(
          title = "Complexidade do modelo vs λ (log10)",
          subtitle = "Número de coeficientes ≠ 0 (sem intercepto). Ridge (L2) quase não zera coeficientes",
          x = "log10(λ)", y = "# coeficientes ≠ 0"
        ) +
        theme_andre(15) +
        theme(legend.position = "right")
              
      # 10b) Heatmaps de CV — 3 gráficos separados e agr legíveis
      fmt_alpha <- function(a) sprintf("%.2f", a)

      # Tabelas base (cv mse em alvo log1p)
      df_ridge <- tibble(
        Modelo  = "Ridge",
        alpha   = 0,
        alpha_f = fmt_alpha(0),
        lambda  = self$fit_ridge$lambda,
        cvm     = self$fit_ridge$cvm
      )

      df_lasso <- tibble(
        Modelo  = "LASSO",
        alpha   = 1,
        alpha_f = fmt_alpha(1),
        lambda  = self$fit_lasso$lambda,
        cvm     = self$fit_lasso$cvm
      )

      df_enet <- bind_rows(lapply(seq_along(self$alpha_grid), function(i) {
        tibble(
          Modelo  = "Elastic Net",
          alpha   = self$alpha_grid[i],
          alpha_f = fmt_alpha(self$alpha_grid[i]),
          lambda  = cv_list[[i]]$lambda,
          cvm     = cv_list[[i]]$cvm
        )
      }))

      # Helper: calcula largura dos tiles por linha (evita "hairlines")
      add_width_per_row <- function(df) {
        df %>%
          mutate(log_lambda = log10(lambda)) %>%
          group_by(alpha_f) %>%
          arrange(log_lambda, .by_group = TRUE) %>%
          mutate(
            w = dplyr::coalesce(
              stats::median(diff(log_lambda)),
              diff(range(log_lambda)) / max(n() - 1, 1)
            ),
            w = ifelse(is.na(w) | !is.finite(w) | w <= 0, 0.03, w)
          ) %>% ungroup()
      }

      # Helper: plota um único heatmap (escala de cor independente por modelo)
      mk_heat_single <- function(df, titulo, s_lambda, s_alpha = NULL) {
        df2   <- add_width_per_row(df)
        lims  <- range(df2$cvm, finite = TRUE)
        y_lab <- if (length(unique(df2$alpha_f)) == 1L)
          paste0("α = ", unique(df2$alpha_f), " (L1/L2)")
        else "α (mistura L1/L2)"

        p <- ggplot(df2, aes(x = log_lambda, y = factor(alpha_f), fill = cvm)) +
          geom_tile(aes(width = w, height = 0.9)) +
          scale_x_continuous(breaks = scales::pretty_breaks(6), name = "log10(λ)") +
          scale_y_discrete(name = y_lab) +
          scale_fill_viridis_c(direction = -1, begin = 0.08, end = 0.98,
                               limits = lims, name = "CV MSE",
                               guide = guide_colorbar(title.position = "top")) +
          labs(title = titulo,
               subtitle = paste0("k=", kfold, "; s = ", self$s_rule, "; alvo: log1p(Price)")) +
          theme_andre(15) + theme(legend.position = "right")

        # Marca o lambda* (lambda.min ou lambda.1se)
        if (!is.null(s_lambda) && is.finite(s_lambda)) {
          p <- p + geom_vline(xintercept = log10(s_lambda),
                              linetype = "dashed", linewidth = 0.6, color = COL_TEXT)
        }
        # No ENet, marca (alpha*, lambda*)
        if (!is.null(s_alpha)) {
          p <- p + geom_point(
            data = data.frame(log_lambda = log10(s_lambda), alpha_f = fmt_alpha(s_alpha)),
            aes(x = log_lambda, y = alpha_f),
            inherit.aes = FALSE, shape = 21, size = 3.2,
            fill = "white", color = COL_TEXT, stroke = 0.6
          )
        }
        p
      }

      # níveis precisam ser únicos! o bug estava aqui
      df_ridge$alpha_f <- factor(df_ridge$alpha_f, levels = unique(df_ridge$alpha_f))
      df_lasso$alpha_f <- factor(df_lasso$alpha_f, levels = unique(df_lasso$alpha_f))
      df_enet$alpha_f  <- factor(df_enet$alpha_f,
                                 levels = rev(fmt_alpha(sort(unique(df_enet$alpha)))))

      # 3 gráficos separados (penalizados)
      self$plot_cv_ridge <- mk_heat_single(
        df_ridge, "CV MSE — Ridge", s_lambda = self$fit_ridge[[self$s_rule]]
      )
      self$plot_cv_lasso <- mk_heat_single(
        df_lasso, "CV MSE — LASSO", s_lambda = self$fit_lasso[[self$s_rule]]
      )
      self$plot_cv_enet <- mk_heat_single(
        df_enet,  "CV MSE — Elastic Net", s_lambda = self$fit_enet[[self$s_rule]], s_alpha = self$alpha_enet
      )

      # 10c) Heatmap Random Forest (OOB MSE por mtry × min.node.size)
      #     Nota: como RF não tem alpha/lambda, o heatmap usa erro OOB c/ log1p.
      p_vars <- length(self$pred_cols)
      mtry_vals <- unique(pmax(1L, round(c(sqrt(p_vars), p_vars/3, p_vars/2))))
      minnode_vals <- if (fast) c(5L, 10L, 20L, 40L) else c(3L, 5L, 10L, 15L, 20L, 30L, 50L)

      rf_grid <- tidyr::crossing(mtry = mtry_vals, min.node.size = minnode_vals) %>%
        mutate(oob_mse = purrr::pmap_dbl(
          list(mtry, min.node.size),
          ~{
            fit <- ranger::ranger(
              dependent.variable.name   = "Price_log1p",
              data                      = self$train[, c("Price_log1p", self$pred_cols), drop = FALSE],
              num.trees                 = trees_total,
              mtry                      = ..1,
              min.node.size             = ..2,
              max.depth                 = maxdepth_rf,
              sample.fraction           = samp_frac,
              replace                   = TRUE,
              splitrule                 = "variance",
              respect.unordered.factors = "order",
              importance                = "none",
              num.threads               = n_threads,
              seed                      = self$seed,
              verbose                   = FALSE
            )
            # ranger retorna MSE OOB em 'prediction.error' para regressão
            as.numeric(fit$prediction.error)
          }
        ))

      # Destaque do set "padrão" usado no fit principal
      rf_sel <- tibble(mtry = mtry_default, min.node.size = minbucket_rf)

      self$plot_cv_rf <- ggplot(rf_grid, aes(x = factor(mtry), y = factor(min.node.size), fill = oob_mse)) +
        geom_tile(width = 0.9, height = 0.9) +
        scale_fill_viridis_c(direction = -1, begin = 0.08, end = 0.98,
                             name = "OOB MSE (log1p)",
                             guide = guide_colorbar(title.position = "top")) +
        geom_point(data = rf_sel, shape = 21, size = 3.2,
                   color = COL_TEXT, fill = "white", stroke = 0.6) +
        labs(title = "OOB MSE — Random Forest",
             subtitle = paste0("Grade (mtry × min.node.size); ntree=", trees_total,
                               ", max.depth=", maxdepth_rf, ", sample.frac=", samp_frac),
             x = "mtry (nº preditores por split)", y = "min.node.size") +
        theme_andre(15) + theme(legend.position = "right")

      # 11) Metadados
      self$meta <- list(
        n_total = nrow(d), n_train = nrow(self$train), n_test = nrow(self$test),
        n_test_used = sum(keep_te),
        split_ratio = self$split_ratio, seed = self$seed,
        s_rule = self$s_rule, kfold = kfold,
        formula = deparse(self$form_lm), p_after_dummy = ncol(x_tr),
        tree_cp_1se = cp_1se,
        alpha_grid = self$alpha_grid, alpha_enet = self$alpha_enet,   # estava faltando
        mtry_rf = mtry_default, num_trees_rf = trees_total, 
        rf_importance = "none", rf_respect_unordered = "order",
        parallel_cores = n_threads,
        smearing = TRUE, boot_times = times_boot,
        district_lump_n = 50,
        coords_filter = list(lat_range = c(-24.5, -23.2), lon_range = c(-47.4, -46.0))
      )


      message("ExE: done.")
      invisible(self)
    },

    as_md = function(image_dir) {
      message("ExE: salvando figuras ...")
      dir.create(image_dir, showWarnings = FALSE, recursive = TRUE)
      p1  <- file.path(image_dir, "e_metrics_bar_rmse.png")
      p1b <- file.path(image_dir, "e_metrics_bar_rmsle.png")
      p2  <- file.path(image_dir, "e_pred_vs_real.png")
      p3    <- file.path(image_dir, "e_nnz_vs_lambda.png")
      p4_r  <- file.path(image_dir, "e_cv_heatmap_ridge.png")
      p4_l  <- file.path(image_dir, "e_cv_heatmap_lasso.png")
      p4_en <- file.path(image_dir, "e_cv_heatmap_elasticnet.png")
      p4_rf <- file.path(image_dir, "e_cv_heatmap_rf.png")
      
      ggsave(p1,   plot = self$plot_metrics,     width = 8.5, height = 5.5, dpi = 320, bg = "white")
      ggsave(p1b,  plot = self$plot_metrics_log, width = 8.5, height = 5.5, dpi = 320, bg = "white")
      ggsave(p2,   plot = self$plot_pred,        width = 9.5, height = 6.2, dpi = 320, bg = "white")
      ggsave(p3,    plot = self$plot_path,        width = 8.5, height = 5.5, dpi = 320, bg = "white")
      ggsave(p4_r,  plot = self$plot_cv_ridge,    width = 8.8, height = 3.8, dpi = 320, bg = "white")
      ggsave(p4_l,  plot = self$plot_cv_lasso,    width = 8.8, height = 3.8, dpi = 320, bg = "white")
      ggsave(p4_en, plot = self$plot_cv_enet,     width = 8.8, height = 6.0, dpi = 320, bg = "white")
      ggsave(p4_rf, plot = self$plot_cv_rf,       width = 8.8, height = 5.2, dpi = 320, bg = "white")

      tbl <- self$metrics %>%
        mutate(RMSE = fmt_real(RMSE), MAE = fmt_real(MAE), R2 = round(R2, 3),
               RMSLE = round(RMSLE, 4),
               lambda = ifelse(is.na(lambda), "—", format(lambda, digits = 4, scientific = TRUE)),
               `#coef≠0` = ifelse(is.na(n_coef_neq0), "—", as.character(n_coef_neq0))) %>%
        select(Modelo, RMSE, MAE, R2, RMSLE, lambda, `#coef≠0`)
      tbl_md <- knitr::kable(tbl, format = "pipe")

      c(
        "# Exercício 1e (i–vi): OLS, Ridge, LASSO, Elastic Net, Árvore e Floresta (rent)",

        "",

        "### Setup",
        paste0("- Amostra: **rent**; Price > 0; *hold-out* **", round(100*self$meta$split_ratio),
              "%/**", round(100*(1-self$meta$split_ratio)), "%** (seed=", self$meta$seed, ")."),
        paste0("- Teste efetivamente usado: **", self$meta$n_test_used, "/", self$meta$n_test,
              "** linhas (após alinhamento de dummies)."),
        paste0("- **Alvo**: `log1p(Price)`; Fórmula base: `", self$meta$formula, "`."),
        paste0("- Tunning: `glmnet` k=", self$meta$kfold,
               " (", self$s_rule, "); **Árvore** 1‑SE (cp=",
               format(self$meta$tree_cp_1se, digits = 3), "); **Floresta** `ranger` ",
               "(ntree=", self$meta$num_trees_rf, ", mtry=", self$meta$mtry_rf,
               ", importance=", self$meta$rf_importance, ", threads=", self$meta$parallel_cores, ")."),

        paste0("- **Elastic Net**: grade de α = {", paste(self$meta$alpha_grid, collapse = ", "),
              "}; α* selecionado = **", sprintf("%.2f", self$meta$alpha_enet), "**."),


        paste0("- `District` reduzido via `forcats::fct_lump_n(n=", self$meta$district_lump_n, ")`."),
        paste0("- Coordenadas filtradas p/ SP (lat ",
               paste(self$meta$coords_filter$lat_range, collapse = ".."),
               ", lon ", paste(self$meta$coords_filter$lon_range, collapse = ".."), ")."),
        "",
        "### Métricas (teste — mesma amostra)",
        paste(tbl_md, collapse = "\n"),
        "",

        "### Gráficos",
        paste0("![](", basename(p1), ")"), "",
        paste0("![](", basename(p1b), ")"), "",
        paste0("![](", basename(p2), ")"), "",
        paste0("![](", basename(p3), ")"), "",
        paste0("![](", basename(p4_r), ")"), "",     # Ridge
        paste0("![](", basename(p4_l), ")"), "",     # LASSO
        paste0("![](", basename(p4_en), ")"), "",    # Elastic Net
        paste0("![](", basename(p4_rf), ")"),        # RF (OOB)
        
        "",
        "### Notas",
        "- Sem cluster PSOCK: evita travamentos em macOS. `ranger` segue multi‑thread.",
        "- Para resultados finais, rode `run(fast = FALSE)` (10‑fold + importance = permutation)."
      )
    }
  )
)

# Montagem do MD 
Relatorio <- R6::R6Class(
  "Relatorio",
  public = list(
    blocos = NULL,
    initialize = function() { self$blocos <- list() },
    add = function(md_vec) { self$blocos <- c(self$blocos, list(md_vec)); invisible(self) },
    save = function(out_path) {
      dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
      writeLines(unlist(self$blocos), con = out_path)
      message("Relatório final: ", normalizePath(out_path))
      invisible(out_path)
    }
  )
)

# Add os exercicios e salva (agr c/ msgs p/ saber o que está rodando)
out_dir <- script_dir

message("Running ExA ..."); ex_a <- ExA$new(sp)$run(); message("ExA done.")
message("Running ExB ..."); ex_b <- ExB$new(sp)$run(); message("ExB done.")
message("Running ExC ..."); ex_c <- ExC$new(sp)$run(); message("ExC done.")
message("Running ExD ..."); ex_d <- ExD$new(sp)$run(); message("ExD done.")
message("Running ExE (fast=TRUE) ..."); ex_e <- ExE$new(sp)$run(fast = TRUE); message("ExE done.")


# Painel com patchwork 
panel_bc <- ex_b$plot +
  patchwork::plot_layout(ncol = 1, heights = 1, guides = "collect") +
  patchwork::plot_annotation(
    title = "Condomínio vs Preço — visão geral (B)",
    theme = theme_andre(16)
  ) &
  ggplot2::theme(legend.position = "right")

panel_path <- file.path(out_dir, "bc_patchwork_panel.png")
ggplot2::ggsave(filename = panel_path, plot = panel_bc,
                width = 10, height = 7, dpi = 320, bg = "white")

# Painel 2 (cividis) — também só o B
ex_b_alt <- ex_b$plot + ggplot2::scale_color_viridis_c(option = "cividis", name = "Densidade")

panel_bc_alt <- ex_b_alt +
  patchwork::plot_layout(ncol = 1, heights = 1, guides = "collect") +
  patchwork::plot_annotation(
    title = "Condomínio vs Preço — (B) em paleta cividis (alternativa)",
    theme = theme_andre(16)
  ) &
  ggplot2::theme(legend.position = "right")

panel_path_alt <- file.path(out_dir, "bc_patchwork_panel_cividis.png")
ggplot2::ggsave(filename = panel_path_alt, plot = panel_bc_alt,
                width = 10, height = 7, dpi = 320, bg = "white")

# Monta o relatório MD com tudo
rel <- Relatorio$new()
rel$add(ex_a$as_md())
rel$add(ex_b$as_md(out_dir))
rel$add(ex_c$as_md(out_dir))
rel$add(c("## Painel combinado — Patchwork (B)", "",
          paste0("![](", basename(panel_path), ")"), ""))
rel$add(c("## Painel alternativo — Paleta cividis (B)", "",
          paste0("![](", basename(panel_path_alt), ")"), ""))
rel$add(ex_d$as_md(out_dir))
rel$add(ex_e$as_md(out_dir))  

rel_path <- file.path(out_dir, "relatorio_exercicios_abcde.md")

rel$save(rel_path)

cat("\nOK\n",
    "Relatório: ", rel_path, "\n",
    "Fig B: ", file.path(out_dir, "b_scatter_condo_price_scatter_loglog.png"), "\n",
    "Fig C: ", file.path(out_dir, "c_scatter_condo_price_facets_scatter_loglog.png"), "\n",
    "Painel B (patchwork): ", panel_path, "\n",
    "Painel B (cividis): ", panel_path_alt, "\n",
    "Fig D: ", file.path(out_dir, "d_top10_distritos.png"), "\n",
    "Fig E (bar RMSE): ",   file.path(out_dir, "e_metrics_bar_rmse.png"), "\n",
    "Fig E (bar RMSLE): ",  file.path(out_dir, "e_metrics_bar_rmsle.png"), "\n",
    "Fig E (ŷ vs y): ",  file.path(out_dir, "e_pred_vs_real.png"), "\n",
    "Fig E (nnz vs lambda): ",   file.path(out_dir, "e_nnz_vs_lambda.png"), "\n",
    "Fig E (CV heatmap Ridge): ",       file.path(out_dir, "e_cv_heatmap_ridge.png"), "\n",
    "Fig E (CV heatmap LASSO): ",       file.path(out_dir, "e_cv_heatmap_lasso.png"), "\n",
    "Fig E (CV heatmap Elastic Net): ", file.path(out_dir, "e_cv_heatmap_elasticnet.png"), "\n",
    "Fig E (RF heatmap mtry×min.node.size): ", file.path(out_dir, "e_cv_heatmap_rf.png"), "\n",
    sep = "")
