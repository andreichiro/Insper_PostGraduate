#' @title Visualização
#' @description Charts com dados da coorte (script modular e reusável)
#' @date 12/Feb/2026
#' @course APRÁTICA AVANÇADA EM DATA SCIENCE E VISUALIZATION - PADSONL07PADSV

# Pacotes

required_pkgs <- c(
  "readr", "dplyr", "ggplot2", "stringr", "forcats", "scales", "tibble", "R6"
)

install_missing <- function(pkgs) {
  missing <- pkgs[!pkgs %in% rownames(installed.packages())]
  if (length(missing) > 0) {
    message("Instalando pacotes ausentes: ", paste(missing, collapse = ", "))
    install.packages(missing, repos = "https://cloud.r-project.org")
  }
}

install_missing(required_pkgs)
invisible(lapply(required_pkgs, library, character.only = TRUE))

# Configuração

CFG <- list(
  seed = 42,
  base_size = 12,
  plot_dpi = 320,
  wrap_x = 18,
  wrap_y = 26,
  wrap_bar = 28,
  pct_tol = 1e-8,
  pct_accuracy = 0.1,
  hist_min_bins = 1L,
  alpha_sig = 0.05,
  expected_digits = 1,
  stdres_digits = 2,
  d1_point_size = 6.2,
  d1_label_min = 2L,
  h_jitter_height = 0.12,
  h_jitter_width = 0.00
)

set.seed(CFG$seed)

# Estilo

PALETTE_5 <- c(
  blue   = "#4C78A8",
  teal   = "#72B7B2",
  green  = "#54A24B",
  amber  = "#F58518",
  purple = "#B279A2"
)

NEUTRAL <- list(
  gray_900 = "#111827",
  gray_700 = "#374151",
  gray_500 = "#6B7280",
  gray_200 = "#E5E7EB",
  gray_050 = "#F9FAFB",
  white    = "#FFFFFF",
  black    = "#000000"
)

COLOR_VIZ_EXPERIENCE <- c(
  "Iniciante"      = PALETTE_5[["teal"]],
  "Intermediário"  = PALETTE_5[["blue"]],
  "Avançado"       = PALETTE_5[["green"]],
  "Não informado"  = NEUTRAL$gray_500
)

blend_hex <- function(hex, with, alpha) {
  rgb1 <- grDevices::col2rgb(hex) / 255
  rgb2 <- grDevices::col2rgb(with) / 255
  out  <- rgb1 * (1 - alpha) + rgb2 * alpha
  grDevices::rgb(out[1], out[2], out[3])
}

COLOR_VIZ_EXPERIENCE_BLUE <- c(
  "Iniciante"      = blend_hex(PALETTE_5[["blue"]], NEUTRAL$white, alpha = 0.45),
  "Intermediário"  = PALETTE_5[["blue"]],
  "Avançado"       = blend_hex(PALETTE_5[["blue"]], NEUTRAL$black, alpha = 0.25),
  "Não informado"  = NEUTRAL$gray_500
)

COLOR_GRAD_COUNT_BLUE  <- list(low = NEUTRAL$gray_050, high = PALETTE_5[["blue"]])
COLOR_GRAD_COUNT_GREEN <- list(low = NEUTRAL$gray_050, high = PALETTE_5[["green"]])
COLOR_GRAD_STDRES      <- list(low = PALETTE_5[["purple"]], mid = NEUTRAL$gray_200, high = PALETTE_5[["amber"]])

theme_coorte <- function(base_size = CFG$base_size) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title.position = "plot",
      plot.title   = element_text(face = "bold", color = NEUTRAL$gray_900),
      plot.subtitle= element_text(color = NEUTRAL$gray_700),
      plot.caption = element_text(color = NEUTRAL$gray_700),
      axis.title   = element_text(color = NEUTRAL$gray_900),
      axis.text    = element_text(color = NEUTRAL$gray_700),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      legend.position = "bottom",
      legend.title = element_text(face = "bold", color = NEUTRAL$gray_900),
      legend.text  = element_text(color = NEUTRAL$gray_700),
      plot.margin  = margin(12, 14, 10, 12)
    )
}

theme_set(theme_coorte())

wrap_text <- function(x, width = CFG$wrap_y) stringr::str_wrap(x, width = width)

scale_fill_experience <- function(name = "Experiência", drop = FALSE) {
  scale_fill_manual(values = COLOR_VIZ_EXPERIENCE, name = name, drop = drop)
}

scale_color_experience <- function(name = "Experiência", drop = FALSE) {
  scale_color_manual(values = COLOR_VIZ_EXPERIENCE, name = name, drop = drop)
}

scale_fill_experience_blue <- function(name = "Experiência", drop = FALSE) {
  scale_fill_manual(values = COLOR_VIZ_EXPERIENCE_BLUE, name = name, drop = drop)
}

scale_color_experience_blue <- function(name = "Experiência", drop = FALSE) {
  scale_color_manual(values = COLOR_VIZ_EXPERIENCE_BLUE, name = name, drop = drop)
}

# Caminhos e export

`%||%` <- function(x, y) if (is.null(x)) y else x

root_dir <- getwd()
data_file <- file.path(root_dir, "data", "coorte.csv")
out_dir   <- file.path(root_dir, "outputs", "plots")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

save_plot <- function(p, filename, out_dir, width = 9, height = 5.5, dpi = CFG$plot_dpi) {
  ggsave(
    filename = file.path(out_dir, filename),
    plot = p,
    width = width,
    height = height,
    dpi = dpi,
    bg = "white"
  )
  message("Salvo: ", file.path(out_dir, filename))
}

# Leitura e validação de dados

required_cols <- c(
  "Carimbo de data/hora",
  "Nome",
  "Idade",
  "Formação Acadêmica (área/curso)",
  "Área de atuação (banco, varejo, educação, etc)",
  "Empresa",
  "Qual seu nível de experiência com visualização de dados?",
  "Qual ferramenta você mais usa para visualização de dados? (ggplot, matplotlib, powerBI, etc)",
  "Quantos anos de experiência você tem em Data Science?"
)

read_coorte_csv <- function(path, required_cols) {
  if (!file.exists(path)) {
    stop(
      paste0(
        "Arquivo de dados não encontrado: ", path, "\n",
        "Coloque o CSV em: <projeto>/data/coorte.csv (UTF-8) e rode novamente."
      ),
      call. = FALSE
    )
  }

  readers <- list(
    list(fn = readr::read_csv,  id = "read_csv (vírgula)"),
    list(fn = readr::read_csv2, id = "read_csv2 (ponto-e-vírgula)")
  )

  for (r in readers) {
    dat <- tryCatch(
      r$fn(
        path,
        show_col_types = FALSE,
        locale = readr::locale(encoding = "UTF-8"),
        col_types = readr::cols(.default = readr::col_character())
      ),
      error = function(e) NULL
    )
    if (!is.null(dat) && all(required_cols %in% names(dat))) {
      return(list(data = dat, reader = r$id))
    }
  }

  last_try <- tryCatch(
    readr::read_delim(
      path,
      delim = ";",
      show_col_types = FALSE,
      locale = readr::locale(encoding = "UTF-8"),
      col_types = readr::cols(.default = readr::col_character())
    ),
    error = function(e) NULL
  )

  cols_found <- if (is.null(last_try)) character(0) else names(last_try)
  missing_cols <- setdiff(required_cols, cols_found)

  stop(
    paste0(
      "Falha ao ler o CSV com delimitadores comuns.\n",
      "Colunas ausentes:\n- ", paste(missing_cols, collapse = "\n- "), "\n",
      "Dica: confira se o arquivo é o export correto do Forms e se está em UTF-8."
    ),
    call. = FALSE
  )
}

# Limpeza, parsing e regex

clean_text <- function(x) {
  x <- as.character(x)
  x <- stringr::str_replace_all(x, "\\s+", " ")
  x <- stringr::str_trim(x)
  dplyr::na_if(x, "")
}

parse_number_flexible <- function(x) {
  s <- clean_text(x)
  if (all(is.na(s))) return(rep(NA_real_, length(s)))

  a <- suppressWarnings(readr::parse_number(
    s, locale = readr::locale(decimal_mark = ",", grouping_mark = ".")
  ))
  b <- suppressWarnings(readr::parse_number(
    s, locale = readr::locale(decimal_mark = ".", grouping_mark = ",")
  ))

  dplyr::if_else(!is.na(a), a, b)
}

regex_map_first <- function(x, pattern_map, default = NA_character_) {
  s <- stringr::str_to_lower(clean_text(x))
  out <- rep(default, length(s))
  if (length(pattern_map) == 0) return(out)

  for (nm in names(pattern_map)) {
    hit <- !is.na(s) & is.na(out) & stringr::str_detect(s, pattern_map[[nm]])
    out[hit] <- nm
  }

  out
}

title_pt <- function(x) {
  s <- clean_text(x)
  ifelse(is.na(s), NA_character_, stringr::str_to_title(stringr::str_to_lower(s), locale = "pt"))
}

EXP_PATTERNS <- c(
  "Iniciante"     = "\\binic",
  "Intermediário" = "\\binter",
  "Avançado"      = "\\bavan"
)

standardize_experience <- function(x) {
  s <- stringr::str_to_lower(clean_text(x))
  mapped <- regex_map_first(s, EXP_PATTERNS, default = NA_character_)
  dplyr::case_when(
    is.na(s) ~ "Não informado",
    !is.na(mapped) ~ mapped,
    TRUE ~ "Não informado"
  )
}

parse_years_ds <- function(x) {
  s <- stringr::str_to_lower(clean_text(x))
  num <- parse_number_flexible(s)

  dplyr::case_when(
    is.na(s) ~ NA_real_,
    stringr::str_detect(s, "menos\\s+de\\s+um") ~ 0.5,
    stringr::str_detect(s, "\\bmes") & !is.na(num) ~ num / 12,
    !is.na(num) ~ num,
    TRUE ~ NA_real_
  )
}

EDU_PATTERNS <- c(
  "Computação/TI"           = "ci[eê]ncia\\s+da\\s+computa|comput[aá]ção|computacao|sistemas\\s+de\\s+informa|engenharia\\s+de\\s+software|software|dados|data\\b|ti\\b",
  "Engenharias (não TI)"    = "\\bengenharia\\b",
  "Administração/Negócios"  = "administra|neg[oó]ci|gest[aã]o|marketing|publicidade|propaganda",
  "Economia/Políticas"      = "economia|pol[ií]ticas\\s+p[úu]blicas|politicas\\s+publicas|rela[cç][oõ]es\\s+internacionais",
  "Estatística/Matemática"  = "estat[ií]stica|matem[aá]tica|probabilidade",
  "Direito"                 = "\\bdireito\\b|jur[ií]d",
  "Design/Artes"            = "design|artes|arquitetura",
  "Saúde/Biológicas"        = "biolog|biotec|sa[uú]de|medicin|farm[aá]c|enferm|biomed",
  "Comunicação/Sociais"     = "comunica|jornal|sociolog|antropolog|psicolog|hist[oó]ria|filosof"
)

rollup_education <- function(x) {
  s <- clean_text(x)
  mapped <- regex_map_first(s, EDU_PATTERNS, default = NA_character_)
  dplyr::case_when(
    is.na(s) ~ "Não informado",
    !is.na(mapped) ~ mapped,
    TRUE ~ title_pt(s)
  )
}

INDU_PATTERNS <- c(
  "Banco"                = "\\bbanco\\b|\\bbanc[aá]rio",
  "Fintech"              = "fintech",
  "Agronegócio"          = "\\bagro\\b|agroneg[oó]cio",
  "Varejo/E-commerce"    = "varejo|e-?commerce",
  "Indústria"            = "ind[uú]str|telecom",
  "Consultoria"          = "consult",
  "Educação/Governo/NGO" = "governo|\\bngo\\b|educa|educa[cç][aã]o|insper",
  "Tecnologia/Software"  = "software|desenvolvedor|engenheir[oa]\\s+de\\s+dados|\\bdata\\b|tecnolog"
)

rollup_industry <- function(x) {
  s <- clean_text(x)
  mapped <- regex_map_first(s, INDU_PATTERNS, default = NA_character_)
  dplyr::case_when(
    is.na(s) ~ "Não informado",
    !is.na(mapped) ~ mapped,
    TRUE ~ "Outros"
  )
}

TOOL_GROUP_PATTERNS <- c(
  "R/ggplot"    = "\\bggplot2?\\b|\\br\\b|pacotes\\s+de\\s+r",
  "Python"      = "\\bpython\\b|matplotlib|seaborn|plotly",
  "BI"          = "power\\s*bi|quicksight|looker",
  "Cloud/Stack" = "aws|athena|snowflake|databricks|sigma"
)

rollup_viz_tool_group <- function(x) {
  s <- stringr::str_to_lower(clean_text(x))
  out <- rep("Não informado", length(s))
  ok <- !is.na(s)
  out[ok] <- "Outra/Não classificada"

  hits <- vapply(
    names(TOOL_GROUP_PATTERNS),
    function(nm) ok & stringr::str_detect(s, TOOL_GROUP_PATTERNS[[nm]]),
    logical(length(s))
  )

  n_match <- rowSums(hits)
  out[ok & n_match > 1] <- "Mix de ferramentas"
  single_idx <- which(ok & n_match == 1)

  if (length(single_idx) > 0) {
    for (i in single_idx) out[i] <- names(TOOL_GROUP_PATTERNS)[which(hits[i, ])]
  }

  out
}

# Helpers

assert_int_equal <- function(x, target, msg) {
  if (as.integer(x) != as.integer(target)) stop(msg, call. = FALSE)
  invisible(TRUE)
}

assert_pct <- function(pct, tol = CFG$pct_tol) {
  s <- sum(pct, na.rm = TRUE)
  if (is.na(s) || abs(s - 1) > tol) stop(paste0("Percentuais não fecham em 100%. Soma=", s), call. = FALSE)
  invisible(TRUE)
}

complete_grid_counts <- function(data, x_col, y_col, extra_cols = NULL) {
  x_levels <- levels(data[[x_col]])
  y_levels <- levels(data[[y_col]])

  grid_list <- list()
  grid_list[[x_col]] <- x_levels
  grid_list[[y_col]] <- y_levels
  if (!is.null(extra_cols)) for (nm in names(extra_cols)) grid_list[[nm]] <- extra_cols[[nm]]

  base_grid <- expand.grid(grid_list, stringsAsFactors = FALSE) |> as_tibble()
  by_cols <- names(grid_list)

  counts <- data |>
    group_by(across(all_of(by_cols))) |>
    summarise(n = dplyr::n(), .groups = "drop")

  base_grid |>
    left_join(counts, by = by_cols) |>
    mutate(n = ifelse(is.na(n), 0L, as.integer(n)))
}

plot_barh_counts <- function(count_df, y_col, title, subtitle = NULL, fill_color = PALETTE_5[["blue"]], wrap_w = CFG$wrap_bar) {
  ggplot(count_df, aes(x = n, y = forcats::fct_rev(.data[[y_col]]))) +
    geom_col(fill = fill_color, alpha = 0.92, width = 0.72) +
    geom_text(aes(label = n), hjust = -0.12, color = NEUTRAL$gray_900, size = 3.8) +
    scale_x_continuous(breaks = scales::pretty_breaks(), expand = expansion(mult = c(0.02, 0.18))) +
    scale_y_discrete(labels = ~ wrap_text(., wrap_w)) +
    labs(title = title, subtitle = subtitle, x = "Pessoas (n)", y = NULL) +
    coord_cartesian(clip = "off")
}

make_axis_totals_label_map <- function(count_df, key_col, n_col = "n") {
  key <- as.character(count_df[[key_col]])
  n   <- count_df[[n_col]]
  setNames(paste0(key, "\n(n=", n, ")"), key)
}

compute_assoc_cells_stdres <- function(data_full, row_col = "education", col_col = "industry") {
  f <- stats::as.formula(paste0("n ~ ", row_col, " + ", col_col))
  ct <- xtabs(f, data = data_full)
  chi <- suppressWarnings(chisq.test(ct, correct = FALSE))

  stdres_df <- as.data.frame(as.table(chi$stdres), responseName = "stdres", stringsAsFactors = FALSE)
  exp_df    <- as.data.frame(as.table(chi$expected), responseName = "expected", stringsAsFactors = FALSE)
  n_df      <- as.data.frame(as.table(ct), responseName = "n", stringsAsFactors = FALSE)

  names(stdres_df)[1:2] <- c(row_col, col_col)
  names(exp_df)[1:2]    <- c(row_col, col_col)
  names(n_df)[1:2]      <- c(row_col, col_col)

  stdres_df |>
    left_join(exp_df, by = c(row_col, col_col)) |>
    left_join(n_df,   by = c(row_col, col_col))
}

compute_area_exp_tables <- function(df) {
  area_exp <- df |> count(industry, viz_experience, name = "n", .drop = FALSE)

  area_exp_pct <- area_exp |>
    group_by(industry) |>
    mutate(total = sum(n), pct = ifelse(total > 0, n / total, 0)) |>
    ungroup()

  adv_order <- area_exp_pct |>
    filter(viz_experience == "Avançado") |>
    transmute(industry, adv_pct = pct)

  adv_order <- tibble::tibble(industry = levels(df$industry)) |>
    left_join(adv_order, by = "industry") |>
    mutate(adv_pct = ifelse(is.na(adv_pct), 0, adv_pct)) |>
    arrange(desc(adv_pct), industry)

  list(
    area_exp = area_exp,
    area_exp_pct = area_exp_pct,
    industry_order = adv_order$industry
  )
}

# Classes

ChartRegistry <- R6::R6Class(
  "ChartRegistry",
  public = list(
    out_dir = NULL,
    specs = NULL,
    defaults = NULL,

    initialize = function(out_dir, width = 9, height = 5.5, dpi = CFG$plot_dpi) {
      self$out_dir <- out_dir
      self$specs <- list()
      self$defaults <- list(width = width, height = height, dpi = dpi)
      dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    },

    add = function(id, plot_fn, file, width = NULL, height = NULL, dpi = NULL) {
      self$specs[[id]] <- list(
        id = id, plot_fn = plot_fn, file = file,
        width = width, height = height, dpi = dpi
      )
      invisible(self)
    },

    render_all = function(ctx) {
      ids <- names(self$specs)
      if (length(ids) == 0) {
        message("Nenhum gráfico registrado.")
        return(invisible(NULL))
      }

      for (id in ids) {
        s <- self$specs[[id]]
        p <- s$plot_fn(ctx)

        save_plot(
          p = p,
          filename = s$file,
          out_dir = self$out_dir,
          width = s$width %||% self$defaults$width,
          height = s$height %||% self$defaults$height,
          dpi = s$dpi %||% self$defaults$dpi
        )
      }

      invisible(NULL)
    }
  )
)

CohortVizApp <- R6::R6Class(
  "CohortVizApp",
  public = list(
    data_file = NULL,
    out_dir = NULL,
    raw = NULL,
    df = NULL,
    df_age = NULL,
    df_ds = NULL,
    reader_used = NULL,
    registry = NULL,

    initialize = function(data_file = data_file, out_dir = out_dir) {
      self$data_file <- data_file
      self$out_dir <- out_dir
      self$registry <- ChartRegistry$new(out_dir = out_dir)
    },

    load = function() {
      res <- read_coorte_csv(self$data_file, required_cols)
      self$raw <- res$data
      self$reader_used <- res$reader
      message("CSV carregado via: ", self$reader_used)
      message("Linhas: ", nrow(self$raw), " | Colunas: ", ncol(self$raw))
      invisible(self)
    },

    prepare = function() {
      df <- self$raw |>
        transmute(
          timestamp = clean_text(`Carimbo de data/hora`),
          name      = clean_text(Nome),
          age       = parse_number_flexible(Idade),
          education_raw = clean_text(`Formação Acadêmica (área/curso)`),
          industry_raw  = clean_text(`Área de atuação (banco, varejo, educação, etc)`),
          company   = clean_text(Empresa),
          viz_experience = standardize_experience(`Qual seu nível de experiência com visualização de dados?`),
          viz_tool  = clean_text(`Qual ferramenta você mais usa para visualização de dados? (ggplot, matplotlib, powerBI, etc)`),
          ds_years  = parse_years_ds(`Quantos anos de experiência você tem em Data Science?`)
        ) |>
        mutate(
          education_raw = dplyr::coalesce(education_raw, "Não informado"),
          industry_raw  = dplyr::coalesce(industry_raw,  "Não informado"),
          company       = dplyr::coalesce(company,       "Não informado"),
          viz_tool      = dplyr::coalesce(viz_tool,      "Não informado"),
          viz_experience = factor(
            viz_experience,
            levels = names(COLOR_VIZ_EXPERIENCE),
            ordered = TRUE
          ),
          education = rollup_education(education_raw),
          industry  = rollup_industry(industry_raw),
          viz_tool_group = rollup_viz_tool_group(viz_tool)
        )

      education_levels <- df |> count(education, sort = TRUE) |> arrange(desc(n), education) |> pull(education)
      industry_levels  <- df |> count(industry,  sort = TRUE) |> arrange(desc(n), industry)  |> pull(industry)

      df <- df |>
        mutate(
          education = factor(education, levels = education_levels),
          industry  = factor(industry,  levels = industry_levels)
        )

      tool_levels <- df |>
        count(viz_tool_group, sort = TRUE) |>
        pull(viz_tool_group)

      df <- df |>
        mutate(viz_tool_group = factor(viz_tool_group, levels = tool_levels))

      self$df <- df
      self$df_age <- df |> filter(!is.na(age))
      self$df_ds  <- df |> filter(!is.na(ds_years))

      message("Linhas no dataset final: ", nrow(self$df))
      invisible(self)
    },

    context = function() {
      list(
        df = self$df,
        df_age = self$df_age,
        df_ds = self$df_ds,
        N_total = nrow(self$df)
      )
    },

    register_charts = function() {
      self$registry$specs <- list()

      self$registry$add(
        id = "A",
        file = "A_idade_histograma_rug.png",
        width = 9.2, height = 5.6,
        plot_fn = function(ctx) {
          # A: hist + rug pra ver idade sem “chute” de bin fixo
          bins_fd <- grDevices::nclass.FD(ctx$df_age$age)
          bins <- max(as.integer(CFG$hist_min_bins), as.integer(ifelse(is.finite(bins_fd), bins_fd, CFG$hist_min_bins)))

          med_age <- median(ctx$df_age$age, na.rm = TRUE)

          ggplot(ctx$df_age, aes(x = age)) +
            geom_histogram(
              bins = bins,
              fill = PALETTE_5[["blue"]],
              color = NEUTRAL$gray_050,
              alpha = 0.86
            ) +
            geom_rug(sides = "b", alpha = 0.65, color = NEUTRAL$gray_900) +
            geom_vline(xintercept = med_age, linetype = "dashed", color = NEUTRAL$gray_900, linewidth = 0.6) +
            annotate(
              "text",
              x = med_age, y = Inf,
              label = paste0("mediana: ", round(med_age, 1)),
              vjust = 1.4, hjust = -0.05,
              color = NEUTRAL$gray_900, size = 3.6
            ) +
            scale_y_continuous(breaks = scales::pretty_breaks(), expand = expansion(mult = c(0.02, 0.18))) +
            labs(
              title = "Distribuição de idade",
              subtitle = "Histograma (bins automáticos) + rug",
              x = "Idade", y = "Pessoas (n)"
            ) +
            coord_cartesian(clip = "off")
        }
      )

      self$registry$add(
        id = "B",
        file = "B_experiencia_visualizacao_barra.png",
        width = 9.2, height = 4.9,
        plot_fn = function(ctx) {
          # B: barras com n e % (percentual sempre fecha em 100% porque usa total calculado)
          exp_counts <- ctx$df |>
            count(viz_experience, name = "n", .drop = FALSE) |>
            mutate(
              total = sum(n),
              pct = ifelse(total > 0, n / total, 0),
              label = paste0(n, " (", scales::percent(pct, accuracy = CFG$pct_accuracy), ")")
            )

          assert_int_equal(sum(exp_counts$n), ctx$N_total, "B: soma das contagens != N_total (tem NA/valor fora do padrão).")
          if (ctx$N_total > 0) assert_pct(exp_counts$pct)

          exp_counts_plot <- exp_counts |> filter(n > 0)

          ggplot(exp_counts_plot, aes(x = n, y = forcats::fct_rev(viz_experience), fill = viz_experience)) +
            geom_col(alpha = 0.92, width = 0.72) +
            geom_text(aes(label = label), hjust = -0.12, color = NEUTRAL$gray_900, size = 3.8) +
            scale_fill_experience(name = NULL, drop = TRUE) +
            scale_x_continuous(breaks = scales::pretty_breaks(), expand = expansion(mult = c(0.02, 0.18))) +
            labs(
              title = "Experiência em visualização",
              subtitle = "Contagem e % do total",
              x = "Pessoas (n)", y = NULL
            ) +
            theme(legend.position = "none") +
            coord_cartesian(clip = "off")
        }
      )

      self$registry$add(
        id = "C1",
        file = "C1_area_atuacao_barra.png",
        width = 9.6, height = 6.0,
        plot_fn = function(ctx) {
          # C1: ranking das áreas (agrupado por regex) pra bater o olho
          industry_counts <- ctx$df |> count(industry, name = "n", .drop = FALSE)
          plot_barh_counts(
            industry_counts, "industry",
            title = "Área de atuação",
            subtitle = NULL,
            fill_color = PALETTE_5[["blue"]],
            wrap_w = 24
          )
        }
      )

      self$registry$add(
        id = "C2",
        file = "C2_formacao_barra.png",
        width = 9.6, height = 5.8,
        plot_fn = function(ctx) {
          # C2: ranking da formação (sem “Outros”: se não casar, vira título do texto)
          education_counts <- ctx$df |> count(education, name = "n", .drop = FALSE)
          plot_barh_counts(
            education_counts, "education",
            title = "Formação acadêmica",
            subtitle = NULL,
            fill_color = PALETTE_5[["teal"]],
            wrap_w = 28
          )
        }
      )

      self$registry$add(
        id = "D1",
        file = "D1_formacao_area_dotmatrix_counts.png",
        width = 12.0, height = 7.0,
        plot_fn = function(ctx) {
          # D1: bolinhas do mesmo tamanho; cor = pessoas na célula (n), simples e direto
          D_full <- complete_grid_counts(ctx$df, x_col = "industry", y_col = "education")
          assert_int_equal(sum(D_full$n), ctx$N_total, "D1: soma das células != N_total (tem algo fora da matriz).")

          x_tot <- ctx$df |> count(industry, name = "n")
          y_tot <- ctx$df |> count(education, name = "n")
          x_label_map <- make_axis_totals_label_map(x_tot, "industry")
          y_label_map <- make_axis_totals_label_map(y_tot, "education")

          max_n <- max(D_full$n, na.rm = TRUE)

          ggplot() +
            geom_tile(
              data = D_full,
              aes(x = industry, y = forcats::fct_rev(education)),
              fill = NEUTRAL$gray_050,
              color = NEUTRAL$gray_200,
              linewidth = 0.4
            ) +
            geom_point(
              data = D_full |> filter(n > 0),
              aes(x = industry, y = forcats::fct_rev(education), fill = n),
              shape = 21,
              color = NEUTRAL$gray_200,
              stroke = 0.6,
              size = CFG$d1_point_size,
              alpha = 0.95
            ) +
            geom_text(
              data = D_full |> filter(n >= CFG$d1_label_min),
              aes(x = industry, y = forcats::fct_rev(education), label = n),
              color = NEUTRAL$gray_900,
              size = 3.0
            ) +
            scale_fill_gradient(
              low = COLOR_GRAD_COUNT_BLUE$low,
              high = COLOR_GRAD_COUNT_BLUE$high,
              limits = c(0, max_n),
              breaks = sort(unique(D_full$n)),
              name = "Pessoas na célula (n)"
            ) +
            scale_x_discrete(labels = function(x) wrap_text(x_label_map[x] %||% x, CFG$wrap_x)) +
            scale_y_discrete(labels = function(y) wrap_text(y_label_map[rev(y)] %||% rev(y), CFG$wrap_y)) +
            labs(
              title = "Formação × Área (contagens)",
              subtitle = "Cor mais forte = mais gente na célula; célula sem bolinha = zero",
              x = NULL, y = NULL
            ) +
            theme(panel.grid = element_blank(), axis.text.x = element_text(angle = 25, hjust = 1))
        }
      )

      self$registry$add(
        id = "D2",
        file = "D2_formacao_area_associacoes_stdres_heatmap.png",
        width = 12.0, height = 7.0,
        plot_fn = function(ctx) {
          # D2: heatmap de resíduos (stdres) pra ver acima/abaixo do esperado com cor divergente
          D_full <- complete_grid_counts(ctx$df, x_col = "industry", y_col = "education")
          assert_int_equal(sum(D_full$n), ctx$N_total, "D2: soma das células != N_total (tem algo fora da matriz).")

          assoc <- compute_assoc_cells_stdres(D_full, row_col = "education", col_col = "industry") |>
            mutate(
              diff = n - expected
            )

          exp_sum <- sum(assoc$expected, na.rm = TRUE)
          assert_int_equal(round(exp_sum), ctx$N_total, "D2: soma do esperado != N_total (checagem do qui-quadrado falhou).")

          z_thr <- stats::qnorm(1 - CFG$alpha_sig / 2)
          assoc <- assoc |>
            mutate(
              sig = abs(stdres) >= z_thr,
              label = ifelse(
                sig,
                paste0("n=", n, "\nexp=", round(expected, CFG$expected_digits)),
                ""
              )
            )

          lim <- max(abs(assoc$stdres), na.rm = TRUE)
          lim <- ifelse(is.finite(lim) && lim > 0, lim, 1)

          ggplot(assoc, aes(x = industry, y = forcats::fct_rev(education))) +
            geom_tile(aes(fill = stdres), color = NEUTRAL$gray_200, linewidth = 0.4) +
            geom_text(aes(label = label), color = NEUTRAL$gray_900, size = 3.0, lineheight = 0.95) +
            scale_fill_gradient2(
              low = COLOR_GRAD_STDRES$low,
              mid = COLOR_GRAD_STDRES$mid,
              high = COLOR_GRAD_STDRES$high,
              midpoint = 0,
              limits = c(-lim, lim),
              name = "Resíduo (stdres)"
            ) +
            scale_x_discrete(labels = ~ wrap_text(., CFG$wrap_x)) +
            scale_y_discrete(labels = ~ wrap_text(., CFG$wrap_y)) +
            labs(
              title = "Formação × Área (associação vs. esperado)",
              subtitle = "Laranja = acima do esperado; roxo = abaixo; cinza = perto do esperado",
              caption = paste0("Rótulo aparece quando |stdres| ≥ ", round(z_thr, CFG$stdres_digits), " (via qnorm)."),
              x = NULL, y = NULL
            ) +
            theme(panel.grid = element_blank(), axis.text.x = element_text(angle = 25, hjust = 1))
        }
      )

      self$registry$add(
        id = "E1",
        file = "E1_area_para_experiencia_contagem.png",
        width = 11.2, height = 6.2,
        plot_fn = function(ctx) {
          # E1: contagem absoluta (tamanho da barra importa), com tons de azul pra experiência
          tabs <- compute_area_exp_tables(ctx$df)
          area_exp <- tabs$area_exp |>
            mutate(industry = factor(industry, levels = tabs$industry_order))

          assert_int_equal(sum(area_exp$n), ctx$N_total, "E1: soma das contagens != N_total.")

          area_totals <- area_exp |> group_by(industry) |> summarise(total = sum(n), .groups = "drop")

          ggplot(area_exp, aes(x = industry, y = n, fill = viz_experience)) +
            geom_col(alpha = 0.94, width = 0.82) +
            geom_text(
              data = area_totals,
              aes(x = industry, y = total, label = total),
              inherit.aes = FALSE,
              vjust = -0.35,
              color = NEUTRAL$gray_900,
              size = 3.3
            ) +
            scale_fill_experience_blue(name = "Experiência", drop = FALSE) +
            scale_y_continuous(breaks = scales::pretty_breaks(), expand = expansion(mult = c(0.02, 0.18))) +
            scale_x_discrete(labels = ~ wrap_text(., 16)) +
            labs(
              title = "Área → Experiência (contagens absolutas)",
              subtitle = "Altura total = tamanho da área (n); cor = nível (tons de azul)",
              x = NULL, y = "Pessoas (n)"
            ) +
            theme(axis.text.x = element_text(angle = 25, hjust = 1)) +
            coord_cartesian(clip = "off")
        }
      )

      self$registry$add(
        id = "E2",
        file = "E2_area_para_experiencia_100pct_ordenado.png",
        width = 11.2, height = 6.2,
        plot_fn = function(ctx) {
          # E2: composição percentual (sempre 100%), ideal pra comparar “mix” sem confundir com tamanho
          tabs <- compute_area_exp_tables(ctx$df)
          area_exp_pct <- tabs$area_exp_pct |>
            mutate(industry = factor(industry, levels = tabs$industry_order))

          check <- area_exp_pct |> group_by(industry) |> summarise(s = sum(pct), .groups = "drop")
          if (any(abs(check$s - 1) > CFG$pct_tol)) stop("E2: tem indústria com soma de pct != 1 (erro de normalização).", call. = FALSE)

          area_totals2 <- area_exp_pct |> distinct(industry, total)

          ggplot(area_exp_pct, aes(x = industry, y = pct, fill = viz_experience)) +
            geom_col(alpha = 0.94, width = 0.82) +
            geom_text(
              data = area_totals2,
              aes(x = industry, y = 1.02, label = paste0("n=", total)),
              inherit.aes = FALSE,
              color = NEUTRAL$gray_900,
              size = 3.2
            ) +
            scale_fill_experience_blue(name = "Experiência", drop = FALSE) +
            scale_y_continuous(labels = scales::percent_format(accuracy = CFG$pct_accuracy), expand = expansion(mult = c(0.02, 0.10))) +
            scale_x_discrete(labels = ~ wrap_text(., 16)) +
            labs(
              title = "Área → Experiência (composição 100%)",
              subtitle = "Cada barra soma 100%: compara composição (não tamanho) e ordena por % de Avançado",
              x = NULL, y = "Proporção na área"
            ) +
            theme(axis.text.x = element_text(angle = 25, hjust = 1)) +
            coord_cartesian(clip = "off")
        }
      )

      self$registry$add(
        id = "F",
        file = "F_idade_por_experiencia_box_pontos.png",
        width = 9.2, height = 5.8,
        plot_fn = function(ctx) {
          # F: box + pontos pra comparar idade por nível de viz sem esconder outliers no “barulho”
          ggplot(ctx$df_age, aes(x = viz_experience, y = age)) +
            geom_boxplot(
              fill = NEUTRAL$gray_200,
              color = NEUTRAL$gray_500,
              alpha = 0.35,
              width = 0.5,
              outlier.shape = NA
            ) +
            geom_jitter(
              aes(color = viz_experience),
              width = 0.12,
              height = 0,
              size = 2.6,
              alpha = 0.88
            ) +
            scale_color_experience(name = NULL, drop = TRUE) +
            labs(
              title = "Idade por experiência em visualização",
              subtitle = NULL,
              x = NULL, y = "Idade"
            ) +
            theme(legend.position = "none")
        }
      )

      self$registry$add(
        id = "G",
        file = "G_formacao_area_experiencia_heatmap_facets.png",
        width = 13.5, height = 7.2,
        plot_fn = function(ctx) {
          # G: heatmap facetado por experiência (verde mais escuro = mais gente na célula)
          all_exp <- levels(ctx$df$viz_experience)

          G_full <- complete_grid_counts(
            ctx$df,
            x_col = "industry",
            y_col = "education",
            extra_cols = list(viz_experience = all_exp)
          ) |>
            mutate(
              viz_experience = factor(viz_experience, levels = all_exp, ordered = TRUE),
              label = ifelse(n > 0, as.character(n), "")
            )

          assert_int_equal(sum(G_full$n), ctx$N_total, "G: soma das células != N_total (checagem do 3-way falhou).")

          facet_keep <- G_full |>
            group_by(viz_experience) |>
            summarise(total = sum(n), .groups = "drop") |>
            filter(total > 0) |>
            pull(viz_experience)

          G_plot <- G_full |> filter(viz_experience %in% facet_keep)
          max_n <- max(G_plot$n, na.rm = TRUE)

          ggplot(G_plot, aes(x = industry, y = forcats::fct_rev(education))) +
            geom_tile(aes(fill = n), color = NEUTRAL$gray_200, linewidth = 0.4) +
            geom_text(
              data = G_plot |> filter(n > 0),
              aes(label = label),
              color = NEUTRAL$gray_900,
              size = 3.0
            ) +
            scale_fill_gradient(
              low = COLOR_GRAD_COUNT_GREEN$low,
              high = COLOR_GRAD_COUNT_GREEN$high,
              limits = c(0, max_n),
              breaks = sort(unique(G_plot$n)),
              name = "Pessoas na célula (n)"
            ) +
            scale_x_discrete(labels = ~ wrap_text(., CFG$wrap_x)) +
            scale_y_discrete(labels = ~ wrap_text(., CFG$wrap_y)) +
            facet_wrap(~ viz_experience, nrow = 1) +
            labs(
              title = "Formação × Área por nível de experiência",
              subtitle = "Cada painel é um nível; a cor (verde) mostra a intensidade de contagem",
              x = NULL, y = NULL
            ) +
            theme(panel.grid = element_blank(), axis.text.x = element_text(angle = 25, hjust = 1))
        }
      )

      self$registry$add(
        id = "H",
        file = "H_ds_vs_viz_strip_summary.png",
        width = 10.8, height = 5.8,
        plot_fn = function(ctx) {
          # H: DS × viz sem “bolha”: pontos + mediana/IQR pra ficar bem legível
          dfp <- ctx$df_ds |> filter(viz_experience != "Não informado")
          if (nrow(dfp) == 0) stop("H: não tem ds_years válido pra plotar.", call. = FALSE)

          overall_med <- median(dfp$ds_years, na.rm = TRUE)

          sum_df <- dfp |>
            group_by(viz_experience) |>
            summarise(
              n = dplyr::n(),
              med = median(ds_years, na.rm = TRUE),
              q25 = stats::quantile(ds_years, probs = 0.25, na.rm = TRUE),
              q75 = stats::quantile(ds_years, probs = 0.75, na.rm = TRUE),
              .groups = "drop"
            )

          ggplot(dfp, aes(x = ds_years, y = viz_experience)) +
            geom_vline(xintercept = overall_med, linetype = "dashed", color = NEUTRAL$gray_700, linewidth = 0.6) +
            geom_linerange(
              data = sum_df,
              aes(y = viz_experience, xmin = q25, xmax = q75),
              inherit.aes = FALSE,
              color = NEUTRAL$gray_700,
              linewidth = 2.1,
              alpha = 0.75
            ) +
            geom_point(
              data = sum_df,
              aes(x = med, y = viz_experience),
              inherit.aes = FALSE,
              shape = 21,
              fill = NEUTRAL$gray_900,
              color = NEUTRAL$gray_900,
              size = 3.4
            ) +
            geom_jitter(
              aes(color = viz_experience),
              height = CFG$h_jitter_height,
              width = CFG$h_jitter_width,
              size = 2.8,
              alpha = 0.90
            ) +
            scale_color_experience_blue(name = NULL, drop = TRUE) +
            scale_x_continuous(breaks = scales::pretty_breaks()) +
            labs(
              title = "Experiência em Data Science × experiência em visualização",
              subtitle = "Pontos = pessoas; faixa = 25–75%; bolinha preta = mediana (linha tracejada = mediana geral de DS)",
              x = "Anos de experiência em Data Science",
              y = NULL
            ) +
            theme(legend.position = "none")
        }
      )

      self$registry$add(
        id = "I",
        file = "I_ds_vs_viz_tool_violin_facets.png",
        width = 13.0, height = 6.8,
        plot_fn = function(ctx) {
          # I: violino + pontos (facet por grupo de ferramenta detectado por regex, sem lista “fixa” de níveis)
          dfp <- ctx$df_ds |> filter(viz_experience != "Não informado")
          if (nrow(dfp) == 0) stop("I: não tem ds_years válido pra plotar.", call. = FALSE)

          # FIX: evita stat_ydensity quebrar em grupos com pouca variação (ex.: tudo 0) ou com n<2
          df_violin <- dfp |>
            group_by(viz_tool_group, viz_experience) |>
            filter(dplyr::n() >= 2, dplyr::n_distinct(ds_years) >= 2) |>
            ungroup()

          p <- ggplot(dfp, aes(x = viz_experience, y = ds_years, fill = viz_experience))

          if (nrow(df_violin) > 0) {
            p <- p + geom_violin(
              data = df_violin,
              alpha = 0.22,
              color = NEUTRAL$gray_500,
              linewidth = 0.5,
              trim = TRUE
            )
          }

          p +
            geom_jitter(
              aes(color = viz_experience),
              width = 0.10,
              height = 0,
              size = 2.2,
              alpha = 0.90
            ) +
            scale_fill_experience(name = NULL, drop = TRUE) +
            scale_color_experience(name = NULL, drop = TRUE) +
            scale_y_continuous(breaks = scales::pretty_breaks()) +
            facet_wrap(~ viz_tool_group, nrow = 1, scales = "fixed", drop = TRUE) +
            labs(
              title = "Data Science × Visualização × Ferramenta (por grupo)",
              subtitle = "Cada painel é um grupo detectado por regex; violino = distribuição e pontos = pessoas",
              x = NULL,
              y = "Anos de experiência em Data Science"
            ) +
            theme(
              legend.position = "none",
              panel.grid.major.x = element_blank()
            )
        }
      )

      invisible(self)
    },

    run = function() {
      self$load()
      self$prepare()

      ctx <- self$context()
      self$register_charts()
      self$registry$render_all(ctx)

      message("Concluído. Plots salvos em: ", self$out_dir)
      invisible(self)
    }
  )
)

# Execução

app <- CohortVizApp$new(data_file = data_file, out_dir = out_dir)
app$run()
