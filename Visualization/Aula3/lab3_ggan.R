suppressPackageStartupMessages({
  library(readxl)
  library(dplyr)
  library(lubridate)
  library(ggplot2)
  library(gganimate)
  library(scales)
  library(htmltools)
  library(glue)
  library(knitr)
  library(tidyr)
})

base_dir <- "/Users/akatsurada/Documents/INSPER/Visualization/Aula3"
script_path <- file.path(base_dir, "lab3_ggan.R")
gif_path <- file.path(base_dir, "lab3_ggan.gif")
html_path <- file.path(base_dir, "lab3_ggan.html")

config <- list(
  frames_estaticos = 3L,
  frames_transicao = 2L,
  frames_pulso_inicial = 14L,
  frames_pulso_final = 18L,
  tolerancia_relativa = 0.05,
  cor_alta = "#16a34a",
  cor_baixa = "#dc2626",
  cor_neutra = "#64748b"
)

fmt_brl <- label_number(
  prefix = "R$ ",
  big.mark = ".",
  decimal.mark = ",",
  accuracy = 1
)

encontrar_input_path <- function(base_dir) {
  arquivos_excel <- Sys.glob(file.path(base_dir, "dados_luis*.xlsx"))

  if (length(arquivos_excel) == 0) {
    stop("Nenhum arquivo 'dados_luis*.xlsx' foi encontrado em ", base_dir)
  }

  arquivos_excel[which.max(file.info(arquivos_excel)$mtime)]
}

carregar_dados_mensais <- function(input_path) {
  dados_mensais_estado <- read_excel(input_path) |>
    mutate(
      Data = as.Date(Data),
      mes = floor_date(Data, "month"),
      Estado = if_else(is.na(Estado), NA_character_, paste("Estado", as.character(Estado)))
    ) |>
    filter(!is.na(Estado)) |>
    group_by(mes, Estado) |>
    summarise(
      juros_medio_diario = sum(juros_diarios, na.rm = TRUE) / n_distinct(Data),
      .groups = "drop"
    )

  if (nrow(dados_mensais_estado) == 0) {
    stop("A base não gerou nenhuma série mensal por estado após o tratamento.")
  }

  dados_mensais_estado
}

preparar_serie_animacao <- function(dados_mensais_estado, config) {
  mes_inicio <- min(dados_mensais_estado$mes, na.rm = TRUE)
  mes_fim <- max(dados_mensais_estado$mes, na.rm = TRUE)

  estados_comuns <- intersect(
    dados_mensais_estado |> filter(mes == mes_inicio) |> pull(Estado),
    dados_mensais_estado |> filter(mes == mes_fim) |> pull(Estado)
  )

  if (length(estados_comuns) == 0) {
    stop("Nenhum estado aparece ao mesmo tempo no primeiro e no último mês da série.")
  }

  dados_animacao <- dados_mensais_estado |>
    filter(Estado %in% estados_comuns) |>
    mutate(Estado = factor(Estado, levels = sort(estados_comuns))) |>
    arrange(Estado, mes)

  meses_animacao <- tibble(
    mes = sort(unique(dados_animacao$mes)),
    mes_id = seq_along(mes),
    frame_label = format(mes, "%m/%Y")
  )

  dados_animacao <- dados_animacao |>
    left_join(meses_animacao, by = "mes") |>
    arrange(Estado, mes_id)

  resumo_estados <- dados_animacao |>
    filter(mes %in% c(mes_inicio, mes_fim)) |>
    mutate(periodo = if_else(mes == mes_inicio, "inicio", "fim")) |>
    select(Estado, periodo, juros_medio_diario) |>
    pivot_wider(names_from = periodo, values_from = juros_medio_diario) |>
    left_join(
      dados_animacao |>
        group_by(Estado) |>
        summarise(
          min_val = min(juros_medio_diario),
          min_mes = format(mes[which.min(juros_medio_diario)][1], "%m/%Y"),
          max_val = max(juros_medio_diario),
          max_mes = format(mes[which.max(juros_medio_diario)][1], "%m/%Y"),
          .groups = "drop"
        ),
      by = "Estado"
    ) |>
    mutate(
      delta = fim - inicio,
      recuperacao = fim - min_val,
      tolerancia = abs(inicio) * config$tolerancia_relativa,
      cor_halo = case_when(
        delta > tolerancia ~ config$cor_alta,
        delta < -tolerancia ~ config$cor_baixa,
        TRUE ~ config$cor_neutra
      )
    )

  list(
    mes_inicio = mes_inicio,
    mes_fim = mes_fim,
    meses_animacao = meses_animacao,
    dados_animacao = dados_animacao,
    resumo_estados = resumo_estados,
    ponto_inicio = dados_animacao |> filter(mes == mes_inicio),
    ponto_fim = dados_animacao |> filter(mes == mes_fim),
    maior_queda = resumo_estados |> slice_min(delta, n = 1, with_ties = FALSE),
    maior_alta = resumo_estados |> slice_max(delta, n = 1, with_ties = FALSE),
    maior_recuperacao = resumo_estados |> slice_max(recuperacao, n = 1, with_ties = FALSE)
  )
}

montar_frames <- function(meses_animacao, mes_inicio, mes_fim, config) {
  novo_frame <- function(frame_id, segment_id, progress, mes_plot, frame_label, pulse_type, halo_alpha, halo_size) {
    tibble(
      frame_id = frame_id,
      segment_id = segment_id,
      progress = progress,
      mes_plot = mes_plot,
      frame_label = frame_label,
      pulse_type = pulse_type,
      halo_alpha = halo_alpha,
      halo_size = halo_size
    )
  }

  frame_specs <- list()
  frame_cursor <- 1L

  forca_pulso_inicial <- sin(seq(0, pi, length.out = config$frames_pulso_inicial))
  for (j in seq_len(config$frames_pulso_inicial)) {
    frame_specs[[length(frame_specs) + 1L]] <- novo_frame(
      frame_id = frame_cursor,
      segment_id = 1L,
      progress = 0,
      mes_plot = mes_inicio,
      frame_label = format(mes_inicio, "%m/%Y"),
      pulse_type = "inicio",
      halo_alpha = 0.08 + 0.26 * forca_pulso_inicial[j],
      halo_size = 6 + 4 * forca_pulso_inicial[j]
    )
    frame_cursor <- frame_cursor + 1L
  }

  for (i in seq_len(nrow(meses_animacao))) {
    for (j in seq_len(config$frames_estaticos)) {
      frame_specs[[length(frame_specs) + 1L]] <- novo_frame(
        frame_id = frame_cursor,
        segment_id = i,
        progress = 0,
        mes_plot = meses_animacao$mes[i],
        frame_label = meses_animacao$frame_label[i],
        pulse_type = "nenhum",
        halo_alpha = 0,
        halo_size = 0
      )
      frame_cursor <- frame_cursor + 1L
    }

    if (i < nrow(meses_animacao)) {
      for (j in seq_len(config$frames_transicao)) {
        progress <- j / config$frames_transicao
        label_transicao <- if (j < config$frames_transicao) {
          meses_animacao$frame_label[i]
        } else {
          meses_animacao$frame_label[i + 1L]
        }

        frame_specs[[length(frame_specs) + 1L]] <- novo_frame(
          frame_id = frame_cursor,
          segment_id = i,
          progress = progress,
          mes_plot = as.Date(
            as.numeric(meses_animacao$mes[i]) +
              (as.numeric(meses_animacao$mes[i + 1L]) - as.numeric(meses_animacao$mes[i])) * progress,
            origin = "1970-01-01"
          ),
          frame_label = label_transicao,
          pulse_type = "nenhum",
          halo_alpha = 0,
          halo_size = 0
        )
        frame_cursor <- frame_cursor + 1L
      }
    }
  }

  forca_pulso_final <- sin(seq(0, pi, length.out = config$frames_pulso_final))
  for (j in seq_len(config$frames_pulso_final)) {
    frame_specs[[length(frame_specs) + 1L]] <- novo_frame(
      frame_id = frame_cursor,
      segment_id = nrow(meses_animacao),
      progress = 0,
      mes_plot = mes_fim,
      frame_label = format(mes_fim, "%m/%Y"),
      pulse_type = "fim",
      halo_alpha = 0.06 + 0.24 * forca_pulso_final[j],
      halo_size = 5.8 + 3.8 * forca_pulso_final[j]
    )
    frame_cursor <- frame_cursor + 1L
  }

  bind_rows(frame_specs)
}

montar_camadas_animacao <- function(dados_animacao, resumo_estados, ponto_inicio, ponto_fim, frame_specs, config) {
  frame_labels <- frame_specs |>
    distinct(frame_id, frame_label) |>
    arrange(frame_id) |>
    pull(frame_label)

  segmentos_estado <- dados_animacao |>
    group_by(Estado) |>
    arrange(mes_id, .by_group = TRUE) |>
    mutate(
      juros_proximo = lead(juros_medio_diario, default = last(juros_medio_diario))
    ) |>
    ungroup() |>
    select(Estado, mes_id, juros_medio_diario, juros_proximo)

  pontos_frames <- frame_specs |>
    select(frame_id, segment_id, progress, mes_plot) |>
    left_join(segmentos_estado, by = c("segment_id" = "mes_id"), relationship = "many-to-many") |>
    transmute(
      frame_id,
      Estado,
      mes = mes_plot,
      juros_medio_diario = juros_medio_diario + (juros_proximo - juros_medio_diario) * progress
    ) |>
    left_join(
      resumo_estados |> select(Estado, cor_halo),
      by = "Estado"
    )

  marcador_tempo <- frame_specs |>
    transmute(
      frame_id,
      mes = mes_plot
    )

  halo_inicio <- frame_specs |>
    filter(pulse_type == "inicio") |>
    select(frame_id, halo_alpha, halo_size) |>
    tidyr::crossing(
      ponto_inicio |>
        select(Estado, mes, juros_medio_diario) |>
        mutate(cor_halo = config$cor_neutra)
    )

  halo_final <- frame_specs |>
    filter(pulse_type == "fim") |>
    select(frame_id, halo_alpha, halo_size) |>
    tidyr::crossing(
      ponto_fim |>
        select(Estado, mes, juros_medio_diario) |>
        left_join(
          resumo_estados |> select(Estado, cor_halo),
          by = "Estado"
        )
    )

  list(
    frame_labels = frame_labels,
    marcador_tempo = marcador_tempo,
    halos_animados = bind_rows(halo_inicio, halo_final),
    ponto_inicio = ponto_inicio,
    ponto_fim = ponto_fim |>
      left_join(
        resumo_estados |> select(Estado, cor_halo),
        by = "Estado"
      ),
    pontos_frames = pontos_frames
  )
}

criar_grafico_animado <- function(dados_animacao, camadas_animacao, frame_specs, fmt_brl) {
  ggplot() +
    geom_vline(
      data = camadas_animacao$marcador_tempo,
      aes(xintercept = as.numeric(mes)),
      colour = "#e2e8f0",
      linewidth = 0.75,
      linetype = "dashed"
    ) +
    geom_line(
      data = dados_animacao,
      aes(x = mes, y = juros_medio_diario, group = Estado),
      colour = "#cbd5e1",
      linewidth = 1.15,
      lineend = "round"
    ) +
    geom_point(
      data = camadas_animacao$ponto_inicio,
      aes(x = mes, y = juros_medio_diario),
      shape = 21,
      fill = "#ffffff",
      colour = "#64748b",
      stroke = 1.4,
      size = 3.5
    ) +
    geom_point(
      data = camadas_animacao$ponto_fim,
      aes(x = mes, y = juros_medio_diario, colour = cor_halo),
      shape = 21,
      fill = "#ffffff",
      stroke = 1.5,
      size = 3.5
    ) +
    geom_point(
      data = camadas_animacao$halos_animados,
      aes(x = mes, y = juros_medio_diario, size = halo_size, alpha = halo_alpha, colour = cor_halo),
      shape = 16,
      stroke = 0
    ) +
    geom_point(
      data = camadas_animacao$pontos_frames,
      aes(x = mes, y = juros_medio_diario, colour = cor_halo),
      shape = 21,
      fill = "#0f172a",
      stroke = 1.2,
      size = 3.7
    ) +
    facet_wrap(~Estado, ncol = 2, scales = "free_y") +
    scale_x_date(
      date_breaks = "1 year",
      date_labels = "%Y",
      expand = expansion(mult = c(0.04, 0.06))
    ) +
    scale_y_continuous(
      labels = fmt_brl,
      expand = expansion(mult = c(0.12, 0.18))
    ) +
    scale_size_identity() +
    scale_alpha_identity() +
    scale_colour_identity() +
    labs(
      title = "Juro médio diário por estado",
      subtitle = "Mês destacado: {camadas_animacao$frame_labels[pmax(1, pmin(length(camadas_animacao$frame_labels), round(frame_time)))]}",
      x = "Ano",
      y = "Juro médio diário"
    ) +
    theme_minimal(base_size = 15) +
    theme(
      plot.title.position = "plot",
      plot.title = element_text(face = "bold", size = 18, colour = "#0f172a", margin = margin(b = 20)),
      plot.subtitle = element_text(size = 12.5, colour = "#334155", lineheight = 1.15, margin = margin(t = 4, b = 10)),
      plot.caption = element_text(size = 10.5, colour = "#475569", margin = margin(t = 12)),
      plot.margin = margin(18, 24, 24, 32),
      axis.title.x = element_text(size = 12.5, margin = margin(t = 16)),
      axis.title.y = element_text(size = 12.5, margin = margin(r = 18)),
      axis.text.x = element_text(size = 11, margin = margin(t = 8)),
      axis.text.y = element_text(size = 10.5, margin = margin(r = 6)),
      strip.text = element_text(face = "bold", size = 13, colour = "#0f172a", margin = margin(t = 6, b = 6)),
      strip.background = element_rect(fill = "#f8fafc", colour = NA),
      panel.spacing.x = grid::unit(1.5, "cm"),
      panel.spacing.y = grid::unit(1.35, "cm"),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(colour = "#e2e8f0"),
      panel.grid.major.y = element_line(colour = "#f1f5f9")
    ) +
    transition_time(frame_id) +
    ease_aes("linear")
}

criar_pagina_html <- function(gif_path, mes_inicio, mes_fim) {
  tags$html(
    tags$head(
      tags$meta(charset = "utf-8"),
      tags$title("Laboratório 3 - gganimate"),
      tags$style(HTML(
        "body{margin:0;background:#f8fafc;color:#0f172a;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
         .wrap{max-width:1100px;margin:0 auto;padding:32px 24px 48px}
         .chart{margin:24px 0 28px;background:#fff;border-radius:16px;padding:16px;box-shadow:0 12px 30px rgba(15,23,42,.08)}
         img{display:block;width:100%;height:auto;border-radius:12px}
         p,li{line-height:1.6}
         a{color:#0f766e}"
      ))
    ),
    tags$body(
      tags$div(
        class = "wrap",
        tags$h1("Extensão 4 - gganimate (Animação)"),
        tags$p(
          tags$strong("Documentação: "),
          tags$a(
            href = "https://gganimate.com/",
            target = "_blank",
            rel = "noopener noreferrer",
            "https://gganimate.com/"
          )
        ),
        tags$div(
          class = "chart",
          tags$img(
            src = knitr::image_uri(gif_path),
            alt = "Animação temporal do juro médio diário por estado"
          )
        )
      )
    )
  )
}

input_path <- encontrar_input_path(base_dir)
dados_mensais_estado <- carregar_dados_mensais(input_path)
serie_animacao <- preparar_serie_animacao(dados_mensais_estado, config)
frame_specs <- montar_frames(
  meses_animacao = serie_animacao$meses_animacao,
  mes_inicio = serie_animacao$mes_inicio,
  mes_fim = serie_animacao$mes_fim,
  config = config
)
camadas_animacao <- montar_camadas_animacao(
  dados_animacao = serie_animacao$dados_animacao,
  resumo_estados = serie_animacao$resumo_estados,
  ponto_inicio = serie_animacao$ponto_inicio,
  ponto_fim = serie_animacao$ponto_fim,
  frame_specs = frame_specs,
  config = config
)
grafico_animado <- criar_grafico_animado(
  dados_animacao = serie_animacao$dados_animacao,
  camadas_animacao = camadas_animacao,
  frame_specs = frame_specs,
  fmt_brl = fmt_brl
)

gif_animado <- animate(
  grafico_animado,
  nframes = max(frame_specs$frame_id),
  fps = 12,
  duration = max(frame_specs$frame_id) / 12,
  width = 1400,
  height = 980,
  res = 130,
  renderer = gifski_renderer(loop = TRUE)
)

anim_save(gif_path, animation = gif_animado)

pagina_html <- criar_pagina_html(
  gif_path = gif_path,
  mes_inicio = serie_animacao$mes_inicio,
  mes_fim = serie_animacao$mes_fim
)

save_html(pagina_html, file = html_path)

cat("entrada:", normalizePath(input_path, winslash = "/"), "\n")
cat("R script         :", normalizePath(script_path, winslash = "/"), "\n")
cat("GIF              :", normalizePath(gif_path, winslash = "/"), "\n")
cat("HTML             :", normalizePath(html_path, winslash = "/"), "\n")
