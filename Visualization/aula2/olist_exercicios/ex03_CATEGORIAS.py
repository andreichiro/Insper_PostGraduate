from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from scipy.stats import gaussian_kde

from TEMPLATE import ConfigTemplate, Template


@dataclass
class ConfiguracaoExercicio03(ConfigTemplate):
    arquivo_saida: str = "ex03_categorias_preco_python.png"
    pasta_calibracao: str = "calibracao_ex03"
    fonte_fixa: str = "DejaVu Sans"


class Exercicio03CategoriasPreco(Template):
    CATEGORIAS_TOP_BOTTOM = [
        "relogios_presentes",
        "automotivo",
        "informatica_acessorios",
        "brinquedos",
        "beleza_saude",
        "cama_mesa_banho",
        "esporte_lazer",
        "moveis_decoracao",
        "utilidades_domesticas",
        "telefonia",
    ]
    ROTULOS_PT = {
        "relogios_presentes": "Relogios Presentes",
        "automotivo": "Automotivo",
        "informatica_acessorios": "Informatica Acessorios",
        "brinquedos": "Brinquedos",
        "beleza_saude": "Beleza Saude",
        "cama_mesa_banho": "Cama Mesa Banho",
        "esporte_lazer": "Esporte Lazer",
        "moveis_decoracao": "Moveis Decoracao",
        "utilidades_domesticas": "Utilidades Domesticas",
        "telefonia": "Telefonia",
    }

    def __init__(self, config: ConfiguracaoExercicio03) -> None:
        super().__init__(config=config)
        self.metricas_qa: dict[str, int] = {}
        self.mediana_global_preco: float = float("nan")

    def preparar_dados(self) -> None:
        if self.df_raw is None:
            raise RuntimeError("Dados não carregados.")

        raw = self.df_raw.copy()
        self.metricas_qa = {
            "linhas": int(len(raw)),
            "na_categoria": int(raw["product_category_name"].isna().sum()),
            "na_preco": int(raw["price"].isna().sum()),
            "duplicadas_total": int(raw.duplicated().sum()),
            "duplicadas_chave": int(raw.duplicated(subset=["order_id", "order_item_id", "product_id", "seller_id"]).sum()),
        }

        self.mediana_global_preco = float(pd.to_numeric(raw["price"], errors="coerce").dropna().median())
        if not np.isfinite(self.mediana_global_preco):
            raise RuntimeError("Mediana global de preço inválida após limpeza.")

        df = raw.dropna(subset=["product_category_name", "price"]).copy()
        df["product_category_name"] = df["product_category_name"].astype(str).str.strip().str.lower()
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna(subset=["price"]).copy()

        df = df.loc[df["product_category_name"].isin(self.CATEGORIAS_TOP_BOTTOM)].copy()
        df = df.loc[df["price"] > 0].copy()
        df["price_kde"] = np.where(df["price"] <= 300, df["price"], np.nan)

        df["categoria_slug"] = pd.Categorical(
            df["product_category_name"],
            categories=self.CATEGORIAS_TOP_BOTTOM,
            ordered=True,
        )
        self.df_plot = df.sort_values(["categoria_slug", "price"], ignore_index=True)

    @staticmethod
    def _parametros_base() -> dict[str, float]:
        return {
            "density_scale": 60.0,
            "bw_adjust": 0.66,
            "line_w": 1.6,
            "titulo_fs": 33,
            "titulo_x": 0.07,
            "titulo_y": 0.93,
            "eixo_fs": 22,
            "categoria_fs": 16,
            "ticks_fs": 14,
            "mediana_fs": 14,
            "left": 0.24,
            "right": 0.965,
            "top": 0.86,
            "bottom": 0.14,
            "mediana_y": -0.22,
        }

    def gerar_grade_parametros(self, max_combinacoes: int) -> list[dict[str, float]]:
        # GRID 1 - chute no escuro (pessimo)
        grid_1 = {
            "density_scale": [50, 55, 60],
            "bw_adjust": [0.70, 0.80, 0.90],
            "titulo_y": [0.92, 0.93],
            "left": [0.21, 0.22],
            "bottom": [0.13, 0.14],
        }

        # GRID 2 - torto
        grid_2 = {
            "density_scale": [55, 57, 59],
            "bw_adjust": [0.78, 0.82, 0.86],
            "titulo_y": [0.925, 0.93, 0.935],
            "left": [0.215, 0.22, 0.225],
            "bottom": [0.135, 0.14, 0.145],
        }

        # GRID 3 - quase
        grid_3 = {
            "density_scale": [56, 57, 58],
            "bw_adjust": [0.80, 0.82, 0.84],
            "titulo_y": [0.928, 0.932, 0.936],
            "left": [0.218, 0.22, 0.222],
            "bottom": [0.138, 0.14, 0.142],
        }

        # GRID 4 - ajuste final (CANDIDATO CORRETO: cand_001)
        # cand_001 = density_scale 60, bw_adjust 0.66, titulo_y 0.93, left 0.24, bottom 0.14
        grid_4 = {
            "density_scale": [60, 62, 58],
            "bw_adjust": [0.66, 0.62, 0.70],
            "titulo_y": [0.93, 0.928],
            "left": [0.24],
            "bottom": [0.14],
        }

        # a diferença está aqui: ordem pensada para variar bw/density cedo nos primeiros candidatos
        ordem = ["titulo_y", "left", "bottom", "density_scale", "bw_adjust"]
        grade: list[dict[str, float]] = []
        chaves_vistas: set[tuple[float, ...]] = set()

        def adicionar_grid(g: dict[str, list[float]]) -> None:
            for valores in itertools.product(*(g[chave] for chave in ordem)):
                candidato = {ch: float(v) for ch, v in zip(ordem, valores)}
                assinatura = (
                    candidato["density_scale"],
                    candidato["bw_adjust"],
                    candidato["titulo_y"],
                    candidato["left"],
                    candidato["bottom"],
                )
                if assinatura in chaves_vistas:
                    continue
                chaves_vistas.add(assinatura)
                grade.append(candidato)
                if len(grade) >= max_combinacoes:
                    return

        # todos os grids
        for g in (grid_4, grid_3, grid_2, grid_1):
            if len(grade) >= max_combinacoes:
                break
            adicionar_grid(g)

        return grade

    def _kde_curva(self, valores: np.ndarray, x_grid: np.ndarray, bw_adjust: float) -> np.ndarray:
        if len(valores) < 5:
            return np.zeros_like(x_grid)
        kde = gaussian_kde(valores, bw_method=lambda s: s.scotts_factor() * bw_adjust)
        return kde(x_grid)

    def _renderizar(self, parametros: dict[str, float], caminho_saida) -> None:
        if self.df_plot is None:
            raise RuntimeError("Dados para plot não preparados.")

        p = self._parametros_base()
        p.update(parametros)

        plt.rcParams["font.family"] = self.config.fonte_fixa
        fig, ax = plt.subplots(figsize=(13.65, 7.68), facecolor="#FFFFFF")
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")

        #começa antes de zero!
        x_grid = np.linspace(-15, 300, 920)
        n_cat = len(self.CATEGORIAS_TOP_BOTTOM)

        for idx, slug in enumerate(self.CATEGORIAS_TOP_BOTTOM):
            y_base = n_cat - 1 - idx
            parte_full = self.df_plot.loc[self.df_plot["categoria_slug"] == slug, "price"].to_numpy()
            parte_kde = (
                self.df_plot.loc[
                    (self.df_plot["categoria_slug"] == slug) & self.df_plot["price_kde"].notna(),
                    "price_kde",
                ]
                .to_numpy()
                .astype(float)
            )
            dens = self._kde_curva(parte_kde, x_grid, bw_adjust=float(p["bw_adjust"]))
            y_topo = y_base + dens * float(p["density_scale"])

            cor_fill = "#5679DC" if idx == 0 else "#CFCFCF"
            alpha_fill = 0.95 if idx == 0 else 1.0

            ax.fill_between(x_grid, y_base, y_topo, color=cor_fill, alpha=alpha_fill, linewidth=0)
            ax.plot(x_grid, y_topo, color="#1A1A1A", linewidth=float(p["line_w"]))
            ax.hlines(y_base, xmin=-15, xmax=315, color="#1A1A1A", linewidth=0.85)

            med_cat = float(np.median(parte_full))
            dens_med = float(self._kde_curva(parte_kde, np.array([med_cat]), bw_adjust=float(p["bw_adjust"]))[0]) * float(p["density_scale"])
            ax.vlines(med_cat, y_base, y_base + dens_med, color="#1A1A1A", linewidth=float(p["line_w"]))

        ax.axvline(self.mediana_global_preco, color="#F06272", linestyle=(0, (3.5, 4.0)), linewidth=1.6)
        ax.text(
            self.mediana_global_preco + 2.5,
            float(p["mediana_y"]),
            "Mediana",
            color="#F06272",
            fontsize=float(p["mediana_fs"]),
            ha="left",
            va="center",
        )

        y_ticks = [n_cat - 1 - i for i in range(n_cat)]
        y_labels = [self.ROTULOS_PT[slug] for slug in self.CATEGORIAS_TOP_BOTTOM]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)

        ax.set_xlim(-15, 315)
        ax.set_xticks([0, 100, 200, 300])
        ax.set_xticklabels(["R$0", "R$100", "R$200", "R$300"])
        ax.set_ylim(float(p["mediana_y"]) - 0.45, n_cat - 1 + 0.65)

        ax.xaxis.set_minor_locator(MultipleLocator(50))
        ax.grid(which="major", axis="x", color="#D2D2D2", linewidth=1.1)
        ax.grid(which="minor", axis="x", color="#E3E3E3", linewidth=0.7)
        ax.grid(which="major", axis="y", color="#E3E3E3", linewidth=0.7)
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.tick_params(axis="both", colors="#5B5B5B", labelsize=float(p["ticks_fs"]), length=0)
        ax.set_xlabel("Preço", fontsize=float(p["eixo_fs"]), color="#1A1A1A", fontweight="normal")
        ax.set_ylabel("")
        fig.text(
            0.06,
            0.53,
            "Categoria",
            fontsize=float(p["categoria_fs"]),
            color="#1A1A1A",
            ha="center",
            va="center",
            rotation=90,
            fontweight="normal",
        )

        fig.text(
            float(p["titulo_x"]),
            float(p["titulo_y"]),
            "Relógios são caros!!",
            fontsize=float(p["titulo_fs"]),
            color="#1A1A1A",
            ha="left",
            va="center",
            fontweight="normal",
        )

        plt.subplots_adjust(
            left=float(p["left"]),
            right=float(p["right"]),
            top=float(p["top"]),
            bottom=float(p["bottom"]),
        )
        fig.savefig(caminho_saida, dpi=100, facecolor=fig.get_facecolor())
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrar", action="store_true", help="Gera candidatos da grade para inspeção manual.")
    parser.add_argument("--max-combinacoes", type=int, default=24, help="Quantidade de candidatos na calibração.")
    args = parser.parse_args()

    config = ConfiguracaoExercicio03()
    pipeline = Exercicio03CategoriasPreco(config=config)
    saida = pipeline.executar(calibrar=args.calibrar, max_combinacoes=args.max_combinacoes)
    print("qa:", pipeline.metricas_qa)
    print(f"arquivo salvo: {saida}")
