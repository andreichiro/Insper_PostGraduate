from __future__ import annotations

import argparse
import itertools
import urllib.request
from dataclasses import dataclass

import emoji
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

from TEMPLATE import ConfigTemplate, Template


@dataclass
class ConfiguracaoExercicio02(ConfigTemplate):
    arquivo_saida: str = "ex02_produtos_data_estado_python.png"
    pasta_calibracao: str = "calibracao_ex02"
    fonte_fixa: str = "Futura"


class Exercicio02ProdutosPorDataEstado(Template):
    def __init__(self, config: ConfiguracaoExercicio02) -> None:
        super().__init__(config=config)
        self.metricas_qa: dict[str, int] = {}
        self._emoji_img: np.ndarray | None = None

    def preparar_dados(self) -> None:
        if self.df_raw is None:
            raise RuntimeError("s/ dados")

        df = self.df_raw.copy()
        self.metricas_qa = {
            "linhas": int(len(df)),
            "na_seller_state": int(df["seller_state"].isna().sum()),
            "na_order_purchase_timestamp": int(df["order_purchase_timestamp"].isna().sum()),
            "duplicadas_total": int(df.duplicated().sum()),
            "duplicadas_chave": int(df.duplicated(subset=["order_id", "order_item_id", "product_id", "seller_id"]).sum()),
        }

        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["order_purchase_timestamp", "seller_state"]).copy()
        df["seller_state"] = df["seller_state"].astype(str).str.strip().str.upper()
        df = df.loc[df["seller_state"] != ""].copy()

        df["mes_compra"] = df["order_purchase_timestamp"].dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()
        df["estado"] = "Outros"
        df.loc[df["seller_state"] == "RJ", "estado"] = "RJ"
        df.loc[df["seller_state"] == "SP", "estado"] = "SP"

        inicio = pd.Timestamp("2017-01-01")
        fim = pd.Timestamp("2018-07-01")
        df = df.loc[(df["mes_compra"] >= inicio) & (df["mes_compra"] <= fim)].copy()

        base = df.groupby(["mes_compra", "estado"], as_index=False).size().rename(columns={"size": "quantidade"})

        meses = pd.date_range(inicio, fim, freq="MS")
        estados = ["RJ", "SP", "Outros"]
        grade = pd.MultiIndex.from_product([meses, estados], names=["mes_compra", "estado"]).to_frame(index=False)
        base = grade.merge(base, on=["mes_compra", "estado"], how="left")
        base["quantidade"] = base["quantidade"].fillna(0).astype(int)
        base["estado"] = pd.Categorical(base["estado"], categories=estados, ordered=True)
        self.df_plot = base.sort_values(["estado", "mes_compra"], ignore_index=True)

    @staticmethod
    def _rotulo_mes_pt(data: pd.Timestamp) -> str:
        meses_pt = {1: "jan", 2: "fev", 3: "mar", 4: "abr", 5: "mai", 6: "jun", 7: "jul", 8: "ago", 9: "set", 10: "out", 11: "nov", 12: "dez"}
        return f"{meses_pt[data.month]}\n{data.year}"

    def _carregar_emoji_grimacing(self) -> np.ndarray:
        if self._emoji_img is not None:
            return self._emoji_img

        pasta_assets = self.config.pasta_saida / "_assets"
        pasta_assets.mkdir(parents=True, exist_ok=True)
        caminho_emoji = pasta_assets / "emoji_grimacing.png"

        if not caminho_emoji.exists():
            emoji_char = emoji.emojize(":grimacing_face:", language="alias")
            codepoints = "-".join(f"{ord(c):x}" for c in emoji_char)
            url = f"https://cdnjs.cloudflare.com/ajax/libs/twemoji/14.0.2/72x72/{codepoints}.png"
            with urllib.request.urlopen(url, timeout=20) as resp:
                conteudo = resp.read()
            with open(caminho_emoji, "wb") as f:
                f.write(conteudo)

        with Image.open(caminho_emoji).convert("RGBA") as im:
            self._emoji_img = np.array(im)
        return self._emoji_img

    @staticmethod
    def _parametros_base() -> dict[str, float]:
        return {
            "line_lw": 4.5,
            "titulo_fs": 35,
            "subtitulo_fs": 24,
            "xlabel_fs": 20,
            "ylabel_fs": 20,
            "ticks_fs": 14,
            "legend_fs": 16,
            "estado_fs": 20,
            "fonte_fs": 16,
            "legend_y": -0.23,
            "estado_y": 0.082,
            "fonte_y": 0.046,
            "bottom": 0.245,
            "top": 0.735,
            "left": 0.12,
            "right": 0.96,
            "titulo_x": 0.12,
            "titulo_y": 0.875,
            "subtitulo_x": 0.12,
            "subtitulo_y": 0.815,
            "emoji_gap": 0.02,
            "emoji_zoom": 0.285,
        }

    def gerar_grade_parametros(self, max_combinacoes: int) -> list[dict[str, float]]:
        # GRID 1 ruim
        # Candidato-base testado no início: line_lw=4.0, titulo_fs=38, subtitulo_fs=28, legend_y=-0.20, bottom=0.22
        grid_1 = {
            "line_lw": [4.0, 4.5, 5.0],
            "titulo_fs": [38, 40, 42],
            "subtitulo_fs": [28, 30, 32],
            "legend_y": [-0.20, -0.23, -0.26],
            "bottom": [0.22, 0.235, 0.25],
        }

        # GRID 2 ruim
        # Região de busca refinada: line_lw em torno de 4.5 e ajuste fino de legenda/bottom.
        grid_2 = {
            "line_lw": [4.3, 4.5, 4.7],
            "titulo_fs": [39, 40, 41],
            "subtitulo_fs": [29, 30, 31],
            "legend_y": [-0.22, -0.23, -0.24],
            "bottom": [0.23, 0.235, 0.24],
        }

        # GRID 3 - melhor até aqui (CANDIDATO CORRETO: cand_001)
        # cand_001 = line_lw 4.5, titulo_fs 35, subtitulo_fs 24, legend_y -0.23, bottom 0.245
        grid_3 = {
            "line_lw": [4.5, 4.4, 4.6],
            "titulo_fs": [35, 34, 36],
            "subtitulo_fs": [24, 23, 25],
            "legend_y": [-0.23, -0.225, -0.235],
            "bottom": [0.245, 0.24, 0.25],
        }

        ordem = ["line_lw", "titulo_fs", "subtitulo_fs", "legend_y", "bottom"]
        grade: list[dict[str, float]] = []
        chaves_vistas: set[tuple[float, ...]] = set()

        def adicionar_grid(g: dict[str, list[float]]) -> None:
            for valores in itertools.product(*(g[chave] for chave in ordem)):
                candidato = {
                    "line_lw": float(valores[0]),
                    "titulo_fs": float(valores[1]),
                    "subtitulo_fs": float(valores[2]),
                    "legend_y": float(valores[3]),
                    "bottom": float(valores[4]),
                }
                assinatura = (
                    candidato["line_lw"],
                    candidato["titulo_fs"],
                    candidato["subtitulo_fs"],
                    candidato["legend_y"],
                    candidato["bottom"],
                )
                if assinatura in chaves_vistas:
                    continue
                chaves_vistas.add(assinatura)
                grade.append(candidato)
                if len(grade) >= max_combinacoes:
                    return

        # prioridade: primeiro grid de paridade, depois refino, depois histórico
        for g in (grid_3, grid_2, grid_1):
            if len(grade) >= max_combinacoes:
                break
            adicionar_grid(g)

        return grade

    def _renderizar(self, parametros: dict[str, float], caminho_saida) -> None:
        if self.df_plot is None:
            raise RuntimeError("Dados para plot não preparados.")

        p = self._parametros_base()
        p.update(parametros)
        df = self.df_plot

        plt.rcParams["font.family"] = self.config.fonte_fixa
        cores = {"RJ": "#404A97", "SP": "#238E97", "Outros": "#67BD4B"}

        fig, ax = plt.subplots(figsize=(16.6, 9.45), facecolor="white")
        ax.set_facecolor("#F1F1F1")

        for estado in ["RJ", "SP", "Outros"]:
            parte = df.loc[df["estado"] == estado]
            ax.plot(parte["mes_compra"], parte["quantidade"], color=cores[estado], linewidth=float(p["line_lw"]), label=estado, solid_capstyle="round")

        ax.set_ylim(0, 6500)
        ax.set_yticks([0, 2000, 4000, 6000])
        ax.set_xlim(pd.Timestamp("2016-12-05"), pd.Timestamp("2018-07-30"))

        ticks_x = pd.date_range("2017-03-01", "2018-06-01", freq="3MS")
        ax.set_xticks(ticks_x)
        ax.set_xticklabels([self._rotulo_mes_pt(x) for x in ticks_x])

        ax.grid(which="major", axis="both", color="#D0D0D0", linewidth=1.0)
        ax.yaxis.set_minor_locator(plt.MultipleLocator(1000))
        ax.grid(which="minor", axis="y", color="#E5E5E5", linewidth=0.5)
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_color("#CCCCCC")
            spine.set_linewidth(1.6)

        ax.tick_params(axis="both", colors="#6A6A6A", labelsize=float(p["ticks_fs"]))
        ax.set_xlabel("Data", fontsize=float(p["xlabel_fs"]), color="#202020", labelpad=10)
        ax.set_ylabel("Quantidade", fontsize=float(p["ylabel_fs"]), color="#202020", labelpad=10)

        fig.text(
            float(p["titulo_x"]),
            float(p["titulo_y"]),
            "São Paulo tem mais vendas",
            fontsize=float(p["titulo_fs"]),
            color="#1A1A1A",
            ha="left",
            va="center",
        )
        subtitulo_txt = fig.text(
            float(p["subtitulo_x"]),
            float(p["subtitulo_y"]),
            "O que é esperado, pois a população é maior",
            fontsize=float(p["subtitulo_fs"]),
            color="#1A1A1A",
            ha="left",
            va="center",
            fontfamily=self.config.fonte_fixa,
        )
        fig.canvas.draw()
        bbox = subtitulo_txt.get_window_extent(renderer=fig.canvas.get_renderer())
        x_dir_subtitulo, y_meio_subtitulo = fig.transFigure.inverted().transform((bbox.x1, (bbox.y0 + bbox.y1) / 2))
        emoji_img = self._carregar_emoji_grimacing()
        emoji_ab = AnnotationBbox(
            OffsetImage(emoji_img, zoom=float(p["emoji_zoom"])),
            (x_dir_subtitulo + float(p["emoji_gap"]), y_meio_subtitulo),
            xycoords=fig.transFigure,
            frameon=False,
            box_alignment=(0.0, 0.5),
            pad=0.0,
        )
        fig.add_artist(emoji_ab)

        legenda = ax.legend(loc="upper center", bbox_to_anchor=(0.58, float(p["legend_y"])), ncol=3, frameon=False, fontsize=float(p["legend_fs"]), handlelength=0.8, columnspacing=0.9, borderaxespad=0.0)
        for linha in legenda.get_lines():
            linha.set_linewidth(float(p["line_lw"]))

        fig.text(0.415, float(p["estado_y"]), "Estado", fontsize=float(p["estado_fs"]), color="#202020", ha="right", va="center")
        fig.text(0.93, float(p["fonte_y"]), "Fonte: Olist", fontsize=float(p["fonte_fs"]), color="#202020", ha="right", va="center")

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

    config = ConfiguracaoExercicio02()
    pipeline = Exercicio02ProdutosPorDataEstado(config=config)
    saida = pipeline.executar(calibrar=args.calibrar, max_combinacoes=args.max_combinacoes)
    print("QA:", pipeline.metricas_qa)
    print(f"Arquivo salvo em: {saida}")
