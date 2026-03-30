from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from TEMPLATE import ConfigTemplate, Template

# python3 /Users/akatsurada/Documents/INSPER/Visualization/aula2/olist_exercicios/ex01_PAG.py --calibrar --max-combinacoes 30
# testar c/ diffs p/ gerar_grade_parametros
# candidato final
# python3 /Users/akatsurada/Documents/INSPER/Visualization/aula2/olist_exercicios/ex01_PAG.py


@dataclass
class ConfiguracaoExercicio01(ConfigTemplate):
    arquivo_saida: str = "EX1PYHON.png"
    pasta_calibracao: str = "calibracao_ex01"
    fonte_fixa: str = "serif"


class E1FormasPagamento(Template):
    def __init__(self, config: ConfiguracaoExercicio01) -> None:
        super().__init__(config=config)

    def preparar_dados(self) -> None:
        if self.df_raw is None:
            raise RuntimeError("Dados não carregados.")

        base = (
            self.df_raw["types"]
            .value_counts(dropna=True)
            .rename_axis("forma_pagamento")
            .reset_index(name="quantidade")
        )

        base = base.loc[base["quantidade"] > 100].copy()
        base["quantidade_milhar"] = base["quantidade"] / 1000
        base["rotulo"] = base["quantidade_milhar"].map(lambda x: f"{x:.2f}")
        base["x_rotulo"] = base["quantidade_milhar"] / 2
        base.loc[base["quantidade_milhar"] < 6, "x_rotulo"] += 0.65
        self.df_plot = base.sort_values("quantidade_milhar", ascending=True, ignore_index=True)

    @staticmethod
    def _parametros_base() -> dict[str, float]:
        return {
            "largura_barra": 0.52,
            "titulo_x": 0.225,
            "titulo_y": 0.93,
            "subtitulo_x": 0.225,
            "subtitulo_y": 0.877,
            "titulo_fs": 29,
            "subtitulo_fs": 19,
            "ticks_fs": 15,
            "eixo_fs": 20,
            "caption_fs": 15,
            "ajuste_left": 0.225,
            "ajuste_right": 0.97,
            "ajuste_top": 0.84,
            "ajuste_bottom": 0.21,
        }

    def gerar_grade_parametros(self, max_combinacoes: int) -> list[dict[str, float]]:
        # GRID 1 - ficou ruim
        grid_1 = {
            "titulo_fs": [28, 29, 30],
            "subtitulo_fs": [18, 19, 20],
            "ajuste_left": [0.21, 0.225, 0.24],
            "ajuste_bottom": [0.20, 0.21, 0.22],
            "largura_barra": [0.50, 0.52],
        }

        # GRID 2 - quase
        grid_2 = {
            "titulo_fs": [29, 30],
            "subtitulo_fs": [19, 20],
            "ajuste_left": [0.222, 0.225, 0.228],
            "ajuste_bottom": [0.205, 0.21, 0.215],
            "largura_barra": [0.51, 0.52, 0.53],
        }

        # GRID 3 USAR ESSE
        # ->>> titulo_fs=29, subtitulo_fs=19, ajuste_left=0.225,
        # ajuste_bottom=0.21, largura_barra=0.52
        grid_3 = {
            "titulo_fs": [29],
            "subtitulo_fs": [19],
            "ajuste_left": [0.224, 0.225, 0.226],
            "ajuste_bottom": [0.209, 0.21, 0.211],
            "largura_barra": [0.515, 0.52, 0.525],
        }

        ordem = ["titulo_fs", "subtitulo_fs", "ajuste_left", "ajuste_bottom", "largura_barra"]
        grade: list[dict[str, float]] = []
        chaves_vistas: set[tuple[float, ...]] = set()

        def adicionar_grid(g: dict[str, list[float]]) -> None:
            for valores in itertools.product(*(g[chave] for chave in ordem)):
                candidato = {
                    "titulo_fs": float(valores[0]),
                    "subtitulo_fs": float(valores[1]),
                    "ajuste_left": float(valores[2]),
                    "titulo_x": float(valores[2]),
                    "subtitulo_x": float(valores[2]),
                    "ajuste_bottom": float(valores[3]),
                    "largura_barra": float(valores[4]),
                }
                assinatura = (
                    candidato["titulo_fs"],
                    candidato["subtitulo_fs"],
                    candidato["ajuste_left"],
                    candidato["ajuste_bottom"],
                    candidato["largura_barra"],
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

    def _renderizar(self, parametros: dict[str, float], caminho_saida: Path) -> None:
        if self.df_plot is None:
            raise RuntimeError("Dados para plot não preparados.")

        p = self._parametros_base()
        p.update(parametros)
        df = self.df_plot

        fig, ax = plt.subplots(figsize=(13.65, 7.68), facecolor="#101215")
        ax.set_facecolor("#34363D")

        ax.barh(
            y=df["forma_pagamento"],
            width=df["quantidade_milhar"],
            color="#78D4D4",
            edgecolor="#78D4D4",
            height=float(p["largura_barra"]),
        )

        for _, row in df.iterrows():
            ax.text(
                row["x_rotulo"],
                row["forma_pagamento"],
                row["rotulo"],
                ha="center",
                va="center",
                fontsize=12,
                color="#3A3A3A",
                family=self.config.fonte_fixa,
                clip_on=False,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "#F4F4F4",
                    "edgecolor": "#6D6D6D",
                    "linewidth": 1.0,
                },
            )

        ax.set_xlim(0, 88)
        ax.set_xticks([0, 20, 40, 60, 80])
        ax.tick_params(axis="x", colors="#D7D9DB", labelsize=float(p["ticks_fs"]), length=0)
        ax.tick_params(axis="y", colors="#D7D9DB", labelsize=float(p["ticks_fs"]), length=0)
        ax.grid(True, which="major", axis="both", color="#64686D", linewidth=1.0, alpha=0.72)
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.text(
            float(p["titulo_x"]),
            float(p["titulo_y"]),
            "Formas de pagamento mais comuns",
            ha="left",
            va="center",
            fontsize=float(p["titulo_fs"]),
            color="#F1F1F1",
            family=self.config.fonte_fixa,
        )
        fig.text(
            float(p["subtitulo_x"]),
            float(p["subtitulo_y"]),
            "Considerando tipos com mais de 100 observações",
            ha="left",
            va="center",
            fontsize=float(p["subtitulo_fs"]),
            color="#E6E6E6",
            family=self.config.fonte_fixa,
        )

        ax.set_xlabel("Quantidade\n(milhares)", fontsize=float(p["eixo_fs"]), color="#E6E6E6", family=self.config.fonte_fixa)
        ax.set_ylabel("Forma de pagamento", fontsize=float(p["eixo_fs"]), color="#E6E6E6", family=self.config.fonte_fixa)

        fig.text(
            0.97,
            0.045,
            "Fonte: Olist",
            ha="right",
            va="bottom",
            fontsize=float(p["caption_fs"]),
            color="#D7D9DB",
            family=self.config.fonte_fixa,
        )

        plt.subplots_adjust(
            left=float(p["ajuste_left"]),
            right=float(p["ajuste_right"]),
            top=float(p["ajuste_top"]),
            bottom=float(p["ajuste_bottom"]),
        )
        fig.savefig(caminho_saida, dpi=100, facecolor=fig.get_facecolor())
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrar", action="store_true", help="Gera candidatos da grade para inspeção manual.")
    parser.add_argument("--max-combinacoes", type=int, default=36, help="Quantidade de candidatos na calibração.")
    args = parser.parse_args()

    configuracao = ConfiguracaoExercicio01()
    pipeline = E1FormasPagamento(config=configuracao)
    caminho = pipeline.executar(calibrar=args.calibrar, max_combinacoes=args.max_combinacoes)
    print(f"Arquivo salvo em: {caminho}")
