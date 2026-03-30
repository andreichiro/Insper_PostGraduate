from __future__ import annotations

import argparse
import itertools
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class ConfigTemplate:
    # cada ex ajustar aq
    url_dados: str = "https://github.com/padsInsper/202533-padsv/releases/download/dados/olist_items.parquet"
    pasta_saida: Path = Path("/Users/akatsurada/Documents/INSPER/Visualization/aula2/olist_exercicios/saidas")
    arquivo_saida: str = "template_saida.png"
    pasta_calibracao: str = "calibracao_template"
    fonte_fixa: str = "Helvetica"
    top_k_inspecao: int = 5
    candidato_final: str = "base"

    #aqui n precisa ajustar
    @property
    def caminho_saida(self) -> Path:
        return self.pasta_saida / self.arquivo_saida

    @property
    def caminho_pasta_calibracao(self) -> Path:
        return self.pasta_saida / self.pasta_calibracao


class Template:
    def __init__(self, config: ConfigTemplate) -> None:
        self.config = config
        self.df_raw: pd.DataFrame | None = None
        self.df_plot: pd.DataFrame | None = None

    def carregar_dados(self) -> None:
        self.df_raw = pd.read_parquet(self.config.url_dados)

    def preparar_dados(self) -> None:
        if self.df_raw is None:
            raise RuntimeError("Dados não carregados.")

        # TMUDAR AQUI SEMPRE
        self.df_plot = self.df_raw.copy()

    @staticmethod
    def _parametros_base() -> dict[str, float]:
        # TPARAMETROS AJUSTAVEIS AQUI
        return {"param_a": 1.0, "param_b": 2.0}

    def _renderizar(self, parametros: dict[str, float], caminho_saida: Path) -> None:
        if self.df_plot is None:
            raise RuntimeError("Dados de plot não preparados.")

        # grafico  self.df_plot + params
        saida_teste = caminho_saida.with_suffix(".csv")
        self.df_plot.head(10).to_csv(saida_teste, index=False)

    #imutavel
    @staticmethod
    def _normalizar_parametros_linha(linha: pd.Series) -> dict[str, float]:
        parametros_validos = set(Template._parametros_base().keys())
        saida: dict[str, float] = {}
        for chave in parametros_validos:
            if chave in linha.index and pd.notna(linha[chave]):
                saida[chave] = float(linha[chave])
        return saida

    def gerar_grade_parametros(self, max_combinacoes: int) -> list[dict[str, float]]:
        #IDEIA: testar pra ver a ordem de grandeza e ai clocar na grade
        grade = list(itertools.product([0.8, 1.0, 1.2], [1.5, 2.0, 2.5]))

        #RODAR A GRADE
        saida: list[dict[str, float]] = []
        for param_a, param_b in grade[:max_combinacoes]:
            saida.append({"param_a": float(param_a), "param_b": float(param_b)})
        return saida

    @staticmethod
    def _formatar_valor(v: object) -> str:
        if isinstance(v, float):
            return f"{v:.3f}".rstrip("0").rstrip(".")
        return str(v)

    # imutavel
    def gerar_imagem_teste_grade(self, grade_df: pd.DataFrame, n_colunas: int = 5) -> Path:
        caminho_saida = self.config.caminho_pasta_calibracao / "grade_visual_teste.png"
        if grade_df.empty:
            return caminho_saida

        n_total = len(grade_df)
        n_col = max(1, min(n_colunas, n_total))
        n_lin = math.ceil(n_total / n_col)
        fig, axes = plt.subplots(n_lin, n_col, figsize=(n_col * 3.2, n_lin * 2.6), facecolor="white")

        if n_lin == 1 and n_col == 1:
            axes_list = [axes]
        elif n_lin == 1 or n_col == 1:
            axes_list = list(axes)
        else:
            axes_list = [ax for linha in axes for ax in linha]

        for ax in axes_list:
            ax.axis("off")

        col_param = [c for c in grade_df.columns if c not in {"id_candidato", "arquivo"}]
        for i, (_, linha) in enumerate(grade_df.iterrows()):
            ax = axes_list[i]
            img = plt.imread(self.config.caminho_pasta_calibracao / str(linha["arquivo"]))
            ax.imshow(img)
            numero = str(linha["id_candidato"]).replace("cand_", "")
            preview = ", ".join(
                f"{c}={self._formatar_valor(linha[c])}" for c in col_param[:3]
            )
            ax.set_title(f"{numero} | {preview}", fontsize=8, pad=2)
            ax.axis("off")

        fig.suptitle("Grade de teste (candidatos numerados)", fontsize=12, y=0.995)
        fig.tight_layout()
        fig.savefig(caminho_saida, dpi=120)
        plt.close(fig)
        return caminho_saida

    # imutavel
    def calibrar_layout(self, max_combinacoes: int = 24) -> dict[str, float]:
        self.config.caminho_pasta_calibracao.mkdir(parents=True, exist_ok=True)
        arquivos_atuais = list(self.config.caminho_pasta_calibracao.glob("cand_*.png"))
        for nome_extra in ("grade_parametros.csv", "grade_visual_teste.png"):
            caminho_extra = self.config.caminho_pasta_calibracao / nome_extra
            if caminho_extra.exists():
                arquivos_atuais.append(caminho_extra)

        if arquivos_atuais:
            pasta_historico = self.config.caminho_pasta_calibracao / "historico_rodadas"
            pasta_historico.mkdir(parents=True, exist_ok=True)
            id_rodada = datetime.now().strftime("%Y%m%d_%H%M%S")
            pasta_rodada = pasta_historico / f"rodada_{id_rodada}"
            pasta_rodada.mkdir(parents=True, exist_ok=True)
            for arquivo in arquivos_atuais:
                shutil.move(str(arquivo), str(pasta_rodada / arquivo.name))

        grade = self.gerar_grade_parametros(max_combinacoes=max_combinacoes)
        resultados: list[dict[str, float | str]] = []
        for i, parametros in enumerate(grade[:max_combinacoes], start=1):
            id_candidato = f"cand_{i:03d}"
            caminho_candidato = self.config.caminho_pasta_calibracao / f"{id_candidato}.png"

            self._renderizar(parametros=parametros, caminho_saida=caminho_candidato)
            resultados.append({"id_candidato": id_candidato, "arquivo": caminho_candidato.name, **parametros})

        grade_df = pd.DataFrame(resultados)
        grade_df.to_csv(self.config.caminho_pasta_calibracao / "grade_parametros.csv", index=False)
        self.gerar_imagem_teste_grade(grade_df=grade_df)
        return self._normalizar_parametros_linha(grade_df.iloc[0])

    def _carregar_parametros_candidato(self, candidato: str) -> dict[str, float]:
        # metodo p/ cada candidato carregar seus params
        caminho_grade = self.config.caminho_pasta_calibracao / "grade_parametros.csv"
        if not caminho_grade.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado em {caminho_grade}. Rode com --calibrar antes."
            )

        grade_df = pd.read_csv(caminho_grade)
        if grade_df.empty:
            raise RuntimeError("Arquivo grade_parametros.csv está vazio.")

        if candidato == "auto":
            linha = grade_df.iloc[0]
        else:
            filtrado = grade_df.loc[grade_df["id_candidato"] == candidato]
            if filtrado.empty:
                raise ValueError(f"Candidato '{candidato}' não encontrado no grade_parametros.csv.")
            linha = filtrado.iloc[0]

        return self._normalizar_parametros_linha(linha)

    # execute universla
    def executar(self, calibrar: bool = False, max_combinacoes: int = 24) -> Path:
        self.carregar_dados()
        self.preparar_dados()

        parametros_finais: dict[str, float] = {}
        if calibrar:
            self.calibrar_layout(max_combinacoes=max_combinacoes)

        candidato = self.config.candidato_final.strip().lower()
        if candidato != "base":
            parametros_finais = self._carregar_parametros_candidato(self.config.candidato_final.strip())

        self.config.pasta_saida.mkdir(parents=True, exist_ok=True)
        self._renderizar(parametros=parametros_finais, caminho_saida=self.config.caminho_saida)
        return self.config.caminho_saida


# testar
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrar", action="store_true")
    parser.add_argument("--max-combinacoes", type=int, default=24)
    args = parser.parse_args()

    cfg = ConfigTemplate()
    pipe = Template(config=cfg)
    caminho = pipe.executar(
        calibrar=args.calibrar,
        max_combinacoes=args.max_combinacoes,
    )
    print(f"Saída em: {caminho}")
