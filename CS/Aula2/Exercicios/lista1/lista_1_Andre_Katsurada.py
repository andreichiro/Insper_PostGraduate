# lista_1_Andre_Katsurada.py
"""
Lista 1 — Exercícios de 1-10
Autor: André Ichiro Katsurada
Data: 12/08/25
Curso: Aprendizagem Estatística de Máquina I — INSPER
"""

from __future__ import annotations

import csv                                 
import math                                 
from abc import ABC, abstractmethod, ABCMeta      
from dataclasses import dataclass           
from pathlib import Path                   
from typing import ClassVar, Dict, List
from decimal import Decimal, ROUND_HALF_UP
import datetime as _dt

import re
import zipfile
import requests
import io
import warnings
import unicodedata as _ud

import pandas as pd                      

import numpy as np

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.nonparametric.smoothers_lowess import lowess

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from scipy.stats import pearsonr, spearmanr, permutation_test

import pingouin as pg

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from linearmodels.panel import PanelOLS

#Path dos arquivos gerados
BASE_DIR: Path = Path(__file__).resolve().parent

#Classe p/ registro de exercícios
class _RegistroExercicios:
    """Mapeamento de exercícios"""

    _mapa: Dict[int, "Exercicio"] = {}

    @classmethod
    def registrar(cls, numero: int, classe: "Exercicio") -> None:
        existente = cls._mapa.get(numero)
        if existente and existente.__qualname__ != classe.__qualname__:
            raise ValueError(f"Exercício {numero} já registrado por {existente}")
        cls._mapa[numero] = classe

    @classmethod
    def instancias_ordenadas(cls) -> List["Exercicio"]:
        """Instancia e devolve as classes"""
        return [cls._mapa[i]() for i in sorted(cls._mapa)]

class _ExercicioMeta(ABCMeta):
    """Metaclass que faz o registro das subclasses"""

    def __new__(mcls, name, bases, ns, **kwargs):
        cls = super().__new__(mcls, name, bases, ns)
        numero = ns.get("numero")
        if numero is not None:
            _RegistroExercicios.registrar(numero, cls)
        return cls

class Exercicio(ABC, metaclass=_ExercicioMeta):
    """Interface pros exercícios"""

    numero: ClassVar[int]

    @abstractmethod
    def executar(self) -> None:
        """Executa a solução."""

#Helpers p/ facilitar a interação & previnir erros nos exercícios c/ input do usuário
def _ask_int(prompt: str, *, min_value: int | None = None, max_value: int | None = None) -> int:
    """Loop until the user provides a valid integer within optional bounds."""
    while True:
        try:
            raw = input(prompt)
        except EOFError:
            print("\nEntrada encerrada.")
            raise
        try:
            val = int(raw)
        except (ValueError, TypeError):
            print("Input inválido. Digite um número inteiro.")
            continue
        if min_value is not None and val < min_value:
            print(f"Valor mínimo permitido é {min_value}.")
            continue
        if max_value is not None and val > max_value:
            print(f"Valor máximo permitido é {max_value}.")
            continue
        return val

def _ask_float(prompt: str, *, min_value: float | None = None, max_value: float | None = None) -> float:
    """Loop until the user provides a valid float within optional bounds."""
    while True:
        try:
            raw = input(prompt)
        except EOFError:
            print("\nEntrada encerrada.")
            raise
        try:
            val = float(str(raw).replace(",", "."))
        except (ValueError, TypeError):
            print("Input inválido. Digite um número do tipo float (use . ou , para decimais).")
            continue
        if min_value is not None and val < min_value:
            print(f"Valor mínimo permitido é {min_value}.")
            continue
        if max_value is not None and val > max_value:
            print(f"Valor máximo permitido é {max_value}.")
            continue
        return val

#Exercícios 1-7
@dataclass(slots=True, frozen=True)
class Ex01Variaveis(Exercicio):
    """Cria variáveis c/ print """
    numero: ClassVar[int] = 1
    #Gerar CSV e ler com pandas)

    nome: str = "André Ichiro"
    idade: int = 39

    def executar(self) -> None:
        print(f"Olá, me chamo {self.nome} e tenho {self.idade} anos.")

class Ex02Media(Exercicio):
    """Calcula manualmente a média de idades"""
    numero: ClassVar[int] = 2
    _idades: List[int] = [20, 21, 20, 22, 19, 18, 14, 35]

    def executar(self) -> None:
        soma = 0
        for idade in self._idades:
            soma += idade
        media = soma / len(self._idades)
        print(f"Média das idades = {media:.3f}")

class Ex03SomaInteiros(Exercicio):
    numero: ClassVar[int] = 3

    def executar(self) -> None:
        numeros: List[int] = []
        for i in range(4):
            valor = _ask_int(f"Digite o {i+1}º número inteiro: ")
            numeros.append(valor)

        soma = 0
        for n in numeros:
            soma += n

        print(f"Números digitados: {numeros}")
        print(f"Soma = {soma}")

class Ex04ListComp(Exercicio):
    numero: ClassVar[int] = 4

    def executar(self) -> None:
        n = _ask_int("Gerar quadrados pares até n = ", min_value=0)
        quadrados_pares = [x**2 for x in range(n + 1) if (x**2) % 2 == 0]
        print(quadrados_pares)

class Ex05Piramide(Exercicio):
    numero: ClassVar[int] = 5

    def executar(self) -> None:
        n = _ask_int("Número de linhas da pirâmide: ", min_value=1)
        for i in range(1, n + 1):
            print(" ".join([str(i)] * i))

class Ex06LojaTintas(Exercicio):
    numero: ClassVar[int] = 6
    COBERTURA_M2_POR_L = 3
    LITROS_POR_LATA = 18
    PRECO_LATA = 80.0

    def executar(self) -> None:
        area = _ask_float("Área a ser pintada (m²): ", min_value=0.0)
        litros = area / self.COBERTURA_M2_POR_L
        latas = math.ceil(litros / self.LITROS_POR_LATA)
        custo = latas * self.PRECO_LATA
        print(f"{latas} lata(s) — preço total R$ {custo:.2f}")

class Ex07Palindromo(Exercicio):
    numero: ClassVar[int] = 7

    def executar(self) -> None:
        texto = input("Digite a frase/palavra: ")
        normalizado = "".join(
            ch.casefold()
            for ch in _ud.normalize("NFKD", texto)
            if _ud.combining(ch) == 0 and ch.isalnum() 
        )
        msg = "é palíndromo!" if normalizado == normalizado[::-1] \
              else "não é palíndromo!"
        print(f"“{texto}” {msg}")


class Ex08ListaCompras(Exercicio):
    """
    Processa 'supermercado.csv' e responde:
        8a) Valor total da compra: somatório de qtd × preço, c/ frações
        8b) Quantidade total de itens, onde qtd < 1 conta como 1 unid. e qtd ≥ 1 usa a parte inteira
        8c) Valor recalculado onde frações contam como 1 unidade (ceil)
        8d) Produto mais caro em valor unitário
    """
    numero: ClassVar[int] = 8
    _csv_path: Path = BASE_DIR / "supermercado.csv"

    def executar(self) -> None:
        if not self._csv_path.exists():
            print("Arquivo supermercado.csv não encontrado no path.\n"
                  "Suba o arquivo na pasta do script!")
            return

        valor_total = Decimal("0")
        qtd_produtos = 0
        valor_recalc = Decimal("0")
        prod_mais_caro = ("", Decimal("0"))

        with self._csv_path.open(newline="", encoding="utf-8") as arq:
            leitor = csv.reader(arq, delimiter=";")

            #Skip pro cabeçalho
            header = next(leitor, None)

            for row in leitor:
                if not row or len(row) < 3:
                    continue

                produto = row[0].strip()
                qtd_str = row[1].strip().replace(",", ".")
                preco_str = row[2].strip().replace(",", ".")

                try:
                    qtd = Decimal(qtd_str)
                    preco = Decimal(preco_str)
                except Exception:
                    #Linha inválida
                    continue

                #1) Pergunta 1: Valor total com frações
                valor_total += qtd * preco

                #2) Pergunta 2: Contagem de itens onde frações contam como 1 unidade adicional
                unidades = 1 if qtd < 1 else int(qtd) 
                qtd_produtos += unidades

                #3) Valor recalculado com frações arredondadas p/ 1 unidade (ceil)
                valor_recalc += Decimal(unidades) * preco

                #4) Pergunta 4: Produto mais caro (unitário) 
                if preco > prod_mais_caro[1]:
                    prod_mais_caro = (produto, preco)

        # Prints arredondando 2 casas decimais
        v1 = valor_total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        v3 = valor_recalc.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        print(f"Valor total da compra: R$ {v1}")
        print(f"Quantidade de produtos: {qtd_produtos}")
        print("Valor total recalculado (frações arredondadas para 1 unid.): "
              f"R$ {v3}")
        print(f"Produto mais caro (unit.): {prod_mais_caro[0]} — R$ {prod_mais_caro[1].quantize(Decimal('0.01'))}")

#Exercício 9
class Ex09IsPrime(Exercicio):
    """Gera os 100 primeiros números primos."""
    numero: ClassVar[int] = 9

    @staticmethod
    def is_prime(n: int) -> bool:
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        limite = int(math.sqrt(n)) + 1
        for i in range(5, limite, 6):           
            if n % i == 0 or n % (i + 2) == 0:
                return False
        return True

    def executar(self) -> None:
        primos: List[int] = []
        candidato = 2
        while len(primos) < 100:
            if self.is_prime(candidato):
                primos.append(candidato)
            candidato += 1
        print("100 primeiros números primos:")
        print(primos)

#Exercício 10 
class Ex10GeracaoDados(Exercicio):
    """
    Ver se países com maior % de adultos (25+) com ensino superior tendem a ter PIB maior.

    O método 'executar':
        10.1. Faz download de dados do Banco Mundial (ensino superior, PIB per capita, 
        população total e área terrestre);
        10.2. Prepara dois conjuntos de dados de demo ('_dados_llm1'/'_dados_llm2')
        e o dataset real;
        10.3. Limpa, valida faixas e cria features em log ('log_gdp', 'log_pop', 'log_area')
        10.4. Para cada conjunto de países ou subconjuntos como top/bottom 3, 10 e 
        Brasil-EUA-Alemanha), analisa:
           10.4a) Correlações Pearson/Spearman (IC95% via Fisher z) c/ p via permutação 
           10.4b) Correlação parcial controlando 'log_pop'  
           10.4c) Regressão linear (OLS) com erros HC3 e checagens básicas (heterocedasticidade, não linearidade, normalidade, colinearidade e pontos influentes)
           10.4d) Regressão quantílica (tao = 0.5) para robustez  
           10.4e) Geração de gráficos em ex10_figs/

    Observação: trata-se de uma análise estatística exploratória procurando por correlações; não infere causalidade
    """
    
    @staticmethod
    def download_indicator(indicator_code: str) -> pd.DataFrame:
        """
        Baixa os dados Banco Mundial e devolve o DF 
        """
        url = f"https://api.worldbank.org/v2/en/indicator/{indicator_code}?downloadformat=csv"
        resp = requests.get(url)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            data_file = [name for name in zf.namelist()
                        if name.startswith("API_") and name.endswith(".csv")][0]
            df = pd.read_csv(zf.open(data_file), skiprows=4)
        return df

    def _build_worldbank_df(self) -> pd.DataFrame:
        """
        Baixa os dados do Banco Mundial e retorna um DF
        """
        #Lista de anos, agr com os últimos 30 anos (usando apenas 5 anos, a análise estava muito restrita)
        current_year = _dt.date.today().year
        years_w = [str(y) for y in range(current_year, current_year - 30, -1)]

        #Indicadores
        edu_raw  = self.download_indicator("SE.TER.CUAT.BA.ZS")   # % 25+ com superior
        gdp_raw  = self.download_indicator("NY.GDP.MKTP.CD")      # PIB total (US$)
        pop_raw  = self.download_indicator("SP.POP.TOTL")         # população
        area_raw = self.download_indicator("AG.LND.TOTL.K2")      # área (km²)

        def has_val(row: pd.Series, y: str) -> bool:
            return (y in row) and pd.notnull(row.get(y))

        records: list[dict] = []
        for _, gdp_row in gdp_raw.iterrows():
            iso = gdp_row["Country Code"]
            pop_row  = pop_raw[pop_raw["Country Code"] == iso]
            area_row = area_raw[area_raw["Country Code"] == iso]
            edu_row  = edu_raw[edu_raw["Country Code"] == iso]
            if pop_row.empty or area_row.empty or edu_row.empty:
                continue

            prow  = pop_row.iloc[0]
            arow  = area_row.iloc[0]
            erow  = edu_row.iloc[0]

            for y in years_w:
                if all((has_val(gdp_row, y), has_val(prow, y), has_val(arow, y), has_val(erow, y))):
                    records.append({
                        "country": gdp_row["Country Name"],
                        "iso": iso,
                        "year": int(y),
                        "gdp": float(gdp_row.get(y)),
                        "population": float(prow.get(y)),
                        "area": float(arow.get(y)),
                        "higher_ed_pct": float(erow.get(y)),
                    })

        df = pd.DataFrame(records)
        if not df.empty:
            df.to_csv(self._REAL_CSV, index=False)
        return df

    #Aviso p/ tamanhos amostrais muito pequenos. Valor aparentemente arbitrário, pensar em como estipular algo melhor
    _MIN_N = 4

    numero: ClassVar[int] = 10

    #Path pros dados reais
    _REAL_CSV: ClassVar[Path] = BASE_DIR / "education_gdp_dataset.csv"
    #Path pros csvs gerados por LLM
    _LLM1_CSV: ClassVar[Path] = BASE_DIR / "llm1_fake_data.csv"
    _LLM2_CSV: ClassVar[Path] = BASE_DIR / "llm2_fake_data.csv"

    #Dados p/ demo
    _dados_llm1 = {
        "País": ["Brasil", "EUA", "México", "Alemanha", "França", "OCDE", "G20", "América do Sul"],
        "GDP": [1.9, 23.4, 1.3, 4.2, 2.9, 60.0, 90.0, 3.3],           # seria em trilhões 
        "Território": [8.5, 9.8, 1.9, 0.36, 0.55, 42.0, 60.0, 17.8],   # seria em milhões km² 
        "População": [214, 331, 128, 83, 67, 1400, 4800, 430],         # seria em milhões 
        "Superior (%)": [18, 48, 17, 35, 34, 38, 35, 20],  # provavelmente mudar p/ algo tipo população economicamente ativa 
    }
    _dados_llm2 = {
        "País": ["Brasil", "EUA", "México", "Alemanha", "França", "OCDE", "G20", "América do Sul"],
        "GDP": [2.1, 25.0, 1.5, 4.0, 3.0, 55.0, 95.0, 4.0],
        "Território": [8.5, 9.8, 2.0, 0.35, 0.55, 40.0, 61.0, 18.0],
        "População": [210, 333, 129, 84, 65, 1350, 4900, 420],
        "Superior (%)": [20, 47, 19, 36, 33, 40, 34, 22],
    }

    _FIGDIR: Path = BASE_DIR / "ex10_figs"

    #Helpers internos 
    @staticmethod
    def _to_frame(d: dict | "pd.DataFrame") -> "pd.DataFrame":
        import pandas as pd
        df = d.copy() if isinstance(d, pd.DataFrame) else pd.DataFrame(d).copy()
        df.rename(
            columns={
                "País": "country",
                "GDP": "gdp",
                "Território": "area",
                "População": "population",
                "Superior (%)": "higher_ed_pct",
            },
            inplace=True,
        )
        for c in ("gdp", "area", "population", "higher_ed_pct"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def _load_real_dataset(self, source: str | Path | pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza os dados reais usando as colunas country, population, area, higher_ed_pct e gdp (ou gdp_pc).
        Retorna um DF
        """
        if isinstance(source, (str, Path)):
            real = pd.read_csv(source)
        else:
            real = source.copy()

        req = {"country", "population", "area", "higher_ed_pct"}
        missing = req - set(real.columns)
        if missing:
            raise ValueError(f"Dataframe não têm as seguintes colunas obrigatórias: {sorted(missing)}")

        if "gdp" not in real.columns:
            if "gdp_pc" not in real.columns:
                raise ValueError("Dataframe precisa ter 'gdp' ou 'gdp_pc'.")
            real["gdp"] = pd.to_numeric(real["gdp_pc"], errors="coerce") * pd.to_numeric(real["population"], errors="coerce")

        base_cols = ["country", "gdp", "population", "area", "higher_ed_pct"]
        cols = (["country", "iso"] + base_cols[1:]) if "iso" in real.columns else base_cols
        real = real[cols].copy()
        for c in [c for c in cols[1:] if c != "iso"]:
            real[c] = pd.to_numeric(real[c], errors="coerce")
        return real

    def _clean_and_feature(self, df: "pd.DataFrame") -> "pd.DataFrame":
        out = df.copy()

        #Remove NA essenciais
        out = out.dropna(subset=["gdp", "population", "area", "higher_ed_pct"])

        #Valida faixas
        out = out.loc[(out["higher_ed_pct"] >= 0) & (out["higher_ed_pct"] <= 100)].copy()
        out = out.loc[(out["gdp"] > 0) & (out["population"] > 0) & (out["area"] > 0)].copy()

        #Derivações
        out["log_gdp"] = np.log(out["gdp"].astype(float))
        out["log_area"] = np.log(out["area"].astype(float))

        #PIB per capita (para painel FE)
        out["gdp_pc"] = out["gdp"].astype(float) / out["population"].astype(float)
        out = out.loc[out["gdp_pc"] > 0].copy()
        out["log_gdp_pc"] = np.log(out["gdp_pc"].astype(float))

        #Educação
        out["higher_ed_pct"] = out["higher_ed_pct"].astype(float)

        out.reset_index(drop=True, inplace=True)
        return out

    @staticmethod
    def _print_header(title: str) -> None:
        print(f"\n# {title}") 

    #Análises
    def _panel_fe_analysis(self, df_panel: pd.DataFrame) -> None:
        """
        Painel: log(PIB per capita) ~ Educação(%) + efeitos fixos de país e de ano.
        Erros-padrão clusterizados por país
        Inicialmente, usei um corte transvesrsal, ou seja, 1 linha por país. Porém, aparentemente o mais recomendado seria guardar várias linhas por país, uma para cada ano ('painel')
        """
        self._print_header("Painel (FE país + ano): log(PIB per capita) ~ Educação (%)")
        d = df_panel.dropna(subset=["higher_ed_pct", "log_gdp_pc"]).copy()
        if d.empty:
            print("Amostra vazia após limpeza para painel.")
            return

        #Garante tipo do ano e índice 
        d["year"] = d["year"].astype(int)
        d = d.set_index(["country", "year"]).sort_index()

        #Vars
        y = d["log_gdp_pc"]
        X = d[["higher_ed_pct"]]  #sem constante; FE cuidam dos interceptos

        # Ajuste FE com cluster por país — NÃO passar 'clusters=' aqui
        res = PanelOLS(y, X, entity_effects=True, time_effects=True).fit(
            cov_type="clustered",
            cluster_entity=True
        )

        #Output
        beta = float(res.params.get("higher_ed_pct", np.nan))
        se   = float(res.std_errors.get("higher_ed_pct", np.nan))
        pval = float(res.pvalues.get("higher_ed_pct", np.nan))
        try:
            ci_low, ci_high = map(float, res.conf_int().loc["higher_ed_pct"].tolist())
        except Exception:
            ci_low = ci_high = float("nan")

        try:
            r2_within = float(res.rsquared.within)
        except Exception:
            r2_within = float("nan")

        nobs = int(getattr(res, "nobs", len(d)))
        n_countries = int(d.index.get_level_values(0).nunique())
        n_years = int(d.index.get_level_values(1).nunique())

        print(f"β_educ = {beta:.4f}  se={se:.4f}  CI95%[{ci_low:.4f},{ci_high:.4f}]  p={pval:.4g}")
        print(f"R² (within) = {r2_within:.3f} | nobs = {nobs} | países = {n_countries} | anos = {n_years}")
        return res  

    def _correlation_suite(self, df: "pd.DataFrame", y_col: str, x_col: str, label: str) -> None:
        y = df[y_col].to_numpy(dtype=float)
        x = df[x_col].to_numpy(dtype=float)

        n = len(x)

        x_const = np.allclose(x, x[0])
        y_const = np.allclose(y, y[0])

        if not (x_const or y_const):
            res = pg.pairwise_corr(
                df[[x_col, y_col]], columns=[x_col, y_col], method="pearson"
            )
            pr = float(res["r"].iloc[0])
            if "CI95%" in res.columns:
                ci_low, ci_high = res["CI95%"].iloc[0]
            else:
                # Fallback: Fisher z via Pingouin (95% por padrão deste relatório)
                ci_low, ci_high = pg.compute_esci(stat=pr, nx=n, ny=n, eftype="r", confidence=0.95)
            ci_str = f"[{ci_low:.3f}, {ci_high:.3f}] (pg/Fisher)"
        else:
            pr = float("nan")
            ci_str = "[NaN, NaN] (n/a)"

        #Spearman
        #Doc: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
        #Conferir se faz sentido usar esse 'Spearman rank-order correlation' aqui
        #Quando usar? Quando não usar?
        
        # Dessa forma gera warnings sr, p_sr = spearmanr(x, y)
        #Forma correta
        if x_const or y_const:
            sr, p_sr = float("nan"), float("nan")
        else:
            sr, p_sr = spearmanr(x, y)
        
        #Permutação p-value p/ Pearson
        #Doc: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html
        #Conferir se faz sentido usar esse 'Permutation test for Pearson correlation' aqui
        #Quando usar? Quando não usar?
        if not (x_const or y_const):
            perm_res = permutation_test(
                (x, y),
                statistic=lambda u, v: pearsonr(u, v)[0],
                alternative="two-sided", 
                vectorized=False,
                n_resamples=10_000, 
                random_state=123
            )
            p_perm = float(perm_res.pvalue)
        else:
            p_perm = float("nan")

        self._print_header(f"[{label}] Correlações {y_col} ~ {x_col}  (n={n})")
        print(f"Pearson r = {pr:.3f}  (IC95% {ci_str})  p_perm≈{p_perm:.4f}")
        print(f"Spearman ρ = {sr:.3f}  p={p_sr:.4f}")

    def _partial_correlation(self, df: "pd.DataFrame", y: str, x: str, controls: list[str]) -> None:
        """
        Mede a relação entre y e x depois de reduzir o efeito dos controls (no caso, populacao em log)
        """

        Xc = sm.add_constant(df[controls])
        resid_x = sm.OLS(df[x], Xc).fit().resid
        resid_y = sm.OLS(df[y], Xc).fit().resid
        r, p = pearsonr(resid_x, resid_y)

        self._print_header(f"Correlação parcial: {y} ~ {x} | controls={controls}")
        print(f"r_parcial = {r:.3f}  p≈{p:.4f}  (obtida via residualização OLS)")

    def _ols_with_diagnostics(self, df: "pd.DataFrame", y: str, x: str, controls: list[str]) -> "sm.regression.linear_model.RegressionResultsWrapper":
        """
        Regressão linear c/ HC3 e faz checagens p/:
        Breusch–Pagan: vê se a variância do erro muda com o nível das variáveis (heterocedasticidade);
        RESET: sinaliza possível curvatura que o modelo linear não captou;
        Jarque–Bera: verifica se os resíduos lembram uma distribuição normal;
        VIF: mede se as variáveis explicativas "se parecem demais" (colinearidade);
        Cook’s D / alavancagem: observa pontos que puxam muito o ajuste
        """

        #Matriz de regressão
        X = df[[x] + controls].copy()
        X = sm.add_constant(X)
        yv = df[y].astype(float).to_numpy()

        #Fit c/ HC3
        ols = sm.OLS(yv, X).fit(cov_type="HC3")

        rhs = x if not controls else f"{x} + {controls}"
        self._print_header(f"OLS (HC3): {y} ~ {rhs}")

        #Sumário
        coef = float(ols.params[x]); se = float(ols.bse[x]); pval = float(ols.pvalues[x])
        ci_low, ci_high = map(float, ols.conf_int().loc[x])
        print(f"coef_{x}={coef:.3f}  se={se:.3f}  CI95%[{ci_low:.3f},{ci_high:.3f}]  p={pval:.4f}")
        print(f"R² adj = {ols.rsquared_adj:.3f} | F-pvalue = {ols.f_pvalue:.4f} | n = {int(ols.nobs)}")

        #Beta padronizado 
        beta_x = float(ols.params[x]) * float(df[x].std(ddof=0)) / float(df[y].std(ddof=0))
        print(f"beta padronizado (aprox.) de {x} = {beta_x:.3f}")

        #Heterocedasticidade (Breusch–Pagan)
        #Doc: https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.het_breuschpagan.html
        #Conferir se faz sentido usar esse 'Breusch-Pagan Lagrange Multiplier test for heteroscedasticity' aqui
        #Quando usar? Quando não usar?
        bp_lm, bp_lmp, bp_f, bp_fp = het_breuschpagan(ols.resid, ols.model.exog)
        print(f"Breusch–Pagan: LM p={bp_lmp:.4f}")

        #Não linearidade: RESET de Ramsey, potência 2
        #Doc: https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.linear_reset.html
        #Conferir se faz sentido usar esse 'Ramsey RESET test' aqui
        #Quando usar? Quando não usar?
        try:
            if ols.df_resid >= 5 and len(ols.params) >= 2:
                reset_p = float(linear_reset(ols, use_f=True).pvalue)
                print(f"RESET (não linearidade): p={reset_p:.4f}")
            else:
                print("RESET (não linearidade): omitido (amostra muito pequena).")
        except Exception as e:
            print(f"RESET (não linearidade): omitido ({e})")

        #Normalidade dos resíduos
        #https://www.statsmodels.org/dev/generated/statsmodels.stats.stattools.jarque_bera.html
        #Conferir se faz sentido usar esse 'Jarque-Bera test for normality' aqui
        #Quando usar? Quando não usar?
        jb_stat, jb_p, _, _ = jarque_bera(ols.resid)
        print(f"Jarque–Bera (normalidade dos resíduos): p={jb_p:.4f}")

        #Multicolinearidade (VIF)
        #Doc: https://www.statsmodels.org/dev/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html
        #Conferir se faz sentido usar esse 'Variance Inflation Factor' aqui
        #Quando usar? Quando não usar?
        X_no_const = df[[x] + controls].astype(float)
        if X_no_const.shape[1] >= 2:
            X_v = X_no_const.to_numpy()
            vif = [(col, variance_inflation_factor(X_v, i)) for i, col in enumerate(X_no_const.columns)]
            print("VIF:")
            for col, val in vif:
                print(f"  {col:>12}: {val:6.3f}")
        else:
            print("VIF: n/a (somente 1 preditor).")

        #Influência: Cook D e leverage
        #Doc: https://www.statsmodels.org/dev/generated/statsmodels.stats.outliers_influence.OLSInfluence.html
        #Conferir se faz sentido usar esse 'Cook’s distance and leverage values' aqui
        #Quando usar? Quando não usar?
        infl = OLSInfluence(ols)
        cooks_d = infl.cooks_distance[0]
        leverage = infl.hat_matrix_diag

        idx_sorted = np.argsort(cooks_d)[::-1]
        top_k = int(min(3, len(df)))
        flagged = [(int(i), float(cooks_d[i]), float(leverage[i]), df.loc[int(i), "country"])
                for i in idx_sorted[:top_k]]
        if flagged:
            print(f"Pontos mais influentes (top {top_k} por Cook's D):")

        else:
            print("Nenhum ponto influente significativo por Cook's D.")

        return ols

    def _quantile_regression(self, df: "pd.DataFrame", y: str, x: str, controls: list[str]) -> None:
        """
        Regressão quantílica (mediana, 0.5) p/ a outliers na dependente.
        Útil quando há valores muito fora da curva
        """
        X = sm.add_constant(df[[x] + controls])
        yv = df[y].astype(float)
        q = sm.QuantReg(yv, X).fit(q=0.5)
        self._print_header(f"Regressão Quantílica (τ=0.5): {y} ~ {x} + {controls}")
        print(q.summary().tables[1])

    #Helpers p/ gráficos 
    def _ensure_figdir(self) -> None:
        self._FIGDIR.mkdir(parents=True, exist_ok=True)

    def _set_matplotlib_style(self) -> None:
        plt.rcParams.update({
            "figure.dpi": 130,
            "savefig.dpi": 200,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
        })

    @staticmethod
    def _point_style(n: int) -> tuple[int, float]:
        """
        Tamanho/transparência por densidade
        ≤30: maior e mais opaco; 30–100: médio; 100–250: pequeno; 250+: mínimo.
        """
        if n <= 30:
            return 28, 0.85
        elif n <= 100:
            return 20, 0.65
        elif n <= 250:
            return 14, 0.45
        else:
            return 10, 0.35

    #Labels compactos p/ país
    @staticmethod
    def _country_label(row: pd.Series, prefer: str = "iso") -> str:
        """
        Retorna o label 
        Prioriza 'iso' (3 letras); fallback para nome truncado
        """
        if prefer in row and pd.notna(row[prefer]) and str(row[prefer]).strip():
            return str(row[prefer]).upper()
        nm = str(row.get("country", ""))
        return (nm[:12] + "…") if len(nm) > 13 else nm

    def _plot_scatter_fit(
        self, df: "pd.DataFrame", y: str, x: str, label: str, fname: str,
        max_labels: int = 30, *, label_col: str | None = None,
    ) -> None:
        """Dispersão y~x + 2 tendências:
           - Se y já é log (ex.: 'log_gdp_pc'): OLS direto em y~x (eixo linear)
           - Se y está em nível (ex.: 'gdp'): OLS em ln(y)~x + smearing e eixo Y log
           LOWESS em ambos os casos para sugerir não linearidades.
        """
        self._ensure_figdir()

        xv = df[x].astype(float).to_numpy()
        yv = df[y].astype(float).to_numpy()
        n  = len(df)
        is_log_y = str(y).lower().startswith("log_")

        xs = np.linspace(np.nanmin(xv), np.nanmax(xv), 100, dtype=float)
        pt_size, alpha = self._point_style(n)

        fig = plt.figure(figsize=(7.8, 4.8))
        ax  = plt.gca()
        ax.margins(0.02)

        #Regressão e curvas
        X = pd.DataFrame({"const": 1.0, x: xv}, index=df.index)

        if is_log_y:
            #y já em log: OLS direto
            ols_lin = sm.OLS(yv, X).fit()
            b0 = float(ols_lin.params["const"]); b1 = float(ols_lin.params[x])
            yhat = b0 + b1 * xs

            ax.scatter(xv, yv, label=f"Países (n={n})", s=pt_size, alpha=alpha, edgecolor="k", linewidth=0.2)
            ax.plot(xs, yhat, label="Tendência média (OLS em log GDP pc)", linewidth=1.5)

            #LOWESS no espaço log
            lo = None
            if n >= 4:
                try:
                    lo = lowess(yv, xv, frac=0.6, return_sorted=True)
                    ax.plot(lo[:, 0], lo[:, 1], linestyle="--", label="Tendência local (LOWESS)", linewidth=1.2)
                except Exception:
                    pass

            resid = yv - ols_lin.predict(X)
            ax.set_ylabel("log PIB per capita")
            title_rhs = "log PIB pc"
        else:
            #y em nível: ln(y)~x com smearing e eixo Y log
            ln_y = np.log(yv)
            ols_ln = sm.OLS(ln_y, X).fit()
            b0 = float(ols_ln.params["const"]); b1 = float(ols_ln.params[x])
            sf = float(np.mean(np.exp(ols_ln.resid)))  # Duan smearing
            yhat = np.exp(b0 + b1 * xs) * sf

            ax.set_yscale("log", base=10)
            ax.scatter(xv, yv, label=f"Países (n={n})", s=pt_size, alpha=alpha, edgecolor="k", linewidth=0.2)
            ax.plot(xs, yhat, label="Tendência média(OLS)", linewidth=1.0, alpha=0.9)

            lo = None
            if n >= 4:
                try:
                    lo = lowess(yv, xv, frac=0.6, return_sorted=True)
                    ax.plot(lo[:, 0], lo[:, 1], linestyle="--", label="Tendência local (LOWESS)", linewidth=1.0, alpha=0.75, color="0.4")
                except Exception:
                    pass

            resid = ln_y - ols_ln.predict(X)
            ax.set_ylabel("PIB total (US$, eixo log)")
            title_rhs = "PIB"

        #Labels dos maiores resíduos (ou todos se poucos pontos). Melhorar legibilidade
        prefer = label_col if label_col else ("iso" if "iso" in df.columns else "country")
        resid_s = pd.Series(resid, index=df.index)
        sigma   = float(resid_s.std(ddof=1)) if len(resid_s) > 1 else float("nan")
        idx_sigma = resid_s.index[resid_s.abs() >= (2.0 * sigma)] if np.isfinite(sigma) and sigma > 0 else pd.Index([])

        q10_x, q90_x = df[x].quantile([0.10, 0.90])
        idx_xext = df[(df[x] <= q10_x) | (df[x] >= q90_x)].index

        q10_y, q90_y = df[y].quantile([0.10, 0.90])
        idx_yext = df[(df[y] <= q10_y) | (df[y] >= q90_y)].index

        idx_union = pd.Index(idx_sigma).union(idx_xext).union(idx_yext)

        if len(idx_union) == 0:
            idx_to_label = (df.index if n <= 12
                            else resid_s.abs().nlargest(min(max_labels, 10)).index)
        else:
            idx_to_label = (resid_s.loc[idx_union].abs()
                            .nlargest(min(max_labels, len(idx_union), 12)).index)

        for i in idx_to_label:
            r = df.loc[i]
            txt = self._country_label(r, prefer=prefer)
            ax.annotate(txt, (r[x], r[y]), xytext=(3, 3),
                        textcoords="offset points", fontsize=7)

        ax.set_xlabel("Educação superior (% de adultos 25+)")
        ax.set_title(f"{label}: relação educação × {title_rhs}")
        ax.legend(loc="best")
        ax.grid(True, linestyle="--", alpha=0.25)

        #Eixos + legíveis
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        if is_log_y:
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))

        #ISO -> País: mudança p/ n poluir; salva como CSV de apêndice
        if "iso" in df.columns and prefer == "iso":
            map_df = df[["iso", "country"]].dropna().drop_duplicates().sort_values("iso")
            if not map_df.empty:
                appendix = self._FIGDIR / "appendix"
                appendix.mkdir(parents=True, exist_ok=True)
                fname_map = appendix / f"{self._safe_slug(label)}_iso_country_map.csv"
                map_df.to_csv(fname_map, index=False, encoding="utf-8-sig")


        caption = ("Linha cheia: OLS em ln(PIB) c/ smearing; tracejada: LOWESS (não linearidades) "
                   "Pontos rotulados: |resíduo| ≥ 2σ ou extremos (10% inferiores/superiores).")
        plt.figtext(0.01, -0.05, caption, ha="left", va="top", fontsize=7)

        out = self._FIGDIR / fname
        plt.tight_layout(); plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
        print(f"[Plot] {out} — {'log GDP pc' if is_log_y else 'PIB em semilog'} com OLS e LOWESS.")


    def _plot_partial_corr_residuals(
        self, df: "pd.DataFrame", y: str, x: str, controls: list[str],
        label: str, fname: str, max_labels: int = 30
    ) -> None:
        """Mostra a relação entre y e x depois de descontar controls: 
        plota resíduos y contra os resíduos de x e adiciona a linha média (OLS)"""

        self._ensure_figdir()
        Xc = sm.add_constant(df[controls].astype(float))
        res_x = sm.OLS(df[x].astype(float), Xc).fit().resid
        res_y = sm.OLS(df[y].astype(float), Xc).fit().resid

        #Séries com nomes corretos
        res_x = pd.Series(res_x, index=df.index, name="res_x")
        res_y = pd.Series(res_y, index=df.index, name="res_y")

        #Resíduos e nomes das colunas
        Xr = pd.DataFrame({"const": 1.0, "res_x": res_x.astype(float).values}, index=df.index)
        ols = sm.OLS(res_y.astype(float), Xr).fit()
        b0 = float(ols.params["const"])
        b1 = float(ols.params["res_x"])

        xs = np.linspace(res_x.min(), res_x.max(), 100, dtype=float)
        ys = b0 + b1 * xs

        r, p = pearsonr(res_x, res_y)
        n = len(res_x)
        pt_size, alpha = self._point_style(n)

        plt.figure(figsize=(6, 4))
        plt.scatter(res_x, res_y, s=pt_size, alpha=alpha, label=f"Observações (resíduos; n={n})")
        plt.plot(xs, ys, label=f"Tendência média (r={r:.2f})")

        resid_abs = (res_y - ols.predict(Xr)).abs()
        for i in resid_abs.nlargest(min(max_labels, n)).index:
            plt.annotate(df.loc[i, "country"], (res_x.loc[i], res_y.loc[i]), xytext=(3, 3), textcoords="offset points", fontsize=7)

        plt.axhline(0, ls="--", alpha=0.4)

        plt.xlabel(f"Educação (%) após controls {', '.join(controls)}")
        plt.ylabel(f"PIB (log) após controls {', '.join(controls)}")
        plt.title(f"{label}: Relação após controles: {', '.join(controls)}")
        plt.legend(fontsize=8); plt.grid(True, linestyle="--", alpha=0.3)

        #Legenda explicativa
        plt.figtext(0.01, -0.05,
                   f"Associação entre educação e PIB descontando controles ({', '.join(controls)}). "
                    "Se a linha sobe, há relação além do efeito de tamanho",
                    ha="left", va="top", fontsize=8)

        out = self._FIGDIR / fname
        plt.tight_layout(); plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
        print(f"[Plot] {out} — parcial (resíduos) com OLS.")

    def _plot_ols_diagnostics(
        self,
        ols: "sm.regression.linear_model.RegressionResultsWrapper",
        df: "pd.DataFrame",
        y: str,
        x: str,
        controls: list[str],
        label: str,
        slug_prefix: str,
        max_labels: int = 30,
    ) -> None:
        """
        Influência:
        eixo X = alavancagem (quão "isolado" o país está nas variáveis explicativas)
        eixo Y = resíduo padronizado (o quão distante ficou da linha)
        tamanho do ponto = impacto no ajuste (Cook’s D)
        Ou seja, ajuda a entender países que podem gerar distorções nos resultados
        """
        self._ensure_figdir()
        idx = df.index

        infl = OLSInfluence(ols)
        lev   = pd.Series(infl.hat_matrix_diag,            index=idx)
        stud  = pd.Series(infl.resid_studentized_external, index=idx)
        cooks = pd.Series(infl.cooks_distance[0],          index=idx)

        n = len(stud)
        pt_size, alpha = self._point_style(n)
        den = cooks.replace(0, np.nan).max()
        den = float(den) if (den is not None and np.isfinite(den) and den > 0) else 1.0
        size = (200 * (cooks / den).clip(lower=0.0).fillna(0.0)) + 10

        fig = plt.figure(figsize=(6, 4))
        ax = plt.gca()
        ax.scatter(lev, stud, s=size, alpha=alpha)

        #Identificar os mais influentes
        top_idx = cooks.nlargest(min(max_labels, n)).index
        for i in top_idx:
            txt = self._country_label(df.loc[i])  # ISO preferível
            ax.annotate(txt, (lev.loc[i], stud.loc[i]), xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.axhline(0, ls="--", alpha=0.5)
        ax.set_xlabel("Alavancagem (leverage)")
        ax.set_ylabel("Resíduo studentizado")
        ax.set_title(f"{label}: Observações influentes (Cook's D)")

        #Se ISO foi usado para rótulos, mostre Iso -> Nome
        if "iso" in df.columns:
            map_df = df.loc[top_idx, ["iso", "country"]].dropna().drop_duplicates().sort_values("iso")
            if not map_df.empty:
                fig.subplots_adjust(right=0.78)
                mapping_text = "ISO -> País\n" + "\n".join(f"{r.iso}: {r.country}" for r in map_df.itertuples(index=False))
                ax.text(1.01, 0.98, mapping_text, transform=ax.transAxes, va="top", ha="left", fontsize=7,
                        bbox=dict(boxstyle="round", alpha=0.4))

        #Legendas explicativas
        plt.figtext(0.01, -0.05,
                    "Circulos maiores indicam alto Cook's D (alto impacto no ajuste). "
                    "Pontos com alta leverage e resíduo alto talvez indiquem questões de qualidade dos dados",
                    ha="left", va="top", fontsize=8)
        
        out = self._FIGDIR / f"{slug_prefix}_ols_influence.png"
        plt.tight_layout(); plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
        print(f"[Plot] {out} — alavancagem vs resíduos (Cook's D).")

    def _plot_education_quartiles(self, df_med: pd.DataFrame, *, label_root: str = "WB – quartis (mediana país)") -> None:
        """Dois painéis lado a lado: Q1 (baixo) e Q4 (alto) de educação, Y = log_gdp_pc, eixos idênticos."""
        self._ensure_figdir()
        if df_med.empty:
            print("Quartis: amostra vazia.")
            return

        q25, q75 = df_med["higher_ed_pct"].quantile([0.25, 0.75])
        df_q1 = df_med[df_med["higher_ed_pct"] <= q25].copy()
        df_q4 = df_med[df_med["higher_ed_pct"] >= q75].copy()

        # Limites comuns dos eixos
        x_all = df_med["higher_ed_pct"].astype(float)
        y_all = df_med["log_gdp_pc"].astype(float)
        xlim = (float(x_all.min()), float(x_all.max()))
        ylim = (float(y_all.min()), float(y_all.max()))

        # Helper interno: desenha um painel (usa mesma lógica do caso is_log_y em _plot_scatter_fit)
        def _panel(ax, d, title):
            xv = d["higher_ed_pct"].astype(float).to_numpy()
            yv = d["log_gdp_pc"].astype(float).to_numpy()
            n  = len(d)
            X  = sm.add_constant(d[["higher_ed_pct"]].astype(float))
            ols_lin = sm.OLS(yv, X).fit()
            b0, b1 = float(ols_lin.params["const"]), float(ols_lin.params["higher_ed_pct"])
            xs = np.linspace(xlim[0], xlim[1], 100, dtype=float)
            yhat = b0 + b1 * xs
            pt_size, alpha = self._point_style(n)

            ax.scatter(xv, yv, s=pt_size, alpha=alpha, label=f"Países (n={n})")
            ax.plot(xs, yhat, linewidth=1.5, label="Tendência média (OLS em log GDP pc)")
            if n >= 4:
                try:
                    lo = lowess(yv, xv, frac=0.6, return_sorted=True)
                    ax.plot(lo[:, 0], lo[:, 1], linestyle="--", linewidth=1.2, label="LOWESS")
                except Exception:
                    pass
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            ax.set_title(title); ax.grid(True, linestyle="--", alpha=0.3)

        fig = plt.figure(figsize=(10, 4.2))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        _panel(ax1, df_q1, f"{label_root} — Q1 (baixo)")
        _panel(ax2, df_q4, f"{label_root} — Q4 (alto)")
        for ax in (ax1, ax2):
            ax.set_xlabel("Educação superior (% de adultos 25+)")
        ax1.set_ylabel("log PIB per capita")
        ax2.legend(fontsize=8, loc="best")

        out = self._FIGDIR / "quartis_educacao_2painels_log_gdppc.png"
        plt.tight_layout(); plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
        print(f"[Plot] {out} — Q1 vs Q4 (eixos idênticos).")

    def _plot_within_dynamics(self, df_panel: pd.DataFrame, countries: list[str] | None = None) -> None:
        """Trajetória anual: X = (educação - média do país), Y = (log_gdp_pc - média do país) para BR/USA/DE."""
        self._ensure_figdir()
        if countries is None:
            countries = ["Brazil", "United States", "Germany"]

        d = df_panel.dropna(subset=["higher_ed_pct", "log_gdp_pc", "country", "year"]).copy()
        d["year"] = d["year"].astype(int)

        import unicodedata as _ud
        def _norm(s: str) -> str:
            x = str(s).strip().casefold()
            return "".join(ch for ch in _ud.normalize("NFKD", x) if not _ud.combining(ch))
        want = {_norm(c) for c in countries}
        d = d[d["country"].apply(lambda s: _norm(s) in want)].copy()
        if d.empty:
            print("Dinâmica within: países solicitados não encontrados.")
            return

        key = "iso" if "iso" in d.columns else "country"
        d["dev_ed"] = d["higher_ed_pct"] - d.groupby(key)["higher_ed_pct"].transform("mean")
        d["dev_lg"] = d["log_gdp_pc"]    - d.groupby(key)["log_gdp_pc"].transform("mean")

        fig = plt.figure(figsize=(7.6, 4.8))
        ax  = plt.gca()
        for g, sub in d.sort_values("year").groupby(key):
            ax.plot(sub["dev_ed"], sub["dev_lg"], marker="o", linewidth=1.0, label=str(g), alpha=0.9)
            # seta no último segmento para indicar direção
            if len(sub) >= 2:
                p1 = sub.iloc[-2][["dev_ed", "dev_lg"]].values
                p2 = sub.iloc[-1][["dev_ed", "dev_lg"]].values
                ax.annotate("", xy=(p2[0], p2[1]), xytext=(p1[0], p1[1]),
                            arrowprops=dict(arrowstyle="->", lw=1.0, alpha=0.8))
            first = sub.iloc[0]; last = sub.iloc[-1]
            ax.annotate(f"{int(first['year'])}", (first["dev_ed"], first["dev_lg"]),
                        xytext=(3,3), textcoords="offset points", fontsize=7)
            ax.annotate(f"{int(last['year'])}",  (last["dev_ed"],  last["dev_lg"]),
                        xytext=(3,3), textcoords="offset points", fontsize=7)

        ax.axhline(0, ls="--", alpha=0.4); ax.axvline(0, ls="--", alpha=0.4)
        ax.set_xlabel("Educação (% acima/abaixo da média do país)")
        ax.set_ylabel("log GDP pc (acima/abaixo da média do país)")
        ax.set_title("Dinâmica dentro do país: educação × log GDP pc (desvios à média)")
        ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.3)

        out = self._FIGDIR / "within_dynamics_br_usa_de.png"
        plt.tight_layout(); plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
        print(f"[Plot] {out} — dinâmicas within (BR/USA/DE).")

    def _plot_fe_bars(self, res_fe) -> None:
        """Bar‑plot dos efeitos fixos de país α̂ᵢ (com CI95% quando disponíveis)."""
        self._ensure_figdir()

        # 1) Extrai efeitos e coloca em formato simples (idx = país)
        try:
            eff = res_fe.estimated_effects
        except Exception as e:
            print(f"Efeitos fixos: não foi possível extrair estimated_effects ({e}).")
            return

        alpha = None
        # Tenta o formato "largo": colunas = tipos de efeito (alpha, lambda, ...)
        try:
            wide = eff.unstack(level=0)
            for col in ("alpha", "entity", "Alpha", "Entity", "mu"):
                if isinstance(wide, pd.DataFrame) and (col in wide.columns):
                    alpha = wide[col]
                    break
        except Exception:
            alpha = None

        # Fallback: fatiar MultiIndex no nível do tipo de efeito
        if alpha is None:
            try:
                alpha = eff.xs("alpha", level=0)
            except Exception:
                try:
                    alpha = eff["alpha"]
                except Exception:
                    if isinstance(eff, (pd.Series, pd.DataFrame)):
                        alpha = eff.squeeze()

        if alpha is None or len(alpha) == 0:
            print("Efeitos fixos: estrutura inesperada; gráfico omitido.")
            return

        alpha = pd.Series(alpha).dropna().sort_values()

        # 2) Achata o índice sem alterar dtype de MultiIndex (evita o erro)
        idx_raw = alpha.index
        if isinstance(idx_raw, pd.MultiIndex):
            lvl = "country" if ("country" in (idx_raw.names or [])) else idx_raw.names[-1]
            labels = idx_raw.get_level_values(lvl)
        else:
            labels = idx_raw
        idx = pd.Index([str(v) for v in labels], dtype=object)  # <- seguro para MultiIndex
        x = np.arange(len(alpha))

        # 3) Intervalos de confiança (se existirem)
        ci_lo = ci_hi = None
        try:
            ci = res_fe.conf_int()
            # Espera MultiIndex (tipo_de_efeito, entidade)
            ci_alpha = None
            try:
                ci_alpha = ci.xs("alpha", level=0)
            except Exception:
                if isinstance(ci, pd.DataFrame) and "alpha" in ci.columns:
                    ci_alpha = ci["alpha"]
            if ci_alpha is not None and len(ci_alpha) > 0:
                # Ordena como alpha
                ci_lo = ci_alpha.iloc[:, 0].reindex(alpha.index)
                ci_hi = ci_alpha.iloc[:, 1].reindex(alpha.index)
        except Exception:
            pass

        # 4) Destaques (funciona para ISO ou nomes de países)
        iso_hi = {"BRA", "USA", "DEU"}
        name_hi = {"brazil", "united states", "germany"}
        def _is_highlight(s: str) -> bool:
            return (s.upper() in iso_hi) or (s.strip().casefold() in name_hi)
        colors = ["tab:orange" if _is_highlight(s) else "tab:blue" for s in idx]

        # 5) Plot
        fig = plt.figure(figsize=(9.5, 5.2))
        ax  = plt.gca()
        ax.bar(x, alpha.values, color=colors, alpha=0.85)

        if (ci_lo is not None) and (ci_hi is not None):
            yerr = np.vstack([alpha.values - ci_lo.values, ci_hi.values - alpha.values])
            ax.errorbar(x, alpha.values, yerr=yerr, fmt="none", ecolor="black", elinewidth=0.8, capsize=2)

        step = max(1, len(x)//15)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(idx[::step], rotation=45, ha="right")
        ax.set_ylabel("Efeito fixo de país α̂ᵢ")
        ax.set_title("Efeitos fixos de país (FE), com BR/USA/DE destacados")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

        out = self._FIGDIR / "country_fixed_effects_bars.png"
        plt.tight_layout(); plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
        print(f"[Plot] {out} — FE por país (com CI quando disponível).")

    #Helpers para pipeline 
    def _safe_slug(self, s: str) -> str:
        return re.sub(r"[^a-z0-9_]+", "_", s.lower()).strip("_")

    def _select_country_subsets(
        self,
        df: pd.DataFrame,
        *,
        metric: str = "higher_ed_pct",
        n_small: int = 3,
        n_large: int = 10,
    ) -> dict[str, pd.DataFrame]:
        """
        Seleciona subconjuntos de países da seguinte forma:
        ordena por 'metric' (padrão: 'higher_ed_pct', % com ensino superior);
        pega bottom3, 3 próximos da mediana e top3
        pega bottom10 e top10
        inclui Brasil, EUA e Alemanha 
        """
        if metric not in df.columns:
            raise ValueError(f"Subset metric '{metric}' not found in columns: {list(df.columns)}")

        #Sem NA
        d = df.dropna(subset=[metric]).copy()
        if d.empty:
            return {}

        #Sort ascendente
        d_sorted = d.sort_values(by=metric, ascending=True).reset_index(drop=True)

        #Head e tail
        def take_head(tbl: pd.DataFrame, k: int) -> pd.DataFrame:
            return tbl.head(min(k, len(tbl)))

        def take_tail(tbl: pd.DataFrame, k: int) -> pd.DataFrame:
            return tbl.tail(min(k, len(tbl)))

        #Bottom/top 
        bottom_k = take_head(d_sorted, n_small)
        top_k    = take_tail(d_sorted, n_small)
        bottom10 = take_head(d_sorted, n_large)
        top10    = take_tail(d_sorted, n_large)

        #Pertos da mediana
        exclude_idx = set(bottom_k.index.tolist() + top_k.index.tolist())
        mid_pool = d_sorted.loc[~d_sorted.index.isin(exclude_idx)].copy()
        if not mid_pool.empty:
            med_val = float(mid_pool[metric].median())
            mid_pool["__dist__"] = (mid_pool[metric] - med_val).abs()
            mid3 = mid_pool.sort_values("__dist__").drop(columns="__dist__").head(min(n_small, len(mid_pool)))
        else:
            mid3 = pd.DataFrame(columns=d_sorted.columns)

        #BR‑USA‑DE 
        def _norm(s: str) -> str:
            x = str(s).strip().casefold()
            return "".join(ch for ch in _ud.normalize("NFKD", x) if not _ud.combining(ch))
        aliases_raw = {
            "brazil": {"brazil", "brasil"},
            "united states": {"united states", "usa", "us", "u.s.", "u.s.a.", "eua", "united states of america", "estados unidos"},
            "germany": {"germany", "alemanha", "alemania", "deutschland"},
        }
        aliases = {k: {_norm(v) for v in vals} for k, vals in aliases_raw.items()}
        def name_matches(nm: str, targets: set[str]) -> bool:
            return _norm(nm) in targets

        mask_br = df["country"].apply(lambda s: name_matches(s, aliases["brazil"]))
        mask_us = df["country"].apply(lambda s: name_matches(s, aliases["united states"]))
        mask_de = df["country"].apply(lambda s: name_matches(s, aliases["germany"]))
        br_usa_de = df.loc[mask_br | mask_us | mask_de].copy()

        subsets: dict[str, pd.DataFrame] = {}

        def add_if_ok(key: str, tbl: pd.DataFrame) -> None:
            if len(tbl) >= 2:
                subsets[key] = tbl.reset_index(drop=True)

        add_if_ok("countries_bottom3", bottom_k)
        add_if_ok("countries_mid3",    mid3)
        add_if_ok("countries_top3",    top_k)
        add_if_ok("countries_br_usa_de", br_usa_de)
        add_if_ok("countries_bottom10", bottom10)
        add_if_ok("countries_top10",    top10)

        return subsets

    def _analyze_fake_data(self, df1: pd.DataFrame, df2: pd.DataFrame) -> None:
        """
        Compara os dois conjuntos de demo (LLM1 e LLM2) da seguinte maneira:
        calcula a correlação entre % com ensino superior e log do PIB de maneira simples
        mostra médias e dispersão de PIB e de educação
        Esse seria o exercício original
        """
        self._print_header("LLM1: relação educação superior × PIB")
        if len(df1) >= 2:
            self._correlation_suite(df1, y_col="log_gdp", x_col="higher_ed_pct", label="LLM1")
        else:
            print("Amostra LLM1 insuficiente para correlação.")

        self._print_header("LLM2: relação educação superior × PIB")
        if len(df2) >= 2:
            self._correlation_suite(df2, y_col="log_gdp", x_col="higher_ed_pct", label="LLM2")
        else:
            print("Amostra LLM2 insuficiente para correlação.")

        #Comparação simples de médias e variação (para ter noção de nível e dispersão).
        self._print_header("Comparação simples LLM1 vs LLM2")
        def stats(tbl: pd.DataFrame, col: str) -> tuple[float, float]:
            v = pd.to_numeric(tbl[col], errors="coerce")
            return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan")

        m1, s1 = stats(df1, "gdp")
        m2, s2 = stats(df2, "gdp")
        e1, e2 = float(df1["higher_ed_pct"].mean()), float(df2["higher_ed_pct"].mean())
        print(f"PIB (nível): média LLM1={m1:.3g} ± {s1:.3g} | LLM2={m2:.3g} ± {s2:.3g}")
        print(f"Educação superior (%): média LLM1={e1:.2f} | LLM2={e2:.2f}")
        print("Observação: estes dados são de demo")

    def _build_views(self, df: pd.DataFrame, tag: str) -> dict[str, tuple[pd.DataFrame, str]]:
        """
        Facilita as análises
        DEMO (LLM1/LLM2): usa apenas 'countries'
        Dados reais: usa subconjuntos 
        Retorna um dict: {nome_da_visão: (DF, rótulo)}.
        """
        #DEMO
        if str(tag).upper() != "REAL":
            countries = df.copy()
            return {"countries": (countries, f"{tag}")}

        #Dados reais (usar corte transversal por país via mediana 1994–2023)
        cols = ["higher_ed_pct", "log_gdp_pc", "log_area"]
        key  = "iso" if "iso" in df.columns else "country"

        df_med = (
            df.dropna(subset=cols + ["country"])
              .groupby([key, "country"], as_index=False)
              .median(numeric_only=True)[[key, "country"] + cols]
        )

        #Conjunto completo (substituindo as análises anteriores que estavam erradas)
        return {
            "countries_all": (
                df_med.reset_index(drop=True),
                "WB – todos os países (mediana 1994–2023)"
            )
        }


    def _cluster_analysis(
        self,
        df: pd.DataFrame,
        *,
        label: str,
        features: list[str],
        k_candidates: range = range(2, 7),
    ) -> pd.DataFrame:
        """
        Clusterização KMeans sobre o conjunto completo.
        - Padroniza 'features'
        - Escolhe k por maior silhouette em k_candidates
        - Salva gráfico (dispersão higher_ed_pct × PIB com Y em log), colorido por cluster
        - Imprime resumo por cluster
        """
        self._ensure_figdir()

        # Seleção e padronização
        d = df.dropna(subset=features).copy()
        if d.empty or len(d) < 5:
            self._print_header(f"[{label}] Clusterização: amostra insuficiente.")
            return df

        X_raw = d[features].astype(float).to_numpy()
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        # Escolha de k por silhouette
        scores = []
        best_k = None
        best_labels = None
        for k in k_candidates:
            try:
                km = KMeans(n_clusters=k, n_init="auto", random_state=42)
                lbl = km.fit_predict(X)
                sc = silhouette_score(X, lbl)
                scores.append((k, sc))
                if best_k is None or sc > max(s for _, s in scores[:-1]):
                    best_k, best_labels = k, lbl
            except Exception:
                continue

        if best_k is None:
            self._print_header(f"[{label}] Clusterização: falha na seleção de k.")
            return df

        d = d.copy()
        d["cluster"] = best_labels

        # Resumo por cluster
        self._print_header(f"[{label}] Clusterização KMeans (k={best_k})")
        print("Silhouette (k ↦ score): " + ", ".join(f"{k}:{sc:.3f}" for k, sc in scores))
        summary = (
            d.groupby("cluster")
            .agg(
                n=("country", "count"),
                med_higher_ed=("higher_ed_pct", "median"),
                p25_higher_ed=("higher_ed_pct", lambda v: float(np.percentile(v, 25))),
                p75_higher_ed=("higher_ed_pct", lambda v: float(np.percentile(v, 75))),
                med_log_gdp_pc=("log_gdp_pc", "median"),
                med_log_area=("log_area", "median"),
            )
            .sort_index()
        )

        print("Resumo por cluster:")
        print(summary.to_string())

        # Plot: educação (%) × PIB (US$, log), colorindo por cluster
        xv = d["higher_ed_pct"].astype(float).to_numpy()
        yv = d["log_gdp_pc"].astype(float).to_numpy()

        n = len(d)
        pt_size, alpha = self._point_style(n)

        fig = plt.figure(figsize=(7.8, 4.8))
        ax = plt.gca()
        # y já está em log (log_gdp_pc) → eixo linear
        sc = ax.scatter(xv, yv, c=d["cluster"], s=pt_size, alpha=alpha, edgecolor="k", linewidth=0.2)

        ax.set_xlabel("Educação superior (% de adultos 25+)")
        ax.set_ylabel("log PIB per capita")

        ax.set_title(f"{label}: clusters KMeans (k={best_k})")

        ax.grid(True, linestyle="--", alpha=0.3)

        #Anota alguns países por cluster (até 3 por cluster)
        for cl in sorted(d["cluster"].unique()):
            sub = d[d["cluster"] == cl]
            cand = pd.concat([
                sub.nsmallest(1, "higher_ed_pct"),
                sub.nlargest(1, "higher_ed_pct"),
                sub.nlargest(1, "log_gdp_pc"),
            ]).drop_duplicates()
            for r in cand.itertuples():
                ax.annotate(self._country_label(pd.Series({"iso": getattr(r, "iso", ""), "country": r.country})),
                            (r.higher_ed_pct, r.log_gdp_pc),
                            xytext=(3, 3), textcoords="offset points", fontsize=7)

        out = self._FIGDIR / "countries_all_clusters_loggdppc.png"

        plt.tight_layout(); plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
        print(f"[Plot] {out} — clusters (log GDP pc).")

        return d

    def _run_grouped_pipeline(self, views: dict[str, tuple[pd.DataFrame, str]], controls: list[str]) -> None:
        for gkey, (d, label) in views.items():
            n = len(d)
            if n < self._MIN_N:
                warnings.warn(f"{label}: amostra pequena (n={n}). Resultados exploratórios.")

            # (A) Correlações + gráfico principal (precisa de pelo menos 2 pontos)
            if n >= 2:
                slug = self._safe_slug(label)

                # Escolhe DV disponível (REAL: log_gdp_pc; DEMO: log_gdp)
                y_corr = "log_gdp_pc" if "log_gdp_pc" in d.columns else ("log_gdp" if "log_gdp" in d.columns else None)
                if y_corr is None:
                    warnings.warn(f"{label}: variável dependente não encontrada para correlação.")
                else:
                    self._correlation_suite(d, y_col=y_corr, x_col="higher_ed_pct", label=label)

                #Dispersão: usar log_gdp_pc 
                y_plot = "log_gdp_pc" if "log_gdp_pc" in d.columns else "gdp"
                self._plot_scatter_fit(
                    d, y=y_plot, x="higher_ed_pct", label=label,
                    fname=f"{slug}_{gkey}_scatter.png",
                    max_labels=5,
                    label_col=("iso" if "iso" in d.columns else "country"),
                )

            # (B) Modelagem (mantém OLS + quantílica; sem controles)
            if gkey.startswith("countries") and n >= 4:
                slug = self._safe_slug(label)

                # OLS/Quantílica: mesma DV escolhida acima (prioriza log_gdp_pc)
                y_model = "log_gdp_pc" if "log_gdp_pc" in d.columns else "log_gdp"
                if y_model not in d.columns:
                    warnings.warn(f"{label}: variável dependente não encontrada para modelagem.")
                else:
                    ols = self._ols_with_diagnostics(d, y=y_model, x="higher_ed_pct", controls=[])
                    self._plot_ols_diagnostics(
                        ols, d, y=y_model, x="higher_ed_pct", controls=[],
                        label=label, slug_prefix=f"{slug}_{gkey}"
                    )
                    self._quantile_regression(d, y=y_model, x="higher_ed_pct", controls=[])

            # (C) Clusterização somente no conjunto completo
            if gkey == "countries_all" and n >= 20:
                feats = ["higher_ed_pct", "log_gdp_pc", "log_area"] if "log_gdp_pc" in d.columns else ["higher_ed_pct", "log_gdp", "log_area"]
                self._cluster_analysis(
                    d, label=label,
                    features=feats,
                    k_candidates=range(2, 7)
                )

    #Execução principal - poderíamos pensar em tests
    def executar(self) -> None:

        #Padronizando os gráficos
        self._set_matplotlib_style()
        
        #Salva os datasets fictícios em CSV c/ as colunas 
        pd.DataFrame(self._dados_llm1).to_csv(self._LLM1_CSV, index=False, encoding="utf-8-sig")
        pd.DataFrame(self._dados_llm2).to_csv(self._LLM2_CSV, index=False, encoding="utf-8-sig")

        #Preparação dos dois datasets de demo
        df1_raw = self._to_frame(self._dados_llm1)
        df2_raw = self._to_frame(self._dados_llm2)

        df1 = self._clean_and_feature(df1_raw)
        df2 = self._clean_and_feature(df2_raw)

        #Análise dos dados de demo
        self._analyze_fake_data(df1, df2)

        #Sanity checks
        if len(df1) < self._MIN_N or len(df2) < self._MIN_N:
            warnings.warn(f"Amostras muito pequenas após limpeza (n1={len(df1)}, n2={len(df2)}). "
                          "Resultados têm baixa potência; trate como exploração.")

        #Pipeline 
        controls = []

        #Análise no dados do WB
        df_real = None
        try:
            if not self._REAL_CSV.exists():
                self._print_header("Download dos dados do World Bank")
                _ = self._build_worldbank_df()

            df_real_raw = self._load_real_dataset(self._REAL_CSV)
            df_real = self._clean_and_feature(df_real_raw)

            # Views (REAL agora é corte transversal por país via mediana)
            views_real = self._build_views(df_real, tag="REAL")
            self._run_grouped_pipeline(views_real, controls=controls)

            # Quartis (Q1 vs Q4) no corte transversal (mediana por país)
            key = "iso" if "iso" in df_real.columns else "country"
            cols = ["higher_ed_pct", "log_gdp_pc", "log_area"]
            df_med = (
                df_real.dropna(subset=cols + ["country"])
                       .groupby([key, "country"], as_index=False)
                       .median(numeric_only=True)[[key, "country"] + cols]
            )
            self._plot_education_quartiles(df_med, label_root="WB – quartis educação (mediana país)")

            #Agora vamos tentar construir um painel dos últimos 30 anos: FE país + ano em log(PIB per capita) já que o corte não estava funcionando
            try:
                self._print_header("Download dos dados World Bank")
                df_panel_raw = pd.read_csv(self._REAL_CSV)
                df_panel = self._clean_and_feature(df_panel_raw)
                res_fe = self._panel_fe_analysis(df_panel)  # agora retorna o fit
                if res_fe is not None:
                    self._plot_within_dynamics(df_panel, countries=["Brazil", "United States", "Germany"])
                    self._plot_fe_bars(res_fe)
            except Exception as e:
                warnings.warn(f"Falha no painel FE: {e}")
        except Exception as e:
            warnings.warn(f"Falha ao obter/analisar REAL: {e}")

        #Discrepâncias entre versões 
        self._print_header("Discrepâncias entre versões (LLM1 vs LLM2)")
        merged = pd.merge(df1_raw, df2_raw, on="country", suffixes=("_1", "_2"))

        denom = merged["gdp_1"].replace(0, np.nan)
        discrep = (merged["gdp_2"] - merged["gdp_1"]).abs() / denom

        pct = float((discrep > 0.10).mean() * 100)
        print("Métrica: fração de unidades (países/agregados) c/ PIB2 − PIB1| / PIB1 > 10%")
        print(f"Unidades avaliadas: {len(merged)} | % com discrepância > 10%: {pct:.1f}%")

        #7) Prints diversos 
        self._print_header("Observações")
        print("Este ex10 não infere causalidade. Rodamos correlações, OLS (HC3), regressão quantílica e, com dados do WB em painel (30 anos), efeitos dos países e de ano com erros-padrão clusterizados por país.")
        print("Para causalidade, seriam necessários delineamentos experimentais ou quase‑experimentais.")
        print("De qualquer forma, fizemos uma análise extra, usando dados reais.")

#Main
def main() -> None:
    """Executa todos os exercícios"""
    exs = _RegistroExercicios.instancias_ordenadas()
    numeros = [ex.numero for ex in exs]
    try:
        escolha = input(f"Pressione Enter para executar 1..{max(numeros)} "
                        f"ou digite um número para começar a partir dele: ").strip()
    except EOFError:
        escolha = ""
    start_idx = 0
    if escolha:
        try:
            start_num = int(escolha)
            if start_num in numeros:
                start_idx = numeros.index(start_num)
            else:
                print(f"Número {start_num} não encontrado. Executando fluxo completo.")
        except ValueError:
            print(f"Entrada '{escolha}' inválida. Executando fluxo completo.")
    for ex in exs[start_idx:]:
        print(f"\nExercício {ex.numero:02d}")
        ex.executar()

if __name__ == "__main__":
    main()
