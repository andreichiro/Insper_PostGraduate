# lista_1_Andre_Katsurada.py
"""
Lista 1 — Exercícios de 1-10
Autor: André Ichiro Katsurada
Data: 06/08/25
Curso: Aprendizagem Estatística de Máquina I — INSPER
"""

from __future__ import annotations

import csv                                 
import math                                 
from abc import ABC, abstractmethod, ABCMeta      
from dataclasses import dataclass           
from pathlib import Path                   
from typing import ClassVar, Dict, List, Tuple
import re
import zipfile
import requests
import io
import warnings
import unicodedata as _ud

import pandas as pd                      

import numpy as np

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from statsmodels.nonparametric.smoothers_lowess import lowess

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler 

from scipy.stats import pearsonr, spearmanr, kendalltau, bootstrap, permutation_test

import pingouin as pg

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
        normalizado = "".join(ch.lower() for ch in texto if ch.isalnum())
        msg = "é palíndromo!" if normalizado == normalizado[::-1] \
              else "não é palíndromo!"
        print(f"“{texto}” {msg}")

class Ex08ListaCompras(Exercicio):
    """
    Processa 'supermercado.csv' e retorna:
        valor total da compra: qtd x preço, usando valores fracionários
        quantidade de produtos distintos 
        valor recalculado onde cada fração conta como 1 unidade
        produto + caro em valor unitário
    """
    numero: ClassVar[int] = 8

    _csv_path: Path = Path("supermercado.csv")

    def executar(self) -> None:
        if not self._csv_path.exists():
            print("Arquivo supermercado.csv não encontrado no path.\n"
                  "Suba o arquivo na pasta do script!")
            return

        valor_total = 0.0
        qtd_produtos = 0
        valor_recalc = 0.0
        prod_mais_caro = ("", 0.0)

        with self._csv_path.open(encoding="utf-8") as arq:
            leitor = csv.reader(arq, delimiter=";")
            for produto, qtd_str, preco_str in leitor:
                qtd = float(qtd_str.replace(",", "."))
                preco = float(preco_str.replace(",", "."))
                valor_total += qtd * preco
                qtd_produtos += 1 if qtd < 1 else int(qtd)
                valor_recalc += (1 if qtd < 1 else qtd) * preco
                if preco > prod_mais_caro[1]:
                    prod_mais_caro = (produto, preco)

        print(f"Valor total da compra: R$ {valor_total:.2f}")
        print(f"Quantidade de produtos: {qtd_produtos}")
        print("Valor total recalculado (frações arredondadas para 1 unid.): "
              f"R$ {valor_recalc:.2f}")
        print(f"Produto mais caro (unit.): {prod_mais_caro[0]} — "
              f"R$ {prod_mais_caro[1]:.2f}")


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
        #Lista de anos, customizável
        years_priority = ["2024","2023","2022","2021","2020","2019","2018"]

        #Variáveis relevantes
        edu_raw  = self.download_indicator("SE.TER.CUAT.BA.ZS") #% da populacao 25+ com diploma
        gdp_raw  = self.download_indicator("NY.GDP.MKTP.CD") #PIB total (US$ correntes)
        pop_raw  = self.download_indicator("SP.POP.TOTL") #populacao total
        area_raw = self.download_indicator("AG.LND.TOTL.K2") #area do pais

        #Helper p/ retornar valores recentes/por ano e não nulos
        def avail_years(row: pd.Series) -> set[str]:
            return {y for y in years_priority if y in row and pd.notnull(row.get(y))}

        #Lista de registros
        records: list[dict] = []
        for _, row in gdp_raw.iterrows():
            iso = row["Country Code"]

            #Filtrar por pais
            pop_row  = pop_raw[pop_raw["Country Code"] == iso]
            area_row = area_raw[area_raw["Country Code"] == iso]
            edu_row  = edu_raw[edu_raw["Country Code"] == iso]
            if pop_row.empty or area_row.empty or edu_row.empty:
                continue

            #Filtrar por ano
            yrs_gdp  = avail_years(row)
            yrs_pop  = avail_years(pop_row.iloc[0])
            yrs_area = avail_years(area_row.iloc[0])
            yrs_edu  = avail_years(edu_row.iloc[0])

            common = [y for y in years_priority if (y in yrs_gdp and y in yrs_pop and y in yrs_area and y in yrs_edu)]
            if not common:
                continue
            y = common[0]

            #Valores
            gdp_val = row.get(y)
            pop_val = pop_row.iloc[0].get(y)
            area_val = area_row.iloc[0].get(y)
            edu_val  = edu_row.iloc[0].get(y)
            if pd.isnull(gdp_val) or pd.isnull(pop_val) or pd.isnull(area_val) or pd.isnull(edu_val):
                continue

            #Consolidado
            records.append({
                "country": row["Country Name"],
                "iso": iso,
                "year": int(y),
                "gdp": float(gdp_val),
                "population": float(pop_val),
                "area": float(area_val),
                "higher_ed_pct": float(edu_val),
            })


        #DF
        df = pd.DataFrame(records)

        if not df.empty:
           df.to_csv("education_gdp_dataset.csv", index=False)
        return df

    #Aviso p/ tamanhos amostrais muito pequenos. Valor aparentemente arbitrário, pensar em como estipular algo melhor
    _MIN_N = 4

    numero: ClassVar[int] = 10

    #Dados reais
    _REAL_CSV: ClassVar[Path] = Path("education_gdp_dataset.csv")

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

    _FIGDIR: Path = Path("ex10_figs")

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

        #Remove NA
        out = out.dropna(subset=["gdp", "population", "area", "higher_ed_pct"])

        #Valida faixas
        out = out.loc[(out["higher_ed_pct"] >= 0) & (out["higher_ed_pct"] <= 100)].copy()
        out = out.loc[(out["gdp"] > 0) & (out["population"] > 0) & (out["area"] > 0)].copy()

        #Fazer log dos valores (escala)
        out["log_gdp"] = np.log(out["gdp"].astype(float))
        out["log_pop"] = np.log(out["population"].astype(float))
        out["log_area"] = np.log(out["area"].astype(float))

        #Educação superior - verificar o nome da coluna no dataset real
        out["higher_ed_pct"] = out["higher_ed_pct"].astype(float)

        out.reset_index(drop=True, inplace=True)
        return out

    @staticmethod
    def _print_header(title: str) -> None:
        print(f"\n# {title}") 

    #Análises
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
        sr, p_sr = spearmanr(x, y)

        #Permutação p-value p/ Pearson
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

        self._print_header(f"OLS (HC3): {y} ~ {x} + {controls}")
        #Sumário
        coef = float(ols.params[x])
        se = float(ols.bse[x])
        pval = float(ols.pvalues[x])
        ci_low, ci_high = map(float, ols.conf_int().loc[x])
        print(f"coef_{x}={coef:.3f}  se={se:.3f}  CI95%[{ci_low:.3f},{ci_high:.3f}]  p={pval:.4f}")

        print(f"R² adj = {ols.rsquared_adj:.3f} | F-pvalue = {ols.f_pvalue:.4f} | n = {int(ols.nobs)}")

        #Beta padronizado 
        beta_x = float(ols.params[x]) * float(df[x].std(ddof=0)) / float(df[y].std(ddof=0))
        print(f"beta padronizado (aprox.) de {x} = {beta_x:.3f}")

        #Heterocedasticidade (Breusch–Pagan)
        bp_lm, bp_lmp, bp_f, bp_fp = het_breuschpagan(ols.resid, ols.model.exog)
        print(f"Breusch–Pagan: LM p={bp_lmp:.4f}")


        #Não linearidade: RESET de Ramsey, potência 2
        try:
            if ols.df_resid >= 5 and len(ols.params) >= 2:
                reset_p = float(linear_reset(ols, use_f=True).pvalue)
                print(f"RESET (não linearidade): p={reset_p:.4f}")
            else:
                print("RESET (não linearidade): omitido (amostra muito pequena).")
        except Exception as e:
            print(f"RESET (não linearidade): omitido ({e})")

        #Normalidade dos resíduos
        jb_stat, jb_p, _, _ = jarque_bera(ols.resid)
        print(f"Jarque–Bera (normalidade dos resíduos): p={jb_p:.4f}")

        #Multicolinearidade (VIF)
        X_no_const = df[[x] + controls].astype(float)
        X_v = X_no_const.to_numpy()
        vif = [(col, variance_inflation_factor(X_v, i)) for i, col in enumerate(X_no_const.columns)]
        print("VIF:")
        for col, val in vif:
            print(f"  {col:>12}: {val:6.3f}")

        #Influência: Cook D e leverage
        infl = OLSInfluence(ols)
        cooks_d = infl.cooks_distance[0]
        leverage = infl.hat_matrix_diag

        idx_sorted = np.argsort(cooks_d)[::-1]
        top_k = int(min(5, len(df)))
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

    @staticmethod
    def _point_style(n: int) -> tuple[int, float]:
        """
        Define tamanho e transparência dos pontos em função de n.
        Retorna (size, alpha)
        """
        return 18, 0.7

    #Rótulo compacto de país
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

    def _plot_scatter_fit(self, df: "pd.DataFrame", y: str, x: str, label: str, fname: str, max_labels: int = 30, *, label_col: str | None = None,) -> None:
        """Gráfico de dispersão com Y em escala log (y~x) e duas linhas de tendência:
        Linha cheia: média prevista por um modelo simples em log c/ correção de viés (smearing)
        Linha tracejada (LOWESS): tendência local que mostra curvaturas sem impor forma fixa
        Labels: código ISO do país & nome 
        """
        self._ensure_figdir()

        #Dados em float
        xv = df[x].astype(float).to_numpy()
        yv = df[y].astype(float).to_numpy()
        n = len(df)

        #Ajuste OLS em ln(y) ~ x 
        X = pd.DataFrame({"const": 1.0, x: xv}, index=df.index)
        ln_y = np.log(yv)  # seguro porque y>0 após limpeza
        ols_ln = sm.OLS(ln_y, X).fit()
        b0 = float(ols_ln.params["const"]); b1 = float(ols_ln.params[x])

        #Curvas para desenhar 
        xs = np.linspace(np.nanmin(xv), np.nanmax(xv), 100, dtype=float)

        #Duan smearing: aproxima a média em y
        sf = float(np.mean(np.exp(ols_ln.resid)))
        yhat_semilog = np.exp(b0 + b1 * xs) * sf

        #LOWESS no espaço original (ajuda a ver não linearidades)
        frac = 0.6
        lo = lowess(yv, xv, frac=frac, return_sorted=True)

        #Aparência dos gráficos 
        pt_size, alpha = self._point_style(n)
        fig = plt.figure(figsize=(7.8, 4.8))
        ax = plt.gca()
        ax.set_yscale("log", base=10)  # unifica visões linear/log 
        ax.scatter(xv, yv, label=f"Países (n={n})", s=pt_size, alpha=alpha)
        ax.plot(xs, yhat_semilog, label="Tendência média (OLS em ln(PIB) + smearing)", linewidth=1.5)
        ax.plot(lo[:, 0], lo[:, 1], linestyle="--", label="Tendência local (LOWESS)", linewidth=1.2)

        #Labels ISO top10 e bottom10
        prefer = label_col if label_col else ("iso" if "iso" in df.columns else "country")
        
        #Se poucos pontos (≤15), rotular todos; caso contrário, rotular os maiores resíduos em ln(y)
        resid_ln = ln_y - ols_ln.predict(X)
        idx_to_label = df.index if n <= 15 else pd.Series(np.abs(resid_ln), index=df.index).nlargest(min(max_labels, n)).index
        for i in idx_to_label:
            r = df.loc[i]
            txt = self._country_label(r, prefer=prefer)
            ax.annotate(txt, (r[x], r[y]), xytext=(3, 3), textcoords="offset points", fontsize=7)

        #Labels pros gráficos
        ax.set_xlabel("Educação superior (% de adultos 25+)")
        ax.set_ylabel("PIB total (US$, eixo log)")
        ax.set_title(f"{label}: PIB vs educação (países selecionados)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)

        #Legenda ISO -> nome
        if "iso" in df.columns and prefer == "iso":

            #Se muitos pontos, mostra só os rotulados; caso contrário, todos .
            if n <= 15:
                map_df = df[["iso", "country"]].dropna().drop_duplicates().sort_values("iso")
            else:
                map_df = df.loc[idx_to_label, ["iso", "country"]].dropna().drop_duplicates().sort_values("iso")
            if not map_df.empty:
                fig.subplots_adjust(right=0.78)  # reserva espaço p/ painel lateral
                mapping_text = "ISO -> País\n" + "\n".join(f"{r.iso}: {r.country}" for r in map_df.itertuples(index=False))
                ax.text(1.01, 0.98, mapping_text, transform=ax.transAxes, va="top", ha="left", fontsize=7,
                        bbox=dict(boxstyle="round", alpha=0.4))  # caixa leve; sem cores específicas

        #Legenda explicativa 
        caption = (
            "O eixo Y em log melhora a leitura entre países\n"
            "A linha cheia aproxima E[PIB|x] via smearing. A LOWESS revela possíveis não linearidades."
        )
        plt.figtext(0.01, -0.05, caption, ha="left", va="top", fontsize=8)



        out = self._FIGDIR / fname
        plt.tight_layout()
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Plot] {out} — semi-log (unificada) com OLS(back‑transform) e LOWESS.")

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
        size = (200 * (cooks / cooks.replace(0, np.nan).max()).clip(lower=0.0).fillna(0.0)) + 10

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
        print("Observação: estes dados são de demonstração; resultados não inferem causalidade.")

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

        #Dados reais
        subsets = self._select_country_subsets(df, metric="higher_ed_pct")

        #Rótulos
        labeled: dict[str, tuple[pd.DataFrame, str]] = {}
        labels_human = {
            "countries_bottom3":  "WB – 3 piores (educação superior %)",
            "countries_mid3":     "WB – 3 medianos (educação superior %)",
            "countries_top3":     "WB – 3 melhores (educação superior %)",
            "countries_br_usa_de":"WB – Brasil, USA e Alemanha",
            "countries_bottom10": "WB – bottom 10 (educação superior %)",
            "countries_top10":    "WB – top 10 (educação superior %)",
        }
        for k, dsub in subsets.items():
            labeled[k] = (dsub, labels_human.get(k, f"REAL – {k}"))

        return labeled

    def _run_grouped_pipeline(self, views: dict[str, tuple[pd.DataFrame, str]], controls: list[str]) -> None:

        for gkey, (d, label) in views.items():
            n = len(d)
            if n < self._MIN_N:
                warnings.warn(f"{label}: amostra pequena (n={n}). Resultados exploratórios.")

            # (A) Correlações (sempre que houver ≥2 pontos)
            if n >= 2:
                slug = self._safe_slug(label)

                # (ÚNICA) correlação principal em ln(PIB) ~ educação%; gráfico semi‑log unificado
                self._correlation_suite(d, y_col="log_gdp", x_col="higher_ed_pct", label=label)
                self._plot_scatter_fit(
                    d, y="gdp", x="higher_ed_pct", label=label,
                    fname=f"{slug}_{gkey}_scatter_semilog.png",
                    max_labels=12,
                    label_col=("iso" if "iso" in d.columns else "country"),  # ISO melhora legibilidade em top/bottom10
                )

            # (B) Modelagem (qualquer subset countries, n≥4)
            if gkey.startswith("countries") and n >= 4:
                slug = self._safe_slug(label)

                # 3) PARCIAL — imprime e gera gráfico (resíduos)
                self._partial_correlation(d, y="log_gdp", x="higher_ed_pct", controls=controls)
                self._plot_partial_corr_residuals(
                    d, y="log_gdp", x="higher_ed_pct", controls=controls,
                    label=label, fname=f"{slug}_{gkey}_partial_residuals.png"
                )

                # 4) OLS + DIAGNÓSTICOS — imprime e gera 4 gráficos
                ols = self._ols_with_diagnostics(d, y="log_gdp", x="higher_ed_pct", controls=controls)
                self._plot_ols_diagnostics(
                    ols, d, y="log_gdp", x="higher_ed_pct", controls=controls,
                    label=label, slug_prefix=f"{slug}_{gkey}"
                )

                #5) Regressão por quantis
                self._quantile_regression(d, y="log_gdp", x="higher_ed_pct", controls=controls)

    #Execução principal - poderíamos pensar em tests
    def executar(self) -> None:
        import warnings
        import pandas as pd

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
        controls = ["log_pop"]

        #Análise no dados do WB
        df_real = None
        try:
            if not self._REAL_CSV.exists():
                self._print_header("Download dos dados do World Bank")
                _ = self._build_worldbank_df()
            df_real_raw = self._load_real_dataset(self._REAL_CSV)
            df_real = self._clean_and_feature(df_real_raw)
            views_real = self._build_views(df_real, tag="REAL")
            self._run_grouped_pipeline(views_real, controls=controls)
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
        print("Não inferimos causalidade")
        print("O que fizemos: controle de confundidor (log_pop), correlação parcial, diagnósticos,")
        print("OLS robusto e quantílica. Para causalidade, seria necessário painel temporal, eventos/quase-experimentos")
        print("ou IV com instrumento plausível (ex.: reformas educacionais exógenas).")

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
