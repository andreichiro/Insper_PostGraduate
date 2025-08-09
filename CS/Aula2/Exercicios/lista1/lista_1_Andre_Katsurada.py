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
from typing import ClassVar, Dict, List    
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
from statsmodels.sandbox.regression.gmm import IV2SLS 
from statsmodels.nonparametric.smoothers_lowess import lowess

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler 

from scipy.stats import kendalltau, bootstrap, pearsonr, spearmanr

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
    # arquivos temporários exigidos pelo enunciado (gerar CSV e ler com pandas)

    nome: str = "André Ichiro"
    idade: int = 28

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
    Análise de correlação e regressão entre GDP (PIB agregado) e % de ensino superior.
    Projetado para dados demo e para migração futura a um dataset real, onde podemos pensar em GDP per capita.

    Pipeline:
    10a) Higienização e features (log-transforma GDP/pop/área, valida faixas)
    10b) Correlações Pearson/Spearman/Kendall + IC bootstrap + p permutação 
    10c) Regressão OLS com HC3 controlando por log_pop e log_area; betas padronizados; CIs bootstrap
    10d) Diagnósticos BP/White, RESET, JB, DW, multicolinearidade (VIF) e influência (Cook/leverage)
    10e) Robustez: regressão quantílica (mediana)
    10f) Opcional: 2SLS se existir coluna "instrument" no dataset e a implementação estiver disponível
    """

    def download_indicator(indicator_code: str) -> pd.DataFrame:
        """
        Faz download de um indicador do Banco Mundial e devolve o CSV como DataFrame.
        """
        url = f"https://api.worldbank.org/v2/en/indicator/{indicator_code}?downloadformat=csv"
        resp = requests.get(url)
        resp.raise_for_status()
        # lê o ZIP em memória e encontra o arquivo de dados
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            data_file = [name for name in zf.namelist()
                        if name.startswith("API_") and name.endswith(".csv")][0]
            df = pd.read_csv(zf.open(data_file), skiprows=4)
        return df

    # baixa cada indicador
    edu_raw  = download_indicator("SE.TER.CUAT.BA.ZS")  # % da população 25+ com diploma de bacharel ou equivalente:contentReference[oaicite:0]{index=0}
    gdp_raw  = download_indicator("NY.GDP.PCAP.CD")     # PIB per capita (US$ correntes):contentReference[oaicite:1]{index=1}
    pop_raw  = download_indicator("SP.POP.TOTL")        # População total:contentReference[oaicite:2]{index=2}
    area_raw = download_indicator("AG.LND.TOTL.K2")     # Área terrestre (km²):contentReference[oaicite:3]{index=3}

    # função auxiliar: retorna o valor mais recente não nulo a partir de uma lista de anos
    def latest_value(row: pd.Series, years: list[str]):
        for y in years:
            v = row.get(y)
            if pd.notnull(v):
                return v
        return None

    # define a lista de anos a procurar – do mais recente para trás
    years_priority = ["2024","2023","2022","2021","2020","2019","2018"]

    records = []
    for _, row in gdp_raw.iterrows():
        iso = row["Country Code"]
        # procura o valor mais recente de PIB per capita
        gdp_pc = latest_value(row, years_priority)
        if gdp_pc is None:
            continue
        # procura população, área e percentagem de ensino superior
        pop_row  = pop_raw[pop_raw["Country Code"] == iso]
        area_row = area_raw[area_raw["Country Code"] == iso]
        edu_row  = edu_raw[edu_raw["Country Code"] == iso]
        if pop_row.empty or area_row.empty or edu_row.empty:
            continue
        pop_val  = latest_value(pop_row.iloc[0], years_priority)
        area_val = latest_value(area_row.iloc[0], years_priority)
        edu_val  = latest_value(edu_row.iloc[0], years_priority)
        if pop_val is None or area_val is None or edu_val is None:
            continue
        # cria o registro consolidado
        records.append({
            "country": row["Country Name"],
            "iso": iso,
            "gdp_pc": float(gdp_pc),
            "population": float(pop_val),
            "area": float(area_val),
            "higher_ed_pct": float(edu_val)
        })

    df = pd.DataFrame(records)
    # calcula o PIB total (PIB per capita × população)
    df["gdp"] = df["gdp_pc"] * df["population"]
    # ordena e salva
    df.to_csv("education_gdp_dataset.csv", index=False)
    # Removido print de preview do DataFrame para evitar poluição de saída

    #Aviso p/ tamanhos amostrais muito pequenos
    _MIN_N = 8  

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
        Normaliza o dataset REAL para o esquema interno:
        requer colunas: country, population, area, higher_ed_pct e gdp (ou gdp_pc).
        Retorna um DataFrame *raw* (pré-clean), com colunas canônicas.
        """
        if isinstance(source, (str, Path)):
            real = pd.read_csv(source)
        else:
            real = source.copy()

        req = {"country", "population", "area", "higher_ed_pct"}
        missing = req - set(real.columns)
        if missing:
            raise ValueError(f"Dataset REAL faltando colunas obrigatórias: {sorted(missing)}")

        if "gdp" not in real.columns:
            if "gdp_pc" not in real.columns:
                raise ValueError("Dataset REAL precisa ter 'gdp' ou 'gdp_pc'.")
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

        # (Opcional futuro) GDP per capita quando escala real estiver definida. Algo assim:
        # out["gdp_pc"] = out["gdp"] * 1e6 / out["population"]  # ~ mil unidades per capita se GDP ~ trilhões e pop ~ milhões

        out.reset_index(drop=True, inplace=True)
        return out

    @staticmethod
    def _print_header(title: str) -> None:
        print(f"\n# {title}") 

    #Helpers p/ estatística
    @staticmethod
    def _pearson_ci_fisher(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
        """
        IC aproximado via transformação de Fisher z (estável para n pequeno).
        Clipa em [-1, 1].
        """
        if n < 4 or not np.isfinite(r):
            return float("nan"), float("nan")
        z = np.arctanh(max(min(r, 0.999999999), -0.999999999))
        se = 1.0 / (n - 3) ** 0.5
        z_lo = z - 1.96 * se
        z_hi = z + 1.96 * se
        return float(np.tanh(z_lo)), float(np.tanh(z_hi))

    @staticmethod
    def _bootstrap_corr_ci(x, y, B: int = 10_000, seed: int = 123) -> tuple[float, float, bool]:
        """
        Bootstrap simples para Pearson r com descarte de reamostras degeneradas.
        Retorna (low, high, ok_flag). Se ok_flag=False, usar Fisher como fallback.
        """
        rng = np.random.default_rng(seed)
        n = len(x)
        rs = []
        for _ in range(B):
            idx = rng.integers(0, n, n)
            xv = x[idx]
            yv = y[idx]
            #Evita vetores constantes que tornam r indefinido
            if (xv == xv[0]).all() or (yv == yv[0]).all():
                continue
            r, _ = pearsonr(xv, yv)
            if np.isfinite(r):
                rs.append(r)
        if len(rs) < max(100, int(0.1 * B)):  #Instável
            return float("nan"), float("nan"), False
        rs = np.sort(np.asarray(rs))
        lo = float(rs[int(0.025 * len(rs))])
        hi = float(rs[int(0.975 * len(rs))])
        #Clip de segurança
        lo = max(lo, -1.0)
        hi = min(hi, 1.0)
        return lo, hi, True

    #Análises
    def _correlation_suite(self, df: "pd.DataFrame", y_col: str, x_col: str, label: str) -> None:
        import numpy as np

        y = df[y_col].to_numpy(dtype=float)
        x = df[x_col].to_numpy(dtype=float)

        n = len(x)

        x_const = np.allclose(x, x[0])
        y_const = np.allclose(y, y[0])

        if x_const or y_const:
            pr, p_pr = (float("nan"), float("nan"))
        else:
            pr, p_pr = pearsonr(x, y)

        sr, p_sr = spearmanr(x, y)  #Lida melhor c/ empates 
        kr, p_kr = kendalltau(x, y)

        #IC para Pearson c/ fallback p/ Fisher se bootstrap ficar instável
        lo_b, hi_b, ok = self._bootstrap_corr_ci(x, y)
        if ok:
            ci_str = f"[{lo_b:.3f}, {hi_b:.3f}] (bootstrap)"
        else:
            lo_f, hi_f = self._pearson_ci_fisher(pr, n)
            ci_str = f"[{lo_f:.3f}, {hi_f:.3f}] (Fisher z)"

        #Permutação (two-sided): exato se n<=9; caso contrário MC 10k
        if not (x_const or y_const):
            obs = abs(pr)
            if n <= 9:
                from itertools import permutations
                count = 0; total = 0
                for perm in permutations(y, n):
                    r_perm, _ = pearsonr(x, np.asarray(perm))
                    if abs(r_perm) >= obs:
                        count += 1
                    total += 1
                p_perm = count / total
            else:
                rng = np.random.default_rng(123)
                n_perm = 10_000
                count = 0
                for _ in range(n_perm):
                    r_perm, _ = pearsonr(x, rng.permutation(y))
                    if abs(r_perm) >= obs:
                        count += 1
                p_perm = (count + 1) / (n_perm + 1)
        else:
            p_perm = float("nan")

        self._print_header(f"[{label}] Correlações {y_col} ~ {x_col}  (n={n})")
        print(f"Pearson r = {pr:.3f}  (IC95% {ci_str})  p={p_pr:.4f} | p_perm≈{p_perm:.4f}")
        print(f"Spearman ρ = {sr:.3f}  p={p_sr:.4f}")
        print(f"Kendall τ = {kr:.3f}   p={p_kr:.4f}")

    def _partial_correlation(self, df: "pd.DataFrame", y: str, x: str, controls: list[str]) -> None:
        """
        Correlação parcial de y~x controlando 'controls' por residualização OLS.
        """

        Xc = sm.add_constant(df[controls])
        resid_x = sm.OLS(df[x], Xc).fit().resid
        resid_y = sm.OLS(df[y], Xc).fit().resid
        r, p = pearsonr(resid_x, resid_y)

        self._print_header(f"Correlação parcial: {y} ~ {x} | controls={controls}")
        print(f"r_parcial = {r:.3f}  p≈{p:.4f}  (obtida via residualização OLS)")

    def _ols_with_diagnostics(self, df: "pd.DataFrame", y: str, x: str, controls: list[str]) -> "sm.regression.linear_model.RegressionResultsWrapper":
        """
        OLS com HC3; diagnósticos: BP/White, RESET, JB, DW; VIF; influência.
        Retorna o modelo ajustado
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

        #Betas padronizados (z scale)
        scaler = StandardScaler()
        X_std = pd.DataFrame(scaler.fit_transform(df[[x] + controls]), columns=[x] + controls)
        X_std = sm.add_constant(X_std)
        ols_std = sm.OLS((df[y] - df[y].mean()) / df[y].std(ddof=0), X_std).fit()
        beta_x = ols_std.params[x]
        print(f"beta padronizado (não-HC3) de {x} = {beta_x:.3f}")

        #CIs bootstrap para coeficiente principal
        rng = np.random.default_rng(123)
        B = 5000
        coefs = np.empty(B)
        n = len(df)
        X_np = X.to_numpy(dtype=float)
        for b in range(B):
            idx = rng.integers(0, n, n)
            coefs[b] = sm.OLS(yv[idx], X_np[idx]).fit().params[1]  # posição 1 => coef de x (após const)
        ci_low, ci_high = np.percentile(coefs, [2.5, 97.5])
        print(f"IC95% bootstrap para coef de {x}: [{ci_low:.4f}, {ci_high:.4f}]")

        #Heterocedasticidade
        bp_lm, bp_lmp, bp_f, bp_fp = het_breuschpagan(ols.resid, ols.model.exog)
        w_lm, w_lmp, w_f, w_fp = het_white(ols.resid, ols.model.exog)
        print(f"Breusch–Pagan: LM p={bp_lmp:.4f} | White: LM p={w_lmp:.4f}")

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

        #Autocorrelação
        dw = sm.stats.stattools.durbin_watson(ols.resid)
        print(f"Durbin–Watson: {dw:.3f}")

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
        thresh = 4 / ols.nobs
        flagged = [(i, cooks_d[i], leverage[i], df.loc[i, "country"])
                   for i in range(len(df)) if cooks_d[i] > thresh]
        if flagged:
            print(f"Pontos influentes (Cook's D > {thresh:.3f}):")
            for i, cd, lev, name in flagged:
                print(f"  idx={i:02d}  CookD={cd:.3f}  leverage={lev:.3f}  país={name}")
        else:
            print("Nenhum ponto influente significativo por Cook's D.")

        return ols

    def _quantile_regression(self, df: "pd.DataFrame", y: str, x: str, controls: list[str]) -> None:
        """
        Regressão quantílica (mediana, 0.5) p/ a outliers na dependente.
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
        if n < 100:
            return 20, 0.8
        if n < 300:
            return 10, 0.6
        return 6, 0.4

    #Rótulo compacto de país
    @staticmethod
    def _country_label(row: pd.Series, prefer: str = "iso") -> str:
        """
        Retorna um rótulo curto e legível para anotar o ponto.
        Prioriza 'iso' (3 letras); fallback para nome truncado.
        """
        if prefer in row and pd.notna(row[prefer]) and str(row[prefer]).strip():
            return str(row[prefer]).upper()
        nm = str(row.get("country", ""))
        return (nm[:12] + "…") if len(nm) > 13 else nm

    def _plot_scatter_fit(self, df: "pd.DataFrame", y: str, x: str, label: str, fname: str, max_labels: int = 30, *, label_col: str | None = None,) -> None:
        """Dispersão SEMI-LOG (eixo y em log10) de y~x + OLS (ajuste em ln(y)) + LOWESS.
        - Linha cheia: E[ln(y)|x] retransformada para y (exp(β0 + β1 x)).
        - Tracejado: LOWESS em escala original de y.
        - Rótulos: ISO se disponível; caso contrário, nome curto do país."""
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
        yhat_semilog = np.exp(b0 + b1 * xs)  # back-transform

        #LOWESS no espaço original (ajuda a ver não linearidades)
        frac = float(min(0.8, max(0.4, 6 / max(n, 1))))
        lo = lowess(yv, xv, frac=frac, return_sorted=True)

        #Aparência dos gráficos 
        pt_size, alpha = self._point_style(n)
        fig = plt.figure(figsize=(7.8, 4.8))
        ax = plt.gca()
        ax.set_yscale("log", base=10)  # unifica visões linear/log num gráfico legível
        ax.scatter(xv, yv, label=f"Países (n={n})", s=pt_size, alpha=alpha)
        ax.plot(xs, yhat_semilog, label="Tendência média (OLS em ln(PIB))", linewidth=1.5)
        ax.plot(lo[:, 0], lo[:, 1], linestyle="--", label="Tendência local (LOWESS)", linewidth=1.2)

        #Labels ISO top10 e bottom10
        prefer = label_col if label_col else ("iso" if "iso" in df.columns else "country")
        # Se poucos pontos (≤15), rotular todos; caso contrário, rotular os maiores resíduos em ln(y)
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
            "A linha cheia resume a tendência média (efeito percentual em PIB). A LOWESS revela possíveis não linearidades."
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
        """Correlação parcial: gráfico de resíduos (y|controls) vs (x|controls) + linha OLS."""

        self._ensure_figdir()
        Xc = sm.add_constant(df[controls].astype(float))
        res_x = sm.OLS(df[x].astype(float), Xc).fit().resid
        res_y = sm.OLS(df[y].astype(float), Xc).fit().resid

        # Ensure Series with names and aligned index
        res_x = pd.Series(res_x, index=df.index, name="res_x")
        res_y = pd.Series(res_y, index=df.index, name="res_y")

        # Regress residuals with explicit column names
        Xr = pd.DataFrame({"const": 1.0, "res_x": res_x.astype(float).values}, index=df.index)
        ols = sm.OLS(res_y.astype(float), Xr).fit()
        b0 = float(ols.params["const"])
        b1 = float(ols.params["res_x"])

        xs = np.linspace(res_x.min(), res_x.max(), 100, dtype=float)
        ys = b0 + b1 * xs

        r, p = pearsonr(res_x, res_y)
        n = len(res_x)
        pt_size = 20 if n < 100 else 10 if n < 300 else 6
        alpha = 0.8 if n < 100 else 0.6 if n < 300 else 0.4

        plt.figure(figsize=(6, 4))
        plt.scatter(res_x, res_y, s=pt_size, alpha=alpha, label=f"Observações (resíduos; n={n})")
        plt.plot(xs, ys, label=f"Tendência média (r={r:.2f})")

        resid_abs = (res_y - ols.predict(Xr)).abs()
        for i in resid_abs.nlargest(min(max_labels, n)).index:
            plt.annotate(df.loc[i, "country"], (res_x.loc[i], res_y.loc[i]), xytext=(3, 3), textcoords="offset points", fontsize=7)

        plt.axhline(0, ls="--", alpha=0.4)

        plt.xlabel(f"Educação (%) após controls {', '.join(controls)}")
        plt.ylabel(f"PIB (log) após controls {', '.join(controls)}")
        plt.title(f"{label}: Relação após controles de tamanho e área")
        plt.legend(fontsize=8); plt.grid(True, linestyle="--", alpha=0.3)

        #Legenda explicativa
        plt.figtext(0.01, -0.05,
                    "Mostra a associação entre educação e PIB descontando população e área. "
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
        Diagnóstico gráfico: apenas Influência (leverage × resíduo studentizado; tamanho ∝ Cook's D).
        QQ/resíduos-ajustados e VIF em gráfico foram removidos por baixa utilidade/legibilidade.
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

    def _plot_qr_vs_ols_residualized(
        self, df: "pd.DataFrame", y: str, x: str, controls: list[str],
        label: str, fname: str, quantile: float = 0.5, max_labels: int = 30
    ) -> None:
        """Compara QR(τ) e OLS no espaço residualizado (controls aplicados)."""
        self._ensure_figdir()
        Xc = sm.add_constant(df[controls].astype(float))
        res_x = sm.OLS(df[x].astype(float), Xc).fit().resid
        res_y = sm.OLS(df[y].astype(float), Xc).fit().resid

        res_x = pd.Series(res_x, index=df.index, name="res_x")
        res_y = pd.Series(res_y, index=df.index, name="res_y")

        Xr = pd.DataFrame({"const": 1.0, "res_x": res_x.astype(float).values}, index=df.index)
        ols = sm.OLS(res_y.astype(float), Xr).fit()
        qreg = sm.QuantReg(res_y.astype(float), Xr).fit(q=quantile)

        xs = np.linspace(res_x.min(), res_x.max(), 100, dtype=float)
        Xgrid = pd.DataFrame({"const": 1.0, "res_x": xs})
        y_ols = ols.predict(Xgrid)
        y_qr  = qreg.predict(Xgrid)

        n = len(res_x)
        pt_size = 20 if n < 100 else 10 if n < 300 else 6
        alpha = 0.8 if n < 100 else 0.6 if n < 300 else 0.4

        plt.figure(figsize=(6, 4))

        plt.scatter(res_x, res_y, s=pt_size, alpha=alpha, label="Observações (resíduos)")
        plt.plot(xs, y_ols, label=f"Média (OLS, Beta={ols.params['res_x']:.3f})")
        plt.plot(xs, y_qr, linestyle="--", label=f"Mediana (QR τ={quantile:.1f}, Beta={qreg.params['res_x']:.3f})")

        resid_abs = (res_y - ols.predict(Xr)).abs()
        for i in resid_abs.nlargest(min(max_labels, n)).index:
            plt.annotate(df.loc[i, "country"], (res_x.loc[i], res_y.loc[i]), xytext=(3, 3), textcoords="offset points", fontsize=7)

        plt.axhline(0, ls="--", alpha=0.5)

        plt.xlabel(f"Educação (%) após controle")
        plt.ylabel(f"PIB (log) após controle")
        plt.title(f"{label}: Mediana vs média após controle (robustez)")
        plt.legend(fontsize=8); plt.grid(True, linestyle="--", alpha=0.3)
        plt.figtext(0.01, -0.05,
                    "Compara efeitos na média (OLS) e na mediana (QR). "
                    "Diferenças relevantes sugerem sensibilidade a outliers/assimetria.",
                    ha="left", va="top", fontsize=8)

        out = self._FIGDIR / fname
        plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()
        print(f"[Plot] {out} — OLS vs QR em espaço residualizado.")

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
        Create smaller, readable subsets for plotting on REAL data only.
        Subsets:
          - top3 / mid3 / bottom3     (by `metric`; mid3 = nearest to median excluding top/bottom)
          - br_usa_de                 (Brazil, USA, Germany; robust to common aliases)
          - top10 / bottom10          (by `metric`)
        Notes:
          - Only returns subsets with at least 2 rows (plots/correlations need ≥2).
          - `metric` defaults to 'higher_ed_pct' so subsets are aligned to education share.
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
        Provide a compact comparison between the two demo datasets and
        analyze the higher_ed_pct ↔ GDP relationship for each.
        Reuses existing correlation suite for consistency.
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

        # Simple, interpretable comparison of central tendencies and spreads.
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
        Returns canonical 'views' for analysis.

        For DEMO (LLM1/LLM2): keep a single 'countries' view (small and readable already).
        For REAL: restrict to smaller, readable subsets only:
          - bottom3 / mid3 / top3  (by higher_ed_pct)
          - BR+USA+DE
          - bottom10 / top10       (by higher_ed_pct)
        """
        # DEMO datasets remain a single compact view.
        if str(tag).upper() != "REAL":
            countries = df.copy()
            return {"countries": (countries, f"{tag}")}

        # REAL dataset: generate targeted subsets by education (%).
        subsets = self._select_country_subsets(df, metric="higher_ed_pct")

        # Label each subset clearly (used in file names and plot titles).
        labeled: dict[str, tuple[pd.DataFrame, str]] = {}
        labels_human = {
            "countries_bottom3":  "REAL – 3 piores (educação superior %)",
            "countries_mid3":     "REAL – 3 medianos (educação superior %)",
            "countries_top3":     "REAL – 3 melhores (educação superior %)",
            "countries_br_usa_de":"REAL – Brasil, USA e Alemanha",
            "countries_bottom10": "REAL – bottom 10 (educação superior %)",
            "countries_top10":    "REAL – top 10 (educação superior %)",
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

                # 5) QUANTÍLICA — imprime e gera gráfico vs OLS em base residualizada
                self._quantile_regression(d, y="log_gdp", x="higher_ed_pct", controls=controls)
                self._plot_qr_vs_ols_residualized(
                    d, y="log_gdp", x="higher_ed_pct", controls=controls,
                    label=label, fname=f"{slug}_{gkey}_qr_vs_ols.png", quantile=0.5
                )

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
        controls = ["log_pop", "log_area"]
        views1 = self._build_views(df1, tag="LLM1")
        views2 = self._build_views(df2, tag="LLM2")

        # Dados reais — roda TODA a análise (mesmo pipeline) e gera gráficos
        df_real = None
        if self._REAL_CSV.exists():
            try:
                df_real_raw = self._load_real_dataset(self._REAL_CSV)
                df_real = self._clean_and_feature(df_real_raw)
                views_real = self._build_views(df_real, tag="REAL")
                self._run_grouped_pipeline(views_real, controls=controls)
            except Exception as e:
                warnings.warn(f"Falha ao carregar/analisar REAL: {e}")
        else:
            warnings.warn(f"Arquivo '{self._REAL_CSV}' não encontrado; pulando dataset REAL.")

        #Discrepâncias entre versões 
        self._print_header("Discrepâncias entre versões (LLM1 vs LLM2)")
        merged = pd.merge(df1_raw, df2_raw, on="country", suffixes=("_1", "_2"))

        denom = merged["gdp_1"].replace(0, np.nan)
        discrep = (merged["gdp_2"] - merged["gdp_1"]).abs() / denom

        pct = float((discrep > 0.10).mean() * 100)
        print("Métrica: fração de unidades (países/agregados) com |GDP₂ − GDP₁| / GDP₁ > 10%")
        print(f"Unidades avaliadas: {len(merged)} | % com discrepância > 10%: {pct:.1f}%")

        #7) Prints diversos 
        self._print_header("Observações")
        print("Não inferimos causalidade")
        print("O que fizemos: controle de confundidores (log_pop, log_area), correlação parcial, diagnósticos,")
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
