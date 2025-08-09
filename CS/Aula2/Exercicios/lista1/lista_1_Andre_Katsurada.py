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
    numero: ClassVar[int] = 10

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

    _saida1: Path = Path("llm1.csv")
    _saida2: Path = Path("llm2.csv")
    _AGG_LABELS = {"OCDE", "G20", "América do Sul"}  
    _MIN_N = 8  #Aviso p/ tamanhos amostrais muito pequenos
    _REGION_MAP = {
            "Brasil": "América do Sul",
            "México": "América do Norte",
            "EUA": "América do Norte",
            "Alemanha": "Europa",
            "França": "Europa",
        }
    _FIGDIR: Path = Path("ex10_figs")

    #Helpers internos 
    def _gravar_csv(self, dados: dict, destino: Path) -> None:
        pd.DataFrame(dados).to_csv(destino, index=False, encoding="utf-8")

    @staticmethod
    def _corr_gdp_superior(caminho: Path) -> float:
        df = pd.read_csv(caminho)
        return df["GDP"].corr(df["Superior (%)"])

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

    def _clean_and_feature(self, df: "pd.DataFrame") -> "pd.DataFrame":
        out = df.copy()

        #Remove NA
        out = out.dropna(subset=["gdp", "population", "area", "higher_ed_pct"])

        #Valida faixas
        out = out.loc[(out["higher_ed_pct"] >= 0) & (out["higher_ed_pct"] <= 100)].copy()
        out = out.loc[(out["gdp"] > 0) & (out["population"] > 0) & (out["area"] > 0)].copy()

        #tipo de unidade: país vs agregado (UE, OCDE, G20, América do Sul etc)
        out["unit_type"] = np.where(out["country"].isin(self._AGG_LABELS),
                                    "provided_aggregate", "country")

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

    #Helper para gerar agregados de regiões 
    def _compute_region_aggregates(self, df_countries: "pd.DataFrame") -> "pd.DataFrame":
        """
        Agrega países -> regiões (calculadas).
        Preferência: se existir coluna 'region' no df, agrupar por ela.
        Caso contrário, usar o fallback _REGION_MAP (apenas para demonstração).
        GDP/área/pop: soma ; higher_ed_pct: média ponderada por população.
        """
        if df_countries.empty:
            return df_countries.copy()

        work = df_countries.copy()

        if "region" in work.columns:
            work["region_calc"] = work["region"]
        else:
            # Fallback apenas para demo — no dataset real, seria uma coluna 'region' ou algo assim
            work["region_calc"] = work["country"].map(self._REGION_MAP)

        work = work.dropna(subset=["region_calc"])
        if work.empty:
            return work

        g = work.groupby("region_calc", as_index=False).apply(
            lambda gdf: pd.Series({
                "country": f"{gdf.name} (calc)",  # rótulo diferenciado do agregado “fornecido”
                "gdp": gdf["gdp"].sum(),
                "area": gdf["area"].sum(),
                "population": gdf["population"].sum(),
                "higher_ed_pct": np.average(gdf["higher_ed_pct"], weights=gdf["population"]),
            }),
            include_groups=False
        ).reset_index(drop=True)

        g["unit_type"] = "computed_aggregate"
        g["log_gdp"] = np.log(g["gdp"])
        g["log_pop"] = np.log(g["population"])
        g["log_area"] = np.log(g["area"])
        return g[["country", "gdp", "area", "population", "higher_ed_pct",
                  "log_gdp", "log_pop", "log_area", "unit_type"]]

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
    def _bootstrap_corr_ci(x, y, B: int = 10_000, seed: int = 123) -> tuple[float, float] | tuple[float, float, bool]:
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

        #Permutação (two-sided) apenas se não for constante
        if not (x_const or y_const):
            rng = np.random.default_rng(123)
            obs = pr
            n_perm = 10_000
            count = 0
            for _ in range(n_perm):
                perm = rng.permutation(y)
                r_perm, _ = pearsonr(x, perm)
                if abs(r_perm) >= abs(obs):
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
        print(f"β padronizado de {x} = {beta_x:.3f}  (aprox. variação em σ_y por 1 σ_x)")

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
        vif = []
        X_no_const = df[[x] + controls].copy().astype(float)
        X_v = sm.add_constant(X_no_const).to_numpy()
        for i, col in enumerate(["const", x] + controls):
            if col == "const":
                continue
            vif.append((col, variance_inflation_factor(X_v, i)))
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

    def _maybe_iv_2sls(self, df: "pd.DataFrame", y: str, x: str, controls: list[str]) -> None:
        """
        Se houver coluna 'instrument', tenta 2SLS (só para dados reais com instrumento plausível).
        """
        if "instrument" not in df.columns:
            print("\n[IV-2SLS] Nenhum instrumento fornecido; saltando.")
            return

        Z = sm.add_constant(df[["instrument"] + controls])
        X_endog = df[[x]]
        Y = df[y]
        try:
            iv = IV2SLS(Y, X_endog, Z).fit()
            self._print_header(f"IV-2SLS: {y} ~ {x} (instrumento=instrument) + {controls}")
            print(iv.summary())
        except Exception as e:
            print(f"[IV-2SLS] Falhou ao ajustar: {e}")

    #Helpers p/ gráficos 
    def _ensure_figdir(self) -> None:
        self._FIGDIR.mkdir(parents=True, exist_ok=True)

    def _plot_scatter_fit(self, df: "pd.DataFrame", y: str, x: str, label: str, fname: str) -> None:
        """Dispersion y~x + OLS simples + LOWESS; anota países/agregados."""

        self._ensure_figdir()
        X = sm.add_constant(df[x].astype(float))
        yv = df[y].astype(float)
        ols = sm.OLS(yv, X).fit()
        xs = np.linspace(df[x].min(), df[x].max(), 100)
        ys = ols.params["const"] + ols.params[x] * xs
        lo = lowess(yv, df[x], frac=0.8, return_sorted=True)

        # Figura de dispersão com duas linhas de tendência
        plt.figure(figsize=(6, 4))
        plt.scatter(df[x], df[y], color="black", label="Observações")
        plt.plot(xs, ys, color="tab:blue", label="Ajuste OLS")
        plt.plot(lo[:, 0], lo[:, 1], color="tab:orange", linestyle="--", label="Curva LOWESS")

        for _, r in df.iterrows():
            plt.annotate(r["country"], (r[x], r[y]), xytext=(3, 3), textcoords="offset points", fontsize=8)

        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f"{label}: {y} vs {x}")
        plt.legend(loc="best", fontsize=8)
        plt.grid(True, linestyle="--", alpha=0.3)

        out = self._FIGDIR / fname
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"[Plot] {out} — Interpretação: OLS (linear) vs LOWESS (padrões não lineares).")

    def _plot_partial_reg(self, df: "pd.DataFrame", y: str, x: str, controls: list[str], label: str, fname: str) -> None:
        """Gráfico de regressão parcial (residualização)."""
        self._ensure_figdir()
        Xc = sm.add_constant(df[controls])
        ry = sm.OLS(df[y], Xc).fit().resid
        rx = sm.OLS(df[x], Xc).fit().resid

        rx = pd.Series(rx, name=x)
        X = sm.add_constant(rx)
        fit = sm.OLS(ry, X).fit()
        xs = np.linspace(rx.min(), rx.max(), 100)
        ys = fit.params["const"] + fit.params[x] * xs

        plt.figure(figsize=(6, 4))
        plt.scatter(rx, ry, color="black", label="Resíduos")
        plt.plot(xs, ys, color="tab:blue", label="Linha de regressão")
        plt.axhline(0, color="grey", linestyle="--", lw=0.7)
        plt.axvline(0, color="grey", linestyle="--", lw=0.7)
        plt.xlabel(f"{x} (residualizado)")
        plt.ylabel(f"{y} (residualizado)")
        plt.title(f"{label}: regressão parcial")
        plt.legend(loc="best", fontsize=8)
        plt.grid(True, linestyle="--", alpha=0.3)

        out = self._FIGDIR / fname
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"[Plot] {out} — Interpretação: inclinação ≈ efeito de {x} após controles.")

    def _plot_residuals_and_qq(self, model, label: str, basefname: str) -> None:
        """Resíduos vs ajustados e QQ-plot."""
        self._ensure_figdir()
        # Resíduos vs ajustados
        plt.figure(figsize=(6, 4))
        plt.scatter(model.fittedvalues, model.resid, color="black")
        plt.axhline(0, color="grey", linestyle="--", lw=0.7)
        plt.xlabel("Ajustados")
        plt.ylabel("Resíduos")
        plt.title(f"{label}: Resíduos vs Ajustados")
        plt.grid(True, linestyle="--", alpha=0.3)
        out1 = self._FIGDIR / f"{basefname}_resid_fitted.png"
        plt.tight_layout()
        plt.savefig(out1, dpi=150)
        plt.close()
        print(f"[Plot] {out1} — Interpretação: padrões sugerem hetero/não linearidade se presentes.")

        # QQ plot
        fig = sm.qqplot(model.resid, line="45")
        fig.suptitle(f"{label}: QQ-plot dos resíduos")
        out2 = self._FIGDIR / f"{basefname}_qq.png"
        fig.savefig(out2, dpi=150)
        print(f"[Plot] {out2} — Interpretação: desvios indicam não-normalidade residual.")
        plt.close(fig)
    #Helpers para pipeline 
    def _safe_slug(self, s: str) -> str:
        return re.sub(r"[^a-z0-9_]+", "_", s.lower()).strip("_")

    def _build_views(self, df: pd.DataFrame, tag: str) -> dict[str, tuple[pd.DataFrame, str]]:
        """
        Retorna 'views' canônicas p/ análise:
        - countries: somente países
        - provagg: agregados fornecidos (OCDE/G20/etc.)
        - regcalc: agregados calculados a partir de países (dinâmico por 'region' quando disponível)
        """
        countries = df.loc[df["unit_type"] == "country"].copy()
        provagg   = df.loc[df["unit_type"] == "provided_aggregate"].copy()
        regcalc   = self._compute_region_aggregates(countries)
        return {
            "countries": (countries, f"{tag} (países)"),
            "provagg":   (provagg,   f"{tag} (agregados fornecidos)"),
            "regcalc":   (regcalc,   f"{tag} (regiões calculadas)"),
        }

    def _run_grouped_pipeline(self, views: dict[str, tuple[pd.DataFrame, str]], controls: list[str]) -> None:
        import warnings as _warnings

        for gkey, (d, label) in views.items():
            n = len(d)
            if n < self._MIN_N:
                _warnings.warn(f"{label}: amostra pequena (n={n}). Resultados exploratórios.")

            # (A) Correlações (sempre que houver ≥2 pontos)
            if n >= 2:
                self._correlation_suite(d, y_col="gdp",     x_col="higher_ed_pct", label=label)
                self._correlation_suite(d, y_col="log_gdp", x_col="higher_ed_pct", label=label)
                self._plot_scatter_fit(d, y="log_gdp", x="higher_ed_pct",
                                       label=label, fname=f"{self._safe_slug(label)}_{gkey}_scatter.png")

            # (B) Modelagem (somente países, n≥4)
            if gkey == "countries" and n >= 4:
                self._partial_correlation(d, y="log_gdp", x="higher_ed_pct", controls=controls)
                ols = self._ols_with_diagnostics(d, y="log_gdp", x="higher_ed_pct", controls=controls)
                self._quantile_regression(d, y="log_gdp", x="higher_ed_pct", controls=controls)
                self._plot_partial_reg(d, y="log_gdp", x="higher_ed_pct",
                                       controls=controls, label=label, fname=f"{self._safe_slug(label)}_{gkey}_partial.png")
                self._plot_residuals_and_qq(ols, label=label, basefname=f"{self._safe_slug(label)}_{gkey}_diag")
                self._maybe_iv_2sls(d, y="log_gdp", x="higher_ed_pct", controls=controls)

    #Execução principal - poderíamos pensar em tests
    def executar(self) -> None:
        import warnings
        import pandas as pd

        #Preparação dos dois datasets de demo
        df1_raw = self._to_frame(self._dados_llm1)
        df2_raw = self._to_frame(self._dados_llm2)

        df1 = self._clean_and_feature(df1_raw)
        df2 = self._clean_and_feature(df2_raw)

        #Sanity checks
        if len(df1) < self._MIN_N or len(df2) < self._MIN_N:
            warnings.warn(f"Amostras muito pequenas após limpeza (n1={len(df1)}, n2={len(df2)}). "
                          "Resultados têm baixa potência; trate como exploração.")

        #Pipeline 
        controls = ["log_pop", "log_area"]
        views1 = self._build_views(df1, tag="LLM1")
        views2 = self._build_views(df2, tag="LLM2")

        self._run_grouped_pipeline(views1, controls=controls)
        self._run_grouped_pipeline(views2, controls=controls)

        #Discrepâncias entre versões 
        self._print_header("Discrepâncias entre versões (LLM1 vs LLM2)")
        merged = pd.merge(df1_raw, df2_raw, on="country", suffixes=("_1", "_2"))
        discrep = (merged["gdp_2"] - merged["gdp_1"]).abs() / merged["gdp_1"].replace(0, pd.NA)
        pct = float((discrep > 0.10).mean() * 100)
        print("Métrica: fração de unidades (países/agregados) com |GDP₂ − GDP₁| / GDP₁ > 10%")
        print(f"Unidades avaliadas: {len(merged)} | % com discrepância > 10%: {pct:.1f}%")

        #Nota sobre os agregados
        self._print_header("Nota: países vs agregados")
        print("Agregados (p.ex., OCDE, G20) combinam múltiplos países e podem mascarar heterogeneidade.")
        print("Resultados para agregados são analisados separadamente e, quando possível, com regiões calculadas a partir de países.")

        #7) Prints diversos 
        self._print_header("Observação sobre causalidade")
        print("Com dados de corte transversal e sem instrumento exógeno válido, não inferimos causalidade.")
        print("O que fizemos: controle de confundidores (log_pop, log_area), correlação parcial, diagnósticos,")
        print("OLS robusto e quantílica. Para causalidade, será necessário painel temporal, eventos/quase-experimentos")
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
