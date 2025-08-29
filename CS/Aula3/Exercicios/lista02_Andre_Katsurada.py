"""
Mini‑teste — Readmissões em Hospitais Medicare — Lista 02
Autor: André Ichiro Katsurada
Data: 22/08/2025
Curso: Programa Avançado em Data Science e Decisão, Computação para a Ciência de Dados, INSPER

- Q1–Q5: cria números e gráficos para entender readmissões
- Q6: mostra os estados que mais/menos se destacam dentro de cada tipo de medida
- Q7: calcula faixas de incerteza por estado usando um modelo que "puxa" os números para perto da média do país
- Q8: usa um corte de probabilidade que limita falsos alarmes em comparações c/ mtos estados
- Q9: traz + estabilidade para estados onde há poucos casos ("puxa" para a média)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple
import sys

import numpy as np

import pandas as pd
from pandas.api.types import CategoricalDtype 

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from scipy.stats import pearsonr, norm
from scipy.special import logit, expit

import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.meta_analysis import combine_effects
from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.stats.multitest import fdrcorrection

import bambi as bmb

import arviz as az

import logging

# P/ logar
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
logging.getLogger("pymc.stats.convergence").setLevel(logging.CRITICAL)

@dataclass(slots=True, frozen=True)
class Config:
    """Configurações padrão do projeto (n. de amostras, figuras, links, paths e dirs)
    """
    DATA_URL: str = "https://raw.githubusercontent.com/amkaris/EDA/master/cms_hospital_readmissions.csv"
    BASE_DIR: Path = Path(__file__).resolve().parent
    FIG_DIR: Path = Path(__file__).resolve().parent / "lista02_figs"
    CLEAN_CSV: Path = Path(__file__).resolve().parent / "hospital_clean.csv"

    # Q5
    Q5_AGG_MODE: str = "median"
    Q5_MIN_N: int = 1000

    # Q6
    Q6_MIN_STATE_MEASURE_N: int = 500
    Q6_TOPK: int = 5

    # Q8 (Bayes-FDR) 
    FDR_ALPHA: float = 0.05
    FDR_TEST_VALUE: float = 1.0     #usado p/ comparação
    FDR_MIN_STATE_N: int = 1000

    # Q9 (plot opcional do resultado Bayes)
    EB_PLOT: bool = True

    # Bayes (Bambi/PyMC) 
    BAYES_TUNE: int = 1000
    BAYES_DRAWS: int = 500
    BAYES_CHAINS: int = 4
    BAYES_SEED: int = 20250813
    BAYES_TARGET_ACCEPT: float = 0.95
    BAYES_MAX_RHAT: float = 1.01
    BAYES_MIN_ESS: int = 400

CFG = Config()
CFG.FIG_DIR.mkdir(parents=True, exist_ok=True)

# Estilo dos gráficos
def _set_matplotlib_style() -> None:
    """Gráficos minimamente padronizados: grades, fontes, tamanhos e bordas"""
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

def add_takeaway(ax, text: str, loc: str = "upper left", *, fontsize: int = 8) -> None:
    """Pequena interpretação de cada gráfico"""
    ha = "left" if "left" in loc else "right"
    va = "top" if "upper" in loc else "bottom"
    x = 0.01 if ha == "left" else 0.99
    y = 0.98 if va == "top" else 0.02
    ax.text(
        x, y, text,
        transform=ax.transAxes, ha=ha, va=va,
        fontsize=fontsize,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="none")
    )

# Classes p/ registro/execução
class _RegistroExercicios:
    """Registro e execução dos exercícios (Q1–Q9).

    Execução na main()
    """
    _mapa: Dict[int, "Exercicio"] = {}

    @classmethod
    def registrar(cls, numero: int, classe: "Exercicio") -> None:
        existente = cls._mapa.get(numero)
        if existente and existente.__qualname__ != classe.__qualname__:
            raise ValueError(f"Exercício {numero} já registrado por {existente}")
        cls._mapa[numero] = classe

    @classmethod
    def instancias_ordenadas(cls) -> List["Exercicio"]:
        return [cls._mapa[i]() for i in sorted(cls._mapa)]

class _ExercicioMeta(type):
    """Metaclasse que registra automaticamente as classes dos exercícios (precisa ter ex com número)"""
    def __new__(mcls, name, bases, ns, **kwargs):
        cls = super().__new__(mcls, name, bases, ns)
        numero = ns.get("numero")
        if numero is not None:
            _RegistroExercicios.registrar(numero, cls)
        return cls

class Exercicio(metaclass=_ExercicioMeta):
    """Classe base pras atividades (Q1–Q9).

    Cada questão é uma unidade modular c/ assinatura em comum
    """
    numero: ClassVar[int]
    def executar(self, ctx: "Contexto") -> None:
        raise NotImplementedError

@dataclass(slots=True)
class Contexto:
    """Contexto compartilhado entre exercícios (inicialmente o DataFrame limpo).

    A ideia é carregar/limpar só uma vez e reaproveitar 
    """
    df: pd.DataFrame

# Carregamento & Limpeza
MIN_COLS = [
    # essenciais para Q1–Q5
    "State", "Measure Name",
    "Number of Discharges", "Number of Readmissions",
    "Predicted Readmission Rate", "Expected Readmission Rate",
    "Excess Readmission Ratio",
]
OPTIONAL_COLS = [
    #extras pros demais exercícios 
    "Hospital Name", "Provider Number", "Start Date", "End Date",
]

def _to_num(s: pd.Series) -> pd.Series:
    """Converte coluna para número, removendo sinais de “%” e espaços; onde não der, vira vazio (NaN)."""
    return pd.to_numeric(s.astype(str).str.rstrip("%").str.strip(), errors="coerce")

#Ajusta escala das taxas p/ todas ficarem em "%" qdo vierem fração
def _harmonize_rate_units(df: pd.DataFrame) -> pd.DataFrame:
    """Garante que as taxas “Prevista” e “Esperada” fiquem na mesma escala (0–100)."""
    for col in ["Predicted Readmission Rate", "Expected Readmission Rate"]:
        if col not in df.columns:
            continue
        v = _to_num(df[col])
        if v.notna().any():
            frac_share = np.mean((v <= 1.5) & np.isfinite(v))
            df[col] = (v * 100.0) if frac_share >= 0.95 else v
        else:
            df[col] = v
    return df

def carregar_higienizar() -> pd.DataFrame:
    """Chama URL do Config e higieniza"""
    #1) Carrega o csv
    try:
        raw = pd.read_csv(CFG.DATA_URL)
    except Exception as e:
        raise RuntimeError(f"Falha ao ler URL {CFG.DATA_URL}: {e}") from e

    #2) Verifica a presença das cols minimas
    keep_cols = [c for c in (MIN_COLS + OPTIONAL_COLS) if c in raw.columns]
    must = set(MIN_COLS)

    if not must.issubset(set(raw.columns)):
        raise ValueError("Dataset não contém as colunas mínimas exigidas para Q1–Q5.")

    #3) Cópia imutável do bruto 
    df = raw.copy()

    #4) Normaliza valores ausentes (de not available p/ nan)
    df.replace({"Not Available": np.nan, "Not Available ": np.nan, "": np.nan}, inplace=True)

    #5) Converte string c/ numeros p/ numérico usando to_num
    for c in ["Number of Discharges", "Number of Readmissions",
              "Predicted Readmission Rate", "Expected Readmission Rate",
              "Excess Readmission Ratio"]:
        df[c] = _to_num(df[c])

    #6) Aplica a escala em % 
    df = _harmonize_rate_units(df)

    #7) Converte datas de string p/ datetime
    for c in ("Start Date", "End Date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    #8) Q1: taxa observada = 100 * (Readmissions/Discharges). Fix da divisão por zero
    with np.errstate(divide="ignore", invalid="ignore"):
        den = df["Number of Discharges"].replace({0: np.nan})
        df["True Admission Rate"] = (df["Number of Readmissions"] / den) * 100.0

    #9) Q2: Diferença OBS e Expected em pontos percentuais
    df["Diff True_minus_Expected"] = df["True Admission Rate"] - df["Expected Readmission Rate"]

    # 10) Remove linhas que n tem cols fundamentais
    df = df.dropna(subset=["Excess Readmission Ratio", "State", "Measure Name"]).copy()

    # 11) Labels de Estado e Medida como categorias 
    df["State"] = df["State"].astype("category")
    if "Measure" not in df.columns:
        df["Measure"] = df["Measure Name"].astype("category")

    #12) Salva versão 'clean' em csv
    _safe_to_csv(
            df[keep_cols + ["True Admission Rate", "Diff True_minus_Expected"]],
            CFG.CLEAN_CSV,
            desc="[Q1] CSV 'clean' salvo",
            index=False,
            encoding="utf-8-sig",
        )
    return df

# Helpers 
def _write_summary_markdown(fig_dir: Path = CFG.FIG_DIR, filename: str = "lista02_resumo.md") -> Path:
    """
    Gera um arquivo Markdown com a interpretação dos 5 gráficos e
    links embutidos para as imagens salvas em `fig_dir`.
    Retorna o Path do arquivo criado.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)

    md_lines = [
        "# Readmissões em Hospitais Medicare — Lista 02 (Resumo)",
        "",
        "As figuras citadas abaixo estão nesta mesma pasta. Clique para abrir.",
        "",
        "---",
        "",
        "## 1) Difference: Observed Readmission Rate − Expected Readmission Rate",
        "![Difference — boxplot](q2_box_true_minus_expected.png)",
        "A maioria dos hospitais está muito perto do zero (diferença pequena).",
        "",
        "A mediana fica levemente abaixo de zero e a média levemente acima: ou seja, no geral o esperado e o observado batem com pequenas diferenças.",
        "",
        "Existem poucos hospitais bem acima (+15 a +25 p.p.) e alguns bem abaixo (até uns −13 p.p.). Essas bolinhas são exceções.",
        "",
        "No geral, o que se esperava está alinhado com o que aconteceu. Há alguns casos extremos (para melhor e para pior) que merecem análise individual.",
        "",
        "---",
        "",
        "## 2) Confiabilidade: Observado vs Esperado (por decil)",
        "![Reliability — deciles](q3_reliability_observed_vs_expected.png)",
        "Os pontos ficam quase colados na diagonal. Há leve tendência do observado ficar um pouco acima do esperado nas faixas mais altas, mas é um desvio pequeno.",
        "",
        "---",
        "",
        "## 3) Excess Readmission Ratio × Number of Discharges",
        "![ERR × volume — scatter](q4_scatter_err_vs_discharges.png)",
        "",
        "Hospitais pequenos espalham mais (pontos altos e baixos).",
        "",
        "Hospitais grandes ficam mais perto do esperado.",
        "",
        "Há alguns pontos laranja tanto acima quanto abaixo, inclusive com volumes médios/grandes.",
        "",
        "Ou seja, nos hospitais com poucos casos a medida oscila mais. Já em volumes maiores, sair muito de 1 chama mais atenção, já que ali a oscilação normal é bem menor.",
        "",
        "---",
        "",
        "## 4) Funnel plot — ERR × volume (faixas 95% e 99,8%)",
        "![Funnel plot](q4_funnel_err_vs_discharges.png)",
        "Maioria dos hospitais dentro da faixa de 95%.",
        "",
        "Poucos saem para fora das faixas de 99,8%, esses são os que mais valem investigação.",
        "",
        "Quando o volume é médio/alto, aí a chance de ser só 'sorte ou azar' é pequena, então valeria investigar",
        "",
        "---",
        "",
        "## 5) ERR por Estado — Top 5 vs Bottom 5 (EB‑selected)",
        "![Top/Bottom por estado](q5_box_err_by_state_top_bottom.png)",
        "Melhores (à esquerda): MT, SD, UT, ID, CO. Em geral abaixo de 1, com o grosso dos hospitais desses estados performando melhor que o esperado.",
        "",
        "Piores (à direita): MD, KY, NJ, WV, NY. Em geral acima de 1, com mais hospitais pior que o esperado; há outliers altos (alguns bem acima de 1,3–1,4).",
        "",
        "A altura das caixas e das “antenas” mostra a variação dentro do estado: alguns têm dispersão maior (mais desigualdade entre hospitais).",
        "",
        "Há tendências por estado: um grupo consistentemente melhor e outro pior que o esperado. As diferenças não são gigantes, mas há outliers e estados com vários hospitais acima de 1 que talvez precisem de mais investigação. Do outro lado, estados à esquerda que podem ter coisas tipo boas práticas para serem entendidas e talvez copiadas.",
        "",
    ]

    md_path = fig_dir / filename
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[MD] Resumo salvo em: {md_path}")
    return md_path


def _posterior_state_draws(model, idata) -> np.ndarray:
    """P/ cada estado, calcula um conjunto de probabilidades simuladas.

    Ideia: vai da média geral e faz cada estado "andar" um pouco
    p/ cima/baixo, devolvendo várias simulações por estado
    Será q essa abordagem faz sentido na prática?
    """
    post = idata.posterior

    # Junta cadeias e rodadas em um só eixo (várias simulações)
    intercept = np.asarray(post["Intercept"])
    intercept = intercept.reshape(intercept.shape[0] * intercept.shape[1], 1)

    # Ajuste por estado. Não estava funcionando, então refiz o reshape manualmente
    if "1|State" in post:
        re = np.asarray(post["1|State"])                      #[C, R, S]
        re = re.reshape(re.shape[0] * re.shape[1], re.shape[2])  #[D, S]
        eta = intercept + re
        return expit(eta)                                      #[D, S]

    # Alternativa: quando o ajuste vem separado em tamanho × deslocamento
    if "1|State_offset" in post and "1|State_sigma" in post:
        offset = np.asarray(post["1|State_offset"])            # [C, R, S]
        sigma  = np.asarray(post["1|State_sigma"])             # [C, R]
        D = offset.shape[0] * offset.shape[1]
        S = offset.shape[2]
        offset = offset.reshape(D, S)                          # [D, S]
        sigma  = sigma.reshape(D, 1)                           # [D, 1]
        eta = intercept + sigma * offset
        return expit(eta)                                      # [D, S]

    raise KeyError(
        "Posterior não contém '1|State' nem ('1|State_offset' & '1|State_sigma'). "
        f"Variáveis disponíveis: {list(post.data_vars)}"
    )


def _posterior_p_draws(model, idata, data: pd.DataFrame, *, predictive: bool = False) -> np.ndarray:
    """Pega, para cada linha da tabela, as simulações do estado.
    Pensar em alternativas pras simulações
    RRetorna matriz [amostras, linhas].
    """
    data = _align_categories_like_model(data.copy(), model)
    mdata = getattr(model, "data", None)
    if not (isinstance(mdata, pd.DataFrame) and "State" in mdata.columns and isinstance(mdata["State"].dtype, CategoricalDtype)):
        raise RuntimeError("Model.data precisa expor 'State' categórico com categorias ordenadas.")

    states_cat = mdata["State"].cat.categories
    codes = pd.Categorical(data["State"], categories=states_cat).codes
    if (codes < 0).any():
        unknown = sorted(set(data["State"].astype(str)) - set(states_cat.astype(str)))
        raise ValueError(f"States não vistos no fit: {unknown}")

    p_state = _posterior_state_draws(model, idata)  # [draws, S]
    return p_state[:, codes]                        # [draws, n_obs]

def _posterior_state_standardized_p(df: pd.DataFrame, *, model, idata) -> Dict[str, np.ndarray]:
    """Devolve um dict {Estado: simulações de probabilidade}."""
    states = [str(s) for s in model.data["State"].cat.categories]
    p_state = _posterior_state_draws(model, idata)
    return {s: p_state[:, i] for i, s in enumerate(states)}


def _bambi_fit(model, *, target_accept: float, max_treedepth: int):
    """Ajusta o modelo de probabilidade com as configurações definidas.
    Rodar o modelo usando a Config
    Nesse caso, o modelo é https://github.com/bambinos/bambi
    É possível reaproveitar o código e pensar em outros modelos/libraries
    """
    return model.fit(
        nuts_sampler="pymc",
        nuts={"max_treedepth": max_treedepth},
        tune=CFG.BAYES_TUNE,
        draws=CFG.BAYES_DRAWS,
        chains=CFG.BAYES_CHAINS,
        target_accept=target_accept,
        random_seed=CFG.BAYES_SEED,
        init="jitter+adapt_diag",
    )

def _safe_to_csv(df: pd.DataFrame, path: Path, *, desc: str, index: bool = False, encoding: str = "utf-8-sig") -> None:
    """Salva a tabela em CSV ou logga o erro
    """
    try:
        df.to_csv(path, index=index, encoding=encoding)
        print(f"{desc}: {path}")
    except Exception as e:
        logger.warning(f"Falha ao salvar {desc} em {path}: {e}")

def _print_convergence_brief(tag: str, idata) -> None:
    """Mostra um resumo de qualidade das simulações """
    try:
        rhat = float(az.rhat(idata).to_array().max())
        ess  = float(az.ess(idata, method="bulk").to_array().min())
        div  = int(np.asarray(getattr(idata.sample_stats, "diverging", 0)).sum())
        print(f"[{tag}] r̂_max={rhat:.3f} | ESS_bulk_min={ess:.0f} | divergences={div}")
    except Exception:
        pass

def _eb_shrinkage_random_effects(m: np.ndarray, v: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Calcula o qto vai ser puxado p/ a média nacional
    Devolve: média geral, força do "puxão" e, para cada estado, o quanto foi puxado.
    """
    m = np.asarray(m, float)
    v = np.asarray(v, float)
    eps = 1e-12
    ok = np.isfinite(m) & np.isfinite(v) & (v > eps)
    if not ok.any():
        return (np.nan, np.nan, np.full_like(m, np.nan, dtype=float))

    m_ok, v_ok = m[ok], v[ok]
    w_fe = 1.0 / v_ok
    mu_fe = float(np.sum(w_fe * m_ok) / np.sum(w_fe))

    res = combine_effects(effect=m_ok, variance=v_ok, method_re="dl") 
    tau2 = float(max(res.tau2, 0.0))

    B_ok = v_ok / (v_ok + tau2)
    B = np.full_like(m, np.nan, dtype=float)
    B[ok] = B_ok
    return mu_fe, tau2, B

def _fast_state_standardized_rate_eb(df: pd.DataFrame) -> pd.DataFrame:
    """Estima a taxa de readmissão por estado ajustada pelo 'mix' de medidas do país,
    puxando p/ a média onde os números são mais instáveis.

    Retorna estado, taxa ajustada, incerteza e intervalo (baixo/alto).
    """
    d = _ensure_measure_column(df).dropna(
        subset=["State", "Number of Readmissions", "Number of Discharges", "Measure"]
    ).copy()
    d["y"] = d["Number of Readmissions"].astype(float)
    d["n"] = d["Number of Discharges"].astype(float)

    # médias nacionais com pesos p/ ser proporcional a discharges
    w_m = _measure_weights_national(d)

    per_measure_rows = []
    for meas, g in d.groupby("Measure", observed=True):
        gg = (g.groupby("State", observed=True)
                .agg(y=("y", "sum"), n=("n", "sum"))
                .reset_index())
        if gg.empty:
            continue
        mu0 = float(gg["y"].sum() / gg["n"].sum())
        v_i = mu0 * (1.0 - mu0) / np.clip(gg["n"].to_numpy(float), 1.0, np.inf)
        p_hat = (gg["y"] / gg["n"]).to_numpy(float)
        _, tau2, B = _eb_shrinkage_random_effects(p_hat, v_i)
        p_shrunk = (1.0 - B) * p_hat + B * mu0
        # posterior variance (approx)
        v_post = (v_i * tau2) / (v_i + tau2) if tau2 > 0 else np.zeros_like(v_i)

        per_measure_rows.append(pd.DataFrame({
            "Measure": str(meas),
            "State": gg["State"].astype(str).to_numpy(),
            "p_shrunk": p_shrunk,
            "v_post": v_post
        }))

    if not per_measure_rows:
        return pd.DataFrame(columns=["State", "p_std", "var_p_std", "lo", "hi"])

    tab = pd.concat(per_measure_rows, ignore_index=True)

    tab["w"] = tab["Measure"].map(lambda me: w_m.get(str(me), 0.0)).astype(float)
    agg = (tab.groupby("State", observed=True)
              .agg(p_std=("p_shrunk", lambda v: float(np.sum(tab.loc[v.index, "w"] * v))),
                   var_p_std=("v_post", lambda v: float(np.sum((tab.loc[v.index, "w"] ** 2) * v)))))
    agg = agg.reset_index()
    z = 1.96
    agg["lo"] = agg["p_std"] - z * np.sqrt(np.clip(agg["var_p_std"], 0.0, np.inf))
    agg["hi"] = agg["p_std"] + z * np.sqrt(np.clip(agg["var_p_std"], 0.0, np.inf))
    return agg[["State", "p_std", "var_p_std", "lo", "hi"]]

class _BayesCache:
    """Cache do output do modelo, essa etapa estava demorando muito"""
    model_by_meas: Dict[Optional[str], tuple] = {}

def _assert_bambi() -> None:
    """Garante que as libraries necessárias estão lá"""
    if (bmb is None) or (az is None):
        raise RuntimeError("Bambi/ArviZ are required for the single-path Bayesian workflow.")

def _prep_binomial_df(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Prepara as colunas: reinternações (y) e altas (n) e checa se existe um id do hospital.

    Retorna a tabela preparada e um sim/não dizendo se temos esse ID
    """
    sub = df.dropna(subset=["Number of Readmissions", "Number of Discharges", "State"]).copy()
    sub["y"] = sub["Number of Readmissions"].astype(int)
    sub["n"] = sub["Number of Discharges"].astype(int)
    sub = sub[sub["n"] > 0]
    sub = _ensure_measure_column(sub)
    fac_col = _get_facility_id_col(sub)
    has_fac = fac_col is not None and sub[fac_col].notna().any()
    if has_fac:
        sub["fac"] = sub[fac_col].astype(str)
    return sub, has_fac

def _fit_bayes_hlm(sub: pd.DataFrame, *, measure: Optional[str]) -> tuple:
    """Ajusta o modelo de simulação por estado"""
    _assert_bambi()
    key = None if measure is None else str(measure)
    if key in _BayesCache.model_by_meas:
        return _BayesCache.model_by_meas[key]

    data, _ = _prep_binomial_df(sub)
    data["State"] = data["State"].astype("category")

    p0 = float(np.clip(data["y"].sum() / data["n"].sum(), 1e-4, 1 - 1e-4))
    mu0 = float(np.log(p0 / (1 - p0)))

    prior_ctor = bmb.Prior if hasattr(bmb, "Prior") else bmb.priors.Prior
    priors = {
        "Intercept": prior_ctor("Normal", mu=mu0, sigma=0.6),
        "State|sd": prior_ctor("HalfNormal", sigma=0.3),
    }

    model = bmb.Model("p(y, n) ~ 1 + (1|State)", data=data, family="binomial", priors=priors, noncentered=True)
    idata = _bambi_fit(model, target_accept=max(CFG.BAYES_TARGET_ACCEPT, 0.99), max_treedepth=14)

    _BayesCache.model_by_meas[key] = (model, idata, data)
    return model, idata, data

def _fit_bayes_hlm_joint(df: pd.DataFrame) -> tuple:
    """Ajusta o modelo por estado juntando todos os tipos de medida"""
    _assert_bambi()
    key = "__STATE_ONLY__"
    if key in _BayesCache.model_by_meas:
        return _BayesCache.model_by_meas[key]

    data, _ = _prep_binomial_df(df)
    # Agrega por estado
    data = (data.groupby(["State"], observed=True)
                .agg(y=("y", "sum"), n=("n", "sum"))
                .reset_index())
    data["State"] = data["State"].astype("category")

    p0 = float(np.clip(data["y"].sum() / data["n"].sum(), 1e-4, 1 - 1e-4))
    mu0 = float(np.log(p0 / (1 - p0)))

    prior_ctor = bmb.Prior if hasattr(bmb, "Prior") else bmb.priors.Prior
    priors = {
        # ajustar aqui com CV seria possível
        "Intercept":  prior_ctor("Normal",     mu=mu0, sigma=0.5),
        "State|sd":   prior_ctor("HalfNormal", sigma=0.2),
    }

    model = bmb.Model("p(y, n) ~ 1 + (1|State)", data=data, family="binomial",
                      priors=priors, noncentered=True)
    idata = _bambi_fit(model, target_accept=CFG.BAYES_TARGET_ACCEPT, max_treedepth=12)

    _BayesCache.model_by_meas[key] = (model, idata, data)
    return model, idata, data

def _measure_weights_national(df: pd.DataFrame) -> Dict[str, float]:
    """Calcula quanto cada medida pesa no país (proporção de altas)."""
    d = _ensure_measure_column(df)
    w = d.groupby("Measure", observed=True)["Number of Discharges"].sum().astype(float)
    w = w / w.sum()
    return {str(k): float(v) for k, v in w.items()}

def _measure_expected_national(df: pd.DataFrame) -> Dict[str, float]:
    """Calcula a taxa esperada no país para cada medida, ponderando pelo volume de readmission e discharges"""
    d = (_ensure_measure_column(df)
        .dropna(subset=["Expected Readmission Rate", "Number of Discharges"])
        .copy())
    d["e"] = d["Expected Readmission Rate"].astype(float) / 100.0
    d["n"] = d["Number of Discharges"].astype(float)

    tmp = d.assign(ne=d["e"] * d["n"])
    agg = tmp.groupby("Measure", observed=True).agg(ne_sum=("ne","sum"), n_sum=("n","sum"))
    res = (agg["ne_sum"] / agg["n_sum"]).astype(float)
    return {str(k): float(v) for k, v in res.to_dict().items()}

def _align_categories_like_model(grid: pd.DataFrame, model) -> pd.DataFrame:
    """Faz a nova tabela usar as mesmas categorias (Estados/Medidas) do modelo"""
    g = grid.copy()
    mdata = getattr(model, "data", None)
    if isinstance(mdata, pd.DataFrame):
        for col in ("State", "Measure", "fac"):
            if col in g.columns and col in mdata.columns and isinstance(mdata[col].dtype, CategoricalDtype):
                g[col] = pd.Categorical(g[col], categories=mdata[col].cat.categories)
    return g


def _posterior_state_aggregate(p_row_draws: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Agrega simulacoes por estado, pesando pelos volumes de atendimento.

    Resultado {Estado ->  simulações}.
    """
    states = data["State"].astype(str).to_numpy()
    n = data["n"].to_numpy().astype(float)
    uniq = np.unique(states)
    out: Dict[str, np.ndarray] = {}
    for st in uniq:
        idx = (states == st)
        w = n[idx]
        num = (p_row_draws[:, idx] * w[None, :]).sum(axis=1)
        den = w.sum()
        out[st] = (num / den)
    return out

def _posterior_expected_by_state(df_sub: pd.DataFrame) -> Dict[str, float]:
    """Calcula, para cada estado, a taxa esperada ponderando pelo volume de atendimentos."""
    d = df_sub.dropna(subset=["Expected Readmission Rate", "Number of Discharges", "State"]).copy()
    d["e"] = d["Expected Readmission Rate"].astype(float) / 100.0
    d["n"] = d["Number of Discharges"].astype(float)
    tmp = d.assign(ne=d["e"] * d["n"])
    agg = tmp.groupby("State", observed=True).agg(ne_sum=("ne", "sum"), n_sum=("n", "sum"))
    res = (agg["ne_sum"] / agg["n_sum"]).astype(float)
    return {str(k): float(v) for k, v in res.to_dict().items()}


def _summarize_posterior_vec(x: np.ndarray, cred: float = 0.95) -> Dict[str, float]:
    """Resumo simples das simulações: média e 95%"""
    lo = (1.0 - cred) / 2.0
    hi = 1.0 - lo
    return {
        "mean": float(np.nanmean(x)),
        "lo":   float(np.nanquantile(x, lo)),
        "hi":   float(np.nanquantile(x, hi)),
    }

def _bayes_fdr_cut(q: np.ndarray, alpha: float) -> float:
    """Escolhe um corte de probabilidade que limita falsos alarmes nas comparações de muitos lugares 

    Ou seja, recebe uma lista de probabilidades possievlmente com valores 'acima do esperado'
    E retorna  o valor de corte.
    """
    q = np.asarray(q, float)
    if q.size == 0 or not np.isfinite(q).any():
        return 1.0
    reject, _ = fdrcorrection(1.0 - np.clip(q, 0.0, 1.0), alpha=alpha)
    return float(np.min(q[reject])) if reject.any() else 1.0

def _safe_slug(s: str) -> str:
    """Transforma um texto em nome de arquivo limpo"""
    import re
    return re.sub(r"[^a-z0-9_]+", "_", str(s).strip().lower()).strip("_")

# Agrega o indicador ERR por estado c/ várias métricas (mediana, média ponderada etc)
def _aggregate_err_by_state(df: pd.DataFrame, *, min_exposure: float = 0.0) -> pd.DataFrame:
    """Para cada estado, resume o indicador ERR com diferentes resumos (mediana, média com peso pelo volume).
    Filtra estados com volume mínimo caso faça sentido
    """
    d = df.dropna(subset=["State", "Excess Readmission Ratio", "Number of Discharges"]).copy()
    d["w"] = d["Number of Discharges"].astype(float).clip(lower=1.0)
    g = (
        d.groupby("State", observed=True)
         .agg(
            n_rows=("Excess Readmission Ratio", "count"),
            exposure=("Number of Discharges", "sum"),
            med_ERR=("Excess Readmission Ratio", "median"),
            wmean_ERR=("Excess Readmission Ratio",
                       lambda v: np.average(v.astype(float), weights=d.loc[v.index, "w"])),
            wmed_ERR=("Excess Readmission Ratio",
                      lambda v: _weighted_median(v.astype(float).values, d.loc[v.index, "w"].values)),
         )
         .reset_index()
    )
    if min_exposure > 0:
        g = g[g["exposure"] >= min_exposure]
    return g

def _get_facility_id_col(df: pd.DataFrame) -> Optional[str]:
    """Indica como identificar os hospitais, provider number = ID"""
    return "Provider Number" if ("Provider Number" in df.columns and df["Provider Number"].notna().any()) else None

def _ensure_measure_column(df: pd.DataFrame) -> pd.DataFrame:
    """Garante a existência da coluna 'Measure' como categoria) usando o 'Measure Name'"""
    if "Measure" not in df.columns and "Measure Name" in df.columns:
        df = df.copy()
        df["Measure"] = df["Measure Name"].astype("category")
    return df

#Modelos
def _weighted_quantile(x, w, q):
    """Calcula quartil levando em conta pesos (volumes)."""
    x = np.asarray(x, float); w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not m.any():
        return float("nan")
    ds = DescrStatsW(x[m], weights=w[m], ddof=0)
    # ndarray qdo return_pandas=False
    return float(np.asarray(ds.quantile([float(q)], return_pandas=False))[0])

def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    """Calcula a mediana levando em conta pesos (volumes)."""
    return _weighted_quantile(x, w, 0.5)

def _state_standardized_raw_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Taxa 'crua' por estado ajustada pelo mix de tipos de medida do país (sem puxar pra média)

    Retorna tabela com Estado e essa taxa.
    """
    d = _ensure_measure_column(df).dropna(
        subset=["State", "Number of Readmissions", "Number of Discharges", "Measure"]
    ).copy()
    d["y"] = d["Number of Readmissions"].astype(float)
    d["n"] = d["Number of Discharges"].astype(float)

    tbl = pd.pivot_table(
        d, index="Measure", columns="State", values=["y", "n"],
        aggfunc="sum", observed=True
    )
    y = tbl["y"].astype(float)
    n = tbl["n"].astype(float).clip(lower=1.0)
    p_hat = (y / n).fillna(0.0)  

    w = pd.Series(_measure_weights_national(d), name="w").reindex(p_hat.index).fillna(0.0)
    # Produto interno -> vetor por estado
    p = w.to_numpy() @ p_hat.to_numpy()  # shape: [States]
    return pd.DataFrame({"State": p_hat.columns.astype(str), "p_std_raw": p})

# Exercícios (Q1–Q5 + extras)

class Ex01CriarTrueAdmissionRate(Exercicio):
    """Q1 — Cria a taxa observada de readmissions e mostra um métricas (mínimo, meio e máximo)."""
    numero: ClassVar[int] = 1
    def executar(self, ctx: Contexto) -> None:
        df = ctx.df
        n_total = len(df)
        n_valid = df["True Admission Rate"].notna().sum()
        print(f"[Q1] True Admission Rate criado. Linhas = {n_total:,} | válidas = {n_valid:,} ({n_valid/n_total:.1%})")
        s = df["True Admission Rate"].dropna()
        if not s.empty:
            desc = s.describe(percentiles=[0.25, 0.5, 0.75])
            print("[Q1] Observed Readmission Rate (%) — resumo:\n" + desc.to_string(float_format=lambda v: f"{v:,.3f}"))
        print(f"[Q1] CSV 'clean' salvo em: {CFG.CLEAN_CSV}")


class Ex02BoxplotDiffTrueExpected(Exercicio):
    """Q2 — Compara 'observado – esperado' em percentual

    Mostra valores centrais e espalhamento e salva um boxplot.
    """
    numero: ClassVar[int] = 2
    def executar(self, ctx: Contexto) -> None:
        df = ctx.df.dropna(subset=["Diff True_minus_Expected"]).copy()
        if df.empty:
            print("[Q2] Sem dados.")
            return

        vals = df["Diff True_minus_Expected"].astype(float).values
        med = float(np.nanmedian(vals))
        q1  = float(np.nanpercentile(vals, 25))
        q3  = float(np.nanpercentile(vals, 75))
        iqr = q3 - q1

        #Dispersão de um paciente tipico, quantis ponderados
        w = df["Number of Discharges"].astype(float).clip(lower=1.0).to_numpy()
        w_q1 = _weighted_quantile(vals, w, 0.25)
        w_q3 = _weighted_quantile(vals, w, 0.75)
        w_iqr = w_q3 - w_q1
        
        w_med = _weighted_median(vals, w)
        w_mad = _weighted_median(np.abs(vals - w_med), w)

        print(f"[Q2] n={len(vals)} | mediana={med:.3f} p.p. | mediana ponderada={w_med:.3f} p.p. | "
            f"Q1={q1:.3f} | Q3={q3:.3f} | IQR={iqr:.3f} | "
            f"Weighted IQR={w_iqr:.3f} p.p. | Weighted MAD={w_mad:.3f} p.p.")
        print("A idéia é que as medidas ponderadas refletem um paciente típico, e hospitais pequenos n dominam a dispersão.")
        print("Mediana ~0 sugere boa calibração do 'Expected'; caudas largas indicam ruído ou variação")

        _set_matplotlib_style()
        plt.figure(figsize=(6.2, 4.2))
        plt.boxplot(vals, vert=True, showmeans=True)
        plt.axhline(0, ls="--", lw=1, alpha=0.6)
        plt.ylabel("Observed − Expected (pp)")
        plt.title("Difference: Observed Readmission Rate − Expected Readmission Rate")

        plt.tight_layout()
        out = CFG.FIG_DIR / "q2_box_true_minus_expected.png"
        plt.savefig(out, dpi=200); plt.close()
        print(f"[Q2] Figura: {out}")

class Ex03ScatterPredictedVsExpected(Exercicio):
    """Q3 — Compara taxas previstas versus esperadas e faz um check

    Cria um gráfico de dispersão por tipo de medida e um gráfico de 'esperado vs. observado' por faixas.
    Depois mede se uma linha reta explica bem a relação entre os dois ou não
    """
    numero: ClassVar[int] = 3
    def executar(self, ctx: Contexto) -> None:
        # 1) Check rápido: o quanto previsto e esperado andam juntos
        df0 = ctx.df.dropna(subset=["Predicted Readmission Rate",
                                    "Expected Readmission Rate",
                                    "Measure Name"]).copy()
        if df0.empty:
            print("[Q3] Sem dados.")
            return

        r, p = pearsonr(df0["Expected Readmission Rate"].astype(float),
                        df0["Predicted Readmission Rate"].astype(float))
        print(f"[Q3] Pearson r(Predicted, Expected) = {r:.3f}  p={p:.3g}")

        # 2) Teste de linha reta entre uma transformação do “esperado” e o “observado” (com peso pelo volume)
        # 2) Teste de linha reta (glm binomial)entre uma transformação do 'esperado' e o 'observado' (com peso pelo volume)
        req = ["Number of Readmissions", "Number of Discharges",
               "Expected Readmission Rate", "Provider Number"]
        d = (df0.dropna(subset=req)
                 .loc[lambda x: x["Number of Discharges"].astype(float) > 0,
                      req]  # apenascolunas necessárias
                 .assign(
                     y=lambda x: x["Number of Readmissions"].astype(float),
                     n=lambda x: x["Number of Discharges"].astype(float),
                     e=lambda x: (x["Expected Readmission Rate"].astype(float) / 100.0)
                                   .clip(1e-6, 1 - 1e-6),
                 ))
        if d.empty:
            print("[Q3] Sem dados válidos para o GLM.")
            return

        # Garante que a proporção fique entre quase 0 e quase 1; estava bugando
        d["prop"] = d["y"] / d["n"]
        d["logit_expected"] = logit(d["e"])
        d = d[np.isfinite(d["prop"]) & np.isfinite(d["logit_expected"])]

        glm = smf.glm(
            formula="prop ~ logit_expected",
            data=d,
            family=sm.families.Binomial(),
            freq_weights=d["n"],
        ).fit(cov_type="cluster", cov_kwds={"groups": d["Provider Number"].astype(str)})

        b0 = float(glm.params["Intercept"])
        b1 = float(glm.params["logit_expected"])
        ci = glm.conf_int()
        ci_b0 = ci.loc["Intercept"].to_numpy()
        ci_b1 = ci.loc["logit_expected"].to_numpy()

        print("[Q3] Calibração (GLM Binomial, logit):")
        print(f" logit(Observado) = {b0:.3f} + {b1:.3f} · logit(Esperado)")
        print(f" IC95% intercepto [{ci_b0[0]:.3f}, {ci_b0[1]:.3f}]  |  inclinação [{ci_b1[0]:.3f}, {ci_b1[1]:.3f}]")
        print(" Interpretação: intercepto≈0 e inclinação≈1 ⇒ boa calibração; desvios sugerem viés/miscalibração.")

        # 3) Gráfico de 'confiabilidade': faixas do 'esperado' com o que de fato aconteceu
        rel = (d.assign(bin=pd.qcut(d["e"], 10, duplicates="drop"),
                        ne=lambda x: x["n"] * x["e"])
                 .groupby("bin", observed=True)
                 .agg(ne_sum=("ne", "sum"), y_sum=("y", "sum"), n_sum=("n", "sum"))
                 .assign(exp=lambda g: g["ne_sum"] / g["n_sum"],
                         obs=lambda g: g["y_sum"] / g["n_sum"],
                         n=lambda g: g["n_sum"].astype(int))
                 .reset_index()[["bin", "exp", "obs", "n"]])

        # Plot do gráfico de 'confiabilidade', com a linha x=y como ref
        _set_matplotlib_style()
        plt.figure(figsize=(6.4, 4.6))
        plt.scatter(rel["exp"], rel["obs"], s=30, alpha=0.9)
        xs = np.linspace(rel["exp"].min(), rel["exp"].max(), 100)
        plt.plot(xs, xs, lw=1.2, alpha=0.9)
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        for _, r_ in rel.iterrows():
            plt.text(r_["exp"], r_["obs"], f"n={int(r_['n'])}",
                     fontsize=7, ha="left", va="bottom", alpha=0.75)
        plt.xlabel("Esperado (média ponderada no decil)")
        plt.ylabel("Observado (razão de somas)")
        plt.title("Confiabilidade: Observado vs Esperado (por decil)")

        # Mostra no canto os números da linha reta (ponto de partida e inclinação)
        txt = (f"Calibração (GLM, logit):\n"
               f"intercepto = {b0:.2f} [{ci_b0[0]:.2f}, {ci_b0[1]:.2f}]\n"
               f"inclinação = {b1:.2f} [{ci_b1[0]:.2f}, {ci_b1[1]:.2f}]")
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, ha="left", va="top",
                fontsize=8, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

        plt.tight_layout()
        out_rel = CFG.FIG_DIR / "q3_reliability_observed_vs_expected.png"
        plt.savefig(out_rel, dpi=200); plt.close()
        print(f"[Q3] Figura (confiabilidade): {out_rel}")


class Ex04ScatterErrVsDischarges(Exercicio):
    """Q4 — Vê como o indicador ERR muda com o volume de atendimentos.

    Mostra um gráfico de pontos e um 'funil' que delimita a faixa esperada. Também marca quem ficou fora dessa faixa.
    """
    numero: ClassVar[int] = 4

    @staticmethod
    def _funnel_lines(n_min: float, n_max: float, p_expected: float) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Desenha faixas de referência (95% e 99,8%) ao redor do valor‑alvo (ERR=1).

        Quanto menor o volume, mais larga a faixa — por isso o formato parece um funil.
        """
        n_grid = np.linspace(max(5.0, n_min), n_max, 250)
        p = float(np.clip(p_expected, 1e-4, 0.99))
        # Estimar a largura das faixas com base no volume e numa taxa média
        se = np.sqrt((1 - p) / (n_grid * p))
        z95, z998 = 1.96, 3.09
        bands = {
            "low95":  np.maximum(1.0 - z95  * se, 0.0),
            "up95":   1.0 + z95  * se,
            "low998": np.maximum(1.0 - z998 * se, 0.0),
            "up998":  1.0 + z998 * se,
        }
        return n_grid, bands

    def executar(self, ctx: Contexto) -> None:
        df = ctx.df.dropna(subset=["Excess Readmission Ratio", "Number of Discharges", "Expected Readmission Rate"]).copy()
        if df.empty:
            print("[Q4] Sem dados.")
            return

        x = df["Number of Discharges"].astype(float).to_numpy()
        y = df["Excess Readmission Ratio"].astype(float).to_numpy()
        r, p = pearsonr(x, y)
        print(f"[Q4] Pearson r(ERR, #Discharges) = {r:.3f}  p={p:.3g}")
        print("Interpretação: baixa correlação é esperada; variância do ERR aumenta em volumes pequenos (funil).")

        # Marca pontos fora da faixa de 95% (investigar melhor)
        p_i = df["Expected Readmission Rate"].astype(float).to_numpy() / 100.0
        n_safe = np.clip(x, 1.0, np.inf)
        # Garante que a proporção fique entre quase 0 e quase 1 para evitar contas instáveis (bug no cáculo resolvido)
        p_safe = np.clip(p_i, 1e-6, 1-1e-6)
        se_i = np.sqrt((1 - p_safe) / (n_safe * p_safe))

        low_i = 1.0 - 1.96 * se_i
        up_i  = 1.0 + 1.96 * se_i
        flags = (y < low_i) | (y > up_i)

        _set_matplotlib_style()
        plt.figure(figsize=(6.8, 4.4))
        plt.scatter(df["Number of Discharges"][~flags], df["Excess Readmission Ratio"][~flags],
                    s=12, alpha=0.6, label="Dentro 95%")
        if flags.any():
            plt.scatter(df["Number of Discharges"][flags], df["Excess Readmission Ratio"][flags],
                        s=16, alpha=0.85, label="Fora 95%")
        plt.axhline(1.0, ls="--", lw=1, alpha=0.7)
        plt.xlabel("Number of Discharges")
        plt.ylabel("Excess Readmission Ratio (Pred/Exp)")
        plt.title("Excess Readmission Ratio × Number of Discharges")
        plt.legend(fontsize=8)
        plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(6, integer=True))
        plt.tight_layout()

        out = CFG.FIG_DIR / "q4_scatter_err_vs_discharges.png"
        plt.savefig(out, dpi=200); plt.close()
        print(f"[Q4] Figura: {out}")

        # Calcula a taxa média ponderada pelo volume para usar como referência no funil (p0)
        w = df["Number of Discharges"].astype(float)
        e = (df["Expected Readmission Rate"].astype(float) / 100.0)
        p0 = float((w * e).sum() / np.clip(w.sum(), 1.0, np.inf))

        nmin, nmax = float(np.nanmin(x)), float(np.nanmax(x))
        n_grid, bands = self._funnel_lines(nmin, nmax, p0)

        _set_matplotlib_style()
        plt.figure(figsize=(7.2, 4.6))
        plt.scatter(x, y, s=10, alpha=0.5, label="Hospitais")
        plt.axhline(1.0, ls="--", lw=1, alpha=0.7, label="Alvo (ERR=1)")
        for k, lstyle in [("low95", ":"), ("up95", ":"), ("low998", "--"), ("up998", "--")]:
            plt.plot(n_grid, bands[k], lw=1.0, ls=lstyle, alpha=0.9, label=("95%" if "95" in k else "99,8%"))
        plt.xlabel("Number of Discharges")
        plt.ylabel("Excess Readmission Ratio (Pred/Exp)")
        plt.title("Funnel plot — ERR × volume (faixas 95% e 99,8%)")
        ax = plt.gca()
        ax.text(0.02, 0.95, f"p₀ (ponderado) = {p0:.1%}", transform=ax.transAxes,
                ha="left", va="top", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        plt.legend(fontsize=8, ncols=2)
        plt.tight_layout()
        out2 = CFG.FIG_DIR / "q4_funnel_err_vs_discharges.png"
        plt.savefig(out2, dpi=200); plt.close()
        print(f"[Q4] Figura (extra): {out2}")

class Ex05TopBottomEstadosErr(Exercicio):
    """Q5 — Lista os 5 estados 'piores”'e “melhores” depois de ajustar o mix de medidas e estabilizar números mais instáveis."""
    numero: ClassVar[int] = 5

    @staticmethod
    def _rank_states(df: pd.DataFrame) -> Tuple[List[str], List[str], pd.DataFrame]:
        # Usa os pesos do país para comparar estados de forma 'justa' entre medidas diferentes
        std = _fast_state_standardized_rate_eb(df)
        if std.empty:
            raise RuntimeError("Sem dados suficientes para EB padronizado.")

        w_m = _measure_weights_national(df)
        e_m = _measure_expected_national(df)
        e_nat = float(sum(w_m[m] * e_m[m] for m in w_m.keys() if m in e_m))
        e_nat = max(e_nat, 1e-9)

        exp_by_state = df.groupby("State", observed=True)["Number of Discharges"].sum()

        out = std.copy()
        out["post_mean_p_std"] = out["p_std"]
        out["post_mean_ratio_std"] = out["p_std"] / e_nat

        se_ratio = np.sqrt(out["var_p_std"].to_numpy(float)) / e_nat
        z = (out["post_mean_ratio_std"].to_numpy(float) - 1.0) / np.where(se_ratio > 0, se_ratio, np.nan)
        out["Pr_excess>0"] = norm.sf(0 - z)

        out["n_discharges"] = out["State"].map(exp_by_state).fillna(0).astype(int)
        out = out.loc[out["n_discharges"] >= CFG.Q5_MIN_N].copy()
        if out.empty:
            raise RuntimeError(f"Nenhum estado com exposição ≥ {CFG.Q5_MIN_N}.")

        res = out.sort_values("post_mean_p_std")
        low  = res.head(5)["State"].tolist()
        high = res.tail(5)["State"].tolist()
        return low, high, res[["State","post_mean_p_std","lo","hi","post_mean_ratio_std","Pr_excess>0","n_discharges"]]

    def executar(self, ctx: Contexto) -> None:
        df = ctx.df.dropna(subset=["State", "Excess Readmission Ratio", "Number of Discharges"]).copy()
        df["w"] = df["Number of Discharges"].astype(float).clip(lower=1.0)

        bot5, top5, summary = self._rank_states(ctx.df)
        print("[Q5] Estados (Top 5 pior risco):   " + ", ".join(top5))
        print("[Q5] Estados (Bottom 5 melhor risco): " + ", ".join(bot5))
        out_csv = CFG.BASE_DIR / "q5_state_eb_standardized_summary.csv"
        _safe_to_csv(summary, out_csv, desc="[Q5] CSV (EB, padronizado)", index=False, encoding="utf-8-sig")

        # Boxplot de ERR nos estados selecionados (descritivo)
        sel = top5 + bot5
        dsel = df[df["State"].isin(sel)].copy()
        _set_matplotlib_style()
        plt.figure(figsize=(10.0, 4.8))
        order = (summary.set_index("State").loc[sel].sort_values("post_mean_p_std").index.tolist())
        data_bp = [dsel.loc[dsel["State"] == st, "Excess Readmission Ratio"].astype(float).dropna().values for st in order]
        plt.boxplot(data_bp, showmeans=True)
        plt.xticks(np.arange(1, len(order)+1), order)
        plt.axhline(1.0, ls="--", lw=1, alpha=0.7)
        plt.ylabel("Excess Readmission Ratio (Pred/Exp)")
        plt.title("ERR por Estado — Top 5 vs Bottom 5 (EB-selected)")
        plt.tight_layout()
        out = CFG.FIG_DIR / "q5_box_err_by_state_top_bottom.png"
        plt.savefig(out, dpi=200); plt.close()
        print(f"[Q5] Figura: {out}")

class Ex06RankWithinMeasure(Exercicio):
    """Q6 — Dentro de cada tipo de medida, mostra os estados que mais/menos se destacam (considerando volume mínimo)."""
    numero: ClassVar[int] = 6

    def executar(self, ctx: Contexto) -> None:
        df = ctx.df.dropna(subset=["State", "Measure Name", "Excess Readmission Ratio",
                                   "Number of Discharges"]).copy()
        if df.empty:
            print("[Q6] Sem dados.")
            return

        # Parte frequente: resumo simples sem usar o modelo de simulação
        out_rows = []
        for meas, sub in df.groupby("Measure Name"):
            g = _aggregate_err_by_state(sub, min_exposure=CFG.Q6_MIN_STATE_MEASURE_N)
            if g.empty:
                continue

            # Ranqueia pela mediana do ERR (c/ contagens e volume)
            top  = g.nlargest(CFG.Q6_TOPK, "med_ERR").assign(rank_kind="top")
            bot  = g.nsmallest(CFG.Q6_TOPK, "med_ERR").assign(rank_kind="bottom")
            both = pd.concat([top, bot], ignore_index=True)
            both.insert(0, "Measure Name", meas)
            out_rows.append(both)

            _set_matplotlib_style()
            slug = _safe_slug(meas)
            plt.figure(figsize=(8.6, 4.6))
            plot_df = both.sort_values("med_ERR").copy()

            # Soma o total de altas por estado para mostrar nas labels
            # usar observed=True para tirar o FutureWarning
            disch_by_state = (sub.groupby("State", observed=True)["Number of Discharges"]
                    .sum()
                    .rename("discharges")
                    .astype(int)
                    .reset_index())
            plot_df = plot_df.merge(disch_by_state, on="State", how="left").fillna({"discharges": 0})

            x = np.arange(len(plot_df))
            plt.bar(x, plot_df["med_ERR"].astype(float).values, alpha=0.85)
            plt.axhline(1.0, ls="--", lw=1, alpha=0.8)
            plt.xticks(x, plot_df["State"].tolist(), rotation=0)
            plt.ylabel("Median ERR")
            plt.title(f"Top/Bottom {CFG.Q6_TOPK} states — {meas} (median ERR; Top = pior risco, #discharges≥{CFG.Q6_MIN_STATE_MEASURE_N})")

            # Escreve acima de cada barra quantas linhas e o total de altas
            for i, r in plot_df.reset_index(drop=True).iterrows():
                plt.text(i, r["med_ERR"] + 0.01, f"n obs={int(r['n_rows'])}\nN={int(r['discharges'])}",
                        ha="center", va="bottom", fontsize=7)

            plt.tight_layout()
            outp = CFG.FIG_DIR / f"q6_{slug}_state_top_bottom_median.png"
            plt.savefig(outp, dpi=200); plt.close()
            print(f"[Q6] Figura: {outp}")

        if out_rows:
            res = pd.concat(out_rows, ignore_index=True)
            out_csv = CFG.BASE_DIR / "q6_rank_by_measure.csv"
            _safe_to_csv(res, out_csv, desc="[Q6] CSV (freq)", index=False, encoding="utf-8-sig")

        else:
            print(f"[Q6] Nenhuma medida com estados ≥ {CFG.Q6_MIN_STATE_MEASURE_N}.")

class Ex07BootstrapStateCIs(Exercicio):
    """Q7 — Faixas de incerteza por estado usando o modelo de simulação e o mix nacional de medidas."""
    numero: ClassVar[int] = 7

    def executar(self, ctx: Contexto) -> None:
        # Estados-alvo = Top/Bottom do Q5 (reutiliza a função)
        bot5, top5, _summary = Ex05TopBottomEstadosErr._rank_states(ctx.df)
        states_sel = top5 + bot5

        d = _ensure_measure_column(ctx.df).dropna(subset=[
            "State", "Number of Readmissions", "Number of Discharges"
        ])
        model, idata, data = _fit_bayes_hlm_joint(d)
        pst = _posterior_state_standardized_p(d, model=model, idata=idata)

        rows = []
        for st in states_sel:
            draws = pst[st]   # will KeyError if missing -> explicit data issue
            s = _summarize_posterior_vec(draws)
            n_st = int(data["State"].astype(str).eq(st).sum())
            rows.append({"State": st, "n": n_st, "post_mean_p_std": s["mean"], "lo": s["lo"], "hi": s["hi"]})

        out_df = pd.DataFrame(rows).sort_values("post_mean_p_std")
        out_csv = CFG.BASE_DIR / "q7_state_posterior_ci_standardized.csv"
        _safe_to_csv(out_df, out_csv, desc="[Q7] CSV (Bayes, padronizado)", index=False, encoding="utf-8-sig")

        _set_matplotlib_style()
        plt.figure(figsize=(10.5, 5.0))
        x = np.arange(len(out_df))
        y = out_df["post_mean_p_std"].to_numpy(float)
        lo = y - out_df["lo"].to_numpy(float)
        hi = out_df["hi"].to_numpy(float) - y
        plt.errorbar(x, y, yerr=[lo, hi], fmt="o", capsize=3)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=1))
        plt.xticks(x, out_df["State"].tolist(), rotation=0)
        plt.ylabel("Prob. de readmissão p — IC credível 95% (padronizado)")
        plt.title("Estados (selecionados): p padronizado com IC95% (Bayes)")
        plt.tight_layout()
        out_fig = CFG.FIG_DIR / "q7_state_posterior_ci_standardized.png"
        plt.savefig(out_fig, dpi=200); plt.close()
        print(f"[Q7] Figura (Bayes, padronizado): {out_fig}")

class Ex08FDR_BenjaminiHochberg(Exercicio):
    """Q8 — Quando comparamos muitos estados/medidas, escolhemos um corte de probabilidade
    para limitar falsos alarmes (tanto no geral por estado quanto dentro de cada medida)."""
    numero: ClassVar[int] = 8

    def executar(self, ctx: Contexto) -> None:
        d = _ensure_measure_column(ctx.df).dropna(subset=[
            "State", "Number of Readmissions", "Number of Discharges", "Expected Readmission Rate"
        ]).copy()

        model, idata, _ = _fit_bayes_hlm_joint(d)
        pst = _posterior_state_standardized_p(d, model=model, idata=idata)

        w_m = _measure_weights_national(d)
        e_m = _measure_expected_national(d)
        e_nat = float(sum(w_m[m] * e_m[m] for m in w_m.keys() if m in e_m))

        exp_by_state = d.groupby("State", observed=True)["Number of Discharges"].sum()
        counts = d["State"].astype(str).value_counts()
        rows = []
        for st, draws in pst.items():
            if float(exp_by_state.get(st, 0.0)) < float(CFG.FDR_MIN_STATE_N):
                continue
            ratio = draws / max(e_nat, 1e-9)
            q = (ratio > 1.0).mean()
            s = _summarize_posterior_vec(ratio)
            rows.append({
                "State": st,
                "n_rows": int(counts.get(st, 0)),
                "n_discharges": int(exp_by_state.get(st, 0.0)),
                "post_mean_ratio_std": s["mean"], "lo": s["lo"], "hi": s["hi"],
                "Pr_excess>0": float(q)
            })
        if rows:
            dfp = pd.DataFrame(rows).sort_values("Pr_excess>0", ascending=False)
            tau = _bayes_fdr_cut(dfp["Pr_excess>0"].to_numpy(), CFG.FDR_ALPHA)
            dfp["discover@alpha"] = dfp["Pr_excess>0"] >= tau
            out_csv = CFG.BASE_DIR / "q8_state_bayes_fdr_standardized.csv"
            _safe_to_csv(dfp, out_csv, desc="[Q8A] CSV (Bayes, padronizado)", index=False, encoding="utf-8-sig")
            print(f"[Q8A] Bayes FDR (padronizado): τ={tau:.3f} ⇒ {int(dfp['discover@alpha'].sum())} descobertas a α={CFG.FDR_ALPHA}.")

        # (B) Por medida × estado
        rows2 = []
        for meas, sub in d.groupby("Measure Name"):
            model_m, idata_m, data_m = _fit_bayes_hlm(sub, measure=meas)
            p_row_m = _posterior_p_draws(model_m, idata_m, data_m, predictive=False)

            pst_m = _posterior_state_aggregate(p_row_m, data_m)
            e_by_state_m = _posterior_expected_by_state(sub)
            exp_by_state_m = data_m.groupby("State", observed=True)["n"].sum()
            counts_m = data_m["State"].astype(str).value_counts()

            for st, draws in pst_m.items():
                if float(exp_by_state_m.get(st, 0.0)) < float(CFG.Q6_MIN_STATE_MEASURE_N):
                    continue
                e = e_by_state_m.get(st, np.nan)
                ratio = draws / e
                q = (ratio > 1.0).mean()
                s = _summarize_posterior_vec(ratio)
                rows2.append({
                    "Measure Name": meas,
                    "State": st,
                    "n_rows": int(counts_m.get(st, 0)),
                    "n_discharges": int(exp_by_state_m.get(st, 0.0)),
                    "post_mean_ratio": s["mean"], "lo": s["lo"], "hi": s["hi"],
                    "Pr_excess>0": float(q)
                })
        if rows2:
            dfpm = pd.DataFrame(rows2)
            tau2 = _bayes_fdr_cut(dfpm["Pr_excess>0"].to_numpy(), CFG.FDR_ALPHA)
            dfpm["discover@alpha"] = dfpm["Pr_excess>0"] >= tau2
            out_csv2 = CFG.BASE_DIR / "q8_measure_state_bayes_fdr.csv"
            dfpm2 = dfpm.sort_values(["Measure Name", "Pr_excess>0", "State"], ascending=[True, False, True])
            _safe_to_csv(dfpm2, out_csv2, desc="[Q8B] CSV (Bayes)", index=False, encoding="utf-8-sig")
            print(f"[Q8B] Bayes-FDR (medida×estado): τ={tau2:.3f} ⇒ {int(dfpm['discover@alpha'].sum())} descobertas a α={CFG.FDR_ALPHA}.")


class Ex09EmpiricalBayesShrinkage(Exercicio):
    """Q9 — Deixa as estimativas por estado mais estáveis, puxando para a média nacional principalmente onde há poucos casos."""
    numero: ClassVar[int] = 9
    def executar(self, ctx: Contexto) -> None:
        d = _ensure_measure_column(ctx.df).dropna(
            subset=["State", "Number of Readmissions", "Number of Discharges"]
        ).copy()

        model, idata, data = _fit_bayes_hlm_joint(d)
        pst = _posterior_state_standardized_p(d, model=model, idata=idata)
        raw_std = _state_standardized_raw_rate(d)

        rows = [{"State": st, **_summarize_posterior_vec(draws)} for st, draws in pst.items()]
        est = (pd.DataFrame(rows)
                .rename(columns={"mean": "post_mean_p_std"})
                .merge(raw_std, on="State", how="left"))

        out_csv = CFG.BASE_DIR / "q9_state_model_shrinkage_standardized.csv"
        est2 = est.sort_values("post_mean_p_std")
        _safe_to_csv(est2, out_csv, desc="[Q9] CSV (Bayes shrinkage, padronizado)", index=False, encoding="utf-8-sig")
        print(f"[Q9] Shrinkage hierárquico (Bayes) concluído. CSV: {out_csv}")

        if CFG.EB_PLOT:
            _set_matplotlib_style()
            plt.figure(figsize=(7.2, 5.0))
            exposure = d.groupby("State", observed=True)["Number of Discharges"].sum().reset_index(name="n_discharges")
            est_plot = est.merge(exposure, on="State", how="left")
            sizes = 20 + 80 * (est_plot["n_discharges"] / est_plot["n_discharges"].max())
            plt.scatter(est_plot["p_std_raw"], est_plot["post_mean_p_std"], s=sizes, alpha=0.7)
            for frac, label in [(0.25, "25% of max N"), (0.50, "50%"), (1.00, "100%")]:
                plt.scatter([], [], s=20 + 80*frac, label=label)
            plt.legend(title="State exposure (discharges)", loc="lower right", frameon=False, fontsize=8)
            lims = [float(np.nanmin(est_plot[["p_std_raw", "post_mean_p_std"]].to_numpy())),
                    float(np.nanmax(est_plot[["p_std_raw", "post_mean_p_std"]].to_numpy()))]
            plt.plot(lims, lims, lw=1.2, alpha=0.8)
            ax = plt.gca()
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=1))
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=1))
            plt.xlabel("Taxa padronizada por estado (raw, mix nacional)")
            plt.ylabel("p padronizado — média a posteriori (shrinkage)")
            plt.title("Shrinkage hierárquico — Estados (padronizado)")
            plt.tight_layout()
            out_fig = CFG.FIG_DIR / "q9_state_model_shrinkage_std_raw_vs_post.png"
            plt.savefig(out_fig, dpi=200); plt.close()
            print(f"[Q9] Figura (Bayes): {out_fig}")


class Ex10ExecutiveTakeaway(Exercicio):
    """Q10 — Painel resumido"""
    numero: ClassVar[int] = 10

    def executar(self, ctx: Contexto) -> None:
        # Collect a few key numbers to display
        # (Reuses your Q5 ranking function and a lightweight GLM calibration)
        df = ctx.df.dropna(subset=["Predicted Readmission Rate", "Expected Readmission Rate"]).copy()

        # Top/Bottom (sempre via Q5 helper)
        bot5, top5, _summary = Ex05TopBottomEstadosErr._rank_states(ctx.df)

        # ---- Deterministic GLM (logit) with provider clustering; identical build as Q3 ----
        req = ["Number of Readmissions", "Number of Discharges", "Expected Readmission Rate", "Provider Number"]
        df_glm = df.dropna(subset=req).copy()
        df_glm = df_glm[df_glm["Number of Discharges"].astype(float) > 0].copy()

        df_glm["prop"] = (df_glm["Number of Readmissions"].astype(float) /
                          df_glm["Number of Discharges"].astype(float))
        p_exp = (df_glm["Expected Readmission Rate"].astype(float) / 100.0).clip(1e-6, 1-1e-6)
        df_glm["logit_expected"] = logit(p_exp)

        m = (np.isfinite(df_glm["prop"]) & np.isfinite(df_glm["logit_expected"]))
        dglm = df_glm.loc[m, ["prop", "logit_expected", "Number of Discharges", "Provider Number"]].copy()

        X = sm.add_constant(dglm[["logit_expected"]], has_constant="add")
        y_glm = dglm["prop"]
        w_glm = dglm["Number of Discharges"].astype(float)
        groups = dglm["Provider Number"].astype(str)

        glm = sm.GLM(y_glm, X, family=sm.families.Binomial(), freq_weights=w_glm).fit(
            cov_type="cluster", cov_kwds={"groups": groups}
        )
        b0 = float(glm.params["const"]); b1 = float(glm.params["logit_expected"])
        ci = glm.conf_int()
        ci_b0 = ci.loc["const"].to_numpy(); ci_b1 = ci.loc["logit_expected"].to_numpy()

        calib_text = (f"Calibration (GLM): intercept={b0:.2f} [{ci_b0[0]:.2f},{ci_b0[1]:.2f}] "
                      f"| slope={b1:.2f} [{ci_b1[0]:.2f},{ci_b1[1]:.2f}]")
        print("[Q10]", calib_text)

#Execução principal
def main() -> None:
    
    _set_matplotlib_style()
    # 1) Carrega e higieniza o dataset
    try:
        df = carregar_higienizar()
    except Exception as e:
        print(f"Falha ao preparar dataset: {e}", file=sys.stderr)
        sys.exit(1)

    # 2) Sanity checks (tamanho e cardinalidade)
    print(f"Dataset pronto: {len(df):,} linhas | colunas: {list(df.columns)}")
    print("Estados únicos:", df["State"].nunique(), " | Medidas únicas:", df["Measure Name"].nunique())

    # 3) Monta o contexto e executa as questões 
    ctx = Contexto(df=df)
    for ex in _RegistroExercicios.instancias_ordenadas():
        print(f"\nExercício {ex.numero:02d}")
        ex.executar(ctx)

    # 4) Gera o Markdown 'lista02_resumo.md'
    _write_summary_markdown()  

# Wrappers q mantém os nomes alinhados 
class _E1(Ex01CriarTrueAdmissionRate): pass
class _E2(Ex02BoxplotDiffTrueExpected): pass
class _E3(Ex03ScatterPredictedVsExpected): pass
class _E4(Ex04ScatterErrVsDischarges): pass
class _E5(Ex05TopBottomEstadosErr): pass
class _E6(Ex06RankWithinMeasure): pass
class _E7(Ex07BootstrapStateCIs): pass
class _E8(Ex08FDR_BenjaminiHochberg): pass
class _E9(Ex09EmpiricalBayesShrinkage): pass
class _E10(Ex10ExecutiveTakeaway): pass

if __name__ == "__main__":
    main()
