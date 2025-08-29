"""
Mini‑teste — Readmissões em Hospitais Medicare — Lista 02
Autor: André Ichiro Katsurada
Data: 29/08/2025
Curso: Programa Avançado em Data Science e Decisão, Computação para a Ciência de Dados, INSPER

- Parte 1: respostas da LLM
- Parte 2: sugestões da LLM para melhorar meu código
"""

#Parte 1: respostas da LLM
#No geral o codigo funciona. 
#Há imports que não estão sendo usados (typing) e a LLM esqueceu completamente a parte de interpretação dos gráficos
#Além disso, os gráficos não estão sendo salvos

# %% -------------------- Setup -------------------- #
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Iterable, List, Tuple

# Load dataset
url = "https://raw.githubusercontent.com/amkaris/EDA/master/cms_hospital_readmissions.csv"
df = pd.read_csv(url)

# Ensure numeric conversion where necessary
for col in [
    "Number of Discharges",
    "Number of Readmissions",
    "Excess Readmission Ratio",
    "Predicted Readmission Rate",
    "Expected Readmission Rate",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# %% -------------------- Q1 -------------------- #
# Create True Admission Rate
df["True Admission Rate"] = df["Number of Readmissions"] / df["Number of Discharges"]

# %% -------------------- Q2 -------------------- #
# Boxplot of True Admission Rate − Expected Readmission Rate
df["Diff True−Expected"] = df["True Admission Rate"] - df["Expected Readmission Rate"]

plt.figure(figsize=(6, 5))
sns.boxplot(y=df["Diff True−Expected"], color="skyblue")
plt.axhline(0, color="red", linestyle="--")
plt.title("Q2 — Difference: True vs Expected Readmission Rate")
plt.ylabel("True − Expected Rate")
plt.show()

# %% -------------------- Q3 -------------------- #
# Scatter: Predicted vs Expected, colored by Measure Name
plt.figure(figsize=(7, 6))
sns.scatterplot(
    data=df,
    x="Predicted Readmission Rate",
    y="Expected Readmission Rate",
    hue="Measure Name",
    alpha=0.7
)
plt.plot([df["Expected Readmission Rate"].min(), df["Expected Readmission Rate"].max()],
         [df["Expected Readmission Rate"].min(), df["Expected Readmission Rate"].max()],
         color="black", linestyle="--")  # diagonal line
plt.title("Q3 — Predicted vs Expected Readmission Rate")
plt.show()

# %% -------------------- Q4 -------------------- #
# Scatter: Excess Readmission Ratio vs Number of Discharges
plt.figure(figsize=(7, 6))
sns.scatterplot(
    data=df,
    x="Number of Discharges",
    y="Excess Readmission Ratio",
    alpha=0.6
)
plt.title("Q4 — Excess Readmission Ratio vs Number of Discharges")
plt.xlabel("Number of Discharges")
plt.ylabel("Excess Readmission Ratio")
plt.show()

# %% -------------------- Q5 -------------------- #
# Top 5 and bottom 5 states by Excess Readmission Ratio
state_avg = df.groupby("State")["Excess Readmission Ratio"].mean().dropna()
top5 = state_avg.nlargest(5).index
bottom5 = state_avg.nsmallest(5).index
selected_states = df[df["State"].isin(top5.union(bottom5))]

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=selected_states,
    x="State",
    y="Excess Readmission Ratio",
    order=list(top5) + list(bottom5),
    palette="Set2"
)
plt.title("Q5 — Top & Bottom 5 States by Excess Readmission Ratio")
plt.show()

#Parte 2: melhorias da LLM no meu código

#Sugestão da LLM para Q1
# Essa sugestão não é necessária, já que limpei e tratei os dados caso alguns hospitais usassem %
# Unit consistency check: you’re multiplying by 100.0, which assumes the denominator is absolute counts. If some hospitals already report percentages, you might double-scale. Add a safeguard:
# if df["Number of Readmissions"].max() <= 1.0:
#    logger.warning("Possible mis-scaled readmission counts (already in %?)")

#Sugestão da LLM para Q2
# A 1a sugestão da LLM, para trocar pelo seaborn, é ótima; a 2a sugestão é ruim, deixa o gráfico bem dificil de ler

# Use seaborn boxplot with jitter or swarm points → makes outliers visible:
# sns.boxplot(y="Diff True_minus_Expected", data=df, showmeans=True)
# sns.stripplot(y="Diff True_minus_Expected", data=df, color="0.25", size=2, alpha=0.5)

# → More informative: not just quartiles, but also the distribution.
# Add median annotation:
# ax.axhline(med, ls="--", color="red", lw=1)
# ax.text(0.1, med, f"Median={med:.2f}", color="red")

#Sugestão da LLM para Q3
# As sugestões da LLM são boas, me lembraram de colorir por Measure Name e para adicionar a linha de 45; porém, os gráficos ficaram estranhos

# Color by Measure Name (as per requirement):
# sns.scatterplot(
#     data=df0,
#     x="Expected Readmission Rate",
#     y="Predicted Readmission Rate",
#     hue="Measure Name",
#     alpha=0.6
# )

#Sugestão da LLM para Q4
# Essas são as melhores sugestões. A 1a para colocar a escala em log, o que faz muito sentido. A 2a sugestão também é boa, sugerindo para colorir os outliers

# Log-scale x-axis: discharges vary by orders of magnitude; linear scale compresses small hospitals:
# plt.xscale("log")

# Highlight outliers: instead of just flagging with color, label a few extreme hospitals:
# outliers = df.loc[flags].nlargest(5, "Excess Readmission Ratio")
# for _, r in outliers.iterrows():
#     ax.text(r["Number of Discharges"], r["Excess Readmission Ratio"], r["State"], fontsize=6)


#Sugestão da LLM para Q5
# A sugestão para usar um violin plot é ótima, porém o gráfico em sí ficou estranho. A sugestão de print não é muito relevante.

# Plot improvement: right now you boxplot ERR by state. Consider a violin plot or stripplot overlay:
# sns.violinplot(x="State", y="Excess Readmission Ratio", data=dsel, inner="box")
# sns.stripplot(x="State", y="Excess Readmission Ratio", data=dsel, color="0.3", size=2, jitter=True)

# → Shows both distribution shape and outliers.
# Ranking clarity: in the printout, explicitly report whether the “top5” are better or worse risk. Right now you print “Top 5 pior risco” but then sort ascending, which is slightly confusing. Suggest:
# print("[Q5] Bottom 5 (best):", ", ".join(bot5))
# print("[Q5] Top 5 (worst):", ", ".join(top5))

# Como ficou o código melhorado pela LLM:
from pathlib import Path
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Tuple
import sys
import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import pearsonr, norm
from scipy.special import logit

# --------------------- logging ---------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

# --------------------- config ---------------------
@dataclass(slots=True, frozen=True)
class Config:
    DATA_URL: str = "https://raw.githubusercontent.com/amkaris/EDA/master/cms_hospital_readmissions.csv"
    BASE_DIR: Path = Path(__file__).resolve().parent
    FIG_DIR: Path = Path(__file__).resolve().parent / "lista02_figs"
    CLEAN_CSV: Path = Path(__file__).resolve().parent / "hospital_clean.csv"
    Q5_MIN_N: int = 1000

CFG = Config()
CFG.FIG_DIR.mkdir(parents=True, exist_ok=True)

# --------------------- helpers ---------------------
def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.rstrip("%").str.strip(), errors="coerce")

def _safe_to_csv(df: pd.DataFrame, path: Path, *, desc: str, index: bool = False) -> None:
    try:
        df.to_csv(path, index=index, encoding="utf-8-sig")
        print(f"{desc}: {path}")
    except Exception as e:
        logger.warning(f"Falha ao salvar {desc}: {e}")

def _set_matplotlib_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

def _weighted_quantile(x, w, q):
    x, w = np.asarray(x, float), np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not m.any():
        return np.nan
    sorter = np.argsort(x[m])
    x_sorted, w_sorted = x[m][sorter], w[m][sorter]
    cumsum = np.cumsum(w_sorted) - 0.5 * w_sorted
    return np.interp(q * w_sorted.sum(), cumsum, x_sorted)

def _weighted_median(x, w): return _weighted_quantile(x, w, 0.5)

# --------------------- data prep ---------------------
def carregar_higienizar() -> pd.DataFrame:
    raw = pd.read_csv(CFG.DATA_URL)
    df = raw.copy()
    df.replace({"Not Available": np.nan, "": np.nan}, inplace=True)
    for c in ["Number of Discharges", "Number of Readmissions",
              "Predicted Readmission Rate", "Expected Readmission Rate",
              "Excess Readmission Ratio"]:
        df[c] = _to_num(df[c])
    with np.errstate(divide="ignore", invalid="ignore"):
        den = df["Number of Discharges"].replace({0: np.nan})
        df["True Admission Rate"] = (df["Number of Readmissions"] / den) * 100.0
    df["Diff True_minus_Expected"] = df["True Admission Rate"] - df["Expected Readmission Rate"]
    df["State"] = df["State"].astype("category")
    _safe_to_csv(df, CFG.CLEAN_CSV, desc="Clean CSV", index=False)
    return df

# --------------------- framework ---------------------
class Contexto: 
    def __init__(self, df: pd.DataFrame): self.df = df
class Exercicio: numero: ClassVar[int]; 
def executar(self, ctx: Contexto): raise NotImplementedError

# --------------------- Q1 ---------------------
class Ex01CriarTrueAdmissionRate(Exercicio):
    numero: ClassVar[int] = 1
    def executar(self, ctx: Contexto) -> None:
        df = ctx.df
        if df["Number of Readmissions"].max() <= 1.0:
            logger.warning("Possible mis-scaled readmission counts (already in %?)")
        df["True Admission Rate"] = df["True Admission Rate"].astype("float32")
        n_total, n_valid = len(df), df["True Admission Rate"].notna().sum()
        print(f"[Q1] True Admission Rate criado. Linhas={n_total} | válidas={n_valid} ({n_valid/n_total:.1%})")

# --------------------- Q2 ---------------------
class Ex02BoxplotDiffTrueExpected(Exercicio):
    numero: ClassVar[int] = 2
    def executar(self, ctx: Contexto) -> None:
        df = ctx.df.dropna(subset=["Diff True_minus_Expected"]).copy()
        if df.empty: return
        vals, med = df["Diff True_minus_Expected"].to_numpy(float), float(np.nanmedian(df["Diff True_minus_Expected"]))
        w = df["Number of Discharges"].to_numpy(float).clip(1.0)
        w_med, w_mad = _weighted_median(vals, w), _weighted_median(np.abs(vals - med), w)
        _safe_to_csv(pd.DataFrame({"median":[med],"w_median":[w_med],"w_mad":[w_mad]}),
                     CFG.BASE_DIR/"q2_summary.csv", desc="Q2 Summary")
        _set_matplotlib_style()
        plt.figure(figsize=(6.4,4.2))
        ax = sns.boxplot(y="Diff True_minus_Expected", data=df, showmeans=True)
        sns.stripplot(y="Diff True_minus_Expected", data=df, color="0.25", size=2, alpha=0.5)
        ax.axhline(med, ls="--", color="red"); ax.text(0.1, med, f"Median={med:.2f}", color="red")
        plt.savefig(CFG.FIG_DIR/"q2_box.png"); plt.close()

# --------------------- Q3 ---------------------
class Ex03ScatterPredictedVsExpected(Exercicio):
    numero: ClassVar[int] = 3
    def executar(self, ctx: Contexto) -> None:
        df = ctx.df.dropna(subset=["Predicted Readmission Rate","Expected Readmission Rate","Measure Name"]).copy()
        if df.empty: return
        r,_ = pearsonr(df["Expected Readmission Rate"], df["Predicted Readmission Rate"])
        print(f"[Q3] Pearson r={r:.3f}")
        _set_matplotlib_style()
        plt.figure(figsize=(6.4,4.6))
        ax = sns.scatterplot(data=df,x="Expected Readmission Rate",y="Predicted Readmission Rate",hue="Measure Name",alpha=0.6)
        lims=[df["Expected Readmission Rate"].min(),df["Expected Readmission Rate"].max()]
        ax.plot(lims,lims,ls="--",color="grey")
        plt.savefig(CFG.FIG_DIR/"q3_scatter.png"); plt.close()

# --------------------- Q4 ---------------------
class Ex04ScatterErrVsDischarges(Exercicio):
    numero: ClassVar[int] = 4
    @staticmethod
    def _funnel_lines(n_min,n_max,p):
        n=np.linspace(max(5,n_min),n_max,200); se=np.sqrt((1-p)/(n*p))
        return n,{"low95":1-1.96*se,"up95":1+1.96*se,"low998":1-3.09*se,"up998":1+3.09*se}
    def executar(self,ctx:Contexto)->None:
        df=ctx.df.dropna(subset=["Excess Readmission Ratio","Number of Discharges","Expected Readmission Rate"]).copy()
        if df.empty:return
        x,y=df["Number of Discharges"].to_numpy(float),df["Excess Readmission Ratio"].to_numpy(float)
        p=df["Expected Readmission Rate"].to_numpy(float)/100
        se=np.sqrt((1-p)/(np.clip(x,1,np.inf)*np.clip(p,1e-6,1-1e-6)))
        flags=(y<1-1.96*se)|(y>1+1.96*se)
        _set_matplotlib_style(); plt.figure(figsize=(7,4.6)); ax=plt.gca()
        plt.scatter(x[~flags],y[~flags],s=12,alpha=0.6,label="Dentro 95%")
        plt.scatter(x[flags],y[flags],s=16,alpha=0.85,label="Fora 95%")
        for _,r in df.loc[flags].nlargest(5,"Excess Readmission Ratio").iterrows():
            ax.text(r["Number of Discharges"],r["Excess Readmission Ratio"],r["State"],fontsize=6)
        ax.set_xscale("log"); ax.axhline(1,ls="--"); plt.savefig(CFG.FIG_DIR/"q4_scatter.png"); plt.close()
        n_grid,bands=self._funnel_lines(x.min(),x.max(),p.mean())
        plt.figure(); plt.scatter(x,y,s=10,alpha=0.5); plt.fill_between(n_grid,bands["low95"],bands["up95"],alpha=0.2)
        plt.xscale("log"); plt.savefig(CFG.FIG_DIR/"q4_funnel.png"); plt.close()

# --------------------- Q5 ---------------------
class Ex05TopBottomEstadosErr(Exercicio):
    numero: ClassVar[int] = 5
    @staticmethod
    def _rank_states(df:pd.DataFrame)->Tuple[List[str],List[str]]:
        g=df.groupby("State")["Excess Readmission Ratio"].median().dropna()
        g=g[df.groupby("State")["Number of Discharges"].sum()>=CFG.Q5_MIN_N]
        return g.nsmallest(5).index.tolist(), g.nlargest(5).index.tolist()
    def executar(self,ctx:Contexto)->None:
        bot,top=self._rank_states(ctx.df); print("Bottom5(best):",bot); print("Top5(worst):",top)
        sel=bot+top; d=ctx.df[ctx.df["State"].isin(sel)]
        _set_matplotlib_style(); plt.figure(figsize=(10,4.8))
        sns.violinplot(x="State",y="Excess Readmission Ratio",data=d,inner="box")
        sns.stripplot(x="State",y="Excess Readmission Ratio",data=d,color="0.3",size=2,jitter=True)
        plt.axhline(1,ls="--"); plt.savefig(CFG.FIG_DIR/"q5_violin.png"); plt.close()

# --------------------- main ---------------------
def main():
    df=carregar_higienizar(); ctx=Contexto(df)
    for ex in [Ex01CriarTrueAdmissionRate(),Ex02BoxplotDiffTrueExpected(),
               Ex03ScatterPredictedVsExpected(),Ex04ScatterErrVsDischarges(),
               Ex05TopBottomEstadosErr()]:
        ex.executar(ctx)

if __name__=="__main__":
    main()
