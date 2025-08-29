
"""
Readmissões em Hospitais Medicare — Exercícío em dupla
Autor: André Ichiro Katsurada e Danilo Guimarães da Silva
Data: 29/08/2025
Curso: Programa Avançado em Data Science e Decisão, Computação para a Ciência de Dados, INSPER

- PI:
- PII: 

"""

import pandas as pd, seaborn as sns, matplotlib.pyplot as plt

from pathlib import Path

# Exercício em dupla PI
sns.set_theme(style="ticks")
 
import pandas as pd
 
hospital_df = pd.read_csv('https://raw.githubusercontent.com/amkaris/EDA/master/cms_hospital_readmissions.csv')
 
num_cols = [
    "Number of Readmissions",
    "Number of Discharges",
    "Expected Readmission Rate",
    "Predicted Readmission Rate",
    "Excess Readmission Ratio",
]
for c in num_cols:
    if c in hospital_df.columns:
        hospital_df[c] = pd.to_numeric(hospital_df[c], errors="coerce")
 
hospital_df["True Admission Rate"] = hospital_df["Number of Readmissions"] / hospital_df["Number of Discharges"].replace(0, pd.NA)

hospital_df['True Admission Rate'] = pd.to_numeric(hospital_df['True Admission Rate'], errors='coerce')
 
hospital_df2 = hospital_df.nlargest(5, 'True Admission Rate')

g = sns.jointplot(
    data=hospital_df,
    x="Number of Discharges", y="True Admission Rate", hue="State",
    kind="scatter",
)

# Exercício em dupla PII
def heatmap_err(df, top_states=20, save_csv=None, save_fig=None):
    """
    ERR (Pred/Exp) ponderado por altas, por Estado × Medida.
    """
    cols = ["Predicted Readmission Rate","Expected Readmission Rate","Number of Discharges"]

    # Labels mais claros p/ ler
    MEASURE_MAP = {
        "READM-30-AMI-HRRP": "Infarto (AMI)",
        "READM-30-COPD-HRRP": "Doença pulmonar (DPOC/COPD)",
        "READM-30-HF-HRRP": "Insuficiência cardíaca (HF)",
        "READM-30-HIP-KNEE-HRRP": "Quadril/Joelho",
        "READM-30-PN-HRRP": "Pneumonia (PN)",
    }

    df = df.copy()
    df[cols] = df[cols].replace({'%': ''}, regex=True).apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["State","Measure Name"] + cols)
    df = df[df["Number of Discharges"] > 0]

    pt = (df.assign(P=lambda x: x["Predicted Readmission Rate"] * x["Number of Discharges"],
                    E=lambda x: x["Expected Readmission Rate"] * x["Number of Discharges"])
            .pivot_table(index="State", columns="Measure Name",
                         values={"P": "sum", "E": "sum", "Number of Discharges": "sum"}))

    #ERR = soma(Pred*Altas) / soma(Esp*Altas)
    mat = (pt["P"] / pt["E"]).where(pt["E"] > 0)

    #Top-N estados por volume (altas)
    state_order = (pt["Number of Discharges"].sum(axis=1)
                   .sort_values(ascending=False).head(top_states).index)
    mat = mat.loc[state_order]

    #Colunas e uso dos labels 
    col_order = [k for k in MEASURE_MAP if k in mat.columns] + [c for c in mat.columns if c not in MEASURE_MAP]
    mat = mat.reindex(columns=col_order).rename(columns=MEASURE_MAP)

    #Salvar
    if save_csv:
        mat.to_csv(save_csv, float_format="%.5f")

    # Plot c/ seaborn
    plt.figure(figsize=(8, 0.5*len(mat) + 2))
    ax = sns.heatmap(mat, cmap="vlag", center=1.0, robust=True,
                     linewidths=.25, linecolor="white")
    ax.set(title="ERR (Pred/Exp) — Top estados por volume (1,0 = esperado)",
           xlabel="Medida", ylabel="Estado")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center") 
    plt.tight_layout()
    if save_fig:
        plt.savefig(save_fig, dpi=200, bbox_inches="tight")
    plt.show()

    return mat  #devolve a matriz 

def write_err_md(mat, save_md, measures_pt=None, artifacts=None):
    """
    Gera um README em MD sobre o heatmap.
    """
    medidas = measures_pt or list(mat.columns)
    md = f"""# ERR (Predito/Esperado) — Estado e Medida

## Pressupostos
- Para cada **Estado × Medida**, calculamos a relação **Predito/Esperado (ERR)** **ponderada pelo número de altas**.
- **Cores**: branco ~ **1,0** (esperado), vermelho **> 1,0** (pior), azul **< 1,0** (melhor).
- **Medidas**: {", ".join(medidas)}.
- Seleciona os **{len(mat.index)}** estados com mais altas e centraliza o mapa em **1,0**.

## Interpretação 
- **Quadril/Joelho**. Melhor que o esperado: vários estados aparecem em azul (ND, VT, IN, NY, IA, NH, MD, TN). Apesar de ter 10 azuis e 10 vermelhos, os azuis são mais intensos e os vermelhos menos. 
- **Doença pulmonar** e **Pneumonia**. Abaixo do esperado: muitos estados em vermelho; são focos claros de pior desempenho. Pulmonar (6 cinzas x 14 vermelhos, com vermelhos intensos) e pneumonia (3 azuis claros e 2 cinzas x 15 vermelhos) 
 **Insuficiência cardíaca** e **Infarto**. Misto com predominância abaixo do esperado: variação relevante entre estados, mas mais vermelhos claros no geral.
"""
    if artifacts:
        md += "### Artefatos\n"
        if artifacts.get("csv"):
            md += f"- Matriz usada no gráfico: `{artifacts['csv']}`\n"
        if artifacts.get("fig"):
            md += f"- Figura do heatmap: `{artifacts['fig']}`\n"

    with open(save_md, "w", encoding="utf-8") as f:
        f.write(md)

if __name__ == "__main__":
    try:
        df = pd.read_csv("hospital_clean.csv")
    except Exception:
        df = pd.read_csv("https://raw.githubusercontent.com/amkaris/EDA/master/cms_hospital_readmissions.csv")

    SCRIPT_DIR = Path(__file__).resolve().parent
    csv_path = SCRIPT_DIR / "err_state_measure.csv"
    fig_path = SCRIPT_DIR / "err_heatmap.png"
    md_path  = SCRIPT_DIR / "interpretacao_ERR.md"

    mat = heatmap_err(df, top_states=20, save_csv=csv_path, save_fig=fig_path)
    write_err_md(
        mat,
        save_md=md_path,
        artifacts={"csv": str(csv_path), "fig": str(fig_path)}
    )