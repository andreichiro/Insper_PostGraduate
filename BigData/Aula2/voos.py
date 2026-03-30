# python3
# utf-8
"""
voo.py

Objetivos:
- Descobrir o maior atraso de partida (DepDelay) em todos os arquivos.
- Para a rota JFK->LAX (excluindo cancelados): contagem por AnoxMês,
  heatmap c/ nomes de meses, e identificação do mês
  com + voos no período.
- Execução distribuída c/ Dask (LocalCluster via Client()).
"""

from __future__ import annotations

import glob
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 

import numpy as np  
import pandas as pd  

import dask  
import dask.dataframe as dd  
from dask.distributed import Client  

#Todo: better logging, classes for different parts of the code
#This will not be sent, so I guess it's not a big deal

# Defaults 
DEFAULT_PATHS = [
    "/Users/akatsurada/Documents/INSPER/BigData/Aula2/voos_nyc/*.csv",  # path
    "data/nycflights/*.csv",                                           
]
ORIGIN = "JFK"
DEST = "LAX"
OUTPUT_DIR = Path("./output")

MESES_PT: Dict[int, str] = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril",
    5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto",
    9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
}

# Utils/Helpers
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

def find_first_existing_glob(candidates: List[str]) -> List[str]:
    """Return sorted files from the first glob that yields results."""
    for pattern in candidates:
        files = sorted(glob.glob(pattern))
        if files:
            logging.info(f"Usando glob: {pattern} ({len(files)} arquivos).")
            return files
        else:
            logging.info(f"Nenhum arquivo encontrado em: {pattern}")
    # Maybe a class for this? Custom exception
    # Other stuff might apply too
    raise FileNotFoundError("Nenhum CSV encontrado nos padrões definidos em DEFAULT_PATHS.")

def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

# Core pipeline
def load_flights(files: List[str]) -> dd.DataFrame:
    """Load minimal columns with safe dtypes and missing handling."""
    cols = ["Year", "Month", "Origin", "Dest", "DepDelay", "Cancelled"]
    dtypes = {
        "Year": "Int64",          
        "Month": "Int64",
        "Origin": "object",
        "Dest": "object",
        "DepDelay": "float64",    # tlz tenha NaN
        "Cancelled": "Int8",      # 0/1/NaN
    }
    df = dd.read_csv(
        files,
        usecols=cols,
        na_values=["NA", ""],
        assume_missing=True,
        dtype=dtypes,
    )
    return df

def compute_max_depdelay(df: dd.DataFrame) -> float:
    """Largest departure delay across all files."""
    if "DepDelay" not in df.columns:
        raise KeyError("Coluna 'DepDelay' não encontrada.")
    max_delay = df["DepDelay"].max().compute()
    return float(max_delay)

def jfk_lax_monthly_counts(df: dd.DataFrame, split_out: int) -> pd.DataFrame:
    """Filter JFK→LAX (non-cancelled) and count YearxMonth."""
    #NaN as not cancelled 
    cancelled = df["Cancelled"].fillna(0)
    mask = (df["Origin"] == ORIGIN) & (df["Dest"] == DEST) & (cancelled == 0)
    df_route = df.loc[mask, ["Year", "Month"]]

    counts = (
        df_route
        .groupby(["Year", "Month"])
        .size(split_out=split_out)
        .rename("num_flights")
    )
    pdf = counts.reset_index().compute()

    # Month names as ordered categorical
    pdf["MonthName"] = pd.Categorical(
        pdf["Month"].map(MESES_PT), # PT 
        categories=[MESES_PT[m] for m in range(1, 12 + 1)],
        ordered=True,
    )
    pdf = pdf.sort_values(["Year", "Month"])
    return pdf

def plot_heatmap(pdf_counts: pd.DataFrame, out_path: Path) -> None:
    """Year as rows, MonthName as columns; cells = counts (single figure, default colors)."""
    if pdf_counts.empty:
        logging.warning("Sem dados para plot; pulando heatmap.")
        return
    pivot = (
        pdf_counts
        .pivot(index="Year", columns="MonthName", values="num_flights")
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(pivot.values, aspect="auto") 
    ax.set_title("JFK -> LAX • Número de voos por mês e ano")
    ax.set_xlabel("Mês")
    ax.set_ylabel("Ano")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(list(pivot.columns), rotation=45, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(list(pivot.index))
    fig.colorbar(im, ax=ax, label="nº de voos")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info(f"Heatmap salvo em: {out_path}")

def summarize_top_month(pdf_counts: pd.DataFrame) -> dict:
    """Return dict with top MonthName (sum across years) and total flights."""
    if pdf_counts.empty:
        return {"top_month_name": None, "top_month_total_flights": 0}
    monthly_totals = (
        pdf_counts.groupby("MonthName", observed=True)["num_flights"]
        .sum()
        .sort_values(ascending=False)
    )
    top_name = str(monthly_totals.index[0])
    top_value = int(monthly_totals.iloc[0])
    return {"top_month_name": top_name, "top_month_total_flights": top_value}

# Main
    setup_logging()
    ensure_output_dir(OUTPUT_DIR)

    # Shuffle. I need to read more about this.. is this equal Spark shuffle? How dask specifically bypass pythons limitations?
    dask.config.set({"dataframe.shuffle.method": "tasks"})

    # Discover files
    files = find_first_existing_glob(DEFAULT_PATHS)

    # Distributed execution (LocalCluster.. it seems Dask have a helm chart!)
    with Client() as client:
        n_workers = len(client.ncores())
        # A bit arbitrary, but it works
        split_out = max(8, n_workers * 2)

        # Load once, then persist for reuse
        df = load_flights(files)
        df = df.persist()  # cache in cluster.. avoids double IO

        # 1) Maior DepDelay (todos os voos)
        max_delay = compute_max_depdelay(df)
        logging.info(f"Maior DepDelay (todos os arquivos): {max_delay}")

        # 2) JFK->LAX: contagem AnoxMês
        pdf_counts = jfk_lax_monthly_counts(df, split_out=split_out)

    #  artifacts
    counts_csv = OUTPUT_DIR / "jfk_lax_counts.csv"
    pdf_counts.to_csv(counts_csv, index=False)
    logging.info(f"Contagens salvas em: {counts_csv}")

    chart_path = OUTPUT_DIR / "jfk_lax_heatmap.png"
    plot_heatmap(pdf_counts, chart_path)

    top_info = summarize_top_month(pdf_counts)

    summary = {
        "origin": ORIGIN,
        "dest": DEST,
        "max_depdelay": None if np.isnan(max_delay) else float(max_delay),
        "top_month_name": top_info["top_month_name"],
        "top_month_total_flights": top_info["top_month_total_flights"],
        "counts_csv": str(counts_csv),
        "chart_path": str(chart_path),
    }
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    logging.info(f"Resumo salvo em: {summary_path}")

    # Answers 
    print(f"Maior DepDelay (todos os voos): {summary['max_depdelay']}")
    if summary["top_month_name"]:
        print(f"Mês com mais voos (JFK→LAX; total no período): "
              f"{summary['top_month_name']} — {summary['top_month_total_flights']} voos")
    else:
        print("Sem dados suficientes para determinar o mês com mais voos (JFK→LAX).")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
