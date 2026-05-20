"""
Lab 2 — exploratory checks for the Distribuidora BR acquisition case.

Run: python tests.py

## Reconstruction (finance + ML framing)

Naive sum BR+adquirida is a *baseline counterfactual*, not consolidation
accounting. Stronger options implemented here:

1. **Multiplicative pro-forma (scale alignment)**
   Same identity y ≈ m·(BR+adq) before T, with m chosen so that a *post-regime*
   scale matches a *pre-regime* scale (ratio of trimmed means over chosen
   windows). Interpret m as “one constant bundle of unobserved consolidation
   adjustments (eliminations, reclass, synergies) + measurement bias”.

2. **Month-of-year calibration (heterogeneous scaling)**
   For each calendar month, m_k = median(consolidated in month k, stable post
   years) / median(raw sum in month k, stable pre years). Pre history is
   rebuilt as s_t · m_{month(t)}. This is closer to *reconciling seasonal
   shapes* across regimes without assuming time-homogeneous additivity.

Both remain *proxies*: validate with temporal CV + MASE vs Seasonal Naive,
and compare to “post-merger only” / intervention models in the notebook.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import adfuller, kpss
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install statsmodels: pip install statsmodels\n" + str(e)) from e

LAB_DIR = Path(__file__).resolve().parent
MAIN_PATH = LAB_DIR / "distribr_serie.txt"
ACQ_PATH = LAB_DIR / "distribr_adquirida.txt"

# Assignment narrative: consolidation from this month onward (check jump in série principal).
ACQUISITION_START = pd.Timestamp("2019-07-01")


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    main = pd.read_csv(MAIN_PATH, parse_dates=["data"]).sort_values("data")
    acq = pd.read_csv(ACQ_PATH, parse_dates=["data"]).sort_values("data")
    main = main.rename(columns={"valor": "br_publicado"})
    acq = acq.rename(columns={"valor": "adquirida_separada"})
    return main, acq


def merged_panel(main: pd.DataFrame, acq: pd.DataFrame) -> pd.DataFrame:
    """Timeline alinhada: BR, adquirida e soma bruta pré-consolidação."""
    merged = main.merge(acq, on="data", how="left")
    merged["soma_bruta"] = merged["br_publicado"] + merged["adquirida_separada"].fillna(0.0)
    return merged


def reconciled_pre_merger(main: pd.DataFrame, acq: pd.DataFrame) -> pd.DataFrame:
    """Legacy column name: pré = soma ingénua, pós = série publicada consolidada."""
    merged = merged_panel(main, acq)
    merged["reconstruido_ou_consolidado"] = np.where(
        merged["data"] < ACQUISITION_START,
        merged["soma_bruta"],
        merged["br_publicado"],
    )
    return merged


def reconstruct_multiplicative_scale(
    df: pd.DataFrame,
    *,
    pre_year_max: int = 2018,
    post_year_min: int = 2021,
    trim: float = 0.1,
) -> pd.Series:
    """
    y_t = m * (BR+adq)_t for t < T; y_t = consolidado for t >= T.

    m = robust central tendency(consolidado | post stable) /
        robust central tendency(soma_bruta | pre stable).

    Exclui 2019 (transição) e 2020 (choque) por defeito nos extratos de média.
    """
    pre_mask = (df["data"] < ACQUISITION_START) & (df["data"].dt.year <= pre_year_max)
    post_mask = (df["data"] >= ACQUISITION_START) & (df["data"].dt.year >= post_year_min)
    num = df.loc[post_mask, "br_publicado"].to_numpy(dtype=float)
    den = df.loc[pre_mask, "soma_bruta"].to_numpy(dtype=float)
    num = num[np.isfinite(num)]
    den = den[np.isfinite(den) & (den > 0)]
    if len(num) < 6 or len(den) < 6:
        return pd.Series(np.nan, index=df.index)

    def _trimmed_mean(a: np.ndarray) -> float:
        lo, hi = np.quantile(a, [trim, 1.0 - trim])
        b = a[(a >= lo) & (a <= hi)]
        return float(b.mean()) if len(b) else float(a.mean())

    m = _trimmed_mean(num) / _trimmed_mean(den)
    out = df["br_publicado"].astype(float).copy()
    out.loc[df["data"] < ACQUISITION_START] = m * df.loc[
        df["data"] < ACQUISITION_START, "soma_bruta"
    ].astype(float)
    return out.rename("recon_multiplicativa")


def reconstruct_monthly_ratio(
    df: pd.DataFrame,
    *,
    pre_year_max: int = 2018,
    post_year_min: int = 2021,
) -> tuple[pd.Series, pd.Series]:
    """
    For each calendar month k, r_k = median(consolidado_post em k) / median(soma_pré em k).

    Reconstrução pré: y_t = soma_bruta_t * r_{month(t)}. Pós: consolidado.

    Returns (reconstructed_series, ratio_by_month_index_1..12).
    """
    pre = df["data"] < ACQUISITION_START
    post = df["data"] >= ACQUISITION_START
    ratios = pd.Series(index=range(1, 13), dtype=float)
    for k in range(1, 13):
        mp = pre & (df["data"].dt.month == k) & (df["data"].dt.year <= pre_year_max)
        mo = post & (df["data"].dt.month == k) & (df["data"].dt.year >= post_year_min)
        sp = df.loc[mp, "soma_bruta"]
        co = df.loc[mo, "br_publicado"]
        if len(sp) < 2 or len(co) < 2:
            ratios[k] = np.nan
            continue
        ratios[k] = float(np.median(co)) / float(np.median(sp))

    filled = ratios.copy()
    pre_stable = pre & (df["data"].dt.year <= pre_year_max)
    post_stable = post & (df["data"].dt.year >= post_year_min)
    den_med = float(np.median(df.loc[pre_stable, "soma_bruta"].astype(float)))
    num_med = float(np.median(df.loc[post_stable, "br_publicado"].astype(float)))
    _m_fallback = num_med / den_med if den_med > 0 else np.nan
    overall = reconstruct_multiplicative_scale(
        df, pre_year_max=pre_year_max, post_year_min=post_year_min
    )
    _m = float(
        np.nanmedian(
            overall.loc[pre].to_numpy() / df.loc[pre, "soma_bruta"].to_numpy().astype(float)
        )
    )
    if not np.isfinite(_m):
        _m = _m_fallback
    filled = filled.fillna(_m)

    recon = df["br_publicado"].astype(float).copy()
    for k in range(1, 13):
        mk = pre & (df["data"].dt.month == k)
        recon.loc[mk] = df.loc[mk, "soma_bruta"].astype(float) * float(filled[k])
    return recon.rename("recon_mensal"), filled


def continuity_jump_jun_to_jul(df: pd.DataFrame, y: pd.Series) -> float:
    """|y(jul consolidado) - y(jun recon)| na linha de tempo unificada."""
    j0 = df["data"] == pd.Timestamp("2019-06-01")
    j1 = df["data"] == pd.Timestamp("2019-07-01")
    if not j0.any() or not j1.any():
        return float("nan")
    return float(abs(float(y.loc[j1].iloc[0]) - float(y.loc[j0].iloc[0])))


def adf_report(series: pd.Series, title: str, alpha: float = 0.05) -> None:
    s = series.dropna()
    adf_stat, adf_p, *_ = adfuller(s, autolag="AIC")
    print(f"\n--- ADF: {title} (n={len(s)}) ---")
    print(f"  statistic={adf_stat:.4f}, p-value={adf_p:.4f}")
    print(
        "  conclusion:",
        "reject unit root (nível ~estacionário)"
        if adf_p < alpha
        else "não rejeita raiz unitária (série pode ser não estacionária em nível)",
    )


def kpss_report(series: pd.Series, title: str, alpha: float = 0.05) -> None:
    s = series.dropna()
    kpss_stat, kpss_p, *_ = kpss(s, regression="c", nlags="auto")
    print(f"\n--- KPSS (level / constant): {title} (n={len(s)}) ---")
    print(f"  statistic={kpss_stat:.4f}, p-value={kpss_p:.4f}")
    print(
        "  conclusion:",
        "rejeita estacionariedade em nível (evidência de tendência)"
        if kpss_p < alpha
        else "não rejeita estacionariedade em nível",
    )


def month_of_year_cv_rmse(
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    min_train_months: int = 36,
    horizon: int = 12,
) -> float:
    """
    Esboço de validação temporal: em cada origem, prevê h=1..horizon com
    Seasonal Naive (valor do mesmo mês no ano anterior). Retorna RMSE médio
    sobre todos os passos — útil para comparar *estratégias de série* com o
    mesmo esquema de CV (não substitui o MASE oficial do trabalho).
    """
    n = len(y)
    errors: list[float] = []
    t = min_train_months
    while t + horizon <= n:
        train_y = y[:t]
        train_dates = dates[:t]
        for h in range(1, horizon + 1):
            idx = t + h - 1
            target_date = dates[idx]
            lag_date = target_date - pd.DateOffset(years=1)
            cand = np.where(train_dates == lag_date)[0]
            if cand.size == 0:
                continue
            pred = train_y[cand[-1]]
            errors.append(float(y[idx] - pred))
        t += 1
    if not errors:
        return float("nan")
    e = np.array(errors)
    return float(np.sqrt(np.mean(e**2)))


def main() -> None:
    main_df, acq_df = load_frames()
    full = reconciled_pre_merger(main_df, acq_df)

    br = full["br_publicado"]
    recon = full["reconstruido_ou_consolidado"]
    dates = pd.DatetimeIndex(full["data"])

    pre = full["data"] < ACQUISITION_START
    post = ~pre

    print("=== Janelas ===")
    print(
        f"BR publicado: {full['data'].min().date()} .. {full['data'].max().date()} (n={len(full)})"
    )
    print(
        f"Adquirida (separada): {acq_df['data'].min().date()} .. {acq_df['data'].max().date()} (n={len(acq_df)})"
    )
    print(f"Pós-aquisição (série principal = consolidada a partir de): {ACQUISITION_START.date()}")

    print("\n=== Níveis: média e desvio antes vs depois (série BR publicada) ===")
    for label, mask in [("pré-aquisição", pre.values), ("pós-aquisição", post.values)]:
        seg = br.values[mask]
        print(f"  {label}: mean={seg.mean():.2f}, std={seg.std():.2f}, n={len(seg)}")

    jun_2019 = full.loc[full["data"] == pd.Timestamp("2019-06-01"), "br_publicado"]
    jul_2019 = full.loc[full["data"] == pd.Timestamp("2019-07-01"), "br_publicado"]
    jun_acq = full.loc[full["data"] == pd.Timestamp("2019-06-01"), "adquirida_separada"]
    if not jun_2019.empty and not jul_2019.empty:
        print("\n=== Checagem informal pico jul/2019 ===")
        print(f"  BR jun/2019: {float(jun_2019.iloc[0]):.2f}")
        print(
            f"  Adquirida jun/2019: {float(jun_acq.iloc[0]):.2f}"
            if not jun_acq.empty
            else "  Adquirida jun/2019: (sem dado)"
        )
        print(
            f"  Soma jun/2019 (aprox.): {float(jun_2019.iloc[0]) + (float(jun_acq.iloc[0]) if not jun_acq.empty else 0):.2f}"
        )
        print(f"  BR jul/2019 (consolidado): {float(jul_2019.iloc[0]):.2f}")

    print("\n=== Sazonalidade (desvio por mês, série BR pós-aquisição) ===")
    post_df = full.loc[post, ["data", "br_publicado"]].copy()
    post_df["mes"] = post_df["data"].dt.month
    by_m = post_df.groupby("mes")["br_publicado"].agg(["mean", "std", "count"])
    print(by_m.to_string())

    print("\n\n### Reconstrução: soma ingénua vs calibrada (finance / ML de fusão) ###")
    print(
        "Nota: escalas são estimadas só com anos estáveis "
        "(pré <= 2018, pós >= 2021) para reduzir viés de 2019 e COVID abrupto."
    )
    recon_mult = reconstruct_multiplicative_scale(full)
    recon_mo, ratios_k = reconstruct_monthly_ratio(full)

    for label, s in [
        ("Soma ingénua → consolidado", recon),
        ("Multiplicativa global (trimmed mean scale)", recon_mult),
        ("Razão por mês civil (k=1..12)", recon_mo),
    ]:
        s_idx = s.copy()
        s_idx.index = full.index
        j = continuity_jump_jun_to_jul(full, s_idx)
        print(f"\n  [{label}]")
        print(f"    |Δ| jun→jul 2019 (continuidade visual): {j:.4f}")

    print("\n  Razões m_k (median pós / median pré por mês) — índice 1=jan … 12=dez:")
    print(ratios_k.to_string())

    # Unit roots on comparable transforms — interpret with care after structural breaks.
    print(
        "\n\n### Testes de raiz unitária (complementares; ver texto da resposta sobre limites) ###"
    )
    adf_report(br, "BR publicado — série completa")
    kpss_report(br, "BR publicado — série completa")
    adf_report(br.loc[post], "BR publicado — apenas pós-aquisição")
    kpss_report(br.loc[post], "BR publicado — apenas pós-aquisição")
    adf_report(recon, "Recon: soma ingénua pré + consolidado pós")
    kpss_report(recon, "Recon: soma ingénua pré + consolidado pós")
    adf_report(recon_mult, "Recon: multiplicativa global pré + consolidado pós")
    adf_report(recon_mo, "Recon: razão mensal pré + consolidado pós")

    print("\n\n=== Esboço CV: RMSE multi-step com Seasonal Naive (comparar definições de y_t) ===")
    candidates: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("BR publicado (série completa)", br.values, np.ones(len(full), dtype=bool)),
        ("BR apenas pós-aquisição", br.values, post.values),
        ("Recon soma ingénua", recon.values, np.ones(len(full), dtype=bool)),
        ("Recon multiplicativa global", recon_mult.values, np.ones(len(full), dtype=bool)),
        ("Recon razão por mês civil", recon_mo.values, np.ones(len(full), dtype=bool)),
    ]
    for name, series_vals, mask in candidates:
        dts = dates[mask]
        y = series_vals[mask]
        rmse = month_of_year_cv_rmse(y, dts, min_train_months=36, horizon=12)
        print(f"  {name}: RMSE (Seasonal Naive CV) ≈ {rmse:.4f}")


if __name__ == "__main__":
    main()
