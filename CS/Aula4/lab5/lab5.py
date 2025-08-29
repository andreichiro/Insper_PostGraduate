"""
EDA + Modelagem — Bike Sharing (UCI)
Autor: Adaptado para André Ichiro (estrutura, princípios e sintaxe preservados)
Data: 22/08/2025

Objetivo:
- EDA completa e modelagem preditiva de contagem (hourly e daily).
- Gráficos com matplotlib (1 por figura), explicações no console.
- Divisão temporal (2011->train, 2012->test), métricas robustas.

Algoritmos:
- GLM (Poisson e Negative Binomial) p/ dados de contagem, via statsmodels.
- Random Forest Regressor (sklearn).
- HistGradientBoostingRegressor (sklearn) como substituto “GBDT de alta velocidade”.

Bibliotecas (restritas):
- numpy, pandas, matplotlib, statsmodels, scikit-learn.

Fonte do dataset:
- UCI Bike Sharing (hour.csv, day.csv) — baixado do zip oficial.

Observação:
- Denormaliza temp/atemp/hum/windspeed p/ unidades humanas (°C, %, km/h),
  mas mantém as versões normalizadas para modelagem quando útil.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Tuple
import sys
import zipfile
import urllib.request
import logging
import warnings

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

# Logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

# Config
@dataclass(slots=True, frozen=True)
class Config:
    # Link do zip
    DATA_ZIP_URL: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
    BASE_DIR: Path = Path(__file__).resolve().parent
    OUT_DIR: Path = Path(__file__).resolve().parent / "bike_out"
    FIG_DIR: Path = OUT_DIR / "figs"
    CLEAN_DIR: Path = OUT_DIR / "clean"
    RANDOM_STATE: int = 20250822

    # Split temporal: 2011 (train) -> 2012 (test)
    TEST_YEAR: int = 2012

    # Plots
    DPI_FIG: int = 140
    DPI_SAVE: int = 200

CFG = Config()
CFG.OUT_DIR.mkdir(parents=True, exist_ok=True)
CFG.FIG_DIR.mkdir(parents=True, exist_ok=True)
CFG.CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# Estilo matplotlib
def _set_matplotlib_style() -> None:
    plt.rcParams.update({
        "figure.dpi": CFG.DPI_FIG,
        "savefig.dpi": CFG.DPI_SAVE,
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

# Registro de exercícios
class _RegistroExercicios:
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
    def __new__(mcls, name, bases, ns, **kwargs):
        cls = super().__new__(mcls, name, bases, ns)
        numero = ns.get("numero")
        if numero is not None:
            _RegistroExercicios.registrar(numero, cls)
        return cls

class Exercicio(metaclass=_ExercicioMeta):
    numero: ClassVar[int]
    def executar(self, ctx: "Contexto") -> None:
        raise NotImplementedError

@dataclass(slots=True)
class Contexto:
    day: pd.DataFrame
    hour: pd.DataFrame
    hour_model: pd.DataFrame  # features já enriquecidas (hourly)

# Utils
def _safe_to_csv(df: pd.DataFrame, path: Path, desc: str, index: bool = False) -> None:
    try:
        df.to_csv(path, index=index, encoding="utf-8-sig")
        print(f"{desc}: {path}")
    except Exception as e:
        logger.warning(f"Falha ao salvar {desc} em {path}: {e}")

def _download_zip(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        logger.info("ZIP já existe, pulando download: %s", dest)
        return dest
    logger.info("Baixando ZIP do UCI: %s", url)
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    dest.write_bytes(data)
    logger.info("ZIP salvo: %s (%.1f MB)", dest, dest.stat().st_size / (1024**2))
    return dest

def _read_from_zip(zip_path: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as f:
            return pd.read_csv(f)

def _slug(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9_]+", "_", str(s).strip().lower()).strip("_")

def _denorm_temp(x: pd.Series, tmin: float, tmax: float) -> pd.Series:
    return x * (tmax - tmin) + tmin

def _denorm_to_pct(x: pd.Series, maxv: float) -> pd.Series:
    return x * maxv

def _denorm_wind_kmh(x: pd.Series, max_m_s: float) -> pd.Series:
    # UCI hum: divided by 100; windspeed divided by 67 (m/s). Convert to km/h.
    m_s = x * max_m_s
    return m_s * 3.6

def _time_aware_split(df_hour: pd.DataFrame, test_year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr = df_hour[df_hour["yr"] < (test_year - df_hour["dteday"].dt.year.min())]
    te = df_hour[df_hour["yr"] >= (test_year - df_hour["dteday"].dt.year.min())]
    return tr.copy(), te.copy()

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _mape_safe(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    y = np.array(y_true, dtype=float)
    yhat = np.array(y_pred, dtype=float)
    denom = np.clip(np.abs(y), eps, None)
    return float(np.mean(np.abs(y - yhat) / denom) * 100.0)

def _set_dt_index(df: pd.DataFrame, dcol: str, hcol: Optional[str] = None) -> pd.DatetimeIndex:
    if hcol is None:
        return pd.to_datetime(df[dcol], errors="coerce")
    return pd.to_datetime(df[dcol], errors="coerce") + pd.to_timedelta(df[hcol], unit="h")

# Helpers (load, enriquecimento e limpeza)
def carregar_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    zip_path = _download_zip(CFG.DATA_ZIP_URL, CFG.OUT_DIR / "Bike-Sharing-Dataset.zip")
    day = _read_from_zip(zip_path, "day.csv")
    hour = _read_from_zip(zip_path, "hour.csv")

    # Tipos e normalização básica
    for df in (day, hour):
        # datas e categorias
        df["dteday"] = pd.to_datetime(df["dteday"], errors="coerce")
        cat_cols = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]
        for c in cat_cols:
            if c in df.columns:
                df[c] = df[c].astype("int64", errors="ignore")
        # contagens inteiras
        for c in ["casual", "registered", "cnt"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # Enriquecimentos interpretáveis
    # UCI doc: temp in °C normalized: (t - t_min) / (t_max - t_min), t_min=-8, t_max=+39 (hourly)
    # atemp in °C normalized: (t - t_min)/(t_max - t_min), t_min=-16, t_max=+50 (hourly)
    # hum: divided by 100; windspeed: divided by 67 (m/s). (Denormalizamos p/ leitura)
    def enrich(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "temp" in out:
            out["temp_c"] = _denorm_temp(out["temp"].astype(float), -8.0, 39.0)
        if "atemp" in out:
            out["atemp_c"] = _denorm_temp(out["atemp"].astype(float), -16.0, 50.0)
        if "hum" in out:
            out["hum_pct"] = _denorm_to_pct(out["hum"].astype(float), 100.0)
        if "windspeed" in out:
            out["wind_kmh"] = _denorm_wind_kmh(out["windspeed"].astype(float), 67.0)
        # mapeamentos rótulos úteis
        season_map = {1: "winter", 2: "spring", 3: "summer", 4: "fall"}
        we_map = {
            1: "clear",
            2: "mist_cloudy",
            3: "light_snow_rain",
            4: "heavy_rain_snow",
        }
        out["season_lbl"] = out.get("season", pd.Series([np.nan]*len(out))).map(season_map)
        out["weathersit_lbl"] = out.get("weathersit", pd.Series([np.nan]*len(out))).map(we_map)
        # auxiliares temporais
        out["dow"] = out["weekday"] if "weekday" in out else out["dteday"].dt.dayofweek
        out["month"] = out["mnth"] if "mnth" in out else out["dteday"].dt.month
        if "hr" in out:
            out["hour"] = out["hr"]
        # índices cíclicos (evitar descontinuidade 23->0)
        if "month" in out:
            out["mon_sin"] = np.sin(2*np.pi*(out["month"].astype(float)/12.0))
            out["mon_cos"] = np.cos(2*np.pi*(out["month"].astype(float)/12.0))
        if "hr" in out:
            out["hr_sin"] = np.sin(2*np.pi*(out["hr"].astype(float)/24.0))
            out["hr_cos"] = np.cos(2*np.pi*(out["hr"].astype(float)/24.0))
        # picos de commute (proxy de padrão)
        if "hr" in out:
            out["is_peak"] = ((out["hr"].isin([7,8,9,17,18,19]))).astype(int)
        return out

    day_e = enrich(day)
    hour_e = enrich(hour)

    return day_e, hour_e

def preparar_model_hour(hour: pd.DataFrame) -> pd.DataFrame:
    # remove colunas proibidas ou alvo-dependentes
    df = hour.copy()
    # alvo
    df["cnt"] = df["cnt"].astype(float)
    # não usar componentes do alvo como features
    for c in ("casual", "registered", "instant"):
        if c in df.columns:
            df.drop(columns=c, inplace=True)
    # garantia de target válido e datas válidas
    df = df.dropna(subset=["cnt", "dteday"]).copy()

    # colunas categóricas canônicas
    cat_cols = []
    for c in ["season_lbl", "weathersit_lbl", "holiday", "workingday", "dow", "month", "hr"]:
        if c in df.columns:
            cat_cols.append(c)

    # numéricas úteis
    num_cols = []
    for c in ["temp", "atemp", "hum", "windspeed", "temp_c", "atemp_c", "hum_pct", "wind_kmh",
              "hr_sin", "hr_cos", "mon_sin", "mon_cos", "is_peak"]:
        if c in df.columns:
            num_cols.append(c)

    # guardamos colunas listas para pipeline posterior
    df.attrs["cat_cols"] = cat_cols
    df.attrs["num_cols"] = num_cols

    return df

# Exercícios
class Ex01CarregarLimpar(Exercicio):
    numero: ClassVar[int] = 1
    def executar(self, ctx: Contexto) -> None:
        print("[E1] Linhas (daily, hourly):", len(ctx.day), len(ctx.hour))
        print("[E1] Colunas (hourly):", list(ctx.hour.columns))
        # Missingness (esperado ~none pela doc; confirmamos)
        miss_hour = ctx.hour.isna().mean().sort_values(ascending=False)
        print("[E1] Max % faltantes (hourly):", f"{100*float(miss_hour.max()):.2f}%")
        # Gráfico: % faltantes (hourly)
        _set_matplotlib_style()
        plt.figure(figsize=(7.2, 4.2))
        miss_hour.mul(100.0).plot(kind="bar")
        plt.ylabel("% faltantes")
        plt.title("Missingness por coluna (hourly)")
        plt.tight_layout()
        out = CFG.FIG_DIR / "e1_missingness_hour.png"
        plt.savefig(out); plt.close()
        print(f"[E1] Figura: {out}")

class Ex02SazonalidadeSeries(Exercicio):
    numero: ClassVar[int] = 2
    def executar(self, ctx: Contexto) -> None:
        # Serie diária — tendência e sazonalidade calendário
        day = ctx.day.sort_values("dteday").copy()
        day["dt_index"] = _set_dt_index(day, "dteday")
        day = day.set_index("dt_index")
        day["cnt"] = day["cnt"].astype(float)

        # Smooth (media movel 7d)
        day["cnt_7dma"] = day["cnt"].rolling(7, min_periods=1).mean()

        _set_matplotlib_style()
        plt.figure(figsize=(8.8, 4.2))
        plt.plot(day.index, day["cnt"], lw=0.6, alpha=0.5, label="Diário")
        plt.plot(day.index, day["cnt_7dma"], lw=1.2, label="7d MA")
        plt.title("Contagem diária de bikes — tendência (2011–2012)")
        plt.ylabel("cnt (bikes/dia)")
        plt.legend()
        plt.tight_layout()
        out = CFG.FIG_DIR / "e2_daily_trend.png"
        plt.savefig(out); plt.close()
        print(f"[E2] Figura: {out}")
        
        # Perfil por hora × dia-da-semana (média)
        hour = ctx.hour.copy()

        # Garante que 'cnt' é float e que não carregamos extension dtypes p/ o pivot
        hour["cnt"] = hour["cnt"].astype(float)

        prof = (hour.groupby(["dow", "hr"])["cnt"]
                      .mean()
                      .reset_index()
                      .pivot(index="hr", columns="dow", values="cnt"))

        # Reindexa para garantir grade completa (0–23 horas, 0–6 weekday) e float puro
        prof = (prof
                .reindex(index=range(24))           # horas 0..23
                .reindex(columns=range(7))           # dow 0..6
                .astype(float))                      # <- chave: tira extension dtypes

        # Converte para ndarray float e mascara NaNs para o imshow
        arr = prof.to_numpy(dtype=float)
        arr = np.ma.masked_invalid(arr)

        _set_matplotlib_style()
        plt.figure(figsize=(8.8, 4.8))
        im = plt.imshow(arr, aspect="auto", origin="lower")
        plt.colorbar(im, fraction=0.046, pad=0.04, label="cnt médio")
        plt.xticks(ticks=np.arange(prof.shape[1]), labels=[str(c) for c in prof.columns])
        plt.yticks(ticks=np.arange(prof.shape[0]), labels=[str(r) for r in prof.index])
        plt.xlabel("weekday (0=Dom)")
        plt.ylabel("hour")
        plt.title("Heatmap cnt médio — hour × weekday")
        plt.tight_layout()
        out = CFG.FIG_DIR / "e2_hour_weekday_heatmap.png"
        plt.savefig(out); plt.close()
        print(f"[E2] Figura: {out}")

class Ex03TempoClimaRelacoes(Exercicio):
    numero: ClassVar[int] = 3
    def executar(self, ctx: Contexto) -> None:
        df = ctx.hour.copy()
        # Scatter atemp_c vs cnt (hexbin reduz sobreposição)
        _set_matplotlib_style()
        plt.figure(figsize=(6.8,4.6))
        plt.hexbin(df["atemp_c"], df["cnt"].astype(float), gridsize=40, mincnt=1)
        plt.xlabel("atemp (°C, 'feels like')")
        plt.ylabel("cnt (bikes/h)")
        plt.title("Demanda × sensação térmica (hourly)")
        cb = plt.colorbar()
        cb.set_label("n pontos")
        plt.tight_layout()
        out = CFG.FIG_DIR / "e3_hex_atemp_cnt.png"
        plt.savefig(out); plt.close()
        print(f"[E3] Figura: {out}")

        # Efeito weathersit (boxplot)
        _set_matplotlib_style()
        orders = ["clear","mist_cloudy","light_snow_rain","heavy_rain_snow"]
        grp = [df.loc[df["weathersit_lbl"]==lab, "cnt"].dropna().astype(float).values for lab in orders]
        plt.figure(figsize=(7.6,4.6))
        plt.boxplot(grp, showmeans=True, labels=orders)
        plt.ylabel("cnt (bikes/h)")
        plt.title("Demanda por condição climática (weathersit)")
        plt.tight_layout()
        out = CFG.FIG_DIR / "e3_box_cnt_weathersit.png"
        plt.savefig(out); plt.close()
        print(f"[E3] Figura: {out}")

class Ex04CorrelacoesNumericas(Exercicio):
    numero: ClassVar[int] = 4
    def executar(self, ctx: Contexto) -> None:
        df = ctx.hour.copy()
        numcols = ["cnt","temp","atemp","hum","windspeed","temp_c","atemp_c","hum_pct","wind_kmh"]
        numcols = [c for c in numcols if c in df.columns]
        corr = df[numcols].corr(method="pearson")
        print("[E4] Correlações com cnt:")
        for c in numcols:
            if c == "cnt": continue
            r = float(corr.loc["cnt", c])
            print(f"  r(cnt, {c}) = {r:.3f}")
        _set_matplotlib_style()
        plt.figure(figsize=(6.2,5.2))
        im = plt.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(numcols)), numcols, rotation=45, ha="right")
        plt.yticks(range(len(numcols)), numcols)
        plt.title("Matriz de correlação (hourly)")
        plt.tight_layout()
        out = CFG.FIG_DIR / "e4_corr_heatmap.png"
        plt.savefig(out); plt.close()
        print(f"[E4] Figura: {out}")

class Ex05GLM_CountModels(Exercicio):
    numero: ClassVar[int] = 5
    def executar(self, ctx: Contexto) -> None:
        # GLM: Poisson e NegBin (overdispersion expected)
        df = ctx.hour_model.copy()
        # treino 2011, teste 2012
        tr, te = _time_aware_split(df, CFG.TEST_YEAR)

        # Fórmula com efeitos categóricos e algumas s‑features
        # Nota: evitar colinearidade forte temp vs atemp; preferimos atemp_c (interpretação)
        formula = "cnt ~ atemp_c + hum_pct + wind_kmh + C(weathersit_lbl) + C(season_lbl) + C(dow) + C(hr)"

        # Poisson
        m_pois = smf.glm(formula=formula, data=tr, family=sm.families.Poisson()).fit()
        print("[E5] GLM Poisson — AIC=", f"{float(m_pois.aic):.1f}")
        # Diagnóstico de sobredispersão
        mu_hat = m_pois.fittedvalues
        pearson = np.sum(((tr["cnt"] - mu_hat)**2 / np.clip(mu_hat, 1e-6, None)))
        df_resid = tr.shape[0] - int(m_pois.df_model) - 1
        disp = pearson / max(df_resid, 1)
        print(f"[E5] Poisson — razão de Pearson/df ≈ {disp:.2f} (>>1 indica sobredispersão)")

        # Negative Binomial
        m_nb = smf.glm(formula=formula, data=tr, family=sm.families.NegativeBinomial()).fit()
        print("[E5] GLM NegBin — AIC=", f"{float(m_nb.aic):.1f}")

        # Avaliação out‑of‑time (2012)
        def _eval(model, name: str) -> None:
            pred = model.predict(te)
            y = te["cnt"].to_numpy(float)
            rmse = _rmse(y, pred)
            mae  = mean_absolute_error(y, pred)
            r2   = r2_score(y, pred)
            mape = _mape_safe(y, pred)
            print(f"[E5] {name}: RMSE={rmse:.1f}  MAE={mae:.1f}  R2={r2:.3f}  MAPE~{mape:.1f}%")

        _eval(m_pois, "Poisson")
        _eval(m_nb,   "NegBin")

        # Gráfico: resíduos vs ajustados (NegBin)
        _set_matplotlib_style()
        plt.figure(figsize=(6.8,4.6))
        fit = m_nb.fittedvalues
        resid = m_nb.resid_pearson
        plt.scatter(fit, resid, s=8, alpha=0.5)
        plt.axhline(0, ls="--", lw=1)
        plt.xlabel("Ajustado (Treino, NegBin)")
        plt.ylabel("Resíduo de Pearson")
        plt.title("Resíduos × Ajustados — GLM NegBin (hourly)")
        plt.tight_layout()
        out = CFG.FIG_DIR / "e5_nb_residuals.png"
        plt.savefig(out); plt.close()
        print(f"[E5] Figura: {out}")

class Ex06ML_RandomForest_HGB(Exercicio):
    numero: ClassVar[int] = 6

    @staticmethod
    def _build_pipeline(cat_cols: List[str], num_cols: List[str], model) -> Pipeline:
        # Árvores toleram numéricas sem escalar; mantemos StandardScaler só p/ modelos lineares (não usados aqui).
        ct = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                ("num", "passthrough", num_cols)
            ],
            remainder="drop",
            verbose_feature_names_out=False
        )
        pipe = Pipeline(steps=[("prep", ct), ("model", model)])
        return pipe

    def executar(self, ctx: Contexto) -> None:
        df = ctx.hour_model.copy()
        tr, te = _time_aware_split(df, CFG.TEST_YEAR)

        y_tr = tr["cnt"].to_numpy(float)
        y_te = te["cnt"].to_numpy(float)
        cat_cols: List[str] = tr.attrs["cat_cols"]
        num_cols: List[str] = tr.attrs["num_cols"]

        models = {
            "RandomForest": RandomForestRegressor(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=CFG.RANDOM_STATE
            ),
            "HistGBR": HistGradientBoostingRegressor(
                max_depth=None,
                learning_rate=0.08,
                max_iter=600,
                l2_regularization=1.0,
                random_state=CFG.RANDOM_STATE
            )
        }

        results = []
        for name, base in models.items():
            pipe = self._build_pipeline(cat_cols, num_cols, base)
            pipe.fit(tr[cat_cols + num_cols], y_tr)
            pred = pipe.predict(te[cat_cols + num_cols])
            rmse = _rmse(y_te, pred)
            mae  = mean_absolute_error(y_te, pred)
            r2   = r2_score(y_te, pred)
            mape = _mape_safe(y_te, pred)
            results.append((name, rmse, mae, r2, mape))
            print(f"[E6] {name}: RMSE={rmse:.1f}  MAE={mae:.1f}  R2={r2:.3f}  MAPE~{mape:.1f}%")

            # Importância por permutação (mais estável) — top 15
            # Importância por permutação no ESPAÇO TRANSFORMADO (granular por dummy)
            prep = pipe.named_steps["prep"]
            est  = pipe.named_steps["model"]

            # X de teste TRANSFORMADO
            X_te_tr = prep.transform(te[cat_cols + num_cols])

            # Evita ruído do ResourceTracker no macOS → n_jobs=1 aqui
            pi = permutation_importance(
                est, X_te_tr, y_te,
                n_repeats=10,
                random_state=CFG.RANDOM_STATE,
                n_jobs=1  # <- evita "No child processes" do multiprocessing no macOS
                # opcional: scoring="neg_mean_squared_error" para alinhar com RMSE
            )

            # Nomes das colunas transformadas (OHE expandido + numéricas)
            try:
                feat_names = prep.get_feature_names_out()
            except Exception:
                # Fallback (raras versões antigas): reconstrói manualmente
                feat_names = []
                for name, trans, cols in prep.transformers_:
                    if name == "cat" and hasattr(trans, "get_feature_names_out"):
                        feat_names.extend(trans.get_feature_names_out(cols))
                    elif name == "num":
                        feat_names.extend(list(cols))

            importances = (
                pd.DataFrame({
                    "feature": list(feat_names),
                    "importance": pi.importances_mean
                })
                .sort_values("importance", ascending=False)
                .head(15)
            )

            _set_matplotlib_style()
            plt.figure(figsize=(7.6, 5.0))
            plt.barh(importances["feature"][::-1], importances["importance"][::-1])
            plt.xlabel("Δ score (perm. importance)")
            plt.title(f"Top 15 importâncias — {name}")
            plt.tight_layout()
            out = CFG.FIG_DIR / f"e6_importances_{_slug(name)}.png"
            plt.savefig(out); plt.close()
            print(f"[E6] Figura: {out}")

            # Série temporal (2012): real vs previsto (zoom mensal)
            te_dt = te.copy()
            te_dt["ts"] = _set_dt_index(te_dt, "dteday", "hr")
            plot = te_dt[["ts"]].copy()
            plot["y"] = y_te
            plot["yhat"] = pred
            plot = plot.sort_values("ts")
            _set_matplotlib_style()
            plt.figure(figsize=(9.6, 4.0))
            plt.plot(plot["ts"], plot["y"], lw=0.8, alpha=0.6, label="real")
            plt.plot(plot["ts"], plot["yhat"], lw=0.9, label="previsto")
            plt.title(f"2012 — Real vs Previsto ({name})")
            plt.ylabel("cnt (bikes/h)")
            plt.legend()
            plt.tight_layout()
            out = CFG.FIG_DIR / f"e6_ts_2012_{_slug(name)}.png"
            plt.savefig(out); plt.close()
            print(f"[E6] Figura: {out}")

        # Comparação de métricas entre modelos
        resdf = pd.DataFrame(results, columns=["model","RMSE","MAE","R2","MAPE"])
        _safe_to_csv(resdf, CFG.OUT_DIR / "e6_model_metrics.csv", desc="[E6] Métricas (2012)")
        _set_matplotlib_style()
        plt.figure(figsize=(7.6, 4.4))
        x = np.arange(len(resdf))
        plt.bar(x-0.2, resdf["RMSE"], width=0.4, label="RMSE")
        plt.bar(x+0.2, resdf["MAE"],  width=0.4, label="MAE")
        plt.xticks(x, resdf["model"].tolist())
        plt.ylabel("erro")
        plt.title("Comparação de erros — 2012 (menor = melhor)")
        plt.legend()
        plt.tight_layout()
        out = CFG.FIG_DIR / "e6_model_compare_errors.png"
        plt.savefig(out); plt.close()
        print(f"[E6] Figura: {out}")

class Ex07PDP_Parciais(Exercicio):
    numero: ClassVar[int] = 7
    def executar(self, ctx: Contexto) -> None:
        # PDP no melhor modelo da E6 (recarrega métricas)
        metrics = pd.read_csv(CFG.OUT_DIR / "e6_model_metrics.csv")
        best = metrics.sort_values("RMSE").iloc[0]["model"]
        print(f"[E7] Melhor por RMSE: {best}")

        df = ctx.hour_model.copy()
        tr, te = _time_aware_split(df, CFG.TEST_YEAR)
        y_tr = tr["cnt"].to_numpy(float)
        cat_cols: List[str] = tr.attrs["cat_cols"]
        num_cols: List[str] = tr.attrs["num_cols"]

        model = RandomForestRegressor(
            n_estimators=600,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=CFG.RANDOM_STATE
        ) if best == "RandomForest" else HistGradientBoostingRegressor(
            max_iter=600,
            learning_rate=0.08,
            l2_regularization=1.0,
            random_state=CFG.RANDOM_STATE
        )
        pipe = Ex06ML_RandomForest_HGB._build_pipeline(cat_cols, num_cols, model)
        pipe.fit(tr[cat_cols + num_cols], y_tr)

        # PDP de variáveis climáticas (interpretabilidade)
        feats = [f for f in ["atemp_c","hum_pct","wind_kmh"] if f in num_cols]
        if not feats:
            print("[E7] Sem variáveis climáticas numéricas disponíveis para PDP.")
            return
        _set_matplotlib_style()
        for f in feats:
            plt.figure(figsize=(6.6,4.4))
            try:
                PartialDependenceDisplay.from_estimator(
                    pipe, tr[cat_cols + num_cols], [f]
                )
            except Exception:
                # fallback simples: grid e médias preditas
                xs = np.linspace(tr[f].quantile(0.02), tr[f].quantile(0.98), 40)
                base = tr[cat_cols + num_cols].iloc[:1].copy()
                grid = pd.concat([base]*len(xs), ignore_index=True)
                grid[f] = xs
                yh = pipe.predict(grid)
                plt.plot(xs, yh)
                plt.xlabel(f); plt.ylabel("cnt pred")
            plt.title(f"Dependência parcial — {best} — {f}")
            plt.tight_layout()
            out = CFG.FIG_DIR / f"e7_pdp_{_slug(best)}_{_slug(f)}.png"
            plt.savefig(out); plt.close()
            print(f"[E7] Figura: {out}")

class Ex08AuditoriaAlgoritmos(Exercicio):
    numero: ClassVar[int] = 8
    def executar(self, ctx: Contexto) -> None:
        # Comentário programático (console) + gráfico simples de coerência
        items = [
            ("GLM Poisson", 5, "Dados de contagem; baseline estatístico."),
            ("GLM NegBin", 5, "Lida com sobredispersão; tipicamente melhor que Poisson aqui."),
            ("Random Forest", 4, "Captura não linearidades e interações; robusto."),
            ("HistGradientBoosting", 5, "GBDT moderno e rápido no sklearn."),
            ("Survival Analysis", 1, "Inadequado: faltam tempos individuais de evento."),
        ]
        print("[E8] Adequação (1–5):")
        for n, s, why in items:
            print(f"  {n}: {s}/5 — {why}")

        _set_matplotlib_style()
        plt.figure(figsize=(7.6, 4.2))
        names = [n for n,_,_ in items]
        scores = [s for _,s,_ in items]
        plt.bar(names, scores)
        plt.ylim(0,5)
        plt.ylabel("Adequação (1–5)")
        plt.title("Adequação por algoritmo — Bike Sharing")
        plt.tight_layout()
        out = CFG.FIG_DIR / "e8_algo_coherence.png"
        plt.savefig(out); plt.close()
        print(f"[E8] Figura: {out}")

class Ex09RelatorioResumo(Exercicio):
    numero: ClassVar[int] = 9
    def executar(self, ctx: Contexto) -> None:
        # Consolida principais achados em CSV curto (para automação)
        lines = []
        # Carrega métricas de E6 e resume
        try:
            met = pd.read_csv(CFG.OUT_DIR / "e6_model_metrics.csv")
            best = met.sort_values("RMSE").iloc[0].to_dict()
            lines.append(f"Melhor modelo (2012): {best['model']} | RMSE={best['RMSE']:.1f} | MAE={best['MAE']:.1f} | R2={best['R2']:.3f} | MAPE~{best['MAPE']:.1f}%")
        except Exception:
            lines.append("Métricas indisponíveis.")
        # Insights EDA (programático)
        hour = ctx.hour.copy()
        commute = hour[hour["is_peak"]==1]["cnt"].mean() if "is_peak" in hour else np.nan
        offpeak = hour[hour["is_peak"]==0]["cnt"].mean() if "is_peak" in hour else np.nan
        lines.append(f"Peak vs Off-peak (cnt médio): {commute:.0f} vs {offpeak:.0f}")

        out = CFG.OUT_DIR / "e9_resumo.txt"
        out.write_text("\n".join(lines), encoding="utf-8")
        print("[E9] Resumo salvo:", out)

# Wrappers (mesma simetria do seu script)
class _E1(Ex01CarregarLimpar): pass
class _E2(Ex02SazonalidadeSeries): pass
class _E3(Ex03TempoClimaRelacoes): pass
class _E4(Ex04CorrelacoesNumericas): pass
class _E5(Ex05GLM_CountModels): pass
class _E6(Ex06ML_RandomForest_HGB): pass
class _E7(Ex07PDP_Parciais): pass
class _E8(Ex08AuditoriaAlgoritmos): pass
class _E9(Ex09RelatorioResumo): pass

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    warnings.filterwarnings("ignore")
    _set_matplotlib_style()

    # Carregar, enriquecer e preparar
    try:
        day, hour = carregar_dataset()
    except Exception as e:
        print(f"Falha ao baixar/ler dataset UCI: {e}", file=sys.stderr)
        sys.exit(1)

    hour_model = preparar_model_hour(hour)

    # Persistência "clean"
    _safe_to_csv(day,  CFG.CLEAN_DIR / "day_clean.csv",  desc="[MAIN] day clean")
    _safe_to_csv(hour, CFG.CLEAN_DIR / "hour_clean.csv", desc="[MAIN] hour clean")

    ctx = Contexto(day=day, hour=hour, hour_model=hour_model)

    print(f"Dataset pronto. day={len(day):,} | hour={len(hour):,}")
    cols_info = {"cat": hour_model.attrs.get("cat_cols", []), "num": hour_model.attrs.get("num_cols", [])}
    print("[MAIN] Colunas categóricas:", cols_info["cat"])
    print("[MAIN] Colunas numéricas:", cols_info["num"])

    for ex in _RegistroExercicios.instancias_ordenadas():
        print(f"\nExercício {ex.numero:02d}")
        ex.executar(ctx)

if __name__ == "__main__":
    main()
