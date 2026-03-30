from __future__ import annotations

import calendar
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.base import clone
from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Configurações gerais do projeto
RANDOM_STATE = 42
LAGS = [1, 7, 14, 28]
ROLL_WINDOWS = [7, 14]


@dataclass
class ModelResult:
    name: str
    rmse: float
    mae: float
    mape: float


def nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    """Retorna a data do n-ésimo dia da semana em um mês (ex.: 4ª quinta-feira de novembro)."""
    month_cal = calendar.monthcalendar(year, month)
    weekday_days = [week[weekday] for week in month_cal if week[weekday] != 0]
    return pd.Timestamp(year=year, month=month, day=weekday_days[n - 1])


def last_weekday_of_month(year: int, month: int, weekday: int) -> pd.Timestamp:
    """Retorna a data da última ocorrência de um dia da semana em um mês."""
    month_cal = calendar.monthcalendar(year, month)
    weekday_days = [week[weekday] for week in month_cal if week[weekday] != 0]
    return pd.Timestamp(year=year, month=month, day=weekday_days[-1])


def obter_data_feriado(name: str, year: int, default_day: int, default_month: int) -> Optional[pd.Timestamp]:
    """Converte o nome do feriado em uma regra calendária apropriada para o ano."""
    nome = str(name).strip().lower()

    if "juneteenth" in nome and year < 2021:
        # Juneteenth virou feriado federal em 2021; antes disso não é marcado.
        return None

    if "martin luther king" in nome:
        return nth_weekday_of_month(year=year, month=1, weekday=0, n=3)  # 3ª segunda de janeiro
    if "george washington" in nome or "president" in nome:
        return nth_weekday_of_month(year=year, month=2, weekday=0, n=3)  # 3ª segunda de fevereiro
    if "memorial" in nome:
        return last_weekday_of_month(year=year, month=5, weekday=0)  # última segunda de maio
    if "labor" in nome:
        return nth_weekday_of_month(year=year, month=9, weekday=0, n=1)  # 1ª segunda de setembro
    if "columbus" in nome:
        return nth_weekday_of_month(year=year, month=10, weekday=0, n=2)  # 2ª segunda de outubro
    if "thanksgiving" in nome:
        return nth_weekday_of_month(year=year, month=11, weekday=3, n=4)  # 4ª quinta de novembro

    # Para feriados de data fixa
    return pd.Timestamp(year=year, month=default_month, day=default_day)


def classificar_tipo_feriado(name: str) -> str:
    """Classifica feriado em fixo ou móvel com base no nome."""
    nome = str(name).strip().lower()
    moveis = [
        "martin luther king",
        "george washington",
        "memorial",
        "labor",
        "columbus",
        "thanksgiving",
    ]
    if any(chave in nome for chave in moveis):
        return "movel"
    return "fixo"


def calcular_data_feriado_observado(data_feriado: pd.Timestamp, tipo_feriado: str) -> Optional[pd.Timestamp]:
    """Gera data observada para feriados fixos que caem no fim de semana."""
    if tipo_feriado != "fixo":
        return None

    # Regra federal típica dos EUA: sábado observa na sexta; domingo observa na segunda.
    if data_feriado.weekday() == 5:
        return data_feriado - pd.Timedelta(days=1)
    if data_feriado.weekday() == 6:
        return data_feriado + pd.Timedelta(days=1)
    return None


def preparar_diretorios(base_dir: Path) -> Dict[str, Path]:
    """Cria a estrutura de pastas para armazenar entregáveis."""
    out_dir = base_dir / "entregaveis"
    figs_dir = out_dir / "visualizacoes"
    tables_dir = out_dir / "tabelas"
    preds_dir = out_dir / "previsoes"

    for path in [out_dir, figs_dir, tables_dir, preds_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        "out": out_dir,
        "figs": figs_dir,
        "tables": tables_dir,
        "preds": preds_dir,
    }


def carregar_bases(base_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lê as bases de treino, teste e feriados a partir de arquivos Excel."""
    train = pd.read_excel(base_dir / "chicago_df.xlsx")
    test = pd.read_excel(base_dir / "chicago_df_test.xlsx")
    holiday = pd.read_excel(base_dir / "us_holiday.xlsx")

    for df in [train, test]:
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

    return train, test, holiday


def criar_calendario_feriados(holiday_df: pd.DataFrame, years: List[int]) -> pd.DataFrame:
    """Expande feriados por ano usando regras móveis (quando aplicável)."""
    registros: List[Dict[str, object]] = []

    for year in years:
        for _, row in holiday_df.iterrows():
            try:
                holiday_name = row["Name"]
                holiday_type = classificar_tipo_feriado(holiday_name)
                data_feriado = obter_data_feriado(
                    name=holiday_name,
                    year=year,
                    default_day=int(row["day"]),
                    default_month=int(row["month"]),
                )
                if data_feriado is not None:
                    registros.append({
                        "date": data_feriado,
                        "holiday_name": holiday_name,
                        "holiday_type": holiday_type,
                        "is_holiday_observed": 0,
                    })

                    # Adiciona data observada quando aplicável
                    data_observada = calcular_data_feriado_observado(data_feriado, holiday_type)
                    if data_observada is not None:
                        registros.append({
                            "date": data_observada,
                            "holiday_name": f"{holiday_name} (Observed)",
                            "holiday_type": "observado",
                            "is_holiday_observed": 1,
                        })
            except ValueError:
                # Ignora combinações inválidas de dia/mês para determinado ano
                continue

    calendario = pd.DataFrame(registros).drop_duplicates(subset=["date", "holiday_name"])
    # Se houver mais de um feriado no mesmo dia, prioriza observado > movel > fixo
    prioridade = {"observado": 3, "movel": 2, "fixo": 1}
    calendario["priority"] = calendario["holiday_type"].map(prioridade).fillna(0)
    calendario.sort_values(["date", "priority"], ascending=[True, False], inplace=True)
    calendario = calendario.drop_duplicates(subset=["date"], keep="first")
    calendario.drop(columns=["priority"], inplace=True)
    calendario.sort_values("date", inplace=True)
    calendario.reset_index(drop=True, inplace=True)
    return calendario


def calcular_parametros_economia(train_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Calcula parâmetros de imputação sem vazamento de informação futura."""
    params: Dict[str, Dict[str, float]] = {}
    for col in ["l14_gas_price", "l30_unemployment_rate"]:
        serie = train_df[col].dropna()
        if serie.empty:
            params[col] = {"train_fill": 0.0, "test_fill": 0.0}
        else:
            params[col] = {
                "train_fill": float(serie.median()),
                "test_fill": float(serie.iloc[-1]),
            }
    return params


def adicionar_features_exogenas(
    df: pd.DataFrame,
    holiday_calendar: pd.DataFrame,
    reference_date: pd.Timestamp,
    econ_fill_values: Dict[str, float],
    previous_exog_row: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Cria features de calendário, clima, economia e eventos sem usar a variável resposta."""
    feat = df.copy()

    # Tratamento de nulos em variáveis econômicas
    for col in ["l14_gas_price", "l30_unemployment_rate"]:
        if col in feat.columns:
            feat[col] = feat[col].ffill()
            if col in econ_fill_values:
                feat[col] = feat[col].fillna(econ_fill_values[col])

    # Features de calendário
    feat["year"] = feat["date"].dt.year
    feat["month"] = feat["date"].dt.month
    feat["day"] = feat["date"].dt.day
    feat["day_of_week"] = feat["date"].dt.dayofweek
    feat["week_of_year"] = feat["date"].dt.isocalendar().week.astype(int)
    feat["day_of_year"] = feat["date"].dt.dayofyear
    feat["is_weekend"] = (feat["day_of_week"] >= 5).astype(int)
    feat["is_month_start"] = feat["date"].dt.is_month_start.astype(int)
    feat["is_month_end"] = feat["date"].dt.is_month_end.astype(int)

    # Codificação cíclica para padrões sazonais
    feat["dow_sin"] = np.sin(2 * np.pi * feat["day_of_week"] / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * feat["day_of_week"] / 7)
    feat["month_sin"] = np.sin(2 * np.pi * feat["month"] / 12)
    feat["month_cos"] = np.cos(2 * np.pi * feat["month"] / 12)

    # Tendência temporal global
    feat["days_since_start"] = (feat["date"] - reference_date).dt.days

    # Features de clima
    feat["temp_mean"] = (feat["temp_min"] + feat["temp_max"]) / 2
    feat["temp_range"] = feat["temp_max"] - feat["temp_min"]
    feat["is_freezing_day"] = (feat["temp_mean"] < 32).astype(int)
    feat["heavy_precip"] = (feat["percip_max"] > 0.2).astype(int)
    feat["precip_log"] = np.log1p(feat["percip_max"])

    # Deltas climáticos para capturar mudança de regime entre dias consecutivos
    prev_temp_mean = feat["temp_mean"].shift(1)
    prev_temp_range = feat["temp_range"].shift(1)
    prev_precip = feat["percip_max"].shift(1)
    if previous_exog_row is not None and len(feat) > 0:
        prev_temp_mean.iloc[0] = (previous_exog_row["temp_min"] + previous_exog_row["temp_max"]) / 2
        prev_temp_range.iloc[0] = previous_exog_row["temp_max"] - previous_exog_row["temp_min"]
        prev_precip.iloc[0] = previous_exog_row["percip_max"]

    feat["temp_mean_delta_1d"] = (feat["temp_mean"] - prev_temp_mean).fillna(0.0)
    feat["temp_range_delta_1d"] = (feat["temp_range"] - prev_temp_range).fillna(0.0)
    feat["precip_delta_1d"] = (feat["percip_max"] - prev_precip).fillna(0.0)

    feat["weather_severity_index"] = (
        0.30 * feat["weather_rain"]
        + 0.30 * feat["weather_snow"]
        + 0.25 * feat["weather_storm"]
        + 0.15 * feat["weather_cloud"]
    )

    # Features de eventos esportivos
    home_cols = ["Blackhawks_Home", "Bulls_Home", "Bears_Home"]
    away_cols = ["Blackhawks_Away", "Bulls_Away", "Bears_Away"]

    feat["sports_home_games"] = feat[home_cols].sum(axis=1)
    feat["sports_away_games"] = feat[away_cols].sum(axis=1)
    feat["sports_any_game"] = (feat["sports_home_games"] + feat["sports_away_games"] > 0).astype(int)

    # A base contém colunas de baseball redundantes (idênticas entre si)
    feat["baseball_game_flag"] = feat["WhiteSox_Home"].astype(int)

    # Features de interação úteis para capturar efeitos condicionais
    feat["home_game_weekend"] = feat["sports_home_games"] * feat["is_weekend"]
    feat["weather_x_weekend"] = feat["weather_severity_index"] * feat["is_weekend"]

    # Merge de feriados
    feat = feat.merge(holiday_calendar, how="left", on="date")
    feat["holiday_name"] = feat["holiday_name"].fillna("Sem_feriado")
    feat["holiday_type"] = feat["holiday_type"].fillna("sem_feriado")
    feat["is_holiday_observed"] = feat["is_holiday_observed"].fillna(0).astype(int)
    feat["is_holiday"] = (feat["holiday_name"] != "Sem_feriado").astype(int)
    feat["is_holiday_fixed"] = (feat["holiday_type"] == "fixo").astype(int)
    feat["is_holiday_movable"] = (feat["holiday_type"] == "movel").astype(int)

    # Distâncias para feriado: absoluta e direcional (pré/pós)
    holiday_dates = holiday_calendar["date"].drop_duplicates().sort_values().to_numpy(dtype="datetime64[D]")
    current_dates = feat["date"].to_numpy(dtype="datetime64[D]")
    dist_abs = []
    dist_next = []
    dist_prev = []
    for date_value in current_dates:
        diffs = (holiday_dates - date_value).astype("timedelta64[D]").astype(int)
        distancia_abs = int(np.min(np.abs(diffs)))
        futuros = diffs[diffs >= 0]
        passados = diffs[diffs <= 0]
        distancia_next = int(np.min(futuros)) if len(futuros) > 0 else 9999
        distancia_prev = int(np.abs(np.max(passados))) if len(passados) > 0 else 9999
        dist_abs.append(distancia_abs)
        dist_next.append(distancia_next)
        dist_prev.append(distancia_prev)

    feat["days_to_nearest_holiday"] = np.array(dist_abs, dtype=int)
    feat["days_to_next_holiday"] = np.array(dist_next, dtype=int)
    feat["days_since_prev_holiday"] = np.array(dist_prev, dtype=int)
    feat["holiday_window_1d"] = (feat["days_to_nearest_holiday"] <= 1).astype(int)
    feat["holiday_window_3d"] = (feat["days_to_nearest_holiday"] <= 3).astype(int)
    feat["pre_holiday_1d"] = (feat["days_to_next_holiday"] == 1).astype(int)
    feat["pre_holiday_3d"] = feat["days_to_next_holiday"].between(1, 3).astype(int)
    feat["post_holiday_1d"] = (feat["days_since_prev_holiday"] == 1).astype(int)
    feat["post_holiday_3d"] = feat["days_since_prev_holiday"].between(1, 3).astype(int)

    # Interações explícitas por dia da semana
    for dow in range(7):
        mask_dow = (feat["day_of_week"] == dow).astype(int)
        feat[f"is_holiday_dow_{dow}"] = feat["is_holiday"] * mask_dow
        feat[f"sports_home_dow_{dow}"] = feat["sports_home_games"] * mask_dow

    # Features de regime para capturar comportamentos extremos
    feat["temp_extreme_low"] = (feat["temp_mean"] <= 20).astype(int)
    feat["temp_extreme_high"] = (feat["temp_mean"] >= 80).astype(int)
    feat["temp_extreme_any"] = ((feat["temp_extreme_low"] == 1) | (feat["temp_extreme_high"] == 1)).astype(int)
    feat["long_holiday_regime"] = (
        (feat["is_holiday"] == 1) | (feat["pre_holiday_3d"] == 1) | (feat["post_holiday_3d"] == 1)
    ).astype(int)
    feat["month_boundary_weekday"] = (
        ((feat["is_month_start"] == 1) | (feat["is_month_end"] == 1)) & (feat["is_weekend"] == 0)
    ).astype(int)

    # Remove colunas redundantes de baseball para reduzir colinearidade
    redundant_cols = ["WhiteSox_Away", "WhiteSox_Home", "Cubs_Away", "Cubs_Home"]
    feat.drop(columns=redundant_cols, inplace=True)

    return feat


def adicionar_lags_historicos(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Gera variáveis de defasagem e estatísticas móveis a partir da série alvo."""
    data = df.copy()

    for lag in LAGS:
        data[f"lag_{lag}"] = data[target_col].shift(lag)

    for window in ROLL_WINDOWS:
        data[f"roll_mean_{window}"] = data[target_col].shift(1).rolling(window=window).mean()
        data[f"roll_std_{window}"] = data[target_col].shift(1).rolling(window=window).std(ddof=0)

    return data


def montar_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    """Define as colunas finais de entrada do modelo."""
    blocked = {"date", target_col, "holiday_name", "holiday_type"}
    feature_cols = [col for col in df.columns if col not in blocked]
    return feature_cols


def calcular_metricas(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula métricas padrão de regressão para comparação de modelos."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None))) * 100
    return {"rmse": float(rmse), "mae": float(mae), "mape": float(mape)}


def construir_modelos() -> Dict[str, RegressorMixin]:
    """Cria o conjunto de modelos candidatos para seleção."""
    modelos: Dict[str, RegressorMixin] = {
        "ridge": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=600,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hist_gb": HistGradientBoostingRegressor(
            max_depth=4,
            learning_rate=0.03,
            max_iter=700,
            min_samples_leaf=20,
            l2_regularization=3.0,
            max_leaf_nodes=31,
            max_bins=128,
            random_state=RANDOM_STATE,
        ),
    }
    return modelos


def predicao_recursiva(
    model: RegressorMixin,
    history_targets: List[float],
    exog_df: pd.DataFrame,
    feature_cols: List[str],
) -> np.ndarray:
    """Realiza previsão multi-passo recursiva utilizando lags da própria série prevista."""
    preds: List[float] = []
    historico = list(history_targets)

    for _, row in exog_df.iterrows():
        lag_features: Dict[str, float] = {}

        for lag in LAGS:
            lag_features[f"lag_{lag}"] = historico[-lag]

        for window in ROLL_WINDOWS:
            janela = np.array(historico[-window:], dtype=float)
            lag_features[f"roll_mean_{window}"] = float(np.mean(janela))
            lag_features[f"roll_std_{window}"] = float(np.std(janela, ddof=0))

        row_dict = row.to_dict()
        row_dict.update(lag_features)

        x_pred = pd.DataFrame([row_dict], columns=feature_cols)
        y_pred = float(model.predict(x_pred)[0])

        preds.append(y_pred)
        historico.append(y_pred)

    return np.array(preds)


def validar_modelo_rolling(
    model: RegressorMixin,
    train_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    horizon: int,
    n_folds: int,
) -> ModelResult:
    """Valida o modelo com backtesting rolling no mesmo horizonte do conjunto de teste."""
    minimo_historico = max(max(LAGS), max(ROLL_WINDOWS)) + 30
    inicio = max(minimo_historico, len(train_df) - horizon * n_folds)

    all_true: List[float] = []
    all_pred: List[float] = []

    for cutoff in range(inicio, len(train_df) - horizon + 1, horizon):
        train_part = train_df.iloc[:cutoff].copy()
        valid_part = train_df.iloc[cutoff:cutoff + horizon].copy()

        train_lagged = adicionar_lags_historicos(train_part, target_col).dropna().reset_index(drop=True)
        x_train = train_lagged[feature_cols]
        y_train = train_lagged[target_col]
        model.fit(x_train, y_train)

        history_targets = train_part[target_col].tolist()
        valid_exog = valid_part[
            [col for col in feature_cols if not col.startswith("lag_") and not col.startswith("roll_")]
        ]
        y_pred = predicao_recursiva(
            model=model,
            history_targets=history_targets,
            exog_df=valid_exog,
            feature_cols=feature_cols,
        )

        all_true.extend(valid_part[target_col].tolist())
        all_pred.extend(y_pred.tolist())

    metrics = calcular_metricas(np.array(all_true), np.array(all_pred))
    return ModelResult(name="", rmse=metrics["rmse"], mae=metrics["mae"], mape=metrics["mape"])


def gerar_diagnostico_validacao(
    model: RegressorMixin,
    train_df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    horizon: int,
    n_folds: int,
    tables_dir: Path,
) -> pd.DataFrame:
    """Gera diagnóstico detalhado de validação recursiva para o modelo final."""
    minimo_historico = max(max(LAGS), max(ROLL_WINDOWS)) + 30
    inicio = max(minimo_historico, len(train_df) - horizon * n_folds)

    registros: List[pd.DataFrame] = []
    fold_idx = 0

    for cutoff in range(inicio, len(train_df) - horizon + 1, horizon):
        fold_idx += 1
        train_part = train_df.iloc[:cutoff].copy()
        valid_part = train_df.iloc[cutoff:cutoff + horizon].copy()

        train_lagged = adicionar_lags_historicos(train_part, target_col).dropna().reset_index(drop=True)
        x_train = train_lagged[feature_cols]
        y_train = train_lagged[target_col]

        model_fold = clone(model)
        model_fold.fit(x_train, y_train)

        history_targets = train_part[target_col].tolist()
        valid_exog = valid_part[
            [col for col in feature_cols if not col.startswith("lag_") and not col.startswith("roll_")]
        ]
        y_pred = predicao_recursiva(
            model=model_fold,
            history_targets=history_targets,
            exog_df=valid_exog,
            feature_cols=feature_cols,
        )

        fold_df = valid_part[["date", "day_of_week", "is_weekend", "is_holiday", target_col]].copy()
        fold_df["fold"] = fold_idx
        fold_df["y_pred"] = y_pred
        fold_df["erro"] = fold_df[target_col] - fold_df["y_pred"]
        fold_df["erro_abs"] = np.abs(fold_df["erro"])
        fold_df["erro_pct_abs"] = np.abs(fold_df["erro"]) / np.clip(fold_df[target_col], 1e-6, None) * 100
        registros.append(fold_df)

    diagnostico = pd.concat(registros, ignore_index=True)
    map_pt = {
        0: "Segunda",
        1: "Terça",
        2: "Quarta",
        3: "Quinta",
        4: "Sexta",
        5: "Sábado",
        6: "Domingo",
    }
    diagnostico["dia_semana"] = diagnostico["day_of_week"].map(map_pt)
    diagnostico.to_csv(tables_dir / "diagnostico_validacao_modelo_final.csv", index=False)

    resumo_geral = pd.DataFrame(
        [
            {
                "rmse": float(np.sqrt(np.mean(np.square(diagnostico["erro"])))),
                "mae": float(np.mean(np.abs(diagnostico["erro"]))),
                "mape_percent": float(np.mean(diagnostico["erro_pct_abs"])),
                "n_obs": int(len(diagnostico)),
            }
        ]
    )
    resumo_geral.to_csv(tables_dir / "diagnostico_validacao_resumo_geral.csv", index=False)

    resumo_dia = (
        diagnostico.groupby("dia_semana")["erro_abs"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .rename(columns={"mean": "mae_medio", "median": "mae_mediana"})
    )
    ordem = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]
    resumo_dia["dia_semana"] = pd.Categorical(resumo_dia["dia_semana"], categories=ordem, ordered=True)
    resumo_dia.sort_values("dia_semana", inplace=True)
    resumo_dia.to_csv(tables_dir / "diagnostico_validacao_por_dia_semana.csv", index=False)

    resumo_feriado = (
        diagnostico.groupby("is_holiday")["erro_abs"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .rename(columns={"mean": "mae_medio", "median": "mae_mediana"})
    )
    resumo_feriado["is_holiday"] = resumo_feriado["is_holiday"].map({0: "Não", 1: "Sim"})
    resumo_feriado.to_csv(tables_dir / "diagnostico_validacao_feriado.csv", index=False)

    resumo_weekend = (
        diagnostico.groupby("is_weekend")["erro_abs"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .rename(columns={"mean": "mae_medio", "median": "mae_mediana"})
    )
    resumo_weekend["is_weekend"] = resumo_weekend["is_weekend"].map({0: "Não", 1: "Sim"})
    resumo_weekend.to_csv(tables_dir / "diagnostico_validacao_fim_semana.csv", index=False)

    return diagnostico


def gerar_visualizacoes(df: pd.DataFrame, figs_dir: Path) -> None:
    """Cria gráficos que justificam as principais hipóteses de engenharia de features."""
    sns.set_theme(style="whitegrid")

    # 1) Série temporal da demanda
    plt.figure(figsize=(14, 5))
    plt.plot(df["date"], df["s_40380"], alpha=0.5, linewidth=1, label="Demanda diária")
    plt.plot(
        df["date"],
        df["s_40380"].rolling(30, min_periods=1).mean(),
        color="black",
        linewidth=2,
        label="Média móvel 30 dias",
    )
    plt.title("Demanda diária na estação Clark/Lake")
    plt.xlabel("Data")
    plt.ylabel("Passageiros (s_40380)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs_dir / "01_serie_temporal_demanda.png", dpi=180)
    plt.close()

    # 2) Efeito de dia da semana
    plt.figure(figsize=(10, 5))
    ordem = [0, 1, 2, 3, 4, 5, 6]
    labels = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
    sns.boxplot(data=df, x="day_of_week", y="s_40380", order=ordem, color="#4C78A8")
    plt.xticks(ticks=ordem, labels=labels)
    plt.title("Distribuição da demanda por dia da semana")
    plt.xlabel("Dia da semana")
    plt.ylabel("Passageiros (s_40380)")
    plt.tight_layout()
    plt.savefig(figs_dir / "02_boxplot_dia_semana.png", dpi=180)
    plt.close()

    # 3) Efeito de feriados
    plt.figure(figsize=(8, 5))
    agg_holiday = df.groupby("is_holiday", as_index=False)["s_40380"].mean()
    agg_holiday["is_holiday_label"] = agg_holiday["is_holiday"].map({0: "Não", 1: "Sim"})
    sns.barplot(data=agg_holiday, x="is_holiday_label", y="s_40380", color="#4C78A8")
    plt.title("Demanda média em feriados vs. não feriados")
    plt.xlabel("É feriado?")
    plt.ylabel("Passageiros médios")
    plt.tight_layout()
    plt.savefig(figs_dir / "03_feriado_vs_nao_feriado.png", dpi=180)
    plt.close()

    # 4) Temperatura média vs demanda
    plt.figure(figsize=(9, 5))
    sns.regplot(
        data=df.sample(min(3000, len(df)), random_state=RANDOM_STATE),
        x="temp_mean",
        y="s_40380",
        scatter_kws={"alpha": 0.25, "s": 18},
        line_kws={"color": "red", "lw": 2},
    )
    plt.title("Relação entre temperatura média e demanda")
    plt.xlabel("Temperatura média (°F)")
    plt.ylabel("Passageiros (s_40380)")
    plt.tight_layout()
    plt.savefig(figs_dir / "04_temp_media_vs_demanda.png", dpi=180)
    plt.close()

    # 5) Índice de clima severo vs demanda
    plt.figure(figsize=(9, 5))
    sns.regplot(
        data=df.sample(min(3000, len(df)), random_state=RANDOM_STATE),
        x="weather_severity_index",
        y="s_40380",
        scatter_kws={"alpha": 0.25, "s": 18},
        line_kws={"color": "darkgreen", "lw": 2},
    )
    plt.title("Relação entre severidade do clima e demanda")
    plt.xlabel("Índice de severidade climática")
    plt.ylabel("Passageiros (s_40380)")
    plt.tight_layout()
    plt.savefig(figs_dir / "05_clima_severidade_vs_demanda.png", dpi=180)
    plt.close()

    # 6) Jogos em casa e demanda
    plt.figure(figsize=(8, 5))
    agg_games = df.groupby("sports_home_games", as_index=False)["s_40380"].mean()
    sns.barplot(data=agg_games, x="sports_home_games", y="s_40380", color="#F58518")
    plt.title("Demanda média por número de jogos em casa")
    plt.xlabel("Quantidade de jogos em casa no dia")
    plt.ylabel("Passageiros médios")
    plt.tight_layout()
    plt.savefig(figs_dir / "06_jogos_casa_vs_demanda.png", dpi=180)
    plt.close()

    # 7) Lag semanal vs demanda
    lag_df = adicionar_lags_historicos(df[["date", "s_40380"]], "s_40380").dropna()
    plt.figure(figsize=(9, 5))
    sns.regplot(
        data=lag_df.sample(min(3000, len(lag_df)), random_state=RANDOM_STATE),
        x="lag_7",
        y="s_40380",
        scatter_kws={"alpha": 0.25, "s": 18},
        line_kws={"color": "purple", "lw": 2},
    )
    plt.title("Autocorrelação semanal: demanda atual vs demanda de 7 dias atrás")
    plt.xlabel("Lag 7 dias")
    plt.ylabel("Passageiros (s_40380)")
    plt.tight_layout()
    plt.savefig(figs_dir / "07_lag7_vs_demanda.png", dpi=180)
    plt.close()

    # 8) Distância para feriado vs demanda
    df_plot = df.copy()
    bins = [-1, 0, 1, 3, 7, 30, 500]
    labels = ["0", "1", "2-3", "4-7", "8-30", "31+"]
    df_plot["holiday_dist_bucket"] = pd.cut(df_plot["days_to_nearest_holiday"], bins=bins, labels=labels)
    plt.figure(figsize=(10, 5))
    agg_dist = df_plot.groupby("holiday_dist_bucket", as_index=False)["s_40380"].mean()
    sns.barplot(data=agg_dist, x="holiday_dist_bucket", y="s_40380", color="#54A24B")
    plt.title("Demanda média por proximidade de feriados")
    plt.xlabel("Dias até o feriado mais próximo")
    plt.ylabel("Passageiros médios")
    plt.tight_layout()
    plt.savefig(figs_dir / "08_proximidade_feriado_vs_demanda.png", dpi=180)
    plt.close()


def gerar_tabelas_exploratorias(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    holiday_raw: pd.DataFrame,
    train_feat: pd.DataFrame,
    tables_dir: Path,
) -> None:
    """Gera tabelas exploratórias usadas na análise e no relatório final."""
    perfil = pd.DataFrame(
        [
            {
                "base": "treino",
                "linhas": len(train_raw),
                "colunas": train_raw.shape[1],
                "data_min": str(train_raw["date"].min().date()),
                "data_max": str(train_raw["date"].max().date()),
                "faltantes_total": int(train_raw.isna().sum().sum()),
            },
            {
                "base": "teste",
                "linhas": len(test_raw),
                "colunas": test_raw.shape[1],
                "data_min": str(test_raw["date"].min().date()),
                "data_max": str(test_raw["date"].max().date()),
                "faltantes_total": int(test_raw.isna().sum().sum()),
            },
            {
                "base": "feriados",
                "linhas": len(holiday_raw),
                "colunas": holiday_raw.shape[1],
                "data_min": "N/A",
                "data_max": "N/A",
                "faltantes_total": int(holiday_raw.isna().sum().sum()),
            },
        ]
    )
    perfil.to_csv(tables_dir / "perfil_bases.csv", index=False)

    # Distribuição por dia da semana
    map_pt = {
        0: "Segunda",
        1: "Terça",
        2: "Quarta",
        3: "Quinta",
        4: "Sexta",
        5: "Sábado",
        6: "Domingo",
    }
    dow = (
        train_feat.groupby("day_of_week")["s_40380"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    dow["dia_semana"] = dow["day_of_week"].map(map_pt)
    dow = dow[["day_of_week", "dia_semana", "count", "mean", "median", "std"]]
    dow.to_csv(tables_dir / "demanda_por_dia_semana.csv", index=False)

    feriados = (
        train_feat.groupby("is_holiday")["s_40380"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    feriados["is_holiday"] = feriados["is_holiday"].map({0: "Não", 1: "Sim"})
    feriados.to_csv(tables_dir / "demanda_feriado_vs_nao.csv", index=False)

    # Correlação dos principais lags
    lag_df = adicionar_lags_historicos(train_feat[["date", "s_40380"]], "s_40380")
    corr_lags = lag_df[["s_40380", "lag_1", "lag_7", "lag_14", "lag_28"]].corr()["s_40380"].reset_index()
    corr_lags.columns = ["feature", "correlacao_com_target"]
    corr_lags.to_csv(tables_dir / "correlacao_lags.csv", index=False)

    # Checagem de colunas esportivas idênticas
    esporte_cols = [
        "Blackhawks_Away",
        "Blackhawks_Home",
        "Bulls_Away",
        "Bulls_Home",
        "Bears_Away",
        "Bears_Home",
        "WhiteSox_Away",
        "WhiteSox_Home",
        "Cubs_Away",
        "Cubs_Home",
    ]
    iguais: List[Dict[str, object]] = []
    for i, col1 in enumerate(esporte_cols):
        for col2 in esporte_cols[i + 1:]:
            if (train_raw[col1] == train_raw[col2]).all():
                iguais.append({"coluna_1": col1, "coluna_2": col2, "identicas": 1})
    pd.DataFrame(iguais).to_csv(tables_dir / "colunas_esportivas_identicas.csv", index=False)

    # Tendência anual da demanda
    anual = (
        train_feat.assign(year=train_feat["date"].dt.year)
        .groupby("year")["s_40380"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    anual.to_csv(tables_dir / "demanda_media_anual.csv", index=False)

    # Proximidade de feriado
    dist = train_feat.copy()
    bins_dist = [-1, 0, 1, 3, 7, 30, 500]
    labels_dist = ["0", "1", "2-3", "4-7", "8-30", "31+"]
    dist["faixa_dist_feriado"] = pd.cut(dist["days_to_nearest_holiday"], bins=bins_dist, labels=labels_dist)
    tabela_dist = (
        dist.groupby("faixa_dist_feriado")["s_40380"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    tabela_dist.to_csv(tables_dir / "demanda_por_distancia_feriado.csv", index=False)

    # Faixas de temperatura
    temp = train_feat.copy()
    bins_temp = [-30, 20, 32, 45, 60, 80, 120]
    labels_temp = ["<20F", "20-32F", "32-45F", "45-60F", "60-80F", ">80F"]
    temp["faixa_temp"] = pd.cut(temp["temp_mean"], bins=bins_temp, labels=labels_temp)
    tabela_temp = (
        temp.groupby("faixa_temp")["s_40380"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    tabela_temp.to_csv(tables_dir / "demanda_por_faixa_temperatura.csv", index=False)

    # Severidade climática por quartil
    clima = train_feat.copy()
    clima["quartil_clima"] = pd.qcut(
        clima["weather_severity_index"],
        q=4,
        labels=["Q1_baixa", "Q2", "Q3", "Q4_alta"],
    )
    tabela_clima = (
        clima.groupby("quartil_clima")["s_40380"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    tabela_clima.to_csv(tables_dir / "demanda_por_quartil_severidade_clima.csv", index=False)

    # Jogos em casa
    tabela_home = (
        train_feat.groupby("sports_home_games")["s_40380"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    tabela_home.to_csv(tables_dir / "demanda_por_jogos_casa.csv", index=False)


def gerar_tabelas_apoio(
    train_feat: pd.DataFrame,
    train_supervised: pd.DataFrame,
    metrics_df: pd.DataFrame,
    selected_features: List[str],
    tables_dir: Path,
    best_model: RegressorMixin,
) -> None:
    """Exporta tabelas auxiliares para documentação dos resultados."""
    # Resumo do alvo
    resumo_target = train_feat["s_40380"].describe().to_frame(name="s_40380")
    resumo_target.to_csv(tables_dir / "resumo_target.csv", index=True)

    # Métricas de validação
    metrics_df.to_csv(tables_dir / "comparacao_modelos_validacao.csv", index=False)

    # Lista de features finais
    pd.DataFrame({"feature": selected_features}).to_csv(
        tables_dir / "features_finais_modelo.csv", index=False
    )

    # Importância de variáveis para modelo com feature_importances_
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feat_imp = (
            pd.DataFrame({"feature": selected_features, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        feat_imp.to_csv(tables_dir / "importancia_features_modelo_final.csv", index=False)
    else:
        # Para modelos sem feature_importances_, usa importância por permutação
        val_n = min(365, max(90, int(len(train_supervised) * 0.2)))
        train_part = train_supervised.iloc[:-val_n].copy()
        valid_part = train_supervised.iloc[-val_n:].copy()

        if len(train_part) > 0 and len(valid_part) > 0:
            model_clone = clone(best_model)
            model_clone.fit(train_part[selected_features], train_part["s_40380"])
            perm = permutation_importance(
                model_clone,
                valid_part[selected_features],
                valid_part["s_40380"],
                n_repeats=5,
                random_state=RANDOM_STATE,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
            )
            feat_imp = (
                pd.DataFrame(
                    {
                        "feature": selected_features,
                        "importance_mean": perm.importances_mean,
                        "importance_std": perm.importances_std,
                    }
                )
                .sort_values("importance_mean", ascending=False)
                .reset_index(drop=True)
            )
            feat_imp.to_csv(tables_dir / "importancia_permutacao_modelo_final.csv", index=False)


def validar_transformacoes(
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    train_supervised: pd.DataFrame,
    feature_cols: List[str],
    reference_date: pd.Timestamp,
    tables_dir: Path,
) -> None:
    """Executa validações de consistência das transformações e salva relatório de QA."""
    checks: List[Dict[str, object]] = []

    expected_test_start = int((test_feat["date"].min() - reference_date).days)
    checks.append(
        {
            "check": "days_since_start_alinhado",
            "status": bool(test_feat["days_since_start"].min() == expected_test_start),
            "detalhe": f"esperado={expected_test_start}; obtido={int(test_feat['days_since_start'].min())}",
        }
    )

    checks.append(
        {
            "check": "features_sem_nulos_no_treino_supervisionado",
            "status": bool(train_supervised[feature_cols].isna().sum().sum() == 0),
            "detalhe": f"n_nulos={int(train_supervised[feature_cols].isna().sum().sum())}",
        }
    )

    checks.append(
        {
            "check": "is_holiday_consistente_com_distancia",
            "status": bool(((train_feat["is_holiday"] == 1) == (train_feat["days_to_nearest_holiday"] == 0)).all()),
            "detalhe": "is_holiday deve ser 1 apenas quando a distância para feriado for 0",
        }
    )

    checks.append(
        {
            "check": "sports_home_games_em_faixa_valida",
            "status": bool(train_feat["sports_home_games"].between(0, 3).all()),
            "detalhe": "faixa esperada: 0 a 3",
        }
    )

    checks.append(
        {
            "check": "sports_away_games_em_faixa_valida",
            "status": bool(train_feat["sports_away_games"].between(0, 3).all()),
            "detalhe": "faixa esperada: 0 a 3",
        }
    )

    checks.append(
        {
            "check": "feriado_observado_implca_feriado",
            "status": bool((train_feat.loc[train_feat["is_holiday_observed"] == 1, "is_holiday"] == 1).all()),
            "detalhe": "quando is_holiday_observed=1, is_holiday deve ser 1",
        }
    )

    checks.append(
        {
            "check": "pre_holiday_1d_consistente",
            "status": bool(
                (train_feat.loc[train_feat["pre_holiday_1d"] == 1, "days_to_next_holiday"] == 1).all()
            ),
            "detalhe": "pre_holiday_1d deve marcar apenas dias com days_to_next_holiday=1",
        }
    )

    delta_cols = [c for c in ["temp_mean_delta_1d", "temp_range_delta_1d", "precip_delta_1d"] if c in train_feat.columns]
    checks.append(
        {
            "check": "deltas_climaticos_sem_nulos",
            "status": bool(train_feat[delta_cols].isna().sum().sum() == 0),
            "detalhe": f"n_nulos={int(train_feat[delta_cols].isna().sum().sum())}",
        }
    )

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(tables_dir / "validacao_transformacoes.csv", index=False)

    if not checks_df["status"].all():
        failed = checks_df.loc[~checks_df["status"], ["check", "detalhe"]]
        raise ValueError(f"Falha em validações de transformação:\n{failed.to_string(index=False)}")


def main() -> None:
    """Executa o fluxo completo de engenharia de features e previsão."""
    base_dir = Path(__file__).resolve().parent
    dirs = preparar_diretorios(base_dir)

    train_df, test_df, holiday_df = carregar_bases(base_dir)

    all_years = sorted(set(train_df["date"].dt.year.tolist() + test_df["date"].dt.year.tolist()))
    holiday_calendar = criar_calendario_feriados(holiday_df, all_years)
    reference_date = train_df["date"].min()

    econ_params = calcular_parametros_economia(train_df)
    train_fill_values = {col: params["train_fill"] for col, params in econ_params.items()}
    test_fill_values = {col: params["test_fill"] for col, params in econ_params.items()}

    train_feat = adicionar_features_exogenas(
        train_df,
        holiday_calendar,
        reference_date=reference_date,
        econ_fill_values=train_fill_values,
        previous_exog_row=None,
    )
    previous_exog_row = train_df.iloc[-1][["temp_min", "temp_max", "percip_max"]].to_dict()
    test_feat = adicionar_features_exogenas(
        test_df,
        holiday_calendar,
        reference_date=reference_date,
        econ_fill_values=test_fill_values,
        previous_exog_row=previous_exog_row,
    )

    gerar_tabelas_exploratorias(
        train_raw=train_df,
        test_raw=test_df,
        holiday_raw=holiday_df,
        train_feat=train_feat,
        tables_dir=dirs["tables"],
    )

    # Cria lags no treino para treinamento supervisionado
    train_lagged = adicionar_lags_historicos(train_feat, "s_40380")

    feature_cols = montar_feature_columns(train_lagged, target_col="s_40380")

    # Separa exógenas (sem lags) para previsão recursiva
    exog_cols = [c for c in feature_cols if not c.startswith("lag_") and not c.startswith("roll_")]

    # Avalia modelos candidatos com backtesting rolling no horizonte real do teste
    modelos = construir_modelos()
    resultados: List[ModelResult] = []
    validation_horizon = len(test_feat)
    validation_folds = min(
        12,
        max(3, (len(train_feat) - max(LAGS) - max(ROLL_WINDOWS) - 1) // validation_horizon),
    )

    for model_name, model in modelos.items():
        resultado = validar_modelo_rolling(
            model=model,
            train_df=train_feat,
            target_col="s_40380",
            feature_cols=feature_cols,
            horizon=validation_horizon,
            n_folds=validation_folds,
        )
        resultado.name = model_name
        resultados.append(resultado)

    metrics_df = pd.DataFrame(
        [
            {
                "modelo": r.name,
                "rmse": r.rmse,
                "mae": r.mae,
                "mape_percent": r.mape,
            }
            for r in resultados
        ]
    ).sort_values("rmse")

    best_model_name = metrics_df.iloc[0]["modelo"]
    best_model = modelos[best_model_name]

    # Treina modelo final em todo o histórico disponível
    train_supervised = train_lagged.dropna().reset_index(drop=True)

    validar_transformacoes(
        train_feat=train_feat,
        test_feat=test_feat,
        train_supervised=train_supervised,
        feature_cols=feature_cols,
        reference_date=reference_date,
        tables_dir=dirs["tables"],
    )

    x_final = train_supervised[feature_cols]
    y_final = train_supervised["s_40380"]
    best_model.fit(x_final, y_final)

    # Diagnóstico de ajuste (overfit/underfit): treino vs validação temporal
    y_train_pred = best_model.predict(x_final)
    train_metrics = calcular_metricas(y_final.to_numpy(), y_train_pred)
    val_row = metrics_df.loc[metrics_df["modelo"] == best_model_name].iloc[0]
    val_rmse = float(val_row["rmse"])
    ratio_train_val_rmse = train_metrics["rmse"] / max(val_rmse, 1e-9)
    if ratio_train_val_rmse < 0.55:
        ajuste_status = "Sinal de overfitting forte"
    elif ratio_train_val_rmse < 0.75:
        ajuste_status = "Sinal de overfitting moderado"
    elif ratio_train_val_rmse <= 1.05:
        ajuste_status = "Ajuste equilibrado"
    else:
        ajuste_status = "Sinal de underfitting"

    pd.DataFrame(
        [
            {
                "modelo": best_model_name,
                "train_rmse": train_metrics["rmse"],
                "train_mae": train_metrics["mae"],
                "train_mape_percent": train_metrics["mape"],
                "val_rmse": val_rmse,
                "val_mae": float(val_row["mae"]),
                "val_mape_percent": float(val_row["mape_percent"]),
                "ratio_train_val_rmse": ratio_train_val_rmse,
                "diagnostico_ajuste": ajuste_status,
            }
        ]
    ).to_csv(dirs["tables"] / "diagnostico_overfit_underfit.csv", index=False)

    # Comparação antes/depois da engenharia adicional de features
    novas_features_exatas = {
        "is_holiday_observed",
        "is_holiday_fixed",
        "is_holiday_movable",
        "days_to_next_holiday",
        "days_since_prev_holiday",
        "pre_holiday_1d",
        "pre_holiday_3d",
        "post_holiday_1d",
        "post_holiday_3d",
        "temp_mean_delta_1d",
        "temp_range_delta_1d",
        "precip_delta_1d",
        "temp_extreme_low",
        "temp_extreme_high",
        "temp_extreme_any",
        "long_holiday_regime",
        "month_boundary_weekday",
    }
    novas_features_prefixo = ("is_holiday_dow_", "sports_home_dow_")
    feature_cols_baseline = [
        c for c in feature_cols
        if c not in novas_features_exatas and not any(c.startswith(pref) for pref in novas_features_prefixo)
    ]

    baseline_metrics = validar_modelo_rolling(
        model=clone(best_model),
        train_df=train_feat,
        target_col="s_40380",
        feature_cols=feature_cols_baseline,
        horizon=validation_horizon,
        n_folds=validation_folds,
    )

    pd.DataFrame(
        [
            {
                "cenario": "baseline_sem_novas_features",
                "n_features": len(feature_cols_baseline),
                "rmse": baseline_metrics.rmse,
                "mae": baseline_metrics.mae,
                "mape_percent": baseline_metrics.mape,
            },
            {
                "cenario": "enhanced_com_novas_features",
                "n_features": len(feature_cols),
                "rmse": float(val_row["rmse"]),
                "mae": float(val_row["mae"]),
                "mape_percent": float(val_row["mape_percent"]),
            },
            {
                "cenario": "delta_enhanced_menos_baseline",
                "n_features": len(feature_cols) - len(feature_cols_baseline),
                "rmse": float(val_row["rmse"]) - baseline_metrics.rmse,
                "mae": float(val_row["mae"]) - baseline_metrics.mae,
                "mape_percent": float(val_row["mape_percent"]) - baseline_metrics.mape,
            },
        ]
    ).to_csv(dirs["tables"] / "comparacao_feature_engineering_upgrade.csv", index=False)

    # Salva diagnóstico detalhado da validação do modelo campeão
    gerar_diagnostico_validacao(
        model=best_model,
        train_df=train_feat,
        target_col="s_40380",
        feature_cols=feature_cols,
        horizon=validation_horizon,
        n_folds=validation_folds,
        tables_dir=dirs["tables"],
    )

    # Gera previsão recursiva para o horizonte de teste
    history_targets = train_feat["s_40380"].tolist()
    test_exog = test_feat[exog_cols].copy()
    test_preds = predicao_recursiva(
        model=best_model,
        history_targets=history_targets,
        exog_df=test_exog,
        feature_cols=feature_cols,
    )

    # Garante previsões não negativas
    test_preds = np.clip(test_preds, a_min=0, a_max=None)

    previsoes_df = pd.DataFrame(
        {
            "date": test_feat["date"],
            "s_40380": test_preds,
        }
    )

    previsoes_csv = dirs["preds"] / "previsoes_teste_s40380.csv"
    previsoes_xlsx = dirs["preds"] / "previsoes_teste_s40380.xlsx"
    previsoes_df.to_csv(previsoes_csv, index=False)
    previsoes_df.to_excel(previsoes_xlsx, index=False)

    # Versão com contexto de calendário para interpretação de negócio
    previsoes_contexto = test_feat[["date", "day_of_week", "is_weekend", "is_holiday"]].copy()
    map_pt = {
        0: "Segunda",
        1: "Terça",
        2: "Quarta",
        3: "Quinta",
        4: "Sexta",
        5: "Sábado",
        6: "Domingo",
    }
    previsoes_contexto["dia_semana"] = previsoes_contexto["day_of_week"].map(map_pt)
    previsoes_contexto["s_40380_pred"] = test_preds
    previsoes_contexto["is_weekend"] = previsoes_contexto["is_weekend"].map({0: "Não", 1: "Sim"})
    previsoes_contexto["is_holiday"] = previsoes_contexto["is_holiday"].map({0: "Não", 1: "Sim"})
    previsoes_contexto.to_csv(dirs["preds"] / "previsoes_teste_s40380_com_contexto.csv", index=False)

    # Gera visualizações e tabelas de apoio
    gerar_visualizacoes(train_feat, dirs["figs"])
    gerar_tabelas_apoio(
        train_feat=train_feat,
        train_supervised=train_supervised,
        metrics_df=metrics_df,
        selected_features=feature_cols,
        tables_dir=dirs["tables"],
        best_model=best_model,
    )

    # Salva resumo da execução para reutilização no relatório
    if hasattr(best_model, "get_params"):
        model_params = best_model.get_params()
    else:
        model_params = {}
    if best_model_name == "hist_gb":
        best_params_export = {
            "max_depth": model_params.get("max_depth"),
            "learning_rate": model_params.get("learning_rate"),
            "max_iter": model_params.get("max_iter"),
            "min_samples_leaf": model_params.get("min_samples_leaf"),
            "l2_regularization": model_params.get("l2_regularization"),
            "max_bins": model_params.get("max_bins"),
        }
    elif best_model_name == "random_forest":
        best_params_export = {
            "n_estimators": model_params.get("n_estimators"),
            "max_depth": model_params.get("max_depth"),
            "min_samples_leaf": model_params.get("min_samples_leaf"),
            "max_features": model_params.get("max_features"),
        }
    else:
        best_params_export = model_params

    resumo_execucao = {
        "best_model": str(best_model_name),
        "metrics": metrics_df.to_dict(orient="records"),
        "n_train": int(len(train_feat)),
        "n_test": int(len(test_feat)),
        "train_date_min": str(train_feat["date"].min().date()),
        "train_date_max": str(train_feat["date"].max().date()),
        "test_date_min": str(test_feat["date"].min().date()),
        "test_date_max": str(test_feat["date"].max().date()),
        "n_features": int(len(feature_cols)),
        "validation_horizon_days": int(validation_horizon),
        "validation_folds": int(validation_folds),
        "best_model_params": best_params_export,
    }

    with open(dirs["out"] / "resumo_execucao.json", "w", encoding="utf-8") as f:
        json.dump(resumo_execucao, f, indent=2, ensure_ascii=False)

    print("Pipeline concluído com sucesso.")
    print(f"Modelo selecionado: {best_model_name}")
    print(f"Previsões salvas em: {previsoes_csv}")


if __name__ == "__main__":
    main()
