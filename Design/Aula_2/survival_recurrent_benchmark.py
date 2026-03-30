#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.metrics import (
        brier_score,
        concordance_index_censored,
        concordance_index_ipcw,
        integrated_brier_score,
    )
except Exception as exc:
    raise SystemExit(
        "Missing dependency 'scikit-survival'. "
        "Run './setup_survival_env.sh' and execute this script inside '.venv'."
    ) from exc


LOGGER = logging.getLogger("survival_recurrent_benchmark")
DEFAULT_BASE_DIR = Path("/Users/akatsurada/Documents/INSPER/Design/Aula_2")


@dataclass(frozen=True)
class SurvivalBenchmarkConfig:
    base_dir: Path
    data_dir: Path
    output_dir: Path
    min_session_seconds: int = 5
    tau_quantile: float = 0.90
    min_train_months: int = 6
    max_backtest_months: int = 10
    capacity_grid: Tuple[float, float] = (0.05, 0.10)
    random_seed: int = 42
    tuning_trials: int = 8
    max_train_rows: int = 220_000
    max_panel_rows: int = 220_000
    max_users: int = 0


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark survival recorrente (RSF vs XGB-AFT).")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-session-seconds", type=int, default=5)
    parser.add_argument("--tau-quantile", type=float, default=0.90)
    parser.add_argument("--min-train-months", type=int, default=6)
    parser.add_argument("--max-backtest-months", type=int, default=10)
    parser.add_argument("--tuning-trials", type=int, default=8)
    parser.add_argument("--max-train-rows", type=int, default=220000)
    parser.add_argument("--max-panel-rows", type=int, default=220000)
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SurvivalBenchmarkConfig:
    base_dir = args.base_dir
    data_dir = args.data_dir if args.data_dir is not None else base_dir / "base_aprendizap"
    output_dir = args.output_dir if args.output_dir is not None else base_dir / "analysis_output"
    return SurvivalBenchmarkConfig(
        base_dir=base_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        min_session_seconds=int(args.min_session_seconds),
        tau_quantile=float(args.tau_quantile),
        min_train_months=int(args.min_train_months),
        max_backtest_months=int(args.max_backtest_months),
        random_seed=int(args.random_seed),
        tuning_trials=int(args.tuning_trials),
        max_train_rows=int(args.max_train_rows),
        max_panel_rows=int(args.max_panel_rows),
        max_users=int(args.max_users),
    )


def q(path: Path) -> str:
    return str(path).replace("'", "''")


def normalize_utm(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "missing"
    s = str(x).strip().lower()
    if s in {"", "none", "<na>"}:
        return "missing"
    if "google ads" in s or "seo ads" in s:
        return "paid_search"
    if "seo org" in s:
        return "organic_search"
    if "landing" in s:
        return "landing"
    if "blog" in s:
        return "blog"
    if "mídias sociais" in s or "midias sociais" in s or "social" in s:
        return "social"
    if "convite_escola" in s:
        return "school_invite"
    if "push" in s or "notificacao" in s:
        return "push_or_notification"
    if "mari" in s:
        return "mari"
    return "other"


def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return a / np.maximum(b, eps)


def _end_of_next_month(ts: pd.Timestamp) -> pd.Timestamp:
    next_start = (ts + pd.offsets.MonthBegin(1)).normalize()
    after_next = (next_start + pd.offsets.MonthBegin(1)).normalize()
    return after_next - pd.Timedelta(seconds=1)


def _build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_features = [
        "prior_events",
        "prior_events_7d",
        "prior_events_30d",
        "prior_events_90d",
        "tenure_days",
        "prev_gap_days",
        "prior_device_mobile_share",
        "prior_device_desktop_share",
        "prior_device_tablet_share",
        "start_month_num",
        "start_dow",
        "total_alunos",
    ]
    categorical_features = ["utm_group", "estado_group", "stage_group", "last_device_group"]

    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                min_frequency=0.002,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    return pre, numeric_features, categorical_features


def _structured_y(event: np.ndarray, duration: np.ndarray) -> np.ndarray:
    y = np.zeros(len(event), dtype=[("event", "bool"), ("time", "f8")])
    y["event"] = event.astype(bool)
    y["time"] = duration.astype(float)
    return y


def _aft_survival_prob(mu: np.ndarray, t: np.ndarray, sigma: float) -> np.ndarray:
    t_clip = np.clip(t.astype(float), 1e-6, None)
    z = (np.log(t_clip) - mu) / max(sigma, 1e-6)
    return 1.0 - norm.cdf(z)


def _make_time_grid(max_t: float) -> np.ndarray:
    mx = max(2.0, float(max_t))
    grid = np.linspace(1.0, mx, num=15)
    return np.unique(np.maximum(grid, 1.0))


def _capacity_metrics(
    y_true_no_return: np.ndarray,
    score: np.ndarray,
    capacities: Iterable[float],
    tp_value: float = 3.0,
    fp_cost: float = 1.0,
    fn_cost: float = 2.0,
) -> List[Dict[str, Any]]:
    n = len(score)
    out: List[Dict[str, Any]] = []
    order = np.argsort(-score)
    for cap in capacities:
        k = max(1, int(math.floor(cap * n)))
        idx = order[:k]
        pred = np.zeros(n, dtype=int)
        pred[idx] = 1
        y = y_true_no_return.astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        net_total = tp * tp_value - fp * fp_cost - fn * fn_cost
        out.append(
            {
                "capacity": float(cap),
                "k": int(k),
                "precision_at_k": precision,
                "recall_at_k": recall,
                "net_value_per_1000": float(net_total * 1000.0 / max(n, 1)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "n": int(n),
            }
        )
    return out


def _build_user_score_table(
    panel_eval: pd.DataFrame,
    y_true_tau: np.ndarray,
    y_true_month_end: np.ndarray,
    month_end_observable: np.ndarray,
    risk_tau: np.ndarray,
    risk_month_end: np.ndarray,
    capacities: Iterable[float],
    model_name: str,
    test_month: pd.Timestamp,
    tau_days: float,
) -> pd.DataFrame:
    n = len(risk_tau)
    if n == 0:
        return pd.DataFrame()

    order = np.argsort(-risk_tau)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1, dtype=int)

    out = panel_eval[
        [
            "panel_row_id",
            "unique_id",
            "as_of_ts",
            "as_of_month",
            "elapsed_days",
            "horizon_days_month_end",
            "prior_events",
            "prior_events_30d",
            "prior_events_90d",
        ]
    ].copy()
    out["test_month"] = str(pd.Timestamp(test_month).date())
    out["model"] = str(model_name)
    out["score_policy"] = "conditional_survival_continuous"
    out["tau_days"] = float(tau_days)
    out["risk_non_return_tau"] = np.asarray(risk_tau, dtype=float)
    out["risk_non_return_month_end"] = np.asarray(risk_month_end, dtype=float)
    out["risk_rank"] = rank.astype(int)
    out["risk_rank_pct"] = out["risk_rank"].astype(float) / float(n)
    out["actual_no_return_tau"] = np.asarray(y_true_tau, dtype=float)
    out["actual_no_return_month_end"] = np.asarray(y_true_month_end, dtype=float)
    out["month_end_observable"] = np.asarray(month_end_observable, dtype=int)
    out["n_eval_rows_fold"] = int(n)
    out["n_eval_positive_tau_fold"] = int(np.nansum(y_true_tau))

    for cap in capacities:
        cap_int = int(round(float(cap) * 100))
        k = max(1, int(math.floor(float(cap) * n)))
        out[f"top_{cap_int}pct"] = (out["risk_rank"] <= k).astype(int)

    return out


def _assert_no_leakage(df: pd.DataFrame, as_of_col: str = "as_of_ts") -> None:
    checks = [
        ("spell_start_day", as_of_col),
        ("last_event_before_as_of", as_of_col),
    ]
    for left, right in checks:
        if left in df.columns and right in df.columns:
            bad = df[pd.to_datetime(df[left]) > pd.to_datetime(df[right])]
            if len(bad) > 0:
                raise ValueError(f"Leakage detected: {left} > {right} for {len(bad)} rows")


def _assert_spell_validity(spells: pd.DataFrame, snapshot_day: pd.Timestamp) -> None:
    if spells.empty:
        raise ValueError("Spell validity check failed: empty spells dataframe.")
    durations = pd.to_numeric(spells.get("duration_days"), errors="coerce")
    if durations.isna().any() or (durations < 1).any():
        raise ValueError("Spell validity check failed: duration_days must be >= 1 with no nulls.")

    spell_start = pd.to_datetime(spells.get("spell_start_day"), errors="coerce")
    spell_end = pd.to_datetime(spells.get("spell_end_day"), errors="coerce")
    if ((spell_end < spell_start).fillna(False)).any():
        raise ValueError("Spell validity check failed: spell_end_day < spell_start_day found.")
    if ((spell_end > pd.to_datetime(snapshot_day)).fillna(False)).any():
        raise ValueError("Spell validity check failed: spell_end_day exceeds snapshot day.")

    by_user = spells.sort_values(["unique_id", "spell_idx"]).groupby("unique_id", sort=False)
    invalid_users = []
    for uid, grp in by_user:
        n_censored = int((grp["event_observed"].astype(int) == 0).sum())
        last_is_censored = int(grp.iloc[-1]["event_observed"]) == 0
        if n_censored != 1 or not last_is_censored:
            invalid_users.append(uid)
    if invalid_users:
        raise ValueError(
            f"Spell validity check failed for {len(invalid_users)} users: "
            "each path must end with exactly one terminal censored spell."
        )


def build_event_days(cfg: SurvivalBenchmarkConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, Dict[str, int]]:
    conn = duckdb.connect()
    conn.execute("PRAGMA threads=4")
    conn.execute(
        f"CREATE VIEW dim AS SELECT * FROM read_csv('{q(cfg.data_dir / 'dim_teachers.csv')}', delim=';', header=true, ignore_errors=true)"
    )
    conn.execute(
        f"CREATE VIEW entries AS SELECT * FROM read_csv_auto('{q(cfg.data_dir / 'fct_teachers_entries.csv')}', header=true)"
    )
    conn.execute(
        f"CREATE VIEW interactions AS SELECT * FROM read_csv_auto('{q(cfg.data_dir / 'fct_teachers_contents_interactions.csv')}', header=true)"
    )

    events = conn.execute(
        f"""
        WITH entries_clean AS (
            SELECT
                e.unique_id,
                e.data_fim AS ts,
                DATE(e.data_fim) AS event_day,
                'entry_clean' AS src,
                'unknown' AS device_group
            FROM entries e
            INNER JOIN dim d USING(unique_id)
            WHERE e.data_inicio IS NOT NULL
              AND e.data_fim IS NOT NULL
              AND (epoch(e.data_fim)-epoch(e.data_inicio)) > {cfg.min_session_seconds}
        ),
        inter_clean AS (
            SELECT
                i.unique_id,
                i.data_inicio AS ts,
                DATE(i.data_inicio) AS event_day,
                'interaction' AS src,
                CASE
                    WHEN lower(coalesce(i.user_agent_device_type, '')) LIKE '%mobile%' THEN 'mobile'
                    WHEN lower(coalesce(i.user_agent_device_type, '')) LIKE '%tablet%' THEN 'tablet'
                    WHEN lower(coalesce(i.user_agent_device_type, '')) LIKE '%desktop%'
                         OR lower(coalesce(i.user_agent_device_type, '')) LIKE '%computer%' THEN 'desktop'
                    WHEN trim(coalesce(i.user_agent_device_type, '')) = '' THEN 'unknown'
                    ELSE 'other'
                END AS device_group
            FROM interactions i
            INNER JOIN dim d USING(unique_id)
            WHERE i.data_inicio IS NOT NULL
        ),
        all_events AS (
            SELECT * FROM entries_clean
            UNION ALL
            SELECT * FROM inter_clean
        )
        SELECT
            unique_id,
            event_day,
            MIN(ts) AS first_ts_in_day,
            COUNT(*) AS raw_events_in_day,
            SUM(CASE WHEN src='interaction' THEN 1 ELSE 0 END)::BIGINT AS interaction_events_in_day,
            SUM(CASE WHEN src='entry_clean' THEN 1 ELSE 0 END)::BIGINT AS entry_events_in_day,
            SUM(CASE WHEN device_group='mobile' THEN 1 ELSE 0 END)::BIGINT AS mobile_events_in_day,
            SUM(CASE WHEN device_group='desktop' THEN 1 ELSE 0 END)::BIGINT AS desktop_events_in_day,
            SUM(CASE WHEN device_group='tablet' THEN 1 ELSE 0 END)::BIGINT AS tablet_events_in_day,
            SUM(CASE WHEN device_group='other' THEN 1 ELSE 0 END)::BIGINT AS other_events_in_day,
            SUM(CASE WHEN device_group='unknown' THEN 1 ELSE 0 END)::BIGINT AS unknown_events_in_day
        FROM all_events
        GROUP BY unique_id, event_day
        ORDER BY unique_id, event_day
        """
    ).fetchdf()

    event_quality_row = conn.execute(
        f"""
        SELECT
            COUNT(*)::BIGINT AS entries_rows_joined_dim,
            SUM(
                CASE
                    WHEN e.data_inicio IS NOT NULL
                     AND e.data_fim IS NOT NULL
                     AND (epoch(e.data_fim)-epoch(e.data_inicio)) <= {cfg.min_session_seconds}
                    THEN 1 ELSE 0
                END
            )::BIGINT AS entries_ping_filtered_rows,
            SUM(
                CASE
                    WHEN e.data_inicio IS NOT NULL
                     AND e.data_fim IS NOT NULL
                     AND (epoch(e.data_fim)-epoch(e.data_inicio)) > {cfg.min_session_seconds}
                    THEN 1 ELSE 0
                END
            )::BIGINT AS entries_clean_rows
        FROM entries e
        INNER JOIN dim d USING(unique_id)
        """
    ).fetchone()

    teacher_dim = conn.execute(
        """
        SELECT
            unique_id,
            estado,
            utm_origin,
            currentstage,
            total_alunos,
            data_entrada
        FROM dim
        """
    ).fetchdf()

    snapshot_ts = conn.execute(
        """
        WITH snap AS (
            SELECT GREATEST(
                (SELECT MAX(data_fim) FROM entries),
                (SELECT MAX(data_inicio) FROM interactions)
            ) AS snapshot_ts
        )
        SELECT snapshot_ts FROM snap
        """
    ).fetchone()[0]
    conn.close()

    events["event_day"] = pd.to_datetime(events["event_day"], errors="coerce")
    teacher_dim["data_entrada"] = pd.to_datetime(teacher_dim["data_entrada"], errors="coerce")
    event_quality = {
        "entries_rows_joined_dim": int(event_quality_row[0]) if event_quality_row is not None else 0,
        "entries_ping_filtered_rows": int(event_quality_row[1]) if event_quality_row is not None else 0,
        "entries_clean_rows": int(event_quality_row[2]) if event_quality_row is not None else 0,
    }
    return events, teacher_dim, pd.to_datetime(snapshot_ts), event_quality


def build_spells_and_panel(
    events: pd.DataFrame,
    teacher_dim: pd.DataFrame,
    snapshot_ts: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = events.sort_values(["unique_id", "event_day"]).copy()
    snapshot_day = snapshot_ts.normalize()

    teacher_dim = teacher_dim.copy()
    teacher_dim["utm_group"] = teacher_dim["utm_origin"].apply(normalize_utm)
    teacher_dim["estado_group"] = teacher_dim["estado"].fillna("missing").replace("", "missing")
    teacher_dim["stage_group"] = teacher_dim["currentstage"].fillna("missing").replace("", "missing")
    teacher_dim["total_alunos"] = pd.to_numeric(teacher_dim["total_alunos"], errors="coerce")
    teacher_lookup = teacher_dim.set_index("unique_id")

    spell_rows: List[Dict[str, Any]] = []
    risk_rows: List[Dict[str, Any]] = []

    for uid, grp in events.groupby("unique_id", sort=False):
        grp = grp.copy()
        grp["event_day"] = pd.to_datetime(grp["event_day"], errors="coerce").dt.normalize()
        grp = grp.dropna(subset=["event_day"]).sort_values("event_day").reset_index(drop=True)
        if grp.empty:
            continue

        first_day = pd.to_datetime(grp.loc[0, "event_day"]).normalize()
        last_day: Optional[pd.Timestamp] = None
        prior_events = 0
        window_7d: deque[pd.Timestamp] = deque()
        window_30d: deque[pd.Timestamp] = deque()
        window_90d: deque[pd.Timestamp] = deque()
        device_hist: Dict[str, float] = {
            "mobile": 0.0,
            "desktop": 0.0,
            "tablet": 0.0,
            "other": 0.0,
            "unknown": 0.0,
        }
        last_device_group = "unknown"

        for j, row in grp.iterrows():
            start_day = pd.to_datetime(row["event_day"]).normalize()
            next_day = pd.to_datetime(grp.loc[j + 1, "event_day"]).normalize() if j + 1 < len(grp) else None

            while window_7d and (start_day - window_7d[0]).days > 7:
                window_7d.popleft()
            while window_30d and (start_day - window_30d[0]).days > 30:
                window_30d.popleft()
            while window_90d and (start_day - window_90d[0]).days > 90:
                window_90d.popleft()

            event_observed = 1 if next_day is not None else 0
            if next_day is not None:
                duration = max(int((next_day - start_day).days), 1)
                spell_end = next_day
            else:
                duration = max(int((snapshot_day - start_day).days), 1)
                spell_end = snapshot_day

            prev_gap = np.nan
            if last_day is not None:
                prev_gap = max(int((start_day - last_day).days), 0)

            prior_7d = len(window_7d)
            prior_30d = len(window_30d)
            prior_90d = len(window_90d)

            hist_total = float(sum(device_hist.values()))
            prior_device_mobile_share = float(device_hist["mobile"] / hist_total) if hist_total > 0 else 0.0
            prior_device_desktop_share = float(device_hist["desktop"] / hist_total) if hist_total > 0 else 0.0
            prior_device_tablet_share = float(device_hist["tablet"] / hist_total) if hist_total > 0 else 0.0

            dim = teacher_lookup.loc[uid] if uid in teacher_lookup.index else None
            total_alunos = float(dim["total_alunos"]) if dim is not None and pd.notna(dim["total_alunos"]) else np.nan
            utm_group = str(dim["utm_group"]) if dim is not None else "missing"
            estado_group = str(dim["estado_group"]) if dim is not None else "missing"
            stage_group = str(dim["stage_group"]) if dim is not None else "missing"

            spell_rows.append(
                {
                    "unique_id": uid,
                    "spell_idx": j,
                    "spell_start_day": start_day,
                    "spell_end_day": spell_end,
                    "next_event_day": next_day,
                    "duration_days": float(duration),
                    "event_observed": int(event_observed),
                    "prior_events": float(prior_events),
                    "prior_events_7d": float(prior_7d),
                    "prior_events_30d": float(prior_30d),
                    "prior_events_90d": float(prior_90d),
                    "prior_device_mobile_share": prior_device_mobile_share,
                    "prior_device_desktop_share": prior_device_desktop_share,
                    "prior_device_tablet_share": prior_device_tablet_share,
                    "last_device_group": last_device_group,
                    "prev_gap_days": float(prev_gap) if pd.notna(prev_gap) else np.nan,
                    "tenure_days": float(max(int((start_day - first_day).days), 0)),
                    "start_month": pd.Period(start_day, freq="M").to_timestamp(),
                    "start_month_num": int(start_day.month),
                    "start_dow": int(start_day.dayofweek),
                    "utm_group": utm_group,
                    "estado_group": estado_group,
                    "stage_group": stage_group,
                    "total_alunos": total_alunos,
                }
            )

            next_month_end = _end_of_next_month(start_day)
            horizon_days_month_end = max(int((next_month_end.normalize() - start_day.normalize()).days), 1)
            risk_rows.append(
                {
                    "unique_id": uid,
                    "as_of_ts": start_day,
                    "as_of_month": pd.Period(start_day, freq="M").to_timestamp(),
                    "last_event_before_as_of": start_day,
                    "current_spell_idx": j,
                    "next_event_after_last": next_day,
                    "elapsed_days": 0.0,
                    "horizon_days_month_end": float(horizon_days_month_end),
                    "prior_events": float(prior_events),
                    "prior_events_7d": float(prior_7d),
                    "prior_events_30d": float(prior_30d),
                    "prior_events_90d": float(prior_90d),
                    "prior_device_mobile_share": prior_device_mobile_share,
                    "prior_device_desktop_share": prior_device_desktop_share,
                    "prior_device_tablet_share": prior_device_tablet_share,
                    "last_device_group": last_device_group,
                    "prev_gap_days": float(prev_gap) if pd.notna(prev_gap) else np.nan,
                    "tenure_days": float(max(int((start_day - first_day).days), 0)),
                    "start_month_num": int(start_day.month),
                    "start_dow": int(start_day.dayofweek),
                    "utm_group": utm_group,
                    "estado_group": estado_group,
                    "stage_group": stage_group,
                    "total_alunos": total_alunos,
                    "snapshot_ts": snapshot_ts,
                }
            )

            mobile_events = pd.to_numeric(row.get("mobile_events_in_day", 0), errors="coerce")
            desktop_events = pd.to_numeric(row.get("desktop_events_in_day", 0), errors="coerce")
            tablet_events = pd.to_numeric(row.get("tablet_events_in_day", 0), errors="coerce")
            other_events = pd.to_numeric(row.get("other_events_in_day", 0), errors="coerce")
            unknown_events = pd.to_numeric(row.get("unknown_events_in_day", 0), errors="coerce")
            day_device = {
                "mobile": float(mobile_events) if pd.notna(mobile_events) else 0.0,
                "desktop": float(desktop_events) if pd.notna(desktop_events) else 0.0,
                "tablet": float(tablet_events) if pd.notna(tablet_events) else 0.0,
                "other": float(other_events) if pd.notna(other_events) else 0.0,
                "unknown": float(unknown_events) if pd.notna(unknown_events) else 0.0,
            }
            for k in device_hist:
                device_hist[k] += max(day_device.get(k, 0.0), 0.0)
            interaction_events = pd.to_numeric(row.get("interaction_events_in_day", 0), errors="coerce")
            has_interaction_on_day = pd.notna(interaction_events) and float(interaction_events) > 0
            if has_interaction_on_day:
                day_main_device = max(day_device, key=day_device.get)
                if day_device[day_main_device] > 0:
                    last_device_group = day_main_device

            prior_events += 1
            last_day = start_day
            window_7d.append(start_day)
            window_30d.append(start_day)
            window_90d.append(start_day)

    spells = pd.DataFrame(spell_rows)
    panel = pd.DataFrame(risk_rows)
    panel["panel_row_id"] = np.arange(len(panel), dtype=np.int64)

    if spells.empty or panel.empty:
        raise ValueError("Spells/panel vazios: sem dados para benchmark de survival.")

    spells["start_month"] = pd.to_datetime(spells["start_month"])  # fold key
    panel["as_of_month"] = pd.to_datetime(panel["as_of_month"])
    _assert_no_leakage(panel)
    _assert_spell_validity(spells, snapshot_day=snapshot_day)

    # Derived outcomes for operational evaluation filled later per tau.
    truth = panel[
        [
            "panel_row_id",
            "unique_id",
            "as_of_ts",
            "as_of_month",
            "next_event_after_last",
            "snapshot_ts",
            "horizon_days_month_end",
        ]
    ].copy()
    truth["next_event_after_last"] = pd.to_datetime(truth["next_event_after_last"], errors="coerce")
    truth["snapshot_ts"] = pd.to_datetime(truth["snapshot_ts"], errors="coerce")

    return spells, panel, truth


def _choose_months(spells: pd.DataFrame, cfg: SurvivalBenchmarkConfig) -> List[pd.Timestamp]:
    months = sorted(pd.to_datetime(spells["start_month"]).dropna().dt.to_period("M").dt.to_timestamp().unique())
    if len(months) > cfg.max_backtest_months:
        months = months[-cfg.max_backtest_months :]
    return [pd.Timestamp(m) for m in months]


def _split_fold(
    months: List[pd.Timestamp],
    idx: int,
) -> Tuple[List[pd.Timestamp], pd.Timestamp, pd.Timestamp]:
    test_month = months[idx]
    val_month = months[idx - 1]
    train_months = months[: idx - 1]
    return train_months, val_month, test_month


def _subsample_train(df: pd.DataFrame, event_col: str, max_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    pos = df[df[event_col] == 1]
    neg = df[df[event_col] == 0]
    if pos.empty or neg.empty:
        return df.sample(n=max_rows, random_state=seed)
    half = max_rows // 2
    n_pos = min(len(pos), half)
    n_neg = min(len(neg), max_rows - n_pos)
    out = pd.concat(
        [
            pos.sample(n=n_pos, random_state=seed),
            neg.sample(n=n_neg, random_state=seed),
        ],
        ignore_index=True,
    )
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _subsample_rows(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed).sort_values("as_of_ts").reset_index(drop=True)


def _fit_rsf_with_budget(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    trials: int,
    seed: int,
) -> Tuple[RandomSurvivalForest, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    best_score = -np.inf
    best_model: Optional[RandomSurvivalForest] = None
    best_params: Dict[str, Any] = {}

    for _ in range(trials):
        params = {
            "n_estimators": int(rng.choice([150, 250, 350])),
            "max_depth": int(rng.choice([6, 10, 14, 18])),
            "min_samples_leaf": int(rng.choice([8, 16, 32, 64])),
            "max_features": float(rng.choice([0.3, 0.5, 0.7, 1.0])),
            "n_jobs": 1,
            "random_state": seed,
        }
        model = RandomSurvivalForest(**params)
        model.fit(x_train, y_train)
        rs = model.predict(x_val)
        cidx = concordance_index_censored(y_val["event"], y_val["time"], rs)[0]
        if cidx > best_score:
            best_score = float(cidx)
            best_model = model
            best_params = params

    if best_model is None:
        raise RuntimeError("RSF tuning did not produce a model.")
    return best_model, {"val_cindex_censored": best_score, **best_params}


def _fit_xgb_aft_with_budget(
    x_train: np.ndarray,
    event_train: np.ndarray,
    dur_train: np.ndarray,
    x_val: np.ndarray,
    event_val: np.ndarray,
    dur_val: np.ndarray,
    trials: int,
    seed: int,
) -> Tuple[xgb.Booster, Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    best_loss = np.inf
    best_model: Optional[xgb.Booster] = None
    best_params: Dict[str, Any] = {}

    lower_train = dur_train.astype(float).copy()
    upper_train = dur_train.astype(float).copy()
    upper_train[event_train == 0] = np.inf

    lower_val = dur_val.astype(float).copy()
    upper_val = dur_val.astype(float).copy()
    upper_val[event_val == 0] = np.inf

    dtrain = xgb.DMatrix(x_train)
    dtrain.set_float_info("label_lower_bound", lower_train)
    dtrain.set_float_info("label_upper_bound", upper_train)

    dval = xgb.DMatrix(x_val)
    dval.set_float_info("label_lower_bound", lower_val)
    dval.set_float_info("label_upper_bound", upper_val)

    for _ in range(trials):
        scale = float(rng.choice([0.8, 1.0, 1.2, 1.5]))
        params = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "tree_method": "hist",
            "learning_rate": float(rng.choice([0.03, 0.05, 0.08])),
            "max_depth": int(rng.choice([4, 6, 8])),
            "min_child_weight": float(rng.choice([1.0, 3.0, 6.0])),
            "subsample": float(rng.choice([0.7, 0.85, 1.0])),
            "colsample_bytree": float(rng.choice([0.6, 0.8, 1.0])),
            "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": scale,
            "seed": seed,
        }
        evals_result: Dict[str, Dict[str, List[float]]] = {}
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=450,
            evals=[(dval, "validation")],
            evals_result=evals_result,
            early_stopping_rounds=35,
            verbose_eval=False,
        )
        loss = float(evals_result["validation"]["aft-nloglik"][booster.best_iteration])
        if loss < best_loss:
            best_loss = loss
            best_model = booster
            best_params = {
                **params,
                "best_iteration": int(booster.best_iteration),
                "val_aft_nloglik": loss,
            }

    if best_model is None:
        raise RuntimeError("XGB-AFT tuning did not produce a model.")
    return best_model, best_params


def _rsf_predict_array(model: RandomSurvivalForest, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    unique_times = np.asarray(model.unique_times_, dtype=float)
    surv_array = np.asarray(model.predict_survival_function(x, return_array=True), dtype=float)
    return unique_times, np.clip(surv_array, 1e-8, 1.0)


def _rsf_eval_matrix(unique_times: np.ndarray, surv_array: np.ndarray, times: np.ndarray) -> np.ndarray:
    grid = np.asarray(times, dtype=float)
    idx = np.searchsorted(unique_times, grid, side="right") - 1
    idx_clipped = np.clip(idx, 0, max(len(unique_times) - 1, 0))
    out = surv_array[:, idx_clipped]
    if np.any(idx < 0):
        out[:, idx < 0] = 1.0
    return np.clip(out, 1e-8, 1.0)


def _rsf_eval_vector(unique_times: np.ndarray, surv_array: np.ndarray, row_times: np.ndarray) -> np.ndarray:
    t = np.asarray(row_times, dtype=float)
    if len(surv_array) != len(t):
        raise ValueError("RSF per-row evaluation mismatch between surv_array and row_times length.")
    idx = np.searchsorted(unique_times, t, side="right") - 1
    idx_clipped = np.clip(idx, 0, max(len(unique_times) - 1, 0))
    out = surv_array[np.arange(len(t)), idx_clipped]
    out[idx < 0] = 1.0
    return np.clip(out, 1e-8, 1.0)


def _xgb_survival(booster: xgb.Booster, x: np.ndarray, times: np.ndarray, sigma: float) -> np.ndarray:
    d = xgb.DMatrix(x)
    mu = booster.predict(d)
    surv = np.zeros((len(mu), len(times)), dtype=float)
    for j, t in enumerate(times):
        surv[:, j] = _aft_survival_prob(mu, np.full(len(mu), float(t)), sigma=sigma)
    return np.clip(surv, 1e-8, 1.0)


def _conditional_no_return(
    survival_at_elapsed: np.ndarray,
    survival_at_elapsed_plus_h: np.ndarray,
) -> np.ndarray:
    return np.clip(_safe_div(survival_at_elapsed_plus_h, survival_at_elapsed), 0.0, 1.0)


def _calibration_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    df = pd.DataFrame({"y_true": y_true.astype(float), "y_pred": y_pred.astype(float)})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return pd.DataFrame(columns=["bin", "mean_pred", "obs_rate", "n"])

    unique_pred = df["y_pred"].nunique()
    bins = max(2, min(int(n_bins), int(unique_pred)))
    if bins < 2:
        return pd.DataFrame(columns=["bin", "mean_pred", "obs_rate", "n"])

    try:
        df["bin"] = pd.qcut(df["y_pred"], q=bins, labels=False, duplicates="drop")
    except Exception:
        return pd.DataFrame(columns=["bin", "mean_pred", "obs_rate", "n"])

    out = (
        df.groupby("bin", as_index=False)
        .agg(
            mean_pred=("y_pred", "mean"),
            obs_rate=("y_true", "mean"),
            n=("y_true", "size"),
        )
        .sort_values("bin")
        .reset_index(drop=True)
    )
    return out


def run_benchmark(cfg: SurvivalBenchmarkConfig) -> Dict[str, Any]:
    out_dir = cfg.output_dir / "survival_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    events, teacher_dim, snapshot_ts, event_quality = build_event_days(cfg)
    if cfg.max_users > 0:
        top_users = (
            events.groupby("unique_id", as_index=False)
            .size()
            .sort_values("size", ascending=False)
            .head(cfg.max_users)["unique_id"]
            .tolist()
        )
        events = events[events["unique_id"].isin(top_users)].copy()
        teacher_dim = teacher_dim[teacher_dim["unique_id"].isin(top_users)].copy()
        LOGGER.info("Smoke cap applied | max_users=%d | users_kept=%d", cfg.max_users, len(top_users))

    spells, panel, truth = build_spells_and_panel(events, teacher_dim, snapshot_ts)

    months = _choose_months(spells, cfg)
    if len(months) < cfg.min_train_months + 2:
        raise ValueError("Not enough months for walk-forward (train + val + test).")

    feature_cols = [
        "prior_events",
        "prior_events_7d",
        "prior_events_30d",
        "prior_events_90d",
        "tenure_days",
        "prev_gap_days",
        "prior_device_mobile_share",
        "prior_device_desktop_share",
        "prior_device_tablet_share",
        "start_month_num",
        "start_dow",
        "utm_group",
        "estado_group",
        "stage_group",
        "last_device_group",
        "total_alunos",
    ]

    fold_rows: List[Dict[str, Any]] = []
    capacity_rows: List[Dict[str, Any]] = []
    brier_curve_rows: List[Dict[str, Any]] = []
    calibration_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []
    user_score_frames: List[pd.DataFrame] = []
    truth_lookup = truth.set_index("panel_row_id", drop=False)

    for idx in range(cfg.min_train_months + 1, len(months)):
        train_months, val_month, test_month = _split_fold(months, idx)

        train_spells = spells[spells["start_month"].isin(train_months)].copy()
        val_spells = spells[spells["start_month"] == val_month].copy()
        test_spells = spells[spells["start_month"] == test_month].copy()

        train_spells = _subsample_train(train_spells, "event_observed", cfg.max_train_rows, cfg.random_seed)

        if train_spells.empty or val_spells.empty or test_spells.empty:
            continue

        tau_fold = float(np.nanquantile(train_spells.loc[train_spells["event_observed"] == 1, "duration_days"], cfg.tau_quantile))
        tau_fold = max(tau_fold, 7.0)

        val_panel = panel[panel["as_of_month"] == val_month].copy()
        test_panel = panel[panel["as_of_month"] == test_month].copy()

        val_panel = _subsample_rows(val_panel, cfg.max_panel_rows, cfg.random_seed + idx * 11 + 1)
        test_panel = _subsample_rows(test_panel, cfg.max_panel_rows, cfg.random_seed + idx * 11 + 7)

        if val_panel.empty or test_panel.empty:
            continue

        for dfx in [train_spells, val_spells, test_spells, val_panel, test_panel]:
            for c in feature_cols:
                if c not in dfx.columns:
                    dfx[c] = np.nan

        pre, _, _ = _build_preprocessor(train_spells[feature_cols])

        x_train = pre.fit_transform(train_spells[feature_cols])
        x_val = pre.transform(val_spells[feature_cols])
        x_test = pre.transform(test_spells[feature_cols])

        y_train = _structured_y(train_spells["event_observed"].to_numpy(), train_spells["duration_days"].to_numpy())
        y_val = _structured_y(val_spells["event_observed"].to_numpy(), val_spells["duration_days"].to_numpy())
        y_test = _structured_y(test_spells["event_observed"].to_numpy(), test_spells["duration_days"].to_numpy())

        rsf, rsf_params = _fit_rsf_with_budget(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            trials=cfg.tuning_trials,
            seed=cfg.random_seed,
        )

        aft, aft_params = _fit_xgb_aft_with_budget(
            x_train=x_train,
            event_train=train_spells["event_observed"].to_numpy().astype(int),
            dur_train=train_spells["duration_days"].to_numpy().astype(float),
            x_val=x_val,
            event_val=val_spells["event_observed"].to_numpy().astype(int),
            dur_val=val_spells["duration_days"].to_numpy().astype(float),
            trials=cfg.tuning_trials,
            seed=cfg.random_seed,
        )

        # Survival-first metrics on spell test set.
        test_time_max = float(np.nanmax(y_test["time"]))
        time_upper = min(float(np.nanquantile(train_spells["duration_days"], 0.95)), max(test_time_max - 1e-6, 1.0))
        if time_upper <= 1.0:
            continue
        time_grid = _make_time_grid(time_upper)
        tau_eval = float(min(tau_fold, float(time_grid[-1])))
        rsf_test_times, rsf_test_surv_array = _rsf_predict_array(rsf, x_test)
        rsf_surv = _rsf_eval_matrix(rsf_test_times, rsf_test_surv_array, time_grid)
        aft_sigma = float(aft_params.get("aft_loss_distribution_scale", 1.0))
        aft_surv = _xgb_survival(aft, x_test, time_grid, sigma=aft_sigma)

        try:
            rsf_uno = float(concordance_index_ipcw(y_train, y_test, 1.0 - rsf_surv[:, min(len(time_grid) - 1, 5)], tau=float(time_grid[-1]))[0])
        except Exception:
            rsf_uno = float(concordance_index_censored(y_test["event"], y_test["time"], 1.0 - rsf_surv[:, min(len(time_grid) - 1, 5)])[0])

        try:
            aft_uno = float(concordance_index_ipcw(y_train, y_test, 1.0 - aft_surv[:, min(len(time_grid) - 1, 5)], tau=float(time_grid[-1]))[0])
        except Exception:
            aft_uno = float(concordance_index_censored(y_test["event"], y_test["time"], 1.0 - aft_surv[:, min(len(time_grid) - 1, 5)])[0])

        rsf_ibs = float(integrated_brier_score(y_train, y_test, rsf_surv, time_grid))
        aft_ibs = float(integrated_brier_score(y_train, y_test, aft_surv, time_grid))

        rsf_bs_tau = float(
            brier_score(
                y_train,
                y_test,
                rsf_surv[:, [np.argmin(np.abs(time_grid - tau_eval))]],
                np.array([tau_eval]),
            )[1][0]
        )
        aft_bs_tau = float(
            brier_score(
                y_train,
                y_test,
                aft_surv[:, [np.argmin(np.abs(time_grid - tau_eval))]],
                np.array([tau_eval]),
            )[1][0]
        )
        rsf_brier_curve = brier_score(y_train, y_test, rsf_surv, time_grid)[1]
        aft_brier_curve = brier_score(y_train, y_test, aft_surv, time_grid)[1]
        for horizon_days, rsf_brier, aft_brier in zip(time_grid, rsf_brier_curve, aft_brier_curve):
            brier_curve_rows.append(
                {
                    "test_month": str(test_month.date()),
                    "model": "rsf",
                    "horizon_days": float(horizon_days),
                    "brier_score": float(rsf_brier),
                }
            )
            brier_curve_rows.append(
                {
                    "test_month": str(test_month.date()),
                    "model": "xgb_aft",
                    "horizon_days": float(horizon_days),
                    "brier_score": float(aft_brier),
                }
            )

        # Operational scoring on panel rows for val/test.
        x_val_panel = pre.transform(val_panel[feature_cols])
        x_test_panel = pre.transform(test_panel[feature_cols])

        e_val = val_panel["elapsed_days"].to_numpy(dtype=float)
        e_test = test_panel["elapsed_days"].to_numpy(dtype=float)

        # Build conditional risks for tau and month-end.
        rsf_val_times, rsf_val_surv_array = _rsf_predict_array(rsf, x_val_panel)
        rsf_s_val_e = _rsf_eval_vector(rsf_val_times, rsf_val_surv_array, np.maximum(e_val, 1e-3))
        rsf_s_val_tau = _rsf_eval_vector(rsf_val_times, rsf_val_surv_array, np.maximum(e_val + tau_fold, 1e-3))
        rsf_risk_val = _conditional_no_return(rsf_s_val_e, rsf_s_val_tau)

        aft_mu_val = aft.predict(xgb.DMatrix(x_val_panel))
        aft_s_val_e = _aft_survival_prob(aft_mu_val, np.maximum(e_val, 1e-3), aft_sigma)
        aft_s_val_tau = _aft_survival_prob(aft_mu_val, np.maximum(e_val + tau_fold, 1e-3), aft_sigma)
        aft_risk_val = _conditional_no_return(aft_s_val_e, aft_s_val_tau)

        val_truth = truth_lookup.reindex(val_panel["panel_row_id"]).reset_index(drop=True).copy()
        test_truth = truth_lookup.reindex(test_panel["panel_row_id"]).reset_index(drop=True).copy()

        for dfx in [val_truth, test_truth]:
            dfx["tau_days"] = tau_fold
            horizon_end = pd.to_datetime(dfx["as_of_ts"]) + pd.to_timedelta(dfx["tau_days"], unit="D")
            dfx["tau_observable"] = horizon_end <= pd.to_datetime(dfx["snapshot_ts"])
            next_evt = pd.to_datetime(dfx["next_event_after_last"], errors="coerce")
            dfx["no_return_tau"] = np.where(
                dfx["tau_observable"],
                ((next_evt.isna()) | (next_evt > horizon_end)).astype(int),
                np.nan,
            )

            me_end = pd.to_datetime(dfx["as_of_ts"]) + pd.to_timedelta(dfx["horizon_days_month_end"], unit="D")
            dfx["month_end_observable"] = me_end <= pd.to_datetime(dfx["snapshot_ts"])
            dfx["no_return_month_end"] = np.where(
                dfx["month_end_observable"],
                ((next_evt.isna()) | (next_evt > me_end)).astype(int),
                np.nan,
            )

        val_idx = val_truth["no_return_tau"].notna().to_numpy()
        test_idx = test_truth["no_return_tau"].notna().to_numpy()
        if not val_idx.any() or not test_idx.any():
            continue

        y_val_nr = val_truth.loc[val_idx, "no_return_tau"].to_numpy(dtype=float)
        y_test_nr = test_truth.loc[test_idx, "no_return_tau"].to_numpy(dtype=float)
        test_panel_eval = test_panel.loc[
            test_idx,
            [
                "panel_row_id",
                "unique_id",
                "as_of_ts",
                "as_of_month",
                "elapsed_days",
                "horizon_days_month_end",
                "prior_events",
                "prior_events_30d",
                "prior_events_90d",
            ],
        ].reset_index(drop=True)
        test_truth_eval = test_truth.loc[test_idx].reset_index(drop=True)
        if len(test_panel_eval) != len(y_test_nr):
            raise ValueError("Apple-to-apple integrity failed: test panel rows and labels are misaligned.")
        if not np.array_equal(test_panel_eval["panel_row_id"].to_numpy(), test_truth_eval["panel_row_id"].to_numpy()):
            raise ValueError("Apple-to-apple integrity failed: panel_row_id mismatch in test fold.")

        iso_rsf = IsotonicRegression(out_of_bounds="clip")
        iso_aft = IsotonicRegression(out_of_bounds="clip")
        iso_rsf.fit(rsf_risk_val[val_idx], y_val_nr)
        iso_aft.fit(aft_risk_val[val_idx], y_val_nr)

        rsf_test_panel_times, rsf_test_panel_surv_array = _rsf_predict_array(rsf, x_test_panel)
        rsf_s_test_e = _rsf_eval_vector(rsf_test_panel_times, rsf_test_panel_surv_array, np.maximum(e_test, 1e-3))
        rsf_s_test_tau = _rsf_eval_vector(rsf_test_panel_times, rsf_test_panel_surv_array, np.maximum(e_test + tau_fold, 1e-3))
        rsf_risk_test = _conditional_no_return(rsf_s_test_e, rsf_s_test_tau)

        aft_mu_test = aft.predict(xgb.DMatrix(x_test_panel))
        aft_s_test_e = _aft_survival_prob(aft_mu_test, np.maximum(e_test, 1e-3), aft_sigma)
        aft_s_test_tau = _aft_survival_prob(aft_mu_test, np.maximum(e_test + tau_fold, 1e-3), aft_sigma)
        aft_risk_test = _conditional_no_return(aft_s_test_e, aft_s_test_tau)

        rsf_cal = np.clip(iso_rsf.predict(rsf_risk_test), 0.0, 1.0)
        aft_cal = np.clip(iso_aft.predict(aft_risk_test), 0.0, 1.0)

        # Secondary month-end risk
        h_month = test_panel["horizon_days_month_end"].to_numpy(dtype=float)
        rsf_s_test_me = _rsf_eval_vector(rsf_test_panel_times, rsf_test_panel_surv_array, np.maximum(e_test + h_month, 1e-3))
        aft_s_test_me = _aft_survival_prob(aft_mu_test, np.maximum(e_test + h_month, 1e-3), aft_sigma)
        rsf_risk_me = _conditional_no_return(rsf_s_test_e, rsf_s_test_me)
        aft_risk_me = _conditional_no_return(aft_s_test_e, aft_s_test_me)

        month_end_truth_eval = pd.to_numeric(test_truth_eval["no_return_month_end"], errors="coerce").to_numpy(dtype=float)
        month_end_observable_eval = (
            pd.Series(test_truth_eval["month_end_observable"])
            .fillna(False)
            .astype(int)
            .to_numpy(dtype=int)
        )

        user_score_frames.extend(
            [
                _build_user_score_table(
                    panel_eval=test_panel_eval,
                    y_true_tau=y_test_nr,
                    y_true_month_end=month_end_truth_eval,
                    month_end_observable=month_end_observable_eval,
                    risk_tau=rsf_cal[test_idx],
                    risk_month_end=rsf_risk_me[test_idx],
                    capacities=cfg.capacity_grid,
                    model_name="rsf",
                    test_month=test_month,
                    tau_days=tau_fold,
                ),
                _build_user_score_table(
                    panel_eval=test_panel_eval,
                    y_true_tau=y_test_nr,
                    y_true_month_end=month_end_truth_eval,
                    month_end_observable=month_end_observable_eval,
                    risk_tau=aft_cal[test_idx],
                    risk_month_end=aft_risk_me[test_idx],
                    capacities=cfg.capacity_grid,
                    model_name="xgb_aft",
                    test_month=test_month,
                    tau_days=tau_fold,
                ),
            ]
        )

        # Operational capacity metrics (primary continuous risk).
        for model_name, score in [("rsf", rsf_cal), ("xgb_aft", aft_cal)]:
            caps = _capacity_metrics(y_test_nr, score[test_idx], cfg.capacity_grid)
            for c in caps:
                capacity_rows.append(
                    {
                        "test_month": str(test_month.date()),
                        "model": model_name,
                        "score_policy": "conditional_survival_continuous",
                        "tau_days": float(tau_fold),
                        **c,
                    }
                )
            calib_df = _calibration_table(y_true=y_test_nr, y_pred=score[test_idx], n_bins=10)
            if not calib_df.empty:
                for _, crow in calib_df.iterrows():
                    calibration_rows.append(
                        {
                            "test_month": str(test_month.date()),
                            "model": model_name,
                            "bin": int(crow["bin"]),
                            "mean_pred": float(crow["mean_pred"]),
                            "obs_rate": float(crow["obs_rate"]),
                            "n": int(crow["n"]),
                        }
                    )

        # Calibration / brier at tau on operational panel target.
        rsf_brier_oper = float(np.mean((rsf_cal[test_idx] - y_test_nr) ** 2))
        aft_brier_oper = float(np.mean((aft_cal[test_idx] - y_test_nr) ** 2))

        fold_rows.extend(
            [
                {
                    "test_month": str(test_month.date()),
                    "model": "rsf",
                    "tau_days": float(tau_fold),
                    "uno_cindex": rsf_uno,
                    "ibs": rsf_ibs,
                    "brier_tau_spell": rsf_bs_tau,
                    "brier_tau_operational": rsf_brier_oper,
                    "calibration_mean_pred": float(np.mean(rsf_cal[test_idx])),
                    "calibration_obs_rate": float(np.mean(y_test_nr)),
                    "val_tuning_metric": float(rsf_params.get("val_cindex_censored", np.nan)),
                    "risk_month_end_mean": float(np.nanmean(rsf_risk_me)),
                },
                {
                    "test_month": str(test_month.date()),
                    "model": "xgb_aft",
                    "tau_days": float(tau_fold),
                    "uno_cindex": aft_uno,
                    "ibs": aft_ibs,
                    "brier_tau_spell": aft_bs_tau,
                    "brier_tau_operational": aft_brier_oper,
                    "calibration_mean_pred": float(np.mean(aft_cal[test_idx])),
                    "calibration_obs_rate": float(np.mean(y_test_nr)),
                    "val_tuning_metric": float(aft_params.get("val_aft_nloglik", np.nan)),
                    "risk_month_end_mean": float(np.nanmean(aft_risk_me)),
                },
            ]
        )

        manifest_rows.append(
            {
                "test_month": str(test_month.date()),
                "train_month_start": str(min(train_months).date()) if train_months else None,
                "train_month_end": str(max(train_months).date()) if train_months else None,
                "val_month": str(val_month.date()),
                "n_train_spells": int(len(train_spells)),
                "n_val_spells": int(len(val_spells)),
                "n_test_spells": int(len(test_spells)),
                "n_val_panel": int(len(val_panel)),
                "n_test_panel": int(len(test_panel)),
                "tau_days": float(tau_fold),
            }
        )

    if not fold_rows or not capacity_rows:
        pd.DataFrame(
            columns=[
                "test_month",
                "model",
                "tau_days",
                "uno_cindex",
                "ibs",
                "brier_tau_spell",
                "brier_tau_operational",
                "calibration_mean_pred",
                "calibration_obs_rate",
                "val_tuning_metric",
                "risk_month_end_mean",
            ]
        ).to_csv(out_dir / "survival_model_metrics_folds.csv", index=False)
        pd.DataFrame(
            columns=[
                "model",
                "uno_cindex_mean",
                "uno_cindex_std",
                "ibs_mean",
                "ibs_std",
                "brier_oper_mean",
                "brier_oper_std",
            ]
        ).to_csv(out_dir / "survival_model_metrics_summary.csv", index=False)
        pd.DataFrame(
            columns=[
                "test_month",
                "model",
                "score_policy",
                "tau_days",
                "capacity",
                "k",
                "precision_at_k",
                "recall_at_k",
                "net_value_per_1000",
                "tp",
                "fp",
                "fn",
                "n",
            ]
        ).to_csv(out_dir / "survival_operational_capacity_by_month.csv", index=False)
        pd.DataFrame(
            columns=[
                "model",
                "capacity",
                "precision_at_k_mean",
                "recall_at_k_mean",
                "net_value_per_1000_mean",
                "net_value_per_1000_std",
            ]
        ).to_csv(out_dir / "survival_operational_capacity_summary.csv", index=False)
        pd.DataFrame(columns=["test_month", "model", "horizon_days", "brier_score"]).to_csv(
            out_dir / "survival_horizon_brier_curve_by_fold.csv",
            index=False,
        )
        pd.DataFrame(columns=["test_month", "model", "bin", "mean_pred", "obs_rate", "n"]).to_csv(
            out_dir / "survival_calibration_curve_by_fold.csv",
            index=False,
        )
        pd.DataFrame(
            columns=[
                "test_month",
                "train_month_start",
                "train_month_end",
                "val_month",
                "n_train_spells",
                "n_val_spells",
                "n_test_spells",
                "n_val_panel",
                "n_test_panel",
                "tau_days",
            ]
        ).to_csv(out_dir / "survival_fold_manifest.csv", index=False)
        spell_sample_cols = [
            "unique_id",
            "spell_idx",
            "spell_start_day",
            "spell_end_day",
            "duration_days",
            "event_observed",
            "prior_events",
            "prior_events_30d",
            "prior_events_90d",
            "prior_device_mobile_share",
            "prior_device_desktop_share",
            "prior_device_tablet_share",
            "last_device_group",
        ]
        panel_sample_cols = [
            "unique_id",
            "as_of_ts",
            "as_of_month",
            "elapsed_days",
            "horizon_days_month_end",
            "prior_events",
            "prior_events_30d",
            "prior_events_90d",
            "prior_device_mobile_share",
            "prior_device_desktop_share",
            "prior_device_tablet_share",
            "last_device_group",
        ]
        sample_limit = int(min(max(cfg.max_panel_rows, 1), 250000))
        spells[[c for c in spell_sample_cols if c in spells.columns]].head(sample_limit).to_csv(
            out_dir / "survival_spells_sample.csv",
            index=False,
        )
        panel[[c for c in panel_sample_cols if c in panel.columns]].head(sample_limit).to_csv(
            out_dir / "survival_panel_sample.csv",
            index=False,
        )
        top_cols = [f"top_{int(round(float(c) * 100))}pct" for c in cfg.capacity_grid]
        pd.DataFrame(
            columns=[
                "panel_row_id",
                "unique_id",
                "as_of_ts",
                "as_of_month",
                "elapsed_days",
                "horizon_days_month_end",
                "prior_events",
                "prior_events_30d",
                "prior_events_90d",
                "test_month",
                "model",
                "score_policy",
                "tau_days",
                "risk_non_return_tau",
                "risk_non_return_month_end",
                "risk_rank",
                "risk_rank_pct",
                "actual_no_return_tau",
                "actual_no_return_month_end",
                "month_end_observable",
                "n_eval_rows_fold",
                "n_eval_positive_tau_fold",
                *top_cols,
            ]
        ).to_csv(out_dir / "survival_user_scores_by_month.csv", index=False)
        pd.DataFrame(
            columns=[
                "panel_row_id",
                "unique_id",
                "as_of_ts",
                "as_of_month",
                "elapsed_days",
                "horizon_days_month_end",
                "prior_events",
                "prior_events_30d",
                "prior_events_90d",
                "test_month",
                "model",
                "score_policy",
                "tau_days",
                "risk_non_return_tau",
                "risk_non_return_month_end",
                "risk_rank",
                "risk_rank_pct",
                "actual_no_return_tau",
                "actual_no_return_month_end",
                "month_end_observable",
                "n_eval_rows_fold",
                "n_eval_positive_tau_fold",
                *top_cols,
            ]
        ).to_csv(out_dir / "survival_user_scores_champion_latest.csv", index=False)
        info = {
            "output_dir": str(out_dir),
            "snapshot_ts": str(snapshot_ts),
            "models_compared": ["rsf", "xgb_aft"],
            "primary_metric": "net_value_at_capacity_with_stability",
            "survival_primary_score": "conditional_non_return_risk=S(e+tau)/S(e)",
            "risk_policy": "conditional_survival_continuous",
            "tau_quantile": float(cfg.tau_quantile),
            "tau_source": "train_uncensored_duration_quantile",
            "granularity": "user_day",
            "ping_filter_applied": True,
            "min_session_seconds": int(cfg.min_session_seconds),
            "champion_model": None,
            "n_folds": 0,
            "capacity_grid": list(cfg.capacity_grid),
            "user_scores_monthly_file": "survival_user_scores_by_month.csv",
            "user_scores_champion_latest_file": "survival_user_scores_champion_latest.csv",
            "champion_latest_month": None,
            "champion_latest_rows": 0,
            "status": "no_valid_folds",
            "selection_policy": (
                "1) net_value_5_10_stable, "
                "2) calibration_and_ibs, "
                "3) recent_temporal_robustness"
            ),
            "entries_rows_joined_dim": int(event_quality.get("entries_rows_joined_dim", 0)),
            "entries_ping_filtered_rows": int(event_quality.get("entries_ping_filtered_rows", 0)),
            "entries_clean_rows": int(event_quality.get("entries_clean_rows", 0)),
            "horizon_brier_curve_file": "survival_horizon_brier_curve_by_fold.csv",
            "calibration_curve_file": "survival_calibration_curve_by_fold.csv",
        }
        (out_dir / "survival_benchmark_summary.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out_dir / "survival_benchmark_details.md").write_text(
            "# Survival Recurrent Benchmark\n\nNo valid folds were produced for the current configuration.",
            encoding="utf-8",
        )
        LOGGER.warning("No valid folds produced survival benchmark outputs; wrote empty artifacts with schema.")
        return info

    fold_df = pd.DataFrame(fold_rows)
    cap_df = pd.DataFrame(capacity_rows)
    manifest_df = pd.DataFrame(manifest_rows)

    summary = (
        fold_df.groupby("model", as_index=False)
        .agg(
            uno_cindex_mean=("uno_cindex", "mean"),
            uno_cindex_std=("uno_cindex", "std"),
            ibs_mean=("ibs", "mean"),
            ibs_std=("ibs", "std"),
            brier_oper_mean=("brier_tau_operational", "mean"),
            brier_oper_std=("brier_tau_operational", "std"),
        )
        .sort_values(["ibs_mean", "uno_cindex_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )

    cap_summary = (
        cap_df.groupby(["model", "capacity"], as_index=False)
        .agg(
            precision_at_k_mean=("precision_at_k", "mean"),
            recall_at_k_mean=("recall_at_k", "mean"),
            net_value_per_1000_mean=("net_value_per_1000", "mean"),
            net_value_per_1000_std=("net_value_per_1000", "std"),
        )
    )

    # Champion rule:
    # 1) best stable net value at 5% and 10%
    # 2) better calibration / IBS
    # 3) better temporal robustness on recent folds
    cap_pivot_mean = cap_summary.pivot(index="model", columns="capacity", values="net_value_per_1000_mean")
    cap_pivot_std = cap_summary.pivot(index="model", columns="capacity", values="net_value_per_1000_std")
    required_caps = [float(c) for c in cfg.capacity_grid]
    model_ranking_rows: List[Dict[str, Any]] = []
    for model in sorted(summary["model"].unique()):
        means = []
        stds = []
        for cap in required_caps:
            means.append(float(cap_pivot_mean.loc[model, cap]) if (model in cap_pivot_mean.index and cap in cap_pivot_mean.columns) else -np.inf)
            stds.append(float(cap_pivot_std.loc[model, cap]) if (model in cap_pivot_std.index and cap in cap_pivot_std.columns) else np.inf)
        primary_net_score = float(min(means)) if means else -np.inf
        stability_penalty = float(np.nansum(stds)) if stds else np.inf

        calib_gap = fold_df[fold_df["model"] == model]
        calib_gap_mean = float(np.mean(np.abs(calib_gap["calibration_mean_pred"] - calib_gap["calibration_obs_rate"])))

        recent_months = sorted(pd.to_datetime(cap_df["test_month"], errors="coerce").dropna().unique())
        recent_months = list(recent_months[-3:]) if recent_months else []
        if recent_months:
            recent_mask = pd.to_datetime(cap_df["test_month"], errors="coerce").isin(recent_months)
            recent_net = float(cap_df[(cap_df["model"] == model) & recent_mask]["net_value_per_1000"].mean())
        else:
            recent_net = -np.inf

        model_ranking_rows.append(
            {
                "model": model,
                "primary_net_score": primary_net_score,
                "stability_penalty": stability_penalty,
                "calibration_gap_mean": calib_gap_mean,
                "recent_net_value_mean": recent_net,
            }
        )

    ranking_df = pd.DataFrame(model_ranking_rows).merge(summary, on="model", how="left")
    ranking_df = ranking_df.sort_values(
        [
            "primary_net_score",
            "stability_penalty",
            "ibs_mean",
            "calibration_gap_mean",
            "recent_net_value_mean",
            "uno_cindex_mean",
        ],
        ascending=[False, True, True, True, False, False],
    ).reset_index(drop=True)
    champion = str(ranking_df.iloc[0]["model"])

    fold_df.to_csv(out_dir / "survival_model_metrics_folds.csv", index=False)
    summary.to_csv(out_dir / "survival_model_metrics_summary.csv", index=False)
    cap_df.to_csv(out_dir / "survival_operational_capacity_by_month.csv", index=False)
    cap_summary.to_csv(out_dir / "survival_operational_capacity_summary.csv", index=False)
    pd.DataFrame(
        brier_curve_rows,
        columns=["test_month", "model", "horizon_days", "brier_score"],
    ).to_csv(out_dir / "survival_horizon_brier_curve_by_fold.csv", index=False)
    pd.DataFrame(
        calibration_rows,
        columns=["test_month", "model", "bin", "mean_pred", "obs_rate", "n"],
    ).to_csv(out_dir / "survival_calibration_curve_by_fold.csv", index=False)
    manifest_df.to_csv(out_dir / "survival_fold_manifest.csv", index=False)

    spell_sample_cols = [
        "unique_id",
        "spell_idx",
        "spell_start_day",
        "spell_end_day",
        "duration_days",
        "event_observed",
        "prior_events",
        "prior_events_30d",
        "prior_events_90d",
        "prior_device_mobile_share",
        "prior_device_desktop_share",
        "prior_device_tablet_share",
        "last_device_group",
    ]
    panel_sample_cols = [
        "unique_id",
        "as_of_ts",
        "as_of_month",
        "elapsed_days",
        "horizon_days_month_end",
        "prior_events",
        "prior_events_30d",
        "prior_events_90d",
        "prior_device_mobile_share",
        "prior_device_desktop_share",
        "prior_device_tablet_share",
        "last_device_group",
    ]
    sample_limit = int(min(max(cfg.max_panel_rows, 1), 250000))
    spells[[c for c in spell_sample_cols if c in spells.columns]].head(sample_limit).to_csv(
        out_dir / "survival_spells_sample.csv",
        index=False,
    )

    panel[[c for c in panel_sample_cols if c in panel.columns]].head(sample_limit).to_csv(
        out_dir / "survival_panel_sample.csv",
        index=False,
    )

    top_cols = [f"top_{int(round(float(c) * 100))}pct" for c in cfg.capacity_grid]
    if user_score_frames:
        user_scores_df = (
            pd.concat(user_score_frames, ignore_index=True)
            .sort_values(["test_month", "model", "risk_rank", "unique_id"])
            .reset_index(drop=True)
        )
    else:
        user_scores_df = pd.DataFrame(
            columns=[
                "panel_row_id",
                "unique_id",
                "as_of_ts",
                "as_of_month",
                "elapsed_days",
                "horizon_days_month_end",
                "prior_events",
                "prior_events_30d",
                "prior_events_90d",
                "test_month",
                "model",
                "score_policy",
                "tau_days",
                "risk_non_return_tau",
                "risk_non_return_month_end",
                "risk_rank",
                "risk_rank_pct",
                "actual_no_return_tau",
                "actual_no_return_month_end",
                "month_end_observable",
                "n_eval_rows_fold",
                "n_eval_positive_tau_fold",
                *top_cols,
            ]
        )
    user_scores_df.to_csv(out_dir / "survival_user_scores_by_month.csv", index=False)

    champion_latest_month = None
    if not user_scores_df.empty:
        champion_scores = user_scores_df[user_scores_df["model"] == champion].copy()
        if not champion_scores.empty:
            champion_latest_month = str(champion_scores["test_month"].max())
            champion_latest = (
                champion_scores[champion_scores["test_month"] == champion_latest_month]
                .sort_values(["risk_rank", "unique_id"])
                .reset_index(drop=True)
            )
        else:
            champion_latest = user_scores_df.head(0).copy()
    else:
        champion_latest = user_scores_df.copy()
    champion_latest.to_csv(out_dir / "survival_user_scores_champion_latest.csv", index=False)

    info = {
        "output_dir": str(out_dir),
        "snapshot_ts": str(snapshot_ts),
        "models_compared": ["rsf", "xgb_aft"],
        "primary_metric": "net_value_at_capacity_with_stability",
        "survival_primary_score": "conditional_non_return_risk=S(e+tau)/S(e)",
        "risk_policy": "conditional_survival_continuous",
        "tau_quantile": float(cfg.tau_quantile),
        "tau_source": "train_uncensored_duration_quantile",
        "granularity": "user_day",
        "ping_filter_applied": True,
        "min_session_seconds": int(cfg.min_session_seconds),
        "champion_model": champion,
        "n_folds": int(manifest_df.shape[0]),
        "capacity_grid": list(cfg.capacity_grid),
        "user_scores_monthly_file": "survival_user_scores_by_month.csv",
        "user_scores_champion_latest_file": "survival_user_scores_champion_latest.csv",
        "champion_latest_month": champion_latest_month,
        "champion_latest_rows": int(champion_latest.shape[0]),
        "selection_policy": (
            "1) net_value_5_10_stable, "
            "2) calibration_and_ibs, "
            "3) recent_temporal_robustness"
        ),
        "entries_rows_joined_dim": int(event_quality.get("entries_rows_joined_dim", 0)),
        "entries_ping_filtered_rows": int(event_quality.get("entries_ping_filtered_rows", 0)),
        "entries_clean_rows": int(event_quality.get("entries_clean_rows", 0)),
        "horizon_brier_curve_file": "survival_horizon_brier_curve_by_fold.csv",
        "calibration_curve_file": "survival_calibration_curve_by_fold.csv",
    }
    (out_dir / "survival_benchmark_summary.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    md = [
        "# Survival Recurrent Benchmark",
        "",
        f"- Snapshot: `{snapshot_ts}`",
        f"- Models: `{', '.join(info['models_compared'])}`",
        f"- Champion: `{champion}`",
        f"- Primary score policy: `{info['survival_primary_score']}`",
        f"- Tau quantile: `{cfg.tau_quantile}`",
        f"- User scores (monthly): `survival_user_scores_by_month.csv`",
        f"- User scores (champion latest): `survival_user_scores_champion_latest.csv`",
        f"- Brier curve (by fold): `{info['horizon_brier_curve_file']}`",
        f"- Calibration curve (by fold): `{info['calibration_curve_file']}`",
        "",
        "## Notes",
        "- Sem janelas arbitrárias de ativo (7/14/30).",
        "- População de risco por elegibilidade observada no as_of (counting-process).",
        "- Sessões ping (<=5s) removidas do stream de survival.",
        f"- Entries filtrados como ping: {int(event_quality.get('entries_ping_filtered_rows', 0))}.",
    ]
    (out_dir / "survival_benchmark_details.md").write_text("\n".join(md), encoding="utf-8")

    return info


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = build_config(args)
    LOGGER.info("Running survival recurrent benchmark | data_dir=%s | output_dir=%s", cfg.data_dir, cfg.output_dir)
    info = run_benchmark(cfg)
    LOGGER.info("Survival benchmark finished | champion=%s", info["champion_model"])


if __name__ == "__main__":
    main()
