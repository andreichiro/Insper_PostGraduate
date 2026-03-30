#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bisnode Firms (Roofing subset) — Reproducible, skeptical EDA pipeline.

Goals
-----
1) Enforce & validate schema: types, ranges, coherence across dates and periods.
2) Produce ≥10 diagnostics per column (skim).
3) Detect missingness, inconsistencies, outliers, asymmetries; quantify proportions.
4) Business-specific audits:
   - Negative sales and interpretation flags
   - Spells of zero sales; recoveries and failures
   - Balance sheet period logic (begin < end; length; partial-year flag)
   - Accounting identities plausibility checks
   - Revenue thresholds & proportions (1k, 10k, 100k, 1M, 10M EUR)
   - Employment dynamics (decreasing, zero staff)
   - Profit/asset/wage coherency; per-employee ratios
   - Exit/founded date sanity; founded after exit?
   - NACE translation; geography distributions
   - Feature leakage risk for prediction
5) Persist tabular artifacts for downstream modeling QA.

Notes
-----
- Designed to be robust to missing columns. All checks are conditional.
- If you have a 4-digit NACE mapping file, pass via --nace4-csv and we will enrich nace_main_name.
"""

from __future__ import annotations
import argparse
import math
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype, is_datetime64_any_dtype, is_integer_dtype

from scipy import stats

import matplotlib.pyplot as plt

# ------------------------- CONFIGURATION -------------------------

REVENUE_THRESHOLDS = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]
Z_OUTLIER = 3.0  # absolute z-score threshold
IQR_FACTOR = 1.5
EXPECTED_YEAR_MIN = 2005
EXPECTED_YEAR_MAX = 2016

# Columns expected & their preferred dtypes
# (we will attempt to coerce and report divergences)
PREFERRED_DTYPES = {
    # IDs & periods
    "comp_id": "Int64",
    "year": "Int64",
    "founded_year": "Int64",
    "exit_year": "Int64",
    "founded_date": "datetime64[ns]",
    "exit_date": "datetime64[ns]",
    "begin": "datetime64[ns]",
    "end": "datetime64[ns]",
    "balsheet_length": "Int64",
    "balsheet_flag": "Int64",        # binary (0/1) if present
    "balsheet_notfullyear": "Int64", # binary (0/1) if present

    # Financials
    "sales": "float",
    "profit_loss_year": "float",
    "inc_bef_tax": "float",
    "COGS": "float",
    "material_exp": "float",
    "personnel_exp": "float",
    "wages": "float",
    "extra_exp": "float",
    "extra_inc": "float",
    "extra_profit_loss": "float",
    "amort": "float",
    "curr_assets": "float",
    "fixed_assets": "float",
    "intang_assets": "float",
    "tang_assets": "float",
    "liq_assets": "float",
    "inventories": "float",
    "curr_liab": "float",
    "share_eq": "float",
    "subscribed_cap": "float",
    "finished_prod": "float",
    "net_dom_sales": "float",
    "net_exp_sales": "float",

    # Workforce & management
    "labor_avg": "float",
    "ceo_count": "float",
    "foreign": "float",  # share in [0,1]
    "female": "float",   # share in [0,1]
    "birth_year": "Int64",
    "inoffice_days": "float",

    # Categoricals
    "gender": "string",     # {'female','male','mix'} (firm-level leadership composition)
    "origin": "string",     # {'domestic','foreign','mix'}
    "nace_main": "string",  # 4-digit NACE Rev.2
    "ind2": "string",       # 2-digit NACE
    "ind": "string",        # broad industry code
    "urban_m": "string",    # {'1','2','3'} or integers; we will coerce to string labels
    "region_m": "string",   # {'Central','West','East'}
    "D": "Int64",           # binary flag column; ensure not nullable boolean
}

# Domain rules: columns that should be nonnegative
NONNEGATIVE_COLS = {
    "sales", "COGS", "material_exp", "personnel_exp", "wages", "extra_exp",
    "extra_inc", "amort", "curr_assets", "fixed_assets", "intang_assets",
    "tang_assets", "liq_assets", "inventories", "curr_liab", "subscribed_cap",
    "finished_prod", "ceo_count", "labor_avg"
}
# Can be negative by nature:
ALLOW_NEGATIVE = {"profit_loss_year", "inc_bef_tax", "extra_profit_loss", "share_eq"}

# NACE 2-digit mapping (Rev.2) — trimmed to the real 2-digit range 01..99.
# We cover the full 2-digit namespace commonly seen. Extend if your data uses others.
NACE2_NAME = {
    # A: Agriculture, forestry and fishing
    "01": "Crop and animal production, hunting and related service activities",
    "02": "Forestry and logging",
    "03": "Fishing and aquaculture",
    # B: Mining and quarrying
    "05": "Mining of coal and lignite",
    "06": "Extraction of crude petroleum and natural gas",
    "07": "Mining of metal ores",
    "08": "Other mining and quarrying",
    "09": "Mining support service activities",
    # C: Manufacturing
    "10": "Manufacture of food products",
    "11": "Manufacture of beverages",
    "12": "Manufacture of tobacco products",
    "13": "Manufacture of textiles",
    "14": "Manufacture of wearing apparel",
    "15": "Manufacture of leather and related products",
    "16": "Manufacture of wood and of products of wood and cork, except furniture; manufacture of articles of straw and plaiting materials",
    "17": "Manufacture of paper and paper products",
    "18": "Printing and reproduction of recorded media",
    "19": "Manufacture of coke and refined petroleum products",
    "20": "Manufacture of chemicals and chemical products",
    "21": "Manufacture of basic pharmaceutical products and pharmaceutical preparations",
    "22": "Manufacture of rubber and plastic products",
    "23": "Manufacture of other non-metallic mineral products",
    "24": "Manufacture of basic metals",
    "25": "Manufacture of fabricated metal products, except machinery and equipment",
    "26": "Manufacture of computer, electronic and optical products",
    "27": "Manufacture of electrical equipment",
    "28": "Manufacture of machinery and equipment n.e.c.",
    "29": "Manufacture of motor vehicles, trailers and semi-trailers",
    "30": "Manufacture of other transport equipment",
    "31": "Manufacture of furniture",
    "32": "Other manufacturing",
    "33": "Repair and installation of machinery and equipment",
    # D: Electricity, gas, steam and air conditioning supply
    "35": "Electricity, gas, steam and air conditioning supply",
    # E: Water supply; sewerage; waste management and remediation activities
    "36": "Water collection, treatment and supply",
    "37": "Sewerage",
    "38": "Waste collection, treatment and disposal activities; materials recovery",
    "39": "Remediation activities and other waste management services",
    # F: Construction
    "41": "Construction of buildings",
    "42": "Civil engineering",
    "43": "Specialised construction activities",
    # G: Wholesale and retail trade; repair of motor vehicles and motorcycles
    "45": "Wholesale and retail trade and repair of motor vehicles and motorcycles",
    "46": "Wholesale trade, except of motor vehicles and motorcycles",
    "47": "Retail trade, except of motor vehicles and motorcycles",
    # H: Transportation and storage
    "49": "Land transport and transport via pipelines",
    "50": "Water transport",
    "51": "Air transport",
    "52": "Warehousing and support activities for transportation",
    "53": "Postal and courier activities",
    # I: Accommodation and food service activities
    "55": "Accommodation",
    "56": "Food and beverage service activities",
    # J: Information and communication
    "58": "Publishing activities",
    "59": "Motion picture, video and television programme production, sound recording and music publishing activities",
    "60": "Programming and broadcasting activities",
    "61": "Telecommunications",
    "62": "Computer programming, consultancy and related activities",
    "63": "Information service activities",
    # K: Financial and insurance activities
    "64": "Financial service activities, except insurance and pension funding",
    "65": "Insurance, reinsurance and pension funding, except compulsory social security",
    "66": "Activities auxiliary to financial services and insurance activities",
    # L: Real estate activities
    "68": "Real estate activities",
    # M: Professional, scientific and technical activities
    "69": "Legal and accounting activities",
    "70": "Activities of head offices; management consultancy activities",
    "71": "Architectural and engineering activities; technical testing and analysis",
    "72": "Scientific research and development",
    "73": "Advertising and market research",
    "74": "Other professional, scientific and technical activities",
    "75": "Veterinary activities",
    # N: Administrative and support service activities
    "77": "Rental and leasing activities",
    "78": "Employment activities",
    "79": "Travel agency, tour operator and other reservation service and related activities",
    "80": "Security and investigation activities",
    "81": "Services to buildings and landscape activities",
    "82": "Office administrative, office support and other business support activities",
    # O: Public administration and defence; compulsory social security
    "84": "Public administration and defence; compulsory social security",
    # P: Education
    "85": "Education",
    # Q: Human health and social work activities
    "86": "Human health activities",
    "87": "Residential care activities",
    "88": "Social work activities without accommodation",
    # R: Arts, entertainment and recreation
    "90": "Creative, arts and entertainment activities",
    "91": "Libraries, archives, museums and other cultural activities",
    "92": "Gambling and betting activities",
    "93": "Sports activities and amusement and recreation activities",
    # S: Other service activities
    "94": "Activities of membership organisations",
    "95": "Repair of computers and personal and household goods",
    "96": "Other personal service activities",
    # T: Activities of households as employers; etc.
    "97": "Activities of households as employers of domestic personnel",
    "98": "Undifferentiated goods- and services-producing activities of private households for own use",
    # U: Activities of extraterritorial organisations and bodies
    "99": "Activities of extraterritorial organisations and bodies",
}

# ------------------------- UTILITIES -------------------------

def script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def to_int(series: pd.Series, tol: float = 1e-6) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    whole = s.round()
    s = s.where(s.isna() | (np.abs(s - whole) <= tol), np.nan)
    return whole.astype("Int64")

def to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)

def clip01(series: pd.Series) -> pd.Series:
    return series.where(series.between(0, 1), np.nan)

def coerce_dtype(df: pd.DataFrame, col: str, target: str) -> pd.DataFrame:
    if col not in df.columns:
        return df
    try:
        if target in ("float", "Float64"):
            df[col] = to_numeric(df[col]).astype("Float64")
        elif target in ("int", "Int64"):
            df[col] = to_int(df[col])
        elif target.startswith("datetime64"):
            df[col] = to_datetime(df[col])
        elif target == "string":
            df[col] = df[col].astype("string")
        else:
            # fallback
            df[col] = df[col].astype(target)
    except Exception as e:
        warnings.warn(f"[TYPE] Could not coerce {col} to {target}: {e}")
    return df

def add_ind2_name(df: pd.DataFrame) -> pd.DataFrame:
    if "ind2" not in df.columns:
        return df
    # Normalize to 2-digit string
    s = df["ind2"].astype("string").str.strip()
    s2 = s.str.extract(r"(\d{2})", expand=False)
    df["ind2_2d"] = s2
    df["ind2_name"] = df["ind2_2d"].map(NACE2_NAME)
    return df

def add_nace_main_2d(df: pd.DataFrame) -> pd.DataFrame:
    # derive 2-digit from 4-digit NACE when available
    if "nace_main" in df.columns:
        s = df["nace_main"].astype("string").str.strip()
        s2 = s.str.extract(r"(\d{2})", expand=False)
        df["nace_main_2d"] = s2
        df["nace_main_2d_name"] = df["nace_main_2d"].map(NACE2_NAME)
    return df

def _bowley_skew(q1: float, med: float, q3: float) -> float:
    """Quartile (Bowley) skewness; robust to heavy tails. Returns np.nan if IQR==0 or NaN inputs."""
    if any(map(lambda v: v is None or pd.isna(v), (q1, med, q3))):
        return np.nan
    iqr = q3 - q1
    return np.nan if iqr == 0 else float(((q3 + q1) - 2.0 * med) / iqr)


def asymmetry_audit(df: pd.DataFrame,
                    skew_threshold: float = 1.0,
                    bowley_threshold: float = 0.30,
                    min_non_null: int = 30) -> pd.DataFrame:
    """
    Req. 1 — For each numeric (non-boolean) column, compute classical skewness (Fisher-Pearson g1),
    robust quartile skewness (Bowley), and log-skewness (if strictly positive).
    """
    rows = []
    for col in df.columns:
        s = df[col]

        # Only real numeric columns; treat booleans as categorical and SKIP them here
        if (not is_numeric_dtype(s)) or is_bool_dtype(s):
            continue

        # Coerce to float explicitly to avoid numpy boolean/int corner cases in SciPy
        x = pd.to_numeric(s, errors="coerce").astype("Float64").dropna()
        nn = int(x.size)

        if nn < min_non_null:
            rows.append({
                "column": col, "non_null": nn,
                "skew": np.nan, "bowley_skew": np.nan, "log_skew": np.nan,
                "is_asymmetric": False, "reason": "insufficient_non_null"
            })
            continue

        # classical skew
        x_arr = x.to_numpy(dtype=float)
        sk = float(stats.skew(x_arr, nan_policy="omit"))

        # robust skew via quartiles
        q1, med, q3 = np.percentile(x_arr, [25, 50, 75])
        bsk = _bowley_skew(q1, med, q3)

        # log skew where strictly positive support
        x_pos = x[x > 0]
        log_sk = float(stats.skew(np.log(x_pos.to_numpy(dtype=float)), nan_policy="omit")) \
                 if x_pos.size >= min_non_null else np.nan

        # decision
        reasons = []
        if np.isfinite(sk) and abs(sk) >= skew_threshold: reasons.append(f"|skew|≥{skew_threshold}")
        if (not np.isnan(bsk)) and abs(bsk) >= bowley_threshold: reasons.append(f"|bowley|≥{bowley_threshold}")
        if np.isfinite(log_sk) and abs(log_sk) >= skew_threshold: reasons.append(f"|log_skew|≥{skew_threshold}")

        rows.append({
            "column": col,
            "non_null": nn,
            "skew": sk, "bowley_skew": bsk, "log_skew": log_sk,
            "is_asymmetric": bool(len(reasons) > 0),
            "reason": ";".join(reasons) if reasons else ""
        })

    out = pd.DataFrame(rows).sort_values(["is_asymmetric", "column"], ascending=[False, True]).reset_index(drop=True)
    return out

def inconsistency_scan_all_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Req. 2 — Column-wise inconsistency scan for ALL columns.
    Includes:
      - Nonnegativity violations (NONNEGATIVE_COLS plus auto-detected names)
      - Binary columns not in {0,1}
      - Shares/probabilities outside [0,1] (known share cols + '*_share', '*_pct')
      - Extreme common-size ratios for '*_bs' and '*_pl' (abs(value) > 5)
      - Year plausibility is already handled elsewhere (not repeated here)
    Emits a compact per-column table with counts/percents by issue.
    """
    # auto-detect helpers
    def _is_binary_like(name: str, s: pd.Series) -> bool:
        if is_bool_dtype(s): return True
        # int-like low-card numeric or columns clearly flagged by name
        if is_numeric_dtype(s) and s.dropna().nunique() <= 3:
            return True
        lname = name.lower()
        return any(k in lname for k in ("flag", "indicator", "problem", "issue")) or name in {"D"}

    def _is_share_like(name: str) -> bool:
        lname = name.lower()
        return (name in {"foreign","female"}) or ("share" in lname or "pct" in lname or "percent" in lname)

    def _auto_nonneg(name: str) -> bool:
        lname = name.lower()
        # do not auto-apply to explicitly allowed negatives
        if name in ALLOW_NEGATIVE: return False
        # heuristics for accounting magnitudes that are >= 0
        return any(k in lname for k in [
            "asset","liab","sales","revenue","exp","wage","payroll","inventory",
            "employees","labor","count","amount","cash","debt","cap"
        ]) or (name in NONNEGATIVE_COLS)

    rows = []
    nrows = len(df)
    for col in df.columns:
        s = df[col]
        nn = int(s.notna().sum())

        rec = {"column": col, "dtype": str(s.dtype), "non_null": nn, "non_null_pct": (100.0*nn/nrows if nrows else np.nan)}

        if is_numeric_dtype(s) or is_bool_dtype(s):
            x = to_numeric(s)

            # nonnegativity
            if _auto_nonneg(col):
                neg = int((x < 0).sum(skipna=True))
                rec["neg_count"] = neg
                rec["neg_pct_non_null"] = (100.0*neg/nn if nn else np.nan)

            # binary domain
            if _is_binary_like(col, s):
                bad = int(((x.notna()) & (~x.isin([0,1]))).sum())
                rec["binary_outside_01_count"] = bad
                rec["binary_outside_01_pct_non_null"] = (100.0*bad/nn if nn else np.nan)

            # shares in [0,1]
            if _is_share_like(col):
                out01 = int(((x.notna()) & (~x.between(0,1))).sum())
                rec["share_outside_[0,1]_count"] = out01
                rec["share_outside_[0,1]_pct_non_null"] = (100.0*out01/nn if nn else np.nan)

            # common-size ratios: *_bs (balance-sheet), *_pl (P&L / sales)
            lname = col.lower()
            if lname.endswith("_bs") or lname.endswith("_pl") or "ratio" in lname or "rate" in lname:
                extreme = int(((x.notna()) & (x.abs() > 5)).sum())
                rec["extreme_ratio_|val|>5_count"] = extreme
                rec["extreme_ratio_|val|>5_pct_non_null"] = (100.0*extreme/nn if nn else np.nan)

            # infinities (should not exist)
            infc = int(np.isinf(x).sum()) if nn else 0
            rec["infinite_count"] = infc
            rec["infinite_pct_non_null"] = (100.0*infc/nn if nn else np.nan)

        else:
            # Categorical-like: encode parsing failures? not needed; use rare/class checks elsewhere
            pass

        rows.append(rec)

    out = pd.DataFrame(rows)
    # keep only useful columns (drop all-NaN)
    keep = [c for c in out.columns if c in {"column","dtype","non_null","non_null_pct"} or not out[c].isna().all()]
    return out[keep].sort_values(["column"]).reset_index(drop=True)


def revenue_bracket_flags(df: pd.DataFrame, low_eur: float, high_eur: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Req. 3 — Row- and firm-level revenue bracket flags.
    Row-level columns: revenue_lt_low, revenue_gt_high (with year & comp_id for traceability)
    Firm-level: one row per comp_id if condition holds in ANY row, with counts & first/last years.
    """
    if "sales" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    s = to_numeric(df["sales"])
    rows = pd.DataFrame({
        "comp_id": df.get("comp_id"),
        "year": to_int(df.get("year")),
        "revenue_lt_low": (s.notna() & (s < low_eur)),
        "revenue_gt_high": (s.notna() & (s > high_eur)),
    }).copy()

    # Firm-level collapse
    f = []
    for which, col in [("lt", "revenue_lt_low"), ("gt", "revenue_gt_high")]:
        tmp = rows[rows[col]].copy()
        if tmp.empty: 
            continue
        grp = tmp.groupby("comp_id").agg(
            flag_count=(col, "sum"),
            first_year=("year", "min"),
            last_year=("year", "max"),
        ).reset_index()
        grp["which"] = which
        f.append(grp)
    firms = pd.concat(f, ignore_index=True) if f else pd.DataFrame()
    return rows, firms


def few_observations_and_rare_values(df: pd.DataFrame,
                                     min_column_nonnull: int = 100,
                                     min_column_nonnull_pct: float = 1.0,
                                     rare_value_min_count: int = 20,
                                     rare_value_min_pct: float = 0.5,
                                     max_levels_for_numeric_as_categorical: int = 30
                                     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Req. 4 —
      A) Columns with 'few observations': non-null coverage below absolute or percentage threshold.
      B) Rare values per column (categorical or integer-like low-cardinality numerics):
         values with count <= rare_value_min_count OR pct <= rare_value_min_pct.
    Returns:
      - low_coverage (column-level)
      - rare_values_long (value-level; flagged only)
    """
    n = len(df)
    cov_rows = []
    rare_rows = []

    # helper to decide if numeric column should be treated as categorical
    def _numeric_is_categorical(s: pd.Series) -> bool:
        if not is_numeric_dtype(s): return False
        x = to_numeric(s).dropna()
        if x.empty: return False
        # integer-like?
        is_int_like = bool(np.allclose(x, np.round(x)))
        uniq = int(x.nunique())
        return is_int_like and (uniq <= max_levels_for_numeric_as_categorical)

    for col in df.columns:
        s = df[col]
        nn = int(s.notna().sum())
        nn_pct = 100.0 * nn / n if n else np.nan

        # A) low coverage columns
        low = (nn < min_column_nonnull) or (not np.isnan(nn_pct) and nn_pct < min_column_nonnull_pct)
        if low:
            cov_rows.append({"column": col, "dtype": str(s.dtype), "non_null": nn, "non_null_pct": nn_pct})

        # B) rare values per column: categorical or int-like low-card numeric
        is_cat_candidate = (str(s.dtype) in {"object", "string"}) or is_bool_dtype(s) or _numeric_is_categorical(s)
        if is_cat_candidate:
            vc = s.value_counts(dropna=True)
            if not vc.empty:
                total = int(vc.sum())
                for v, c in vc.items():
                    pct = 100.0 * c / total if total else np.nan
                    if (c <= rare_value_min_count) or (not np.isnan(pct) and pct <= rare_value_min_pct):
                        rare_rows.append({
                            "column": col,
                            "value": v,
                            "count": int(c),
                            "pct_within_column": pct
                        })

    low_cov = pd.DataFrame(cov_rows).sort_values(["non_null_pct","non_null"]).reset_index(drop=True) if cov_rows else pd.DataFrame()
    rare_vals = pd.DataFrame(rare_rows).sort_values(["column","pct_within_column"]).reset_index(drop=True) if rare_rows else pd.DataFrame()
    return low_cov, rare_vals


def class_imbalance_all_columns(df: pd.DataFrame,
                                majority_threshold_pct: float = 90.0,
                                max_levels: int = 30,
                                treat_integer_as_categorical_if_unique_le: int = 30) -> pd.DataFrame:
    """
    Req. 5 — For every categorical/boolean/low-cardinality integer column, compute class distribution
    and flag if the largest class exceeds majority_threshold_pct of non-null values.

    Returns rows:
      column, dtype, non_null, unique, top1_value, top1_count, top1_pct, is_imbalanced
    """
    rows = []
    for col in df.columns:
        s = df[col]

        # candidate columns
        is_cat = (str(s.dtype) in {"string", "object"} or is_bool_dtype(s))
        is_int_low_card = (is_integer_dtype(s) and s.dropna().nunique() <= treat_integer_as_categorical_if_unique_le)
        if not (is_cat or is_int_low_card):
            continue

        x = s.dropna()
        nn = int(x.size)
        if nn == 0:
            continue

        vc = x.value_counts()
        uniq = int(vc.size)
        if uniq == 0 or uniq > max_levels:
            continue

        top1_val = vc.index[0]
        top1_cnt = int(vc.iloc[0])
        top1_pct = 100.0 * top1_cnt / nn

        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "non_null": nn,
            "unique": uniq,
            "top1_value": top1_val,
            "top1_count": top1_cnt,
            "top1_pct": top1_pct,
            "is_imbalanced": bool(top1_pct >= majority_threshold_pct)
        })
    return pd.DataFrame(rows).sort_values(["is_imbalanced","top1_pct"], ascending=[False,False]).reset_index(drop=True)

# ------------------------- SCHEMA & TYPE ENFORCEMENT -------------------------

def enforce_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Coerce columns to preferred types where present. Return (df, coercion_report)."""
    coercion_logs = []
    for col, target in PREFERRED_DTYPES.items():
        if col in df.columns:
            before = str(df[col].dtype)
            df = coerce_dtype(df, col, target)
            after = str(df[col].dtype)
            if before != after:
                coercion_logs.append((col, before, after, "OK"))
    # Shares clamped to [0,1]
    for share_col in ("foreign", "female"):
        if share_col in df.columns:
            df[share_col] = clip01(to_numeric(df[share_col]).astype("Float64"))
    # Urban mapping canonicalization (keep original if already labels)
    if "urban_m" in df.columns:
        # Map {1,2,3} -> labels while preserving existing labels if non-numeric
        raw = df["urban_m"].astype("string").str.strip()
        numeric_mask = raw.str.fullmatch(r"\d+")
        mapped = raw.copy()
        mapped.loc[numeric_mask & (raw == "1")] = "capital"
        mapped.loc[numeric_mask & (raw == "2")] = "big_city"
        mapped.loc[numeric_mask & (raw == "3")] = "other"
        df["urban_m_std"] = mapped

    # Year handling: coerce to Int64 here; range validation is handled in flags inventory
    if "year" in df.columns:
        y = to_int(df["year"])
        df["year"] = y

    report = pd.DataFrame(coercion_logs, columns=["column", "dtype_before", "dtype_after", "status"])

    return df, report

def _top_barh(df: pd.DataFrame, xcol: str, ycol: str, n: int, title: str, xlabel: str, out_png: str) -> None:
    plt.figure(figsize=(10, max(4, 0.35 * n)))
    top = df.sort_values(xcol, ascending=False).head(n)
    if top.empty:
        plt.text(0.5, 0.5, "No data to plot", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return
    plt.barh(top[ycol].astype(str), top[xcol])
    plt.xlabel(xlabel); plt.title(title)
    plt.gca().invert_yaxis()
    _safe_savefig(out_png)

def plot_asymmetry_audit(asym_df: pd.DataFrame, out_png: str, topn: int = 20) -> None:
    if asym_df is None or asym_df.empty:
        plt.figure(); plt.text(0.5,0.5,"No numeric columns for asymmetry",ha="center",va="center"); plt.axis("off")
        _safe_savefig(out_png); return
    df = asym_df.copy()
    df["abs_skew"] = df["skew"].abs()
    df["abs_bowley"] = df["bowley_skew"].abs()
    df["abs_log"] = df["log_skew"].abs()
    df["max_abs"] = df[["abs_skew","abs_bowley","abs_log"]].max(axis=1, skipna=True)
    keep = df[df["max_abs"].notna()].copy()
    _top_barh(
        keep.assign(column=keep["column"].astype(str)),
        xcol="max_abs", ycol="column", n=topn,
        title="Top asymmetric columns (max of |skew|, |bowley|, |log_skew|)",
        xlabel="max abs skewness", out_png=out_png
    )

def plot_flags_inventory(flags_df: pd.DataFrame, out_png: str, title: str, topn: int = 30) -> None:
    plt.figure(figsize=(10, max(4, 0.35 * min(topn, len(flags_df) if flags_df is not None else 0))))
    if flags_df is None or flags_df.empty:
        plt.text(0.5, 0.5, "No flags to display", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return
    df = flags_df.sort_values("flagged_pct", ascending=False).head(topn)
    if df.empty:
        plt.text(0.5, 0.5, "No flags to display", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return
    plt.barh(df["flag"].astype(str), df["flagged_pct"])
    plt.xlabel("Flagged rows (%)")
    plt.title(title)
    plt.gca().invert_yaxis()
    _safe_savefig(out_png)

def plot_inconsistency_panels(inc_df: pd.DataFrame, figs_dir: str) -> list:
    paths = []
    if inc_df is None or inc_df.empty:
        return paths

    def _panel(df_sub: pd.DataFrame, title: str, xlabel: str, out_path: str, topn: int = 25) -> None:
        if df_sub is None or df_sub.empty:
            plt.figure(); plt.text(0.5,0.5,"No data to plot",ha="center",va="center"); plt.axis("off"); _safe_savefig(out_path); return
        vals = df_sub["value"].astype(float).fillna(0.0)
        if (vals > 0).any():
            _top_barh(df_sub.assign(column=df_sub["column"].astype(str)), "value", "column", topn, title, xlabel, out_path)
        else:
            plt.figure(); plt.text(0.5,0.5,f"No {title.lower()} detected",ha="center",va="center"); plt.axis("off"); _safe_savefig(out_path)

    # Nonnegativity
    if {"neg_pct_non_null","column"}.issubset(inc_df.columns):
        p = os.path.join(figs_dir, "inconsistency_nonnegative.png")
        dfsub = inc_df[inc_df["neg_pct_non_null"].notna()][["column","neg_pct_non_null"]].rename(
            columns={"neg_pct_non_null":"value"})
        _panel(dfsub, "Nonnegativity violations (share of non‑null)", "violations (% of non‑null)", p)
        paths.append(p)
    # Binary outside {0,1}
    if {"binary_outside_01_pct_non_null","column"}.issubset(inc_df.columns):
        p = os.path.join(figs_dir, "inconsistency_binary.png")
        dfsub = inc_df[inc_df["binary_outside_01_pct_non_null"].notna()][["column","binary_outside_01_pct_non_null"]].rename(
            columns={"binary_outside_01_pct_non_null":"value"})
        _panel(dfsub, "Binary domain issues (outside {0,1})", "issues (% of non‑null)", p)
        paths.append(p)
    # Shares outside [0,1]
    if {"share_outside_[0,1]_pct_non_null","column"}.issubset(inc_df.columns):
        p = os.path.join(figs_dir, "inconsistency_shares.png")
        dfsub = inc_df[inc_df["share_outside_[0,1]_pct_non_null"].notna()][["column","share_outside_[0,1]_pct_non_null"]].rename(
            columns={"share_outside_[0,1]_pct_non_null":"value"})
        _panel(dfsub, "Share variables outside [0,1]", "issues (% of non‑null)", p)
        paths.append(p)
    # Extreme ratios
    if {"extreme_ratio_|val|>5_pct_non_null","column"}.issubset(inc_df.columns):
        p = os.path.join(figs_dir, "inconsistency_extreme_ratios.png")
        dfsub = inc_df[inc_df["extreme_ratio_|val|>5_pct_non_null"].notna()][["column","extreme_ratio_|val|>5_pct_non_null"]].rename(
            columns={"extreme_ratio_|val|>5_pct_non_null":"value"})
        _panel(dfsub, "Extreme ratios (|value| > 5)", "extremes (% of non‑null)", p)
        paths.append(p)
    return paths

def plot_revenue_bracket_timeseries(rb_rows: pd.DataFrame, out_png: str) -> None:
    if rb_rows is None or rb_rows.empty or "year" not in rb_rows.columns:
        plt.figure(); plt.text(0.5,0.5,"No revenue bracket rows",ha="center",va="center"); plt.axis("off"); _safe_savefig(out_png); return
    x = rb_rows.copy()
    x["year"] = to_int(x["year"])
    x = x.dropna(subset=["year"])
    if x.empty:
        plt.figure(); plt.text(0.5,0.5,"No usable years for revenue brackets",ha="center",va="center"); plt.axis("off"); _safe_savefig(out_png); return
    by = x.groupby("year")[["revenue_lt_low","revenue_gt_high"]].sum().reset_index()
    plt.figure(figsize=(10,4.5))
    plt.plot(by["year"], by["revenue_lt_low"], marker="o", label="< €1k")
    plt.plot(by["year"], by["revenue_gt_high"], marker="o", label="> €10m")
    plt.legend()
    plt.xlabel("Year"); plt.ylabel("Row count")
    plt.title("Revenue bracket flags by year")
    _safe_savefig(out_png)

def plot_low_coverage(low_cov: pd.DataFrame, out_png: str, topn: int = 25) -> None:
    if low_cov is None or low_cov.empty:
        plt.figure(); plt.text(0.5,0.5,"No low‑coverage columns",ha="center",va="center"); plt.axis("off"); _safe_savefig(out_png); return
    df = low_cov.copy()
    df["non_null_pct"] = df["non_null_pct"].fillna(0.0).clip(0, 100)
    df["coverage_gap_pct"] = 100.0 - df["non_null_pct"]
    _top_barh(df, xcol="coverage_gap_pct", ycol="column", n=topn,
              title="Columns with lowest coverage", xlabel="coverage gap (% missing)", out_png=out_png)

def plot_rare_values_by_column(rare_vals: pd.DataFrame, out_png: str, topn: int = 25) -> None:
    if rare_vals is None or rare_vals.empty:
        plt.figure(); plt.text(0.5,0.5,"No rare values flagged",ha="center",va="center"); plt.axis("off"); _safe_savefig(out_png); return
    cnt = rare_vals.groupby("column").size().rename("rare_values").reset_index()
    _top_barh(cnt, xcol="rare_values", ycol="column", n=topn,
              title="Columns with most rare values flagged", xlabel="rare values (count)", out_png=out_png)

def plot_class_imbalance(imb: pd.DataFrame, out_png: str, topn: int = 25, threshold: float = 90.0) -> None:
    if imb is None or imb.empty:
        plt.figure(); plt.text(0.5,0.5,"No class imbalance candidates",ha="center",va="center"); plt.axis("off"); _safe_savefig(out_png); return
    df = imb.copy()
    df["top1_pct"] = df["top1_pct"].astype(float)
    plt.figure(figsize=(10, max(4, 0.35 * topn)))
    top = df.sort_values("top1_pct", ascending=False).head(topn)
    plt.barh(top["column"].astype(str), top["top1_pct"])
    plt.xlabel("Top‑class share (% of non‑null)")
    plt.title("Class imbalance — highest top‑class shares")
    plt.axvline(threshold, linestyle="--")
    plt.gca().invert_yaxis()
    _safe_savefig(out_png)

def plot_drops_by_reason(drop_log: pd.DataFrame, out_png: str) -> None:
    plt.figure(figsize=(10, 6))
    if drop_log is None or drop_log.empty:
        plt.text(0.5, 0.5, "No rows dropped", ha="center", va="center", fontsize=14)
        plt.axis("off")
        _safe_savefig(out_png); return
    reasons = (drop_log["reasons"].str.split(";", expand=True)
               .stack().reset_index(drop=True).value_counts())
    if reasons.empty:
        plt.text(0.5, 0.5, "No rows dropped", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return
    reasons.sort_values(ascending=True).plot(kind="barh")
    plt.xlabel("Rows dropped")
    plt.title("What we dropped — and why")
    _safe_savefig(out_png)

def plot_fixes_by_column(fix_summary: pd.DataFrame, out_png: str) -> None:
    plt.figure(figsize=(10, 6))
    if fix_summary is None or fix_summary.empty:
        plt.text(0.5, 0.5, "No in-place value fixes applied", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return
    fs = fix_summary.sort_values("rows_affected", ascending=True)
    plt.barh(fs["column_fixed"].astype(str), fs["rows_affected"])
    plt.xlabel("Rows affected")
    plt.title("What we fixed in place (negative → 0; shares outside [0,1] → NaN)")
    _safe_savefig(out_png)

def plot_type_coercions(coercion_report: pd.DataFrame, out_png: str) -> None:
    plt.figure(figsize=(10, 6))
    if coercion_report is None or coercion_report.empty:
        plt.text(0.5, 0.5, "No dtype coercions were needed", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return
    cr = coercion_report.copy()
    if "status" in cr.columns:
        cr["change"] = cr["dtype_before"] + " → " + cr["dtype_after"] + " (" + cr["status"] + ")"
    else:
        cr["change"] = cr["dtype_before"] + " → " + cr["dtype_after"]
    cr = cr.groupby("change")["column"].count().sort_values(ascending=True)
    cr.plot(kind="barh")
    plt.xlabel("Columns affected")
    plt.title("Type coercions applied")
    _safe_savefig(out_png)

def plot_missingness_top(miss_series: pd.Series, out_png: str, title: str, topn: int = 30) -> None:
    if miss_series is None or miss_series.empty:
        plt.text(0.5, 0.5, "No missingness to display", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return

    m = miss_series.sort_values(ascending=False).head(topn)
    plt.figure(figsize=(10, max(4, 0.25*len(m))))
    m.plot(kind="barh")

    plt.xlabel("Missing (%)")
    plt.title(title)
    _safe_savefig(out_png)

def plot_sales_hist_after(df: pd.DataFrame, out_png: str) -> None:
    plt.figure(figsize=(9, 5))
    if "sales" not in df.columns:
        plt.text(0.5, 0.5, "No 'sales' column", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return
    s = to_numeric(df["sales"])
    sp = s[s > 0]
    if not sp.notna().any():
        plt.text(0.5, 0.5, "No positive sales to plot", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return
    plt.hist(np.log10(sp.dropna()), bins=50)
    plt.xlabel("log10(sales €)")
    plt.ylabel("count")
    neg = int((s < 0).sum()); zero = int((s == 0).sum()); pos = int((s > 0).sum())
    plt.title(f"Sales distribution after cleaning (neg={neg}, zero={zero}, pos={pos})")
    _safe_savefig(out_png)

def plot_residual_issues(flags_df_after: pd.DataFrame,
                         out_png: str,
                         drop_like_flags: Optional[List[str]] = None) -> None:
    """
    Always render a bar chart. If all drop-eligible checks are 0% after cleaning,
    draw zero-height bars instead of writing text.
    """
    # 1) Determine which flags are "drop-eligible"
    if drop_like_flags is None:
        base = {
            "comp_id_missing", "year_missing", "begin_after_or_eq_end",
            "duplicate_comp_id_year", "duplicate_comp_id_year_noncanonical",
            "row_after_exit_year", "founded_after_exit"
        }
        yearouts = {f for f in flags_df_after["flag"].astype(str) if f.startswith("year_out_of_[")}
        drop_like_flags = sorted(base | yearouts)


    # 2) Build the dataframe to plot (include zeros for any missing flags)
    resid = flags_df_after[flags_df_after["flag"].isin(drop_like_flags)][["flag","flagged_pct"]].copy()
    missing = [f for f in drop_like_flags if f not in set(resid["flag"])]
    if missing:
        resid = pd.concat([resid, pd.DataFrame({"flag": missing, "flagged_pct": 0.0})], ignore_index=True)

    resid = resid.sort_values("flagged_pct", ascending=True)

    # 3) Plot as bar chart (always)
    plt.figure(figsize=(10, max(3, 0.35 * max(1, len(resid)))))
    plt.barh(resid["flag"].astype(str), resid["flagged_pct"])
    plt.xlabel("Flagged rows (%) AFTER cleaning")

    # Helpful title + x-limits so value labels are visible even when all zeros
    if float(resid["flagged_pct"].max()) <= 0.0:
        plt.title("Residual issues after cleaning — all drop-eligible checks at 0%")
        plt.xlim(0, 1)  # give a little room for labels on zero bars
    else:
        plt.title("Residual issues after cleaning (should be 0 for drop-eligible rules)")

    # Optional: annotate percentages to the right of each bar
    ax = plt.gca()
    xmax = ax.get_xlim()[1]
    offset = 0.02 * (xmax if xmax > 0 else 1.0)
    for y, v in enumerate(resid["flagged_pct"].tolist()):
        ax.text(v + offset, y, f"{v:.2f}%", va="center", fontsize=8)

    _safe_savefig(out_png)

def plot_nonnegativity_audit(nonneg_df: pd.DataFrame, out_png: str, topn: int = 25) -> None:
    plt.figure(figsize=(10, max(4, 0.35 * topn)))
    if nonneg_df is None or nonneg_df.empty:
        plt.text(0.5, 0.5, "No nonnegativity audit to display", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return

    df = nonneg_df.copy()

    # Recompute share if missing and drop columns with 0 non-null (no information)
    if "neg_pct" not in df.columns and {"negatives", "non_null"}.issubset(df.columns):
        df["neg_pct"] = np.where(df["non_null"] > 0, 100.0 * df["negatives"] / df["non_null"], np.nan)

    # Keep only columns where we have any coverage
    if "non_null" in df.columns:
        df = df[df["non_null"].fillna(0) > 0]

    if df.empty or "neg_pct" not in df.columns:
        plt.text(0.5, 0.5, "No negatives detected (no non‑null values)", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return

    df["neg_pct"] = pd.to_numeric(df["neg_pct"], errors="coerce")
    df = df[df["neg_pct"].notna()].sort_values("neg_pct", ascending=False).head(topn)

    if df.empty:
        plt.text(0.5, 0.5, "No negatives detected", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return

    plt.barh(df["column"].astype(str), df["neg_pct"])
    plt.xlabel("Share of negatives among non‑null (%)")
    topv = float(df["neg_pct"].max()) if not df["neg_pct"].empty else 0.0
    if topv <= 0.0:
        # Make zero-height bars visible and informative
        plt.xlim(0, 1)
        plt.title("Nonnegativity audit — all 0% (no negatives found)")
    else:
        plt.title("Nonnegativity audit — worst offenders")
    plt.gca().invert_yaxis()
    _safe_savefig(out_png)

def plot_sales_negative_by_year(neg_rows: pd.DataFrame, out_png: str) -> None:
    plt.figure(figsize=(9, 4.5))
    if neg_rows is None or neg_rows.empty or "year" not in neg_rows.columns:
        plt.text(0.5, 0.5, "No negative sales rows", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return
    y = to_int(neg_rows["year"]).dropna()
    if y.empty:
        plt.text(0.5, 0.5, "No year info for negative sales", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return
    by = y.value_counts().sort_index()
    plt.plot(by.index.astype(int), by.values, marker="o")
    plt.xlabel("Year"); plt.ylabel("Negative‑sales rows"); plt.title("Negative sales — count by year")
    _safe_savefig(out_png)

def plot_zero_sales_spells_long(zlong: pd.DataFrame, out_png: str) -> None:
    vals = pd.to_numeric(zlong["spell_len"], errors="coerce").dropna()
    if vals.empty:
        plt.text(0.5, 0.5, "No long zero‑sales spells", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return

    plt.hist(vals, bins=range(1, int(vals.max())+2))
    plt.figure(figsize=(9, 4.5))
    if zlong is None or zlong.empty or "spell_len" not in zlong.columns:
        plt.text(0.5, 0.5, "No long zero‑sales spells", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return

    vals = pd.to_numeric(zlong["spell_len"], errors="coerce").dropna()
    if vals.empty:
        plt.text(0.5, 0.5, "No long zero‑sales spells", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return
    plt.hist(vals, bins=range(1, int(vals.max()) + 2))

    plt.xlabel("Spell length (years)"); plt.ylabel("Firms")
    rec = "recovered_after_zero_spell" in zlong.columns and bool(zlong["recovered_after_zero_spell"].any())
    plt.title("Long zero‑sales spells" + (" (recovery annotated in CSV)" if rec else ""))
    _safe_savefig(out_png)

def plot_firms_ever_zero_sales(fez: pd.DataFrame, out_png: str) -> None:
    plt.figure(figsize=(7, 4))
    if fez is None or fez.empty or "ever_zero" not in fez.columns:
        plt.text(0.5, 0.5, "No 'ever zero sales' info", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return
    vc = fez["ever_zero"].value_counts(dropna=False)
    plt.bar(vc.index.astype(str), vc.values)
    plt.xlabel("Ever had zero sales?"); plt.ylabel("Firms"); plt.title("Firms with any zero‑sales year")
    _safe_savefig(out_png)

def plot_partial_year_consistency(partial_df: pd.DataFrame, out_png: str) -> None:
    plt.figure(figsize=(9, 5))
    if partial_df is None or partial_df.empty or \
       not {"flag_is_partial", "expected_partial_by_duration"}.issubset(partial_df.columns):
        plt.text(0.5, 0.5, "No partial‑year consistency data", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return

    df = partial_df.copy()
    # Normalize to {0,1} ints and drop rows where either side is missing
    f = pd.to_numeric(df["flag_is_partial"], errors="coerce").astype("Int64")
    e = pd.to_numeric(df["expected_partial_by_duration"], errors="coerce").astype("Int64")
    valid = pd.DataFrame({"flag": f, "expected": e}).dropna()
    if valid.empty:
        plt.text(0.5, 0.5, "No comparable rows (both expected & flag missing)", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return

    # Confusion-style counts
    ct = pd.crosstab(valid["expected"], valid["flag"], dropna=False)
    ct = ct.reindex(index=[0,1], columns=[0,1], fill_value=0)  # ensure full 2x2
    ct.index = ["expected: full (≥360d)", "expected: partial (<360d)"]
    ct.columns = ["flag=0", "flag=1"]

    ct.plot(kind="bar", stacked=True)
    plt.ylabel("Rows")
    mism = int((valid["flag"] != valid["expected"]).sum())
    denom = int(len(valid))
    share = 100.0 * mism / denom if denom else 0.0
    plt.title(f"Partial‑year flag vs duration — mismatch: {mism}/{denom} ({share:.2f}%)")
    plt.xticks(rotation=0)
    _safe_savefig(out_png)

def plot_identity_residual_hist(df: pd.DataFrame, out_png: str, title: str) -> None:
    plt.figure(figsize=(8, 4.5))
    if df is None or df.empty or "residual" not in df.columns:
        plt.text(0.5, 0.5, "No residuals to plot", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return
    r = pd.to_numeric(df["residual"], errors="coerce").dropna()
    if r.empty:
        plt.text(0.5, 0.5, "Residuals are all NaN", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return
    plt.hist(r, bins=40)
    plt.xlabel("Residual"); plt.ylabel("Rows"); plt.title(title)
    _safe_savefig(out_png)

def plot_hist_of_column(df: pd.DataFrame, col: str, out_png: str, title: str, xlabel: str) -> None:
    plt.figure(figsize=(8, 4.5))
    if df is None or df.empty or col not in df.columns:
        plt.text(0.5, 0.5, f"No '{col}' to plot", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        plt.text(0.5, 0.5, f"'{col}' is empty/NaN", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return
    plt.hist(s, bins=40)
    plt.xlabel(xlabel); plt.ylabel("Rows"); plt.title(title)
    _safe_savefig(out_png)

def plot_temporal_logic(templog: pd.DataFrame, figs_dir: str) -> list:
    paths = []
    if templog is None or templog.empty:
        return paths

    flags = [c for c in ["row_after_exit_year", "founded_after_exit", "row_before_founded_year"] if c in templog.columns]
    p = os.path.join(figs_dir, "temporal_logic_violations.png")
    plt.figure(figsize=(9, 4.5))

    if not flags:
        plt.text(0.5, 0.5, "No temporal-logic fields present", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(p); paths.append(p); return paths

    shares = []
    for c in flags:
        v = templog[c]
        denom = int(v.notna().sum())
        numer = int(v.fillna(False).astype(bool).sum())
        share = (100.0 * numer / denom) if denom else 0.0
        shares.append(share)

    plt.bar(flags, shares)
    for i, v in enumerate(shares):
        plt.text(i, max(v, 0) + 0.02, f"{v:.2f}%", ha="center", va="bottom", fontsize=8)

    plt.ylabel("Share of rows (%)"); plt.xticks(rotation=20, ha="right")

    if max(shares) <= 0.0:
        plt.ylim(0, 1)  # make zero-height bars visible
        plt.title("Temporal logic violations — all 0%")
    else:
        plt.title("Temporal logic violations")
    _safe_savefig(p); paths.append(p)

    # Age histogram (unchanged, but keep robust)
    if "age_at_year" in templog.columns:
        p2 = os.path.join(figs_dir, "age_at_year_hist.png")
        plot_hist_of_column(templog, "age_at_year", p2, "Age at year (when defined)", "age (years)")
        paths.append(p2)

    return paths

def plot_revenue_thresholds_summary(rts: pd.DataFrame, out_png: str) -> None:
    plt.figure(figsize=(9, 4.5))
    if rts is None or rts.empty: 
        plt.text(0.5, 0.5, "No revenue threshold summary", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return
    df = rts[rts.columns.intersection(["threshold_eur",">=threshold_pct"])].dropna()
    if df.empty:
        plt.text(0.5, 0.5, "No >=threshold data to plot", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return
    plt.bar(df["threshold_eur"].astype(int).astype(str), df[">=threshold_pct"])
    plt.xlabel("Threshold (€)"); plt.ylabel("Share ≥ threshold (%)")
    plt.title("Revenue thresholds — coverage")
    _safe_savefig(out_png)

def plot_urban_counts(uc: pd.DataFrame, out_png: str) -> None:
    plt.figure(figsize=(8, 4.5))
    if uc is None or uc.empty or "urban_m_std" not in uc.columns:
        plt.text(0.5, 0.5, "No urban counts to display", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return
    plt.bar(uc["urban_m_std"].astype(str), uc["count"])
    plt.title("Urbanisation bands"); plt.ylabel("count")
    _safe_savefig(out_png)

def plot_employment_flags(df: pd.DataFrame, col: str, out_png: str, title: str) -> None:
    plt.figure(figsize=(7, 4))
    if df is None or df.empty or col not in df.columns:
        plt.text(0.5, 0.5, f"No '{col}' data", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return
    vc = df[col].value_counts(dropna=True)
    plt.bar(vc.index.astype(str), vc.values)
    plt.title(title); plt.ylabel("Firms")
    _safe_savefig(out_png)

def plot_failure_proxies(fp: pd.DataFrame, figs_dir: str) -> list:
    paths = []
    if fp is None or fp.empty: return paths
    if "zero_sales_longest_spell" in fp.columns:
        p1 = os.path.join(figs_dir, "failure_zero_sales_longest_spell.png")
        plot_hist_of_column(fp, "zero_sales_longest_spell", p1, "Longest zero‑sales spell per firm", "years"); paths.append(p1)
    for col, ttl in [("zero_sales_Xyrs","≥X yrs zero‑sales (share)"), ("dormant_Xyrs","Dormant ≥X yrs (share)")]:
        if col in fp.columns:
            p = os.path.join(figs_dir, f"failure_flag_{col}.png")
            plot_employment_flags(fp, col, p, ttl); paths.append(p)
    return paths

def plot_leakage_risk_counts(lr: pd.DataFrame, out_png: str) -> None:
    plt.figure(figsize=(10, 5))
    if lr is None or lr.empty or "risk" not in lr.columns:
        plt.text(0.5, 0.5, "No leakage risk table", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return
    vc = lr["risk"].value_counts()
    vc.sort_values(ascending=True).plot(kind="barh")
    plt.xlabel("Columns"); plt.title("Columns by leakage risk tag")
    _safe_savefig(out_png)

def plot_sales_sign_summary(ss: pd.DataFrame, out_png: str) -> None:
    plt.figure(figsize=(8, 4.5))
    if ss is None or ss.empty:
        plt.text(0.5, 0.5, "No sales sign summary", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return
    row = ss.iloc[0]
    labels = ["neg", "zero", "pos"]
    vals = [float(row.get("neg_count", 0)), float(row.get("zero_count", 0)), float(row.get("pos_count", 0))]
    plt.bar(labels, vals); plt.ylabel("Rows"); plt.title("Sales sign summary")
    _safe_savefig(out_png)

def plot_revenue_bracket_firm_flags(rbf: pd.DataFrame, out_png: str) -> None:
    plt.figure(figsize=(7, 4))
    if rbf is None or rbf.empty or "which" not in rbf.columns:
        plt.text(0.5, 0.5, "No firm‑level revenue flags", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return
    vc = rbf["which"].value_counts()
    plt.bar(vc.index.astype(str), vc.values)
    plt.xlabel("Bracket"); plt.ylabel("Firms"); plt.title("Firms flagged by revenue bracket")
    _safe_savefig(out_png)

def plot_skim_overview(skim: pd.DataFrame, out_png: str, topn: int = 25) -> None:
    plt.figure(figsize=(10, max(4, 0.35 * topn)))
    if skim is None or skim.empty or "unique" not in skim.columns:
        plt.text(0.5, 0.5, "No skim overview to display", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return
    df = skim.reset_index()[["column","unique"]].sort_values("unique", ascending=False).head(topn)
    plt.barh(df["column"].astype(str), df["unique"])
    plt.xlabel("Distinct values"); plt.title("Skim — columns with most unique values")
    plt.gca().invert_yaxis()
    _safe_savefig(out_png)

def plot_dedup_ledger(drop_log: pd.DataFrame, out_png: str, raw_df: Optional[pd.DataFrame] = None) -> None:
    plt.figure(figsize=(8, 4.5))

    # Nothing to read from drop log → fallback to raw DF (if available)
    if drop_log is None or drop_log.empty or "reasons" not in drop_log.columns:
        if raw_df is not None and {"comp_id","year"}.issubset(raw_df.columns):
            g = raw_df[["comp_id","year"]].dropna().groupby(["comp_id","year"]).size().reset_index(name="n")
            dup = g[g["n"] > 1]
            if dup.empty:
                plt.text(0.5, 0.5, "No duplicate firm‑years (raw data)", ha="center", va="center", fontsize=14)
                plt.axis("off"); _safe_savefig(out_png); return
            byn = dup["n"].value_counts().sort_index()
            plt.bar(byn.index.astype(int).astype(str), byn.values)
            plt.xlabel("Duplicate cluster size"); plt.ylabel("Count of clusters")
            plt.title("Raw duplicates by cluster size (no drops were applied)")
            _safe_savefig(out_png); return

        plt.text(0.5, 0.5, "No deduplicated rows", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return

    # Parse duplicate reasons from full drop log
    toks = drop_log["reasons"].fillna("").str.split(";").explode()
    sel = toks[toks.str.contains(r"^duplicate_comp_id_year", regex=True)]
    if sel.empty:
        # Fallback to raw DF analysis if provided
        if raw_df is not None and {"comp_id","year"}.issubset(raw_df.columns):
            g = raw_df[["comp_id","year"]].dropna().groupby(["comp_id","year"]).size().reset_index(name="n")
            dup = g[g["n"] > 1]
            if dup.empty:
                plt.text(0.5, 0.5, "No duplicate firm‑years (raw data)", ha="center", va="center", fontsize=14)
                plt.axis("off"); _safe_savefig(out_png); return
            byn = dup["n"].value_counts().sort_index()
            plt.bar(byn.index.astype(int).astype(str), byn.values)
            plt.xlabel("Duplicate cluster size"); plt.ylabel("Count of clusters")
            plt.title("Raw duplicates by cluster size (no duplicate drops logged)")
            _safe_savefig(out_png); return

        plt.text(0.5, 0.5, "No duplicate_* reasons found", ha="center", va="center", fontsize=14)
        plt.axis("off"); _safe_savefig(out_png); return

    vc = sel.value_counts()
    vc.sort_values(ascending=True).plot(kind="barh")
    plt.xlabel("Rows dropped"); plt.title("Dedup ledger — dropped by duplicate reason")
    _safe_savefig(out_png)

# ------------------------- COLUMN SKIM -------------------------

def shannon_entropy(series: pd.Series) -> float:
    vc = series.value_counts(dropna=True)
    p = (vc / vc.sum()).values
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(p * np.log2(p)).sum() if p.size else np.nan
    return float(ent)

def numeric_skim(s: pd.Series) -> Dict[str, float]:
    # Force numeric float array; guard against pandas BooleanDtype
    if is_bool_dtype(s):
        s = s.astype("Float64")
    else:
        s = pd.to_numeric(s, errors="coerce").astype("Float64")
    n = int(s.shape[0])

    nn = int(s.notna().sum())
    miss = n - nn
    x = s.dropna()

    # Precompute mean/std once
    mean_val = float(x.mean()) if nn else np.nan
    std_val = float(x.std(ddof=1)) if nn > 1 else np.nan

    # Basic stats
    stats_dict = {
        "count": n,
        "non_null": nn,
        "missing": miss,
        "missing_pct": 100 * miss / n if n else np.nan,
        "unique": int(x.nunique()),
        "zeros": int((x == 0).sum()),
        "negatives": int((x < 0).sum()),
        "positives": int((x > 0).sum()),
        "mean": mean_val,
        "std": std_val,
        "min": float(x.min()) if nn else np.nan,
        "p01": float(x.quantile(0.01)) if nn else np.nan,
        "p05": float(x.quantile(0.05)) if nn else np.nan,
        "q25": float(x.quantile(0.25)) if nn else np.nan,
        "median": float(x.median()) if nn else np.nan,
        "q75": float(x.quantile(0.75)) if nn else np.nan,
        "p95": float(x.quantile(0.95)) if nn else np.nan,
        "p99": float(x.quantile(0.99)) if nn else np.nan,
        "max": float(x.max()) if nn else np.nan,
        "cv": (std_val / mean_val) if (nn > 1 and pd.notna(mean_val) and mean_val != 0) else np.nan,
        "skew": float(stats.skew(x.to_numpy(dtype=float), nan_policy="omit")) if nn > 2 else np.nan,
        "kurtosis": float(stats.kurtosis(x.to_numpy(dtype=float), nan_policy="omit")) if nn > 2 else np.nan,
    }

    # Outliers (IQR)
    if nn:
        q1, q3 = stats_dict["q25"], stats_dict["q75"]
        if not (math.isnan(q1) or math.isnan(q3)):
            iqr = q3 - q1
            lb, ub = q1 - IQR_FACTOR * iqr, q3 + IQR_FACTOR * iqr
            stats_dict.update({
                "iqr": float(iqr),
                "fence_low": float(lb),
                "fence_high": float(ub),
                "outliers_iqr": int(((x < lb) | (x > ub)).sum()),
            })

    # Outliers (z-score)
    z_key = f"outliers_z>={Z_OUTLIER:.1f}"
    if nn > 1 and not math.isnan(std_val) and std_val > 0 and pd.notna(mean_val):
        z = (x - mean_val) / std_val
        stats_dict[z_key] = int((z.abs() >= Z_OUTLIER).sum())
    else:
        stats_dict[z_key] = np.nan

    return stats_dict

def categorical_skim(s: pd.Series, topk: int = 10) -> Dict[str, object]:
    n = int(s.shape[0])
    nn = int(s.notna().sum())
    miss = n - nn
    x = s.dropna().astype("string")
    vc = x.value_counts()
    top_items = vc.head(topk)
    ent = shannon_entropy(x)
    top1 = vc.iloc[0] if not vc.empty else np.nan
    top1_pct = float(top1/nn*100) if nn and not np.isnan(top1) else np.nan
    top2 = vc.iloc[1] if vc.size > 1 else np.nan
    top2_pct = float(top2/nn*100) if nn and not np.isnan(top2) else np.nan
    # Gini impurity & Herfindahl-Hirschman Index (HHI) for dispersion
    p = (vc/nn) if nn else pd.Series(dtype=float)
    gini = float(1 - (p**2).sum()) if nn else np.nan
    hhi = float((p**2).sum()) if nn else np.nan
    rare_rate = float((vc[vc==1].sum())/nn*100) if nn else np.nan

    return {
        "count": n, "non_null": nn, "missing": miss, "missing_pct": 100*miss/n if n else np.nan,
        "unique": int(x.nunique()), "entropy_bits": ent,
        "top1_count": top1, "top1_pct": top1_pct, "top2_count": top2, "top2_pct": top2_pct,
        "gini_impurity": gini, "hhi": hhi, "rare_value_pct": rare_rate,
        "top_values": top_items.to_dict()
    }

def skim_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)

        if is_bool_dtype(s):
            # treat as categorical to avoid numeric quantile on booleans
            row = categorical_skim(s.astype("string"))
        elif is_numeric_dtype(s):
            row = numeric_skim(s)
        elif is_datetime64_any_dtype(s):
            # basic datetime skim
            n = int(s.shape[0]); nn = int(s.notna().sum()); miss = n - nn
            x = s.dropna()
            row = {
                "count": n, "non_null": nn, "missing": miss, "missing_pct": 100*miss/n if n else np.nan,
                "min": x.min() if nn else pd.NaT,
                "q25": x.quantile(0.25) if nn else pd.NaT,
                "median": x.quantile(0.5) if nn else pd.NaT,
                "q75": x.quantile(0.75) if nn else pd.NaT,
                "max": x.max() if nn else pd.NaT,
                "unique": int(x.nunique()),
            }
        else:
            row = categorical_skim(s.astype("string"))

        row["column"] = col
        row["dtype"] = dtype
        rows.append(row)
    skim = pd.DataFrame(rows).set_index("column").sort_index()
    return skim

# ------------------------- VALUE CONSISTENCY & CONSTRAINTS -------------------------

def check_nonnegative(df: pd.DataFrame) -> pd.DataFrame:
    findings = []
    for col in sorted(NONNEGATIVE_COLS):
        if col in df.columns:
            s = to_numeric(df[col])
            neg_count = int((s.dropna() < 0).sum())
            total = int(s.notna().sum())
            findings.append({
                "column": col,
                "negatives": neg_count,
                "non_null": total,
                "neg_pct": 100*neg_count/total if total else np.nan
            })
    return pd.DataFrame(findings)


def check_sales_negatives_and_zero_spells(df: pd.DataFrame, min_zero_years: int = 2,
                                          zero_eps: float = 1e-9) -> Dict[str, pd.DataFrame]:
    out = {}
    if not {"comp_id", "year", "sales"}.issubset(df.columns):
        return out

    work = df[["comp_id", "year", "sales"]].dropna(subset=["comp_id", "year"]).copy()
    work["sales"] = to_numeric(work["sales"])
    work["year"] = to_int(work["year"])

    # Negative sales diagnostics
    neg = work[work["sales"] < 0]
    out["sales_negative_rows"] = neg.sort_values(["comp_id", "year"])

    # Zero sales spells per firm (must be consecutive years)
    g = work.sort_values(["comp_id", "year"]).groupby("comp_id", as_index=False, group_keys=False)

    def _spells(gr: pd.DataFrame) -> pd.DataFrame:
        # In pandas >= 2.2 with include_groups=False, the group key is in gr.name
        comp = gr.name if hasattr(gr, "name") else (int(gr["comp_id"].iloc[0]) if "comp_id" in gr.columns else None)

        g2 = gr.sort_values("year").copy()
        years = g2["year"].to_numpy()
        sales = g2["sales"].fillna(0).to_numpy()
        zero = (np.abs(sales) <= zero_eps)

        spells = []
        start_idx = None
        for i in range(len(years)):
            if zero[i]:
                if start_idx is None:
                    start_idx = i
                else:
                    if years[i] - years[i-1] != 1:  # break if not consecutive year
                        end_idx = i - 1
                        spells.append((int(years[start_idx]), int(years[end_idx]), int(years[end_idx] - years[start_idx] + 1)))
                        start_idx = i
            if (not zero[i]) and (start_idx is not None):
                end_idx = i - 1
                spells.append((int(years[start_idx]), int(years[end_idx]), int(years[end_idx] - years[start_idx] + 1)))
                start_idx = None
        # close final spell
        if start_idx is not None:
            end_idx = len(years) - 1
            spells.append((int(years[start_idx]), int(years[end_idx]), int(years[end_idx] - years[start_idx] + 1)))

        res = pd.DataFrame(spells, columns=["spell_start_year", "spell_end_year", "spell_len"])
        if not res.empty:
            res["comp_id"] = comp
        return res

    # Pandas 2.2+: include_groups=False; older versions: fall back without kwarg
    try:
        spells = g.apply(_spells, include_groups=False)
    except TypeError:
        spells = g.apply(_spells)

    out["zero_sales_spells"] = spells if isinstance(spells, pd.DataFrame) else pd.DataFrame()

    # Firms with spell >= min_zero_years and whether they recovered later
    if not out["zero_sales_spells"].empty:
        long_spells = out["zero_sales_spells"].query("spell_len >= @min_zero_years").copy()
        sales_pos = work.assign(pos=lambda d: d["sales"] > 0)
        later_pos = sales_pos.merge(long_spells[["comp_id", "spell_end_year"]], on="comp_id", how="inner")
        recovered = later_pos[later_pos["year"] > later_pos["spell_end_year"]].groupby("comp_id")["pos"].any().reset_index()
        recovered.rename(columns={"pos": "recovered_after_zero_spell"}, inplace=True)
        long_spells = long_spells.merge(recovered, on="comp_id", how="left").fillna({"recovered_after_zero_spell": False})
        out["zero_sales_spells_long"] = long_spells

    # Firms that ever had zero sales
    ever_zero = work.groupby("comp_id")["sales"].apply(lambda s: bool((np.abs(s) <= zero_eps).any())).rename("ever_zero").reset_index()
    out["firms_ever_zero_sales"] = ever_zero

    return out

def check_balance_sheet_period(df: pd.DataFrame) -> pd.DataFrame:
    if not {"begin", "end"}.issubset(df.columns):
        return pd.DataFrame()
    b = to_datetime(df["begin"])
    e = to_datetime(df["end"])
    # Duration in days (NaT-safe)
    dur = (e - b).dt.days
    return pd.DataFrame({
        "begin_is_null": b.isna(),
        "end_is_null": e.isna(),
        "begin_after_end": (b.notna() & e.notna() & (b >= e)),
        "duration_days": dur,
        "duration_lt_360": dur < 360,
        "duration_gt_370": dur > 370,
    })

def check_partial_year_flag(df: pd.DataFrame, period_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if period_df is None or period_df.empty or "balsheet_notfullyear" not in df.columns:
        return pd.DataFrame()
    flag = df["balsheet_notfullyear"].astype("Int64")
    dur = period_df["duration_days"]
    # Heuristic: < 360 days => should be flagged as partial (1)
    expected_partial = (dur.notna() & (dur < 360)).astype("Int64")
    mism = (flag.notna() & expected_partial.notna() & (flag != expected_partial))
    return pd.DataFrame({
        "flag_is_partial": flag,
        "duration_days": dur,
        "expected_partial_by_duration": expected_partial,
        "flag_mismatch": mism
    })

def check_accounting_identities(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    We do not assume exact identities (different reporting standards),
    but we test plausible relationships and quantify residuals.
    """
    out = {}
    cols = df.columns

    def _nonneg(s):
        return to_numeric(s).fillna(0)

    # (A) Fixed assets plausibility with tangible + intangible (if both present)
    if {"fixed_assets", "tang_assets", "intang_assets"}.issubset(cols):
        fa = _nonneg(df["fixed_assets"])
        ta = _nonneg(df["tang_assets"])
        ia = _nonneg(df["intang_assets"])
        resid = fa - (ta + ia)
        out["identity_fixed_vs_tang_intang"] = pd.DataFrame({
            "fixed_assets": fa, "tang_assets": ta, "intang_assets": ia, "residual": resid
        })

    # (B) Total assets proxy:
    # Case-study suggestion: total_assets_bs ≈ intang + fixed + curr_assets
    if {"intang_assets", "fixed_assets", "curr_assets"}.issubset(cols):
        ia = _nonneg(df["intang_assets"]); fa = _nonneg(df["fixed_assets"]); ca = _nonneg(df["curr_assets"])
        total_proxy = ia + fa + ca
        out["total_assets_proxy"] = pd.DataFrame({
            "intang_assets": ia, "fixed_assets": fa, "curr_assets": ca, "total_assets_proxy": total_proxy
        })

    # (C) Fixed assets vs tangible only (some reporters)
    if {"fixed_assets", "tang_assets"}.issubset(cols):
        fa = _nonneg(df["fixed_assets"]); ta = _nonneg(df["tang_assets"])
        out["fixed_vs_tangible"] = pd.DataFrame({"fixed_assets": fa, "tang_assets": ta, "residual": fa - ta})

    # (D) Liquidity sanity during zero sales: do firms have cash?
    if {"sales", "liq_assets"}.issubset(cols):
        s = to_numeric(df["sales"]).fillna(0)
        la = _nonneg(df["liq_assets"])
        out["liquidity_when_no_sales"] = pd.DataFrame({"zero_sales": (s == 0), "liq_assets": la})

    return out

def check_temporal_logic(df: pd.DataFrame) -> pd.DataFrame:
    """Check founded/exit vs year coherence."""
    out = {}
    if "year" in df.columns:
        y = to_int(df["year"])
    else:
        y = pd.Series([pd.NA] * len(df), dtype="Int64")

    if "exit_year" in df.columns:
        ey = to_int(df["exit_year"])
        out["row_after_exit_year"] = (y.notna() & ey.notna() & (y > ey))
    else:
        out["row_after_exit_year"] = pd.Series([pd.NA] * len(df), dtype="boolean")

    if "founded_year" in df.columns and "exit_year" in df.columns:
        fy = to_int(df["founded_year"])
        ey = to_int(df["exit_year"])
        out["founded_after_exit"] = (fy.notna() & ey.notna() & (fy > ey))
    else:
        out["founded_after_exit"] = pd.Series([pd.NA] * len(df), dtype="boolean")

    if "founded_year" in df.columns and "year" in df.columns:
        fy = to_int(df["founded_year"])
        out["age_at_year"] = (y - fy).where((y.notna() & fy.notna()) & ((y - fy) >= 0), 0)

    return pd.DataFrame(out)

# ------------------------- REVENUE THRESHOLDS & SUMMARIES -------------------------

def revenue_threshold_summary(df: pd.DataFrame, thresholds: List[int]) -> pd.DataFrame:
    if "sales" not in df.columns:
        return pd.DataFrame()
    s = to_numeric(df["sales"])
    rows = []
    total = int(s.notna().sum())
    for t in thresholds:
        ge = int((s >= t).sum())
        lt = int((s < t).sum())
        rows.append({
            "threshold_eur": t,
            ">=threshold_count": ge,
            ">=threshold_pct": 100 * ge / total if total else np.nan,
            "<threshold_count": lt,
            "<threshold_pct": 100 * lt / total if total else np.nan
        })
    # also very low revenue slices
    for t in thresholds:
        le = int((s <= t).sum())
        rows.append({
            "threshold_eur_max": t,
            "<=threshold_count": le,
            "<=threshold_pct": 100 * le / total if total else np.nan
        })
    return pd.DataFrame(rows)

def yearly_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if "year" in df.columns:
        years = to_int(df["year"])
        rows.append(pd.DataFrame({"year": years, "_n": 1}).groupby("year")["_n"].sum().rename("rows").reset_index())
    if {"year", "comp_id"}.issubset(df.columns):
        rows.append(df[["year", "comp_id"]].dropna().drop_duplicates().groupby("year")["comp_id"].nunique().rename("unique_firms").reset_index())
    if "sales" in df.columns and "year" in df.columns:
        tmp = df[["year", "sales"]].copy()
        tmp["sales"] = to_numeric(tmp["sales"])
        rows.append(tmp.groupby("year")["sales"].agg(["count", "median", "mean", "std", "min", "max"]).reset_index())
    if not rows:
        return pd.DataFrame()
    out = rows[0]
    for r in rows[1:]:
        out = out.merge(r, on="year", how="outer")
    out = out.sort_values("year")
    return out

# ------------------------- GEOGRAPHY & INDUSTRY -------------------------

def geography_industry_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out = {}
    if "region_m" in df.columns:
        vc = df["region_m"].astype("string").value_counts(dropna=False)
        out["region_counts"] = vc.reset_index(name="count").rename(columns={"index": "region_m"})
    if "urban_m_std" in df.columns:
        vc = df["urban_m_std"].astype("string").value_counts(dropna=False)
        out["urban_counts"] = vc.reset_index(name="count").rename(columns={"index": "urban_m_std"})
    if "ind2_name" in df.columns:
        vc = df["ind2_name"].astype("string").value_counts(dropna=False)
        out["ind2_counts"] = vc.reset_index(name="count").rename(columns={"index": "ind2_name"})
    elif "ind2" in df.columns:
        vc = df["ind2"].astype("string").value_counts(dropna=False)
        out["ind2_counts"] = vc.reset_index(name="count").rename(columns={"index": "ind2"})
    return out

# ------------------------- WORKFORCE, PAY & CEO -------------------------

def workforce_pay_diagnostics(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out = {}
    cols = df.columns
    if {"personnel_exp", "labor_avg"}.issubset(cols):
        # Cost per employee — careful with zeros
        pe = to_numeric(df["personnel_exp"])
        le = to_numeric(df["labor_avg"])
        per_emp = pe / le.replace({0: np.nan})
        out["personnel_cost_per_employee"] = pd.DataFrame({"personnel_exp": pe, "labor_avg": le, "per_employee": per_emp})
    if {"wages", "personnel_exp"}.issubset(cols):
        w = to_numeric(df["wages"])
        pe = to_numeric(df["personnel_exp"])
        out["wages_to_personnel_ratio"] = pd.DataFrame({"wages": w, "personnel_exp": pe, "ratio": w / pe.replace({0: np.nan})})
    # CEO counts and birth year (age)
    if "birth_year" in cols and "year" in cols:
        by = to_int(df["birth_year"]); y = to_int(df["year"])
        out["ceo_age"] = pd.DataFrame({"ceo_age": (y - by).where(y.notna() & by.notna())})
    if "ceo_count" in cols and {"sales", "personnel_exp"}.issubset(cols):
        cc = to_numeric(df["ceo_count"])
        pe = to_numeric(df["personnel_exp"])
        out["personnel_exp_per_ceo"] = pd.DataFrame({"ceo_count": cc, "personnel_exp": pe, "per_ceo": pe / cc.replace({0: np.nan})})
    return out

def employment_dynamics(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out = {}
    if not {"comp_id", "year", "labor_avg"}.issubset(df.columns):
        return out
    tmp = df[["comp_id", "year", "labor_avg"]].dropna(subset=["comp_id", "year"]).copy()
    tmp["year"] = to_int(tmp["year"])
    tmp["labor_avg"] = to_numeric(tmp["labor_avg"])
    tmp = tmp.sort_values(["comp_id", "year"])
    # firm-level trend: decreasing employees flag
    tmp["delta_labor"] = tmp.groupby("comp_id")["labor_avg"].diff()

    firm_decreasing = (
        tmp.groupby("comp_id")["delta_labor"]
          .apply(lambda s: (s.dropna().size > 0) and (s.dropna() < 0).all())
          .rename("always_decreasing").reset_index()
    )

    out["firm_decreasing_employment"] = firm_decreasing
    # zero employees
    zero_emp = tmp.assign(is_zero=lambda d: d["labor_avg"].fillna(0) == 0).groupby("comp_id")["is_zero"].any().rename("ever_zero_employees").reset_index()
    out["firms_ever_zero_employees"] = zero_emp
    return out

# ------------------------- FAILURE PROXIES & LEAKAGE -------------------------

def failure_proxies(df: pd.DataFrame, min_zero_years: int) -> pd.DataFrame:
    """
    Construct heuristic failure flags complementary to exit_year:
    - zero_sales_Xyrs: at least X consecutive years with zero sales
    - zero_sales_and_zero_employees: dormancy proxy
    - persistent_losses: >= X years negative profits
    """
    if not {"comp_id", "year"}.issubset(df.columns):
        return pd.DataFrame()
    x = df.copy()
    x["sales"] = to_numeric(x.get("sales"))
    x["labor_avg"] = to_numeric(x.get("labor_avg"))
    x["profit_loss_year"] = to_numeric(x.get("profit_loss_year"))
    x["year"] = to_int(x["year"])
    x = x.sort_values(["comp_id", "year"])
    grp = x.groupby("comp_id", group_keys=False)

    # helper: length of max consecutive condition
    def max_consec(cond: pd.Series) -> int:
        best = cur = 0
        for v in cond.fillna(False):
            if v:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best

    def _featrow(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "zero_sales_longest_spell": max_consec((g["sales"].fillna(0) == 0)),
            "zero_sales_and_zero_employees_longest": max_consec(
                (g["sales"].fillna(0) == 0) & (g["labor_avg"].fillna(0) == 0)
            ),
            "years_negative_profit": int((g["profit_loss_year"].fillna(0) < 0).sum()),
        })

    # Newer pandas: include_groups=False; fall back for older versions
    try:
        feats = grp.apply(_featrow, include_groups=False).reset_index()
    except TypeError:
        feats = grp.apply(_featrow).reset_index()

    feats["zero_sales_Xyrs"] = feats["zero_sales_longest_spell"] >= min_zero_years
    feats["dormant_Xyrs"] = feats["zero_sales_and_zero_employees_longest"] >= min_zero_years
    return feats

def leakage_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Heuristic labeling of columns that are likely unsafe as predictors for bankruptcy/exit models.
    """
    cols = set(df.columns)
    unsafe_exact = {"exit_year", "exit_date"} & cols
    ids = {"comp_id"}
    times = {"year", "begin", "end", "founded_year", "founded_date"}
    # Keep financials but recommend **lagging** if modeling
    recommend_lag = {
        c for c in cols if c in {
            "sales","profit_loss_year","inc_bef_tax","COGS","material_exp","personnel_exp","wages",
            "extra_exp","extra_inc","extra_profit_loss","amort","curr_assets","fixed_assets","intang_assets",
            "tang_assets","liq_assets","inventories","curr_liab","share_eq","subscribed_cap","finished_prod",
            "labor_avg","foreign","female","ceo_count","birth_year","inoffice_days","gender","origin",
            "ind2","nace_main","region_m","urban_m"
        }
    }
    rows = []
    for c in sorted(cols):
        if c in unsafe_exact:
            risk = "LEAKAGE_TARGET"
            note = "Directly encodes outcome timing; exclude."
        elif c in ids:
            risk = "IDENTIFIER"
            note = "Use only for grouping; exclude as feature."
        elif c in times:
            risk = "TIME_INDEX"
            note = "Use for splitting/lagging; not a raw predictor."
        elif c in recommend_lag:
            risk = "OK_IF_LAGGED"
            note = "Use lagged values to avoid peeking."
        else:
            risk = "UNKNOWN"
            note = "Review."
        rows.append({"column": c, "risk": risk, "note": note})
    return pd.DataFrame(rows).sort_values(["risk","column"])

# ------------------------- FLAGS INVENTORY & CLEANING -------------------------
def build_flags_inventory(df: pd.DataFrame,
                          var_dict: Optional[Dict[str, str]] = None,
                          expected_year_min: int = EXPECTED_YEAR_MIN,
                          expected_year_max: int = EXPECTED_YEAR_MAX
                          ) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """Detect source and derived flags, attach definitions & recommended actions."""
    flags: Dict[str, pd.Series] = {}

    # Year coverage & key presence
    if "year" in df.columns:
        y = to_int(df["year"])
        flags[f"year_out_of_[{expected_year_min},{expected_year_max}]"] = ~(y.between(expected_year_min, expected_year_max))
    flags["comp_id_missing"] = ~df.get("comp_id", pd.Series([pd.NA]*len(df))).notna()
    flags["year_missing"] = ~df.get("year", pd.Series([pd.NA]*len(df))).notna()

    # Duplicates (two masks):
    #  - duplicate_comp_id_year: any row beyond the first in each firm-year cluster
    #  - duplicate_comp_id_year_noncanonical: rows to DROP when keeping "most complete & latest end"
    if {"comp_id", "year"}.issubset(df.columns):
        flags["duplicate_comp_id_year"] = df.duplicated(subset=["comp_id", "year"], keep="first").fillna(False)

        x = df.copy()
        x["comp_id"] = to_int(x["comp_id"])
        x["year"] = to_int(x["year"])
        dup_all = x.duplicated(subset=["comp_id", "year"], keep=False)
        drop_mask = pd.Series(False, index=df.index)
        if dup_all.any():
            g = x.loc[dup_all].copy()
            g["_nonnull"] = g.notna().sum(axis=1)
            if "end" in g.columns:
                g["_end"] = to_datetime(g["end"])
                sort_cols, asc = ["_nonnull", "_end"], [False, False]
            else:
                sort_cols, asc = ["_nonnull"], [False]
            for _, gr in g.groupby(["comp_id", "year"], sort=True):
                keep = gr.sort_values(by=sort_cols, ascending=asc, kind="mergesort").index[0]
                drop_mask.loc[gr.index.difference([keep])] = True
        flags["duplicate_comp_id_year_noncanonical"] = drop_mask.fillna(False)

    # Period logic
    if {"begin","end"}.issubset(df.columns):
        b = to_datetime(df["begin"]); e = to_datetime(df["end"]); dur = (e - b).dt.days
        flags["begin_after_or_eq_end"] = (b.notna() & e.notna() & (b >= e))
        flags["balsheet_duration_lt_360"] = (dur < 360).fillna(False)
        flags["balsheet_duration_gt_370"] = (dur > 370).fillna(False)
        if "balsheet_notfullyear" in df.columns:
            f = to_int(df["balsheet_notfullyear"])
            expected_partial = (dur.notna() & (dur < 360)).astype("Int64")
            mismatch = (f.notna() & expected_partial.notna() & (f != expected_partial))
            flags["partial_year_flag_mismatch"] = mismatch.fillna(False)

    # Temporal logic
    if "exit_year" in df.columns and "year" in df.columns:
        ey = to_int(df["exit_year"]); y = to_int(df["year"])
        flags["row_after_exit_year"] = (y.notna() & ey.notna() & (y > ey)).fillna(False)
    if "founded_year" in df.columns and "exit_year" in df.columns:
        fy = to_int(df["founded_year"]); ey = to_int(df["exit_year"])
        flags["founded_after_exit"] = (fy.notna() & ey.notna() & (fy > ey)).fillna(False)
    if {"year","founded_year"}.issubset(df.columns):
        y = to_int(df["year"]); fy = to_int(df["founded_year"])
        flags["row_before_founded_year"] = (y.notna() & fy.notna() & (y < fy)).fillna(False)

    # Non-negative domain checks
    for col in sorted(NONNEGATIVE_COLS):
        if col in df.columns:
            s = to_numeric(df[col])
            flags[f"negative_{col}"] = (s < 0).fillna(False)

    # Shares outside [0,1] — do NOT count NaNs as outside
    for col in ("foreign","female"):
        if col in df.columns:
            s = to_numeric(df[col])
            flags[f"{col}_outside_[0,1]"] = (s.notna()) & (~s.between(0,1))

    # Sales sign/missing + revenue brackets
    if "sales" in df.columns:
        s = to_numeric(df["sales"])
        flags["sales_negative"] = (s < 0).fillna(False)
        flags["sales_zero"] = (s == 0).fillna(False)
        flags["sales_missing"] = s.isna()

        # New: explicit bracket flags 
        flags["revenue_lt_1k"] = (s.notna() & (s < 1_000)).fillna(False)
        flags["revenue_gt_10m"] = (s.notna() & (s > 10_000_000)).fillna(False)

    # Auto-discover source flags by name/shape
    name_keys = ("flag","problem","error","missing","mismatch","notfull","partial","invalid","issue","anomaly","outlier","bad")
    for col in df.columns:
        if col in flags:
            continue
        lname = str(col).lower()
        if any(k in lname for k in name_keys):
            s = df[col]
            try:
                if s.dropna().map(lambda x: isinstance(x, (bool, np.bool_))).all():
                    flags[col] = s.fillna(False)
                else:
                    num = pd.to_numeric(s, errors="coerce")
                    if num.notna().any():
                        flags[col] = (num == 1).fillna(False)
            except Exception:
                pass

    # Summarize + definitions + actions
    def _stats(name: str, s: pd.Series) -> Dict[str, object]:
        n = int(len(s)); nn = int(pd.Series(s).notna().sum())
        b = pd.Series(s).fillna(False).astype(bool)
        flagged = int(b.sum())
        return {"flag": name, "rows": n, "non_null": nn, "missing": n-nn,
                "missing_pct": (n-nn)*100/n if n else np.nan,
                "flagged_count": flagged, "flagged_pct": flagged*100/n if n else np.nan}

    flags_df = pd.DataFrame([_stats(k, v) for k, v in flags.items()]) \
                 .sort_values("flagged_pct", ascending=False).reset_index(drop=True)

    def _defn(flag_name: str) -> str:
        if var_dict and flag_name in var_dict: return var_dict[flag_name]
        f = flag_name.lower()
        if f == "balsheet_notfullyear": return "Financial statement covers <12 months (partial year)."
        if f.startswith("negative_"): return f"{flag_name.split('negative_',1)[1]} should be nonnegative but observed < 0."
        if f == "begin_after_or_eq_end": return "Balance-sheet period start is on/after end — invalid period."
        if f == "partial_year_flag_mismatch": return "Duration < 360 days but balsheet_notfullyear != 1 (or vice versa)."
        if f.startswith("year_out_of_"): return "Observation year outside expected coverage."
        if f in {"comp_id_missing","year_missing"}: return "Primary key component is missing."
        if f == "duplicate_comp_id_year": return "Duplicate firm-year combination."
        if f == "duplicate_comp_id_year_noncanonical": return "Duplicate firm-year: this row is non-canonical (kept row is most complete & latest end)."
        if f == "row_after_exit_year": return "Row year exceeds exit_year for the firm."
        if f == "founded_after_exit": return "founded_year > exit_year — impossible ordering."
        if f.endswith("_outside_[0,1]"): return f"{flag_name.replace('_outside_[0,1]','')} share is outside [0,1]."
        if f == "sales_negative": return "Sales < 0 (could reflect returns; treat with care)."
        if f == "sales_zero": return "Sales equal to 0."
        if f == "sales_missing": return "Sales is missing (NA)."
        if f == "row_before_founded_year": return "row year < founded_year."
        if f == "revenue_lt_1k": return "Row revenue (sales) < €1,000."
        if f == "revenue_gt_10m": return "Row revenue (sales) > €10,000,000."

        if any(k in f for k in ("flag","error","problem","missing","mismatch","invalid","issue","anomaly","outlier","bad")):
            return "Flag/indicator present in source data."
        return "Binary/flag-like indicator (inferred)."

    def _action(flag_name: str) -> str:
        f = flag_name.lower()
        if f in {
            "comp_id_missing","year_missing","begin_after_or_eq_end",
            "duplicate_comp_id_year","duplicate_comp_id_year_noncanonical",
            "row_after_exit_year","founded_after_exit",
        }:
            return "DROP ROW"
        if f.startswith("year_out_of_"):
            return "DROP ROW"
        if f in {"revenue_lt_1k","revenue_gt_10m"}:
            return "KEEP (review; revenue bracket)"
        if f.startswith("negative_"):
            return "FIX VALUE (set to 0) + KEEP"
        if f.endswith("_outside_[0,1]"):
            return "SET NaN + KEEP"
        if f == "sales_negative":
            return "KEEP (cap to 1 for logs)"
        if f in {"sales_zero","sales_missing"}:
            return "KEEP (impute/indicator as needed)"
        if f == "partial_year_flag_mismatch":
            return "KEEP (audit; consider fixing flag)"
        if f == "balsheet_notfullyear":
            return "KEEP (modeling note)"
        return "KEEP (review downstream)"

    flags_df["definition"] = flags_df["flag"].map(_defn)
    flags_df["recommended_action"] = flags_df["flag"].map(_action)
    return flags_df, flags

def apply_cleaning(df: pd.DataFrame,
                   flags: Dict[str, pd.Series],
                   fix_nonnegatives: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply strict row drops and safe value fixes. Returns (df_clean, drop_log, fix_summary).
    """
    # Prefer smarter duplicate drop if available
    dup_key = "duplicate_comp_id_year_noncanonical" if "duplicate_comp_id_year_noncanonical" in flags else "duplicate_comp_id_year"

    drop_keys = [
        "comp_id_missing", "year_missing", "begin_after_or_eq_end",
        dup_key, "row_after_exit_year", "founded_after_exit"
    ]
    # dynamic year_out_of_[MIN,MAX]
    drop_keys.extend([k for k in flags.keys() if k.startswith("year_out_of_[")])

    drop_mask = pd.Series(False, index=df.index)
    for k in drop_keys:
        if k in flags:
            s_flag = pd.Series(flags[k]).reindex(df.index)
            s_flag = s_flag.fillna(False).astype(bool)
            drop_mask = drop_mask | s_flag


    # Value fixes: negatives -> 0 for nonnegative columns
    fix_records: List[Dict[str, int]] = []
    if fix_nonnegatives:
        SAFE_NONNEG_COLS = sorted(set(NONNEGATIVE_COLS) - {"sales"})
        for col in SAFE_NONNEG_COLS:
            if col in df.columns:
                s = to_numeric(df[col])
                neg_mask = (s < 0)
                nfix = int(neg_mask.fillna(False).sum())
                if nfix > 0:
                    df[col] = s.mask(neg_mask, 0)
                    fix_records.append({"column_fixed": col, "rows_affected": nfix})

    # Drop log (use .loc — index labels, not positions)
    drop_idx = df.index[drop_mask]
    drop_reasons = []
    for i in drop_idx:
        rs = ";".join([k for k in drop_keys if k in flags and bool(flags[k].loc[i])])
        drop_reasons.append(rs)
    drop_log = pd.DataFrame({"row_index": drop_idx, "reasons": drop_reasons})

    df_clean = df.loc[~drop_mask].copy()
    fix_summary = (pd.DataFrame(fix_records).sort_values("rows_affected", ascending=False)
                   if fix_records else pd.DataFrame(columns=["column_fixed","rows_affected"]))
    return df_clean, drop_log, fix_summary

# ------------------------- MAIN PIPELINE -------------------------
def _safe_savefig(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def make_visuals(df: pd.DataFrame, artifacts: Dict[str, pd.DataFrame], out_dir: str) -> List[str]:
    """Generate overview plots — now one figure per CSV artifact saved."""
    figs_dir = os.path.join(out_dir, "figs")
    ensure_dir(figs_dir)
    paths = []

    # 0) Type coercions (CLEAN)
    tc = artifacts.get("type_coercion_report")
    if tc is not None:
        p = os.path.join(figs_dir, "type_coercions_clean.png"); plot_type_coercions(tc, p); paths.append(p)

    # 1) Skim
    skim = artifacts.get("skim_per_column")
    if skim is not None and not skim.empty:
        p = os.path.join(figs_dir, "skim_unique_top.png"); plot_skim_overview(skim, p); paths.append(p)

    # 2) Missingness – top 30 columns
    miss = artifacts.get("missingness")
    if miss is not None and "missing_pct" in miss.columns:
        m = miss.sort_values("missing_pct", ascending=False).head(30)
        plt.figure(figsize=(10, max(4, 0.25*len(m))))
        plt.barh(m.index if m.index.name else m.reset_index()["index"], m["missing_pct"])
        plt.xlabel("Missing (%)"); plt.title("Missingness – Top 30 columns")
        p = os.path.join(figs_dir, "missingness_top30.png"); _safe_savefig(p); paths.append(p)

    # 3) Sales distribution (log10 positives)
    if "sales" in df.columns:
        s = to_numeric(df["sales"]); sp = s[s > 0]
        if sp.notna().any():
            plt.figure(figsize=(8, 5))
            plt.hist(np.log10(sp), bins=50)
            plt.xlabel("log10(sales €)"); plt.ylabel("count")
            neg = int((s < 0).sum()); zero = int((s == 0).sum()); pos = int((s > 0).sum())
            plt.title(f"Sales distribution (positives). neg={neg}, zero={zero}, pos={pos}")
            p = os.path.join(figs_dir, "sales_log10_hist.png"); _safe_savefig(p); paths.append(p)

    # 4) Yearly summary — unique firms & median sales
    ys = artifacts.get("yearly_summary")
    if ys is not None and "unique_firms" in ys.columns:
        plt.figure(figsize=(9, 4.5))
        plt.plot(ys["year"], ys["unique_firms"], marker="o")
        plt.xlabel("Year"); plt.ylabel("Unique firms"); plt.title("Unique firms by year")
        p = os.path.join(figs_dir, "firms_by_year.png"); _safe_savefig(p); paths.append(p)
    if ys is not None and "median" in ys.columns:
        plt.figure(figsize=(9, 4.5))
        plt.plot(ys["year"], ys["median"], marker="o")
        plt.xlabel("Year"); plt.ylabel("Median sales (€)"); plt.title("Median sales by year")
        p = os.path.join(figs_dir, "median_sales_by_year.png"); _safe_savefig(p); paths.append(p)

    # 5) Nonnegativity audit
    nonneg = artifacts.get("nonnegativity_audit")
    if nonneg is not None and not nonneg.empty:
        p = os.path.join(figs_dir, "nonnegativity_audit.png"); plot_nonnegativity_audit(nonneg, p); paths.append(p)

    # 6) Negative sales rows by year
    negrows = artifacts.get("sales_negative_rows")
    if negrows is not None and not negrows.empty:
        p = os.path.join(figs_dir, "negative_sales_by_year.png"); plot_sales_negative_by_year(negrows, p); paths.append(p)

    # 7) Zero‑sales spells (lengths + long spells)
    zsp = artifacts.get("zero_sales_spells")
    if zsp is not None and not zsp.empty and "spell_len" in zsp.columns:
        plt.figure(figsize=(8, 4.5))
        plt.hist(zsp["spell_len"], bins=range(1, int(zsp["spell_len"].max()) + 2))
        plt.xlabel("Spell length (years with zero sales, consecutive)"); plt.ylabel("count")
        plt.title("Zero‑sales spell length distribution")
        p = os.path.join(figs_dir, "zero_sales_spells_hist.png"); _safe_savefig(p); paths.append(p)
    zlong = artifacts.get("zero_sales_spells_long")
    if zlong is not None and not zlong.empty:
        p = os.path.join(figs_dir, "zero_sales_spells_long.png"); plot_zero_sales_spells_long(zlong, p); paths.append(p)
    fez = artifacts.get("firms_ever_zero_sales")
    if fez is not None and not fez.empty:
        p = os.path.join(figs_dir, "firms_ever_zero_sales.png"); plot_firms_ever_zero_sales(fez, p); paths.append(p)

    # 8) Region & Urban & Industry
    rc = artifacts.get("region_counts")
    if rc is not None and not rc.empty:
        plt.figure(figsize=(8, 4.5))
        plt.bar(rc["region_m"].astype(str), rc["count"]); plt.xticks(rotation=45, ha="right")
        plt.ylabel("count"); plt.title("HQ regions")
        p = os.path.join(figs_dir, "region_counts.png"); _safe_savefig(p); paths.append(p)
    uc = artifacts.get("urban_counts")
    if uc is not None and not uc.empty:
        p = os.path.join(figs_dir, "urban_counts.png"); plot_urban_counts(uc, p); paths.append(p)
    ic = artifacts.get("ind2_counts")
    if ic is not None and not ic.empty:
        top = ic.head(15)
        plt.figure(figsize=(10, 5))
        plt.bar(top.iloc[:, 0].astype(str), top["count"]); plt.xticks(rotation=45, ha="right")
        plt.ylabel("count"); plt.title("Top industries (ind2)")
        p = os.path.join(figs_dir, "ind2_top15.png"); _safe_savefig(p); paths.append(p)

    # 9) Balance‑sheet period + partial‑year consistency
    period = artifacts.get("balance_sheet_period_audit")
    if period is not None and "duration_days" in period.columns:
        dd = period["duration_days"].dropna()
        if dd.size:
            plt.figure(figsize=(8, 4.5))
            plt.hist(dd, bins=40)
            plt.xlabel("Duration (days)"); plt.ylabel("count"); plt.title("Balance‑sheet period duration")
            p = os.path.join(figs_dir, "balsheet_duration.png"); _safe_savefig(p); paths.append(p)
    partial = artifacts.get("partial_year_flag_consistency")
    if partial is not None and not partial.empty:
        p = os.path.join(figs_dir, "partial_year_flag_consistency.png"); plot_partial_year_consistency(partial, p); paths.append(p)

    # 10) Accounting identities
    id_fixed_ti = artifacts.get("identity_fixed_vs_tang_intang")
    if id_fixed_ti is not None and not id_fixed_ti.empty:
        p = os.path.join(figs_dir, "identity_fixed_vs_tang_intang_residual.png")
        plot_identity_residual_hist(id_fixed_ti, p, "Fixed vs Tangible+Intangible — residual"); paths.append(p)
    id_fixed_t = artifacts.get("identity_fixed_vs_tangible")
    if id_fixed_t is not None and not id_fixed_t.empty:
        p = os.path.join(figs_dir, "identity_fixed_vs_tangible_residual.png")
        plot_identity_residual_hist(id_fixed_t, p, "Fixed vs Tangible — residual"); paths.append(p)
    id_total = artifacts.get("identity_total_assets_proxy")
    if id_total is not None and not id_total.empty:
        p = os.path.join(figs_dir, "total_assets_proxy_dist.png")
        plot_hist_of_column(id_total, "total_assets_proxy", p, "Total assets proxy (IA+FA+CA)", "€"); paths.append(p)
    id_liq = artifacts.get("identity_liquidity_when_no_sales")
    if id_liq is not None and not id_liq.empty and "zero_sales" in id_liq.columns:
        # distribution of liquidity when sales == 0
        z = id_liq[id_liq["zero_sales"] == True]
        p = os.path.join(figs_dir, "liq_assets_when_zero_sales.png")
        plot_hist_of_column(z, "liq_assets", p, "Liquidity when sales == 0", "€"); paths.append(p)

    # 11) Temporal logic
    tl = artifacts.get("temporal_logic_audit")
    paths.extend(plot_temporal_logic(tl, figs_dir))

    # 12) Revenue thresholds + brackets
    rts = artifacts.get("revenue_thresholds_summary")
    if rts is not None and not rts.empty:
        p = os.path.join(figs_dir, "revenue_thresholds_summary.png"); plot_revenue_thresholds_summary(rts, p); paths.append(p)
    rb = artifacts.get("revenue_bracket_row_flags")
    if rb is not None and not rb.empty:
        p = os.path.join(figs_dir, "revenue_brackets_by_year.png"); plot_revenue_bracket_timeseries(rb, p); paths.append(p)
    rbf = artifacts.get("revenue_bracket_firm_flags")
    if rbf is not None and not rbf.empty:
        p = os.path.join(figs_dir, "revenue_bracket_firm_flags.png"); plot_revenue_bracket_firm_flags(rbf, p); paths.append(p)

    # 13) Workforce & pay
    pce = artifacts.get("personnel_cost_per_employee")
    if pce is not None and not pce.empty:
        p = os.path.join(figs_dir, "personnel_cost_per_employee_hist.png")
        plot_hist_of_column(pce, "per_employee", p, "Personnel cost per employee", "€ / employee"); paths.append(p)
    wpr = artifacts.get("wages_to_personnel_ratio")
    if wpr is not None and not wpr.empty:
        p = os.path.join(figs_dir, "wages_to_personnel_ratio_hist.png")
        plot_hist_of_column(wpr, "ratio", p, "Wages / Personnel expenses", "ratio"); paths.append(p)
    ceo_age = artifacts.get("ceo_age")
    if ceo_age is not None and not ceo_age.empty:
        p = os.path.join(figs_dir, "ceo_age_hist.png")
        plot_hist_of_column(ceo_age, "ceo_age", p, "CEO age at year", "years"); paths.append(p)
    ppc = artifacts.get("personnel_exp_per_ceo")
    if ppc is not None and not ppc.empty:
        p = os.path.join(figs_dir, "personnel_exp_per_ceo_hist.png")
        plot_hist_of_column(ppc, "per_ceo", p, "Personnel expenses per CEO", "€ / CEO"); paths.append(p)

    # 14) Employment dynamics
    dec = artifacts.get("firm_decreasing_employment")
    if dec is not None and not dec.empty:
        p = os.path.join(figs_dir, "firm_decreasing_employment_share.png")
        plot_employment_flags(dec, "always_decreasing", p, "Firms with always‑decreasing employment"); paths.append(p)
    zez = artifacts.get("firms_ever_zero_employees")
    if zez is not None and not zez.empty:
        p = os.path.join(figs_dir, "firms_ever_zero_employees.png")
        plot_employment_flags(zez, "ever_zero_employees", p, "Firms that ever had zero employees"); paths.append(p)

    # 15) Failure proxies
    fp = artifacts.get("failure_proxies_by_firm")
    paths.extend(plot_failure_proxies(fp, figs_dir))

    # 16) Leakage risk table
    lr = artifacts.get("leakage_risk_table")
    if lr is not None and not lr.empty:
        p = os.path.join(figs_dir, "leakage_risk_counts.png"); plot_leakage_risk_counts(lr, p); paths.append(p)

    # 17) Sales sign summary
    ssign = artifacts.get("sales_sign_summary")
    if ssign is not None and not ssign.empty:
        p = os.path.join(figs_dir, "sales_sign_summary.png"); plot_sales_sign_summary(ssign, p); paths.append(p)

    # 18) Asymmetry & inconsistency & few obs & class imbalance (already had)
    asym = artifacts.get("asymmetry_per_column")
    if asym is not None and not asym.empty:
        p = os.path.join(figs_dir, "asymmetry_top_max_abs_skew.png"); plot_asymmetry_audit(asym, p); paths.append(p)
    inc = artifacts.get("inconsistency_per_column")
    if inc is not None and not inc.empty:
        paths.extend(plot_inconsistency_panels(inc, figs_dir))
    low_cov = artifacts.get("columns_low_coverage")
    if low_cov is not None and not low_cov.empty:
        p = os.path.join(figs_dir, "columns_low_coverage.png"); plot_low_coverage(low_cov, p); paths.append(p)
    rare_vals = artifacts.get("rare_values_long")
    if rare_vals is not None and not rare_vals.empty:
        p = os.path.join(figs_dir, "rare_values_by_column.png"); plot_rare_values_by_column(rare_vals, p); paths.append(p)
    imb = artifacts.get("class_imbalance_by_column")
    if imb is not None and not imb.empty:
        p = os.path.join(figs_dir, "class_imbalance_top.png"); plot_class_imbalance(imb, p); paths.append(p)

    # 19) Zero‑sales resource crosscheck (already had)
    cross = artifacts.get("zero_sales_resource_crosscheck")
    if cross is not None and not cross.empty and "zero_sales" in cross.columns:
        zs = cross[cross["zero_sales"] == True].copy()
        if not zs.empty:
            grp = zs.assign(
                liq=lambda d: d["has_liq_assets"].map({True:"liq>0", False:"liq==0"}),
                pay=lambda d: d["has_personnel_exp"].map({True:"payroll>0", False:"payroll==0"})
            ).groupby(["liq","pay"]).size().reset_index(name="count")
            cats = ["liq>0|payroll>0","liq>0|payroll==0","liq==0|payroll>0","liq==0|payroll==0"]
            grp["cat"] = grp["liq"] + "|" + grp["pay"]
            grp = grp.set_index("cat").reindex(cats).fillna(0).reset_index()
            plt.figure(figsize=(8, 4.5))
            plt.bar(grp["cat"], grp["count"]); plt.xticks(rotation=20, ha="right"); plt.ylabel("count")
            plt.title("Resource status while sales == 0")
            p = os.path.join(figs_dir, "zero_sales_resources.png"); _safe_savefig(p); paths.append(p)

    return paths

def compute_core_artifacts(df: pd.DataFrame,
                           min_zero_sales_years: int,
                           nace4_csv: Optional[str]) -> Dict[str, pd.DataFrame]:
    artifacts: Dict[str, pd.DataFrame] = {}

    # Schema & coercion
    df2, coercion_report = enforce_schema(df.copy())
    artifacts["type_coercion_report"] = coercion_report

    # NACE enrichments
    df2 = add_ind2_name(df2)
    df2 = add_nace_main_2d(df2)
    if nace4_csv and os.path.exists(nace4_csv) and "nace_main" in df2.columns:
        try:
            m = pd.read_csv(nace4_csv, dtype={"code": "string", "name": "string"})
            code2name = dict(zip(m["code"].str.zfill(4), m["name"]))
            df2["nace_main_4d"] = df2["nace_main"].astype("string").str.extract(r"(\d{4})", expand=False)
            df2["nace_main_name"] = df2["nace_main_4d"].map(code2name)
        except Exception as e:
            warnings.warn(f"Could not load NACE 4-digit mapping: {e}")

    # Skim & missingness
    artifacts["skim_per_column"] = skim_dataframe(df2)
    miss = df2.isna().sum().rename("missing").to_frame()
    miss["non_missing"] = df2.shape[0] - miss["missing"]
    miss["missing_pct"] = 100 * miss["missing"] / df2.shape[0]
    artifacts["missingness"] = miss.sort_values("missing", ascending=False)

    # Nonnegativity, sales spells, balance sheet, identities
    artifacts["nonnegativity_audit"] = check_nonnegative(df2)
    spells = check_sales_negatives_and_zero_spells(df2, min_zero_years=min_zero_sales_years)
    for k, v in spells.items():
        artifacts[k] = v

    period = check_balance_sheet_period(df2)
    if not period.empty:
        artifacts["balance_sheet_period_audit"] = period
        partial = check_partial_year_flag(df2, period)
        if not partial.empty:
            artifacts["partial_year_flag_consistency"] = partial

    identities = check_accounting_identities(df2)
    for k, v in identities.items():
        key = k if str(k).startswith("identity_") else f"identity_{k}"
        artifacts[key] = v

    # Temporal logic, revenue thresholds, yearly summaries
    artifacts["temporal_logic_audit"] = check_temporal_logic(df2)
    artifacts["revenue_thresholds_summary"] = revenue_threshold_summary(df2, REVENUE_THRESHOLDS)
    artifacts["yearly_summary"] = yearly_counts(df2)

    # Geography, workforce, employment, failure proxies, leakage risk
    geo_ind = geography_industry_tables(df2)
    for k, v in geo_ind.items():
        artifacts[k] = v
    wf = workforce_pay_diagnostics(df2)
    for k, v in wf.items():
        artifacts[k] = v
    empdyn = employment_dynamics(df2)
    for k, v in empdyn.items():
        artifacts[k] = v
    artifacts["failure_proxies_by_firm"] = failure_proxies(df2, min_zero_sales_years)
    artifacts["leakage_risk_table"] = leakage_risk(df2)

    # Sales sign summary
    if "sales" in df2.columns:
        s = to_numeric(df2["sales"])
        total_nn = int(s.notna().sum())
        neg = int((s < 0).sum()); zero = int((s == 0).sum()); pos = int((s > 0).sum())
        artifacts["sales_sign_summary"] = pd.DataFrame([{
            "non_null": total_nn,
            "neg_count": neg, "neg_pct": 100*neg/total_nn if total_nn else np.nan,
            "zero_count": zero, "zero_pct": 100*zero/total_nn if total_nn else np.nan,
            "pos_count": pos, "pos_pct": 100*pos/total_nn if total_nn else np.nan
        }])

    # Cross-conditions during zero sales
    if {"sales", "liq_assets", "personnel_exp"}.issubset(df2.columns):
        s = to_numeric(df2["sales"]).fillna(0)
        la = to_numeric(df2["liq_assets"])
        pe = to_numeric(df2["personnel_exp"])
        z = (s == 0)
        cross = pd.DataFrame({
            "zero_sales": z,
            "has_liq_assets": la > 0,
            "has_personnel_exp": pe > 0
        })
        cross["any_resources_while_zero_sales"] = cross["has_liq_assets"] | cross["has_personnel_exp"]
        artifacts["zero_sales_resource_crosscheck"] = cross

    # === NEW: Asymmetry audit (Req. 1) ===
    artifacts["asymmetry_per_column"] = asymmetry_audit(
        df2, skew_threshold=1.0, bowley_threshold=0.30, min_non_null=30
    )

    # === NEW: Inconsistency scan for ALL columns (Req. 2) ===
    inc_df = inconsistency_scan_all_columns(df2)
    if not inc_df.empty:
        artifacts["inconsistency_per_column"] = inc_df

    # === NEW: Revenue brackets (Req. 3) ===
    rb_rows, rb_firms = revenue_bracket_flags(df2, low_eur=1_000, high_eur=10_000_000)
    if not rb_rows.empty:
        artifacts["revenue_bracket_row_flags"] = rb_rows
    if not rb_firms.empty:
        artifacts["revenue_bracket_firm_flags"] = rb_firms

    # === NEW: Few observations — columns & rare values (Req. 4) ===
    low_cov, rare_vals = few_observations_and_rare_values(
        df2,
        # "Low coverage" now means: fewer than 100 non‑nulls (to catch tiny datasets)
        # OR < 80% non‑null coverage (i.e., >20% missing).
        min_column_nonnull=100, min_column_nonnull_pct=80.0,
        rare_value_min_count=20, rare_value_min_pct=0.5,
        max_levels_for_numeric_as_categorical=30
    )


    if not low_cov.empty:
        artifacts["columns_low_coverage"] = low_cov
    if not rare_vals.empty:
        artifacts["rare_values_long"] = rare_vals

    # === NEW: Class imbalance across columns (Req. 5) ===
    imb = class_imbalance_all_columns(
        df2,
        majority_threshold_pct=90.0,
        max_levels=30,
        treat_integer_as_categorical_if_unique_le=30
    )
    if not imb.empty:
        artifacts["class_imbalance_by_column"] = imb

    return artifacts

def run_pipeline(file: str,
                 min_zero_sales_years: int = 2,
                 nace4_csv: Optional[str] = None,
                 var_dict_xls: Optional[str] = None,
                 overwrite: bool = False) -> None:
    # Always save next to this .py
    out_base_dir = script_dir()
    figs_dir = os.path.join(out_base_dir, "figs")
    ensure_dir(figs_dir)

    # 1) Load
    if str(file).lower().endswith(".parquet"):
        df = pd.read_parquet(file)  # read is fine; we never write Parquet
    else:
        df = pd.read_csv(file, low_memory=False, na_values=["NA","NaN","","null","None"])

    # 2) Schema & enrichments (quiet)
    df, coercion_report = enforce_schema(df)
    df = add_ind2_name(df)
    df = add_nace_main_2d(df)
    if nace4_csv and os.path.exists(nace4_csv) and "nace_main" in df.columns:
        try:
            m = pd.read_csv(nace4_csv, dtype={"code": "string", "name": "string"})
            code2name = dict(zip(m["code"].str.zfill(4), m["name"]))
            df["nace_main_4d"] = df["nace_main"].astype("string").str.extract(r"(\d{4})", expand=False)
            df["nace_main_name"] = df["nace_main_4d"].map(code2name)
        except Exception:
            pass

    # 3) Optional var-dict (quiet)
    var_dict = None
    if var_dict_xls and os.path.exists(var_dict_xls):
        try:
            try:
                var_df = pd.read_excel(var_dict_xls, engine="xlrd")
            except Exception:
                var_df = pd.read_excel(var_dict_xls)
            cands = {c.lower().strip(): c for c in var_df.columns}
            var_col = cands.get("variable", list(var_df.columns)[0])
            desc_col = cands.get("description", list(var_df.columns)[1] if var_df.shape[1] > 1 else list(var_df.columns)[0])
            var_df = var_df[[var_col, desc_col]].copy()
            var_df.columns = ["variable","description"]
            var_df["variable"] = var_df["variable"].astype(str).str.strip()
            var_df["description"] = var_df["description"].astype(str).str.strip()
            var_dict = dict(zip(var_df["variable"], var_df["description"]))
        except Exception:
            pass

    # 4) Build flags BEFORE cleaning
    flags_df_before, flags_before = build_flags_inventory(df, var_dict=var_dict)
    _drop_like_flags = sorted(
        {
            "comp_id_missing", "year_missing", "begin_after_or_eq_end",
            "duplicate_comp_id_year", "duplicate_comp_id_year_noncanonical",
            "row_after_exit_year", "founded_after_exit"
        } | {f for f in flags_df_before["flag"].astype(str) if f.startswith("year_out_of_[")}
    )

    # BEFORE-cleaning snapshots
    miss_before = df.isna().mean().mul(100.0).sort_values(ascending=False)
    plot_missingness_top(
        miss_before,
        os.path.join(figs_dir, "missingness_top30_before_clean.png"),
        "Missingness — Top 30 columns (BEFORE cleaning)"
    )

    # Nonnegativity BEFORE cleaning (helps explain “after” being all zero)
    nonneg_before = check_nonnegative(df)
    plot_nonnegativity_audit(nonneg_before, os.path.join(figs_dir, "nonnegativity_audit_before.png"))

    plot_residual_issues(
        flags_df_before,
        os.path.join(figs_dir, "residual_issues_before_clean.png"),
        drop_like_flags=_drop_like_flags
    )
    # Full inventory BEFORE cleaning
    plot_flags_inventory(
        flags_df_before,
        os.path.join(figs_dir, "flags_inventory_before_clean.png"),
        title="All checks — top flagged (BEFORE cleaning)"
    )


    # 5) Clean (drop invalid rows, fix negatives, clip shares)
    df_clean, drop_log, fix_summary = apply_cleaning(df.copy(), flags_before, fix_nonnegatives=True)

    # 6) Flags AFTER cleaning (to prove nothing drop-eligible remains)
    flags_df_after, _ = build_flags_inventory(df_clean, var_dict=var_dict)

    # 7) QA charts (what & why)
    plot_drops_by_reason(drop_log, os.path.join(figs_dir, "drops_by_reason.png"))
    plot_fixes_by_column(fix_summary, os.path.join(figs_dir, "fixes_by_column.png"))
    plot_type_coercions(coercion_report, os.path.join(figs_dir, "type_coercions.png"))

    # NEW: if we will later save a dedup ledger CSV, produce its chart now as well
    plot_dedup_ledger(drop_log, os.path.join(figs_dir, "dedup_dropped_rows.png"), raw_df=df)

    plot_residual_issues(
        flags_df_after,
        os.path.join(figs_dir, "residual_issues_after_clean.png"),
        drop_like_flags=_drop_like_flags
    )
    # Full inventory AFTER cleaning
    plot_flags_inventory(
        flags_df_after,
        os.path.join(figs_dir, "flags_inventory_after_clean.png"),
        title="All checks — top flagged (AFTER cleaning)"
    )
    miss_after = df_clean.isna().mean().mul(100.0).sort_values(ascending=False)


    plot_missingness_top(miss_after, os.path.join(figs_dir, "missingness_top30_after_clean.png"),
                         "Missingness — Top 30 columns (AFTER cleaning)")
    plot_sales_hist_after(df_clean, os.path.join(figs_dir, "sales_log10_hist_after_clean.png"))
    artifacts_clean = compute_core_artifacts(df_clean, min_zero_sales_years, nace4_csv)
    _ = make_visuals(df_clean, artifacts_clean, out_dir=script_dir())

    # Persist artifacts (Goal 5)
    arts_dir = os.path.join(out_base_dir, "artifacts")
    ensure_dir(arts_dir)
    try:
        flags_df_before.to_csv(os.path.join(arts_dir, "flags_before.csv"), index=False)
        flags_df_after.to_csv(os.path.join(arts_dir, "flags_after.csv"), index=False)
        if drop_log is not None and not drop_log.empty:
            drop_log.to_csv(os.path.join(arts_dir, "drop_log.csv"), index=False)
            # Dedup ledger (subset)
            dd = drop_log[drop_log["reasons"].str.contains("duplicate_comp_id_year", na=False)]
            if not dd.empty:
                dd.to_csv(os.path.join(arts_dir, "dedup_dropped_rows.csv"), index=False)
        if fix_summary is not None and not fix_summary.empty:
            fix_summary.to_csv(os.path.join(arts_dir, "fix_summary.csv"), index=False)
        for name, dfobj in artifacts_clean.items():
            if isinstance(dfobj, pd.DataFrame) and not dfobj.empty:
                dfobj.to_csv(os.path.join(arts_dir, f"{name}.csv"), index=False)
    except Exception as e:
        warnings.warn(f"[WRITE] Could not persist some artifacts: {e}")

    # 8) Save ONLY the cleaned dataset next to the .py
    in_base = os.path.splitext(os.path.basename(file))[0]
    cleaned_name = f"{in_base}__CLEAN.csv"
    cleaned_path = os.path.join(out_base_dir, cleaned_name)
    if overwrite:
        cleaned_path = os.path.join(out_base_dir, os.path.basename(file))  # overwrite filename in script folder
    df_clean.to_csv(cleaned_path, index=False)

    # 9) Console summary (no warnings)
    total = len(df); dropped = len(drop_log) if drop_log is not None else 0
    print(f"[OK] Cleaned dataset saved at: {cleaned_path}")
    print(f"[OK] Rows: raw={total:,}  dropped={dropped:,}  kept={len(df_clean):,}")
    print(f"[OK] Charts written to: {figs_dir}")
    residual_drop_eligible = flags_df_after[flags_df_after["recommended_action"]=="DROP ROW"]["flagged_count"].sum()
    if residual_drop_eligible == 0:
        print("[OK] Post-clean check: no drop-eligible issues remain.")
    else:
        print("[ATTN] Post-clean residual drop-eligible issues detected (see residual_issues_after_clean.png).")

# ------------------------- CLI -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Skeptical EDA + Cleaning for Bisnode panel (cleaned-only)")
    p.add_argument("--file", required=True, help="Path to input CSV/Parquet file")
    p.add_argument("--min-zero-sales-years", type=int, default=2, help="Spell length for zero-sales analysis")
    p.add_argument("--nace4-csv", default=None, help="Optional CSV with columns {code,name} for 4-digit NACE")
    p.add_argument("--var-dict-xls", default=None, help="Optional Excel with variable->description mapping")
    p.add_argument("--overwrite", action="store_true", help="Overwrite dataset name with the input basename (dangerous)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        file=args.file,
        min_zero_sales_years=args.min_zero_sales_years,
        nace4_csv=args.nace4_csv,
        var_dict_xls=args.var_dict_xls,
        overwrite=args.overwrite
    )