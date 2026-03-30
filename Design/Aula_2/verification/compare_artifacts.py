#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from common import (
    align_and_sort_df,
    collect_in_scope_files,
    html_to_text,
    is_ml_like_path,
    load_manifest_spec,
    load_table,
    normalize_summary_json,
    read_json,
    safe_float,
    sha256_file,
    strip_survival_sections_from_html,
    utc_now_iso,
    write_json,
)


def parse_args() -> argparse.Namespace:
    default_base = Path("/Users/akatsurada/Documents/INSPER/Design/Aula_2")
    parser = argparse.ArgumentParser(description="Compare baseline and reproduced non-survival artifacts.")
    parser.add_argument("--base-dir", type=Path, default=default_base)
    parser.add_argument("--baseline-output-dir", type=Path, required=True)
    parser.add_argument("--repro-output-dir", type=Path, required=True)
    parser.add_argument("--spec-file", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def _add_row(rows: List[Dict[str, Any]], artifact: str, metric: str, baseline: Any, repro: Any, delta: Any, status: str, severity: str) -> None:
    rows.append(
        {
            "artifact": artifact,
            "metric": metric,
            "baseline": baseline,
            "repro": repro,
            "delta": delta,
            "status": status,
            "severity": severity,
        }
    )


def _compare_summary_json(
    rel_path: str,
    b_path: Path,
    r_path: Path,
    spec: Dict[str, Any],
    rows: List[Dict[str, Any]],
) -> None:
    cfg = spec.get("summary_json", {})
    exclude_prefixes = cfg.get("exclude_key_prefixes", [])
    ignore_keys = cfg.get("ignore_keys", [])

    b = normalize_summary_json(read_json(b_path), exclude_prefixes=exclude_prefixes, ignore_keys=ignore_keys)
    r = normalize_summary_json(read_json(r_path), exclude_prefixes=exclude_prefixes, ignore_keys=ignore_keys)

    keys = sorted(set(b.keys()) | set(r.keys()))
    for k in keys:
        if k not in b:
            _add_row(rows, rel_path, f"summary.{k}", None, r.get(k), None, "fail", "critical")
            continue
        if k not in r:
            _add_row(rows, rel_path, f"summary.{k}", b.get(k), None, None, "fail", "critical")
            continue

        bv = b[k]
        rv = r[k]
        if isinstance(bv, (int, float)) and isinstance(rv, (int, float)):
            diff = abs(float(bv) - float(rv))
            status = "pass" if diff <= 1e-9 else "fail"
            severity = "info" if status == "pass" else "major"
            _add_row(rows, rel_path, f"summary.{k}", bv, rv, diff, status, severity)
        else:
            status = "pass" if bv == rv else "fail"
            severity = "info" if status == "pass" else "major"
            _add_row(rows, rel_path, f"summary.{k}", bv, rv, None if status == "pass" else "different", status, severity)


def _compare_html_non_survival(rel_path: str, b_path: Path, r_path: Path, rows: List[Dict[str, Any]]) -> None:
    b_raw = b_path.read_text(encoding="utf-8", errors="ignore")
    r_raw = r_path.read_text(encoding="utf-8", errors="ignore")
    b = strip_survival_sections_from_html(b_raw)
    r = strip_survival_sections_from_html(r_raw)

    bh = sha256_file(b_path)
    rh = sha256_file(r_path)
    # Raw HTML hash is informative only; Plotly element ids/scripts can vary even
    # when analytical content is unchanged.
    _add_row(
        rows,
        rel_path,
        "file.sha256_raw",
        bh,
        rh,
        None if bh == rh else "different_nonsemantic_allowed",
        "pass",
        "info",
    )

    b_text = html_to_text(b)
    r_text = html_to_text(r)
    bn = str(abs(hash(b_text)))
    rn = str(abs(hash(r_text)))
    _add_row(
        rows,
        rel_path,
        "html.non_survival_text_hash",
        bn,
        rn,
        None if bn == rn else "different",
        "pass" if bn == rn else "fail",
        "info" if bn == rn else "major",
    )


def _compare_consolidated_json(rel_path: str, b_path: Path, r_path: Path, rows: List[Dict[str, Any]]) -> None:
    b = read_json(b_path)
    r = read_json(r_path)

    checks = [
        (
            "run_metadata.config",
            b.get("run_metadata", {}).get("config"),
            r.get("run_metadata", {}).get("config"),
        ),
        (
            "run_metadata.snapshot_ts",
            b.get("run_metadata", {}).get("snapshot_ts"),
            r.get("run_metadata", {}).get("snapshot_ts"),
        ),
        (
            "hypotheses.status_counts",
            b.get("hypotheses", {}).get("status_counts"),
            r.get("hypotheses", {}).get("status_counts"),
        ),
        (
            "causal_diagnostic_assessment.causal_claim_allowed",
            b.get("causal_diagnostic_assessment", {}).get("causal_claim_allowed"),
            r.get("causal_diagnostic_assessment", {}).get("causal_claim_allowed"),
        ),
        (
            "ml.status",
            b.get("ml", {}).get("status"),
            r.get("ml", {}).get("status"),
        ),
        (
            "temporal_backtest.status",
            b.get("temporal_backtest", {}).get("status"),
            r.get("temporal_backtest", {}).get("status"),
        ),
    ]

    for metric, bv, rv in checks:
        status = "pass" if bv == rv else "fail"
        severity = "info" if status == "pass" else "major"
        _add_row(rows, rel_path, metric, bv, rv, None if status == "pass" else "different", status, severity)


def _compare_tabular(
    rel_path: str,
    b_path: Path,
    r_path: Path,
    spec: Dict[str, Any],
    rows: List[Dict[str, Any]],
) -> None:
    tol = spec.get("tolerances", {})
    det_tol = float(tol.get("deterministic_float_abs", 1e-9))
    ml_tol = float(tol.get("ml_float_abs", 5e-3))
    ml_markers = spec.get("ml_path_markers", [])
    use_ml_tol = is_ml_like_path(rel_path, ml_markers)

    bdf = align_and_sort_df(load_table(b_path))
    rdf = align_and_sort_df(load_table(r_path))

    if rel_path.endswith("deep_dive_cluster_profiles_detailed.csv"):
        if "cluster" in bdf.columns and "cluster" in rdf.columns:
            bdf = bdf.drop(columns=["cluster"])
            rdf = rdf.drop(columns=["cluster"])

    ignore_cols: set[str] = set()
    if rel_path.endswith("hypothesis_results.csv") or rel_path.endswith("parquet/hypotheses.parquet"):
        # Narrative text can vary due tiny rounded-number differences while the
        # statistical decision/status remains identical.
        ignore_cols.update({"evidence", "interpretation"})

    _add_row(rows, rel_path, "table.rows", int(len(bdf)), int(len(rdf)), int(len(rdf) - len(bdf)), "pass" if len(bdf) == len(rdf) else "fail", "info" if len(bdf) == len(rdf) else "critical")

    bcols = list(bdf.columns)
    rcols = list(rdf.columns)
    if bcols != rcols:
        _add_row(rows, rel_path, "table.columns", bcols, rcols, "different", "fail", "critical")
        return

    if len(bdf) != len(rdf):
        return

    for c in bcols:
        if c in ignore_cols:
            continue
        bs = bdf[c]
        rs = rdf[c]

        if pd.api.types.is_numeric_dtype(bs) and pd.api.types.is_numeric_dtype(rs):
            # Force float arithmetic to avoid boolean subtraction errors in NumPy/Pandas.
            bnum = pd.to_numeric(bs, errors="coerce").astype(float)
            rnum = pd.to_numeric(rs, errors="coerce").astype(float)

            nan_mismatch = (bnum.isna() ^ rnum.isna()).sum()
            if int(nan_mismatch) > 0:
                _add_row(rows, rel_path, f"col.{c}.nan_mismatch", int(0), int(nan_mismatch), int(nan_mismatch), "fail", "major")
                continue

            valid = ~(bnum.isna() | rnum.isna())
            if int(valid.sum()) == 0:
                _add_row(rows, rel_path, f"col.{c}.all_nan", None, None, None, "pass", "info")
                continue

            diff = (bnum[valid] - rnum[valid]).abs()
            max_diff = float(diff.max())

            is_integer_like = (
                (pd.api.types.is_integer_dtype(bs) or pd.api.types.is_bool_dtype(bs))
                and (pd.api.types.is_integer_dtype(rs) or pd.api.types.is_bool_dtype(rs))
            )
            if not is_integer_like:
                b_intlike = np.all(np.isclose(bnum[valid].to_numpy(), np.round(bnum[valid].to_numpy()), atol=1e-12))
                r_intlike = np.all(np.isclose(rnum[valid].to_numpy(), np.round(rnum[valid].to_numpy()), atol=1e-12))
                is_integer_like = bool(b_intlike and r_intlike)

            if is_integer_like:
                status = "pass" if max_diff == 0.0 else "fail"
                severity = "info" if status == "pass" else "major"
                _add_row(rows, rel_path, f"col.{c}.exact_int", 0.0, max_diff, max_diff, status, severity)
            else:
                tol_use = ml_tol if use_ml_tol else det_tol
                status = "pass" if max_diff <= tol_use else "fail"
                severity = "info" if status == "pass" else "major"
                _add_row(rows, rel_path, f"col.{c}.max_abs_diff", tol_use, max_diff, max_diff, status, severity)
        else:
            btxt = bs.fillna("<NA>").astype(str)
            rtxt = rs.fillna("<NA>").astype(str)
            mismatches = int((btxt != rtxt).sum())
            status = "pass" if mismatches == 0 else "fail"
            severity = "info" if status == "pass" else "major"
            _add_row(rows, rel_path, f"col.{c}.string_mismatches", 0, mismatches, mismatches, status, severity)

    if rel_path.endswith("hypothesis_results.csv"):
        h_cols = ["hypothesis_id", "status", "decision_rule"]
        if all(c in bdf.columns for c in h_cols) and all(c in rdf.columns for c in h_cols):
            bx = bdf[h_cols].sort_values("hypothesis_id").reset_index(drop=True)
            rx = rdf[h_cols].sort_values("hypothesis_id").reset_index(drop=True)
            mismatch = int((bx != rx).any(axis=1).sum())
            status = "pass" if mismatch == 0 else "fail"
            severity = "info" if status == "pass" else "critical"
            _add_row(rows, rel_path, "hypothesis.status_decision_parity", 0, mismatch, mismatch, status, severity)


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    spec_file = (args.spec_file or (base_dir / "verification" / "spec" / "non_survival_manifest.yaml")).resolve()
    spec = load_manifest_spec(spec_file)

    baseline_output_dir = args.baseline_output_dir.resolve()
    repro_output_dir = args.repro_output_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scope = spec.get("scope", {})
    patterns = scope.get("output_patterns", [])
    exclude_contains = scope.get("exclude_path_contains", [])
    exclude_regex = scope.get("exclude_regex", [])

    b_files = collect_in_scope_files(
        root=baseline_output_dir,
        patterns=patterns,
        exclude_contains=exclude_contains,
        exclude_regex=exclude_regex,
    )
    r_files = collect_in_scope_files(
        root=repro_output_dir,
        patterns=patterns,
        exclude_contains=exclude_contains,
        exclude_regex=exclude_regex,
    )

    b_map = {str(p.relative_to(baseline_output_dir)): p for p in b_files}
    r_map = {str(p.relative_to(repro_output_dir)): p for p in r_files}

    all_paths = sorted(set(b_map.keys()) | set(r_map.keys()))

    rows: List[Dict[str, Any]] = []

    for rel in all_paths:
        bp = b_map.get(rel)
        rp = r_map.get(rel)
        if bp is None:
            _add_row(rows, rel, "file.presence", "missing", "present", "missing_baseline", "fail", "critical")
            continue
        if rp is None:
            _add_row(rows, rel, "file.presence", "present", "missing", "missing_repro", "fail", "critical")
            continue

        b_type = bp.suffix.lower()
        if rel.endswith("analise_inicial_dos_dados_summary.json"):
            _compare_summary_json(rel, bp, rp, spec, rows)
            continue
        if rel.endswith("consolidated_status.json"):
            _compare_consolidated_json(rel, bp, rp, rows)
            continue
        if rel.endswith("analise_inicial_dos_dados_interativa.html"):
            _compare_html_non_survival(rel, bp, rp, rows)
            continue
        if b_type in {".csv", ".parquet"}:
            _compare_tabular(rel, bp, rp, spec, rows)
            continue

        b_hash = sha256_file(bp)
        r_hash = sha256_file(rp)
        status = "pass" if b_hash == r_hash else "fail"
        severity = "info" if status == "pass" else "major"
        _add_row(rows, rel, "file.sha256", b_hash, r_hash, None if status == "pass" else "different", status, severity)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "artifact_diff_summary.csv", index=False)

    severity_order = {"critical": 3, "major": 2, "minor": 1, "info": 0}
    worst_row = None
    if not df.empty:
        fail_df = df[df["status"] == "fail"].copy()
        if not fail_df.empty:
            fail_df["sev_rank"] = fail_df["severity"].map(severity_order).fillna(0)
            worst_row = fail_df.sort_values("sev_rank", ascending=False).iloc[0].to_dict()

    summary = {
        "generated_at_utc": utc_now_iso(),
        "baseline_output_dir": str(baseline_output_dir),
        "repro_output_dir": str(repro_output_dir),
        "checked_artifacts": int(df["artifact"].nunique()) if not df.empty else 0,
        "checks_total": int(len(df)),
        "checks_failed": int((df["status"] == "fail").sum()) if not df.empty else 0,
        "checks_passed": int((df["status"] == "pass").sum()) if not df.empty else 0,
        "failed_by_severity": (
            df[df["status"] == "fail"]["severity"].value_counts().to_dict() if not df.empty else {}
        ),
        "worst_failure": worst_row,
        "comparison_passed": bool(((df["status"] == "fail").sum() == 0) if not df.empty else True),
    }

    write_json(out_dir / "artifact_diff_summary.json", summary)
    print(str(out_dir / "artifact_diff_summary.json"))


if __name__ == "__main__":
    main()
