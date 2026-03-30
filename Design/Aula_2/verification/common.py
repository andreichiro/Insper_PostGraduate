from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import duckdb
import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest_spec(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Spec file {path} must be JSON-compatible YAML (JSON content). Parse error: {exc}"
        ) from exc


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def relative_to(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def collect_paths_by_patterns(root: Path, patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pat in patterns:
        paths.extend(root.glob(pat))
    uniq = sorted({p.resolve() for p in paths if p.is_file()})
    return [Path(p) for p in uniq]


def _path_is_excluded(rel_path: str, exclude_contains: Sequence[str], exclude_regex: Sequence[str]) -> bool:
    lower_rel = rel_path.lower()
    for token in exclude_contains:
        if token.lower() in lower_rel:
            return True
    for pat in exclude_regex:
        if re.search(pat, rel_path):
            return True
    return False


def collect_in_scope_files(root: Path, patterns: Sequence[str], exclude_contains: Sequence[str], exclude_regex: Sequence[str]) -> List[Path]:
    candidates = collect_paths_by_patterns(root, patterns)
    out: List[Path] = []
    for p in candidates:
        rel = relative_to(p, root)
        if _path_is_excluded(rel, exclude_contains=exclude_contains, exclude_regex=exclude_regex):
            continue
        out.append(p)
    return sorted(out)


def infer_file_type(path: Path) -> str:
    sfx = path.suffix.lower()
    if sfx == ".csv":
        return "csv"
    if sfx == ".parquet":
        return "parquet"
    if sfx == ".json":
        return "json"
    if sfx == ".html":
        return "html"
    if sfx == ".md":
        return "markdown"
    if sfx in {".xlsx", ".xls"}:
        return "excel"
    return "binary"


def _table_profile_with_duckdb(path: Path) -> Optional[Dict[str, Any]]:
    file_type = infer_file_type(path)
    if file_type not in {"csv", "parquet"}:
        return None

    conn = duckdb.connect()
    p = str(path).replace("'", "''")
    if file_type == "csv":
        relation = f"read_csv_auto('{p}', header=true)"
    else:
        relation = f"read_parquet('{p}')"

    try:
        row_count = int(conn.execute(f"SELECT COUNT(*) AS n FROM {relation}").fetchone()[0])
        columns_df = conn.execute(f"DESCRIBE SELECT * FROM {relation}").fetchdf()
        schema_rows = []
        for _, row in columns_df.iterrows():
            schema_rows.append(
                {
                    "name": str(row.get("column_name")),
                    "duckdb_type": str(row.get("column_type")),
                    "null": str(row.get("null", "")),
                }
            )
        schema_fingerprint = hashlib.sha256(
            json.dumps(schema_rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {
            "row_count": row_count,
            "columns_count": int(len(schema_rows)),
            "schema": schema_rows,
            "schema_fingerprint": schema_fingerprint,
        }
    finally:
        conn.close()


def build_file_manifest(root: Path, files: Sequence[Path], include_tabular_profile: bool = True) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in files:
        st = p.stat()
        entry: Dict[str, Any] = {
            "relative_path": relative_to(p, root),
            "absolute_path": str(p.resolve()),
            "file_type": infer_file_type(p),
            "size_bytes": int(st.st_size),
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            "sha256": sha256_file(p),
        }
        if include_tabular_profile:
            profile = _table_profile_with_duckdb(p)
            if profile is not None:
                entry["tabular_profile"] = profile
        out.append(entry)
    return out


def collect_environment_manifest(python_exec: str) -> Dict[str, Any]:
    code = r"""
import importlib
import json
import platform
import sys
mods = [
    "numpy","pandas","duckdb","scipy","sklearn","matplotlib","seaborn","plotly","openpyxl","xgboost"
]
out = {
    "python_executable": sys.executable,
    "python_version": sys.version,
    "platform": platform.platform(),
    "modules": {},
}
for m in mods:
    try:
        mod = importlib.import_module(m)
        out["modules"][m] = getattr(mod, "__version__", "unknown")
    except Exception:
        out["modules"][m] = None
print(json.dumps(out, ensure_ascii=False))
"""
    proc = subprocess.run([python_exec, "-c", code], check=True, capture_output=True, text=True)
    return json.loads(proc.stdout)


def run_cmd(cmd: Sequence[str], cwd: Path, log_file: Optional[Path] = None) -> Dict[str, Any]:
    started = utc_now_iso()
    proc = subprocess.run(list(cmd), cwd=str(cwd), capture_output=True, text=True)
    ended = utc_now_iso()
    rec = {
        "cmd": list(cmd),
        "cwd": str(cwd),
        "returncode": int(proc.returncode),
        "started_at_utc": started,
        "ended_at_utc": ended,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(
            "\n".join(
                [
                    f"$ {' '.join(cmd)}",
                    "\n[stdout]\n",
                    proc.stdout,
                    "\n[stderr]\n",
                    proc.stderr,
                ]
            ),
            encoding="utf-8",
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (code={proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout[-2000:]}\n"
            f"stderr:\n{proc.stderr[-2000:]}"
        )
    return rec


def normalize_summary_json(obj: Dict[str, Any], exclude_prefixes: Sequence[str], ignore_keys: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in obj.items():
        if any(k.startswith(pref) for pref in exclude_prefixes):
            continue
        if k in ignore_keys:
            continue
        out[k] = v
    return out


def strip_survival_sections_from_html(html: str) -> str:
    patterns = [
        r"<section>\s*<h2>5\)\s*Survival recorrente de inatividade .*?</section>",
        r"<section>\s*<h2>10\)\s*Risco operacional de inatividade .*?</section>",
        r"<section>\s*<h2>11\)\s*Benchmark survival .*?</section>",
    ]
    out = html
    for pat in patterns:
        out = re.sub(pat, "", out, flags=re.S | re.I)

    # Paths and timestamps are non-semantic for this audit.
    out = re.sub(r"/Users/[^\s'\"<]+", "<ABS_PATH>", out)
    out = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)", "<ISO_TS>", out)
    return out


def html_to_text(html: str) -> str:
    text = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_ml_like_path(rel_path: str, markers: Sequence[str]) -> bool:
    low = rel_path.lower()
    return any(m.lower() in low for m in markers)


def safe_float(x: Any) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def align_and_sort_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.reindex(sorted(out.columns), axis=1)

    # Stable ordering across runs for deterministic comparisons.
    sort_cols = list(out.columns)
    for c in sort_cols:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].astype("datetime64[ns]")
    try:
        out = out.sort_values(sort_cols, kind="mergesort", na_position="first")
    except Exception:
        # Fallback for unorderable mixed object columns.
        for c in sort_cols:
            out[c] = out[c].astype(str)
        out = out.sort_values(sort_cols, kind="mergesort", na_position="first")

    return out.reset_index(drop=True)


def load_table(path: Path) -> pd.DataFrame:
    t = infer_file_type(path)
    if t == "csv":
        conn = duckdb.connect()
        try:
            p = str(path).replace("'", "''")
            return conn.execute(f"SELECT * FROM read_csv_auto('{p}', header=true)").fetchdf()
        finally:
            conn.close()
    if t == "parquet":
        conn = duckdb.connect()
        try:
            p = str(path).replace("'", "''")
            return conn.execute(f"SELECT * FROM read_parquet('{p}')").fetchdf()
        finally:
            conn.close()
    raise ValueError(f"Unsupported table type for {path}")
