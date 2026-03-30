from __future__ import annotations

import os

import sys
import re
import json
import math
import time
import csv
import shutil
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt

try:
    from wordcloud import WordCloud
except Exception as _e:
    WordCloud = None
    _WORDCLOUD_IMPORT_ERROR = _e

from pyspark.sql import SparkSession, DataFrame, functions as F, types as T
from pyspark.storagelevel import StorageLevel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF, IDFModel, Normalizer, PCA, SQLTransformer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator, ClusteringEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.functions import vector_to_array  # Spark 3.x
from pyspark.mllib.evaluation import BinaryClassificationMetrics

try:
    import torch  # type: ignore
except Exception as _e:
    torch = None  # type: ignore
    _TORCH_IMPORT_ERROR = _e

logger = logging.getLogger("amazon_nlp_pipeline")

TEST_CSV = "/Users/akatsurada/Documents/INSPER/BigData/ProjetoFinal/test.csv"
TRAIN_CSV = None 
OUTPUT_DIR = "./models_output"
FORCE_RERUN = False
RUN_MODERNBERT = True

default_profile = "score_only" if (Path(OUTPUT_DIR) / "models" / "supervised_pipeline").exists() else "dev"
os.environ.setdefault("AMAZON_NLP_RUN_PROFILE", default_profile)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Ensure Spark workers use same Python as the notebook/kernel
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["AMAZON_NLP_CSV_MULTILINE"] = "0"


# Utils
def _next_power_of_two(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << ((x - 1).bit_length())


def _rel_to_output(cfg: "Config", p: Path) -> str:
    try:
        return str(p.relative_to(Path(cfg.output_dir)))
    except Exception:
        return str(p)


def _save_fig(cfg: "Config", fig: plt.Figure, out_path: Path, *, dpi: int = 140) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=dpi)
    plt.close(fig)
    return _rel_to_output(cfg, out_path)


def _plot_bar(cfg: "Config", labels: List[str], values: List[float], *, title: str, xlabel: str, ylabel: str,
              out_path: Path, rotate: int = 0, figsize: Tuple[int, int] = (9, 4)) -> str:
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return _save_fig(cfg, fig, out_path)


def _plot_barh(cfg: "Config", labels: List[str], values: List[float], *, title: str, xlabel: str, ylabel: str,
               out_path: Path, figsize: Tuple[int, int] = (9, 6)) -> str:
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(labels))[::-1]
    ax.barh(y, list(values)[::-1])
    ax.set_yticks(y)
    ax.set_yticklabels(list(labels)[::-1])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return _save_fig(cfg, fig, out_path)


def _plot_grouped_bars(cfg: "Config", x_labels: List[str], series: Dict[str, List[float]], *,
                       title: str, xlabel: str, ylabel: str, out_path: Path,
                       rotate: int = 0, figsize: Tuple[int, int] = (10, 4)) -> str:
    fig, ax = plt.subplots(figsize=figsize)
    n = len(x_labels)
    s = len(series)
    x = np.arange(n)
    width = 0.8 / max(1, s)
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + (i - (s - 1) / 2) * width, vals, width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=rotate, ha="right" if rotate else "center")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return _save_fig(cfg, fig, out_path)


def _plot_confusion_2x2(cfg: "Config", *, tn: int, fp: int, fn: int, tp: int, title: str, out_path: Path) -> str:
    mat = np.array([[tn, fp], [fn, tp]], dtype=float)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(mat, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"])
    ax.set_yticklabels(["0", "1"])
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{int(v)}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save_fig(cfg, fig, out_path)


# Helpers p/ HTML
def _html_esc(x: Any) -> str:
    return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _html_begin(title: str) -> List[str]:
    css = """
    :root{--fg:#222;--muted:#555;--line:#e6e6e6;--bg:#fff;--note:#f9fbff;--note-b:#9bbcff;--warn:#fff9f0;--warn-b:#ffb155}
    *{box-sizing:border-box}
    body{font-family:-apple-system,system-ui,Segoe UI,Roboto,Arial,sans-serif;margin:24px;color:var(--fg);background:var(--bg)}
    h1,h2,h3{margin:1.0em 0 .5em}
    p{margin:.6em 0;line-height:1.45}
    table{border-collapse:collapse;width:100%;margin:8px 0 16px}
    th,td{border:1px solid var(--line);padding:6px 8px;font-size:13px;vertical-align:top}
    th{text-align:left;background:#f7f7f7;position:sticky;top:0}
    tbody tr:nth-child(even){background:#fbfbfb}
    .small{color:var(--muted);font-size:.9em}
    .note{background:var(--note);border-left:4px solid var(--note-b);padding:.6em .8em;margin:.6em 0}
    .warn{background:var(--warn);border-left:4px solid var(--warn-b);padding:.6em .8em;margin:.6em 0}
    img{max-width:100%;height:auto;border:1px solid #eee;box-shadow:0 1px 2px rgba(0,0,0,.05);margin:6px 0}
    code{background:#f6f6f6;padding:1px 4px;border-radius:4px}
    .kpi{display:inline-block;margin:.2em .6em .2em 0;padding:.2em .6em;background:#f5f7ff;border:1px solid #dfe6ff;border-radius:6px}
    .hr{height:1px;background:var(--line);margin:16px 0}
    """
    out = []
    out.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    out.append(f"<title>{_html_esc(title)}</title>")
    out.append(f"<style>{css}</style></head><body>")
    return out


def _html_end(html: List[str]) -> None:
    html.append("</body></html>")


def _html_h1(html: List[str], txt: str) -> None:
    html.append(f"<h1>{_html_esc(txt)}</h1>")


def _html_h2(html: List[str], txt: str) -> None:
    html.append(f"<h2>{_html_esc(txt)}</h2>")


def _html_h3(html: List[str], txt: str) -> None:
    html.append(f"<h3>{_html_esc(txt)}</h3>")


def _html_p(html: List[str], txt: str) -> None:
    html.append(f"<p>{txt}</p>")


def _html_note(html: List[str], txt: str) -> None:
    html.append(f"<p class='note'>{txt}</p>")


def _html_warn(html: List[str], txt: str) -> None:
    html.append(f"<p class='warn'>{txt}</p>")


def _html_img(html: List[str], src: str, caption: Optional[str] = None) -> None:
    cap = f"<b>{_html_esc(caption)}</b><br/>" if caption else ""
    html.append(f"<p>{cap}<img src='{_html_esc(src)}' alt='img'></p>")


def _html_table(html: List[str], rows: List[Dict[str, Any]], caption: Optional[str] = None, max_rows: int = 50) -> None:
    if caption:
        html.append(f"<h3>{_html_esc(caption)}</h3>")
    if not rows:
        html.append("<p class='small'>No rows.</p>")
        return
    rows = rows[:max_rows]
    cols = list(rows[0].keys())
    html.append("<table><thead><tr>" + "".join(f"<th>{_html_esc(c)}</th>" for c in cols) + "</tr></thead><tbody>")
    for r in rows:
        html.append("<tr>" + "".join(f"<td>{_html_esc(r.get(c,''))}</td>" for c in cols) + "</tr>")
    html.append("</tbody></table>")

    def top_tokens(where_sentiment: Optional[int]) -> List[Dict[str, Any]]:
        d0 = cleaned if where_sentiment is None else cleaned.where(F.col("sentiment") == F.lit(where_sentiment))
        rows = d0.groupBy("token").count().orderBy(F.desc("count")).limit(int(cfg.eda_top_tokens)).collect()
        return [{"token": r["token"], "count": int(r["count"])} for r in rows]

    top_all = top_tokens(None)
    top_pos = top_tokens(1)
    top_neg = top_tokens(0)

    # Top-token bar charts (no Spark work: uses collected lists)
    def plot_top(lst: List[Dict[str, Any]], name: str, title: str) -> Optional[str]:
        if not lst:
            return None
        head = lst[:20]
        labels = [d["token"] for d in head]
        values = [int(d["count"]) for d in head]
        return _plot_barh(
            cfg, labels, values, title=title, xlabel="count", ylabel="token",
            out_path=eda_dir / name
        )

    files["top_tokens_all"] = plot_top(top_all, "top_tokens_all.png", "Top tokens (all)")
    files["top_tokens_pos"] = plot_top(top_pos, "top_tokens_positive.png", "Top tokens (sentiment=1)")
    files["top_tokens_neg"] = plot_top(top_neg, "top_tokens_negative.png", "Top tokens (sentiment=0)")

    # Wordclouds
    def make_wordcloud(freq: Dict[str, int], fname: str) -> Optional[str]:
        if not freq:
            return None
        if WordCloud is None:
            logger.warning("wordcloud not installed; skipping %s (error=%s)", fname, str(_WORDCLOUD_IMPORT_ERROR))
            return None
        wc = WordCloud(width=1200, height=600, background_color="white", max_words=int(cfg.wordcloud_max_words))
        wc = wc.generate_from_frequencies(freq)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        return _save_fig(cfg, fig, eda_dir / fname)

    files["wordcloud_all"] = make_wordcloud({d["token"]: d["count"] for d in top_all}, "wordcloud_all.png")
    files["wordcloud_pos"] = make_wordcloud({d["token"]: d["count"] for d in top_pos}, "wordcloud_positive.png")
    files["wordcloud_neg"] = make_wordcloud({d["token"]: d["count"] for d in top_neg}, "wordcloud_negative.png")

    cleaned.unpersist()

    out = {
        "generated_at": _utc_now_iso(),
        "rows": int(total_rows),
        "text_len_quantiles": {"p50": p50, "p90": p90, "p95": p95, "p99": p99, "bin_width": int(bin_width)},
        "token_stats": {"sample_frac_used": float(frac), "sample_cap_rows": int(target_sample_rows), "approx_vocab": int(approx_vocab)},
        "files": files,
    }
    _write_json(Path(cfg.output_dir) / "metrics" / "eda_advanced.json", out)
    return out

def _binary_metrics_from_pred(pred_df: DataFrame) -> Dict[str, Any]:
    # agregação
    rows = (
        pred_df.select(F.col("label").cast("int").alias("label"), F.col("prediction").cast("int").alias("pred"))
        .groupBy("label", "pred")
        .count()
        .collect()
    )

    m = {(int(r["label"]), int(r["pred"])): int(r["count"]) for r in rows}
    tn = m.get((0, 0), 0)
    fp = m.get((0, 1), 0)
    fn = m.get((1, 0), 0)
    tp = m.get((1, 1), 0)
    n = tn + fp + fn + tp

    def _safe_div(a: float, b: float) -> float:
        return float(a / b) if b else float("nan")

    acc = _safe_div(tp + tn, n)
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    spec = _safe_div(tn, tn + fp)
    f1 = float("nan") if not (prec == prec and rec == rec and (prec + rec) > 0) else float(2 * prec * rec / (prec + rec))
    bal_acc = float("nan") if not (rec == rec and spec == spec) else float((rec + spec) / 2.0)

    return {
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "n": n},
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": f1,
        "balanced_accuracy": bal_acc,
    }

def _save_fig_basic(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(path), dpi=140)
    plt.close(fig)
    return path.name


def plot_bar(labels, values, *, title: str, xlabel: str, ylabel: str, out_path: Path) -> str:
    labels = [str(x) for x in labels]
    values = [float(v) for v in values]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(labels))
    ax.bar(x, values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    return _save_fig_basic(fig, out_path)


def plot_line(xs, ys, *, title: str, xlabel: str, ylabel: str, out_path: Path, vline: Optional[float] = None) -> str:
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    fig, ax = plt.subplots(figsize=(8, 4))
    marker = "o" if len(xs) <= 60 else None
    ax.plot(xs, ys, marker=marker)
    if vline is not None:
        ax.axvline(float(vline), linestyle="--")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return _save_fig_basic(fig, out_path)


def plot_heatmap(mat, xlabels, ylabels, *, title: str, out_path: Path) -> str:
    mat = np.asarray(mat, dtype=float)
    xlabels = [str(x) for x in xlabels]
    ylabels = [str(y) for y in ylabels]
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(mat, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=30, ha="right")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    fig.colorbar(im, ax=ax)
    return _save_fig_basic(fig, out_path)


def plot_confusion_2x2(cm, *, title: str, out_path: Path) -> str:
    # cm = [[tn, fp], [fn, tp]]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, aspect="auto")
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i][j])), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    return _save_fig_basic(fig, out_path)


def plot_two_lines(
    xs,
    ys1,
    ys2,
    *,
    label1: str,
    label2: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> str:
    xs = [float(x) for x in xs]
    ys1 = [float(y) for y in ys1]
    ys2 = [float(y) for y in ys2]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, ys1, marker="o" if len(xs) <= 60 else None, label=label1)
    ax.plot(xs, ys2, marker="o" if len(xs) <= 60 else None, label=label2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return _save_fig_basic(fig, out_path)


# Utilities
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or not str(v).strip() else str(v).strip()


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return default
    try:
        return int(str(v).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return default
    try:
        return float(str(v).strip())
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    return default


def setup_logging(output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / "pipeline.log"
    if getattr(setup_logging, "_configured", False):
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(str(log_path), mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    setup_logging._configured = True 


def _best_torch_device() -> str:
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _maybe_call(obj: Any, attr: str) -> Any:
    v = getattr(obj, attr, None)
    if v is None:
        return None
    return v() if callable(v) else v


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _next_pow2(n: int) -> int:
    n = max(1, int(n))
    return 1 << (n - 1).bit_length()


def pick_seq_len(p: int, allowed=(32, 64, 128, 256, 512), cap: int = 256) -> int:
    allowed = [a for a in allowed if a <= cap]
    for a in allowed:
        if a >= int(p):
            return int(a)
    return int(allowed[-1]) if allowed else int(cap)


_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kKmMgGtT]?)\s*$")


def _parse_spark_size_bytes(s: str) -> int:
    if s is None:
        return 0
    s = str(s).strip().lower()
    if s == "0":
        return 0
    m = _SIZE_RE.match(s.replace("b", ""))
    if not m:
        return 0
    val = float(m.group(1))
    unit = m.group(2)
    mult = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}.get(unit, 1)
    return int(val * mult)


def _prev_power_of_two(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x.bit_length() - 1)


def _safe_pca_input_dim(spark: SparkSession, requested_dim: int, requested_k: int) -> int:
    max_res = _parse_spark_size_bytes(spark.conf.get("spark.driver.maxResultSize", "1g"))
    drv_mem = _parse_spark_size_bytes(spark.conf.get("spark.driver.memory", "4g"))

    cap = int(drv_mem * 0.60) if drv_mem > 0 else 0
    if max_res > 0:
        cap = min(cap, int(max_res * 0.60)) if cap > 0 else int(max_res * 0.60)

    if cap <= 0:
        return max(requested_dim, requested_k)

    def tri_bytes(d: int) -> int:
        # Rough triangular covariance buffer size estimate (float32-ish, conservative).
        return 4 * d * (d + 1)

    if tri_bytes(requested_dim) <= cap:
        return max(requested_dim, requested_k)

    B = cap // 4
    disc = 1 + 4 * B
    d_max = int((math.isqrt(disc) - 1) // 2)

    d_max = max(d_max, requested_k)
    d_safe = min(requested_dim, d_max)
    d_safe = _prev_power_of_two(d_safe)
    return max(d_safe, requested_k)


# =========================
# Config
# =========================
@dataclass
class Config:
    polarity_test_path: str
    polarity_train_path: Optional[str] = None
    output_dir: str = "./models_output"

    run_profile: str = "dev"  # dev | train_full | score_only

    # CSV ingestion
    csv_multiline: bool = True
    enable_csv_sharding: bool = False
    csv_shard_rows: int = 200_000
    enable_parquet_cache: bool = False

    # caps
    max_polarity_rows: int = 100_000
    rf_max_train_rows: int = 30_000
    embedding_max_rows: int = 10_000
    torch_max_rows: int = 10_000

    # features
    hashing_num_features: int = 1 << 15
    cluster_hashing_num_features: int = 1 << 12

    # split + CV
    deterministic_split_mod: int = 10_000
    train_split_fraction: float = 0.8
    random_state: int = 42
    cv_folds: int = 3
    cv_parallelism: int = 4

    logreg_reg_params: Tuple[float, ...] = (0.01, 0.1)
    logreg_l1_ratios: Tuple[float, ...] = (0.0, 0.5)

    enable_rf: bool = True
    rf_num_trees: Tuple[int, ...] = (50, 100)
    rf_max_depths: Tuple[int, ...] = (10, 20)
    rf_max_bins: int = 64
    rf_subsampling_rate: float = 0.8
    rf_feature_subset_strategy: str = "sqrt"

    # clustering
    enable_clustering: bool = True
    embedding_pca_k: int = 256 
    embedding_pca_k_max: int = 256
    pca_var_target: float = 0.90
    kmeans_k: int = 20 
    kmeans_k_min: int = 2
    kmeans_k_max: int = 30
    kmeans_max_iter: int = 30

    # ModernBERT
    enable_modernbert: bool = True
    modernbert_model_name: str = "answerdotai/ModernBERT-base"
    modernbert_max_seq_len: int = 64
    modernbert_batch_size: int = 64
    modernbert_epochs: int = 1
    modernbert_lr: float = 2e-5

    # EDA
    enable_eda: bool = True
    eda_sample_frac: float = 0.05
    wordcloud_max_words: int = 200
    eda_top_tokens: int = 1000
    hist_target_bins: int = 40
    html_report_name: str = "amazon_reviews_report.html"

    # evaluation
    eval_curve_cap_rows: int = 200_000
    calibration_bins: int = 10

    # spark
    spark_app_name: str = "AmazonNLP"
    spark_master: str = "local[*]"
    spark_driver_memory: str = "8g"
    spark_shuffle_partitions: int = 0
    spark_default_parallelism: int = 0
    enable_aqe: bool = True

    def ensure_output_dirs(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        for sub in ("models", "torch_models", "metrics", "artifacts", "eda", "predictions"):
            Path(self.output_dir, sub).mkdir(parents=True, exist_ok=True)

    @property
    def parquet_cache_root(self) -> Path:
        return Path(self.output_dir) / "artifacts" / "parquet_cache"

    @property
    def csv_shards_root(self) -> Path:
        return Path(self.output_dir) / "artifacts" / "csv_shards"


def tune_config(cfg: Config) -> None:
    cfg.run_profile = _env_str("AMAZON_NLP_RUN_PROFILE", cfg.run_profile).lower()
    if cfg.run_profile not in ("dev", "train_full", "score_only"):
        logger.warning("Unknown run_profile=%s; defaulting to dev.", cfg.run_profile)
        cfg.run_profile = "dev"

    cfg.csv_multiline = _env_bool("AMAZON_NLP_CSV_MULTILINE", cfg.csv_multiline)
    cfg.enable_csv_sharding = _env_bool("AMAZON_NLP_ENABLE_CSV_SHARDING", cfg.enable_csv_sharding)
    cfg.csv_shard_rows = _env_int("AMAZON_NLP_CSV_SHARD_ROWS", cfg.csv_shard_rows)

    cores = os.cpu_count() or 4
    if cfg.spark_default_parallelism <= 0:
        cfg.spark_default_parallelism = _env_int("AMAZON_NLP_DEFAULT_PARALLELISM", max(2, cores))
    if cfg.spark_shuffle_partitions <= 0:
        default_shuffle = min(64, max(16, int(cfg.spark_default_parallelism) * 2))
        cfg.spark_shuffle_partitions = _env_int("AMAZON_NLP_SHUFFLE_PARTITIONS", default_shuffle)

    if "AMAZON_NLP_MAX_POLARITY_ROWS" in os.environ:
        cfg.max_polarity_rows = _env_int("AMAZON_NLP_MAX_POLARITY_ROWS", cfg.max_polarity_rows)
    else:
        cfg.max_polarity_rows = cfg.max_polarity_rows if cfg.run_profile == "dev" else 0

    cfg.rf_max_train_rows = _env_int("AMAZON_NLP_RF_MAX_TRAIN_ROWS", cfg.rf_max_train_rows)
    cfg.embedding_max_rows = _env_int("AMAZON_NLP_EMBED_MAX_ROWS", cfg.embedding_max_rows)
    cfg.torch_max_rows = _env_int("AMAZON_NLP_TORCH_MAX_ROWS", cfg.torch_max_rows)

    cfg.cv_parallelism = _env_int("AMAZON_NLP_CV_PARALLELISM", cfg.cv_parallelism)
    if str(cfg.spark_master).startswith("local"):
        cfg.cv_parallelism = max(1, min(int(cfg.cv_parallelism), 4))

    cfg.enable_parquet_cache = _env_bool(
        "AMAZON_NLP_ENABLE_PARQUET_CACHE",
        default=(cfg.run_profile in ("train_full", "score_only")),
    )

    # Params adicionais
    cfg.pca_var_target = _env_float("AMAZON_NLP_PCA_VAR_TARGET", cfg.pca_var_target)
    cfg.kmeans_k_max = _env_int("AMAZON_NLP_KMEANS_K_MAX", cfg.kmeans_k_max)
    cfg.eval_curve_cap_rows = _env_int("AMAZON_NLP_EVAL_CURVE_CAP_ROWS", cfg.eval_curve_cap_rows)

    if cfg.run_profile in ("train_full", "score_only"):
        cfg.enable_eda = False
        cfg.enable_clustering = False
        cfg.enable_modernbert = False
        cfg.enable_rf = False
        cfg.cv_parallelism = 1

    logger.info(
        "Tune: profile=%s | cores=%d | parallelism=%d | shuffle=%d | cv_parallelism=%d | "
        "max_rows=%d | parquet_cache=%s | csv_multiline=%s | csv_sharding=%s",
        cfg.run_profile,
        cores,
        cfg.spark_default_parallelism,
        cfg.spark_shuffle_partitions,
        cfg.cv_parallelism,
        cfg.max_polarity_rows,
        cfg.enable_parquet_cache,
        cfg.csv_multiline,
        cfg.enable_csv_sharding,
    )


# Spark session
def _stop_existing_local_spark_if_any() -> None:
    try:
        active = SparkSession.getActiveSession()
        if active is not None:
            active.stop()
    except Exception:
        pass
    try:
        from pyspark import SparkContext  # type: ignore

        sc = getattr(SparkContext, "_active_spark_context", None)
        if sc is not None:
            sc.stop()
    except Exception:
        pass


def create_spark(cfg: Config) -> SparkSession:
    tune_config(cfg)
    if str(cfg.spark_master).startswith("local"):
        _stop_existing_local_spark_if_any()
        os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

    py = sys.executable
    builder = (
        SparkSession.builder.appName(cfg.spark_app_name)
        .master(cfg.spark_master)
        .config("spark.driver.memory", cfg.spark_driver_memory)
        .config("spark.default.parallelism", str(int(cfg.spark_default_parallelism)))
        .config("spark.sql.shuffle.partitions", str(int(cfg.spark_shuffle_partitions)))
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.python.worker.reuse", "true")
        .config("spark.ui.enabled", "false")
        .config("spark.pyspark.python", py)
        .config("spark.pyspark.driver.python", py)
        .config("spark.sql.files.maxPartitionBytes", str(64 * 1024 * 1024))
        .config("spark.sql.files.openCostInBytes", str(4 * 1024 * 1024))
    )
    if cfg.enable_aqe:
        builder = builder.config("spark.sql.adaptive.enabled", "true").config(
            "spark.sql.adaptive.coalescePartitions.enabled", "true"
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


@contextmanager
def spark_session(cfg: Config):
    spark = create_spark(cfg)
    try:
        yield spark
    finally:
        try:
            spark.stop()
        except Exception:
            pass


# =========================
# Input ingestion: CSV sharding + Parquet cache
# =========================
def _polarity_schema() -> T.StructType:
    return T.StructType(
        [
            T.StructField("polarity", T.IntegerType(), nullable=False),
            T.StructField("title", T.StringType(), nullable=True),
            T.StructField("text", T.StringType(), nullable=True),
        ]
    )


def _csv_signature(path: str) -> str:
    try:
        p = Path(path)
        st = p.stat()
        return f"{p.resolve()}|size={st.st_size}|mtime={int(st.st_mtime)}"
    except Exception:
        return f"{path}|unknown"


def shard_csv_multiline_safe(src_path: str, dst_dir: str, *, rows_per_shard: int) -> str:
    src = Path(src_path)
    out_dir = Path(dst_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    marker = out_dir / "_SUCCESS"
    if marker.exists():
        return str(out_dir)

    meta_path = out_dir / "_meta.json"
    sig = _csv_signature(src_path)
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("src_signature") == sig and meta.get("rows_per_shard") == int(rows_per_shard):
                if any(out_dir.glob("part-*.csv")):
                    marker.write_text("ok", encoding="utf-8")
                    return str(out_dir)
        except Exception:
            pass

    logger.info("Sharding multiline CSV: %s -> %s (rows_per_shard=%d)", src_path, dst_dir, int(rows_per_shard))

    try:
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    except Exception:
        pass

    part = 0
    rows_in_part = 0
    total = 0

    def _open_writer(part_idx: int):
        fpath = out_dir / f"part-{part_idx:05d}.csv"
        f = fpath.open("w", encoding="utf-8", newline="")
        w = csv.writer(
            f,
            delimiter=",",
            quotechar='"',
            doublequote=True,
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        return f, w

    cur_f, cur_w = _open_writer(part)
    try:
        with src.open("r", encoding="utf-8", newline="") as fin:
            reader = csv.reader(fin, delimiter=",", quotechar='"', doublequote=True)
            for row in reader:
                if len(row) < 3:
                    row = (row + [""] * 3)[:3]
                elif len(row) > 3:
                    row = [row[0], row[1], ",".join(row[2:])]

                cur_w.writerow(row)
                rows_in_part += 1
                total += 1

                if rows_in_part >= rows_per_shard:
                    cur_f.close()
                    part += 1
                    rows_in_part = 0
                    cur_f, cur_w = _open_writer(part)

        cur_f.close()
    finally:
        try:
            cur_f.close()
        except Exception:
            pass

    _write_json(meta_path, {"src_signature": sig, "rows_per_shard": int(rows_per_shard), "rows": int(total)})
    marker.write_text("ok", encoding="utf-8")
    logger.info("Sharding done: %d rows across %d shard files", int(total), int(part) + 1)
    return str(out_dir)


def _maybe_shard_csv_for_multiline(cfg: Config, csv_path: str, split_name: str) -> str:
    p = Path(csv_path)
    if not cfg.csv_multiline or not cfg.enable_csv_sharding:
        return csv_path
    if not p.exists() or not p.is_file():
        return csv_path

    sig = _csv_signature(csv_path)
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", p.stem)[:80]
    shard_dir = cfg.csv_shards_root / split_name / f"{safe_name}_{abs(hash(sig)) % (10**10)}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    return shard_csv_multiline_safe(str(p), str(shard_dir), rows_per_shard=int(cfg.csv_shard_rows))


def _read_polarity_csv_raw(spark: SparkSession, csv_path: str, cfg: Config) -> DataFrame:
    return (
        spark.read.option("multiLine", str(bool(cfg.csv_multiline)).lower())
        .option("quote", '"')
        .option("escape", '"')
        .csv(csv_path, header=False, schema=_polarity_schema())
    )


def _transform_raw_to_dataset(df_raw: DataFrame, split_name: str) -> DataFrame:
    df = df_raw.withColumn("dataset_split", F.lit(split_name))
    df = (
        df.withColumn(
            "sentiment",
            F.when(F.col("polarity") == 1, F.lit(0)).when(F.col("polarity") == 2, F.lit(1)).otherwise(F.lit(None).cast("int")),
        )
        .drop("polarity")
        .where(F.col("sentiment").isNotNull())
    )
    df = df.withColumn(
        "text_full",
        F.concat_ws(" ", F.coalesce(F.col("title"), F.lit("")), F.coalesce(F.col("text"), F.lit(""))),
    )
    return df.select("dataset_split", "sentiment", "text_full")


def _parquet_cache_dir(cfg: Config, split_name: str) -> Path:
    return cfg.parquet_cache_root / split_name


def _parquet_meta_path(cache_dir: Path) -> Path:
    return cache_dir / "_meta.json"


def _parquet_cache_valid(cache_dir: Path, *, src_signature: str, multiline: bool) -> bool:
    meta_path = _parquet_meta_path(cache_dir)
    if not cache_dir.exists() or not any(cache_dir.glob("*.parquet")):
        return False
    if not meta_path.exists():
        return True
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta.get("src_signature") == src_signature and bool(meta.get("csv_multiline", True)) == bool(multiline)
    except Exception:
        return False


def load_split_dataset(spark: SparkSession, cfg: Config, *, path: str, split_name: str) -> DataFrame:
    if not path:
        raise ValueError(f"Empty path for split={split_name}")

    cache_dir = _parquet_cache_dir(cfg, split_name)
    src_sig = _csv_signature(path)

    if cfg.enable_parquet_cache and _parquet_cache_valid(cache_dir, src_signature=src_sig, multiline=cfg.csv_multiline):
        logger.info("Loading %s from Parquet cache: %s", split_name, str(cache_dir))
        return spark.read.parquet(str(cache_dir))

    read_path = _maybe_shard_csv_for_multiline(cfg, path, split_name)
    df_raw = _read_polarity_csv_raw(spark, read_path, cfg)
    df = _transform_raw_to_dataset(df_raw, split_name)

    if cfg.enable_parquet_cache:
        logger.info("Writing Parquet cache for %s -> %s", split_name, str(cache_dir))
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.write.mode("overwrite").parquet(str(cache_dir))
        _write_json(
            _parquet_meta_path(cache_dir),
            {"src_signature": src_sig, "csv_multiline": bool(cfg.csv_multiline), "written_at": _utc_now_iso()},
        )
        return spark.read.parquet(str(cache_dir))

    return df


def load_polarity_dataset(spark: SparkSession, cfg: Config) -> DataFrame:
    test_df = load_split_dataset(spark, cfg, path=cfg.polarity_test_path, split_name="test")

    if cfg.polarity_train_path:
        train_df = load_split_dataset(spark, cfg, path=cfg.polarity_train_path, split_name="train")

        # IMPORTANT: cap per split (otherwise train dominates and test disappears)
        if cfg.max_polarity_rows and cfg.max_polarity_rows > 0:
            max_rows = int(cfg.max_polarity_rows)

            train_cap = int(max_rows * float(cfg.train_split_fraction))
            train_cap = max(0, min(train_cap, max_rows))
            test_cap = max_rows - train_cap

            # keep both splits non-empty when possible
            if max_rows >= 2:
                if train_cap == 0:
                    train_cap, test_cap = 1, max_rows - 1
                elif test_cap == 0:
                    test_cap, train_cap = 1, max_rows - 1

            if train_cap > 0:
                train_df = train_df.limit(int(train_cap))
            if test_cap > 0:
                test_df = test_df.limit(int(test_cap))

        df = train_df.unionByName(test_df)
    else:
        df = test_df
        if cfg.max_polarity_rows and cfg.max_polarity_rows > 0:
            df = df.limit(int(cfg.max_polarity_rows))

    df = df.withColumn("review_id", F.monotonically_increasing_id())
    return df.select("review_id", "dataset_split", "sentiment", "text_full")

def _maybe_fix_underpartitioning_dev(df: DataFrame, cfg: Config) -> DataFrame:
    parts = int(min(max(2, cfg.spark_default_parallelism), 64))
    try:
        cur = int(df.rdd.getNumPartitions())
        if cur < parts:
            df = df.repartition(parts)
    except Exception:
        df = df.repartition(parts)
    return df


# EDA
def run_basic_eda(df: DataFrame, cfg: Config, total_rows: int) -> Dict[str, Any]:
    eda_dir = Path(cfg.output_dir) / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    by_split = df.groupBy("dataset_split").count().orderBy("dataset_split").collect()
    by_sent = df.groupBy("sentiment").count().orderBy("sentiment").collect()

    split_labels = [r["dataset_split"] for r in by_split]
    split_counts = [int(r["count"]) for r in by_split]
    sent_labels = [str(int(r["sentiment"])) for r in by_sent]
    sent_counts = [int(r["count"]) for r in by_sent]

    f_split = plot_bar(
        split_labels,
        split_counts,
        title="Rows by dataset split",
        xlabel="Split",
        ylabel="Rows",
        out_path=eda_dir / "eda_split_counts.png",
    )
    f_sent = plot_bar(
        sent_labels,
        sent_counts,
        title="Class balance (sentiment)",
        xlabel="Sentiment",
        ylabel="Rows",
        out_path=eda_dir / "eda_sentiment_counts.png",
    )

    miss_row = (
        df.select(
            *[F.mean(F.col(c).isNull().cast("double")).alias(c) for c in df.columns],
            F.mean((F.length(F.trim(F.coalesce(F.col("text_full"), F.lit("")))) == 0).cast("double")).alias("text_empty_rate"),
        )
        .collect()[0]
        .asDict(True)
    )
    miss = {k: float(v) if v is not None else 0.0 for k, v in miss_row.items()}
    miss_items = sorted(miss.items(), key=lambda x: x[1], reverse=True)

    miss_labels = [k for k, _ in miss_items]
    miss_vals = [v for _, v in miss_items]
    f_miss = plot_bar(
        miss_labels,
        miss_vals,
        title="Missing/empty rates",
        xlabel="Column",
        ylabel="Rate",
        out_path=eda_dir / "eda_missing_rates.png",
    )

    out = {
        "generated_at": _utc_now_iso(),
        "total_rows": int(total_rows),
        "by_dataset_split": [r.asDict(True) for r in by_split],
        "by_sentiment": [r.asDict(True) for r in by_sent],
        "missing_rates": miss,
        "plots": {"split_counts": f_split, "sentiment_counts": f_sent, "missing_rates": f_miss},
    }
    _write_json(Path(cfg.output_dir) / "metrics" / "eda_summary.json", out)
    return out


def run_advanced_eda(df: DataFrame, cfg: Config, total_rows: int) -> Dict[str, Any]:
    eda_dir = Path(cfg.output_dir) / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    df_len = df.withColumn("text_len_tokens", F.size(F.split(F.col("text_full"), r"\s+")))
    qs = (
        df_len.agg(
            F.expr(
                "percentile_approx(text_len_tokens, array(0.5, 0.9, 0.95, 0.99), 10000)"
            ).alias("qs")
        )
        .collect()[0]["qs"]
    )
    p50, p90, p95, p99 = [int(x) for x in qs] if qs else [0, 0, 0, 0]

    bin_width = max(1, int(math.ceil(max(1, p99) / float(max(10, cfg.hist_target_bins)))))
    binned = (
        df_len.select((F.floor(F.col("text_len_tokens") / F.lit(bin_width)) * F.lit(bin_width)).alias("bin_start"))
        .groupBy("bin_start")
        .count()
        .orderBy("bin_start")
    )
    rows = binned.collect()
    x = [int(r["bin_start"]) for r in rows]
    y = [int(r["count"]) for r in rows]

    suggested_seq_len = pick_seq_len(p95, cap=256)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x, y, width=bin_width)
    if p90:
        ax.axvline(p90, linestyle="--")
    if p95:
        ax.axvline(p95, linestyle="--")
    if suggested_seq_len:
        ax.axvline(suggested_seq_len, linestyle="--")
    ax.set_title("Text length distribution (tokens)")
    ax.set_xlabel("Tokens (binned)")
    ax.set_ylabel("Count")
    hist_file = _save_fig_basic(fig, eda_dir / "eda_text_len_hist.png")

    # token EDA 
    sample_target = min(200_000, max(20_000, int(float(cfg.eda_sample_frac) * float(total_rows))))
    sample_frac = min(1.0, float(sample_target) / float(max(1, total_rows)))

    sample = (
        df.select("text_full", "sentiment")
        .where(F.col("text_full").isNotNull())
        .sample(False, sample_frac, seed=int(cfg.random_state))
        .limit(int(sample_target))
    )

    tok = RegexTokenizer(inputCol="text_full", outputCol="tokens", pattern=r"\W+", toLowercase=True)
    sw = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", locale="en_US")
    cleaned = (
        sw.transform(tok.transform(sample))
        .select("sentiment", F.explode("filtered_tokens").alias("token"))
        .select("sentiment", F.lower(F.col("token")).alias("token"))
        .where(F.length("token") > 2)
        .where(F.col("token").rlike("^[a-zA-Z]+$"))
    )

    vocab_est = int(cleaned.agg(F.approx_count_distinct("token").alias("v")).collect()[0]["v"] or 0)
    suggested_hash_dim = _next_pow2(min(max(2 * vocab_est, 2**14), 2**18)) if vocab_est > 0 else int(cfg.hashing_num_features)

    def top_tokens(where_sentiment: Optional[int], n: int) -> List[Tuple[str, int]]:
        d0 = cleaned if where_sentiment is None else cleaned.where(F.col("sentiment") == F.lit(where_sentiment))
        rows0 = d0.groupBy("token").count().orderBy(F.desc("count")).limit(int(n)).collect()
        return [(r["token"], int(r["count"])) for r in rows0]

    top_n = int(max(50, cfg.eda_top_tokens))
    top_all_full = top_tokens(None, top_n)
    top_pos_full = top_tokens(1, top_n)
    top_neg_full = top_tokens(0, top_n)

    top_all_plot = top_all_full[:30]
    top_pos_plot = top_pos_full[:30]
    top_neg_plot = top_neg_full[:30]

    f_top_all = plot_bar(
        [t for t, _ in top_all_plot],
        [c for _, c in top_all_plot],
        title="Top tokens (all)",
        xlabel="Token",
        ylabel="Count",
        out_path=eda_dir / "eda_top_tokens_all.png",
    )
    f_top_pos = plot_bar(
        [t for t, _ in top_pos_plot],
        [c for _, c in top_pos_plot],
        title="Top tokens (positive)",
        xlabel="Token",
        ylabel="Count",
        out_path=eda_dir / "eda_top_tokens_pos.png",
    )
    f_top_neg = plot_bar(
        [t for t, _ in top_neg_plot],
        [c for _, c in top_neg_plot],
        title="Top tokens (negative)",
        xlabel="Token",
        ylabel="Count",
        out_path=eda_dir / "eda_top_tokens_neg.png",
    )

    wc_all = wc_pos = wc_neg = None
    if WordCloud is not None:
        def make_wordcloud(freq: Dict[str, int], fname: str) -> Optional[str]:
            if not freq:
                return None
            wc = WordCloud(width=1200, height=600, background_color="white", max_words=int(cfg.wordcloud_max_words))
            wc = wc.generate_from_frequencies(freq)
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.imshow(wc, interpolation="bilinear")
            ax2.axis("off")
            return _save_fig_basic(fig2, eda_dir / fname)

        wc_all = make_wordcloud(dict(top_all_full), "eda_wordcloud_all.png")
        wc_pos = make_wordcloud(dict(top_pos_full), "eda_wordcloud_pos.png")
        wc_neg = make_wordcloud(dict(top_neg_full), "eda_wordcloud_neg.png")
    else:
        logger.warning("wordcloud not installed; skipping wordclouds (error=%s)", str(getattr(globals(), "_WORDCLOUD_IMPORT_ERROR", "")))

    out = {
        "generated_at": _utc_now_iso(),
        "rows": int(total_rows),
        "quantiles_text_len_tokens": {"p50": p50, "p90": p90, "p95": p95, "p99": p99},
        "vocab_est_sample": int(vocab_est),
        "suggested": {
            "modernbert_max_seq_len": int(suggested_seq_len),
            "hashing_num_features": int(suggested_hash_dim),
            "eda_sample_target": int(sample_target),
        },
        "plots": {
            "text_len_hist": hist_file,
            "top_tokens_all": f_top_all,
            "top_tokens_pos": f_top_pos,
            "top_tokens_neg": f_top_neg,
            "wordcloud_all": wc_all,
            "wordcloud_pos": wc_pos,
            "wordcloud_neg": wc_neg,
        },
    }
    _write_json(Path(cfg.output_dir) / "metrics" / "eda_advanced.json", out)
    return out


def apply_data_driven_config(cfg: Config, eda_adv: Dict[str, Any]) -> None:
    sug = (eda_adv or {}).get("suggested") or {}

    if sug.get("modernbert_max_seq_len"):
        cfg.modernbert_max_seq_len = int(sug["modernbert_max_seq_len"])
    if sug.get("hashing_num_features"):
        cfg.hashing_num_features = int(sug["hashing_num_features"])

    decisions = {
        "applied_at": _utc_now_iso(),
        "modernbert_max_seq_len": int(cfg.modernbert_max_seq_len),
        "hashing_num_features": int(cfg.hashing_num_features),
    }
    _write_json(Path(cfg.output_dir) / "metrics" / "eda_decisions_applied.json", decisions)
    _write_json(Path(cfg.output_dir) / "metrics" / "config_final.json", asdict(cfg))


# Evaluation
def evaluate_binary_classifier(pred_df: DataFrame, *, name: str, cfg: Config, split: str) -> Dict[str, Any]:
    out_dir = Path(cfg.output_dir)
    eda_dir = out_dir / "eda"
    metrics_dir = out_dir / "metrics"
    eda_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    p = pred_df

    if "label" not in p.columns:
        raise ValueError(f"{name}: expected label column 'label'.")

    # Ensure prob1 exists
    if "probability" in p.columns:
        p = p.withColumn("prob1", vector_to_array(F.col("probability")).getItem(1))
    elif "prob1" not in p.columns:
        raise ValueError(f"{name}: need probability or prob1 column for curves/calibration.")

    # Prediction
    if "prediction" not in p.columns:
        p = p.withColumn("prediction", F.when(F.col("prob1") >= F.lit(0.5), F.lit(1.0)).otherwise(F.lit(0.0)))

    # Normalization
    p = (
        p.withColumn("label", F.col("label").cast("double"))
        .withColumn("prediction", F.col("prediction").cast("double"))
        .withColumn("prob1", F.col("prob1").cast("double"))
    )

    # Confusion matrix + precision/recall
    cm_rows = p.withColumn("label_i", F.col("label").cast("int")).withColumn("pred_i", F.col("prediction").cast("int")).groupBy("label_i", "pred_i").count().collect()
    counts = {(int(r["label_i"]), int(r["pred_i"])): int(r["count"]) for r in cm_rows}
    tn = counts.get((0, 0), 0)
    fp = counts.get((0, 1), 0)
    fn = counts.get((1, 0), 0)
    tp = counts.get((1, 1), 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    # Scalars: AUC via BinaryClassificationMetrics (no dependency on rawPrediction col)
    cap = int(min(int(cfg.eval_curve_cap_rows), int(cfg.max_polarity_rows or cfg.eval_curve_cap_rows)) if (cfg.max_polarity_rows and cfg.max_polarity_rows > 0) else int(cfg.eval_curve_cap_rows))
    curve_df = p.select("prob1", "label").limit(int(cap))
    scoreAndLabels = curve_df.rdd.map(lambda r: (float(r["prob1"]), float(r["label"])))
    bcm = BinaryClassificationMetrics(scoreAndLabels)

    auc_roc = float(bcm.areaUnderROC) if hasattr(bcm, "areaUnderROC") else float("nan")
    auc_pr = float(bcm.areaUnderPR) if hasattr(bcm, "areaUnderPR") else float("nan")

    acc = float(MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy").evaluate(p))
    f1 = float(MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1").evaluate(p))

    # Brier + calibration bins
    brier = float(p.select(F.mean(F.pow(F.col("prob1") - F.col("label"), F.lit(2))).alias("brier")).collect()[0]["brier"] or 0.0)

    bins = int(max(2, cfg.calibration_bins))
    calib = (
        p.withColumn("bin", (F.floor(F.col("prob1") * F.lit(bins)) / F.lit(float(bins))))
        .groupBy("bin")
        .agg(F.count("*").alias("n"), F.avg("prob1").alias("avg_p"), F.avg("label").alias("avg_y"))
        .orderBy("bin")
        .collect()
    )
    calib_x = [float(r["avg_p"]) for r in calib]
    calib_y = [float(r["avg_y"]) for r in calib]
    calib_n = [int(r["n"]) for r in calib]

    # ROC
    roc_pts = bcm.roc().collect()
    pr_pts = bcm.pr().collect()

    roc_x = [float(x) for x, _ in roc_pts]
    roc_y = [float(y) for _, y in roc_pts]
    pr_x = [float(x) for x, _ in pr_pts]   # recall
    pr_y = [float(y) for _, y in pr_pts]   # precision

    # F1 vs threshold 
    thr_file = None
    best_thr = None
    best_f1_thr = None
    try:
        f1_thr = bcm.fMeasureByThreshold().collect()  
        if f1_thr:
            f1_thr_sorted = sorted(((float(t), float(v)) for t, v in f1_thr), key=lambda x: x[0])
            xs = [t for t, _ in f1_thr_sorted]
            ys = [v for _, v in f1_thr_sorted]
            best_thr, best_f1_thr = max(f1_thr_sorted, key=lambda x: x[1])
            thr_file = plot_line(
                xs,
                ys,
                title=f"F1 vs threshold – {name} ({split})",
                xlabel="Threshold",
                ylabel="F1",
                out_path=eda_dir / f"eval_{name}_{split}_f1_thr.png",
                vline=float(best_thr),
            )
    except Exception:
        pass

    # Plots
    cm_file = plot_confusion_2x2(
        [[tn, fp], [fn, tp]],
        title=f"Confusion matrix – {name} ({split})",
        out_path=eda_dir / f"eval_{name}_{split}_cm.png",
    )

    roc_file = plot_line(
        roc_x,
        roc_y,
        title=f"ROC – {name} ({split})",
        xlabel="FPR",
        ylabel="TPR",
        out_path=eda_dir / f"eval_{name}_{split}_roc.png",
    )

    pr_file = plot_line(
        pr_x,
        pr_y,
        title=f"PR – {name} ({split})",
        xlabel="Recall",
        ylabel="Precision",
        out_path=eda_dir / f"eval_{name}_{split}_pr.png",
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.plot(calib_x, calib_y, marker="o" if len(calib_x) <= 60 else None)
    ax.set_title(f"Calibration – {name} ({split})")
    ax.set_xlabel("Avg predicted prob")
    ax.set_ylabel("Observed positive rate")
    calib_file = _save_fig_basic(fig, eda_dir / f"eval_{name}_{split}_calib.png")

    out = {
        "generated_at": _utc_now_iso(),
        "model": name,
        "split": split,
        "metrics": {
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "accuracy": acc,
            "f1": f1,
            "precision_pos": float(precision),
            "recall_pos": float(recall),
            "brier": brier,
        },
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "calibration_bins": [{"avg_p": float(x), "avg_y": float(y), "n": int(n)} for x, y, n in zip(calib_x, calib_y, calib_n)],
        "threshold_tuning": {"best_thr": float(best_thr) if best_thr is not None else None, "best_f1": float(best_f1_thr) if best_f1_thr is not None else None},
        "plots": {"cm": cm_file, "roc": roc_file, "pr": pr_file, "calibration": calib_file, "f1_threshold": thr_file},
        "curve_rows_cap": int(cap),
    }
    _write_json(metrics_dir / f"eval_{name}_{split}.json", out)
    return out


def _collect_eval_jsons(cfg: Config) -> List[Dict[str, Any]]:
    metrics_dir = Path(cfg.output_dir) / "metrics"
    out: List[Dict[str, Any]] = []
    for p in sorted(metrics_dir.glob("eval_*_*.json")):
        j = _read_json(p)
        if isinstance(j, dict) and j.get("metrics"):
            out.append(j)
    return out


def plot_model_leaderboard(eval_jsons: List[Dict[str, Any]], cfg: Config, *, metric_key: str = "auc_roc") -> Optional[str]:
    eda_dir = Path(cfg.output_dir) / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in eval_jsons:
        try:
            m = e.get("metrics") or {}
            val = m.get(metric_key)
            if val is None:
                continue
            rows.append((str(e.get("model", "unknown")), float(val)))
        except Exception:
            continue
    if not rows:
        return None
    rows = sorted(rows, key=lambda x: x[1], reverse=True)
    models = [m for m, _ in rows]
    vals = [v for _, v in rows]
    return plot_bar(
        models,
        vals,
        title=f"Model comparison – {metric_key}",
        xlabel="Model",
        ylabel=metric_key,
        out_path=eda_dir / f"models_leaderboard_{metric_key}.png",
    )


def refresh_leaderboards(cfg: Config) -> Dict[str, Optional[str]]:
    evals = _collect_eval_jsons(cfg)
    out = {
        "auc_roc": plot_model_leaderboard(evals, cfg, metric_key="auc_roc"),
        "auc_pr": plot_model_leaderboard(evals, cfg, metric_key="auc_pr"),
        "f1": plot_model_leaderboard(evals, cfg, metric_key="f1"),
        "accuracy": plot_model_leaderboard(evals, cfg, metric_key="accuracy"),
    }
    _write_json(Path(cfg.output_dir) / "metrics" / "leaderboards.json", {"generated_at": _utc_now_iso(), "files": out})
    return out


# Supervisionado
def _add_split_bucket(df: DataFrame, cfg: Config) -> DataFrame:
    mod = int(cfg.deterministic_split_mod)
    seed = int(cfg.random_state)
    bucket = F.pmod(F.abs(F.xxhash64(F.col("text_full"), F.lit(seed))), F.lit(mod)).cast("int")
    return df.withColumn("split_bucket", bucket)


def _train_test_filters(cfg: Config) -> Tuple[Any, Any]:
    mod = int(cfg.deterministic_split_mod)
    cut = int(mod * float(cfg.train_split_fraction))
    if cfg.polarity_train_path:
        train_cond = (F.col("dataset_split") == F.lit("train"))
        test_cond = (F.col("dataset_split") == F.lit("test"))
    else:
        train_cond = (F.col("split_bucket") < F.lit(cut))
        test_cond = (F.col("split_bucket") >= F.lit(cut))
    return train_cond, test_cond


def fit_tfidf_features(df: DataFrame, cfg: Config) -> Tuple[DataFrame, IDFModel]:
    tok = RegexTokenizer(inputCol="text_full", outputCol="tokens", pattern=r"\W+", toLowercase=True)
    sw = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", locale="en_US")
    hashing = HashingTF(inputCol="filtered_tokens", outputCol="raw_features", numFeatures=int(cfg.hashing_num_features))

    raw = (
        hashing.transform(sw.transform(tok.transform(df)))
        .select("dataset_split", "sentiment", F.col("sentiment").cast("double").alias("label"), "split_bucket", "raw_features")
        .where(F.col("label").isNotNull())
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    _ = raw.count()

    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
    idf_model: IDFModel = idf.fit(raw)
    features = idf_model.transform(raw).select("dataset_split", "sentiment", "label", "split_bucket", "tfidf_features")

    raw.unpersist()
    return features, idf_model


def train_supervised_models_fast(features_df: DataFrame, cfg: Config) -> Tuple[Dict[str, Any], Any, str]:
    plots_dir = Path(cfg.output_dir) / "metrics" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    base = features_df.select("tfidf_features", "label", "dataset_split", "split_bucket").persist(StorageLevel.MEMORY_AND_DISK)
    _ = base.count()

    mod = int(cfg.deterministic_split_mod)
    cut = int(mod * float(cfg.train_split_fraction))

    if cfg.polarity_train_path:
        train_df = base.where(F.col("dataset_split") == F.lit("train")).select("tfidf_features", "label")
        test_df = base.where(F.col("dataset_split") == F.lit("test")).select("tfidf_features", "label")
    else:
        train_df = base.where(F.col("split_bucket") < F.lit(cut)).select("tfidf_features", "label")
        test_df = base.where(F.col("split_bucket") >= F.lit(cut)).select("tfidf_features", "label")

    train_df = train_df.persist(StorageLevel.MEMORY_AND_DISK)
    test_df = test_df.persist(StorageLevel.MEMORY_AND_DISK)
    _ = train_df.count()
    _ = test_df.count()

    out: Dict[str, Any] = {}

    # LR CV 
    lr = LogisticRegression(featuresCol="tfidf_features", labelCol="label", maxIter=20)
    lr_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [float(x) for x in cfg.logreg_reg_params])
        .addGrid(lr.elasticNetParam, [float(x) for x in cfg.logreg_l1_ratios])
        .build()
    )
    lr_eval_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

    logger.info("Supervised: starting LR CV (grid=%d, folds=%d)", len(lr_grid), int(cfg.cv_folds))
    t0 = time.time()
    lr_cv = CrossValidator(
        estimator=lr,
        estimatorParamMaps=lr_grid,
        evaluator=lr_eval_auc,
        numFolds=int(cfg.cv_folds),
        parallelism=int(cfg.cv_parallelism),
    )
    lr_cv_model = lr_cv.fit(train_df)
    lr_best = lr_cv_model.bestModel
    lr_pred = lr_best.transform(test_df).persist(StorageLevel.MEMORY_AND_DISK)
    _ = lr_pred.count()

    lr_auc = float(lr_eval_auc.evaluate(lr_pred))
    lr_bin = _binary_metrics_from_pred(lr_pred)

    lr_cm = lr_bin["confusion"]
    lr_cm_file = _plot_confusion_2x2(
        cfg,
        tn=lr_cm["tn"], fp=lr_cm["fp"], fn=lr_cm["fn"], tp=lr_cm["tp"],
        title="LogReg (dev) – Confusion Matrix",
        out_path=plots_dir / "confusion_logreg_cv.png",
    )

    out["logreg_cv"] = {
        "auc_roc": lr_auc,
        **{k: v for k, v in lr_bin.items() if k != "confusion"},
        "confusion": lr_cm,
        "best_params": {
            "regParam": float(_maybe_call(lr_best, "getRegParam")),
            "elasticNetParam": float(_maybe_call(lr_best, "getElasticNetParam")),
        },
        "grid_size": int(len(lr_grid)),
        "cv_folds": int(cfg.cv_folds),
        "seconds": float(time.time() - t0),
        "files": {"confusion": lr_cm_file},
    }
    logger.info("Supervised: LR done | auc=%.4f acc=%.4f f1=%.4f", lr_auc, out["logreg_cv"]["accuracy"], out["logreg_cv"]["f1"])

    best_model = lr_best
    best_name = "logreg_cv"
    best_auc = lr_auc

    lr_pred.unpersist()

    # RF CV 
    if cfg.enable_rf:
        rf_train = train_df
        if cfg.rf_max_train_rows and cfg.rf_max_train_rows > 0:
            rf_train = rf_train.limit(int(cfg.rf_max_train_rows))
        rf_train = rf_train.persist(StorageLevel.MEMORY_AND_DISK)
        _ = rf_train.count()

        rf = RandomForestClassifier(
            featuresCol="tfidf_features",
            labelCol="label",
            seed=int(cfg.random_state),
            maxBins=int(cfg.rf_max_bins),
            subsamplingRate=float(cfg.rf_subsampling_rate),
            featureSubsetStrategy=str(cfg.rf_feature_subset_strategy),
        )
        rf_grid = (
            ParamGridBuilder()
            .addGrid(rf.numTrees, [int(x) for x in cfg.rf_num_trees])
            .addGrid(rf.maxDepth, [int(x) for x in cfg.rf_max_depths])
            .build()
        )
        rf_eval_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

        logger.info("Supervised: starting RF CV (grid=%d, folds=%d, cap_rows=%d)", len(rf_grid), int(cfg.cv_folds), int(cfg.rf_max_train_rows))
        t1 = time.time()
        rf_cv = CrossValidator(
            estimator=rf,
            estimatorParamMaps=rf_grid,
            evaluator=rf_eval_auc,
            numFolds=int(cfg.cv_folds),
            parallelism=int(cfg.cv_parallelism),
        )
        rf_cv_model = rf_cv.fit(rf_train)
        rf_best = rf_cv_model.bestModel
        rf_pred = rf_best.transform(test_df).persist(StorageLevel.MEMORY_AND_DISK)
        _ = rf_pred.count()

        rf_auc = float(rf_eval_auc.evaluate(rf_pred))
        rf_bin = _binary_metrics_from_pred(rf_pred)

        rf_cm = rf_bin["confusion"]
        rf_cm_file = _plot_confusion_2x2(
            cfg,
            tn=rf_cm["tn"], fp=rf_cm["fp"], fn=rf_cm["fn"], tp=rf_cm["tp"],
            title="RandomForest (dev) – Confusion Matrix",
            out_path=plots_dir / "confusion_random_forest_cv.png",
        )

        out["random_forest_cv"] = {
            "auc_roc": rf_auc,
            **{k: v for k, v in rf_bin.items() if k != "confusion"},
            "confusion": rf_cm,
            "best_params": {
                "numTrees": int(_maybe_call(rf_best, "getNumTrees")),
                "maxDepth": int(_maybe_call(rf_best, "getMaxDepth")),
            },
            "grid_size": int(len(rf_grid)),
            "cv_folds": int(cfg.cv_folds),
            "train_rows_cap": int(cfg.rf_max_train_rows),
            "seconds": float(time.time() - t1),
            "files": {"confusion": rf_cm_file},
        }
        logger.info("Supervised: RF done | auc=%.4f acc=%.4f f1=%.4f", rf_auc, out["random_forest_cv"]["accuracy"], out["random_forest_cv"]["f1"])

        if rf_auc > best_auc:
            best_model = rf_best
            best_name = "random_forest_cv"
            best_auc = rf_auc

        rf_pred.unpersist()
        rf_train.unpersist()

    # Comparações
    names = list(out.keys())
    aucs = [float(out[n].get("auc_roc", float("nan"))) for n in names]
    accs = [float(out[n].get("accuracy", float("nan"))) for n in names]
    f1s  = [float(out[n].get("f1", float("nan"))) for n in names]

    comp_file = _plot_grouped_bars(
        cfg,
        x_labels=names,
        series={"AUC_ROC": aucs, "Accuracy": accs, "F1": f1s},
        title="Supervised models (dev) – metric comparison",
        xlabel="model",
        ylabel="metric",
        out_path=plots_dir / "supervised_metric_comparison.png",
        rotate=15,
        figsize=(11, 4),
    )

    _write_json(Path(cfg.output_dir) / "metrics" / "supervised_models.json", {**out, "_files": {"comparison": comp_file}})

    best_spec = {
        "generated_at": _utc_now_iso(),
        "winner": best_name,
        "winner_metric": "auc_roc",
        "winner_value": float(best_auc),
        "files": {"comparison": comp_file},
        "logreg_best_params": (out.get("logreg_cv") or {}).get("best_params"),
        "rf_best_params": (out.get("random_forest_cv") or {}).get("best_params"),
    }
    _write_json(Path(cfg.output_dir) / "metrics" / "supervised_best.json", best_spec)

    train_df.unpersist()
    test_df.unpersist()
    base.unpersist()
    return out, best_model, best_name

def save_best_supervised_pipeline(idf_model: IDFModel, best_model: Any, cfg: Config) -> Path:
    tok = RegexTokenizer(inputCol="text_full", outputCol="tokens", pattern=r"\W+", toLowercase=True)
    sw = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", locale="en_US")
    hashing = HashingTF(inputCol="filtered_tokens", outputCol="raw_features", numFeatures=int(cfg.hashing_num_features))

    pm = PipelineModel(stages=[tok, sw, hashing, idf_model, best_model])

    out_dir = Path(cfg.output_dir) / "models" / "supervised_pipeline"
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    pm.write().overwrite().save(str(out_dir))
    logger.info("Saved supervised pipeline: %s", str(out_dir))
    return out_dir


# Clustering 
def fit_and_save_cluster_pipeline(spark: SparkSession, df: DataFrame, cfg: Config) -> Dict[str, Any]:
    out_dir = Path(cfg.output_dir)
    eda_dir = out_dir / "eda"
    metrics_dir = out_dir / "metrics"
    eda_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    base = df.select("text_full", "sentiment").where(F.col("text_full").isNotNull()).where(F.col("sentiment").isNotNull())
    if cfg.embedding_max_rows and cfg.embedding_max_rows > 0:
        base = base.limit(int(cfg.embedding_max_rows))

    parts = int(min(max(2, cfg.spark_default_parallelism), 64))
    base = base.repartition(parts).persist(StorageLevel.MEMORY_AND_DISK)
    base_rows = int(base.count())

    requested_dim = int(cfg.cluster_hashing_num_features)
    safe_dim = _safe_pca_input_dim(spark, requested_dim=requested_dim, requested_k=int(min(cfg.embedding_pca_k_max, requested_dim)))
    if safe_dim != requested_dim:
        logger.warning("Clustering: clamping hash_dim %d -> %d (driver safety)", requested_dim, safe_dim)

    tok = RegexTokenizer(inputCol="text_full", outputCol="tokens", pattern=r"\W+", toLowercase=True)
    sw = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", locale="en_US")
    hashing = HashingTF(inputCol="filtered_tokens", outputCol="raw_features", numFeatures=int(safe_dim))
    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
    norm = Normalizer(inputCol="tfidf_features", outputCol="tfidf_norm", p=2.0)

    # tf-idf c/ normalização
    logger.info("Clustering: building normalized TF-IDF (rows=%d, hash_dim=%d) ...", base_rows, safe_dim)
    t0 = time.time()
    tf = hashing.transform(sw.transform(tok.transform(base))).select("sentiment", "raw_features").persist(StorageLevel.MEMORY_AND_DISK)
    _ = tf.count()
    idf_model = idf.fit(tf)
    tfidf = idf_model.transform(tf).select("sentiment", "tfidf_features").persist(StorageLevel.MEMORY_AND_DISK)
    _ = tfidf.count()
    tfidf_norm = norm.transform(tfidf).select("sentiment", "tfidf_norm").persist(StorageLevel.MEMORY_AND_DISK)
    _ = tfidf_norm.count()

    # PCA k por variancia explicada
    pca_max_k = int(max(2, min(cfg.embedding_pca_k_max, safe_dim)))
    pca_var_target = float(max(0.0, min(0.999, cfg.pca_var_target)))
    use_pca = pca_max_k >= 2 and pca_var_target > 0.0

    pca_k_chosen = 0
    pca_cum = []
    pca_plot = None
    pca_model = None

    if use_pca:
        logger.info("Clustering: fitting PCA (max_k=%d) for variance curve ...", pca_max_k)
        pca_tmp = PCA(k=int(pca_max_k), inputCol="tfidf_norm", outputCol="pca_tmp")
        pca_tmp_model = pca_tmp.fit(tfidf_norm)

        ev = pca_tmp_model.explainedVariance
        ev_arr = np.array(ev.toArray() if hasattr(ev, "toArray") else list(ev), dtype=float)
        ev_arr = np.nan_to_num(ev_arr, nan=0.0, posinf=0.0, neginf=0.0)
        cum = np.cumsum(ev_arr).tolist()
        pca_cum = [float(x) for x in cum]

        pca_k_chosen = int(next((i + 1 for i, v in enumerate(pca_cum) if v >= pca_var_target), pca_max_k))
        pca_k_chosen = int(max(2, min(pca_k_chosen, pca_max_k)))

        pca_plot = plot_line(
            list(range(1, len(pca_cum) + 1)),
            pca_cum,
            title=f"PCA cumulative explained variance (target={pca_var_target:.2f})",
            xlabel="k",
            ylabel="Cumulative variance",
            out_path=eda_dir / "cluster_pca_cumvar.png",
            vline=float(pca_k_chosen),
        )

        logger.info("Clustering: chosen PCA k=%d (target_var=%.2f)", pca_k_chosen, pca_var_target)

        pca = PCA(k=int(pca_k_chosen), inputCol="tfidf_norm", outputCol="embedding_vec")
        pca_model = pca.fit(tfidf_norm)
        embed_df = pca_model.transform(tfidf_norm).select("sentiment", "embedding_vec").persist(StorageLevel.MEMORY_AND_DISK)
        _ = embed_df.count()
    else:
        embed_df = tfidf_norm.select("sentiment", F.col("tfidf_norm").alias("embedding_vec")).persist(StorageLevel.MEMORY_AND_DISK)
        _ = embed_df.count()

    # KMeans k
    kmin = int(max(2, cfg.kmeans_k_min))
    kmax = int(max(kmin, cfg.kmeans_k_max))
    candidates = list(range(kmin, kmax + 1))
    if len(candidates) > 15:
        step = max(1, len(candidates) // 15)
        candidates = candidates[::step]
        if candidates[-1] != kmax:
            candidates.append(kmax)

    sil_scores: List[Tuple[int, float]] = []
    logger.info("Clustering: tuning kmeans k via silhouette | candidates=%s", candidates)

    try_cosine = True
    for k in candidates:
        km = KMeans(featuresCol="embedding_vec", predictionCol="cluster_id", k=int(k), maxIter=int(cfg.kmeans_max_iter), seed=int(cfg.random_state))
        km_model = km.fit(embed_df)
        pred = km_model.transform(embed_df).select("embedding_vec", "cluster_id").persist(StorageLevel.MEMORY_AND_DISK)
        _ = pred.count()

        try:
            if try_cosine:
                evl = ClusteringEvaluator(featuresCol="embedding_vec", predictionCol="cluster_id", metricName="silhouette", distanceMeasure="cosine")
                s = float(evl.evaluate(pred))
            else:
                raise Exception("skip cosine")
        except Exception:
            try_cosine = False
            evl = ClusteringEvaluator(featuresCol="embedding_vec", predictionCol="cluster_id", metricName="silhouette")
            s = float(evl.evaluate(pred))

        sil_scores.append((int(k), float(s)))
        pred.unpersist()

    sil_scores_sorted = sorted(sil_scores, key=lambda x: x[0])
    ks = [k for k, _ in sil_scores_sorted]
    ss = [s for _, s in sil_scores_sorted]
    best_k, best_s = max(sil_scores_sorted, key=lambda x: x[1]) if sil_scores_sorted else (int(cfg.kmeans_k), float("nan"))

    sil_plot = plot_line(
        ks,
        ss,
        title="KMeans silhouette vs k",
        xlabel="k",
        ylabel="Silhouette",
        out_path=eda_dir / "cluster_silhouette_by_k.png",
        vline=float(best_k),
    )

    logger.info("Clustering: chosen kmeans k=%d (silhouette=%.4f)", best_k, best_s)

    # Fit kmeans
    kmeans = KMeans(featuresCol="embedding_vec", predictionCol="cluster_id", k=int(best_k), maxIter=int(cfg.kmeans_max_iter), seed=int(cfg.random_state))
    kmeans_model = kmeans.fit(embed_df)

    clustered = kmeans_model.transform(embed_df).select("sentiment", "embedding_vec", "cluster_id").persist(StorageLevel.MEMORY_AND_DISK)
    _ = clustered.count()

    try:
        ev_final = ClusteringEvaluator(featuresCol="embedding_vec", predictionCol="cluster_id", metricName="silhouette", distanceMeasure="cosine")
        silhouette = float(ev_final.evaluate(clustered))
    except Exception:
        ev_final = ClusteringEvaluator(featuresCol="embedding_vec", predictionCol="cluster_id", metricName="silhouette")
        silhouette = float(ev_final.evaluate(clustered))

    prof = (
        clustered.groupBy("cluster_id")
        .agg(F.count("*").alias("n"), F.avg(F.col("sentiment").cast("double")).alias("mean_sentiment"))
        .orderBy("cluster_id")
        .collect()
    )
    cluster_sizes = [(int(r["cluster_id"]), int(r["n"])) for r in prof]
    size_plot = plot_bar(
        [str(k) for k, _ in cluster_sizes],
        [n for _, n in cluster_sizes],
        title="Cluster sizes",
        xlabel="cluster_id",
        ylabel="n",
        out_path=eda_dir / "cluster_sizes.png",
    )

    # pipeline model (transformers + fitted models)
    stages: List[Any] = [tok, sw, hashing, idf_model, norm]
    if pca_model is not None:
        stages.append(pca_model)
    else:
        stages.append(SQLTransformer(statement="SELECT *, tfidf_norm AS embedding_vec FROM __THIS__"))
    stages.append(kmeans_model)

    pipeline_model = PipelineModel(stages=stages)
    model_path = out_dir / "models" / "cluster_pipeline"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline_model.write().overwrite().save(str(model_path))
    logger.info("Saved cluster pipeline: %s", str(model_path))

    out = {
        "generated_at": _utc_now_iso(),
        "rows_fit": int(base_rows),
        "hash_dim": int(safe_dim),
        "pca_used": bool(pca_model is not None),
        "pca_k_chosen": int(pca_k_chosen) if pca_model is not None else 0,
        "pca_var_target": float(pca_var_target),
        "pca_cumvar": pca_cum[: min(len(pca_cum), 512)],
        "kmeans_k_candidates": ks,
        "kmeans_silhouette_scores": ss,
        "kmeans_k_chosen": int(best_k),
        "kmeans_max_iter": int(cfg.kmeans_max_iter),
        "silhouette_final": float(silhouette),
        "cluster_profiles": [r.asDict(True) for r in prof],
        "plots": {"pca_cumvar": pca_plot, "silhouette_by_k": sil_plot, "cluster_sizes": size_plot},
        "model_path": str(model_path),
        "seconds": float(time.time() - t0),
    }
    _write_json(metrics_dir / "cluster_kmeans.json", out)

    #  Charts
    plots_dir = Path(cfg.output_dir) / "metrics" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    prof_rows = [r.asDict(True) for r in prof]
    if prof_rows:
        labels = [str(r["cluster_id"]) for r in prof_rows]
        sizes = [int(r["n"]) for r in prof_rows]
        means = [float(r["mean_sentiment"]) for r in prof_rows]

        out["files"] = {}
        out["files"]["cluster_sizes"] = _plot_bar(
            cfg, labels, sizes,
            title="Cluster sizes", xlabel="cluster_id", ylabel="n",
            out_path=plots_dir / "cluster_sizes.png",
            rotate=0,
            figsize=(11, 4),
        )
        out["files"]["cluster_mean_sentiment"] = _plot_bar(
            cfg, labels, means,
            title="Mean sentiment by cluster", xlabel="cluster_id", ylabel="mean(sentiment)",
            out_path=plots_dir / "cluster_mean_sentiment.png",
            rotate=0,
            figsize=(11, 4),
        )

    clustered.unpersist()
    embed_df.unpersist()
    tfidf_norm.unpersist()
    tfidf.unpersist()
    tf.unpersist()
    base.unpersist()
    return out

# Score-only
def score_with_saved_supervised_model(spark: SparkSession, df: DataFrame, cfg: Config) -> DataFrame:
    model_path = Path(cfg.output_dir) / "models" / "supervised_pipeline"
    if not model_path.exists():
        raise RuntimeError(f"Missing supervised model at {model_path}. Run dev/train_full first.")
    pm = PipelineModel.load(str(model_path))
    pred = pm.transform(df)
    return pred.select("review_id", "dataset_split", "sentiment", F.col("prediction").cast("int").alias("prediction"))


def score_with_saved_cluster_model(spark: SparkSession, df: DataFrame, cfg: Config) -> DataFrame:
    model_path = Path(cfg.output_dir) / "models" / "cluster_pipeline"
    if not model_path.exists():
        raise RuntimeError(f"Missing cluster model at {model_path}. Run dev first.")
    pm = PipelineModel.load(str(model_path))
    scored = pm.transform(df)
    return scored.select("review_id", "dataset_split", F.col("cluster_id").cast("int").alias("cluster_id"))

# Train full 
def train_full_supervised_pipeline(spark: SparkSession, df: DataFrame, cfg: Config) -> Dict[str, Any]:
    best_path = Path(cfg.output_dir) / "metrics" / "supervised_best.json"
    if not best_path.exists():
        raise RuntimeError(f"Missing {best_path}. Run dev first.")

    best = json.loads(best_path.read_text(encoding="utf-8")) or {}
    lr_params = best.get("logreg_best_params") or {}
    reg_param = float(lr_params.get("regParam", 0.1))
    enet = float(lr_params.get("elasticNetParam", 0.0))

    d0 = (
        df.select("dataset_split", "sentiment", "text_full")
        .where(F.col("sentiment").isNotNull())
        .where(F.col("text_full").isNotNull())
    )
    d0 = _add_split_bucket(d0, cfg)

    train_cond, test_cond = _train_test_filters(cfg)

    train_raw = d0.where(train_cond).select("text_full", F.col("sentiment").cast("double").alias("label"))
    test_raw  = d0.where(test_cond).select("text_full", F.col("sentiment").cast("double").alias("label"))

    tok = RegexTokenizer(inputCol="text_full", outputCol="tokens", pattern=r"\W+", toLowercase=True)
    sw = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", locale="en_US")
    hashing = HashingTF(inputCol="filtered_tokens", outputCol="raw_features", numFeatures=int(cfg.hashing_num_features))
    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
    preproc = Pipeline(stages=[tok, sw, hashing, idf])

    logger.info("train_full: fitting preprocessing (tok->sw->hash->idf) ...")
    t0 = time.time()
    preproc_model = preproc.fit(train_raw.select("text_full"))

    logger.info("train_full: transforming TF-IDF + caching train features (DISK_ONLY) ...")
    train_feat = (
        preproc_model.transform(train_raw)
        .select("tfidf_features", "label")
        .persist(StorageLevel.DISK_ONLY)
    )
    _ = train_feat.count()  # materialize cache

    # No need to cache test unless reused
    test_feat = preproc_model.transform(test_raw).select("tfidf_features", "label")

    lr = LogisticRegression(
        featuresCol="tfidf_features",
        labelCol="label",
        maxIter=20,
        regParam=reg_param,
        elasticNetParam=enet,
    )
    logger.info("train_full: fitting LogisticRegression (regParam=%.4f elasticNetParam=%.4f) ...", reg_param, enet)
    lr_model = lr.fit(train_feat)

    pred = lr_model.transform(test_feat)

    eval_bundle = evaluate_binary_classifier(
        pred.select("label", "prediction", "probability"),
        name="logreg_full",
        cfg=cfg,
        split="test",
    )

    final_model = PipelineModel(stages=list(preproc_model.stages) + [lr_model])
    out_dir = Path(cfg.output_dir) / "models" / "supervised_pipeline"
    final_model.write().overwrite().save(str(out_dir))
    logger.info("train_full: saved supervised pipeline: %s", str(out_dir))

    metrics = {
        "generated_at": _utc_now_iso(),
        "model": "logistic_regression_full",
        "regParam": reg_param,
        "elasticNetParam": enet,
        "eval_json": "eval_logreg_full_test.json",
        "eval_metrics": eval_bundle.get("metrics"),
        "seconds": float(time.time() - t0),
        "notes": "train_full overwrites models/supervised_pipeline with the full-data version.",
    }
    _write_json(Path(cfg.output_dir) / "metrics" / "supervised_full.json", metrics)

    train_feat.unpersist()
    refresh_leaderboards(cfg)
    return metrics

# ModernBERT 
@dataclass
class TorchSplitBundle:
    train_texts: List[str]
    train_labels: List[int]
    test_texts: List[str]
    test_labels: List[int]
    meta: Dict[str, Any]


def collect_train_test_texts_labels(df_bucket: DataFrame, cfg: Config) -> TorchSplitBundle:
    train_cond, test_cond = _train_test_filters(cfg)

    base = df_bucket.select("text_full", "sentiment", "dataset_split", "split_bucket").where(F.col("text_full").isNotNull()).where(F.col("sentiment").isNotNull())

    train_df = base.where(train_cond).select("text_full", "sentiment")
    test_df = base.where(test_cond).select("text_full", "sentiment")

    train_n = int(train_df.count())
    test_n = int(test_df.count())
    total = train_n + test_n

    cap_total = int(cfg.torch_max_rows) if (cfg.torch_max_rows and cfg.torch_max_rows > 0) else total
    cap_total = min(cap_total, total)

    if total > 0:
        cap_train = int(round(cap_total * (train_n / total))) if train_n > 0 else 0
        cap_test = cap_total - cap_train
    else:
        cap_train = cap_test = 0

    if train_n > 0:
        cap_train = max(1, min(cap_train, train_n))
    if test_n > 0:
        cap_test = max(1, min(cap_test, test_n))

    while (cap_train + cap_test) > cap_total and cap_train > 1:
        cap_train -= 1
    while (cap_train + cap_test) > cap_total and cap_test > 1:
        cap_test -= 1

    train_rows = train_df.orderBy(F.xxhash64(F.col("text_full"), F.lit(int(cfg.random_state)))).limit(int(cap_train)).collect() if cap_train > 0 else []
    test_rows = test_df.orderBy(F.xxhash64(F.col("text_full"), F.lit(int(cfg.random_state)))).limit(int(cap_test)).collect() if cap_test > 0 else []

    train_texts = [str(r["text_full"]) for r in train_rows]
    train_labels = [int(r["sentiment"]) for r in train_rows]
    test_texts = [str(r["text_full"]) for r in test_rows]
    test_labels = [int(r["sentiment"]) for r in test_rows]

    meta = {
        "train_rows_total": train_n,
        "test_rows_total": test_n,
        "torch_cap_total": cap_total,
        "train_used": len(train_texts),
        "test_used": len(test_texts),
        "split_logic": "dataset_split" if cfg.polarity_train_path else "deterministic_split_bucket",
    }
    return TorchSplitBundle(train_texts=train_texts, train_labels=train_labels, test_texts=test_texts, test_labels=test_labels, meta=meta)


def _auc_roc_from_scores(labels: List[int], scores: List[float]) -> float:
    # ROC AUC, sorting + trapezoid.
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.size == 0:
        return float("nan")
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]

    tp = 0
    fp = 0
    prev = None

    fpr = [0.0]
    tpr = [0.0]

    for yi, si in zip(y_sorted, s_sorted):
        if prev is None:
            prev = si
        if si != prev:
            fpr.append(fp / neg)
            tpr.append(tp / pos)
            prev = si
        if yi == 1:
            tp += 1
        else:
            fp += 1

    fpr.append(fp / neg)
    tpr.append(tp / pos)

    auc = 0.0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0
    return float(auc)


def _pr_curve_from_scores(labels: List[int], scores: List[float]) -> Tuple[List[float], List[float]]:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.size == 0:
        return [], []
    pos = int(np.sum(y == 1))
    if pos == 0:
        return [], []

    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]

    tp = 0
    fp = 0
    prev = None

    recall = [0.0]
    precision = [1.0]

    for yi, si in zip(y_sorted, s_sorted):
        if prev is None:
            prev = si
        if si != prev:
            r = tp / pos
            p = tp / (tp + fp) if (tp + fp) else 1.0
            recall.append(float(r))
            precision.append(float(p))
            prev = si
        if yi == 1:
            tp += 1
        else:
            fp += 1

    r = tp / pos
    p = tp / (tp + fp) if (tp + fp) else 1.0
    recall.append(float(r))
    precision.append(float(p))

    return recall, precision


def _auc_pr_from_curve(recall: List[float], precision: List[float]) -> float:
    if not recall or not precision or len(recall) != len(precision):
        return float("nan")
    auc = 0.0
    for i in range(1, len(recall)):
        auc += (recall[i] - recall[i - 1]) * (precision[i] + precision[i - 1]) / 2.0
    return float(auc)


def evaluate_binary_from_arrays(
    *,
    probs: List[float],
    labels: List[int],
    name: str,
    cfg: Config,
    split: str,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    out_dir = Path(cfg.output_dir)
    eda_dir = out_dir / "eda"
    metrics_dir = out_dir / "metrics"
    eda_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probs, dtype=float)
    if y.size != p.size:
        raise ValueError(f"{name}: probs/labels size mismatch ({p.size} vs {y.size})")

    pred = (p >= float(threshold)).astype(int)

    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tp = int(np.sum((y == 1) & (pred == 1)))

    acc = float(np.mean((pred == y).astype(float))) if y.size else float("nan")
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0

    auc_roc = _auc_roc_from_scores(labels, probs)
    rec_curve, prec_curve = _pr_curve_from_scores(labels, probs)
    auc_pr = _auc_pr_from_curve(rec_curve, prec_curve)

    brier = float(np.mean((p - y.astype(float)) ** 2)) if y.size else float("nan")

    # Calibration 
    bins = int(max(2, cfg.calibration_bins))
    edges = np.linspace(0.0, 1.0, bins + 1)
    calib_bins = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < (bins - 1) else (p >= lo) & (p <= hi)
        n = int(np.sum(mask))
        if n <= 0:
            continue
        avg_p = float(np.mean(p[mask]))
        avg_y = float(np.mean(y[mask].astype(float)))
        calib_bins.append((avg_p, avg_y, n))

    calib_x = [x for x, _, _ in calib_bins]
    calib_y = [y0 for _, y0, _ in calib_bins]
    calib_n = [n for _, _, n in calib_bins]

    # F1 vs threshold
    best_thr = None
    best_f1_thr = None
    thr_file = None
    try:
        # Evaluate on unique score thresholds (descending)
        uniq = np.unique(p)
        uniq = np.sort(uniq)[::-1]
        best = (-1.0, None)
        xs = []
        ys = []
        for thr in uniq:
            pr0 = (p >= thr).astype(int)
            tp0 = int(np.sum((y == 1) & (pr0 == 1)))
            fp0 = int(np.sum((y == 0) & (pr0 == 1)))
            fn0 = int(np.sum((y == 1) & (pr0 == 0)))
            prec0 = tp0 / (tp0 + fp0) if (tp0 + fp0) else 0.0
            rec0 = tp0 / (tp0 + fn0) if (tp0 + fn0) else 0.0
            f10 = (2 * prec0 * rec0) / (prec0 + rec0) if (prec0 + rec0) else 0.0
            xs.append(float(thr))
            ys.append(float(f10))
            if f10 > best[0]:
                best = (float(f10), float(thr))
        if best[1] is not None:
            best_f1_thr, best_thr = best[0], best[1]
            thr_file = plot_line(
                xs[::-1],
                ys[::-1],
                title=f"F1 vs threshold – {name} ({split})",
                xlabel="Threshold",
                ylabel="F1",
                out_path=eda_dir / f"eval_{name}_{split}_f1_thr.png",
                vline=float(best_thr),
            )
    except Exception:
        pass

    # Plots
    cm_file = plot_confusion_2x2(
        [[tn, fp], [fn, tp]],
        title=f"Confusion matrix – {name} ({split})",
        out_path=eda_dir / f"eval_{name}_{split}_cm.png",
    )

    # ROC curve from thresholds
    def roc_points(labels_: List[int], scores_: List[float]) -> Tuple[List[float], List[float]]:
        y_ = np.asarray(labels_, dtype=int)
        s_ = np.asarray(scores_, dtype=float)
        pos_ = int(np.sum(y_ == 1))
        neg_ = int(np.sum(y_ == 0))
        if pos_ == 0 or neg_ == 0:
            return [], []
        order_ = np.argsort(-s_, kind="mergesort")
        y_s = y_[order_]
        s_s = s_[order_]
        tp_ = 0
        fp_ = 0
        prev_ = None
        fpr_ = [0.0]
        tpr_ = [0.0]
        for yi, si in zip(y_s, s_s):
            if prev_ is None:
                prev_ = si
            if si != prev_:
                fpr_.append(fp_ / neg_)
                tpr_.append(tp_ / pos_)
                prev_ = si
            if yi == 1:
                tp_ += 1
            else:
                fp_ += 1
        fpr_.append(fp_ / neg_)
        tpr_.append(tp_ / pos_)
        return [float(x) for x in fpr_], [float(x) for x in tpr_]

    roc_x, roc_y = roc_points(labels, probs)
    roc_file = plot_line(
        roc_x,
        roc_y,
        title=f"ROC – {name} ({split})",
        xlabel="FPR",
        ylabel="TPR",
        out_path=eda_dir / f"eval_{name}_{split}_roc.png",
    )

    pr_file = plot_line(
        rec_curve,
        prec_curve,
        title=f"PR – {name} ({split})",
        xlabel="Recall",
        ylabel="Precision",
        out_path=eda_dir / f"eval_{name}_{split}_pr.png",
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.plot(calib_x, calib_y, marker="o" if len(calib_x) <= 60 else None)
    ax.set_title(f"Calibration – {name} ({split})")
    ax.set_xlabel("Avg predicted prob")
    ax.set_ylabel("Observed positive rate")
    calib_file = _save_fig_basic(fig, eda_dir / f"eval_{name}_{split}_calib.png")

    out = {
        "generated_at": _utc_now_iso(),
        "model": name,
        "split": split,
        "metrics": {
            "auc_roc": float(auc_roc),
            "auc_pr": float(auc_pr),
            "accuracy": float(acc),
            "f1": float(f1),
            "precision_pos": float(precision),
            "recall_pos": float(recall),
            "brier": float(brier),
        },
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "calibration_bins": [{"avg_p": float(x), "avg_y": float(y0), "n": int(n)} for x, y0, n in zip(calib_x, calib_y, calib_n)],
        "threshold_tuning": {"best_thr": float(best_thr) if best_thr is not None else None, "best_f1": float(best_f1_thr) if best_f1_thr is not None else None},
        "plots": {"cm": cm_file, "roc": roc_file, "pr": pr_file, "calibration": calib_file, "f1_threshold": thr_file},
        "curve_rows_cap": int(len(labels)),
    }
    _write_json(metrics_dir / f"eval_{name}_{split}.json", out)
    return out


def train_modernbert_sentiment_from_splits(bundle: TorchSplitBundle, cfg: Config) -> Dict[str, Any]:
    loss_trace: List[Dict[str, Any]] = []
    epoch_trace: List[Dict[str, Any]] = []
    global_step = 0

    if torch is None:
        raise RuntimeError(f"torch import failed: {_TORCH_IMPORT_ERROR}")

    from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
    from torch.utils.data import TensorDataset, DataLoader  # type: ignore
    from torch.optim import AdamW  # type: ignore

    train_texts, train_labels = bundle.train_texts, bundle.train_labels
    test_texts, test_labels = bundle.test_texts, bundle.test_labels

    if not train_texts or not test_texts:
        raise RuntimeError("ModernBERT requires both train and test splits (non-empty).")

    device = _best_torch_device()
    use_amp = (device == "cuda")
    pin = (device == "cuda")

    batch_size = int(cfg.modernbert_batch_size)
    if device == "cpu":
        batch_size = min(batch_size, 16)
    if device == "mps":
        batch_size = min(batch_size, 32)

    logger.info(
        "ModernBERT: device=%s | amp=%s | batch_size=%d | max_len=%d | train=%d | test=%d",
        device,
        use_amp,
        batch_size,
        int(cfg.modernbert_max_seq_len),
        len(train_texts),
        len(test_texts),
    )

    # Threading
    try:
        if device == "cpu":
            cores = os.cpu_count() or 4
            torch.set_num_interop_threads(1)
            torch.set_num_threads(max(1, min(8, cores)))
        else:
            torch.set_num_interop_threads(1)
            torch.set_num_threads(1)
    except Exception:
        pass

    rng = np.random.RandomState(int(cfg.random_state))
    idx = np.arange(len(train_texts))
    rng.shuffle(idx)
    val_n = max(1, int(0.1 * len(idx)))
    val_idx = idx[:val_n]
    tr_idx = idx[val_n:]

    tokenizer = AutoTokenizer.from_pretrained(str(cfg.modernbert_model_name), trust_remote_code=True)

    # Encode pro train + val 
    enc = tokenizer(
        train_texts,
        truncation=True,
        padding="max_length",
        max_length=int(cfg.modernbert_max_seq_len),
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    y = torch.tensor(train_labels, dtype=torch.long)

    tr_ds = TensorDataset(input_ids[tr_idx], attention_mask[tr_idx], y[tr_idx])
    va_ds = TensorDataset(input_ids[val_idx], attention_mask[val_idx], y[val_idx])

    train_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    model = AutoModelForSequenceClassification.from_pretrained(
        str(cfg.modernbert_model_name),
        num_labels=2,
        trust_remote_code=True,
    ).to(device)

    optim = AdamW(model.parameters(), lr=float(cfg.modernbert_lr))

    scaler = None
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True) 
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=True) 

    def _eval_loader(loader: DataLoader) -> Tuple[float, float, float, List[float], List[int]]:
        model.eval()
        losses = []
        probs: List[float] = []
        gold: List[int] = []
        with torch.inference_mode():
            for ids, mask, yy in loader:
                ids = ids.to(device, non_blocking=pin)
                mask = mask.to(device, non_blocking=pin)
                yy = yy.to(device, non_blocking=pin)

                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = model(input_ids=ids, attention_mask=mask, labels=yy)
                else:
                    out = model(input_ids=ids, attention_mask=mask, labels=yy)

                loss = out.loss
                logits = out.logits
                p1 = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy().tolist()

                losses.append(float(loss.detach().cpu()))
                probs.extend([float(v) for v in p1])
                gold.extend([int(v) for v in yy.detach().cpu().numpy().tolist()])

        pred = [1 if v >= 0.5 else 0 for v in probs]
        acc = float(np.mean([int(a == b) for a, b in zip(gold, pred)])) if gold else float("nan")
        auc = _auc_roc_from_scores(gold, probs)
        avg_loss = float(np.mean(losses)) if losses else float("nan")
        return acc, auc, avg_loss, probs, gold

    epochs = int(cfg.modernbert_epochs)
    steps_total = len(train_loader)
    log_every = max(1, steps_total // 20)

    train_loss_ep: List[float] = []
    val_loss_ep: List[float] = []
    val_acc_ep: List[float] = []
    val_auc_ep: List[float] = []

    best_val_acc = -1.0
    best_state_dir = Path(cfg.output_dir) / "torch_models" / "modernbert_sentiment"
    best_state_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_steps = 0

        for step, (ids, mask, yy) in enumerate(train_loader, start=1):
            ids = ids.to(device, non_blocking=pin)
            mask = mask.to(device, non_blocking=pin)
            yy = yy.to(device, non_blocking=pin)

            optim.zero_grad(set_to_none=True)

            if use_amp:
                assert scaler is not None
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(input_ids=ids, attention_mask=mask, labels=yy)
                    loss = out.loss
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                out = model(input_ids=ids, attention_mask=mask, labels=yy)
                loss = out.loss
                loss.backward()
                optim.step()

            running_loss += float(loss.detach().cpu())
            running_steps += 1
            global_step += 1

            if step % log_every == 0 or step == steps_total:
                it_s = step / max(1e-9, (time.time() - t0))
                avg_loss = running_loss / max(1, running_steps)

                loss_trace.append({
                    "epoch": ep + 1,
                    "step": int(step),
                    "global_step": int(global_step),
                    "avg_loss": float(avg_loss),
                    "it_per_s": float(it_s),
                })

                logger.info(
                    "ModernBERT ep=%d step=%d/%d | avg_loss=%.4f | it/s=%.2f | device=%s",
                    ep + 1, step, steps_total, avg_loss, it_s, device,
                )

        # epoch aggregates
        avg_train_loss = float(running_loss / max(1, running_steps))
        val_acc, val_auc, val_loss, _, _ = _eval_loader(val_loader)

        epoch_trace.append({
            "epoch": ep + 1,
            "train_loss": float(avg_train_loss),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_auc": (float(val_auc) if not math.isnan(val_auc) else None),
        })


        train_loss_ep.append(avg_train_loss)
        val_loss_ep.append(float(val_loss))
        val_acc_ep.append(float(val_acc))
        val_auc_ep.append(float(val_auc))

        logger.info(
            "ModernBERT epoch=%d done in %.1fs | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | val_auc=%s",
            ep + 1,
            time.time() - t0,
            avg_train_loss,
            float(val_loss),
            float(val_acc),
            f"{val_auc:.4f}" if not math.isnan(val_auc) else "nan",
        )

        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            model.save_pretrained(str(best_state_dir))
            tokenizer.save_pretrained(str(best_state_dir))

    # Load best for test eval (re-load is optional; keep in-memory best is fine; we assume last saved is best)
    # Test eval
    test_enc = tokenizer(
        test_texts,
        truncation=True,
        padding="max_length",
        max_length=int(cfg.modernbert_max_seq_len),
        return_tensors="pt",
    )
    test_ids = test_enc["input_ids"]
    test_mask = test_enc["attention_mask"]
    test_y = torch.tensor(test_labels, dtype=torch.long)
    test_ds = TensorDataset(test_ids, test_mask, test_y)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    test_acc, test_auc, test_loss, test_probs, test_gold = _eval_loader(test_loader)

    # Training curves plots
    eda_dir = Path(cfg.output_dir) / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    epochs_x = list(range(1, len(train_loss_ep) + 1))
    loss_plot = plot_two_lines(
        epochs_x,
        train_loss_ep,
        val_loss_ep,
        label1="train_loss",
        label2="val_loss",
        title="ModernBERT training curve – loss",
        xlabel="Epoch",
        ylabel="Loss",
        out_path=eda_dir / "modernbert_loss_curve.png",
    )
    acc_plot = plot_two_lines(
        epochs_x,
        val_acc_ep,
        val_auc_ep,
        label1="val_acc",
        label2="val_auc",
        title="ModernBERT validation curve – metrics",
        xlabel="Epoch",
        ylabel="Value",
        out_path=eda_dir / "modernbert_val_metrics_curve.png",
    )

    # Test eval bundle (same file format as Spark eval)
    eval_bundle = evaluate_binary_from_arrays(
        probs=[float(x) for x in test_probs],
        labels=[int(x) for x in test_gold],
        name="modernbert",
        cfg=cfg,
        split="test",
        threshold=0.5,
    )

    metrics = {
        "generated_at": _utc_now_iso(),
        "model_name": str(cfg.modernbert_model_name),
        "device": device,
        "amp": bool(use_amp),
        "epochs": int(cfg.modernbert_epochs),
        "batch_size_used": int(batch_size),
        "max_seq_len": int(cfg.modernbert_max_seq_len),
        "train_split": bundle.meta,
        "train_curve": {
            "epochs": epochs_x,
            "train_loss": train_loss_ep,
            "val_loss": val_loss_ep,
            "val_acc": val_acc_ep,
            "val_auc": val_auc_ep,
            "plots": {"loss_curve": loss_plot, "val_metrics_curve": acc_plot},
        },
        "test_eval": {
            "accuracy": float(test_acc),
            "auc": float(test_auc),
            "loss": float(test_loss),
            "eval_json": "eval_modernbert_test.json",
            "eval_metrics": eval_bundle.get("metrics"),
        },
        "model_dir": str(best_state_dir),
    }

    _write_json(Path(cfg.output_dir) / "metrics" / "modernbert_sentiment.json", metrics)
    refresh_leaderboards(cfg)
    plots_dir = Path(cfg.output_dir) / "metrics" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    files = {}

    if loss_trace:
        xs = [d["global_step"] for d in loss_trace]
        ys = [d["avg_loss"] for d in loss_trace]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(xs, ys)
        ax.set_title("ModernBERT training loss (logged)")
        ax.set_xlabel("global_step")
        ax.set_ylabel("avg_loss")
        files["loss_curve"] = _save_fig(cfg, fig, plots_dir / "modernbert_loss_curve.png")

    if epoch_trace:
        ex = [d["epoch"] for d in epoch_trace]
        accs = [d["val_acc"] for d in epoch_trace]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ex, accs, marker="o")
        ax.set_title("ModernBERT validation accuracy by epoch")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val_accuracy")
        files["val_acc"] = _save_fig(cfg, fig, plots_dir / "modernbert_val_acc.png")

    return metrics


# =========================
# Report + manifest (renders all artifacts)
# =========================
def generate_html_report(cfg: Config) -> Path:
    metrics_dir = Path(cfg.output_dir) / "metrics"

    def _load(name: str) -> Optional[Dict[str, Any]]:
        p = metrics_dir / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    eda_basic = _load("eda_summary.json")
    eda_adv = _load("eda_advanced.json")
    sup = _load("supervised_models.json")
    sup_best = _load("supervised_best.json")
    sup_full = _load("supervised_full.json")
    clu = _load("cluster_kmeans.json")
    mb = _load("modernbert_sentiment.json")
    manifest = _load("run_manifest.json")

    html = _html_begin("Amazon Polarity – NLP Report")
    _html_h1(html, "Amazon Review Polarity – NLP Report")

    # --- Header KPIs (data-driven only)
    kpis = []
    if eda_basic and "total_rows" in eda_basic:
        kpis.append(f"<span class='kpi'>rows: <b>{_html_esc(eda_basic['total_rows'])}</b></span>")
    if sup_best and "winner" in sup_best:
        kpis.append(f"<span class='kpi'>supervised winner: <b>{_html_esc(sup_best['winner'])}</b></span>")
    if clu and "silhouette" in clu:
        kpis.append(f"<span class='kpi'>clustering silhouette: <b>{_html_esc(clu['silhouette'])}</b></span>")
    if mb and "best_val_accuracy" in mb:
        kpis.append(f"<span class='kpi'>ModernBERT best val acc: <b>{_html_esc(mb['best_val_accuracy'])}</b></span>")
    if kpis:
        _html_p(html, "".join(kpis))

    if manifest:
        _html_note(html, f"Generated at <b>{_html_esc(manifest.get('generated_at'))}</b>. Output dir: <code>{_html_esc(cfg.output_dir)}</code>")

    html.append("<div class='hr'></div>")

    # =========================
    # 1) EDA
    # =========================
    _html_h2(html, "1. EDA")

    if eda_basic:
        _html_note(html, "Basic EDA includes dataset split distribution, class balance, and missingness KPIs.")
        _html_table(html, eda_basic.get("by_dataset_split") or [], caption="Rows by dataset_split", max_rows=20)
        _html_table(html, eda_basic.get("by_sentiment") or [], caption="Rows by sentiment", max_rows=20)

        files = eda_basic.get("files") or {}
        for key, cap in [
            ("by_split_bar", "Rows by dataset_split"),
            ("by_sentiment_bar", "Rows by sentiment"),
            ("missing_bar", "Missing values per column"),
        ]:
            if files.get(key):
                _html_img(html, files[key], cap)

        qual = eda_basic.get("quality") or {}
        if qual:
            qual_rows = [{"metric": k, "value": v} for k, v in qual.items()]
            _html_table(html, qual_rows, caption="Quality KPIs", max_rows=50)
    else:
        _html_warn(html, "EDA basic metrics not found (dev profile may not have been run).")

    if eda_adv:
        q = eda_adv.get("text_len_quantiles") or {}
        if q:
            _html_table(html, [{"stat": k, "value": v} for k, v in q.items()], caption="Text length quantiles (tokens)", max_rows=20)
        ts = eda_adv.get("token_stats") or {}
        if ts:
            _html_table(html, [{"stat": k, "value": v} for k, v in ts.items()], caption="Token stats (sampled)", max_rows=20)

        files = eda_adv.get("files") or {}
        if files.get("text_len_hist"):
            _html_img(html, files["text_len_hist"], "Text length histogram (tokens)")

        for key, cap in [
            ("top_tokens_all", "Top tokens (all)"),
            ("top_tokens_pos", "Top tokens (positive)"),
            ("top_tokens_neg", "Top tokens (negative)"),
            ("wordcloud_all", "Wordcloud (all)"),
            ("wordcloud_pos", "Wordcloud (positive)"),
            ("wordcloud_neg", "Wordcloud (negative)"),
        ]:
            if files.get(key):
                _html_img(html, files[key], cap)
    else:
        _html_warn(html, "EDA advanced metrics not found.")

    html.append("<div class='hr'></div>")

    # =========================
    # 2) Supervised
    # =========================
    _html_h2(html, "2. Supervised models")

    if sup:
        # Build a compact table (data-driven)
        rows = []
        for name, m in sup.items():
            if name.startswith("_"):
                continue
            if not isinstance(m, dict):
                continue
            rows.append({
                "model": name,
                "auc_roc": m.get("auc_roc"),
                "accuracy": m.get("accuracy"),
                "f1": m.get("f1"),
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "seconds": m.get("seconds"),
            })
        rows = sorted(rows, key=lambda r: (r.get("auc_roc") is None, -(r.get("auc_roc") or -1)))
        _html_table(html, rows, caption="Supervised metrics (dev)", max_rows=20)

        # Comparison chart
        files = (sup.get("_files") or {})
        if files.get("comparison"):
            _html_img(html, files["comparison"], "Supervised metric comparison (AUC/Acc/F1)")

        # Confusion matrices
        for name, m in sup.items():
            if name.startswith("_") or not isinstance(m, dict):
                continue
            f = (m.get("files") or {}).get("confusion")
            if f:
                _html_img(html, f, f"Confusion matrix – {name}")

    else:
        _html_warn(html, "supervised_models.json not found.")

    if sup_best:
        _html_note(html, f"Winner (dev): <b>{_html_esc(sup_best.get('winner'))}</b> by <code>{_html_esc(sup_best.get('winner_metric'))}</code> = <b>{_html_esc(sup_best.get('winner_value'))}</b>.")

    if sup_full:
        _html_h3(html, "2.1 Full-data supervised (train_full)")
        _html_table(html, [{"k": k, "v": v} for k, v in sup_full.items() if k != "notes"], caption="train_full metrics", max_rows=50)

    html.append("<div class='hr'></div>")

    # =========================
    # 3) Clustering
    # =========================
    _html_h2(html, "3. Clustering")

    if clu:
        _html_note(html, f"Silhouette: <b>{_html_esc(clu.get('silhouette'))}</b> (distanceMeasure may vary by fallback).")
        prof = clu.get("cluster_profiles") or []
        _html_table(html, prof, caption="Cluster profiles", max_rows=50)

        files = clu.get("files") or {}
        if files.get("cluster_sizes"):
            _html_img(html, files["cluster_sizes"], "Cluster sizes")
        if files.get("cluster_mean_sentiment"):
            _html_img(html, files["cluster_mean_sentiment"], "Mean sentiment by cluster")
    else:
        _html_warn(html, "cluster_kmeans.json not found.")

    html.append("<div class='hr'></div>")

    # =========================
    # 4) ModernBERT
    # =========================
    _html_h2(html, "4. ModernBERT")

    if mb:
        _html_table(
            html,
            [{"metric": k, "value": v} for k, v in mb.items() if k not in ("training_history", "files")],
            caption="ModernBERT metrics",
            max_rows=50
        )
        files = mb.get("files") or {}
        if files.get("loss_curve"):
            _html_img(html, files["loss_curve"], "Training loss curve (logged)")
        if files.get("val_acc"):
            _html_img(html, files["val_acc"], "Validation accuracy by epoch")
    else:
        _html_warn(html, "modernbert_sentiment.json not found (ModernBERT likely disabled/skipped).")

    html.append("<div class='hr'></div>")

    # =========================
    # 5) Run config snapshot (traceability)
    # =========================
    _html_h2(html, "5. Run manifest (traceability)")
    if manifest:
        _html_note(html, "This section is purely traceability: it shows the actual config and file paths used in this run.")
        _html_p(html, "<details><summary>Config (raw)</summary><pre>" + _html_esc(json.dumps(manifest.get("config", {}), indent=2)) + "</pre></details>")
        _html_p(html, "<details><summary>Paths (raw)</summary><pre>" + _html_esc(json.dumps(manifest.get("paths", {}), indent=2)) + "</pre></details>")

    _html_end(html)

    out = Path(cfg.output_dir) / cfg.html_report_name
    out.write_text("".join(html), encoding="utf-8")
    return out


def write_run_manifest(cfg: Config) -> None:
    manifest = {
        "generated_at": _utc_now_iso(),
        "config": asdict(cfg),
        "paths": {
            "models_supervised": str(Path(cfg.output_dir) / "models" / "supervised_pipeline"),
            "models_cluster": str(Path(cfg.output_dir) / "models" / "cluster_pipeline"),
            "torch_modernbert": str(Path(cfg.output_dir) / "torch_models" / "modernbert_sentiment"),
            "metrics_dir": str(Path(cfg.output_dir) / "metrics"),
            "predictions_dir": str(Path(cfg.output_dir) / "predictions"),
            "report_html": str(Path(cfg.output_dir) / cfg.html_report_name),
        },
    }
    _write_json(Path(cfg.output_dir) / "metrics" / "run_manifest.json", manifest)

# =========================
# Report + manifest (renders all artifacts)
# Drop-in replacement for your current HTML helpers + generate_html_report + write_run_manifest.
# =========================

def _html_esc(x: Any) -> str:
    s = "" if x is None else str(x)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def _html_begin(title: str) -> List[str]:
    css = """
    :root{
      --fg:#222;--muted:#555;--line:#e6e6e6;--bg:#fff;
      --note:#f9fbff;--note-b:#9bbcff;--warn:#fff9f0;--warn-b:#ffb155
    }
    *{box-sizing:border-box}
    body{
      font-family:-apple-system,system-ui,Segoe UI,Roboto,Arial,sans-serif;
      margin:24px;color:var(--fg);background:var(--bg)
    }
    h1,h2,h3{margin:1.0em 0 .5em}
    p{margin:.6em 0;line-height:1.45}
    ul{margin:.2em 0 .8em 1.2em}
    table{border-collapse:collapse;width:100%;margin:8px 0 16px}
    th,td{border:1px solid var(--line);padding:6px 8px;font-size:13px;vertical-align:top}
    th{text-align:left;background:#f7f7f7;position:sticky;top:0}
    td.num{text-align:right;font-variant-numeric:tabular-nums}
    tbody tr:nth-child(even){background:#fbfbfb}
    .small{color:var(--muted);font-size:.9em}
    .note{background:var(--note);border-left:4px solid var(--note-b);padding:.6em .8em;margin:.6em 0}
    .warn{background:var(--warn);border-left:4px solid var(--warn-b);padding:.6em .8em;margin:.6em 0}
    .pill{display:inline-block;background:#eef;border:1px solid #dde;padding:2px 8px;border-radius:999px;margin:2px}
    .kpi{display:inline-block;margin:.2em .6em .2em 0;padding:.2em .6em;background:#f5f7ff;border:1px solid #dfe6ff;border-radius:6px}
    .hr{height:1px;background:var(--line);margin:16px 0}
    img{max-width:100%;height:auto;border:1px solid #eee;box-shadow:0 1px 2px rgba(0,0,0,.05);margin:6px 0}
    code{background:#f6f6f6;padding:1px 4px;border-radius:4px}
    pre{background:#f6f6f6;border:1px solid #eee;padding:10px 12px;border-radius:10px;overflow:auto}
    details{margin:.6em 0}
    summary{cursor:pointer;color:#1a4b9a}
    a{color:#1a4b9a;text-decoration:none}
    a:hover{text-decoration:underline}
    """
    out: List[str] = []
    out.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    out.append("<meta name='viewport' content='width=device-width, initial-scale=1'/>")
    out.append(f"<title>{_html_esc(title)}</title>")
    out.append(f"<style>{css}</style></head><body>")
    return out


def _html_end(html: List[str]) -> None:
    html.append("</body></html>")


def _html_hr(html: List[str]) -> None:
    html.append("<div class='hr'></div>")


def _html_h1(html: List[str], txt: str, *, anchor: Optional[str] = None) -> None:
    aid = f" id='{_html_esc(anchor)}'" if anchor else ""
    html.append(f"<h1{aid}>{_html_esc(txt)}</h1>")


def _html_h2(html: List[str], txt: str, *, anchor: Optional[str] = None) -> None:
    aid = f" id='{_html_esc(anchor)}'" if anchor else ""
    html.append(f"<h2{aid}>{_html_esc(txt)}</h2>")


def _html_h3(html: List[str], txt: str, *, anchor: Optional[str] = None) -> None:
    aid = f" id='{_html_esc(anchor)}'" if anchor else ""
    html.append(f"<h3{aid}>{_html_esc(txt)}</h3>")


def _html_p(html: List[str], txt: str, *, raw: bool = True) -> None:
    html.append(f"<p>{txt if raw else _html_esc(txt)}</p>")


def _html_note(html: List[str], txt: str, *, raw: bool = True) -> None:
    html.append(f"<p class='note'>{txt if raw else _html_esc(txt)}</p>")


def _html_warn(html: List[str], txt: str, *, raw: bool = True) -> None:
    html.append(f"<p class='warn'>{txt if raw else _html_esc(txt)}</p>")


def _html_list(html: List[str], items: List[str], *, raw: bool = True) -> None:
    if not items:
        html.append("<p class='small'>Lista vazia.</p>")
        return
    li = "".join(f"<li>{it if raw else _html_esc(it)}</li>" for it in items)
    html.append(f"<ul>{li}</ul>")


def _html_codeblock(html: List[str], code: str) -> None:
    html.append("<pre><code>" + _html_esc(code) + "</code></pre>")


def _looks_like_number(v: Any) -> bool:
    if isinstance(v, (int, float, np.integer, np.floating)):
        return True
    if v is None:
        return False
    s = str(v).strip()
    if not s:
        return False
    try:
        float(s)
        return True
    except Exception:
        return False


def _fmt_num(v: Any, *, digits: int = 4) -> str:
    if v is None:
        return ""
    if isinstance(v, (np.integer,)):
        return str(int(v))
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Inf" if v > 0 else "-Inf"
        return f"{v:.{digits}f}"
    return str(v)


def _html_table(
    html: List[str],
    rows: List[Dict[str, Any]],
    caption: Optional[str] = None,
    *,
    max_rows: int = 50,
    digits: int = 4,
) -> None:
    if caption:
        html.append(f"<h3>{_html_esc(caption)}</h3>")
    if not rows:
        html.append("<p class='small'>Sem linhas para exibir.</p>")
        return

    rows = rows[:max_rows]
    cols = list(rows[0].keys())

    # Decide numeric columns (right align) if most values look numeric.
    num_cols: set[str] = set()
    for c in cols:
        vals = [r.get(c) for r in rows]
        ok = sum(1 for v in vals if v is not None)
        if ok == 0:
            continue
        numeric_like = sum(1 for v in vals if _looks_like_number(v))
        if numeric_like / ok >= 0.8:
            num_cols.add(c)

    html.append("<table><thead><tr>" + "".join(f"<th>{_html_esc(c)}</th>" for c in cols) + "</tr></thead><tbody>")
    for r in rows:
        tds = []
        for c in cols:
            v = r.get(c, "")
            cls = " class='num'" if c in num_cols else ""
            cell = _fmt_num(v, digits=digits) if c in num_cols else ("" if v is None else str(v))
            tds.append(f"<td{cls}>{_html_esc(cell)}</td>")
        html.append("<tr>" + "".join(tds) + "</tr>")
    html.append("</tbody></table>")


def _html_img(html: List[str], src: str, caption: Optional[str] = None) -> None:
    if not src:
        return
    cap = f"<b>{_html_esc(caption)}</b><br/>" if caption else ""
    html.append(f"<p>{cap}<img src='{_html_esc(src)}' alt='img'></p>")


def _rel_to_output(cfg: "Config", p: Path) -> str:
    try:
        return str(p.relative_to(Path(cfg.output_dir)))
    except Exception:
        return str(p)


def _resolve_asset(cfg: "Config", ref: Optional[str]) -> Optional[str]:
    """
    Makes image refs robust across your code paths:
    - accepts absolute paths
    - accepts "eda/foo.png" relative paths
    - accepts bare filenames ("foo.png") and searches common output dirs.
    Returns path relative to cfg.output_dir (so HTML at output_dir can resolve it).
    """
    if not ref:
        return None
    s = str(ref).strip()
    if not s:
        return None
    if s.startswith("http://") or s.startswith("https://"):
        return s

    base = Path(cfg.output_dir).resolve()

    # Absolute path?
    try:
        p = Path(s)
        if p.is_absolute() and p.exists():
            return _rel_to_output(cfg, p.resolve())
    except Exception:
        pass

    # Direct relative to output_dir?
    p0 = (base / s).resolve()
    if p0.exists():
        return _rel_to_output(cfg, p0)

    # If ref is just a filename, search common subdirs
    name = Path(s).name
    candidates = [
        base / name,
        base / "eda" / name,
        base / "metrics" / name,
        base / "metrics" / "plots" / name,
        base / "artifacts" / name,
        base / "predictions" / name,
        base / "torch_models" / name,
    ]
    for c in candidates:
        if c.exists():
            return _rel_to_output(cfg, c.resolve())

    # Last resort: shallow recursive search (keeps report resilient)
    try:
        for c in base.rglob(name):
            if c.is_file():
                return _rel_to_output(cfg, c.resolve())
    except Exception:
        pass

    return s  # fallback (may still work if user opened HTML elsewhere)


def _read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        return v
    except Exception:
        return None


class _ReportBuilder:
    def __init__(self, cfg: "Config", title: str) -> None:
        self.cfg = cfg
        self.html = _html_begin(title)
        self.fig_no = 0
        self.tbl_no = 0

    def h1(self, t: str) -> None:
        _html_h1(self.html, t)

    def h2(self, t: str) -> None:
        _html_h2(self.html, t)

    def h3(self, t: str) -> None:
        _html_h3(self.html, t)

    def p(self, t: str) -> None:
        _html_p(self.html, t, raw=True)

    def note(self, t: str) -> None:
        _html_note(self.html, t, raw=True)

    def warn(self, t: str) -> None:
        _html_warn(self.html, t, raw=True)

    def lst(self, items: List[str]) -> None:
        _html_list(self.html, items, raw=True)

    def code(self, code: str) -> None:
        _html_codeblock(self.html, code)

    def hr(self) -> None:
        _html_hr(self.html)

    def fig(self, ref: Optional[str], caption: str) -> None:
        src = _resolve_asset(self.cfg, ref)
        if not src:
            return
        self.fig_no += 1
        _html_img(self.html, src, f"Figura {self.fig_no} – {caption}")

    def table(self, rows: List[Dict[str, Any]], caption: str, *, max_rows: int = 50, digits: int = 4) -> None:
        self.tbl_no += 1
        _html_table(self.html, rows, f"Tabela {self.tbl_no} – {caption}", max_rows=max_rows, digits=digits)

    def finish(self, out_path: Path) -> Path:
        _html_end(self.html)
        out_path.write_text("".join(self.html), encoding="utf-8")
        return out_path


def generate_html_report(cfg: "Config") -> Path:
    """
    Generates a full HTML report with an R-like structure (1..7 sections),
    resilient to missing files and to inconsistent artifact paths.
    """
    out_dir = Path(cfg.output_dir)
    metrics_dir = out_dir / "metrics"

    # Load what exists (report stays stable even when some runs skipped)
    eda_basic = _read_json(metrics_dir / "eda_summary.json") or {}
    eda_adv = _read_json(metrics_dir / "eda_advanced.json") or {}
    eda_decisions = _read_json(metrics_dir / "eda_decisions_applied.json") or {}
    cfg_final = _read_json(metrics_dir / "config_final.json") or {}
    sup = _read_json(metrics_dir / "supervised_models.json") or {}
    sup_best = _read_json(metrics_dir / "supervised_best.json") or {}
    sup_full = _read_json(metrics_dir / "supervised_full.json") or {}
    clu = _read_json(metrics_dir / "cluster_kmeans.json") or {}
    mb = _read_json(metrics_dir / "modernbert_sentiment.json") or {}
    leader = _read_json(metrics_dir / "leaderboards.json") or {}
    manifest = _read_json(metrics_dir / "run_manifest.json") or {}

    rb = _ReportBuilder(cfg, "Amazon Review Polarity – Relatório Final")
    rb.h1("Amazon Review Polarity – Relatório Final")

    # =========================
    # 1. Introdução e contexto
    # =========================
    rb.h2("1. Introdução e contexto")
    rb.p(
        "Este relatório consolida a execução do pipeline de NLP para <b>classificação de polaridade</b> em reviews. "
        "O fluxo cobre ingestão/limpeza, EDA, modelos supervisionados (TF‑IDF + ML), análise não supervisionada (clusters) "
        "e (opcionalmente) fine‑tuning de um Transformer (<code>ModernBERT</code>)."
    )

    rb.h3("1.1 Objetivos do projeto")
    rb.lst(
        [
            "Padronizar e versionar a ingestão dos CSVs (com suporte a multiline e cache em Parquet).",
            "Explorar dados (split, balanceamento, missingness, tamanho de texto, vocabulário) com artefatos reproduzíveis.",
            "Treinar e comparar modelos supervisionados (LogReg e RandomForest) em TF‑IDF (HashingTF + IDF).",
            "Opcional: segmentar o espaço de textos via PCA + k‑means e inspecionar perfis por cluster.",
            "Opcional: treinar ModernBERT para benchmark de um modelo contextual (torch/transformers).",
            "Gerar <code>models/</code>, <code>metrics/</code>, <code>eda/</code> e um HTML final para auditoria."
        ]
    )

    rb.h3("1.2 Visão geral (resultados desta execução)")
    kpis: List[str] = []

    total_rows = eda_basic.get("total_rows") or eda_basic.get("rows") or None
    if total_rows is not None:
        kpis.append(f"<span class='kpi'>rows: <b>{_html_esc(total_rows)}</b></span>")

    # Best supervised (dev winner)
    winner = sup_best.get("winner")
    winner_val = sup_best.get("winner_value")
    if winner:
        kpis.append(f"<span class='kpi'>winner (dev): <b>{_html_esc(winner)}</b></span>")
    if winner_val is not None:
        kpis.append(f"<span class='kpi'>AUC (dev): <b>{_html_esc(_fmt_num(winner_val, digits=4))}</b></span>")

    # Clustering silhouette (key differs across versions)
    sil = clu.get("silhouette_final")
    if sil is None:
        sil = clu.get("silhouette")
    if sil is not None:
        kpis.append(f"<span class='kpi'>silhouette: <b>{_html_esc(_fmt_num(sil, digits=4))}</b></span>")

    # ModernBERT test metrics
    mb_test = (mb.get("test_eval") or {}).get("eval_metrics") or (mb.get("test_eval") or {}).get("metrics") or {}
    mb_acc = _safe_float(mb_test.get("accuracy"))
    mb_auc = _safe_float(mb_test.get("auc_roc")) or _safe_float((mb.get("test_eval") or {}).get("auc"))
    if mb_acc is not None:
        kpis.append(f"<span class='kpi'>ModernBERT acc: <b>{_html_esc(_fmt_num(mb_acc, digits=4))}</b></span>")
    if mb_auc is not None:
        kpis.append(f"<span class='kpi'>ModernBERT AUC: <b>{_html_esc(_fmt_num(mb_auc, digits=4))}</b></span>")

    if kpis:
        rb.p("".join(kpis))

    gen_at = manifest.get("generated_at") or eda_basic.get("generated_at") or sup_best.get("generated_at") or _utc_now_iso()
    rb.note(
        f"Gerado em <b>{_html_esc(gen_at)}</b>. Output dir: <code>{_html_esc(cfg.output_dir)}</code>."
    )

    rb.hr()

    # =========================
    # 2. Descrição dos dados e EDA
    # =========================
    rb.h2("2. Descrição dos dados e EDA")

    rb.h3("2.1 Estrutura geral dos dados")
    rb.table(
        [
            {"linhas": int(total_rows) if total_rows is not None else "", "colunas_modeladas": 4},
        ],
        "Dimensão (visão do pipeline)",
        max_rows=10,
        digits=0,
    )
    rb.note(
        "O dataset modelado pelo pipeline utiliza <code>review_id</code>, <code>dataset_split</code>, "
        "<code>sentiment</code> (0=neg, 1=pos) e <code>text_full</code> (title+text)."
    )

    # Basic EDA tables
    by_split = eda_basic.get("by_dataset_split") or []
    by_sent = eda_basic.get("by_sentiment") or []
    if by_split:
        rb.table(by_split, "Distribuição por dataset_split", max_rows=20, digits=0)
    if by_sent:
        rb.table(by_sent, "Distribuição por sentiment", max_rows=20, digits=0)

    # Basic EDA plots (support both schemas: "files" or "plots")
    plots_basic = eda_basic.get("files") or eda_basic.get("plots") or {}
    rb.fig(plots_basic.get("by_split_bar") or plots_basic.get("split_counts"), "Rows por dataset_split")
    rb.fig(plots_basic.get("by_sentiment_bar") or plots_basic.get("sentiment_counts"), "Balanceamento de classes (sentiment)")
    rb.fig(plots_basic.get("missing_bar") or plots_basic.get("missing_rates"), "Missingness (colunas)")

    # Quality KPIs
    qual = eda_basic.get("quality") or {}
    if qual:
        rb.table([{"metric": k, "value": v} for k, v in qual.items()], "KPIs de qualidade (agregados)", max_rows=80, digits=4)
    miss_rates = eda_basic.get("missing_rates") or {}
    if miss_rates and not qual:
        rb.table([{"coluna": k, "taxa_missing": v} for k, v in miss_rates.items()], "Taxa de missing (fração)", max_rows=80, digits=4)

    rb.h3("2.2 Distribuição de tamanho de texto e tokens (amostragem)")
    # Advanced EDA tables/plots (support both key variants)
    q = eda_adv.get("text_len_quantiles") or eda_adv.get("quantiles_text_len_tokens") or {}
    if q:
        rb.table([{"estatística": k, "valor": v} for k, v in q.items()], "Quantis de tamanho do texto (tokens)", max_rows=20, digits=0)

    sug = eda_adv.get("suggested") or {}
    if sug:
        rb.note(
            "Decisões sugeridas pela EDA (data-driven): "
            f"<span class='pill'>modernbert_max_seq_len={_html_esc(sug.get('modernbert_max_seq_len'))}</span> "
            f"<span class='pill'>hashing_num_features={_html_esc(sug.get('hashing_num_features'))}</span>"
        )

    tok_stats = eda_adv.get("token_stats") or {}
    if tok_stats:
        rb.table([{"estatística": k, "valor": v} for k, v in tok_stats.items()], "Estatísticas de tokens (amostra)", max_rows=30, digits=4)

    files_adv = eda_adv.get("files") or eda_adv.get("plots") or {}
    rb.fig(files_adv.get("text_len_hist") or files_adv.get("text_len_histogram") or files_adv.get("text_len_hist_plot"), "Histograma de tamanho de texto (tokens)")

    rb.h3("2.3 EDA de texto – tokens mais frequentes")
    rb.fig(files_adv.get("top_tokens_all") or files_adv.get("top_tokens_all.png"), "Top tokens (geral)")
    rb.fig(files_adv.get("top_tokens_pos") or files_adv.get("top_tokens_positive") or files_adv.get("top_tokens_positive.png"), "Top tokens (sentiment=1)")
    rb.fig(files_adv.get("top_tokens_neg") or files_adv.get("top_tokens_negative") or files_adv.get("top_tokens_negative.png"), "Top tokens (sentiment=0)")
    rb.fig(files_adv.get("wordcloud_all"), "Wordcloud (geral)")
    rb.fig(files_adv.get("wordcloud_pos"), "Wordcloud (sentiment=1)")
    rb.fig(files_adv.get("wordcloud_neg"), "Wordcloud (sentiment=0)")

    rb.hr()

    # =========================
    # 3. Pré-processamento e decisões
    # =========================
    rb.h2("3. Pré-processamento e decisões")

    rb.h3("3.1 Decisões aplicadas (EDA → config)")
    if eda_decisions:
        rb.table([{"chave": k, "valor": v} for k, v in eda_decisions.items()], "Snapshot de decisões aplicadas", max_rows=80, digits=4)
    else:
        rb.warn("Arquivo <code>eda_decisions_applied.json</code> não encontrado (provável execução sem EDA).")

    rb.h3("3.2 Config final (traceável)")
    cfg_src = cfg_final or manifest.get("config") or {}
    if cfg_src:
        # Show a compact subset first
        keys_focus = [
            "run_profile",
            "max_polarity_rows",
            "hashing_num_features",
            "train_split_fraction",
            "cv_folds",
            "cv_parallelism",
            "enable_rf",
            "enable_clustering",
            "pca_var_target",
            "kmeans_k_max",
            "enable_modernbert",
            "modernbert_model_name",
            "modernbert_max_seq_len",
            "modernbert_batch_size",
        ]
        compact = [{"param": k, "value": cfg_src.get(k)} for k in keys_focus if k in cfg_src]
        if compact:
            rb.table(compact, "Parâmetros principais", max_rows=80, digits=6)
        rb.p("<details><summary>Config (raw)</summary><pre>" + _html_esc(json.dumps(cfg_src, indent=2)) + "</pre></details>")
    else:
        rb.warn("Não foi possível carregar snapshot de configuração (config_final.json / run_manifest.json ausentes).")

    rb.hr()

    # =========================
    # 4. Modelagem supervisionada
    # =========================
    rb.h2("4. Modelagem supervisionada – classificação de sentimento")

    rb.h3("4.1 Comparação em dev (CV)")
    if sup:
        rows = []
        for name, m in sup.items():
            if name.startswith("_") or not isinstance(m, dict):
                continue
            bp = m.get("best_params") or {}
            rows.append(
                {
                    "model": name,
                    "auc_roc": m.get("auc_roc"),
                    "accuracy": m.get("accuracy"),
                    "f1": m.get("f1"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "specificity": m.get("specificity"),
                    "balanced_acc": m.get("balanced_accuracy"),
                    "seconds": m.get("seconds"),
                    "best_params": ", ".join(f"{k}={v}" for k, v in bp.items()) if bp else "",
                }
            )
        rows = sorted(rows, key=lambda r: (r.get("auc_roc") is None, -(r.get("auc_roc") or -1.0)))
        rb.table(rows, "Métricas (dev) – modelos supervisionados", max_rows=20, digits=6)

        # Comparison plot (supports both keys)
        sup_files = sup.get("_files") or {}
        rb.fig(sup_files.get("comparison"), "Comparação (AUC/Accuracy/F1)")

        # Confusion matrices
        for name, m in sup.items():
            if name.startswith("_") or not isinstance(m, dict):
                continue
            cm = (m.get("files") or {}).get("confusion")
            rb.fig(cm, f"Matriz de confusão – {name}")

        if sup_best:
            rb.note(
                "Vencedor em dev: "
                f"<b>{_html_esc(sup_best.get('winner'))}</b> por "
                f"<code>{_html_esc(sup_best.get('winner_metric'))}</code> = "
                f"<b>{_html_esc(_fmt_num(sup_best.get('winner_value'), digits=6))}</b>."
            )
    else:
        rb.warn("Arquivo <code>supervised_models.json</code> não encontrado.")

    rb.h3("4.2 Treino full (train_full) e avaliação detalhada (quando disponível)")
    if sup_full:
        # Show high-level fields
        keep = {k: v for k, v in sup_full.items() if k not in ("notes",)}
        rb.table([{"campo": k, "valor": v} for k, v in keep.items()], "Resumo train_full", max_rows=80, digits=6)

        # If eval bundle exists, show plots (they live under eda/)
        eval_json_name = sup_full.get("eval_json")
        if isinstance(eval_json_name, str) and eval_json_name:
            eval_obj = _read_json(metrics_dir / eval_json_name) or {}
            if eval_obj.get("metrics"):
                rb.table(
                    [{"métrica": k, "valor": v} for k, v in (eval_obj.get("metrics") or {}).items()],
                    "Métricas de teste (train_full)",
                    max_rows=50,
                    digits=6,
                )
            plots = eval_obj.get("plots") or {}
            rb.fig(plots.get("cm"), "Confusion matrix (train_full)")
            rb.fig(plots.get("roc"), "ROC (train_full)")
            rb.fig(plots.get("pr"), "PR (train_full)")
            rb.fig(plots.get("calibration"), "Calibração (train_full)")
            rb.fig(plots.get("f1_threshold"), "F1 vs threshold (train_full)")
    else:
        rb.note(
            "Nenhum resultado de <code>train_full</code> foi encontrado. "
            "Se você executar com <code>AMAZON_NLP_RUN_PROFILE=train_full</code>, "
            "o relatório passa a incluir ROC/PR/calibração."
        )

    rb.hr()

    # =========================
    # 5. Análise não supervisionada
    # =========================
    rb.h2("5. Análise não supervisionada – clusters")
    if clu:
        k_chosen = clu.get("kmeans_k_chosen")
        pca_used = clu.get("pca_used")
        pca_k = clu.get("pca_k_chosen")
        sil = clu.get("silhouette_final") if clu.get("silhouette_final") is not None else clu.get("silhouette")
        rb.note(
            "Pipeline: TF‑IDF normalizado"
            + (" + PCA" if pca_used else "")
            + f" + k‑means. "
            f"k escolhido = <b>{_html_esc(k_chosen)}</b> | "
            f"silhouette ≈ <b>{_html_esc(_fmt_num(sil, digits=6))}</b> | "
            f"PCA k = <b>{_html_esc(pca_k)}</b>."
        )

        prof = clu.get("cluster_profiles") or []
        if prof:
            rb.table(prof, "Perfil por cluster (n, mean_sentiment)", max_rows=80, digits=6)

        # Plots may be under "plots" or "files" depending on your run; support both.
        cplots = clu.get("files") or clu.get("plots") or {}
        rb.fig(cplots.get("pca_cumvar"), "PCA – variância acumulada (clustering)")
        rb.fig(cplots.get("silhouette_by_k"), "k‑means – silhouette por k")
        rb.fig(cplots.get("cluster_sizes"), "Tamanho dos clusters")
        rb.fig(cplots.get("cluster_mean_sentiment"), "Sentimento médio por cluster")

        # If the code saved extra charts but didn’t write them to JSON, try to pick them up by name.
        rb.fig("cluster_mean_sentiment.png", "Sentimento médio por cluster (fallback)")
    else:
        rb.note(
            "Nenhuma métrica de clustering foi encontrada. "
            "Em <code>dev</code>, certifique-se de que <code>enable_clustering=True</code>."
        )

    rb.hr()

    # =========================
    # 6. Aplicação: scoring e artefatos
    # =========================
    rb.h2("6. Aplicação: scoring e artefatos")

    paths = (manifest.get("paths") or {}) if manifest else {}
    rb.table(
        [
            {"artefato": "pipeline supervisionado (Spark)", "path": paths.get("models_supervised") or str(out_dir / "models" / "supervised_pipeline")},
            {"artefato": "pipeline clustering (Spark)", "path": paths.get("models_cluster") or str(out_dir / "models" / "cluster_pipeline")},
            {"artefato": "ModernBERT (torch)", "path": paths.get("torch_modernbert") or str(out_dir / "torch_models" / "modernbert_sentiment")},
            {"artefato": "metrics dir", "path": paths.get("metrics_dir") or str(out_dir / "metrics")},
            {"artefato": "predictions dir", "path": paths.get("predictions_dir") or str(out_dir / "predictions")},
        ],
        "Principais paths gerados",
        max_rows=20,
        digits=4,
    )

    rb.h3("6.1 Como carregar e pontuar novos textos (Spark)")
    rb.code(
        """from pyspark.ml import PipelineModel
from pyspark.sql import functions as F

pm = PipelineModel.load("<output_dir>/models/supervised_pipeline")
df_new = spark.createDataFrame([("some review text here",)], ["text_full"])

pred = pm.transform(df_new)
pred.select(
    "text_full",
    F.col("prediction").cast("int").alias("prediction"),
    F.col("probability")
).show(truncate=False)"""
    )

    rb.h3("6.2 Scoring em lote (score_only)")
    rb.p(
        "Para pontuar rapidamente (sem re-treino), use o perfil <code>score_only</code>. "
        "O pipeline escreve previsões em <code>predictions/</code> (Parquet)."
    )
    rb.code(
        """import os
os.environ["AMAZON_NLP_RUN_PROFILE"] = "score_only"
os.environ["AMAZON_NLP_ENABLE_PARQUET_CACHE"] = "1"
os.environ["AMAZON_NLP_SCORE_CLUSTERS"] = "1"   # opcional
pipeline.run(force=False, run_modernbert=False)"""
    )

    rb.hr()

    # =========================
    # 7. Conclusões, limitações e trabalhos futuros
    # =========================
    rb.h2("7. Conclusões, limitações e trabalhos futuros")

    # Build a data-driven summary sentence
    summary_bits: List[str] = []
    if winner:
        summary_bits.append(f"Em <code>dev</code>, o melhor supervisionado foi <b>{_html_esc(winner)}</b>")
    if winner_val is not None:
        summary_bits.append(f"(AUC ≈ <b>{_html_esc(_fmt_num(winner_val, digits=4))}</b>)")
    if sil is not None:
        summary_bits.append(f"Clustering apresentou silhouette ≈ <b>{_html_esc(_fmt_num(sil, digits=4))}</b>")
    if mb_acc is not None:
        summary_bits.append(f"ModernBERT atingiu accuracy ≈ <b>{_html_esc(_fmt_num(mb_acc, digits=4))}</b> em teste (se executado)")
    if summary_bits:
        rb.note(" | ".join(summary_bits) + ".")

    rb.h3("7.1 Limitações")
    rb.lst(
        [
            "TF‑IDF com HashingTF sofre colisões; dimensões maiores reduzem colisões ao custo de memória/tempo.",
            "A avaliação em dev usa split determinístico por hash (quando não há train/test explícito); isso facilita reprodutibilidade, mas pode induzir viés se houver duplicatas.",
            "Curvas completas (ROC/PR/calibração) só aparecem quando há <code>eval_*.json</code> (ex.: train_full e ModernBERT).",
            "Clustering em textos curtos tende a silhouette baixo: clusters são úteis como segmentação exploratória, não como classes rígidas."
        ]
    )

    rb.h3("7.2 Próximos passos")
    rb.lst(
        [
            "Adicionar avaliação detalhada (ROC/PR/calibração) também para os modelos de dev (LR/RF) para fechar o loop de comparação.",
            "Persistir amostras de textos representativos por cluster (ex.: top‑N por proximidade ao centróide) para interpretação humana.",
            "Adicionar validação de duplicatas/near-duplicates (ex.: MinHash/LSH) antes do split.",
            "Se GPU disponível, ampliar treino do ModernBERT (epochs>1) e considerar early stopping por AUC."
        ]
    )

    # Traceability (appendix-like, still within section 7 to keep 1..7 like the R structure)
    rb.h3("7.3 Traceabilidade (manifest)")
    if manifest:
        rb.note("Esta seção existe apenas para auditoria: config e paths reais usados na execução.")
        rb.p("<details><summary>Manifest (raw)</summary><pre>" + _html_esc(json.dumps(manifest, indent=2)) + "</pre></details>")
    else:
        rb.warn("run_manifest.json não encontrado.")

    out_path = out_dir / cfg.html_report_name
    return rb.finish(out_path)

def write_run_manifest(cfg: "Config") -> None:
    manifest = {
        "generated_at": _utc_now_iso(),
        "config": asdict(cfg),
        "paths": {
            "models_supervised": str(Path(cfg.output_dir) / "models" / "supervised_pipeline"),
            "models_cluster": str(Path(cfg.output_dir) / "models" / "cluster_pipeline"),
            "torch_modernbert": str(Path(cfg.output_dir) / "torch_models" / "modernbert_sentiment"),
            "metrics_dir": str(Path(cfg.output_dir) / "metrics"),
            "predictions_dir": str(Path(cfg.output_dir) / "predictions"),
            "report_html": str(Path(cfg.output_dir) / cfg.html_report_name),
        },
    }
    _write_json(Path(cfg.output_dir) / "metrics" / "run_manifest.json", manifest)


# Orchestrator
class AmazonPolarityNotebookPipeline:
    def __init__(self, *, test_csv: str, train_csv: Optional[str], output_dir: str):
        self.test_csv = test_csv
        self.train_csv = train_csv
        self.output_dir = output_dir

    def _make_cfg(self, run_modernbert: bool) -> Config:
        cfg = Config(
            polarity_test_path=self.test_csv,
            polarity_train_path=self.train_csv,
            output_dir=self.output_dir,
            enable_modernbert=bool(run_modernbert),
        )
        cfg.ensure_output_dirs()
        setup_logging(cfg.output_dir)
        tune_config(cfg)
        return cfg

    def run(self, *, force: bool = False, run_modernbert: bool = True) -> None:
        if force and Path(self.output_dir).exists():
            shutil.rmtree(self.output_dir, ignore_errors=True)
            cfg.ensure_output_dirs()

        cfg = self._make_cfg(run_modernbert=run_modernbert)
        torch_bundle: Optional[TorchSplitBundle] = None

        with spark_session(cfg) as spark:
            if cfg.run_profile == "dev":
                torch_bundle = self._run_dev(spark, cfg)
            elif cfg.run_profile == "train_full":
                self._run_train_full(spark, cfg)
            elif cfg.run_profile == "score_only":
                self._run_score_only(spark, cfg)
            else:
                raise ValueError(f"Unsupported run_profile: {cfg.run_profile}")

        # Spark is stopped here ✅
        if cfg.run_profile == "dev" and cfg.enable_modernbert and torch_bundle is not None:
            device = _best_torch_device()
            if device == "cpu" and not _env_bool("AMAZON_NLP_ENABLE_MODERNBERT_CPU", False):
                logger.warning("ModernBERT skipped on CPU (set AMAZON_NLP_ENABLE_MODERNBERT_CPU=1 to force).")
            else:
                train_modernbert_sentiment_from_splits(torch_bundle, cfg)

        # Ensure leaderboards include everything that ran (incl. ModernBERT)
        refresh_leaderboards(cfg)

        write_run_manifest(cfg)
        report = generate_html_report(cfg)
        logger.info("Report: %s", str(report))

    def _run_dev(self, spark: SparkSession, cfg: Config) -> Optional[TorchSplitBundle]:
        logger.info("=== RUN: dev ===")
        df = load_polarity_dataset(spark, cfg)
        df = _maybe_fix_underpartitioning_dev(df, cfg).persist(StorageLevel.MEMORY_AND_DISK)

        logger.info("dev: df partitions=%d", int(df.rdd.getNumPartitions()))
        total_rows = int(df.count())
        logger.info("dev: dataset rows=%d", total_rows)

        eda_adv: Optional[Dict[str, Any]] = None
        if cfg.enable_eda:
            logger.info("dev: EDA ...")
            run_basic_eda(df, cfg, total_rows)
            eda_adv = run_advanced_eda(df, cfg, total_rows)
            apply_data_driven_config(cfg, eda_adv)

        logger.info("dev: supervised CV ...")
        df_bucket = _add_split_bucket(df, cfg).select("dataset_split", "sentiment", "text_full", "split_bucket")

        # Collect train/test texts for ModernBERT using the same split logic as supervised
        torch_bundle: Optional[TorchSplitBundle] = None
        if cfg.enable_modernbert:
            logger.info("dev: collecting train/test splits for ModernBERT (training happens after Spark stops) ...")
            torch_bundle = collect_train_test_texts_labels(df_bucket, cfg)
            _write_json(Path(cfg.output_dir) / "metrics" / "modernbert_split_meta.json", {"generated_at": _utc_now_iso(), **torch_bundle.meta})

        features_df, idf_model = fit_tfidf_features(df_bucket, cfg)
        _, best_model, best_name = train_supervised_models_fast(features_df, cfg)
        save_best_supervised_pipeline(idf_model, best_model, cfg)
        logger.info("dev: supervised winner=%s", best_name)

        if cfg.enable_clustering:
            logger.info("dev: clustering pipeline (data-driven) ...")
            fit_and_save_cluster_pipeline(spark, df, cfg)

        df.unpersist()
        return torch_bundle

    def _run_train_full(self, spark: SparkSession, cfg: Config) -> None:
        logger.info("=== RUN: train_full ===")
        df = load_polarity_dataset(spark, cfg)
        train_full_supervised_pipeline(spark, df, cfg)

    def _run_score_only(self, spark: SparkSession, cfg: Config) -> None:
        logger.info("=== RUN: score_only ===")
        df = load_polarity_dataset(spark, cfg)

        pred = score_with_saved_supervised_model(spark, df, cfg)
        out_path = Path(cfg.output_dir) / "predictions" / "supervised_predictions.parquet"
        pred.write.mode("overwrite").parquet(str(out_path))
        logger.info("score_only: wrote supervised predictions: %s", str(out_path))

        if _env_bool("AMAZON_NLP_SCORE_CLUSTERS", False):
            clu = score_with_saved_cluster_model(spark, df, cfg)
            clu_path = Path(cfg.output_dir) / "predictions" / "cluster_assignments.parquet"
            clu.write.mode("overwrite").parquet(str(clu_path))
            logger.info("score_only: wrote cluster assignments: %s", str(clu_path))


pipeline = AmazonPolarityNotebookPipeline(test_csv=TEST_CSV, train_csv=TRAIN_CSV, output_dir=OUTPUT_DIR)

# First time (dev trains + saves models; ModernBERT trains after Spark stops)
os.environ["AMAZON_NLP_RUN_PROFILE"] = "dev"
pipeline.run(force=FORCE_RERUN, run_modernbert=RUN_MODERNBERT)

# Later runs (fast scoring)
# os.environ["AMAZON_NLP_RUN_PROFILE"] = "score_only"
# os.environ["AMAZON_NLP_ENABLE_PARQUET_CACHE"] = "1"
# os.environ["AMAZON_NLP_SCORE_CLUSTERS"] = "1"
# pipeline.run(force=False, run_modernbert=False)
