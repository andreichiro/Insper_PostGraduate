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
import inspect
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
from pyspark.ml.feature import (
    RegexTokenizer, StopWordsRemover, HashingTF, IDF, IDFModel, Normalizer, PCA, SQLTransformer
)
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator, ClusteringEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.functions import vector_to_array  # Spark 3.x
from pyspark.mllib.evaluation import BinaryClassificationMetrics

try:
    from pyspark.ml.functions import array_to_vector  # spark 3 1 ou mais
except Exception:
    array_to_vector = None  

from pyspark.ml.linalg import Vectors, VectorUDT


import platform
import threading
import gc
import uuid
import multiprocessing as mp

# importa torch so quando precisa pra evitar crash no mac
torch = None  
_TORCH_IMPORT_ERROR: Optional[BaseException] = None

def _ensure_torch_imported() -> None:
    """Importa torch so quando precisa pra evitar dor de cabeca no mac."""
    global torch, _TORCH_IMPORT_ERROR
    if torch is not None:
        return
    try:
        import torch as _torch  
        torch = _torch
        _TORCH_IMPORT_ERROR = None
    except Exception as e:
        torch = None  
        _TORCH_IMPORT_ERROR = e

logger = logging.getLogger("amazon_nlp_pipeline")

# entradas e padroes
TEST_CSV = "/Users/akatsurada/Documents/INSPER/BigData/ProjetoFinal/test.csv"
TRAIN_CSV: Optional[str] = None
OUTPUT_DIR = "./models_output"
FORCE_RERUN = False
RUN_MODERNBERT = True

default_profile = "score_only" if (Path(OUTPUT_DIR) / "models" / "supervised_pipeline").exists() else "dev"
os.environ.setdefault("AMAZON_NLP_RUN_PROFILE", default_profile)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# usa o mesmo python no driver e nos workers pra nao dar conflito
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ.setdefault("AMAZON_NLP_CSV_MULTILINE", "0")

# utilitarios
def _register_vector_udfs(spark: SparkSession) -> None:
    """Registra udfs de sql pra o pipeline salvo rodar sem surpresa."""
    def to_dense_vector(v):
        """Faz to dense vector pra manter o pipeline organizado."""
        if v is None:
            return None
        try:
# funciona pra vetor esparso e denso
            return Vectors.dense(v.toArray())
        except Exception:
# ultimo plano se o vetor vier estranho
            try:
                return Vectors.dense(list(v))
            except Exception:
                return None

# pode registrar de novo sem problema
    spark.udf.register("to_dense_vector", to_dense_vector, VectorUDT())

def _densify_vector_col(df: DataFrame, in_col: str, out_col: str) -> DataFrame:
    """Cria um vetor denso pro pca funcionar sem erro chato."""
    if array_to_vector is not None:
        return df.withColumn(out_col, array_to_vector(vector_to_array(F.col(in_col))))

# quebra galho no spark antigo e roda de boa no dev
    to_dense = F.udf(
        lambda v: Vectors.dense(v.toArray().tolist() if hasattr(v, "toArray") else list(v)),
        VectorUDT(),
    )
    return df.withColumn(out_col, to_dense(F.col(in_col)))

def _patch_threading_delete_dummy() -> None:
    """Faz um remendo no python 3 13 no fim da execucao pra nao estourar erro."""
    try:
        if sys.version_info < (3, 13):
            return
        cls = getattr(threading, "_DeleteDummyThreadOnDel", None)
        if cls is None:
            return
        orig_del = getattr(cls, "__del__", None)
        if orig_del is None:
            return

        def safe_del(self):  
            """Faz seguro del pra manter o pipeline organizado."""
            try:
                orig_del(self)
            except TypeError:
                # isso aparece no fim do processo quando as travas ja sumiram
                try:
                    ident = getattr(self, "ident", None)
                    active = getattr(threading, "_active", None)
                    if ident is not None and isinstance(active, dict):
                        active.pop(ident, None)
                except Exception:
                    pass

        cls.__del__ = safe_del
    except Exception:
        return

_patch_threading_delete_dummy()

def _utc_now_iso() -> str:
    """Ajuda interna de utc now iso pra deixar a execucao mais lisa."""
    return datetime.now(timezone.utc).isoformat()

def _env_str(name: str, default: str) -> str:
    """Ajuda interna de ambiente texto pra deixar a execucao mais lisa."""
    v = os.environ.get(name)
    return default if v is None or not str(v).strip() else str(v).strip()

def _env_int(name: str, default: int) -> int:
    """Ajuda interna de ambiente inteiro pra deixar a execucao mais lisa."""
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return default
    try:
        return int(str(v).strip())
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    """Ajuda interna de ambiente float pra deixar a execucao mais lisa."""
    v = os.environ.get(name)
    if v is None or not str(v).strip():
        return default
    try:
        return float(str(v).strip())
    except Exception:
        return default

def _env_bool(name: str, default: bool) -> bool:
    """Ajuda interna de ambiente booleano pra deixar a execucao mais lisa."""
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
    """Configura logs em arquivo e na tela pra acompanhar a execucao."""
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
    setup_logging._configured = True  # type: ignore[attr-defined]

def _best_torch_device() -> str:
    """Ajuda interna de melhor torch device pra deixar a execucao mais lisa."""
    _ensure_torch_imported()
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _maybe_call(obj: Any, attr: str) -> Any:
    """Ajuda interna de talvez chama pra deixar a execucao mais lisa."""
    v = getattr(obj, attr, None)
    if v is None:
        return None
    return v() if callable(v) else v

def _write_json(path: Path, obj: Any) -> None:
    """Ajuda interna de escreve json pra deixar a execucao mais lisa."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def _read_json(path: Path) -> Optional[Any]:
    """Ajuda interna de le json pra deixar a execucao mais lisa."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def _install_sitecustomize_threading_patch(output_dir: str) -> None:
    """Garante um sitecustomize pra todos os processos aplicarem o remendo do threading."""
    try:
        if sys.version_info < (3, 13):
            return

        site_dir = Path(output_dir) / "artifacts" / "py_site"
        site_dir.mkdir(parents=True, exist_ok=True)
        sc_path = site_dir / "sitecustomize.py"

        if not sc_path.exists():
            sc_path.write_text(
# deixa isso bem simples pra rodar em qualquer processo
                "import sys, threading\n"
                "if sys.version_info >= (3, 13):\n"
                "    cls = getattr(threading, '_DeleteDummyThreadOnDel', None)\n"
                "    if cls is not None:\n"
                "        _orig = getattr(cls, '__del__', None)\n"
                "        def _safe_del(self):\n"
                "            try:\n"
                "                if _orig is not None:\n"
                "                    _orig(self)\n"
                "            except Exception:\n"
                "                # Never allow shutdown-time errors to be printed\n"
                "                return\n"
                "        try:\n"
                "            cls.__del__ = _safe_del\n"
                "        except Exception:\n"
                "            pass\n",
                encoding="utf-8",
            )

# coloca no comeco do pythonpath pros workers acharem o sitecustomize
        cur = os.environ.get("PYTHONPATH", "")
        parts = [p for p in cur.split(os.pathsep) if p]
        if str(site_dir) not in parts:
            os.environ["PYTHONPATH"] = str(site_dir) + (os.pathsep + cur if cur else "")

    except Exception:
# so um tapa pra ajudar
        return

def _next_power_of_two(x: int) -> int:
    """Ajuda interna de proximo power of duas pra deixar a execucao mais lisa."""
    if x <= 1:
        return 1
    return 1 << ((x - 1).bit_length())

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kKmMgGtT]?)\s*$")

def _parse_spark_size_bytes(s: str) -> int:
    """Ajuda interna de converte spark tamanho bytes pra deixar a execucao mais lisa."""
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
    """Ajuda interna de anterior power of duas pra deixar a execucao mais lisa."""
    if x <= 1:
        return 1
    return 1 << (x.bit_length() - 1)

def pick_seq_len(p: int, allowed: Tuple[int, ...] = (32, 64, 128, 256, 512), cap: int = 256) -> int:
    """Faz escolhe seq len pra manter o pipeline organizado."""
    allowed2 = [a for a in allowed if a <= cap]
    for a in allowed2:
        if a >= int(p):
            return int(a)
    return int(allowed2[-1]) if allowed2 else int(cap)

def _safe_pca_input_dim(spark: SparkSession, requested_dim: int, requested_k: int) -> int:
    """Ajuda interna de seguro pca input dim pra deixar a execucao mais lisa."""
    max_res = _parse_spark_size_bytes(spark.conf.get("spark.driver.maxResultSize", "1g"))
    drv_mem = _parse_spark_size_bytes(spark.conf.get("spark.driver.memory", "4g"))

    cap = int(drv_mem * 0.60) if drv_mem > 0 else 0
    if max_res > 0:
        cap = min(cap, int(max_res * 0.60)) if cap > 0 else int(max_res * 0.60)
    if cap <= 0:
        return max(int(requested_dim), int(requested_k))

    def tri_bytes(d: int) -> int:
        """Faz tri bytes pra manter o pipeline organizado."""
# estimativa simples de memoria pra buffer triangular pra nao exagerar
        return 4 * d * (d + 1)

    if tri_bytes(int(requested_dim)) <= cap:
        return max(int(requested_dim), int(requested_k))

    B = cap // 4
    disc = 1 + 4 * B
    d_max = int((math.isqrt(int(disc)) - 1) // 2)
    d_max = max(int(d_max), int(requested_k))
    d_safe = min(int(requested_dim), int(d_max))
    d_safe = _prev_power_of_two(int(d_safe))
    return max(int(d_safe), int(requested_k))

def _save_fig(cfg: "Config", fig: plt.Figure, out_path: Path, *, dpi: int = 140) -> str:
    """Ajuda interna de salva figura pra deixar a execucao mais lisa."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=dpi)
    plt.close(fig)
    return _rel_to_output(cfg, out_path)

def _plot_bar(
    cfg: "Config", labels: List[str], values: List[float], *, title: str, xlabel: str, ylabel: str,
    out_path: Path, rotate: int = 0, figsize: Tuple[int, int] = (9, 4)
) -> str:
    """Ajuda interna de plota barras pra deixar a execucao mais lisa."""
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return _save_fig(cfg, fig, out_path)

def _plot_barh(
    cfg: "Config", labels: List[str], values: List[float], *, title: str, xlabel: str, ylabel: str,
    out_path: Path, figsize: Tuple[int, int] = (9, 6)
) -> str:
    """Ajuda interna de plota barras pra deixar a execucao mais lisa."""
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(labels))[::-1]
    ax.barh(y, list(values)[::-1])
    ax.set_yticks(y)
    ax.set_yticklabels(list(labels)[::-1])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return _save_fig(cfg, fig, out_path)

def _plot_grouped_bars(
    cfg: "Config", x_labels: List[str], series: Dict[str, List[float]], *, title: str, xlabel: str, ylabel: str,
    out_path: Path, rotate: int = 0, figsize: Tuple[int, int] = (10, 4)
) -> str:
    """Ajuda interna de plota agrupado bars pra deixar a execucao mais lisa."""
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
    """Ajuda interna de plota confusao 2x2 pra deixar a execucao mais lisa."""
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

# atalhos de plot pra eda e avaliacao
def plot_bar(
    labels: List[Any],
    values: List[Any],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    cfg: "Config",
    rotate: int = 30,
    figsize: Tuple[int, int] = (9, 4),
) -> str:
    """Gera plota barras e salva pra entrar no relatorio."""

    labels2 = [str(x) for x in labels]
    values2 = [float(v) for v in values]
    return _plot_bar(cfg, labels2, values2, title=title, xlabel=xlabel, ylabel=ylabel, out_path=out_path, rotate=rotate, figsize=figsize)

def plot_barh(
    labels: List[Any],
    values: List[Any],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    cfg: "Config",
    figsize: Tuple[int, int] = (9, 6),
) -> str:
    """Gera plota barras e salva pra entrar no relatorio."""
    labels2 = [str(x) for x in labels]
    values2 = [float(v) for v in values]
    return _plot_barh(cfg, labels2, values2, title=title, xlabel=xlabel, ylabel=ylabel, out_path=out_path, figsize=figsize)

def plot_grouped_bars(
    cfg: "Config",
    x_labels: List[Any],
    series: Dict[str, List[Any]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    rotate: int = 0,
    figsize: Tuple[int, int] = (10, 4),
) -> str:
    """Gera plota agrupado bars e salva pra entrar no relatorio."""
    x_labels2 = [str(x) for x in x_labels]
    series2: Dict[str, List[float]] = {str(k): [float(v) for v in vals] for k, vals in series.items()}
    return _plot_grouped_bars(cfg, x_labels2, series2, title=title, xlabel=xlabel, ylabel=ylabel, out_path=out_path, rotate=rotate, figsize=figsize)


def plot_line(xs: List[Any], ys: List[Any], *, title: str, xlabel: str, ylabel: str, out_path: Path, cfg: "Config", vline: Optional[float] = None) -> str:
    """Gera plota linha e salva pra entrar no relatorio."""
    xs2 = [float(x) for x in xs]
    ys2 = [float(y) for y in ys]
    fig, ax = plt.subplots(figsize=(8, 4))
    marker = "o" if len(xs2) <= 60 else None
    ax.plot(xs2, ys2, marker=marker)
    if vline is not None:
        ax.axvline(float(vline), linestyle="--")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return _save_fig(cfg, fig, out_path)

def plot_two_lines(
    xs: List[Any], ys1: List[Any], ys2: List[Any], *,
    label1: str, label2: str, title: str, xlabel: str, ylabel: str, out_path: Path, cfg: "Config"
) -> str:
    """Gera plota duas lines e salva pra entrar no relatorio."""
    xs2 = [float(x) for x in xs]
    y1 = [float(y) for y in ys1]
    y2 = [float(y) for y in ys2]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs2, y1, marker="o" if len(xs2) <= 60 else None, label=label1)
    ax.plot(xs2, y2, marker="o" if len(xs2) <= 60 else None, label=label2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return _save_fig(cfg, fig, out_path)

def plot_confusion_2x2(cm: List[List[int]], *, title: str, out_path: Path, cfg: "Config") -> str:
    """Gera plota confusao 2x2 e salva pra entrar no relatorio."""
    tn, fp = int(cm[0][0]), int(cm[0][1])
    fn, tp = int(cm[1][0]), int(cm[1][1])
    return _plot_confusion_2x2(cfg, tn=tn, fp=fp, fn=fn, tp=tp, title=title, out_path=out_path)


# Config

@dataclass
class Config:
    polarity_test_path: str
    polarity_train_path: Optional[str] = None
    output_dir: str = "./models_output"

    run_profile: str = "dev"  # dev ou train full ou score only

    # CSV ingestion
    csv_multiline: bool = True
    enable_csv_sharding: bool = False
    csv_shard_rows: int = 200_000
    enable_parquet_cache: bool = False

    # caps
    max_polarity_rows: int = 100_000
    lr_max_train_rows: int = 50_000  
    rf_max_train_rows: int = 30_000
    embedding_max_rows: int = 10_000
    torch_max_rows: int = 10_000
    xgb_max_train_rows: int = 50_000
    xgb_max_test_rows: int = 50_000

    # features
    hashing_num_features: int = 1 << 15
    cluster_hashing_num_features: int = 1 << 12

    # split e validacao cruzada
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

    # xgboost no driver como linha base com tfidf
    enable_xgboost: bool = True
    enable_xgb_sparse: bool = True  
    xgb_val_fraction: float = 0.10
    xgb_pca_k: int = 256
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_min_child_weight: float = 1.0
    xgb_reg_lambda: float = 1.0
    xgb_reg_alpha: float = 0.0
    xgb_gamma: float = 0.0
    xgb_early_stopping_rounds: int = 30
    xgb_eval_metric: str = "auc"
    xgb_tree_method: str = "hist"  # hist ou approx ou exact ou gpu_hist
    xgb_scale_pos_weight: Optional[float] = None  # se for none calcula no split de treino

    # clustering
    enable_clustering: bool = True
    embedding_pca_k: int = 256
    embedding_pca_k_max: int = 256
    pca_var_target: float = 0.90
    kmeans_k: int = 20
    kmeans_k_min: int = 2
    kmeans_k_max: int = 30
    kmeans_max_iter: int = 30

    # modernbert
    enable_modernbert: bool = True
    modernbert_model_name: str = "answerdotai/ModernBERT-base"
    modernbert_max_seq_len: int = 64
    modernbert_batch_size: int = 64
    modernbert_epochs: int = 1
    modernbert_lr: float = 2e-5

    # eda
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
        """Faz garante output dirs pra manter o pipeline organizado."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        for sub in ("models", "torch_models", "metrics", "artifacts", "eda", "predictions"):
            Path(self.output_dir, sub).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir, "models", "xgboost").mkdir(parents=True, exist_ok=True)

    @property
    def parquet_cache_root(self) -> Path:
        """Faz parquet cache root pra manter o pipeline organizado."""
        return Path(self.output_dir) / "artifacts" / "parquet_cache"

    @property
    def csv_shards_root(self) -> Path:
        """Faz csv shards root pra manter o pipeline organizado."""
        return Path(self.output_dir) / "artifacts" / "csv_shards"

def tune_config(cfg: Config) -> None:
    """Faz ajusta config pra manter o pipeline organizado."""
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
    cfg.xgb_max_train_rows = _env_int("AMAZON_NLP_XGB_MAX_TRAIN_ROWS", cfg.xgb_max_train_rows)
    cfg.xgb_max_test_rows = _env_int("AMAZON_NLP_XGB_MAX_TEST_ROWS", cfg.xgb_max_test_rows)

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

    # chaves do xgboost pra ligar e desligar
    cfg.enable_xgboost = _env_bool("AMAZON_NLP_ENABLE_XGBOOST", cfg.enable_xgboost)
    cfg.enable_xgb_sparse = _env_bool("AMAZON_NLP_XGB_ENABLE_SPARSE", cfg.enable_xgb_sparse)
    cfg.xgb_val_fraction = float(max(0.05, min(0.5, _env_float("AMAZON_NLP_XGB_VAL_FRACTION", cfg.xgb_val_fraction))))
    cfg.xgb_pca_k = int(max(2, _env_int("AMAZON_NLP_XGB_PCA_K", cfg.xgb_pca_k)))

    # chaves do modernbert pra ligar e ajustar
    cfg.enable_modernbert = _env_bool("AMAZON_NLP_ENABLE_MODERNBERT", cfg.enable_modernbert)
    cfg.modernbert_model_name = _env_str("AMAZON_NLP_MODERNBERT_MODEL_NAME", cfg.modernbert_model_name)
    cfg.modernbert_max_seq_len = _env_int("AMAZON_NLP_MODERNBERT_MAX_SEQ_LEN", cfg.modernbert_max_seq_len)
    cfg.modernbert_batch_size = _env_int("AMAZON_NLP_MODERNBERT_BATCH_SIZE", cfg.modernbert_batch_size)
    cfg.modernbert_epochs = _env_int("AMAZON_NLP_MODERNBERT_EPOCHS", cfg.modernbert_epochs)
    cfg.modernbert_lr = _env_float("AMAZON_NLP_MODERNBERT_LR", cfg.modernbert_lr)

    # no train full e no score only fica bem enxuto pra rodar rapido
    if cfg.run_profile in ("train_full", "score_only"):
        cfg.enable_eda = False
        cfg.enable_clustering = False
        cfg.enable_modernbert = False
        cfg.enable_rf = False
        cfg.enable_xgboost = False
        cfg.cv_parallelism = 1

    logger.info(
        "Tune: profile=%s | cores=%d | parallelism=%d | shuffle=%d | cv_parallelism=%d | "
        "max_rows=%d | parquet_cache=%s | csv_multiline=%s | csv_sharding=%s | xgboost=%s",
        cfg.run_profile,
        cores,
        cfg.spark_default_parallelism,
        cfg.spark_shuffle_partitions,
        cfg.cv_parallelism,
        cfg.max_polarity_rows,
        cfg.enable_parquet_cache,
        cfg.csv_multiline,
        cfg.enable_csv_sharding,
        cfg.enable_xgboost,
    )


# sessao do spark
def _stop_existing_local_spark_if_any() -> None:
    """Ajuda interna de para existente local spark if any pra deixar a execucao mais lisa."""
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
    """Cria a sessao do spark com configs boas pra rodar estavel."""
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

# agora o builder existe entao isso e seguro
    pp = os.environ.get("PYTHONPATH", "")
    if pp:
        builder = builder.config("spark.executorEnv.PYTHONPATH", pp)

    if cfg.enable_aqe:
        builder = (
            builder.config("spark.sql.adaptive.enabled", "true")
                   .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        )

    spark = builder.getOrCreate()
    _register_vector_udfs(spark)  
    spark.sparkContext.setLogLevel("WARN")
    return spark

@contextmanager
def spark_session(cfg: Config):
    """Abre e fecha o spark direitinho pra nao ficar coisa presa."""
    spark = create_spark(cfg)
    try:
        yield spark
    finally:
        try:
            spark.stop()
        except Exception:
            pass


# entrada de dados com shards de csv e cache em parquet
def _polarity_schema() -> T.StructType:
    """Ajuda interna de polarity schema pra deixar a execucao mais lisa."""
    return T.StructType([
        T.StructField("polarity", T.IntegerType(), nullable=False),
        T.StructField("title", T.StringType(), nullable=True),
        T.StructField("text", T.StringType(), nullable=True),
    ])

def _csv_signature(path: str) -> str:
    """Ajuda interna de csv assinatura pra deixar a execucao mais lisa."""
    try:
        p = Path(path)
        st = p.stat()
        head = p.open("rb").read(256)
        return f"{p.resolve()}|size={st.st_size}|mtime={int(st.st_mtime)}|head={hash(head)}"
    except Exception:
        return f"{path}|unknown"

def shard_csv_multiline_safe(src_path: str, dst_dir: str, *, rows_per_shard: int) -> str:
    """Faz shard csv multilinha seguro pra manter o pipeline organizado."""
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
        """Ajuda interna de open writer pra deixar a execucao mais lisa."""
        fpath = out_dir / f"part-{part_idx:05d}.csv"
        f = fpath.open("w", encoding="utf-8", newline="")
        w = csv.writer(
            f, delimiter=",", quotechar='"', doublequote=True, quoting=csv.QUOTE_MINIMAL, lineterminator="\n"
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
    """Ajuda interna de talvez shard csv for multilinha pra deixar a execucao mais lisa."""
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
    """Ajuda interna de le polarity csv raw pra deixar a execucao mais lisa."""
    return (
        spark.read.option("multiLine", str(bool(cfg.csv_multiline)).lower())
        .option("quote", '"')
        .option("escape", '"')
        .csv(csv_path, header=False, schema=_polarity_schema())
    )

def _transform_raw_to_dataset(df_raw: DataFrame, split_name: str) -> DataFrame:
    """Ajuda interna de transforma raw to dataset pra deixar a execucao mais lisa."""
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
    """Ajuda interna de parquet cache dir pra deixar a execucao mais lisa."""
    return cfg.parquet_cache_root / split_name

def _parquet_meta_path(cache_dir: Path) -> Path:
    """Ajuda interna de parquet meta path pra deixar a execucao mais lisa."""
    return cache_dir / "_meta.json"

def _parquet_cache_valid(cache_dir: Path, *, src_signature: str, multiline: bool) -> bool:
    """Ajuda interna de parquet cache valido pra deixar a execucao mais lisa."""
    meta_path = _parquet_meta_path(cache_dir)
    if not cache_dir.exists() or not any(cache_dir.glob("*.parquet")):
        return False
    if not meta_path.exists():
# safer do not trust a cache without signature information
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta.get("src_signature") == src_signature and bool(meta.get("csv_multiline", True)) == bool(multiline)
    except Exception:
        return False

def load_split_dataset(spark: SparkSession, cfg: Config, *, path: str, split_name: str) -> DataFrame:
    """Carrega carrega split dataset e deixa pronto pro resto do fluxo."""
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
        _write_json(_parquet_meta_path(cache_dir), {"src_signature": src_sig, "csv_multiline": bool(cfg.csv_multiline), "written_at": _utc_now_iso()})
        return spark.read.parquet(str(cache_dir))

    return df

def load_polarity_dataset(spark: SparkSession, cfg: Config) -> DataFrame:
    """Carrega o dataset de reviews e deixa pronto pro resto do pipeline."""
    test_df = load_split_dataset(spark, cfg, path=cfg.polarity_test_path, split_name="test")

    if cfg.polarity_train_path:
        train_df = load_split_dataset(spark, cfg, path=cfg.polarity_train_path, split_name="train")

        # importante limita por split pra nao sumir o teste
        if cfg.max_polarity_rows and cfg.max_polarity_rows > 0:
            max_rows = int(cfg.max_polarity_rows)
            train_cap = int(max_rows * float(cfg.train_split_fraction))
            train_cap = max(0, min(train_cap, max_rows))
            test_cap = max_rows - train_cap

            if max_rows >= 2:
                if train_cap == 0:
                    train_cap, test_cap = 1, max_rows - 1
                elif test_cap == 0:
                    test_cap, train_cap = 1, max_rows - 1

            if train_cap > 0:
                train_df = _deterministic_cap(train_df, "text_full", train_cap, seed=int(cfg.random_state))

            if test_cap > 0:
                test_df = _deterministic_cap(test_df, "text_full", test_cap, seed=cfg.random_state)

        df = train_df.unionByName(test_df)
    else:
        df = test_df
        if cfg.max_polarity_rows and cfg.max_polarity_rows > 0:
            df = _deterministic_cap(df, "text_full", int(cfg.max_polarity_rows), seed=cfg.random_state)

    df = df.withColumn("review_id", F.monotonically_increasing_id())
    return df.select("review_id", "dataset_split", "sentiment", "text_full")

def _maybe_fix_underpartitioning_dev(df: DataFrame, cfg: Config) -> DataFrame:
    """Ajuda interna de talvez fix underpartitioning dev pra deixar a execucao mais lisa."""
    parts = int(min(max(2, cfg.spark_default_parallelism), 64))
    try:
        cur = int(df.rdd.getNumPartitions())
        if cur < parts:
            df = df.repartition(parts)
    except Exception:
        df = df.repartition(parts)

    return df

def _deterministic_cap(df: DataFrame, col: str, n: int, seed: int) -> DataFrame:
    """Ajuda interna de deterministico limite pra deixar a execucao mais lisa."""
    return df.orderBy(F.xxhash64(F.col(col), F.lit(seed))).limit(int(n))


# EDA

def run_basic_eda(df: DataFrame, cfg: Config, total_rows: int) -> Dict[str, Any]:
    """Roda run basico eda e guarda resultado pra guiar decisoes."""
    eda_dir = Path(cfg.output_dir) / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    by_split = df.groupBy("dataset_split").count().orderBy("dataset_split").collect()
    by_sent = df.groupBy("sentiment").count().orderBy("sentiment").collect()

    split_labels = [r["dataset_split"] for r in by_split]
    split_counts = [int(r["count"]) for r in by_split]
    sent_labels = [str(int(r["sentiment"])) for r in by_sent]
    sent_counts = [int(r["count"]) for r in by_sent]

    f_split = plot_bar(split_labels, split_counts, title="Rows by dataset split", xlabel="Split", ylabel="Rows", out_path=eda_dir / "eda_split_counts.png", cfg=cfg, rotate=0)
    f_sent = plot_bar(sent_labels, sent_counts, title="Class balance (sentiment)", xlabel="Sentiment", ylabel="Rows", out_path=eda_dir / "eda_sentiment_counts.png", cfg=cfg, rotate=0)

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
    f_miss = plot_bar(miss_labels, miss_vals, title="Missing/empty rates", xlabel="Column", ylabel="Rate", out_path=eda_dir / "eda_missing_rates.png", cfg=cfg, rotate=30)

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
    """Roda run avancado eda e guarda resultado pra guiar decisoes."""
    eda_dir = Path(cfg.output_dir) / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    df_len = df.withColumn("text_len_tokens", F.size(F.split(F.col("text_full"), r"\s+")))
    qs = (
        df_len.agg(F.expr("percentile_approx(text_len_tokens, array(0.5, 0.9, 0.95, 0.99), 10000)").alias("qs"))
        .collect()[0]["qs"]
    )
    p50, p90, p95, p99 = [int(x) for x in qs] if qs else [0, 0, 0, 0]

    bin_width = max(1, int(math.ceil(max(1, p99) / float(max(10, cfg.hist_target_bins)))))
    binned = (
        df_len.select((F.floor(F.col("text_len_tokens") / F.lit(bin_width)) * F.lit(bin_width)).alias("bin_start"))
        .groupBy("bin_start").count().orderBy("bin_start")
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
    hist_file = _save_fig(cfg, fig, eda_dir / "eda_text_len_hist.png")

    # eda de tokens com amostra pra rodar rapido
    sample_target = min(200_000, max(20_000, int(float(cfg.eda_sample_frac) * float(total_rows))))
    sample_frac = min(1.0, float(sample_target) / float(max(1, total_rows)))

    sample = (
        df.select("text_full", "sentiment")
        .where(F.col("text_full").isNotNull())
        .sample(False, sample_frac, seed=int(cfg.random_state))
    )
    sample = _deterministic_cap(sample, "text_full", int(sample_target), seed=cfg.random_state)

    tok = RegexTokenizer(inputCol="text_full", outputCol="tokens", pattern=r"\W+", toLowercase=True)
    sw = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", locale="en_US")
    cleaned = (
        sw.transform(tok.transform(sample))
        .select("sentiment", F.explode("filtered_tokens").alias("token"))
        .select("sentiment", F.lower(F.col("token")).alias("token"))
        .where(F.length("token") > 2)
        .where(F.col("token").rlike("^[a-zA-Z]+$"))
    )
    cleaned = cleaned.persist(StorageLevel.MEMORY_AND_DISK)
    _ = cleaned.count()


    vocab_est = int(cleaned.agg(F.approx_count_distinct("token").alias("v")).collect()[0]["v"] or 0)
    suggested_hash_dim = _next_power_of_two(min(max(2 * vocab_est, 2**14), 2**18)) if vocab_est > 0 else int(cfg.hashing_num_features)

    def top_tokens(where_sentiment: Optional[int], n: int) -> List[Tuple[str, int]]:
        """Faz top tokens pra manter o pipeline organizado."""
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

    f_top_all = plot_bar([t for t, _ in top_all_plot], [c for _, c in top_all_plot], title="Top tokens (all)", xlabel="Token", ylabel="Count", out_path=eda_dir / "eda_top_tokens_all.png", cfg=cfg, rotate=30)
    f_top_pos = plot_bar([t for t, _ in top_pos_plot], [c for _, c in top_pos_plot], title="Top tokens (positive)", xlabel="Token", ylabel="Count", out_path=eda_dir / "eda_top_tokens_pos.png", cfg=cfg, rotate=30)
    f_top_neg = plot_bar([t for t, _ in top_neg_plot], [c for _, c in top_neg_plot], title="Top tokens (negative)", xlabel="Token", ylabel="Count", out_path=eda_dir / "eda_top_tokens_neg.png", cfg=cfg, rotate=30)

    wc_all = wc_pos = wc_neg = None
    if WordCloud is not None:
        def make_wordcloud(freq: Dict[str, int], fname: str) -> Optional[str]:
            """Gera make wordcloud e salva pra entrar no relatorio."""
            if not freq:
                return None
            wc = WordCloud(width=1200, height=600, background_color="white", max_words=int(cfg.wordcloud_max_words))
            wc = wc.generate_from_frequencies(freq)
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.imshow(wc, interpolation="bilinear")
            ax2.axis("off")
            return _save_fig(cfg, fig2, eda_dir / fname)

        wc_all = make_wordcloud(dict(top_all_full), "eda_wordcloud_all.png")
        wc_pos = make_wordcloud(dict(top_pos_full), "eda_wordcloud_pos.png")
        wc_neg = make_wordcloud(dict(top_neg_full), "eda_wordcloud_neg.png")
    else:
        logger.warning("wordcloud not installed; skipping wordclouds (error=%s)", str(getattr(globals(), "_WORDCLOUD_IMPORT_ERROR", "")))
    cleaned.unpersist()

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
    """Faz aplica dados guiado config pra manter o pipeline organizado."""
    sug = (eda_adv or {}).get("suggested") or {}
    if sug.get("modernbert_max_seq_len"):
        cfg.modernbert_max_seq_len = int(sug["modernbert_max_seq_len"])

# limita o max len do modernbert pra rodar mais estavel no dev
    cap = _env_int("AMAZON_NLP_MODERNBERT_MAX_LEN_CAP", 128)
    if int(cfg.modernbert_max_seq_len) > int(cap):
        logger.info(
            "Clamping modernbert_max_seq_len %d -> %d based on AMAZON_NLP_MODERNBERT_MAX_LEN_CAP.",
            int(cfg.modernbert_max_seq_len),
            int(cap),
        )
        cfg.modernbert_max_seq_len = int(cap)



    if sug.get("hashing_num_features"):
        cfg.hashing_num_features = int(sug["hashing_num_features"])

    decisions = {
        "applied_at": _utc_now_iso(),
        "modernbert_max_seq_len": int(cfg.modernbert_max_seq_len),
        "hashing_num_features": int(cfg.hashing_num_features),
    }
    _write_json(Path(cfg.output_dir) / "metrics" / "eda_decisions_applied.json", decisions)
    _write_json(Path(cfg.output_dir) / "metrics" / "config_final.json", asdict(cfg))



# avaliacao
def _downsample_curve(xs: List[float], ys: List[float], *, max_points: int = 2000) -> Tuple[List[float], List[float]]:
    """Ajuda interna de reduz curva pra deixar a execucao mais lisa."""
    n = min(len(xs), len(ys))
    if n <= max_points:
        return xs[:n], ys[:n]
    idx = np.linspace(0, n - 1, num=max_points, dtype=int)
    return [float(xs[i]) for i in idx], [float(ys[i]) for i in idx]

def _roc_curve_from_scores(labels: List[int], scores: List[float]) -> Tuple[List[float], List[float]]:
    """Ajuda interna de roc curva de scores pra deixar a execucao mais lisa."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.size == 0:
        return [], []
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return [], []

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
    return [float(x) for x in fpr], [float(x) for x in tpr]

def evaluate_binary_classifier(pred_df: DataFrame, *, name: str, cfg: Config, split: str) -> Dict[str, Any]:
    """Calcula metricas binarias e salva arquivos pra comparar modelos."""
    out_dir = Path(cfg.output_dir)
    eda_dir = out_dir / "eda"
    metrics_dir = out_dir / "metrics"
    eda_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    p = pred_df
    if "label" not in p.columns:
        raise ValueError(f"{name}: expected label column 'label'.")

    # garante prob1 pra fazer curvas e calibracao sem dor
    if "probability" in p.columns:
        p = p.withColumn("prob1", vector_to_array(F.col("probability")).getItem(1))
    elif "prob1" not in p.columns:
        raise ValueError(f"{name}: need probability or prob1 column for curves/calibration.")

    # garante prediction pra ter classe final mesmo sem coluna pronta
    if "prediction" not in p.columns:
        p = p.withColumn("prediction", F.when(F.col("prob1") >= F.lit(0.5), F.lit(1.0)).otherwise(F.lit(0.0)))

    p = (
        p.withColumn("label", F.col("label").cast("double"))
         .withColumn("prediction", F.col("prediction").cast("double"))
         .withColumn("prob1", F.col("prob1").cast("double"))
         .persist(StorageLevel.MEMORY_AND_DISK)
    )

    try:
        # matriz de confusao pra enxergar acertos e erros
        cm_rows = (
            p.withColumn("label_i", F.col("label").cast("int"))
             .withColumn("pred_i", F.col("prediction").cast("int"))
             .groupBy("label_i", "pred_i").count().collect()
        )
        counts = {(int(r["label_i"]), int(r["pred_i"])): int(r["count"]) for r in cm_rows}
        tn = counts.get((0, 0), 0)
        fp = counts.get((0, 1), 0)
        fn = counts.get((1, 0), 0)
        tp = counts.get((1, 1), 0)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        bal_acc = (recall + specificity) / 2.0
        n = tn + fp + fn + tp
        acc = float((tp + tn) / n) if n else float("nan")
        f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0

        # aucs no spark usando prob1 pra ficar leve
        auc_roc = float(BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prob1", metricName="areaUnderROC").evaluate(p))
        auc_pr  = float(BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prob1", metricName="areaUnderPR").evaluate(p))

        # brier e bins de calibracao
        brier = float(
            p.select(F.mean(F.pow(F.col("prob1") - F.col("label"), F.lit(2))).alias("brier"))
             .collect()[0]["brier"] or 0.0
        )

        bins = int(max(2, cfg.calibration_bins))
        calib = (
            p.withColumn("bin", (F.floor(F.col("prob1") * F.lit(bins)) / F.lit(float(bins))))
             .groupBy("bin")
             .agg(F.count("*").alias("n"), F.avg("prob1").alias("avg_p"), F.avg("label").alias("avg_y"))
             .orderBy("bin").collect()
        )
        calib_x = [float(r["avg_p"]) for r in calib]
        calib_y = [float(r["avg_y"]) for r in calib]
        calib_n = [int(r["n"]) for r in calib]

        # curvas no driver com limite de linhas
        cap = int(cfg.eval_curve_cap_rows)
        if cfg.max_polarity_rows and cfg.max_polarity_rows > 0:
            cap = int(min(cap, int(cfg.max_polarity_rows)))
        cap = int(max(1, cap))

        base = (
            p.select("prob1", "label")
             .where(F.col("prob1").isNotNull() & F.col("label").isNotNull())
        )
        n = int(base.count())

        if n == 0:
            # sem pontos entao a curva fica vazia
            scores: List[float] = []
            labels: List[int] = []
        else:
            frac = min(1.0, float(cap) / float(n))
            sample_df = (
                base.sample(False, frac, seed=int(cfg.random_state))
                    .limit(int(cap))
            )
            curve_rows = sample_df.collect()
            scores = [float(r["prob1"]) for r in curve_rows]
            labels = [int(r["label"]) for r in curve_rows]

        roc_x, roc_y = _roc_curve_from_scores(labels, scores)
        pr_x, pr_y = _pr_curve_from_scores(labels, scores)  # recall, precision

        roc_x, roc_y = _downsample_curve(roc_x, roc_y, max_points=2000)
        pr_x, pr_y   = _downsample_curve(pr_x, pr_y, max_points=2000)


        # f1 por limiar pra achar um corte bom
        thr_file = None
        best_thr = None
        best_f1_thr = None
        try:
            if scores and len(set(labels)) == 2:
                y_np = np.asarray(labels, dtype=int)
                s_np = np.asarray(scores, dtype=float)
                thr_grid = np.linspace(0.0, 1.0, num=101, dtype=float)
                f1s: List[float] = []
                for thr in thr_grid:
                    pred = (s_np >= thr)
                    tp0 = int(np.sum(pred & (y_np == 1)))
                    fp0 = int(np.sum(pred & (y_np == 0)))
                    fn0 = int(np.sum((~pred) & (y_np == 1)))
                    prec0 = tp0 / (tp0 + fp0) if (tp0 + fp0) else 0.0
                    rec0  = tp0 / (tp0 + fn0) if (tp0 + fn0) else 0.0
                    f10   = (2 * prec0 * rec0) / (prec0 + rec0) if (prec0 + rec0) else 0.0
                    f1s.append(float(f10))
                best_idx = int(np.argmax(np.asarray(f1s)))
                best_thr = float(thr_grid[best_idx])
                best_f1_thr = float(f1s[best_idx])
                thr_file = plot_line(
                    thr_grid.tolist(), f1s,
                    title=f"F1 por limiar – {name} ({split})",
                    xlabel="Limiar", ylabel="F1",
                    out_path=eda_dir / f"eval_{name}_{split}_f1_thr.png",
                    cfg=cfg, vline=float(best_thr)
                )
        except Exception:
            pass

        cm_file  = plot_confusion_2x2([[tn, fp], [fn, tp]], title=f"Matriz de confusao – {name} ({split})", out_path=eda_dir / f"eval_{name}_{split}_cm.png", cfg=cfg)
        roc_file = plot_line(roc_x, roc_y, title=f"ROC – {name} ({split})", xlabel="FPR", ylabel="TPR", out_path=eda_dir / f"eval_{name}_{split}_roc.png", cfg=cfg)
        pr_file  = plot_line(pr_x, pr_y, title=f"PR – {name} ({split})", xlabel="Recall", ylabel="Precisao", out_path=eda_dir / f"eval_{name}_{split}_pr.png", cfg=cfg)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.plot(calib_x, calib_y, marker="o" if len(calib_x) <= 60 else None)
        ax.set_title(f"Calibration – {name} ({split})")
        ax.set_xlabel("Avg predicted prob")
        ax.set_ylabel("Observed positive rate")
        calib_file = _save_fig(cfg, fig, eda_dir / f"eval_{name}_{split}_calib.png")

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
                "specificity_neg": float(specificity),
                "balanced_accuracy": float(bal_acc),
                "brier": brier,
            },
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "calibration_bins": [{"avg_p": float(x), "avg_y": float(y0), "n": int(n0)} for x, y0, n0 in zip(calib_x, calib_y, calib_n)],
            "threshold_tuning": {"best_thr": float(best_thr) if best_thr is not None else None, "best_f1": float(best_f1_thr) if best_f1_thr is not None else None},
            "plots": {"cm": cm_file, "roc": roc_file, "pr": pr_file, "calibration": calib_file, "f1_threshold": thr_file},
            "curve_rows_cap": int(cap),
            "curve_rows_used": int(len(scores)),
        }
        _write_json(metrics_dir / f"eval_{name}_{split}.json", out)
        return out
    finally:
        try:
            p.unpersist()
        except Exception:
            pass

def _auc_roc_from_scores(labels: List[int], scores: List[float]) -> float:
    """Ajuda interna de auc roc de scores pra deixar a execucao mais lisa."""
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
    """Ajuda interna de pr curva de scores pra deixar a execucao mais lisa."""
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
    """Ajuda interna de auc pr de curva pra deixar a execucao mais lisa."""
    if not recall or not precision or len(recall) != len(precision):
        return float("nan")
    auc = 0.0
    for i in range(1, len(recall)):
        auc += (recall[i] - recall[i - 1]) * (precision[i] + precision[i - 1]) / 2.0
    return float(auc)

def evaluate_binary_from_arrays(*, probs: List[float], labels: List[int], name: str, cfg: Config, split: str, threshold: float = 0.5) -> Dict[str, Any]:
    """Calcula metricas binarias direto de arrays pra usar fora do spark."""
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
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    bal_acc = float((recall + specificity) / 2.0)

    auc_roc = _auc_roc_from_scores(labels, probs)
    rec_curve, prec_curve = _pr_curve_from_scores(labels, probs)
    auc_pr = _auc_pr_from_curve(rec_curve, prec_curve)
    brier = float(np.mean((p - y.astype(float)) ** 2)) if y.size else float("nan")

# bins de calibracao
    bins = int(max(2, cfg.calibration_bins))
    edges = np.linspace(0.0, 1.0, bins + 1)
    calib_bins: List[Tuple[float, float, int]] = []
    for i in range(bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (p >= lo) & (p < hi) if i < (bins - 1) else (p >= lo) & (p <= hi)
        n = int(np.sum(mask))
        if n <= 0:
            continue
        avg_p = float(np.mean(p[mask]))
        avg_y = float(np.mean(y[mask].astype(float)))
        calib_bins.append((avg_p, avg_y, n))

    calib_x = [x for x, _, _ in calib_bins]
    calib_y = [yy for _, yy, _ in calib_bins]
    calib_n = [n for _, _, n in calib_bins]

# f1 por limiar
    best_thr = None
    best_f1_thr = None
    thr_file = None
    try:
        uniq = np.unique(p)
        uniq = np.sort(uniq)[::-1]
        best = (-1.0, None)
        xs: List[float] = []
        ys: List[float] = []
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
            thr_file = plot_line(xs[::-1], ys[::-1], title=f"F1 por limiar – {name} ({split})", xlabel="Limiar", ylabel="F1", out_path=eda_dir / f"eval_{name}_{split}_f1_thr.png", cfg=cfg, vline=float(best_thr))
    except Exception:
        pass

    cm_file = plot_confusion_2x2([[tn, fp], [fn, tp]], title=f"Confusion matrix – {name} ({split})", out_path=eda_dir / f"eval_{name}_{split}_cm.png", cfg=cfg)

    # pontos do roc
    def roc_points(labels_: List[int], scores_: List[float]) -> Tuple[List[float], List[float]]:
        """Faz roc points pra manter o pipeline organizado."""
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
    roc_file = plot_line(roc_x, roc_y, title=f"ROC – {name} ({split})", xlabel="FPR", ylabel="TPR", out_path=eda_dir / f"eval_{name}_{split}_roc.png", cfg=cfg)
    pr_file = plot_line(rec_curve, prec_curve, title=f"PR – {name} ({split})", xlabel="Recall", ylabel="Precision", out_path=eda_dir / f"eval_{name}_{split}_pr.png", cfg=cfg)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.plot(calib_x, calib_y, marker="o" if len(calib_x) <= 60 else None)
    ax.set_title(f"Calibration – {name} ({split})")
    ax.set_xlabel("Avg predicted prob")
    ax.set_ylabel("Observed positive rate")
    calib_file = _save_fig(cfg, fig, eda_dir / f"eval_{name}_{split}_calib.png")

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
            "specificity_neg": float(specificity),
            "balanced_accuracy": float(bal_acc),
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

def _collect_eval_jsons(cfg: Config) -> List[Dict[str, Any]]:
    """Ajuda interna de coleta eval jsons pra deixar a execucao mais lisa."""
    metrics_dir = Path(cfg.output_dir) / "metrics"
    out: List[Dict[str, Any]] = []
    for p in sorted(metrics_dir.glob("eval_*.json")):
        j = _read_json(p)
        if isinstance(j, dict) and isinstance(j.get("metrics"), dict):
            out.append(j)
    return out

def plot_model_leaderboard(eval_jsons: List[Dict[str, Any]], cfg: Config, *, metric_key: str) -> Optional[str]:
    """Gera plota model leaderboard e salva pra entrar no relatorio."""
    eda_dir = Path(cfg.output_dir) / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Tuple[str, float]] = []
    for e in eval_jsons:
        m = e.get("metrics") or {}
        val = m.get(metric_key)
        try:
            if val is None:
                continue
            rows.append((str(e.get("model", "unknown")), float(val)))
        except Exception:
            continue
    if not rows:
        return None
    reverse = (metric_key != "brier")
    rows = sorted(rows, key=lambda x: x[1], reverse=reverse)
    models = [m for m, _ in rows]
    vals = [v for _, v in rows]
    return plot_bar(models, vals, title=f"Model comparison – {metric_key}", xlabel="Model", ylabel=metric_key, out_path=eda_dir / f"models_leaderboard_{metric_key}.png", cfg=cfg, rotate=15)

def refresh_leaderboards(cfg: Config) -> Dict[str, Optional[str]]:
    """Faz refresh leaderboards pra manter o pipeline organizado."""
    evals = _collect_eval_jsons(cfg)
    out = {
        "auc_roc": plot_model_leaderboard(evals, cfg, metric_key="auc_roc"),
        "auc_pr": plot_model_leaderboard(evals, cfg, metric_key="auc_pr"),
        "f1": plot_model_leaderboard(evals, cfg, metric_key="f1"),
        "accuracy": plot_model_leaderboard(evals, cfg, metric_key="accuracy"),
        "brier": plot_model_leaderboard(evals, cfg, metric_key="brier"),
    }
    _write_json(Path(cfg.output_dir) / "metrics" / "leaderboards.json", {"generated_at": _utc_now_iso(), "files": out})
    return out


# modelos supervisionados
def _add_split_bucket(df: DataFrame, cfg: Config) -> DataFrame:
    """Ajuda interna de add split bucket pra deixar a execucao mais lisa."""
    mod = int(cfg.deterministic_split_mod)
    seed = int(cfg.random_state)
    bucket = F.pmod(F.abs(F.xxhash64(F.col("text_full"), F.lit(seed))), F.lit(mod)).cast("int")
    return df.withColumn("split_bucket", bucket)

def _train_test_filters(cfg: Config) -> Tuple[Any, Any]:
    """Ajuda interna de treina test filters pra deixar a execucao mais lisa."""
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
    """Monta tfidf no spark pra treinar e pontuar modelos supervisionados."""
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

def _spark_vector_to_csr(rows: List[Any], *, num_features: int) -> "sp.csr_matrix":
    """Ajuda interna de spark vector to csr pra deixar a execucao mais lisa."""
    if sp is None:
        raise RuntimeError(f"scipy is required for sparse XGBoost, but import failed: {_SCIPY_IMPORT_ERROR}")

    n_rows = int(len(rows))
    if n_rows <= 0:
        return sp.csr_matrix((0, int(num_features)), dtype=np.float32)

    # primeira passada pra contar nnz e checar limites pra nao corromper e travar
    row_nnz: List[int] = []
    max_idx = -1
    for v in rows:
        if hasattr(v, "indices") and hasattr(v, "values"):
            idx = v.indices  # type: ignore[attr-defined]
            nnz = int(len(idx))
            row_nnz.append(nnz)
            if nnz:
                try:
                    max_idx = max(max_idx, int(max(idx)))
                except Exception:
                    max_idx = max(max_idx, int(np.max(np.asarray(idx))))
        else:
            arr = np.asarray(v, dtype=np.float32)
            nz = np.nonzero(arr)[0]
            nnz = int(nz.size)
            row_nnz.append(nnz)
            if nnz:
                max_idx = max(max_idx, int(nz.max()))

    if max_idx >= int(num_features):
        raise RuntimeError(
            f"Invalid TF-IDF vector indices for CSR build: max_index={max_idx} >= num_features={int(num_features)}"
        )

    total_nnz = int(sum(row_nnz))
    data = np.empty(total_nnz, dtype=np.float32)
    indices = np.empty(total_nnz, dtype=np.int32)
    indptr = np.empty(n_rows + 1, dtype=np.int32)
    indptr[0] = 0

    cursor = 0
    for i, v in enumerate(rows):
        nnz = int(row_nnz[i])
        indptr[i + 1] = int(indptr[i] + nnz)
        if nnz <= 0:
            continue

        if hasattr(v, "indices") and hasattr(v, "values"):
            idx = np.asarray(v.indices, dtype=np.int32)  # type: ignore[attr-defined]
            vals = np.asarray(v.values, dtype=np.float32)  # type: ignore[attr-defined]
            if idx.size != nnz or vals.size != nnz:
            # defensivo pra manter consistente se vier vetor estranho do spark
                nnz = int(min(idx.size, vals.size, nnz))
                idx = idx[:nnz]
                vals = vals[:nnz]
                indptr[i + 1] = int(indptr[i] + nnz)
            indices[cursor : cursor + nnz] = idx
            data[cursor : cursor + nnz] = vals
        else:
            arr = np.asarray(v, dtype=np.float32)
            nz = np.nonzero(arr)[0].astype(np.int32, copy=False)
            if nz.size != nnz:
                nnz = int(nz.size)
                indptr[i + 1] = int(indptr[i] + nnz)
            indices[cursor : cursor + nnz] = nz
            data[cursor : cursor + nnz] = arr[nz].astype(np.float32, copy=False)

        cursor += nnz

    X = sp.csr_matrix((data, indices, indptr), shape=(n_rows, int(num_features)), dtype=np.float32)
    try:
        X.sum_duplicates()
        X.sort_indices()
    except Exception:
        pass
    return X

def _collect_xgb_matrices(train_df: DataFrame, test_df: DataFrame, cfg: Config) -> Tuple[Any, Any, np.ndarray, np.ndarray, Any, np.ndarray, str]:
    """Ajuda interna de coleta xgb matrizes pra deixar a execucao mais lisa."""


    # Limit rows
    tr = train_df
    te = test_df
    if cfg.xgb_max_train_rows and cfg.xgb_max_train_rows > 0:
        tr = tr.limit(int(cfg.xgb_max_train_rows))
    if cfg.xgb_max_test_rows and cfg.xgb_max_test_rows > 0:
        te = te.limit(int(cfg.xgb_max_test_rows))

    # da pra forcar pca denso pra ficar mais seguro no mac
    force_dense_default = (platform.system() == "Darwin") and (sp is None)

    force_dense = _env_bool("AMAZON_NLP_XGB_FORCE_DENSE", force_dense_default)
    use_sparse = (sp is not None) and bool(getattr(cfg, "enable_xgb_sparse", True)) and (not force_dense)

    if use_sparse:

    # checa a dimensao do tfidf pra evitar csr torto e travar
        sample = tr.select("tfidf_features").limit(1).collect()
        if sample:
            v0 = sample[0]["tfidf_features"]
            dim0 = int(getattr(v0, "size", len(v0)))
            if dim0 != int(cfg.hashing_num_features):
                raise RuntimeError(
                    f"Dimension mismatch for XGBoost: tfidf_features dim={dim0}, cfg.hashing_num_features={cfg.hashing_num_features}. "
                    "Check env/config reuse across runs."
                )

        # caminho csr coleta vetores esparsos pra matriz do scipy
        tr_rows = tr.select("tfidf_features", F.col("label").cast("int").alias("y")).collect()
        te_rows = te.select("tfidf_features", F.col("label").cast("int").alias("y")).collect()

        X_all = _spark_vector_to_csr(
            [r["tfidf_features"] for r in tr_rows],
            num_features=int(cfg.hashing_num_features),
        )
        y_all = np.array([int(r["y"]) for r in tr_rows], dtype=np.int32)

        X_test = _spark_vector_to_csr(
            [r["tfidf_features"] for r in te_rows],
            num_features=int(cfg.hashing_num_features),
        )
        y_test = np.array([int(r["y"]) for r in te_rows], dtype=np.int32)

        rep = "sparse"

        # segura o tamanho do csr pra nao estourar o driver
        try:
            nnz_tr = int(X_all.nnz)
            nnz_te = int(X_test.nnz)
            max_nnz_tr = _env_int("AMAZON_NLP_XGB_MAX_NNZ_TRAIN", 12_000_000)
            max_nnz_te = _env_int("AMAZON_NLP_XGB_MAX_NNZ_TEST", 12_000_000)
            if nnz_tr > max_nnz_tr or nnz_te > max_nnz_te:
                raise RuntimeError(
                    f"XGBoost CSR too large: nnz_train={nnz_tr} (limit={max_nnz_tr}), nnz_test={nnz_te} (limit={max_nnz_te})."
                )
        except Exception as e:
            raise RuntimeError(f"XGBoost sparse matrix safety check failed: {e}") from e

    else:
        safe_k = _safe_pca_input_dim(
            spark=tr.sparkSession,
            requested_dim=int(cfg.hashing_num_features),
            requested_k=int(cfg.xgb_pca_k),
        )
        pca_k = int(max(2, min(safe_k, int(cfg.hashing_num_features))))

        tr_pca = _densify_vector_col(tr, "tfidf_features", "_tfidf_dense")
        te_pca = _densify_vector_col(te, "tfidf_features", "_tfidf_dense")

        pca = PCA(k=int(pca_k), inputCol="_tfidf_dense", outputCol="xgb_pca")
        pca_model = pca.fit(tr_pca)

        tr_rows = (
            pca_model.transform(tr_pca)
            .select(vector_to_array(F.col("xgb_pca")).alias("x"),
                    F.col("label").cast("int").alias("y"))
            .collect()
        )
        te_rows = (
            pca_model.transform(te_pca)
            .select(vector_to_array(F.col("xgb_pca")).alias("x"),
                    F.col("label").cast("int").alias("y"))
            .collect()
        )


        X_all = np.array([r["x"] for r in tr_rows], dtype=np.float32)
        y_all = np.array([int(r["y"]) for r in tr_rows], dtype=np.int32)
        X_test = np.array([r["x"] for r in te_rows], dtype=np.float32)
        y_test = np.array([int(r["y"]) for r in te_rows], dtype=np.int32)
        rep = "dense_pca"

    # segura o tamanho do denso pra nao estourar memoria
        cells_tr = int(X_all.shape[0]) * int(X_all.shape[1])
        cells_te = int(X_test.shape[0]) * int(X_test.shape[1])
        max_cells_tr = _env_int("AMAZON_NLP_XGB_MAX_CELLS_TRAIN", 50_000_000)
        max_cells_te = _env_int("AMAZON_NLP_XGB_MAX_CELLS_TEST", 50_000_000)
        if cells_tr > max_cells_tr or cells_te > max_cells_te:
            raise RuntimeError(
                f"XGBoost dense matrices too large: X_train={X_all.shape} (cells={cells_tr}, limit={max_cells_tr}), "
                f"X_test={X_test.shape} (cells={cells_te}, limit={max_cells_te})."
            )
 
    # divide train em train/val
    rng = np.random.RandomState(int(cfg.random_state))
    idx = np.arange(len(y_all))
    rng.shuffle(idx)
    n_all = int(len(idx))
    if n_all < 2:
        raise ValueError("XGBoost needs >= 2 training rows to create a train/val split.")
    if int(len(y_test)) < 1:
        raise ValueError("XGBoost needs >= 1 test row to evaluate.")
    frac = float(cfg.xgb_val_fraction)
    frac = max(0.05, min(0.5, frac))
    val_n = int(round(frac * n_all))
    val_n = max(1, min(val_n, n_all - 1))

    val_idx = idx[:val_n]
    tr_idx = idx[val_n:]
    if len(tr_idx) == 0 and len(val_idx) > 1:
        tr_idx = val_idx[1:]
        val_idx = val_idx[:1]

    X_train = X_all[tr_idx]
    y_train = y_all[tr_idx]
    X_val = X_all[val_idx]
    y_val = y_all[val_idx]

    return (X_train, X_val, y_train, y_val, X_test, y_test, rep)

# dependencias opcionais como scipy e xgboost
try:
    import scipy.sparse as sp  
except Exception as _e:
    sp = None  
    _SCIPY_IMPORT_ERROR = _e

try:
    import xgboost as xgb 
    from xgboost import XGBClassifier 
except Exception as _e:
    xgb = None  
    XGBClassifier = None  
    _XGBOOST_IMPORT_ERROR = _e

def _xgb_worker_train(payload: Dict[str, Any]) -> None:
    """Ajuda interna de xgb worker treina pra deixar a execucao mais lisa."""
    try:
        cfg_dict = payload["cfg"]
        cfg = Config(**cfg_dict)
        cfg.ensure_output_dirs()

        rep = str(payload["rep"])
        paths = payload["paths"]

        if rep == "sparse":
            if sp is None:
                raise RuntimeError("scipy is required to load sparse matrices in worker but is not available.")
            X_train = sp.load_npz(paths["X_train"])
            X_val = sp.load_npz(paths["X_val"])
            X_test = sp.load_npz(paths["X_test"])
        else:
            X_train = np.load(paths["X_train"], allow_pickle=False)
            X_val   = np.load(paths["X_val"],   allow_pickle=False)
            X_test  = np.load(paths["X_test"],  allow_pickle=False)
            
        y_train = np.load(paths["y_train"], allow_pickle=False)
        y_val   = np.load(paths["y_val"],   allow_pickle=False)
        y_test  = np.load(paths["y_test"],  allow_pickle=False)

        _ = _train_xgboost_from_matrices(
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            rep=rep,
            cfg=cfg,
        )
    except Exception as e:
        # tenta escrever um json de skip pra dar um motivo claro para o usuario
        try:
            out_dir = Path(payload.get("output_dir") or payload["cfg"].get("output_dir") or "./models_output")
            metrics_dir = out_dir / "metrics"
            out = {"generated_at": _utc_now_iso(), "status": "skipped", "reason": f"XGBoost worker exception: {e}"}
            _write_json(metrics_dir / "xgboost.json", out)
            _write_json(metrics_dir / "xgboost_baseline.json", out)
        except Exception:
            pass
        raise

def _train_xgboost_from_matrices(
    *,
    X_train: Any,
    X_val: Any,
    y_train: np.ndarray,
    y_val: np.ndarray,
    X_test: Any,
    y_test: np.ndarray,
    rep: str,
    cfg: Config,
) -> Dict[str, Any]:
    """Ajuda interna de treina xgboost de matrizes pra deixar a execucao mais lisa."""
    metrics_dir = Path(cfg.output_dir) / "metrics"
    models_dir = Path(cfg.output_dir) / "models" / "xgboost"
    eda_dir = Path(cfg.output_dir) / "eda"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    eda_dir.mkdir(parents=True, exist_ok=True)

    if XGBClassifier is None or xgb is None:
        out = {"generated_at": _utc_now_iso(), "status": "skipped", "reason": "xgboost import unavailable"}
        _write_json(metrics_dir / "xgboost.json", out)
        _write_json(metrics_dir / "xgboost_baseline.json", out)
        return out

    t0 = time.time()

    # desbalanceamento de classes
    pos = int(np.sum(y_train == 1))
    neg = int(np.sum(y_train == 0))
    if cfg.xgb_scale_pos_weight is not None and float(cfg.xgb_scale_pos_weight) > 0:
        scale_pos_weight = float(cfg.xgb_scale_pos_weight)
    else:
        scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    # metodo de arvore / threads conservador defaults no mac
    tree_method = str(cfg.xgb_tree_method)
    try:
        if platform.system() == "Darwin" and tree_method.lower() == "hist":
            tree_method = "approx"
    except Exception:
        pass
    max_jobs = _env_int("AMAZON_NLP_XGB_MAX_JOBS", 4)
    n_jobs = int(max(1, min(os.cpu_count() or 4, int(max_jobs))))

    params = {
        "n_estimators": int(cfg.xgb_n_estimators),
        "max_depth": int(cfg.xgb_max_depth),
        "learning_rate": float(cfg.xgb_learning_rate),
        "subsample": float(cfg.xgb_subsample),
        "colsample_bytree": float(cfg.xgb_colsample_bytree),
        "min_child_weight": float(cfg.xgb_min_child_weight),
        "reg_lambda": float(cfg.xgb_reg_lambda),
        "reg_alpha": float(cfg.xgb_reg_alpha),
        "gamma": float(cfg.xgb_gamma),
        "objective": "binary:logistic",
        "eval_metric": str(cfg.xgb_eval_metric),
        "tree_method": tree_method,
        "n_jobs": n_jobs,
        "random_state": int(cfg.random_state),
        "scale_pos_weight": float(scale_pos_weight),
    }

    # logica de early stopping pra funcionar em versoes diferentes do xgboost
    xgb_version = getattr(xgb, "__version__", "unknown")
    early_req = int(cfg.xgb_early_stopping_rounds or 0)
    early_req = int(max(0, early_req))
    eval_set = [(X_val, y_val)]
    fit_common = {"eval_set": eval_set, "verbose": False}
    maximize = str(cfg.xgb_eval_metric).lower() in {"auc", "aucpr", "map", "ndcg"}

    early_used = False
    early_mode = "none"

    def _unexpected_kwarg(e: TypeError) -> bool:
        """Ajuda interna de unexpected kwarg pra deixar a execucao mais lisa."""
        msg = str(e).lower()
        return ("unexpected keyword argument" in msg) or ("got an unexpected keyword argument" in msg)

    def _try_init(extra_kwargs: Dict[str, Any]) -> Optional[XGBClassifier]:
        """Ajuda interna de try init pra deixar a execucao mais lisa."""
        try:
            return XGBClassifier(**params, **extra_kwargs)
        except TypeError as e:
            if _unexpected_kwarg(e):
                return None
            raise

    def _try_fit(model: XGBClassifier, extra_kwargs: Dict[str, Any]) -> bool:
        """Ajuda interna de try fit pra deixar a execucao mais lisa."""
        try:
            model.fit(X_train, y_train, **fit_common, **extra_kwargs)
            return True
        except TypeError as e:
            if _unexpected_kwarg(e):
                return False
            raise

    def _make_es_callback(rounds: int) -> Optional[Any]:
        """Ajuda interna de make es callback pra deixar a execucao mais lisa."""
        cb = getattr(getattr(xgb, "callback", None), "EarlyStopping", None)
        if cb is None:
            return None
        try:
            return cb(
                rounds=int(rounds),
                metric_name=str(cfg.xgb_eval_metric),
                data_name="validation_0",
                save_best=True,
                maximize=bool(maximize),
            )
        except TypeError:
            try:
                return cb(int(rounds))
            except Exception:
                return None

    if early_req > 0:
        clf0 = _try_init({"early_stopping_rounds": int(early_req)})
        if clf0 is not None and _try_fit(clf0, {}):
            clf = clf0
            early_used = True
            early_mode = "ctor:early_stopping_rounds"
        else:
            clf1 = XGBClassifier(**params)
            if _try_fit(clf1, {"early_stopping_rounds": int(early_req)}):
                clf = clf1
                early_used = True
                early_mode = "fit:early_stopping_rounds"
            else:
                es = _make_es_callback(int(early_req))
                if es is not None:
                    clf2 = _try_init({"callbacks": [es]})
                    if clf2 is not None and _try_fit(clf2, {}):
                        clf = clf2
                        early_used = True
                        early_mode = "ctor:callbacks"
                    else:
                        clf3 = XGBClassifier(**params)
                        if _try_fit(clf3, {"callbacks": [es]}):
                            clf = clf3
                            early_used = True
                            early_mode = "fit:callbacks"
                        else:
                            logger.warning(
                                "XGBoost early stopping requested (rounds=%d) but not supported; continuing without it. xgboost_version=%s",
                                int(early_req), str(xgb_version),
                            )
                            clf = XGBClassifier(**params)
                            try:
                                clf.fit(X_train, y_train, **fit_common)
                            except TypeError as e:
                                if _unexpected_kwarg(e):
                                    clf.fit(X_train, y_train, verbose=False)
                                else:
                                    raise
                else:
                    logger.warning(
                        "XGBoost early stopping requested (rounds=%d) but callback unavailable; continuing without it. xgboost_version=%s",
                        int(early_req), str(xgb_version),
                    )
                    clf = XGBClassifier(**params)
                    try:
                        clf.fit(X_train, y_train, **fit_common)
                    except TypeError as e:
                        if _unexpected_kwarg(e):
                            clf.fit(X_train, y_train, verbose=False)
                        else:
                            raise
    else:
        clf = XGBClassifier(**params)
        try:
            clf.fit(X_train, y_train, **fit_common)
        except TypeError as e:
            if _unexpected_kwarg(e):
                clf.fit(X_train, y_train, verbose=False)
            else:
                raise

    best_iter = getattr(clf, "best_iteration", None)
    if isinstance(best_iter, int) and best_iter >= 0:
        try:
            probs = clf.predict_proba(X_test, iteration_range=(0, best_iter + 1))[:, 1]
        except TypeError:
            ntree = getattr(clf, "best_ntree_limit", None)
            if ntree is not None:
                probs = clf.predict_proba(X_test, ntree_limit=int(ntree))[:, 1]
            else:
                probs = clf.predict_proba(X_test)[:, 1]
    else:
        probs = clf.predict_proba(X_test)[:, 1]

    probs_test = probs.astype(float).tolist()
    labels_test = [int(x) for x in y_test.tolist()]
    eval_bundle = evaluate_binary_from_arrays(probs=probs_test, labels=labels_test, name="xgboost", cfg=cfg, split="test", threshold=0.5)

    eval_curve_file = None
    try:
        res = clf.evals_result()
        if isinstance(res, dict):
            ds0 = next(iter(res.values())) if res else None
            if isinstance(ds0, dict) and ds0:
                metric_name = next(iter(ds0.keys()))
                ys = ds0.get(metric_name) or []
                xs = list(range(1, len(ys) + 1))
                eval_curve_file = plot_line(xs, ys, title=f"XGBoost validation {metric_name} vs iteration", xlabel="Iteration", ylabel=metric_name, out_path=eda_dir / "xgboost_val_curve.png", cfg=cfg)
    except Exception:
        pass

    importance_file = None
    try:
        booster = clf.get_booster()
        score = booster.get_score(importance_type="gain") or {}
        items = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:20]
        if items:
            labels = [k for k, _ in items]
            values = [float(v) for _, v in items]
            importance_file = plot_barh(labels, values, title="XGBoost feature importance (gain) – top 20", xlabel="gain", ylabel="feature", out_path=eda_dir / "xgboost_feature_importance.png", cfg=cfg)
    except Exception:
        pass

    model_path = models_dir / "xgboost_model.json"
    try:
        clf.get_booster().save_model(str(model_path))
    except Exception:
        model_path = models_dir / "xgboost_model.ubj"
        try:
            clf.get_booster().save_model(str(model_path))
        except Exception:
            model_path = models_dir / "xgboost_model.bin"
            clf.get_booster().save_model(str(model_path))

    out = {
        "generated_at": _utc_now_iso(),
        "status": "ok",
        "representation": str(rep),
        "train_rows": int(len(y_train) + len(y_val)),
        "val_rows": int(len(y_val)),
        "test_rows": int(len(y_test)),
        "scale_pos_weight": float(scale_pos_weight),
        "params": params,
        "early_stopping": {"requested_rounds": int(early_req), "used": bool(early_used), "mode": str(early_mode), "xgboost_version": str(xgb_version)},
        "best_iteration": int(getattr(clf, "best_iteration", -1)) if getattr(clf, "best_iteration", None) is not None else None,
        "best_score": float(getattr(clf, "best_score", float("nan"))) if getattr(clf, "best_score", None) is not None else None,
        "eval_json": "eval_xgboost_test.json",
        "eval_metrics": eval_bundle.get("metrics"),
        "plots": {"val_curve": eval_curve_file, "feature_importance": importance_file},
        "model_path": _rel_to_output(cfg, model_path),
        "seconds": float(time.time() - t0),
    }
    _write_json(metrics_dir / "xgboost.json", out)
    _write_json(metrics_dir / "xgboost_baseline.json", out)
    return out

def _persist_xgb_worker_inputs(
    *,
    X_train: Any,
    X_val: Any,
    X_test: Any,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    rep: str,
    cfg: Config,
) -> Tuple[Path, Dict[str, str]]:
    """Ajuda interna de salva xgb worker entradas pra deixar a execucao mais lisa."""
    root = Path(cfg.output_dir) / "artifacts" / "xgb_worker"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, str] = {}

    if rep == "sparse":
        if sp is None:
            raise RuntimeError("scipy is required to persist sparse XGBoost matrices but is not available.")
        xtr = run_dir / "X_train.npz"
        xva = run_dir / "X_val.npz"
        xte = run_dir / "X_test.npz"
        sp.save_npz(str(xtr), X_train)
        sp.save_npz(str(xva), X_val)
        sp.save_npz(str(xte), X_test)
        paths["X_train"] = str(xtr)
        paths["X_val"] = str(xva)
        paths["X_test"] = str(xte)
    else:
        xtr = run_dir / "X_train.npy"
        xva = run_dir / "X_val.npy"
        xte = run_dir / "X_test.npy"
        np.save(str(xtr), np.asarray(X_train, dtype=np.float32))
        np.save(str(xva), np.asarray(X_val, dtype=np.float32))
        np.save(str(xte), np.asarray(X_test, dtype=np.float32))
        paths["X_train"] = str(xtr)
        paths["X_val"] = str(xva)
        paths["X_test"] = str(xte)

    ytr = run_dir / "y_train.npy"
    yva = run_dir / "y_val.npy"
    yte = run_dir / "y_test.npy"
    np.save(str(ytr), np.asarray(y_train, dtype=np.int32))
    np.save(str(yva), np.asarray(y_val, dtype=np.int32))
    np.save(str(yte), np.asarray(y_test, dtype=np.int32))
    paths["y_train"] = str(ytr)
    paths["y_val"] = str(yva)
    paths["y_test"] = str(yte)

    return run_dir, paths


def _train_xgboost_in_isolated_process(
    *,
    X_train: Any,
    X_val: Any,
    y_train: np.ndarray,
    y_val: np.ndarray,
    X_test: Any,
    y_test: np.ndarray,
    rep: str,
    cfg: Config,
) -> Dict[str, Any]:
    """Ajuda interna de treina xgboost in isolado processo pra deixar a execucao mais lisa."""
    metrics_dir = Path(cfg.output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    keep = _env_bool("AMAZON_NLP_XGB_KEEP_WORKER_ARTIFACTS", False)
    run_dir: Optional[Path] = None

    try:
        run_dir, paths = _persist_xgb_worker_inputs(
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            rep=rep, cfg=cfg,
        )

        payload = {
            "cfg": asdict(cfg),
            "rep": str(rep),
            "paths": paths,
            "output_dir": cfg.output_dir,
        }

# usa spawn pra reduzir risco de travar com libs nativas
        ctx = mp.get_context("spawn")
        proc = ctx.Process(target=_xgb_worker_train, args=(payload,), daemon=False)

        proc.start()
        proc.join()

        if proc.exitcode != 0:
# o worker devia ter escrito um json de skip se nao escreve aqui
            existing = _read_json(metrics_dir / "xgboost.json") or _read_json(metrics_dir / "xgboost_baseline.json")
            if isinstance(existing, dict):
                out = dict(existing)
# deixa a falha bem clara
                if out.get("status") == "ok":
                    out["status"] = "skipped"
                out["reason"] = str(out.get("reason") or "XGBoost worker failed") + f" (exitcode={proc.exitcode})"
            else:
                out = {
                    "generated_at": _utc_now_iso(),
                    "status": "skipped",
                    "reason": f"XGBoost worker failed (exitcode={proc.exitcode}).",
                }
            _write_json(metrics_dir / "xgboost.json", out)
            _write_json(metrics_dir / "xgboost_baseline.json", out)
            return out

        out = _read_json(metrics_dir / "xgboost.json") or _read_json(metrics_dir / "xgboost_baseline.json")
        if not isinstance(out, dict):
            out = {
                "generated_at": _utc_now_iso(),
                "status": "skipped",
                "reason": "XGBoost worker finished but metrics/xgboost.json was not produced.",
            }
            _write_json(metrics_dir / "xgboost.json", out)
            _write_json(metrics_dir / "xgboost_baseline.json", out)
        return out

    finally:
        if run_dir is not None and (not keep):
            shutil.rmtree(run_dir, ignore_errors=True)


def train_xgboost_baseline(train_df: DataFrame, test_df: DataFrame, cfg: Config) -> Dict[str, Any]:
    """Treina uma linha base de xgboost e salva modelo e metricas."""
    metrics_dir = Path(cfg.output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.enable_xgboost:
        out = {"generated_at": _utc_now_iso(), "status": "disabled"}
        _write_json(metrics_dir / "xgboost.json", out)
        _write_json(metrics_dir / "xgboost_baseline.json", out)
        return out

    if XGBClassifier is None or xgb is None:
        raise RuntimeError(
            "XGBoost is enabled (enable_xgboost=True) but the 'xgboost' package could not be imported. "
            "Install it (e.g., pip install xgboost) or disable with AMAZON_NLP_ENABLE_XGBOOST=0."
        ) from _XGBOOST_IMPORT_ERROR

# coleta as matrizes
    try:
        X_train, X_val, y_train, y_val, X_test, y_test, rep = _collect_xgb_matrices(train_df, test_df, cfg)
    except Exception as e:
        out = {"generated_at": _utc_now_iso(), "status": "skipped", "reason": str(e)}
        _write_json(metrics_dir / "xgboost.json", out)
        _write_json(metrics_dir / "xgboost_baseline.json", out)
        return out

    # politica padrao de isolamento
    # no mac isola por padrao pra evitar instabilidade
    # no resto roda no processo principal por padrao
    isolate_default = (platform.system() == "Darwin")
    isolate = _env_bool("AMAZON_NLP_XGB_ISOLATE_PROCESS", isolate_default)

    try:
        if isolate:
            out = _train_xgboost_in_isolated_process(
                X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val,
                X_test=X_test, y_test=y_test, rep=rep, cfg=cfg,
            )
        else:
            out = _train_xgboost_from_matrices(
                X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val,
                X_test=X_test, y_test=y_test, rep=rep, cfg=cfg,
            )
    except Exception as e:
    # linha base nao pode derrubar o resto entao marca como pulado
        out = {"generated_at": _utc_now_iso(), "status": "skipped", "reason": f"XGBoost failed: {e}"}
        _write_json(metrics_dir / "xgboost.json", out)
        _write_json(metrics_dir / "xgboost_baseline.json", out)
        return out
    finally:
    # libera memoria depois de objetos grandes
        try:
            gc.collect()
        except Exception:
            pass

    refresh_leaderboards(cfg)
    return out

def train_supervised_models_fast(features_df: DataFrame, cfg: Config) -> Tuple[Dict[str, Any], Any, str]:
    """Treina modelos rapidos e gera metricas pro relatorio."""
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

    def _fnum(v: Any) -> float:
        """Ajuda interna de fnum pra deixar a execucao mais lisa."""
        try:
            return float(v) if v is not None else float("nan")
        except Exception:
            return float("nan")



    # Logistic Regression CV

    lr = LogisticRegression(featuresCol="tfidf_features", labelCol="label", maxIter=20)
    lr_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [float(x) for x in cfg.logreg_reg_params])
        .addGrid(lr.elasticNetParam, [float(x) for x in cfg.logreg_l1_ratios])
        .build()
    )
    lr_eval_auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    logger.info("Supervised(dev): starting LR CV (grid=%d folds=%d)", len(lr_grid), int(cfg.cv_folds))
    t0 = time.time()
    lr_cv = CrossValidator(
        estimator=lr,
        estimatorParamMaps=lr_grid,
        evaluator=lr_eval_auc,
        numFolds=int(cfg.cv_folds),
        parallelism=int(cfg.cv_parallelism),
    )
    
    lr_train = train_df
    if cfg.lr_max_train_rows and cfg.lr_max_train_rows > 0:
        lr_train = lr_train.limit(int(cfg.lr_max_train_rows))

    lr_cv_model = lr_cv.fit(lr_train)

    lr_best = lr_cv_model.bestModel
    lr_pred = lr_best.transform(test_df).persist(StorageLevel.MEMORY_AND_DISK)
    _ = lr_pred.count()

    lr_eval_bundle = evaluate_binary_classifier(lr_pred.select("label", "prediction", "probability"), name="logreg_cv", cfg=cfg, split="test")
    m = lr_eval_bundle.get("metrics") or {}
    plots = lr_eval_bundle.get("plots") or {}
    out["logreg_cv"] = {
        "auc_roc": m.get("auc_roc"),
        "auc_pr": m.get("auc_pr"),
        "accuracy": m.get("accuracy"),
        "f1": m.get("f1"),
        "precision": m.get("precision_pos"),
        "recall": m.get("recall_pos"),
        "brier": m.get("brier"),
        "confusion": lr_eval_bundle.get("confusion"),
        "threshold_tuning": lr_eval_bundle.get("threshold_tuning"),
        "calibration_bins": lr_eval_bundle.get("calibration_bins"),
        "curve_rows_cap": lr_eval_bundle.get("curve_rows_cap"),
        "eval_json": "eval_logreg_cv_test.json",
        "best_params": {
            "regParam": float(_maybe_call(lr_best, "getRegParam") or 0.0),
            "elasticNetParam": float(_maybe_call(lr_best, "getElasticNetParam") or 0.0),
        },
        "grid_size": int(len(lr_grid)),
        "cv_folds": int(cfg.cv_folds),
        "seconds": float(time.time() - t0),
        "files": {"confusion": (plots or {}).get("cm")},

    }
    logger.info("Supervised(dev): LR done | auc_roc=%.4f acc=%.4f f1=%.4f", _fnum(m.get("auc_roc")), _fnum(m.get("accuracy")), _fnum(m.get("f1")))


    best_model = lr_best
    best_name = "logreg_cv"
    
    best_auc = _fnum(m.get("auc_roc"))

    lr_pred.unpersist()


    # random forest com validacao cruzada se estiver ligado
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

        logger.info("Supervised(dev): starting RF CV (grid=%d folds=%d cap_rows=%d)", len(rf_grid), int(cfg.cv_folds), int(cfg.rf_max_train_rows))
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

        rf_eval_bundle = evaluate_binary_classifier(rf_pred.select("label", "prediction", "probability"), name="random_forest_cv", cfg=cfg, split="test")
        m2 = rf_eval_bundle.get("metrics") or {}
        plots2 = rf_eval_bundle.get("plots") or {}
        out["random_forest_cv"] = {
            "auc_roc": m2.get("auc_roc"),
            "auc_pr": m2.get("auc_pr"),
            "accuracy": m2.get("accuracy"),
            "f1": m2.get("f1"),
            "precision": m2.get("precision_pos"),
            "recall": m2.get("recall_pos"),
            "brier": m2.get("brier"),
            "confusion": rf_eval_bundle.get("confusion"),
            "threshold_tuning": rf_eval_bundle.get("threshold_tuning"),
            "calibration_bins": rf_eval_bundle.get("calibration_bins"),
            "curve_rows_cap": rf_eval_bundle.get("curve_rows_cap"),
            "eval_json": "eval_random_forest_cv_test.json",
            "best_params": {
                "numTrees": int(_maybe_call(rf_best, "getNumTrees") or 0),
                "maxDepth": int(_maybe_call(rf_best, "getMaxDepth") or 0),
            },
            "grid_size": int(len(rf_grid)),
            "cv_folds": int(cfg.cv_folds),
            "train_rows_cap": int(cfg.rf_max_train_rows),
            "seconds": float(time.time() - t1),
            "plots": plots2,
        }

        logger.info("Supervised(dev): RF done | auc_roc=%.4f acc=%.4f f1=%.4f", _fnum(m2.get("auc_roc")), _fnum(m2.get("accuracy")), _fnum(m2.get("f1")))

        rf_auc = _fnum(m2.get("auc_roc"))
        if (best_auc != best_auc) or (rf_auc == rf_auc and rf_auc > best_auc):  # handle NaN
            best_model = rf_best
            best_name = "random_forest_cv"
            best_auc = rf_auc

        rf_pred.unpersist()
        rf_train.unpersist()


    # linha base do xgboost opcional

    xgb_metrics = train_xgboost_baseline(train_df, test_df, cfg) if cfg.enable_xgboost else {"status": "disabled"}

    if isinstance(xgb_metrics, dict) and xgb_metrics.get("status") == "ok":
        # le o json de avaliacao que o avaliador escreveu para o xgboost
        xgb_eval = _read_json(Path(cfg.output_dir) / "metrics" / "eval_xgboost_test.json") or {}
        xm = (xgb_eval.get("metrics") or {}) if isinstance(xgb_eval, dict) else {}
        out["xgboost"] = {
            "auc_roc": xm.get("auc_roc"),
            "auc_pr": xm.get("auc_pr"),
            "accuracy": xm.get("accuracy"),
            "f1": xm.get("f1"),
            "precision": xm.get("precision_pos"),
            "recall": xm.get("recall_pos"),
            "brier": xm.get("brier"),
            "eval_json": "eval_xgboost_test.json",
            "seconds": xgb_metrics.get("seconds"),
            "plots": (xgb_eval.get("plots") or {}),
            "model_path": xgb_metrics.get("model_path"),
        }


    # grafico de comparacao pra bater o olho no resultado dos modelos supervisionados

    names = [k for k in out.keys() if not k.startswith("_")]
    aucs = [_fnum(out[n].get("auc_roc")) for n in names]
    aucprs = [_fnum(out[n].get("auc_pr")) for n in names]
    accs = [_fnum(out[n].get("accuracy")) for n in names]
    f1s = [_fnum(out[n].get("f1")) for n in names]

    plots_dir = Path(cfg.output_dir) / "eda"
    plots_dir.mkdir(parents=True, exist_ok=True)
    comp_file = plot_grouped_bars(cfg, x_labels=names, series={"AUC_ROC": aucs, "AUC_PR": aucprs, "Acuracia": accs, "F1": f1s}, title="Modelos supervisionados (dev) – comparação", xlabel="modelo", ylabel="metrica", out_path=plots_dir / "supervised_metric_comparison.png", rotate=15, figsize=(12, 4))

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

# Refresh leaderboards now that eval_.json exist
    refresh_leaderboards(cfg)
    return out, best_model, best_name

def save_best_supervised_pipeline(idf_model: IDFModel, best_model: Any, cfg: Config) -> Path:
    """Salva salva melhor supervisionado pipeline pra reusar depois sem retrabalho."""
    tok = RegexTokenizer(inputCol="text_full", outputCol="tokens", pattern=r"\W+", toLowercase=True)
    sw = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", locale="en_US")
    hashing = HashingTF(inputCol="filtered_tokens", outputCol="raw_features", numFeatures=int(cfg.hashing_num_features))
    pm = PipelineModel(stages=[tok, sw, hashing, idf_model, best_model])
    out_dir = Path(cfg.output_dir) / "models" / "supervised_pipeline"
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    pm.write().overwrite().save(str(out_dir))
    logger.info("Saved supervised pipeline: %s", str(out_dir))
    return out_dir


# ajudinhas pro modo score only
def score_with_saved_supervised_model(spark: SparkSession, df: DataFrame, cfg: Config) -> DataFrame:
    """Aplica o pipeline supervisionado salvo pra gerar predicao sem retreinar."""
    model_path = Path(cfg.output_dir) / "models" / "supervised_pipeline"
    if not model_path.exists():
        raise RuntimeError(f"Missing supervised model at {model_path}. Run dev/train_full first.")
    pm = PipelineModel.load(str(model_path))
    pred = pm.transform(df)
    return pred.select("review_id", "dataset_split", "sentiment", F.col("prediction").cast("int").alias("prediction"), F.col("probability"))


# treino completo do supervisionado
def train_full_supervised_pipeline(spark: SparkSession, df: DataFrame, cfg: Config) -> Dict[str, Any]:
    """Treina o supervisionado completo e salva tudo pra usar depois."""
    best_path = Path(cfg.output_dir) / "metrics" / "supervised_best.json"
    if not best_path.exists():
        raise RuntimeError(f"Missing {best_path}. Run dev first.")

    best = json.loads(best_path.read_text(encoding="utf-8")) or {}
    lr_params = best.get("logreg_best_params") or {}
    reg_param = float(lr_params.get("regParam", 0.1))
    enet = float(lr_params.get("elasticNetParam", 0.0))

    d0 = df.select("dataset_split", "sentiment", "text_full").where(F.col("sentiment").isNotNull()).where(F.col("text_full").isNotNull())
    d0 = _add_split_bucket(d0, cfg)
    train_cond, test_cond = _train_test_filters(cfg)

    train_raw = d0.where(train_cond).select("text_full", F.col("sentiment").cast("double").alias("label"))
    test_raw = d0.where(test_cond).select("text_full", F.col("sentiment").cast("double").alias("label"))

    tok = RegexTokenizer(inputCol="text_full", outputCol="tokens", pattern=r"\W+", toLowercase=True)
    sw = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", locale="en_US")
    hashing = HashingTF(inputCol="filtered_tokens", outputCol="raw_features", numFeatures=int(cfg.hashing_num_features))
    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
    preproc = Pipeline(stages=[tok, sw, hashing, idf])

    logger.info("train_full: fitting preprocessing (tok->sw->hash->idf) ...")
    t0 = time.time()
    preproc_model = preproc.fit(train_raw.select("text_full"))

    logger.info("train_full: transforming TF-IDF + caching train features (DISK_ONLY) ...")
    train_feat = preproc_model.transform(train_raw).select("tfidf_features", "label").persist(StorageLevel.DISK_ONLY)
    _ = train_feat.count()
    test_feat = preproc_model.transform(test_raw).select("tfidf_features", "label")

    lr = LogisticRegression(featuresCol="tfidf_features", labelCol="label", maxIter=20, regParam=reg_param, elasticNetParam=enet)
    logger.info("train_full: fitting LogisticRegression (regParam=%.4f elasticNetParam=%.4f) ...", reg_param, enet)
    lr_model = lr.fit(train_feat)
    pred = lr_model.transform(test_feat)

    eval_bundle = evaluate_binary_classifier(pred.select("label", "prediction", "probability"), name="logreg_full", cfg=cfg, split="test")

    final_model = PipelineModel(stages=list(preproc_model.stages) + [lr_model])
    out_dir = Path(cfg.output_dir) / "models" / "supervised_pipeline"
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    final_model.write().overwrite().save(str(out_dir))
    logger.info("train_full: saved supervised pipeline: %s", str(out_dir))

    metrics = {
        "generated_at": _utc_now_iso(),
        "model": "logreg_full",
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


# pipeline de clustering com pca e kmeans

def fit_and_save_cluster_pipeline(spark: SparkSession, df: DataFrame, cfg: Config) -> Dict[str, Any]:
    """Treina o clustering e salva o pipeline pra reusar depois."""
    out_dir = Path(cfg.output_dir)
    eda_dir = out_dir / "eda"
    metrics_dir = out_dir / "metrics"
    eda_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)


# base com limite e persistida pra nao recalcular

    base = (
        df.select("text_full", "sentiment")
          .where(F.col("text_full").isNotNull())
          .where(F.col("sentiment").isNotNull())
    )
    if cfg.embedding_max_rows and cfg.embedding_max_rows > 0:
        base = base.limit(int(cfg.embedding_max_rows))

    parts = int(min(max(2, cfg.spark_default_parallelism), 64))
    base = base.repartition(parts).persist(StorageLevel.MEMORY_AND_DISK)
    base_rows = int(base.count())


    # limita o hash dim pra nao dar overflow
    requested_dim = int(cfg.cluster_hashing_num_features)
    safe_dim = _safe_pca_input_dim(
        spark,
        requested_dim=requested_dim,
        requested_k=int(min(cfg.embedding_pca_k_max, requested_dim)),
    )
    if safe_dim != requested_dim:
        logger.warning("Clustering: clamping hash_dim %d -> %d (driver safety)", requested_dim, safe_dim)


    # pipeline de features com tfidf e normalizacao
    tok = RegexTokenizer(inputCol="text_full", outputCol="tokens", pattern=r"\W+", toLowercase=True)
    sw = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", locale="en_US")
    hashing = HashingTF(inputCol="filtered_tokens", outputCol="raw_features", numFeatures=int(safe_dim))
    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
    norm = Normalizer(inputCol="tfidf_features", outputCol="tfidf_norm", p=2.0)

    logger.info("Clustering: building normalized TF-IDF (rows=%d, hash_dim=%d) ...", base_rows, safe_dim)
    t0 = time.time()

    tf = (
        hashing.transform(sw.transform(tok.transform(base)))
        .select("sentiment", "raw_features")
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    _ = tf.count()

    idf_model = idf.fit(tf)

    tfidf = (
        idf_model.transform(tf)
        .select("sentiment", "tfidf_features")
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    _ = tfidf.count()

    tfidf_norm = (
        norm.transform(tfidf)
        .select("sentiment", "tfidf_norm")
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    _ = tfidf_norm.count()


    # ajuste densifica so depois do tfidf norm pra o pca funcionar
    densify_norm: Optional[SQLTransformer] = None
    tfidf_norm_for_pca: Optional[DataFrame] = None

    # escolha do pca pela curva de variancia explicada
    pca_max_k = int(min(int(cfg.embedding_pca_k_max), int(safe_dim)))
    pca_var_target = float(max(0.0, min(0.999, cfg.pca_var_target)))
    use_pca = (pca_max_k >= 2) and (pca_var_target > 0.0)

    # o pca do spark centraliza a media e pede vetor denso
    # so liga pca se tiver array_to_vector
    if use_pca and array_to_vector is None:
        logger.warning("Clustering: array_to_vector unavailable (Spark < 3.1). Disabling PCA for clustering.")
        use_pca = False

    if use_pca:
    # garante a udf nessa sessao pra nao falhar na hora
        _register_vector_udfs(spark)

        densify_norm = SQLTransformer(
            statement="SELECT *, to_dense_vector(tfidf_norm) AS tfidf_norm_dense FROM __THIS__"
        )
        tfidf_norm_for_pca = densify_norm.transform(tfidf_norm).persist(StorageLevel.MEMORY_AND_DISK)
        _ = tfidf_norm_for_pca.count()
    

    # pca opcional pra criar embedding vec
    pca_k_chosen = 0
    pca_cum: List[float] = []
    pca_plot = None
    pca_model = None

    if use_pca:
        assert tfidf_norm_for_pca is not None

        logger.info("Clustering: fitting PCA (max_k=%d) for variance curve ...", pca_max_k)

        # primeiro pca so pra pegar a variancia explicada
        pca_tmp = PCA(k=int(pca_max_k), inputCol="tfidf_norm_dense", outputCol="pca_tmp")
        pca_tmp_model = pca_tmp.fit(tfidf_norm_for_pca)

        ev = pca_tmp_model.explainedVariance
        ev_arr = np.array(ev.toArray() if hasattr(ev, "toArray") else list(ev), dtype=float)
        ev_arr = np.nan_to_num(ev_arr, nan=0.0, posinf=0.0, neginf=0.0)

        pca_cum = [float(x) for x in np.cumsum(ev_arr).tolist()]
        pca_k_chosen = int(next((i + 1 for i, v in enumerate(pca_cum) if v >= pca_var_target), pca_max_k))
        pca_k_chosen = int(max(2, min(pca_k_chosen, pca_max_k)))

        pca_plot = plot_line(
            list(range(1, len(pca_cum) + 1)),
            pca_cum,
            title=f"PCA cumulative explained variance (target={pca_var_target:.2f})",
            xlabel="k",
            ylabel="Cumulative variance",
            out_path=eda_dir / "cluster_pca_cumvar.png",
            cfg=cfg,
            vline=float(pca_k_chosen),
        )
        logger.info("Clustering: chosen PCA k=%d (target_var=%.2f)", pca_k_chosen, pca_var_target)

    # pca final com a coluna certa pro kmeans
        pca = PCA(k=int(pca_k_chosen), inputCol="tfidf_norm_dense", outputCol="embedding_vec")
        pca_model = pca.fit(tfidf_norm_for_pca)

        embed_df = (
            pca_model.transform(tfidf_norm_for_pca)
            .select("sentiment", "embedding_vec")
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        _ = embed_df.count()
    else:
        embed_df = (
            tfidf_norm
            .select("sentiment", F.col("tfidf_norm").alias("embedding_vec"))
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        _ = embed_df.count()


    # teste de k do kmeans usando silhouette

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
        km = KMeans(
            featuresCol="embedding_vec",
            predictionCol="cluster_id",
            k=int(k),
            maxIter=int(cfg.kmeans_max_iter),
            seed=int(cfg.random_state),
        )
        km_model = km.fit(embed_df)
        pred = km_model.transform(embed_df).select("embedding_vec", "cluster_id").persist(StorageLevel.MEMORY_AND_DISK)
        _ = pred.count()
        try:
            if try_cosine:
                evl = ClusteringEvaluator(
                    featuresCol="embedding_vec",
                    predictionCol="cluster_id",
                    metricName="silhouette",
                    distanceMeasure="cosine",
                )
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
        cfg=cfg,
        vline=float(best_k),
    )
    logger.info("Clustering: chosen kmeans k=%d (silhouette=%.4f)", best_k, best_s)


    # treino final do kmeans

    kmeans = KMeans(
        featuresCol="embedding_vec",
        predictionCol="cluster_id",
        k=int(best_k),
        maxIter=int(cfg.kmeans_max_iter),
        seed=int(cfg.random_state),
    )
    kmeans_model = kmeans.fit(embed_df)

    clustered = (
        kmeans_model.transform(embed_df)
        .select("sentiment", "embedding_vec", "cluster_id")
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    _ = clustered.count()

    try:
        ev_final = ClusteringEvaluator(
            featuresCol="embedding_vec",
            predictionCol="cluster_id",
            metricName="silhouette",
            distanceMeasure="cosine",
        )
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
    prof_rows = [r.asDict(True) for r in prof]

    size_plot = None
    mean_plot = None
    if prof_rows:
        labels = [str(r["cluster_id"]) for r in prof_rows]
        sizes = [int(r["n"]) for r in prof_rows]
        means = [float(r["mean_sentiment"]) for r in prof_rows]
        size_plot = plot_bar(
            labels, sizes,
            title="Cluster sizes",
            xlabel="cluster_id",
            ylabel="n",
            out_path=eda_dir / "cluster_sizes.png",
            cfg=cfg,
            rotate=0,
            figsize=(11, 4),
        )
        mean_plot = plot_bar(
            labels, means,
            title="Mean sentiment by cluster",
            xlabel="cluster_id",
            ylabel="mean(sentiment)",
            out_path=eda_dir / "cluster_mean_sentiment.png",
            cfg=cfg,
            rotate=0,
            figsize=(11, 4),
        )


    # salva o modelo do pipeline de clustering

    stages: List[Any] = [tok, sw, hashing, idf_model, norm]
    if pca_model is not None:
    # densify stage is required so scoring also creates tfidf_norm_dense for PCA
        assert densify_norm is not None
        stages.append(densify_norm)
        stages.append(pca_model)
    else:
        # fornece embedding_vec diretamente quando o pca esta desligado
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
        "cluster_profiles": prof_rows,
        "plots": {
            "pca_cumvar": pca_plot,
            "silhouette_by_k": sil_plot,
            "cluster_sizes": size_plot,
            "cluster_mean_sentiment": mean_plot,
        },
        "files": {
            "pca_cumvar": pca_plot,
            "silhouette_by_k": sil_plot,
            "cluster_sizes": size_plot,
            "cluster_mean_sentiment": mean_plot,
        },
        "model_path": str(model_path),
        "seconds": float(time.time() - t0),
    }
    _write_json(metrics_dir / "cluster_kmeans.json", out)


    # limpeza

    try:
        clustered.unpersist()
    except Exception:
        pass
    try:
        embed_df.unpersist()
    except Exception:
        pass
    try:
        if tfidf_norm_for_pca is not None:
            tfidf_norm_for_pca.unpersist()
    except Exception:
        pass
    try:
        tfidf_norm.unpersist()
    except Exception:
        pass
    try:
        tfidf.unpersist()
    except Exception:
        pass
    try:
        tf.unpersist()
    except Exception:
        pass
    try:
        base.unpersist()
    except Exception:
        pass

    return out

def score_with_saved_cluster_model(spark: SparkSession, df: DataFrame, cfg: Config) -> DataFrame:
    """Aplica o modelo de clustering salvo pra gerar clusters."""
    _register_vector_udfs(spark)  #  precisa porque o sqltransformer salvo usa a udf

    model_path = Path(cfg.output_dir) / "models" / "cluster_pipeline"
    if not model_path.exists():
        raise RuntimeError(f"Missing cluster model at {model_path}. Run dev first.")
    pm = PipelineModel.load(str(model_path))
    scored = pm.transform(df)
    return scored.select("review_id", "dataset_split", F.col("cluster_id").cast("int").alias("cluster_id"))


# modernbert opcional

@dataclass
class TorchSplitBundle:
    train_texts: List[str]
    train_labels: List[int]
    test_texts: List[str]
    test_labels: List[int]
    meta: Dict[str, Any]


def collect_train_test_texts_labels(df_bucket: DataFrame, cfg: Config) -> TorchSplitBundle:
    """Coleta textos e labels em split fixo pra treinar modernbert depois."""
    train_cond, test_cond = _train_test_filters(cfg)
    base = df_bucket.select("text_full", "sentiment", "dataset_split", "split_bucket")
    base = base.where(F.col("text_full").isNotNull()).where(F.col("sentiment").isNotNull())
    train_df = base.where(train_cond).select("text_full", "sentiment")
    test_df = base.where(test_cond).select("text_full", "sentiment")

    train_n = int(train_df.count())
    test_n = int(test_df.count())
    total = train_n + test_n

    cap_total = int(cfg.torch_max_rows) if (cfg.torch_max_rows and cfg.torch_max_rows > 0) else total
    cap_total = min(cap_total, total) if total else 0

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

# amostra deterministica pra bater entre runs
    seed = int(cfg.random_state)
    train_rows = (train_df.orderBy(F.xxhash64(F.col("text_full"), F.lit(seed))).limit(int(cap_train)).collect() if cap_train > 0 else [])
    test_rows = (test_df.orderBy(F.xxhash64(F.col("text_full"), F.lit(seed))).limit(int(cap_test)).collect() if cap_test > 0 else [])

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


def train_modernbert_sentiment_from_splits(bundle: TorchSplitBundle, cfg: Config) -> Dict[str, Any]:
    """Treina modernbert simples pra ter um baseline neural."""
    _ensure_torch_imported()
    if torch is None:
        raise RuntimeError(f"torch import failed: {_TORCH_IMPORT_ERROR}")

    from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
    from torch.utils.data import TensorDataset, DataLoader  # type: ignore
    from torch.optim import AdamW  # type: ignore

    train_texts, train_labels = bundle.train_texts, bundle.train_labels
    test_texts, test_labels = bundle.test_texts, bundle.test_labels

    if len(train_texts) < 3 or not test_texts:
        raise RuntimeError(
            f"ModernBERT requires at least 3 train samples and 1 test sample; "
            f"got train={len(train_texts)}, test={len(test_texts)}."
        )

    device = _best_torch_device()
    use_amp = (device == "cuda")
    pin = (device == "cuda")

    batch_size = int(cfg.modernbert_batch_size)
    if device == "cpu":
        batch_size = min(batch_size, 16)
    if device == "mps":
        batch_size = min(batch_size, 32)
    elif device == "cuda":
    # heuristica bem simples pelo tamanho da sequencia
        if cfg.modernbert_max_seq_len > 128:
            batch_size = min(batch_size, 32)
        if cfg.modernbert_max_seq_len > 256:
            batch_size = min(batch_size, 16)

    use_amp = (device == "cuda") and _env_bool("AMAZON_NLP_ENABLE_AMP", True)

    logger.info(
        "ModernBERT: device=%s | amp=%s | batch_size=%d | max_len=%d | train=%d | test=%d",
        device, use_amp, batch_size, int(cfg.modernbert_max_seq_len), len(train_texts), len(test_texts)
    )

    # ajuste de threads pra nao brigar
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

    enc = tokenizer(train_texts, truncation=True, padding="max_length", max_length=int(cfg.modernbert_max_seq_len), return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    y = torch.tensor(train_labels, dtype=torch.long)

    tr_ds = TensorDataset(input_ids[tr_idx], attention_mask[tr_idx], y[tr_idx])
    va_ds = TensorDataset(input_ids[val_idx], attention_mask[val_idx], y[val_idx])

    train_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    # onde guarda o melhor checkpoint
    best_val_acc = -1.0
    best_state_dir = Path(cfg.output_dir) / "torch_models" / "modernbert_sentiment"
    best_state_dir.mkdir(parents=True, exist_ok=True)

    # modelo comeca do pretrained base
    model = AutoModelForSequenceClassification.from_pretrained(
        str(cfg.modernbert_model_name),
        num_labels=2,
        trust_remote_code=True
    ).to(device)

    # importante nao recarrega o tokenizer daqui no primeiro run
    # usa o tokenizer que ja foi carregado la em cima pra nao confundir

    optim = AdamW(model.parameters(), lr=float(cfg.modernbert_lr))

    scaler = None
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)  # type: ignore[attr-defined]
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=True)  # type: ignore[attr-defined]

    def _eval_loader(loader: DataLoader) -> Tuple[float, float, float, List[float], List[int]]:
        """Ajuda interna de eval loader pra deixar a execucao mais lisa."""
        model.eval()
        losses: List[float] = []
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

    t_start = time.time()
    global_step = 0
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
                logger.info("ModernBERT ep=%d step=%d/%d | avg_loss=%.4f | it/s=%.2f | device=%s", ep + 1, step, steps_total, avg_loss, it_s, device)

        avg_train_loss = float(running_loss / max(1, running_steps))
        val_acc, val_auc, val_loss, _, _ = _eval_loader(val_loader)
        train_loss_ep.append(avg_train_loss)
        val_loss_ep.append(float(val_loss))
        val_acc_ep.append(float(val_acc))
        val_auc_ep.append(float(val_auc))
        logger.info("ModernBERT epoch=%d done in %.1fs | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f | val_auc=%s", ep + 1, time.time() - t0, avg_train_loss, float(val_loss), float(val_acc), f"{val_auc:.4f}" if not math.isnan(val_auc) else "nan")
        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            model.save_pretrained(str(best_state_dir))
            tokenizer.save_pretrained(str(best_state_dir))

    # avaliacao no teste
    # recarrega o melhor checkpoint salvo no treino
    try:
        model = AutoModelForSequenceClassification.from_pretrained(str(best_state_dir), trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(str(best_state_dir), trust_remote_code=True)
    except Exception as e:
        logger.warning(
            "ModernBERT: could not reload best checkpoint from %s (%s). Evaluating last epoch weights instead.",
            str(best_state_dir),
            str(e),
        )

    # avaliacao no teste
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

# graficos das curvas de treino
    eda_dir = Path(cfg.output_dir) / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    epochs_x = list(range(1, len(train_loss_ep) + 1))
    loss_plot = plot_two_lines(epochs_x, train_loss_ep, val_loss_ep, label1="train_loss", label2="val_loss", title="ModernBERT training curve – loss", xlabel="Epoch", ylabel="Loss", out_path=eda_dir / "modernbert_loss_curve.png", cfg=cfg)
    metrics_plot = plot_two_lines(epochs_x, val_acc_ep, val_auc_ep, label1="val_acc", label2="val_auc", title="ModernBERT validation curve – metrics", xlabel="Epoch", ylabel="Value", out_path=eda_dir / "modernbert_val_metrics_curve.png", cfg=cfg)

# escreve eval json no mesmo formato dos modelos do spark
    eval_bundle = evaluate_binary_from_arrays(probs=[float(x) for x in test_probs], labels=[int(x) for x in test_gold], name="modernbert", cfg=cfg, split="test", threshold=0.5)

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
            "plots": {"loss_curve": loss_plot, "val_metrics_curve": metrics_plot},
        },
        "test_eval": {
            "accuracy": float(test_acc),
            "auc": float(test_auc),
            "loss": float(test_loss),
            "eval_json": "eval_modernbert_test.json",
            "eval_metrics": eval_bundle.get("metrics"),
        },
        "seconds": float(time.time() - t_start),
        "model_dir": str(best_state_dir),
        "files": {"loss_curve": loss_plot, "val_metrics_curve": metrics_plot},
    }
    _write_json(Path(cfg.output_dir) / "metrics" / "modernbert_sentiment.json", metrics)
    refresh_leaderboards(cfg)
    return metrics


# relatorio e manifest

def _html_esc(x: Any) -> str:
    """Ajuda interna de html esc pra deixar a execucao mais lisa."""
    s = "" if x is None else str(x)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def _html_begin(title: str) -> List[str]:
    """Ajuda interna de html begin pra deixar a execucao mais lisa."""
    css = """
    :root{--fg:#222;--muted:#555;--line:#e6e6e6;--bg:#fff;--note:#f9fbff;--note-b:#9bbcff;--warn:#fff9f0;--warn-b:#ffb155}
    *{box-sizing:border-box}
    body{font-family:-apple-system,system-ui,Segoe UI,Roboto,Arial,sans-serif;margin:24px;color:var(--fg);background:var(--bg)}
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
    """Ajuda interna de html end pra deixar a execucao mais lisa."""
    html.append("</body></html>")


def _html_hr(html: List[str]) -> None:
    """Ajuda interna de html hr pra deixar a execucao mais lisa."""
    html.append("<div class='hr'></div>")


def _html_h1(html: List[str], txt: str) -> None:
    """Ajuda interna de html h1 pra deixar a execucao mais lisa."""
    html.append(f"<h1>{_html_esc(txt)}</h1>")


def _html_h2(html: List[str], txt: str) -> None:
    """Ajuda interna de html h2 pra deixar a execucao mais lisa."""
    html.append(f"<h2>{_html_esc(txt)}</h2>")


def _html_h3(html: List[str], txt: str) -> None:
    """Ajuda interna de html h3 pra deixar a execucao mais lisa."""
    html.append(f"<h3>{_html_esc(txt)}</h3>")


def _html_p(html: List[str], txt: str, *, raw: bool = True) -> None:
    """Ajuda interna de html p pra deixar a execucao mais lisa."""
    html.append(f"<p>{txt if raw else _html_esc(txt)}</p>")


def _html_note(html: List[str], txt: str, *, raw: bool = True) -> None:
    """Ajuda interna de html note pra deixar a execucao mais lisa."""
    html.append(f"<p class='note'>{txt if raw else _html_esc(txt)}</p>")


def _html_warn(html: List[str], txt: str, *, raw: bool = True) -> None:
    """Ajuda interna de html warn pra deixar a execucao mais lisa."""
    html.append(f"<p class='warn'>{txt if raw else _html_esc(txt)}</p>")


def _looks_like_number(v: Any) -> bool:
    """Ajuda interna de looks like number pra deixar a execucao mais lisa."""
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


def _fmt_num(v: Any, *, digits: int = 6) -> str:
    """Ajuda interna de fmt num pra deixar a execucao mais lisa."""
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


def _html_table(html: List[str], rows: List[Dict[str, Any]], caption: Optional[str] = None, *, max_rows: int = 50, digits: int = 6) -> None:
    """Ajuda interna de html table pra deixar a execucao mais lisa."""
    if caption:
        html.append(f"<h3>{_html_esc(caption)}</h3>")
    if not rows:
        html.append("<p class='small'>Sem linhas para exibir.</p>")
        return
    rows = rows[:max_rows]
    cols = list(rows[0].keys())
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


def _html_list(html: List[str], items: List[str], *, raw: bool = True) -> None:
    """Ajuda interna de html list pra deixar a execucao mais lisa."""
    if not items:
        html.append("<p class='small'>Lista vazia.</p>")
        return
    li = "".join(f"<li>{it if raw else _html_esc(it)}</li>" for it in items)
    html.append(f"<ul>{li}</ul>")


def _html_img(html: List[str], src: str, caption: Optional[str] = None) -> None:
    """Ajuda interna de html img pra deixar a execucao mais lisa."""
    if not src:
        return
    cap = f"<b>{_html_esc(caption)}</b><br/>" if caption else ""
    html.append(f"<p>{cap}<img src='{_html_esc(src)}' alt='img'></p>")


def _rel_to_output(cfg: Config, p: Path) -> str:
    """Ajuda interna de rel to output pra deixar a execucao mais lisa."""
    try:
        return str(p.relative_to(Path(cfg.output_dir)))
    except Exception:
        return str(p)


def _resolve_asset(cfg: Config, ref: Optional[str]) -> Optional[str]:
    """Ajuda interna de resolve asset pra deixar a execucao mais lisa."""
    if not ref:
        return None
    s = str(ref).strip()
    if not s:
        return None
    if s.startswith("http://") or s.startswith("https://"):
        return s
    base = Path(cfg.output_dir).resolve()
    try:
        p = Path(s)
        if p.is_absolute() and p.exists():
            return _rel_to_output(cfg, p.resolve())
    except Exception:
        pass
    p0 = (base / s).resolve()
    if p0.exists():
        return _rel_to_output(cfg, p0)
    name = Path(s).name
    candidates = [
        base / name,
        base / "eda" / name,
        base / "metrics" / name,
        base / "metrics" / "plots" / name,
        base / "artifacts" / name,
        base / "predictions" / name,
        base / "torch_models" / name,
        base / "xgboost_models" / name,
        base / "models" / name,
        base / "models" / "xgboost" / name,
    ]
    for c in candidates:
        if c.exists():
            return _rel_to_output(cfg, c.resolve())
    try:
        for c in base.rglob(name):
            if c.is_file():
                return _rel_to_output(cfg, c.resolve())
    except Exception:
        pass
    return s


class _ReportBuilder:
    def __init__(self, cfg: Config, title: str) -> None:
        """Ajuda interna de init pra deixar a execucao mais lisa."""
        self.cfg = cfg
        self.html = _html_begin(title)
        self.fig_no = 0
        self.tbl_no = 0

    def h1(self, t: str) -> None:
        """Faz h1 pra manter o pipeline organizado."""
        _html_h1(self.html, t)

    def h2(self, t: str) -> None:
        """Faz h2 pra manter o pipeline organizado."""
        _html_h2(self.html, t)

    def h3(self, t: str) -> None:
        """Faz h3 pra manter o pipeline organizado."""
        _html_h3(self.html, t)

    def p(self, t: str) -> None:
        """Faz p pra manter o pipeline organizado."""
        _html_p(self.html, t, raw=True)

    def note(self, t: str) -> None:
        """Faz note pra manter o pipeline organizado."""
        _html_note(self.html, t, raw=True)

    def warn(self, t: str) -> None:
        """Faz warn pra manter o pipeline organizado."""
        _html_warn(self.html, t, raw=True)

    def lst(self, items: List[str]) -> None:
        """Faz lst pra manter o pipeline organizado."""
        _html_list(self.html, items, raw=True)

    def hr(self) -> None:
        """Faz hr pra manter o pipeline organizado."""
        _html_hr(self.html)

    def fig(self, ref: Optional[str], caption: str) -> None:
        """Faz figura pra manter o pipeline organizado."""
        src = _resolve_asset(self.cfg, ref)
        if not src:
            return
        self.fig_no += 1
        _html_img(self.html, src, f"Figura {self.fig_no} – {caption}")

    def table(self, rows: List[Dict[str, Any]], caption: str, *, max_rows: int = 50, digits: int = 6) -> None:
        """Faz table pra manter o pipeline organizado."""
        self.tbl_no += 1
        _html_table(self.html, rows, f"Tabela {self.tbl_no} – {caption}", max_rows=max_rows, digits=digits)

    def finish(self, out_path: Path) -> Path:
        """Faz finish pra manter o pipeline organizado."""
        _html_end(self.html)
        out_path.write_text("".join(self.html), encoding="utf-8")
        return out_path


def _load_eval_bundles(metrics_dir: Path) -> List[Dict[str, Any]]:
    """Ajuda interna de carrega eval bundles pra deixar a execucao mais lisa."""
    out: List[Dict[str, Any]] = []
    for p in sorted(metrics_dir.glob("eval_*.json")):
        j = _read_json(p)
        if isinstance(j, dict) and isinstance(j.get("metrics"), dict):
            j["_eval_file"] = p.name
            out.append(j)
    return out


def generate_html_report(cfg: Config) -> Path:
    """Gera o html final pra juntar resultados e imagens num lugar so."""
    out_dir = Path(cfg.output_dir)
    metrics_dir = out_dir / "metrics"

    eda_basic = _read_json(metrics_dir / "eda_summary.json") or {}
    eda_adv = _read_json(metrics_dir / "eda_advanced.json") or {}
    eda_decisions = _read_json(metrics_dir / "eda_decisions_applied.json") or {}
    cfg_final = _read_json(metrics_dir / "config_final.json") or {}
    sup = _read_json(metrics_dir / "supervised_models.json") or {}
    sup_best = _read_json(metrics_dir / "supervised_best.json") or {}
    sup_full = _read_json(metrics_dir / "supervised_full.json") or {}
    xgb = _read_json(metrics_dir / "xgboost.json") or _read_json(metrics_dir / "xgboost_baseline.json") or {}

    clu = _read_json(metrics_dir / "cluster_kmeans.json") or {}
    mb = _read_json(metrics_dir / "modernbert_sentiment.json") or {}
    leader = _read_json(metrics_dir / "leaderboards.json") or {}
    manifest = _read_json(metrics_dir / "run_manifest.json") or {}

    rb = _ReportBuilder(cfg, "Amazon Review Polarity – Relatório Final")
    rb.h1("Amazon Review Polarity – Relatório Final")

    # introducao
    rb.h2("1. Introdução e contexto")
    rb.p(
        "Este relatório consolida a execução do pipeline de NLP para <b>classificação de polaridade</b> em reviews. "
        "O fluxo cobre ingestão/limpeza, EDA, modelos supervisionados (TF‑IDF + ML), baseline com <b>XGBoost</b>, "
        "análise não supervisionada (clusters) e (opcionalmente) fine‑tuning de um Transformer (<code>ModernBERT</code>)."
    )

    # kpis
    kpis: List[str] = []
    total_rows = eda_basic.get("total_rows") or eda_basic.get("rows")
    if total_rows is not None:
        kpis.append(f"<span class='kpi'>rows: <b>{_html_esc(total_rows)}</b></span>")
    if sup_best.get("winner"):
        kpis.append(f"<span class='kpi'>winner (dev): <b>{_html_esc(sup_best.get('winner'))}</b></span>")
    if sup_best.get("winner_value") is not None:
        kpis.append(f"<span class='kpi'>AUC (dev): <b>{_html_esc(_fmt_num(sup_best.get('winner_value'), digits=4))}</b></span>")
    sil = clu.get("silhouette_final") if clu else None
    if sil is None and isinstance(clu, dict):
        sil = clu.get("silhouette")
    if sil is not None:
        kpis.append(f"<span class='kpi'>silhouette: <b>{_html_esc(_fmt_num(sil, digits=4))}</b></span>")
    if xgb and isinstance(xgb, dict) and xgb.get("eval_metrics"):
        try:
            kpis.append(f"<span class='kpi'>xgboost AUC: <b>{_html_esc(_fmt_num((xgb.get('eval_metrics') or {}).get('auc_roc'), digits=4))}</b></span>")
        except Exception:
            pass
    if mb and isinstance(mb, dict):
        mb_test = (mb.get("test_eval") or {}).get("eval_metrics") or {}
        if mb_test.get("accuracy") is not None:
            kpis.append(f"<span class='kpi'>ModernBERT acc: <b>{_html_esc(_fmt_num(mb_test.get('accuracy'), digits=4))}</b></span>")
    if kpis:
        rb.p("".join(kpis))

    gen_at = manifest.get("generated_at") or eda_basic.get("generated_at") or _utc_now_iso()
    rb.note(f"Gerado em <b>{_html_esc(gen_at)}</b>. Output dir: <code>{_html_esc(cfg.output_dir)}</code>.")
    rb.hr()

    # eda
    rb.h2("2. Descrição dos dados e EDA")
    rb.note("Seções e gráficos abaixo são gerados automaticamente a partir dos artefatos em <code>metrics/</code> e <code>eda/</code>.")
    by_split = eda_basic.get("by_dataset_split") or []
    by_sent = eda_basic.get("by_sentiment") or []
    if by_split:
        rb.table(by_split, "Distribuição por dataset_split", max_rows=50, digits=0)
    if by_sent:
        rb.table(by_sent, "Distribuição por sentiment", max_rows=50, digits=0)
    plots_basic = eda_basic.get("plots") or eda_basic.get("files") or {}
    rb.fig(plots_basic.get("split_counts") or plots_basic.get("by_split_bar"), "Rows por dataset_split")
    rb.fig(plots_basic.get("sentiment_counts") or plots_basic.get("by_sentiment_bar"), "Balanceamento de classes (sentiment)")
    rb.fig(plots_basic.get("missing_rates") or plots_basic.get("missing_bar"), "Missing/empty rates")

    q = eda_adv.get("quantiles_text_len_tokens") or eda_adv.get("text_len_quantiles") or {}
    if q:
        rb.table([{"stat": k, "value": v} for k, v in q.items()], "Quantis de tamanho do texto (tokens)", max_rows=50, digits=0)
    files_adv = eda_adv.get("plots") or eda_adv.get("files") or {}
    rb.fig(files_adv.get("text_len_hist") or files_adv.get("text_len_histogram"), "Histograma de tamanho de texto (tokens)")
    rb.fig(files_adv.get("top_tokens_all"), "Top tokens (geral)")
    rb.fig(files_adv.get("top_tokens_pos"), "Top tokens (sentiment=1)")
    rb.fig(files_adv.get("top_tokens_neg"), "Top tokens (sentiment=0)")
    rb.fig(files_adv.get("wordcloud_all"), "Wordcloud (geral)")
    rb.fig(files_adv.get("wordcloud_pos"), "Wordcloud (sentiment=1)")
    rb.fig(files_adv.get("wordcloud_neg"), "Wordcloud (sentiment=0)")

    rb.hr()

# config e decisoes
    rb.h2("3. Configuração e decisões (traceabilidade)")
    if eda_decisions:
        rb.table([{"chave": k, "valor": v} for k, v in eda_decisions.items()], "Decisões aplicadas (EDA → config)", max_rows=200, digits=6)
    cfg_src = cfg_final or (manifest.get("config") if isinstance(manifest, dict) else {}) or {}
    if cfg_src:
        keys_focus = ["run_profile","max_polarity_rows","hashing_num_features","train_split_fraction","cv_folds","cv_parallelism","enable_rf","enable_xgboost","enable_clustering","pca_var_target","kmeans_k_max","enable_modernbert","modernbert_model_name","modernbert_max_seq_len","modernbert_batch_size"]
        compact = [{"param": k, "value": cfg_src.get(k)} for k in keys_focus if k in cfg_src]
        if compact:
            rb.table(compact, "Parâmetros principais", max_rows=200, digits=6)
        rb.p("<details><summary>Config (raw)</summary><pre>" + _html_esc(json.dumps(cfg_src, indent=2)) + "</pre></details>")
    else:
        rb.warn("Não foi possível carregar snapshot de configuração (config_final.json / run_manifest.json ausentes).")

    rb.hr()

# supervisionado e xgboost
    rb.h2("4. Modelos supervisionados (TF‑IDF + ML) e XGBoost")
    if sup:
        rows = []
        for name, m in sup.items():
            if name.startswith("_") or not isinstance(m, dict):
                continue
            rows.append({
                "model": name,
                "auc_roc": (m.get("auc_roc") if m.get("auc_roc") is not None else (m.get("eval_metrics") or {}).get("auc_roc")),
                "auc_pr": (m.get("auc_pr") if m.get("auc_pr") is not None else (m.get("eval_metrics") or {}).get("auc_pr")),
                "accuracy": (m.get("accuracy") if m.get("accuracy") is not None else (m.get("eval_metrics") or {}).get("accuracy")),
                "f1": (m.get("f1") if m.get("f1") is not None else (m.get("eval_metrics") or {}).get("f1")),
                "precision_pos": (m.get("precision_pos") if m.get("precision_pos") is not None else m.get("precision")),
                "recall_pos": (m.get("recall_pos") if m.get("recall_pos") is not None else m.get("recall")),
                "brier": m.get("brier"),
                "seconds": m.get("seconds"),
                "eval_json": m.get("eval_json"),
            })
        rows = sorted(rows, key=lambda r: (r.get("auc_roc") is None, -(r.get("auc_roc") or -1.0)))
        rb.table(rows, "Resumo (dev) – métricas principais", max_rows=50, digits=6)
        sup_files = sup.get("_files") or {}
        rb.fig(sup_files.get("comparison"), "Comparação (AUC/Acuracia/F1) – modelos supervisionados")
        # matrizes de confusao pra compatibilidade com os modelos supervisionados
        for name, m in sup.items():
            if name.startswith("_") or not isinstance(m, dict):
                continue
            
            pm = (m.get("plots") or m.get("files") or {})
            cm = pm.get("cm") or pm.get("confusion") or pm.get("confusion_matrix")
            rb.fig(cm, f"Matriz de confusão – {name}")

    else:
        rb.warn("Arquivo <code>supervised_models.json</code> não encontrado.")

    if xgb:
        rb.h3("4.1 XGBoost (baseline)")
        rb.table([{"campo": k, "valor": v} for k, v in xgb.items() if k not in ("eval_metrics",)], "Resumo XGBoost", max_rows=80, digits=6)
        if xgb.get("eval_metrics"):
            rb.table([{"métrica": k, "valor": v} for k, v in (xgb.get("eval_metrics") or {}).items()], "Métricas XGBoost (teste)", max_rows=50, digits=6)
    else:
        rb.note("XGBoost não foi executado (dependência ausente ou desabilitado).")

    if sup_best:
        rb.note(f"Winner (dev): <b>{_html_esc(sup_best.get('winner'))}</b> por <code>{_html_esc(sup_best.get('winner_metric'))}</code> = <b>{_html_esc(_fmt_num(sup_best.get('winner_value'), digits=6))}</b>.")

    if sup_full:
        rb.h3("4.2 Treino full (train_full)")
        rb.table([{"campo": k, "valor": v} for k, v in sup_full.items()], "Resumo train_full", max_rows=120, digits=6)

    rb.hr()

    # clustering
    rb.h2("5. Análise não supervisionada – Clustering (PCA + k‑means)")
    if clu:
        silv = clu.get("silhouette_final") if clu.get("silhouette_final") is not None else clu.get("silhouette")
        rb.note(f"Silhouette ≈ <b>{_html_esc(_fmt_num(silv, digits=6))}</b> | k escolhido = <b>{_html_esc(clu.get('kmeans_k_chosen'))}</b> | PCA usado = <b>{_html_esc(clu.get('pca_used'))}</b>.")
        prof = clu.get("cluster_profiles") or []
        if prof:
            rb.table(prof, "Perfil por cluster (n, mean_sentiment)", max_rows=200, digits=6)
        cplots = clu.get("plots") or clu.get("files") or {}
        rb.fig(cplots.get("pca_cumvar"), "PCA – variância acumulada")
        rb.fig(cplots.get("silhouette_by_k"), "Silhouette vs k")
        rb.fig(cplots.get("cluster_sizes"), "Tamanho dos clusters")
        rb.fig(cplots.get("cluster_mean_sentiment"), "Sentimento médio por cluster")
    else:
        rb.note("Nenhuma métrica de clustering encontrada (provável execução fora de dev ou clustering desabilitado).")

    rb.hr()

# modernbert
    rb.h2("6. ModernBERT (opcional)")
    if mb:
        rb.table([{"campo": k, "valor": v} for k, v in mb.items() if k not in ("train_curve", "files")], "Resumo ModernBERT", max_rows=120, digits=6)
        rb.fig((mb.get("files") or {}).get("loss_curve"), "Training curve – loss")
        rb.fig((mb.get("files") or {}).get("val_metrics_curve"), "Validation curve – metrics")
    else:
        rb.note("ModernBERT não foi executado (desabilitado ou dependências ausentes).")

    rb.hr()

# artefatos e uso
    rb.h2("7. Artefatos gerados e como usar")
    paths = (manifest.get("paths") or {}) if isinstance(manifest, dict) else {}
    rb.table([
        {"artefato": "pipeline supervisionado (Spark)", "path": paths.get("models_supervised") or str(out_dir / "models" / "supervised_pipeline")},
        {"artefato": "pipeline clustering (Spark)", "path": paths.get("models_cluster") or str(out_dir / "models" / "cluster_pipeline")},
        {"artefato": "modelo XGBoost", "path": paths.get("xgboost_model") or str(out_dir / "models" / "xgboost")},
        {"artefato": "ModernBERT (torch)", "path": paths.get("torch_modernbert") or str(out_dir / "torch_models" / "modernbert_sentiment")},
        {"artefato": "metrics dir", "path": paths.get("metrics_dir") or str(out_dir / "metrics")},
        {"artefato": "eda dir", "path": paths.get("eda_dir") or str(out_dir / "eda")},
        {"artefato": "predictions dir", "path": paths.get("predictions_dir") or str(out_dir / "predictions")},
    ], "Paths principais", max_rows=50, digits=0)

    rb.hr()

# leaderboards
    rb.h2("8. Comparação global entre modelos (Leaderboards)")
    lfiles = (leader.get("files") or {}) if isinstance(leader, dict) else {}
    if any(lfiles.values()):
        rb.fig(lfiles.get("auc_roc"), "Leaderboard – AUC ROC")
        rb.fig(lfiles.get("auc_pr"),  "Leaderboard – AUC PR")
        rb.fig(lfiles.get("f1"),      "Leaderboard – F1")
        rb.fig(lfiles.get("accuracy"),"Leaderboard – Acuracia")
    else:
        rb.note("Nenhum leaderboard disponível (provável ausência de arquivos eval_*.json).")

    rb.hr()

# avaliacao detalhada por modelo
    rb.h2("9. Avaliação detalhada por modelo (eval_*.json)")
    evals = _load_eval_bundles(metrics_dir)
    if evals:
        summary = []
        for e in evals:
            m = e.get("metrics") or {}
            thr = e.get("threshold_tuning") or {}
            c = e.get("confusion") or {}
            tn, fp, fn, tp = (c.get("tn", 0), c.get("fp", 0), c.get("fn", 0), c.get("tp", 0))
            n = tn + fp + fn + tp
            baseline = (max(tp + fn, tn + fp) / n) if n else None
            summary.append({
                "model": e.get("model"),
                "split": e.get("split"),
                "auc_roc": m.get("auc_roc"),
                "auc_pr": m.get("auc_pr"),
                "accuracy": m.get("accuracy"),
                "baseline_majority_acc": baseline,
                "f1": m.get("f1"),
                "precision_pos": m.get("precision_pos"),
                "recall_pos": m.get("recall_pos"),
                "brier": m.get("brier"),
                "best_thr": thr.get("best_thr"),
                "best_f1": thr.get("best_f1"),
                "eval_json": e.get("_eval_file"),
            })
        rb.table(summary, "Resumo (fonte: metrics/eval_*.json)", max_rows=300, digits=6)

        for i, e in enumerate(sorted(evals, key=lambda x: (str(x.get("split")), str(x.get("model")))), start=1):
            model = str(e.get("model", "unknown"))
            split = str(e.get("split", "unknown"))
            rb.h3(f"9.{i} {model} ({split})")
            rb.note(f"Fonte: <code>metrics/{_html_esc(e.get('_eval_file'))}</code>")
            m = e.get("metrics") or {}
            rb.table([{"métrica": k, "valor": v} for k, v in m.items()], "Métricas", max_rows=80, digits=6)
            plots = e.get("plots") or {}
            rb.fig(plots.get("cm"), "Matriz de confusao")
            rb.fig(plots.get("roc"), "ROC")
            rb.fig(plots.get("pr"), "Precisao e recall")
            rb.fig(plots.get("calibration"), "Curva de calibração")
            rb.fig(plots.get("f1_threshold"), "F1 por limiar")
    else:
        rb.warn("Nenhum arquivo <code>eval_*.json</code> encontrado. Gere-os para todos os modelos (dev + train_full + XGBoost + ModernBERT).")

    rb.hr()

    # glossario de metricas
    rb.h2("10. Métricas Geradas")
    rb.h3("10.1 Classificação (Spark + XGBoost + ModernBERT)")
    rb.lst([
        "<b>AUC ROC</b>: separação global de classes.",
        "<b>AUC PR</b>: robusto quando classes estão desbalanceadas.",
        "<b>Acuracia</b>: porcentagem de acertos total (comparar com baseline de maioria).",
        "<b>F1 / Precision / Recall</b>: detalham equilíbrio entre erros positivos/negativos.",
        "<b>Matriz de confusao</b>: explicita tipos de erro (FP alto = otimismo excessivo; FN alto = conservador demais).",
        "<b>Brier Score</b>: verifica calibração das probabilidades (quanto menor, melhor).",
        "<b>Curva de calibração</b>: compara probabilidades previstas vs observadas em bins.",
        "<b>F1 por limiar</b>: sugere limiares alternativos a 0.5 conforme objetivo.",
        "<b>Plots + JSONs</b>: ROC, PR, calibração, matriz de confusão e manifestos são salvos para auditoria.",
    ])

    rb.h3("10.2 Clustering (PCA + k‑means)")
    rb.lst([
        "<b>Silhouette</b>: qualidade dos clusters (1 = bem definidos, 0 = confusos, &lt;0 = ruim).",
        "<b>Curva silhouette vs k</b>: auxilia a escolher k.",
        "<b>Variância explicada (PCA)</b>: indica quanta informação é preservada ao reduzir dimensão.",
        "<b>Perfil por cluster</b>: <code>n</code> e <code>mean_sentiment</code> ajudam a interpretar grupos.",
    ])

    out_path = out_dir / cfg.html_report_name
    return rb.finish(out_path)


def write_run_manifest(cfg: Config) -> None:
    """Escreve um manifest com config e caminhos pra facilitar reuso."""

    out_dir = Path(cfg.output_dir)
    xgb_meta = _read_json(out_dir / "metrics" / "xgboost.json") or {}
    xgb_model_rel = xgb_meta.get("model_path")
    if xgb_model_rel:
        xgb_model_abs = str((out_dir / xgb_model_rel).resolve())
    else:
        xgb_model_abs = str(out_dir / "models" / "xgboost" / "xgboost_model.json")

    manifest = {
        "generated_at": _utc_now_iso(),
        "config": asdict(cfg),
        "paths": {
            "models_supervised": str(out_dir / "models" / "supervised_pipeline"),
            "models_cluster": str(out_dir / "models" / "cluster_pipeline"),
            "xgboost_model_dir": str(out_dir / "models" / "xgboost"),
            "xgboost_model": xgb_model_abs,
            "torch_modernbert": str(out_dir / "torch_models" / "modernbert_sentiment"),
            "metrics_dir": str(out_dir / "metrics"),
            "eda_dir": str(out_dir / "eda"),
            "predictions_dir": str(out_dir / "predictions"),
            "report_html": str(out_dir / cfg.html_report_name),
        },
    }
    _write_json(out_dir / "metrics" / "run_manifest.json", manifest)


# orquestrador
class AmazonPolarityNotebookPipeline:
    def __init__(self, *, test_csv: str, train_csv: Optional[str], output_dir: str):
        """Ajuda interna de init pra deixar a execucao mais lisa."""
        self.test_csv = test_csv
        self.train_csv = train_csv
        self.output_dir = output_dir

    def _make_cfg(self, run_modernbert: bool) -> Config:
        """Ajuda interna de make cfg pra deixar a execucao mais lisa."""
        cfg = Config(
            polarity_test_path=self.test_csv,
            polarity_train_path=self.train_csv,
            output_dir=self.output_dir,
            enable_modernbert=bool(run_modernbert),
        )
        cfg.ensure_output_dirs()
        setup_logging(cfg.output_dir)
        tune_config(cfg)

# precisa acontecer antes de criar a sessao do spark
        _install_sitecustomize_threading_patch(cfg.output_dir)

        return cfg

    def run(self, *, force: bool = False, run_modernbert: bool = True) -> None:
        """Roda o pipeline inteiro e gera modelos metricas e relatorio."""
        if force and _env_str("AMAZON_NLP_RUN_PROFILE", "").lower() == "score_only":
            logger.warning("force=True with run_profile=score_only is inconsistent; switching to dev.")
            os.environ["AMAZON_NLP_RUN_PROFILE"] = "dev"

        # force rerun apaga a pasta de saida pra recomecar limpo
        if force and Path(self.output_dir).exists():
            shutil.rmtree(self.output_dir, ignore_errors=True)

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

        # spark parou aqui para o modernbert
        if cfg.run_profile == "dev" and cfg.enable_modernbert and torch_bundle is not None:
            device = _best_torch_device()
            if device == "cpu" and not _env_bool("AMAZON_NLP_ENABLE_MODERNBERT_CPU", False):
                logger.warning("ModernBERT skipped on CPU (set AMAZON_NLP_ENABLE_MODERNBERT_CPU=1 to force).")
            else:
                train_modernbert_sentiment_from_splits(torch_bundle, cfg)

        # leaderboards juntando tudo que saiu nos eval json
        refresh_leaderboards(cfg)

        write_run_manifest(cfg)
        report = generate_html_report(cfg)
        logger.info("Report: %s", str(report))

    def _run_dev(self, spark: SparkSession, cfg: Config) -> Optional[TorchSplitBundle]:
        """Ajuda interna de run dev pra deixar a execucao mais lisa."""
        logger.info("=== RUN: dev ===")
        df = load_polarity_dataset(spark, cfg)
        df = _maybe_fix_underpartitioning_dev(df, cfg).persist(StorageLevel.MEMORY_AND_DISK)

        try:
            parts = int(df.rdd.getNumPartitions())
        except Exception:
            parts = -1
        logger.info("dev: df partitions=%s", parts)
        total_rows = int(df.count())
        logger.info("dev: dataset rows=%d", total_rows)

        if cfg.enable_eda:
            logger.info("dev: EDA ...")
            run_basic_eda(df, cfg, total_rows)
            eda_adv = run_advanced_eda(df, cfg, total_rows)
            apply_data_driven_config(cfg, eda_adv)

        logger.info("dev: supervised CV ...")

        df_bucket = (
            _add_split_bucket(df, cfg)
            .select("dataset_split", "sentiment", "text_full", "split_bucket")
            .persist(StorageLevel.MEMORY_AND_DISK)
        )
        _ = df_bucket.count() 

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
            logger.info("dev: clustering pipeline ...")
            fit_and_save_cluster_pipeline(spark, df, cfg)

        df.unpersist()
        return torch_bundle

    def _run_train_full(self, spark: SparkSession, cfg: Config) -> None:
        """Ajuda interna de run treina completo pra deixar a execucao mais lisa."""
        logger.info("=== RUN: train_full ===")
        df = load_polarity_dataset(spark, cfg)
        train_full_supervised_pipeline(spark, df, cfg)

    def _run_score_only(self, spark: SparkSession, cfg: Config) -> None:
        """Ajuda interna de run score only pra deixar a execucao mais lisa."""
        logger.info("=== RUN: score_only ===")
        df = load_polarity_dataset(spark, cfg)

        pred = score_with_saved_supervised_model(spark, df, cfg)
        out_path = Path(cfg.output_dir) / "predictions" / "supervised_predictions.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pred.write.mode("overwrite").parquet(str(out_path))
        logger.info("score_only: wrote supervised predictions: %s", str(out_path))

        if _env_bool("AMAZON_NLP_SCORE_CLUSTERS", False):
            clu = score_with_saved_cluster_model(spark, df, cfg)
            clu_path = Path(cfg.output_dir) / "predictions" / "cluster_assignments.parquet"
            clu_path.parent.mkdir(parents=True, exist_ok=True)
            clu.write.mode("overwrite").parquet(str(clu_path))
            logger.info("score_only: wrote cluster assignments: %s", str(clu_path))


if __name__ == "__main__":
    pipeline = AmazonPolarityNotebookPipeline(test_csv=TEST_CSV, train_csv=TRAIN_CSV, output_dir=OUTPUT_DIR)
    os.environ["AMAZON_NLP_RUN_PROFILE"] = _env_str("AMAZON_NLP_RUN_PROFILE", default_profile)
    pipeline.run(force=FORCE_RERUN, run_modernbert=RUN_MODERNBERT)

