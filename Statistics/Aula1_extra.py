"""
Aula1.py  ─  statistics & ML evaluation toolbox
----------------------------------------------
Testing different ML algorithms
Hyper‑parameter tuning with RandomizedSearchCV
Full metric suite + ROC / PR / calibration / lift plots
Optional deep‑learning optimiser demo (AdamW vs Lamb)

Run  :  python Aula1.py
Reqs :  numpy, pandas, scikit‑learn, matplotlib, scipy
Opt. :  xgboost, lightgbm, catboost, torch, torchopt
"""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from typing import Callable, Dict, Mapping, Protocol, Sequence, Tuple
import random, math
from itertools import cycle
import itertools   

import numpy as np
import pandas as pd
from scipy import integrate, stats
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss, log_loss,
    ConfusionMatrixDisplay, RocCurveDisplay, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, RandomizedSearchCV
)
try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier 
except ModuleNotFoundError:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier 
except ModuleNotFoundError:
    CatBoostClassifier = None
try:
    from torchopt import AdamW, Lamb  
    import torch 
except ModuleNotFoundError:
    AdamW = Lamb = None
    torch = None

#1) Global config
@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """High‑level settings shared across the toolbox."""
    positive_label: str = "yes"
    threshold: float = 0.5
    n_boot: int = 1_000
    random_state: int = 42
    cv_folds: int = 5

#2) Prediction strategies
class PredictionStrategy(Protocol):
    """Maps raw model output columns into hard class labels."""
    def labels(self, df: pd.DataFrame) -> np.ndarray: ...

class ProvidedLabelStrategy:
    """Use the column `pred` already present in the DataFrame."""
    def __init__(self, pos: str) -> None:
        self._pos = pos

    def labels(self, df: pd.DataFrame) -> np.ndarray:
        return df["pred"].replace({"sim": self._pos}).to_numpy()

class ThresholdStrategy:
    """Turn a probability column into labels via fixed threshold."""
    def __init__(self, thr: float, pos: str) -> None:
        self._thr = thr
        self._pos = pos

    def labels(self, df: pd.DataFrame) -> np.ndarray:
        return np.where(df["prob"] >= self._thr, self._pos, "no")

#3) ModelStrategy 
class ModelStrategy(Protocol):
    """Minimal interface every wrapped estimator must expose."""
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ModelStrategy": ...
    def prob(self, X: pd.DataFrame) -> np.ndarray: ...

class _SkWrap:
    """Uniform adapter around any scikit‑learn binary classifier."""
    def __init__(self, est: BaseEstimator & ClassifierMixin):
        self._est = est

    def fit(self, X, y): 
        self._est.fit(X, y)
        return self

    def prob(self, X):  
        return self._est.predict_proba(X)[:, 1]

class LogReg(_SkWrap):
    def __init__(self) -> None:
        super().__init__(LogisticRegression(max_iter=2_000, solver="lbfgs"))

class GBoost(_SkWrap):
    def __init__(self) -> None:
        super().__init__(GradientBoostingClassifier())

class RF(_SkWrap):
    def __init__(self) -> None:
        super().__init__(RandomForestClassifier())

class XGB(_SkWrap):
    def __init__(self) -> None:
        if XGBClassifier is None:
            raise ModuleNotFoundError("Install xgboost to use XGBStrategy")
        super().__init__(XGBClassifier(eval_metric="logloss", use_label_encoder=False))

class LGBM(_SkWrap):
    def __init__(self) -> None:
        if LGBMClassifier is None:
            raise ModuleNotFoundError("Install lightgbm to use LGBMStrategy")
        super().__init__(LGBMClassifier())

class CatBoost(_SkWrap):
    def __init__(self) -> None:
        if CatBoostClassifier is None:
            raise ModuleNotFoundError("Install catboost to use CatBoostStrategy")
        super().__init__(CatBoostClassifier(verbose=0))

#Registry
_MODEL_REG: Dict[str, Callable[[], ModelStrategy]] = {
    "logreg": LogReg, "gboost": GBoost, "rf": RF,
    "xgb": XGB, "lgbm": LGBM, "cat": CatBoost,
}

#4) Statistical summaries & distribution helpers
class StatSummary:
    """Compute mean/variance/stdev for a 1‑D array‑like."""
    def __init__(self, s: pd.Series | np.ndarray):
        self._arr = np.asarray(s, dtype=float)

    def mean(self): return float(np.mean(self._arr))
    def var(self):  return float(np.var(self._arr, ddof=1))
    def std(self):  return float(np.std(self._arr, ddof=1))
    def n(self):    return int(self._arr.size)
    def df(self):   return self.n() - 1

    def describe(self) -> Mapping[str, float]:
        mu, v, sd = self.mean(), self.var(), self.std()
        return {"n": self.n(), "df": self.df(), "mean": mu, "var": v, "sd": sd}

#5) Distributions: Normal, Binomial, Poisson
class DistributionToolkit:
    """PDF/CDF/PMF wrappers + calculus utilities."""
    @staticmethod
    def normal_pdf(x: float, mu=0.0, sigma=1.0) -> float:
        return stats.norm(mu, sigma).pdf(x)

    @staticmethod
    def normal_cdf(x: float, mu=0.0, sigma=1.0) -> float:
        return stats.norm(mu, sigma).cdf(x)

    @staticmethod
    def binom_pmf(k: int, n: int, p: float) -> float:
        return stats.binom(n, p).pmf(k)

    @staticmethod
    def poisson_pmf(k: int, lam: float) -> float:
        return stats.poisson(lam).pmf(k)

    @staticmethod
    def integrate_pdf(func: Callable[[float], float], a: float, b: float) -> float:
        val, _ = integrate.quad(func, a, b)
        return val

    @staticmethod
    def derivative(func: Callable[[float], float], x: float, eps=1e-6) -> float:
        return (func(x + eps) - func(x - eps)) / (2 * eps)

#5) Linear‑algebra helpers
class MatrixOps:
    """Tiny matrix algebra helpers."""
    @staticmethod
    def mul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return A @ B

    @staticmethod
    def grad_logistic(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        z = X @ w
        pred = 1 / (1 + np.exp(-z))
        return X.T @ (pred - y) / len(y)

#6) Hyper‑parameter tuner
class HyperTuner:
    """RandomisedSearchCV wrapper with optional stratified splitting."""
    def __init__(
        self,
        param_distributions: Mapping[str, Sequence],
        n_iter: int = 25,
        random_state: int = 42,
        cv: int | StratifiedKFold | None = None,
    ) -> None:
        self._params = param_distributions
        self._n_iter = n_iter
        self._rs = random_state
        self._cv = (
            StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
            if cv is None else cv
        )

    def tune(self, strat: ModelStrategy, X: pd.DataFrame, y: pd.Series) -> ModelStrategy:
        base = strat._est                              
        rcv = RandomizedSearchCV(
            base, self._params, n_iter=self._n_iter,
            cv=self._cv, scoring="f1", random_state=self._rs, n_jobs=-1
        )
        rcv.fit(X, y)
        strat._est = rcv.best_estimator_             
        return strat

#7) Evaluator
class Evaluator:
    """Calculate all scalar metrics & common diagnostic plots."""

    def __init__(self, df: pd.DataFrame, cfg: EvaluationConfig,
                 label_strat: PredictionStrategy):
        self._cfg = cfg
        self._df = df.copy()
        self._df["obs"] = self._df["obs"].replace({"sim": cfg.positive_label})
        self._df["pred"] = label_strat.labels(self._df)

    #Core 
    def _arr(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        yt = (self._df["obs"] == self._cfg.positive_label).astype(int).to_numpy()
        yp = (self._df["pred"] == self._cfg.positive_label).astype(int).to_numpy()
        pr = self._df["prob"].to_numpy()
        return yt, yp, pr

    #Scalar metrics 
    def metrics(self) -> Mapping[str, float]:
        yt, yp, pr = self._arr()
        tp, fn, fp, tn = confusion_matrix(yt, yp, labels=[1, 0]).ravel()
        safe_ll = log_loss if "eps" not in signature(log_loss).parameters \
            else lambda a, b: log_loss(a, b, eps=1e-15)
        return dict(
            TP=tp, FP=fp, TN=tn, FN=fn,
            accuracy=accuracy_score(yt, yp),
            precision=precision_score(yt, yp, zero_division=0),
            recall=recall_score(yt, yp, zero_division=0),
            f1=f1_score(yt, yp, zero_division=0),
            mcc=matthews_corrcoef(yt, yp),
            roc_auc=roc_auc_score(yt, pr),
            pr_auc=average_precision_score(yt, pr),
            brier=brier_score_loss(yt, pr),
            log_loss=safe_ll(yt, pr)
        )

    #Bootstrap
    def bootstrap_ci(self, stat=f1_score) -> Tuple[float, Tuple[float, float]]:
        yt, yp, _ = self._arr()
        rng = random.Random(self._cfg.random_state)
        idx = np.arange(len(yt))
        vals = [stat(yt[s := rng.choices(idx, k=len(idx))], yp[s], zero_division=0)
                for _ in range(self._cfg.n_boot)]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return float(np.mean(vals)), (float(lo), float(hi))

    #Best threshold search 
    def best_thr(self, metric=f1_score) -> float:
        yt, _, pr = self._arr()
        grid = np.linspace(0.01, 0.99, 500)
        scores = [metric(yt, (pr >= t).astype(int), zero_division=0) for t in grid]
        return float(grid[int(np.argmax(scores))])

    #Plotting helpers 
    def plot_confusion(self):
        yt, yp, _ = self._arr()
        ConfusionMatrixDisplay.from_predictions(yt, yp)
        plt.title("Confusion matrix"); plt.tight_layout(); plt.show()

    def plot_roc(self):
        yt, _, pr = self._arr()
        RocCurveDisplay.from_predictions(yt, pr)
        plt.tight_layout(); plt.show()

    def plot_calibration(self, bins: int = 5):
        yt, _, pr = self._arr()
        frac_pos, mean_pred = calibration_curve(yt, pr, n_bins=bins)
        plt.plot(mean_pred, frac_pos, "o-")
        plt.plot([0, 1], [0, 1], "--")
        plt.title("Calibration curve"); plt.tight_layout(); plt.show()

    def plot_lift(self):
        yt, _, pr = self._arr()
        order = np.argsort(-pr)
        lift = np.cumsum(yt[order]) / (np.arange(1, len(yt) + 1) * yt.mean())
        plt.plot(lift); plt.title("Cumulative lift"); plt.tight_layout(); plt.show()

    #Helpers to build and fit the model
    @staticmethod
    def build(model_name: str) -> ModelStrategy:
        try:
            return _MODEL_REG[model_name]()
        except KeyError as exc:
            raise ValueError(f"{model_name} not in {_MODEL_REG.keys()}") from exc

    def fit(self, X: pd.DataFrame, y: pd.Series,
            model_name: str = "logreg", tuner: HyperTuner | None = None) -> ModelStrategy:
        model = self.build(model_name)
        if tuner:
            model = tuner.tune(model, X, y)
        model.fit(X, y)
        self._df["prob"] = model.prob(X)
        self._df["pred"] = ThresholdStrategy(self._cfg.threshold,
                                             self._cfg.positive_label).labels(self._df)
        return model

    def cv_scores(self, X: pd.DataFrame, y: pd.Series,
                  model_name: str = "logreg") -> Tuple[float, float]:
        model = self.build(model_name)
        y_prob = cross_val_predict(model._est, X, y,          # type: ignore[attr-defined]
                                   cv=self._cfg.cv_folds,
                                   method="predict_proba")[:, 1]
        y_pred = (y_prob >= self._cfg.threshold).astype(int)
        return accuracy_score(y, y_pred), f1_score(y, y_pred, zero_division=0)


#8) ModelComparator 
class ModelComparator:
    """Fit several ModelStrategy objects, collect OOF metrics & plot dashboard."""
    def __init__(
        self,
        cfg: EvaluationConfig,
        models: Sequence[str],
        tuner_map: Mapping[str, HyperTuner] | None = None,
        palette: Sequence[str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.models = list(models)
        self.tuner_map = tuner_map or {}
        self._cv = StratifiedKFold(
            n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state
        )
        self.palette = palette or plt.rcParams["axes.prop_cycle"].by_key()["color"]

    #Internal helpes for the dashboard
    def _oof_probs(self, strat: ModelStrategy,
                   X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        return cross_val_predict(
            strat._est, X, y,                        # type: ignore[attr-defined]
            cv=self._cv, method="predict_proba"
        )[:, 1]

    #Public entry point 
    def compare(self, X: pd.DataFrame, y: pd.Series,
                param_grids: Mapping[str, Mapping[str, Sequence]]) -> pd.DataFrame:

        records, prob_map = [], {}
        for i, name in enumerate(self.models):
            try:
                strat = Evaluator.build(name)
            except ModuleNotFoundError as exc:                 
                print(f"⋄ {name} skipped – {exc}")
                continue

            if tuner := self.tuner_map.get(name):
                tuner._params = param_grids[name]   # inject grid
                strat = tuner.tune(strat, X, y)

            proba = self._oof_probs(strat, X, y)
            pred = (proba >= self.cfg.threshold).astype(int)
            prob_map[name] = proba

            records.append({
                "model": name,
                "accuracy": accuracy_score(y, pred),
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
                "f1": f1_score(y, pred, zero_division=0),
                "mcc": matthews_corrcoef(y, pred),
                "roc_auc": roc_auc_score(y, proba),
                "pr_auc": average_precision_score(y, proba),
                "brier": brier_score_loss(y, proba),
                "log_loss": log_loss(y, np.c_[1 - proba, proba]),
            })

        #Dashboard for ROC / PR / CAL / LIFT
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        ax_roc, ax_pr, ax_cal, ax_lift = axs.ravel()

        for i, (name, proba) in enumerate(prob_map.items()):
            col = self.palette[i % len(self.palette)]
            RocCurveDisplay.from_predictions(y, proba, ax=ax_roc, name=name, color=col)

            prec, rec, _ = precision_recall_curve(y, proba)
            ax_pr.plot(rec, prec, label=name, color=col)

            frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10)
            ax_cal.plot(mean_pred, frac_pos, "o-", label=name, color=col)
            ax_cal.plot([0, 1], [0, 1], "--k")

            order = np.argsort(-proba)
            lift = np.cumsum(y.iloc[order]) / (np.arange(1, len(y) + 1) * y.mean())
            ax_lift.plot(lift, label=name, color=col)

        ax_roc.set_title("ROC"); ax_pr.set_title("Precision‑Recall")
        ax_cal.set_title("Calibration"); ax_lift.set_title("Cumulative lift")
        for ax in (ax_pr, ax_cal, ax_lift):
            ax.legend()
        plt.tight_layout(); plt.show()

        return (pd.DataFrame(records)
                .set_index("model")
                .sort_values("roc_auc", ascending=False))
    
#9) Optimiser demo 
def optimiser_demo(seed: int = 0, steps: int = 100) -> None:
    """Plot loss curves for AdamW vs Lamb on a toy logistic‑regression task."""
    if AdamW is None or Lamb is None or torch is None:        # pragma: no cover
        print("⋄ optimiser_demo skipped – torch/torchopt not installed")
        return

    from sklearn.datasets import load_breast_cancer
    ds = load_breast_cancer()
    X = torch.tensor(ds["data"], dtype=torch.float32)
    y = torch.tensor(ds["target"], dtype=torch.float32).view(-1, 1)

    def loss_fn(w: torch.Tensor) -> torch.Tensor:
        logits = X @ w
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, y)

    torch.manual_seed(seed)
    w_adam = torch.zeros((X.shape[1], 1), requires_grad=True)
    w_lamb = torch.zeros_like(w_adam, requires_grad=True)
    opt_adam = AdamW(lr=1e-2); opt_lamb = Lamb(lr=1e-2)

    losses_a, losses_l = [], []
    for _ in range(steps):
        opt_adam.zero_grad();  loss_a = loss_fn(w_adam); loss_a.backward();  opt_adam.step(w_adam)
        opt_lamb.zero_grad();  loss_l = loss_fn(w_lamb); loss_l.backward();  opt_lamb.step(w_lamb)
        losses_a.append(loss_a.item()); losses_l.append(loss_l.item())

    plt.figure(figsize=(6, 4))
    plt.plot(losses_a, label="AdamW"); plt.plot(losses_l, label="Lamb")
    plt.yscale("log"); plt.xlabel("epoch"); plt.ylabel("BCE loss (log)")
    plt.title("Optimiser convergence"); plt.legend(); plt.tight_layout(); plt.show()

#10) Demo
if __name__ == "__main__":
    print("═" * 60, "\nPart A  –  scalar metrics & basic plots")
    df_demo = pd.DataFrame({
        "prob": np.linspace(0.05, 0.95, 20),
        "pred": ["no"] * 10 + ["yes"] * 10,
        "obs":  ["no", "no", "sim", "no", "no", "sim", "no", "no", "yes", "yes"] * 2,
    })
    cfg = EvaluationConfig()
    evaluator = Evaluator(df_demo, cfg, ProvidedLabelStrategy(cfg.positive_label))
    print("Metrics:", evaluator.metrics())
    print("F1 bootstrap:", evaluator.bootstrap_ci())
    print("Best‑F1 threshold:", evaluator.best_thr())
    evaluator.plot_confusion(); evaluator.plot_roc()
    evaluator.plot_calibration(); evaluator.plot_lift()

    print("\nExtras:")
    print("StatSummary(prob):", StatSummary(df_demo["prob"]).describe())
    area = DistributionToolkit.integrate_pdf(
        lambda x: DistributionToolkit.normal_pdf(x, 0, 1), -1, 1
    )
    print("∫ N(0,1) from –1 to 1 ≈", round(area, 3))
    d_dx = DistributionToolkit.derivative(
        lambda z: DistributionToolkit.normal_cdf(z, 0, 1), 0.0
    )
    print("Derivative of CDF at 0 ≈ PDF(0) =", round(d_dx, 3))
    A = np.eye(2); B = np.ones((2, 1))
    print("MatrixOps.mul(A, B):\n", MatrixOps.mul(A, B))
    X_demo = np.array([[1, 2], [3, 4]], dtype=float)
    y_demo = np.array([0, 1], dtype=float)
    w_demo = np.zeros(2)
    print("grad_logistic demo:", MatrixOps.grad_logistic(X_demo, y_demo, w_demo))

    print("\n", "═" * 60, "\nPart B  –  multi‑model comparison dashboard")
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, pd.Series(data.target, name="target")
    X = (X - X.mean()) / X.std()   # crude scaling for tree parity

    #Tiny demo for search space
    search_spaces = {
         "logreg": {"C": np.logspace(-3, 1, 10)},
         "rf":     {"n_estimators": [100, 200], "max_depth": [5, 10]},
         "gboost": {"learning_rate": np.logspace(-3, -1, 5)},
         "xgb":    {"max_depth": [3, 5], "eta": [0.05, 0.1]},
         "lgbm":   {"num_leaves": [15, 31], "learning_rate": [0.05, 0.1]},
         "cat":    {"depth": [4, 6], "learning_rate": [0.05, 0.1]},
     }

    available_models = []
    for m in list(search_spaces):         
        try:
            _MODEL_REG[m]()                
            available_models.append(m)
        except ModuleNotFoundError as exc:
            print(f"⋄ {m} demo skipped – {exc}")
            del search_spaces[m]        

    tuners = {name: HyperTuner({}, n_iter=5, random_state=cfg.random_state)
              for name in available_models}

    comparator = ModelComparator(cfg,
                                 models=tuple(available_models),
                                 tuner_map=tuners)

    score_tbl = comparator.compare(X, y, param_grids=search_spaces)
    print(score_tbl.round(3))

    #Logistic Regression baseline
    base_acc, base_f1 = evaluator.cv_scores(X, y, model_name="logreg")
    print("\nLogReg 5‑fold CV – accuracy:", round(base_acc, 3), "| F1:", round(base_f1, 3))

    print("\n", "═" * 60, "\nPart C  –  optimiser demo (AdamW vs Lamb)")
    optimiser_demo()
    print("\nExtra sanity:", math.sqrt(2), "– cycle demo:", next(cycle('✓')))
    print("itertools.combinations(‘ABCD’,2) sample →",
          list(itertools.islice(itertools.combinations("ABCD", 2), 3)))
