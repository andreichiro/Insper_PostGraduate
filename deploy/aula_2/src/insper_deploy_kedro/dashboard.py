"""Dashboard Streamlit — comparação de modelos e inferência ao vivo."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from insper_deploy_kedro.pipelines.data_engineering.nodes import (
    add_features,
    clean_data,
    transform_encoders,
    transform_scalers,
)
from insper_deploy_kedro.pipelines.inference.nodes import predict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "06_models"
OUTPUT_DIR = DATA_DIR / "07_model_output"

FEATURE_DESCRIPTIONS: dict[str, tuple[str, float, float, float]] = {
    "Pregnancies": ("Gestações", 0.0, 20.0, 6.0),
    "Glucose": ("Glicose (mg/dL)", 0.0, 250.0, 148.0),
    "BloodPressure": ("Pressão arterial (mm Hg)", 0.0, 140.0, 72.0),
    "SkinThickness": ("Espessura da pele (mm)", 0.0, 100.0, 35.0),
    "Insulin": ("Insulina (mu U/ml)", 0.0, 900.0, 0.0),
    "BMI": ("IMC (kg/m²)", 0.0, 70.0, 33.6),
    "DiabetesPedigreeFunction": ("Função pedigree", 0.0, 2.5, 0.627),
    "Age": ("Idade", 18.0, 100.0, 50.0),
}

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "baseline": "Logistic Regression",
    "optimized": "CatBoost (Optuna)",
    "xgboost": "XGBoost (Optuna)",
}


def _find_artifact(path: Path) -> Path | None:
    """Resolve artefato Kedro — suporta datasets versionados (subdiretório c/ timestamp)."""
    if path.is_file():
        return path
    if path.is_dir():
        versions = sorted(path.iterdir(), reverse=True)
        for version_dir in versions:
            candidate = version_dir / path.name
            if candidate.is_file():
                return candidate
    return None


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301


@st.cache_data
def load_test_report() -> dict[str, Any] | None:
    """Carrega relatório de teste (cacheado). Tenta test_report, senão junta as métricas individuais."""
    report_path = _find_artifact(OUTPUT_DIR / "test_report.pkl")
    if report_path:
        return _load_pickle(report_path)

    individual: dict[str, Any] = {}
    for key in ("baseline", "optimized", "xgboost"):
        p = _find_artifact(OUTPUT_DIR / f"{key}_metrics.pkl")
        if p:
            individual[key] = _load_pickle(p)
    return individual or None


@st.cache_resource
def load_production_artifacts() -> dict[str, Any] | None:
    """Carrega encoders, scalers e modelo de produção (cacheado)."""
    paths = {
        "encoders": MODELS_DIR / "production_encoders.pkl",
        "scalers": MODELS_DIR / "production_scalers.pkl",
        "model": MODELS_DIR / "production_model.pkl",
    }
    artifacts: dict[str, Any] = {}
    for key, path in paths.items():
        resolved = _find_artifact(path)
        if resolved is None:
            return None
        artifacts[key] = _load_pickle(resolved)

    artifacts["inference_raw_columns"] = {
        "categorical": [],
        "numerical": list(FEATURE_DESCRIPTIONS.keys()),
    }
    return artifacts


# ── Métricas ────────────────────────────────────────────────────────────


def render_metrics_tab(report: dict[str, Any]) -> None:
    """Aba de comparação de métricas entre modelos."""
    st.header("Comparação de Modelos")

    metrics_data: dict[str, dict[str, float]] = {}
    for model_key, metrics in report.items():
        display_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        metrics_data[display_name] = {
            "ROC AUC": metrics.get("roc_auc", 0),
            "F1": metrics.get("f1", 0),
            "Accuracy": metrics.get("accuracy", 0),
            "Precision": metrics.get("precision", 0),
            "Recall": metrics.get("recall", 0),
            "R²": metrics.get("r2", 0),
            "MAPE (%)": metrics.get("mape", 0),
            "Amostras": metrics.get("n_samples", 0),
        }

    df = pd.DataFrame(metrics_data).T
    best_model = str(df["ROC AUC"].idxmax())

    col1, col2, col3 = st.columns(3)
    col1.metric("Melhor Modelo", best_model)
    col2.metric("ROC AUC", f"{df.loc[best_model, 'ROC AUC']:.4f}")
    col3.metric("F1 Score", f"{df.loc[best_model, 'F1']:.4f}")

    st.divider()
    st.subheader("Tabela de Métricas (split de teste)")
    styled = (
        df.style.format(
            {
                "ROC AUC": "{:.4f}",
                "F1": "{:.4f}",
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "R²": "{:.4f}",
                "MAPE (%)": "{:.2f}%",
                "Amostras": "{:.0f}",
            }
        )
        .highlight_max(
            subset=["ROC AUC", "F1", "Accuracy", "Precision", "Recall", "R²"],
            color="#2ecc71",
        )
        .highlight_min(subset=["MAPE (%)"], color="#2ecc71")
    )
    st.dataframe(styled, use_container_width=True)

    st.divider()
    st.subheader("Comparação Visual")
    chart_metrics = ["ROC AUC", "F1", "Precision", "Recall"]
    chart_df = df[chart_metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(chart_metrics))
    width = 0.25
    for i, (model_name, row) in enumerate(chart_df.iterrows()):
        ax.bar(x + i * width, row.values, width, label=str(model_name))

    ax.set_xticks(x + width)
    ax.set_xticklabels(chart_metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── Matrizes de Confusão ────────────────────────────────────────────────


def render_confusion_matrix_tab(report: dict[str, Any]) -> None:
    """Aba de matrizes de confusão."""
    st.header("Matrizes de Confusão (Teste)")

    cols = st.columns(len(report))
    for col, (model_key, metrics) in zip(cols, report.items()):
        display_name = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        cm_raw = metrics.get("confusion_matrix", [])
        cm = np.array(cm_raw)
        if cm.size == 0:
            col.warning(f"{display_name}: sem dados")
            continue

        with col:
            st.subheader(display_name)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            im = ax.imshow(cm, cmap="Blues")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    color = "white" if cm[i, j] > cm.max() / 2 else "black"
                    ax.text(
                        j, i, str(cm[i, j]),
                        ha="center", va="center", color=color, fontsize=14,
                    )

            ax.set_xlabel("Predito")
            ax.set_ylabel("Real")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Não", "Sim"])
            ax.set_yticklabels(["Não", "Sim"])
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            tn, fp, fn, tp = cm.ravel()
            st.caption(f"VP={tp}  VN={tn}  FP={fp}  FN={fn}")


# ── Inferência ──────────────────────────────────────────────────────────


def render_inference_tab(artifacts: dict[str, Any] | None) -> None:
    """Aba de inferência ao vivo."""
    st.header("Predição ao Vivo")

    if artifacts is None:
        st.error(
            "Artefatos de produção não encontrados. "
            "Rode `uv run kedro run` primeiro pra treinar o modelo."
        )
        return

    st.markdown("Preencha os dados do paciente e clique em **Prever**.")

    col1, col2 = st.columns(2)
    inputs: dict[str, float] = {}
    half = len(FEATURE_DESCRIPTIONS) // 2
    for i, (feature, (label, min_val, max_val, default)) in enumerate(
        FEATURE_DESCRIPTIONS.items()
    ):
        target_col = col1 if i < half else col2
        inputs[feature] = target_col.number_input(
            label,
            min_value=min_val,
            max_value=max_val,
            value=default,
            step=1.0 if feature in ("Pregnancies", "Age") else 0.1,
            key=feature,
        )

    if st.button("Prever", type="primary", use_container_width=True):
        raw_df = pd.DataFrame([inputs])
        cleaned = clean_data(raw_df, artifacts["inference_raw_columns"])
        featured = add_features(cleaned)
        encoded = transform_encoders(featured, artifacts["encoders"])
        scaled = transform_scalers(encoded, artifacts["scalers"])
        result = predict(scaled, artifacts["model"])

        prediction = result["prediction"].iloc[0]
        proba = (
            result["prediction_proba"].iloc[0]
            if "prediction_proba" in result.columns
            else None
        )

        st.divider()
        if str(prediction) == "1":
            st.error("Resultado: **Positivo para diabetes**", icon="⚠️")
        else:
            st.success("Resultado: **Negativo para diabetes**", icon="✅")

        if proba is not None:
            st.metric("Probabilidade", f"{proba:.1%}")
            st.progress(float(proba))


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="Diabetes — Dashboard ML",
        page_icon="🩺",
        layout="wide",
    )

    st.title("Dashboard ML — Predição de Diabetes")

    report = load_test_report()
    artifacts = load_production_artifacts()

    if report is None and artifacts is None:
        st.error(
            "Nenhum artefato encontrado. Rode `uv run kedro run` "
            "pra treinar o pipeline primeiro."
        )
        return

    tab_metrics, tab_cm, tab_inference = st.tabs([
        "📊 Métricas",
        "🔢 Matrizes de Confusão",
        "🩺 Predição ao Vivo",
    ])

    with tab_metrics:
        if report:
            render_metrics_tab(report)
        else:
            st.warning("Nenhum relatório de teste encontrado. Rode o pipeline completo.")

    with tab_cm:
        if report:
            render_confusion_matrix_tab(report)
        else:
            st.warning("Nenhum relatório de teste encontrado. Rode o pipeline completo.")

    with tab_inference:
        render_inference_tab(artifacts)

    with st.sidebar:
        st.markdown("### Sobre")
        if artifacts and "model" in artifacts:
            model_info = artifacts["model"]
            st.markdown(f"**Modelo em produção:** `{model_info.get('class_path', '?')}`")
            if "best_params" in model_info:
                with st.expander("Hiperparâmetros"):
                    st.json(model_info["best_params"])
        st.markdown(f"**Dados:** `{MODELS_DIR.relative_to(PROJECT_ROOT)}`")
        st.divider()
        if st.button("Limpar cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        st.caption("Projeto Insper — Deploy de ML")


if __name__ == "__main__":
    main()
