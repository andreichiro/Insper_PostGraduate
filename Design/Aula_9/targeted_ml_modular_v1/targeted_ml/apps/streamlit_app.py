from __future__ import annotations

import difflib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Support direct `streamlit run .../streamlit_app.py` execution without
# requiring the project to be pre-installed as a package.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional dependency
    st = None

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "build"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data"
DEFAULT_SPEC_NAME = "activity.yaml"
SPECS_DIR = PROJECT_ROOT / "specs"
PROGRESS_RE = re.compile(r"\[progress\]\s+(\d+(?:\.\d+)?)%")
APP_TITLE = "Machine Learning APP"
APP_SUBTITLE = "Fundação 1bi"
TRAINING_ACTIONS = [
    ("Validar spec", "validate-spec"),
    ("Gerar modelada", "build-modelled"),
    ("Rodar ML", "build-ml"),
    ("Gerar relatório", "build-report"),
    ("Rodar tudo", "build"),
    ("Exportar serving", "export-serving"),
]
INFERENCE_MODE_LABELS = {
    "Base modelada (DuckDB)": "modelled_duckdb",
    "Arquivo para score (CSV/Parquet)": "scoring_frame_file",
    "Raw do projeto": "raw_dataset_root",
}
DELIVERY_PREVIEW_COLUMNS = [
    "teacher_unique_id",
    "risk_score",
    "risk_rank",
    "flag_top_10_percent",
    "flag_tercis",
    "flag_score_ge_0_70",
]
DELIVERY_RUN_ARTIFACTS = [
    ("all_scored_clients.parquet", "Base inteira rankeada"),
    ("high_risk_clients_top10.parquet", "Fila filtrada: top 10%"),
    ("high_risk_clients_tercis.parquet", "Fila filtrada: tercis"),
    ("high_risk_clients_score_ge_0_70.parquet", "Fila filtrada: risk_score >= 0,70"),
]
SPEC_LABELS = {
    "activity.yaml": "Atividade (principal)",
    "churn_m1.yaml": "Churn M1",
    "return_m1.yaml": "Retorno M1",
}


def _ensure_streamlit() -> None:
    if st is None:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "streamlit is not installed. Install the optional app dependencies first: "
            "pip install -e .[app]"
        )


def _render_resolved_spec_yaml(selected_spec: Path) -> str:
    from targeted_ml.config.loader import render_resolved_spec_yaml

    return render_resolved_spec_yaml(selected_spec)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _list_specs() -> list[Path]:
    return sorted(SPECS_DIR.glob("*.yaml"))


def _list_runnable_specs() -> list[Path]:
    return [path for path in _list_specs() if path.name != "base.yaml"]


def _spec_label(path: Path) -> str:
    return SPEC_LABELS.get(path.name, path.stem.replace("_", " ").title())


def _spec_options_map() -> dict[str, Path]:
    return {_spec_label(path): path for path in _list_runnable_specs()}


def _default_spec_index(spec_options: list[Path]) -> int:
    for index, path in enumerate(spec_options):
        if path.name == DEFAULT_SPEC_NAME:
            return index
    return 0


def _paths(output_root: str) -> dict[str, Path]:
    root = Path(output_root).expanduser().resolve()
    return {
        "output_root": root,
        "reports_dir": root / "reports",
        "metadata_dir": root / "metadata",
        "tables_dir": root / "tables",
        "serving_dir": root / "serving",
        "inference_runs_dir": root / "inference_runs",
        "modelled_duckdb": root / "modelled" / "duckdb" / "base_modelada_v2.duckdb",
        "app_specs_dir": root / "app_specs",
        "app_uploads_dir": root / "app_uploads",
    }


def _stage_progress(line: str) -> float | None:
    match = PROGRESS_RE.search(line)
    if not match:
        return None
    return max(0.0, min(100.0, float(match.group(1))))


def _run_cli(command: list[str], cwd: Path) -> tuple[int, list[str]]:
    progress = st.progress(0.0)
    status = st.empty()
    log_box = st.empty()
    lines: list[str] = []
    status.info("Executando comando...")
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        lines.append(line)
        percent = _stage_progress(line)
        if percent is not None:
            progress.progress(percent / 100.0)
            status.info(f"Progresso detectado: {percent:0.1f}%")
        elif line.startswith("[serving]") or line.startswith("[inference]"):
            status.info(line)
        log_box.code("\n".join(lines[-200:]), language="text")
    return_code = proc.wait()
    progress.progress(1.0 if return_code == 0 else 0.0)
    if return_code == 0:
        status.success("Comando concluído com sucesso.")
    else:
        status.error(f"Comando falhou com código {return_code}.")
    return return_code, lines


def _write_temp_spec(output_root: Path, spec_name: str, spec_text: str) -> Path:
    app_specs_dir = output_root / "app_specs"
    app_specs_dir.mkdir(parents=True, exist_ok=True)
    slug = Path(spec_name).stem
    spec_path = app_specs_dir / f"{slug}__streamlit.yaml"
    spec_path.write_text(spec_text, encoding="utf-8")
    return spec_path


def _build_cli_command(
    spec_path: Path,
    output_root: Path,
    command_name: str,
    extra_args: list[str] | None = None,
) -> list[str]:
    args = [
        sys.executable,
        "-m",
        "targeted_ml",
        command_name,
        "--analysis-spec",
        str(spec_path),
        "--output-root",
        str(output_root),
    ]
    if extra_args:
        args.extend(extra_args)
    return args


def _append_optional_arg(command: list[str], flag: str, value: str) -> None:
    if value.strip():
        command.extend([flag, value.strip()])


def _download_binary_button(label: str, path: Path, *, file_name: str | None = None, mime: str = "application/octet-stream") -> None:
    st.download_button(
        label,
        data=path.read_bytes(),
        file_name=file_name or path.name,
        mime=mime,
        use_container_width=True,
    )


def _preview_delivery_frame(frame: pd.DataFrame) -> pd.DataFrame:
    preferred = [col for col in DELIVERY_PREVIEW_COLUMNS if col in frame.columns]
    if preferred:
        return frame[preferred].copy()
    return frame.copy()


def _render_run_delivery_artifacts(run_dir: Path, *, section_title: str) -> None:
    if not run_dir.exists():
        st.info("Run de inferência não encontrado.")
        return
    st.subheader(section_title)
    run_manifest_path = run_dir / "run_manifest.json"
    if run_manifest_path.exists():
        st.markdown("**Manifesto do run**")
        st.json(_read_json(run_manifest_path))
        _download_binary_button("Baixar manifesto do run", run_manifest_path, mime="application/json")
    for filename, label in DELIVERY_RUN_ARTIFACTS:
        artifact_path = run_dir / filename
        if not artifact_path.exists():
            continue
        frame = pd.read_parquet(artifact_path)
        st.markdown(f"**{label}**")
        st.caption(f"{len(frame):,} linhas")
        preview = _preview_delivery_frame(frame).head(100)
        st.dataframe(preview, use_container_width=True, hide_index=True)
        _download_binary_button(f"Baixar {label.lower()}", artifact_path, mime="application/octet-stream")


def _render_latest_delivery_preview(paths: dict[str, Path]) -> None:
    latest_run_path = paths["inference_runs_dir"] / "latest.json"
    if not latest_run_path.exists():
        return
    latest_run = _read_json(latest_run_path).get("latest_run_dir")
    if not latest_run:
        return
    run_dir = Path(latest_run)
    st.info(f"Último run: {run_dir}")
    _render_run_delivery_artifacts(run_dir, section_title="Artefatos de entrega do último run")


def _render_serving_artifacts(paths: dict[str, Path]) -> None:
    serving_manifest_path = paths["serving_dir"] / "serving_manifest.json"
    if not serving_manifest_path.exists():
        st.warning("Ainda não existe serving_manifest.json. Use Export serving primeiro.")
        return

    manifest = _read_json(serving_manifest_path)
    st.success(f"Serving pronto: {manifest.get('serving_status', 'desconhecido')}")
    st.json(
        {
            "primary_model_artifact_id": manifest.get("primary_model_artifact_id"),
            "export_id": manifest.get("export_id"),
            "inference_contract_path": manifest.get("inference_contract_path"),
            "exported_model_count": manifest.get("exported_model_count"),
        }
    )

    model_id = manifest.get("primary_model_artifact_id")
    if model_id:
        model_path = paths["serving_dir"] / "models" / f"{model_id}.joblib"
        if model_path.exists():
            st.markdown("**Modelo salvo**")
            st.code(str(model_path), language="text")
            _download_binary_button("Baixar modelo salvo (.joblib)", model_path)


def _render_training_tab(paths: dict[str, Path], selected_spec: Path) -> None:
    source_text = selected_spec.read_text(encoding="utf-8")
    resolved_text = _render_resolved_spec_yaml(selected_spec)
    spec_key = f"spec_text::{selected_spec.name}"
    if spec_key not in st.session_state:
        st.session_state[spec_key] = resolved_text
    st.caption(
        f"Spec base selecionado: `{selected_spec.name}`. "
        "A UI edita a configuração efetiva resolvida, já com herança de `base.yaml` aplicada."
    )
    with st.expander("Ver arquivo-fonte selecionado"):
        st.code(source_text, language="yaml")
    edited_text = st.text_area(
        "Configuração efetiva editável",
        value=st.session_state[spec_key],
        height=360,
        key=f"editor::{selected_spec.name}",
    )
    st.session_state[spec_key] = edited_text
    if edited_text != resolved_text:
        diff = "".join(
            difflib.unified_diff(
                resolved_text.splitlines(keepends=True),
                edited_text.splitlines(keepends=True),
                fromfile=f"{selected_spec.stem}__resolved.yaml",
                tofile=f"{selected_spec.stem}__streamlit.yaml",
            )
        )
        with st.expander("Diff do spec"):
            st.code(diff or "Sem diff.", language="diff")
    spec_path = _write_temp_spec(paths["output_root"], selected_spec.name, edited_text)
    st.session_state["active_app_spec_path"] = str(spec_path)
    st.caption(f"Spec usado pela UI: `{spec_path}`")
    cols = st.columns(6)
    for col, (label, command_name) in zip(cols, TRAINING_ACTIONS):
        if col.button(label, use_container_width=True):
            _run_cli(_build_cli_command(spec_path, paths["output_root"], command_name), PROJECT_ROOT)


def _render_inference_tab(paths: dict[str, Path], selected_spec: Path) -> None:
    active_spec = Path(st.session_state.get("active_app_spec_path", str(selected_spec)))
    _render_serving_artifacts(paths)

    contract_path = paths["serving_dir"] / "inference_contract.json"
    if contract_path.exists():
        contract = _read_json(contract_path)
        with st.expander("Contrato de inferência"):
            st.json(contract)
        raw_contract = contract.get("raw_dataset_root_contract", {})
        if raw_contract.get("supported"):
            st.subheader("Contrato para raw_dataset_root")
            st.json(raw_contract)
        required_cols = contract.get("required_scoring_columns_union", [])
        if required_cols:
            st.subheader("Colunas mínimas aceitas para scoring_frame")
            st.dataframe(pd.DataFrame({"required_scoring_column": required_cols}), use_container_width=True, hide_index=True)
        template_path = paths["serving_dir"] / "scoring_frame_template.csv"
        if template_path.exists():
            _download_binary_button("Baixar template de scoring frame", template_path, mime="text/csv")

    score_mode_label = st.radio("Entrada para inferência", list(INFERENCE_MODE_LABELS.keys()), horizontal=True)
    score_mode = INFERENCE_MODE_LABELS[score_mode_label]
    run_name = st.text_input("Nome opcional do run", value="")
    if score_mode == "modelled_duckdb":
        modelled_path = st.text_input(
            "Caminho do modelled_duckdb",
            value=str(paths["modelled_duckdb"]),
        )
        if st.button("Rodar score na base modelada", use_container_width=True):
            command = _build_cli_command(
                active_spec,
                paths["output_root"],
                "score-modelled",
                ["--modelled-duckdb", modelled_path],
            )
            _append_optional_arg(command, "--run-name", run_name)
            _run_cli(command, PROJECT_ROOT)
    elif score_mode == "raw_dataset_root":
        raw_dataset_root = st.text_input(
            "Caminho do dataset_root raw",
            value=str(DEFAULT_DATASET_ROOT),
        )
        st.caption("O builder espera encontrar os arquivos raw no caminho relativo configurado na spec, por padrão `raw/base_aprendizap`.")
        if st.button("Gerar modelada do raw e rodar score", use_container_width=True):
            command = _build_cli_command(
                active_spec,
                paths["output_root"],
                "score-raw",
                ["--dataset-root", raw_dataset_root],
            )
            _append_optional_arg(command, "--run-name", run_name)
            _run_cli(command, PROJECT_ROOT)
    else:
        latest_observed_ts = st.text_input("latest_observed_ts (opcional se o arquivo já vier com score_window_ready_flag)", value="")
        uploaded = st.file_uploader("Upload CSV ou Parquet", type=["csv", "parquet"])
        if uploaded is not None and st.button("Rodar score no arquivo enviado", use_container_width=True):
            uploads_dir = paths["app_uploads_dir"]
            uploads_dir.mkdir(parents=True, exist_ok=True)
            upload_path = uploads_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{Path(uploaded.name).name}"
            upload_path.write_bytes(uploaded.getbuffer())
            command = _build_cli_command(
                active_spec,
                paths["output_root"],
                "score-frame",
                ["--scoring-frame", str(upload_path)],
            )
            _append_optional_arg(command, "--latest-observed-ts", latest_observed_ts)
            _append_optional_arg(command, "--run-name", run_name)
            _run_cli(command, PROJECT_ROOT)

    _render_latest_delivery_preview(paths)


def _render_saved_outputs_tab(paths: dict[str, Path]) -> None:
    run_dirs = sorted([p for p in paths["inference_runs_dir"].iterdir() if p.is_dir()], reverse=True) if paths["inference_runs_dir"].exists() else []
    if not run_dirs:
        st.info("Ainda não existem inference runs materializados.")
        return
    selected = st.selectbox("Inference run salvo", run_dirs, format_func=lambda p: p.name)
    run_manifest_path = selected / "run_manifest.json"
    if run_manifest_path.exists():
        st.json(_read_json(run_manifest_path))
    validation_path = selected / "validation_report.parquet"
    _render_run_delivery_artifacts(selected, section_title="Artefatos de entrega do run selecionado")
    if validation_path.exists():
        st.subheader("Validation report")
        st.dataframe(pd.read_parquet(validation_path), use_container_width=True, hide_index=True)


def _render_reports_tab(paths: dict[str, Path]) -> None:
    html_candidates = sorted(paths["reports_dir"].glob("*.html"))
    if not html_candidates:
        st.info("Nenhum HTML encontrado no output_root atual.")
        return
    selected = st.selectbox("Relatório HTML", html_candidates, format_func=lambda p: p.name)
    st.code(str(selected), language="text")
    html_text = selected.read_text(encoding="utf-8")
    st.download_button("Baixar HTML", data=html_text.encode("utf-8"), file_name=selected.name, mime="text/html")
    with st.expander("Preview bruto (primeiras 2000 chars)"):
        st.code(html_text[:2000], language="html")


def run_app() -> None:
    _ensure_streamlit()
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)
    output_root_value = st.sidebar.text_input("Pasta de saída (output_root)", value=str(DEFAULT_OUTPUT_ROOT))
    spec_options_map = _spec_options_map()
    spec_labels = list(spec_options_map.keys())
    if not spec_labels:
        st.error("Nenhum spec encontrado em specs/.")
        return
    selected_label = st.sidebar.selectbox(
        "Estudo ativo",
        spec_labels,
        index=_default_spec_index(list(spec_options_map.values())),
    )
    selected_spec = spec_options_map[selected_label]
    st.sidebar.caption("O editor mostra a configuração efetiva resolvida do estudo selecionado.")
    if st.session_state.get("selected_spec_name") != selected_spec.name:
        st.session_state["selected_spec_name"] = selected_spec.name
        st.session_state["active_app_spec_path"] = str(selected_spec)
    paths = _paths(output_root_value)
    tabs = st.tabs(["Relatórios", "Inferência", "Treinamento", "Saídas salvas"])
    with tabs[0]:
        _render_reports_tab(paths)
    with tabs[1]:
        _render_inference_tab(paths, selected_spec)
    with tabs[2]:
        _render_training_tab(paths, selected_spec)
    with tabs[3]:
        _render_saved_outputs_tab(paths)


if __name__ == "__main__":  # pragma: no cover - streamlit entrypoint
    run_app()
