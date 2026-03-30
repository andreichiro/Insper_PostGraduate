#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


LOGGER = logging.getLogger("executar_pipeline_analytics_v2")
DEFAULT_BASE_DIR = Path("/Users/akatsurada/Documents/INSPER/Design/Aula_2")


@dataclass(frozen=True)
class FullPipelineV2Config:
    base_dir: Path
    data_dir: Path
    output_dir: Path
    python_exec: str


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roda o pipeline analítico v2 completo.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--python-exec",
        type=str,
        default=None,
        help="Interpretador Python para executar as etapas. Se omitido, tenta detectar um ambiente com pandas e duckdb.",
    )
    return parser.parse_args()


def python_has_analytics_deps(python_exec: str) -> bool:
    try:
        subprocess.run(
            [
                python_exec,
                "-c",
                "import pandas, duckdb",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def detect_python_exec(preferred: str | None) -> str:
    candidates: List[str] = []
    if preferred:
        candidates.append(preferred)
    env_python = os.environ.get("ANALYTICS_V2_PYTHON")
    if env_python:
        candidates.append(env_python)
    candidates.extend(
        [
            sys.executable,
            "/opt/anaconda3/bin/python3",
            "/opt/homebrew/bin/python3.12",
            "/opt/homebrew/bin/python3.11",
            "/opt/homebrew/bin/python3",
            "/usr/bin/python3",
        ]
    )
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if python_has_analytics_deps(candidate):
            return candidate
    raise RuntimeError(
        "Nenhum interpretador Python com pandas e duckdb foi encontrado. "
        "Use --python-exec ou a variável ANALYTICS_V2_PYTHON para apontar para um ambiente válido."
    )


def build_config(args: argparse.Namespace) -> FullPipelineV2Config:
    base_dir = args.base_dir.resolve()
    data_dir = (args.data_dir if args.data_dir is not None else base_dir / "base_aprendizap").resolve()
    output_dir = (args.output_dir if args.output_dir is not None else base_dir / "analysis_output_v2").resolve()
    return FullPipelineV2Config(
        base_dir=base_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        python_exec=detect_python_exec(args.python_exec),
    )


def run_step(config: FullPipelineV2Config, script_name: str, script_args: List[str]) -> None:
    script = config.base_dir / script_name
    if not script.exists():
        raise FileNotFoundError(f"Script não encontrado: {script}")
    cmd = [config.python_exec, str(script), *script_args]
    LOGGER.info("Executando etapa v2: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    setup_logging()
    config = build_config(parse_args())
    common_args = [
        "--base-dir",
        str(config.base_dir),
        "--data-dir",
        str(config.data_dir),
        "--output-dir",
        str(config.output_dir),
    ]
    for script_name in [
        "etapa_00_legado_v2.py",
        "etapa_01_qualidade_joins_v2.py",
        "etapa_02_star_schema_v2.py",
        "etapa_03_eda_v2.py",
        "etapa_04_analytics_usuarios_v2.py",
        "etapa_05_relatorios_v2.py",
        "etapa_06_onboarding_validado_v2.py",
        "etapa_07_ux_diagnostico_v2.py",
        "etapa_08_ux_flow_prioritization_v2.py",
        "etapa_09_prediction_drift_v2.py",
        "etapa_10_product_ux_backlog_v2.py",
    ]:
        run_step(config, script_name, common_args)

    LOGGER.info("Pipeline analytics v2 concluído com sucesso.")


if __name__ == "__main__":
    main()
