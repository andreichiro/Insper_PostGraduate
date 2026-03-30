#!/usr/bin/env python3
from __future__ import annotations

"""
Entrypoint único para reproduzir o pipeline analítico atual (sem legado).

Etapas:
1) etapa_01_base.py
2) etapa_02_deep_dive.py
3) etapa_03_relatorio.py
4) etapa_04_metricas_mensais.py
"""

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

LOGGER = logging.getLogger("executar_pipeline_analytics")
DEFAULT_BASE_DIR = Path('/Users/akatsurada/Documents/INSPER/Design/Aula_2')


@dataclass(frozen=True)
class FullPipelineConfig:
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
    parser = argparse.ArgumentParser(description="Roda o pipeline completo (etapas 01, 02, 03 e 04).")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Diretório base do projeto.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Diretório de dados fonte. Default: <base-dir>/base_aprendizap",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Diretório de saída. Default: <base-dir>/analysis_output",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> FullPipelineConfig:
    base_dir = args.base_dir.resolve()
    data_dir = (args.data_dir if args.data_dir is not None else base_dir / 'base_aprendizap').resolve()
    output_dir = (args.output_dir if args.output_dir is not None else base_dir / 'analysis_output').resolve()
    return FullPipelineConfig(
        base_dir=base_dir,
        data_dir=data_dir,
        output_dir=output_dir,
        python_exec=sys.executable,
    )


def run_step(config: FullPipelineConfig, script_name: str, script_args: List[str]) -> None:
    script = config.base_dir / script_name
    if not script.exists():
        raise FileNotFoundError(f'Script not found: {script}')
    cmd = [config.python_exec, str(script), *script_args]
    LOGGER.info("Running step: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    setup_logging()
    config = build_config(parse_args())

    common_args = [
        '--base-dir', str(config.base_dir),
        '--data-dir', str(config.data_dir),
        '--output-dir', str(config.output_dir),
    ]

    run_step(
        config,
        'etapa_01_base.py',
        ['--data-dir', str(config.data_dir), '--output-dir', str(config.output_dir)],
    )
    run_step(config, 'etapa_02_deep_dive.py', common_args)
    run_step(config, 'etapa_03_relatorio.py', common_args)
    run_step(config, 'etapa_04_metricas_mensais.py', common_args)

    LOGGER.info("Full analytics pipeline completed successfully.")


if __name__ == '__main__':
    main()
