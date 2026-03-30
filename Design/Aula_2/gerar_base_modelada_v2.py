#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analytics_v2_common import DEFAULT_BASE_DIR, V2Config, build_config, setup_logging
from etapa_02_star_schema_v2 import run_stage_02_modelagem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera a base modelada v2 a partir dos dados raw.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg: V2Config = build_config(args.base_dir, args.data_dir, args.output_dir)
    summary = run_stage_02_modelagem(cfg)
    payload = {
        "table_name": "base_modelada_v2",
        "csv": summary["base_modelada_exports"]["csv"],
        "parquet": summary["base_modelada_exports"]["parquet"],
        "validation_status": summary["base_modelada_validation_status"],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
