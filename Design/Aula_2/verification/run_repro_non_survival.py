#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from common import (
    build_file_manifest,
    collect_environment_manifest,
    collect_in_scope_files,
    ensure_dir,
    load_manifest_spec,
    run_cmd,
    utc_now_iso,
    write_json,
)


@dataclass(frozen=True)
class ReproConfig:
    base_dir: Path
    data_dir: Path
    baseline_output_dir: Path
    repro_root: Path
    run_id: str
    python_exec: str
    spec_file: Path
    skip_pipeline: bool


def parse_args() -> argparse.Namespace:
    default_base = Path("/Users/akatsurada/Documents/INSPER/Design/Aula_2")
    parser = argparse.ArgumentParser(description="Reproduce non-survival pipeline outputs in isolated directory.")
    parser.add_argument("--base-dir", type=Path, default=default_base)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--baseline-output-dir", type=Path, default=None)
    parser.add_argument("--repro-root", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--python-exec", type=str, default=None)
    parser.add_argument("--spec-file", type=Path, default=None)
    parser.add_argument("--skip-pipeline", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ReproConfig:
    base_dir = args.base_dir.resolve()
    data_dir = (args.data_dir or (base_dir / "base_aprendizap")).resolve()
    baseline_output_dir = (args.baseline_output_dir or (base_dir / "analysis_output")).resolve()
    repro_root = (args.repro_root or (base_dir / "analysis_output_repro")).resolve()
    run_id = args.run_id or f"run_{utc_now_iso().replace(':', '').replace('-', '').replace('.', '').replace('+00:00', 'Z')}"
    python_exec = args.python_exec or str((base_dir / ".venv" / "bin" / "python").resolve())
    spec_file = (args.spec_file or (base_dir / "verification" / "spec" / "non_survival_manifest.yaml")).resolve()
    return ReproConfig(
        base_dir=base_dir,
        data_dir=data_dir,
        baseline_output_dir=baseline_output_dir,
        repro_root=repro_root,
        run_id=run_id,
        python_exec=python_exec,
        spec_file=spec_file,
        skip_pipeline=bool(args.skip_pipeline),
    )


def load_stage1_config(baseline_output_dir: Path) -> Dict[str, Any]:
    consolidated_path = baseline_output_dir / "consolidated_status.json"
    if not consolidated_path.exists():
        raise FileNotFoundError(f"Missing consolidated_status.json at {consolidated_path}")
    payload = json.loads(consolidated_path.read_text(encoding="utf-8"))
    cfg = payload.get("run_metadata", {}).get("config", {})
    if not cfg:
        raise ValueError("Could not find run_metadata.config in consolidated_status.json")
    return cfg


def freeze_manifest(label: str, root: Path, files: List[Path], out_path: Path) -> Dict[str, Any]:
    manifest = {
        "label": label,
        "root_dir": str(root),
        "generated_at_utc": utc_now_iso(),
        "files": build_file_manifest(root=root, files=files, include_tabular_profile=True),
    }
    write_json(out_path, manifest)
    return manifest


def stage1_command(cfg: ReproConfig, stage1_params: Dict[str, Any], repro_output_dir: Path) -> List[str]:
    cmd = [
        cfg.python_exec,
        str((cfg.base_dir / "etapa_01_base_rigorosa.py").resolve()),
        "--data-dir",
        str(cfg.data_dir),
        "--output-dir",
        str(repro_output_dir),
        "--random-seed",
        str(stage1_params.get("random_seed", 42)),
        "--churn-days",
        str(stage1_params.get("churn_days", 30)),
        "--conversion-days",
        str(stage1_params.get("conversion_days", 30)),
        "--alpha",
        str(stage1_params.get("alpha", 0.05)),
        "--min-segment-n",
        str(stage1_params.get("min_segment_n", 200)),
        "--temporal-backtest-months",
        str(stage1_params.get("temporal_backtest_months", 18)),
        "--temporal-min-train-months",
        str(stage1_params.get("temporal_min_train_months", 6)),
        "--temporal-train-row-cap",
        str(stage1_params.get("temporal_train_row_cap", 300000)),
        "--temporal-min-test-rows",
        str(stage1_params.get("temporal_min_test_rows", 5000)),
        "--threshold-grid-size",
        str(stage1_params.get("threshold_grid_size", 99)),
        "--intervention-capacity-pct",
        str(stage1_params.get("intervention_capacity_pct", 0.2)),
        "--min-precision-for-trigger",
        str(stage1_params.get("min_precision_for_trigger", 0.8)),
        "--churn-tp-value",
        str(stage1_params.get("churn_tp_value", 3.0)),
        "--churn-fp-cost",
        str(stage1_params.get("churn_fp_cost", 1.0)),
        "--churn-fn-cost",
        str(stage1_params.get("churn_fn_cost", 2.0)),
    ]
    if not bool(stage1_params.get("run_temporal_backtest", True)):
        cmd.append("--no-temporal-backtest")
    return cmd


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    spec = load_manifest_spec(cfg.spec_file)

    run_dir = ensure_dir(cfg.repro_root / cfg.run_id)
    repro_output_dir = ensure_dir(run_dir / "analysis_output_repro")
    manifests_dir = ensure_dir(run_dir / "manifests")
    logs_dir = ensure_dir(run_dir / "logs")

    stage1_params = load_stage1_config(cfg.baseline_output_dir)

    scope = spec.get("scope", {})
    patterns = scope.get("output_patterns", [])
    freeze_patterns = scope.get("freeze_output_patterns", patterns)
    exclude_contains = scope.get("exclude_path_contains", [])
    exclude_regex = scope.get("exclude_regex", [])
    required_inputs = scope.get("required_inputs", [])

    for rel in required_inputs:
        path = cfg.baseline_output_dir / rel
        if not path.exists():
            raise FileNotFoundError(f"Required input missing for stage 3: {path}")

    baseline_files = collect_in_scope_files(
        root=cfg.baseline_output_dir,
        patterns=freeze_patterns,
        exclude_contains=exclude_contains,
        exclude_regex=exclude_regex,
    )

    baseline_data_files = sorted(
        [*cfg.data_dir.glob("*.csv"), *cfg.data_dir.glob("*.xlsx")],
        key=lambda p: p.name,
    )

    freeze_manifest(
        label="baseline_outputs_non_survival",
        root=cfg.baseline_output_dir,
        files=baseline_files,
        out_path=manifests_dir / "baseline_outputs_manifest.json",
    )
    freeze_manifest(
        label="baseline_data_sources",
        root=cfg.data_dir,
        files=baseline_data_files,
        out_path=manifests_dir / "baseline_data_manifest.json",
    )

    env_manifest = collect_environment_manifest(cfg.python_exec)
    write_json(manifests_dir / "environment_manifest.json", env_manifest)

    commands_log: List[Dict[str, Any]] = []

    if not cfg.skip_pipeline:
        cmd1 = stage1_command(cfg=cfg, stage1_params=stage1_params, repro_output_dir=repro_output_dir)
        commands_log.append(
            run_cmd(cmd1, cwd=cfg.base_dir, log_file=logs_dir / "01_etapa_01.log")
        )

        cmd2 = [
            cfg.python_exec,
            str((cfg.base_dir / "etapa_02_deep_dive.py").resolve()),
            "--base-dir",
            str(cfg.base_dir),
            "--data-dir",
            str(cfg.data_dir),
            "--output-dir",
            str(repro_output_dir),
        ]
        commands_log.append(
            run_cmd(cmd2, cwd=cfg.base_dir, log_file=logs_dir / "02_etapa_02.log")
        )

        src_q = cfg.baseline_output_dir / "executive_quadro_por_item.csv"
        dst_q = repro_output_dir / "executive_quadro_por_item.csv"
        ensure_dir(dst_q.parent)
        shutil.copy2(src_q, dst_q)

        cmd3 = [
            cfg.python_exec,
            str((cfg.base_dir / "etapa_03_relatorio_interativo.py").resolve()),
            "--base-dir",
            str(cfg.base_dir),
            "--data-dir",
            str(cfg.data_dir),
            "--output-dir",
            str(repro_output_dir),
            "--run-survival-benchmark",
            "0",
        ]
        commands_log.append(
            run_cmd(cmd3, cwd=cfg.base_dir, log_file=logs_dir / "03_etapa_03.log")
        )

    repro_files = collect_in_scope_files(
        root=repro_output_dir,
        patterns=freeze_patterns,
        exclude_contains=exclude_contains,
        exclude_regex=exclude_regex,
    )

    freeze_manifest(
        label="reproduced_outputs_non_survival",
        root=repro_output_dir,
        files=repro_files,
        out_path=manifests_dir / "repro_outputs_manifest.json",
    )

    metadata = {
        "run_id": cfg.run_id,
        "created_at_utc": utc_now_iso(),
        "base_dir": str(cfg.base_dir),
        "data_dir": str(cfg.data_dir),
        "baseline_output_dir": str(cfg.baseline_output_dir),
        "repro_output_dir": str(repro_output_dir),
        "python_exec": cfg.python_exec,
        "spec_file": str(cfg.spec_file),
        "skip_pipeline": cfg.skip_pipeline,
        "stage1_params": stage1_params,
        "commands": commands_log,
    }
    write_json(run_dir / "run_metadata.json", metadata)

    print(str(run_dir))


if __name__ == "__main__":
    main()
