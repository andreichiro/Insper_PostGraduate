from __future__ import annotations

import argparse
from pathlib import Path

from targeted_ml.config.loader import load_analysis_spec
from targeted_ml.orchestration.pipeline import build_html, build_ml, build_modelled, resolve_paths


def _common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--analysis-spec", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--override", action="append", default=[])
    return parser


def _load_spec(args: argparse.Namespace):
    overrides = list(args.override or [])
    if args.dataset_root:
        overrides.append(f"data.dataset_root={args.dataset_root}")
    return load_analysis_spec(args.analysis_spec, overrides=overrides)


def raw_to_modelled_main() -> None:
    parser = _common_parser("Prepare modeled data for the modular targeted ML pipeline.")
    args = parser.parse_args()
    spec = _load_spec(args)
    paths = resolve_paths(project_root=Path(__file__).resolve().parents[1], output_root=args.output_root)
    build_modelled(spec, paths)


def modelled_to_ml_main() -> None:
    parser = _common_parser("Run modeled-to-ML for the modular targeted ML pipeline.")
    parser.add_argument("--skip-post-model-refit", dest="skip_post_model_refit", action="store_true")
    parser.add_argument("--skip-post-model-outputs", dest="skip_post_model_refit", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    spec = _load_spec(args)
    if args.skip_post_model_refit:
        spec.modeling.skip_post_model_refit = True
    paths = resolve_paths(project_root=Path(__file__).resolve().parents[1], output_root=args.output_root)
    build_ml(spec, paths)


def ml_to_html_main() -> None:
    parser = _common_parser("Render HTML for the modular targeted ML pipeline.")
    args = parser.parse_args()
    spec = _load_spec(args)
    paths = resolve_paths(project_root=Path(__file__).resolve().parents[1], output_root=args.output_root)
    build_html(spec, paths)
