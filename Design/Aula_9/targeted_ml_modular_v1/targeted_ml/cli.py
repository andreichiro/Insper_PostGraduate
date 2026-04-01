from __future__ import annotations

import argparse
from pathlib import Path

import typer

from targeted_ml.config.loader import load_analysis_spec
from targeted_ml.orchestration.compatibility import CompatibilityError, compare_contracts
from targeted_ml.orchestration.pipeline import (
    build_all,
    build_html,
    build_ml,
    build_modelled,
    export_serving,
    resolve_paths,
    score_modelled_input,
    score_raw_input,
    score_scoring_frame_input,
)


CTX_SETTINGS = {"allow_extra_args": True, "ignore_unknown_options": True}

app = typer.Typer(add_completion=False, no_args_is_help=True, context_settings=CTX_SETTINGS)


def _make_parser(name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"targeted-ml {name}")
    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--analysis-spec", required=True)
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--override", action="append", default=[])


def _parse_ctx(ctx: typer.Context, parser: argparse.ArgumentParser) -> argparse.Namespace:
    return parser.parse_args(list(ctx.args))


def _as_optional_path(value: str) -> Path | None:
    return Path(value).resolve() if value else None


def _load_spec_from_args(args: argparse.Namespace):
    spec_path = Path(args.analysis_spec).resolve()
    if not spec_path.exists():
        raise typer.BadParameter(f"analysis spec not found: {spec_path}")
    overrides = list(args.override or [])
    if getattr(args, "dataset_root", ""):
        overrides.append(f"data.dataset_root={Path(args.dataset_root).resolve()}")
    return load_analysis_spec(spec_path, overrides=overrides)


@app.command("validate-spec", context_settings=CTX_SETTINGS)
def validate_spec(ctx: typer.Context) -> None:
    parser = _make_parser("validate-spec")
    _add_common_args(parser)
    args = _parse_ctx(ctx, parser)
    spec = _load_spec_from_args(args)
    typer.echo(spec.model_dump_json(indent=2))


@app.command("build-modelled", context_settings=CTX_SETTINGS)
def cli_build_modelled(ctx: typer.Context) -> None:
    parser = _make_parser("build-modelled")
    _add_common_args(parser)
    args = _parse_ctx(ctx, parser)
    spec = _load_spec_from_args(args)
    paths = resolve_paths(Path(__file__).resolve().parents[1], _as_optional_path(args.output_root))
    build_modelled(spec, paths)
    typer.echo(str(paths.modelled_duckdb))


@app.command("build-ml", context_settings=CTX_SETTINGS)
def cli_build_ml(ctx: typer.Context) -> None:
    parser = _make_parser("build-ml")
    _add_common_args(parser)
    parser.add_argument("--skip-post-model-refit", dest="skip_post_model_refit", action="store_true")
    parser.add_argument("--skip-post-model-outputs", dest="skip_post_model_refit", action="store_true", help=argparse.SUPPRESS)
    args = _parse_ctx(ctx, parser)
    spec = _load_spec_from_args(args)
    if args.skip_post_model_refit:
        spec.modeling.skip_post_model_refit = True
    paths = resolve_paths(Path(__file__).resolve().parents[1], _as_optional_path(args.output_root))
    build_ml(spec, paths)
    typer.echo(str(paths.build_dir))


@app.command("build-report", context_settings=CTX_SETTINGS)
def cli_build_report(ctx: typer.Context) -> None:
    parser = _make_parser("build-report")
    _add_common_args(parser)
    args = _parse_ctx(ctx, parser)
    spec = _load_spec_from_args(args)
    paths = resolve_paths(Path(__file__).resolve().parents[1], _as_optional_path(args.output_root))
    output_html = build_html(spec, paths)
    typer.echo(str(output_html))


@app.command("export-serving", context_settings=CTX_SETTINGS)
def cli_export_serving(ctx: typer.Context) -> None:
    parser = _make_parser("export-serving")
    _add_common_args(parser)
    parser.add_argument("--problem-key", action="append", default=[])
    parser.add_argument("--model-name", action="append", default=[])
    args = _parse_ctx(ctx, parser)
    spec = _load_spec_from_args(args)
    paths = resolve_paths(Path(__file__).resolve().parents[1], _as_optional_path(args.output_root))
    manifest_path = export_serving(
        spec,
        paths,
        problem_keys=list(args.problem_key or []),
        model_names=list(args.model_name or []),
    )
    typer.echo(str(manifest_path))


@app.command("score-modelled", context_settings=CTX_SETTINGS)
def cli_score_modelled(ctx: typer.Context) -> None:
    parser = _make_parser("score-modelled")
    _add_common_args(parser)
    parser.add_argument("--modelled-duckdb", default="")
    parser.add_argument("--problem-key", action="append", default=[])
    parser.add_argument("--model-name", action="append", default=[])
    parser.add_argument("--run-name", default="")
    args = _parse_ctx(ctx, parser)
    spec = _load_spec_from_args(args)
    paths = resolve_paths(Path(__file__).resolve().parents[1], _as_optional_path(args.output_root))
    modelled_duckdb = Path(args.modelled_duckdb).resolve() if args.modelled_duckdb else paths.modelled_duckdb
    run_dir = score_modelled_input(
        spec,
        paths,
        modelled_duckdb=modelled_duckdb,
        problem_keys=list(args.problem_key or []),
        model_names=list(args.model_name or []),
        run_name=args.run_name or None,
    )
    typer.echo(str(run_dir))


@app.command("score-frame", context_settings=CTX_SETTINGS)
def cli_score_frame(ctx: typer.Context) -> None:
    parser = _make_parser("score-frame")
    _add_common_args(parser)
    parser.add_argument("--scoring-frame", required=True)
    parser.add_argument("--latest-observed-ts", default="")
    parser.add_argument("--problem-key", action="append", default=[])
    parser.add_argument("--model-name", action="append", default=[])
    parser.add_argument("--run-name", default="")
    args = _parse_ctx(ctx, parser)
    spec = _load_spec_from_args(args)
    paths = resolve_paths(Path(__file__).resolve().parents[1], _as_optional_path(args.output_root))
    run_dir = score_scoring_frame_input(
        spec,
        paths,
        scoring_frame_path=Path(args.scoring_frame).resolve(),
        latest_observed_ts=args.latest_observed_ts or None,
        problem_keys=list(args.problem_key or []),
        model_names=list(args.model_name or []),
        run_name=args.run_name or None,
    )
    typer.echo(str(run_dir))


@app.command("score-raw", context_settings=CTX_SETTINGS)
def cli_score_raw(ctx: typer.Context) -> None:
    parser = _make_parser("score-raw")
    _add_common_args(parser)
    parser.add_argument("--problem-key", action="append", default=[])
    parser.add_argument("--model-name", action="append", default=[])
    parser.add_argument("--run-name", default="")
    args = _parse_ctx(ctx, parser)
    if not args.dataset_root:
        raise typer.BadParameter("--dataset-root is required for score-raw")
    spec = _load_spec_from_args(args)
    paths = resolve_paths(Path(__file__).resolve().parents[1], _as_optional_path(args.output_root))
    run_dir = score_raw_input(
        spec,
        paths,
        dataset_root=Path(args.dataset_root).resolve(),
        problem_keys=list(args.problem_key or []),
        model_names=list(args.model_name or []),
        run_name=args.run_name or None,
    )
    typer.echo(str(run_dir))


@app.command("build", context_settings=CTX_SETTINGS)
def cli_build_all(ctx: typer.Context) -> None:
    parser = _make_parser("build")
    _add_common_args(parser)
    parser.add_argument("--skip-modelled", action="store_true")
    parser.add_argument("--skip-post-model-refit", dest="skip_post_model_refit", action="store_true")
    parser.add_argument("--skip-post-model-outputs", dest="skip_post_model_refit", action="store_true", help=argparse.SUPPRESS)
    args = _parse_ctx(ctx, parser)
    spec = _load_spec_from_args(args)
    paths = build_all(
        spec,
        project_root=Path(__file__).resolve().parents[1],
        output_root=_as_optional_path(args.output_root),
        skip_modelled=args.skip_modelled,
        skip_post_model_refit=args.skip_post_model_refit,
    )
    typer.echo(str(paths.build_dir))


@app.command("list-datasets")
def cli_list_datasets() -> None:
    base = Path(__file__).resolve().parents[1] / "data"
    for child in sorted(base.iterdir()):
        typer.echo(str(child))


@app.command("list-policies", context_settings=CTX_SETTINGS)
def cli_list_policies(ctx: typer.Context) -> None:
    parser = _make_parser("list-policies")
    _add_common_args(parser)
    args = _parse_ctx(ctx, parser)
    spec = _load_spec_from_args(args)
    for policy in spec.post_model_outputs.band_policies:
        typer.echo(f"band::{policy.policy_name}::{policy.parameter_json}")
    for policy in spec.post_model_outputs.heavy_user_policies:
        typer.echo(f"heavy::{policy.policy_name}::{policy.parameter_json}")


@app.command("check-compatibility", context_settings=CTX_SETTINGS)
def cli_check_compatibility(ctx: typer.Context) -> None:
    parser = _make_parser("check-compatibility")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--baseline-contract", default="")
    args = _parse_ctx(ctx, parser)
    root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root).resolve()
    current_contract = output_root / "metadata" / "compatibility_contract_current.json"
    baseline = Path(args.baseline_contract).resolve() if args.baseline_contract else (root / "baseline" / "compatibility_contract_v2.json")
    baseline_build_dir = root / "baseline" / "build_v2"
    if not current_contract.exists():
        raise typer.BadParameter(f"compatibility contract not found: {current_contract}")
    try:
        result = compare_contracts(
            baseline,
            current_contract,
            baseline_build_dir=baseline_build_dir,
            current_build_dir=output_root,
        )
    except CompatibilityError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=1) from exc
    typer.echo(result.model_dump_json(indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
