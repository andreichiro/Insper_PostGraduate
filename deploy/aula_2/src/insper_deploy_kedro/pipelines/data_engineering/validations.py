"""Validações de qualidade — Great Expectations guiado por YAML (classes e limites declarativos)."""

from __future__ import annotations

import logging
from typing import Any

import great_expectations as gx
import pandas as pd

from insper_deploy_kedro.constants import SPLIT_COLUMN

logger = logging.getLogger(__name__)


def _ge_expectation_class(class_name: str) -> type:
    """Resolve nome de classe em gx.expectations (mesmo espírito do class_path da modelagem)."""
    try:
        return getattr(gx.expectations, class_name)
    except AttributeError as exc:
        raise ValueError(
            f"Expectation '{class_name}' não existe em great_expectations.expectations"
        ) from exc


def _instantiate_expectation(class_name: str, kwargs: dict[str, Any]) -> Any:
    """Instancia uma expectation GX a partir do nome declarado no YAML."""
    cls = _ge_expectation_class(class_name)
    return cls(**kwargs)


def _make_batch(df: pd.DataFrame, asset_name: str) -> Any:
    """Cria um batch GX efêmero a partir de um DataFrame."""
    context = gx.get_context()
    ds_name = f"pandas_{asset_name}"
    data_source = context.data_sources.add_pandas(ds_name)
    data_asset = data_source.add_dataframe_asset(name=asset_name)
    batch_def = data_asset.add_batch_definition_whole_dataframe(f"{asset_name}_batch")
    return batch_def.get_batch(batch_parameters={"dataframe": df})


def _run_expectations(batch: Any, expectations: list[Any]) -> dict[str, Any]:
    """Roda lista de expectations e retorna resumo."""
    results: list[dict[str, Any]] = []
    all_passed = True

    for exp in expectations:
        result = batch.validate(exp)
        passed = result["success"]
        if not passed:
            all_passed = False
        results.append({
            "expectation": type(exp).__name__,
            "kwargs": {
                k: v
                for k, v in result["expectation_config"]["kwargs"].items()
                if k != "batch_id"
            },
            "success": passed,
            "result_detail": result.get("result", {}),
        })

    return {"success": all_passed, "results": results}


def _flatten_column_groups(column_groups: dict[str, list[str]]) -> list[str]:
    return [
        column_name
        for group_columns in column_groups.values()
        for column_name in group_columns
    ]


def validate_clean_data(
    cleaned_data: pd.DataFrame,
    raw_columns: dict[str, list[str]],
    data_quality: dict[str, Any],
) -> pd.DataFrame:
    """Roda GE conforme `data_quality` no YAML — classes e ranges sem hardcode."""
    cfg = data_quality["cleaned"]
    classes = cfg["classes"]

    logger.info(
        "validate_clean_data: %d linhas, expectations declaradas no YAML",
        len(cleaned_data),
    )

    batch = _make_batch(cleaned_data, "cleaned_data")
    expectations: list[Any] = []

    exist_cls = classes["column_to_exist"]
    for col in _flatten_column_groups(raw_columns):
        expectations.append(
            _instantiate_expectation(exist_cls, {"column": col}),
        )

    not_null_cls = classes["column_not_null"]
    not_null_sev = cfg.get("not_null_severity", "critical")
    for col in raw_columns.get("numerical", []) + raw_columns.get("target", []):
        expectations.append(
            _instantiate_expectation(
                not_null_cls,
                {"column": col, "severity": not_null_sev},
            ),
        )

    between_cls = classes["column_between"]
    between_sev = cfg.get("between_severity", "warning")
    ranges = cfg.get("numerical_ranges", {})
    for col, bounds in ranges.items():
        if col not in cleaned_data.columns:
            continue
        lo, hi = bounds[0], bounds[1]
        expectations.append(
            _instantiate_expectation(
                between_cls,
                {
                    "column": col,
                    "min_value": lo,
                    "max_value": hi,
                    "severity": between_sev,
                },
            ),
        )

    min_rows = int(cfg["min_rows"])
    table_cls = classes["table_min_rows"]
    table_sev = cfg.get("table_min_rows_severity", "critical")
    expectations.append(
        _instantiate_expectation(
            table_cls,
            {"value": min_rows, "severity": table_sev},
        ),
    )

    target_cols = raw_columns.get("target", [])
    if target_cols:
        target_cls = classes["target_distinct_in_set"]
        allowed = cfg.get("target_allowed_values", [0, 1])
        expectations.append(
            _instantiate_expectation(
                target_cls,
                {"column": target_cols[0], "value_set": list(allowed)},
            ),
        )

    for spec in cfg.get("extra_expectations", []) or []:
        exp_class = spec["expectation_class"]
        kwargs = dict(spec.get("kwargs", {}))
        expectations.append(_instantiate_expectation(exp_class, kwargs))

    report = _run_expectations(batch, expectations)

    critical_failures = [
        r
        for r in report["results"]
        if not r["success"] and r.get("kwargs", {}).get("severity") == "critical"
    ]
    warning_failures = [
        r for r in report["results"] if not r["success"] and r not in critical_failures
    ]

    for w in warning_failures:
        logger.warning(
            "validate_clean_data WARNING: %s %s",
            w["expectation"],
            w["kwargs"],
        )

    if critical_failures:
        msgs = [f"{f['expectation']} {f['kwargs']}" for f in critical_failures]
        raise ValueError(
            "validate_clean_data: validação(ões) crítica(s) falharam: "
            + "; ".join(msgs)
        )

    logger.info(
        "validate_clean_data: %d expectativas OK, %d warnings",
        sum(1 for r in report["results"] if r["success"]),
        len(warning_failures),
    )
    return cleaned_data


def validate_split_data(
    split_data: pd.DataFrame,
    split_ratio: dict[str, float],
    stratify_column: str | None,
    data_quality: dict[str, Any],
) -> pd.DataFrame:
    """Checagens pós-split — thresholds só no YAML."""
    cfg = data_quality["split"]
    min_minority = float(cfg["min_minority_ratio"])
    warn_below = int(cfg["warn_when_split_rows_below"])

    logger.info(
        "validate_split_data: %d linhas, %d splits",
        len(split_data),
        len(split_ratio),
    )

    issues: list[str] = []

    for split_name in split_ratio:
        split_df = split_data[split_data[SPLIT_COLUMN] == split_name]
        n = len(split_df)

        if n == 0:
            issues.append(f"split '{split_name}' está vazio")
            continue

        if n < warn_below:
            logger.warning(
                "validate_split_data: split '%s' tem só %d linhas (< %d)",
                split_name,
                n,
                warn_below,
            )

        if stratify_column and stratify_column in split_df.columns:
            value_counts = split_df[stratify_column].value_counts(normalize=True)
            minority = float(value_counts.min())
            if minority < min_minority:
                logger.warning(
                    "validate_split_data: split '%s' classe minoritária %.1f%% (< %.0f%%)",
                    split_name,
                    minority * 100,
                    min_minority * 100,
                )

    if issues:
        raise ValueError(
            "validate_split_data: " + "; ".join(issues),
        )

    logger.info("validate_split_data: splits OK")
    return split_data
