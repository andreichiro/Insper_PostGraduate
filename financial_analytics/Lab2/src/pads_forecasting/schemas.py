"""Pydantic schemas for YAML-driven configuration."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProjectConfig(BaseModel):
    name: str
    run_id: str
    seed: int = 42


class DataConfig(BaseModel):
    date_col: str = "data"
    value_col: str = "valor"
    freq: str = "MS"
    acquisition_date: date
    final_forecast_start: date
    horizon: int = Field(gt=0)
    expected_main_start: date
    expected_main_end: date
    expected_main_rows: int = Field(gt=0)
    expected_acquired_start: date
    expected_acquired_end: date
    expected_acquired_rows: int = Field(gt=0)


class CovidConfig(BaseModel):
    enabled: bool = True
    shock_window: tuple[date, date]
    recovery_window: tuple[date, date]
    aftershock_window: tuple[date, date] | None = None
    future_value: int = 0


class InterventionsConfig(BaseModel):
    covid: CovidConfig


class ReconstructionConfig(BaseModel):
    alpha_grid: list[float]
    alpha_penalty_lambda: float = Field(ge=0)
    alpha_prefer_one_margin_pct: float = Field(ge=0)
    beta_grid: list[float]

    @field_validator("alpha_grid", "beta_grid")
    @classmethod
    def _non_empty_grid(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("Grid must not be empty.")
        return value


class FoldConfig(BaseModel):
    name: str
    train_end: date
    valid_start: date
    valid_end: date
    role: Literal["stress", "normal"]

    @field_validator("valid_end")
    @classmethod
    def _valid_end_after_start(cls, value: date, info: Any) -> date:
        valid_start = info.data.get("valid_start")
        if valid_start and value < valid_start:
            raise ValueError("valid_end must be >= valid_start.")
        return value


class ValidationConfig(BaseModel):
    horizon: int = Field(gt=0)
    season_length: int = Field(gt=0)
    folds: list[FoldConfig]
    common_mase_reference_strategy: str = "observed_post_merger"
    mase_uncertainty_bootstrap_samples: int = Field(default=1000, gt=0)
    robust_alpha: dict[str, Any] | None = None
    covid_adjustment: dict[str, Any] | None = None
    robustness_top_n: int = Field(default=3, gt=0)
    robustness_rolling_origins: dict[str, Any] | None = None


class SelectionConfig(BaseModel):
    primary_metric: str = "normal_mean_common_mase"
    normal_fold_roles: list[str]
    old_data_min_improvement_pct: float = Field(default=0.0, ge=0)
    complexity_min_improvement_pct: float | None = Field(default=None, ge=0)
    max_cv_mase: float | None = Field(default=None, gt=0)
    train_valid_ratio_warning: float = Field(gt=0)
    train_valid_ratio_reject: float = Field(gt=0)
    residual_diagnostics_max_ratio: float = Field(gt=0)
    prefer_alpha_1_if_margin_below_pct: float | None = Field(default=None, ge=0)
    stability_iqr_multiplier: float = Field(default=1.5, gt=0)
    stability_min_fold_variation_epsilon: float = Field(default=1e-12, ge=0)


class MetricsConfig(BaseModel):
    primary: str
    reported: list[str]

    @field_validator("reported")
    @classmethod
    def _reported_metrics_non_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("At least one reported metric is required.")
        return value


class ModelFamilyConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    optional: bool = False
    season_length: int | None = Field(default=None, gt=0)
    covid_modes: list[str] | None = None
    grid: dict[str, list[Any]] | None = None

    @field_validator("covid_modes")
    @classmethod
    def _covid_modes_non_empty(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and not value:
            raise ValueError("covid_modes must not be empty when provided.")
        return value

    @field_validator("grid")
    @classmethod
    def _grid_values_non_empty(
        cls, value: dict[str, list[Any]] | None
    ) -> dict[str, list[Any]] | None:
        if value is None:
            return value
        empty_keys = [key for key, entries in value.items() if not entries]
        if empty_keys:
            raise ValueError(f"Model grid entries must not be empty: {empty_keys}")
        return value


class ModelsConfig(BaseModel):
    seasonal_naive: ModelFamilyConfig
    ets: ModelFamilyConfig
    sarimax: ModelFamilyConfig
    prophet: ModelFamilyConfig
    lightgbm: ModelFamilyConfig
    ridge: ModelFamilyConfig
    elasticnet: ModelFamilyConfig | None = None
    bvar: ModelFamilyConfig


class HpoConfig(BaseModel):
    enabled: bool
    optimizer: Literal["grid", "optuna"]
    objective: str


class OutputsConfig(BaseModel):
    reporting_dir: str
    figures_dir: str
    previsao_path: str
    forecast_intervals_path: str
    mlruns_dir: str
    html_report_path: str
    decision_html_report_path: str


def validate_parameter_groups(
    project: dict[str, Any],
    data: dict[str, Any],
    interventions: dict[str, Any],
    reconstruction: dict[str, Any],
    validation: dict[str, Any],
    selection: dict[str, Any],
    outputs: dict[str, Any],
    models: dict[str, Any],
    hpo: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    """Validate top-level parameter groups used by the pipelines."""

    ProjectConfig.model_validate(project)
    DataConfig.model_validate(data)
    InterventionsConfig.model_validate(interventions)
    ReconstructionConfig.model_validate(reconstruction)
    ValidationConfig.model_validate(validation)
    SelectionConfig.model_validate(selection)
    OutputsConfig.model_validate(outputs)
    ModelsConfig.model_validate(models)
    HpoConfig.model_validate(hpo)
    MetricsConfig.model_validate(metrics)
