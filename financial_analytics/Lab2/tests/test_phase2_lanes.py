from pathlib import Path

import pandas as pd
import yaml

from pads_forecasting.pipelines.data_engineering.nodes import build_canonical_panel
from pads_forecasting.pipelines.eda.nodes import generate_eda_outputs
from pads_forecasting.pipelines.reconstruction.nodes import build_target_strategies

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "conf/base"


def _load_yaml(name: str) -> dict:
    with (CONF / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _phase2_inputs(tmp_path: Path):
    base = _load_yaml("parameters.yml")
    models = _load_yaml("parameters_models.yml")
    validation = _load_yaml("parameters_validation.yml")
    outputs = {
        **_load_yaml("parameters_outputs.yml")["outputs"],
        "reporting_dir": str(tmp_path),
        "figures_dir": str(tmp_path / "figures"),
    }
    main_raw = pd.read_csv(ROOT / "data/01_raw/distribr_serie.txt")
    acquired_raw = pd.read_csv(ROOT / "data/01_raw/distribr_adquirida.txt")
    panel, data_validation = build_canonical_panel(
        main_raw,
        acquired_raw,
        base["project"],
        base["data"],
        base["interventions"],
        base["reconstruction"],
        validation["validation"],
        validation["selection"],
        outputs,
        models["models"],
        models["hpo"],
        validation["metrics"],
    )
    return base, validation, outputs, panel, data_validation


def test_phase2_track_a_builds_canonical_panel_and_covid_flags(tmp_path):
    base, _, _, panel, data_validation = _phase2_inputs(tmp_path)

    expected_columns = {
        "data",
        "br_publicado",
        "adquirida_separada",
        "consolidado_observado",
        "is_pre_acquisition",
        "is_post_acquisition",
        "target_source",
        "month",
        "trend_index",
        "covid_shock",
        "covid_recovery",
    }
    acquisition = pd.Timestamp(base["data"]["acquisition_date"])

    assert expected_columns.issubset(panel.columns)
    assert len(panel) == 120
    assert panel["covid_shock"].sum() == 4
    assert panel["covid_recovery"].sum() == 6
    assert panel.loc[panel["data"] >= acquisition, "adquirida_separada"].isna().all()
    assert data_validation["passed"].all()


def test_phase2_track_b_generates_eda_tables_and_required_figures(tmp_path):
    base, validation, outputs, panel, _ = _phase2_inputs(tmp_path)
    target_strategies, *_ = build_target_strategies(
        panel,
        base["data"],
        base["reconstruction"],
        validation["validation"],
        outputs,
    )

    eda_summary, stationarity_tests = generate_eda_outputs(
        panel,
        target_strategies,
        base["data"],
        outputs,
    )

    figures = Path(outputs["figures_dir"])
    expected_figures = {
        "series_acquisition_covid.png",
        "acquired_company_series.png",
        "target_reconstruction_overlay.png",
        "seasonality_month_profile.png",
        "decomposition.png",
        "outliers_covid.png",
    }

    assert expected_figures.issubset({path.name for path in figures.glob("*.png")})
    assert {"main_rows", "covid_shock_months", "covid_recovery_months"}.issubset(
        set(eda_summary["item"])
    )
    assert set(stationarity_tests["series"]) == {"raw_full", "post_only", "proforma_sum"}
    assert set(stationarity_tests["test"]) == {"ADF", "KPSS"}
    assert len(stationarity_tests) == 6


def test_phase2_track_c_builds_all_reconstruction_candidates(tmp_path):
    base, validation, outputs, panel, _ = _phase2_inputs(tmp_path)
    target_strategies, summary, alpha_sensitivity, leave_one_fold_alpha = build_target_strategies(
        panel,
        base["data"],
        base["reconstruction"],
        validation["validation"],
        outputs,
    )

    strategies = target_strategies["strategies"]
    two_weight_candidates = target_strategies["two_weight_candidates"]
    expected_two_weight_count = len(base["reconstruction"]["alpha_grid"]) * len(
        base["reconstruction"]["beta_grid"]
    )
    acquisition = pd.Timestamp(base["data"]["acquisition_date"])

    assert set(strategies) == {"raw_full", "post_only", "proforma_sum", "calibrated_alpha"}
    assert len(target_strategies["alpha_candidates"]) == len(base["reconstruction"]["alpha_grid"])
    assert len(two_weight_candidates) == expected_two_weight_count
    assert (
        summary["target_strategy"].eq("two_weight_sensitivity").sum() == expected_two_weight_count
    )
    assert not alpha_sensitivity.empty
    assert len(leave_one_fold_alpha) == len(validation["validation"]["folds"])
    assert strategies["post_only"]["data"].min() == acquisition
    assert {
        "br_component_observed",
        "acquired_component_observed",
        "consolidated_observed",
    }.issubset(strategies["proforma_sum"].columns)
    assert (
        strategies["proforma_sum"]
        .loc[strategies["proforma_sum"]["data"] < acquisition, "br_component_observed"]
        .notna()
        .all()
    )
    assert (
        strategies["proforma_sum"]
        .loc[strategies["proforma_sum"]["data"] >= acquisition, "consolidated_observed"]
        .notna()
        .all()
    )
    for name in ["raw_full", "proforma_sum", "calibrated_alpha"]:
        pre_rows = strategies[name][strategies[name]["data"] < acquisition].copy()
        component_sum = pre_rows["br_component_observed"].astype(float) + pre_rows[
            "acquired_component_observed"
        ].astype(float)
        pd.testing.assert_series_equal(
            component_sum.reset_index(drop=True),
            pre_rows["y"].astype(float).reset_index(drop=True),
            check_names=False,
        )

    post_proforma = strategies["proforma_sum"][strategies["proforma_sum"]["data"] >= acquisition]
    post_raw = strategies["raw_full"][strategies["raw_full"]["data"] >= acquisition]
    pd.testing.assert_series_equal(
        post_proforma["y"].reset_index(drop=True),
        post_raw["y"].reset_index(drop=True),
        check_names=False,
    )
