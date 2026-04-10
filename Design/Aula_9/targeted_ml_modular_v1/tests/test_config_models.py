from __future__ import annotations

import pytest

from targeted_ml.config.models import AnalysisSpec


def test_definition_a_strategy_normalizes_univariate_alias() -> None:
    spec = AnalysisSpec.model_validate(
        {
            "analysis_name": "test_activity",
            "analysis_kind": "activity",
            "data": {"dataset_root": "."},
            "label": {"definition_a": {"strategy": "univariate"}},
        }
    )
    assert spec.label.definition_a.strategy == "univariate_exact"


def test_definition_a_rejects_removed_combinatorial_strategy() -> None:
    with pytest.raises(ValueError, match="definition_a.strategy"):
        AnalysisSpec.model_validate(
            {
                "analysis_name": "test_activity",
                "analysis_kind": "activity",
                "data": {"dataset_root": "."},
                "label": {"definition_a": {"strategy": "combinatorial_beam"}},
            }
        )


def test_definition_a_accepts_screened_pairwise_compound_weighted_strategy() -> None:
    spec = AnalysisSpec.model_validate(
        {
            "analysis_name": "test_activity",
            "analysis_kind": "activity",
            "data": {"dataset_root": "."},
            "label": {"definition_a": {"strategy": "screened_pairwise_compound_weighted"}},
        }
    )
    assert spec.label.definition_a.strategy == "screened_pairwise_compound_weighted"


def test_data_modeled_source_supports_raw() -> None:
    spec = AnalysisSpec.model_validate(
        {
            "analysis_name": "test_activity",
            "analysis_kind": "activity",
            "data": {"dataset_root": ".", "modeled_source": "raw"},
            "label": {},
        }
    )
    assert spec.data.modeled_source == "raw"


def test_population_filter_supports_same_month_entry_only() -> None:
    spec = AnalysisSpec.model_validate(
        {
            "analysis_name": "test_activity",
            "analysis_kind": "activity",
            "data": {"dataset_root": "."},
            "population": {"official_population": "same_month_entry_only"},
            "label": {},
        }
    )
    assert spec.population.official_population == "same_month_entry_only"


def test_modeling_supports_definition_selection_holdout_months() -> None:
    spec = AnalysisSpec.model_validate(
        {
            "analysis_name": "test_activity",
            "analysis_kind": "activity",
            "data": {"dataset_root": "."},
            "label": {},
            "modeling": {"definition_selection_holdout_months": 6},
        }
    )
    assert spec.modeling.definition_selection_holdout_months == 6


def test_definition_a_supports_promoted_candidate_limit() -> None:
    spec = AnalysisSpec.model_validate(
        {
            "analysis_name": "test_activity",
            "analysis_kind": "activity",
            "data": {"dataset_root": "."},
            "label": {"definition_a": {"promoted_candidate_limit": 3}},
        }
    )
    assert spec.label.definition_a.promoted_candidate_limit == 3


def test_modeling_supports_six_workers() -> None:
    spec = AnalysisSpec.model_validate(
        {
            "analysis_name": "test_activity",
            "analysis_kind": "activity",
            "data": {"dataset_root": "."},
            "label": {},
            "modeling": {"workers": 6},
        }
    )
    assert spec.modeling.workers == 6


def test_modeling_supports_configurable_definition_lock_bootstrap_gate() -> None:
    spec = AnalysisSpec.model_validate(
        {
            "analysis_name": "test_activity",
            "analysis_kind": "activity",
            "data": {"dataset_root": "."},
            "label": {},
            "modeling": {
                "definition_lock_bootstrap_gate": {
                    "column_name": "lock_gap_sustained_active_2of3_post_label_ci_width",
                    "operator": "<",
                    "threshold": 0.20,
                }
            },
        }
    )
    assert spec.modeling.definition_lock_bootstrap_gate.column_name == "lock_gap_sustained_active_2of3_post_label_ci_width"
    assert spec.modeling.definition_lock_bootstrap_gate.operator == "<"
    assert spec.modeling.definition_lock_bootstrap_gate.threshold == 0.20
