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
