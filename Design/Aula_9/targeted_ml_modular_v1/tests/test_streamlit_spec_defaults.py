from __future__ import annotations

from pathlib import Path

from targeted_ml.apps.streamlit_app import _default_spec_index, _list_runnable_specs, _spec_label
from targeted_ml.config.loader import render_resolved_spec_yaml


def test_list_runnable_specs_excludes_base() -> None:
    names = [path.name for path in _list_runnable_specs()]
    assert "base.yaml" not in names
    assert "activity.yaml" in names
    assert "churn_m1.yaml" in names
    assert "return_m1.yaml" in names


def test_default_spec_prefers_activity() -> None:
    specs = _list_runnable_specs()
    default_path = specs[_default_spec_index(specs)]
    assert default_path.name == "activity.yaml"


def test_spec_labels_are_user_friendly() -> None:
    assert _spec_label(Path("activity.yaml")) == "Atividade (principal)"
    assert _spec_label(Path("churn_m1.yaml")) == "Churn M1"
    assert _spec_label(Path("return_m1.yaml")) == "Retorno M1"


def test_resolved_activity_yaml_is_not_sparse() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    yaml_text = render_resolved_spec_yaml(repo_root / "specs" / "activity.yaml")
    assert "defaults:" not in yaml_text
    assert "analysis_name: targeted_ml_activity" in yaml_text
    assert "analysis_kind: activity" in yaml_text
    assert "dataset_root: data" in yaml_text
    assert "model_families:" in yaml_text
