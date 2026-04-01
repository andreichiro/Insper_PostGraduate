from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from targeted_ml.config.models import AnalysisSpec


def _resolve_defaults(spec_path: Path) -> list[Path]:
    raw = OmegaConf.load(spec_path)
    defaults = raw.get("defaults", [])
    resolved: list[Path] = []
    for item in defaults:
        if isinstance(item, str):
            candidate = spec_path.parent / f"{item}.yaml"
        elif isinstance(item, dict):
            _, value = next(iter(item.items()))
            candidate = spec_path.parent / f"{value}.yaml"
        else:
            continue
        resolved.append(candidate.resolve())
    return resolved


def load_analysis_config(spec_path: Path, overrides: list[str] | None = None):
    spec_path = spec_path.resolve()
    cfgs = [OmegaConf.load(path) for path in _resolve_defaults(spec_path)]
    cfgs.append(OmegaConf.load(spec_path))
    merged = OmegaConf.merge(*cfgs)
    if "defaults" in merged:
        del merged["defaults"]
    if overrides:
        merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(overrides))
    return merged


def render_resolved_spec_yaml(spec_path: Path, overrides: list[str] | None = None) -> str:
    merged = load_analysis_config(spec_path, overrides=overrides)
    return OmegaConf.to_yaml(merged, resolve=True, sort_keys=False)


def load_analysis_spec(spec_path: Path, overrides: list[str] | None = None) -> AnalysisSpec:
    merged = load_analysis_config(spec_path, overrides=overrides)
    payload = OmegaConf.to_container(merged, resolve=True)
    return AnalysisSpec.model_validate(payload)
