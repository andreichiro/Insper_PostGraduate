"""Lazy orchestration exports to preserve package API without import cycles."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from targeted_ml.config.models import AnalysisSpec
    from targeted_ml.orchestration.artifacts import ProjectPaths


def resolve_paths(project_root: Path, output_root: Path | None = None) -> "ProjectPaths":
    from .pipeline import resolve_paths as _resolve_paths

    return _resolve_paths(project_root, output_root)


def build_modelled(spec: "AnalysisSpec", paths: "ProjectPaths") -> Path:
    from .pipeline import build_modelled as _build_modelled

    return _build_modelled(spec, paths)


def build_ml(spec: "AnalysisSpec", paths: "ProjectPaths") -> Path:
    from .pipeline import build_ml as _build_ml

    return _build_ml(spec, paths)


def build_html(spec: "AnalysisSpec", paths: "ProjectPaths") -> Path:
    from .pipeline import build_html as _build_html

    return _build_html(spec, paths)


def build_all(
    spec: "AnalysisSpec",
    project_root: Path,
    output_root: Path | None = None,
    skip_modelled: bool = False,
    skip_post_model_refit: bool = False,
) -> "ProjectPaths":
    from .pipeline import build_all as _build_all

    return _build_all(spec, project_root, output_root, skip_modelled, skip_post_model_refit)


__all__ = ["resolve_paths", "build_modelled", "build_ml", "build_html", "build_all"]
