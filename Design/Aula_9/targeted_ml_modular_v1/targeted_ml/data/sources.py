from __future__ import annotations

from pathlib import Path


def resolve_dataset_root(dataset_root: Path) -> Path:
    root = dataset_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"dataset root not found: {root}")
    return root
