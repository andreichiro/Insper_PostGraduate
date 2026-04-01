from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DefinitionStrategySpec:
    strategy_name: str
    metric_names: list[str] = field(default_factory=list)
    external_validators: list[str] = field(default_factory=list)
