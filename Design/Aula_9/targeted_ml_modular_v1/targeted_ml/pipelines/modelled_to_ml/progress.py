from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter


@dataclass(frozen=True)
class ProgressStageSpec:
    key: str
    label: str
    weight: float


class BuildProgressTracker:
    def __init__(
        self,
        stage_specs: list[ProgressStageSpec],
        reference_total_minutes: float | None = None,
        bar_width: int = 24,
    ) -> None:
        self.stage_specs = stage_specs
        self.reference_total_minutes = reference_total_minutes
        self.bar_width = max(8, bar_width)
        self.started_at = perf_counter()
        self.stage_progress = {spec.key: 0.0 for spec in stage_specs}
        self.stage_lookup = {spec.key: spec for spec in stage_specs}
        self.total_weight = sum(max(spec.weight, 0.0) for spec in stage_specs) or 1.0

    def start_stage(self, stage_key: str, detail: str = "", total: int | None = None) -> None:
        self._emit(stage_key, detail=detail or "iniciado", current=0 if total else None, total=total)

    def update_stage(
        self,
        stage_key: str,
        current: int,
        total: int,
        detail: str = "",
    ) -> None:
        progress = 1.0 if total <= 0 else min(max(current / total, 0.0), 1.0)
        self.stage_progress[stage_key] = progress
        self._emit(stage_key, detail=detail, current=current, total=total)

    def complete_stage(self, stage_key: str, detail: str = "") -> None:
        self.stage_progress[stage_key] = 1.0
        self._emit(stage_key, detail=detail or "concluído")

    def _overall_percent(self) -> float:
        weighted = 0.0
        for spec in self.stage_specs:
            weighted += spec.weight * self.stage_progress.get(spec.key, 0.0)
        return max(0.0, min(100.0, 100.0 * weighted / self.total_weight))

    def _render_bar(self, percent: float) -> str:
        filled = int(round((percent / 100.0) * self.bar_width))
        filled = min(max(filled, 0), self.bar_width)
        return "[" + ("#" * filled) + ("-" * (self.bar_width - filled)) + "]"

    def _emit(self, stage_key: str, detail: str = "", current: int | None = None, total: int | None = None) -> None:
        spec = self.stage_lookup[stage_key]
        percent = self._overall_percent()
        bar = self._render_bar(percent)
        elapsed_minutes = (perf_counter() - self.started_at) / 60.0
        eta_suffix = ""
        if self.reference_total_minutes is not None:
            remaining = max(0.0, self.reference_total_minutes * (1.0 - percent / 100.0))
            eta_suffix = f" | eta_ref={remaining:0.1f}m"
        unit_suffix = ""
        if current is not None and total is not None:
            unit_suffix = f" | {current}/{total}"
        detail_suffix = f" | {detail}" if detail else ""
        print(
            f"[progress] {percent:05.1f}% {bar} | {spec.label}{unit_suffix} | elapsed={elapsed_minutes:0.1f}m{eta_suffix}{detail_suffix}",
            flush=True,
        )
