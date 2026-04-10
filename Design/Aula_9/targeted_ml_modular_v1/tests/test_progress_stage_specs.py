from __future__ import annotations

import inspect
import re

from targeted_ml.pipelines.modelled_to_ml import analysis_setup as setup
from targeted_ml.pipelines.modelled_to_ml import runner


def test_runner_progress_stage_keys_are_declared_in_stage_specs() -> None:
    runner_source = inspect.getsource(runner)
    used_stage_keys = set(
        re.findall(r'progress\.(?:start_stage|complete_stage|update_stage)\("([^"]+)"', runner_source)
    )
    declared_stage_keys = {row["key"] for row in setup.BUILD_PROGRESS_STAGE_SPECS}
    assert used_stage_keys <= declared_stage_keys
