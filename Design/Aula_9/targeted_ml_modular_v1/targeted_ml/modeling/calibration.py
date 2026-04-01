from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV


@dataclass(frozen=True)
class TemporalCalibrationSpec:
    """Política de calibração temporal do build oficial.

    O outer fold continua sendo a fronteira real de teste.
    Dentro do treino do outer fold, o calibrador usa um bloco temporal final.
    """

    method: str = "sigmoid"
    requires_two_classes: bool = True


def build_temporal_calibration_holdout(
    train: pd.DataFrame,
    month_col: str,
    target_col: str,
) -> tuple[np.ndarray | None, np.ndarray | None, pd.DataFrame]:
    """Separa um bloco temporal final do treino para calibração.

    Regra prática:
    - o modelo aprende no bloco mais antigo
    - a calibração usa o bloco mais recente ainda dentro do treino
    - o teste externo não entra em nenhum dos dois
    """

    months = pd.to_datetime(train[month_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    unique_months = np.array(sorted(months.dropna().unique()))
    audit_rows: list[dict[str, Any]] = []
    if len(unique_months) < 2:
        audit_rows.append(
            {
                "inner_fold_id": 1,
                "split_strategy": "temporal_calibration_holdout",
                "train_rows": int(len(train)),
                "test_rows": 0,
                "train_positives": int(pd.to_numeric(train[target_col], errors="coerce").fillna(0).sum()),
                "test_positives": 0,
                "valid_inner_split_flag": 0,
                "invalid_reason": "not_enough_months_for_calibration_holdout",
            }
        )
        return None, None, pd.DataFrame(audit_rows)

    for start_idx in range(len(unique_months) - 1, 0, -1):
        fit_months = unique_months[:start_idx]
        calibration_months = unique_months[start_idx:]
        fit_idx = np.flatnonzero(months.isin(fit_months).to_numpy())
        calibration_idx = np.flatnonzero(months.isin(calibration_months).to_numpy())
        y_fit = train.iloc[fit_idx][target_col]
        y_calibration = train.iloc[calibration_idx][target_col]
        invalid_reason = ""
        if y_fit.nunique() < 2:
            invalid_reason = "calibration_fit_single_class"
        elif y_calibration.nunique() < 2:
            invalid_reason = "calibration_holdout_single_class"
        audit_rows.append(
            {
                "inner_fold_id": int(len(audit_rows) + 1),
                "split_strategy": "temporal_calibration_holdout",
                "calibration_start_month": pd.Timestamp(calibration_months[0]).strftime("%Y-%m-%d"),
                "calibration_month_count": int(len(calibration_months)),
                "train_rows": int(len(fit_idx)),
                "test_rows": int(len(calibration_idx)),
                "train_positives": int(y_fit.sum()),
                "test_positives": int(y_calibration.sum()),
                "valid_inner_split_flag": int(not invalid_reason),
                "invalid_reason": invalid_reason,
            }
        )
        if not invalid_reason:
            return fit_idx, calibration_idx, pd.DataFrame(audit_rows)
    return None, None, pd.DataFrame(audit_rows)


def build_temporal_calibrator(
    estimator,
    train: pd.DataFrame,
    target_col: str,
    fit_idx: np.ndarray,
    calibration_idx: np.ndarray,
    method: str = "sigmoid",
):
    """Monta um calibrador 100% library-native sem usar `prefit`.

    A sacada aqui é passar um split temporal explícito para o próprio
    ``CalibratedClassifierCV``. Assim, o sklearn faz o fit/calibration no fluxo
    esperado, sem `FrozenEstimator` e sem `cv='prefit'`.
    """

    cv = [(fit_idx, calibration_idx)]
    calibrator = CalibratedClassifierCV(
        estimator=estimator,
        method=method,
        cv=cv,
        ensemble=True,
    )
    calibrator.fit(train, train[target_col].to_numpy())
    return calibrator
