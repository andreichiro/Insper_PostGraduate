from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


@dataclass
class ExpandingMonthSplit(BaseCrossValidator):
    """Splitter temporal por mês no estilo expanding window.

    É pequeno e manual porque o caso é painel com muitas linhas por mês.
    O sklearn puro com ``TimeSeriesSplit`` trabalha sobre ordem amostral e não
    garante respeito às bordas de mês do jeito que esta análise precisa.

    Em linguagem simples:
    - treino: meses acumulados até aqui
    - teste: próximo mês
    """

    month_col: str = "first_month"
    min_train_periods: int = 1
    test_periods: int = 1
    max_splits: int | None = 5

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: N803
        months = self._unique_months(X=X, groups=groups)
        usable = max(0, len(months) - self.min_train_periods)
        if self.max_splits is None:
            return usable
        return min(self.max_splits, usable)

    def split(self, X, y=None, groups=None):  # noqa: N803
        months = self._unique_months(X=X, groups=groups)
        if len(months) <= self.min_train_periods:
            return

        start_fold = self.min_train_periods
        stop_fold = len(months) - self.test_periods + 1
        candidate_starts = list(range(start_fold, stop_fold))
        if self.max_splits is not None:
            candidate_starts = candidate_starts[-self.max_splits :]

        month_series = self._month_series(X=X, groups=groups)
        for end_train in candidate_starts:
            train_months = months[:end_train]
            test_months = months[end_train : end_train + self.test_periods]
            if len(test_months) < self.test_periods:
                continue
            train_mask = month_series.isin(train_months)
            test_mask = month_series.isin(test_months)
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue
            yield np.flatnonzero(train_mask.to_numpy()), np.flatnonzero(test_mask.to_numpy())

    def _month_series(self, X, groups=None) -> pd.Series:
        if groups is not None:
            series = pd.Series(groups)
        elif isinstance(X, pd.DataFrame):
            if self.month_col not in X.columns:
                raise ValueError(f"ExpandingMonthSplit requires month column '{self.month_col}' in X.")
            series = X[self.month_col]
        elif isinstance(X, pd.Series):
            series = X
        else:
            raise TypeError("ExpandingMonthSplit expects a pandas DataFrame, Series, or groups array.")

        return pd.to_datetime(series, errors="coerce").dt.to_period("M").dt.to_timestamp()

    def _unique_months(self, X, groups=None) -> list[pd.Timestamp]:
        series = self._month_series(X=X, groups=groups)
        return sorted(series.dropna().unique().tolist())
