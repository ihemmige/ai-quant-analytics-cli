from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from quantcli.data.price_provider import PriceProvider, PriceProviderError

FixtureName = Literal["monotonic_up", "drawdown", "short_1", "invalid_non_positive"]


@dataclass
class FakePriceProvider(PriceProvider):
    fixture: FixtureName = "monotonic_up"
    fail: bool = False
    calls: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_fixtures",
            {
                "monotonic_up": self._monotonic_up,
                "drawdown": self._drawdown,
                "short_2": self._short_2,
                "invalid_non_positive": self._invalid_non_positive,
            },
        )

    def name(self) -> str:
        return "FakePriceProvider"

    def get_adjusted_close(self, ticker: str, n_days: int) -> np.ndarray:
        self.calls += 1
        if n_days < 0:
            raise ValueError("n_days must be >= 0")
        if self.fail:
            raise PriceProviderError("Injected provider failure for testing.")
        arr = self._fixtures[self.fixture](ticker, n_days)
        return np.array(arr, dtype=np.float64)

    def _monotonic_up(self, ticker: str, n_days: int) -> np.ndarray:
        base = 100.0
        return np.array([base + float(i) for i in range(n_days)], dtype=np.float64)

    def _drawdown(self, ticker: str, n_days: int) -> np.ndarray:
        if n_days == 0:
            return np.array([], dtype=np.float64)
        if n_days == 1:
            return np.array([100.0], dtype=np.float64)

        peak = 120.0
        trough = 80.0
        peak_idx = n_days // 2

        out: list[float] = []
        for i in range(n_days):
            # n_days >= 2 here, so denominators are safe
            if i <= peak_idx:
                val = 100.0 + (peak - 100.0) * (i / peak_idx)
            else:
                val = peak + (trough - peak) * (
                    (i - peak_idx) / (n_days - 1 - peak_idx)
                )
            out.append(float(val))
        return np.array(out, dtype=np.float64)

    def _short_2(self, ticker: str, n_days: int) -> np.ndarray:
        return np.array([100.0, 101.0], dtype=np.float64)

    def _invalid_non_positive(self, ticker: str, n_days: int) -> np.ndarray:
        return np.array([0.0 for _ in range(n_days)], dtype=np.float64)
