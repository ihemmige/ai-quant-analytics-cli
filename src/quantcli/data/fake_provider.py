from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Literal, Sequence

from .price_provider import PriceProvider


class PriceProviderError(RuntimeError):
    pass


FixtureName = Literal["monotonic_up", "drawdown", "short_1", "invalid_non_positive"]


@dataclass(frozen=True)
class FakePriceProvider(PriceProvider):
    fixture: FixtureName = "monotonic_up"
    fail: bool = False

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

    def get_adjusted_close(self, ticker: str, n_days: int) -> Sequence[float]:
        if n_days < 0:
            raise ValueError("n_days must be >= 0")
        if self.fail:
            raise PriceProviderError("Injected provider failure for testing.")
        return self._fixtures[self.fixture](ticker, n_days)

    def _monotonic_up(self, ticker: str, n_days: int) -> Sequence[float]:
        base = 100.0
        return [base + float(i) for i in range(n_days)]

    def _drawdown(self, ticker: str, n_days: int) -> Sequence[float]:
        if n_days == 0:
            return []
        if n_days == 1:
            return [100.0]

        peak = 120.0
        trough = 80.0
        peak_idx = n_days // 2

        out: list[float] = []
        for i in range(n_days):
            # n_days >= 2 here, so denominators are safe
            if i <= peak_idx:
                val = 100.0 + (peak - 100.0) * (i / peak_idx)
            else:
                val = peak + (trough - peak) * ((i - peak_idx) / (n_days - 1 - peak_idx))
            out.append(float(val))
        return out

    def _short_2(self, ticker: str, n_days: int) -> Sequence[float]:
        return [100.0, 101.0]

    def _invalid_non_positive(self, ticker: str, n_days: int) -> Sequence[float]:
        return [0.0 for _ in range(n_days)]
