from typing import Protocol, Sequence
import numpy as np


class PriceProviderError(RuntimeError):
    pass


class PriceProvider(Protocol):
    def get_adjusted_close(
        self,
        ticker: str,
        n_days: int,  # trading days
    ) -> np.ndarray:
        """Return adjusted close prices as np.ndarray[float64], oldest->newest.
        Raises PriceProviderError on failure.
        """
        ...

    def name(self) -> str:
        """Return a human-readable provider name."""
        ...
