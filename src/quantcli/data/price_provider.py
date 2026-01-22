from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class PriceProviderError(RuntimeError):
    pass


class PriceProvider(Protocol):
    def get_adjusted_close(
        self,
        ticker: str,
        n_days: int,  # trading days
    ) -> NDArray[np.float64]:
        """Return adjusted close prices as np.ndarray[float64], oldest->newest.
        Raises PriceProviderError on failure.
        """
        ...

    def name(self) -> str:
        """Return a human-readable provider name."""
        ...
