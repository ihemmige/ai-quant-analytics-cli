from typing import Protocol, Sequence


class PriceProvider(Protocol):
    def get_adjusted_close(
        self,
        ticker: str,
        n_days: int,  # trading days
    ) -> Sequence[float]: ...

    def name(self) -> str:
        """Return a human-readable provider name."""
        ...
