# quantcli/data/yfinance_price_provider.py
import contextlib
import io

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from quantcli.data.price_provider import PriceProvider, PriceProviderError


class YFinancePriceProvider(PriceProvider):
    def name(self) -> str:
        return "yfinance"

    def get_adjusted_close(self, ticker: str, n_days: int) -> NDArray[np.float64]:
        try:
            import yfinance as yf

            # yfinance/Yahoo prints warnings/errors; never leak to CLI stdout/stderr
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                df = yf.Ticker(ticker).history(
                    period=f"{n_days}d",
                    auto_adjust=True,  # adjusted close
                )
        except Exception as e:
            raise PriceProviderError("yfinance history failed") from e

        if df is None or getattr(df, "empty", True):
            raise PriceProviderError("no price data")

        if "Close" not in df.columns:
            raise PriceProviderError("missing close column")

        prices = pd.to_numeric(df["Close"], errors="coerce").dropna()

        if len(prices) < 2:
            raise PriceProviderError("insufficient price points")

        arr = np.asarray(prices.values, dtype=np.float64)

        if not np.isfinite(arr).all():
            raise PriceProviderError("non finite prices")

        if (arr <= 0).any():
            raise PriceProviderError("non positive prices")

        return arr
