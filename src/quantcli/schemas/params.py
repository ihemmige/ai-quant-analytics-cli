from pydantic import BaseModel, Field


class Params(BaseModel):
    window: int | None = Field(
        default=None,
        gt=0,
        le=5000,
        description="Number of return observations to use for metrics "
        "that require a window (e.g. realized volatility).",
    )

    annualization_factor: int = Field(
        default=252,
        gt=0,
        description="Factor used to annualize volatility-based metrics.",
    )

    risk_free_rate: float = Field(
        default=0.0,
        description=(
            "Annualized continuously-compounded (log) risk-free rate used for "
            "Sharpe ratio calculation. Expressed as a decimal "
            "(e.g. 0.01 for 1% annualized)."
        ),
    )
