from pydantic import BaseModel, Field


class Params(BaseModel):
    window: int | None = Field(
        default=None,
        gt=0,
        le=5000,
        description="Number of return observations to use for metrics that require a window (e.g. realized volatility).",
    )

    annualization_factor: int = Field(
        default=252,
        gt=0,
        description="Factor used to annualize volatility-based metrics.",
    )
