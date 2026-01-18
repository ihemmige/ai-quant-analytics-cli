from pydantic import BaseModel, Field


class TimeRange(BaseModel):
    n_days: int = Field(
        ...,
        gt=0,
        le=5000,  # ~20 years of trading days
        description="Number of most recent trading days to include",
    )
