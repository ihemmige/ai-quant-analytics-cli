from quantcli.schemas import Intent, Result, Refusal, ToolName, Params
import quantcli.tools as tools
from quantcli.data import PriceProvider, PriceProviderError
from typing import Callable, Sequence
from quantcli.tools import total_return, max_drawdown, realized_volatility
from __future__ import annotations

MetricFn = Callable[[Sequence[float], Params], float]

METRICS: dict[ToolName, MetricFn] = {
    ToolName.total_return: total_return,
    ToolName.max_drawdown: max_drawdown,
    ToolName.realized_volatility: realized_volatility,
}


def run(intent: Intent, provider: PriceProvider) -> Result | Refusal:
    validated_intent: Intent | Refusal = tools.validate_intent(intent)
    if isinstance(validated_intent, Refusal):
        return validated_intent
    try:
        prices = provider.get_prices(
            ticker=validated_intent.tickers[0],
            n_days=validated_intent.time_range.n_days,
        )
    except PriceProviderError as e:
        return Refusal(reason=f"Failed to fetch price data: {str(e)}")

    metric_fn = METRICS[validated_intent.tool]
    if not metric_fn:
        return Refusal(
            reason=f"Unsupported tool: {validated_intent.tool}",
            clarifying_question=None,
            allowed_capabilities=list(ToolName),
        )

    ret_value = metric_fn(prices, validated_intent.params)

    return Result(
        tool=validated_intent.tool,
        tickers=validated_intent.tickers,
        value=ret_value,
        metadata={
            "range_n_days": validated_intent.time_range.n_days,
            "window": validated_intent.params.window,
            "annualization_factor": validated_intent.params.annualization_factor,
            "data_points": len(prices),
            "price_source": provider.name(),
            "tool_version": "1.0.0",  # TODO
            "interpretation_notes": None,  # TODO
        },
    )
