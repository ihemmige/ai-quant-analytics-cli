from __future__ import annotations
from quantcli.schemas import Intent, Result, Refusal, ToolName, Params
from quantcli.data import PriceProvider, PriceProviderError
from typing import Callable, Sequence
from quantcli.tools import total_return, max_drawdown, realized_volatility
from quantcli.validate_intent import validate_intent
from quantcli.router.llm_client import LLMClient
from quantcli.router.router import route_query


MetricFn = Callable[[Sequence[float], Params], float]

METRICS: dict[ToolName, MetricFn] = {
    ToolName.total_return: total_return,
    ToolName.max_drawdown: max_drawdown,
    ToolName.realized_volatility: realized_volatility,
}


def run_intent(intent: Intent, provider: PriceProvider) -> Result | Refusal:
    validated_intent: Intent | Refusal = validate_intent(intent)
    if isinstance(validated_intent, Refusal):
        return validated_intent

    try:
        prices = provider.get_adjusted_close(
            ticker=validated_intent.tickers[0],
            n_days=validated_intent.time_range.n_days,
        )
    except PriceProviderError as e:
        return Refusal(
            reason=f"Failed to fetch price data: {str(e)}",
            clarifying_question=None,
            allowed_capabilities=list(ToolName),
        )

    metric_fn = METRICS.get(validated_intent.tool)
    if metric_fn is None:
        return Refusal(
            reason=f"Unsupported tool: {validated_intent.tool}",
            clarifying_question=None,
            allowed_capabilities=list(ToolName),
        )

    ret_value = metric_fn(prices, validated_intent.params)

    annualization = (
        validated_intent.params.annualization_factor
        if validated_intent.tool == ToolName.realized_volatility
        else None
    )

    return Result(
        tool=validated_intent.tool,
        tickers=validated_intent.tickers,
        value=ret_value,
        metadata={
            "range_n_days": validated_intent.time_range.n_days,
            "window": validated_intent.params.window,
            "annualization_factor": annualization,
            "data_points": len(prices),
            "price_source": provider.name(),
            "tool_version": "1.0.0",  # TODO
            "interpretation_notes": None,  # TODO
        },
    )


def run_query(
    user_query: str, llm_client: LLMClient, price_provider: PriceProvider
) -> Result | Refusal:
    intent_or_refusal = route_query(user_query, llm_client)
    if isinstance(intent_or_refusal, Refusal):
        return intent_or_refusal

    return run_intent(intent_or_refusal, price_provider)
