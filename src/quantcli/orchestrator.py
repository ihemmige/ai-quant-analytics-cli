from quantcli.data.price_provider import PriceProvider, PriceProviderError
from quantcli.llm.llm_client import LLMClient
from quantcli.refusals import make_refusal
from quantcli.router.router import route_query
from quantcli.schemas.intent import Intent
from quantcli.schemas.refusal import Refusal
from quantcli.schemas.result import Result
from quantcli.schemas.tool_name import ToolName
from quantcli.tools.registry import get_metric
from quantcli.validate_intent import validate_intent


def run_intent(intent: Intent, provider: PriceProvider) -> Result | Refusal:
    validated_intent = validate_intent(intent)
    if isinstance(validated_intent, Refusal):
        return validated_intent

    metric_fn = get_metric(validated_intent.tool)
    if metric_fn is None:
        return make_refusal(reason="Requested tool is not supported.")

    try:
        prices = provider.get_adjusted_close(
            ticker=validated_intent.tickers[0],
            n_days=validated_intent.time_range.n_days,
        )
    except PriceProviderError:
        return make_refusal(reason="Unable to retrieve valid price data.")

    try:
        ret_value = metric_fn(prices, validated_intent.params)
    except ValueError:
        return make_refusal(reason="Unable to compute metric.")

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
