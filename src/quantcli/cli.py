import argparse
from typing import Sequence

from quantcli.data.yfinance_price_provider import YFinancePriceProvider
from quantcli.schemas.refusal import Refusal
from quantcli.schemas.result import Result
from quantcli.orchestrator import run_query
from quantcli.runtime import anthropic_client_from_env, ConfigError
from quantcli.tools.registry import supported_tools
from quantcli.llm.llm_client import LLMClient
from typing import Callable
from quantcli.data.price_provider import PriceProvider


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quantcli")
    p.add_argument("query", nargs="+", help="Natural language query")
    return p


def cli(
    argv: Sequence[str] | None,
    *,
    llm_factory: Callable[[], LLMClient] = anthropic_client_from_env,
    provider_factory: Callable[[], PriceProvider] = YFinancePriceProvider,
    run_query_fn: Callable[
        [str, LLMClient, PriceProvider], Result | Refusal
    ] = run_query,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    query = " ".join(args.query)

    try:
        llm_client = llm_factory()
    except ConfigError as e:
        refusal = Refusal(
            reason=str(e),
            clarifying_question="Set ANTHROPIC_API_KEY.",
            allowed_capabilities=supported_tools(),
        )
        print(refusal.model_dump_json())
        return 2

    try:
        yfinance_provider = provider_factory()
        out = run_query_fn(
            user_query=query,
            llm_client=llm_client,
            price_provider=yfinance_provider,
        )
        print(out.model_dump_json())
        return 2 if isinstance(out, Refusal) else 0
    except Exception as e:
        refusal = Refusal(
            reason="Unexpected internal error.",
            clarifying_question=None,
            allowed_capabilities=supported_tools(),
        )
        print(refusal.model_dump_json())
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    return cli(argv)
