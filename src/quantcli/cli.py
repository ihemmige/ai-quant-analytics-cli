import argparse
from collections.abc import Callable, Sequence

from quantcli.data.price_provider import PriceProvider
from quantcli.data.yfinance_price_provider import YFinancePriceProvider
from quantcli.llm.llm_client import LLMClient
from quantcli.orchestrator import run_query
from quantcli.refusals import make_refusal
from quantcli.runtime import ConfigError, anthropic_client_from_env
from quantcli.schemas.refusal import Refusal
from quantcli.schemas.result import Result


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
        refusal = make_refusal(
            reason=str(e),
            clarifying_question="Set ANTHROPIC_API_KEY.",
        )
        print(refusal.model_dump_json())
        return 2

    try:
        yfinance_provider = provider_factory()
        out = run_query_fn(query, llm_client, yfinance_provider)
        print(out.model_dump_json())
        return 2 if isinstance(out, Refusal) else 0
    except Exception:
        refusal = make_refusal(
            reason="Unexpected internal error.",
            clarifying_question=None,
        )
        print(refusal.model_dump_json())
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    return cli(argv)
