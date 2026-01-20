import argparse
import json
from typing import Sequence

from quantcli.data.yfinance_price_provider import YFinancePriceProvider
from quantcli.schemas.refusal import Refusal
from quantcli.orchestrator import run_query

from quantcli.llm.fake_llm_client import FakeLLMClient


HARDCODED_ROUTER_RESPONSE = (
    '{"type":"intent","intent":{"tickers":["AAPL"],"time_range":{"n_days":10},'
    '"tool":"max_drawdown","params":{"window":null,"annualization_factor":252}}}'
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quantcli")
    p.add_argument("query", nargs="+", help="Natural language query")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    query = " ".join(args.query)

    yfinance_provider = YFinancePriceProvider()
    llm_client = FakeLLMClient(HARDCODED_ROUTER_RESPONSE)  # TODO replace
    out = run_query(
        user_query=query, llm_client=llm_client, price_provider=yfinance_provider
    )

    print(out.model_dump_json())

    return 2 if isinstance(out, Refusal) else 0
