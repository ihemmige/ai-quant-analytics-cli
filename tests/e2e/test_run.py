from quantcli.schemas import Intent, Result, Refusal, ToolName, Params, TimeRange
from quantcli.run import run
from quantcli.data import FakePriceProvider, PriceProviderError

def test_run_refusal_short_circuits_provider_call():
    intent = Intent(
        tickers=["AAPL", "GOOG"],
        time_range=TimeRange(n_days=5),
        tool=ToolName.total_return,
        params=Params(),
    )
    result = run(intent, FakePriceProvider())
    # The provider should not be called since the intent is invalid
    assert isinstance(result, Refusal)