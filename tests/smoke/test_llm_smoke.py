# LLM smoke test.
#
# Disabled by default.
# Enable explicitly with:
#   QUANTCLI_LLM_SMOKE=1 pytest tests/smoke/test_llm_network_smoke.py
#
# Test checks:
# - The real LLM call runs end-to-end without crashing
# - Routing and decoding complete successfully
# - The result is either an Intent or Refusal
#
# Test does NOT check:
# - Semantic correctness of the output
# - Deterministic or stable behavior
# - Anything that should run in CI

import os

import pytest

from quantcli.llm.anthropic_client import AnthropicLLMClient
from quantcli.router.router import route_query
from quantcli.schemas.intent import Intent
from quantcli.schemas.refusal import Refusal

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(
        os.getenv("QUANTCLI_LLM_SMOKE") != "1",
        reason="Network-gated smoke test. Set QUANTCLI_LLM_SMOKE=1 to enable.",
    ),
]


def test_llm_network_smoke_no_crash_and_decodes() -> None:
    user_query = "total return AAPL 10 days"

    llm_client = AnthropicLLMClient(api_key=os.getenv("ANTHROPIC_API_KEY"))

    result = route_query(user_query, llm_client)

    assert result is not None
    assert isinstance(result, Intent | Refusal)
    assert hasattr(result, "model_dump")
