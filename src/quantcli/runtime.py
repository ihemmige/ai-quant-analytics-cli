from quantcli.llm.anthropic_client import AnthropicLLMClient
import os


class ConfigError(Exception):
    pass


def anthropic_client_from_env() -> AnthropicLLMClient:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ConfigError("LLM authentication failed.")
    anthropic_model = os.getenv("QUANTCLI_ANTHROPIC_MODEL")
    return (
        AnthropicLLMClient(api_key=api_key, model=anthropic_model)
        if anthropic_model
        else AnthropicLLMClient(api_key=api_key)
    )
