# QuantCLI

A guardrail-first CLI for financial analytics, using an LLM for intent parsing and deterministic code for all computation.

## Features
- Natural language → strict, schema-validated intent
- Deterministic, unit-tested financial metrics (no AI in computation)
- Validation-first execution with explicit failure modes (`Result` or `Refusal`)
- JSON-only output with exit codes (automation-safe)
- Internal observability without leaking internals


## Install & Run
```bash
git clone git@github.com:ihemmige/ai-quant-analytics-cli.git
cd ai-quant-analytics-cli
pip install -e .

export ANTHROPIC_API_KEY=...
quantcli "max drawdown AAPL last 10 days"
```

## Example Queries
```bash
quantcli "Compute the max drawdown for TSLA over the last 60 days."
quantcli "Compute realized volatility for MSFT over the last 90 days with a 20 day window."
quantcli "What was AAPL’s total return over the last 30 days?"
quantcli "Compute the Sharpe ratio for AAPL over the last 60 days with a 20 day window."
quantcli "Compute the Sharpe ratio for AAPL over the last 60 days with a 20 day window and a risk free rate of 0.05."
```

### Example invalid query (returns a structured refusal)
```bash
quantcli "Compute realized volatility for MSFT over the last 90 days."
quantcli "Compute Sharpe ratio for AAPL over the last 60 days."
```

## Implemented Metrics
The following metrics are currently supported for single-asset analysis:

- `max_drawdown`
- `realized_volatility` (requires `window`)
- `total_return`
- `sharpe_ratio` (requires `window`)

`n_days` refers to the number of price observations used in the calculation (minimum 2). For metrics requiring a window, the window must be less than `n_days`.

## Architecture
### Stage 1: Probabilistic Routing
```mermaid
flowchart LR
  U["User query"] --> CLI["quantcli"]
  CLI --> LLM["LLM router (Anthropic Claude)"]
  LLM --> D["Strict decoder"]

  D -->|valid wrapper + schema| I["Intent"]
  D -->|otherwise| Rf["Refusal"]
```

### Stage 2: Deterministic Execution
```mermaid
flowchart LR
  I["Intent"] --> V["Validator"]

  V -->|invalid / unsupported| Rf["Refusal"]
  V -->|valid| P["Price provider (yfinance)"]

  P -->|provider error| Rf
  P --> M["Metrics"]

  M -->|metric error| Rf
  M --> Rs["Result"]
```
Once decoding completes, no LLM output is consulted again.

## Observability & Debug Logging

QuantCLI provides **internal structured debug logging**, designed to preserve strict CLI output guarantees.

### Properties
- Off by default; enabled via `QUANTCLI_DEBUG=1`
- Logs are emitted as JSONL (one event per line)
- Logs are written to `stderr` or an optional file (never to `stdout`)
- No prompt text, user input, or raw LLM output is logged

### Enable debug logging (stderr)
```bash
export QUANTCLI_DEBUG=1
quantcli "max drawdown AAPL last 10 days"
```

### Log to a file instead of stderr
```bash
export QUANTCLI_DEBUG=1
export QUANTCLI_DEBUG_PATH=/tmp/quantcli.debug.log
quantcli "max drawdown AAPL last 10 days"
```

## Guarantees
- No guessing, retries, or JSON repair from LLM
- Ambiguous or unsupported queries return an explicit, structured `Refusal` (no silent fallbacks)
- Strict boundary between probabilistic routing and deterministic execution
- JSON-only stdout (third-party stdout/stderr suppressed)

## Development
### Setup
```bash
pip install -e .[dev]
pip install pre-commit
pre-commit install
```

### Checks
```bash
make lint      # ruff + black + mypy
make test      # pytest
```

All checks are enforced automatically on git commit via pre-commit.

## Future Enhancements
- Additional metrics (e.g. rolling returns, downside risk)
- Deterministic multi-asset and portfolio analytics
- Internal logging for improved observability
