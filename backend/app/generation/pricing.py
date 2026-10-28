"""Cost of a Claude response, from its `usage` and per-model prices (USD per million tokens).

With server-side fallbacks, `usage.iterations` lists every attempt (the declined one and the
fallback) with its own model and token counts; top-level `usage` covers only the attempt that
produced the message. So when iterations are present, each one is priced at its own model.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import structlog

from app.generation.types import Usage

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float
    cache_write: float  # 5-minute TTL writes: 1.25x the input price


DEFAULT_PRICES: dict[str, Price] = {
    "claude-opus-5-5": Price(input=4.0, output=20.0, cache_read=0.20, cache_write=5.0),
    "claude-opus-5": Price(input=5.0, output=25.0, cache_read=0.50, cache_write=6.25),
    "claude-opus-4-8": Price(input=5.0, output=25.0, cache_read=0.50, cache_write=6.25),
    "claude-sonnet-5-5": Price(input=2.0, output=10.0, cache_read=0.20, cache_write=2.5),
    "claude-haiku-5-5": Price(input=0.10, output=0.50, cache_read=0.01, cache_write=0.125),
}


def price_for(model: str, overrides: Mapping[str, Mapping[str, float]]) -> Price | None:
    if model in overrides:
        return Price(**overrides[model])
    return DEFAULT_PRICES.get(model)


def _tokens(entry: Any) -> tuple[int, int, int, int]:
    return (
        getattr(entry, "input_tokens", 0) or 0,
        getattr(entry, "output_tokens", 0) or 0,
        getattr(entry, "cache_read_input_tokens", 0) or 0,
        getattr(entry, "cache_creation_input_tokens", 0) or 0,
    )


def usage_cost(
    usage: Any, model: str, overrides: Mapping[str, Mapping[str, float]] | None = None
) -> Usage:
    """Total tokens and cost of a response, across fallback attempts when there were any."""
    overrides = overrides or {}
    attempts = [
        (getattr(entry, "model", None) or model, entry)
        for entry in (getattr(usage, "iterations", None) or [])
    ] or [(model, usage)]

    totals = [0, 0, 0, 0]
    cost = 0.0
    for attempt_model, entry in attempts:
        tokens = _tokens(entry)
        totals = [a + b for a, b in zip(totals, tokens, strict=True)]
        price = price_for(attempt_model, overrides)
        if price is None:
            log.warning("pricing.unknown_model", model=attempt_model)
            continue
        uncached, output, cache_read, cache_write = tokens
        cost += (
            uncached * price.input
            + output * price.output
            + cache_read * price.cache_read
            + cache_write * price.cache_write
        ) / 1_000_000
    return Usage(
        input_tokens=totals[0],
        output_tokens=totals[1],
        cache_read_tokens=totals[2],
        cache_write_tokens=totals[3],
        cost_usd=round(cost, 6),
    )
