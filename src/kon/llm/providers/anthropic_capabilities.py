"""
Per-model capability registry for Anthropic Messages API.

Encodes facts about each Claude model in one place instead of scattering
substring matching across provider code. Inspired by pi-mono's
`models.generated.ts` + `compat.forceAdaptiveThinking` pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Default thinking budgets (tokens) for non-adaptive models.
# These match pi-mono's `adjustMaxTokensForThinking` defaults.
DEFAULT_THINKING_BUDGETS: dict[str, int] = {
    "none": 0,
    "minimal": 1024,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
    "xhigh": 16384,  # non-adaptive models clamp xhigh down to high
}

# Default level -> effort mapping for adaptive models that don't specify their own.
DEFAULT_EFFORT_MAP: dict[str, str] = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
}


@dataclass(frozen=True)
class AnthropicCapabilities:
    # If True, the model uses Anthropic's adaptive thinking
    # (`thinking={type:"adaptive"}` + `output_config.effort`) instead of
    # budget-based thinking.
    adaptive_thinking: bool = False

    # For adaptive models, how each Kon thinking level maps to the model's
    # effort vocabulary. Opus 4.6 uses "max", Opus 4.7 uses "xhigh".
    effort_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_EFFORT_MAP))

    # For non-adaptive thinking models, custom per-level token budgets.
    thinking_budgets: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_THINKING_BUDGETS)
    )

    # Non-adaptive thinking models need the interleaved-thinking beta header
    # to use thinking between tool calls. Adaptive models include it natively.
    supports_interleaved_thinking_beta: bool = False


# Specific capability profiles keyed by canonical Anthropic model id substrings.
# Matching is performed by `lookup_capabilities()` below — first match wins,
# so list more specific entries first.
_OPUS_4_7 = AnthropicCapabilities(
    adaptive_thinking=True,
    effort_map={
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "xhigh",  # Opus 4.7 uses "xhigh"
    },
)

_OPUS_4_6 = AnthropicCapabilities(
    adaptive_thinking=True,
    effort_map={
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "max",  # Opus 4.6 uses "max"
    },
)

_SONNET_4_6 = AnthropicCapabilities(
    adaptive_thinking=True,
    effort_map={
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "max",
    },
)

_LEGACY_THINKING = AnthropicCapabilities(
    adaptive_thinking=False, supports_interleaved_thinking_beta=True
)


# Order matters: more specific patterns first. Patterns are matched against
# the lowercased model id with both "-" and "." variants normalized to "-".
_MODEL_PATTERNS: list[tuple[str, AnthropicCapabilities]] = [
    ("opus-4-7", _OPUS_4_7),
    ("opus-4-6", _OPUS_4_6),
    ("sonnet-4-6", _SONNET_4_6),
]


def lookup_capabilities(model_id: str) -> AnthropicCapabilities:
    """Look up the capability profile for an Anthropic model.

    Falls back to the legacy thinking profile (budget-based with interleaved
    beta) for unknown models — this matches behavior for Claude 3.x / 4.x
    non-adaptive thinking models.
    """
    normalized = model_id.lower().replace(".", "-")
    for pattern, caps in _MODEL_PATTERNS:
        if pattern in normalized:
            return caps
    return _LEGACY_THINKING


def supports_adaptive_thinking(model_id: str) -> bool:
    """Backwards-compatible helper used by existing callers/tests."""
    return lookup_capabilities(model_id).adaptive_thinking
