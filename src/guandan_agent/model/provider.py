"""Single-transmission provider boundary shared by online and offline callers."""

from typing import Protocol

from guandan_agent.contracts import ErrorInfo, ProviderInput, ProviderReply, Usage


class ProviderPort(Protocol):
    async def complete(self, input: ProviderInput) -> ProviderReply: ...

    async def aclose(self) -> None: ...


def safe_error(code: str, message: str) -> ErrorInfo:
    return ErrorInfo(code=code, component="model", safe_message=message, related_event_id=None)


def unknown_usage() -> Usage:
    return Usage(input_tokens=None, output_tokens=None, total_tokens=None,
                 cost_amount=None, currency=None, cost_basis=None)


def normalize_usage(value: object) -> Usage:
    """Preserve partial provider evidence; never infer a missing token total."""
    if not isinstance(value, dict):
        return unknown_usage()
    result: dict[str, int | None] = {}
    for field, aliases in {
        "input_tokens": ("input_tokens", "prompt_tokens"),
        "output_tokens": ("output_tokens", "completion_tokens"),
        "total_tokens": ("total_tokens",),
    }.items():
        result[field] = next((value[name] for name in aliases
                              if type(value.get(name)) is int and value[name] >= 0), None)
    return Usage(**result, cost_amount=None, currency=None, cost_basis=None)
