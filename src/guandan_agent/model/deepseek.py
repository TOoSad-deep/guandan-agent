"""Explicit-credential DeepSeek JSON transport; no game semantics or retries."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar

import httpx
from langchain_deepseek import ChatDeepSeek
from langsmith import tracing_context
from openai import (
    APIConnectionError, APIStatusError, APITimeoutError,
    ContentFilterFinishReasonError, LengthFinishReasonError,
)

from guandan_agent.clock import Clock, SystemClock
from guandan_agent.contracts import ProviderInput, ProviderReply
from guandan_agent.model.provider import normalize_usage, safe_error, unknown_usage


class DeepSeekProvider:
    """Owns its HTTP client; credentials are injected, never read from .env.

    HTTP request hooks count observable attempted transports per invocation.
    Cancellation may already have incurred remote work, and is allowed to escape
    so the caller can close its own authorization and record unknown usage.
    """

    def __init__(
        self, *, api_key: str, model: str = "deepseek-flash", clock: Clock | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("An explicit provider credential is required.")
        self._clock = clock or SystemClock()
        self._requests: ContextVar[list[int] | None] = ContextVar("provider_requests", default=None)
        self._closed = False

        async def count_request(_request: httpx.Request) -> None:
            counter = self._requests.get()
            if counter is not None:
                counter[0] += 1

        self._http = httpx.AsyncClient(
            transport=transport, trust_env=False, event_hooks={"request": [count_request]},
        )
        self._model = ChatDeepSeek(
            model=model, api_key=api_key, api_base="https://api.deepseek.com",
            max_retries=0, http_async_client=self._http, cache=False,
            extra_body={"thinking": {"type": "disabled"}}, callbacks=[],
            include_response_headers=False,
        )

    async def complete(self, input: ProviderInput) -> ProviderReply:
        start = self._clock.monotonic()
        counter = [0]
        token = self._requests.set(counter)
        content = None
        usage = unknown_usage()
        error = None
        try:
            remaining = input.deadline_mono - start
            if self._closed:
                error = safe_error("MODEL_TRANSPORT", "Provider has been closed.")
            elif remaining <= 0:
                error = safe_error("MODEL_TIMEOUT", "Provider deadline has expired.")
            else:
                with tracing_context(enabled=False):
                    async with asyncio.timeout(remaining):
                        raw = await self._model.ainvoke(
                            [(message.role, message.content) for message in input.messages],
                            config={"callbacks": []},
                            response_format={"type": "json_object"},
                            max_tokens=input.max_output_tokens,
                        )
                content = raw.content if isinstance(raw.content, str) else None
                # response_metadata preserves partial usage without inventing totals.
                raw_usage = raw.response_metadata.get("token_usage")
                usage = normalize_usage(raw_usage if isinstance(raw_usage, dict)
                                        else raw.usage_metadata)
                if self._clock.monotonic() >= input.deadline_mono:
                    content = None
                    error = safe_error("MODEL_TIMEOUT", "Provider reply arrived after its deadline.")
        except TimeoutError:
            error = safe_error("MODEL_TIMEOUT", "Provider request exceeded its time limit.")
        except asyncio.CancelledError:
            raise
        except LengthFinishReasonError as exc:
            # JSON-mode parsing may raise after a successful HTTP response.
            # Retain its reported usage, never the truncated text or repr(exc).
            observed = getattr(getattr(exc, "completion", None), "usage", None)
            if observed is not None:
                usage = normalize_usage(observed.model_dump())
            error = safe_error("MODEL_OUTPUT_LIMIT", "Provider output reached the configured token limit.")
        except ContentFilterFinishReasonError as exc:
            observed = getattr(getattr(exc, "completion", None), "usage", None)
            if observed is not None:
                usage = normalize_usage(observed.model_dump())
            error = safe_error("MODEL_CONTENT_FILTER", "Provider did not return usable output because of a content filter.")
        except APIStatusError as exc:
            status = exc.status_code
            message = (f"Provider request failed with HTTP status {status}."
                       if type(status) is int and 100 <= status <= 599 else "Provider request failed with an HTTP status error.")
            error = safe_error("MODEL_TRANSPORT", message)
        except APITimeoutError:
            error = safe_error("MODEL_TIMEOUT", "Provider network request timed out.")
        except APIConnectionError:
            error = safe_error("MODEL_TRANSPORT", "Provider connection could not be established.")
        except Exception:
            error = safe_error("MODEL_TRANSPORT", "Provider response could not be processed.")
        finally:
            self._requests.reset(token)
        return ProviderReply(
            call_id=input.call_id, content=content, usage=usage,
            transport_requests=counter[0], elapsed_ms=max(0.0, self._clock.monotonic() - start) * 1000,
            error=error,
        )

    async def aclose(self) -> None:
        self._closed = True
        await self._http.aclose()
