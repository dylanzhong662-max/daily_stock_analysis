"""Small, fail-open Langfuse helpers for Agent and tool observability.

Only compact, non-sensitive diagnostics are sent for tools.  Raw portfolio
data, news bodies, arguments, and tool outputs remain in the application.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Optional


def _enabled() -> bool:
    return (
        os.getenv("LANGFUSE_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
        and bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
        and bool(os.getenv("LANGFUSE_SECRET_KEY"))
    )


CAPTURE_CONTENT_ENV = "TRADINGAGENTS_LANGFUSE_CAPTURE_CONTENT"
# 脱敏范围内，超过这个长度的字符串视为“内容”（prompt、新闻、报告、工具输出），只上报长度
_MASK_MIN_CHARS = 200
_initialized = False
_redact_scope: ContextVar[bool] = ContextVar("langfuse_redact_scope", default=False)


def capture_content() -> bool:
    return os.getenv(CAPTURE_CONTENT_ENV, "false").strip().lower() in {"1", "true", "yes", "on"}


def _redact(data: Any) -> Any:
    if isinstance(data, str):
        return f"[omitted: {len(data)} chars]" if len(data) >= _MASK_MIN_CHARS else data
    if isinstance(data, dict):
        return {key: _redact(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [_redact(value) for value in data]
    return data


def mask_long_content(*, data: Any, **_kwargs: Any) -> Any:
    """Langfuse mask hook: inside ``redacted_content()`` replace long text with its length.

    Framework integrations (the LangChain callback used by TradingAgents) export full
    prompts and outputs; the scope keeps them to counts while the Agent's own compact
    observations elsewhere in the process stay as they are.
    """
    return _redact(data) if _redact_scope.get() else data


@contextmanager
def redacted_content() -> Iterator[None]:
    """Observations created in this context (and its child contexts) carry no long content."""
    token = _redact_scope.set(not capture_content())
    try:
        yield
    finally:
        _redact_scope.reset(token)


def langfuse_client() -> Optional[Any]:
    """Process-wide Langfuse client, or None when tracing is disabled or unavailable.

    The first construction fixes the mask for the whole process (the SDK keeps one
    client per public key), so every caller must come through here.
    """
    global _initialized
    if not _enabled():
        return None
    try:
        from langfuse import Langfuse, get_client

        if not _initialized:
            Langfuse(mask=mask_long_content)
            _initialized = True
        return get_client()
    except Exception:
        return None


def _client() -> Optional[Any]:
    return langfuse_client()


def _update(observation: Any, **values: Any) -> None:
    if observation is None:
        return
    try:
        observation.update(**values)
    except Exception:
        # Observability must never fail a trading-analysis request.
        pass


@contextmanager
def observe_agent_run(*, session_id: str, user_message: str) -> Iterator[Optional[Any]]:
    """Create a logical Agent root with only the user message as input."""
    client = _client()
    if client is None:
        yield None
        return
    try:
        with client.start_as_current_observation(as_type="agent", name="stock-analysis-agent") as observation:
            _update(
                observation,
                input={"message": user_message[:2000]},
                metadata={"session_id": session_id, "feature": "agent_chat"},
            )
            yield observation
    except Exception:
        yield None


@contextmanager
def observe_tool_call(*, tool_name: str, arguments_summary: str) -> Iterator[Optional[Any]]:
    """Create a tool child observation without recording raw results."""
    client = _client()
    if client is None:
        yield None
        return
    try:
        with client.start_as_current_observation(as_type="tool", name=tool_name) as observation:
            _update(observation, input={"arguments": arguments_summary[:500]})
            yield observation
    except Exception:
        yield None


def finish_tool_call(
    observation: Optional[Any], *, success: bool, duration_s: float, result_length: int,
    cached: bool, timeout: bool = False, error_code: Optional[str] = None,
) -> None:
    _update(
        observation,
        output={"success": success, "result_bytes": result_length},
        metadata={
            "duration_s": round(duration_s, 3),
            "cached": cached,
            "timeout": timeout,
            "error_code": error_code or "",
        },
        level="DEFAULT" if success else "ERROR",
        status_message=None if success else (error_code or "tool_failed"),
    )


def finish_agent_run(observation: Optional[Any], *, success: bool, total_steps: int,
                     total_tool_calls: int, error_code: Optional[str] = None) -> None:
    _update(
        observation,
        output={"success": success},
        metadata={"total_steps": total_steps, "total_tool_calls": total_tool_calls},
        level="DEFAULT" if success else "ERROR",
        status_message=None if success else (error_code or "agent_failed"),
    )


@contextmanager
def observe_generation(*, model: str, message_count: int, tools_available: int) -> Iterator[Optional[Any]]:
    """Record an LLM turn without exporting prompts, tool results, or completions.

    LiteLLM's OTEL callback serializes the full messages array, which includes
    potentially large RAG/news tool responses on the second agent turn.  The
    explicit observation keeps the useful operational signals while preserving
    the application's data-minimisation boundary.
    """
    client = _client()
    if client is None:
        yield None
        return
    try:
        with client.start_as_current_observation(
            as_type="generation", name="generate-agent-response", model=model,
        ) as observation:
            _update(
                observation,
                input={
                    "message_count": message_count,
                    "tools_available": tools_available,
                },
            )
            yield observation
    except Exception:
        yield None


def finish_generation(
    observation: Optional[Any], *, success: bool, content_length: int,
    requested_tools: list[str], usage: Optional[dict[str, Any]] = None,
    error_code: Optional[str] = None,
) -> None:
    """Finish a compact generation observation with counts only."""
    usage = usage or {}
    usage_details = {
        key: usage[key]
        for key in ("input", "output", "total", "input_tokens", "output_tokens", "total_tokens")
        if isinstance(usage.get(key), (int, float))
    }
    _update(
        observation,
        output={
            "content_chars": content_length,
            "requested_tools": requested_tools[:10],
        },
        usage_details=usage_details or None,
        level="DEFAULT" if success else "ERROR",
        status_message=None if success else (error_code or "generation_failed"),
    )
