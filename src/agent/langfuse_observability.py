"""Small, fail-open Langfuse helpers for Agent and tool observability.

Only compact, non-sensitive diagnostics are sent for tools.  Raw portfolio
data, news bodies, arguments, and tool outputs remain in the application.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional


def _enabled() -> bool:
    return (
        os.getenv("LANGFUSE_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
        and bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
        and bool(os.getenv("LANGFUSE_SECRET_KEY"))
    )


def _client() -> Optional[Any]:
    if not _enabled():
        return None
    try:
        from langfuse import get_client

        return get_client()
    except Exception:
        return None


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
