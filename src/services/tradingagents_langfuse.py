# -*- coding: utf-8 -*-
"""TradingAgents 研究的 Langfuse 观测。

一次研究 = 一条 trace：根观测是 ``run-tradingagents-research``（agent 类型），
TradingAgents 的 LangGraph 图通过 Langfuse 官方 LangChain ``CallbackHandler`` 接入，
每个角色节点、每次模型调用（含 token 用量）和工具调用都挂在根观测下。

- 复用 ``src/agent/langfuse_observability`` 的启用开关（``LANGFUSE_ENABLED`` + key）与进程级客户端。
- 默认只上报计数：长文本（prompt、新闻、报告、工具输出）替换为长度；
  ``TRADINGAGENTS_LANGFUSE_CAPTURE_CONTENT=true`` 时上报原文。
- 观测是 fail-open 的：任何 Langfuse 异常都只记日志，不影响研究。
"""

from __future__ import annotations

import logging
from contextlib import ExitStack, contextmanager
from typing import Any, Dict, Iterator, List, Optional

from src.agent.langfuse_observability import langfuse_client, redacted_content

logger = logging.getLogger(__name__)

ROOT_NAME = "run-tradingagents-research"
TRACE_NAME = "tradingagents-research"

# 分析师子图里的节点名在所有分析师之间重复（都叫 agent/tools），trace 里加上所属角色
_SUBGRAPH_NODE_LABELS = {"agent": "agent", "tools": "tools", "wrap_up": "wrap-up", "LangGraph": "subgraph"}


def is_role_node(name: str) -> bool:
    """TradingAgents 顶层图里的角色节点（Market Analyst、Bull Researcher、Trader…）。"""
    return name == "Trader" or name.endswith((" Analyst", " Researcher", " Manager"))


def role_scoped_name(name: str, role: Optional[str]) -> Optional[str]:
    """子图节点的新名字；不需要改名时返回 None。"""
    label = _SUBGRAPH_NODE_LABELS.get(name)
    if not role or not label or name == role:
        return None
    return f"{role}: {label}"


def _role_named_handler_class(base: type) -> type:
    """在官方 CallbackHandler 上只改子图节点的显示名，其余行为不变。"""

    class RoleNamedCallbackHandler(base):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._dsa_roles: Dict[Any, str] = {}

        def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
            try:
                name = self.get_langchain_run_name(serialized, **kwargs)
                role = name if is_role_node(name) else self._dsa_roles.get(parent_run_id)
                if role:
                    self._dsa_roles[run_id] = role
                renamed = role_scoped_name(name, role)
                if renamed:
                    kwargs["name"] = renamed
            except Exception as exc:  # 改名失败就用原名
                logger.debug("[TradingAgents/Langfuse] node rename skipped: %s", exc)
            return super().on_chain_start(serialized, inputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

        def on_chain_end(self, outputs, *, run_id, **kwargs):
            self._dsa_roles.pop(run_id, None)
            return super().on_chain_end(outputs, run_id=run_id, **kwargs)

        def on_chain_error(self, error, *, run_id, **kwargs):
            self._dsa_roles.pop(run_id, None)
            return super().on_chain_error(error, run_id=run_id, **kwargs)

    return RoleNamedCallbackHandler


class ResearchTrace:
    """一次研究的观测句柄；未启用时所有方法都是空操作。"""

    def __init__(self, client: Any = None, root: Any = None, handler: Any = None, trace_id: Optional[str] = None):
        self.client = client
        self.root = root
        self.handler = handler
        self.trace_id = trace_id

    @property
    def enabled(self) -> bool:
        return self.handler is not None

    def attach(self, graph: Any) -> None:
        """把回调挂到图的调用配置上（图级回调会被节点内的模型和工具调用继承）。"""
        if not self.enabled:
            return
        propagator = getattr(graph, "propagator", None)
        original = getattr(propagator, "get_graph_args", None)
        if original is None:
            logger.info("[TradingAgents/Langfuse] graph has no propagator.get_graph_args; tracing skipped")
            return
        handler = self.handler

        def get_graph_args(callbacks: Optional[List[Any]] = None):
            return original(callbacks=[*(callbacks or []), handler])

        propagator.get_graph_args = get_graph_args

    def finish(self, *, rating: Optional[str] = None, report_id: Optional[str] = None,
               error: Optional[BaseException] = None) -> None:
        if self.root is None:
            return
        try:
            if error is None:
                self.root.update(output={"rating": rating, "report_id": report_id})
            else:
                self.root.update(level="ERROR", status_message=f"{type(error).__name__}: {error}"[:500])
        except Exception as exc:
            logger.info("[TradingAgents/Langfuse] could not record result: %s", exc)


def trace_url(trace: ResearchTrace) -> Optional[str]:
    try:
        return trace.client.get_trace_url() if trace.enabled else None
    except Exception:
        return None


@contextmanager
def research_trace(*, ticker: str, trade_date: str, analysts: List[str],
                   metadata: Dict[str, Any]) -> Iterator[ResearchTrace]:
    """包住一次研究；退出时 flush，保证一次性容器退出前把数据发出去。"""
    client = langfuse_client()
    if client is None:
        yield ResearchTrace()
        return
    stack = ExitStack()
    trace = ResearchTrace()
    try:
        from langfuse import propagate_attributes
        from langfuse.langchain import CallbackHandler

        stack.enter_context(redacted_content())
        stack.enter_context(propagate_attributes(
            trace_name=TRACE_NAME,
            tags=["tradingagents", f"data:{metadata.get('data_source') or 'unknown'}"],
            metadata={key: str(value) for key, value in {"ticker": ticker, "trade_date": trade_date,
                                                          **metadata}.items() if value is not None},
        ))
        root = stack.enter_context(client.start_as_current_observation(
            as_type="agent", name=ROOT_NAME,
            input={"ticker": ticker, "trade_date": trade_date, "analysts": analysts},
        ))
        trace = ResearchTrace(client, root, _role_named_handler_class(CallbackHandler)(),
                              client.get_current_trace_id())
    except Exception as exc:  # 观测失败不影响研究
        logger.warning("[TradingAgents/Langfuse] tracing disabled for this run: %s", exc)
        stack.close()
        stack = ExitStack()
        trace = ResearchTrace()
    try:
        yield trace
    finally:
        try:
            stack.close()
            if trace.enabled:
                client.flush()
        except Exception as exc:
            logger.info("[TradingAgents/Langfuse] flush failed: %s", exc)
