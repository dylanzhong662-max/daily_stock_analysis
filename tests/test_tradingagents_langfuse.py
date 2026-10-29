# -*- coding: utf-8 -*-
"""Tests for Langfuse tracing of TradingAgents research runs."""

from __future__ import annotations

import contextlib
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.agent import langfuse_observability as obs
from src.services import tradingagents_langfuse as tl


def _fake_langfuse_modules(attributes):
    langfuse = types.ModuleType("langfuse")

    @contextlib.contextmanager
    def propagate_attributes(**kwargs):
        attributes.update(kwargs)
        yield

    langfuse.propagate_attributes = propagate_attributes
    langchain = types.ModuleType("langfuse.langchain")
    langchain.CallbackHandler = _FakeHandler
    return {"langfuse": langfuse, "langfuse.langchain": langchain}


class _FakeHandler:
    """官方 CallbackHandler 的替身：记录 on_chain_start 收到的名字。"""

    def __init__(self):
        self.started = []

    def get_langchain_run_name(self, serialized, **kwargs):
        return kwargs.get("name") or (serialized or {}).get("name") or "<unknown>"

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        self.started.append(self.get_langchain_run_name(serialized, **kwargs))

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        pass

    def on_chain_error(self, error, *, run_id, **kwargs):
        pass


def _fake_client(root):
    client = MagicMock()

    @contextlib.contextmanager
    def start_as_current_observation(**kwargs):
        client.root_kwargs = kwargs
        yield root

    client.start_as_current_observation.side_effect = start_as_current_observation
    client.get_trace_url.return_value = "https://cloud.langfuse.com/trace/abc"
    return client


class MaskTest(unittest.TestCase):
    def test_long_content_redacted_only_inside_scope(self):
        data = {"messages": [{"role": "system", "content": "x" * 5000}], "ticker": "GOOGL"}
        self.assertEqual(obs.mask_long_content(data=data), data)  # Agent 自己的观测不受影响
        with patch.dict(os.environ, {obs.CAPTURE_CONTENT_ENV: "false"}), obs.redacted_content():
            masked = obs.mask_long_content(data=data)
        self.assertEqual(masked, {"messages": [{"role": "system", "content": "[omitted: 5000 chars]"}],
                                  "ticker": "GOOGL"})
        with patch.dict(os.environ, {obs.CAPTURE_CONTENT_ENV: "true"}), obs.redacted_content():
            self.assertEqual(obs.mask_long_content(data=data), data)


class ResearchTraceTest(unittest.TestCase):
    def test_disabled_is_noop(self):
        with patch.object(tl, "langfuse_client", return_value=None), \
                tl.research_trace(ticker="GOOGL", trade_date="2026-10-26", analysts=["market"], metadata={}) as trace:
            graph = SimpleNamespace(propagator=SimpleNamespace(get_graph_args=lambda callbacks=None: {}))
            trace.attach(graph)
            trace.finish(rating="Hold")
        self.assertFalse(trace.enabled)
        self.assertIsNone(tl.trace_url(trace))
        self.assertEqual(graph.propagator.get_graph_args(), {})

    def test_enabled_trace_wraps_run_and_flushes(self):
        attributes = {}
        root = MagicMock()
        client = _fake_client(root)
        original_args = MagicMock(return_value={"config": {}})
        graph = SimpleNamespace(propagator=SimpleNamespace(get_graph_args=original_args))
        with patch.dict(sys.modules, _fake_langfuse_modules(attributes)), \
                patch.object(tl, "langfuse_client", return_value=client):
            with tl.research_trace(ticker="GOOGL", trade_date="2026-10-26", analysts=["market", "news"],
                                   metadata={"data_source": "fmp", "deep_think_llm": "deepseek-reasoner",
                                             "unset": None}) as trace:
                trace.attach(graph)
                graph.propagator.get_graph_args()
                trace.finish(rating="Underweight", report_id="GOOGL_20261026_105901")
                self.assertEqual(tl.trace_url(trace), "https://cloud.langfuse.com/trace/abc")
        self.assertTrue(trace.enabled)
        original_args.assert_called_once_with(callbacks=[trace.handler])
        self.assertIsInstance(trace.handler, _FakeHandler)
        self.assertEqual(client.root_kwargs["as_type"], "agent")
        self.assertEqual(client.root_kwargs["name"], tl.ROOT_NAME)
        self.assertEqual(client.root_kwargs["input"]["analysts"], ["market", "news"])
        self.assertEqual(attributes["trace_name"], tl.TRACE_NAME)
        self.assertIn("data:fmp", attributes["tags"])
        self.assertEqual(attributes["metadata"]["ticker"], "GOOGL")
        self.assertNotIn("unset", attributes["metadata"])
        root.update.assert_called_once_with(output={"rating": "Underweight", "report_id": "GOOGL_20261026_105901"})
        client.flush.assert_called_once()

    def test_subgraph_nodes_are_named_after_their_role(self):
        handler = tl._role_named_handler_class(_FakeHandler)()
        handler.on_chain_start(None, {}, run_id="g", name="LangGraph")
        handler.on_chain_start(None, {}, run_id="m", parent_run_id="g", name="Market Analyst")
        handler.on_chain_start(None, {}, run_id="s", parent_run_id="m", name="LangGraph")
        handler.on_chain_start(None, {}, run_id="a", parent_run_id="s", name="agent")
        handler.on_chain_start(None, {}, run_id="p", parent_run_id="a", name="ChatPromptTemplate")
        handler.on_chain_start(None, {}, run_id="t", parent_run_id="s", name="tools")
        handler.on_chain_end({}, run_id="m")
        handler.on_chain_start(None, {}, run_id="b", parent_run_id="g", name="Bull Researcher")
        self.assertEqual(handler.started, ["LangGraph", "Market Analyst", "Market Analyst: subgraph",
                                           "Market Analyst: agent", "ChatPromptTemplate", "Market Analyst: tools",
                                           "Bull Researcher"])
        self.assertTrue(tl.is_role_node("Trader") and tl.is_role_node("Portfolio Manager"))
        self.assertFalse(tl.is_role_node("Memory Log"))

    def test_error_marks_root_and_setup_failure_degrades(self):
        root = MagicMock()
        tl.ResearchTrace(MagicMock(), root, "handler").finish(error=RuntimeError("boom"))
        self.assertEqual(root.update.call_args.kwargs["level"], "ERROR")

        client = MagicMock()
        client.start_as_current_observation.side_effect = RuntimeError("langfuse down")
        with patch.dict(sys.modules, _fake_langfuse_modules({})), \
                patch.object(tl, "langfuse_client", return_value=client), \
                tl.research_trace(ticker="X", trade_date="2026-10-26", analysts=[], metadata={}) as trace:
            pass
        self.assertFalse(trace.enabled)
        client.flush.assert_not_called()


if __name__ == "__main__":
    unittest.main()
