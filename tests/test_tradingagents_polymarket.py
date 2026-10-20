# -*- coding: utf-8 -*-
"""Tests for the automatic Polymarket reference block."""

from __future__ import annotations

import json
import os
import sys
import types
import unittest
from unittest.mock import patch

from src.services import tradingagents_polymarket as pm


def _market(mid, question, volume, prob="0.7", end="2099-01-01T00:00:00Z", wk=0.02):
    return {
        "id": mid, "question": question, "volumeNum": volume, "endDate": end,
        "outcomePrices": json.dumps([prob, "0.3"]), "outcomes": json.dumps(["Yes", "No"]),
        "oneWeekPriceChange": wk, "closed": False,
    }


def _fake_modules(search_results, historical=False):
    polymarket = types.ModuleType("tradingagents.dataflows.vendors.polymarket")
    calls = []

    def _request(path, params):
        calls.append(params["q"])
        result = search_results.get(params["q"])
        if isinstance(result, Exception):
            raise result
        return {"events": [{"markets": result or []}]}

    polymarket._request = _request
    polymarket._is_forward_looking = lambda m, now: not m.get("closed")
    polymarket._parse_json_list = lambda v: json.loads(v) if isinstance(v, str) else (v or [])
    date_window = types.ModuleType("tradingagents.dataflows.date_window")
    date_window.is_historical = lambda d: historical
    vendors = types.ModuleType("tradingagents.dataflows.vendors")
    vendors.polymarket = polymarket
    modules = {
        "tradingagents": types.ModuleType("tradingagents"),
        "tradingagents.dataflows": types.ModuleType("tradingagents.dataflows"),
        "tradingagents.dataflows.vendors": vendors,
        "tradingagents.dataflows.vendors.polymarket": polymarket,
        "tradingagents.dataflows.date_window": date_window,
    }
    return modules, calls


class PolymarketReferenceTest(unittest.TestCase):
    def _env(self, **env):
        base = {k: v for k, v in os.environ.items() if k not in (pm.TOPICS_ENV, pm.TICKER_ENV)}
        return patch.dict(os.environ, {**base, **env}, clear=True)

    def test_topic_resolution_defaults_ticker_and_user(self):
        with self._env():
            topics = pm.resolve_topics("NVDA", ["anthropic ipo", "Nvidia GTC"], use_defaults=True)
        names = [t for t, _ in topics]
        self.assertEqual(names[: len(pm.DEFAULT_TOPICS)], list(pm.DEFAULT_TOPICS))
        self.assertIn(("NVDA", "ticker"), topics)
        self.assertIn(("Nvidia GTC", "user"), topics)
        self.assertEqual(sum(1 for n in names if n.lower() == "anthropic ipo"), 1)  # 与默认去重

        with self._env(**{pm.TOPICS_ENV: "none", pm.TICKER_ENV: "false"}):
            self.assertEqual(pm.resolve_topics("NVDA", [], use_defaults=True), [])
        with self._env(**{pm.TOPICS_ENV: "Fed decision, AI bubble"}):
            self.assertEqual([t for t, k in pm.resolve_topics("AVGO", [], True) if k == "default"], ["Fed decision", "AI bubble"])
        with self._env():
            self.assertEqual(pm.resolve_topics("NVDA", ["X"], use_defaults=False), [("X", "user")])

    def test_build_reference_dedupes_ranks_and_tolerates_failures(self):
        shared = _market("m1", "Will Anthropic or OpenAI IPO first?", 400_000)
        results = {
            "Fed decision": [_market("f1", "No change after October meeting?", 14_000_000)],
            "Anthropic IPO": [shared, _market("a1", "Anthropic IPO by Dec 31?", 1_200_000)],
            "OpenAI IPO": [shared, _market("o1", "OpenAI IPO market cap above $1T?", 1_000_000)],
            "rate hike": RuntimeError("timeout"),
            "NVDA": [
                _market("n1", "Will NVIDIA (NVDA) hit (LOW) $192 in October?", 13_000),
                _market("n2", "Will NVIDIA (NVDA) Q3 Data Center revenue be above $100B?", 400),
            ],
        }
        modules, calls = _fake_modules(results)
        with self._env(**{pm.TOPICS_ENV: "Fed decision,rate hike,Anthropic IPO,OpenAI IPO"}), \
                patch.dict(sys.modules, modules):
            text, used = pm.build_reference("NVDA", "2026-10-20", [], use_defaults=True)
        self.assertEqual(used, ["Fed decision", "rate hike", "Anthropic IPO", "OpenAI IPO", "NVDA"])
        self.assertEqual(text.count("Will Anthropic or OpenAI IPO first?"), 1)
        self.assertIn('## "rate hike"\nUnavailable right now (RuntimeError)', text)
        nvda = text[text.index('## "NVDA"'):]
        self.assertLess(nvda.index("Data Center revenue"), nvda.index("hit (LOW)"))  # 财报类优先
        self.assertIn("Yes 70%", text)
        self.assertIn("1-week +2.0pp", text)

    def test_historical_date_withholds_without_network(self):
        modules, calls = _fake_modules({}, historical=True)
        with self._env(), patch.dict(sys.modules, modules):
            text, used = pm.build_reference("NVDA", "2026-01-05", [], use_defaults=True)
        self.assertIn("Withheld for 2026-01-05", text)
        self.assertEqual(calls, [])
        self.assertTrue(used)

    def test_no_topics_returns_empty(self):
        with self._env(**{pm.TOPICS_ENV: "none", pm.TICKER_ENV: "false"}):
            self.assertEqual(pm.build_reference("NVDA", "2026-10-20", [], use_defaults=True), ("", []))


if __name__ == "__main__":
    unittest.main()
