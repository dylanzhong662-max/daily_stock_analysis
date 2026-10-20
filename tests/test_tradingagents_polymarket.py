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


def _market(mid, question, volume, prob="0.7", end="2099-01-01T00:00:00Z", wk=0.02, group=None):
    return {
        "id": mid, "question": question, "volumeNum": volume, "endDate": end, "groupItemTitle": group,
        "outcomePrices": json.dumps([prob, "0.3"]), "outcomes": json.dumps(["Yes", "No"]),
        "oneWeekPriceChange": wk, "closed": False,
    }


def _event(eid, title, markets, neg_risk=False, volume=None):
    return {"id": eid, "title": title, "negRisk": neg_risk, "markets": markets,
            "volume": volume if volume is not None else sum(m["volumeNum"] for m in markets)}


def _fake_modules(search_results, historical=False):
    polymarket = types.ModuleType("tradingagents.dataflows.vendors.polymarket")
    calls = []

    def _request(path, params):
        calls.append(params["q"])
        result = search_results.get(params["q"])
        if isinstance(result, Exception):
            raise result
        return {"events": result or []}

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

    def test_build_reference_groups_events_and_filters(self):
        shared = _event("e-first", "Will Anthropic or OpenAI IPO first?", [_market("m1", "Will Anthropic or OpenAI IPO first?", 400_000)])
        buckets = _event("e-cap", "Anthropic IPO Closing Market Cap", [
            _market("c1", "cap <1.25T", 300_000, prob="0.04", group="<$1.25T"),
            _market("c2", "cap 1.25-1.5T", 300_000, prob="0.09", group="$1.25–$1.5T"),
            _market("c3", "cap 1.5-1.75T", 300_000, prob="0.17", group="$1.5–$1.75T"),
        ], neg_risk=True)
        hikes = _event("e-hikes", "How many Fed rate hikes in 2026?", [
            _market("h1", "1 hike", 400_000, prob="0.205", group="1"),
            _market("h2", "2 hikes", 180_000, prob="0.685", group="2"),
        ], neg_risk=True)
        results = {
            "Fed rate hike": [hikes, _event("e-boc", "Bank of Canada Rate Hike in 2026?", [_market("b", "BoC", 9_000)])],
            "Anthropic IPO": [buckets, shared],
            "OpenAI IPO": [shared, _event("e-oai", "OpenAI IPO closing market cap above $1T?", [_market("o1", "OpenAI >1T", 1_000_000)])],
            "AI bubble": RuntimeError("timeout"),
            "NVDA": [
                _event("e-px", "NVIDIA (NVDA) hit price in October?", [_market("n1", "hit $192", 13_000, group="↓ $192")], neg_risk=True),
                _event("e-dc", "NVIDIA Q3 Data Center revenue?", [_market("n2", "DC revenue above $100B?", 900)]),
                _event("e-tiny", "NVIDIA tiny market?", [_market("n3", "tiny", 100)]),
            ],
        }
        modules, calls = _fake_modules(results)
        with self._env(**{pm.TOPICS_ENV: "Fed rate hike,Anthropic IPO,OpenAI IPO,AI bubble"}), \
                patch.dict(sys.modules, modules):
            text, used = pm.build_reference("NVDA", "2026-10-20", [], use_defaults=True)
        self.assertEqual(used, ["Fed rate hike", "Anthropic IPO", "OpenAI IPO", "AI bubble", "NVDA"])
        # 分档事件以事件标题 + 完整分布展示
        self.assertIn("**How many Fed rate hikes in 2026?**", text)
        self.assertIn("mutually exclusive outcomes", text)
        self.assertIn("  - 2: 68%", text)
        self.assertIn("  - $1.5–$1.75T: 17%", text)
        # 低成交量事件被过滤；跨主题事件去重
        self.assertNotIn("Bank of Canada", text)
        self.assertNotIn("tiny", text)
        self.assertEqual(text.count("Will Anthropic or OpenAI IPO first?"), 1)
        self.assertIn('## "AI bubble"\nUnavailable right now (RuntimeError)', text)
        nvda = text[text.index('## "NVDA"'):]
        self.assertLess(nvda.index("DC revenue"), nvda.index("hit $192"))  # 财报类优先
        self.assertIn("thin market", nvda)  # $900 < THIN_VOLUME

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
