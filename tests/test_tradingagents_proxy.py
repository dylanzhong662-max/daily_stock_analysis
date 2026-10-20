# -*- coding: utf-8 -*-
"""Tests for routing TradingAgents social sources through a proxy."""

from __future__ import annotations

import importlib.util
import os
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

_HAS_TA = importlib.util.find_spec("tradingagents") is not None


class ProxyUrlopenTest(unittest.TestCase):
    def test_retries_429_then_succeeds(self):
        from src.services import tradingagents_proxy as px

        opener = MagicMock()
        ok = object()
        opener.open.side_effect = [HTTPError("u", 429, "Too Many", {}, None), ok]
        with patch.object(px, "build_opener", return_value=opener), patch.object(px.time, "sleep") as sleep:
            urlopen = px._proxied_urlopen("http://proxy:8118", retry_429=True)
            self.assertIs(urlopen(MagicMock(host="www.reddit.com"), timeout=5), ok)
        self.assertEqual(opener.open.call_count, 2)
        sleep.assert_called_once()

    def test_non_429_and_no_retry_mode_raise_immediately(self):
        from src.services import tradingagents_proxy as px

        opener = MagicMock()
        opener.open.side_effect = HTTPError("u", 403, "Forbidden", {}, None)
        with patch.object(px, "build_opener", return_value=opener):
            with self.assertRaises(HTTPError):
                px._proxied_urlopen("http://proxy:8118", retry_429=True)(MagicMock(), timeout=5)
        self.assertEqual(opener.open.call_count, 1)

    def test_requests_shim_adds_proxies(self):
        from src.services import tradingagents_proxy as px

        with patch("requests.get") as get:
            px._proxied_requests("http://proxy:8118").get("https://gamma-api.polymarket.com/markets", timeout=3)
        self.assertEqual(get.call_args.kwargs["proxies"], {"http": "http://proxy:8118", "https": "http://proxy:8118"})

    def test_no_env_means_no_install(self):
        from src.services import tradingagents_proxy as px

        with patch.dict(os.environ, {px.SOCIAL_PROXY_ENV: ""}):
            self.assertIsNone(px.install_social_proxy())

    def test_company_search_name_strips_suffixes(self):
        from src.services import tradingagents_proxy as px

        self.assertEqual(px.company_search_name("NVIDIA Corporation"), "NVIDIA")
        self.assertEqual(px.company_search_name("Broadcom Inc."), "Broadcom")
        self.assertEqual(px.company_search_name("Micron Technology, Inc."), "Micron Technology")
        self.assertEqual(px.company_search_name("Taiwan Semiconductor Manufacturing Company Limited"),
                         "Taiwan Semiconductor Manufacturing")

    @unittest.skipUnless(_HAS_TA, "tradingagents not installed")
    def test_tune_reddit_widens_search(self):
        from urllib.parse import parse_qs

        from tradingagents.agents.analysts import sentiment_analyst
        from tradingagents.dataflows.vendors import reddit

        from src.services import tradingagents_proxy as px

        saved = (reddit._search_qs, sentiment_analyst.subreddits_for, sentiment_analyst.fetch_reddit_posts)

        def restore():
            reddit._search_qs, sentiment_analyst.subreddits_for, sentiment_analyst.fetch_reddit_posts = saved
            sentiment_analyst._dsa_reddit_tuned = False
            px._search_names.clear()

        self.addCleanup(restore)
        sentiment_analyst._dsa_reddit_tuned = False
        with patch.object(reddit, "fetch_reddit_posts") as fetch:
            px.tune_reddit("NVDA", "NVIDIA Corporation")
            subs = sentiment_analyst.subreddits_for("NVDA")
            self.assertIn("semiconductors", subs)
            self.assertIn("NVDA_Stock", subs)
            self.assertNotIn("NVDA_Stock", sentiment_analyst.subreddits_for("AVGO"))
            self.assertEqual(parse_qs(reddit._search_qs("NVDA", 100))["q"], ['NVDA OR "NVIDIA"'])
            self.assertEqual(parse_qs(reddit._search_qs("ZZZ", 100))["q"], ["ZZZ"])
            sentiment_analyst.fetch_reddit_posts("NVDA", subs, start_date="2026-10-13", end_date="2026-10-20")
        self.assertEqual(fetch.call_args.kwargs["limit_per_sub"], px.REDDIT_POSTS_PER_SUB)
        # 加密货币版块映射不受影响
        self.assertIn("CryptoCurrency", sentiment_analyst.subreddits_for("BTC-USD"))

    @unittest.skipUnless(_HAS_TA, "tradingagents not installed")
    def test_disable_stocktwits_only_when_configured(self):
        from tradingagents.agents.analysts import sentiment_analyst

        from src.services import tradingagents_proxy as px

        original = sentiment_analyst.fetch_stocktwits_messages
        self.addCleanup(setattr, sentiment_analyst, "fetch_stocktwits_messages", original)
        with patch.dict(os.environ, {px.DISABLE_STOCKTWITS_ENV: ""}):
            self.assertFalse(px.disable_stocktwits_if_configured())
        self.assertIs(sentiment_analyst.fetch_stocktwits_messages, original)
        with patch.dict(os.environ, {px.DISABLE_STOCKTWITS_ENV: "true"}):
            self.assertTrue(px.disable_stocktwits_if_configured())
        self.assertIn("not queried for NVDA", sentiment_analyst.fetch_stocktwits_messages("NVDA", limit=30))

    @unittest.skipUnless(_HAS_TA, "tradingagents not installed")
    def test_install_patches_vendor_modules(self):
        from tradingagents.dataflows.vendors import polymarket, reddit, stocktwits

        from src.services import tradingagents_proxy as px

        originals = (reddit.urlopen, stocktwits.urlopen, polymarket.requests)

        def restore():
            reddit.urlopen, stocktwits.urlopen, polymarket.requests = originals

        self.addCleanup(restore)
        with patch.object(px, "_installed_proxy", None):
            self.assertEqual(px.install_social_proxy("http://proxy:8118"), "http://proxy:8118")
            self.assertIsNot(reddit.urlopen, originals[0])
            self.assertIsNot(polymarket.requests, originals[2])
            self.assertTrue(hasattr(polymarket.requests, "RequestException"))


if __name__ == "__main__":
    unittest.main()
