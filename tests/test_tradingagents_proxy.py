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
