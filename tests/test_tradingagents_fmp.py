# -*- coding: utf-8 -*-
"""Tests for the FMP data adapter used by the TradingAgents integration."""

from __future__ import annotations

import importlib.util
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_HAS_TA = importlib.util.find_spec("tradingagents") is not None


def _response(status=200, payload=None, text=None):
    resp = MagicMock()
    resp.status_code = status
    if payload is None:
        resp.json.side_effect = ValueError("not json")
        resp.text = text or "Premium Query Parameter"
    else:
        resp.json.return_value = payload
    return resp


@unittest.skipUnless(_HAS_TA, "tradingagents not installed")
class FmpAdapterTest(unittest.TestCase):
    def setUp(self):
        from src.services import tradingagents_fmp as fmp

        self.fmp = fmp
        self.client = fmp.FmpClient("test-key")
        self.client.session = MagicMock()
        patcher = patch.object(fmp, "_client", self.client)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_history_maps_adjusted_columns_and_end_is_exclusive(self):
        self.client.session.get.return_value = _response(payload=[
            {"date": "2026-10-15", "adjOpen": 3, "adjHigh": 4, "adjLow": 2, "adjClose": 3.5, "volume": 30},
            {"date": "2026-10-14", "adjOpen": 1, "adjHigh": 2, "adjLow": 1, "adjClose": 1.5, "volume": 10},
            {"date": "2026-10-16", "adjOpen": 9, "adjHigh": 9, "adjLow": 9, "adjClose": 9, "volume": 9},
        ])
        frame = self.fmp._FmpYfShim.Ticker("aapl").history(start="2026-10-14", end="2026-10-16")
        self.assertEqual(list(frame.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertEqual([d.strftime("%Y-%m-%d") for d in frame.index], ["2026-10-14", "2026-10-15"])
        self.assertEqual(frame["Close"].iloc[-1], 3.5)
        params = self.client.session.get.call_args.kwargs["params"]
        self.assertEqual(params["symbol"], "AAPL")
        self.assertEqual(params["apikey"], "test-key")

    def test_empty_history_is_no_market_data(self):
        from tradingagents.dataflows.errors import NoMarketDataError

        self.client.session.get.return_value = _response(payload=[])
        with self.assertRaises(NoMarketDataError):
            self.fmp.fetch_history("ZZZZQ", "2026-10-01", "2026-10-15")

    def test_plan_restricted_or_http_error_is_vendor_unavailable(self):
        from tradingagents.dataflows.errors import VendorUnavailableError

        self.client.session.get.return_value = _response(payload=None, text="Premium Query Parameter")
        with self.assertRaises(VendorUnavailableError):
            self.client.get("historical-price-eod/dividend-adjusted", symbol="0700.HK")
        self.client.session.get.return_value = _response(status=402, payload={})
        with self.assertRaises(VendorUnavailableError):
            self.client.get("profile", symbol="0700.HK")

    def test_missing_key_is_not_configured(self):
        from tradingagents.dataflows.errors import VendorNotConfiguredError

        with patch.dict(os.environ, {"FMP_API_KEY": ""}):
            client = self.fmp.FmpClient(None)
        with self.assertRaises(VendorNotConfiguredError):
            client.get("profile", symbol="AAPL")

    def test_statements_filter_by_filing_date(self):
        self.client.session.get.return_value = _response(payload=[
            {"date": "2026-06-27", "filingDate": "2026-07-31", "revenue": 2},
            {"date": "2026-03-28", "filingDate": "2026-05-01", "revenue": 1},
        ])
        out = self.fmp.get_income_statement("AAPL", "quarterly", "2026-06-30")
        self.assertIn("2026-03-28", out)
        self.assertNotIn("2026-06-27", out)

    def test_news_window_and_insider_point_in_time(self):
        self.client.session.get.return_value = _response(payload=[
            {"publishedDate": "2026-10-15 21:12:13", "title": "in", "publisher": "CNBC", "text": "t", "url": "u"},
            {"publishedDate": "2026-10-20 08:00:00", "title": "future", "publisher": "WSJ"},
        ])
        with patch("tradingagents.dataflows.config.get_config", return_value={"news_article_limit": 5}):
            out = self.fmp.get_news("AAPL", "2026-10-08", "2026-10-15")
        self.assertIn("### in (source: CNBC", out)
        self.assertNotIn("future", out)

        self.client.session.get.return_value = _response(payload=[
            {"filingDate": "2026-10-15", "transactionDate": "2026-10-13", "reportingName": "A"},
            {"filingDate": "2026-10-20", "transactionDate": "2026-10-18", "reportingName": "LATE"},
        ])
        out = self.fmp.get_insider_transactions("AAPL", "2026-10-15")
        self.assertIn("A", out)
        self.assertNotIn("LATE", out)

    def test_company_profile_uses_yahoo_field_names(self):
        self.client.session.get.return_value = _response(payload=[
            {"companyName": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics",
             "exchange": "NASDAQ", "isEtf": False, "isFund": False},
        ])
        profile = self.fmp.get_company_profile("AAPL")
        self.assertEqual(profile["longName"], "Apple Inc.")
        self.assertEqual(profile["quoteType"], "EQUITY")

    def test_apply_config_keeps_user_tool_vendors(self):
        cfg = self.fmp.apply_fmp_config({"tool_vendors": {"get_news": "alpha_vantage"}})
        self.assertEqual(cfg["tool_vendors"]["get_news"], "alpha_vantage")
        self.assertEqual(cfg["tool_vendors"]["get_fundamentals"], "fmp")
        self.assertEqual(cfg["tool_vendors"]["get_income_statement"], "sec_edgar,fmp")

    def test_install_patches_tradingagents_modules(self):
        from tradingagents.agents import context
        from tradingagents.dataflows import router
        from tradingagents.dataflows.vendors.yahoo import market, ohlcv

        originals = (ohlcv.yf, ohlcv.yf_retry, market.yf, market.yf_retry, context.get_company_profile)
        saved_methods = {m: dict(v) for m, v in router.VENDOR_METHODS.items()}

        def restore():
            ohlcv.yf, ohlcv.yf_retry, market.yf, market.yf_retry, context.get_company_profile = originals
            for method, impls in saved_methods.items():
                router.VENDOR_METHODS[method] = impls

        self.addCleanup(restore)
        with patch.object(self.fmp, "_installed", False):
            self.fmp.install_fmp_vendor("k")
        self.assertIs(ohlcv.yf, self.fmp._FmpYfShim)
        self.assertIs(market.yf, self.fmp._FmpYfShim)
        self.assertIs(context.get_company_profile, self.fmp.get_company_profile)
        self.assertIs(router.VENDOR_METHODS["get_news"]["fmp"], self.fmp.get_news)


class DataSourceSelectionTest(unittest.TestCase):
    def _resolve(self, env, fmp_key=None):
        from src.services.tradingagents_service import resolve_data_source

        base = {k: v for k, v in os.environ.items() if k not in ("FMP_API_KEY", "TRADINGAGENTS_DATA_SOURCE")}
        with patch.dict(os.environ, {**base, **env}, clear=True):
            return resolve_data_source(SimpleNamespace(fmp_api_key=fmp_key))

    def test_auto_prefers_fmp_when_key_present(self):
        self.assertEqual(self._resolve({"TRADINGAGENTS_DATA_SOURCE": "auto"}, fmp_key="k"), "fmp")
        self.assertEqual(self._resolve({"TRADINGAGENTS_DATA_SOURCE": ""}), "yfinance")

    def test_explicit_choices(self):
        from src.services.tradingagents_service import TradingAgentsError

        self.assertEqual(self._resolve({"TRADINGAGENTS_DATA_SOURCE": "yfinance"}, fmp_key="k"), "yfinance")
        with self.assertRaises(TradingAgentsError):
            self._resolve({"TRADINGAGENTS_DATA_SOURCE": "fmp"})
        with self.assertRaises(TradingAgentsError):
            self._resolve({"TRADINGAGENTS_DATA_SOURCE": "polygon"}, fmp_key="k")


if __name__ == "__main__":
    unittest.main()
