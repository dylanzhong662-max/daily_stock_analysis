# -*- coding: utf-8 -*-
"""Tests for the TradingAgents data quality patches."""

from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import MagicMock

from src.services import tradingagents_quality as quality

_HAS_TA = importlib.util.find_spec("tradingagents") is not None

# GOOGL 季度利润表（SEC EDGAR 形状）：Q4 不单独申报，2025-09-30 之后直接是 2026-03-31
_SEC_INCOME = (
    "# Income Statement for GOOGL (quarterly), USD in millions unless the row says otherwise\n"
    "# SEC EDGAR facts filed on or before 2026-10-26, at the values filed then\n\n"
    ",2024-06-30,2024-09-30,2025-03-31,2025-06-30,2025-09-30,2026-03-31,2026-06-30\n"
    "Revenue,84742,88268,90234,96428,102346,109896,119796\n"
    "Gross Profit,unavailable (not tagged by this filer),unavailable (not tagged by this filer),"
    "unavailable (not tagged by this filer),unavailable (not tagged by this filer),"
    "unavailable (not tagged by this filer),unavailable (not tagged by this filer),"
    "unavailable (not tagged by this filer)\n"
    "Diluted EPS (USD/shares),1.89,2.12,2.81,2.31,2.87,5.11,9.11\n"
)

_SEC_CASHFLOW = (
    "# Cash Flow Statement for GOOGL (quarterly), USD in millions unless the row says otherwise\n"
    "# SEC EDGAR facts filed on or before 2026-10-26, at the values filed then\n\n"
    ",2025-03-31,2025-06-30 (6 months),2025-09-30 (9 months),2026-03-31,2026-06-30 (6 months)\n"
    "Capital Expenditure,17197,39643,63596,35674,80598\n"
    "Investing Cash Flow,-29356,-40738,-68515,-63389,-145822\n"
    "Financing Cash Flow,-20201,-26033,-44416,25077,86320\n"
)

_FMP_INCOME = (
    "# Income Statement data for SKHY (quarterly, source: FMP, as filed by 2026-10-26)\n"
    ",2026-06-30,2026-03-31,2025-12-31,2025-09-30,2025-06-30\n"
    "period,Q2,Q1,Q4,Q3,Q2\n"
    "revenue,79.32,52.58,32.83,24.45,22.23\n"
    "costOfRevenue,13.33,10.9,10.25,10.42,10.25\n"
)

_CPI = (
    "## FRED: Consumer Price Index (CPIAUCSL)\n"
    "- Units: Index 1982-1984=100\n"
    "- Frequency: Monthly (SA)\n"
    "- Window: 2025-10-26 to 2026-10-26\n\n"
    "**Latest:** 336.298 (2026-09-01) | **Change from 2025-11-01 to 2026-09-01:** +11.24 (+3.46%), from 325.063\n\n"
    "| Date | Value |\n| --- | --- |\n"
    "| 2025-11-01 | 325.063 |\n| 2026-08-01 | 334.131 |\n| 2026-09-01 | 336.298 |\n"
)

_CPI_LONG = _CPI.replace("| 2025-11-01 | 325.063 |", "| 2025-09-01 | 324.245 |\n| 2025-11-01 | 325.063 |")

_FEDFUNDS = (
    "## FRED: Federal Funds Effective Rate (FEDFUNDS)\n"
    "- Units: %\n"
    "- Frequency: Monthly (NSA)\n\n"
    "| Date | Value |\n| --- | --- |\n"
    "| 2025-09-01 | 4.22 |\n| 2026-08-01 | 3.63 |\n| 2026-09-01 | 3.75 |\n"
)

_SNAPSHOT = (
    "## Verified market data snapshot for SKHY\n\n"
    "- Requested analysis date: 2026-10-26\n"
    "- Latest trading row used: 2026-10-23\n\n"
    "| Indicator | Value |\n|---|---:|\n"
    "| close_10_ema | 146.06 |\n| close_50_sma | 169.86 |\n| close_200_sma | 164.28 |\n| rsi | 34.51 |"
)


class RequiredRowsTest(unittest.TestCase):
    def test_windows(self):
        self.assertEqual(quality.required_rows("close_200_sma"), 200)
        self.assertEqual(quality.required_rows("close_10_ema"), 10)
        self.assertEqual(quality.required_rows("boll_lb"), 20)
        self.assertEqual(quality.required_rows("macdh"), 35)
        self.assertEqual(quality.required_rows("unknown"), 1)


class IndicatorPatchTest(unittest.TestCase):
    def test_bulk_marks_rows_before_window(self):
        original = MagicMock(return_value={f"2026-10-0{i}": str(i) for i in range(1, 6)})
        values = quality._wrap_bulk(original)("SKHY", "close_3_sma", "2026-10-05")
        self.assertTrue(values["2026-10-01"].startswith("N/A: insufficient price history (needs 3"))
        self.assertTrue(values["2026-10-02"].startswith("N/A"))
        self.assertEqual(values["2026-10-03"], "3")

    def test_snapshot_rewrites_long_window_and_states_history(self):
        text = quality.rewrite_snapshot(_SNAPSHOT, 75, "2026-07-10")
        self.assertIn("| close_200_sma | N/A: insufficient price history (needs 200 trading rows, has 75) |", text)
        self.assertIn("| close_50_sma | 169.86 |", text)
        self.assertIn("- Price history available: 75 trading rows since 2026-07-10 (less than one full year", text)
        full = quality.rewrite_snapshot(_SNAPSHOT, 1254, "2021-10-26")
        self.assertIn("| close_200_sma | 164.28 |", full)
        self.assertNotIn("less than one full year", full)


class StatementGuideTest(unittest.TestCase):
    def test_sec_quarter_gap_and_same_quarter_yoy(self):
        guide = quality.statement_guide(_SEC_INCOME)
        self.assertIn("- 2026-06-30: quarter, Apr 2026 – Jun 2026", guide)
        self.assertIn("Not filed as a standalone quarter: Oct 2024 – Dec 2024; Oct 2025 – Dec 2025.", guide)
        self.assertIn("NOT consecutive quarters", guide)
        # 2026-06-30 对 2025-06-30（+24.2%），而不是对 2025-03-31（+32.8%）
        self.assertIn("- Revenue: 2026-06-30 +24.2%, 2026-03-31 +21.8%", guide)
        self.assertNotIn("+32.8%", guide)
        self.assertNotIn("Gross Profit", guide)  # 未申报的行不出同比
        self.assertIn("not adjusted for later stock splits", guide)

    def test_year_to_date_columns_pair_with_same_span(self):
        guide = quality.statement_guide(_SEC_CASHFLOW)
        self.assertIn("2026-06-30: 6 months year to date, Jan 2026 – Jun 2026", guide)
        self.assertIn("2026-06-30 (6m) +103.3%", guide)
        self.assertIn("2026-03-31 +107.4%", guide)
        self.assertNotIn("Not filed as a standalone quarter", guide)
        # 两期均为流出按流出规模表述；正负号翻转不给百分比
        self.assertIn("- Investing Cash Flow: 2026-06-30 (6m) outflow +258.0%, 2026-03-31 outflow +115.9%", guide)
        self.assertNotIn("Financing Cash Flow", guide)

    def test_fmp_whitelist_and_annual_skip(self):
        guide = quality.statement_guide(_FMP_INCOME)
        self.assertIn("- revenue: 2026-06-30 +256.8%", guide)
        self.assertNotIn("costOfRevenue", guide)
        self.assertEqual(quality.statement_guide(_FMP_INCOME.replace("quarterly", "annual")), "")

    def test_trim_keeps_recent_columns_but_yoy_uses_older_base(self):
        ends = [f"{year}-{month}" for year in range(2022, 2027) for month in ("03-31", "06-30", "09-30")][:14]
        text = ("# Income Statement for X (quarterly), USD in millions unless the row says otherwise\n\n"
                "," + ",".join(ends) + "\nRevenue," + ",".join(str(100 + i) for i in range(len(ends))) + "\n")
        trimmed = quality.trim_statement_columns(text, keep=4)
        self.assertIn("# Showing the most recent 4 of 14 period columns.", trimmed)
        self.assertIn("," + ",".join(ends[-4:]) + "\n", trimmed)
        self.assertNotIn(ends[0], trimmed)
        self.assertEqual(quality.trim_statement_columns(text, keep=20), text)
        out = quality._wrap_statement(MagicMock(return_value=text))("X", "quarterly", "2026-10-26")
        self.assertNotIn(ends[0] + ",", out.split("## Period guide")[0])
        self.assertIn("# Showing the most recent 12 of 14 period columns.", out)
        self.assertIn("- Revenue: 2026-06-30 +2.7%", out)  # 113 vs 2025-06-30 的 110
        balance = quality._wrap_statement(MagicMock(return_value=text), guide=False)("X")
        self.assertNotIn("Period guide", balance)

    def test_wrapper_keeps_latest_quarter_section_out_of_table(self):
        extra = "\n## Latest quarter not yet in SEC XBRL (source: FMP)\n\n,2026-09-30 (quarter)\nRevenue,127577\n"
        out = quality._wrap_statement(MagicMock(return_value=_SEC_INCOME + extra))("GOOGL", "quarterly", "2026-10-29")
        self.assertTrue(out.endswith(extra))
        guide = out[:-len(extra)]
        self.assertIn("- Revenue: 2026-06-30 +24.2%", guide)
        self.assertNotIn("127577", guide)  # 补充小节不参与主表同比

    def test_wrapper_appends_and_survives_bad_input(self):
        wrapped = quality._wrap_statement(MagicMock(return_value=_SEC_INCOME))
        self.assertTrue(wrapped("GOOGL", "quarterly", "2026-10-26").startswith(_SEC_INCOME))
        self.assertIn("Period guide", wrapped("GOOGL", "quarterly", "2026-10-26"))
        plain = quality._wrap_statement(MagicMock(return_value="no data"))
        self.assertEqual(plain("X"), "no data")


class FredSummaryTest(unittest.TestCase):
    def test_monthly_index_yoy_uses_same_month(self):
        summary = quality.fred_summary(_CPI_LONG)
        self.assertIn("**Year-over-year:** +3.7% (2026-09-01 vs 2025-09-01, from 324.245)", summary)
        self.assertIn("**Latest period change:** +0.6% (2026-08-01 -> 2026-09-01)", summary)
        self.assertIn("not a year-over-year rate", summary)

    def test_rate_series_uses_percentage_points(self):
        summary = quality.fred_summary(_FEDFUNDS)
        self.assertIn("**Year-over-year:** -0.47 pp", summary)
        self.assertIn("**Latest period change:** +0.12 pp", summary)

    def test_daily_series_untouched(self):
        self.assertEqual(quality.fred_summary(_CPI.replace("Monthly (SA)", "Daily")), "")

    def test_wrapper_fetches_year_ago_base_when_window_lacks_it(self):
        original = MagicMock(side_effect=[_CPI, _CPI_LONG])
        text = quality._wrap_fred(original)("cpi", "2026-10-26")
        self.assertIn("+3.7% (2026-09-01 vs 2025-09-01", text)
        self.assertEqual(original.call_args_list[1].args, ("cpi", "2026-10-26", 400))

    def test_wrapper_single_call_when_base_present(self):
        original = MagicMock(return_value=_CPI_LONG)
        quality._wrap_fred(original)("cpi", "2026-10-26", 450)
        original.assert_called_once_with("cpi", "2026-10-26", 450)


class MemoryPatchTest(unittest.TestCase):
    def test_cross_ticker_lessons_disabled(self):
        original = MagicMock(return_value="ctx")
        wrapped = quality._wrap_past_context(original)
        self.assertEqual(wrapped("log", "GOOGL", as_of="2026-10-26"), "ctx")
        original.assert_called_once_with("log", "GOOGL", n_same=5, n_cross=0, as_of="2026-10-26")


@unittest.skipUnless(_HAS_TA, "tradingagents not installed")
class InstallTest(unittest.TestCase):
    def test_install_is_idempotent_and_wraps_vendors(self):
        from tradingagents.agents import tools as ta_tools
        from tradingagents.dataflows import router
        from tradingagents.dataflows.vendors.yahoo import market, snapshot
        from tradingagents.memory.log import TradingMemoryLog

        originals = (market._get_stock_stats_bulk, market.get_stock_stats, snapshot.build_verified_market_snapshot,
                     ta_tools.build_verified_market_snapshot, TradingMemoryLog.get_past_context)
        saved_methods = {m: dict(v) for m, v in router.VENDOR_METHODS.items()}

        def restore():
            (market._get_stock_stats_bulk, market.get_stock_stats, snapshot.build_verified_market_snapshot,
             ta_tools.build_verified_market_snapshot, TradingMemoryLog.get_past_context) = originals
            for method, impls in saved_methods.items():
                router.VENDOR_METHODS[method] = impls
            quality._installed = False

        self.addCleanup(restore)
        quality._installed = False
        quality.install_quality_patches()
        bulk = market._get_stock_stats_bulk
        income = router.VENDOR_METHODS["get_income_statement"]["sec_edgar"]
        quality.install_quality_patches()
        self.assertIs(market._get_stock_stats_bulk, bulk)
        self.assertIs(router.VENDOR_METHODS["get_income_statement"]["sec_edgar"], income)
        for func in (bulk, income, ta_tools.build_verified_market_snapshot, TradingMemoryLog.get_past_context,
                     router.VENDOR_METHODS["get_macro_indicators"]["fred"]):
            self.assertTrue(quality._is_patched(func))


if __name__ == "__main__":
    unittest.main()
