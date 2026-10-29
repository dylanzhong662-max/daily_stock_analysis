# -*- coding: utf-8 -*-
"""Tests for the offline evaluation of TradingAgents reports."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.services import tradingagents_eval as ev

# 来自 2026-10-26 两份真实报告（修复前）的句子
_SKHY_MA = "- **中期阻力**：200 SMA **164.28** 与 Bollinger 中轨 **164.04** 高度重合 → 构成 164 左右的\"重合阻力带\"；"
_SKHY_MA_OK = "1. **200 日均线不可用**（仅有 75 个交易日数据，2026-07-10 起），因此无法判断长期趋势是否已破坏。"
_GOOGL_YOY_BAD = "- **2026Q1 收入 1,197.96 亿，同比 +32.8%，四年最快**。"
_GOOGL_YOY_OK = "2026Q2 营收 1198.0 亿，同比 +24.2%，营业利润同比 +30.4%。"
_GOOGL_SEGMENT = "Google Cloud 收入同比 +82%，季度增量已超过 Search。"
_GOOGL_COST = "2026Q2 收入成本同比仅 +17.7%（低于收入 24.2%）。"
_SKHY_CPI_BAD = "| CPI（CPIAUCSL，SA） | **336.298（2026-09）** | 同比 **+3.46%** | 通胀重新加速 |"
_CROSS = "但我明确不把技术面当作承重梁：历史经验（SKHY，Underweight，+0.1%）表明，把评级锚在短周期技术触发上会被打脸。"
_PEER = "Motley Fool 指出最便宜的 Mag 7 就是 Alphabet（非NVDA、非MSFT）。"
_TOUCH = ("**10 月市场**：\"↓$320：91%\"、\"↓$310：61%\"。- **读法**：Polymarket 上市场普遍认为 10 月收盘价将低于 320（91%），"
          "且跌破 310 的概率过半。")
_TRADING_RULE = "财报前若收盘跌破330.38（10-21低点），无条件再减仓；预测市场仅作参考。"

_GOOGL_REVENUE_YOY = [24.2, 21.8, 15.9, 13.8, 12.0, 15.1, 13.6, 15.4]


class RatingHitTest(unittest.TestCase):
    def test_direction_and_hold_band(self):
        self.assertTrue(ev.rating_hit("Buy", 0.01))
        self.assertFalse(ev.rating_hit("Overweight", -0.001))
        self.assertTrue(ev.rating_hit("Underweight", -0.03))
        self.assertFalse(ev.rating_hit("Sell", 0.02))
        self.assertTrue(ev.rating_hit("Hold", 0.02))
        self.assertTrue(ev.rating_hit("Hold", -0.019))
        self.assertFalse(ev.rating_hit("Hold", 0.021))
        self.assertIsNone(ev.rating_hit("REVIEW", 0.05))

    def test_outcomes_skip_unfinished_windows_and_done_ones(self):
        report = ev.Report("SKHY_20261016_203424", "SKHY", "2026-10-16", "Underweight", Path("."))

        def fetch(ticker, date, days, benchmark):
            return (0.01, -0.012, 5, "2026-10-23") if days == 5 else (None, None, None, None)

        outcomes = ev.compute_outcomes(report, fetch, "SPY", {})
        self.assertEqual(list(outcomes), ["5"])
        self.assertTrue(outcomes["5"]["hit"])
        self.assertEqual(ev.compute_outcomes(report, MagicMock(side_effect=AssertionError), "SPY",
                                             {"5": {}, "20": {}}), {})


class MisreadRuleTest(unittest.TestCase):
    def test_short_history_ma(self):
        text = "\n".join([_SKHY_MA, _SKHY_MA_OK])
        self.assertEqual(ev.check_short_history_ma(text, 75), [_SKHY_MA])
        self.assertEqual(ev.check_short_history_ma(text, 1254), [])
        self.assertEqual(ev.check_short_history_ma(text, None), [])

    def test_revenue_yoy_against_filed_quarters(self):
        text = "\n".join([_GOOGL_YOY_BAD, _GOOGL_YOY_OK, _GOOGL_SEGMENT, _GOOGL_COST])
        hits = ev.check_revenue_yoy(text, _GOOGL_REVENUE_YOY)
        self.assertEqual(len(hits), 1)
        self.assertIn("cited +32.8%", hits[0])
        self.assertEqual(ev.check_revenue_yoy(text, []), [])

    def test_new_false_positive_shapes(self):
        self.assertEqual(ev.check_short_history_ma("只有77个交易日价格历史，200日均线都算不出来，溢价42.1%。", 77), [])
        margin = "2025Q2 毛利率仅 27.5%（成本同比暴增 12.4% 而收入近乎零增长），2026Q2 已回到 40.4%。"
        receivable = "应收账款占单季营收 60% 且同比 +257%，若需求转弱将转化为坏账。"
        self.assertEqual(ev.check_revenue_yoy("\n".join([margin, receivable]), [256.8, 25.4]), [])
        self.assertEqual(ev.check_cpi_yoy("| CPI 指数（SA） | 336.298 | 8→9 月 +0.6% |", 3.7), [])

    def test_unchanged_value_after_version_bump_is_not_rewritten(self):
        client = MagicMock()
        report = ev.Report("X_20261026_000000", "X", "2026-10-26", "Hold", Path("."))
        state = {"trace_id": "t", "scores": {"misread_yoy@rules-v1": 2}}
        ev.ScoreSink(client).score(report, state, "misread_yoy", 2, "NUMERIC", version="rules-v2")
        ev.ScoreSink(client).score(report, state, "misread_yoy", 3, "NUMERIC", version="rules-v3")
        self.assertEqual(client.create_score.call_count, 1)
        self.assertEqual(state["scores"]["misread_yoy@rules-v2"], 2)

    def test_cpi_yoy(self):
        hits = ev.check_cpi_yoy(_SKHY_CPI_BAD, 3.72)
        self.assertEqual(len(hits), 1)
        self.assertIn("FRED +3.72%", hits[0])
        self.assertEqual(ev.check_cpi_yoy(_SKHY_CPI_BAD.replace("3.46", "3.72"), 3.717), [])

    def test_cross_ticker_needs_lesson_context(self):
        self.assertEqual(ev.check_cross_ticker(_CROSS, "GOOGL", ["SKHY", "NVDA"]), [_CROSS])
        self.assertEqual(ev.check_cross_ticker(_PEER, "GOOGL", ["NVDA", "MSFT"]), [])
        self.assertEqual(ev.check_cross_ticker(_CROSS, "SKHY", ["SKHY"]), [])

    def test_touch_market_misread(self):
        self.assertEqual(len(ev.check_touch_market(_TOUCH)), 1)
        self.assertEqual(ev.check_touch_market(_TRADING_RULE), [])

    def test_parse_reference_values(self):
        guide = ("## Year-over-year (computed: each period vs the same period one year earlier)\n"
                 "- Revenue: 2026-06-30 +24.2%, 2026-03-31 +21.8%\n- Net Income: 2026-06-30 +297.9%\n")
        self.assertEqual(ev.parse_revenue_yoy(guide), [24.2, 21.8])
        self.assertEqual(ev.parse_revenue_yoy("- revenue: 2026-06-30 +256.8%"), [256.8])
        self.assertEqual(ev.parse_fred_yoy("**Year-over-year:** +3.7% (2026-09-01 vs 2025-09-01"), 3.7)
        self.assertIsNone(ev.parse_fred_yoy("no yoy"))


class JudgeTest(unittest.TestCase):
    def _payload(self, **overrides):
        data = {key: {"label": "acceptable", "reason": "ok"} for key in (*ev.DIMENSIONS, "overall")}
        data.update(overrides)
        return json.dumps(data)

    def test_parse_valid_and_fenced_output(self):
        payload = self._payload(evidence_grounding={"label": "Poor", "reason": "x"}, overall={"label": "good"})
        result = ev.parse_judge_output("```json\n" + payload + "\n```")
        self.assertEqual(result["evidence_grounding"]["label"], "poor")
        self.assertEqual(result["overall"]["label"], "poor")  # 总体由维度推导，不信模型自报
        self.assertEqual(result["balance"]["label"], "acceptable")

    def test_overall_follows_rubric(self):
        labels = dict.fromkeys(ev.DIMENSIONS, "acceptable")
        self.assertEqual(ev.overall_label(labels), "acceptable")
        self.assertEqual(ev.overall_label({**labels, "actionability": "good", "balance": "good"}), "good")
        self.assertEqual(ev.overall_label({**labels, "balance": "poor", "actionability": "good",
                                           "consistency": "good"}), "acceptable")
        self.assertEqual(ev.overall_label({**labels, "consistency": "poor"}), "poor")

    def test_balance_capped_when_final_rating_drifts(self):
        result = ev.parse_judge_output(self._payload(balance={"label": "good", "reason": "weighed"},
                                                     actionability={"label": "good", "reason": "r"}))
        self.assertEqual(result["overall"]["label"], "good")
        capped = ev.apply_rubric_caps(result, -1.5)
        self.assertEqual(capped["balance"]["label"], "acceptable")
        self.assertIn("capped", capped["balance"]["reason"])
        self.assertEqual(capped["overall"]["label"], "acceptable")
        kept = ev.parse_judge_output(self._payload(balance={"label": "good", "reason": "ok"}))
        self.assertEqual(ev.apply_rubric_caps(kept, -0.5)["balance"]["label"], "good")
        self.assertEqual(ev.apply_rubric_caps(kept, None)["balance"]["label"], "good")

    def test_invalid_output_is_dropped(self):
        self.assertIsNone(ev.parse_judge_output("not json"))
        self.assertIsNone(ev.parse_judge_output(self._payload(balance={"label": "excellent"})))
        self.assertIsNone(ev.parse_judge_output(json.dumps({"overall": {"label": "good"}})))

    def test_judge_input_uses_sections_and_truncates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "5_portfolio").mkdir()
            (path / "5_portfolio" / "decision.md").write_text("**Rating**: Sell\n" + "x" * 7000, encoding="utf-8")
            report = ev.Report("X_20261026_000000", "X", "2026-10-26", "Sell", path)
            text = ev.judge_input(report)
        self.assertIn("## Portfolio manager decision", text)
        self.assertIn("[...truncated]", text)
        self.assertNotIn("## Trader plan", text)

    def _role_report(self, tmp):
        path = Path(tmp)
        files = {
            "1_analysts/market.md": "**Rating**: Hold（偏多）\n", "1_analysts/sentiment.md": "**Rating**: Mildly Bullish \n",
            "1_analysts/news.md": "FINAL TRANSACTION PROPOSAL: **HOLD**", "1_analysts/fundamentals.md": "**Rating**: Buy\n",
            "2_research/manager.md": "**Recommendation**: Underweight\n", "3_trading/trader.md": "**Action**: Sell\n",
            "5_portfolio/decision.md": "**Rating**: Underweight\n",
        }
        for relative, body in files.items():
            (path / relative).parent.mkdir(parents=True, exist_ok=True)
            (path / relative).write_text(body, encoding="utf-8")
        return ev.Report("SKHY_20261028_091641", "SKHY", "2026-10-28", "Underweight", path)

    def test_role_ratings_and_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            ratings = ev.role_ratings(self._role_report(tmp))
        self.assertEqual(ratings["market analyst"], "Hold")
        self.assertEqual(ratings["sentiment analyst"], "Mildly Bullish")
        self.assertEqual(ratings["news analyst"], "Hold")
        self.assertEqual(ratings["trader"], "Sell")
        self.assertNotIn("bull researcher", ratings)
        # 分析师平均 (0+1+0+2)/4=0.75，最终 Underweight=-1
        self.assertAlmostEqual(ev.rating_drift(ratings), -1.75)
        self.assertIsNone(ev.rating_drift({"portfolio manager": "Hold"}))

    def test_judge_facts_carry_verified_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self._role_report(tmp)
            facts = ev.judge_facts(report, history=(77, "2026-10-27"), revenue_yoy=[256.8, 198.1],
                                   cpi_yoy=3.7, rule_hits={"misread_short_history_ma": [_SKHY_MA]})
            stale = ev.judge_facts(report, history=(623, "2025-02-06"), revenue_yoy=[], cpi_yoy=None, rule_hits={})
            text = ev.judge_input(report, facts)
        self.assertIn("77 bars", facts)
        self.assertIn("Too short for a 200-day average", facts)
        self.assertIn("+256.8%", facts)
        self.assertIn("-1.8 notches", facts)
        self.assertIn("Flagged by misread_short_history_ma", facts)
        self.assertIn("stale", stale)
        self.assertNotIn("Too short", stale)
        self.assertLess(text.index("VERIFIED FACTS"), text.index("## Portfolio manager decision"))


class JudgeModelTest(unittest.TestCase):
    _ENV = (ev.JUDGE_MODEL_ENV, ev.JUDGE_API_KEY_ENV, ev.JUDGE_BASE_URL_ENV, ev.JUDGE_PROXY_ENV)

    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in self._ENV}

    def tearDown(self):
        for key, value in self._saved.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value

    def test_dedicated_judge_model(self):
        os.environ[ev.JUDGE_MODEL_ENV] = "openai/gpt-6.1-sol"
        os.environ[ev.JUDGE_API_KEY_ENV] = "sk-test"
        os.environ[ev.JUDGE_BASE_URL_ENV] = "https://aihubmix.com/v1"
        config = MagicMock(litellm_model="deepseek/deepseek-reasoner", llm_model_list=[])
        self.assertEqual(ev.judge_model_kwargs(config),
                         {"model": "openai/gpt-6.1-sol", "api_key": "sk-test", "api_base": "https://aihubmix.com/v1"})
        self.assertEqual(ev.judge_version(), f"{ev.JUDGE_VERSION}@gpt-6.1-sol")

    def test_judge_proxy_only_on_judge_client(self):
        os.environ[ev.JUDGE_MODEL_ENV] = "openai/gpt-6.1-sol"
        os.environ[ev.JUDGE_API_KEY_ENV] = "sk-test"
        os.environ[ev.JUDGE_BASE_URL_ENV] = "https://aihubmix.com/v1"
        os.environ[ev.JUDGE_PROXY_ENV] = "http://127.0.0.1:8118"
        kwargs = ev.judge_model_kwargs(MagicMock(litellm_model="deepseek/deepseek-reasoner", llm_model_list=[]))
        self.assertEqual(str(kwargs["client"].base_url).rstrip("/"), "https://aihubmix.com/v1")
        self.assertNotIn("HTTPS_PROXY", os.environ)

    def test_falls_back_to_litellm_model(self):
        config = MagicMock(litellm_model="alias", llm_model_list=[
            {"model_name": "alias", "litellm_params": {"model": "deepseek/deepseek-reasoner", "api_key": "k"}}])
        self.assertEqual(ev.judge_model_kwargs(config), {"model": "deepseek/deepseek-reasoner", "api_key": "k"})
        self.assertEqual(ev.judge_version(), ev.JUDGE_VERSION)


class ReviewSheetTest(unittest.TestCase):
    def test_export_hides_judge_and_import_validates(self):
        report = ev.Report("GOOGL_20261026_152922", "GOOGL", "2026-10-26", "Sell", Path("/r/GOOGL_20261026_152922"))
        with tempfile.TemporaryDirectory() as tmp:
            sheet = ev.write_review_sheet([report], Path(tmp) / "sheet.csv")
            content = sheet.read_text(encoding="utf-8-sig")
            self.assertNotIn("acceptable", content)
            self.assertIn("/api/v1/tradingagents/reports/GOOGL_20261026_152922/html", content)
            filled = content.replace("Sell,/api", "Sell,/api").splitlines()
            filled[1] += "good,poor,acceptable,good,acceptable,太偏空"
            filled[1] = filled[1].replace(",,,,,,", ",")
            sheet.write_text("\n".join(filled) + "\n", encoding="utf-8")
            rows = ev.read_review_sheet(sheet)
            self.assertEqual(rows["GOOGL_20261026_152922"]["consistency"], "poor")
            sheet.write_text(content.replace(",,,,,,", ",great,,,,,") , encoding="utf-8")
            with self.assertRaises(ValueError):
                ev.read_review_sheet(sheet)

    def test_agreement(self):
        judge = {"a": {d: {"label": "good"} for d in (*ev.DIMENSIONS, "overall")},
                 "b": {d: {"label": "poor"} for d in (*ev.DIMENSIONS, "overall")}}
        human = {"a": {"overall": "good", "balance": "poor"}, "b": {"overall": "poor"}, "c": {"overall": "good"}}
        result = ev.agreement(judge, human)
        self.assertEqual(result["per_dimension"]["overall"], {"n": 2, "agree": 2, "rate": 1.0})
        self.assertEqual(result["per_dimension"]["balance"]["rate"], 0.0)
        self.assertEqual(result["mismatches"], [("a", "balance", "poor", "good")])


class ScoreSinkTest(unittest.TestCase):
    def _report(self, tmp, linked=None):
        path = Path(tmp) / "SKHY_20261016_203424"
        (path / "0_reference").mkdir(parents=True)
        if linked:
            (path / "0_reference" / "langfuse.json").write_text(json.dumps({"trace_id": linked}), encoding="utf-8")
        return ev.Report(path.name, "SKHY", "2026-10-16", "Underweight", path)

    def test_scores_written_once_on_linked_trace(self):
        client = MagicMock()
        sink = ev.ScoreSink(client)
        with tempfile.TemporaryDirectory() as tmp:
            report = self._report(tmp, linked="abc123")
            state = {}
            sink.score(report, state, "rating_alpha_5d", -1.2, "NUMERIC", "c")
            sink.score(report, state, "rating_alpha_5d", -1.2, "NUMERIC", "c")
        client.create_score.assert_called_once()
        kwargs = client.create_score.call_args.kwargs
        self.assertEqual(kwargs["trace_id"], "abc123")
        self.assertEqual(kwargs["score_id"], ev._score_id(report.report_id, "rating_alpha_5d"))
        client.start_observation.assert_not_called()
        self.assertEqual(state["scores"], {"rating_alpha_5d": -1.2})

    def test_backfill_trace_is_deterministic(self):
        import contextlib
        import sys
        import types

        langfuse = types.ModuleType("langfuse")
        langfuse.propagate_attributes = lambda **kw: contextlib.nullcontext()
        client = MagicMock()
        client.create_trace_id.side_effect = lambda seed: f"tid-{seed}"
        sink = ev.ScoreSink(client)
        with tempfile.TemporaryDirectory() as tmp, unittest.mock.patch.dict(sys.modules, {"langfuse": langfuse}):
            report = self._report(tmp)
            state = {}
            self.assertEqual(sink.trace_for(report, state), f"tid-{report.report_id}")
            self.assertEqual(sink.trace_for(report, state), f"tid-{report.report_id}")
        client.start_observation.assert_called_once()
        self.assertEqual(client.start_observation.call_args.kwargs["trace_context"],
                         {"trace_id": f"tid-{report.report_id}"})
        self.assertTrue(state["backfilled"])

    def test_disabled_sink_only_records_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self._report(tmp)
            state = {}
            ev.ScoreSink(None).score(report, state, "data_misread_count", 3, "NUMERIC")
        self.assertEqual(state["scores"], {"data_misread_count": 3})


class EvaluateReportTest(unittest.TestCase):
    def test_full_pass_scores_each_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "GOOGL_20261026_105901"
            (path / "0_reference").mkdir(parents=True)
            (path / "0_reference" / "langfuse.json").write_text('{"trace_id": "t1"}', encoding="utf-8")
            (path / "complete_report.md").write_text("\n".join([_GOOGL_YOY_BAD, _CROSS, _TOUCH]), encoding="utf-8")
            report = ev.Report(path.name, "GOOGL", "2026-10-26", "Underweight", path)
            ctx = ev.EvalContext(
                fetch_returns=lambda *a: (0.02, -0.01, 5, "2026-11-02"),
                benchmark_for=lambda t: "SPY",
                revenue_yoy=lambda t, d: _GOOGL_REVENUE_YOY,
                cpi_yoy=lambda d: 3.72,
                judge=lambda r, facts: {k: {"label": "acceptable", "reason": facts[:10]}
                                        for k in (*ev.DIMENSIONS, "overall")},
                known_tickers=["SKHY"],
            )
            client = MagicMock()
            state = {}
            original = ev.history_rows_at
            ev.history_rows_at = lambda t, d: 1254
            try:
                ev.evaluate_report(report, state, ctx, ev.ScoreSink(client))
                calls = client.create_score.call_count
                ev.evaluate_report(report, state, ctx, ev.ScoreSink(client))  # 第二次不重复写
            finally:
                ev.history_rows_at = original
        names = {c.kwargs["name"] for c in client.create_score.call_args_list}
        self.assertEqual(client.create_score.call_count, calls)
        self.assertTrue({"rating_alpha_5d", "rating_hit_5d", "rating_alpha_20d", "data_misread_count",
                         "misread_yoy", "misread_cross_ticker", "misread_touch_market", "report_quality",
                         "report_balance"} <= names)
        scores = state["scores"]
        self.assertEqual(scores[f"data_misread_count@{ev.RULES_VERSION}"], 3)
        self.assertEqual(scores["rating_hit_5d"], 1)


if __name__ == "__main__":
    unittest.main()
