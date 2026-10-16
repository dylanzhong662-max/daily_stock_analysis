# -*- coding: utf-8 -*-
"""Tests for the optional TradingAgents deep-research integration."""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

try:
    import litellm  # noqa: F401
except ModuleNotFoundError:
    sys.modules["litellm"] = MagicMock()

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.v1.endpoints import tradingagents as ta_endpoint
from src.services import tradingagents_service as svc
from src.services.tradingagents_service import (
    TradingAgentsError,
    TradingAgentsService,
    build_tradingagents_config,
    rating_to_operation_advice,
    resolve_llm_route,
    to_yfinance_ticker,
)

_TA_ENV_KEYS = [
    "TRADINGAGENTS_LLM_PROVIDER",
    "TRADINGAGENTS_DEEP_THINK_LLM",
    "TRADINGAGENTS_QUICK_THINK_LLM",
    "TRADINGAGENTS_LLM_BACKEND_URL",
    "TRADINGAGENTS_OUTPUT_LANGUAGE",
    "TRADINGAGENTS_RESULTS_DIR",
    "TRADINGAGENTS_CACHE_DIR",
    "TRADINGAGENTS_MEMORY_LOG_PATH",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_COMPATIBLE_API_KEY",
    "FMP_API_KEY",
    "TRADINGAGENTS_DATA_SOURCE",
]


def _clean_env():
    env = {k: v for k, v in os.environ.items() if k not in _TA_ENV_KEYS}
    return patch.dict(os.environ, env, clear=True)


def _config(**overrides):
    base = dict(
        litellm_model="",
        llm_model_list=[],
        report_language="zh",
        gemini_api_keys=[],
        anthropic_api_keys=[],
        deepseek_api_keys=[],
        openai_api_keys=[],
        openai_base_url=None,
        tradingagents_enabled=True,
        tradingagents_analysts=["market", "news"],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _default_ta_config():
    return {
        "results_dir": "~/.tradingagents/logs",
        "data_cache_dir": "~/.tradingagents/cache",
        "memory_log_path": "~/.tradingagents/memory/trading_memory.md",
        "llm_provider": "openai",
        "deep_think_llm": "gpt-6-sol",
        "quick_think_llm": "gpt-6-luna",
        "backend_url": None,
        "output_language": "English",
    }


def _fake_tradingagents_modules(graph_cls):
    pkg = types.ModuleType("tradingagents")
    pkg.__version__ = "0.6.0"
    graph_module = types.ModuleType("tradingagents.graph.trading_graph")
    graph_module.TradingAgentsGraph = graph_cls
    config_module = types.ModuleType("tradingagents.default_config")
    config_module.build_default_config = _default_ta_config
    return {
        "tradingagents": pkg,
        "tradingagents.graph": types.ModuleType("tradingagents.graph"),
        "tradingagents.graph.trading_graph": graph_module,
        "tradingagents.default_config": config_module,
    }


class TickerConversionTest(unittest.TestCase):
    def test_converts_dsa_codes_to_yahoo(self):
        self.assertEqual(to_yfinance_ticker("600519"), "600519.SS")
        self.assertEqual(to_yfinance_ticker("000001"), "000001.SZ")
        self.assertEqual(to_yfinance_ticker("hk00700"), "0700.HK")
        self.assertEqual(to_yfinance_ticker("aapl"), "AAPL")

    def test_empty_code_rejected(self):
        with self.assertRaises(TradingAgentsError) as ctx:
            to_yfinance_ticker("  ")
        self.assertEqual(ctx.exception.code, "tradingagents_invalid_code")


class LlmRouteTest(unittest.TestCase):
    def test_gemini_maps_to_google(self):
        route = resolve_llm_route(_config(litellm_model="gemini/gemini-3-pro", gemini_api_keys=["g" * 20]))
        self.assertEqual(route["provider"], "google")
        self.assertEqual(route["model"], "gemini-3-pro")
        self.assertEqual(route["api_key"], "g" * 20)

    def test_anthropic_and_deepseek(self):
        self.assertEqual(resolve_llm_route(_config(litellm_model="anthropic/claude-sonnet-5"))["provider"], "anthropic")
        self.assertEqual(resolve_llm_route(_config(litellm_model="deepseek/deepseek-chat"))["provider"], "deepseek")

    def test_openai_with_base_url(self):
        route = resolve_llm_route(
            _config(
                litellm_model="openai/gpt-6-luna",
                openai_api_keys=["sk-" + "x" * 20],
                openai_base_url="https://relay.example.com/v1",
            )
        )
        self.assertEqual(route["provider"], "openai")
        self.assertEqual(route["backend_url"], "https://relay.example.com/v1")

    def test_channel_alias_resolves_through_model_list(self):
        config = _config(
            litellm_model="my-alias",
            llm_model_list=[
                {
                    "model_name": "my-alias",
                    "litellm_params": {
                        "model": "dashscope/qwen-max",
                        "api_key": "k" * 20,
                        "api_base": "https://dashscope.example.com/v1",
                    },
                }
            ],
        )
        route = resolve_llm_route(config)
        self.assertEqual(route["provider"], "openai_compatible")
        self.assertEqual(route["model"], "qwen-max")
        self.assertEqual(route["backend_url"], "https://dashscope.example.com/v1")

    def test_unknown_provider_without_base_url_rejected(self):
        with self.assertRaises(TradingAgentsError) as ctx:
            resolve_llm_route(_config(litellm_model="dashscope/qwen-max"))
        self.assertEqual(ctx.exception.code, "tradingagents_llm_unsupported")

    def test_missing_model_rejected(self):
        with self.assertRaises(TradingAgentsError) as ctx:
            resolve_llm_route(_config())
        self.assertEqual(ctx.exception.code, "tradingagents_llm_not_configured")


class BuildConfigTest(unittest.TestCase):
    def test_maps_dsa_model_and_isolates_paths(self):
        with _clean_env():
            cfg = build_tradingagents_config(
                _config(litellm_model="gemini/gemini-3-pro", gemini_api_keys=["g" * 20]),
                _default_ta_config,
            )
            self.assertEqual(os.environ.get("GOOGLE_API_KEY"), "g" * 20)
        self.assertEqual(cfg["llm_provider"], "google")
        self.assertEqual(cfg["deep_think_llm"], "gemini-3-pro")
        self.assertEqual(cfg["quick_think_llm"], "gemini-3-pro")
        self.assertEqual(cfg["output_language"], "Chinese")
        for key in ("results_dir", "data_cache_dir", "memory_log_path"):
            self.assertTrue(Path(cfg[key]).as_posix().startswith("data/tradingagents/"), cfg[key])

    def test_existing_api_key_env_not_overwritten(self):
        with _clean_env():
            os.environ["GOOGLE_API_KEY"] = "user-key"
            build_tradingagents_config(
                _config(litellm_model="gemini/gemini-3-pro", gemini_api_keys=["g" * 20]),
                _default_ta_config,
            )
            self.assertEqual(os.environ["GOOGLE_API_KEY"], "user-key")

    def test_tradingagents_env_overrides_take_priority(self):
        def defaults_with_env():
            cfg = _default_ta_config()
            cfg.update(deep_think_llm="claude-opus-5-5", results_dir="/custom/logs")
            return cfg

        with _clean_env():
            os.environ["TRADINGAGENTS_DEEP_THINK_LLM"] = "claude-opus-5-5"
            os.environ["TRADINGAGENTS_RESULTS_DIR"] = "/custom/logs"
            cfg = build_tradingagents_config(
                _config(litellm_model="anthropic/claude-sonnet-5"), defaults_with_env
            )
        self.assertEqual(cfg["deep_think_llm"], "claude-opus-5-5")
        self.assertEqual(cfg["quick_think_llm"], "claude-sonnet-5")
        self.assertEqual(cfg["results_dir"], "/custom/logs")

    def test_explicit_provider_skips_dsa_mapping(self):
        with _clean_env():
            os.environ["TRADINGAGENTS_LLM_PROVIDER"] = "openai"
            # 未配置 LITELLM_MODEL 也不应报错
            cfg = build_tradingagents_config(_config(), _default_ta_config)
        self.assertEqual(cfg["deep_think_llm"], "gpt-6-sol")


class RatingMappingTest(unittest.TestCase):
    def test_five_tier_ratings(self):
        expected = {"Buy": "买入", "Overweight": "加仓", "Hold": "持有", "Underweight": "减仓", "Sell": "卖出"}
        for rating, advice in expected.items():
            self.assertEqual(rating_to_operation_advice(rating, "zh"), advice)

    def test_review_maps_to_wait(self):
        self.assertEqual(rating_to_operation_advice("REVIEW", "zh"), "观望")

    def test_english_report_language(self):
        self.assertEqual(rating_to_operation_advice("Buy", "en"), "Buy")


class ServiceTest(unittest.TestCase):
    def test_disabled(self):
        with self.assertRaises(TradingAgentsError) as ctx:
            TradingAgentsService(_config(tradingagents_enabled=False)).ensure_ready()
        self.assertEqual(ctx.exception.code, "tradingagents_disabled")
        self.assertEqual(ctx.exception.status_code, 403)

    def test_not_installed(self):
        with patch.object(svc.importlib, "import_module", side_effect=ImportError("no module")):
            with self.assertRaises(TradingAgentsError) as ctx:
                TradingAgentsService(_config()).ensure_ready()
        self.assertEqual(ctx.exception.code, "tradingagents_not_installed")

    def test_status_reports_availability(self):
        with patch.object(svc.importlib, "import_module", side_effect=ImportError("no module")):
            status = TradingAgentsService(_config(tradingagents_enabled=False)).status()
        self.assertEqual(status["enabled"], False)
        self.assertEqual(status["available"], False)

    def test_invalid_analysts_and_dates(self):
        graph_cls = MagicMock()
        with patch.dict(sys.modules, _fake_tradingagents_modules(graph_cls)):
            service = TradingAgentsService(_config(litellm_model="anthropic/claude-sonnet-5"))
            with self.assertRaises(TradingAgentsError) as ctx:
                service.analyze("AAPL", analysts=["astrology"])
            self.assertEqual(ctx.exception.code, "tradingagents_invalid_analysts")
            future = (date.today() + timedelta(days=2)).isoformat()
            for bad_date in ("2026/01/01", future):
                with self.assertRaises(TradingAgentsError) as ctx:
                    service.analyze("AAPL", trade_date=bad_date)
                self.assertEqual(ctx.exception.code, "tradingagents_invalid_date")
        graph_cls.assert_not_called()

    def test_analyze_runs_graph_and_collects_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_md = Path(tmp) / "complete_report.md"
            report_md.write_text("# report", encoding="utf-8")
            report_md.with_suffix(".html").write_text("<html></html>", encoding="utf-8")

            graph = MagicMock()
            graph.propagate.return_value = (
                {
                    "market_report": "市场",
                    "news_report": "新闻",
                    "final_trade_decision": "**Rating**: Overweight",
                },
                "Overweight",
            )
            graph.save_reports.return_value = report_md
            graph_cls = MagicMock(return_value=graph)

            with _clean_env(), patch.dict(sys.modules, _fake_tradingagents_modules(graph_cls)):
                service = TradingAgentsService(_config(litellm_model="anthropic/claude-sonnet-5"))
                result = service.analyze("hk00700", trade_date="2026-09-01")

        graph_cls.assert_called_once()
        self.assertEqual(graph_cls.call_args.kwargs["selected_analysts"], ("market", "news"))
        self.assertEqual(graph_cls.call_args.kwargs["config"]["llm_provider"], "anthropic")
        graph.propagate.assert_called_once_with("0700.HK", "2026-09-01")
        self.assertEqual(result["rating"], "Overweight")
        self.assertEqual(result["operation_advice"], "加仓")
        self.assertEqual(result["ticker"], "0700.HK")
        self.assertEqual(result["market_report"], "市场")
        self.assertEqual(result["sentiment_report"], "")
        self.assertTrue(result["report_html_path"].endswith("complete_report.html"))
        self.assertIn("Overweight", svc.format_notification(result))


class EndpointTest(unittest.TestCase):
    def _client(self, config):
        app = FastAPI()
        app.include_router(ta_endpoint.router, prefix="/api/v1/tradingagents")
        app.dependency_overrides[ta_endpoint.get_config_dep] = lambda: config
        return TestClient(app)

    def test_start_task_rejected_when_disabled(self):
        response = self._client(_config(tradingagents_enabled=False)).post(
            "/api/v1/tradingagents/tasks", json={"stock_code": "AAPL"}
        )
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"]["error"], "tradingagents_disabled")

    def test_start_task_submits_background_job(self):
        queue = MagicMock()
        queue.submit_background_task.return_value = SimpleNamespace(
            task_id="t1", trace_id="t1", status="pending", message="TradingAgents 研究任务已提交"
        )
        with (
            patch.object(ta_endpoint.TradingAgentsService, "ensure_ready", return_value=None),
            patch.object(ta_endpoint, "get_task_queue", return_value=queue),
        ):
            response = self._client(_config()).post(
                "/api/v1/tradingagents/tasks", json={"stock_code": "AAPL", "analysts": ["market"]}
            )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["task_id"], "t1")
        kwargs = queue.submit_background_task.call_args.kwargs
        self.assertEqual(kwargs["report_type"], "tradingagents")
        self.assertEqual(kwargs["stock_code"], "AAPL")

    def test_task_status_not_found_for_other_report_type(self):
        queue = MagicMock()
        queue.get_task.return_value = SimpleNamespace(report_type="screening_screen")
        with patch.object(ta_endpoint, "get_task_queue", return_value=queue):
            response = self._client(_config()).get("/api/v1/tradingagents/tasks/abc")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
