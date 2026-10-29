# -*- coding: utf-8 -*-
"""TradingAgents 多智能体深度研究集成。

TradingAgents (https://github.com/TauricResearch/TradingAgents) 是可选依赖，
需要单独安装 ``requirements-tradingagents.txt``；未安装时本模块仍可导入，
只在真正运行分析时报错。

LLM 配置默认从 DSA 的 ``LITELLM_MODEL`` / API Key / ``OPENAI_BASE_URL`` 推导，
用户显式设置的 ``TRADINGAGENTS_*`` 环境变量优先。
"""

from __future__ import annotations

import gc
import importlib
import json
import logging
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.config import (
    Config,
    _get_litellm_provider,
    extra_litellm_params,
    get_api_keys_for_model,
)
from src.report_language import localize_operation_advice, normalize_report_language
from src.services.tradingagents_polymarket import default_topics, ticker_topic_enabled

logger = logging.getLogger(__name__)

TRADINGAGENTS_DATA_DIR = Path("data") / "tradingagents"
SUPPORTED_ANALYSTS = ("market", "social", "news", "fundamentals")

# LiteLLM provider prefix -> TradingAgents provider
_PROVIDER_MAP = {
    "gemini": "google",
    "vertex_ai": "google",
    "anthropic": "anthropic",
    "deepseek": "deepseek",
    "openai": "openai",
    "openrouter": "openrouter",
    "xai": "xai",
    "groq": "groq",
    "mistral": "mistral",
    "ollama": "ollama",
}

# TradingAgents provider -> API key env var it reads
# (tradingagents/llm_clients/api_key_env.py)
_PROVIDER_KEY_ENV = {
    "google": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "xai": "XAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "openai_compatible": "OPENAI_COMPATIBLE_API_KEY",
}

_REPORT_LANGUAGE_TO_OUTPUT = {"zh": "Chinese", "en": "English", "ko": "Korean"}

# TradingAgents 5 档评级 -> DSA 操作建议（中文 canonical，再按报告语言本地化）
_RATING_TO_ADVICE = {
    "buy": "买入",
    "overweight": "加仓",
    "hold": "持有",
    "underweight": "减仓",
    "sell": "卖出",
}
_REVIEW_ADVICE = "观望"

_STATE_REPORT_KEYS = (
    "market_report",
    "sentiment_report",
    "news_report",
    "fundamentals_report",
    "investment_plan",
    "trader_investment_plan",
    "final_trade_decision",
)


# 报告目录结构（tradingagents/reporting.py::write_report_tree）
REPORT_SECTIONS = (
    ("polymarket", "Polymarket 预测市场参考", "0_reference/polymarket.md"),
    ("market", "技术面分析师", "1_analysts/market.md"),
    ("sentiment", "情绪分析师", "1_analysts/sentiment.md"),
    ("news", "新闻分析师", "1_analysts/news.md"),
    ("fundamentals", "基本面分析师", "1_analysts/fundamentals.md"),
    ("bull", "多方研究员", "2_research/bull.md"),
    ("bear", "空方研究员", "2_research/bear.md"),
    ("research_manager", "研究经理", "2_research/manager.md"),
    ("trader", "交易员", "3_trading/trader.md"),
    ("aggressive", "激进风控", "4_risk/aggressive.md"),
    ("conservative", "保守风控", "4_risk/conservative.md"),
    ("neutral", "中性风控", "4_risk/neutral.md"),
    ("decision", "组合经理决策", "5_portfolio/decision.md"),
)
_REPORT_ID_RE = re.compile(r"^(?P<ticker>[A-Za-z0-9.^=-]+)_(?P<stamp>\d{8}_\d{6})$")
_RATING_LINE_RE = re.compile(r"\*\*Rating\*\*\s*[:：]\s*([A-Za-z]+)")


def reports_root() -> Path:
    results_dir = (os.environ.get("TRADINGAGENTS_RESULTS_DIR") or "").strip()
    base = Path(results_dir) if results_dir else TRADINGAGENTS_DATA_DIR / "logs"
    return base / "reports"


def _parse_rating(decision_path: Path) -> Optional[str]:
    try:
        match = _RATING_LINE_RE.search(decision_path.read_text(encoding="utf-8")[:4000])
    except OSError:
        return None
    return match.group(1) if match else None


def _report_dir(report_id: str) -> Path:
    match = _REPORT_ID_RE.match(report_id or "")
    root = reports_root().resolve()
    path = (root / report_id).resolve() if match else None
    if path is None or path.parent != root or not path.is_dir():
        raise TradingAgentsError("tradingagents_report_not_found", f"报告不存在: {report_id}", status_code=404)
    return path


def _report_summary(path: Path, report_language: str) -> Dict[str, Any]:
    match = _REPORT_ID_RE.match(path.name)
    rating = _parse_rating(path / "5_portfolio" / "decision.md")
    created_at = datetime.strptime(match.group("stamp"), "%Y%m%d_%H%M%S").isoformat() if match else None
    return {
        "report_id": path.name,
        "ticker": match.group("ticker") if match else path.name,
        "created_at": created_at,
        "rating": rating,
        "operation_advice": rating_to_operation_advice(rating or "", report_language),
        "has_html": (path / "complete_report.html").is_file(),
    }


class TradingAgentsError(Exception):
    """可直接展示给用户的 TradingAgents 集成错误。"""

    def __init__(self, code: str, message: str, *, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _env_is_set(name: str) -> bool:
    return bool((os.environ.get(name) or "").strip())


def _import_tradingagents() -> Tuple[Any, Any]:
    """Return (TradingAgentsGraph, build_default_config) or raise TradingAgentsError."""
    try:
        graph_module = importlib.import_module("tradingagents.graph.trading_graph")
        config_module = importlib.import_module("tradingagents.default_config")
    except ImportError as exc:
        raise TradingAgentsError(
            "tradingagents_not_installed",
            "未安装 TradingAgents，请执行 pip install -r requirements-tradingagents.txt（需要 Python 3.11+、pandas 3）",
            status_code=503,
        ) from exc
    return graph_module.TradingAgentsGraph, config_module.build_default_config


def to_yfinance_ticker(code: str) -> str:
    """把 DSA 股票代码（600519 / hk00700 / AAPL）转换成 TradingAgents 使用的 Yahoo 代码。"""
    from data_provider.yfinance_fetcher import YfinanceFetcher

    normalized = (code or "").strip()
    if not normalized:
        raise TradingAgentsError("tradingagents_invalid_code", "股票代码不能为空")
    return YfinanceFetcher()._convert_stock_code(normalized)


def resolve_llm_route(config: Config) -> Dict[str, Any]:
    """从 DSA 的主模型配置推导 TradingAgents 的 provider / model / backend_url / api_key。

    优先查 LLM_CHANNELS / LITELLM_CONFIG 生成的 Router model_list（model_name 别名），
    否则按 legacy 单模型配置解析。
    """
    model = (getattr(config, "litellm_model", "") or "").strip()
    if not model:
        raise TradingAgentsError(
            "tradingagents_llm_not_configured",
            "未配置 LITELLM_MODEL，且未设置 TRADINGAGENTS_LLM_PROVIDER / TRADINGAGENTS_*_LLM",
        )

    real_model = model
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    for entry in getattr(config, "llm_model_list", None) or []:
        if not isinstance(entry, dict) or entry.get("model_name") != model:
            continue
        params = entry.get("litellm_params") or {}
        real_model = str(params.get("model") or model)
        api_key = params.get("api_key") or None
        api_base = params.get("api_base") or None
        break
    else:
        keys = get_api_keys_for_model(model, config)
        api_key = keys[0] if keys else None
        api_base = extra_litellm_params(model, config).get("api_base")

    prefix = _get_litellm_provider(real_model)
    model_name = real_model.split("/", 1)[1] if "/" in real_model else real_model
    provider = _PROVIDER_MAP.get(prefix)
    if provider is None:
        if not api_base:
            raise TradingAgentsError(
                "tradingagents_llm_unsupported",
                f"无法把 LiteLLM 模型 {real_model!r} 映射到 TradingAgents provider，"
                "请设置 TRADINGAGENTS_LLM_PROVIDER 等变量",
            )
        provider = "openai_compatible"

    return {
        "provider": provider,
        "model": model_name,
        "backend_url": api_base,
        "api_key": api_key,
    }


def build_tradingagents_config(config: Config, build_default_config: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    """生成 TradingAgents 配置：以其默认配置（已含 TRADINGAGENTS_* 覆盖）为底，只补未显式设置的项。"""
    ta_config = build_default_config()

    if not _env_is_set("TRADINGAGENTS_RESULTS_DIR"):
        ta_config["results_dir"] = str(TRADINGAGENTS_DATA_DIR / "logs")
    if not _env_is_set("TRADINGAGENTS_CACHE_DIR"):
        ta_config["data_cache_dir"] = str(TRADINGAGENTS_DATA_DIR / "cache")
    if not _env_is_set("TRADINGAGENTS_MEMORY_LOG_PATH"):
        ta_config["memory_log_path"] = str(TRADINGAGENTS_DATA_DIR / "memory" / "trading_memory.md")

    if not _env_is_set("TRADINGAGENTS_OUTPUT_LANGUAGE"):
        language = normalize_report_language(getattr(config, "report_language", "zh"))
        ta_config["output_language"] = _REPORT_LANGUAGE_TO_OUTPUT.get(language, "English")

    if _env_is_set("TRADINGAGENTS_LLM_PROVIDER"):
        # 用户完全接管 LLM 配置
        return ta_config

    route = resolve_llm_route(config)
    ta_config["llm_provider"] = route["provider"]
    if not _env_is_set("TRADINGAGENTS_DEEP_THINK_LLM"):
        ta_config["deep_think_llm"] = route["model"]
    if not _env_is_set("TRADINGAGENTS_QUICK_THINK_LLM"):
        ta_config["quick_think_llm"] = route["model"]
    if route["backend_url"] and not _env_is_set("TRADINGAGENTS_LLM_BACKEND_URL"):
        ta_config["backend_url"] = route["backend_url"]

    # TradingAgents 只从环境变量读取 API Key；仅在对应变量未设置时补齐，不覆盖用户值。
    key_env = _PROVIDER_KEY_ENV.get(route["provider"])
    if key_env and route["api_key"] and not _env_is_set(key_env):
        os.environ[key_env] = route["api_key"]
    return ta_config


def rating_to_operation_advice(rating: str, report_language: str = "zh") -> str:
    """5 档评级转成操作建议。

    中文直接用 5 档映射（DSA 的 localize_operation_advice 会把「加仓」归一成「买入」，
    丢失 Overweight 与 Buy 的区别）；其他语言保留 TradingAgents 原评级，REVIEW 走本地化的「观望」。
    """
    key = (rating or "").strip().lower()
    if normalize_report_language(report_language) == "zh":
        return _RATING_TO_ADVICE.get(key, _REVIEW_ADVICE)
    if key in _RATING_TO_ADVICE:
        return key.capitalize()
    return localize_operation_advice(_REVIEW_ADVICE, report_language)


def _resolve_analysts(analysts: Optional[List[str]], default: List[str]) -> List[str]:
    selected = [a.strip().lower() for a in (analysts or default) if a and a.strip()]
    unknown = [a for a in selected if a not in SUPPORTED_ANALYSTS]
    if unknown or not selected:
        raise TradingAgentsError(
            "tradingagents_invalid_analysts",
            f"analysts 只能取 {','.join(SUPPORTED_ANALYSTS)}，收到: {','.join(unknown) or '(空)'}",
        )
    return selected


def resolve_data_source(config: Config) -> str:
    """TRADINGAGENTS_DATA_SOURCE=auto|fmp|yfinance；auto 在配置了 FMP_API_KEY 时用 FMP。"""
    from src.services.tradingagents_fmp import resolve_fmp_api_key

    requested = (os.environ.get("TRADINGAGENTS_DATA_SOURCE") or "auto").strip().lower()
    has_fmp = bool(resolve_fmp_api_key(config))
    if requested == "fmp":
        if not has_fmp:
            raise TradingAgentsError(
                "tradingagents_fmp_not_configured",
                "TRADINGAGENTS_DATA_SOURCE=fmp 但未配置 FMP_API_KEY",
            )
        return "fmp"
    if requested == "yfinance":
        return "yfinance"
    if requested != "auto":
        raise TradingAgentsError(
            "tradingagents_invalid_data_source",
            f"TRADINGAGENTS_DATA_SOURCE 只能是 auto/fmp/yfinance，收到: {requested}",
        )
    return "fmp" if has_fmp else "yfinance"


def _resolve_trade_date(trade_date: Optional[str]) -> str:
    if not trade_date:
        return date.today().isoformat()
    try:
        parsed = date.fromisoformat(trade_date.strip())
    except ValueError as exc:
        raise TradingAgentsError("tradingagents_invalid_date", f"日期格式应为 YYYY-MM-DD，收到: {trade_date}") from exc
    if parsed > date.today():
        raise TradingAgentsError("tradingagents_invalid_date", f"分析日期不能晚于今天: {trade_date}")
    return parsed.isoformat()


# 页面可选的 deep 档模型（研究经理/组合经理）。服务端白名单，浏览器只能从中选，不能传任意地址或 key。
# 格式：逗号分隔的 provider:model[@proxy]，proxy 为 "proxy" 时沿用 TRADINGAGENTS_DEEP_THINK_PROXY；
# 第一项之外的 openai_compatible 选项沿用 TRADINGAGENTS_DEEP_THINK_BACKEND_URL / OPENAI_COMPATIBLE_API_KEY。
# 例：openai_compatible:gpt-6.1-sol@proxy,deepseek:deepseek-v4-pro,deepseek:deepseek-flash
DEEP_MODEL_OPTIONS_ENV = "TRADINGAGENTS_DEEP_MODEL_OPTIONS"
_PROVIDER_LABELS = {"openai_compatible": "aihubmix", "deepseek": "DeepSeek", "openai": "OpenAI",
                    "anthropic": "Anthropic", "google": "Google"}


def deep_model_options() -> List[Dict[str, Any]]:
    """解析 ``TRADINGAGENTS_DEEP_MODEL_OPTIONS``；未配置时返回空列表（页面不显示选择框）。"""
    options = []
    for raw in (os.environ.get(DEEP_MODEL_OPTIONS_ENV) or "").split(","):
        item = raw.strip()
        if not item or ":" not in item:
            continue
        spec, _, proxy = item.partition("@")
        provider, _, model = spec.partition(":")
        provider, model = provider.strip().lower(), model.strip()
        if not provider or not model:
            continue
        options.append({
            "id": f"{provider}:{model}",
            "provider": provider,
            "model": model,
            "proxied": proxy.strip().lower() == "proxy",
            "label": f"{model}（{_PROVIDER_LABELS.get(provider, provider)}）",
        })
    return options


def apply_deep_model(ta_config: Dict[str, Any], deep_model: Optional[str]) -> Dict[str, Any]:
    """按页面选择覆盖 deep 档 provider/model/代理；未选择时保持环境变量配置。"""
    if not deep_model:
        return ta_config
    option = next((o for o in deep_model_options() if o["id"] == deep_model), None)
    if option is None:
        raise TradingAgentsError(
            "tradingagents_invalid_model",
            f"不支持的模型: {deep_model}，可选: {', '.join(o['id'] for o in deep_model_options()) or '(未配置)'}",
        )
    ta_config["deep_think_provider"] = option["provider"]
    ta_config["deep_think_llm"] = option["model"]
    if option["provider"] != "openai_compatible":
        # 回到原生 provider（如 DeepSeek）时不要沿用网关地址
        ta_config["deep_think_backend_url"] = None
    ta_config["deep_think_proxy"] = (os.environ.get("TRADINGAGENTS_DEEP_THINK_PROXY") or "").strip() or None \
        if option["proxied"] else None
    return ta_config


MAX_POLYMARKET_TOPICS = 5


def _normalize_topics(topics: Optional[List[str]]) -> List[str]:
    cleaned: List[str] = []
    for topic in topics or []:
        text = " ".join(str(topic or "").split())[:80]
        if text and text.lower() not in {t.lower() for t in cleaned}:
            cleaned.append(text)
    if len(cleaned) > MAX_POLYMARKET_TOPICS:
        raise TradingAgentsError(
            "tradingagents_invalid_polymarket_topics",
            f"Polymarket 参考主题最多 {MAX_POLYMARKET_TOPICS} 个",
        )
    return cleaned


def _inject_context(graph: Any, extra: str) -> None:
    """把参考块（带标记）追加到 instrument_context；只有新闻分析师与组合经理读到全文。"""
    from src.services.tradingagents_polymarket import install_reference_scope, mark_reference

    try:
        install_reference_scope()
    except Exception as exc:  # 作用域补丁失败时退回旧行为：所有角色都读到参考块
        logger.warning("[TradingAgents] Polymarket reference scope skipped: %s", exc)
    original = graph.create_run_state
    block = mark_reference(extra)

    def create_run_state(*args, **kwargs):
        state = original(*args, **kwargs)
        state["instrument_context"] = f"{state.get('instrument_context') or ''}\n\n{block}".strip()
        return state

    graph.create_run_state = create_run_state


def _inject_shared_context(graph: Any, extra: str) -> None:
    """把说明块原样追加到 instrument_context，所有角色都能读到（如盘后跳空提示）。"""
    original = graph.create_run_state

    def create_run_state(*args, **kwargs):
        state = original(*args, **kwargs)
        state["instrument_context"] = f"{state.get('instrument_context') or ''}\n\n{extra}".strip()
        return state

    graph.create_run_state = create_run_state


def _extended_hours_note(ticker: str, run_date: str, data_source: str) -> str:
    """当天研究且用 FMP 时，取盘前/盘后跳空提示；历史日期或取不到时返回空串。"""
    if data_source != "fmp" or run_date != date.today().isoformat():
        return ""
    from src.services.tradingagents_fmp import extended_hours_note

    try:
        return extended_hours_note(ticker)
    except Exception as exc:  # 只是补充提示，失败不影响研究
        logger.info("[TradingAgents] extended-hours quote unavailable for %s: %s", ticker, exc)
        return ""


class TradingAgentsService:
    """调用 TradingAgents 完成单只股票的多智能体研究。"""

    def __init__(self, config: Config):
        self.config = config

    def status(self) -> Dict[str, Any]:
        try:
            module = importlib.import_module("tradingagents")
            available, version = True, getattr(module, "__version__", None)
        except ImportError:
            available, version = False, None
        return {
            "enabled": bool(getattr(self.config, "tradingagents_enabled", False)),
            "available": available,
            "version": version,
            "analysts": list(getattr(self.config, "tradingagents_analysts", SUPPORTED_ANALYSTS)),
            "polymarket_default_topics": default_topics(),
            "polymarket_ticker_topic": ticker_topic_enabled(),
            "deep_model_options": [{"id": o["id"], "label": o["label"]} for o in deep_model_options()],
            "deep_model_default": self._default_deep_model(),
        }

    @staticmethod
    def _default_deep_model() -> Optional[str]:
        """环境变量当前配置的 deep 档对应的选项 id（不在白名单里时为 None）。"""
        model = (os.environ.get("TRADINGAGENTS_DEEP_THINK_LLM") or "").strip()
        provider = (os.environ.get("TRADINGAGENTS_DEEP_THINK_PROVIDER") or os.environ.get("TRADINGAGENTS_LLM_PROVIDER")
                    or "").strip().lower()
        for option in deep_model_options():
            if option["model"] == model and (not provider or option["provider"] == provider):
                return option["id"]
        return None

    def ensure_ready(self) -> None:
        if not getattr(self.config, "tradingagents_enabled", False):
            raise TradingAgentsError(
                "tradingagents_disabled",
                "TradingAgents 未启用，请设置 TRADINGAGENTS_ENABLED=true",
                status_code=403,
            )
        _import_tradingagents()

    def analyze(
        self,
        code: str,
        *,
        trade_date: Optional[str] = None,
        analysts: Optional[List[str]] = None,
        polymarket_topics: Optional[List[str]] = None,
        use_default_polymarket: bool = True,
        deep_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        self.ensure_ready()
        topics = _normalize_topics(polymarket_topics)
        graph_cls, build_default_config = _import_tradingagents()

        ticker = to_yfinance_ticker(code)
        selected = _resolve_analysts(analysts, list(getattr(self.config, "tradingagents_analysts", [])))
        run_date = _resolve_trade_date(trade_date)
        ta_config = apply_deep_model(build_tradingagents_config(self.config, build_default_config), deep_model)
        data_source = resolve_data_source(self.config)
        if data_source == "fmp":
            from src.services.tradingagents_fmp import apply_fmp_config, install_fmp_vendor, resolve_fmp_api_key

            install_fmp_vendor(resolve_fmp_api_key(self.config))
            apply_fmp_config(ta_config)
        from src.services.tradingagents_quality import install_quality_patches

        # 在 FMP vendor 之后安装，才能一并包装它的报表实现
        try:
            install_quality_patches()
        except Exception as exc:  # TradingAgents 内部结构变化时降级为原始工具输出，研究照常进行
            logger.warning("[TradingAgents] data quality patches skipped: %s", exc)
        from src.services.tradingagents_proxy import (
            disable_stocktwits_if_configured,
            install_gpt_structured_output,
            install_llm_tier_proxy,
            install_social_proxy,
        )

        install_social_proxy()
        # 不吞异常：配了档位代理却没装上，该档模型必然连不上，直接报错比跑到一半失败更清楚
        install_llm_tier_proxy()
        install_gpt_structured_output()
        company_name = None
        if data_source == "fmp":
            from src.services.tradingagents_fmp import get_company_profile

            try:
                company_name = get_company_profile(ticker).get("longName")
            except Exception as exc:  # 公司名只用于扩大 Reddit 搜索，失败就只按代码搜
                logger.info("[TradingAgents] company name lookup failed for %s: %s", ticker, exc)
        from src.services.tradingagents_proxy import tune_reddit

        try:
            tune_reddit(ticker, company_name)
        except Exception as exc:  # 只是扩大情绪数据覆盖，失败不影响研究
            logger.warning("[TradingAgents] Reddit tuning skipped: %s", exc)
        disable_stocktwits_if_configured()

        logger.info(
            "[TradingAgents] start code=%s ticker=%s date=%s provider=%s deep=%s quick=%s analysts=%s data=%s",
            code, ticker, run_date, ta_config.get("llm_provider"),
            ta_config.get("deep_think_llm"), ta_config.get("quick_think_llm"), ",".join(selected), data_source,
        )
        graph = graph_cls(selected_analysts=tuple(selected), config=ta_config)
        from src.services.tradingagents_polymarket import build_reference

        try:
            reference, used_topics = build_reference(ticker, run_date, topics, use_default_polymarket)
        except Exception as exc:  # 参考信息是增强项，失败不影响研究
            logger.warning("[TradingAgents] Polymarket reference unavailable: %s", exc)
            reference, used_topics = "", []
        if reference:
            _inject_context(graph, reference)
        extended = _extended_hours_note(ticker, run_date, data_source)
        if extended:
            logger.info("[TradingAgents] extended-hours move for %s injected", ticker)
            _inject_shared_context(graph, extended)
        from src.services.tradingagents_langfuse import research_trace, trace_url

        trace_meta = {
            "llm_provider": ta_config.get("llm_provider"),
            "deep_think_llm": ta_config.get("deep_think_llm"),
            "quick_think_llm": ta_config.get("quick_think_llm"),
            "data_source": data_source,
            "polymarket_topics": ",".join(used_topics),
        }
        with research_trace(ticker=ticker, trade_date=run_date, analysts=selected, metadata=trace_meta) as trace:
            trace.attach(graph)
            try:
                state, rating = graph.propagate(ticker, run_date)
                report_md_path = Path(graph.save_reports(state, ticker))
            except Exception as exc:
                trace.finish(error=exc)
                raise
            trace.finish(rating=rating, report_id=report_md_path.parent.name)
            url = trace_url(trace)
            if url:
                logger.info("[TradingAgents] Langfuse trace: %s", url)
            if trace.trace_id:
                # 评估任务据此把分数挂到这次研究的 trace 上
                ref_dir = report_md_path.parent / "0_reference"
                ref_dir.mkdir(parents=True, exist_ok=True)
                (ref_dir / "langfuse.json").write_text(
                    json.dumps({"trace_id": trace.trace_id, "trace_url": url}), encoding="utf-8")
        for name, block in (("polymarket.md", reference), ("extended_hours.md", extended)):
            if block:
                ref_dir = report_md_path.parent / "0_reference"
                ref_dir.mkdir(parents=True, exist_ok=True)
                (ref_dir / name).write_text(block, encoding="utf-8")
        report_html_path = report_md_path.with_suffix(".html")
        del graph
        gc.collect()

        report_language = normalize_report_language(getattr(self.config, "report_language", "zh"))
        result: Dict[str, Any] = {
            "code": code,
            "ticker": ticker,
            "trade_date": run_date,
            "analysts": selected,
            "polymarket_topics": used_topics,
            "rating": rating,
            "operation_advice": rating_to_operation_advice(rating, report_language),
            "llm_provider": ta_config.get("llm_provider"),
            "deep_think_llm": ta_config.get("deep_think_llm"),
            "quick_think_llm": ta_config.get("quick_think_llm"),
            "data_source": data_source,
            "report_id": report_md_path.parent.name,
            "report_md_path": str(report_md_path),
            "report_html_path": str(report_html_path) if report_html_path.exists() else None,
        }
        for key in _STATE_REPORT_KEYS:
            value = state.get(key) if isinstance(state, dict) else None
            result[key] = value if isinstance(value, str) else ""
        logger.info("[TradingAgents] done ticker=%s rating=%s report=%s", ticker, rating, report_md_path)
        return result


def list_reports(config: Config, limit: int = 50) -> List[Dict[str, Any]]:
    """最近的 TradingAgents 报告（按目录名时间倒序）。"""
    root = reports_root()
    if not root.is_dir():
        return []
    language = normalize_report_language(getattr(config, "report_language", "zh"))
    dirs = [p for p in root.iterdir() if p.is_dir() and _REPORT_ID_RE.match(p.name)]
    dirs.sort(key=lambda p: _REPORT_ID_RE.match(p.name).group("stamp"), reverse=True)
    return [_report_summary(p, language) for p in dirs[: max(1, limit)]]


def read_report(config: Config, report_id: str) -> Dict[str, Any]:
    """读取单份报告的各阶段 Markdown。"""
    path = _report_dir(report_id)
    language = normalize_report_language(getattr(config, "report_language", "zh"))
    summary = _report_summary(path, language)
    sections = []
    for key, title, rel in REPORT_SECTIONS:
        file = path / rel
        if file.is_file():
            sections.append({"key": key, "title": title, "content": file.read_text(encoding="utf-8")})
    summary["sections"] = sections
    return summary


def report_html_path(report_id: str) -> Path:
    html = _report_dir(report_id) / "complete_report.html"
    if not html.is_file():
        raise TradingAgentsError("tradingagents_report_not_found", f"报告 {report_id} 没有 HTML 版本", status_code=404)
    return html


def format_notification(result: Dict[str, Any], *, max_chars: int = 1500) -> str:
    """生成推送用的简短 Markdown 摘要。"""
    decision = (result.get("final_trade_decision") or "").strip()
    if len(decision) > max_chars:
        decision = decision[:max_chars].rstrip() + "…"
    lines = [
        f"## 🧠 TradingAgents 深度研究：{result.get('code')} ({result.get('ticker')})",
        "",
        f"**评级**：{result.get('rating')}（{result.get('operation_advice')}）",
        f"**分析日期**：{result.get('trade_date')}",
        "",
        decision,
        "",
        f"完整报告：`{result.get('report_html_path') or result.get('report_md_path')}`",
    ]
    return "\n".join(lines)
