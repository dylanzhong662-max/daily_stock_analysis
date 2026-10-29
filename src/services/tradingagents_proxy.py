# -*- coding: utf-8 -*-
"""让 TradingAgents 的境外社交/预测市场数据源走代理。

国内机房直连 Reddit 超时、Polymarket 不可达；设置 ``TRADINGAGENTS_SOCIAL_PROXY``
（如 ``http://host.docker.internal:8118``）后，只有这几个数据源经代理访问，
FMP、SEC、DeepSeek 等仍直连。

- Reddit / StockTwits：替换模块内的 ``urlopen``，经代理请求；Reddit 对代理出口 IP
  的 429 限流较频繁，这里额外做几次退避重试（TradingAgents 自身只重试一次）。
- Polymarket：替换模块内的 ``requests``，给 ``get`` 带上 proxies。

模型档位也可以单独走代理：``TRADINGAGENTS_DEEP_THINK_PROXY`` / ``TRADINGAGENTS_QUICK_THINK_PROXY``
只给对应档位的 LLM 客户端带上代理（例如 deep 档用境外网关上的 GPT，quick 档的 DeepSeek 仍直连）。
"""

from __future__ import annotations

import logging
import os
import random
import re
import threading
import time
from types import SimpleNamespace
from typing import Optional
from urllib.error import HTTPError
from urllib.request import ProxyHandler, build_opener

logger = logging.getLogger(__name__)

SOCIAL_PROXY_ENV = "TRADINGAGENTS_SOCIAL_PROXY"
DISABLE_STOCKTWITS_ENV = "TRADINGAGENTS_DISABLE_STOCKTWITS"
# 匿名 RSS 约每分钟 1 次/IP，共享代理出口常被 429；退避 10s/20s（TradingAgents 自身再按 Retry-After 重试一次）
_RETRY_429_ATTEMPTS = 3
_RETRY_429_BASE_SECONDS = 10.0

_lock = threading.Lock()
_installed_proxy: Optional[str] = None


def resolve_social_proxy() -> Optional[str]:
    value = (os.environ.get(SOCIAL_PROXY_ENV) or "").strip()
    return value or None


def _proxied_urlopen(proxy: str, *, retry_429: bool):
    opener = build_opener(ProxyHandler({"http": proxy, "https": proxy}))

    def urlopen(req, timeout=None):
        attempts = _RETRY_429_ATTEMPTS if retry_429 else 1
        for attempt in range(1, attempts + 1):
            try:
                return opener.open(req, timeout=timeout)
            except HTTPError as exc:
                if exc.code != 429 or attempt == attempts:
                    raise
                wait = _RETRY_429_BASE_SECONDS * attempt * random.uniform(0.9, 1.1)
                logger.info("[TradingAgents] proxy 429 for %s, retry %d/%d in %.1fs",
                            getattr(req, "host", "?"), attempt, attempts - 1, wait)
                time.sleep(wait)
        raise RuntimeError("unreachable")

    return urlopen


def _proxied_requests(proxy: str):
    import requests

    proxies = {"http": proxy, "https": proxy}

    def get(*args, **kwargs):
        kwargs.setdefault("proxies", proxies)
        return requests.get(*args, **kwargs)

    return SimpleNamespace(get=get, RequestException=requests.RequestException)


def install_social_proxy(proxy: Optional[str] = None) -> Optional[str]:
    """按 ``TRADINGAGENTS_SOCIAL_PROXY`` 给 Reddit/StockTwits/Polymarket 装代理（幂等）。"""
    global _installed_proxy
    proxy = proxy or resolve_social_proxy()
    if not proxy:
        return None
    with _lock:
        if _installed_proxy == proxy:
            return proxy
        from tradingagents.dataflows.vendors import polymarket, reddit, stocktwits

        reddit.urlopen = _proxied_urlopen(proxy, retry_429=True)
        stocktwits.urlopen = _proxied_urlopen(proxy, retry_429=False)
        polymarket.requests = _proxied_requests(proxy)
        _installed_proxy = proxy
        logger.info("[TradingAgents] social/prediction-market sources routed via proxy")
        return proxy


TIER_PROXY_ENVS = {"deep": "TRADINGAGENTS_DEEP_THINK_PROXY", "quick": "TRADINGAGENTS_QUICK_THINK_PROXY"}
# 推理模型单次调用可能要几分钟
_LLM_PROXY_TIMEOUT_SECONDS = 600
_TIER_PATCHED_ATTR = "_dsa_tier_proxy"


def resolve_tier_proxies() -> dict:
    return {tier: value for tier, env in TIER_PROXY_ENVS.items()
            if (value := (os.environ.get(env) or "").strip())}


def install_llm_tier_proxy() -> dict:
    """给设置了代理的模型档位注入走代理的 httpx 客户端（幂等；每次建图时按当前环境变量生效）。

    TradingAgents 的 OpenAI 兼容客户端会把 ``http_client`` / ``http_async_client`` 透传给 ChatOpenAI。
    """
    proxies = resolve_tier_proxies()
    if not proxies:
        return {}
    from tradingagents.graph import trading_graph

    with _lock:
        original = trading_graph.create_tier_client
        if getattr(original, _TIER_PATCHED_ATTR, False):
            return proxies

        def create_tier_client(config, tier, **extra):
            proxy = resolve_tier_proxies().get(tier)
            if proxy:
                import httpx

                extra.setdefault("http_client", httpx.Client(proxy=proxy, timeout=_LLM_PROXY_TIMEOUT_SECONDS))
                extra.setdefault("http_async_client",
                                 httpx.AsyncClient(proxy=proxy, timeout=_LLM_PROXY_TIMEOUT_SECONDS))
            return original(config, tier, **extra)

        setattr(create_tier_client, _TIER_PATCHED_ATTR, True)
        trading_graph.create_tier_client = create_tier_client
    logger.info("[TradingAgents] LLM tiers routed via proxy: %s", ",".join(sorted(proxies)))
    return proxies


_GPT_MODEL_RE = re.compile(r"^(?:openai/)?gpt-\d", re.I)
_gpt_structured_installed = False


def install_gpt_structured_output() -> None:
    """经 OpenAI 兼容网关（openai_compatible）调用 GPT 时，结构化输出改用 json_schema（幂等）。

    该 provider 的客户端会去掉 tool_choice；GPT 在长 prompt 下常直接写正文、不调 schema 工具，
    结构化结果为空、每个经理节点白白多一次自由文本重试。json_schema 由服务端强制格式。
    """
    global _gpt_structured_installed
    with _lock:
        if _gpt_structured_installed:
            return
        from tradingagents.llm_clients import capabilities

        original = capabilities.get_capabilities
        json_schema = capabilities.ModelCapabilities(
            supports_tool_choice=True, supports_json_mode=True, supports_json_schema=True,
            preferred_structured_method="json_schema",
        )

        def get_capabilities(model_name: str):
            return json_schema if _GPT_MODEL_RE.match(model_name or "") else original(model_name)

        capabilities.get_capabilities = get_capabilities
        from tradingagents.llm_clients import openai_client

        # openai_client 是 from-import，需要一起替换
        openai_client.get_capabilities = get_capabilities
        _gpt_structured_installed = True


def _stocktwits_disabled_block(ticker: str, *args, **kwargs) -> str:
    return (
        f"<StockTwits not queried for {ticker}: the source blocks this server (HTTP 403), "
        f"so it is disabled. This is not an absence of discussion; rely on the other sources.>"
    )


def disable_stocktwits_if_configured() -> bool:
    """``TRADINGAGENTS_DISABLE_STOCKTWITS=true`` 时跳过 StockTwits（服务器 IP 被拦截时省去必然失败的请求）。"""
    if (os.environ.get(DISABLE_STOCKTWITS_ENV) or "").strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    from tradingagents.agents.analysts import sentiment_analyst

    sentiment_analyst.fetch_stocktwits_messages = _stocktwits_disabled_block
    return True


# ---------------------------------------------------------------------------
# Reddit 覆盖面：默认只搜 3 个版块、只按代码搜、每版块最多 5 帖，样本太小
# ---------------------------------------------------------------------------

REDDIT_SUBREDDITS = ("wallstreetbets", "stocks", "investing", "StockMarket", "semiconductors", "ValueInvesting")
# 已确认存在的个股版块（不存在的版块会让合并搜索整体失败，所以只用白名单）
TICKER_SUBREDDITS = {"NVDA": "NVDA_Stock", "AMD": "AMD_Stock"}
REDDIT_POSTS_PER_SUB = 8
_NAME_SUFFIX_RE = re.compile(
    r"[,.]?\s+(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|holdings?|group|n\.?v|s\.?a|ag)\.?$",
    re.IGNORECASE,
)
_search_names: dict = {}


def company_search_name(company_name: str) -> str:
    """'NVIDIA Corporation' -> 'NVIDIA'，'Broadcom Inc.' -> 'Broadcom'。"""
    name = (company_name or "").strip()
    for _ in range(3):
        stripped = _NAME_SUFFIX_RE.sub("", name).strip()
        if stripped == name:
            break
        name = stripped
    return name


def tune_reddit(ticker: str, company_name: Optional[str] = None) -> None:
    """扩大情绪分析师的 Reddit 覆盖：更多版块、按“代码 OR 公司名”搜、每版块 8 帖。"""
    from urllib.parse import urlencode

    from tradingagents.agents.analysts import sentiment_analyst
    from tradingagents.dataflows.vendors import reddit

    symbol = (ticker or "").split(".")[0].upper()
    name = company_search_name(company_name or "")
    if symbol and name and name.upper() != symbol:
        _search_names[symbol] = name

    if getattr(sentiment_analyst, "_dsa_reddit_tuned", False):
        return
    original_subreddits_for = reddit.subreddits_for
    original_fetch = reddit.fetch_reddit_posts

    def subreddits_for(t: str):
        default = original_subreddits_for(t)
        if tuple(default) != tuple(reddit.DEFAULT_SUBREDDITS):
            return default  # 加密货币等特殊映射保持原样
        extra = TICKER_SUBREDDITS.get((t or "").split(".")[0].upper())
        return REDDIT_SUBREDDITS + ((extra,) if extra else ())

    def search_qs(t: str, limit: int) -> str:
        name_ = _search_names.get((t or "").upper())
        query = f'{t} OR "{name_}"' if name_ else t
        return urlencode({"q": query, "restrict_sr": "on", "sort": "new", "t": "week", "limit": limit})

    def fetch_reddit_posts(t, subreddits=None, **kwargs):
        kwargs.setdefault("limit_per_sub", REDDIT_POSTS_PER_SUB)
        return original_fetch(t, subreddits, **kwargs)

    reddit._search_qs = search_qs
    sentiment_analyst.subreddits_for = subreddits_for
    sentiment_analyst.fetch_reddit_posts = fetch_reddit_posts
    sentiment_analyst._dsa_reddit_tuned = True
