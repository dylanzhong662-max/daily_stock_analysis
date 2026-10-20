# -*- coding: utf-8 -*-
"""让 TradingAgents 的境外社交/预测市场数据源走代理。

国内机房直连 Reddit 超时、Polymarket 不可达；设置 ``TRADINGAGENTS_SOCIAL_PROXY``
（如 ``http://host.docker.internal:8118``）后，只有这几个数据源经代理访问，
FMP、SEC、DeepSeek 等仍直连。

- Reddit / StockTwits：替换模块内的 ``urlopen``，经代理请求；Reddit 对代理出口 IP
  的 429 限流较频繁，这里额外做几次退避重试（TradingAgents 自身只重试一次）。
- Polymarket：替换模块内的 ``requests``，给 ``get`` 带上 proxies。
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from types import SimpleNamespace
from typing import Optional
from urllib.error import HTTPError
from urllib.request import ProxyHandler, build_opener

logger = logging.getLogger(__name__)

SOCIAL_PROXY_ENV = "TRADINGAGENTS_SOCIAL_PROXY"
_RETRY_429_ATTEMPTS = 3
_RETRY_429_BASE_SECONDS = 4.0

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
                wait = _RETRY_429_BASE_SECONDS * attempt * random.uniform(0.8, 1.2)
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
