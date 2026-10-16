# -*- coding: utf-8 -*-
"""
美股选股服务（FMP）
==================
用 Financial Modeling Prep 的 company-screener 按条件筛选全美股候选，
可选做「动量过滤」（价格站上 50/200 日均线）并排序。

供 API 端点 /api/v1/screener/us 使用；仅依赖 requests + FMP_API_KEY。
"""
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

FMP_BASE = "https://financialmodelingprep.com/stable"

# 预设筛选条件（均限定 country=US、剔除 ETF/基金、仅活跃交易）
PRESETS: Dict[str, Dict[str, Any]] = {
    "largecap": {
        "marketCapMoreThan": 50_000_000_000,
        "priceMoreThan": 10,
        "volumeMoreThan": 2_000_000,
    },
    "growth_tech": {
        "marketCapMoreThan": 10_000_000_000,
        "sector": "Technology",
        "priceMoreThan": 10,
        "volumeMoreThan": 1_000_000,
    },
    "value": {
        "marketCapMoreThan": 10_000_000_000,
        "priceMoreThan": 10,
        "dividendMoreThan": 0.01,
        "betaLowerThan": 1.2,
    },
}


def _api_key(config: Any = None) -> str:
    key = ""
    if config is not None:
        key = (getattr(config, "fmp_api_key", None) or "").strip()
    return key or (os.getenv("FMP_API_KEY", "") or "").strip()


def _run_screener(params: Dict[str, Any], limit: int, key: str) -> List[Dict[str, Any]]:
    q = dict(params)
    q.update({
        "country": q.get("country", "US"),
        "isActivelyTrading": "true",
        "isEtf": "false",
        "isFund": "false",
        "limit": limit,
        "apikey": key,
    })
    resp = requests.get(f"{FMP_BASE}/company-screener", params=q, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and data.get("Error Message"):
        raise RuntimeError(f"FMP screener 报错: {data['Error Message']}")
    if not isinstance(data, list):
        raise RuntimeError(f"FMP screener 返回异常: {str(data)[:200]}")
    return data


def _fetch_quote(sym: str, key: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    try:
        q = requests.get(
            f"{FMP_BASE}/quote", params={"symbol": sym, "apikey": key}, timeout=8
        ).json()
        if isinstance(q, list) and q:
            return sym, q[0]
    except Exception:
        pass
    return sym, None


def _momentum_rank(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    """保留站上 50 日均线（且不低于 200 日均线）的上升趋势股，按高出 50 日线幅度排序。

    行情查询并发执行（10 线程），避免逐个串行导致 HTTP 超时。
    """
    symbols = [r.get("symbol") for r in rows if r.get("symbol")]
    if not symbols:
        return []
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(lambda s: _fetch_quote(s, key), symbols))
    quote_map = {sym: q for sym, q in results if q}

    scored: List[Dict[str, Any]] = []
    for row in rows:
        q0 = quote_map.get(row.get("symbol"))
        if not q0:
            continue
        price = q0.get("price")
        ma50 = q0.get("priceAvg50")
        ma200 = q0.get("priceAvg200")
        if not (price and ma50) or price <= ma50:
            continue
        if ma200 and price < ma200:
            continue
        row["above_ma50_pct"] = round((price / ma50 - 1) * 100, 2)
        row["price"] = price
        scored.append(row)
    scored.sort(key=lambda r: r.get("above_ma50_pct", 0), reverse=True)
    return scored


def screen_us_stocks(
    *,
    preset: str = "largecap",
    sector: Optional[str] = None,
    market_cap_min: Optional[float] = None,
    price_min: Optional[float] = None,
    volume_min: Optional[float] = None,
    exchange: Optional[str] = None,
    top: int = 10,
    momentum: bool = False,
    screen_limit: int = 40,
    config: Any = None,
) -> Dict[str, Any]:
    """执行美股选股，返回 {candidates, count, momentum, preset}。"""
    key = _api_key(config)
    if not key:
        raise RuntimeError("未配置 FMP_API_KEY")

    params = dict(PRESETS.get(preset, PRESETS["largecap"]))
    if sector:
        params["sector"] = sector
    if market_cap_min is not None:
        params["marketCapMoreThan"] = int(market_cap_min)
    if price_min is not None:
        params["priceMoreThan"] = price_min
    if volume_min is not None:
        params["volumeMoreThan"] = int(volume_min)
    if exchange:
        params["exchange"] = exchange

    rows = _run_screener(params, screen_limit, key)
    logger.info("[美股选股] preset=%s 初筛=%d momentum=%s", preset, len(rows), momentum)

    if momentum:
        rows = _momentum_rank(rows, key)
    else:
        # 无动量时按市值降序（稳定的默认排序，避免额外请求）
        rows.sort(key=lambda r: (r.get("marketCap") or 0), reverse=True)

    top_rows = rows[: max(1, top)]
    candidates = [
        {
            "symbol": r.get("symbol"),
            "name": r.get("companyName") or "",
            "sector": r.get("sector") or "",
            "industry": r.get("industry") or "",
            "marketCap": r.get("marketCap"),
            "price": r.get("price"),
            "beta": r.get("beta"),
            "exchange": r.get("exchangeShortName") or r.get("exchange") or "",
            "aboveMa50Pct": r.get("above_ma50_pct"),
        }
        for r in top_rows
        if r.get("symbol")
    ]
    return {
        "preset": preset,
        "momentum": momentum,
        "count": len(candidates),
        "candidates": candidates,
    }
