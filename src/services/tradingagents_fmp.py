# -*- coding: utf-8 -*-
"""TradingAgents 的 FMP（Financial Modeling Prep）数据适配。

TradingAgents 默认走 Yahoo Finance，但国内机房 IP 会被 Yahoo 429 限流。
本模块在不修改 TradingAgents 包的前提下，把它的数据来源换成 FMP stable API：

- 行情（OHLCV / 技术指标 / 已核验行情快照 / 历史决策复盘收盘价）：
  替换 ``vendors.yahoo.ohlcv`` 与 ``vendors.yahoo.market`` 模块里的 ``yf``，
  价格请求改由 FMP ``historical-price-eod/dividend-adjusted`` 提供。
- 基本面概览、新闻、全球新闻、内部人交易、三大报表：注册为 ``fmp`` vendor，
  通过 tool_vendors 选用（报表优先 SEC EDGAR as-filed，再回退 FMP）。
- 公司身份（名称/行业/交易所）：替换 ``agents.context.get_company_profile``。

只在 TradingAgents 已安装时导入使用；补丁是进程级、幂等的。
当前 FMP 套餐只覆盖美股，非美股代码会返回“数据不可用”而不是编造。
"""

from __future__ import annotations

import io
import logging
import os
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

FMP_BASE_URL = "https://financialmodelingprep.com/stable"
_TIMEOUT_SECONDS = 20

_install_lock = threading.Lock()
_installed = False


def resolve_fmp_api_key(config: Any = None) -> Optional[str]:
    key = (getattr(config, "fmp_api_key", None) or os.getenv("FMP_API_KEY") or "").strip()
    return key or None


def _errors():
    from tradingagents.dataflows import errors

    return errors


class FmpClient:
    """最小 FMP stable API 客户端；不走系统代理（服务器代理会破坏到 FMP 的 TLS）。"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or resolve_fmp_api_key()
        self.session = requests.Session()
        self.session.trust_env = False
        # trust_env=False 也会忽略 CA 变量，这里显式保留自定义 CA（公司网络 MITM 等场景）
        self.session.verify = os.getenv("REQUESTS_CA_BUNDLE") or os.getenv("SSL_CERT_FILE") or True
        self.session.headers["User-Agent"] = "daily-stock-analysis/tradingagents"

    def get(self, path: str, **params: Any) -> List[Dict[str, Any]]:
        errors = _errors()
        if not self.api_key:
            raise errors.VendorNotConfiguredError("FMP_API_KEY is not configured")
        query = {k: v for k, v in params.items() if v is not None}
        query["apikey"] = self.api_key
        try:
            resp = self.session.get(f"{FMP_BASE_URL}/{path}", params=query, timeout=_TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            raise errors.VendorUnavailableError(f"FMP request failed: {type(exc).__name__}") from exc
        if resp.status_code != 200:
            raise errors.VendorUnavailableError(f"FMP {path} returned HTTP {resp.status_code}")
        try:
            data = resp.json()
        except ValueError as exc:
            # 套餐不覆盖的代码（如港股/A股）返回纯文本提示而非 JSON
            snippet = resp.text.strip()[:120]
            raise errors.VendorUnavailableError(f"FMP {path} is not available on this plan: {snippet}") from exc
        if isinstance(data, dict):
            message = data.get("Error Message") or data.get("error")
            if message:
                raise errors.VendorUnavailableError(f"FMP {path} error: {message}")
            return [data]
        return data if isinstance(data, list) else []


_client: Optional[FmpClient] = None


def _get_client() -> FmpClient:
    global _client
    if _client is None:
        _client = FmpClient()
    return _client


def _fmp_symbol(symbol: str) -> str:
    return (symbol or "").strip().upper()


# ---------------------------------------------------------------------------
# 行情：伪装成 yfinance 的 Ticker.history()
# ---------------------------------------------------------------------------

def fetch_history(symbol: str, start: Optional[str] = None, end: Optional[str] = None,
                  auto_adjust: bool = True) -> pd.DataFrame:
    """FMP 日线，返回与 ``yf.Ticker(...).history()`` 相同形状的 DataFrame。

    ``end`` 与 yfinance 一致为开区间。空结果抛 NoMarketDataError。
    """
    errors = _errors()
    canonical = _fmp_symbol(symbol)
    if auto_adjust:
        rows = _get_client().get("historical-price-eod/dividend-adjusted", symbol=canonical,
                                 **{"from": start, "to": end})
        mapping = {"adjOpen": "Open", "adjHigh": "High", "adjLow": "Low", "adjClose": "Close", "volume": "Volume"}
    else:
        rows = _get_client().get("historical-price-eod/full", symbol=canonical, **{"from": start, "to": end})
        mapping = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    if not rows:
        raise errors.NoMarketDataError(symbol, canonical, "no price rows from FMP")

    frame = pd.DataFrame(rows)
    if "date" not in frame.columns:
        raise errors.NoMarketDataError(symbol, canonical, "FMP price rows carry no date")
    frame = frame.rename(columns=mapping)
    frame["Date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if end:
        frame = frame[frame["Date"] < pd.Timestamp(end)]
    if start:
        frame = frame[frame["Date"] >= pd.Timestamp(start)]
    columns = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in frame.columns]
    frame = frame.sort_values("Date").set_index("Date")[columns]
    if frame.empty:
        raise errors.NoMarketDataError(symbol, canonical, f"no FMP price rows between {start} and {end}")
    return frame


class _FmpTicker:
    def __init__(self, symbol: str):
        self.ticker = symbol

    def history(self, start=None, end=None, auto_adjust=True, actions=False, **_ignored):
        return fetch_history(self.ticker, _date_str(start), _date_str(end), auto_adjust=auto_adjust)


class _FmpYfShim:
    """替换 TradingAgents yahoo 行情模块中的 ``yf``：只实现它们用到的 ``Ticker().history``。"""

    Ticker = _FmpTicker


def _date_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value[:10]
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _passthrough_retry(func, *_args, **_kwargs):
    """替换 yf_retry：让 FMP 抛出的 NoMarketData/VendorUnavailable 原样冒泡给路由层。"""
    return func()


# ---------------------------------------------------------------------------
# 注册到 TradingAgents 路由的 fmp vendor 实现
# ---------------------------------------------------------------------------

def get_company_profile(ticker: str) -> dict:
    """Yahoo ``info`` 字段名的公司身份，供 TradingAgents 锁定分析对象。"""
    rows = _get_client().get("profile", symbol=_fmp_symbol(ticker))
    if not rows:
        return {}
    p = rows[0]
    return {
        "longName": p.get("companyName"),
        "shortName": p.get("companyName"),
        "sector": p.get("sector"),
        "industry": p.get("industry"),
        "exchange": p.get("exchange"),
        "quoteType": "ETF" if p.get("isEtf") else ("MUTUALFUND" if p.get("isFund") else "EQUITY"),
        "currency": p.get("currency"),
    }


def get_fundamentals(ticker: str, as_of_date: Optional[str] = None) -> str:
    from tradingagents.dataflows.date_window import withhold_live_profile

    errors = _errors()
    canonical = _fmp_symbol(ticker)
    withheld = withhold_live_profile(as_of_date, canonical)
    if withheld:
        return withheld

    client = _get_client()
    profile = (client.get("profile", symbol=canonical) or [{}])[0]
    if not profile.get("companyName"):
        raise errors.NoMarketDataError(ticker, canonical, "no FMP profile")

    def optional(path: str) -> Dict[str, Any]:
        try:
            return (client.get(path, symbol=canonical) or [{}])[0]
        except errors.VendorError as exc:
            logger.info("[TradingAgents/FMP] %s unavailable for %s: %s", path, canonical, exc)
            return {}

    quote = optional("quote")
    ratios = optional("ratios-ttm")
    metrics = optional("key-metrics-ttm")
    currency = profile.get("currency")

    def money(value):
        return None if value is None else f"{value} {currency}" if currency else value

    fields = [
        ("Name", profile.get("companyName")),
        ("Sector", profile.get("sector")),
        ("Industry", profile.get("industry")),
        ("Exchange", profile.get("exchange")),
        ("Market Cap", money(profile.get("marketCap"))),
        ("Price", money(profile.get("price"))),
        ("52 Week Range", profile.get("range")),
        ("50 Day Average", money(quote.get("priceAvg50"))),
        ("200 Day Average", money(quote.get("priceAvg200"))),
        ("Beta", profile.get("beta")),
        ("PE Ratio (TTM)", ratios.get("priceToEarningsRatioTTM")),
        ("PEG Ratio (TTM)", ratios.get("priceToEarningsGrowthRatioTTM")),
        ("Price to Book (TTM)", ratios.get("priceToBookRatioTTM")),
        ("Price to Sales (TTM)", ratios.get("priceToSalesRatioTTM")),
        ("EV to EBITDA (TTM)", metrics.get("evToEBITDATTM")),
        ("EPS (TTM)", money(ratios.get("netIncomePerShareTTM"))),
        ("Dividend Yield (TTM, fraction)", ratios.get("dividendYieldTTM")),
        ("Gross Margin (TTM)", ratios.get("grossProfitMarginTTM")),
        ("Operating Margin (TTM)", ratios.get("operatingProfitMarginTTM")),
        ("Net Margin (TTM)", ratios.get("netProfitMarginTTM")),
        ("Return on Equity (TTM)", metrics.get("returnOnEquityTTM")),
        ("Return on Assets (TTM)", metrics.get("returnOnAssetsTTM")),
        ("Debt to Equity (TTM)", ratios.get("debtToEquityRatioTTM")),
        ("Current Ratio (TTM)", ratios.get("currentRatioTTM")),
        ("Free Cash Flow Yield (TTM)", metrics.get("freeCashFlowYieldTTM")),
    ]
    lines = [f"{label}: {value}" for label, value in fields if value not in (None, "")]
    return f"# Company Fundamentals for {canonical} (source: FMP)\n\n" + "\n".join(lines)


def _statement(path: str, title: str, ticker: str, freq: str, as_of_date: Optional[str]) -> str:
    errors = _errors()
    canonical = _fmp_symbol(ticker)
    period = "annual" if (freq or "").lower() == "annual" else "quarter"
    rows = _get_client().get(path, symbol=canonical, period=period, limit=8)
    if as_of_date:
        # 按申报日做 point-in-time 过滤：只给分析日当时已公开的报表
        rows = [r for r in rows if str(r.get("filingDate") or r.get("acceptedDate") or "")[:10] <= as_of_date]
    if not rows:
        raise errors.NoMarketDataError(ticker, canonical, f"no FMP {title.lower()} filed by {as_of_date}")
    frame = pd.DataFrame(rows)
    drop = [c for c in ("symbol", "cik", "link", "finalLink") if c in frame.columns]
    frame = frame.drop(columns=drop).set_index("date").T
    buf = io.StringIO()
    frame.to_csv(buf)
    return f"# {title} data for {canonical} ({freq}, source: FMP, as filed by {as_of_date or 'today'})\n" + buf.getvalue()


def get_balance_sheet(ticker: str, freq: str = "quarterly", as_of_date: Optional[str] = None) -> str:
    return _statement("balance-sheet-statement", "Balance Sheet", ticker, freq, as_of_date)


def get_cashflow(ticker: str, freq: str = "quarterly", as_of_date: Optional[str] = None) -> str:
    return _statement("cash-flow-statement", "Cash Flow", ticker, freq, as_of_date)


def get_income_statement(ticker: str, freq: str = "quarterly", as_of_date: Optional[str] = None) -> str:
    return _statement("income-statement", "Income Statement", ticker, freq, as_of_date)


def _format_news(rows: List[Dict[str, Any]]) -> str:
    out = []
    for row in rows:
        source = row.get("publisher") or row.get("site") or "Unknown"
        published = str(row.get("publishedDate") or "")[:16]
        block = f"### {row.get('title') or 'No title'} (source: {source}, {published})\n"
        if row.get("text"):
            block += f"{row['text']}\n"
        if row.get("url"):
            block += f"Link: {row['url']}\n"
        out.append(block)
    return "\n".join(out)


def _news_window(rows: List[Dict[str, Any]], start_date: str, end_date: str) -> List[Dict[str, Any]]:
    end_exclusive = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    return [r for r in rows if start_date <= str(r.get("publishedDate") or "")[:10] < end_exclusive]


def get_news(ticker: str, start_date: str, end_date: str) -> str:
    from tradingagents.dataflows.config import get_config

    canonical = _fmp_symbol(ticker)
    limit = get_config().get("news_article_limit", 20)
    rows = _get_client().get("news/stock", symbols=canonical, limit=limit, **{"from": start_date, "to": end_date})
    rows = _news_window(rows, start_date, end_date)
    if not rows:
        return f"No news found for {canonical} between {start_date} and {end_date} (source: FMP)"
    return f"## {canonical} News, from {start_date} to {end_date} (source: FMP):\n\n{_format_news(rows)}"


def get_global_news(as_of_date: str, look_back_days: Optional[int] = None, limit: Optional[int] = None) -> str:
    from tradingagents.dataflows.config import get_config

    config = get_config()
    look_back_days = look_back_days if look_back_days is not None else config.get("global_news_lookback_days", 7)
    limit = limit if limit is not None else config.get("global_news_article_limit", 10)
    start_date = (datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=look_back_days)).strftime("%Y-%m-%d")
    rows = _get_client().get("news/general-latest", limit=max(limit * 3, 20), **{"from": start_date, "to": as_of_date})
    rows = _news_window(rows, start_date, as_of_date)[:limit]
    if not rows:
        return f"No global news found between {start_date} and {as_of_date} (source: FMP)"
    return f"## Global Market News, from {start_date} to {as_of_date} (source: FMP):\n\n{_format_news(rows)}"


def get_insider_transactions(ticker: str, as_of_date: Optional[str] = None) -> str:
    canonical = _fmp_symbol(ticker)
    rows = _get_client().get("insider-trading/search", symbol=canonical, limit=50)
    if as_of_date:
        # 按申报日过滤：分析日当时还没披露的交易不能给
        rows = [r for r in rows if str(r.get("filingDate") or "")[:10] <= as_of_date]
    if not rows:
        return f"No insider transactions filed for {canonical} by {as_of_date or 'today'} (source: FMP)"
    keep = ["filingDate", "transactionDate", "reportingName", "typeOfOwner", "transactionType",
            "acquisitionOrDisposition", "securitiesTransacted", "price", "securitiesOwned"]
    frame = pd.DataFrame(rows)[[c for c in keep if c in rows[0]]]
    return f"# Insider transactions for {canonical} (source: FMP, filed by {as_of_date or 'today'})\n" + frame.to_csv(index=False)


# ---------------------------------------------------------------------------
# 安装补丁 + 配置
# ---------------------------------------------------------------------------

FMP_TOOL_VENDORS = {
    "get_fundamentals": "fmp",
    "get_balance_sheet": "sec_edgar,fmp",
    "get_cashflow": "sec_edgar,fmp",
    "get_income_statement": "sec_edgar,fmp",
    "get_news": "fmp",
    "get_global_news": "fmp",
    "get_insider_transactions": "fmp",
}


def install_fmp_vendor(api_key: Optional[str] = None) -> None:
    """把 TradingAgents 的 Yahoo 数据通道切到 FMP（幂等）。"""
    global _installed, _client
    with _install_lock:
        if api_key and (_client is None or _client.api_key != api_key):
            _client = FmpClient(api_key)
        if _installed:
            return
        from tradingagents.agents import context as ta_context
        from tradingagents.dataflows import router
        from tradingagents.dataflows.vendors.yahoo import market as yahoo_market
        from tradingagents.dataflows.vendors.yahoo import ohlcv as yahoo_ohlcv

        for module in (yahoo_ohlcv, yahoo_market):
            module.yf = _FmpYfShim
            module.yf_retry = _passthrough_retry
        ta_context.get_company_profile = get_company_profile

        vendor_impls = {
            "get_fundamentals": get_fundamentals,
            "get_balance_sheet": get_balance_sheet,
            "get_cashflow": get_cashflow,
            "get_income_statement": get_income_statement,
            "get_news": get_news,
            "get_global_news": get_global_news,
            "get_insider_transactions": get_insider_transactions,
        }
        for method, impl in vendor_impls.items():
            router.VENDOR_METHODS[method]["fmp"] = impl
        _installed = True
        logger.info("[TradingAgents] FMP data vendor installed (prices/fundamentals/news via FMP)")


def apply_fmp_config(ta_config: Dict[str, Any]) -> Dict[str, Any]:
    tool_vendors = dict(ta_config.get("tool_vendors") or {})
    for method, chain in FMP_TOOL_VENDORS.items():
        tool_vendors.setdefault(method, chain)
    ta_config["tool_vendors"] = tool_vendors
    return ta_config
