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
import re
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


def get_closes(symbol: str, start_date: str, end_date: str) -> pd.Series:
    """复盘用收盘价：与 Yahoo 版一致，窗口内还没有交易日时返回空序列（留待下次复盘）。"""
    try:
        return fetch_history(symbol, _date_str(start_date), _date_str(end_date))["Close"]
    except _errors().NoMarketDataError:
        return pd.Series(dtype=float)


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


_etf_flags: Dict[str, bool] = {}


def _is_etf(canonical: str, profile: Optional[Dict[str, Any]] = None) -> bool:
    if canonical not in _etf_flags:
        if profile is None:
            try:
                profile = (_get_client().get("profile", symbol=canonical) or [{}])[0]
            except _errors().VendorError:
                return False
        _etf_flags[canonical] = bool(profile.get("isEtf") or profile.get("isFund"))
    return _etf_flags[canonical]


def _etf_fundamentals(canonical: str, profile: Dict[str, Any], quote: Dict[str, Any]) -> str:
    """ETF 没有公司财报：给出规模、费率、资产类别、行业/国家暴露。"""
    client = _get_client()
    errors = _errors()

    def safe(path: str) -> List[Dict[str, Any]]:
        try:
            return client.get(path, symbol=canonical)
        except errors.VendorError as exc:
            logger.info("[TradingAgents/FMP] %s unavailable for %s: %s", path, canonical, exc)
            return []

    info = (safe("etf/info") or [{}])[0]
    currency = profile.get("currency") or ""
    fields = [
        ("Name", info.get("name") or profile.get("companyName")),
        ("Type", "ETF / fund (no corporate financial statements; analyse holdings, exposure, flows and NAV)"),
        ("Issuer", info.get("etfCompany")),
        ("Asset Class", info.get("assetClass")),
        ("Assets Under Management", f"{info.get('assetsUnderManagement')} {currency}" if info.get("assetsUnderManagement") else None),
        ("Expense Ratio (%)", info.get("expenseRatio")),
        ("NAV", f"{info.get('nav')} {info.get('navCurrency') or currency}" if info.get("nav") else None),
        ("Price", f"{profile.get('price')} {currency}" if profile.get("price") is not None else None),
        ("Holdings Count", info.get("holdingsCount")),
        ("Average Volume", info.get("avgVolume")),
        ("Inception Date", info.get("inceptionDate")),
        ("52 Week Range", profile.get("range")),
        ("50 Day Average", quote.get("priceAvg50")),
        ("200 Day Average", quote.get("priceAvg200")),
    ]
    lines = [f"{label}: {value}" for label, value in fields if value not in (None, "")]
    sectors = safe("etf/sector-weightings")
    if sectors:
        lines.append("Sector exposure: " + ", ".join(f"{r.get('sector')} {r.get('weightPercentage')}%" for r in sectors[:8]))
    countries = safe("etf/country-weightings")
    if countries:
        lines.append("Country exposure: " + ", ".join(f"{r.get('country')} {r.get('weightPercentage')}" for r in countries[:6]))
    if info.get("description"):
        lines.append(f"Objective: {str(info['description'])[:600]}")
    return f"# Fund Profile for {canonical} (ETF, source: FMP)\n\n" + "\n".join(lines)


def _price_history_span(canonical: str, as_of_date: Optional[str]) -> Optional[tuple]:
    """价格历史不满一年时返回 (近 400 天交易日行数, 首个交易日)；满一年或取不到时返回 None。"""
    end = datetime.strptime(as_of_date, "%Y-%m-%d") if as_of_date else datetime.now()
    start = end - timedelta(days=400)
    try:
        frame = fetch_history(canonical, start.strftime("%Y-%m-%d"), (end + timedelta(days=1)).strftime("%Y-%m-%d"))
    except _errors().VendorError as exc:
        logger.info("[TradingAgents/FMP] price history check skipped for %s: %s", canonical, exc)
        return None
    first = frame.index[0]
    # 窗口取 400 天：首个交易日落在一年内说明是新上市，而不是窗口截断
    if first > pd.Timestamp(end - timedelta(days=365)):
        return len(frame), first.strftime("%Y-%m-%d")
    return None


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
    if _is_etf(canonical, profile):
        return _etf_fundamentals(canonical, profile, quote)
    ratios = optional("ratios-ttm")
    metrics = optional("key-metrics-ttm")
    currency = profile.get("currency")

    def money(value):
        return None if value is None else f"{value} {currency}" if currency else value

    week_range, avg50, avg200 = profile.get("range"), money(quote.get("priceAvg50")), money(quote.get("priceAvg200"))
    history = _price_history_span(canonical, as_of_date)
    if history:
        # 新上市（如 ADR）不足一年时，FMP 的均线/52 周区间只是上市以来的统计
        rows, first_date = history
        short = f"N/A (only {rows} trading days of price history since {first_date})"
        if rows < 200:
            avg200 = short
        if rows < 50:
            avg50 = short
        if week_range:
            week_range = f"{week_range} (since first trade {first_date}, not a full 52 weeks)"

    fields = [
        ("Name", profile.get("companyName")),
        ("Sector", profile.get("sector")),
        ("Industry", profile.get("industry")),
        ("Exchange", profile.get("exchange")),
        ("Market Cap", money(profile.get("marketCap"))),
        ("Price", money(profile.get("price"))),
        ("52 Week Range", week_range),
        ("50 Day Average", avg50),
        ("200 Day Average", avg200),
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
    if _is_etf(canonical):
        return (f"# {title} for {canonical}: not applicable\n{canonical} is an ETF/fund, which files no corporate "
                f"{title.lower()}. Analyse it through holdings, sector/country exposure, flows and NAV instead.")
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


NEWS_TEXT_MAX_CHARS = 400
# 律所集体诉讼招揽稿：同一案件会被十几家律所反复推送，只保留一条
_LAW_FIRM_RE = re.compile(
    r"class action|investor alert|lead plaintiff|securities fraud|shareholder alert|investors who lost|"
    r"rosen law|faruqi|pomerantz|kaplan fox|levi & korsinsky|bronstein|glancy|bragar|schall law|"
    r"howard g\. smith|block & leviton|bfa law|robbins geller|kessler topaz|gainey mckenna|portnoy law",
    re.IGNORECASE,
)


def _is_law_firm_solicitation(row: Dict[str, Any]) -> bool:
    return bool(_LAW_FIRM_RE.search(f"{row.get('title') or ''} {row.get('text') or ''}"))


def _format_news(rows: List[Dict[str, Any]]) -> str:
    out = []
    solicitations = 0
    for row in rows:
        if _is_law_firm_solicitation(row):
            solicitations += 1
            if solicitations > 1:
                continue
        source = row.get("publisher") or row.get("site") or "Unknown"
        published = str(row.get("publishedDate") or "")[:16]
        block = f"### {row.get('title') or 'No title'} (source: {source}, {published})\n"
        text = " ".join(str(row.get("text") or "").split())
        if text:
            block += f"{text[:NEWS_TEXT_MAX_CHARS]}{'…' if len(text) > NEWS_TEXT_MAX_CHARS else ''}\n"
        if row.get("url"):
            block += f"Link: {row['url']}\n"
        out.append(block)
    if solicitations > 1:
        out.append(f"_{solicitations - 1} more law-firm class-action solicitations about the same matter were "
                   f"omitted; they repeat one lawsuit notice and carry no new information._\n")
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
    header = f"# Insider transactions for {canonical} (source: FMP, filed by {as_of_date or 'today'})\n"
    # 只有公开市场买卖（Form 4 代码 P/S）反映内部人的主动判断；代扣税、行权、赠与、授予只汇总
    market = [r for r in rows if str(r.get("transactionType") or "")[:2] in ("P-", "S-")]
    other = [r for r in rows if r not in market]
    lines = []
    if market:
        keep = ["filingDate", "transactionDate", "reportingName", "typeOfOwner", "transactionType",
                "securitiesTransacted", "price", "securitiesOwned"]
        lines.append("Open-market purchases (P) and sales (S):\n"
                     + pd.DataFrame(market)[[c for c in keep if c in market[0]]].to_csv(index=False))
    else:
        lines.append("No open-market purchases or sales in these filings.\n")
    if other:
        counts: Dict[str, int] = {}
        for row in other:
            kind = str(row.get("transactionType") or "unspecified")
            counts[kind] = counts.get(kind, 0) + 1
        people = sorted({str(r.get("reportingName")) for r in other if r.get("reportingName")})
        summary = ", ".join(f"{kind} ×{n}" for kind, n in sorted(counts.items(), key=lambda kv: -kv[1]))
        lines.append(
            f"Other filings (not open-market trades: tax withholding, option exercise, gifts, awards): "
            f"{summary}; {len(people)} reporting person(s): {', '.join(people[:8])}"
            f"{' …' if len(people) > 8 else ''}.\n"
        )
    return header + "\n".join(lines)


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
        from tradingagents.memory import settlement

        for module in (yahoo_ohlcv, yahoo_market):
            module.yf = _FmpYfShim
            module.yf_retry = _passthrough_retry
        ta_context.get_company_profile = get_company_profile
        settlement.get_closes = get_closes

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
