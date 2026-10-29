# -*- coding: utf-8 -*-
"""TradingAgents 研究的 Polymarket 预测市场参考。

每次研究前自动抓取一组默认主题（宏观利率/衰退/通胀 + AI 算力产业链事件）和本股相关
市场，再加上用户临时指定的主题，拼成一个参考块。参考块全文只给新闻分析师与组合经理，
其余角色只看到一行指引（读新闻报告里的解读），避免每个角色重复读一遍上千 token 的赔率表。

- 默认主题可用 ``TRADINGAGENTS_POLYMARKET_TOPICS`` 覆盖（逗号分隔，``none`` 关闭）。
- ``TRADINGAGENTS_POLYMARKET_TICKER=false`` 关闭按股票代码搜索本股市场。
- 按 Polymarket 事件分组展示（事件标题 + 各档位概率），互斥分档事件给出完整分布，
  避免把同一事件的不同档位误读为相互矛盾；跨主题按事件去重。
- 过滤低成交量事件（默认主题 < $20k、个股 < $500），个股主题优先财报类事件。
- 单个主题失败只记一行“不可用”，不影响研究；历史日期不提供（Polymarket 只有实时赔率）。
"""

from __future__ import annotations

import logging
import os
import re
from datetime import UTC, datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

TOPICS_ENV = "TRADINGAGENTS_POLYMARKET_TOPICS"
TICKER_ENV = "TRADINGAGENTS_POLYMARKET_TICKER"

# 以 AI 算力产业链美股为主的默认参考：利率路径、衰退/通胀、头部 AI 公司 IPO 估值、
# AI 泡沫、台海（台积电供应链）、芯片出口管制
DEFAULT_TOPICS: Tuple[str, ...] = (
    "Fed decision",
    "Fed rate hike",
    "US recession",
    "inflation",
    "Anthropic IPO",
    "OpenAI IPO",
    "AI bubble",
    "China Taiwan",
    "chip export",
)

EVENTS_PER_TOPIC = 2
EVENTS_FOR_TICKER = 3
MARKETS_PER_EVENT = 8
MIN_EVENT_VOLUME = 20_000
MIN_TICKER_EVENT_VOLUME = 500
THIN_VOLUME = 5_000
# 宏观主题只保留美国相关事件（搜索 "inflation" 会带出巴西/英国等）
_MACRO_TOPICS = {"fed decision", "fed rate hike", "us recession", "inflation"}
_NON_US_MACRO_RE = re.compile(
    r"\b(brazil|uk|britain|england|japan|boj|canada|euro|ecb|eurozone|germany|france|australia|rba|india|"
    r"turkey|argentina|mexico|china|korea|switzerland|snb|south africa|new zealand|russia|nigeria|poland|"
    r"sweden|norway|chile|colombia|peru|indonesia|philippines|thailand|vietnam|egypt|israel)\b|\bu\.k\.",
    re.IGNORECASE,
)
_FUNDAMENTAL_RE = re.compile(r"revenue|margin|eps|earnings|guidance|data center", re.IGNORECASE)


def default_topics() -> List[str]:
    raw = os.environ.get(TOPICS_ENV)
    if raw is None:
        return list(DEFAULT_TOPICS)
    if raw.strip().lower() in {"", "none", "off", "false"}:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def ticker_topic_enabled() -> bool:
    return (os.environ.get(TICKER_ENV) or "true").strip().lower() not in {"0", "false", "no", "off"}


def resolve_topics(ticker: str, extra: List[str], use_defaults: bool) -> List[Tuple[str, str]]:
    """返回 [(主题, 类型)]，类型为 default/ticker/user；按小写去重。"""
    topics: List[Tuple[str, str]] = []
    if use_defaults:
        topics += [(t, "default") for t in default_topics()]
        if ticker_topic_enabled() and ticker:
            topics.append((ticker.split(".")[0].upper(), "ticker"))
    topics += [(t, "user") for t in extra]
    seen, unique = set(), []
    for topic, kind in topics:
        if topic.lower() not in seen:
            seen.add(topic.lower())
            unique.append((topic, kind))
    return unique


def _prob(m: Dict, parse) -> Optional[Tuple[str, float]]:
    prices, outcomes = parse(m.get("outcomePrices")), parse(m.get("outcomes"))
    try:
        return (outcomes[0] if outcomes else "Yes"), float(prices[0])
    except (ValueError, IndexError, TypeError):
        return None


def _week(m: Dict) -> str:
    wk = m.get("oneWeekPriceChange")
    return f", 1w {wk * 100:+.1f}pp" if isinstance(wk, (int, float)) and wk else ""


def _event_volume(event: Dict, markets: List[Dict]) -> float:
    return float(event.get("volume") or sum(m.get("volumeNum") or 0 for m in markets))


_STRIKE_RE = re.compile(r"^\s*(?P<side>[↑↓])\s*\$?\s*(?P<strike>[\d,]+(?:\.\d+)?)")
_TOUCH_NOTE = (
    "  - How to read: touch markets. ↑ $X resolves Yes if the price trades at or above X at any time "
    "before resolution, ↓ $X if it trades at or below X; they are not closing-price forecasts. "
    "A strike at or near the current price is close to certain and carries no directional information."
)
# 触及型档位允许的反向误差（报价精度/价差）
_LADDER_TOLERANCE = 0.01


def _strike(label: str) -> Optional[Tuple[str, float]]:
    match = _STRIKE_RE.match(label or "")
    if not match:
        return None
    return match.group("side"), float(match.group("strike").replace(",", ""))


def _ladder_violation(points: List[Tuple[str, float, float]]) -> Optional[str]:
    """触及型档位越远离现价概率应越低；返回第一处反常的描述。"""
    for side in ("↑", "↓"):
        ladder = sorted(((k, p) for s, k, p in points if s == side), reverse=(side == "↓"))
        for (near_k, near_p), (far_k, far_p) in zip(ladder, ladder[1:]):
            if far_p > near_p + _LADDER_TOLERANCE:
                return f"{side} ${far_k:g} {far_p:.0%} > {side} ${near_k:g} {near_p:.0%}"
    return None


def _format_event(event: Dict, markets: List[Dict], parse,
                  last_close: Optional[Tuple[str, float]] = None) -> Optional[str]:
    volume = _event_volume(event, markets)
    thin = " — thin market, low information" if volume < THIN_VOLUME else ""
    title = event.get("title") or markets[0].get("question") or "?"
    resolves = (markets[0].get("endDate") or "")[:10]
    if len(markets) == 1:
        prob = _prob(markets[0], parse)
        if not prob:
            return None
        label, p = prob
        return (f"- **{markets[0].get('question') or title}** — {label} {p:.0%} "
                f"(${volume:,.0f} volume, resolves {resolves}{_week(markets[0])}){thin}")
    exclusive = bool(event.get("negRisk"))
    note = "mutually exclusive outcomes" if exclusive else "separate yes/no markets"
    if exclusive:
        total = sum((_prob(m, parse) or ("", 0.0))[1] for m in markets)
        if abs(total - 1.0) > 0.15:
            # Polymarket 偶有档位概率之和明显偏离 100%（如某档单独挂牌），提示模型谨慎解读
            note += f"; outcome prices sum to {total:.0%}, so read them as loose odds"
    lines = [f"- **{title}** (${volume:,.0f} volume, {note}){thin}"]
    if exclusive:
        # 分档事件保留原始档位顺序；档位过多时只留概率最高的几档
        keep = sorted(markets, key=lambda m: -((_prob(m, parse) or ("", 0.0))[1]))[:MARKETS_PER_EVENT]
        shown = [m for m in markets if m in keep]
    else:
        # “截至某日”类事件按到期日排序，读成一条时间曲线
        shown = sorted(markets, key=lambda m: m.get("endDate") or "")[:MARKETS_PER_EVENT]
    strikes: List[Tuple[str, float, float]] = []
    for m in shown:
        prob = _prob(m, parse)
        if not prob:
            continue
        label = m.get("groupItemTitle") or m.get("question") or "?"
        distance = ""
        strike = _strike(label)
        if strike:
            strikes.append((strike[0], strike[1], prob[1]))
            if last_close and last_close[1] > 0:
                gap = (strike[1] - last_close[1]) / last_close[1]
                distance = f", {abs(gap):.1%} {'above' if gap >= 0 else 'below'} last close"
        lines.append(f"  - {label}: {prob[1]:.0%} (resolves {(m.get('endDate') or '')[:10]}{_week(m)}{distance})")
    if len(lines) <= 1:
        return None
    if strikes:
        # 个股“hit”盘口是触及型，模型容易读成收盘价预测
        note = _TOUCH_NOTE
        if last_close:
            note += f" Last close: {last_close[1]:.2f} ({last_close[0]})."
        violation = _ladder_violation(strikes)
        if violation:
            note += (f" The ladder is not monotonic ({violation}), so quotes are stale; "
                     "do not read a price distribution from it.")
        lines.insert(1, note)
    return "\n".join(lines)


def _last_close(ticker: str, run_date: str) -> Optional[Tuple[str, float]]:
    """本股最新收盘（与研究同一行情通道）；失败返回 None，只少一个距离标注。"""
    try:
        from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv

        data = load_ohlcv(ticker, run_date)
        row = data.dropna(subset=["Close"]).iloc[-1]
        return str(row["Date"])[:10], float(row["Close"])
    except Exception as exc:
        logger.info("[TradingAgents/Polymarket] last close unavailable for %s: %s", ticker, exc)
        return None


# ---------------------------------------------------------------------------
# 参考块作用域：只给新闻分析师与组合经理
# ---------------------------------------------------------------------------

_REF_START = "<!-- polymarket-reference:start -->"
_REF_END = "<!-- polymarket-reference:end -->"
_REF_BLOCK_RE = re.compile(re.escape(_REF_START) + r".*?" + re.escape(_REF_END), re.DOTALL)
REFERENCE_POINTER = (
    "Prediction-market reference: the full Polymarket odds were given to the news analyst and the "
    "portfolio manager. Use the news analyst's report for how they bear on this instrument; do not "
    "cite specific odds that the report does not mention."
)
# 读 instrument_context、但不需要赔率全文的角色模块
_POINTER_ONLY_MODULES = (
    "tradingagents.agents.analysts.market_analyst",
    "tradingagents.agents.analysts.sentiment_analyst",
    "tradingagents.agents.analysts.fundamentals_analyst",
    "tradingagents.agents.researchers.bull_researcher",
    "tradingagents.agents.researchers.bear_researcher",
    "tradingagents.agents.managers.research_manager",
    "tradingagents.agents.trader.trader",
    "tradingagents.agents.risk_mgmt.aggressive_debator",
    "tradingagents.agents.risk_mgmt.conservative_debator",
    "tradingagents.agents.risk_mgmt.neutral_debator",
)
_SCOPED_ATTR = "_dsa_reference_scoped"


def mark_reference(text: str) -> str:
    return f"{_REF_START}\n{text.strip()}\n{_REF_END}"


def strip_reference(context: str) -> str:
    """把参考块换成一行指引；没有参考块时原样返回。"""
    return _REF_BLOCK_RE.sub(REFERENCE_POINTER, context or "")


def install_reference_scope() -> None:
    """让其余角色读到的 instrument_context 去掉参考块全文（幂等）。"""
    import importlib

    for name in _POINTER_ONLY_MODULES:
        module = importlib.import_module(name)
        original = module.get_instrument_context_from_state
        if getattr(original, _SCOPED_ATTR, False):
            continue

        def scoped(state, _original=original):
            return strip_reference(_original(state))

        setattr(scoped, _SCOPED_ATTR, True)
        module.get_instrument_context_from_state = scoped


def build_reference(ticker: str, run_date: str, extra: List[str], use_defaults: bool = True) -> Tuple[str, List[str]]:
    """抓取并格式化参考块；返回 (markdown, 实际使用的主题)。没有主题时返回 ("", [])。"""
    topics = resolve_topics(ticker, extra, use_defaults)
    if not topics:
        return "", []

    from tradingagents.dataflows.date_window import is_historical

    names = [t for t, _ in topics]
    header = (
        "# Prediction-market reference (Polymarket)\n"
        "Live, crowd-priced probabilities on macro and sector events relevant to this instrument "
        "(rate path, recession/inflation, AI-leader IPO valuations, AI bubble, Taiwan and chip-export risk, "
        "and markets on the company itself). Higher volume = more reliable. Treat them as odds, not facts; "
        "explain how, if at all, they bear on the instrument, and ignore markets that are irrelevant.\n\n"
    )
    if is_historical(run_date):
        return header + (
            f"Withheld for {run_date}: Polymarket only serves live odds, so they would leak "
            f"post-decision information into a historical analysis.\n"
        ), names

    from tradingagents.dataflows.vendors import polymarket

    now = datetime.now(UTC)
    seen_events = set()
    sections = []
    for topic, kind in topics:
        try:
            data = polymarket._request("public-search", {"q": topic, "limit_per_type": 20})
        except Exception as exc:  # 单个主题失败不影响研究
            logger.info("[TradingAgents/Polymarket] search failed for %r: %s", topic, exc)
            sections.append(f'## "{topic}"\nUnavailable right now ({type(exc).__name__}).')
            continue
        min_volume = MIN_TICKER_EVENT_VOLUME if kind == "ticker" else MIN_EVENT_VOLUME
        candidates = []
        for event in data.get("events", []):
            key = event.get("id") or event.get("slug") or event.get("title")
            markets = [m for m in event.get("markets", []) if polymarket._is_forward_looking(m, now)]
            if not markets or key in seen_events or _event_volume(event, markets) < min_volume:
                continue
            if topic.lower() in _MACRO_TOPICS and _NON_US_MACRO_RE.search(event.get("title") or ""):
                continue
            candidates.append((key, event, markets))
        if kind == "ticker":
            candidates.sort(key=lambda c: (not _FUNDAMENTAL_RE.search(c[1].get("title") or ""),
                                           -_event_volume(c[1], c[2])))
            limit = EVENTS_FOR_TICKER
        else:
            candidates.sort(key=lambda c: -_event_volume(c[1], c[2]))
            limit = EVENTS_PER_TOPIC
        blocks = []
        close = _last_close(ticker, run_date) if kind == "ticker" and candidates else None
        for key, event, markets in candidates:
            block = _format_event(event, markets, polymarket._parse_json_list, close)
            if block:
                seen_events.add(key)
                blocks.append(block)
            if len(blocks) >= limit:
                break
        if blocks:
            title = f'## "{topic}"' + (" (markets on this company)" if kind == "ticker" else "")
            sections.append(title + "\n" + "\n".join(blocks))
    if not sections:
        return header + "No open prediction markets matched the reference topics.\n", names
    return header + "\n\n".join(sections) + "\n", names
