# -*- coding: utf-8 -*-
"""TradingAgents 研究的 Polymarket 预测市场参考。

每次研究前自动抓取一组默认主题（宏观利率/衰退/通胀 + AI 算力产业链事件）和本股相关
市场，再加上用户临时指定的主题，拼成一个参考块注入给所有角色。

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
    r"turkey|argentina|mexico|china|korea|switzerland|snb)\b",
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


def _format_event(event: Dict, markets: List[Dict], parse) -> Optional[str]:
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
    lines = [f"- **{title}** (${volume:,.0f} volume, {note}){thin}"]
    if exclusive:
        # 分档事件保留原始档位顺序；档位过多时只留概率最高的几档
        keep = sorted(markets, key=lambda m: -((_prob(m, parse) or ("", 0.0))[1]))[:MARKETS_PER_EVENT]
        shown = [m for m in markets if m in keep]
    else:
        # “截至某日”类事件按到期日排序，读成一条时间曲线
        shown = sorted(markets, key=lambda m: m.get("endDate") or "")[:MARKETS_PER_EVENT]
    for m in shown:
        prob = _prob(m, parse)
        if not prob:
            continue
        label = m.get("groupItemTitle") or m.get("question") or "?"
        lines.append(f"  - {label}: {prob[1]:.0%} (resolves {(m.get('endDate') or '')[:10]}{_week(m)})")
    return "\n".join(lines) if len(lines) > 1 else None


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
        for key, event, markets in candidates:
            block = _format_event(event, markets, polymarket._parse_json_list)
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
