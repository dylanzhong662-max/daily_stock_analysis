# -*- coding: utf-8 -*-
"""TradingAgents 研究的 Polymarket 预测市场参考。

每次研究前自动抓取一组默认主题（宏观利率/衰退/通胀 + AI 算力产业链事件）和本股相关
市场，再加上用户临时指定的主题，拼成一个参考块注入给所有角色。

- 默认主题可用 ``TRADINGAGENTS_POLYMARKET_TOPICS`` 覆盖（逗号分隔，``none`` 关闭）。
- ``TRADINGAGENTS_POLYMARKET_TICKER=false`` 关闭按股票代码搜索本股市场。
- 跨主题按 market id 去重；个股主题优先财报类市场（营收/毛利率/EPS）。
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
    "rate hike",
    "recession",
    "inflation",
    "Anthropic IPO",
    "OpenAI IPO",
    "AI bubble",
    "China Taiwan",
    "chip export",
)

MARKETS_PER_TOPIC = 4
MARKETS_FOR_TICKER = 6
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


def _format_market(m: Dict, parse) -> Optional[str]:
    prices, outcomes = parse(m.get("outcomePrices")), parse(m.get("outcomes"))
    try:
        prob = float(prices[0])
    except (ValueError, IndexError, TypeError):
        return None
    label = outcomes[0] if outcomes else "Yes"
    wk = m.get("oneWeekPriceChange")
    wk_str = f", 1-week {wk * 100:+.1f}pp" if isinstance(wk, (int, float)) and wk else ""
    return (
        f"- **{m.get('question')}** — {label} {prob:.0%} "
        f"(${(m.get('volumeNum') or 0):,.0f} volume, resolves {(m.get('endDate') or '')[:10]}{wk_str})"
    )


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
    seen_ids = set()
    sections = []
    for topic, kind in topics:
        try:
            data = polymarket._request("public-search", {"q": topic, "limit_per_type": 20})
        except Exception as exc:  # 单个主题失败不影响研究
            logger.info("[TradingAgents/Polymarket] search failed for %r: %s", topic, exc)
            sections.append(f'## "{topic}"\nUnavailable right now ({type(exc).__name__}).')
            continue
        markets = [
            m for event in data.get("events", []) for m in event.get("markets", [])
            if polymarket._is_forward_looking(m, now)
        ]
        if kind == "ticker":
            markets.sort(key=lambda m: (not _FUNDAMENTAL_RE.search(m.get("question") or ""), -(m.get("volumeNum") or 0)))
            limit = MARKETS_FOR_TICKER
        else:
            markets.sort(key=lambda m: -(m.get("volumeNum") or 0))
            limit = MARKETS_PER_TOPIC
        lines = []
        for m in markets:
            key = m.get("id") or m.get("conditionId") or m.get("question")
            if key in seen_ids:
                continue
            line = _format_market(m, polymarket._parse_json_list)
            if line:
                seen_ids.add(key)
                lines.append(line)
            if len(lines) >= limit:
                break
        if lines:
            title = f'## "{topic}"' + (" (markets on this company)" if kind == "ticker" else "")
            sections.append(title + "\n" + "\n".join(lines))
    if not sections:
        return header + "No open prediction markets matched the reference topics.\n", names
    return header + "\n\n".join(sections) + "\n", names
