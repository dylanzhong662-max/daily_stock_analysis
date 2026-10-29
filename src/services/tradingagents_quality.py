# -*- coding: utf-8 -*-
"""TradingAgents 工具输出的数据质量补丁。

TradingAgents 把工具输出原样交给 LLM，几类输出容易被误读，进而带偏整份报告：

- 价格历史不足时，stockstats 的均线按已有行数计算（min_periods=1）：上市 75 天的
  ADR 会给出一个“200 日均线”，其实是上市以来均价。补丁把窗口不足的指标改为 N/A，
  并在行情快照里写明可用历史行数。
- SEC EDGAR 季报表没有独立的 Q4 列（Q4 只在 10-K 年报里），模型会把相邻列当作
  连续季度、拿错同比基数。补丁在报表后附上每列覆盖的期间、缺失季度和按同期配对算好的同比，
  并把表裁剪到最近 12 个期间列（SEC 会给出十年以上的列）。
- FRED 只给“窗口首尾变化”，模型会把 11 个月的变化当同比。补丁对月度/季度序列补上
  真实同比与最近一期变化。
- 反思记忆默认注入其他股票的复盘教训，组合经理会拿别的股票的结果给本股定仓位。
  补丁只保留本股自己的历史。

补丁是进程级、幂等的，不修改 TradingAgents 包；只在 TradingAgents 已安装时调用。
"""

from __future__ import annotations

import csv
import functools
import io
import logging
import re
import threading
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_install_lock = threading.Lock()
_installed = False
_PATCHED_ATTR = "_dsa_quality_patched"

FULL_YEAR_ROWS = 252

# stockstats 指标 -> 有意义所需的最少交易日行数
_FIXED_WINDOWS = {
    "boll": 20, "boll_ub": 20, "boll_lb": 20,
    "rsi": 14, "atr": 14, "vwma": 14, "mfi": 14,
    "macd": 26, "macds": 35, "macdh": 35,
}
_MA_RE = re.compile(r"^close_(\d+)_(?:sma|ema)$")


def required_rows(indicator: str) -> int:
    """指标需要的最少行数；未知指标返回 1（不拦截）。"""
    match = _MA_RE.match(indicator or "")
    if match:
        return int(match.group(1))
    return _FIXED_WINDOWS.get(indicator, 1)


def _insufficient(need: int, have: int) -> str:
    return f"N/A: insufficient price history (needs {need} trading rows, has {have})"


def _mark(func: Callable) -> Callable:
    setattr(func, _PATCHED_ATTR, True)
    return func


def _is_patched(func: Any) -> bool:
    return bool(getattr(func, _PATCHED_ATTR, False))


# ---------------------------------------------------------------------------
# 技术指标：窗口不足时给 N/A
# ---------------------------------------------------------------------------

def _wrap_bulk(original: Callable) -> Callable:
    @functools.wraps(original)
    def wrapper(symbol, indicator, as_of_date):
        values = original(symbol, indicator, as_of_date)
        need = required_rows(indicator)
        if need <= 1:
            return values
        for index, day in enumerate(sorted(values)):
            if index + 1 < need:
                values[day] = _insufficient(need, index + 1)
        return values

    return _mark(wrapper)


def _wrap_single(original: Callable, load_ohlcv: Callable) -> Callable:
    @functools.wraps(original)
    def wrapper(symbol, indicator, as_of_date):
        need = required_rows(indicator)
        if need > 1:
            import pandas as pd

            data = load_ohlcv(symbol, as_of_date)
            have = int((pd.to_datetime(data["Date"]) <= pd.to_datetime(as_of_date)).sum())
            if have < need:
                return _insufficient(need, have)
        return original(symbol, indicator, as_of_date)

    return _mark(wrapper)


_SNAPSHOT_ROW_RE = re.compile(r"^\| (?P<name>[a-z0-9_]+) \| (?P<value>[^|]*) \|$")


def rewrite_snapshot(text: str, rows: int, first_date: str) -> str:
    """把行情快照里窗口不足的指标改成 N/A，并写明可用历史行数。"""
    out = []
    for line in text.splitlines():
        match = _SNAPSHOT_ROW_RE.match(line)
        if match and required_rows(match.group("name")) > rows:
            line = f"| {match.group('name')} | {_insufficient(required_rows(match.group('name')), rows)} |"
        out.append(line)
        if line.startswith("- Latest trading row used:"):
            note = f"- Price history available: {rows} trading rows since {first_date}"
            if rows < FULL_YEAR_ROWS:
                note += (" (less than one full year: indicators whose window exceeds this are N/A, "
                         "and 52-week highs/lows only cover this period)")
            out.append(note)
    return "\n".join(out)


def _wrap_snapshot(original: Callable, verified_rows: Callable) -> Callable:
    @functools.wraps(original)
    def wrapper(symbol, as_of_date, *args, **kwargs):
        text = original(symbol, as_of_date, *args, **kwargs)
        try:
            frame = verified_rows(symbol, as_of_date)
            first = frame["Date"].iloc[0]
            first_date = first.strftime("%Y-%m-%d") if hasattr(first, "strftime") else str(first)[:10]
            return rewrite_snapshot(text, len(frame), first_date)
        except Exception as exc:  # 快照本身已生成，补充说明失败不影响使用
            logger.info("[TradingAgents/quality] snapshot history check skipped for %s: %s", symbol, exc)
            return text

    return _mark(wrapper)


# ---------------------------------------------------------------------------
# 财报：期间指引 + 同期同比
# ---------------------------------------------------------------------------

_COLUMN_RE = re.compile(r"^(?P<end>\d{4}-\d{2}-\d{2})(?: \((?P<months>\d+) months\))?$")
_FMP_YOY_ROWS = (
    "revenue", "grossProfit", "operatingIncome", "netIncome", "epsDiluted",
    "operatingCashFlow", "capitalExpenditure", "freeCashFlow",
)
RECENT_COLUMNS = 8
_YOY_TOLERANCE_DAYS = 10
_QUARTER_GAP_DAYS = 120


def _month_name(day: date) -> str:
    return day.strftime("%b %Y")


def _months_back(day: date, months: int) -> date:
    """期末所在月往前 months-1 个月的月份（期间起始月）。"""
    index = day.year * 12 + day.month - 1 - (months - 1)
    return date(index // 12, index % 12 + 1, 1)


def _number(cell: str) -> Optional[float]:
    try:
        return float(cell)
    except (TypeError, ValueError):
        return None


def _pct(new: float, old: float) -> Optional[str]:
    if old == 0:
        return None
    change = (new - old) / abs(old) * 100
    return f"{change:+.1f}%"


def _statement_change(new: float, old: float) -> Optional[str]:
    """报表行同比：正负号变化时百分比无意义；两期均为负（如投资现金流出）按流出规模变化表述。"""
    if old == 0 or (new < 0) != (old < 0):
        return None
    if new < 0:
        return f"outflow {(abs(new) - abs(old)) / abs(old) * 100:+.1f}%"
    return _pct(new, old)


def statement_guide(text: str) -> str:
    """为季度报表生成“期间指引 + 同比”附录；无法解析或非季度报表时返回空串。"""
    lines = text.splitlines()
    if not lines or "annual" in lines[0].lower():
        return ""
    body = [line for line in lines if line and not line.startswith("#")]
    if len(body) < 2:
        return ""
    table = list(csv.reader(io.StringIO("\n".join(body))))
    header, rows = table[0], table[1:]
    columns: List[Tuple[int, date, int]] = []
    for index, cell in enumerate(header[1:], start=1):
        match = _COLUMN_RE.match(cell.strip())
        if match:
            end = datetime.strptime(match.group("end"), "%Y-%m-%d").date()
            columns.append((index, end, int(match.group("months") or 3)))
    if len(columns) < 2:
        return ""

    is_fmp = "source: FMP" in lines[0]
    columns.sort(key=lambda c: c[1])
    recent = columns[-RECENT_COLUMNS:]

    guide = ["", "## Period guide (computed by DSA from the column dates above)"]
    for _, end, months in reversed(recent):
        start = _months_back(end, months)
        span = "quarter" if months == 3 else f"{months} months year to date"
        guide.append(f"- {end.isoformat()}: {span}, {_month_name(start)} – {_month_name(end)}")
    gaps = []
    # 现金流表常以年初至今口径申报，季度之间本就隔着 6/9 个月列，只对纯季度列判断缺口
    if all(months == 3 for _, _, months in recent):
        for (_, prev_end, _), (_, end, _) in zip(recent, recent[1:]):
            if (end - prev_end).days > _QUARTER_GAP_DAYS:
                missing_end = _months_back(end, 4)
                gaps.append(f"{_month_name(_months_back(missing_end, 3))} – {_month_name(missing_end)}")
    if gaps:
        guide.append(
            "- Not filed as a standalone quarter: " + "; ".join(gaps)
            + ". A fiscal fourth quarter is reported only inside the annual (10-K) figure, so adjacent "
            "columns are NOT consecutive quarters. Compare a quarter only with the same quarter a year earlier."
        )

    by_key = {(end, months): index for index, end, months in columns}
    yoy_lines = []
    for row in rows:
        if not row:
            continue
        name = row[0].strip()
        if is_fmp and name not in _FMP_YOY_ROWS:
            continue
        cells = []
        for index, end, months in reversed(recent):
            current = _number(row[index]) if index < len(row) else None
            base_index = next(
                (i for (e, m), i in by_key.items()
                 if m == months and abs((end - e).days - 365) <= _YOY_TOLERANCE_DAYS),
                None,
            )
            base = _number(row[base_index]) if base_index is not None and base_index < len(row) else None
            change = _statement_change(current, base) if current is not None and base is not None else None
            if change:
                label = end.isoformat() + ("" if months == 3 else f" ({months}m)")
                cells.append(f"{label} {change}")
        if cells:
            yoy_lines.append(f"- {name}: " + ", ".join(cells))
    if yoy_lines:
        guide.append("")
        guide.append("## Year-over-year (computed: each period vs the same period one year earlier)")
        guide.extend(yoy_lines)
        if any("share" in line.lower() for line in yoy_lines):
            guide.append("- Per-share rows are as filed and are not adjusted for later stock splits.")
    return "\n".join(guide) + "\n"


STATEMENT_MAX_COLUMNS = 12


def trim_statement_columns(text: str, keep: int = STATEMENT_MAX_COLUMNS) -> str:
    """报表只保留最近 ``keep`` 个期间列（SEC 会给出十年以上的列）；无法解析时原样返回。"""
    lines = text.splitlines()
    head = [line for line in lines if not line or line.startswith("#")]
    body = [line for line in lines if line and not line.startswith("#")]
    if len(body) < 2:
        return text
    table = list(csv.reader(io.StringIO("\n".join(body))))
    dated = [(index, _COLUMN_RE.match(cell.strip()).group("end"))
             for index, cell in enumerate(table[0][1:], start=1) if _COLUMN_RE.match(cell.strip())]
    if len(dated) <= keep:
        return text
    recent = sorted(dated, key=lambda c: c[1])[-keep:]
    kept = {index for index, _ in recent}
    columns = [0] + [i for i in range(1, len(table[0])) if i in kept or not _COLUMN_RE.match(table[0][i].strip())]
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for row in table:
        writer.writerow([row[i] if i < len(row) else "" for i in columns])
    note = f"# Showing the most recent {keep} of {len(dated)} period columns."
    return "\n".join(head + [note]).rstrip("\n") + "\n\n" + buf.getvalue()


def _wrap_statement(original: Callable, guide: bool = True) -> Callable:
    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        text = original(*args, **kwargs)
        if not isinstance(text, str):
            return text
        try:
            # 同比要用到更早的基数列，先在完整表上算附录，再裁剪
            appendix = statement_guide(text) if guide else ""
            return trim_statement_columns(text) + appendix
        except Exception as exc:  # 附录只是辅助，解析失败就返回原表
            logger.info("[TradingAgents/quality] statement guide skipped: %s", exc)
            return text

    return _mark(wrapper)


# ---------------------------------------------------------------------------
# FRED：月度/季度序列补真实同比
# ---------------------------------------------------------------------------

_FRED_ROW_RE = re.compile(r"^\| (\d{4}-\d{2}-\d{2}) \| ([^|]+) \|$")
_FRED_BASE_LOOKBACK_DAYS = 400


def _fred_points(text: str) -> Dict[str, float]:
    points = {}
    for line in text.splitlines():
        match = _FRED_ROW_RE.match(line.strip())
        if match:
            value = _number(match.group(2).strip())
            if value is not None:
                points[match.group(1)] = value
    return points


def _fred_field(text: str, field: str) -> str:
    prefix = f"- {field}:"
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return ""


def _year_earlier(day: str) -> str:
    return f"{int(day[:4]) - 1}{day[4:]}"


def fred_summary(text: str, base_points: Optional[Dict[str, float]] = None) -> str:
    """生成同比/最近一期变化说明；非月度/季度或缺基数时返回空串。"""
    frequency = _fred_field(text, "Frequency").lower()
    if not (frequency.startswith("monthly") or frequency.startswith("quarterly")):
        return ""
    points = _fred_points(text)
    if not points:
        return ""
    ordered = sorted(points)
    latest = ordered[-1]
    is_rate = "%" in _fred_field(text, "Units") or "percent" in _fred_field(text, "Units").lower()
    pool = {**(base_points or {}), **points}
    lines = []
    base_day = _year_earlier(latest)
    if base_day in pool:
        base = pool[base_day]
        if is_rate:
            lines.append(f"**Year-over-year:** {points[latest] - base:+.2f} pp ({latest} vs {base_day}, from {base})")
        else:
            change = _pct(points[latest], base)
            if change:
                lines.append(f"**Year-over-year:** {change} ({latest} vs {base_day}, from {base})")
    if len(ordered) >= 2:
        prev = ordered[-2]
        if is_rate:
            lines.append(f"**Latest period change:** {points[latest] - points[prev]:+.2f} pp ({prev} -> {latest})")
        else:
            change = _pct(points[latest], points[prev])
            if change:
                lines.append(f"**Latest period change:** {change} ({prev} -> {latest})")
    if not lines:
        return ""
    lines.append("_The \"Change from ... to ...\" figure above spans the whole window; it is not a "
                 "year-over-year rate. Use the year-over-year line for inflation/growth rates._")
    return "\n" + "\n".join(lines) + "\n"


def _wrap_fred(original: Callable) -> Callable:
    @functools.wraps(original)
    def wrapper(indicator, as_of_date, look_back_days=None):
        text = original(indicator, as_of_date, look_back_days)
        if not isinstance(text, str) or "| Date | Value |" not in text:
            return text
        try:
            points = _fred_points(text)
            base_points = None
            if points and _year_earlier(max(points)) not in points:
                base_points = _fred_points(original(indicator, as_of_date, _FRED_BASE_LOOKBACK_DAYS))
            return text + fred_summary(text, base_points)
        except Exception as exc:  # 同比只是补充，失败返回原文
            logger.info("[TradingAgents/quality] FRED year-over-year skipped for %s: %s", indicator, exc)
            return text

    return _mark(wrapper)


# ---------------------------------------------------------------------------
# 反思记忆：只用本股历史
# ---------------------------------------------------------------------------

def _wrap_past_context(original: Callable) -> Callable:
    @functools.wraps(original)
    def wrapper(self, ticker, n_same=5, n_cross=3, as_of=None):
        # 跨股票教训会被当作本股证据引用（如用 SKHY 的复盘结果给 GOOGL 定仓位）
        return original(self, ticker, n_same=n_same, n_cross=0, as_of=as_of)

    return _mark(wrapper)


# ---------------------------------------------------------------------------
# 安装
# ---------------------------------------------------------------------------

def install_quality_patches() -> None:
    """安装全部数据质量补丁（幂等；FMP vendor 需先安装，才能一并包装它的报表实现）。"""
    global _installed
    with _install_lock:
        from tradingagents.agents import tools as ta_tools
        from tradingagents.dataflows import router
        from tradingagents.dataflows.vendors.yahoo import market as yahoo_market
        from tradingagents.dataflows.vendors.yahoo import snapshot as yahoo_snapshot
        from tradingagents.memory.log import TradingMemoryLog

        if not _installed:
            yahoo_market._get_stock_stats_bulk = _wrap_bulk(yahoo_market._get_stock_stats_bulk)
            yahoo_market.get_stock_stats = _wrap_single(yahoo_market.get_stock_stats, yahoo_market.load_ohlcv)
            snapshot = _wrap_snapshot(yahoo_snapshot.build_verified_market_snapshot, yahoo_snapshot._verified_rows)
            yahoo_snapshot.build_verified_market_snapshot = snapshot
            ta_tools.build_verified_market_snapshot = snapshot
            TradingMemoryLog.get_past_context = _wrap_past_context(TradingMemoryLog.get_past_context)
            _installed = True

        # 报表/宏观实现可能在之后才注册（如 FMP vendor），每次调用都补包装未包装的
        # 资产负债表是时点数，不加期间/同比附录，只裁剪列
        for method, guide in (("get_income_statement", True), ("get_cashflow", True), ("get_balance_sheet", False)):
            for vendor, impl in list(router.VENDOR_METHODS.get(method, {}).items()):
                if vendor in ("sec_edgar", "fmp") and callable(impl) and not _is_patched(impl):
                    router.VENDOR_METHODS[method][vendor] = _wrap_statement(impl, guide=guide)
        fred_impl = router.VENDOR_METHODS.get("get_macro_indicators", {}).get("fred")
        if callable(fred_impl) and not _is_patched(fred_impl):
            router.VENDOR_METHODS["get_macro_indicators"]["fred"] = _wrap_fred(fred_impl)
        logger.info("[TradingAgents] data quality patches installed")
