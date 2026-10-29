# -*- coding: utf-8 -*-
"""TradingAgents 研究报告的离线评估，分数写回 Langfuse。

三类指标：

1. **评级结果**：评级后 5 / 20 个交易日的超额收益（alpha = 个股涨跌 − 基准涨跌，美股基准 SPY），
   命中规则 Buy/Overweight alpha>0、Sell/Underweight alpha<0、Hold |alpha|≤2%。
   复用 TradingAgents 记忆复盘的 ``fetch_returns``，窗口未走完就留待下次。
2. **数据误读**：对报告逐句做确定性规则检查（短历史引用 200 日线、与真实值不符的营收/CPI 同比、
   引用其他股票的复盘结果、把触及型预测盘口读成收盘预测），命中句子写进分数说明。
3. **报告质量**：LLM（``TRADINGAGENTS_EVAL_JUDGE_MODEL``，未设时用 DSA 的 ``LITELLM_MODEL``）按 4 个维度给
   good/acceptable/poor；
   人工在本地评审表打同样的分，导入后计算两者一致率，用来校准 LLM 评审。

报告全文只在本地读取；写进 Langfuse 的只有分数和简短说明。没有 trace 的历史报告
用确定性 trace id 回填一条轻量 trace（标签 ``backfill``），时间戳为回填当天。
每个分数的终值只写一次（Langfuse 只在同 id、同名、同一天时覆盖），状态记在
``data/tradingagents/eval/state.json``。

用法（服务器上）：``sudo ./scripts/tradingagents_eval.sh run|summary|review-sheet|import-review <csv>``。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

EVAL_ENABLED_ENV = "TRADINGAGENTS_EVAL_ENABLED"
EVAL_JUDGE_ENV = "TRADINGAGENTS_EVAL_JUDGE"
# 评审单独用的模型（LiteLLM 写法，OpenAI 兼容网关用 openai/<model>）；不设则用 LITELLM_MODEL
JUDGE_MODEL_ENV = "TRADINGAGENTS_EVAL_JUDGE_MODEL"
JUDGE_API_KEY_ENV = "TRADINGAGENTS_EVAL_JUDGE_API_KEY"
JUDGE_BASE_URL_ENV = "TRADINGAGENTS_EVAL_JUDGE_BASE_URL"
# 评审网关在境外、机房直连不通时走的代理（如 http://host.docker.internal:8118）；只作用于评审调用
JUDGE_PROXY_ENV = "TRADINGAGENTS_EVAL_JUDGE_PROXY"
HORIZONS = (5, 20)
HOLD_BAND = 0.02
BULLISH = {"buy", "overweight"}
BEARISH = {"sell", "underweight"}
LABELS = ("good", "acceptable", "poor")
DIMENSIONS = ("evidence_grounding", "consistency", "actionability", "balance")
JUDGE_VERSION = "report-quality-v2"
# 规则或评审 prompt 改动时升版本：旧分数保留，新版本另写一份
RULES_VERSION = "rules-v5"
# 规则分数的 id 种子固定不变：同一天重算会覆盖旧版本分数（Langfuse 按 id + 名称 + 日期覆盖），
# 版本号写在分数说明里
RULES_SCORE_SEED = "rules-v1"
RULES = ("misread_short_history_ma", "misread_yoy", "misread_cross_ticker", "misread_touch_market")
# 评审表来源：human 人工；claude 为 Claude 代评的参考标签（分数名后缀区分，不冒充人工）
REVIEW_SOURCES = ("human", "claude")
REVIEW_COLUMNS = ("report_id", "ticker", "trade_date", "rating", "report_url", *DIMENSIONS, "overall", "note")

_lock = threading.Lock()


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def eval_root() -> Path:
    from src.services.tradingagents_service import TRADINGAGENTS_DATA_DIR

    return TRADINGAGENTS_DATA_DIR / "eval"


# ---------------------------------------------------------------------------
# 报告索引
# ---------------------------------------------------------------------------

_REPORT_ID_RE = re.compile(r"^(?P<ticker>[A-Za-z0-9.^=-]+)_(?P<stamp>\d{8}_\d{6})$")
_ANALYSIS_DATE_RE = re.compile(r"Analysis date:\s*(\d{4}-\d{2}-\d{2})")


@dataclass
class Report:
    report_id: str
    ticker: str
    trade_date: str
    rating: Optional[str]
    path: Path

    def read(self, relative: str) -> str:
        try:
            return (self.path / relative).read_text(encoding="utf-8")
        except OSError:
            return ""

    @property
    def text(self) -> str:
        return self.read("complete_report.md")


def list_reports(root: Optional[Path] = None) -> List[Report]:
    from src.services.tradingagents_service import _parse_rating, reports_root

    root = root or reports_root()
    reports = []
    for path in sorted(root.iterdir()) if root.is_dir() else []:
        match = _REPORT_ID_RE.match(path.name)
        if not match or not (path / "complete_report.md").is_file():
            continue
        head = (path / "complete_report.md").read_text(encoding="utf-8")[:2000]
        date_match = _ANALYSIS_DATE_RE.search(head)
        trade_date = date_match.group(1) if date_match else datetime.strptime(
            match.group("stamp")[:8], "%Y%m%d").strftime("%Y-%m-%d")
        reports.append(Report(path.name, match.group("ticker").upper(), trade_date,
                              _parse_rating(path / "5_portfolio" / "decision.md"), path))
    return reports


# ---------------------------------------------------------------------------
# 1. 评级结果
# ---------------------------------------------------------------------------

def rating_hit(rating: str, alpha: float) -> Optional[bool]:
    key = (rating or "").strip().lower()
    if key in BULLISH:
        return alpha > 0
    if key in BEARISH:
        return alpha < 0
    if key == "hold":
        return abs(alpha) <= HOLD_BAND
    return None


def compute_outcomes(report: Report, fetch_returns: Callable, benchmark: str,
                     done: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """只算还没定下来的窗口；窗口未走完返回空。"""
    results = {}
    for days in HORIZONS:
        if str(days) in done:
            continue
        raw, alpha, held, resolved = fetch_returns(report.ticker, report.trade_date, days, benchmark)
        if alpha is None:
            continue
        results[str(days)] = {"raw": raw, "alpha": alpha, "resolved": resolved, "benchmark": benchmark,
                              "hit": rating_hit(report.rating or "", alpha)}
    return results


# ---------------------------------------------------------------------------
# 2. 数据误读规则
# ---------------------------------------------------------------------------

_MA200_RE = re.compile(r"200\s*(?:日|天|-day|day)?\s*(?:均线|线|SMA|MA|移动平均)|close_200_sma|MA200", re.I)
_UNAVAILABLE_RE = re.compile(r"N/A|不可用|不足|无数据|无法|算不出|缺少|缺失|insufficient|unavailable|缺乏|没有|不存在|不适用", re.I)
_DECIMAL_RE = re.compile(r"\d+\.\d+")
_YOY_RE = re.compile(r"同比|YoY|year[- ]over[- ]year", re.I)
_YOY_AFTER_RE = re.compile(r"\s*\)?\s*(?:YoY|year[- ]over[- ]year)", re.I)
_REVENUE_RE = re.compile(r"(?:总营收|营收|收入|revenue)(?P<gap>[^%。；;\n|]{0,18}?)(?P<val>[+＋\-−]?\d+(?:\.\d+)?)\s?%", re.I)
# 分部收入、成本口径与 TTM/半年/全年口径不和季度同比比较
_REVENUE_SKIP_NEAR_RE = re.compile(r"云|Cloud|分部|业务|segment|广告|YouTube|搜索|Search|HBM|DRAM|NAND|数据中心|"
                                   r"Data Center|服务|订阅|硬件|Services|成本|cost|商业|政府|美国|国际|commercial|government|部门|CPU|GPU|"
                                   r"产品线|ASP", re.I)
_YOY_SKIP_SENTENCE_RE = re.compile(r"环比|QoQ|预期|预计|预测|预估|预览|即将|能否|延续|一致预期|consensus|estimate|preview|"
                                   r"forecast|guidance|指引|估计|目标|假设|情景|H1|上半年|全年|年度|TTM|annual|年化|翻倍", re.I)
# 数字属于营收之后的另一个指标（“营收同比翻倍、营业利润率 66%”）
_REVENUE_OTHER_METRIC_RE = re.compile(r"、|利润|率|毛利|EPS|capex|资本开支|现金|约", re.I)
# 其他公司的营收（产业链、同行）不和本股申报值比
_OTHER_COMPANY_RE = re.compile(r"TSMC|台积电|SpaceX|Super Micro|美光|Micron|三星|Samsung|英伟达|Nvidia|微软|Microsoft|"
                               r"苹果|Apple|Meta|亚马逊|Amazon|博通|Broadcom|海力士|英特尔|Intel|谷歌|Alphabet|甲骨文|Oracle",
                               re.I)
_CPI_AFTER_RE = re.compile(r"CPI(?P<gap>[^%。；;\n]{0,40}?)(?P<val>[+\-]?\d+(?:\.\d+)?)\s?%")
_CPI_BEFORE_RE = re.compile(r"(?P<val>[+\-]?\d+(?:\.\d+)?)\s?%\s*(?:的)?\s*CPI")
_CPI_MONTHLY_RE = re.compile(r"环比|MoM|月率|month|月度|单月|\d+\s*→\s*\d+\s*月", re.I)
# 数字其实属于后面的利率/收益率，或是预测市场的月度分布
_CPI_OTHER_FIGURE_RE = re.compile(r"\+\s|利率|收益率|美债|bps|yield|定价|分布|上行|下行|长端", re.I)
_VAGUE_PREFIX_RE = re.compile(r"(?:近|接近|逾|超过|超)\s*$")
_CPI_SKIP_RE = re.compile(r"核心|core|PCE|预期|预计|预测|Polymarket|预测市场|概率|>|＞|高于 \d", re.I)
_CROSS_CONTEXT_RE = re.compile(r"复盘|历史经验|上一轮|过往|教训|历史决策|\blesson|(?<![A-Za-z])alpha(?![A-Za-z])", re.I)
_TOUCH_RE = re.compile(r"收盘(?:价)?\s*(?:将|会)\s*(?:低于|高于|跌破|站上)|clos\w*\s+(?:will\s+be\s+)?(?:below|above)\s+\$?\d",
                       re.I)
_PREDICTION_MARKET_RE = re.compile(r"Polymarket|预测市场|盘口", re.I)
REVENUE_TOLERANCE_PP = 1.0
CPI_TOLERANCE_PP = 0.15


def sentences(text: str) -> Iterable[str]:
    for line in text.splitlines():
        for part in re.split(r"(?<=[。；！？!?])", line):
            part = part.strip()
            if part:
                yield part


def _number(raw: str) -> float:
    return float(raw.replace("＋", "+").replace("−", "-"))


def check_short_history_ma(text: str, history_rows: Optional[int]) -> List[str]:
    if history_rows is None or history_rows >= 200:
        return []
    return [s for s in sentences(text)
            if _MA200_RE.search(s) and _DECIMAL_RE.search(s) and not _UNAVAILABLE_RE.search(s)]


def check_revenue_yoy(text: str, true_yoy: List[float], self_names: Iterable[str] = ()) -> List[str]:
    if not true_yoy:
        return []
    own = re.compile("|".join(re.escape(n) for n in self_names if n), re.I) if any(self_names) else None
    hits = []
    for sentence in sentences(text):
        if not _YOY_RE.search(sentence) or _YOY_SKIP_SENTENCE_RE.search(sentence):
            continue
        if any(not (own and own.fullmatch(m.group(0))) for m in _OTHER_COMPANY_RE.finditer(sentence)):
            continue
        for match in _REVENUE_RE.finditer(sentence):
            near = sentence[max(0, match.start() - 8):match.start()] + match.group("gap")
            if _REVENUE_SKIP_NEAR_RE.search(near) or _REVENUE_OTHER_METRIC_RE.search(match.group("gap")):
                continue
            # 同比要直接修饰这个数字：“营收同比 +X%” 或 “营收 …，+X% YoY”
            if not _YOY_RE.search(match.group("gap")) and not _YOY_AFTER_RE.match(sentence, match.end()):
                continue
            value = _number(match.group("val"))
            if not any(abs(value - truth) <= max(REVENUE_TOLERANCE_PP, abs(truth) * 0.02) for truth in true_yoy):
                hits.append(f"{sentence} [cited {value:+g}%, filed quarters: "
                            f"{', '.join(f'{t:+.1f}%' for t in true_yoy[:4])}]")
    return hits


def check_cpi_yoy(text: str, true_cpi: Optional[float]) -> List[str]:
    """CPI 同比引用与 FRED 真实同比不符（表格行、“6.3% 的 CPI”这类写法都算）；环比/核心/预期口径不比。"""
    if true_cpi is None:
        return []
    hits = []
    for sentence in sentences(text):
        if "CPI" not in sentence or _CPI_SKIP_RE.search(sentence):
            continue
        flat = sentence.replace("*", "").replace("|", " ")
        values = [m.group("val") for m in _CPI_AFTER_RE.finditer(flat)
                  if not _CPI_MONTHLY_RE.search(m.group("gap")) and not _CPI_OTHER_FIGURE_RE.search(m.group("gap"))]
        values += [m.group("val") for m in _CPI_BEFORE_RE.finditer(flat)
                   if not _VAGUE_PREFIX_RE.search(flat[:m.start()])]
        for raw in values:
            value = _number(raw)
            if abs(value - true_cpi) > CPI_TOLERANCE_PP:
                hits.append(f"{sentence} [cited {value:+g}%, FRED {true_cpi:+.2f}%]")
                break
    return hits


def check_cross_ticker(text: str, ticker: str, known_tickers: Iterable[str]) -> List[str]:
    others = [t for t in known_tickers if t and t != ticker]
    hits = []
    for sentence in sentences(text):
        for other in others:
            for match in re.finditer(rf"(?<![A-Za-z]){re.escape(other)}(?![A-Za-z])", sentence):
                window = sentence[max(0, match.start() - 40):match.end() + 40]
                if _CROSS_CONTEXT_RE.search(window):
                    hits.append(sentence)
                    break
            else:
                continue
            break
    return hits


def check_touch_market(text: str) -> List[str]:
    hits = []
    for match in _TOUCH_RE.finditer(text):
        if _PREDICTION_MARKET_RE.search(text[max(0, match.start() - 300):match.end() + 50]):
            hits.append(" ".join(text[max(0, match.start() - 80):match.end() + 30].split()))
    return hits


# 本股自己的中英文名（它出现在句子里不算“其他公司”）
_SELF_NAMES = {"GOOGL": ("谷歌", "Alphabet", "Google"), "GOOG": ("谷歌", "Alphabet", "Google"),
               "NVDA": ("英伟达", "Nvidia"), "MSFT": ("微软", "Microsoft"), "AAPL": ("苹果", "Apple"),
               "META": ("Meta",), "AMZN": ("亚马逊", "Amazon"), "AVGO": ("博通", "Broadcom"),
               "SKHY": ("海力士", "SK hynix"), "INTC": ("英特尔", "Intel"), "MU": ("美光", "Micron"),
               "ORCL": ("甲骨文", "Oracle"), "TSM": ("台积电", "TSMC")}


def _self_names(ticker: str) -> Tuple[str, ...]:
    return _SELF_NAMES.get(ticker.upper(), ())


def run_rules(text: str, *, ticker: str, history_rows: Optional[int], true_revenue_yoy: List[float],
              true_cpi_yoy: Optional[float], known_tickers: Iterable[str]) -> Dict[str, List[str]]:
    return {
        "misread_short_history_ma": check_short_history_ma(text, history_rows),
        "misread_yoy": check_revenue_yoy(text, true_revenue_yoy, _self_names(ticker)) + check_cpi_yoy(text, true_cpi_yoy),
        "misread_cross_ticker": check_cross_ticker(text, ticker, known_tickers),
        "misread_touch_market": check_touch_market(text),
    }


# 规则所需的参照数据（按 ticker + 分析日取一次）

_YOY_LINE_RE = re.compile(r"^- (?:Revenue|revenue): (.+)$", re.M)
_YOY_VALUE_RE = re.compile(r"\d{4}-\d{2}-\d{2} ([+\-]\d+(?:\.\d+)?)%")
_FRED_YOY_RE = re.compile(r"\*\*Year-over-year:\*\* ([+\-]\d+(?:\.\d+)?)%")


def parse_revenue_yoy(statement_text: str) -> List[float]:
    """从利润表附录（数据质量补丁生成）里取季度营收同比，最新在前。"""
    match = _YOY_LINE_RE.search(statement_text or "")
    if not match:
        return []
    return [float(v) for v in _YOY_VALUE_RE.findall(match.group(1))]


def parse_fred_yoy(fred_text: str) -> Optional[float]:
    match = _FRED_YOY_RE.search(fred_text or "")
    return float(match.group(1)) if match else None


def history_span_at(ticker: str, trade_date: str) -> Tuple[Optional[int], Optional[str]]:
    """分析日当时可用的日线行数和最后一根 K 线的日期（读研究缓存的行情 CSV）。"""
    from src.services.tradingagents_service import TRADINGAGENTS_DATA_DIR

    path = TRADINGAGENTS_DATA_DIR / "cache" / f"{ticker}-YFin-data.csv"
    try:
        with path.open(encoding="utf-8") as handle:
            dates = [d for d in ((row.get("Date") or "")[:10] for row in csv.DictReader(handle)) if d and d <= trade_date]
    except OSError:
        return None, None
    return len(dates), max(dates) if dates else None


def history_rows_at(ticker: str, trade_date: str) -> Optional[int]:
    return history_span_at(ticker, trade_date)[0]


def load_known_tickers() -> List[str]:
    from src.services.tradingagents_service import TRADINGAGENTS_DATA_DIR

    path = TRADINGAGENTS_DATA_DIR / "memory" / "trading_memory.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    return sorted(set(re.findall(r"^\[\d{4}-\d{2}-\d{2} \| ([A-Z0-9.^=-]+) \|", text, re.M)))


# ---------------------------------------------------------------------------
# 3. LLM 评审 + 人工评审表
# ---------------------------------------------------------------------------

JUDGE_SECTION_LIMIT = 6000
JUDGE_RISK_LIMIT = 2500
JUDGE_SECTIONS = (
    ("Portfolio manager decision", "5_portfolio/decision.md", JUDGE_SECTION_LIMIT),
    ("Research manager plan", "2_research/manager.md", JUDGE_SECTION_LIMIT),
    ("Trader plan", "3_trading/trader.md", JUDGE_SECTION_LIMIT),
    ("Aggressive risk analyst", "4_risk/aggressive.md", JUDGE_RISK_LIMIT),
    ("Conservative risk analyst", "4_risk/conservative.md", JUDGE_RISK_LIMIT),
    ("Neutral risk analyst", "4_risk/neutral.md", JUDGE_RISK_LIMIT),
    ("Market analyst", "1_analysts/market.md", JUDGE_SECTION_LIMIT),
    ("Sentiment analyst", "1_analysts/sentiment.md", JUDGE_SECTION_LIMIT),
    ("News analyst", "1_analysts/news.md", JUDGE_SECTION_LIMIT),
    ("Fundamentals analyst", "1_analysts/fundamentals.md", JUDGE_SECTION_LIMIT),
)
# 各角色自己给出的评级：评审判断“多空漂移”和“标签与动作是否一致”的事实依据
ROLE_FILES = (
    ("market analyst", "1_analysts/market.md"), ("sentiment analyst", "1_analysts/sentiment.md"),
    ("news analyst", "1_analysts/news.md"), ("fundamentals analyst", "1_analysts/fundamentals.md"),
    ("bull researcher", "2_research/bull.md"), ("bear researcher", "2_research/bear.md"),
    ("research manager", "2_research/manager.md"), ("trader", "3_trading/trader.md"),
    ("aggressive risk", "4_risk/aggressive.md"), ("conservative risk", "4_risk/conservative.md"),
    ("neutral risk", "4_risk/neutral.md"), ("portfolio manager", "5_portfolio/decision.md"),
)
ANALYST_ROLES = ("market analyst", "sentiment analyst", "news analyst", "fundamentals analyst")
_ROLE_RATING_RE = re.compile(r"\*\*(?:Rating|Recommendation|Action)\*\*\s*[:：]\s*([A-Za-z][A-Za-z ]*[A-Za-z])")
_ROLE_PROPOSAL_RE = re.compile(r"FINAL TRANSACTION PROPOSAL:?\s*\**\s*([A-Za-z]+)", re.I)
RATING_SCALE = {"buy": 2, "overweight": 1, "mildly bullish": 1, "bullish": 1, "hold": 0, "mixed": 0,
                "neutral": 0, "mildly bearish": -1, "bearish": -1, "underweight": -1, "sell": -2}
JUDGE_PROMPT = """You review an equity research report produced by a team of AI agents (analysts -> bull/bear \
debate -> research manager -> trader -> risk debate -> portfolio manager). Judge the report itself, not whether \
the stock later went up or down. Use the full scale: "poor" for a real defect, "good" when there is no material \
slip. Do not default to the middle label.

The user message starts with VERIFIED FACTS computed from the raw data: the price-history length, the filed \
revenue growth, the FRED CPI figure, each role's own rating, and sentences flagged by deterministic checks. \
Treat these facts as true. A flagged sentence is a candidate: check in the report whether it is really wrong \
and whether the decision relies on it. An error made by one debater that the research manager or portfolio \
manager does not use, or explicitly corrects, is at most "acceptable".

Rate each dimension "good", "acceptable" or "poor":
- evidence_grounding
  good: the numbers that drive the decision trace back to the analysts' data and are read correctly.
  acceptable: minor slips or unverifiable side claims that do not drive the decision.
  poor: a claim the decision relies on is wrong, invented or misread. Typical cases: a 200-day average, \
52-week figure or long drawdown for a stock whose history is too short to have one; a growth rate against the \
wrong base quarter or contradicting the filed figure; a macro figure contradicting FRED; touch-type \
prediction-market odds (probability that the price touches a level) read as closing-price forecasts or summed \
into a distribution; another stock's past trade result used as evidence; cumulative figures quoted as one quarter.
- consistency
  good: rating, entry/stop/targets and position sizes agree across research manager, trader, risk debate and \
portfolio manager; stops and targets sit on the correct side of the price; the label matches the actions.
  acceptable: small numeric mismatches between roles that do not change the action.
  poor: a real contradiction - the final label does not match the prescribed sizing ("Sell" while keeping a \
30-60% position, "Hold" while cutting to half, "Overweight" with a below-standard target), a stop on the wrong \
side of the price, numbers that cannot all be true, or the portfolio manager reversing the plan without saying \
why. The trader's scale is only Buy/Hold/Sell, so a trader "Sell" under an Underweight plan is a mapping and not \
by itself a contradiction; judge the final label against the final actions.
- actionability
  good: concrete price levels, position sizes, triggers and a time horizon.
  acceptable: some of these missing or vague. poor: no usable levels or sizing.
- balance
  good: the final decision weighs the strongest bull AND bear evidence and the conclusion follows; any move \
away from the analysts' own ratings is justified by evidence the analysts had not already weighed.
  acceptable: one side under-weighted but acknowledged, or a one-notch move with a reasoned justification.
  poor: the conclusion drifts away from the analysts without new evidence - e.g. analysts mostly Hold/Buy \
(and even the bear researcher at Hold) but the final call is Underweight/Sell because the research manager, \
trader and risk team kept escalating - or the other side's strongest points are ignored.

Before labelling, fill "checks" from the report: the current price; the final label; the final target \
exposure as stated by the portfolio manager (e.g. "0.6x standard", "under 30%", "0%"); every stop / exit level \
with the side it sits on for the position that remains; and whether the final label fits that exposure \
(Buy/Overweight above 1x standard, Hold about 1x, Underweight clearly below 1x but above 0, Sell 0 or close to it). \
Then list the concrete defects (quote the numbers) and give the labels. Return JSON only:
{"checks": {"price": "...", "final_label": "...", "final_exposure": "...", "stops": "...", \
"label_fits_exposure": true}, "issues": ["<defect>", ...], \
"evidence_grounding": {"label": "...", "reason": "<= 2 sentences"}, "consistency": {...}, "actionability": {...}, \
"balance": {...}, "overall": {"reason": "<= 2 sentences"}}"""


def role_ratings(report: Report) -> Dict[str, str]:
    """每个角色在自己那份报告里给出的评级（取不到的角色不列）。"""
    ratings = {}
    for role, relative in ROLE_FILES:
        body = report.read(relative)
        match = _ROLE_RATING_RE.search(body) or _ROLE_PROPOSAL_RE.search(body)
        if match:
            ratings[role] = match.group(1).strip().title()
    return ratings


def rating_drift(ratings: Dict[str, str]) -> Optional[float]:
    """最终评级相对分析师平均评级偏了几档（正=更看多，负=更看空）。"""
    analysts = [RATING_SCALE[ratings[r].lower()] for r in ANALYST_ROLES if ratings.get(r, "").lower() in RATING_SCALE]
    final = RATING_SCALE.get(ratings.get("portfolio manager", "").lower())
    if not analysts or final is None:
        return None
    return final - sum(analysts) / len(analysts)


def judge_facts(report: Report, *, history: Tuple[Optional[int], Optional[str]], revenue_yoy: List[float],
                cpi_yoy: Optional[float], rule_hits: Dict[str, List[str]]) -> str:
    """评审前置的确定性事实：模型自己核对不了的数据（历史长度、真实同比、CPI、各角色评级、规则命中句）。"""
    lines = []
    rows, last_date = history
    if rows is not None:
        line = f"- Daily price history up to the analysis date: {rows} bars, last bar {last_date}."
        if rows < 200:
            line += " Too short for a 200-day average or a 52-week statistic."
        if last_date and (datetime.strptime(report.trade_date, "%Y-%m-%d")
                          - datetime.strptime(last_date, "%Y-%m-%d")).days > 7:
            line += " The price data is stale (trading stopped or the feed ended)."
        lines.append(line)
    if revenue_yoy:
        lines.append("- Filed total-revenue YoY by quarter, latest first: "
                     + ", ".join(f"{v:+.1f}%" for v in revenue_yoy[:4])
                     + ". A cited figure within about 1pp of one of these is correct; segment, cost or other-company "
                       "growth rates are not comparable.")
    if cpi_yoy is not None:
        lines.append(f"- FRED headline CPI YoY, latest month: {cpi_yoy:+.2f}%. Only a cited headline CPI YoY figure "
                     "that differs from this is an error; the analysts may not have had FRED data, so saying it "
                     "was unavailable is not an error.")
    ratings = role_ratings(report)
    if ratings:
        lines.append("- Rating each role gave: " + "; ".join(f"{role} {rating}" for role, rating in ratings.items()))
        drift = rating_drift(ratings)
        if drift is not None:
            lines.append(f"- Final rating vs the analysts' average: {drift:+.1f} notches "
                         "(Sell -2, Underweight -1, Hold 0, Overweight +1, Buy +2).")
    for name, found in rule_hits.items():
        for sentence in found[:3]:
            lines.append(f"- Flagged by {name}: {sentence[:300]}")
    return "VERIFIED FACTS\n" + "\n".join(lines) if lines else ""


def judge_input(report: Report, facts: str = "") -> str:
    parts = [f"Ticker: {report.ticker}\nAnalysis date: {report.trade_date}\nFinal rating: {report.rating}"]
    if facts:
        parts.append(facts)
    for title, relative, limit in JUDGE_SECTIONS:
        body = report.read(relative).strip()
        if body:
            clipped = body[:limit] + ("\n[...truncated]" if len(body) > limit else "")
            parts.append(f"## {title}\n{clipped}")
    return "\n\n".join(parts)


def overall_label(labels: Dict[str, str]) -> str:
    """总体分按 rubric 推导：证据或一致性 poor 即 poor；没有 poor 且至少两项 good 才 good。"""
    if labels.get("evidence_grounding") == "poor" or labels.get("consistency") == "poor":
        return "poor"
    if "poor" not in labels.values() and sum(1 for v in labels.values() if v == "good") >= 2:
        return "good"
    return "acceptable"


BALANCE_DRIFT_CAP = 1.0


def apply_rubric_caps(result: Dict[str, Any], drift: Optional[float]) -> Dict[str, Any]:
    """rubric 的硬约束：最终评级偏离分析师平均 ≥1 档时，“多空平衡”最多 acceptable；总体分随之重算。"""
    if drift is not None and abs(drift) >= BALANCE_DRIFT_CAP and result["balance"]["label"] == "good":
        reason = f"[capped: final rating {drift:+.1f} notches from the analysts] {result['balance']['reason']}"
        result["balance"] = {"label": "acceptable", "reason": reason[:400]}
        result["overall"]["label"] = overall_label({k: result[k]["label"] for k in DIMENSIONS})
    return result


def parse_judge_output(raw: str) -> Optional[Dict[str, Any]]:
    """解析评审 JSON；维度标签不合法或缺失时返回 None（不写分数，下次重试）。总体标签由维度推导。"""
    try:
        data = json.loads(re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.M).strip())
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    result: Dict[str, Any] = {}
    for key in DIMENSIONS:
        item = data.get(key)
        label = str(item.get("label") or "").strip().lower() if isinstance(item, dict) else ""
        if label not in LABELS:
            return None
        result[key] = {"label": label, "reason": str(item.get("reason") or "")[:400]}
    raw_issues = data.get("issues")
    issues = [str(i)[:200] for i in raw_issues if i][:6] if isinstance(raw_issues, list) else []
    overall = data.get("overall") if isinstance(data.get("overall"), dict) else {}
    result["overall"] = {"label": overall_label({k: result[k]["label"] for k in DIMENSIONS}),
                         "reason": str(overall.get("reason") or "; ".join(issues))[:400]}
    result["issues"] = issues
    return result


def judge_model_kwargs(config: Any) -> Dict[str, Any]:
    """评审模型的 LiteLLM 参数：设了 TRADINGAGENTS_EVAL_JUDGE_MODEL 就单独走它，否则用 LITELLM_MODEL。

    评审和研究用同一个模型会偏向自己的输出，所以评审可以单独指定一个（OpenAI 兼容）模型。
    """
    from src.config import extra_litellm_params, get_api_keys_for_model

    judge_model = os.environ.get(JUDGE_MODEL_ENV, "").strip()
    if judge_model:
        kwargs: Dict[str, Any] = {"model": judge_model}
        api_key = os.environ.get(JUDGE_API_KEY_ENV, "").strip()
        base_url = os.environ.get(JUDGE_BASE_URL_ENV, "").strip()
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["api_base"] = base_url
        proxy = os.environ.get(JUDGE_PROXY_ENV, "").strip()
        if proxy:
            import httpx
            from openai import OpenAI

            kwargs["client"] = OpenAI(api_key=api_key or None, base_url=base_url or None,
                                      http_client=httpx.Client(proxy=proxy, timeout=300))
        return kwargs

    model = (getattr(config, "litellm_model", "") or "").strip()
    if not model:
        raise RuntimeError("LITELLM_MODEL is not configured")
    for entry in getattr(config, "llm_model_list", None) or []:
        if isinstance(entry, dict) and entry.get("model_name") == model:
            kwargs = dict(entry.get("litellm_params") or {})
            break
    else:
        keys = get_api_keys_for_model(model, config)
        kwargs = {"model": model, **({"api_key": keys[0]} if keys else {}), **extra_litellm_params(model, config)}
    kwargs.setdefault("model", model)
    return kwargs


def judge_version() -> str:
    """换评审模型等于换了一把尺子：版本号带上模型名，旧分数保留，全部报告按新模型重评。"""
    judge_model = os.environ.get(JUDGE_MODEL_ENV, "").strip()
    return f"{JUDGE_VERSION}@{judge_model.rsplit('/', 1)[-1]}" if judge_model else JUDGE_VERSION


def llm_complete(config: Any, system: str, user: str) -> str:
    """调一次评审模型（默认与 TradingAgents 的模型路由一致，见 ``judge_model_kwargs``）。"""
    import litellm

    kwargs = judge_model_kwargs(config)
    response = litellm.completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        # GPT-5+ 推理模型只接受 temperature=1，drop_params 让 LiteLLM 丢掉不支持的参数而不是报错
        temperature=0, response_format={"type": "json_object"}, drop_params=True, **kwargs,
    )
    return response.choices[0].message.content or ""


def write_review_sheet(reports: List[Report], path: Path, existing: Optional[Dict[str, Dict[str, str]]] = None) -> Path:
    """导出人工评审表：不含 LLM 结论；已填过的行保留。"""
    existing = existing or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for report in reports:
            row = {"report_id": report.report_id, "ticker": report.ticker, "trade_date": report.trade_date,
                   "rating": report.rating or "",
                   "report_url": f"/api/v1/tradingagents/reports/{report.report_id}/html"}
            for column in (*DIMENSIONS, "overall", "note"):
                row[column] = (existing.get(report.report_id) or {}).get(column, "")
            writer.writerow(row)
    return path


def read_review_sheet(path: Path) -> Dict[str, Dict[str, str]]:
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            labels = {key: (row.get(key) or "").strip().lower() for key in (*DIMENSIONS, "overall")}
            if any(labels.values()):
                bad = [f"{k}={v}" for k, v in labels.items() if v and v not in LABELS]
                if bad:
                    raise ValueError(f"{row.get('report_id')}: invalid labels {bad}; use {'/'.join(LABELS)}")
                rows[row["report_id"]] = {**labels, "note": (row.get("note") or "").strip()}
    return rows


def agreement(judge: Dict[str, Dict[str, Any]], human: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """LLM 与人工逐维一致率（只算两边都有标签的报告）。"""
    per_dim: Dict[str, Dict[str, int]] = {}
    mismatches = []
    for report_id, labels in human.items():
        result = judge.get(report_id)
        if not result:
            continue
        for key in (*DIMENSIONS, "overall"):
            human_label = labels.get(key)
            judge_label = (result.get(key) or {}).get("label")
            if not human_label or not judge_label:
                continue
            stats = per_dim.setdefault(key, {"n": 0, "agree": 0})
            stats["n"] += 1
            if human_label == judge_label:
                stats["agree"] += 1
            else:
                mismatches.append((report_id, key, human_label, judge_label))
    return {"per_dimension": {k: {**v, "rate": v["agree"] / v["n"]} for k, v in per_dim.items() if v["n"]},
            "mismatches": mismatches}


# ---------------------------------------------------------------------------
# Langfuse：回填 trace + 写分数
# ---------------------------------------------------------------------------

def _score_id(report_id: str, name: str, version: str = "") -> str:
    return hashlib.sha256(f"{report_id}|{name}|{version}".encode()).hexdigest()[:32]


class ScoreSink:
    """把分数写到 Langfuse；未启用时只记本地状态。"""

    def __init__(self, client: Any = None):
        self.client = client

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def trace_for(self, report: Report, state: Dict[str, Any]) -> Optional[str]:
        """报告对应的 trace：研究时记下的真实 trace，或回填一条。"""
        if state.get("trace_id"):
            return state["trace_id"]
        linked = report.read("0_reference/langfuse.json")
        if linked:
            try:
                state["trace_id"] = json.loads(linked)["trace_id"]
                return state["trace_id"]
            except (ValueError, KeyError, TypeError):
                pass
        if not self.enabled:
            return None
        from langfuse import propagate_attributes

        from src.services.tradingagents_langfuse import ROOT_NAME, TRACE_NAME

        trace_id = self.client.create_trace_id(seed=report.report_id)
        with propagate_attributes(
            trace_name=TRACE_NAME, tags=["tradingagents", "backfill"],
            metadata={"ticker": report.ticker, "trade_date": report.trade_date, "report_id": report.report_id},
        ):
            root = self.client.start_observation(
                trace_context={"trace_id": trace_id}, name=ROOT_NAME, as_type="agent",
                input={"ticker": report.ticker, "trade_date": report.trade_date},
                output={"rating": report.rating, "report_id": report.report_id},
                metadata={"backfill": "true", "note": "created after the run; timestamp is the backfill date"},
            )
            root.end()
        state["trace_id"] = trace_id
        state["backfilled"] = True
        return trace_id

    def score(self, report: Report, state: Dict[str, Any], name: str, value: Any, data_type: str,
              comment: str = "", version: str = "", id_seed: Optional[str] = None) -> None:
        written = state.setdefault("scores", {})
        key = f"{name}@{version}" if version else name
        if key in written:
            return
        # 规则/评审升版本后结果没变：只记状态，不在 Langfuse 里再写一条
        if version and any(k.startswith(f"{name}@") and v == value for k, v in written.items()):
            written[key] = value
            return
        trace_id = self.trace_for(report, state)
        if self.enabled and trace_id:
            self.client.create_score(
                name=name, value=value, trace_id=trace_id, data_type=data_type,
                comment=(comment or "")[:1000] or None,
                score_id=_score_id(report.report_id, name, version if id_seed is None else id_seed),
            )
        written[key] = value


# ---------------------------------------------------------------------------
# 编排
# ---------------------------------------------------------------------------

@dataclass
class EvalContext:
    fetch_returns: Callable
    benchmark_for: Callable[[str], str]
    revenue_yoy: Callable[[str, str], List[float]]
    cpi_yoy: Callable[[str], Optional[float]]
    judge: Optional[Callable[[Report, str], Optional[Dict[str, Any]]]] = None
    known_tickers: List[str] = field(default_factory=list)
    judge_version: str = JUDGE_VERSION


def _load_state() -> Dict[str, Any]:
    path = eval_root() / "state.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_state(state: Dict[str, Any]) -> None:
    path = eval_root() / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def evaluate_report(report: Report, state: Dict[str, Any], ctx: EvalContext, sink: ScoreSink) -> None:
    # 评级结果
    outcomes = state.setdefault("outcomes", {})
    if report.rating:
        try:
            outcomes.update(compute_outcomes(report, ctx.fetch_returns, ctx.benchmark_for(report.ticker), outcomes))
        except Exception as exc:  # 行情取不到就下次再算
            logger.info("[TradingAgents/eval] outcome pending for %s: %s", report.report_id, exc)
    for days, result in outcomes.items():
        comment = (f"rating {report.rating}; raw {result['raw'] * 100:+.2f}%, alpha vs {result['benchmark']} "
                   f"{result['alpha'] * 100:+.2f}% through {result['resolved']}")
        sink.score(report, state, f"rating_alpha_{days}d", round(result["alpha"] * 100, 3), "NUMERIC", comment)
        if result["hit"] is not None:
            sink.score(report, state, f"rating_hit_{days}d", 1 if result["hit"] else 0, "BOOLEAN", comment)

    # 数据误读
    if state.get("rules_version") != RULES_VERSION:
        text = report.text
        hits = run_rules(
            text, ticker=report.ticker, history_rows=history_rows_at(report.ticker, report.trade_date),
            true_revenue_yoy=ctx.revenue_yoy(report.ticker, report.trade_date),
            true_cpi_yoy=ctx.cpi_yoy(report.trade_date), known_tickers=ctx.known_tickers,
        )
        state["rules"] = {name: found[:20] for name, found in hits.items()}
        state["rules_version"] = RULES_VERSION
    rules = state["rules"]
    total = sum(len(found) for found in rules.values())
    summary = "; ".join(f"{name}={len(found)}" for name, found in rules.items())
    sink.score(report, state, "data_misread_count", total, "NUMERIC", f"[{RULES_VERSION}] {summary}",
               RULES_VERSION, RULES_SCORE_SEED)
    for name, found in rules.items():
        sink.score(report, state, name, len(found), "NUMERIC",
                   f"[{RULES_VERSION}] " + " | ".join(s[:180] for s in found[:3]), RULES_VERSION, RULES_SCORE_SEED)

    # LLM 评审（前置规则命中和真实数据，模型自己核对不了这些）
    if ctx.judge is not None and (state.get("judge") or {}).get("version") != ctx.judge_version:
        facts = judge_facts(report, history=history_span_at(report.ticker, report.trade_date),
                            revenue_yoy=ctx.revenue_yoy(report.ticker, report.trade_date),
                            cpi_yoy=ctx.cpi_yoy(report.trade_date), rule_hits=rules)
        try:
            result = ctx.judge(report, facts)
        except Exception as exc:
            logger.warning("[TradingAgents/eval] judge failed for %s: %s", report.report_id, exc)
            result = None
        if result:
            state["judge"] = {"version": ctx.judge_version, **result}
    judged = state.get("judge")
    if judged:
        version = judged.get("version", JUDGE_VERSION)
        sink.score(report, state, "report_quality", judged["overall"]["label"], "CATEGORICAL",
                   f"[{version}] {judged['overall']['reason']}", version)
        for dim in DIMENSIONS:
            sink.score(report, state, f"report_{dim}", judged[dim]["label"], "CATEGORICAL",
                       f"[{version}] {judged[dim]['reason']}", version)


def build_context(config: Any, with_judge: bool) -> EvalContext:
    """接上真实数据源：FMP 行情、SEC/FMP 报表（带数据质量补丁的同比附录）、FRED。"""
    from tradingagents.default_config import build_default_config
    from tradingagents.dataflows.config import set_config
    from tradingagents.dataflows.router import route_to_vendor
    from tradingagents.memory import settlement

    from src.services.tradingagents_quality import install_quality_patches
    from src.services.tradingagents_service import build_tradingagents_config, resolve_data_source

    ta_config = build_tradingagents_config(config, build_default_config)
    if resolve_data_source(config) == "fmp":
        from src.services.tradingagents_fmp import apply_fmp_config, install_fmp_vendor, resolve_fmp_api_key

        install_fmp_vendor(resolve_fmp_api_key(config))
        apply_fmp_config(ta_config)
    install_quality_patches()
    set_config(ta_config)

    revenue_cache: Dict[Tuple[str, str], List[float]] = {}
    cpi_cache: Dict[str, Optional[float]] = {}

    def revenue_yoy(ticker: str, trade_date: str) -> List[float]:
        key = (ticker, trade_date)
        if key not in revenue_cache:
            try:
                revenue_cache[key] = parse_revenue_yoy(
                    route_to_vendor("get_income_statement", ticker, "quarterly", trade_date))
            except Exception as exc:
                logger.info("[TradingAgents/eval] no statements for %s: %s", ticker, exc)
                revenue_cache[key] = []
        return revenue_cache[key]

    def cpi_yoy(trade_date: str) -> Optional[float]:
        if trade_date not in cpi_cache:
            try:
                cpi_cache[trade_date] = parse_fred_yoy(route_to_vendor("get_macro_indicators", "cpi", trade_date, 400))
            except Exception as exc:
                logger.info("[TradingAgents/eval] FRED CPI unavailable: %s", exc)
                cpi_cache[trade_date] = None
        return cpi_cache[trade_date]

    def judge(report: Report, facts: str) -> Optional[Dict[str, Any]]:
        result = parse_judge_output(llm_complete(config, JUDGE_PROMPT, judge_input(report, facts)))
        return apply_rubric_caps(result, rating_drift(role_ratings(report))) if result else None

    return EvalContext(
        fetch_returns=settlement.fetch_returns,
        benchmark_for=lambda ticker: settlement.resolve_benchmark(ticker, ta_config),
        revenue_yoy=revenue_yoy, cpi_yoy=cpi_yoy,
        judge=judge if with_judge else None, known_tickers=load_known_tickers(),
        judge_version=judge_version(),
    )


def run_evaluation(config: Any, *, with_judge: Optional[bool] = None,
                   report_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """评估全部（或指定）报告；幂等，可每天跑。"""
    from src.agent.langfuse_observability import langfuse_client

    with_judge = _env_flag(EVAL_JUDGE_ENV) if with_judge is None else with_judge
    with _lock:
        reports = [r for r in list_reports() if not report_ids or r.report_id in report_ids]
        ctx = build_context(config, with_judge)
        client = langfuse_client()
        sink = ScoreSink(client)
        state = _load_state()
        for report in reports:
            entry = state.setdefault(report.report_id, {})
            try:
                evaluate_report(report, entry, ctx, sink)
            except Exception as exc:  # 单份报告失败不影响其他
                logger.warning("[TradingAgents/eval] %s failed: %s", report.report_id, exc)
            _save_state(state)
        if client is not None:
            client.flush()
    return summarize(reports, state)


def summarize(reports: List[Report], state: Dict[str, Any]) -> Dict[str, Any]:
    """按评级分组的命中率 / 平均 alpha（同股同日多次研究只取最新一份）、误读分布、评审一致率。"""
    latest: Dict[Tuple[str, str], Report] = {}
    for report in reports:
        key = (report.ticker, report.trade_date)
        if key not in latest or report.report_id > latest[key].report_id:
            latest[key] = report
    outcome_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for report in latest.values():
        for days, result in (state.get(report.report_id) or {}).get("outcomes", {}).items():
            bucket = outcome_stats.setdefault(f"{days}d", {}).setdefault(report.rating or "unknown",
                                                                         {"n": 0, "hits": 0, "alpha_sum": 0.0})
            bucket["n"] += 1
            bucket["hits"] += 1 if result.get("hit") else 0
            bucket["alpha_sum"] += result["alpha"]
    for horizon in outcome_stats.values():
        for bucket in horizon.values():
            bucket["hit_rate"] = bucket["hits"] / bucket["n"]
            bucket["avg_alpha_pct"] = bucket.pop("alpha_sum") / bucket["n"] * 100
    misread = {name: sum(len((state.get(r.report_id) or {}).get("rules", {}).get(name, [])) for r in reports)
               for name in RULES}
    reports_with_misread = sum(1 for r in reports if any((state.get(r.report_id) or {}).get("rules", {}).values()))
    judge = {r.report_id: state[r.report_id]["judge"] for r in reports if (state.get(r.report_id) or {}).get("judge")}
    references = {source: {r.report_id: state[r.report_id][source] for r in reports
                           if (state.get(r.report_id) or {}).get(source)} for source in REVIEW_SOURCES}
    return {
        "reports": len(reports), "unique_ticker_dates": len(latest),
        "outcomes": outcome_stats,
        "misread_sentences": misread, "reports_with_misread": reports_with_misread,
        "judge_labels": {k: sum(1 for j in judge.values() if j["overall"]["label"] == k) for k in LABELS},
        "judge_version": sorted({j.get("version", "") for j in judge.values()}),
        "agreement": {source: agreement(judge, labels) for source, labels in references.items() if labels} or None,
    }


def import_review(config: Any, sheet: Path, source: str = "human") -> Dict[str, Any]:
    """导入评审表。source=human 为人工打分；source=claude 为 Claude 代评的参考标签（分数名带后缀区分）。"""
    from src.agent.langfuse_observability import langfuse_client

    if source not in REVIEW_SOURCES:
        raise ValueError(f"source must be one of {REVIEW_SOURCES}")
    labeled = read_review_sheet(sheet)
    with _lock:
        reports = {r.report_id: r for r in list_reports()}
        client = langfuse_client()
        sink = ScoreSink(client)
        state = _load_state()
        for report_id, labels in labeled.items():
            report = reports.get(report_id)
            if report is None:
                continue
            entry = state.setdefault(report_id, {})
            entry[source] = labels
            # 评审分可能改动：版本号带上标签，改了就写一条新分数
            version = hashlib.sha256(json.dumps(labels, sort_keys=True).encode()).hexdigest()[:8]
            if labels.get("overall"):
                sink.score(report, entry, f"report_quality_{source}", labels["overall"], "CATEGORICAL",
                           labels.get("note", ""), version)
            for dim in DIMENSIONS:
                if labels.get(dim):
                    sink.score(report, entry, f"report_{dim}_{source}", labels[dim], "CATEGORICAL", "", version)
        _save_state(state)
        if client is not None:
            client.flush()
        judge = {rid: state[rid]["judge"] for rid in labeled if (state.get(rid) or {}).get("judge")}
    return agreement(judge, labeled)


def main(argv: Optional[List[str]] = None) -> int:
    from src.config import get_config, setup_env

    setup_env()
    parser = argparse.ArgumentParser(prog="tradingagents_eval")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="evaluate reports and write scores to Langfuse")
    run.add_argument("--report", action="append", help="only this report id (repeatable)")
    run.add_argument("--no-judge", action="store_true", help="skip the LLM review")
    sub.add_parser("summary", help="print hit rates, misread counts and judge agreement")
    sheet = sub.add_parser("review-sheet", help="export the human review sheet (CSV)")
    sheet.add_argument("--out", default=None)
    imp = sub.add_parser("import-review", help="import a filled review sheet")
    imp.add_argument("csv")
    imp.add_argument("--source", choices=REVIEW_SOURCES, default="human",
                     help="human (default) or claude (reference labels produced by Claude)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = get_config()

    if args.command == "run":
        result = run_evaluation(config, with_judge=False if args.no_judge else None, report_ids=args.report)
    elif args.command == "summary":
        result = summarize(list_reports(), _load_state())
    elif args.command == "review-sheet":
        out = Path(args.out) if args.out else eval_root() / "review_sheet.csv"
        state = _load_state()
        existing = {rid: v["human"] for rid, v in state.items() if isinstance(v, dict) and v.get("human")}
        result = {"sheet": str(write_review_sheet(list_reports(), out, existing))}
    else:
        result = import_review(config, Path(args.csv), args.source)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
