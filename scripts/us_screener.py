#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FMP 美股全市场选股器
====================
用 Financial Modeling Prep 的 company-screener 按条件筛选全美股候选，
可选做「动量过滤」（价格在 50/200 日均线之上）并排序，最后可 --analyze
把 Top N 喂给现有分析流水线（main.py）打分并推送飞书。

依赖：requests + 环境变量 FMP_API_KEY（容器内已配）。

示例：
  # 纯筛选（不分析），大盘蓝筹 + 动量，取前 10
  python scripts/us_screener.py --preset largecap --top 10

  # 科技成长股，筛完直接分析并推飞书
  python scripts/us_screener.py --preset growth_tech --top 8 --analyze

  # 自定义条件
  python scripts/us_screener.py --market-cap-min 20e9 --sector Technology --price-min 10 --top 10 --analyze
"""
import argparse
import os
import subprocess
import sys

import requests

FMP_BASE = "https://financialmodelingprep.com/stable"
API_KEY = os.getenv("FMP_API_KEY", "").strip()

# 预设筛选条件（全部限定 country=US、剔除 ETF/基金、仅活跃交易）
PRESETS = {
    # 大盘蓝筹：市值 > 500 亿美元、价 > 10、日均量 > 200 万
    "largecap": {
        "marketCapMoreThan": 50_000_000_000,
        "priceMoreThan": 10,
        "volumeMoreThan": 2_000_000,
    },
    # 科技成长：科技板块、市值 > 100 亿、价 > 10、日均量 > 100 万
    "growth_tech": {
        "marketCapMoreThan": 10_000_000_000,
        "sector": "Technology",
        "priceMoreThan": 10,
        "volumeMoreThan": 1_000_000,
    },
    # 价值/防御：市值 > 100 亿、有分红、低 beta(<1.2)
    "value": {
        "marketCapMoreThan": 10_000_000_000,
        "priceMoreThan": 10,
        "dividendMoreThan": 0.01,
        "betaLowerThan": 1.2,
    },
}


def screen(params: dict, limit: int) -> list:
    """调用 FMP company-screener 返回候选列表。"""
    q = dict(params)
    q.update({
        "country": q.get("country", "US"),
        "isActivelyTrading": "true",
        "isEtf": "false",
        "isFund": "false",
        "limit": limit,
        "apikey": API_KEY,
    })
    resp = requests.get(f"{FMP_BASE}/company-screener", params=q, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and data.get("Error Message"):
        raise SystemExit(f"[FMP] screener 报错: {data['Error Message']}")
    if not isinstance(data, list):
        raise SystemExit(f"[FMP] screener 返回异常: {str(data)[:200]}")
    return data


def momentum_rank(rows: list) -> list:
    """保留价格在 50 日均线之上（且不低于 200 日均线）的上升趋势股，按高出 50 日均线的幅度排序。"""
    scored = []
    for row in rows:
        sym = row.get("symbol")
        if not sym:
            continue
        try:
            q = requests.get(
                f"{FMP_BASE}/quote", params={"symbol": sym, "apikey": API_KEY}, timeout=15
            ).json()
        except Exception:
            continue
        if not isinstance(q, list) or not q:
            continue
        q = q[0]
        price = q.get("price")
        ma50 = q.get("priceAvg50")
        ma200 = q.get("priceAvg200")
        if not (price and ma50):
            continue
        # 上升趋势：站上 50 日线；若有 200 日线也要站上
        if price <= ma50:
            continue
        if ma200 and price < ma200:
            continue
        row["_above_ma50_pct"] = round((price / ma50 - 1) * 100, 2)
        row["_price"] = price
        scored.append(row)
    scored.sort(key=lambda r: r.get("_above_ma50_pct", 0), reverse=True)
    return scored


def fmt_cap(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    if v >= 1e12:
        return f"{v/1e12:.2f}万亿"
    if v >= 1e8:
        return f"{v/1e8:.0f}亿"
    return f"{v:.0f}"


def main() -> int:
    if not API_KEY:
        print("错误：未配置 FMP_API_KEY 环境变量", file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(description="FMP 美股全市场选股器")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="largecap",
                        help="预设筛选条件（默认 largecap）")
    parser.add_argument("--sector", default=None, help="覆盖板块，如 Technology / Healthcare / Financial Services")
    parser.add_argument("--market-cap-min", type=float, default=None, help="最小市值(美元)，如 20e9")
    parser.add_argument("--price-min", type=float, default=None, help="最低股价")
    parser.add_argument("--volume-min", type=float, default=None, help="最低日均成交量")
    parser.add_argument("--exchange", default=None, help="交易所过滤，如 NASDAQ / NYSE")
    parser.add_argument("--screen-limit", type=int, default=60, help="初筛返回条数上限（默认 60）")
    parser.add_argument("--top", type=int, default=10, help="最终候选数（默认 10）")
    parser.add_argument("--no-momentum", action="store_true", help="关闭动量过滤（默认开启）")
    parser.add_argument("--analyze", action="store_true", help="对 Top N 调用 main.py 分析并推送飞书")
    args = parser.parse_args()

    params = dict(PRESETS[args.preset])
    if args.sector:
        params["sector"] = args.sector
    if args.market_cap_min is not None:
        params["marketCapMoreThan"] = int(args.market_cap_min)
    if args.price_min is not None:
        params["priceMoreThan"] = args.price_min
    if args.volume_min is not None:
        params["volumeMoreThan"] = int(args.volume_min)
    if args.exchange:
        params["exchange"] = args.exchange

    print(f"[选股] 预设={args.preset} 条件={ {k: v for k, v in params.items()} }")
    rows = screen(params, args.screen_limit)
    print(f"[选股] 初筛命中 {len(rows)} 只")

    if not args.no_momentum:
        rows = momentum_rank(rows)
        print(f"[选股] 动量过滤后（站上50/200日均线）{len(rows)} 只，按趋势强度排序")

    top = rows[: args.top]
    if not top:
        print("[选股] 无符合条件的候选。")
        return 0

    print(f"\n=== Top {len(top)} 候选美股 ===")
    print(f"{'代码':<8}{'名称':<28}{'板块':<22}{'市值':<10}{'现价':<10}{'高出50日线':<10}")
    for r in top:
        print(
            f"{r.get('symbol',''):<8}"
            f"{(r.get('companyName','') or '')[:26]:<28}"
            f"{(r.get('sector','') or '')[:20]:<22}"
            f"{fmt_cap(r.get('marketCap')):<10}"
            f"{str(r.get('_price') or r.get('price') or '-'):<10}"
            f"{str(r.get('_above_ma50_pct','-'))+'%':<10}"
        )

    symbols = [r["symbol"] for r in top if r.get("symbol")]
    print(f"\n候选代码: {','.join(symbols)}")

    if args.analyze:
        print(f"\n[分析] 对 {len(symbols)} 只候选运行分析流水线并推送飞书...")
        env = dict(os.environ)
        env.update({
            "SCHEDULE_ENABLED": "false",
            "WEBUI_ENABLED": "false",
            "SCHEDULE_RUN_IMMEDIATELY": "true",
        })
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        main_py = os.path.join(app_dir, "main.py")
        cmd = [sys.executable, main_py, "--stocks", ",".join(symbols),
               "--force-run", "--no-market-review"]
        rc = subprocess.run(cmd, env=env).returncode
        print(f"[分析] 完成，退出码 {rc}")
        return rc

    print("\n提示：加 --analyze 可直接对以上候选打分（评分/买卖建议/止盈止损）并推送飞书。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
