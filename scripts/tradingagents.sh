#!/usr/bin/env bash
# TradingAgents 深度研究：在独立的一次性容器里运行，不占用常驻 stock-server 的内存。
#
# 用法（服务器上）：
#   sudo ./scripts/tradingagents.sh AAPL
#   sudo ./scripts/tradingagents.sh NVDA,MSFT --no-notify
#   sudo ./scripts/tradingagents.sh AAPL --tradingagents-date 2026-10-14
#   sudo ./scripts/tradingagents.sh NVDA --tradingagents-polymarket "Anthropic IPO,OpenAI IPO"
#
# 模型 / 数据源 / 分析师等参数读取 .env 中的 TRADINGAGENTS_* 变量，详见 docs/tradingagents-integration.md。
# 报告保存在 data/tradingagents/logs/reports/ 下。
set -euo pipefail

if [[ $# -lt 1 || "$1" == -* ]]; then
  echo "usage: $0 CODES [--no-notify] [--tradingagents-date YYYY-MM-DD]" >&2
  exit 2
fi

codes="$1"
shift

cd "$(dirname "$0")/../docker"
exec docker compose run --rm --no-deps \
  -e TRADINGAGENTS_ENABLED=true \
  -e WEBUI_ENABLED=false \
  server python main.py --tradingagents "$codes" "$@"
