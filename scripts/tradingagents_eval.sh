#!/usr/bin/env bash
# TradingAgents 研究评估：评级命中率、数据误读规则、LLM 报告评审，分数写回 Langfuse。
#
# 用法（服务器上）：
#   sudo ./scripts/tradingagents_eval.sh run                 # 评估全部报告（幂等，可重复跑）
#   sudo ./scripts/tradingagents_eval.sh run --no-judge      # 不跑 LLM 评审
#   sudo ./scripts/tradingagents_eval.sh summary             # 命中率 / 误读 / 评审一致率汇总
#   sudo ./scripts/tradingagents_eval.sh review-sheet        # 导出人工评审表 data/tradingagents/eval/review_sheet.csv
#   sudo ./scripts/tradingagents_eval.sh import-review data/tradingagents/eval/review_sheet.csv
#
# 详见 docs/tradingagents-integration.md「评估」一节。
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 run|summary|review-sheet|import-review <csv> [options]" >&2
  exit 2
fi

cd "$(dirname "$0")/../docker"
exec docker compose run --rm --no-deps \
  -e TRADINGAGENTS_ENABLED=true \
  -e WEBUI_ENABLED=false \
  server python -m src.services.tradingagents_eval "$@"
