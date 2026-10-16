# TradingAgents 多智能体深度研究集成

[TradingAgents](https://github.com/TauricResearch/TradingAgents)（TauricResearch，v0.6.0）是一个基于 LangGraph 的多智能体交易研究框架。它的流程如下：

1. 四类分析师并行出报告：技术、情绪、新闻、基本面。
2. 多方与空方研究员辩论，由研究经理给出投资计划。
3. 交易员提出方案。
4. 激进、保守、中性三位风控分析师讨论。
5. 组合经理给出最终的 5 档评级：Buy / Overweight / Hold / Underweight / Sell。

DSA 把它作为**独立于日常分析的深度研究入口**集成进来，具体约定：

- 不替换现有分析流水线。
- 不写入分析历史，也不改变决策仪表盘的结构。
- 默认关闭，依赖需要单独安装。

> 研究用途，不构成投资建议。LLM 输出具有不确定性，同一标的同一天多次运行的结论可能不同。

## 安装

TradingAgents 未发布到 PyPI，需要 **Python 3.11+**。它额外引入 `langchain-*`、`langgraph` 等依赖，因此不放进主 `requirements.txt`：

```bash
pip install -r requirements.txt
pip install -r requirements-tradingagents.txt   # 固定在 v0.6.0 tag
```

依赖兼容性：

- DSA 主依赖在 Python 3.11+ 上解析出的 pandas、numpy、openai、pydantic 版本与 TradingAgents 一致，都是 pandas 3.x，可以装在同一个环境里。
- Python 3.10 环境无法安装 TradingAgents。

未安装时：

- CLI 和 API 会返回 `tradingagents_not_installed`。
- 其余 DSA 功能不受影响。

## 配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TRADINGAGENTS_ENABLED` | `false` | 总开关 |
| `TRADINGAGENTS_ANALYSTS` | `market,social,news,fundamentals` | 默认启用的分析师，可取其子集。减少分析师可以降低耗时和费用 |

其余参数直接使用 TradingAgents 原生的 `TRADINGAGENTS_*` 环境变量，例如：

- 模型：`TRADINGAGENTS_LLM_PROVIDER`、`TRADINGAGENTS_DEEP_THINK_LLM`、`TRADINGAGENTS_QUICK_THINK_LLM`、`TRADINGAGENTS_LLM_BACKEND_URL`
- 辩论轮数：`TRADINGAGENTS_MAX_DEBATE_ROUNDS`、`TRADINGAGENTS_MAX_RISK_ROUNDS`
- 输出语言：`TRADINGAGENTS_OUTPUT_LANGUAGE`

完整列表见上游 `tradingagents/default_config.py`。

### 模型如何复用 DSA 配置

如果没有设置 `TRADINGAGENTS_LLM_PROVIDER`，DSA 会从主模型配置推导：

1. 读取 `LITELLM_MODEL`。如果它是 `LLM_CHANNELS` 或 `LITELLM_CONFIG` 中的别名，就取该渠道的真实 `model`、`api_key` 和 `api_base`。
2. 按 LiteLLM 前缀映射到 TradingAgents 的 provider：

| LiteLLM 前缀 | TradingAgents provider | API Key 环境变量 |
|---|---|---|
| `gemini/`、`vertex_ai/` | `google` | `GOOGLE_API_KEY` |
| `anthropic/` | `anthropic` | `ANTHROPIC_API_KEY` |
| `deepseek/` | `deepseek` | `DEEPSEEK_API_KEY` |
| `openai/`（可带 `OPENAI_BASE_URL`） | `openai` 加 `backend_url` | `OPENAI_API_KEY` |
| `openrouter/`、`xai/`、`groq/`、`mistral/`、`ollama/` | 同名 provider | 对应变量 |
| 其他前缀，且渠道配置了 `api_base` | `openai_compatible` 加 `backend_url` | `OPENAI_COMPATIBLE_API_KEY` |

3. deep 与 quick 两个档位默认使用同一个模型，可以分别用 `TRADINGAGENTS_DEEP_THINK_LLM` 和 `TRADINGAGENTS_QUICK_THINK_LLM` 覆盖。
   - deep 用于研究经理和组合经理，适合推理更强的模型。
   - quick 用于分析师、辩论和交易员。
4. TradingAgents 只从环境变量读取 API Key。如果对应变量没有设置，DSA 会在运行时把推导出的 Key 写入当前进程的环境变量；已有的值不会被覆盖。
5. `TRADINGAGENTS_OUTPUT_LANGUAGE` 未设置时，按 `REPORT_LANGUAGE` 选择 Chinese、English 或 Korean。

无法映射时，会返回 `tradingagents_llm_unsupported`，例如未知前缀且没有 `api_base`。这时请显式设置 `TRADINGAGENTS_LLM_PROVIDER` 等变量。

### 数据与行情

数据源由 `TRADINGAGENTS_DATA_SOURCE` 决定：

| 取值 | 行为 |
|---|---|
| `auto`（默认） | 配置了 `FMP_API_KEY` 时用 FMP，否则用 TradingAgents 原生的 Yahoo Finance |
| `fmp` | 强制 FMP；未配置 `FMP_API_KEY` 时报 `tradingagents_fmp_not_configured` |
| `yfinance` | 强制 Yahoo Finance |

**FMP 模式**（`src/services/tradingagents_fmp.py`）在不修改 TradingAgents 包的前提下替换它的数据通道，适合 Yahoo 对机房 IP 返回 429 的国内服务器：

| 数据 | 来源 |
|---|---|
| 日线 OHLCV、技术指标、已核验行情快照、历史决策复盘收盘价 | FMP `historical-price-eod/dividend-adjusted` |
| 公司身份（名称、行业、交易所） | FMP `profile` |
| 基本面概览 | FMP `profile` + `quote` + `ratios-ttm` + `key-metrics-ttm`（仅当天运行提供，历史日期按 TradingAgents 规则不提供） |
| 三大报表 | 先 SEC EDGAR（按申报值），再 FMP（按 `filingDate` 过滤到分析日） |
| 个股新闻、全球新闻 | FMP `news/stock`、`news/general-latest` |
| 内部人交易 | FMP `insider-trading/search`（按 `filingDate` 过滤） |

当前 FMP 套餐**只覆盖美股**：港股、A 股代码会得到“数据不可用”提示，不会编造数据。情绪分析师依赖 StockTwits / Reddit，国内机房无法访问，建议把 `TRADINGAGENTS_ANALYSTS` 设为 `market,news,fundamentals`。

**Yahoo 模式**下，TradingAgents 使用自己的数据源，**不走** DSA 的 `data_provider/` fallback 链：

- 行情和新闻：Yahoo Finance
- 美股财报：SEC EDGAR
- 宏观数据：FRED，需要 `FRED_API_KEY`，可选
- 预测市场：Polymarket

DSA 代码会自动转换成 Yahoo 格式：`600519` → `600519.SS`，`hk00700` → `0700.HK`，`AAPL` 保持不变。A 股和港股的基本面数据只有 Yahoo 来源，覆盖度不如美股。

建议设置 `SEC_EDGAR_USER_AGENT="Your Name you@example.com"`，SEC 要求调用方提供联系方式。

### 运行数据目录

默认写入 `data/tradingagents/`，不写 `~/.tradingagents`：

| 内容 | 默认路径 | 覆盖变量 |
|---|---|---|
| 报告（`reports/<TICKER>_<时间>/complete_report.md`、`.html`）和状态日志 | `data/tradingagents/logs` | `TRADINGAGENTS_RESULTS_DIR` |
| 数据缓存 | `data/tradingagents/cache` | `TRADINGAGENTS_CACHE_DIR` |
| 决策记忆日志（跨次运行复盘历史决策收益） | `data/tradingagents/memory/trading_memory.md` | `TRADINGAGENTS_MEMORY_LOG_PATH` |

## 使用

### CLI

```bash
TRADINGAGENTS_ENABLED=true python main.py --tradingagents AAPL,600519
python main.py --tradingagents NVDA --tradingagents-date 2026-09-30 --no-notify
```

- 每只股票依次运行。单只失败不会中断其余股票；全部失败时退出码为 1。
- 结果摘要会打印到终端，包括评级、操作建议、最终决策和报告路径。
- 未加 `--no-notify` 时，摘要会推送到已配置的通知渠道。
- `--tradingagents` 是独立模式，不会执行日常个股分析和大盘复盘。

### API

一次研究大约需要 20-40 次 LLM 调用、耗时数分钟，因此只提供后台任务接口：

```
GET  /api/v1/tradingagents/status            # {enabled, available, version, analysts}
POST /api/v1/tradingagents/tasks             # 202，请求体 {"stock_code": "AAPL", "trade_date": "2026-09-30", "analysts": ["market","news"]}
GET  /api/v1/tradingagents/tasks/{task_id}   # status/progress/error；completed 后 result 为下方结构
```

`result` 的字段：

- 基本信息：`code`、`ticker`、`trade_date`、`analysts`
- 评级：`rating`，取值 Buy / Overweight / Hold / Underweight / Sell / REVIEW
- `operation_advice`：中文报告依次映射为买入 / 加仓 / 持有 / 减仓 / 卖出，REVIEW 映射为观望
- 实际使用的模型：`llm_provider`、`deep_think_llm`、`quick_think_llm`
- 各阶段报告文本：`market_report`、`sentiment_report`、`news_report`、`fundamentals_report`、`investment_plan`、`trader_investment_plan`、`final_trade_decision`
- 报告文件：`report_md_path`、`report_html_path`

错误码：

| 错误码 | HTTP 状态 | 含义 |
|---|---|---|
| `tradingagents_disabled` | 403 | 未启用 |
| `tradingagents_not_installed` | 503 | 未安装依赖 |
| `tradingagents_invalid_*` | 400 | 参数不合法 |

## 费用与耗时

默认配置下，每只股票的调用量如下：

- 4 位分析师：每位多轮工具调用
- 多空辩论：每轮 2 次
- 研究经理：1 次
- 交易员：1 次
- 风控辩论：每轮 3 次
- 组合经理：1 次
- 记忆复盘：每条到期的历史决策 1 次

合计约 20-40 次 LLM 调用。降低成本的方法：

- 减少 `TRADINGAGENTS_ANALYSTS`
- 保持辩论轮数为 1
- quick 档位选择便宜的模型
