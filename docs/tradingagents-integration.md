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
   - quick 用于分析师、辩论、交易员和记忆复盘。
   - DeepSeek 的 `deepseek-chat` / `deepseek-reasoner` 已是废弃别名，都指向 V4.1 Flash；要让两档真正分开，用 `TRADINGAGENTS_DEEP_THINK_LLM=deepseek-v4-pro`、`TRADINGAGENTS_QUICK_THINK_LLM=deepseek-flash`。
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

当前 FMP 套餐**只覆盖美股**：港股、A 股代码会得到“数据不可用”提示，不会编造数据。

ETF/基金（如 SMH、GLD）没有公司财报：FMP 模式下基本面改为基金概况（规模、费率、NAV、持仓数、行业/国家暴露），三大报表返回“不适用”。

情绪分析师除新闻外还读取 StockTwits 和 Reddit，新闻分析师会查询 Polymarket 预测市场；这几个源国内机房直连不可达。设置 `TRADINGAGENTS_SOCIAL_PROXY`（如 `http://host.docker.internal:8118`，compose 已配置 `host.docker.internal` 指向宿主机）后，只有它们经代理访问（`src/services/tradingagents_proxy.py`），Reddit 遇到 429 限流会额外退避重试。StockTwits 对机房/代理 IP 仍可能返回 403，此时情绪分析只基于新闻 + Reddit。可设 `TRADINGAGENTS_DISABLE_STOCKTWITS=true` 直接跳过 StockTwits。

Reddit 覆盖面（`tune_reddit`）：原版只按股票代码在 3 个版块搜、每版块 5 帖；现在按“代码 OR 公司名”（公司名来自 FMP，如 `NVDA OR "NVIDIA"`）在 wallstreetbets / stocks / investing / StockMarket / semiconductors / ValueInvesting（NVDA、AMD 另加个股版块）合并搜索、每版块 8 帖；仍是一次请求，遇 429 退避重试。RSS 不含点赞/评论数，帖子热度无法判断。未配置代理时建议把 `TRADINGAGENTS_ANALYSTS` 设为 `market,news,fundamentals`。

**Yahoo 模式**下，TradingAgents 使用自己的数据源，**不走** DSA 的 `data_provider/` fallback 链：

- 行情和新闻：Yahoo Finance
- 美股财报：SEC EDGAR
- 宏观数据：FRED，需要 `FRED_API_KEY`，可选
- 预测市场：Polymarket

DSA 代码会自动转换成 Yahoo 格式：`600519` → `600519.SS`，`hk00700` → `0700.HK`，`AAPL` 保持不变。A 股和港股的基本面数据只有 Yahoo 来源，覆盖度不如美股。

建议设置 `SEC_EDGAR_USER_AGENT="Your Name you@example.com"`，SEC 要求调用方提供联系方式。

**数据质量补丁**（`src/services/tradingagents_quality.py`，两种数据源都生效，不修改 TradingAgents 包）：TradingAgents 把工具输出原样交给模型，下面几类输出容易被误读，DSA 在交给模型前做了修正或补充说明：

| 问题 | 处理 |
|---|---|
| 价格历史不足（如刚上市的 ADR）时，stockstats 均线按已有行数计算，“200 日均线”其实是上市以来均价 | 窗口超过可用行数的指标改为 `N/A: insufficient price history`；行情快照写明可用历史行数；FMP 基本面里的 50/200 日均线、52 周区间同样标注 |
| SEC EDGAR 季报没有独立 Q4 列，模型把相邻列当连续季度、拿错同比基数 | 季度报表后附“期间指引”（每列覆盖的月份、未单独申报的季度）和按同期配对算好的同比 |
| FRED 只给窗口首尾变化，模型把 11 个月的变化当同比 | 月度/季度序列补上真实同比（利率类用百分点）与最近一期变化 |
| Polymarket 个股 “hit” 盘口是触及型，被读成收盘价预测 | 参考块注明触及型含义、各档位离最新收盘的距离，档位概率不单调时提示报价陈旧；宏观主题过滤 `U.K.` 写法的英国事件 |
| 反思记忆默认注入其他股票的复盘教训，被当作本股证据 | 只注入本股自己的历史决策与复盘 |

补丁安装失败（如 TradingAgents 升级后内部结构变化）只记一条 warning，研究按原始工具输出照常进行。

**注入内容精简**（减少每个角色重复读到的 token）：

| 内容 | 处理 |
|---|---|
| Polymarket 参考块（约 1.8k token） | 全文只给新闻分析师与组合经理；其余 10 个角色只看到一行指引，改读新闻报告里的解读。报告目录的 `0_reference/polymarket.md` 仍保存全文 |
| 个股/全球新闻（FMP） | 每篇正文截断到 400 字；律所集体诉讼招揽稿（同一案件被十几家律所反复推送）只保留 1 条并注明省略条数 |
| 内部人交易（FMP） | 明细只保留公开市场买卖（Form 4 代码 P/S）；代扣税、行权、赠与、授予只汇总为类型计数与涉及人员 |
| 三大报表（SEC / FMP） | 只保留最近 12 个期间列；利润表/现金流的期间指引与同比在裁剪前用完整列计算 |

### 观测（Langfuse）

配置 `LANGFUSE_ENABLED=true`、`LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`、`LANGFUSE_BASE_URL` 后，每次研究在 Langfuse 生成一条 trace（`src/services/tradingagents_langfuse.py`）：

- trace 名 `tradingagents-research`，根观测 `run-tradingagents-research`（agent 类型），输入为代码/日期/分析师，输出为评级与报告 ID；标签 `tradingagents`、`data:<数据源>`，metadata 含模型与 Polymarket 主题。
- TradingAgents 的 LangGraph 图通过 Langfuse 官方 LangChain `CallbackHandler` 接入（依赖 `langchain` 包，已写入 `requirements-tradingagents.txt`）：每个角色节点、每次模型调用（模型名与 token 用量）和工具调用都挂在根观测下。
- **默认只上报计数**：研究期间超过 200 字的文本（prompt、新闻、报告、工具输出）在上报前替换为 `[omitted: N chars]`，token 用量不受影响；排查问题时临时设 `TRADINGAGENTS_LANGFUSE_CAPTURE_CONTENT=true` 上报原文。脱敏只作用于研究过程，Agent 对话自己的精简观测不受影响。
- 可设 `LANGFUSE_TRACING_ENVIRONMENT=production` 区分环境。
- 观测是 fail-open 的：Langfuse 不可达或初始化失败只记日志，研究照常；一次性容器退出前会 flush。研究结束日志会打印 trace 链接。
- 分析师子图里的节点按所属角色命名（如 `Market Analyst: agent`、`News Analyst: tools`）。
- trace id 写在报告目录的 `0_reference/langfuse.json`，评估任务据此把分数挂到对应 trace。

### 评估

`src/services/tradingagents_eval.py` 离线读取服务器上的报告文件，把三类分数写回 Langfuse（报告原文不上传）。随每日结果评估任务自动运行（`TRADINGAGENTS_ENABLED` 且 `TRADINGAGENTS_EVAL_ENABLED=true`，默认开），也可手动：

```bash
sudo ./scripts/tradingagents_eval.sh run            # 幂等，可重复跑；--no-judge 跳过 LLM 评审
sudo ./scripts/tradingagents_eval.sh summary        # 命中率 / 误读 / 评审一致率
sudo ./scripts/tradingagents_eval.sh review-sheet   # 导出 data/tradingagents/eval/review_sheet.csv
sudo ./scripts/tradingagents_eval.sh import-review data/tradingagents/eval/review_sheet.csv
# 非人工的参考标签（例如 Claude 按同一 rubric 代评）用 --source claude，分数名后缀为 _claude
sudo ./scripts/tradingagents_eval.sh import-review data/tradingagents/eval/claude_review.csv --source claude
```

| 分数 | 类型 | 含义 |
|---|---|---|
| `rating_alpha_5d` / `rating_alpha_20d` | NUMERIC（%） | 评级后 5 / 20 个交易日的超额收益：个股涨跌 − 基准（美股 SPY）涨跌，复用 TradingAgents 记忆复盘的算法；窗口未走完不写 |
| `rating_hit_5d` / `rating_hit_20d` | BOOLEAN | Buy/Overweight 要求 alpha>0，Sell/Underweight 要求 alpha<0，Hold 要求 \|alpha\|≤2% |
| `data_misread_count` | NUMERIC | 下面四条规则命中的句子总数；每条规则另有同名分数，说明里附命中句子 |
| `misread_short_history_ma` | NUMERIC | 分析日价格历史不足 200 行，却引用带数值的 200 日均线 |
| `misread_yoy` | NUMERIC | 营收同比与申报季度的真实同比（±1pp）都对不上，或 CPI 同比与 FRED 真实同比差超过 0.15pp（分部、环比、预期、TTM/半年口径不比） |
| `misread_cross_ticker` | NUMERIC | 引用记忆日志里其他股票的复盘结果 |
| `misread_touch_market` | NUMERIC | 把预测市场的触及型盘口写成“收盘价将低于/高于” |
| `report_quality` 及 `report_evidence_grounding` / `report_consistency` / `report_actionability` / `report_balance` | CATEGORICAL | LLM 评审（`TRADINGAGENTS_EVAL_JUDGE_MODEL`，未设时用 `LITELLM_MODEL`；`TRADINGAGENTS_EVAL_JUDGE=true` 时运行）：good / acceptable / poor，说明里是理由 |
| `report_quality_human` 及 `report_<维度>_human` | CATEGORICAL | 人工评审表导入的分数 |
| `report_quality_claude` 及 `report_<维度>_claude` | CATEGORICAL | `import-review --source claude` 导入的 Claude 参考标签（不是人工） |

- 规则是确定性的，命中句子写在分数说明里，可逐条核对；规则或评审 prompt 升版本（`RULES_VERSION` / `JUDGE_VERSION`）时会另写一份新版本分数，旧分数保留。
- 人工评审：`review-sheet` 导出的 CSV 不含 LLM 结论；按报告链接（DSA 网页）阅读后在 4 个维度和总体列填 good/acceptable/poor，`import-review` 导入后输出 LLM 与人工的逐维一致率和不一致明细（参考线：总体约 85% 且每类都不差，才算评审可用）。
- LLM 评审（`JUDGE_VERSION=report-quality-v2`）：
  - 评审前先给模型一段“已核实事实”：分析日可用的日线根数与最后一根日期（不足 200 根、或数据已停更会注明）、申报的季度营收同比、FRED CPI 同比、各角色自己给出的评级与最终评级相对分析师平均的偏移档数、规则命中的句子。模型自己核对不了这些，v1 因此几乎从不给 poor。
  - prompt 按 rubric 给出每个维度 good / acceptable / poor 的锚点，并要求先填“当前价、最终标签、最终目标仓位、止损在哪一侧、标签与仓位是否相符”，再列具体缺陷，最后打标签。
  - 总体分不让模型自报，按 rubric 推导：证据或一致性为 poor 即 poor；没有 poor 且至少两项 good 才 good；否则 acceptable。
  - 最终评级偏离分析师平均 ≥1 档时，“多空平衡”最多 acceptable（说明里注明 capped），总体分随之重算。
  - 校准：20 份报告的 Claude 参考标签对比，总体一致率 v1 6/20 → v2 14/20（全量重跑；证据 14/20、一致性 10/20、多空平衡 16/20、可操作 20/20）；仍未到 85% 参考线，剩余分歧主要在“一致性”。没有人工标签前，评审分只用来看趋势，不用来单份定论。
- 没有 trace 的历史报告会用确定性 trace id 回填一条轻量 trace（标签 `backfill`，只有元数据、没有模型调用），时间戳是回填当天；按分析日统计请看 metadata 的 `trade_date`。
- 评估状态（已写的分数、规则结果、评审结果）在 `data/tradingagents/eval/state.json`；每个分数的终值只写一次。

### 运行数据目录

默认写入 `data/tradingagents/`，不写 `~/.tradingagents`：

| 内容 | 默认路径 | 覆盖变量 |
|---|---|---|
| 报告（`reports/<TICKER>_<时间>/complete_report.md`、`.html`）和状态日志 | `data/tradingagents/logs` | `TRADINGAGENTS_RESULTS_DIR` |
| 数据缓存 | `data/tradingagents/cache` | `TRADINGAGENTS_CACHE_DIR` |
| 决策记忆日志（跨次运行复盘历史决策收益） | `data/tradingagents/memory/trading_memory.md` | `TRADINGAGENTS_MEMORY_LOG_PATH` |

## 使用

### Web 页面

侧边栏「深度研究」（`/tradingagents`）：输入美股代码、可选日期和分析师，点「开始研究」后页面每 3 秒轮询进度，完成后展示评级、组合经理决策和各角色报告；右侧列出历史报告，可打开完整 HTML 报告。

- 研究在 `stock-server` 进程内运行，**同一时间只允许 1 个任务**（再次提交返回 409 `tradingagents_busy`，页面会自动接管正在运行的任务）。
- 需要 `TRADINGAGENTS_ENABLED=true`；Docker 部署时 `server` 服务内存上限已调到 2G。
- 任务状态保存在内存里，服务重启会丢失进行中的任务；已生成的报告不受影响。

### Polymarket 预测市场参考（自动）

每次研究前会**自动**抓取一组默认主题的 Polymarket 预测市场（`src/services/tradingagents_polymarket.py`），以 AI 算力产业链美股为主：

| 主题 | 典型市场 |
|---|---|
| `Fed decision`、`Fed rate hike` | 下次议息会议各结果概率、2026 年加息次数分布 |
| `US recession`、`inflation` | 美国 2026 年衰退、通胀超 4.5%/5% |
| `Anthropic IPO`、`OpenAI IPO` | 年内 IPO 概率、IPO 当日市值区间 |
| `AI bubble` | AI 泡沫 2026 年破裂 |
| `China Taiwan`、`chip export` | 台海冲突、AI 芯片出口管制立法 |
| 本股代码（如 `NVDA`） | 财报类（数据中心营收、毛利率）优先，其次价格区间 |

按 Polymarket **事件**分组：每个主题取成交量最高的 2 个事件（本股 3 个，财报类优先），互斥分档事件（如“2026 年加息次数”“Anthropic IPO 当日市值区间”）给出完整概率分布，“截至某日”类事件按到期日排序；事件成交量低于 $20k（本股 $500）的过滤，低于 $5k 的标注为薄市场；跨主题按事件去重；单个主题失败只标注“不可用”。参考块注入到**所有角色**共享的上下文，并保存为报告的 `0_reference/polymarket.md`。新闻分析师本身也可按需调用 Polymarket 工具。

- `TRADINGAGENTS_POLYMARKET_TOPICS`：覆盖默认主题（逗号分隔），`none` 关闭。
- `TRADINGAGENTS_POLYMARKET_TICKER=false`：不搜索本股市场。
- 单次研究可在页面取消「自动加入」或填写额外主题（API `use_default_polymarket` / `polymarket_topics`，CLI `--tradingagents-polymarket`）。
- 需要 Polymarket 可达（国内服务器配置 `TRADINGAGENTS_SOCIAL_PROXY`）；历史日期不提供（Polymarket 只有实时赔率）。

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
GET  /api/v1/tradingagents/running            # {task_id}，当前运行中的任务（无则为 null）
GET  /api/v1/tradingagents/reports            # 历史报告列表（report_id/ticker/created_at/rating）
GET  /api/v1/tradingagents/reports/{id}       # 单份报告各阶段 Markdown
GET  /api/v1/tradingagents/reports/{id}/html  # 完整 HTML 报告
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
