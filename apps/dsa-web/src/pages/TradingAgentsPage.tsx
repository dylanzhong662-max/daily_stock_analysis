import type React from 'react';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { BrainCircuit, ExternalLink, History, Loader2, RefreshCw } from 'lucide-react';
import { createParsedApiError, getParsedApiError, type ParsedApiError } from '../api/error';
import {
  TRADINGAGENTS_ANALYSTS,
  getBusyTaskId,
  tradingAgentsApi,
  type TradingAgentsAnalyst,
  type TradingAgentsReportDetail,
  type TradingAgentsReportSummary,
  type TradingAgentsStatus,
} from '../api/tradingAgents';
import {
  ApiErrorAlert,
  AppPage,
  Badge,
  Button,
  Card,
  Checkbox,
  Collapsible,
  EmptyState,
  InlineAlert,
  Input,
  PageHeader,
} from '../components/common';
import { ReportMarkdownBody } from '../components/report/ReportMarkdownBody';

const POLL_INTERVAL_MS = 3000;

const ANALYST_LABELS: Record<TradingAgentsAnalyst, string> = {
  market: '技术面',
  social: '情绪面（国内服务器不可用）',
  news: '新闻面',
  fundamentals: '基本面',
};

type BadgeVariant = 'success' | 'danger' | 'default' | 'warning';

function ratingVariant(rating?: string | null): BadgeVariant {
  switch ((rating || '').toLowerCase()) {
    case 'buy':
    case 'overweight':
      return 'success';
    case 'underweight':
    case 'sell':
      return 'danger';
    case 'hold':
      return 'default';
    default:
      return 'warning';
  }
}

function formatTime(value?: string | null): string {
  if (!value) {
    return '';
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', { hour12: false });
}

function formatElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return m > 0 ? `${m} 分 ${s} 秒` : `${s} 秒`;
}

const TradingAgentsPage: React.FC = () => {
  const [status, setStatus] = useState<TradingAgentsStatus | null>(null);
  const [statusError, setStatusError] = useState<ParsedApiError | null>(null);
  const [stockCode, setStockCode] = useState('');
  const [tradeDate, setTradeDate] = useState('');
  const [analysts, setAnalysts] = useState<TradingAgentsAnalyst[]>(['market', 'news', 'fundamentals']);

  const [activeTaskId, setActiveTaskId] = useState<string | null>(null);
  const [taskMessage, setTaskMessage] = useState('');
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<ParsedApiError | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const [reports, setReports] = useState<TradingAgentsReportSummary[]>([]);
  const [reportsLoading, setReportsLoading] = useState(false);
  const [report, setReport] = useState<TradingAgentsReportDetail | null>(null);
  const [reportLoading, setReportLoading] = useState(false);

  const running = activeTaskId !== null;
  const ready = Boolean(status?.enabled && status?.available);

  const loadReports = useCallback(async () => {
    setReportsLoading(true);
    try {
      setReports(await tradingAgentsApi.listReports(50));
    } catch (err) {
      setError(getParsedApiError(err));
    } finally {
      setReportsLoading(false);
    }
  }, []);

  const openReport = useCallback(async (reportId: string) => {
    setReportLoading(true);
    setError(null);
    try {
      setReport(await tradingAgentsApi.getReport(reportId));
    } catch (err) {
      setError(getParsedApiError(err));
    } finally {
      setReportLoading(false);
    }
  }, []);

  const attachTask = useCallback((taskId: string, message: string) => {
    setActiveTaskId(taskId);
    setTaskMessage(message);
    setStartedAt(Date.now());
    setElapsed(0);
  }, []);

  // 初始化：状态、历史报告、恢复正在运行的任务
  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const nextStatus = await tradingAgentsApi.getStatus();
        if (!active) return;
        setStatus(nextStatus);
        if (nextStatus.analysts?.length) {
          setAnalysts(nextStatus.analysts.filter((a) => TRADINGAGENTS_ANALYSTS.includes(a)));
        }
        const runningId = await tradingAgentsApi.getRunningTaskId();
        if (active && runningId) {
          attachTask(runningId, '已有研究任务在运行，正在等待结果');
        }
      } catch (err) {
        if (active) setStatusError(getParsedApiError(err));
      }
    })();
    void loadReports();
    return () => {
      active = false;
    };
  }, [attachTask, loadReports]);

  // 计时
  useEffect(() => {
    if (!running || startedAt === null) return undefined;
    const timer = window.setInterval(() => setElapsed(Math.floor((Date.now() - startedAt) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, [running, startedAt]);

  // 轮询任务状态
  useEffect(() => {
    if (!activeTaskId) return undefined;
    const pollingTaskId = activeTaskId;
    let active = true;
    let timer: ReturnType<typeof window.setTimeout> | undefined;

    const finish = () => {
      setActiveTaskId(null);
      setStartedAt(null);
    };

    const poll = async () => {
      try {
        const task = await tradingAgentsApi.getTask(pollingTaskId);
        if (!active) return;
        if (task.message) setTaskMessage(task.message);
        if (task.status === 'completed') {
          finish();
          setNotice(`${task.result?.ticker ?? ''} 研究完成：${task.result?.rating ?? ''}（${task.result?.operationAdvice ?? ''}）`);
          await loadReports();
          if (task.result?.reportId) {
            await openReport(task.result.reportId);
          }
          return;
        }
        if (task.status === 'failed' || task.status === 'cancelled') {
          finish();
          setError(createParsedApiError({ title: '研究失败', message: task.error || task.message || '任务失败' }));
          return;
        }
      } catch (err) {
        if (!active) return;
        const parsed = getParsedApiError(err);
        if (parsed.status === 404) {
          // 服务重启等原因任务已丢失
          finish();
          setError(parsed);
          return;
        }
      }
      if (active) timer = window.setTimeout(poll, POLL_INTERVAL_MS);
    };

    void poll();
    return () => {
      active = false;
      if (timer) window.clearTimeout(timer);
    };
  }, [activeTaskId, loadReports, openReport]);

  const canSubmit = useMemo(
    () => ready && !running && !submitting && stockCode.trim().length > 0 && analysts.length > 0,
    [ready, running, submitting, stockCode, analysts],
  );

  const toggleAnalyst = (analyst: TradingAgentsAnalyst, checked: boolean) => {
    setAnalysts((prev) => {
      const next = checked ? [...prev, analyst] : prev.filter((a) => a !== analyst);
      return TRADINGAGENTS_ANALYSTS.filter((a) => next.includes(a));
    });
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!canSubmit) return;
    setSubmitting(true);
    setError(null);
    setNotice(null);
    try {
      const accepted = await tradingAgentsApi.startTask({
        stockCode: stockCode.trim().toUpperCase(),
        tradeDate: tradeDate || undefined,
        analysts,
      });
      attachTask(accepted.taskId, accepted.message);
    } catch (err) {
      const busyId = getBusyTaskId(err);
      if (busyId) {
        attachTask(busyId, '已有研究任务在运行，正在等待它完成');
      } else {
        setError(getParsedApiError(err));
      }
    } finally {
      setSubmitting(false);
    }
  };

  const decision = report?.sections.find((s) => s.key === 'decision');
  const otherSections = report?.sections.filter((s) => s.key !== 'decision') ?? [];

  return (
    <AppPage>
      <PageHeader
        eyebrow="TradingAgents"
        title="多智能体深度研究"
        description="分析师团队 → 多空辩论 → 交易员 → 风控评审 → 组合经理，给出 5 档评级。数据来自 FMP（仅美股），模型为 DeepSeek。"
      />

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="flex min-w-0 flex-col gap-6">
          <Card title="发起研究" padding="md">
            {statusError ? <ApiErrorAlert error={statusError} /> : null}
            {status && !ready ? (
              <InlineAlert
                variant="warning"
                title="TradingAgents 不可用"
                message={!status.enabled ? '服务端未启用（TRADINGAGENTS_ENABLED=false）' : '服务端未安装 TradingAgents 依赖'}
              />
            ) : null}
            <form className="flex flex-col gap-4" onSubmit={handleSubmit}>
              <div className="grid gap-4 sm:grid-cols-2">
                <Input
                  label="股票代码"
                  placeholder="如 AAPL、NVDA"
                  hint="当前 FMP 套餐仅支持美股"
                  value={stockCode}
                  onChange={(e) => setStockCode(e.target.value)}
                  disabled={running}
                />
                <Input
                  label="分析日期（可选）"
                  type="date"
                  hint="留空为今天，不能晚于今天"
                  value={tradeDate}
                  onChange={(e) => setTradeDate(e.target.value)}
                  disabled={running}
                />
              </div>
              <div className="flex flex-wrap gap-x-6 gap-y-2">
                {TRADINGAGENTS_ANALYSTS.map((analyst) => (
                  <Checkbox
                    key={analyst}
                    label={ANALYST_LABELS[analyst]}
                    checked={analysts.includes(analyst)}
                    onChange={(e) => toggleAnalyst(analyst, e.target.checked)}
                    disabled={running}
                  />
                ))}
              </div>
              <div className="flex items-center gap-3">
                <Button type="submit" variant="primary" disabled={!canSubmit} isLoading={submitting}>
                  <BrainCircuit className="h-4 w-4" />
                  开始研究
                </Button>
                <span className="text-xs text-secondary-text">每次约 3–5 分钟，同一时间只能运行一个研究任务</span>
              </div>
            </form>
          </Card>

          {running ? (
            <Card padding="md">
              <div className="flex items-center gap-3">
                <Loader2 className="h-5 w-5 animate-spin text-cyan" />
                <div className="min-w-0">
                  <p className="text-sm font-medium text-foreground">{taskMessage || '研究进行中'}</p>
                  <p className="text-xs text-secondary-text">已用时 {formatElapsed(elapsed)}，预计 3–5 分钟</p>
                </div>
              </div>
            </Card>
          ) : null}

          {notice ? <InlineAlert variant="success" message={notice} /> : null}
          {error ? <ApiErrorAlert error={error} onDismiss={() => setError(null)} /> : null}

          {reportLoading ? (
            <Card padding="md">
              <div className="flex items-center gap-2 text-sm text-secondary-text">
                <Loader2 className="h-4 w-4 animate-spin" /> 正在加载报告…
              </div>
            </Card>
          ) : report ? (
            <Card padding="md">
              <div className="mb-4 flex flex-wrap items-center gap-3">
                <h2 className="text-lg font-semibold text-foreground">{report.ticker}</h2>
                <Badge variant={ratingVariant(report.rating)} size="md">
                  {report.rating || '未给出评级'}
                </Badge>
                {report.operationAdvice ? <span className="text-sm text-foreground">{report.operationAdvice}</span> : null}
                <span className="text-xs text-secondary-text">{formatTime(report.createdAt)}</span>
                {report.hasHtml ? (
                  <a
                    className="ml-auto inline-flex items-center gap-1 text-sm text-cyan hover:underline"
                    href={tradingAgentsApi.reportHtmlUrl(report.reportId)}
                    target="_blank"
                    rel="noreferrer"
                  >
                    打开完整报告 <ExternalLink className="h-3.5 w-3.5" />
                  </a>
                ) : null}
              </div>
              {decision ? (
                <ReportMarkdownBody content={decision.content} testId="tradingagents-decision" />
              ) : (
                <p className="text-sm text-secondary-text">报告中没有组合经理决策。</p>
              )}
              {otherSections.length ? (
                <div className="mt-6 flex flex-col gap-3">
                  {otherSections.map((section) => (
                    <Collapsible key={section.key} title={section.title}>
                      <ReportMarkdownBody content={section.content} />
                    </Collapsible>
                  ))}
                </div>
              ) : null}
            </Card>
          ) : (
            <EmptyState
              icon={<BrainCircuit className="h-6 w-6" />}
              title="还没有选中报告"
              description="发起一次研究，或在右侧历史中选择一份报告查看。"
            />
          )}
        </div>

        <Card padding="sm" className="h-fit">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="flex items-center gap-2 text-sm font-semibold text-foreground">
              <History className="h-4 w-4" /> 历史报告
            </h2>
            <Button variant="ghost" size="xsm" onClick={() => void loadReports()} disabled={reportsLoading} aria-label="刷新历史">
              <RefreshCw className={reportsLoading ? 'h-3.5 w-3.5 animate-spin' : 'h-3.5 w-3.5'} />
            </Button>
          </div>
          {reports.length === 0 ? (
            <p className="py-6 text-center text-xs text-secondary-text">{reportsLoading ? '加载中…' : '暂无报告'}</p>
          ) : (
            <ul className="flex max-h-[70vh] flex-col gap-1 overflow-y-auto">
              {reports.map((item) => (
                <li key={item.reportId}>
                  <button
                    type="button"
                    onClick={() => void openReport(item.reportId)}
                    className={`flex w-full items-center justify-between gap-2 rounded-lg px-3 py-2 text-left transition-colors hover:bg-hover ${
                      report?.reportId === item.reportId ? 'bg-hover' : ''
                    }`}
                  >
                    <span className="min-w-0">
                      <span className="block text-sm font-medium text-foreground">{item.ticker}</span>
                      <span className="block text-xs text-secondary-text">{formatTime(item.createdAt)}</span>
                    </span>
                    <Badge variant={ratingVariant(item.rating)}>{item.rating || '—'}</Badge>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>
    </AppPage>
  );
};

export default TradingAgentsPage;
