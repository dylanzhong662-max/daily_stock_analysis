import { beforeEach, describe, expect, it, vi } from 'vitest';
import { getBusyTaskId, tradingAgentsApi } from '../tradingAgents';

const { get, post } = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
}));

vi.mock('../index', () => ({
  default: {
    get,
    post,
  },
}));

describe('tradingAgentsApi', () => {
  beforeEach(() => {
    get.mockReset();
    post.mockReset();
  });

  it('submits a research task with snake_case payload', async () => {
    post.mockResolvedValueOnce({
      data: { task_id: 't1', trace_id: 't1', status: 'pending', message: 'ok', stock_code: 'AAPL' },
    });

    const accepted = await tradingAgentsApi.startTask({ stockCode: 'AAPL', analysts: ['market', 'news'] });

    expect(post).toHaveBeenCalledWith('/api/v1/tradingagents/tasks', {
      stock_code: 'AAPL',
      trade_date: undefined,
      analysts: ['market', 'news'],
      polymarket_topics: undefined,
    });

    post.mockResolvedValueOnce({ data: { task_id: 't2' } });
    await tradingAgentsApi.startTask({ stockCode: 'NVDA', polymarketTopics: ['Anthropic IPO'] });
    expect(post).toHaveBeenLastCalledWith('/api/v1/tradingagents/tasks', expect.objectContaining({
      polymarket_topics: ['Anthropic IPO'],
    }));
    expect(accepted.taskId).toBe('t1');
  });

  it('maps task result and report list to camelCase', async () => {
    get.mockResolvedValueOnce({
      data: {
        task_id: 't1',
        status: 'completed',
        progress: 100,
        result: { ticker: 'AAPL', rating: 'Underweight', operation_advice: '减仓', report_id: 'AAPL_20261016_151221' },
      },
    });
    const task = await tradingAgentsApi.getTask('t1');
    expect(get).toHaveBeenCalledWith('/api/v1/tradingagents/tasks/t1');
    expect(task.result?.reportId).toBe('AAPL_20261016_151221');
    expect(task.result?.operationAdvice).toBe('减仓');

    get.mockResolvedValueOnce({
      data: { reports: [{ report_id: 'AAPL_20261016_151221', ticker: 'AAPL', has_html: true, created_at: '2026-10-16T15:12:21' }] },
    });
    const reports = await tradingAgentsApi.listReports(20);
    expect(get).toHaveBeenLastCalledWith('/api/v1/tradingagents/reports', { params: { limit: 20 } });
    expect(reports[0].hasHtml).toBe(true);
    expect(tradingAgentsApi.reportHtmlUrl('AAPL_20261016_151221')).toBe(
      '/api/v1/tradingagents/reports/AAPL_20261016_151221/html',
    );
  });

  it('reads the running task id and the busy task id from a 409', async () => {
    get.mockResolvedValueOnce({ data: { task_id: null } });
    expect(await tradingAgentsApi.getRunningTaskId()).toBeNull();

    const busy = {
      response: { status: 409, data: { detail: { error: 'tradingagents_busy', detail: { task_id: 'running-1' } } } },
    };
    expect(getBusyTaskId(busy)).toBe('running-1');
    expect(getBusyTaskId({ response: { status: 500, data: {} } })).toBeNull();
  });
});
