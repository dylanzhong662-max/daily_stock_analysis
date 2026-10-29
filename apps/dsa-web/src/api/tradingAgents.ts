import apiClient from './index';
import { API_BASE_URL } from '../utils/constants';
import { toCamelCase } from './utils';

export type TradingAgentsAnalyst = 'market' | 'social' | 'news' | 'fundamentals';

export const TRADINGAGENTS_ANALYSTS: TradingAgentsAnalyst[] = ['market', 'social', 'news', 'fundamentals'];

export type TradingAgentsStatus = {
  enabled: boolean;
  available: boolean;
  version?: string | null;
  analysts: TradingAgentsAnalyst[];
  polymarketDefaultTopics?: string[];
  polymarketTickerTopic?: boolean;
  deepModelOptions?: { id: string; label: string }[];
  deepModelDefault?: string | null;
};

export type TradingAgentsTaskAccepted = {
  taskId: string;
  traceId: string;
  status: string;
  message: string;
  stockCode: string;
};

export type TradingAgentsResult = {
  code: string;
  ticker: string;
  tradeDate: string;
  rating: string;
  operationAdvice: string;
  reportId?: string | null;
  deepThinkLlm?: string | null;
  quickThinkLlm?: string | null;
  dataSource?: string | null;
  finalTradeDecision?: string;
};

export type TradingAgentsTaskStatus = {
  taskId: string;
  traceId?: string | null;
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled' | string;
  progress: number;
  message?: string | null;
  error?: string | null;
  result?: TradingAgentsResult | null;
};

export type TradingAgentsReportSummary = {
  reportId: string;
  ticker: string;
  createdAt?: string | null;
  rating?: string | null;
  operationAdvice?: string | null;
  hasHtml: boolean;
};

export type TradingAgentsReportSection = {
  key: string;
  title: string;
  content: string;
};

export type TradingAgentsReportDetail = TradingAgentsReportSummary & {
  sections: TradingAgentsReportSection[];
};

export type StartTradingAgentsTaskPayload = {
  stockCode: string;
  tradeDate?: string;
  analysts?: TradingAgentsAnalyst[];
  polymarketTopics?: string[];
  useDefaultPolymarket?: boolean;
  deepModel?: string;
};

const BASE = '/api/v1/tradingagents';

export const tradingAgentsApi = {
  getStatus: async (): Promise<TradingAgentsStatus> => {
    const response = await apiClient.get(`${BASE}/status`);
    return toCamelCase<TradingAgentsStatus>(response.data);
  },

  startTask: async (payload: StartTradingAgentsTaskPayload): Promise<TradingAgentsTaskAccepted> => {
    const response = await apiClient.post(`${BASE}/tasks`, {
      stock_code: payload.stockCode,
      trade_date: payload.tradeDate || undefined,
      analysts: payload.analysts,
      polymarket_topics: payload.polymarketTopics?.length ? payload.polymarketTopics : undefined,
      use_default_polymarket: payload.useDefaultPolymarket ?? true,
      deep_model: payload.deepModel || undefined,
    });
    return toCamelCase<TradingAgentsTaskAccepted>(response.data);
  },

  getTask: async (taskId: string): Promise<TradingAgentsTaskStatus> => {
    const response = await apiClient.get(`${BASE}/tasks/${encodeURIComponent(taskId)}`);
    return toCamelCase<TradingAgentsTaskStatus>(response.data);
  },

  getRunningTaskId: async (): Promise<string | null> => {
    const response = await apiClient.get(`${BASE}/running`);
    return toCamelCase<{ taskId: string | null }>(response.data).taskId ?? null;
  },

  listReports: async (limit = 50): Promise<TradingAgentsReportSummary[]> => {
    const response = await apiClient.get(`${BASE}/reports`, { params: { limit } });
    return toCamelCase<{ reports: TradingAgentsReportSummary[] }>(response.data).reports ?? [];
  },

  getReport: async (reportId: string): Promise<TradingAgentsReportDetail> => {
    const response = await apiClient.get(`${BASE}/reports/${encodeURIComponent(reportId)}`);
    return toCamelCase<TradingAgentsReportDetail>(response.data);
  },

  reportHtmlUrl: (reportId: string): string => `${API_BASE_URL}${BASE}/reports/${encodeURIComponent(reportId)}/html`,
};

/** 409 tradingagents_busy 响应里携带的正在运行的任务 id。 */
export function getBusyTaskId(error: unknown): string | null {
  const detail = (error as { response?: { status?: number; data?: { detail?: { error?: string; detail?: { task_id?: string } } } } })
    ?.response;
  if (detail?.status !== 409 || detail.data?.detail?.error !== 'tradingagents_busy') {
    return null;
  }
  return detail.data.detail.detail?.task_id ?? null;
}
