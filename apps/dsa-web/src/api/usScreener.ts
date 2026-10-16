import apiClient from './index';

export interface UsScreenRequest {
  preset?: string;
  sector?: string | null;
  market_cap_min?: number | null;
  price_min?: number | null;
  volume_min?: number | null;
  exchange?: string | null;
  top?: number;
  momentum?: boolean;
}

export interface UsScreenCandidate {
  symbol: string;
  name: string;
  sector: string;
  industry: string;
  marketCap: number | null;
  price: number | null;
  beta: number | null;
  exchange: string;
  aboveMa50Pct: number | null;
}

export interface UsScreenResponse {
  preset: string;
  momentum: boolean;
  count: number;
  candidates: UsScreenCandidate[];
}

export const usScreenerApi = {
  /** 运行 FMP 美股选股 */
  screen: async (payload: UsScreenRequest): Promise<UsScreenResponse> => {
    const resp = await apiClient.post<UsScreenResponse>('/api/v1/screener/us', payload);
    return resp.data;
  },

  /** 把某只股票加入自选 */
  addToWatchlist: async (stockCode: string): Promise<void> => {
    await apiClient.post('/api/v1/stocks/watchlist/add', { stock_code: stockCode });
  },
};
