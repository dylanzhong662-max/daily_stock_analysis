import { useState } from 'react';
import { Search, Loader2, PlusCircle, Check, TrendingUp } from 'lucide-react';

import { usScreenerApi, type UsScreenCandidate } from '../api/usScreener';
import { analysisApi } from '../api/analysis';

const PRESETS = [
  { id: 'largecap', label: '大盘蓝筹（市值>500亿）' },
  { id: 'growth_tech', label: '科技成长' },
  { id: 'value', label: '价值/防御（有分红·低波）' },
];

const SECTORS = [
  '', 'Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical',
  'Consumer Defensive', 'Industrials', 'Energy', 'Communication Services',
  'Basic Materials', 'Real Estate', 'Utilities',
];

function fmtCap(v: number | null): string {
  if (!v) return '-';
  if (v >= 1e12) return `${(v / 1e12).toFixed(2)}万亿`;
  if (v >= 1e8) return `${Math.round(v / 1e8)}亿`;
  return `${v}`;
}

export default function UsScreenerPage() {
  const [preset, setPreset] = useState('largecap');
  const [sector, setSector] = useState('');
  const [marketCapMinB, setMarketCapMinB] = useState('');
  const [top, setTop] = useState(10);
  const [momentum, setMomentum] = useState(true);
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [candidates, setCandidates] = useState<UsScreenCandidate[]>([]);
  const [msg, setMsg] = useState('');
  const [err, setErr] = useState('');
  const [added, setAdded] = useState<Record<string, boolean>>({});

  const runScreen = async () => {
    setLoading(true);
    setErr('');
    setMsg('');
    try {
      const resp = await usScreenerApi.screen({
        preset,
        sector: sector || null,
        market_cap_min: marketCapMinB ? Number(marketCapMinB) * 1e9 : null,
        top,
        momentum,
      });
      setCandidates(resp.candidates);
      setMsg(`筛选完成，命中 ${resp.count} 只候选${momentum ? '（已按上升趋势排序）' : ''}`);
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: { message?: string } } } })?.response?.data?.detail?.message;
      setErr(`筛选失败：${detail || (e as Error).message}`);
    } finally {
      setLoading(false);
    }
  };

  const analyzeAll = async () => {
    if (!candidates.length) return;
    setAnalyzing(true);
    setErr('');
    try {
      const codes = candidates.map((c) => c.symbol);
      await analysisApi.analyzeAsync({ stockCodes: codes });
      setMsg(`已提交 ${codes.length} 只候选分析（异步）。评分/买卖建议/止盈止损将推送飞书，并可在「首页 / 历史」查看。`);
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setErr(`提交分析失败：${detail || (e as Error).message}`);
    } finally {
      setAnalyzing(false);
    }
  };

  const addOne = async (sym: string) => {
    try {
      await usScreenerApi.addToWatchlist(sym);
      setAdded((prev) => ({ ...prev, [sym]: true }));
    } catch (e: unknown) {
      setErr(`加入自选失败（${sym}）：${(e as Error).message}`);
    }
  };

  return (
    <div className="p-6 max-w-6xl mx-auto text-slate-100">
      <div className="mb-6">
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <TrendingUp className="h-6 w-6 text-cyan-400" /> 美股选股（FMP）
        </h1>
        <p className="text-slate-400 mt-1 text-sm">
          按条件从全美股筛选候选 → 一键自动分析（评分/买卖建议/止盈止损，推送飞书）→ 是否加入自选由你决定。
        </p>
      </div>

      {/* 筛选表单 */}
      <div className="rounded-xl border border-slate-700 bg-slate-900/50 p-5 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-slate-400">预设策略</span>
            <select value={preset} onChange={(e) => setPreset(e.target.value)}
              className="bg-slate-800 border border-slate-600 rounded px-3 py-2">
              {PRESETS.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-slate-400">板块（可选）</span>
            <select value={sector} onChange={(e) => setSector(e.target.value)}
              className="bg-slate-800 border border-slate-600 rounded px-3 py-2">
              {SECTORS.map((s) => <option key={s} value={s}>{s || '全部'}</option>)}
            </select>
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-slate-400">最小市值（十亿美元）</span>
            <input type="number" value={marketCapMinB} onChange={(e) => setMarketCapMinB(e.target.value)}
              placeholder="如 50" className="bg-slate-800 border border-slate-600 rounded px-3 py-2" />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="text-slate-400">候选数量 Top</span>
            <input type="number" min={1} max={30} value={top}
              onChange={(e) => setTop(Math.max(1, Math.min(30, Number(e.target.value) || 10)))}
              className="bg-slate-800 border border-slate-600 rounded px-3 py-2" />
          </label>
        </div>
        <div className="flex items-center justify-between mt-4">
          <label className="flex items-center gap-2 text-sm text-slate-300">
            <input type="checkbox" checked={momentum} onChange={(e) => setMomentum(e.target.checked)} />
            动量过滤（只留站上50/200日均线的上升趋势股，稍慢）
          </label>
          <button onClick={runScreen} disabled={loading}
            className="flex items-center gap-2 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 rounded px-5 py-2 font-medium">
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
            筛选
          </button>
        </div>
      </div>

      {msg && <div className="mb-4 text-sm text-emerald-400 bg-emerald-950/40 border border-emerald-800 rounded px-4 py-2">{msg}</div>}
      {err && <div className="mb-4 text-sm text-red-400 bg-red-950/40 border border-red-800 rounded px-4 py-2">{err}</div>}

      {/* 候选表格 */}
      {candidates.length > 0 && (
        <div className="rounded-xl border border-slate-700 bg-slate-900/50 overflow-hidden">
          <div className="flex items-center justify-between px-5 py-3 border-b border-slate-700">
            <span className="font-medium">候选美股（{candidates.length}）</span>
            <button onClick={analyzeAll} disabled={analyzing}
              className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 rounded px-4 py-1.5 text-sm font-medium">
              {analyzing ? <Loader2 className="h-4 w-4 animate-spin" /> : <TrendingUp className="h-4 w-4" />}
              分析全部并推飞书
            </button>
          </div>
          <table className="w-full text-sm">
            <thead className="text-slate-400 bg-slate-800/50">
              <tr>
                <th className="text-left px-4 py-2">代码</th>
                <th className="text-left px-4 py-2">名称</th>
                <th className="text-left px-4 py-2">板块</th>
                <th className="text-right px-4 py-2">市值</th>
                <th className="text-right px-4 py-2">现价</th>
                <th className="text-right px-4 py-2">高出50日线</th>
                <th className="text-center px-4 py-2">操作</th>
              </tr>
            </thead>
            <tbody>
              {candidates.map((c) => (
                <tr key={c.symbol} className="border-t border-slate-800 hover:bg-slate-800/30">
                  <td className="px-4 py-2 font-mono text-cyan-400">{c.symbol}</td>
                  <td className="px-4 py-2">{c.name}</td>
                  <td className="px-4 py-2 text-slate-400">{c.sector}</td>
                  <td className="px-4 py-2 text-right">{fmtCap(c.marketCap)}</td>
                  <td className="px-4 py-2 text-right">{c.price ?? '-'}</td>
                  <td className="px-4 py-2 text-right text-emerald-400">
                    {c.aboveMa50Pct != null ? `${c.aboveMa50Pct}%` : '-'}
                  </td>
                  <td className="px-4 py-2 text-center">
                    <button onClick={() => addOne(c.symbol)} disabled={added[c.symbol]}
                      className="inline-flex items-center gap-1 text-xs rounded px-2 py-1 border border-slate-600 hover:bg-slate-700 disabled:opacity-60">
                      {added[c.symbol] ? <><Check className="h-3 w-3 text-emerald-400" /> 已加入</> : <><PlusCircle className="h-3 w-3" /> 加入自选</>}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
