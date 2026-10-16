# -*- coding: utf-8 -*-
"""
FmpFetcher — US market data source (Financial Modeling Prep, stable API)

Data source: https://financialmodelingprep.com/stable/*
Markets: US only
说明：国内机房可直连 FMP，作为 yfinance(Yahoo 对机房 IP 429 限流) 的美股替代源。
    - 日线：/stable/historical-price-eod/full （完整 OHLC + volume）
    - 实时：/stable/quote （最新价、涨跌、开高低、昨收、52周高低、市值等）
"""

import json
import logging
import os
import subprocess
from typing import Optional
from urllib.parse import urlencode

import pandas as pd
from .base import BaseFetcher, DataFetchError, STANDARD_COLUMNS
from .realtime_types import UnifiedRealtimeQuote, RealtimeSource
from .us_index_mapping import is_us_stock_code

logger = logging.getLogger(__name__)

_FMP_BASE_URL = "https://financialmodelingprep.com/stable"

# 美股主要指数（FMP 符号, 返回 code, 中文名）——用于大盘复盘（Yahoo 机房 IP 429 时的替代源）
_US_INDICES = [
    ("^GSPC", "SPX", "标普500指数"),
    ("^IXIC", "IXIC", "纳斯达克综合指数"),
    ("^DJI", "DJI", "道琼斯工业指数"),
    ("^VIX", "VIX", "VIX恐慌指数"),
]


class FmpFetcher(BaseFetcher):
    name = "FmpFetcher"
    priority = int(os.getenv("FMP_PRIORITY", "2"))

    def __init__(self):
        from src.config import get_config
        config = get_config()
        self._api_key = getattr(config, 'fmp_api_key', None) or os.getenv('FMP_API_KEY')
        if not self._api_key:
            logger.debug("[FMP] API key not configured, fetcher disabled")

    def is_available(self) -> bool:
        return bool(self._api_key)

    def _is_us_stock(self, stock_code: str) -> bool:
        return is_us_stock_code(stock_code)

    def _get_json(self, path: str, params: dict[str, object], *, timeout: int = 15):
        """Call FMP directly with system curl.

        The server's HTTPS proxy is required for some providers but breaks the
        TLS handshake to FMP.  ``--noproxy`` keeps this provider independent of
        that proxy.  The key is passed through curl stdin, never an argv value.
        """
        if not self._api_key:
            raise DataFetchError("[FMP] API key not configured")
        query = urlencode({**params, "apikey": self._api_key})
        try:
            completed = subprocess.run(
                [
                    "curl", "--noproxy", "financialmodelingprep.com", "--fail",
                    "--silent", "--show-error", "--connect-timeout", "10",
                    "--max-time", str(timeout), "--config", "-",
                ],
                input=(
                    f'url = "{_FMP_BASE_URL}/{path}?{query}"\n'
                    'user-agent = "daily-stock-analysis/1.0"\n'
                ),
                text=True,
                capture_output=True,
                timeout=timeout + 5,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr.strip() or f"curl exited {completed.returncode}")
            return json.loads(completed.stdout)
        except Exception as exc:
            raise DataFetchError(f"[FMP] HTTP request failed: {exc}") from exc

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        if not self._api_key:
            raise DataFetchError("[FMP] API key not configured")
        if not self._is_us_stock(stock_code):
            raise DataFetchError(f"[FMP] {stock_code} is not a US stock")

        symbol = stock_code.strip().upper()
        params: dict[str, object] = {
            'symbol': symbol,
            'from': start_date,
            'to': end_date,
        }

        try:
            self.random_sleep(0.2, 0.6)
            data = self._get_json("historical-price-eod/full", params)
        except Exception as e:
            raise DataFetchError(f"[FMP] HTTP request failed for {symbol}: {e}") from e

        if isinstance(data, dict) and data.get("Error Message"):
            raise DataFetchError(f"[FMP] API error for {symbol}: {data['Error Message']}")
        if not isinstance(data, list) or not data:
            raise DataFetchError(f"[FMP] No data returned for {symbol}")

        # FMP 返回按日期倒序；保留原始列，交给 _normalize_data 统一处理
        return pd.DataFrame(data)

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        # FMP 字段已是 open/high/low/close/volume/date，仅需补齐派生列并按时间升序
        df['date'] = pd.to_datetime(df['date']).dt.date
        df = df.sort_values('date').reset_index(drop=True)

        for col in ('open', 'high', 'low', 'close', 'volume'):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['pct_chg'] = df['close'].pct_change() * 100
        df['pct_chg'] = df['pct_chg'].fillna(0).round(2)
        df['amount'] = df['volume'] * df['close']
        df['code'] = stock_code

        keep = ['code'] + STANDARD_COLUMNS
        df = df[[col for col in keep if col in df.columns]]
        return df

    def get_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        if not self._api_key or not self._is_us_stock(stock_code):
            return None

        symbol = stock_code.strip().upper()
        try:
            self.random_sleep(0.2, 0.6)
            data = self._get_json("quote", {'symbol': symbol})
        except Exception as e:
            logger.warning(f"[FMP] Realtime quote failed for {symbol}: {e}")
            return None

        if not isinstance(data, list) or not data:
            return None
        q = data[0]

        price = q.get('price')
        if not price:
            return None

        prev_close = q.get('previousClose')
        high = q.get('dayHigh')
        low = q.get('dayLow')
        change_pct = q.get('changePercentage')
        change_amount = q.get('change')

        amplitude = None
        if high and low and prev_close and prev_close > 0:
            amplitude = round((high - low) / prev_close * 100, 2)

        return UnifiedRealtimeQuote(
            code=symbol,
            name=q.get('name') or "",
            source=RealtimeSource.FALLBACK,
            price=price,
            change_pct=round(change_pct, 2) if change_pct is not None else None,
            change_amount=round(change_amount, 4) if change_amount is not None else None,
            volume=q.get('volume'),
            amount=None,
            volume_ratio=None,
            turnover_rate=None,
            amplitude=amplitude,
            open_price=q.get('open'),
            high=high,
            low=low,
            pre_close=prev_close,
            total_mv=q.get('marketCap'),
            high_52w=q.get('yearHigh'),
            low_52w=q.get('yearLow'),
        )

    def get_main_indices(self, region: str = "cn") -> Optional[list]:
        """美股主要指数行情（region=us）。其他市场返回 None，交给国内源/yfinance。"""
        if region != "us" or not self._api_key:
            return None
        results = []
        for fmp_sym, code, name in _US_INDICES:
            try:
                self.random_sleep(0.1, 0.3)
                data = self._get_json("quote", {'symbol': fmp_sym})
            except Exception as e:
                logger.warning(f"[FMP] 指数 {fmp_sym} 获取失败: {e}")
                continue
            if not isinstance(data, list) or not data:
                continue
            q = data[0]
            price = q.get('price')
            if price is None:
                continue
            prev_close = q.get('previousClose') or 0
            high = q.get('dayHigh')
            low = q.get('dayLow')
            amplitude = ((high - low) / prev_close * 100) if (high and low and prev_close) else 0
            results.append({
                'code': code,
                'name': name,
                'current': price,
                'change': q.get('change'),
                'change_pct': q.get('changePercentage'),
                'open': q.get('open'),
                'high': high,
                'low': low,
                'prev_close': prev_close,
                'volume': q.get('volume') or 0,
                'amount': 0.0,
                'amplitude': round(amplitude, 2) if amplitude else 0.0,
            })
        if results:
            logger.info(f"[FMP] 成功获取 {len(results)} 个美股指数行情")
        return results or None

    def get_stock_name(self, stock_code: str) -> Optional[str]:
        if not self._api_key or not self._is_us_stock(stock_code):
            return None
        symbol = stock_code.strip().upper()
        try:
            data = self._get_json("quote", {'symbol': symbol}, timeout=10)
        except Exception as e:
            logger.debug(f"[FMP] Symbol name lookup failed for {symbol}: {e}")
            return None
        if isinstance(data, list) and data and data[0].get('name'):
            return data[0]['name']
        return None
