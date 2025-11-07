# pages/02_Metrics_&_Performance.py
# -------------------------------------------------------------------
# Metrics & Performance with Agentic Competitive Analysis
# - Pulls tickers from main page (st.session_state.step1) or manual
# - Fetches advanced metrics & optional price performance
# - Adds peer-app metrics: 5y PE/PS averages, 3y CAGRs (rev/NI/OC/FCF),
#   forward PE/PS from analyst estimates, EPS/Revenue growth path,
#   5y price/risk block from yfinance
# - Persists the built table in session_state so the Agentic panel
#   works after button clicks/reruns
# - Agentic panel now accepts free-form questions and uses context
#   from all cached tabs (no pre-prompt/bullets parsing)
# -------------------------------------------------------------------

import os
import time
import math
import json
import numpy as np
import pandas as pd
import requests
import streamlit as st
from datetime import datetime, timezone, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Optional GPT (for Agentic panel)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

import yfinance as yf

st.set_page_config(page_title="Metrics & Performance", layout="wide")

# ============================== Config =========================================
FMP_API_KEY    = os.getenv("FMP_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAI) else None

# --- Batch / rate-limit knobs (env-overridable) ---
FMP_BATCH_SIZE  = int(os.getenv("FMP_BATCH_SIZE", "50"))
FMP_BATCH_SLEEP = float(os.getenv("FMP_BATCH_SLEEP", "1.0"))

def _batch_pause(i: int, batch_size: int = FMP_BATCH_SIZE, sleep_s: float = FMP_BATCH_SLEEP):
    """Sleep every `batch_size` iterations (i is 1-based index)."""
    if batch_size > 0 and i % batch_size == 0:
        time.sleep(max(0.0, sleep_s))

# ============================ HTTP + helpers ===================================
def _session():
    s = requests.Session()
    s.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=5,
                backoff_factor=0.7,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET"],
                raise_on_status=False,
            )
        ),
    )
    return s

SESSION = _session()

def _get_json(url, params=None, timeout=25):
    try:
        r = SESSION.get(url, params=params, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def _today_utc_date():
    return datetime.now(timezone.utc).date()

def _try_variants(symbol: str):
    sym = (symbol or "").strip().upper()
    seen = set()
    for s in [sym, sym.replace("-", "."), sym.replace(".", "-")]:
        if s not in seen:
            seen.add(s)
            yield s

# ============================ Math / format utils ==============================
def _safe_pct(num, den):
    import pandas as pd
    a = pd.to_numeric(num, errors="coerce") if hasattr(num, "__array__") else num
    b = pd.to_numeric(den, errors="coerce") if hasattr(den, "__array__") else den
    if hasattr(a, "__array__") or hasattr(b, "__array__"):
        if not hasattr(a, "__array__"):
            a = np.full_like(b, a, dtype="float64")
        if not hasattr(b, "__array__"):
            b = np.full_like(a, b, dtype="float64")
        return np.where((~pd.isna(a)) & (~pd.isna(b)) & (b != 0), a / b, np.nan)
    try:
        if a is None or b is None or b == 0 or (isinstance(a, float) and not np.isfinite(a)) or (isinstance(b, float) and not np.isfinite(b)):
            return np.nan
        return a / b
    except Exception:
        return np.nan

def _fmt_money(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "N/A"
    try:
        v = float(x)
    except Exception:
        return "N/A"
    abs_v = abs(v)
    if abs_v >= 1e11: return f"${v/1e9:.1f}B"
    if abs_v >= 1e8:  return f"${v/1e9:.2f}B"
    if abs_v >= 1e6:  return f"${v/1e6:.1f}M"
    return f"${v:,.0f}"

def _signed_cagr(current, past, years=3):
    if pd.isna(current) or pd.isna(past) or past == 0:
        return np.nan
    try:
        g = (abs(current)/abs(past))**(1/years) - 1
        return g if abs(current) >= abs(past) else -g
    except Exception:
        return np.nan

# =========================== Base data fetchers =================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_profiles_v3_bulk(tickers, timeout=20):
    rows = []
    for i, t in enumerate(sorted({(t or "").strip().upper() for t in tickers if t}), start=1):
        payload = None
        for tv in _try_variants(t):
            data = _get_json(f"https://financialmodelingprep.com/api/v3/profile/{tv}",
                             params={"apikey": FMP_API_KEY}, timeout=timeout)
            if isinstance(data, list) and data:
                payload = data[0]; break
        rows.append({
            "ticker": t,
            "companyName": (payload or {}).get("companyName"),
            "sector": (payload or {}).get("sector"),
            "industry": (payload or {}).get("industry"),
            "price_profile": (payload or {}).get("price"),
            "beta": (payload or {}).get("beta"),
        })
        time.sleep(0.03)
        _batch_pause(i)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df = df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return df

@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_quote(sym):
    payload = None
    for v in _try_variants(sym):
        data = _get_json(f"https://financialmodelingprep.com/api/v3/quote/{v}",
                         params={"apikey": FMP_API_KEY})
        if isinstance(data, list) and data:
            payload = data[0]; break
    if not payload:
        return pd.DataFrame(columns=["symbol","as_of_quote","yearHigh","yearLow","priceAvg50","priceAvg200","price"])
    df = pd.DataFrame([payload]); df["symbol"] = sym
    df["as_of_quote"] = _today_utc_date()
    return df[["symbol","as_of_quote","yearHigh","yearLow","priceAvg50","priceAvg200","price"]]

@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_ratios_ttm(sym):
    payload = None
    for v in _try_variants(sym):
        data = _get_json(f"https://financialmodelingprep.com/api/v3/ratios-ttm/{v}",
                         params={"apikey": FMP_API_KEY})
        if isinstance(data, list) and data:
            payload = data; break
    if not payload:
        return pd.DataFrame(columns=["symbol","as_of_ratios"])
    df = pd.DataFrame(payload); df["symbol"] = sym

    def _coalesce(df, canon, candidates):
        avail = [c for c in candidates if c in df.columns]
        if not avail: return
        if canon not in df.columns: df[canon] = np.nan
        for c in avail: df[canon] = df[canon].combine_first(df[c])
        for c in avail:
            if c != canon and c in df.columns:
                df.drop(columns=[c], inplace=True, errors="ignore")

    _coalesce(df, "priceToSalesRatioTTM", ["priceToSalesRatioTTM","priceSalesRatioTTM"])
    _coalesce(df, "debtToEquityRatioTTM", ["debtToEquityRatioTTM","debtEquityRatioTTM"])
    _coalesce(df, "peRatioTTM", ["peRatioTTM"])

    if "date" in df.columns:
        df["as_of_ratios"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.date
        df = df.sort_values("as_of_ratios").tail(1)
    keep = ["symbol","as_of_ratios","priceToSalesRatioTTM","priceEarningsToGrowthRatioTTM",
            "priceToFreeCashFlowsRatioTTM","priceToBookRatioTTM","debtToEquityRatioTTM",
            "returnOnEquityTTM","returnOnAssetsTTM","peRatioTTM","currentRatioTTM",
            "operatingProfitMarginTTM","netProfitMarginTTM"]
    return df[[c for c in keep if c in df.columns]]

@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_key_metrics_ttm(sym):
    payload = None
    for v in _try_variants(sym):
        data = _get_json(f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{v}",
                         params={"apikey": FMP_API_KEY})
        if isinstance(data, list) and data:
            payload = data; break
    if not payload:
        return pd.DataFrame(columns=["symbol","as_of_km"])
    df = pd.DataFrame(payload); df["symbol"] = sym
    if "date" in df.columns:
        df["as_of_km"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.date
        df = df.sort_values("as_of_km").tail(1)
    keep = ["symbol","as_of_km","dividendYieldTTM","roicTTM","debtToAssetsTTM","evToSalesTTM"]
    return df[[c for c in keep if c in df.columns]]

@st.cache_data(ttl=1200, show_spinner=False)
def _fetch_rsi_latest_multi(symbols, period=21, interval="1day"):
    frames = []
    for i, s in enumerate(symbols, start=1):
        payload = None
        for v in _try_variants(s):
            data = _get_json(
                f"https://financialmodelingprep.com/api/v3/technical_indicator/{interval}/{v}",
                params={"type":"rsi","period":period,"apikey":FMP_API_KEY}
            )
            if isinstance(data, list) and data:
                payload = data; break
        if payload:
            df = pd.DataFrame(payload)
            df["symbol"] = s
            if "date" in df.columns:
                df["as_of_rsi"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.date
            frames.append(df[["symbol","as_of_rsi","rsi"]])
        time.sleep(0.04)
        _batch_pause(i)
    if not frames:
        return pd.DataFrame(columns=["symbol","as_of_rsi","RSI"])
    all_df = pd.concat(frames, ignore_index=True)
    latest = (all_df.sort_values(["symbol","as_of_rsi"]).groupby("symbol", as_index=False).tail(1))
    latest = latest.rename(columns={"rsi":"RSI"})
    return latest[["symbol","as_of_rsi","RSI"]]

@st.cache_data(ttl=900, show_spinner=False)
def _fetch_price_performance(sym, days=400):
    payload = None
    for v in _try_variants(sym):
        payload = _get_json(
            f"https://financialmodelingprep.com/api/v3/historical-price-full/{v}",
            params={"timeseries": days, "serietype":"line", "apikey": FMP_API_KEY}
        )
        if isinstance(payload, dict) and "historical" in payload and payload["historical"]:
            break
        payload = None
    if not payload:
        return {"symbol": sym, "Perf 1M (%)": np.nan, "Perf 3M (%)": np.nan, "Perf 6M (%)": np.nan, "Perf 1Y (%)": np.nan}

    hist = pd.DataFrame(payload["historical"])
    if hist.empty or "close" not in hist.columns:
        return {"symbol": sym, "Perf 1M (%)": np.nan, "Perf 3M (%)": np.nan, "Perf 6M (%)": np.nan, "Perf 1Y (%)": np.nan}

    hist = hist.sort_values("date")
    closes = hist["close"].astype(float).values
    if len(closes) < 2:
        return {"symbol": sym, "Perf 1M (%)": np.nan, "Perf 3M (%)": np.nan, "Perf 6M (%)": np.nan, "Perf 1Y (%)": np.nan}

    def ret(period):
        if len(closes) <= period:
            return np.nan
        now = closes[-1]
        then = closes[-1 - period]
        return (now/then - 1.0) * 100.0 if then else np.nan

    return {
        "symbol": sym,
        "Perf 1M (%)": ret(21),
        "Perf 3M (%)": ret(63),
        "Perf 6M (%)": ret(126),
        "Perf 1Y (%)": ret(252),
    }

# =========================== Merge per symbol (base) ===========================
def _merge_one_symbol(sym, blocks):
    prof  = blocks["profile"].get(sym)
    quot  = blocks["quote"].get(sym)
    ratio = blocks["ratios"].get(sym)
    km    = blocks["km"].get(sym)
    rsi_all = blocks["rsi"]
    rsi = rsi_all[rsi_all["symbol"] == sym] if (rsi_all is not None and not rsi_all.empty) else pd.DataFrame()

    parts = [p for p in [prof,quot,ratio,km] if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame(columns=["symbol","As Of"])
    base = pd.DataFrame({"symbol":[sym]})
    for p in parts:
        base = base.merge(p, on="symbol", how="left")
    if rsi is not None and not rsi.empty:
        base = base.merge(rsi, on="symbol", how="left")

    date_cols = ["as_of_quote","as_of_ratios","as_of_km","as_of_rsi"]
    dates_df = pd.DataFrame(index=base.index)
    for c in date_cols:
        if c in base.columns:
            dates_df[c] = pd.to_datetime(base[c], errors="coerce")
    base["As Of"] = dates_df.max(axis=1).dt.date if not dates_df.empty else pd.NaT

    if "price" not in base.columns: base["price"] = np.nan
    if "price_profile" in base.columns:
        base["price"] = base["price"].combine_first(base["price_profile"])

    for c in ["yearHigh", "priceAvg50", "priceAvg200"]:
        if c not in base.columns: base[c] = np.nan

    base["pct_from_high"]   = _safe_pct(base.get("price") - base.get("yearHigh"),   base.get("yearHigh"))
    base["pct_above_50ma"]  = _safe_pct(base.get("price") - base.get("priceAvg50"), base.get("priceAvg50"))
    base["pct_above_200ma"] = _safe_pct(base.get("price") - base.get("priceAvg200"),base.get("priceAvg200"))

    pct_cols = [
        "pct_from_high","pct_above_50ma","pct_above_200ma",
        "dividendYieldTTM","returnOnEquityTTM","returnOnAssetsTTM",
        "roicTTM","operatingProfitMarginTTM","netProfitMarginTTM"
    ]
    for c in pct_cols:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce") * 100.0

    front = ["symbol","As Of","price","pct_from_high","pct_above_50ma","pct_above_200ma",
             "beta","yearHigh","yearLow","priceAvg50","priceAvg200","RSI"]
    rest  = [c for c in base.columns if c not in front]
    base  = base[[c for c in front if c in base.columns] + rest]
    return base

def _pretty_rename(df):
    if df.empty: return df
    mapping_basic = {
        "price":"Price","yearHigh":"52W High","yearLow":"52W Low",
        "priceAvg50":"50D MA","priceAvg200":"200D MA","beta":"Beta","RSI":"RSI",
        "pct_from_high":"% From 52W High","pct_above_50ma":"% Above 50D MA","pct_above_200ma":"% Above 200D MA",
        "dividendYieldTTM":"Dividend Yield (%)","returnOnEquityTTM":"ROE (%)","returnOnAssetsTTM":"ROA (%)","roicTTM":"ROIC (%)",
        "operatingProfitMarginTTM":"Operating Margin (%)","netProfitMarginTTM":"Net Margin (%)",
        "priceToSalesRatioTTM":"Price to Sales (TTM)","priceToBookRatioTTM":"Price to Book (TTM)",
        "priceToFreeCashFlowsRatioTTM":"Price to FCF (TTM)","priceEarningsToGrowthRatioTTM":"PEG (TTM)",
        "debtToEquityRatioTTM":"Debt to Equity (TTM)","currentRatioTTM":"Current Ratio (TTM)","peRatioTTM":"P/E (TTM)",
        "debtToAssetsTTM":"Debt to Assets (TTM)","evToSalesTTM":"EV to Sales (TTM)"
    }
    return df.rename(columns={k:v for k,v in mapping_basic.items() if k in df.columns})

@st.cache_data(ttl=1200, show_spinner=False)
def fetch_all_metrics(symbols, rsi_period=21):
    if not symbols:
        return pd.DataFrame(columns=["symbol"])

    profile_blocks = {}
    quote_blocks   = {}
    ratios_blocks  = {}
    km_blocks      = {}
    for i, s in enumerate(symbols, start=1):
        prof_single = fetch_profiles_v3_bulk([s])
        if not prof_single.empty:
            prof_single = prof_single.rename(columns={"ticker":"symbol"})[["symbol","price_profile","beta"]]
        profile_blocks[s] = prof_single
        quote_blocks[s]   = _fetch_quote(s)
        ratios_blocks[s]  = _fetch_ratios_ttm(s)
        km_blocks[s]      = _fetch_key_metrics_ttm(s)
        time.sleep(0.02)
        _batch_pause(i)

    rsi_block = _fetch_rsi_latest_multi(symbols, period=rsi_period)
    blocks = {"profile":profile_blocks, "quote":quote_blocks, "ratios":ratios_blocks, "km":km_blocks, "rsi":rsi_block}

    rows = []
    for i, sym in enumerate(symbols, start=1):
        rows.append(_merge_one_symbol(sym, blocks))
        if i % 200 == 0: time.sleep(1.0)
    df = pd.concat([r for r in rows if r is not None and not r.empty], ignore_index=True) if rows else pd.DataFrame(columns=["symbol"])
    df = _pretty_rename(df)
    ordered = ["symbol"] + [c for c in df.columns if c != "symbol"]
    return df[ordered]

@st.cache_data(ttl=900, show_spinner=False)
def add_price_performance(df_symbols):
    if not df_symbols:
        return pd.DataFrame(columns=["symbol","Perf 1M (%)","Perf 3M (%)","Perf 6M (%)","Perf 1Y (%)"])
    perfs = []
    for i, s in enumerate(df_symbols, start=1):
        perfs.append(_fetch_price_performance(s))
        time.sleep(0.02)
        _batch_pause(i)
    perf_df = pd.DataFrame(perfs)
    return perf_df

# ======================== Peer-app parity fetchers ==============================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_5yr_pe_ps_averages(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(tickers, start=1):
        try:
            url = f'https://financialmodelingprep.com/api/v3/ratios/{t}?period=annual&apikey={FMP_API_KEY}'
            data = _get_json(url)
            if isinstance(data, list):
                for e in data:
                    e['ticker'] = t
                    rows.append(e)
        except Exception:
            pass
        _batch_pause(i)
    if not rows:
        return pd.DataFrame(columns=['ticker','five_year_avg_PE','five_year_avg_PS'])
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.sort_values(['ticker','date'], inplace=True)
    for col in ['priceEarningsRatio','priceToSalesRatio']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['five_year_avg_PE'] = df.groupby('ticker')['priceEarningsRatio'].transform(lambda x: x.mask(x <= 0).rolling(5, min_periods=1).mean())
    df['five_year_avg_PS'] = df.groupby('ticker')['priceToSalesRatio'].transform(lambda x: x.mask(x <= 0).rolling(5, min_periods=1).mean())
    df = df[df['date'] == df.groupby('ticker')['date'].transform('max')]
    return df[['ticker','five_year_avg_PE','five_year_avg_PS']]

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_income_cagr_latest(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(tickers, start=1):
        try:
            url = f'https://financialmodelingprep.com/api/v3/income-statement/{t}?period=annual&apikey={FMP_API_KEY}'
            data = _get_json(url)
            if isinstance(data, list):
                for e in data:
                    e['ticker'] = t
                    rows.append(e)
        except Exception:
            pass
        _batch_pause(i)
    if not rows:
        return pd.DataFrame(columns=['ticker','date','revenue','operatingIncome','eps','netIncome',
                                     'three_year_rev_cagr','op_margin','op_margin_3y_ago','this_year','next_year'])
    df = pd.DataFrame(rows)
    keep = ["ticker","date","revenue","operatingIncome","eps","netIncome"]
    df = df[[c for c in keep if c in df.columns]]
    for c in ["revenue","operatingIncome","eps","netIncome"]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.sort_values(['ticker','date'], inplace=True)
    df['op_margin'] = np.where(df['revenue'].ne(0), df['operatingIncome']/df['revenue'], np.nan)
    df['revenue_3y_ago'] = df.groupby('ticker')['revenue'].shift(3)
    df['op_margin_3y_ago'] = df.groupby('ticker')['op_margin'].shift(3)
    df['three_year_rev_cagr'] = df.apply(lambda r: _signed_cagr(r['revenue'], r['revenue_3y_ago']), axis=1)
    df = df[df['date'] == df.groupby('ticker')['date'].transform('max')].copy()
    df['this_year'] = df['date'] + pd.DateOffset(years=1)
    df['next_year'] = df['date'] + pd.DateOffset(years=2)
    return df[["ticker","date","revenue","operatingIncome","eps","netIncome",
               "op_margin","op_margin_3y_ago","three_year_rev_cagr","this_year","next_year"]]

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_cashflow_cagr_latest(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(tickers, start=1):
        try:
            url = f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{t}?period=annual&apikey={FMP_API_KEY}'
            data = _get_json(url)
            if isinstance(data, list):
                for e in data:
                    e['ticker'] = t
                    rows.append(e)
        except Exception:
            pass
        _batch_pause(i)
    if not rows:
        return pd.DataFrame(columns=['ticker','netIncome','operatingCashFlow','freeCashFlow','ni_cagr','oc_cagr','fcf_cagr'])
    df = pd.DataFrame(rows)
    for c in ['netIncome','operatingCashFlow','freeCashFlow']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.sort_values(['ticker','date'], inplace=True)
    df['ni_3y_ago'] = df.groupby('ticker')['netIncome'].shift(3)
    df['oc_3y_ago'] = df.groupby('ticker')['operatingCashFlow'].shift(3)
    df['fcf_3y_ago'] = df.groupby('ticker')['freeCashFlow'].shift(3)
    df['ni_cagr']  = df.apply(lambda r: _signed_cagr(r['netIncome'],         r['ni_3y_ago']), axis=1)
    df['oc_cagr']  = df.apply(lambda r: _signed_cagr(r['operatingCashFlow'], r['oc_3y_ago']), axis=1)
    df['fcf_cagr'] = df.apply(lambda r: _signed_cagr(r['freeCashFlow'],      r['fcf_3y_ago']), axis=1)
    df = df[df['date'] == df.groupby('ticker')['date'].transform('max')]
    return df[['ticker','netIncome','operatingCashFlow','freeCashFlow','ni_cagr','oc_cagr','fcf_cagr']]

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_analyst_estimates_window(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(tickers, start=1):
        try:
            url = f'https://financialmodelingprep.com/api/v3/analyst-estimates/{t}?apikey={FMP_API_KEY}'
            data = _get_json(url)
            if isinstance(data, list):
                for e in data:
                    e['ticker'] = t
                    rows.append(e)
        except Exception:
            pass
        _batch_pause(i)
    if not rows:
        return pd.DataFrame(columns=['ticker','earnings_date','estimatedRevenueAvg','estimatedEpsAvg','date_start','date_end'])
    df = pd.DataFrame(rows)
    df['earnings_date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values(['ticker','earnings_date'], ascending=[True, False])
    df['date_start'] = df['earnings_date']
    df['date_end']   = df.groupby('ticker')['earnings_date'].shift(1)
    return df[['ticker','earnings_date','estimatedRevenueAvg','estimatedEpsAvg','date_start','date_end']]

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_yfinance_metrics(tickers: list[str]) -> pd.DataFrame:
    try:
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        infos = [yf.Ticker(t).info for t in tickers]
        df_infos = pd.DataFrame(infos)
        if 'symbol' in df_infos.columns:
            df_infos = df_infos.set_index('symbol')
        fundamentals = ['longName','currentPrice','marketCap']
        stock_list_info = df_infos[df_infos.columns[df_infos.columns.isin(fundamentals)]]

        px = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)
        rets = px['Adj Close'].pct_change()
        cumulative_return = (1 + rets).cumprod() - 1
        total_5y = cumulative_return.iloc[[-1]].T.rename(columns={cumulative_return.iloc[[-1]].index[0]:'Total 5 Year Return'})

        one_year_start = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
        one_year_prices = yf.download(tickers, start=one_year_start, end=end_date, auto_adjust=False, progress=False)['Adj Close']
        one_year_return = one_year_prices.pct_change().add(1).cumprod().iloc[-1] - 1
        one_year_return = one_year_return.to_frame(name='Total 1 Year Return')

        price_change = px['Close']
        largest_1d_drop = price_change.pct_change().agg(['min']).T.rename(columns={'min':'Largest 1 Day Drop'})
        largest_5d_drop = price_change.pct_change(-5).agg(['min']).T.rename(columns={'min':'Largest 5 Day Drop'})

        def val_risk_return(returns_df):
            risk = pd.DataFrame({'VAR': returns_df.quantile(0.01)})
            risk['CVAR'] = returns_df[returns_df.lt(returns_df.quantile(0.01))].mean()
            return risk

        def ann_risk_return(returns_df):
            out = returns_df.agg(['std']).T
            out.columns = ['Annualized Volatility']
            out['Annualized Volatility'] = out['Annualized Volatility'] * np.sqrt(252)
            return out

        risk = val_risk_return(rets)
        vol  = ann_risk_return(rets)

        out = stock_list_info.join([total_5y, one_year_return, largest_1d_drop, largest_5d_drop, risk, vol], how='inner')
        out = out.rename(columns={'longName':'Company/ETF Name','currentPrice':'Current Stock Price','marketCap':'Market Cap'})
        out['Annualized Return'] = (1 + out['Total 5 Year Return'])**(1/5) - 1
        out.reset_index(inplace=True)
        out = out.rename(columns={'index':'ticker'}) if 'index' in out.columns else out.rename(columns={'level_0':'ticker'})
        return out
    except Exception:
        return pd.DataFrame(columns=['ticker','Company/ETF Name','Current Stock Price','Market Cap',
                                     'Total 5 Year Return','Total 1 Year Return','Annualized Return',
                                     'Annualized Volatility','VAR','CVAR','Largest 1 Day Drop','Largest 5 Day Drop'])

# ================================ UI ==========================================
st.title("📊 Metrics & Performance")

# Pull from main page if available
screen_df = None
filtered_tickers = []
if "step1" in st.session_state:
    s1 = st.session_state.step1
    screen_df = s1.get("screen_df")
    filtered_tickers = list(s1.get("filtered_tickers") or [])

with st.sidebar:
    st.subheader("Ticker Universe")
    if st.button("Clear page data"):
        st.session_state.pop("metrics_page", None)
        st.experimental_rerun()

    if screen_df is not None and isinstance(screen_df, pd.DataFrame) and not screen_df.empty:
        industries = ["All"] + sorted([x for x in screen_df["industry"].dropna().unique().tolist() if x])
        default_ind = 0
        if "industry_pick" in st.session_state and st.session_state.industry_pick in industries:
            default_ind = industries.index(st.session_state.industry_pick)
        chosen_industry = st.selectbox("Industry (from main screen set):", industries, index=default_ind)

        base = screen_df
        if filtered_tickers:
            base = base[base["ticker"].isin([t.upper() for t in filtered_tickers])]
        if chosen_industry != "All":
            base = base[base["industry"] == chosen_industry]
        from_main_tickers = sorted(base["ticker"].dropna().unique().tolist())
    else:
        chosen_industry = "All"
        from_main_tickers = []

    src = st.radio("Use tickers from:", ["Main Page (recommended)", "Manual input"], index=0, horizontal=True)

    manual_tickers = []
    if src == "Manual input":
        _txt = st.text_area("Paste tickers (comma/space/newline separated):", value="AAPL, MSFT, NVDA")
        manual_tickers = sorted({t.strip().upper() for t in _txt.replace("\n"," ").replace("\t"," ").replace(";",",").replace("  "," ").replace(",", " ").split(" ") if t.strip()})

    tickers = from_main_tickers if src == "Main Page (recommended)" else manual_tickers

    fetch_limit = st.slider("Max tickers to fetch", 5, 200, min(len(tickers), 50) if tickers else 50, 5)
    rsi_period  = st.slider("RSI period", 7, 50, 21, 1)
    add_perf    = st.checkbox("Add price performance (1M/3M/6M/1Y)", value=True)
    add_yf_blk  = st.checkbox("Add 5Y Price & Risk (yfinance)", value=True)
    run_fetch   = st.button("Fetch metrics")

# ============================== FETCH (Step 1) =================================
if run_fetch:
    if not FMP_API_KEY:
        st.error("Missing FMP_API_KEY. Set it in your environment or Space secrets.")
        st.stop()
    if not tickers:
        st.warning("No tickers selected. Choose Industry/Main Page set or enter tickers manually.")
        st.stop()

    work_list = tickers[:fetch_limit]
    with st.spinner(f"Fetching advanced metrics for {len(work_list)} tickers..."):
        df_metrics = fetch_all_metrics(work_list, rsi_period=rsi_period)

    if df_metrics.empty:
        st.warning("No metrics returned. Try a different set of tickers.")
        st.stop()

    with st.spinner("Fetching company profiles..."):
        prof_df = fetch_profiles_v3_bulk(work_list)
    if not prof_df.empty:
        prof_df = prof_df.rename(columns={"ticker":"symbol"})
        df_metrics = df_metrics.merge(prof_df[["symbol","companyName","sector","industry"]], on="symbol", how="left")

    if add_perf:
        with st.spinner("Adding price performance..."):
            perf_df = add_price_performance(work_list)
        if not perf_df.empty:
            df_metrics = df_metrics.merge(perf_df, on="symbol", how="left")

    # ---------- Merge extra peer-app metrics ----------
    work_syms = df_metrics['symbol'].dropna().unique().tolist()

    with st.spinner("Adding peer metrics (5y averages, CAGRs, forward path, price/risk)..."):
        inc_latest   = fetch_income_cagr_latest(work_syms)
        ests_window  = fetch_analyst_estimates_window(work_syms)
        fiveyr_avgs  = fetch_5yr_pe_ps_averages(work_syms)
        cf_latest    = fetch_cashflow_cagr_latest(work_syms)
        yf_block     = fetch_yfinance_metrics(work_syms) if add_yf_blk else pd.DataFrame()

    base = df_metrics.copy()
    base = base.merge(inc_latest.rename(columns={'ticker':'symbol'}), on='symbol', how='left')
    base = base.merge(fiveyr_avgs.rename(columns={'ticker':'symbol'}), on='symbol', how='left')
    base = base.merge(cf_latest.rename(columns={'ticker':'symbol'}), on='symbol', how='left')
    if add_yf_blk and not yf_block.empty:
        base = base.merge(yf_block.rename(columns={'ticker':'symbol'}), on='symbol', how='left')

    # Map analyst windows to this_year / next_year anchors
    if not inc_latest.empty and not ests_window.empty:
        # ✅ keep the critical rename to 'symbol' to avoid KeyError
        w = ests_window.rename(columns={"ticker": "symbol"}).copy()
        w['date_start'] = pd.to_datetime(w['date_start'], errors='coerce')
        w['date_end']   = pd.to_datetime(w['date_end'],   errors='coerce')

        this_merge = inc_latest[['ticker','this_year']].rename(columns={'ticker':'symbol','this_year':'target_date'})
        next_merge = inc_latest[['ticker','next_year']].rename(columns={'ticker':'symbol','next_year':'target_date'})

        def pick_window(m, est):
            j = m.merge(est, on='symbol', how='left')
            j = j[(j['target_date'] >= j['date_start']) & (j['target_date'] < j['date_end'])].copy()
            j = j.sort_values(['symbol','earnings_date'], ascending=[True, False]).drop_duplicates('symbol', keep='first')
            return j[['symbol','estimatedRevenueAvg','estimatedEpsAvg']]

        this_sel = pick_window(this_merge, w).rename(columns={'estimatedRevenueAvg':'revenue_this_year',
                                                              'estimatedEpsAvg':'eps_this_year'})
        next_sel = pick_window(next_merge, w).rename(columns={'estimatedRevenueAvg':'revenue_next_year',
                                                              'estimatedEpsAvg':'eps_next_year'})

        base = base.merge(this_sel, on='symbol', how='left').merge(next_sel, on='symbol', how='left')

    # Derived forward valuation + growth path
    base['pe_fwd'] = pd.to_numeric(base.get('Current Stock Price'), errors='coerce') / pd.to_numeric(base.get('eps_this_year'), errors='coerce')
    base['ps_fwd'] = pd.to_numeric(base.get('Market Cap'), errors='coerce') / pd.to_numeric(base.get('revenue_this_year'), errors='coerce')

    base['EPS % Change'] = (pd.to_numeric(base.get('eps_next_year'), errors='coerce') - pd.to_numeric(base.get('eps_this_year'), errors='coerce')) \
                            / pd.to_numeric(base.get('eps_this_year'), errors='coerce')
    base['Revenue Growth This Year'] = (pd.to_numeric(base.get('revenue_this_year'), errors='coerce') / pd.to_numeric(base.get('revenue'), errors='coerce')) - 1
    base['Revenue Growth Next Year'] = (pd.to_numeric(base.get('revenue_next_year'), errors='coerce') / pd.to_numeric(base.get('revenue_this_year'), errors='coerce')) - 1
    base['Three Year Revenue CAGR']  = base.get('three_year_rev_cagr')

    base = base.rename(columns={
        'op_margin': 'Operating Margin (latest)',
        'op_margin_3y_ago': 'Operating Margin (3Y Ago)',
        'five_year_avg_PE': 'Avg PE 5 Yr',
        'five_year_avg_PS': 'Avg PS 5 Yr',
        'ni_cagr': '3Y Net Income CAGR',
        'oc_cagr': '3Y Operating CF CAGR',
        'fcf_cagr': '3Y FCF CAGR'
    })

    # Persist for later reruns (so agentic panel works)
    st.session_state.metrics_page = {
        "df": base,
        "syms": base["symbol"].tolist(),
        "ts": time.time()
    }

# ========================== DISPLAY & AGENTIC (Step 2) =========================
page_state = st.session_state.get("metrics_page", {})
df_metrics = page_state.get("df")
if isinstance(df_metrics, pd.DataFrame) and not df_metrics.empty:
    st.subheader("Results")
    st.dataframe(df_metrics, use_container_width=True, height=430)

    st.download_button(
        "Download CSV",
        data=df_metrics.to_csv(index=False).encode("utf-8"),
        file_name="metrics_performance.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.subheader("Views")

    t1, t2, t3, t4 = st.tabs(["Valuation (Fwd vs 5Y)", "Growth & Margins (CAGR)", "Revenue/EPS Path", "Price & Risk (5Y)"])

    with t1:
        cols = ["symbol","companyName","pe_fwd","Avg PE 5 Yr","ps_fwd","Avg PS 5 Yr",
                "P/E (TTM)","Price to Sales (TTM)","Price to Book (TTM)","PEG (TTM)",
                "Operating Margin (%)","Net Margin (%)","Debt to Equity (TTM)"]
        cols = [c for c in cols if c in df_metrics.columns]
        st.dataframe(df_metrics[cols].round(3), use_container_width=True)

    with t2:
        cols = ["symbol","companyName","Three Year Revenue CAGR","3Y Net Income CAGR","3Y Operating CF CAGR","3Y FCF CAGR",
                "Operating Margin (latest)","Operating Margin (3Y Ago)","ROE (%)","ROA (%)","ROIC (%)"]
        cols = [c for c in cols if c in df_metrics.columns]
        view = df_metrics[cols].copy()
        for c in cols:
            if c not in ["symbol","companyName"]:
                view[c] = view[c].apply(lambda x: f"{x*100:.1f}%" if isinstance(x,(float,int)) and pd.notna(x) and abs(x) < 5 else (f"{x:.2f}" if isinstance(x,(float,int)) else x))
        st.dataframe(view, use_container_width=True)

    with t3:
        cols = ["symbol","companyName","eps","eps_this_year","eps_next_year",
                "revenue","revenue_this_year","revenue_next_year",
                "Revenue Growth This Year","Revenue Growth Next Year","EPS % Change"]
        cols = [c for c in cols if c in df_metrics.columns]
        def _fmt_pct(v): 
            return f"{v*100:.1f}%" if isinstance(v,(float,int)) and pd.notna(v) else v
        view = df_metrics[cols].copy()
        for c in ["Revenue Growth This Year","Revenue Growth Next Year","EPS % Change"]:
            if c in view.columns: view[c] = view[c].map(_fmt_pct)
        st.dataframe(view, use_container_width=True)

    with t4:
        cols = ["symbol","companyName","Total 5 Year Return","Total 1 Year Return","Annualized Return",
                "Annualized Volatility","VAR","CVAR","Largest 1 Day Drop","Largest 5 Day Drop"]
        cols = [c for c in cols if c in df_metrics.columns]
        view = df_metrics[cols].copy()
        for c in cols:
            if c not in ["symbol","companyName"]:
                view[c] = view[c].apply(lambda x: f"{x*100:.1f}%" if isinstance(x,(float,int)) and pd.notna(x) else x)
        st.dataframe(view, use_container_width=True)

    # ---------------- Agentic Competitive Analysis (natural language) ----------
    st.markdown("---")
    st.subheader("🤖 Agentic Competitive Analysis")

    st.caption("Ask anything in natural language. The analyst will use the tables above and other cached tabs (if available).")

    all_syms = page_state.get("syms", [])
    default_sel = all_syms[: min(12, len(all_syms))]

    selected_syms = st.multiselect("Companies to analyze", options=all_syms, default=default_sel, key="agentic_syms")
    max_rows_for_llm = st.slider("Max companies to send to the analyst", 5, 30, min(12, len(selected_syms) if selected_syms else 12), 1)

    user_query = st.text_area(
        "Ask your financial analyst",
        placeholder="e.g., Compare price risk vs. profitability and valuation; who looks most resilient on downside?",
        height=100,
        key="agentic_query",
    )

    # Build unified context from available tabs/pages
    def _df_to_records(df):
        if isinstance(df, pd.DataFrame) and not df.empty:
            # light rounding to keep token size sane
            df2 = df.copy()


            for c in df2.columns:
                if pd.api.types.is_numeric_dtype(df2[c]):
                    df2[c] = df2[c].round(4)
            for c in df2.columns:
                if pd.api.types.is_datetime64_any_dtype(df2[c]):
                    df2[c] = pd.to_datetime(df2[c], errors="coerce").dt.strftime("%Y-%m-%d")

        # convert any stray python datetime/date objects sitting in object columns
            from datetime import datetime, date
            def _to_str(v):
                return v.isoformat() if isinstance(v, (datetime, date)) else v
            for c in df2.columns:
                if df2[c].dtype == "object":
                    df2[c] = df2[c].map(_to_str)

            return df2.to_dict(orient="records")


        return None

    context = {
        "metrics_page": _df_to_records(df_metrics),
        "main_screen": _df_to_records(st.session_state.get("step1", {}).get("screen_df")),
        # If other tabs store data in session_state, include them here:
        "earnings_tab": _df_to_records(st.session_state.get("earnings_tab", {}).get("df")),
        "risk_tab": _df_to_records(st.session_state.get("risk_tab", {}).get("df")),
    }

    def _slice_records_by_symbols(records, syms):
        if not records:
            return []
        return [row for row in records if str(row.get("symbol","")).upper() in set(syms)]

    def gpt_agentic_analysis(selected_symbols, user_query, context):
        if not user_query or not selected_symbols:
            return "Please select companies and enter a question."

        # Filter each table by selected symbols
        filtered = {}
        for key, recs in context.items():
            if recs:
                filtered[key] = _slice_records_by_symbols(recs, selected_symbols)

        # If no OpenAI client, do a heuristic fallback
        if not client or not OPENAI_API_KEY:
            try:
                df = pd.DataFrame(filtered.get("metrics_page", []))
                if df.empty:
                    return "No data available for analysis."
                score = (
                    df.get("ROIC (%)", 0).fillna(0) * 0.35 +
                    df.get("Operating Margin (%)", 0).fillna(0) * 0.25 -
                    df.get("Debt to Equity (TTM)", 0).fillna(0) * 0.10 +
                    df.get("Perf 1Y (%)", 0).fillna(0) * 0.10
                )
                df["_score"] = score
                df = df.sort_values("_score", ascending=False)
                bullets = [f"- {r['symbol']}: composite score {r['_score']:.2f}" for _, r in df.head(6).iterrows()]
                return "Heuristic view (no GPT key found):\n" + "\n".join(bullets)
            except Exception:
                return "Heuristic view unavailable (missing data)."

        prompt = f"""
You are an autonomous equity research analyst.

Answer the user's question using ONLY the structured data provided (JSON arrays).
Integrate valuation, profitability, leverage, price momentum/risk, growth CAGRs, and forward paths.
Compare peers directly, highlight outliers, and provide decision-ready insights (no fluff).
If a metric is missing, say so rather than inventing numbers.

USER QUESTION:
{user_query}

SELECTED SYMBOLS: {', '.join(selected_symbols)}

CONTEXT DATA (multiple tables):
{json.dumps(filtered, indent=2, ensure_ascii=False, default=str)}
"""
        try:
            r = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"user","content":prompt}],
                temperature=0.35,
                max_tokens=1200,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            return f"GPT analysis failed: {e}"

    if st.button("Run agentic analysis"):
        if not selected_syms:
            st.warning("Pick at least one company to analyze.")
        elif not user_query.strip():
            st.warning("Type a question for the analyst.")
        else:
            with st.spinner("Analyzing across tabs..."):
                out = gpt_agentic_analysis(selected_syms[:max_rows_for_llm], user_query.strip(), context)
            st.markdown(
                '<div style="white-space:pre-wrap;line-height:1.4; background:#fff; border:1px solid #ececf3; '
                'border-radius:10px; padding:14px; font-variant-ligatures:none;">'
                + out + "</div>",
                unsafe_allow_html=True
            )
            st.download_button(
                "Download analysis (.txt)",
                data=out.encode("utf-8"),
                file_name="agentic_competitive_analysis.txt",
                mime="text/plain"
            )
else:
    st.info("Select a ticker universe in the sidebar (preferably from your main page), choose options, then click **Fetch metrics**.")
