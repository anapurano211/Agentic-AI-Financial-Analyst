# app.py — Screener ➜ (optional) Metrics Filters ➜ Industry ➜ Profiles ➜ Agentic Earnings/Sentiment
# -----------------------------------------------------------------------------
# Run:  streamlit run app.py
#
# Required env vars: FMP_API_KEY, OPENAI_API_KEY
# (No hard-coded keys. Earnings only run after you pick Industry.)

import os
import re
import json
import time
import math
import numpy as np
import pandas as pd
import requests
import streamlit as st
from datetime import datetime, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from openai import OpenAI

# =============================== Keys =========================================
FMP_API_KEY    = os.getenv("FMP_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ============================ HTTP w/ retries =================================
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

# =============================== Screener =====================================
SERVER_KEYS = {
    "country","exchange","sector","industry","isActivelyTrading",
    "marketCapMoreThan","marketCapLowerThan","priceMoreThan","priceLowerThan",
    "volumeMoreThan","volumeLowerThan","betaMoreThan","betaLowerThan",
    "dividendMoreThan","dividendLowerThan","limit","page","isEtf","isFund"
}

def _sanitize_params(filters):
    p = {k:v for k,v in (filters or {}).items() if k in SERVER_KEYS and v not in (None,"",[])}
    for bkey in ["isActivelyTrading","isEtf","isFund"]:
        if bkey in p and isinstance(p[bkey], bool):
            p[bkey] = "true" if p[bkey] else "false"
    p["limit"] = max(1, min(int(p.get("limit", 500)), 500))
    return p

@st.cache_data(ttl=3600, show_spinner=False)
def fmp_company_screener_safe(filters, timeout=20, max_pages=10):
    url = "https://financialmodelingprep.com/stable/company-screener"
    params_base = _sanitize_params(filters) | {"apikey": FMP_API_KEY}
    all_rows, meta = [], {"errors": [], "warnings": [], "pages": 0}

    for page in range(max_pages):
        try:
            r = SESSION.get(url, params={**params_base, "page": page}, timeout=timeout)
            if r.status_code == 429:
                time.sleep(1.0); continue
            if r.status_code != 200:
                meta["errors"].append({"page":page, "status":r.status_code}); break
            data = r.json() or []
            if not data: break
            df = pd.DataFrame(data)
            if "symbol" in df.columns: df = df.rename(columns={"symbol":"ticker"})
            all_rows.append(df)
            meta["pages"] = page+1
            if len(df) < params_base["limit"]: break
        except Exception as e:
            meta["warnings"].append({"page":page,"msg":repr(e)}); break

    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if "ticker" not in out.columns: out["ticker"] = pd.Series(dtype=str)
    if not out.empty:
        out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
        # client-side ETF/Fund fallback if server ignored flags
        for col in ["isEtf","isFund"]:
            if col in out.columns:
                out = out[out[col].astype(str).str.lower().isin(["false","0","nan"])]
        out = out.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return out, meta

# ============================ Profiles (v3) ====================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_profiles_v3_bulk(tickers, timeout=20):
    rows = []
    for t in sorted({(t or "").strip().upper() for t in tickers if t}):
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
            "description": (payload or {}).get("description")
        })
        time.sleep(0.04)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df = df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return df

# ===================== Advanced Metrics (multi-ticker) =========================
def _safe_pct(num, den):
    return np.where((pd.notna(den)) & (den != 0), num/den, np.nan)

def _to_date_series(df, col):
    return pd.to_datetime(df[col], errors="coerce", utc=True).dt.date if col in df.columns else pd.Series([pd.NaT]*len(df), index=df.index)

def _latest_one_row(df, date_col=None):
    if df is None or df.empty: return df
    if date_col and date_col in df.columns: return df.sort_values(date_col).tail(1)
    return df.head(1)

def _coalesce_columns(df, mapping):
    df = df.copy()
    for canon, candidates in mapping.items():
        avail = [c for c in candidates if c in df.columns]
        if not avail: continue
        if canon not in df.columns: df[canon] = np.nan
        for c in avail: df[canon] = df[canon].combine_first(df[c])
        for c in avail:
            if c != canon and c in df.columns:
                df.drop(columns=[c], inplace=True, errors="ignore")
    return df

def _fetch_profile(sym):
    payload = None
    for v in _try_variants(sym):
        data = _get_json(f"https://financialmodelingprep.com/api/v3/profile/{v}",
                         params={"apikey": FMP_API_KEY})
        if isinstance(data, list) and data:
            payload = data[0]; break
    if not payload: return pd.DataFrame(columns=["symbol","as_of_profile","price_profile","beta"])
    df = pd.DataFrame([payload]); df["symbol"] = sym
    df["as_of_profile"] = _today_utc_date()
    return df[["symbol","as_of_profile","price","beta"]].rename(columns={"price":"price_profile"})

def _fetch_quote(sym):
    payload = None
    for v in _try_variants(sym):
        data = _get_json(f"https://financialmodelingprep.com/api/v3/quote/{v}",
                         params={"apikey": FMP_API_KEY})
        if isinstance(data, list) and data:
            payload = data[0]; break
    if not payload: return pd.DataFrame(columns=["symbol","as_of_quote","yearHigh","yearLow","priceAvg50","priceAvg200","price"])
    df = pd.DataFrame([payload]); df["symbol"] = sym
    df["as_of_quote"] = _today_utc_date()
    return df[["symbol","as_of_quote","yearHigh","yearLow","priceAvg50","priceAvg200","price"]]

def _fetch_ratios_ttm(sym):
    payload = None
    for v in _try_variants(sym):
        data = _get_json(f"https://financialmodelingprep.com/api/v3/ratios-ttm/{v}",
                         params={"apikey": FMP_API_KEY})
        if isinstance(data, list) and data:
            payload = data; break
    if not payload: return pd.DataFrame(columns=["symbol","as_of_ratios"])
    df = pd.DataFrame(payload); df["symbol"] = sym
    df = _coalesce_columns(df, {
        "priceToSalesRatioTTM": ["priceToSalesRatioTTM","priceSalesRatioTTM"],
        "debtToEquityRatioTTM": ["debtToEquityRatioTTM","debtEquityRatioTTM"],
        "peRatioTTM": ["peRatioTTM"],
    })
    df["as_of_ratios"] = _to_date_series(df, "date")
    df = _latest_one_row(df, "date")
    keep = ["symbol","as_of_ratios","priceToSalesRatioTTM","priceEarningsToGrowthRatioTTM",
            "priceToFreeCashFlowsRatioTTM","priceToBookRatioTTM","debtToEquityRatioTTM",
            "returnOnEquityTTM","returnOnAssetsTTM","peRatioTTM","currentRatioTTM",
            "operatingProfitMarginTTM","netProfitMarginTTM"]
    return df[[c for c in keep if c in df.columns]]

def _fetch_key_metrics_ttm(sym):
    payload = None
    for v in _try_variants(sym):
        data = _get_json(f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{v}",
                         params={"apikey": FMP_API_KEY})
        if isinstance(data, list) and data:
            payload = data; break
    if not payload: return pd.DataFrame(columns=["symbol","as_of_km"] )
    df = pd.DataFrame(payload); df["symbol"] = sym
    df["as_of_km"] = _to_date_series(df, "date")
    df = _latest_one_row(df, "date")
    keep = ["symbol","as_of_km","dividendYieldTTM","roicTTM","debtToAssetsTTM","evToSalesTTM"]
    return df[[c for c in keep if c in df.columns]]

def _fetch_rsi_latest_multi(symbols, period=21, interval="1day"):
    frames = []
    for s in symbols:
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
            df["as_of_rsi"] = _to_date_series(df, "date")
            frames.append(df[["symbol","as_of_rsi","rsi"]])
        time.sleep(0.05)
    if not frames: return pd.DataFrame(columns=["symbol","as_of_rsi","RSI"])
    all_df = pd.concat(frames, ignore_index=True)
    latest = (all_df.sort_values(["symbol","as_of_rsi"]).groupby("symbol", as_index=False).tail(1))
    latest = latest.rename(columns={"rsi":"RSI"})
    return latest[["symbol","as_of_rsi","RSI"]]

def _merge_one_symbol(sym, blocks):
    prof  = blocks["profile"].get(sym)
    quot  = blocks["quote"].get(sym)
    ratio = blocks["ratios"].get(sym)
    km    = blocks["km"].get(sym)
    rsi_all = blocks["rsi"]
    rsi = rsi_all[rsi_all["symbol"] == sym] if rsi_all is not None and not rsi_all.empty else pd.DataFrame()

    parts = [p for p in [prof,quot,ratio,km] if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame(columns=["symbol","As Of"])
    base = pd.DataFrame({"symbol":[sym]})
    for p in parts:
        base = base.merge(p, on="symbol", how="left")
    if rsi is not None and not rsi.empty:
        base = base.merge(rsi, on="symbol", how="left")

    date_cols = ["as_of_quote","as_of_profile","as_of_ratios","as_of_km","as_of_rsi"]
    dates_df = pd.DataFrame(index=base.index)
    for c in date_cols:
        if c in base.columns:
            dates_df[c] = pd.to_datetime(base[c], errors="coerce")
    base["As Of"] = dates_df.max(axis=1).dt.date if not dates_df.empty else pd.NaT

    if "price" not in base.columns: base["price"] = np.nan
    if "price_profile" in base.columns:
        base["price"] = base["price"].combine_first(base["price_profile"])

    base["pct_from_high"]   = _safe_pct(base.get("price") - base.get("yearHigh"),   base.get("yearHigh"))
    base["pct_above_50ma"]  = _safe_pct(base.get("price") - base.get("priceAvg50"), base.get("priceAvg50"))
    base["pct_above_200ma"] = _safe_pct(base.get("price") - base.get("priceAvg200"),base.get("priceAvg200"))

    base.drop(columns=[c for c in date_cols if c in base.columns], inplace=True, errors="ignore")

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

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_all_metrics(symbols, rsi_period=21):
    if not symbols:
        return pd.DataFrame(columns=["symbol"])
    profile_blocks = {s:_fetch_profile(s)    for s in symbols}
    quote_blocks   = {s:_fetch_quote(s)      for s in symbols}
    ratios_blocks  = {s:_fetch_ratios_ttm(s) for s in symbols}
    km_blocks      = {s:_fetch_key_metrics_ttm(s) for s in symbols}
    rsi_block      = _fetch_rsi_latest_multi(symbols, period=rsi_period)
    blocks = {"profile":profile_blocks, "quote":quote_blocks, "ratios":ratios_blocks, "km":km_blocks, "rsi":rsi_block}

    rows = []
    for i, sym in enumerate(symbols, start=1):
        rows.append(_merge_one_symbol(sym, blocks))
        time.sleep(0.03)
        if i % 200 == 0: time.sleep(1.0)
    df = pd.concat([r for r in rows if r is not None and not r.empty], ignore_index=True) if rows else pd.DataFrame(columns=["symbol"])
    df = _pretty_rename(df)
    ordered = ["symbol"] + [c for c in df.columns if c != "symbol"]
    return df[ordered]

# ========================= Company description summary ========================
def _clamp_lines(lines, n=200):
    return [ln[:n] for ln in lines]

def summarize_description(desc, name="", ticker="", sector="", industry="", user_prompt=None):
    if not isinstance(desc, str) or not desc.strip():
        return f"{name or ticker or 'Company'} ({ticker})"
    if not client:
        parts = re.split(r"(?<=[.!?])\s+", desc.strip())
        parts = [p.strip() for p in parts if p.strip()]
        return "\n".join(parts[:2]) if parts else (name or ticker or "Company")

    base = f"""
You are a concise company profiler. Using ONLY the text below, {user_prompt or "write up to three short lines (no bullets): 1) what the company does, 2) key products/segments if present, 3) other details if present."}
Do not invent facts. Keep each line under ~200 characters.

Company: {name or ticker} ({ticker or 'N/A'})
Sector: {sector or 'N/A'}; Industry: {industry or 'N/A'}
Text:
\"\"\"{desc[:4000]}\"\"\""""
    try:
        r = client.chat.completions.create(
            model=OPENAI_MODEL, messages=[{"role":"user","content":base}], temperature=0.2
        )
        text = (r.choices[0].message.content or "").strip()
    except Exception:
        parts = re.split(r"(?<=[.!?])\s+", desc.strip())
        parts = [p.strip() for p in parts if p.strip()]
        return "\n".join(parts[:2]) if parts else (name or ticker or "Company")

    text = re.sub(r"(?i)\badditional details not in source\.?\b","",text)
    lines = [ln.strip(" -•\t") for ln in text.splitlines() if ln.strip()]
    seen, cleaned = set(), []
    for ln in lines:
        if ln and ln not in seen:
            cleaned.append(ln); seen.add(ln)
    if not cleaned:
        parts = re.split(r"(?<=[.!?])\s+", desc.strip())
        parts = [p.strip() for p in parts if p.strip()]
        cleaned = parts[:2] if parts else [name or ticker or "Company"]
    cleaned = _clamp_lines(cleaned, 200)
    return "\n".join(cleaned[:3])

# =========================== Agentic transcript bits ===========================
def _grab_number(txt, *keys):
    for k in keys:
        m = re.search(rf'(\d+)\s*(?:-| )?\s*{k}', txt)
        if m: return int(m.group(1))
    return None

def parse_transcript_instruction(raw: str):
    """Free-text → switches: include_sentiment, sentiment_only, style, count, focus, exclude, want_qoq."""
    txt = (raw or "").strip().lower()

    include_sentiment = not any(p in txt for p in ["no sentiment","without sentiment","skip sentiment","omit sentiment"])
    sentiment_only = ("sentiment only" in txt) or ("only sentiment" in txt)

    count = _grab_number(txt, "bullet","bullets","line","lines","point","points","pt","pts") or 6
    style = "bullets"
    if re.search(r'\bline|lines\b', txt): style = "lines"

    focus, exclude = [], []
    m_focus = re.search(r'(?:focus on|about|on)\s+([^.\n;]+)', txt)
    m_excl  = re.search(r'(?:exclude|avoid)\s+([^.\n;]+)', txt)
    if m_focus:
        focus = [w.strip() for w in re.split(r',|/|;|\band\b', m_focus.group(1)) if w.strip()]
    if m_excl:
        exclude = [w.strip() for w in re.split(r',|/|;|\band\b', m_excl.group(1)) if w.strip()]

    syn_map = {
        "gm": "gross margin", "margins": "margins", "margin": "margins",
        "opex": "operating expenses", "capex": "capital expenditures",
        "fcf": "free cash flow", "free cashflow": "free cash flow",
        "rev": "revenue", "topline": "revenue", "bottom line": "net income", "eps": "eps",
        "demand": "demand", "orders": "bookings", "backlog": "backlog", "guidance": "guidance",
    }
    for k, v in syn_map.items():
        if re.search(rf'\b{k}\b', txt) and v not in focus:
            focus.append(v)

    want_qoq = any(k in txt for k in ["qoq","q/q","quarter over quarter","q-o-q"])

    return {
        "include_sentiment": include_sentiment and not sentiment_only,
        "sentiment_only": sentiment_only,
        "style": style,
        "count": max(1, count),
        "focus": focus,
        "exclude": exclude,
        "want_qoq": want_qoq,
    }

@st.cache_data(ttl=1200, show_spinner=False)
def fetch_income_quarters(symbol: str, limit=6):
    for tv in _try_variants(symbol):
        data = _get_json(
            f"https://financialmodelingprep.com/api/v3/income-statement/{tv}",
            params={"period":"quarter","limit":limit,"apikey":FMP_API_KEY}
        )
        if isinstance(data, list) and data:
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date")
            return df
    return pd.DataFrame()

@st.cache_data(ttl=1200, show_spinner=False)
def fetch_cashflow_quarters(symbol: str, limit=6):
    for tv in _try_variants(symbol):
        data = _get_json(
            f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{tv}",
            params={"period":"quarter","limit":limit,"apikey":FMP_API_KEY}
        )
        if isinstance(data, list) and data:
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date")
            return df
    return pd.DataFrame()

def _pct(a, b):
    try:
        if a is None or b is None or b == 0: return None
        return (a - b) / b * 100.0
    except Exception:
        return None

def _quarter_label_from_date(dt: pd.Timestamp):
    if pd.isna(dt): return None
    m, y = int(dt.month), int(dt.year)
    q = (m - 1)//3 + 1
    return f"Q{q} FY{y}"

def fetch_qoq_fundamentals(symbol: str):
    inc = fetch_income_quarters(symbol, limit=6)
    cf  = fetch_cashflow_quarters(symbol, limit=6)
    if inc.empty and cf.empty:
        return {}

    def last_two(series):
        vals = [v for v in series.tolist() if pd.notna(v)]
        return (vals[-1] if len(vals) >=1 else None, vals[-2] if len(vals) >=2 else None)

    r_now, r_prev = last_two(inc["revenue"] if "revenue" in inc.columns else pd.Series(dtype=float))
    ni_now, ni_prev = last_two(inc["netIncome"] if "netIncome" in inc.columns else pd.Series(dtype=float))

    fcf_now, fcf_prev = None, None
    if not cf.empty:
        if "freeCashFlow" in cf.columns and cf["freeCashFlow"].notna().any():
            fcf_now, fcf_prev = last_two(cf["freeCashFlow"])
        else:
            oc  = cf["netCashProvidedByOperatingActivities"] if "netCashProvidedByOperatingActivities" in cf.columns else pd.Series(dtype=float)
            cap = cf["capitalExpenditure"] if "capitalExpenditure" in cf.columns else pd.Series(dtype=float)
            fcf_series = oc.combine(cap, lambda a, b: a - b if pd.notna(a) and pd.notna(b) else np.nan)
            fcf_now, fcf_prev = last_two(fcf_series)

    dts = inc["date"].dropna() if "date" in inc.columns else pd.Series([], dtype="datetime64[ns]")
    last_dt = dts.iloc[-1] if len(dts) else pd.NaT
    prev_dt = dts.iloc[-2] if len(dts) > 1 else pd.NaT
    return {
        "revenue_now": r_now, "revenue_prev": r_prev, "revenue_qoq": _pct(r_now, r_prev),
        "netincome_now": ni_now, "netincome_prev": ni_prev, "netincome_qoq": _pct(ni_now, ni_prev),
        "fcf_now": fcf_now, "fcf_prev": fcf_prev, "fcf_qoq": _pct(fcf_now, fcf_prev),
        "label_now": _quarter_label_from_date(last_dt), "label_prev": _quarter_label_from_date(prev_dt),
    }

def _fmt_money(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "N/A"
    try: v = float(x)
    except Exception: return "N/A"
    abs_v = abs(v)
    if abs_v >= 1e11: return f"${v/1e9:.1f}B"
    if abs_v >= 1e8:  return f"${v/1e9:.2f}B"
    if abs_v >= 1e6:  return f"${v/1e6:.1f}M"
    return f"${v:,.0f}"

def _fmt_pct(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "N/A"
    return f"{x:+.1f}%"

def qoq_hint_text(q):
    if not q: return None
    parts = []
    if any(q.get(k) is not None for k in ["revenue_now","revenue_prev","revenue_qoq"]):
        parts.append(f"Revenue ({q.get('label_prev') or 'prev'}→{q.get('label_now') or 'now'}): "
                     f"{_fmt_money(q.get('revenue_prev'))} → {_fmt_money(q.get('revenue_now'))} ({_fmt_pct(q.get('revenue_qoq'))})")
    if any(q.get(k) is not None for k in ["netincome_now","netincome_prev","netincome_qoq"]):
        parts.append(f"Net income: {_fmt_money(q.get('netincome_prev'))} → {_fmt_money(q.get('netincome_now'))} ({_fmt_pct(q.get('netincome_qoq'))})")
    if any(q.get(k) is not None for k in ["fcf_now","fcf_prev","fcf_qoq"]):
        parts.append(f"Free cash flow: {_fmt_money(q.get('fcf_prev'))} → {_fmt_money(q.get('fcf_now'))} ({_fmt_pct(q.get('fcf_qoq'))})")
    return "; ".join(parts) if parts else None

def qoq_box_html(q):
    if not q: return ""
    rows = []
    if any(q.get(k) is not None for k in ["revenue_now","revenue_prev","revenue_qoq"]):
        rows.append(f"<div><b>Revenue</b> ({q.get('label_prev') or 'prev'}→{q.get('label_now') or 'now'}): "
                    f"{_fmt_money(q.get('revenue_prev'))} → {_fmt_money(q.get('revenue_now'))} ({_fmt_pct(q.get('revenue_qoq'))})</div>")
    if any(q.get(k) is not None for k in ["netincome_now","netincome_prev","netincome_qoq"]):
        rows.append(f"<div><b>Net income</b>: {_fmt_money(q.get('netincome_prev'))} → {_fmt_money(q.get('netincome_now'))} ({_fmt_pct(q.get('netincome_qoq'))})</div>")
    if any(q.get(k) is not None for k in ["fcf_now","fcf_prev","fcf_qoq"]):
        rows.append(f"<div><b>Free cash flow</b>: {_fmt_money(q.get('fcf_prev'))} → {_fmt_money(q.get('fcf_now'))} ({_fmt_pct(q.get('fcf_qoq'))})</div>")
    return (
        '<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:10px;margin:8px 0 10px 0;'
        'font-size:12px;color:#334155">'
        '<div style="font-weight:600;margin-bottom:6px">QoQ snapshot</div>' + "".join(rows) + '</div>'
    )

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_latest_transcript(symbol: str, limit: int = 1):
    url = f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{symbol}"
    data = _get_json(url, params={"apikey":FMP_API_KEY,"limit":limit})
    if isinstance(data, list) and data:
        d = data[0]
        return {"content": d.get("content",""), "year": d.get("year"), "quarter": d.get("quarter"), "date": d.get("date") or d.get("dateReported")}
    return {}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_specific_transcript(symbol: str, year: int, quarter: int):
    url = f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{symbol}"
    data = _get_json(url, params={"apikey":FMP_API_KEY,"year":year,"quarter":quarter})
    if isinstance(data, list) and data:
        d = data[0]
        return {"content": d.get("content",""), "year": d.get("year") or year, "quarter": d.get("quarter") or quarter, "date": d.get("date") or d.get("dateReported")}
    return {}

def summarize_transcript(text: str, user_instruction: str | None = None,
                         style: str = "bullets", count: int = 6,
                         focus: list[str] | None = None, exclude: list[str] | None = None,
                         qoq_hint: str | None = None) -> str:
    if not isinstance(text, str) or not text.strip():
        return "No transcript available."
    if not client:
        return text[:1200] + ("..." if len(text) > 1200 else "")

    focus_txt = f" Focus ONLY on: {', '.join(focus)}." if focus else ""
    excl_txt  = f" Do NOT discuss: {', '.join(exclude)}." if exclude else ""
    if style == "lines":
        shape = f"Write EXACTLY {count} short lines (no bullets). Each line ≤ 180 characters."
    else:
        shape = f"Write EXACTLY {count} concise bullet points. No preface or closing text."

    hint = f"\nUse these QoQ facts as ground truth when relevant:\n{qoq_hint}\n" if qoq_hint else ""
    extra = (user_instruction or "").strip()

    prompt = f"""
You are an equity analyst. Using ONLY the transcript text below, {shape}{focus_txt}{excl_txt}
Be precise, factual, and avoid boilerplate.{hint}
If numbers are not in the transcript, don't invent them.

User preference: {extra}

Transcript:
\"\"\"{text[:12000]}\"\"\""""
    try:
        r = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return text[:1200] + ("..." if len(text) > 1200 else "")

def analyze_sentiment_gpt(summary_text: str) -> dict:
    if not isinstance(summary_text, str) or not summary_text.strip():
        return {}
    if not client:
        return {"overall_polarity":0.0,"label":"Neutral","confidence":0.5,"subscores":{"results":0,"guidance":0,"demand":0,"margins":0},"drivers":[]}

    prompt = f"""
You are an equity analyst. Rate the sentiment of this earnings call SUMMARY.
Rules:
- overall_polarity in [-1,1] (Bearish=-1, Neutral=0, Bullish=+1)
- label: Bearish if polarity < -0.15; Neutral if -0.15..0.15; Bullish if > 0.15
- confidence in [0,1]
- Provide subscores in [-1,1] for: results, guidance, demand, margins
- Include 2-5 short 'drivers' behind the score
Return ONLY JSON with keys: overall_polarity, label, confidence, subscores, drivers.

SUMMARY:
\"\"\"{summary_text[:8000]}\"\"\""""
    try:
        r = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
            response_format={"type":"json_object"},
        )
        raw = (r.choices[0].message.content or "").strip()
        data = json.loads(raw)
        data["overall_polarity"] = max(-1.0, min(1.0, float(data.get("overall_polarity", 0.0))))
        data["confidence"]      = max(0.0,  min(1.0, float(data.get("confidence", 0.0))))
        data["subscores"] = data.get("subscores") or {}
        data["drivers"]   = data.get("drivers") or []
        return data
    except Exception:
        return {"overall_polarity":0.0,"label":"Neutral","confidence":0.5,"subscores":{"results":0,"guidance":0,"demand":0,"margins":0},"drivers":[]}

# ================================ UI ==========================================
st.set_page_config(page_title="Agentic AI Financial Analyst", layout="wide")

st.markdown("""
<style>
.cardbox{background:#f8f9fb;border:1px solid #e7e8ef;border-radius:14px;padding:14px;margin-bottom:16px;}
.subtle{color:#4b5563;font-size:13px;margin-bottom:6px}
.scrollarea{
  white-space:pre-wrap;line-height:1.38;max-height:300px;overflow:auto;background:white;border:1px solid #ececf3;border-radius:10px;padding:12px;
  font-variant-ligatures:none; word-break:normal; overflow-wrap:anywhere; letter-spacing:normal;
}
</style>
""", unsafe_allow_html=True)

st.title("🧩 Agentic AI Financial Analyst — Screener → Metrics → Industry → Earnings")

# --------------------------- Sidebar controls ---------------------------------
with st.sidebar:
    st.subheader("Base Screener (US common stocks)")
    sector = st.selectbox("Sector", [
        "Technology","Energy","Financial Services","Industrials","Healthcare",
        "Utilities","Materials","Consumer Defensive","Consumer Cyclical",
        "Real Estate","Communication Services"
    ], index=0)

    # ⬇️ NEW: Range sliders for Market Cap & Volume
    cap_b_min, cap_b_max = st.slider("Market Cap Range ($B)", 0, 30000, (0, 30000), 1)
    volume_min, volume_max = st.slider("Avg Volume Range", 0, 200_000_000, (0, 200_000_000), 50_000)

    limit = st.slider("Max companies to fetch", 50, 500, 300, 50)

    st.subheader("Advanced Metrics (optional)")
    enable_metrics = st.checkbox("Enable advanced post-screen filters", value=True)
    rsi_period     = st.slider("RSI period", 7, 50, 21, 1)
    metrics_max    = st.slider("Max tickers to fetch metrics for", 20, 200, 120, 10)

    if enable_metrics:
        st.markdown("**Valuation**")
        pe_min, pe_max = st.slider("P/E (TTM)", -100.0, 800.0, (0.0, 60.0))
        ps_min, ps_max = st.slider("Price to Sales (TTM)", 0.0, 800.0, (0.0, 15.0))
        pb_min, pb_max = st.slider("Price to Book (TTM)", -100.0, 800.0, (0.0, 15.0))
        st.markdown("**Quality / Leverage**")
        de_min, de_max = st.slider("Debt to Equity (TTM)", 0.0, 500.0, (0.0, 2.0))
        cr_min, cr_max = st.slider("Current Ratio (TTM)", 0.0, 100.0, (0.0, 5.0))
        st.markdown("**Momentum / Price Positioning** (percent)")
        p50_min, p50_max   = st.slider("% Above 50D MA", -100.0, 100.0, (-25.0, 25.0))
        p200_min, p200_max = st.slider("% Above 200D MA", -100.0, 100.0, (-25.0, 25.0))
        from_high_min, from_high_max = st.slider("% From 52W High", -100.0, 100.0, (-100.0, 5.0))
        rsi_min, rsi_max = st.slider("RSI", 0.0, 100.0, (20.0, 80.0))
        st.markdown("**Profitability / Yield** (percent)")
        roe_min, roe_max = st.slider("ROE (%)", -50.0, 800.0, (0.0, 60.0))
        roa_min, roa_max = st.slider("ROA (%)", -30.0, 500.0, (0.0, 30.0))
        roic_min, roic_max = st.slider("ROIC (%)", -30.0, 800.0, (0.0, 50.0))
        opm_min, opm_max = st.slider("Operating Margin (%)", -50.0, 100.0, (0.0, 50.0))
        npm_min, npm_max = st.slider("Net Margin (%)", -50.0, 100.0, (0.0, 40.0))
        div_min, div_max = st.slider("Dividend Yield (%)", 0.0, 100.0, (0.0, 10.0))
        ignore_missing = st.checkbox("Ignore missing values when filtering", value=True)
    else:
        # create dummies so code below can reference them
        pe_min=ps_min=pb_min=de_min=cr_min=p50_min=p200_min=from_high_min=rsi_min=roe_min=roa_min=roic_min=opm_min=npm_min=div_min=0.0
        pe_max=800.0; ps_max=pb_max=800.0; de_max=500.0; cr_max=100.0
        p50_max=p200_max=100.0; from_high_max=100.0; rsi_max=100.0
        roe_max=800.0; roa_max=500.0; roic_max=800.0; opm_max=100.0; npm_max=100.0; div_max=100.0
        ignore_missing=True

    st.subheader("Company Summary")
    desc_prompt = st.text_area(
        "Prompt (optional)",
        value="Write up to three short lines: 1) what the company does, 2) key products/segments if present, 3) other details if present.",
        height=90,
    )

    st.subheader("Earnings Transcript (agentic)")
    include_transcripts = st.checkbox("Include earnings call summary / sentiment", value=True)
    mode = st.radio("Transcript mode", ["Latest", "Specific (year/quarter)"], index=0, horizontal=True)
    coly, colq = st.columns(2)
    with coly:
        year = st.number_input("Year (if specific)", min_value=2000, max_value=2100, value=2024, step=1)
    with colq:
        quarter = st.selectbox("Quarter (if specific)", options=[1,2,3,4], index=1)
    transcript_instruction = st.text_area(
        "Tell the analyst what you want",
        value="Summarize in 5–7 bullets: Results, Guidance, Demand, Margins/Costs, Capital returns, Risks.",
        height=90,
    )

    # STEP 1 button — screens + (optional) metrics only
    run_step1 = st.button("Step 1 — Run screener + metrics")

# =============================== FLOW STEP 1 ==================================
if run_step1:
    if not FMP_API_KEY:
        st.error("Missing FMP_API_KEY (set env var or Space secret)."); st.stop()

    # Base screener
    with st.spinner("Screening companies..."):
        filters = {
            "country":"US", "sector": sector,
            "isActivelyTrading":"true", "isEtf":"false", "isFund":"false",
            # ⬇️ Use both min & max for cap and volume
            "marketCapMoreThan": int(cap_b_min) * 1_000_000_000,
            "marketCapLowerThan": int(cap_b_max) * 1_000_000_000,
            "volumeMoreThan": int(volume_min),
            "volumeLowerThan": int(volume_max),
            "limit": int(limit),
        }
        screen_df, meta = fmp_company_screener_safe(filters)
    st.caption(f"Screened {len(screen_df)} tickers (pages: {meta.get('pages',0)}).")
    st.dataframe(screen_df[["ticker","companyName","sector","industry"]], use_container_width=True, height=250)

    # Metrics filter
    filtered_tickers = screen_df["ticker"].dropna().unique().tolist()
    metrics_df = None
    if enable_metrics and filtered_tickers:
        subset = filtered_tickers[: int(metrics_max)]
        with st.spinner(f"Fetching advanced metrics for {len(subset)} tickers..."):
            metrics_df = fetch_all_metrics(subset, rsi_period=rsi_period)

        if not metrics_df.empty:
            st.dataframe(metrics_df.head(60), use_container_width=True, height=280)

        def _rng(col, lo, hi):
            s = pd.to_numeric(metrics_df.get(col), errors="coerce")
            return (s.isna()) | ((s >= lo) & (s <= hi)) if ignore_missing else ((s >= lo) & (s <= hi))

        if metrics_df is not None and not metrics_df.empty:
            mask = np.ones(len(metrics_df), dtype=bool)
            conds = [
                _rng("P/E (TTM)", pe_min, pe_max),
                _rng("Price to Sales (TTM)", ps_min, ps_max),
                _rng("Price to Book (TTM)", pb_min, pb_max),
                _rng("Debt to Equity (TTM)", de_min, de_max),
                _rng("Current Ratio (TTM)", cr_min, cr_max),
                _rng("% Above 50D MA", p50_min, p50_max),
                _rng("% Above 200D MA", p200_min, p200_max),
                _rng("% From 52W High", from_high_min, from_high_max),
                _rng("RSI", rsi_min, rsi_max),
                _rng("ROE (%)", roe_min, roe_max),
                _rng("ROA (%)", roa_min, roa_max),
                _rng("ROIC (%)", roic_min, roic_max),
                _rng("Operating Margin (%)", opm_min, opm_max),
                _rng("Net Margin (%)", npm_min, npm_max),
                _rng("Dividend Yield (%)", div_min, div_max),
            ]
            for c in conds: mask &= c.values
            kept = metrics_df.loc[mask, "symbol"].astype(str).str.upper().tolist()
            filtered_tickers = [t for t in filtered_tickers if t in set(kept)]

    st.caption(f"Metrics filters kept **{len(filtered_tickers)}** of {len(screen_df)} tickers.")

    # Save step 1 results to state for step 2
    st.session_state.step1 = {
        "screen_df": screen_df,
        "filtered_tickers": filtered_tickers,
        "sector": sector,
        "desc_prompt": desc_prompt,
        "include_transcripts": include_transcripts,
        "mode": mode, "year": int(year), "quarter": int(quarter),
        "transcript_instruction": transcript_instruction,
    }

# If Step 1 already done (this run or prior), show Industry picker and Step 2
if "step1" in st.session_state:
    s1 = st.session_state.step1
    screen_df = s1["screen_df"]
    filtered_tickers = s1["filtered_tickers"]

    # Build industries from post-metrics ticker set
    base = screen_df[screen_df["ticker"].isin(filtered_tickers)]
    industries = ["All"] + sorted([x for x in base["industry"].dropna().unique().tolist() if x])
    st.subheader("Industry filter")
    chosen_industry = st.selectbox("Pick industry (applies before profiles/earnings):", industries, index=0, key="industry_pick")

    # Compute final ticker list after industry
    if chosen_industry != "All":
        final_tickers = base[base["industry"] == chosen_industry]["ticker"].tolist()
    else:
        final_tickers = filtered_tickers[:]

    st.caption(f"Industry filter kept **{len(final_tickers)}** tickers.")
    if not final_tickers:
        st.warning("No tickers remain after industry filter.")
    # STEP 2 button — profiles + summaries + (optional) sentiment
    run_step2 = st.button("Step 2 — Fetch profiles + earnings summaries", disabled=(len(final_tickers)==0))

    # ============================ FLOW STEP 2 =================================
    if run_step2:
        if s1["include_transcripts"] and not OPENAI_API_KEY:
            st.warning("OPENAI_API_KEY not found — summaries/sentiment will use fallbacks.")

        # Profiles + description
        with st.spinner(f"Fetching profiles for {len(final_tickers)} companies..."):
            prof = fetch_profiles_v3_bulk(final_tickers)
            if prof.empty:
                st.warning("No profiles found."); st.stop()

        with st.spinner("Summarizing company descriptions..."):
            prof["summary_3_lines"] = prof.apply(
                lambda r: summarize_description(
                    r.get("description"), r.get("companyName"), r.get("ticker"),
                    r.get("sector"), r.get("industry"),
                    user_prompt=s1["desc_prompt"].strip() if s1["desc_prompt"] else None
                ),
                axis=1
            )

        # Agentic earnings
        transcripts = {}
        intent = parse_transcript_instruction(s1["transcript_instruction"] or "")

        if s1["include_transcripts"]:
            with st.spinner("Fetching & summarizing earnings transcripts..."):
                for t in prof["ticker"]:
                    meta_t = fetch_specific_transcript(t, s1["year"], s1["quarter"]) if s1["mode"].startswith("Specific") else fetch_latest_transcript(t, 1)
                    if not meta_t or not meta_t.get("content"):
                        transcripts[t] = {"summary":"No transcript available.","meta":{}, "sentiment":{}, "qoq": None}
                        continue

                    qoq = fetch_qoq_fundamentals(t) if intent["want_qoq"] else None

                    if intent["sentiment_only"]:
                        summ = ""
                        sent = analyze_sentiment_gpt(meta_t["content"])
                    else:
                        summ = summarize_transcript(
                            meta_t["content"],
                            user_instruction=s1["transcript_instruction"],
                            style=intent["style"],
                            count=intent["count"],
                            focus=intent["focus"],
                            exclude=intent["exclude"],
                            qoq_hint=(qoq_hint_text(qoq) if qoq else None),
                        )
                        sent = analyze_sentiment_gpt(summ) if intent["include_sentiment"] else {}

                    transcripts[t] = {"summary": summ or ("" if intent["sentiment_only"] else "No summary requested."),
                                      "meta": meta_t, "sentiment": sent, "qoq": qoq}

        # Cards
        st.subheader("Results")
        prof = prof.sort_values("ticker").reset_index(drop=True)
        cols = st.columns(2)

        def _sent_bar(color: str, val: float) -> str:
            pct = int((max(-1, min(1, val)) + 1) * 50)
            return f'''
            <div style="height:8px;background:#ececf3;border-radius:6px;position:relative;overflow:hidden;">
              <div style="height:8px;width:{pct}%;background:{color};"></div>
            </div>'''

        export_rows = []
        for i, r in prof.iterrows():
            tkr = r["ticker"]
            with cols[i % 2]:
                st.markdown(f"### {tkr} — {r.get('companyName') or ''}")
                st.markdown(
                    f'<div class="cardbox"><div class="subtle">Company overview</div>'
                    f'<div class="scrollarea">{r["summary_3_lines"]}</div></div>',
                    unsafe_allow_html=True
                )
                if s1["include_transcripts"] and "transcripts" in locals():
                    tr = transcripts.get(tkr)
                    meta_t = (tr or {}).get("meta", {})
                    hdr = f"Earnings call ({meta_t.get('year','?')} Q{meta_t.get('quarter','?')})" if meta_t else "Earnings call"

                    qoq_html = qoq_box_html((tr or {}).get("qoq"))
                    body_html = qoq_html if qoq_html else ""

                    sent = (tr or {}).get("sentiment") or {}
                    if sent:
                        label = (sent.get("label") or "Neutral").title()
                        pol   = float(sent.get("overall_polarity", 0.0))
                        conf  = float(sent.get("confidence", 0.0))
                        subs  = sent.get("subscores") or {}
                        drivers = sent.get("drivers") or []
                        color = {"Bullish":"#10b981","Neutral":"#6b7280","Bearish":"#ef4444"}.get(label, "#6b7280")
                        sent_html = f"""
                        <div style="display:flex;gap:8px;align-items:center;margin:6px 0 10px 0;">
                          <span style="padding:3px 10px;border-radius:999px;background:{color};color:white;font-weight:600;font-size:12px;">
                            {label}
                          </span>
                          <span style="font-size:12px;color:#374151;">Polarity {pol:+.2f} · Conf {conf:.2f}</span>
                        </div>
                        <div style="display:grid;grid-template-columns:90px 1fr;gap:6px;align-items:center;font-size:12px;color:#4b5563;margin-bottom:6px;">
                          <div>Results</div><div>{_sent_bar(color, float(subs.get('results',0.0)))}</div>
                          <div>Guidance</div><div>{_sent_bar(color, float(subs.get('guidance',0.0)))}</div>
                          <div>Demand</div><div>{_sent_bar(color, float(subs.get('demand',0.0)))}</div>
                          <div>Margins</div><div>{_sent_bar(color, float(subs.get('margins',0.0)))}</div>
                        </div>
                        """
                        if drivers:
                            sent_html += "<div style='font-size:12px;color:#374151;margin-bottom:8px;'><b>Drivers:</b> " + "; ".join(drivers[:4]) + "</div>"
                        body_html += sent_html

                    body_html += f'<div class="scrollarea">{(tr or {}).get("summary","No transcript available.")}</div>'

                    st.markdown(
                        f'<div class="cardbox"><div class="subtle">{hdr}</div>{body_html}</div>',
                        unsafe_allow_html=True
                    )

                    qoq = (tr or {}).get("qoq") or {}
                    export_rows.append({
                        "ticker": tkr,
                        "companyName": r.get("companyName"),
                        "summary_3_lines": r.get("summary_3_lines"),
                        "transcript_year": meta_t.get("year"),
                        "transcript_quarter": meta_t.get("quarter"),
                        "transcript_summary": (tr or {}).get("summary"),
                        "sentiment_label": (sent.get("label") if sent else None),
                        "sentiment_polarity": (sent.get("overall_polarity") if sent else None),
                        "sentiment_confidence": (sent.get("confidence") if sent else None),
                        "instruction_used": s1["transcript_instruction"],
                        "qoq_revenue_now": qoq.get("revenue_now"),
                        "qoq_revenue_prev": qoq.get("revenue_prev"),
                        "qoq_revenue_pct": qoq.get("revenue_qoq"),
                        "qoq_netincome_now": qoq.get("netincome_now"),
                        "qoq_netincome_prev": qoq.get("netincome_prev"),
                        "qoq_netincome_pct": qoq.get("netincome_qoq"),
                        "qoq_fcf_now": qoq.get("fcf_now"),
                        "qoq_fcf_prev": qoq.get("fcf_prev"),
                        "qoq_fcf_pct": qoq.get("fcf_qoq"),
                        "qoq_label_now": qoq.get("label_now"),
                        "qoq_label_prev": qoq.get("label_prev"),
                    })
                else:
                    export_rows.append({
                        "ticker": tkr,
                        "companyName": r.get("companyName"),
                        "summary_3_lines": r.get("summary_3_lines"),
                        "instruction_used": s1["transcript_instruction"],
                    })

        export_df = pd.DataFrame(export_rows)
        st.download_button(
            "Download CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="analysis_with_metrics_industry_and_agentic_earnings.csv",
            mime="text/csv"
        )

# Initial hint
if "step1" not in st.session_state:
    st.info("Set screener + (optional) metrics in the sidebar, then click **Step 1 — Run screener + metrics**. After that, pick an **Industry** and click **Step 2**.")
