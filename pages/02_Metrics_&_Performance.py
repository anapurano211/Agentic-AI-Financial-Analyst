# pages/02_Metrics_&_Performance.py
# -------------------------------------------------------------------
# Metrics & Performance with Agentic Competitive Analysis
# - Pulls tickers from main page (st.session_state.step1) or manual
# - Fetches advanced metrics & optional price performance
# - Persists the fetched table in session_state so the Agentic panel
#   works after button clicks/reruns
# -------------------------------------------------------------------

import os
import time
import math
import json
import numpy as np
import pandas as pd
import requests
import streamlit as st
from datetime import datetime, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Optional GPT
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ============================== Config =========================================
FMP_API_KEY    = os.getenv("FMP_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAI) else None

st.set_page_config(page_title="Metrics & Performance", layout="wide")

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

# ============================ Math utils =======================================
def _safe_pct(num, den):
    """
    Vectorized safe division for Series/arrays/scalars.
    Returns num/den where den != 0 and both are not NA, else NaN.
    """
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

# =========================== Data fetchers =====================================
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
            "price_profile": (payload or {}).get("price"),
            "beta": (payload or {}).get("beta"),
        })
        time.sleep(0.03)
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
            if "date" in df.columns:
                df["as_of_rsi"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.date
            frames.append(df[["symbol","as_of_rsi","rsi"]])
        time.sleep(0.04)
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

# =========================== Merge metrics per symbol ==========================
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
    for s in symbols:
        prof_single = fetch_profiles_v3_bulk([s])
        if not prof_single.empty:
            prof_single = prof_single.rename(columns={"ticker":"symbol"})[["symbol","price_profile","beta"]]
        profile_blocks[s] = prof_single
        quote_blocks[s]   = _fetch_quote(s)
        ratios_blocks[s]  = _fetch_ratios_ttm(s)
        km_blocks[s]      = _fetch_key_metrics_ttm(s)
        time.sleep(0.02)

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
    for s in df_symbols:
        perfs.append(_fetch_price_performance(s))
        time.sleep(0.02)
    perf_df = pd.DataFrame(perfs)
    return perf_df

# ======================== Agentic Competitive Analysis =========================
def _grab_number(txt, *keys):
    import re
    for k in keys:
        m = re.search(rf'(\d+)\s*(?:-| )?\s*{k}', txt)
        if m: return int(m.group(1))
    return None

def parse_agentic_instruction(raw: str):
    import re
    txt = (raw or "").strip().lower()
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
    return {"style": style, "count": max(1, count), "focus": focus, "exclude": exclude}

def _compact_metrics_table(df: pd.DataFrame, selected: list[str], max_rows: int = 20) -> list[dict]:
    if df is None or df.empty:
        return []
    cols_keep = [c for c in [
        "symbol","companyName","sector","industry",
        "Price","Beta","RSI",
        "% From 52W High","% Above 50D MA","% Above 200D MA",
        "P/E (TTM)","Price to Sales (TTM)","Price to Book (TTM)","PEG (TTM)",
        "Operating Margin (%)","Net Margin (%)","ROE (%)","ROA (%)","ROIC (%)",
        "Dividend Yield (%)","Current Ratio (TTM)","Debt to Equity (TTM)","EV to Sales (TTM)",
        "Perf 1M (%)","Perf 3M (%)","Perf 6M (%)","Perf 1Y (%)"
    ] if c in df.columns]
    sub = df[df["symbol"].isin(selected)][cols_keep].copy().head(max_rows)
    for c in sub.columns:
        if pd.api.types.is_numeric_dtype(sub[c]):
            sub[c] = sub[c].round(2)
    return sub.to_dict(orient="records")

def gpt_competitive_story(data_rows: list[dict], user_instruction: str, style: str, count: int, focus: list[str], exclude: list[str]) -> str:
    if not client or not OPENAI_API_KEY:
        # Fallback heuristic
        try:
            df = pd.DataFrame(data_rows)
            if df.empty:
                return "No data available for analysis."
            score = (
                df.get("ROIC (%)", 0).fillna(0) * 0.35 +
                df.get("Operating Margin (%)", 0).fillna(0) * 0.25 +
                df.get("ROE (%)", 0).fillna(0) * 0.20 -
                df.get("Debt to Equity (TTM)", 0).fillna(0) * 0.10 +
                df.get("Perf 1Y (%)", 0).fillna(0) * 0.10
            )
            df["_score"] = score
            df = df.sort_values("_score", ascending=False)
            top = df.head(min(5, len(df)))
            lines = []
            for _, r in top.iterrows():
                lines.append(
                    f"- {r.get('symbol')} — ROIC {r.get('ROIC (%)', 'NA')}%, OPM {r.get('Operating Margin (%)','NA')}%, "
                    f"D/E {r.get('Debt to Equity (TTM)','NA')}, P/E {r.get('P/E (TTM)','NA')}, Perf 1Y {r.get('Perf 1Y (%)','NA')}%"
                )
            return "Heuristic view (no GPT key found):\n" + "\n".join(lines)
        except Exception:
            return "Heuristic view unavailable (missing data)."

    focus_txt = f" Focus ONLY on: {', '.join(focus)}." if focus else ""
    excl_txt  = f" Avoid: {', '.join(exclude)}." if exclude else ""
    shape = (f"Write EXACTLY {count} concise bullet points." if style == "bullets"
             else f"Write EXACTLY {count} short lines (no bullets). Each line ≤ 180 characters.")
    guidance = (
        "Use ONLY the JSON metrics provided; DO NOT invent numbers. "
        "Cite companies by symbol. Highlight sources of advantage (e.g., high ROIC with strong margins and low leverage), "
        "flag red flags (e.g., high P/E with weak margins or high D/E), and compare peers directly. "
        "Prefer clear, decision-ready phrasing over boilerplate."
    )
    prompt = f"""
You are an equity analyst. {shape}{focus_txt}{excl_txt}
{guidance}

User preference: {user_instruction or ''}

DATA (JSON rows, one per company):
{json.dumps(data_rows, ensure_ascii=False, indent=2)}
"""
    try:
        r = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            temperature=0.25,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"GPT analysis failed: {e}"

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

    # Reorder columns
    front_cols = ["symbol","companyName","sector","industry","As Of","Price","Beta","RSI",
                  "% From 52W High","% Above 50D MA","% Above 200D MA",
                  "P/E (TTM)","Price to Sales (TTM)","Price to Book (TTM)","PEG (TTM)",
                  "Operating Margin (%)","Net Margin (%)","ROE (%)","ROA (%)","ROIC (%)",
                  "Dividend Yield (%)","Current Ratio (TTM)","Debt to Equity (TTM)",
                  "EV to Sales (TTM)","52W High","52W Low","50D MA","200D MA"]
    perf_cols = ["Perf 1M (%)","Perf 3M (%)","Perf 6M (%)","Perf 1Y (%)"]
    cols_order = [c for c in front_cols if c in df_metrics.columns] + \
                 [c for c in perf_cols if c in df_metrics.columns] + \
                 [c for c in df_metrics.columns if c not in front_cols + perf_cols]
    df_metrics = df_metrics[[c for c in cols_order if c in df_metrics.columns]]

    # Persist for later reruns (so agentic panel works)
    st.session_state.metrics_page = {
        "df": df_metrics,
        "syms": df_metrics["symbol"].tolist(),
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
    st.subheader("🤖 Agentic Competitive Analysis")

    st.caption(
        "Ask an analyst-style question using the table’s metrics. "
        "Examples: “8 bullets, focus on ROIC/margins/leverage; call out red flags and relative valuation.” "
        "Or: “5 lines comparing pricing power vs. cost efficiency; avoid commentary on RSI.”"
    )

    all_syms = page_state.get("syms", [])
    default_sel = all_syms[: min(12, len(all_syms))]
    with st.form("agentic_form"):
        selected_syms = st.multiselect("Companies to analyze", options=all_syms, default=default_sel, key="agentic_syms")
        max_rows_for_llm = st.slider("Max companies to send to the analyst", 5, 30, min(12, len(selected_syms) if selected_syms else 12), 1, key="agentic_max")
        user_instruction = st.text_area(
            "Your instruction",
            value="In 6–8 bullets, compare competitive advantages and risks across these companies. "
                  "Focus on ROIC, operating/net margins, leverage (D/E), and relative valuation (P/E, EV/S). "
                  "Call out outliers and any red flags. End with one long/short pair-trade idea.",
            height=110,
            key="agentic_text",
        )
        submitted = st.form_submit_button("Run agentic analysis")

    if submitted:
        if not selected_syms:
            st.warning("Pick at least one company to analyze.")
        else:
            intent = parse_agentic_instruction(user_instruction)
            payload = _compact_metrics_table(df_metrics, selected_syms, max_rows=max_rows_for_llm)
            with st.spinner("Analyzing competitive advantages..."):
                out = gpt_competitive_story(
                    payload,
                    user_instruction=user_instruction,
                    style=intent["style"],
                    count=intent["count"],
                    focus=intent["focus"],
                    exclude=intent["exclude"],
                )
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
