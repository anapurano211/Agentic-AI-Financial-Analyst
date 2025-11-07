# pages/03_Portfolio_Optimizer_&_Backtests.py
# -------------------------------------------------------------------
# Efficient Frontier Sandbox — Crisis Roles + GPT Ideas + K-means
# - GPT idea generator for ETFs / large-cap stocks based on user needs
# - 5-year risk/return analysis of suggested funds from Yahoo Finance
# - K-means risk buckets on tickers (defensive / core / growth / speculative)
# - K-means "crisis fingerprint" clusters for tickers:
#       Crisis-resilient / defensive
#       Mixed / moderate
#       Crisis-vulnerable / pro-cyclical
# - User picks tickers and period (manual or crisis preset)
# - Simulate random long-only portfolios & plot efficient frontier
# - Highlight max-Sharpe and min-vol portfolios
# - User-defined test portfolio (equal-weight or custom)
# -------------------------------------------------------------------

import os
import re
import json
import datetime as dt

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.cluster import KMeans  # ML for risk & crisis roles

# Optional GPT (for idea generator + risk commentary)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# -------------------------------------------------------------------
# Keys / client  (MODEL HARD-LOCKED TO gpt-4o-mini TO KEEP COSTS LOW)
# -------------------------------------------------------------------
def get_key(name: str, default: str = "") -> str:
    """Get a key from env vars or .streamlit/secrets."""
    val = os.getenv(name)
    if val:
        return val
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


FMP_API_KEY    = os.getenv("FMP_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# 🔒 Force the cheap model on this page regardless of env:
OPENAI_MODEL   = "gpt-4o-mini"

client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAI) else None

# soft caps to keep each call inexpensive
MAX_TOKENS_IDEA = 400
MAX_TOKENS_RISK = 450

TRADING_DAYS = 252
INDEX_PROXY  = "SPY"  # used for beta in risk table

# Crisis windows (approximate, tweak if you want)
CRISIS_WINDOWS = [
    ("GFC 2008–2009",        dt.date(2007, 10, 1), dt.date(2009, 3, 9)),
    ("US/EU downgrade 2011", dt.date(2011, 7, 1),  dt.date(2011, 10, 3)),
    ("China/oil 2015–2016",  dt.date(2015, 6, 1),  dt.date(2016, 2, 11)),
    ("COVID crash 2020",     dt.date(2020, 2, 1),  dt.date(2020, 3, 23)),
    ("Rates/inflation 2022", dt.date(2022, 1, 1),  dt.date(2022, 10, 31)),
]

# -------------------------------------------------------------------
# Helper: parse tickers from GPT markdown table
# -------------------------------------------------------------------
def _extract_tickers_from_markdown(md: str):
    """
    Very simple parser for a markdown table produced by GPT.
    Assumes Ticker is the first column.
    """
    if not md:
        return []

    tickers = []
    for line in md.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if line.startswith("| Ticker"):
            continue

        cleaned = line.replace("|", "").strip()
        if cleaned and set(cleaned) == {"-"}:
            continue

        parts = [c.strip() for c in line.split("|") if c.strip()]
        if not parts:
            continue
        t = parts[0].upper()
        if 1 <= len(t) <= 6 and t.replace(".", "").isalnum():
            tickers.append(t)

    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# -------------------------------------------------------------------
# Price / risk helpers
# -------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_prices(tickers_list, start, end):
    """Download adjusted daily close prices for a given date window."""
    if not tickers_list:
        return pd.DataFrame()
    data = yf.download(
        tickers_list,
        start=start,
        end=end + dt.timedelta(days=1),  # yfinance end is exclusive
        auto_adjust=True,
        progress=False,
    )
    if data.empty or "Close" not in data:
        return pd.DataFrame()

    px = data["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px = px.dropna(how="all").ffill().dropna(how="all")
    return px


@st.cache_data(ttl=900, show_spinner=False)
def fetch_20y_prices(ticker_list):
    """Download ~20 years of adjusted close prices for a list of tickers."""
    if not ticker_list:
        return pd.DataFrame()
    data = yf.download(
        ticker_list,
        period="20y",
        auto_adjust=True,
        progress=False,
    )
    if data.empty or "Close" not in data:
        return pd.DataFrame()
    px = data["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px = px.dropna(how="all").ffill().dropna(how="all")
    return px


@st.cache_data(ttl=900, show_spinner=False)
def fetch_5y_prices(ticker_list):
    """Download ~5 years of adjusted close prices for a list of tickers."""
    if not ticker_list:
        return pd.DataFrame()
    data = yf.download(
        ticker_list,
        period="5y",
        auto_adjust=True,
        progress=False,
    )
    if data.empty or "Close" not in data:
        return pd.DataFrame()
    px = data["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px = px.dropna(how="all").ffill().dropna(how="all")
    return px


def _max_drawdown(returns_series: pd.Series) -> float:
    """Max drawdown from a daily return series (negative number)."""
    equity = (1 + returns_series).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return dd.min() if not dd.empty else np.nan


def build_risk_table_from_prices(prices: pd.DataFrame, index_proxy: str = INDEX_PROXY) -> pd.DataFrame:
    """
    Given daily prices, build a risk/return stats table per ticker:
      - AnnReturn, AnnVol, Sharpe, MaxDrawdown
      - Beta vs index_proxy
      - DownsideVol (annualized std of negative returns)
    """
    rets = prices.pct_change().dropna()
    if rets.empty:
        return pd.DataFrame()

    ann_ret = (1 + rets).prod() ** (TRADING_DAYS / len(rets)) - 1
    ann_vol = rets.std() * np.sqrt(TRADING_DAYS)
    sharpe = ann_ret / ann_vol.replace(0, np.nan)

    # max drawdown per asset
    mdd_list = []
    for col in rets.columns:
        mdd_list.append(_max_drawdown(rets[col]))

    # proxy for beta / downside vol
    try:
        proxy_data = yf.download(
            index_proxy,
            start=rets.index.min(),
            end=rets.index.max() + dt.timedelta(days=1),
            auto_adjust=True,
            progress=False,
        )
        proxy_px = proxy_data["Close"].dropna()
        proxy_rets = proxy_px.pct_change().reindex(rets.index).dropna()
    except Exception:
        proxy_rets = None

    betas = []
    downside_vols = []

    for col in rets.columns:
        r = rets[col].dropna()
        if proxy_rets is None or proxy_rets.empty:
            betas.append(np.nan)
            downside_vols.append(np.nan)
            continue

        aligned = pd.concat([r, proxy_rets], axis=1, join="inner").dropna()
        if aligned.shape[0] < 30:
            betas.append(np.nan)
            downside_vols.append(np.nan)
            continue

        rp = aligned.iloc[:, 0]
        rm = aligned.iloc[:, 1]

        var_m = np.var(rm)
        cov_pm = np.cov(rp, rm)[0, 1]
        beta = cov_pm / var_m if var_m > 0 else np.nan
        betas.append(beta)

        neg = rp[rp < 0]
        if len(neg) > 0:
            downside_vols.append(neg.std() * np.sqrt(TRADING_DAYS))
        else:
            downside_vols.append(np.nan)

    df_stats = pd.DataFrame(
        {
            "Ticker": rets.columns,
            "AnnReturn": ann_ret.values,
            "AnnVol": ann_vol.values,
            "Sharpe": sharpe.values,
            "MaxDrawdown": mdd_list,
            "Beta": betas,
            "DownsideVol": downside_vols,
        }
    )
    return df_stats


# --------------------- K-means risk bucketing on tickers ---------------------
def kmeans_risk_buckets(stats_df: pd.DataFrame, n_clusters: int = 3):
    """
    Cluster tickers into risk buckets using K-means on:
    [AnnVol, MaxDrawdown, Beta, DownsideVol].

    Returns a copy of stats_df with columns:
      - Cluster (int)
      - BucketLabel (str)
    or (None) if clustering isn't possible.
    """
    if stats_df is None or stats_df.empty:
        return None

    feat_cols = ["AnnVol", "MaxDrawdown", "Beta", "DownsideVol"]
    df = stats_df.copy().reset_index(drop=True)

    df_feat = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    mask_valid = df_feat.notna().all(axis=1)
    if mask_valid.sum() < 2:
        return None

    df_feat = df_feat[mask_valid]
    X = df_feat.values

    # Standardize
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma

    k = min(n_clusters, df_feat.shape[0])
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(Xs)

    # Attach back to df
    df["Cluster"] = np.nan
    df.loc[mask_valid, "Cluster"] = labels.astype(int)

    # Label buckets by beta & volatility
    summary = (
        df[df["Cluster"].notna()]
        .groupby("Cluster")
        .agg(
            MeanVol=("AnnVol", "mean"),
            MeanBeta=("Beta", "mean"),
            MeanDD=("MaxDrawdown", "mean"),
            Count=("AnnVol", "size"),
        )
        .reset_index()
    )
    summary["Cluster"] = summary["Cluster"].astype(int)

    summary_sorted = summary.sort_values(["MeanBeta", "MeanVol"]).reset_index(drop=True)
    cluster_order = list(summary_sorted["Cluster"].astype(int))

    bucket_names = [
        "Defensive / low beta",
        "Core / market-like",
        "Growth / high beta",
        "Speculative / very high risk",
    ]
    label_map = {}
    for i, cid in enumerate(cluster_order):
        label_map[cid] = bucket_names[min(i, len(bucket_names) - 1)]

    df["BucketLabel"] = df["Cluster"].map(
        lambda x: label_map.get(int(x), "Unclustered") if pd.notna(x) else "Unclustered"
    )

    return df


# -------------------------------------------------------------------
# Portfolio metrics: ANNUALIZED, path-correct using log returns
# -------------------------------------------------------------------
def calc_portfolio_perf(weights, returns_df, risk_free_annual):
    """
    Compute annualized return, vol, Sharpe for a single portfolio.

    - weights: 1D array-like (len = n_assets), sums to 1
    - returns_df: DataFrame of daily SIMPLE returns (T x n_assets)
    - risk_free_annual: annual risk-free rate (e.g. 0.02 for 2%)

    Uses the actual path of portfolio returns via log(1+r).
    """
    R = returns_df.values  # T x N
    w = np.asarray(weights, dtype=float).reshape(-1, 1)  # N x 1
    port_ret = (R @ w).ravel()  # T

    port_ret = pd.Series(port_ret).replace([np.inf, -np.inf], np.nan).dropna().values
    if port_ret.size == 0:
        return np.nan, np.nan, np.nan

    # Path-correct annualized return
    log_r = np.log1p(port_ret)
    g = log_r.mean()                     # mean daily log-return
    ann_ret = np.expm1(g * TRADING_DAYS) # annualized simple return

    # Vol and Sharpe
    ann_vol = port_ret.std(ddof=0) * np.sqrt(TRADING_DAYS)
    sharpe = (ann_ret - risk_free_annual) / ann_vol if ann_vol > 0 else np.nan

    return ann_ret, ann_vol, sharpe


def simulate_random_portfolios(num_portfolios, returns_df, risk_free_annual, ticker_list):
    """
    Simulate random long-only portfolios using the actual daily return path.

    returns_df: DataFrame of daily SIMPLE returns (T x N)
    """
    R = returns_df.values  # T x N
    T, N = R.shape

    # Random weights: N x P
    W = np.random.random((N, num_portfolios))
    W /= W.sum(axis=0, keepdims=True)

    # Portfolio daily returns: T x P
    Rp = R @ W
    Rp = np.where(np.isfinite(Rp), Rp, np.nan)

    # Path-correct annualized returns via log(1+r)
    logRp = np.log1p(Rp)
    g = np.nanmean(logRp, axis=0)               # mean daily log-return per portfolio
    ann_ret = np.expm1(g * TRADING_DAYS)        # annualized simple return

    # Annualized vol & Sharpe
    ann_vol = np.nanstd(Rp, axis=0) * np.sqrt(TRADING_DAYS)
    sharpe = np.where(ann_vol > 0, (ann_ret - risk_free_annual) / ann_vol, np.nan)

    results = np.vstack([ann_ret, ann_vol, sharpe, W])
    cols = ["ret", "stdev", "sharpe"] + list(ticker_list)
    results_df = pd.DataFrame(results.T, columns=cols)
    return results_df


def parse_custom_weights(text, n_assets):
    """
    Parse a string of weights (decimals or percents) into a normalized numpy array.
    Returns (weights, error_message). If error_message is not None, weights is None.
    """
    if not text.strip():
        return None, "Please enter at least one weight."

    parts = re.split(r"[,\s]+", text.strip())
    try:
        has_percent = any("%" in p for p in parts)
        vals = []
        for p in parts:
            if not p:
                continue
            p_clean = p.replace("%", "")
            v = float(p_clean)
            vals.append(v)
        arr = np.array(vals, dtype=float)
        if has_percent:
            arr = arr / 100.0
    except ValueError:
        return None, "Could not parse some of the weights. Use numbers like 0.2 or 20%."

    if arr.size != n_assets:
        return None, f"Expected {n_assets} weights (one per ticker), but got {arr.size}."

    if arr.sum() <= 0:
        return None, "Sum of weights must be positive."

    arr = arr / arr.sum()
    return arr, None


# -------------------------------------------------------------------
# Crisis fingerprint features per ticker (long history)
# -------------------------------------------------------------------
def build_crisis_fingerprint(prices_long: pd.DataFrame):
    """
    From ~20Y daily prices, build per-ticker "crisis fingerprints":

    For each ticker:
      - For each CRISIS_WINDOW: cumulative total return in that window
      - LongTermAnnRet: annualized return over full available history
      - LongTermAnnVol: annualized vol over full history
      - LongTermMaxDD: max drawdown over full history

    Returns:
      df with columns:
        Ticker, LongTermAnnRet, LongTermAnnVol, LongTermMaxDD,
        Ret_<crisis_name> for each crisis
    """
    if prices_long.empty:
        return pd.DataFrame()

    rets = prices_long.pct_change().dropna(how="all")
    if rets.empty:
        return pd.DataFrame()

    tickers = list(rets.columns)
    rows = []
    total_days = rets.shape[0]

    for t in tickers:
        r = rets[t].dropna()
        if r.empty:
            continue

        # Long-term stats
        ann_ret = (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1
        ann_vol = r.std(ddof=0) * np.sqrt(TRADING_DAYS)
        mdd = _max_drawdown(r)

        row = {
            "Ticker": t,
            "LongTermAnnRet": float(ann_ret),
            "LongTermAnnVol": float(ann_vol),
            "LongTermMaxDD": float(mdd),
            "DataDays": int(len(r)),
            "TotalDaysFullPanel": int(total_days),
        }

        # Crisis window returns
        for crisis_name, start_d, end_d in CRISIS_WINDOWS:
            key = f"Ret_{crisis_name}"
            sub = r.loc[(r.index.date >= start_d) & (r.index.date <= end_d)]
            if sub.empty or len(sub) < 10:
                row[key] = np.nan
            else:
                row[key] = float((1 + sub).prod() - 1)

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


def kmeans_crisis_clusters(crisis_df: pd.DataFrame, n_clusters: int = 3):
    """
    K-means on crisis fingerprints.

    Features:
      - all Ret_<crisis> columns
      - LongTermAnnRet
      - LongTermAnnVol
      - LongTermMaxDD
      - AvgCrisisRet (mean of crisis-window returns)

    We then label clusters by AvgCrisisRet:
      lowest  -> Crisis-vulnerable / pro-cyclical
      middle  -> Mixed / moderate
      highest -> Crisis-resilient / defensive

    Returns:
      crisis_df copy with:
        - CrisisCluster (int)
        - CrisisRole (str)
        - AvgCrisisRet (float)
    """
    if crisis_df is None or crisis_df.empty:
        return None

    df = crisis_df.copy()
    crisis_cols = [c for c in df.columns if c.startswith("Ret_")]

    if not crisis_cols:
        return None

    # Avg crisis performance per ticker
    df["AvgCrisisRet"] = df[crisis_cols].mean(axis=1, skipna=True)

    feat_cols = crisis_cols + ["LongTermAnnRet", "LongTermAnnVol", "LongTermMaxDD", "AvgCrisisRet"]
    df_feat = df[feat_cols].replace([np.inf, -np.inf], np.nan)

    # require fully non-null feature rows
    mask_valid = df_feat.notna().all(axis=1)
    if mask_valid.sum() < 2:
        return None

    X = df_feat.loc[mask_valid, feat_cols].values

    # Standardize
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma

    k = min(n_clusters, X.shape[0])
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(Xs)

    df["CrisisCluster"] = np.nan
    df.loc[mask_valid, "CrisisCluster"] = labels.astype(int)

    # Cluster-wise avg crisis return
    cluster_crisis_ret = (
        df[df["CrisisCluster"].notna()]
        .groupby("CrisisCluster")["AvgCrisisRet"]
        .mean()
        .reset_index()
    )
    cluster_crisis_ret["CrisisCluster"] = cluster_crisis_ret["CrisisCluster"].astype(int)

    sorted_by_crisis = cluster_crisis_ret.sort_values("AvgCrisisRet").reset_index(drop=True)

    label_map = {}
    for i, row in enumerate(sorted_by_crisis.itertuples(index=False)):
        cid = int(row.CrisisCluster)
        if i == 0:
            label = "Crisis-vulnerable / pro-cyclical"
        elif i == 1:
            label = "Mixed / moderate"
        else:
            label = "Crisis-resilient / defensive"
        label_map[cid] = label

    df["CrisisRole"] = df["CrisisCluster"].map(
        lambda x: label_map.get(int(x), "Unclustered") if pd.notna(x) else "Unclustered"
    )

    return df


# -------------------------------------------------------------------
# Layout
# -------------------------------------------------------------------
st.set_page_config(page_title="Efficient Frontier — Crisis Roles Sandbox", layout="wide")
st.title("📊 Efficient Frontier — Crisis Roles Sandbox")

st.markdown(
    """
Use this page to **experiment with portfolio construction**:

- Custom date range (any period you want)
- **2008–2009 Financial Crisis**
- **2020 COVID Crash**
- K-means risk buckets on tickers (defensive/core/growth/speculative)
- K-means **crisis fingerprint roles** per ticker:
  which names tend to get hit hardest vs. hold up better in big drawdowns.

We:
1. Pull daily prices for your chosen tickers from Yahoo Finance  
2. Simulate many random long-only portfolios  
3. Plot **annualized return vs. annualized volatility**, colored by **Sharpe ratio**  
4. Highlight the **max-Sharpe** and **min-vol** portfolios  
5. Let you overlay your own test portfolio (equal-weight or custom)  
6. For GPT-suggested funds, analyze ~5 years of data and explain & cluster their risk  
7. For your universe, cluster tickers by their **crisis fingerprints** and visualize the roles
"""
)

# ======================= GPT ETF/STOCK IDEAS ============================
st.markdown("### 🤖 GPT Idea Generator: ETF / Stock Suggestions")

st.caption(
    "Describe your goals (time horizon, risk tolerance, income vs. growth, sectors you like/avoid). "
    "GPT will suggest **liquid, plain-vanilla ETFs** (and optionally large-cap stocks) "
    "you can research and then plug into the simulation below."
)

col_g1, col_g2 = st.columns([2, 1])
with col_g1:
    user_needs = st.text_area(
        "Describe your investment needs",
        height=110,
        placeholder=(
            "Example: I’m a long-term investor (5–10 years), moderate risk, mostly US equities, "
            "diversified core exposure plus a small tilt to technology. Avoid leveraged or inverse ETFs."
        ),
        key="gpt_idea_prompt",
    )
with col_g2:
    instrument_focus = st.selectbox(
        "Instrument focus",
        ["ETFs only", "ETFs + large-cap stocks"],
        index=0,
        key="gpt_idea_instrument_focus",
    )
    n_ideas = st.slider(
        "How many ideas?",
        min_value=3,
        max_value=15,
        value=6,
        step=1,
        key="gpt_idea_count",
    )

col_gb1, col_gb2 = st.columns([1, 3])
with col_gb1:
    run_idea_btn = st.button("Ask GPT for ETF ideas", type="primary")
with col_gb2:
    st.write("⬅️ Use this first, then copy the tickers into the sidebar below.")

# Simple cache key for idea generator to avoid duplicate calls
def _idea_cache_key(needs: str, focus: str, n: int) -> str:
    return f"{needs.strip()}||{focus}||{n}"

if run_idea_btn:
    if not user_needs.strip():
        st.warning("Please describe your investment needs first.")
    else:
        if not client:
            st.info(
                "No OpenAI key found. Here’s a generic example of how ideas might look:\n\n"
                "| Ticker | Name                         | Instrument Type  | Risk Level | Key Rationale |\n"
                "|--------|------------------------------|------------------|-----------|---------------|\n"
                "| VTI    | Vanguard Total Stock Market  | Core Equity ETF  | Medium    | Broad US stock market exposure |\n"
                "| VXUS   | Vanguard Total Intl ex-US    | Intl Equity ETF  | Medium    | International diversification |\n"
                "| BND    | Vanguard Total Bond Market   | Bond ETF         | Low       | Diversifier / income |\n"
            )
        else:
            cache_key = _idea_cache_key(user_needs, instrument_focus, n_ideas)
            cached = st.session_state.get("idea_cache", {})
            if cache_key in cached:
                idea_out, suggested_tickers = cached[cache_key]
            else:
                with st.spinner("Asking GPT for ETF ideas..."):
                    instr_text = (
                        "ONLY ETFs (no individual stocks)."
                        if instrument_focus == "ETFs only"
                        else "Primarily ETFs, but you may also include a few large-cap, highly liquid individual stocks."
                    )

                    sys_prompt = (
                        "You are a portfolio construction assistant. "
                        "Recommend liquid, diversified, plain-vanilla instruments that a retail investor could research further. "
                        "Avoid leveraged, inverse, ultra, or exotic/illiquid funds. Only US-listed instruments. "
                        "Do NOT give personalized financial advice or guarantees—only generic, researchable ideas."
                    )

                    user_prompt = f"""
The user has described their investment needs as:

\"\"\"{user_needs.strip()}\"\"\"


Please recommend **{n_ideas}** ideas focusing on: {instr_text}

Requirements:
- US-listed only.
- Avoid leveraged/inverse/ultra ETFs and complex derivative strategies.
- Prefer large, liquid, low-cost funds where possible.
- Mix of broad "core" exposure and a few reasonable tilts (if it fits the description).
- Output MUST be a markdown table with **exactly these columns**:
  - `Ticker`
  - `Name`
  - `Instrument Type` (e.g., Core Equity ETF, Sector ETF, Bond ETF, Large-cap Stock)
  - `Risk Level` (Low / Medium / High)
  - `Key Rationale` (1–2 short sentences)
- Do NOT include any disclaimers in the table itself; keep those implicit.
"""
                    try:
                        r = client.chat.completions.create(
                            model=OPENAI_MODEL,
                            messages=[
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            temperature=0.3,
                            max_tokens=MAX_TOKENS_IDEA,
                        )
                        idea_out = (r.choices[0].message.content or "").strip()
                    except Exception as e:
                        idea_out = f"GPT idea generator call failed: {e}"

                suggested_tickers = _extract_tickers_from_markdown(idea_out)
                cached[cache_key] = (idea_out, suggested_tickers)
                st.session_state["idea_cache"] = cached

            st.markdown(
                '<div style="white-space:pre-wrap;line-height:1.5;background:#fff;border:1px solid #ececf3;'
                'border-radius:10px;padding:14px;font-variant-ligatures:none;">'
                + idea_out
                + "</div>",
                unsafe_allow_html=True,
            )

            if suggested_tickers:
                tick_str = ", ".join(suggested_tickers)
                st.markdown("**Tickers to research (copy/paste into sidebar):**")
                st.code(tick_str, language=None)
                st.session_state["gpt_suggested_tickers_str"] = tick_str
                st.session_state["gpt_suggested_ticker_list"] = suggested_tickers

# ==================== GPT risk explanation + K-means buckets (5Y) ====================
if "gpt_suggested_ticker_list" in st.session_state:
    st.markdown("### 📊 Risk & Return Snapshot for Suggested Funds")

    analyze_btn = st.button("Analyze risk/return over last 5 years")

    if analyze_btn:
        tickers_for_risk = st.session_state["gpt_suggested_ticker_list"]

        with st.spinner(f"Fetching 5Y price history for {len(tickers_for_risk)} tickers..."):
            px_5y = fetch_5y_prices(tickers_for_risk)

        if px_5y.empty:
            st.warning("Could not fetch enough 5-year history for those tickers.")
        else:
            requested_risk = set(tickers_for_risk)
            available_risk = set(px_5y.columns)
            missing_risk = sorted(requested_risk - available_risk)
            if missing_risk:
                st.warning(
                    "No 5-year price history for: "
                    + ", ".join(missing_risk)
                    + ". They were dropped from the risk table."
                )

            stats_df = build_risk_table_from_prices(px_5y)

            if stats_df.empty:
                st.warning("Could not compute risk statistics.")
            else:
                view = stats_df.copy()
                view["AnnReturn"] = view["AnnReturn"].map(lambda x: f"{x*100:.2f}%")
                view["AnnVol"] = view["AnnVol"].map(lambda x: f"{x*100:.2f}%")
                view["Sharpe"] = view["Sharpe"].map(lambda x: f"{x:.2f}")
                view["MaxDrawdown"] = view["MaxDrawdown"].map(lambda x: f"{x*100:.2f}%")
                view["Beta"] = view["Beta"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "NA")
                view["DownsideVol"] = view["DownsideVol"].map(
                    lambda x: f"{x*100:.2f}%" if pd.notna(x) else "NA"
                )

                st.dataframe(
                    view[
                        [
                            "Ticker",
                            "AnnReturn",
                            "AnnVol",
                            "Sharpe",
                            "MaxDrawdown",
                            "Beta",
                            "DownsideVol",
                        ]
                    ],
                    use_container_width=True,
                    height=260,
                )

                st.markdown("#### 🧬 K-means Risk Buckets (5Y behaviour, beta-aware)")

                bucket_df = kmeans_risk_buckets(stats_df, n_clusters=3)
                if bucket_df is not None:
                    bucket_view = bucket_df.copy()
                    bucket_view["AnnReturn"] = bucket_view["AnnReturn"].map(lambda x: f"{x*100:.2f}%")
                    bucket_view["AnnVol"] = bucket_view["AnnVol"].map(lambda x: f"{x*100:.2f}%")
                    bucket_view["Sharpe"] = bucket_view["Sharpe"].map(lambda x: f"{x:.2f}")
                    bucket_view["MaxDrawdown"] = bucket_view["MaxDrawdown"].map(lambda x: f"{x*100:.2f}%")
                    bucket_view["Beta"] = bucket_view["Beta"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "NA")
                    bucket_view["DownsideVol"] = bucket_view["DownsideVol"].map(
                        lambda x: f"{x*100:.2f}%" if pd.notna(x) else "NA"
                    )

                    st.dataframe(
                        bucket_view[
                            [
                                "Ticker",
                                "BucketLabel",
                                "AnnReturn",
                                "AnnVol",
                                "Sharpe",
                                "MaxDrawdown",
                                "Beta",
                                "DownsideVol",
                            ]
                        ]
                        .sort_values("BucketLabel"),
                        use_container_width=True,
                        height=280,
                    )

                    valid_bucket = bucket_df[bucket_df["Cluster"].notna()].copy()
                    if not valid_bucket.empty:
                        fig_rb, ax_rb = plt.subplots(figsize=(7, 5))
                        sc_rb = ax_rb.scatter(
                            valid_bucket["AnnVol"],
                            valid_bucket["AnnReturn"],
                            c=valid_bucket["Cluster"],
                            cmap="viridis",
                            s=60,
                        )
                        for _, row in valid_bucket.iterrows():
                            ax_rb.annotate(
                                row["Ticker"],
                                (row["AnnVol"], row["AnnReturn"]),
                                textcoords="offset points",
                                xytext=(5, 3),
                                fontsize=8,
                            )
                        ax_rb.set_xlabel("Annualized Volatility")
                        ax_rb.set_ylabel("Annualized Return")
                        ax_rb.set_title("K-means Risk Buckets (5Y stats)")
                        ax_rb.grid(alpha=0.3)
                        cbar_rb = plt.colorbar(sc_rb, ax=ax_rb)
                        cbar_rb.set_label("Risk cluster ID")
                        st.pyplot(fig_rb)
                else:
                    st.info(
                        "Not enough clean 5-year data to run K-means risk buckets on these tickers."
                    )

                if client:
                    risk_table_compact = []
                    for r in stats_df.to_dict(orient="records"):
                        risk_table_compact.append(
                            {
                                "Ticker": r["Ticker"],
                                "AnnReturn": round(float(r["AnnReturn"]), 4),
                                "AnnVol": round(float(r["AnnVol"]), 4),
                                "Sharpe": round(float(r["Sharpe"]), 3)
                                if pd.notna(r["Sharpe"])
                                else None,
                                "MaxDrawdown": round(float(r["MaxDrawdown"]), 4),
                                "Beta": round(float(r["Beta"]), 3)
                                if pd.notna(r["Beta"])
                                else None,
                                "DownsideVol": round(float(r["DownsideVol"]), 4)
                                if pd.notna(r["DownsideVol"])
                                else None,
                            }
                        )

                    risk_context = {
                        "risk_table": risk_table_compact,
                        "description": user_needs.strip(),
                    }

                    rt_key = (
                        tuple(sorted(r["Ticker"] for r in risk_table_compact)),
                        user_needs.strip(),
                    )
                    risk_cache = st.session_state.get("risk_expl_cache", {})

                    if rt_key in risk_cache:
                        explanation = risk_cache[rt_key]
                    else:
                        prompt = f"""
You are a portfolio and risk analyst.

You are given a table of risk/return statistics (computed from about 5 years of daily data) for several ETFs/stocks.
Each row has:
- Ticker
- AnnReturn (annualized total return, decimal)
- AnnVol (annualized volatility, decimal)
- Sharpe (return / vol, risk-free assumed to be 0)
- MaxDrawdown (worst peak-to-trough drawdown, negative decimal)
- Beta (sensitivity to a broad market index)
- DownsideVol (annualized volatility of negative returns only)

The user originally described their needs as:

\"\"\"{user_needs.strip()}\"\"\"


CONTEXT (JSON risk table):
{json.dumps(risk_context, ensure_ascii=False)}

Please:
1. Briefly summarize which funds look more defensive vs. more aggressive (use beta, vol, drawdown).
2. Comment on which ones seem to be return leaders vs. diversifiers.
3. Explain in plain language what the max drawdowns and downside vols say about pain during bad markets.
4. Give 2–3 generic, non-personalized observations on how someone *might* combine different risk buckets (no advice or recommendations).

Keep it concise and easy to understand. Do NOT invent numbers; rely only on the JSON.
"""
                        try:
                            r2 = client.chat.completions.create(
                                model=OPENAI_MODEL,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.25,
                                max_tokens=MAX_TOKENS_RISK,
                            )
                            explanation = (r2.choices[0].message.content or "").strip()
                        except Exception as e:
                            explanation = f"GPT risk explanation failed: {e}"

                        risk_cache[rt_key] = explanation
                        st.session_state["risk_expl_cache"] = risk_cache

                    st.markdown(
                        '<div style="white-space:pre-wrap;line-height:1.5;background:#fff;'
                        'border:1px solid #ececf3;border-radius:10px;padding:14px;'
                        'font-variant-ligatures:none;">'
                        + explanation
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.info(
                        "No OpenAI key configured, so I can’t auto-explain this table, "
                        "but higher AnnReturn + reasonable Sharpe and smaller MaxDrawdown, "
                        "plus moderate Beta and DownsideVol, generally indicate a more attractive risk/return tradeoff."
                    )

st.markdown("---")

# -------------------------------------------------------------------
# Sidebar controls for simulation (frontier + crisis roles + test portfolio)
# -------------------------------------------------------------------
with st.sidebar:
    st.header("Universe & Period")

    default_ticker_str = st.session_state.get(
        "gpt_suggested_tickers_str",
        "VTI, VTV, MGK, JPM, BA, MSFT, PYPL, ARKK, LMT, MPW, CVX, GRPN, SWPPX, C, CRM",
    )

    tickers_text = st.text_area(
        "Tickers (comma/space/newline separated)",
        value=default_ticker_str,
        height=90,
    )

    tickers = sorted(
        {
            t.strip().upper()
            for t in (
                tickers_text.replace("\n", " ")
                .replace("\t", " ")
                .replace(";", ",")
                .replace("  ", " ")
                .replace(",", " ")
                .split(" ")
            )
            if t.strip()
        }
    )

    regime = st.selectbox(
        "Analysis period",
        [
            "Custom period",
            "2008–2009 Financial Crisis",
            "2020 COVID Crash",
        ],
        index=0,
    )

    today = dt.date.today()
    if regime == "Custom period":
        start_date = st.date_input("Start date", value=dt.date(2017, 1, 1))
        end_date = st.date_input("End date", value=min(today, dt.date(2025, 12, 31)))
    elif regime == "2008–2009 Financial Crisis":
        start_date = dt.date(2007, 7, 1)
        end_date = dt.date(2009, 6, 30)
        st.caption(f"Fixed window: **{start_date} → {end_date}** (Financial Crisis)")
    else:  # 2020 COVID Crash
        start_date = dt.date(2020, 2, 1)
        end_date = dt.date(2020, 12, 31)
        st.caption(f"Fixed window: **{start_date} → {end_date}** (COVID Crash)")

    st.markdown("---")
    st.header("Simulation Settings")

    num_portfolios = st.slider(
        "Number of random portfolios",
        min_value=1_000,
        max_value=100_000,
        value=20_000,
        step=1_000,
        help="More portfolios → smoother frontier but slower.",
    )

    rf = (
        st.number_input(
            "Risk-free rate (annual, %)",
            min_value=-2.0,
            max_value=10.0,
            value=0.0,
            step=0.25,
        )
        / 100.0
    )

    st.markdown("---")
    st.header("Test Portfolio")

    test_mode = st.selectbox(
        "Test portfolio weights",
        ["None", "Equal-weight (1/N)", "Custom weights"],
        index=0,
        help="This portfolio will be plotted on top of the random frontier.",
    )

    custom_weights_str = ""
    if test_mode == "Custom weights":
        custom_weights_str = st.text_input(
            "Weights (comma or space separated, same order as tickers). "
            "Can be decimals like 0.2 or percents like 20.",
            value="",
        )

    run_btn = st.button("Run Frontier Simulation", type="primary")

# -------------------------------------------------------------------
# Main frontier run + crisis roles
# -------------------------------------------------------------------
if run_btn:
    if not tickers:
        st.error("Please enter at least one valid ticker.")
        st.stop()

    if start_date is None or end_date is None or start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

    st.subheader("1️⃣ Price Data for Frontier")

    with st.spinner(f"Downloading price data for {len(tickers)} tickers..."):
        prices = fetch_prices(tickers, start_date, end_date)

    regime_desc = f"Period: {start_date} → {end_date} ({regime})"

    if prices.empty or prices.shape[1] < 2:
        st.error("No usable price data returned (or fewer than 2 tickers with data).")
        st.stop()

    requested = set(tickers)
    available = set(prices.columns)
    missing = sorted(requested - available)
    if missing:
        st.warning(
            "No usable price data in this period for: "
            + ", ".join(missing)
            + ". They were dropped from the analysis."
        )

    st.write(
        f"Using **{prices.shape[1]}** tickers with data from **{prices.index.min().date()}** "
        f"to **{prices.index.max().date()}**."
    )
    st.caption(regime_desc)
    st.dataframe(prices.tail().round(2), use_container_width=True, height=180)

    returns = prices.pct_change().dropna()
    if returns.empty:
        st.error("Not enough return data after cleaning to run the simulation.")
        st.stop()

    # ----------------- Simulated portfolios -----------------
    st.subheader("2️⃣ Simulated Portfolios (annualized, path-correct)")

    with st.spinner(f"Simulating {num_portfolios:,} random portfolios..."):
        results_frame = simulate_random_portfolios(
            num_portfolios, returns, rf, prices.columns
        )

    max_sharpe_port = results_frame.iloc[results_frame["sharpe"].idxmax()]
    min_vol_port = results_frame.iloc[results_frame["stdev"].idxmin()]

    # ----------------- Test portfolio -----------------
    test_weights = None
    test_label = None
    test_metrics = None

    if test_mode == "Equal-weight (1/N)":
        n = len(prices.columns)
        test_weights = np.ones(n) / n
        test_label = "Equal-weight (1/N)"
        test_metrics = calc_portfolio_perf(test_weights, returns, rf)

    elif test_mode == "Custom weights":
        test_weights, err = parse_custom_weights(custom_weights_str, len(prices.columns))
        if err:
            st.error(f"Custom weights error: {err}")
        else:
            test_label = "Custom portfolio"
            test_metrics = calc_portfolio_perf(test_weights, returns, rf)

    # ----------------- Frontier plot -----------------
    st.subheader("3️⃣ Efficient Frontier Scatter (annualized)")

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(
        results_frame["stdev"],
        results_frame["ret"],
        c=results_frame["sharpe"],
        cmap="RdYlBu",
        alpha=0.7,
        s=10,
    )
    ax.set_xlabel("Annualized standard deviation")
    ax.set_ylabel("Annualized return")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Sharpe Ratio (annualized)")

    ax.scatter(
        max_sharpe_port["stdev"],
        max_sharpe_port["ret"],
        marker=(5, 1, 0),
        color="red",
        s=200,
        label="Max Sharpe",
    )
    ax.scatter(
        min_vol_port["stdev"],
        min_vol_port["ret"],
        marker=(5, 1, 0),
        color="green",
        s=200,
        label="Min Volatility",
    )

    if test_weights is not None and test_metrics is not None:
        t_ret, t_vol, t_sh = test_metrics
        ax.scatter(
            t_vol,
            t_ret,
            marker="X",
            color="black",
            s=140,
            label=test_label,
        )

    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # ----------------- Crisis fingerprint roles (per ticker) + VIS -----------------
    st.subheader("4️⃣ K-means Crisis Fingerprint Roles (per ticker)")

    with st.spinner("Building crisis fingerprints from long history (~20Y)..."):
        px_long = fetch_20y_prices(list(prices.columns))

    if px_long.empty:
        st.info(
            "Could not fetch enough long history to build crisis fingerprints "
            "for these tickers."
        )
    else:
        crisis_df = build_crisis_fingerprint(px_long)

        if crisis_df.empty:
            st.info("Not enough clean long history to compute crisis fingerprints.")
        else:
            cluster_df = kmeans_crisis_clusters(crisis_df, n_clusters=3)
            if cluster_df is None:
                st.info(
                    "Not enough complete crisis fingerprints to run K-means clustering."
                )
            else:
                # Pretty table view
                view_c = cluster_df.copy()
                view_c["LongTermAnnRet"] = view_c["LongTermAnnRet"].map(lambda x: f"{x*100:.2f}%")
                view_c["LongTermAnnVol"] = view_c["LongTermAnnVol"].map(lambda x: f"{x*100:.2f}%")
                view_c["LongTermMaxDD"] = view_c["LongTermMaxDD"].map(lambda x: f"{x*100:.2f}%")

                for crisis_name, _, _ in CRISIS_WINDOWS:
                    col = f"Ret_{crisis_name}"
                    if col in view_c.columns:
                        view_c[col] = view_c[col].map(
                            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "NA"
                        )

                crisis_cols_display = [
                    c for c in view_c.columns if c.startswith("Ret_")
                ]

                display_cols = (
                    ["Ticker", "CrisisRole", "LongTermAnnRet", "LongTermAnnVol", "LongTermMaxDD"]
                    + crisis_cols_display
                )

                st.dataframe(
                    view_c[display_cols].sort_values("CrisisRole"),
                    use_container_width=True,
                    height=320,
                )

                st.caption(
                    "CrisisRole is derived from K-means clustering on multi-crisis returns "
                    "+ long-term risk. Rough interpretation:\n"
                    "- **Crisis-resilient / defensive**: tends to hold up better in major drawdowns\n"
                    "- **Mixed / moderate**: mixed behaviour across crises\n"
                    "- **Crisis-vulnerable / pro-cyclical**: tends to get hit harder when markets sell off\n"
                )

                # 4a. Scatter: long-term risk/return colored by crisis cluster
                st.markdown("#### 📈 Long-term Risk/Return by Crisis Role")

                cluster_valid = cluster_df[cluster_df["CrisisCluster"].notna()].copy()
                if not cluster_valid.empty:
                    fig_sc, ax_sc = plt.subplots(figsize=(7.5, 5.5))
                    sc2 = ax_sc.scatter(
                        cluster_valid["LongTermAnnVol"],
                        cluster_valid["LongTermAnnRet"],
                        c=cluster_valid["CrisisCluster"],
                        cmap="viridis",
                        s=70,
                        alpha=0.9,
                    )
                    for _, row in cluster_valid.iterrows():
                        ax_sc.annotate(
                            row["Ticker"],
                            (row["LongTermAnnVol"], row["LongTermAnnRet"]),
                            textcoords="offset points",
                            xytext=(5, 3),
                            fontsize=8,
                        )
                    ax_sc.set_xlabel("Long-term annualized volatility")
                    ax_sc.set_ylabel("Long-term annualized return")
                    ax_sc.set_title("Tickers in long-term risk/return space by crisis role")
                    ax_sc.grid(alpha=0.3)
                    cbar2 = plt.colorbar(sc2, ax=ax_sc)
                    cbar2.set_label("Crisis cluster ID")
                    st.pyplot(fig_sc)

                    # 4b. Bar: average crisis return by cluster
                    st.markdown("#### 📊 Average Crisis Performance by Role")

                    avg_crisis_by_cluster = (
                        cluster_valid.groupby("CrisisCluster")["AvgCrisisRet"]
                        .mean()
                        .reset_index()
                    )
                    avg_crisis_by_cluster["Role"] = avg_crisis_by_cluster["CrisisCluster"].map(
                        lambda cid: cluster_valid.loc[
                            cluster_valid["CrisisCluster"] == cid, "CrisisRole"
                        ].iloc[0]
                    )

                    fig_bar, ax_bar = plt.subplots(figsize=(7, 4))
                    x = np.arange(len(avg_crisis_by_cluster))
                    heights = avg_crisis_by_cluster["AvgCrisisRet"].values
                    ax_bar.bar(x, heights)
                    ax_bar.axhline(0, linestyle="--", linewidth=1)
                    ax_bar.set_xticks(x)
                    ax_bar.set_xticklabels(avg_crisis_by_cluster["Role"], rotation=20, ha="right")
                    ax_bar.set_ylabel("Average crisis return")
                    ax_bar.set_title("Average crisis performance by crisis role")
                    for i, h in enumerate(heights):
                        ax_bar.text(
                            i,
                            h,
                            f"{h*100:.1f}%",
                            ha="center",
                            va="bottom" if h >= 0 else "top",
                            fontsize=8,
                        )
                    ax_bar.grid(axis="y", alpha=0.3)
                    st.pyplot(fig_bar)

                    # 4c. Heatmap: cluster × crisis window average returns
                    st.markdown("#### 🌡️ Crisis Window Heatmap by Role")

                    crisis_cols = [c for c in cluster_valid.columns if c.startswith("Ret_")]
                    if crisis_cols:
                        heat_df = (
                            cluster_valid.groupby("CrisisCluster")[crisis_cols]
                            .mean()
                            .sort_index()
                        )

                        # nicer column labels
                        col_labels = [c.replace("Ret_", "") for c in heat_df.columns]
                        row_labels = []
                        for cid in heat_df.index:
                            role = cluster_valid.loc[
                                cluster_valid["CrisisCluster"] == cid, "CrisisRole"
                            ].iloc[0]
                            row_labels.append(f"{cid} – {role}")

                        fig_hm, ax_hm = plt.subplots(figsize=(8, 4))
                        im = ax_hm.imshow(heat_df.values, aspect="auto", cmap="RdYlGn")

                        ax_hm.set_xticks(np.arange(len(col_labels)))
                        ax_hm.set_xticklabels(col_labels, rotation=30, ha="right")
                        ax_hm.set_yticks(np.arange(len(row_labels)))
                        ax_hm.set_yticklabels(row_labels)
                        ax_hm.set_title("Average crisis returns by role and crisis window")

                        for i in range(heat_df.shape[0]):
                            for j in range(heat_df.shape[1]):
                                val = heat_df.values[i, j]
                                text = f"{val*100:.1f}%" if np.isfinite(val) else "NA"
                                ax_hm.text(
                                    j,
                                    i,
                                    text,
                                    ha="center",
                                    va="center",
                                    fontsize=7,
                                )

                        plt.colorbar(im, ax=ax_hm, label="Return")
                        st.pyplot(fig_hm)

    # ----------------- Key portfolios -----------------
    st.subheader("5️⃣ Key Portfolios (annualized stats)")

    def _format_portfolio_row(row):
        base = pd.DataFrame(row).T
        base = base.copy()
        base["ret"] = base["ret"].map(lambda x: f"{x*100:.2f}%")
        base["stdev"] = base["stdev"].map(lambda x: f"{x*100:.2f}%")
        base["sharpe"] = base["sharpe"].map(lambda x: f"{x:.2f}")
        for t in prices.columns:
            base[t] = base[t].map(lambda x: f"{x*100:.2f}%")
        return base

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Max Sharpe Portfolio (annualized)**")
        st.dataframe(
            _format_portfolio_row(max_sharpe_port),
            use_container_width=True,
            height=220,
        )

    with col2:
        st.markdown("**Minimum Volatility Portfolio (annualized)**")
        st.dataframe(
            _format_portfolio_row(min_vol_port),
            use_container_width=True,
            height=220,
        )

    if test_weights is not None and test_metrics is not None:
        st.markdown("**User-Defined Test Portfolio (annualized)**")
        row = pd.Series(
            [test_metrics[0], test_metrics[1], test_metrics[2]] + list(test_weights),
            index=["ret", "stdev", "sharpe"] + list(prices.columns),
            name=test_label,
        )
        st.dataframe(
            _format_portfolio_row(row),
            use_container_width=True,
            height=220,
        )

else:
    st.info(
        "Use the GPT section above if you’d like some ETF/stock ideas, analyze their 5-year "
        "risk profile (with beta-aware K-means risk buckets), copy those tickers into the sidebar, then:\n"
        "- Pick an **analysis period** (custom, 2008–2009 Financial Crisis, or 2020 COVID Crash).\n"
        "Then set your simulation + test portfolio options and click **Run Frontier Simulation**.\n\n"
        "Below the frontier, you’ll also see **K-means crisis fingerprint roles** for each ticker, "
        "plus visualizations of how each role behaves across different crisis windows."
    )
