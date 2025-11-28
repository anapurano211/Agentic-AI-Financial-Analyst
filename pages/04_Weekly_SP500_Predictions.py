# pages/04_Weekly_SP500_Predictions.py
# -------------------------------------------------------------------
# Weekly S&P 500 Prediction Recommendations
#
# - Scrapes current S&P 500 constituents from Wikipedia
# - Rebuilds daily → TA → weekly (W-FRI) features to match training
# - Loads final LogReg pipeline (StandardScaler + LogisticRegression)
# - Scores current (most recent) week for the universe
# - Returns Top-K ideas with Company / Sector / predicted probability
# - GPT Q&A panel to ask questions about this week’s picks & features
# -------------------------------------------------------------------

import os
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import joblib

try:
    import talib as ta
except ImportError:
    ta = None

# Optional GPT client
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ------------------------- Streamlit Config ---------------------------

st.set_page_config(
    page_title="Weekly S&P 500 Prediction Recommendations",
    layout="wide",
)

st.title("📈 Weekly S&P 500 Prediction Recommendations")

st.markdown(
    """
This page runs your **weekly S&P 500 Logistic Regression model** in production mode:

- Universe: **current S&P 500 constituents** scraped live from Wikipedia  
- Features: **12 technical indicators** on daily data, resampled to **weekly (W-FRI)**  
- Model: **Logistic Regression** trained to predict whether a stock will **outperform** next week  
- Output: **Top-K names** ranked by model probability, with company name and sector

> ⚠️ This is a *research / educational* tool and **not** investment advice.
"""
)

# ------------------------- Core Config & Paths ------------------------

DATE_COL   = "Date"
ID_COL     = "Ticker"

FEATURE_COLS = [
    "ret_5d",
    "ret_20d",
    "rsi_14",
    "macd_line",
    "macd_signal",
    "stoch_k_14_3",
    "stoch_d_14_3",
    "bb_pos_20_2",
    "atr14_norm",
    "adx_14",
    "obv_slope_20",
    "vol_zscore_20",
]

WEEKLY_FREQ      = "W-FRI"
TRAIN_START_DATE = "2015-01-01"  # must match your training ETL

# Project root = parent of /pages
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "logreg_tech_final_v2.joblib"
env_path = os.getenv("LOGREG_MODEL_PATH")
MODEL_PATH = Path(env_path) if env_path else DEFAULT_MODEL_PATH

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# ------------------------- GPT Config --------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAI) else None
MAX_TOKENS_WEEKLY_QA = 1200


# ============================ Helpers =================================

@st.cache_resource(show_spinner=False)
def load_final_model(model_path: Path = MODEL_PATH):
    """
    Load the final Logistic Regression pipeline (StandardScaler + LogisticRegression).

    This is the SAME pipeline you trained, so scaling is consistent with your backtest.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at: {model_path}\n"
            "Set LOGREG_MODEL_PATH env var or place the joblib file in app root."
        )
    model = joblib.load(model_path)
    return model


@st.cache_data(ttl=3600, show_spinner=False)
def get_sp500_constituents() -> Tuple[List[str], pd.DataFrame]:
    """
    Scrape current S&P 500 constituents from Wikipedia (robust version).

    Returns
    -------
    tickers_list : list[str]
        Cleaned tickers (BRK.B -> BRK-B, etc.)
    sp500_df : DataFrame
        Full S&P 500 metadata table
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }
    resp = requests.get(SP500_WIKI_URL, headers=headers)
    resp.raise_for_status()

    tables = pd.read_html(resp.text)
    if not tables:
        raise RuntimeError("No tables found on S&P 500 Wikipedia page.")

    candidate_idx = None
    symbol_col_name = None

    for i, tbl in enumerate(tables):
        cols = [str(c).strip() for c in tbl.columns]
        for c in cols:
            c_lower = c.lower()
            if "symbol" in c_lower or "ticker" in c_lower:
                candidate_idx = i
                symbol_col_name = c
                break
        if candidate_idx is not None:
            break

    if candidate_idx is None or symbol_col_name is None:
        df = tables[0].copy()
        df.columns = [str(c).strip() for c in df.columns]
        possible = [
            c for c in df.columns
            if "symbol" in c.lower() or "ticker" in c.lower()
        ]
        if not possible:
            raise RuntimeError(
                "Could not find a 'Symbol' or 'Ticker'-like column in any table. "
                f"First table columns: {list(df.columns)}"
            )
        symbol_col_name = possible[0]
    else:
        df = tables[candidate_idx].copy()

    df.columns = [str(c).strip() for c in df.columns]

    df[symbol_col_name] = (
        df[symbol_col_name]
        .astype(str)
        .str.strip()
        .str.replace(".", "-", regex=False)  # BRK.B -> BRK-B
    )

    tickers = df[symbol_col_name].tolist()

    # Standardize metadata column names if they exist
    col_map = {}
    for col in df.columns:
        low = col.lower()
        if "symbol" in low or "ticker" in low:
            col_map[col] = "Ticker"
        elif "security" in low:
            col_map[col] = "Company"
        elif "gics sector" in low:
            col_map[col] = "Sector"
    df_meta = df.rename(columns=col_map)

    return tickers, df_meta


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a per-ticker OHLCV DataFrame with columns:
    ['Open', 'High', 'Low', 'Close', 'Volume'],
    add technical features and return the enriched DataFrame.

    Mirrors your TRAINING script (TA-Lib based).
    """
    if ta is None:
        raise ImportError(
            "The 'ta-lib' package is required for this page. "
            "Install it in your environment (e.g., `pip install TA-Lib`)."
        )

    df = df.copy()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # 1) Returns
    df["ret_5d"]  = close.pct_change(5)
    df["ret_20d"] = close.pct_change(20)

    # 2) RSI(14)
    df["rsi_14"] = ta.RSI(close, timeperiod=14)

    # 3) MACD (12, 26, 9)
    macd, macd_signal, macd_hist = ta.MACD(
        close,
        fastperiod=12,
        slowperiod=26,
        signalperiod=9,
    )
    df["macd_line"]   = macd
    df["macd_signal"] = macd_signal

    # 4) Stochastic Oscillator (14,3)
    slowk, slowd = ta.STOCH(
        high,
        low,
        close,
        fastk_period=14,
        slowk_period=3,
        slowd_period=3,
        slowk_matype=0,
        slowd_matype=0,
    )
    df["stoch_k_14_3"] = slowk
    df["stoch_d_14_3"] = slowd

    # 5) Bollinger Bands (20, 2)
    upper, middle, lower = ta.BBANDS(
        close,
        timeperiod=20,
        nbdevup=2,
        nbdevdn=2,
        matype=0,
    )
    df["bb_upper_20_2"]  = upper
    df["bb_middle_20_2"] = middle
    df["bb_lower_20_2"]  = lower

    band_width = upper - lower
    df["bb_pos_20_2"] = (close - lower) / band_width
    df.loc[band_width == 0, "bb_pos_20_2"] = np.nan  # avoid /0

    # 6) ATR(14) normalized
    atr14 = ta.ATR(high, low, close, timeperiod=14)
    df["atr14"]      = atr14
    df["atr14_norm"] = atr14 / close

    # 7) ADX(14)
    df["adx_14"] = ta.ADX(high, low, close, timeperiod=14)

    # 8) OBV + 20-day slope
    obv = ta.OBV(close, vol)
    df["obv"]          = obv
    df["obv_slope_20"] = obv.pct_change(20)

    # 9) Volume z-score (20d)
    vol_ma_20  = vol.rolling(20).mean()
    vol_std_20 = vol.rolling(20).std()
    df["vol_zscore_20"] = (vol - vol_ma_20) / vol_std_20

    return df


@st.cache_data(ttl=1800, show_spinner=False)
def build_sp500_current_week_features(
    symbols: List[str],
    start_date: str = TRAIN_START_DATE,
    weekly_freq: str = WEEKLY_FREQ,
) -> pd.DataFrame:
    """
    For a list of tickers, build all technical features using the SAME
    daily → TA-Lib → 60-day warmup → W-FRI resample pipeline as training,
    and then keep ONLY the most recent week available.
    """
    data = yf.download(
        symbols,
        start=start_date,
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    feature_frames = []

    for ticker in symbols:
        try:
            df_ticker = data[ticker]
        except Exception:
            df_ticker = data.get(ticker)

        if df_ticker is None:
            continue

        df_ticker = df_ticker.dropna().copy()
        if df_ticker.empty:
            continue

        df_feat = add_technical_features(df_ticker)
        df_feat["Ticker"] = ticker

        # Warmup cut
        df_feat = df_feat.iloc[60:]
        if df_feat.empty:
            continue

        feature_frames.append(df_feat)

    if not feature_frames:
        raise ValueError("No valid tickers after technical feature engineering.")

    features_df = pd.concat(feature_frames)
    features_df.index.name = "Date"
    features_df.reset_index(inplace=True)
    features_df["Date"] = pd.to_datetime(features_df["Date"])

    # Weekly resample
    weekly_frames = []
    for ticker, grp in features_df.groupby("Ticker"):
        g = grp.set_index("Date").sort_index()
        weekly = g.resample(weekly_freq).last()
        weekly["Ticker"] = ticker
        weekly_frames.append(weekly.reset_index())

    weekly_features_df = pd.concat(weekly_frames, ignore_index=True)

    keep_cols = ["Date", "Ticker"] + FEATURE_COLS
    weekly_features_df = weekly_features_df[keep_cols].copy()
    weekly_features_df["Date"] = pd.to_datetime(weekly_features_df["Date"])

    weekly_features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    weekly_features_df = weekly_features_df.dropna(subset=FEATURE_COLS)

    if weekly_features_df.empty:
        raise RuntimeError("No weekly rows with full features after cleaning.")

    max_week = weekly_features_df["Date"].max()
    current_week_df = weekly_features_df[weekly_features_df["Date"] == max_week].copy()

    return current_week_df


def score_current_week(
    model,
    feature_df: pd.DataFrame,
    top_k: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score the current-week feature_df with the trained pipeline.
    """
    df = feature_df.copy()
    X = df[FEATURE_COLS].values

    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    df["pred_proba"] = probs
    df["pred_label"] = preds

    scores_df = df[[DATE_COL, ID_COL, "pred_proba", "pred_label"] + FEATURE_COLS].copy()

    latest_week = scores_df[DATE_COL].max()
    latest_df = scores_df[scores_df[DATE_COL] == latest_week].copy()
    latest_df = latest_df.sort_values("pred_proba", ascending=False)

    top_df = latest_df.head(top_k).reset_index(drop=True)
    return scores_df, top_df


# ===================== GPT Weekly Q&A Helpers =========================

def _weekly_df_to_records(df: pd.DataFrame, tickers_subset: List[str]):
    """Lightly clean & compress table before sending to GPT."""
    if df is None or df.empty:
        return []

    sub = df[df["Ticker"].isin(tickers_subset)].copy()
    if sub.empty:
        return []

    if "Date" in sub.columns:
        sub["Date"] = pd.to_datetime(sub["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    num_cols = sub.select_dtypes(include=["number"]).columns
    sub[num_cols] = sub[num_cols].round(4)

    return sub.to_dict(orient="records")


def run_weekly_gpt_analysis(
    selected_tickers: List[str],
    question: str,
    full_df: pd.DataFrame,
    max_rows_for_llm: int,
):
    """
    Build a JSON context for the latest week & call GPT
    to explain the model’s signals in plain language.
    """
    if not question.strip() or not selected_tickers:
        return "Please select at least one ticker and enter a question."

    if not client or not OPENAI_API_KEY:
        return (
            "No OpenAI API key is configured, so I can't run GPT here.\n\n"
            "Heuristic tip: higher `pred_proba` combined with strong recent returns "
            "(ret_5d / ret_20d), constructive momentum (MACD above signal), "
            "supportive RSI (not extremely overbought), and reasonable volatility / "
            "volume often indicate stronger technical setups in this model."
        )

    subset = selected_tickers[:max_rows_for_llm]
    records = _weekly_df_to_records(full_df, subset)
    if not records:
        return "No matching rows found for the selected tickers."

    model_context = {
        "description": (
            "Weekly S&P 500 ranking model: Logistic Regression on 12 technical features "
            f"trained from {TRAIN_START_DATE} onwards. Target is next-week outperformance "
            "vs. the market benchmark. Higher `pred_proba` means the model sees a higher "
            "probability of beating the benchmark over the next week."
        ),
        "feature_names": FEATURE_COLS,
        "latest_week_rows": records,
    }

    prompt = f"""
You are an equity quant explaining a weekly stock selection model.

The model is a logistic regression on technical indicators only (no fundamentals),
trained on weekly data (W-FRI). Each row below is ONE stock for the most recent week,
with the following fields:

- Date, Ticker, pred_proba (model's probability of 'outperform' for next week),
- Sector / Company (if available),
- and the 12 technical features used in training:
  {", ".join(FEATURE_COLS)}

CONTEXT (JSON):
{json.dumps(model_context, indent=2, ensure_ascii=False, default=str)}

USER QUESTION:
{question.strip()}

Guidelines:
- Use ONLY the JSON above; do NOT invent numbers or tickers.
- Explain in plain language how the technical features might drive the model's view.
- Compare the selected stocks to each other (why some score higher/lower).
- Call out any names that look technically stretched / overbought / oversold or unusually volatile.
- Highlight 2–3 general *risk considerations* (no advice, no guarantees).
- Do NOT give personalized financial advice; keep everything generic and educational.
"""

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.35,
            max_tokens=MAX_TOKENS_WEEKLY_QA,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"GPT analysis failed: {e}"


# ======================== Sidebar Controls ============================

with st.sidebar:
    st.header("Weekly Scan Settings")

    universe_mode = st.radio(
        "Universe",
        ["Current S&P 500 (from Wikipedia)", "Custom tickers"],
        index=0,
    )

    custom_text = ""
    if universe_mode == "Custom tickers":
        custom_text = st.text_area(
            "Custom tickers (comma / space / newline separated)",
            value="AAPL, MSFT, NVDA, META, JPM, JNJ",
            height=80,
        )

    top_k = st.slider(
        "Top-K recommendations to show",
        min_value=10,
        max_value=100,
        value=30,
        step=5,
    )

    st.markdown("---")
    show_full_scores = st.checkbox(
        "Show full scored universe table",
        value=False,
        help="If checked, also show all scored names (not just Top-K).",
    )

    run_btn = st.button("Run Weekly S&P 500 Scan", type="primary")


# ====================== Load previous scan state ======================

scan_state = st.session_state.get("weekly_scan")


# =========================== Run scan ================================

if run_btn:
    # 1) Check TA-Lib availability
    if ta is None:
        st.error(
            "The 'ta-lib' package is not available. "
            "Install it in your environment before running this page."
        )
        st.stop()

    # 2) Load universe
    if universe_mode == "Current S&P 500 (from Wikipedia)":
        with st.spinner("Scraping current S&P 500 constituents from Wikipedia..."):
            try:
                sp500_symbols, sp500_df = get_sp500_constituents()
            except Exception as e:
                st.error(f"Failed to fetch S&P 500 list: {e}")
                st.stop()

        symbols = sorted(set(sp500_symbols))
        st.success(f"Loaded {len(symbols)} S&P 500 tickers.")
    else:
        tokens = (
            custom_text.replace("\n", " ")
            .replace("\t", " ")
            .replace(";", ",")
            .replace("  ", " ")
            .replace(",", " ")
            .split(" ")
        )
        symbols = sorted({t.strip().upper() for t in tokens if t.strip()})
        if not symbols:
            st.error("Please enter at least one valid ticker.")
            st.stop()
        sp500_df = pd.DataFrame(columns=["Ticker", "Company", "Sector"])

    # 3) Load model
    with st.spinner(f"Loading final Logistic Regression model from `{MODEL_PATH}`..."):
        try:
            model = load_final_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Could not load model: {e}")
            st.stop()

    # 4) Build current-week features
    with st.spinner(
        f"Downloading daily prices from {TRAIN_START_DATE} onward and building TA features..."
    ):
        try:
            current_week_df = build_sp500_current_week_features(symbols)
        except Exception as e:
            st.error(f"Error building current-week features: {e}")
            st.stop()

    latest_week = current_week_df["Date"].max()

    # 5) Score & rank
    with st.spinner("Scoring current week with Logistic Regression model..."):
        scores_df, top_df = score_current_week(model, current_week_df, top_k=top_k)

    # 6) Attach S&P 500 metadata (Company / Sector) if we have it
    if not sp500_df.empty and "Ticker" in sp500_df.columns:
        sp500_meta = sp500_df.copy()
        for c in ["Company", "Sector"]:
            if c not in sp500_meta.columns:
                sp500_meta[c] = np.nan

        top_with_meta = top_df.merge(
            sp500_meta[["Ticker", "Company", "Sector"]],
            on="Ticker",
            how="left",
        )
        scores_with_meta = scores_df.merge(
            sp500_meta[["Ticker", "Company", "Sector"]],
            on="Ticker",
            how="left",
        )
    else:
        top_with_meta = top_df.copy()
        scores_with_meta = scores_df.copy()

    # Save everything to session_state so Q&A survives future reruns
    scan_state = {
        "latest_week": latest_week,
        "current_week_df": current_week_df,
        "top_with_meta": top_with_meta,
        "scores_with_meta": scores_with_meta,
        "top_k": top_k,
    }
    st.session_state["weekly_scan"] = scan_state


# ====================== Render results if scan exists =================

if scan_state is not None:
    latest_week = scan_state["latest_week"]
    current_week_df = scan_state["current_week_df"]
    top_with_meta = scan_state["top_with_meta"]
    scores_with_meta = scan_state["scores_with_meta"]

    st.subheader("1️⃣ Feature Build & Latest Week")
    st.write(
        f"Using **latest completed week** ending: "
        f"**{latest_week.date()}** with "
        f"**{current_week_df[ID_COL].nunique()}** tickers having full features."
    )
    st.dataframe(
        current_week_df[[DATE_COL, ID_COL] + FEATURE_COLS].tail(10).round(4),
        use_container_width=True,
        height=260,
    )

    # 2) Top-K table
    st.subheader("2️⃣ Model Scoring & Top-K Ideas")

    main_cols = ["Date", "Ticker", "Company", "Sector", "pred_proba", "pred_label"]
    main_cols = [c for c in main_cols if c in top_with_meta.columns]
    other_cols = [c for c in top_with_meta.columns if c not in main_cols]

    view_top = top_with_meta[main_cols + other_cols].copy()
    if "pred_proba" in view_top.columns:
        view_top["pred_proba"] = view_top["pred_proba"].map(lambda x: f"{x:.3f}")

    st.markdown(f"**Top {len(view_top)} names ranked by model probability**")
    st.dataframe(
        view_top,
        use_container_width=True,
        height=420,
    )

    st.download_button(
        "Download Top-K as CSV",
        data=view_top.to_csv(index=False).encode("utf-8"),
        file_name="weekly_sp500_top_k_predictions.csv",
        mime="text/csv",
    )

    # -------------------- 🤖 Weekly Model Q&A --------------------
    st.markdown("---")
    st.subheader("🤖 Ask the Weekly Model Assistant")

    st.caption(
        "Ask questions about this week’s S&P 500 picks. "
        "The assistant only sees the model’s probabilities and technical features "
        "for the latest week (it does **not** see or change the model itself)."
    )

    if "Ticker" in top_with_meta.columns:
        tickers_list = top_with_meta["Ticker"].tolist()
    else:
        tickers_list = top_with_meta[ID_COL].tolist()

    default_sel = tickers_list[: min(10, len(tickers_list))]

    selected_ticks = st.multiselect(
        "Which names do you want to talk about?",
        options=tickers_list,
        default=default_sel,
        key="weekly_qna_tickers",
    )

    user_q = st.text_area(
        "Your question for the assistant",
        height=120,
        placeholder=(
            "Examples:\n"
            "- Why is the model so bullish on these stocks this week?\n"
            "- Compare the technical set-up of these names.\n"
            "- Which factors seem to be driving the high probabilities?\n"
            "- What risks might I watch for given these signals?"
        ),
        key="weekly_qna_question",
    )

    max_rows_for_llm = st.slider(
        "Max stocks to send to the assistant",
        min_value=5,
        max_value=40,
        value=min(15, len(selected_ticks) if selected_ticks else 15),
        step=1,
        help="Keeps the context small so the GPT call stays cheap and fast.",
    )

    if st.button("Ask the weekly model assistant"):
        if not selected_ticks:
            st.warning("Pick at least one ticker.")
        elif not user_q.strip():
            st.warning("Type a question for the assistant.")
        else:
            with st.spinner("Let me analyze the weekly features and predictions..."):
                answer = run_weekly_gpt_analysis(
                    selected_ticks,
                    user_q,
                    top_with_meta,
                    max_rows_for_llm,
                )

            st.markdown(
                '<div style="white-space:pre-wrap;line-height:1.5;background:#fff;'
                'border:1px solid #ececf3;border-radius:10px;padding:14px;'
                'font-variant-ligatures:none;">'
                + answer
                + "</div>",
                unsafe_allow_html=True,
            )

    # Optional: full scored universe
    if show_full_scores:
        st.markdown("---")
        st.subheader("3️⃣ Full Scored Universe (all tickers with features)")

        view_full = scores_with_meta.copy()
        if "pred_proba" in view_full.columns:
            view_full["pred_proba"] = view_full["pred_proba"].map(lambda x: f"{x:.3f}")
        st.dataframe(
            view_full.sort_values("pred_proba", ascending=False),
            use_container_width=True,
            height=420,
        )

        st.download_button(
            "Download full scored universe as CSV",
            data=view_full.to_csv(index=False).encode("utf-8"),
            file_name="weekly_sp500_all_scores.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.caption(
        "Probabilities are **model-based estimates** of weekly outperformance using only "
        "technical features. Always sanity-check results against fundamentals, news, and your own judgment."
    )

else:
    st.info(
        "Set your **universe** and **Top-K** size in the sidebar, then click "
        "**Run Weekly S&P 500 Scan** to build features for the latest week and generate recommendations.\n\n"
        "After you run the scan, you can also use the **Weekly Model Assistant** to ask GPT questions "
        "about the top picks and the technical features driving the signals."
    )
