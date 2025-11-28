# 🧠 Agentic AI Financial Analyst — Streamlit App

A multi-page Streamlit application combining **financial data pipelines**, **machine learning-style analysis**, and **agentic GPT reasoning** for intelligent equity research, portfolio screening, and optimization.

> Demo Space: https://huggingface.co/spaces/andrewnap211/agentic-ai-financial-analyst  

---

## 🚀 Overview

This project merges **data-driven stock screening**, **LLM-based summarization**, and **quantitative portfolio construction** into one cohesive tool:

1. **Screener & Agentic Summarizer** — Filter stocks by sector, market cap, valuation, and profitability, then generate concise GPT-based company profiles and earnings call summaries with sentiment.
2. **Metrics & Performance Analyzer** — Build peer comparison tables with growth CAGRs, forward estimates, valuation ratios, and price/risk metrics.
3. **Portfolio Optimizer & Backtests** — Construct random long-only portfolios, visualize the efficient frontier, cluster tickers by risk and “crisis fingerprints,” and generate GPT-based ETF ideas and risk commentary.
4. **Weekly S&P 500 Prediction Recommendations** — Run a production-style weekly Logistic Regression model on the **current S&P 500 universe**, rank the top ideas, and query GPT about the tickers and feature signals.

---

## 🧩 Architecture

| Page | Purpose |
|------|---------|
| **`app.py`** | Stock Screener, Industry filter, Company Profiles, Agentic Earnings/Sentiment |
| **`pages/02_Metrics_&_Performance.py`** | Cross-ticker metrics comparison, forward/backward performance, agentic peer analysis |
| **`pages/03_Portfolio_Optimizer_&_Backtests.py`** | Efficient-frontier simulation, crisis-regime clustering (K-means), GPT idea generator & risk commentary |
| **`pages/04_Weekly_SP500_Predictions.py`** | Weekly S&P 500 Logistic Regression recommender, Top-K stock list, GPT Q&A on model drivers |

---

## ✨ Key Features

### 🧭 1. Screener + Agentic Company/Earnings Summaries (`app.py`)

- **Fundamental & technical filters**
  - Sector, Market Cap ($B), Volume, Valuation (PE, PS, PB)
  - Quality & Leverage (ROE, ROA, ROIC, D/E, Current Ratio)
  - Momentum (52W high, % above MA, RSI)
  - Profitability & Yield (Margins, Dividend Yield)

- **Step-wise flow**
  1. **Step 1:** Run FMP screener and metric filters.
  2. **Step 2:** Select industry → Fetch company profiles → Generate LLM summaries and sentiment.

- **Agentic Earnings Summarizer**
  - Pulls FMP **earnings call transcripts** (latest or user-selected quarter).
  - GPT summarizes **CEO/CFO remarks** following your prompt style (bullets vs paragraphs).
  - Computes **QoQ financial snapshot** (Revenue, Net Income, FCF).
  - Produces structured **JSON sentiment**:
    - polarity (−1 to 1) · label · confidence  
    - subscores (results / guidance / demand / margins) · key narrative drivers.

- **LLM grounding guarantees**
  - Each prompt explicitly restricts GPT to provided transcript/profile text.
  - No external facts or hallucination allowed.

---

### 📈 2. Metrics & Performance Page (`pages/02_Metrics_&_Performance.py`)

- **Flexible universe**
  - Pull tickers from the Screener session or manual input.
  - Choose max tickers and optional RSI period.

- **Integrated data blocks**
  - **Valuation (Fwd vs 5Y)** — PE/PS vs 5Y averages + margins/leverage.
  - **Growth & Margins (CAGR)** — 3Y Revenue/Net Income/OCF/FCF CAGRs, margin deltas, ROE/ROA/ROIC.
  - **Revenue / EPS Path** — Actual vs Analyst estimates (this & next year).
  - **Price & Risk (5Y)** — Total returns, volatility, VAR/CVAR, drawdowns.

- **Price performance** — 1M / 3M / 6M / 1Y from FMP; 5Y history from Yahoo Finance.

- **Agentic Peer Comparison**
  - Select multiple tickers → Ask GPT a free-form question (e.g., “Who balances growth and margins best?”).
  - LLM reasoning is confined to current session JSON tables.
  - Heuristic scoring fallback if no OpenAI key.

---

### ⚙️ 3. Portfolio Optimizer & Backtests (`pages/03_Portfolio_Optimizer_&_Backtests.py`)

#### 🧮 Efficient-Frontier Simulation
- Pulls daily prices (Yahoo Finance) for user-selected tickers and period.
- Simulates up to **100 000 random long-only portfolios**.
- Computes annualized:
  - Return · Volatility · Sharpe ratio (path-correct via log returns).
- Highlights:
  - **Max Sharpe Portfolio**
  - **Minimum Volatility Portfolio**
  - Optional **User-Defined Test Portfolio** (equal-weight or custom weights).

#### 💡 GPT Idea Generator
- Describe your investment goals → GPT suggests **plain-vanilla ETFs or large-cap stocks**.
- Output: Markdown table with Ticker, Name, Instrument Type, Risk Level, Rationale.
- Parser auto-extracts tickers for simulation.

#### 🧬 Risk & Return Snapshot (5Y)
- Computes per-ticker:
  - Ann. Return, Vol, Sharpe, Max Drawdown, Beta (vs SPY), Downside Vol.
- **K-means risk buckets** → Defensive / Core / Growth / Speculative.
- GPT risk commentary explains defensive vs aggressive funds and pain points (Max DD, Downside Vol).

#### 🌩 Crisis Fingerprint Roles (~20Y History)
- Builds per-ticker fingerprints across major crises:  
  2008 GFC · 2011 Downgrade · 2015–16 Oil/China · 2020 COVID · 2022 Inflation.
- K-means clustering on multi-crisis returns + long-term risk metrics →  
  **Crisis-resilient / Mixed / Crisis-vulnerable.**
- Visualizations:
  - Long-term Return vs Volatility scatter (colored by role)
  - Bar chart of avg crisis return per role
  - Heatmap of crisis windows × roles.

---

### 📅 4. Weekly S&P 500 Prediction Recommendations (`pages/04_Weekly_SP500_Predictions.py`)

A production-style weekly alpha engine powered by your trained **Logistic Regression** model on technical indicators.

- **Universe**
  - Default: **current S&P 500 constituents** scraped live from Wikipedia.
  - Optional: custom ticker list.

- **Feature pipeline (mirrors training)**
  - Daily OHLCV from Yahoo Finance since 2015.
  - TA-Lib technicals per ticker:
    - Returns: `ret_5d`, `ret_20d`
    - Momentum: `rsi_14`, `macd_line`, `macd_signal`, `stoch_k_14_3`, `stoch_d_14_3`
    - Volatility/Trend: `bb_pos_20_2`, `atr14_norm`, `adx_14`
    - Volume/Flow: `obv_slope_20`, `vol_zscore_20`
  - 60-day warmup, then **weekly resample (`W-FRI`)** with `.last()`.
  - Only the **most recent completed week** is kept.

- **Model**
  - Joblib-serialized pipeline: **StandardScaler + LogisticRegression**, trained offline.
  - Loaded from `models/logreg_tech_final_v2.joblib` (or `LOGREG_MODEL_PATH`).

- **Outputs**
  - **Top-K stock list** sorted by predicted probability of weekly outperformance.
  - Joined with **Company** and **Sector** metadata from the S&P table.
  - Downloadable CSVs:
    - Top-K recommendations
    - Full scored universe.

- **GPT Q&A on Stocks & Features**
  - Select tickers from the Top-K or full universe.
  - Ask GPT questions like:
    - “Why is the model confident in these names this week?”
    - “Which technical features stand out for AAPL vs MSFT?”
    - “What kind of market regime usually favors this basket?”
  - GPT sees **only the feature matrix and prediction outputs** for the selected names and explains:
    - Which signals are elevated/suppressed,
    - How that relates to trend, momentum, volatility,
    - Caveats and reminders that this is not investment advice.

---

## 🔌 Data Sources & Models

| Source / Model | Used For | Examples |
|----------------|----------|----------|
| **Financial Modeling Prep (FMP)** | Screener, profiles, metrics, transcripts, estimates | `/v3/profile`, `/v3/ratios-ttm`, `/v3/income-statement`, `/v3/earning_call_transcript`, etc. |
| **Yahoo Finance (`yfinance`)** | 5–20 year price history, volatility, beta, drawdowns, weekly features | `yf.download()` |
| **Logistic Regression (joblib pipeline)** | Weekly S&P 500 outperformance classification based on technicals | `models/logreg_tech_final_v2.joblib` |
| **OpenAI GPT-4o-mini** | Summaries, sentiment, ETF ideas, risk commentary, peer analysis, weekly feature Q&A | `chat.completions.create()` |

---

## ⚙️ Environment Variables

| Variable | Purpose |
|---------|---------|
| `FMP_API_KEY` | Financial Modeling Prep access |
| `OPENAI_API_KEY` | Enables LLM summaries & GPT panels |
| `OPENAI_MODEL` | Overrides GPT model (defaults to `gpt-4o-mini`) |
| `LOGREG_MODEL_PATH` | Optional override path for the weekly LogReg pipeline (`.joblib`) |
| `FMP_BATCH_SIZE` / `FMP_BATCH_SLEEP` | Optional rate-limit tuning |

If no OpenAI key is provided, heuristic summaries and risk rules run instead of GPT calls.

---

## 🏗️ Run Locally

```bash
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

export FMP_API_KEY="your_fmp_key"
export OPENAI_API_KEY="sk-..."   # optional but recommended
# export LOGREG_MODEL_PATH="models/logreg_tech_final_v2.joblib"  # optional override

streamlit run app.py

