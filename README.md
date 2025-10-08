# Agentic AI Financial Analyst — Streamlit App

One-click stock screening ➜ metrics filtering ➜ industry selection ➜ concise company profiles ➜ **agentic** earnings summaries & sentiment — plus a Metrics & Performance page for cross-ticker comps and narrative analysis.

> Live demo (example Space): https://huggingface.co/spaces/andrewnap211/agentic-ai-financial-analyst

---

## ✨ What’s New (since v0.1)

- **Multi-page app**: new `pages/02_Metrics_&_Performance.py` for cross-ticker pulls and an *agentic* Competitive Analysis panel.
- **Industry-gated earnings**: transcripts run **after** you choose an **Industry** (saves LLM calls).
- **Range sliders** for **Market Cap** and **Average Volume** (min & max) in the screener.
- **Agentic earnings summarizer**: user controls bullets/lines, count, focus/exclusions, and optional QoQ facts (Revenue, NI, FCF).
- **JSON sentiment**: overall polarity, label, confidence, sub-scores (results/guidance/demand/margins), and drivers.
- **Cross-page state sharing**: the main page’s ticker universe is available on the Metrics page via `st.session_state`.
- **Batch-safe API calling**: env knobs `FMP_BATCH_SIZE` and `FMP_BATCH_SLEEP` throttle high-volume loops to reduce 429s.
- **Resilient HTTP**: session with retries/backoff for 429/5xx; symbol variant normalization (`-`/`.` swaps).
- **Metrics page upgrades**:
  - Forward PE/PS from analyst windows; EPS/Revenue “this/next year”.
  - 3-year CAGRs (rev/NI/OC/FCF) & operating margin change.
  - 5-year PE/PS averages (peer context).
  - Optional 5-year price & risk block (yfinance): total/annualized return, VAR/CVAR, volatility, largest drops.
  - Agentic panel for narrative peer comparison over the current table.
- **CSV exports** from both pages.

---

## 🚦 Current Features

### Main page (`app.py`)
- **Server-side screening (FMP)**: sector, market-cap/volume ranges, active-trading flag, pagination, retries.
- **Optional advanced filters**: valuation (PE/PS/PB), quality/leverage (D/E, Current), momentum (RSI, MA distance), profitability (ROE/ROA/ROIC, margins), dividend yield.
- **Industry gating**: pick **Industry** *after* metrics, then run profiles & earnings.
- **Profiles (FMP v3)** with **LLM summaries**: 1–3 short lines, strictly from the company description (no outside facts).
- **Agentic earnings**:
  - Transcript fetch (latest or specific year/quarter).
  - Free-form instruction with bullets/lines & count; focus/exclusions.
  - Optional **QoQ snapshot** box (Revenue, Net Income, FCF).
  - **Sentiment JSON** + visual bars.
- **Cards UI** and **CSV export**.

### Metrics & Performance page (`pages/02_Metrics_&_Performance.py`)
- **Ticker universe** from the main page or paste your own.
- **Advanced metrics merge**: profile + quote + ratios-TTM + key-metrics-TTM + RSI.
- **Peer-app parity**:
  - **5-yr averages** (PE/PS).
  - **3-yr CAGRs** (rev/NI/OC/FCF); operating margin now vs 3y ago.
  - **Analyst windows** → forward **EPS/Revenue** and **PE_fwd / PS_fwd**.
  - **Price & risk (5y, optional)** with yfinance.
- **Agentic Competitive Analysis**: ask narrative questions; the model uses the **current table** only (no hidden fetches).

---

## 🧱 Project Structure

```
app.py                           # Screener → Metrics → Industry → Profiles → Agentic Earnings/Sentiment
pages/
  └─ 02_Metrics_&_Performance.py # Cross-ticker metrics + agentic competitive analysis
requirements.txt
README.md
```

> Anything in `pages/` becomes a Streamlit page; numeric prefix sets menu order.

---

## 🔌 Data & Models

| Source | Purpose | Endpoints (examples) |
|---|---|---|
| **Financial Modeling Prep (FMP)** | Screener, profiles, metrics, transcripts | `/stable/company-screener`, `/api/v3/profile/{ticker}`, `/api/v3/quote/{ticker}`, `/api/v3/ratios-ttm/{ticker}`, `/api/v3/key-metrics-ttm/{ticker}`, `/api/v3/technical_indicator/...`, `/api/v3/income-statement/{ticker}`, `/api/v3/cash-flow-statement/{ticker}`, `/api/v3/analyst-estimates/{ticker}`, `/api/v3/earning_call_transcript/{ticker}` |
| **OpenAI** | LLM summaries, agentic analysis | Chat Completions (default `gpt-4o-mini`) |
| **yfinance** (optional) | 5-year price & risk stats | Download OHLCV; compute returns, VAR/CVAR, volatility |

> LLM prompts **forbid outside facts**; summaries are grounded strictly in provided text.

---

## ⚙️ Configuration

Set as environment variables (or HF Space Secrets):

- `FMP_API_KEY` (required)
- `OPENAI_API_KEY` (optional, enables LLM features fully; fallback heuristics used otherwise)
- `OPENAI_MODEL` (optional, default: `gpt-4o-mini`)

**Batch / rate-limit knobs (optional)**

- `FMP_BATCH_SIZE` (default: `50`)
- `FMP_BATCH_SLEEP` (seconds between batches, default: `1.0`)

These throttle tight loops (profiles, RSI, metrics, performance & peer pulls). Lower the batch size or increase sleep if you see `429 Too Many Requests`.

---

## ▶️ Run Locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt

# Example (PowerShell)
$env:FMP_API_KEY="your_fmp_key"
$env:OPENAI_API_KEY="sk-..."    # optional
$env:FMP_BATCH_SIZE="50"        # optional
$env:FMP_BATCH_SLEEP="1.0"      # optional

streamlit run app.py
```

Open the URL from Streamlit (usually http://localhost:8501).

---

## 🚀 Deploy to Hugging Face Spaces

1. Create a **Streamlit** Space.
2. Upload: `app.py`, `pages/02_Metrics_&_Performance.py`, `requirements.txt`, `README.md`.
3. In **Settings → Repository secrets**, add:
   - `FMP_API_KEY`
   - `OPENAI_API_KEY` (optional)
   - (Optional) `FMP_BATCH_SIZE`, `FMP_BATCH_SLEEP`, `OPENAI_MODEL`
4. The Space builds and serves automatically.

**requirements.txt** (minimal):

```
streamlit>=1.31
pandas>=2.0
numpy>=1.24
requests>=2.31
urllib3>=1.26,<3
openai>=1.37.0
yfinance>=0.2.38
```

---

## 🧭 Using the App

### Main Page — two steps

1) **Step 1 — Run screener + metrics**  
   - Choose *Sector*, *Market Cap Range*, *Avg Volume Range*, *Max companies*.  
   - (Optional) enable *Advanced Metrics* and set ranges for valuation, leverage, momentum, profitability, yield.  
   - Click **Step 1** to see screened tickers and (optionally) their metrics.

2) **Industry filter**  
   - Pick an **Industry** (e.g., “Semiconductors”). The universe trims accordingly.

3) **Step 2 — Profiles + Earnings**  
   - Mode: latest or specific year/quarter.  
   - Provide instructions (e.g., “6 bullets: results, guidance, demand, margins/costs, cap returns, risks; include QoQ”).  
   - The app renders company cards with a 3-line profile, optional **QoQ** box, **LLM summary**, and **sentiment bars**.  
   - Download a CSV.

### Metrics & Performance Page

- Reads the universe from the main page, or paste custom tickers.  
- Click **Fetch metrics** to build a comprehensive table:
  - Valuation, quality, momentum, profitability, yield.
  - 3-yr CAGRs & margin change.
  - Forward EPS/Revenue windows → **PE_fwd / PS_fwd**.
  - Optional 5-year price & risk (yfinance).  
- Use **🤖 Agentic Competitive Analysis** to ask narrative questions (e.g., “Balance growth + margins vs forward valuation; who’s most resilient on downside risk?”).  
- The analysis uses the **current in-memory table**; re-click **Fetch metrics** to refresh data.

---

## 🧠 How It Works (high level)

- **Robust HTTP**: shared `requests.Session` with retries/backoff for 429/5xx; JSON helpers.
- **Ticker normalization**: tries symbol variants (`-`↔`.`) automatically.
- **Screener**: `fmp_company_screener_safe()` paginates, merges, de-dupes; local ETF/Fund fallback filter.
- **Metrics**: `_fetch_quote`, `_fetch_ratios_ttm`, `_fetch_key_metrics_ttm`, `_fetch_rsi_latest_multi` → merged & prettified; computes `% from high`, `% vs 50/200D MA`, etc.
- **Profiles**: `fetch_profiles_v3_bulk()` pulls v3 profiles; LLM condenses to 1–3 lines.
- **Agentic earnings**: transcript fetch (latest or Y/Q) → `summarize_transcript()` (follows user style); optional `fetch_qoq_fundamentals()` ➜ QoQ box; `analyze_sentiment_gpt()` returns JSON used for visual bars.
- **Metrics page extras**: annual statements & estimates for CAGRs, 5-year averages, forward paths; optional yfinance risk/return.

**Batching**: loops call `_batch_pause()` using `FMP_BATCH_SIZE`/`FMP_BATCH_SLEEP` to play nice with API limits.

---

## ❓ Troubleshooting

- **Empty industry after Step 1** → loosen ranges or switch sector; increase “Max companies”.
- **429 (rate limit)** → reduce **Max tickers**, lower `FMP_BATCH_SIZE`, or increase `FMP_BATCH_SLEEP`.
- **LLM disabled** → set `OPENAI_API_KEY`; otherwise you’ll see fallbacks/heuristics.
- **Metrics page shows old universe** → re-run Step 1 on main page.
- **Price & risk block empty** → enable the yfinance checkbox and ensure tickers are supported.

---

## 🔐 Security & Costs

- Put keys in **Secrets**, not in code.  
- Caching (`@st.cache_data`) reduces repeated API/LLM calls.  
- For research/education only — **not investment advice**.

---

## 🗺️ Roadmap

- Scoring presets (growth/quality/value/momentum) + composite ranks.
- Portfolio optimizer & rolling backtests.
- PDF “Research Brief” export with data citations.
- Tool-using agent that plans and executes multi-step research tasks.

---

## 📄 License

Add a `LICENSE` (e.g., MIT) if you plan to share publicly.

---

## 🙌 Thanks

- Financial Modeling Prep — data APIs  
- Streamlit — rapid UI framework  
- OpenAI — language models  
- yfinance — price history & quick risk stats
