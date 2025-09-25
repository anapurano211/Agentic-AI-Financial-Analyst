# Agentic AI Financial Analyst — Streamlit App

One‑click stock screening ➜ metrics filtering ➜ industry selection ➜ concise company profiles ➜ **agentic** earnings summaries & sentiment.  
Built for expansion into a research assistant with scoring, portfolio optimization, and backtesting.

> Live demo (example Space): https://huggingface.co/spaces/andrewnap211/agentic-ai-financial-analyst

---

## ✨ What’s New (since v0.1)

- **Multi‑page app**: new page `pages/02_Metrics_&_Performance.py` for cross‑ticker metric pulls and an *agentic* “Competitive Advantages” analysis prompt.
- **Two‑step workflow**: earnings only run **after** you choose an **Industry** (prevents wasting LLM calls on the wrong universe).
- **Range sliders** for **Market Cap** and **Average Volume** (min & max) in the screener.
- **Agentic earnings summarizer**: user controls bullets/lines, count, focus/exclusions, and optional QoQ facts (Revenue, NI, FCF).
- **State sharing across pages**: ticker universe from the main page is available on the Metrics page via `st.session_state`.

---

## 🚦 Current Features

### Main page (`app.py`)
- **Server‑side screening (FMP)**: sector + market‑cap/volume ranges, active trading, pagination, retries.
- **Optional advanced filters**: valuation (PE/PS/PB), quality/leverage (D/E, Current), momentum (RSI, MA position), profitability (ROE/ROA/ROIC, margins), dividend yield.
- **Industry gating**: after metrics, pick **Industry** ➜ then run profiles & earnings.
- **Profiles (FMP v3)**: company, sector/industry, description.
- **LLM profile summary (OpenAI)**: 1–3 short lines with no external facts.
- **Agentic earnings**: transcript fetch (latest or Y/Q), custom instruction, bullets vs lines, include/exclude topics, optional QoQ box; JSON sentiment (overall + subscores + drivers).
- **Card UI + CSV export**.

### Metrics & Performance page (`pages/02_Metrics_&_Performance.py`)
- **Pull metrics for the chosen universe** (or paste your own tickers).
- **Agentic “Competitive Advantages” prompt**: GPT analyzes the current metrics table to narrate moat/edge themes (valuation, growth, margins, balance sheet, momentum).

---

## 🧱 Project Structure

```
app.py                          # Main: Screener → Metrics → Industry → Profiles → Agentic Earnings/Sentiment
pages/
  └─ 02_Metrics_&_Performance.py # Secondary: metrics fetch + agentic competitive-advantages analysis
requirements.txt
README.md
```

> Streamlit treats anything in `pages/` as a separate page. The numeric prefix controls menu order.

---

## 🔌 Data & Models

| Source | Purpose | Endpoints (examples) |
|---|---|---|
| **Financial Modeling Prep (FMP)** | Screener + profiles + metrics | `/stable/company-screener`, `/api/v3/profile/{ticker}`, `/api/v3/ratios-ttm/{ticker}`, `/api/v3/key-metrics-ttm/{ticker}`, `/api/v3/earning_call_transcript/{ticker}` |
| **OpenAI** | LLM summaries & agentic analysis | Chat Completions (`gpt-4o-mini` default) |

> The app strictly instructs the LLM to **use only provided text** for summaries; no outside facts.

---

## ⚙️ Configuration

Provide keys via environment variables (or HF Space Secrets):

- `FMP_API_KEY`
- `OPENAI_API_KEY`
- Optional: `OPENAI_MODEL` (default: `gpt-4o-mini`)

---

## ▶️ Run Locally

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt

# Set keys (example)
set FMP_API_KEY=your_fmp_key         # Windows PowerShell: $env:FMP_API_KEY="..."
set OPENAI_API_KEY=sk-...

streamlit run app.py
```

Open the URL printed by Streamlit (usually http://localhost:8501).

---

## 🚀 Deploy to Hugging Face Spaces

1. Create a **Streamlit** Space.
2. Upload: `app.py`, `pages/02_Metrics_&_Performance.py`, `requirements.txt`, `README.md`.
3. In **Settings → Repository secrets**, add `FMP_API_KEY` and `OPENAI_API_KEY`.
4. The build will run and serve automatically.

Minimal `requirements.txt`:

```
streamlit>=1.31
pandas>=2.0
requests>=2.31
urllib3>=1.26,<3
openai>=1.37.0
```

---

## 🧭 Using the App

### Main Page – two steps
1. **Step 1 — Run screener + metrics**  
   - Choose *Sector*, *Market Cap Range*, *Avg Volume Range*, and *Max companies*.
   - (Optional) enable *Advanced Metrics* and set your ranges.
   - Click **Step 1** to see the screened table and (optionally) metrics table.
2. **Industry filter**  
   - Pick an **Industry** (e.g., “Semiconductors”). The selection trims the universe.
3. **Step 2 — Profiles + Earnings**  
   - Choose transcript mode (Latest or Specific Y/Q) and write your instruction (e.g., “7 bullets on guidance, margins, demand; add QoQ”).  
   - Click **Step 2** to render company cards with profiles, QoQ snapshot (if requested), LLM summary, and sentiment bars.  
   - Download a CSV of the outputs.

### Metrics & Performance Page
- The page reads the current universe from `st.session_state.step1.filtered_tickers`.  
- You can also paste a custom list of tickers.  
- Click **Fetch metrics**.  
- Use the **Agentic Competitive Advantages** box to ask narrative questions like:
  - “Who looks advantaged on growth + margins while still reasonably valued?”  
  - “Contrast cash generation and leverage risks for the top 5 by ROIC.”

> This analysis works on the **metrics table currently in memory**; it won’t fetch new data unless you click **Fetch metrics** again.

---

## 🧠 How It Works (high level)

- **Screener** → `fmp_company_screener_safe()` handles retries and pagination; normalizes symbols.
- **Profiles** → `fetch_profiles_v3_bulk()` calls `profile/{ticker}` and clips description length for tokens.
- **Metrics** → `fetch_all_metrics()` merges `profile`, `quote`, `ratios-ttm`, `key-metrics-ttm`, and `RSI`; computes relative price positions.
- **Agentic Earnings** → `summarize_transcript()` shapes the output per user instruction; optional `fetch_qoq_fundamentals()` adds Revenue/NI/FCF QoQ; `analyze_sentiment_gpt()` returns a JSON blob (overall + subscores + drivers).

---

## ❓ Troubleshooting

- **No industries listed after Step 1** → your screen returned zero tickers; loosen ranges or sector.
- **Stale universe on Metrics page** → re‑run Step 1 on the main page; the page reads from `st.session_state`.
- **FMP 429 or empty payload** → try reducing batch size or run again; the app already retries.
- **OpenAI errors** → ensure `OPENAI_API_KEY` is set; summaries/sentiment gracefully fall back but will be generic.
- **Broken layout** → make sure both `app.py` and `pages/02_Metrics_&_Performance.py` are present.

---

## 🔐 Security & Costs

- Keep keys in **Secrets**, not in code.
- Caching (`@st.cache_data`) reduces both API calls and LLM cost.
- This tool is for research/education only — **not financial advice**.

---

## 🗺️ Roadmap

- Metrics scoring & presets (growth, quality, value, momentum).
- Portfolio optimizer and rolling backtests.
- Export as PDF “Research Brief” with data citations.
- Tool‑calling agent that plans and executes multi‑step analyses.

---

## 📄 License

Choose a license that fits your usage (e.g., MIT). Add `LICENSE` to the repo if you plan to share publicly.

---

## 🙌 Thanks

- Financial Modeling Prep — data APIs  
- Streamlit — rapid UI framework  
- OpenAI — language models
