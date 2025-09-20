# app.py — Quick screener + profile summaries (card layout, no filler lines)
import os, time, json, re, requests, pandas as pd, streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from openai import OpenAI

# -------------------- Keys (as provided) --------------------
FMP_API_KEY    = os.getenv("FMP_API_KEY") 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # or "gpt-3.5-turbo"

# -------------------- HTTP session (retries) --------------------
def _session():
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(
        total=3, backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"], raise_on_status=False
    )))
    return s

SESSION = _session()

# -------------------- Screener (graceful) --------------------
SERVER_KEYS = {
    "country","exchange","sector","industry","isActivelyTrading",
    "marketCapMoreThan","marketCapLowerThan",
    "priceMoreThan","priceLowerThan",
    "volumeMoreThan","volumeLowerThan",
    "betaMoreThan","betaLowerThan",
    "dividendMoreThan","dividendLowerThan",
    "limit","page"
}

def _sanitize_params(filters):
    params = {k: v for k, v in (filters or {}).items() if k in SERVER_KEYS and v not in (None, "", [])}
    if isinstance(params.get("isActivelyTrading"), bool):
        params["isActivelyTrading"] = "true" if params["isActivelyTrading"] else "false"
    params["limit"] = max(1, min(int(params.get("limit", 500)), 500))
    return params

@st.cache_data(ttl=3600, show_spinner=False)
def fmp_company_screener_safe(filters, timeout=20, max_pages=10):
    base_url = "https://financialmodelingprep.com/stable/company-screener"
    params_base = _sanitize_params(filters) | {"apikey": FMP_API_KEY}
    all_rows, meta = [], {"errors": [], "warnings": [], "pages": 0}

    for page in range(max_pages):
        try:
            r = SESSION.get(base_url, params={**params_base, "page": page}, timeout=timeout)
            if r.status_code == 429:
                time.sleep(1.0)
                continue
            if r.status_code != 200:
                meta["errors"].append({"page": page, "status": r.status_code})
                break

            data = r.json() or []
            if not data:
                break

            df = pd.DataFrame(data)
            if "symbol" in df.columns:
                df = df.rename(columns={"symbol": "ticker"})
            all_rows.append(df)

            if len(df) < params_base["limit"]:
                meta["pages"] = page + 1
                break
            meta["pages"] = page + 1
        except Exception as e:
            meta["warnings"].append({"page": page, "msg": repr(e)})
            break

    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if "ticker" not in out.columns:
        out["ticker"] = pd.Series(dtype=str)
    if not out.empty:
        out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
        out = out.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return out, meta

# -------------------- Profiles (v3) --------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_profiles_v3_bulk(tickers, timeout=20):
    rows = []
    for t in sorted({(t or "").strip().upper() for t in tickers if t}):
        try:
            r = SESSION.get(
                f"https://financialmodelingprep.com/api/v3/profile/{t}",
                params={"apikey": FMP_API_KEY},
                timeout=timeout
            )
            j = r.json() if r.status_code == 200 else []
            d = (j[0] if j else {}) if isinstance(j, list) else {}
            rows.append({
                "ticker": t,
                "companyName": d.get("companyName"),
                "sector": d.get("sector"),
                "industry": d.get("industry"),
                "description": d.get("description")
            })
            time.sleep(0.04)  # be polite
        except Exception:
            rows.append({"ticker": t, "companyName": None, "sector": None, "industry": None, "description": None})

    df = pd.DataFrame(rows)
    if not df.empty:
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df = df.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    return df

# -------------------- Summarizer (ONLY description, up to 3 lines, no filler) --------------------
def _clamp_lines_to_len(lines, n=200):
    return [ln[:n] for ln in lines]

def summarize_description(desc, name="", ticker="", sector="", industry="", user_prompt=None):
    """
    - Uses ONLY the provided description.
    - Returns up to 3 short lines (no padding or filler).
    - If API fails, falls back to the first 1–2 sentences from the description.
    """
    if not isinstance(desc, str) or not desc.strip():
        return f"{name or ticker or 'Company'} ({ticker})"

    base = f"""
You are a concise company profiler. Using ONLY the text below, {user_prompt or "write up to three short lines (no bullets): 1) what the company does, 2) key products/segments if present, 3) other details if present."}
Do not invent facts. Keep each line under ~200 characters.

Company: {name or ticker} ({ticker or 'N/A'})
Sector: {sector or 'N/A'}; Industry: {industry or 'N/A'}
Text:
\"\"\"{desc[:4000]}\"\"\""""

    try:
        r = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": base}],
            temperature=0.2,
        )
        text = (r.choices[0].message.content or "").strip()
    except Exception:
        parts = re.split(r"(?<=[.!?])\s+", desc.strip())
        parts = [p.strip() for p in parts if p.strip()]
        return "\n".join(parts[:2]) if parts else (name or ticker or "Company")

    # Remove any leftover "additional details..." phrasing if model adds it
    text = re.sub(r"(?i)\badditional details not in source\.?\b", "", text)
    lines = [ln.strip(" -•\t") for ln in text.splitlines() if ln.strip()]
    # Deduplicate lines
    seen, cleaned = set(), []
    for ln in lines:
        if ln and ln not in seen:
            cleaned.append(ln)
            seen.add(ln)
    if not cleaned:
        parts = re.split(r"(?<=[.!?])\s+", desc.strip())
        parts = [p.strip() for p in parts if p.strip()]
        cleaned = parts[:2] if parts else [name or ticker or "Company"]
    cleaned = _clamp_lines_to_len(cleaned, 200)
    return "\n".join(cleaned[:3])

# -------------------- UI --------------------
st.set_page_config(page_title="Quick Company Summaries", layout="wide")
st.title("🧩 Quick Company Summaries (FMP → Profiles → GPT)")

with st.sidebar:
    st.subheader("Screening")
    sector = st.selectbox("Sector", [
        "Energy","Technology","Financial Services","Industrials","Healthcare",
        "Utilities","Materials","Consumer Defensive","Consumer Cyclical",
        "Real Estate","Communication Services"
    ], index=0)
    min_cap_b  = st.slider("Min Market Cap ($B)", 0, 500, 10, 5)
    min_volume = st.slider("Min Avg Volume", 0, 5_000_000, 500_000, 50_000)
    limit      = st.slider("Max companies to fetch", 50, 500, 300, 50)
    top_n      = st.slider("Companies to summarize", 5, 50, 20, 1)
    cards_per_row = st.slider("Cards per row", 1, 3, 2, 1)

    st.subheader("Summarizer prompt")
    user_prompt = st.text_area(
        "Prompt (optional)",
        value="Write up to three short lines: 1) what the company does, 2) key products/segments if present, 3) other details if present.",
        height=90,
    )

    left, right = st.columns(2)
    with left:
        run = st.button("Run screen + summarize")
    with right:
        if st.button("🔁 Regenerate summaries"):
            st.experimental_rerun()

if run:
    with st.spinner("Screening companies..."):
        filters = {
            "country": "US",
            "sector": sector,
            "isActivelyTrading": "true",
            "marketCapMoreThan": int(min_cap_b) * 1_000_000_000,
            "volumeMoreThan": int(min_volume),
            "limit": int(limit),
        }
        screen_df, meta = fmp_company_screener_safe(filters)
        if screen_df.empty:
            st.warning("No tickers returned. Try loosening filters.")
            st.stop()
        st.caption(f"Screened {len(screen_df)} tickers (pages: {meta.get('pages', 0)}).")
        st.dataframe(screen_df[["ticker","companyName","sector","industry"]],
                     use_container_width=True, height=240)

    # Choose first N (you can swap to multiselect later)
    selected = screen_df["ticker"].dropna().unique().tolist()[:top_n]

    with st.spinner(f"Fetching profiles for {len(selected)} companies..."):
        prof = fetch_profiles_v3_bulk(selected)
        if prof.empty:
            st.warning("No profiles found.")
            st.stop()

    with st.spinner("Summarizing descriptions with GPT..."):
        prof["summary_3_lines"] = prof.apply(
            lambda r: summarize_description(
                r.get("description"),
                r.get("companyName"),
                r.get("ticker"),
                r.get("sector"),
                r.get("industry"),
                user_prompt=user_prompt.strip() if user_prompt else None
            ),
            axis=1
        )

    # -------------------- Card / Grid layout (no truncation) --------------------
    st.subheader("Results")
    prof = prof.sort_values("ticker").reset_index(drop=True)
    cols = st.columns(cards_per_row)

    for i, r in prof.iterrows():
        with cols[i % cards_per_row]:
            st.markdown(f"**{r['ticker']} — {r.get('companyName') or ''}**")
            st.markdown(
                f"""
<div style="
  white-space: pre-wrap;
  line-height: 1.35;
  background: #f7f7f9;
  border: 1px solid #e6e6e6;
  border-radius: 10px;
  padding: 12px;
  margin-bottom: 14px;">
{r['summary_3_lines']}
</div>
""",
                unsafe_allow_html=True
            )

    # Export CSV
    show = prof[["ticker","companyName","summary_3_lines"]].copy()
    st.download_button(
        "Download CSV",
        data=show.to_csv(index=False).encode("utf-8"),
        file_name="company_summaries.csv",
        mime="text/csv"
    )
else:
    st.info("Set filters in the sidebar, optionally edit the prompt, then click **Run screen + summarize**.")
