import streamlit as st
import pandas as pd
import json
import plotly.express as px
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Thermodynamic Computing", layout="wide")
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #555; margin-bottom: 2rem; }
    .universe-title { font-size: 1.5rem; font-weight: 600; margin-top: 1rem; margin-bottom: 1rem; padding-left: 0.5rem; border-left: 5px solid #1f77b4; }
    .etf-card { background: linear-gradient(135deg, #1f77b4 0%, #2c3e50 100%); color: white; border-radius: 15px; padding: 1rem; margin: 0.5rem; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
    .etf-ticker { font-size: 1.3rem; font-weight: bold; }
    .etf-score { font-size: 0.9rem; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🔥 Thermodynamic Computing Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Portfolio annealing | Free energy minimisation | Boltzmann distribution | Annealing path | Multi‑window evaluation</div>', unsafe_allow_html=True)

st.sidebar.markdown("## 🔥 Thermodynamic")
st.sidebar.markdown(f"**Run Date:** `{st.session_state.get('run_date', 'Not loaded')}`")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**T_init:** {config.T_INIT} | **T_final:** {config.T_FINAL} | **Steps:** {config.T_STEPS}")
st.sidebar.markdown("**Windows evaluated:** 63, 252, 504, 1008, 2016 days (best per ETF)")

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'thermodynamic_' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error: {data['error']}")
    st.stop()

st.session_state['run_date'] = data['run_date']
universes = data["universes"]

st.header("🏆 Top ETFs by Thermodynamic Portfolio Weight (Lowest Temperature)")

with st.expander("📖 Interpretation", expanded=True):
    st.markdown("""
    - Portfolio optimisation is modelled as a physical annealing process.
    - **Free energy** \(F = U - TS\) where \(U\) = negative expected return, \(T\) = risk aversion (temperature), \(S\) = portfolio entropy.
    - Equilibrium weights are given by the **Boltzmann distribution**: \(w_i \propto \exp(-\mu_i / T)\) where \(\mu_i\) is the expected return.
    - Annealing starts at high temperature (exploration) and gradually cools to low temperature (exploitation).
    - At low temperature, the portfolio concentrates on the ETF with the highest expected return.
    - The **score** is the portfolio weight at the final temperature.
    - For each ETF, the rolling window that gives the **highest weight** is selected.
    """)

for universe_name, uni_data in universes.items():
    top_etfs = uni_data.get("top_etfs", [])
    if not top_etfs:
        continue
    st.markdown(f'<div class="universe-title">{universe_name.replace("_", " ").title()}</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, etf in enumerate(top_etfs):
        with cols[idx]:
            st.markdown(f"""
            <div class="etf-card">
                <div class="etf-ticker">{etf['ticker']}</div>
                <div class="etf-score">weight = {etf['weight']:.4f}</div>
                <div class="etf-score">best window = {etf.get('best_window', 'N/A')}d</div>
            </div>
            """, unsafe_allow_html=True)
    # Show annealing path for the best window
    win_res = uni_data.get("window_results", {})
    if win_res:
        best_win = top_etfs[0]['best_window'] if top_etfs else None
        if best_win is not None and str(best_win) in win_res:
            history = win_res[str(best_win)].get("history", [])
            if history:
                temps = [h[0] for h in history]
                # Free energies
                free_energies = [h[2] for h in history]
                # Top ETF weight over temperature (e.g., weight of the first top ETF)
                top_ticker = top_etfs[0]['ticker']
                top_weights = []
                for _, w_list, _ in history:
                    # w_list is list of weights for all ETFs
                    # Find index of top_ticker
                    idx = list(returns_df.columns).index(top_ticker) if top_ticker in returns_df.columns else 0
                    top_weights.append(w_list[idx] if idx < len(w_list) else 0)
                fig1 = px.line(x=temps, y=free_energies, log_x=True, title=f"Free Energy vs Temperature (annealing)", labels={'x':'Temperature', 'y':'Free Energy'})
                fig2 = px.line(x=temps, y=top_weights, log_x=True, title=f"Weight of {top_ticker} vs Temperature", labels={'x':'Temperature', 'y':'Weight'})
                st.plotly_chart(fig1, use_container_width=True)
                st.plotly_chart(fig2, use_container_width=True)
    with st.expander("📋 Full ranking (all ETFs, best window per ETF)"):
        full = uni_data.get("full_scores", {})
        if full:
            rows = []
            for ticker, info in full.items():
                if isinstance(info, dict):
                    weight = info.get("weight", 0.0)
                    win = info.get("best_window", "N/A")
                else:
                    weight = info
                    win = "N/A"
                rows.append({"ETF": ticker, "Portfolio Weight": weight, "Best Window": win})
            df = pd.DataFrame(rows)
            df["Portfolio Weight"] = pd.to_numeric(df["Portfolio Weight"], errors='coerce')
            df = df.dropna(subset=["Portfolio Weight"]).sort_values("Portfolio Weight", ascending=False)
            st.dataframe(df, use_container_width=True, hide_index=True)
    st.divider()

st.caption("Thermodynamic computing: portfolio weights minimise free energy via simulated annealing. Higher weight → larger allocation in the optimal portfolio → overweight signal.")
