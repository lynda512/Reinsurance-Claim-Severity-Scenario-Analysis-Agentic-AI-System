
# src/app/streamlit_app.py
import streamlit as st
import pandas as pd
import os
import json
from pathlib import Path
BASE = Path(__file__).resolve().parents[2]
DATA_PATH = BASE / "data_out" / "synthetic_reinsurance.csv"
MODEL_PATH = BASE / "models" / "severity_model.pkl"

st.set_page_config(page_title="Reinsurance Agentic AI Demo", layout="wide")

st.title("Reinsurance Claim Severity & Scenario Analysis — Agentic AI Demo")
st.markdown("Demo pipeline: data quality → model → agentic analysis → dashboard")

# Sidebar
st.sidebar.header("Actions")
if st.sidebar.button("Generate fresh synthetic data (25000)"):
    import subprocess, sys
    subprocess.run([sys.executable, os.path.join("data", "generate_data.py")])
    st.sidebar.success("Regenerated data. Refresh the page.")

# Load data
if not DATA_PATH.exists():
    st.warning("No dataset found. Generate data first with `python data/generate_data.py`.")
else:
    df = pd.read_csv(DATA_PATH, parse_dates=['inception_date','loss_date','reported_date'])
    st.sidebar.success(f"Loaded dataset with {len(df):,} rows")

# Tabs
tabs = st.tabs(["Data Quality", "Modeling", "Scenario Simulator", "Agent Chat", "Raw Data"])

with tabs[0]:
    st.header("Data Quality Report")
    from src.data_quality import data_quality_report
    report = data_quality_report(df)
    st.json(report)
    # show missingness table
    missing = (df.isnull().mean()*100).round(2).sort_values(ascending=False)
    st.subheader("Missingness (%)")
    st.table(missing.to_frame("missing_pct"))

with tabs[1]:
    st.header("Train & Evaluate Model (Local)")
    st.write("Train model if not present. Training may take up to a minute.")
    if st.button("Train model now"):
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "src.train_model", "--data", str(DATA_PATH), "--model_out", str(MODEL_PATH)])
        st.experimental_rerun()
    if MODEL_PATH.exists():
        st.success("Model found.")
        st.write("Make quick predictions on a sample")
        idx = st.slider("Sample size for prediction", 1, 200, 20)
        sample = df.sample(idx, random_state=42)
        from src.agent.tools import run_model_on_records
        preds = run_model_on_records(sample.to_dict(orient='records'))
        sample2 = sample.copy()
        sample2['predicted_severity'] = preds
        st.dataframe(sample2[['policy_id','line_of_business','exposure','catastrophe_flag','severity','predicted_severity']].head(50))
    else:
        st.info("Model not trained yet. Click 'Train model now' to train.")

with tabs[2]:
    st.header("Scenario Simulator")
    shock = st.slider("Shock to exposure (%)", 0, 200, 20)
    lob_filter = st.selectbox("Filter by Line of Business (optional)", options=["(all)"] + sorted(df['line_of_business'].unique().tolist()))
    subset = None if lob_filter == "(all)" else {'line_of_business': lob_filter}
    from src.agent.tools import simulate_scenario
    if st.button("Run scenario"):
        with st.spinner("Running scenario..."):
            sim = simulate_scenario(shock_pct=shock/100.0, subset_filters=subset, sample_size=200)
            st.json(sim)

with tabs[3]:
    st.header("Agent Chat (Tool-enabled)")
    st.write("Ask the agent to run data quality checks, predict, or run scenario analyses. Examples: 'Run a 20% exposure shock for Property LOB and summarize results.'")
    user_input = st.text_area("Your prompt", value="Summarize recent data quality and run a 20% shock scenario for Property.")
    if st.button("Run Agent"):
        from src.agent.agent import run_agent
        with st.spinner("Agent running..."):
            resp = run_agent(user_input)
            st.subheader("Agent Summary")
            st.write(resp['summary'])
            st.subheader("Tool outputs")
            st.json(resp['tool_outputs'])

with tabs[4]:
    st.header("Raw Data (sample)")
    st.dataframe(df.sample(200, random_state=42))
