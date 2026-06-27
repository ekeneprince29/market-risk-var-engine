import streamlit as st
import pandas as pd
import numpy as np

from historical_var import historical_var
from parametric_var import parametric_var
from monte_carlo_var import monte_carlo_var
from monte_carlo_correlated import monte_carlo_correlated
from portfolio_var import compute_portfolio_var
from stress_testing import historical_stress, hypothetical_stress, custom_stress
from liquidity_horizon import liquidity_adjusted_var
from backtesting import kupiec_test

# ---------------------------------------------------------
# TITLE + INTRO
# ---------------------------------------------------------
st.title("📊 Market Risk Analytics Platform")

st.write("""
Institutional‑grade dashboard for Market Risk, Treasury Risk, and Trading Controls.  
Includes VaR, Expected Shortfall, Monte Carlo Simulation, Stress Testing, Liquidity Horizon (FRTB),  
Backtesting, and full portfolio analytics with interactive visualizations.
""")

st.divider()

# ---------------------------------------------------------
# DATA INPUT
# ---------------------------------------------------------
st.header("📁 Data Input — Upload CSV or Excel price/portfolio data")
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.divider()

    # ---------------------------------------------------------
    # DATA PREVIEW
    # ---------------------------------------------------------
    st.header("🔍 Data Preview")
    st.dataframe(df.head())

    st.divider()

    # ---------------------------------------------------------
    # SINGLE ASSET RISK METRICS
    # ---------------------------------------------------------
    st.header("📈 Single Asset Risk Metrics")
    price_col = df.select_dtypes(include=["number"]).columns[0]
    prices = df[price_col]
    returns = prices.pct_change().dropna()

    hist_var = historical_var(returns)
    param_var = parametric_var(returns)
    mc_var = monte_carlo_var(returns)
    es_value = returns[returns < hist_var].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Historical VaR (99%)", f"{hist_var:.4f}")
    col2.metric("Parametric VaR (99%)", f"{param_var:.4f}")
    col3.metric("Monte Carlo VaR (99%)", f"{mc_var:.4f}")
    col4.metric("Expected Shortfall (99%)", f"{es_value:.4f}")

    # ---------------------------------------------------------
    # SINGLE ASSET CHARTS
    # ---------------------------------------------------------
    st.subheader("📊 Single Asset Charts")
    st.line_chart(prices, height=250, use_container_width=True)
    st.line_chart(returns, height=250, use_container_width=True)

    st.divider()

    # ---------------------------------------------------------
    # PORTFOLIO VaR
    # ---------------------------------------------------------
    st.header("📊 Portfolio VaR (Multi‑Asset)")
    numeric_cols = df.select_dtypes(include=["number"])
    port_returns = numeric_cols.pct_change().dropna()

    portfolio_var_value = compute_portfolio_var(port_returns)

    st.metric("Portfolio VaR (99%)", f"{portfolio_var_value:.4f}")

    # Portfolio charts
    st.subheader("Portfolio Price Time Series")
    st.line_chart(numeric_cols, height=250, use_container_width=True)

    st.subheader("Portfolio Return Time Series")
    st.line_chart(port_returns, height=250, use_container_width=True)

    st.divider()

    # ---------------------------------------------------------
    # STRESS TESTING
    # ---------------------------------------------------------
    st.header("⚠️ Stress Testing & Scenario Analysis")

    hist_stress_value = historical_stress(prices)
    hypo_stress_value = hypothetical_stress(prices, -0.10)

    col1, col2 = st.columns(2)
    col1.metric("Historical Stress (first 5 days)", f"{hist_stress_value:.4f}")
    col2.metric("Hypothetical Stress Scenario (–10%)", f"{hypo_stress_value:.4f}")

    st.subheader("🧨 Custom Stress Scenarios — Asset‑Level Shocks")
    shocks = {}
    for col in numeric_cols.columns:
        shocks[col] = st.slider(f"{col} shock (%)", -50.0, 50.0, 0.0)

    stressed_value, base_value = custom_stress(numeric_cols, shocks)

    st.header("📉 Stressed Portfolio Impact")
    st.metric("Base Portfolio Value", f"{base_value:.2f}")
    st.metric("Stressed Portfolio Value", f"{stressed_value:.2f}")
    st.metric("One‑Day Stressed Loss", f"{(stressed_value - base_value) / base_value:.4%}")
    st.metric("Approx. Stressed VaR (99%)", f"{portfolio_var_value:.4f}")

    st.divider()

    # ---------------------------------------------------------
    # LIQUIDITY HORIZON (FRTB)
    # ---------------------------------------------------------
    st.header("⏳ Liquidity Horizon (FRTB)")
    lh_var = liquidity_adjusted_var(portfolio_var_value, horizon_days=20)
    st.metric("Liquidity‑Adjusted VaR (20‑day)", f"{lh_var:.4f}")

    st.divider()

    # ---------------------------------------------------------
    # BACKTESTING — KUPIEC TEST
    # ---------------------------------------------------------
    st.header("📉 Backtesting — Kupiec Proportion of Failures Test")

    var_series = pd.Series(portfolio_var_value, index=port_returns.index)
    results = kupiec_test(port_returns.mean(axis=1), var_series)

    col1, col2, col3 = st.columns(3)
    col1.metric("Exceptions", results["exceptions"])
    col2.metric("Likelihood Ratio Statistic", f"{results['LR_statistic']:.4f}")
    col3.metric("Backtest Result", "✅ Passed" if results["passed"] else "❌ Failed")

    st.divider()

    # ---------------------------------------------------------
    # METHODOLOGY
    # ---------------------------------------------------------
    with st.expander("📘 Methodology"):
        st.write("""
**Historical VaR** — Non‑parametric percentile of historical returns  
**Parametric VaR** — Variance‑covariance method assuming normal distribution  
**Monte Carlo VaR** — Simulated returns using normal distribution  
**Monte Carlo (Correlated)** — Cholesky decomposition applied to covariance matrix  
**Expected Shortfall** — Average tail loss beyond VaR threshold  
**Stress Testing** — Historical and hypothetical shocks applied to asset prices  
**Liquidity Horizon (FRTB)** — Regulatory liquidity adjustments to VaR  
**Kupiec Test** — Proportion‑of‑Failures backtest evaluating VaR accuracy  
""")

else:
    st.info("Upload a CSV or Excel file to begin analysis.")
