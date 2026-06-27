import streamlit as st
import pandas as pd
import numpy as np

from src.historical_var import historical_var, expected_shortfall
from src.parametric_var import parametric_var
from src.monte_carlo_var import monte_carlo_var
from src.portfolio_var import portfolio_var

from src.stress_testing import historical_stress, hypothetical_stress
from src.liquidity_horizon import liquidity_adjusted_var
from src.backtesting import kupiec_test

st.set_page_config(
    page_title="Market Risk VaR Engine",
    page_icon="📊",
    layout="wide"
)

st.sidebar.title("📊 Market Risk VaR Engine")

dataset_choice = st.sidebar.selectbox(
    "Sample dataset",
    [
        "None",
        "Sample A (fails backtesting)",
        "Sample B (passes backtesting)",
        "Portfolio sample"
    ]
)

uploaded_file = st.sidebar.file_uploader("Upload your own CSV", type=["csv"])

section = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Single Asset VaR",
        "Portfolio VaR",
        "Stress Testing",
        "Liquidity Horizon",
        "Backtesting",
        "Reports"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("Built for risk analytics and model validation demos.")

df = None
portfolio_df = None

if dataset_choice == "Sample A (fails backtesting)":
    df = pd.read_csv("data/prices.csv")
elif dataset_choice == "Sample B (passes backtesting)":
    df = pd.read_csv("data/prices_pass.csv")
elif dataset_choice == "Portfolio sample":
    portfolio_df = pd.read_csv("data/portfolio_prices.csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

st.markdown("""
# 📊 Market Risk VaR Engine

A streamlined risk analytics environment designed to measure market volatility, quantify tail exposure, and evaluate model behavior under varying conditions.  
The engine brings together statistical VaR estimation, distribution‑based risk measures, Monte Carlo simulation, scenario‑driven stress analysis, liquidity horizon scaling under regulatory frameworks, and formal backtesting diagnostics to provide a clear, comprehensive view of risk performance.

Upload a dataset or select a sample to explore how the engine responds to changing market dynamics and how its risk estimates behave under different assumptions.
""")

if df is not None:
    st.success("Single-asset dataset loaded.")
    st.subheader("🔍 Single-asset data preview")
    st.dataframe(df.head())
elif portfolio_df is not None:
    st.success("Portfolio dataset loaded.")
    st.subheader("🔍 Portfolio data preview")
    st.dataframe(portfolio_df.head())
else:
    st.info("Select a sample dataset in the sidebar or upload your own CSV.")

st.markdown("---")

if section == "Single Asset VaR":
    if df is None or "price" not in df.columns:
        st.warning("Single-asset VaR requires a dataset with a 'price' column.")
    else:
        st.header("📈 Single Asset VaR & Expected Shortfall")

        price_series = pd.to_numeric(df["price"], errors="coerce")
        returns = price_series.pct_change().dropna()

        if returns.empty:
            st.warning("No valid returns could be computed from 'price' column.")
        else:
            colA, colB, colC, colD = st.columns(4)

            hvar = historical_var(returns)
            pvar = parametric_var(returns)
            mvar = monte_carlo_var(returns)
            es = expected_shortfall(returns)

            colA.metric("Historical VaR (99%)", f"{hvar:.4f}")
            colB.metric("Parametric VaR (99%)", f"{pvar:.4f}")
            colC.metric("Monte Carlo VaR (99%)", f"{mvar:.4f}")
            colD.metric("Expected Shortfall (99%)", f"{es:.4f}")

            st.markdown("### Returns vs Historical VaR (99%)")

            var_series = pd.Series(hvar, index=returns.index)
            chart_df = pd.DataFrame({
                "Returns": returns,
                "VaR": var_series
            })

            fig = px.line(chart_df, title="Returns vs Historical VaR (99%)")
            fig.add_hline(y=hvar, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

if section == "Portfolio VaR":
    if portfolio_df is None:
        st.warning("Portfolio VaR requires a portfolio dataset (e.g., Date, Asset_A, Asset_B, Asset_C).")
    else:
        st.header("📊 Portfolio VaR")

        price_data = portfolio_df.iloc[:, 1:].select_dtypes(include=["number"])
        returns_port = price_data.pct_change().dropna()

        if returns_port.empty:
            st.warning("Portfolio returns could not be computed.")
        else:
            weights = np.array([1 / len(returns_port.columns)] * len(returns_port.columns))
            port_var = portfolio_var(returns_port, weights)

            st.metric("Portfolio VaR (99%)", f"{port_var:.4f}")

            st.markdown("### Portfolio Asset Returns Heatmap")
            heat_df = returns_port.copy()
            heat_df.index = pd.to_datetime(portfolio_df.iloc[1:, 0].values)

            fig = px.imshow(
                heat_df.T,
                aspect="auto",
                color_continuous_scale="RdBu",
                title="Portfolio Asset Returns Heatmap"
            )
            st.plotly_chart(fig, use_container_width=True)

if section == "Stress Testing":
    if portfolio_df is None:
        st.warning("Stress testing requires a portfolio dataset.")
    else:
        st.header("⚠️ Stress Testing")

        price_data = portfolio_df.iloc[:, 1:].select_dtypes(include=["number"])
        returns_port = price_data.pct_change().dropna()

        if returns_port.empty:
            st.warning("Portfolio returns could not be computed.")
        else:
            base_series = returns_port.iloc[:, 0]

            hist_stress = historical_stress(base_series, (0, 5))
            hypo_stress = hypothetical_stress(base_series, -0.10)

            col1, col2 = st.columns(2)
            col1.metric("Historical Stress (first 5 days)", f"{hist_stress:.4f}")
            col2.metric("Hypothetical Stress (-10%)", f"{hypo_stress:.4f}")

            st.markdown("### Base Returns (Stress Window Highlighted)")

            stress_df = pd.DataFrame({
                "Returns": base_series
            })

            fig = px.line(stress_df, y="Returns", title="Base Returns")
            st.plotly_chart(fig, use_container_width=True)

if section == "Liquidity Horizon":
    if portfolio_df is None:
        st.warning("Liquidity Horizon requires a portfolio dataset.")
    else:
        st.header("⏳ Liquidity Horizon (FRTB)")

        price_data = portfolio_df.iloc[:, 1:].select_dtypes(include=["number"])
        returns_port = price_data.pct_change().dropna()

        if returns_port.empty:
            st.warning("Portfolio returns could not be computed.")
        else:
            weights = np.array([1 / len(returns_port.columns)] * len(returns_port.columns))
            port_var = portfolio_var(returns_port, weights)

            lh_20 = liquidity_adjusted_var(port_var, 20)
            lh_40 = liquidity_adjusted_var(port_var, 40)
            lh_60 = liquidity_adjusted_var(port_var, 60)

            col1, col2, col3 = st.columns(3)
            col1.metric("LH 20-day VaR", f"{lh_20:.4f}")
            col2.metric("LH 40-day VaR", f"{lh_40:.4f}")
            col3.metric("LH 60-day VaR", f"{lh_60:.4f}")

            st.markdown("### Liquidity-Adjusted VaR vs Horizon")
            lh_df = pd.DataFrame({
                "Horizon (days)": [20, 40, 60],
                "Liquidity-Adjusted VaR": [lh_20, lh_40, lh_60]
            })
            fig = px.bar(
                lh_df,
                x="Horizon (days)",
                y="Liquidity-Adjusted VaR",
                title="Liquidity-Adjusted VaR vs Horizon"
            )
            st.plotly_chart(fig, use_container_width=True)

if section == "Backtesting":
    if df is None or "price" not in df.columns:
        st.warning("Backtesting requires a single-asset dataset with a 'price' column.")
    else:
        st.header("📉 Backtesting (Kupiec Test)")

        price_series = pd.to_numeric(df["price"], errors="coerce")
        returns = price_series.pct_change().dropna()

        if returns.empty:
            st.warning("No valid returns could be computed from 'price' column.")
        else:
            hvar = historical_var(returns)
            var_series = pd.Series(hvar, index=returns.index)

            results = kupiec_test(returns, var_series)

            passed = results["passed"]
            exceptions = results["exceptions"]
            lr_stat = results["LR_statistic"]

            col1, col2, col3 = st.columns(3)
            col1.metric("Exceptions", exceptions)
            col2.metric("LR Statistic", f"{lr_stat:.4f}")
            col3.metric("Passed Test", "✅ Pass" if passed else "❌ Fail")

            if passed:
                st.success("Backtesting result: Model is statistically consistent with the expected breach rate.")
            else:
                st.error("Backtesting result: Model is not statistically consistent with the expected breach rate.")

            st.markdown("### Returns vs VaR (Backtesting)")
            bt_df = pd.DataFrame({
                "Returns": returns,
                "VaR": var_series
            })
            fig = px.line(bt_df, title="Returns vs VaR (Backtesting)")
            fig.add_hline(y=hvar, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

if section == "Reports":
    st.header("📄 Risk Report (Summary)")

    st.markdown("""
This section summarizes the behavior of the risk engine across:
- Single-asset VaR and Expected Shortfall
- Portfolio VaR and cross-asset return structure
- Stress testing scenarios
- Liquidity horizon scaling
- Backtesting diagnostics

Use this view as a narrative starting point for model documentation, validation notes, or interview discussion.
""")
