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

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Market Risk VaR Engine",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("""
# 📊 Market Risk VaR Engine  
A professional dashboard for market risk analytics:
- **Value-at-Risk (VaR)**
- **Expected Shortfall (ES)**
- **Monte Carlo Simulation**
- **Stress Testing & Scenario Analysis**
- **Liquidity Horizon (FRTB)**
- **Backtesting (Kupiec Test)**
- **Risk Visualization (Charts)**  
""")

st.divider()

# ---------------------------------------------------------
# UNIVERSAL FILE LOADER
# ---------------------------------------------------------
def load_data(uploaded_file):
    filename = uploaded_file.name.lower()

    # Excel support
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        try:
            uploaded_file.seek(0)
            return pd.read_excel(uploaded_file)
        except Exception:
            st.error("❌ Unable to read Excel file.")
            return None

    # CSV support attempts
    encodings = ["utf-8", "ISO-8859-1", "utf-16"]
    for enc in encodings:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=enc)
        except Exception:
            pass

    # Auto delimiter detection
    try:
        uploaded_file.seek(0)
        raw = uploaded_file.read().decode("utf-8", errors="ignore")
        import csv
        dialect = csv.Sniffer().sniff(raw.split("\n")[0])
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, delimiter=dialect.delimiter)
    except Exception:
        pass

    # Python engine fallback
    try:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, engine="python", on_bad_lines="skip")
    except Exception:
        pass

    st.error("❌ Unable to parse this file. Please upload a valid CSV or Excel file.")
    return None

# ---------------------------------------------------------
# DATA INPUT
# ---------------------------------------------------------
st.header("📁 Data Input")

col1, col2 = st.columns([1, 2])

with col1:
    use_sample = st.button("📘 Use Sample Data")
    uploaded_file = st.file_uploader("Upload CSV or Excel price/portfolio data", type=["csv", "xlsx", "xls"])

# ---------------------------------------------------------
# SESSION STATE FIX (PREVENTS APP RESTARTS)
# ---------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

# Load sample or uploaded file into session state
if use_sample:
    st.session_state.df = pd.read_csv("data/prices.csv")

elif uploaded_file is not None:
    df_temp = load_data(uploaded_file)
    if df_temp is not None:
        df_temp.columns = (
            df_temp.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace("\ufeff", "", regex=False)
            .str.replace("﻿", "", regex=False)
        )
        if len(df_temp.columns) == 2 and "price" not in df_temp.columns:
            df_temp.columns = ["date", "price"]
        st.session_state.df = df_temp

# Always use df from session state
df = st.session_state.df

# ---------------------------------------------------------
# DATA PREVIEW
# ---------------------------------------------------------
if df is None:
    st.info("Upload a CSV or Excel file, or click **Use Sample Data** to begin.")
    st.stop()

st.subheader("🔍 Data Preview")
st.dataframe(df.head())
st.divider()

# ---------------------------------------------------------
# SINGLE ASSET RISK
# ---------------------------------------------------------
if "price" in df.columns:

    st.header("📈 Single Asset Risk Metrics")

    price_series = pd.to_numeric(df["price"], errors="coerce")
    returns = price_series.pct_change().dropna()

    if not returns.empty:
        colA, colB, colC, colD = st.columns(4)

        colA.metric("Historical VaR (99%)", f"{historical_var(returns):.4f}")
        colB.metric("Parametric VaR (99%)", f"{parametric_var(returns):.4f}")
        colC.metric("Monte Carlo VaR (99%)", f"{monte_carlo_var(returns):.4f}")
        colD.metric("Expected Shortfall (99%)", f"{expected_shortfall(returns):.4f}")

        st.divider()

        # Charts
        st.subheader("📊 Single Asset Charts")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Price Time Series**")
            st.line_chart(price_series)

        with col2:
            st.markdown("**Return Time Series**")
            st.line_chart(returns)

        st.divider()

# ---------------------------------------------------------
# PORTFOLIO RISK
# ---------------------------------------------------------
if len(df.columns) > 2:

    st.header("📊 Portfolio VaR")

    price_data = df.iloc[:, 1:].select_dtypes(include=["number"])
    returns_port = price_data.pct_change().dropna()

    if not returns_port.empty:

        weights = np.array([1 / len(price_data.columns)] * len(price_data.columns))
        port_var = portfolio_var(returns_port, weights)

        st.metric("Portfolio VaR (99%)", f"{port_var:.4f}")
        st.divider()

        # Portfolio charts
        st.subheader("📊 Portfolio Charts")

        port_prices = price_data.dot(weights)
        port_returns = returns_port.dot(weights)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Portfolio Price Time Series**")
            st.line_chart(port_prices)

        with col2:
            st.markdown("**Portfolio Return Time Series**")
            st.line_chart(port_returns)

        st.divider()

        # ---------------------------------------------------------
        # STRESS TESTING
        # ---------------------------------------------------------
        st.header("⚠️ Stress Testing")

        col1, col2 = st.columns(2)

        with col1:
            hist_stress = historical_stress(returns_port.iloc[:, 0], (0, 5))
            st.metric("Historical Stress (first 5 days)", f"{hist_stress:.4f}")

        with col2:
            hypo_stress = hypothetical_stress(returns_port.iloc[:, 0], -0.10)
            st.metric("Hypothetical Stress (-10%)", f"{hypo_stress:.4f}")

        st.divider()

        # ---------------------------------------------------------
        # CUSTOM STRESS SCENARIOS
        # ---------------------------------------------------------
        st.header("🧨 Custom Stress Scenarios")

        st.markdown("Define shocks to each asset (in %) and see the impact on portfolio value and VaR.")

        shock_cols = st.columns(len(price_data.columns))
        shocks = []

        for i, col_name in enumerate(price_data.columns):
            with shock_cols[i]:
                shock = st.slider(
                    f"{col_name} shock (%)",
                    min_value=-50.0,
                    max_value=50.0,
                    value=0.0,
                    step=1.0
                )
                shocks.append(shock / 100.0)

        shocks = np.array(shocks)

        last_prices = price_data.iloc[-1].values
        stressed_prices = last_prices * (1 + shocks)

        base_port_value = np.dot(last_prices, weights)
        stressed_port_value = np.dot(stressed_prices, weights)
        stressed_loss = (base_port_value - stressed_port_value) / base_port_value

        st.markdown("### 📉 Stressed Portfolio Impact")
        col1, col2, col3 = st.columns(3)

        col1.metric("Base Portfolio Value", f"{base_port_value:.2f}")
        col2.metric("Stressed Portfolio Value", f"{stressed_port_value:.2f}")
        col3.metric("One-Day Stressed Loss", f"{stressed_loss:.4%}")

        avg_abs_shock = np.mean(np.abs(shocks))
        stressed_var = port_var * (1 + avg_abs_shock)

        st.metric("Approx. Stressed VaR (99%)", f"{stressed_var:.4f}")
        st.divider()

        # ---------------------------------------------------------
        # LIQUIDITY HORIZON
        # ---------------------------------------------------------
        st.header("⏳ Liquidity Horizon (FRTB)")

        lh_var = liquidity_adjusted_var(port_var, 20)
        st.metric("Liquidity-Adjusted VaR (20-day)", f"{lh_var:.4f}")

        st.divider()

        # ---------------------------------------------------------
        # BACKTESTING
        # ---------------------------------------------------------
        st.header("📉 Backtesting (Kupiec Test)")

        var_series = pd.Series(historical_var(returns_port), index=returns_port.index)
        results = kupiec_test(returns_port, var_series)

        col1, col2, col3 = st.columns(3)

        col1.metric("Exceptions", results["exceptions"])
        col2.metric("LR Statistic", f"{results['LR_statistic']:.4f}")
        col3.metric("Passed Test", "✅ Yes" if results["passed"] else "❌ No")

        st.divider()
