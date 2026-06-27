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
# INTRO
# ---------------------------------------------------------
st.title("📊 Market Risk VaR Engine")

st.markdown("""
A streamlined risk analytics environment designed to measure market volatility, quantify tail exposure, and evaluate model behavior under varying conditions.  
Upload a dataset or use one of the two included price files to explore how the engine responds to changing market dynamics.
""")

# ---------------------------------------------------------
# DATA INPUT
# ---------------------------------------------------------
st.header("Data Input")

uploaded_file = st.file_uploader("Upload CSV price data", type=["csv"])

df = None

# Two sample datasets only
load1 = st.button("Load Price Data 1")
load2 = st.button("Load Price Data 2")

if load1:
    df = pd.read_csv("data/prices.csv")
    st.success("Loaded Price Data 1")

if load2:
    df = pd.read_csv("data/prices_pass.csv")
    st.success("Loaded Price Data 2")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Uploaded custom CSV data")

if df is not None:
    st.subheader("Data Preview")
    st.dataframe(df.head())
    st.divider()

# ---------------------------------------------------------
# SINGLE ASSET RISK
# ---------------------------------------------------------
if df is not None and "price" in df.columns:

    st.header("Single Asset Risk Metrics")

    price_series = pd.to_numeric(df["price"], errors="coerce")
    returns = price_series.pct_change().dropna()

    if returns.empty:
        st.warning("No valid returns could be computed from 'price' column.")
    else:
        hvar = historical_var(returns)
        pvar = parametric_var(returns)
        mvar = monte_carlo_var(returns)
        es = expected_shortfall(returns)

        st.write(f"**Historical VaR (99%)**: {hvar:.4f}")
        st.write(f"**Parametric VaR (99%)**: {pvar:.4f}")
        st.write(f"**Monte Carlo VaR (99%)**: {mvar:.4f}")
        st.write(f"**Expected Shortfall (99%)**: {es:.4f}")

        st.divider()

# ---------------------------------------------------------
# PORTFOLIO VaR
# ---------------------------------------------------------
if df is not None and len(df.columns) > 2:

    st.header("Portfolio VaR")

    price_data = df.iloc[:, 1:].select_dtypes(include=["number"])
    returns_port = price_data.pct_change().dropna()

    if returns_port.empty:
        st.warning("Portfolio returns could not be computed.")
    else:
        weights = np.array([1 / len(returns_port.columns)] * len(returns_port.columns))
        port_var = portfolio_var(returns_port, weights)

        st.write(f"**Portfolio VaR (99%)**: {port_var:.4f}")
        st.divider()

        # ---------------------------------------------------------
        # STRESS TESTING
        # ---------------------------------------------------------
        st.header("Stress Testing")

        hist_stress = historical_stress(returns_port.iloc[:, 0], (0, 5))
        hypo_stress = hypothetical_stress(returns_port.iloc[:, 0], -0.10)

        st.write(f"**Historical Stress (first 5 days)**: {hist_stress:.4f}")
        st.write(f"**Hypothetical Stress (-10%)**: {hypo_stress:.4f}")

        st.divider()

        # ---------------------------------------------------------
        # LIQUIDITY HORIZON
        # ---------------------------------------------------------
        st.header("Liquidity Horizon (FRTB)")

        lh_var = liquidity_adjusted_var(port_var, 20)
        st.write(f"**Liquidity-Adjusted VaR (20-day)**: {lh_var:.4f}")

        st.divider()

        # ---------------------------------------------------------
        # BACKTESTING
        # ---------------------------------------------------------
        if "price" in df.columns:
            st.header("Backtesting (Kupiec Test)")

            var_series = pd.Series(hvar, index=returns.index)
            results = kupiec_test(returns, var_series)

            st.write(f"**Exceptions**: {results['exceptions']}")
            st.write(f"**LR Statistic**: {results['LR_statistic']:.4f}")
            st.write(f"**Passed Test**: {results['passed']}")

            st.divider()
