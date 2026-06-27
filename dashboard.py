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

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Market Risk VaR Engine",
    page_icon="📊",
    layout="wide"
)

# ---------------------- HEADER ----------------------
st.markdown("""
# 📊 Market Risk VaR Engine  
A professional dashboard for market risk analytics:
- **Value-at-Risk (VaR)**
- **Expected Shortfall (ES)**
- **Monte Carlo Simulation**
- **Stress Testing**
- **Liquidity Horizon (FRTB)**
- **Backtesting (Kupiec Test)**  
""")

st.divider()

# ---------------------- UNIVERSAL FILE LOADER ----------------------
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

    # CSV support
    try:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="utf-8")
    except Exception:
        pass

    try:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="ISO-8859-1")
    except Exception:
        pass

    try:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="utf-16")
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

# ---------------------- DATA INPUT ----------------------
st.header("📁 Data Input")

col1, col2 = st.columns([1, 2])

with col1:
    use_sample = st.button("📘 Use Sample Data")
    uploaded_file = st.file_uploader("Upload CSV or Excel price/portfolio data", type=["csv", "xlsx", "xls"])

df = None

if use_sample:
    df = pd.read_csv("data/prices.csv")
    st.success("Loaded sample data from data/prices.csv")

elif uploaded_file is not None:
    df = load_data(uploaded_file)

    if df is not None:
        # FIX COLUMN NAME ISSUES
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace("\ufeff", "", regex=False)
            .str.replace("﻿", "", regex=False)
        )

        # FORCE second column to be 'price' if missing header
        if len(df.columns) == 2 and "price" not in df.columns:
            df.columns = ["date", "price"]

        st.success(f"Loaded file: {uploaded_file.name}")
        st.write("COLUMNS:", df.columns.tolist())  # TEMP DEBUG

else:
    st.info("Upload a CSV or Excel file, or click **Use Sample Data** to begin.")

if df is not None:
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())
    st.divider()

# ---------------------- SINGLE ASSET RISK ----------------------
if df is not None and "price" in df.columns:

    st.header("📈 Single Asset Risk Metrics")

    price_series = pd.to_numeric(df["price"], errors="coerce")
    returns = price_series.pct_change().dropna()

    if returns.empty:
        st.warning("No valid returns could be computed from 'price' column.")
    else:
        colA, colB, colC, colD = st.columns(4)

        colA.metric("Historical VaR (99%)", f"{historical_var(returns):.4f}")
        colB.metric("Parametric VaR (99%)", f"{parametric_var(returns):.4f}")
        colC.metric("Monte Carlo VaR (99%)", f"{monte_carlo_var(returns):.4f}")
        colD.metric("Expected Shortfall (99%)", f"{expected_shortfall(returns):.4f}")

        st.divider()

# ---------------------- PORTFOLIO VaR ----------------------
if df is not None and len(df.columns) > 2:

    st.header("📊 Portfolio VaR")

    price_data = df.iloc[:, 1:].select_dtypes(include=["number"])
    returns_port = price_data.pct_change().dropna()

    if returns_port.empty:
        st.warning("Portfolio returns could not be computed.")
    else:
        weights = np.array([1 / len(returns_port.columns)] * len(returns_port.columns))
        port_var = portfolio_var(returns_port, weights)

        st.metric("Portfolio VaR (99%)", f"{port_var:.4f}")
        st.divider()

        # ---------------------- STRESS TESTING ----------------------
        st.header("⚠️ Stress Testing")

        col1, col2 = st.columns(2)

        with col1:
            hist_stress = historical_stress(returns_port.iloc[:, 0], (0, 5))
            st.metric("Historical Stress (first 5 days)", f"{hist_stress:.4f}")

        with col2:
            hypo_stress = hypothetical_stress(returns_port.iloc[:, 0], -0.10)
            st.metric("Hypothetical Stress (-10%)", f"{hypo_stress:.4f}")

        st.divider()

        # ---------------------- LIQUIDITY HORIZON ----------------------
        st.header("⏳ Liquidity Horizon (FRTB)")

        lh_var = liquidity_adjusted_var(port_var, 20)
        st.metric("Liquidity-Adjusted VaR (20-day)", f"{lh_var:.4f}")

        st.divider()

        # ---------------------- BACKTESTING ----------------------
        if "price" in df.columns:
            st.header("📉 Backtesting (Kupiec Test)")

            var_series = pd.Series(historical_var(returns), index=returns.index)
            results = kupiec_test(returns, var_series)

            col1, col2, col3 = st.columns(3)

            col1.metric("Exceptions", results["exceptions"])
            col2.metric("LR Statistic", f"{results['LR_statistic']:.4f}")
            col3.metric("Passed Test", "✅ Yes" if results["passed"] else "❌ No")

            st.divider()
