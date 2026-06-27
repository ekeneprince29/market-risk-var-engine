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

st.markdown("""
### 📊 Market Risk VaR Engine

Upload a price or portfolio CSV to run:
- Value-at-Risk (VaR)
- Expected Shortfall (ES)
- Monte Carlo simulations
- Stress Testing
- Liquidity Horizon (FRTB)
- Backtesting (Kupiec Test)

You can upload your own data or use sample data to see the engine in action.
""")

st.title("Market Risk VaR Engine Dashboard")
st.subheader("Data Input")

# --- Data selection: sample or upload ---
use_sample = st.button("Use sample data")
uploaded_file = st.file_uploader("Upload CSV price/portfolio data", type=["csv"])

df = None

if use_sample:
    df = pd.read_csv("data/prices.csv")
    st.success("Loaded sample data from data/prices.csv")
    st.write("Data Preview:", df.head())

elif uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Uploaded custom CSV data")
    st.write("Data Preview:", df.head())

else:
    st.info("Upload a CSV file or click 'Use sample data' to begin.")

# --- Only proceed if we actually have data ---
if df is not None:

    # ---------- Single asset VaR (expects a 'price' column) ----------
    if "price" in df.columns:
        st.subheader("Single Asset Risk Metrics")

        price_series = pd.to_numeric(df["price"], errors="coerce")
        returns = price_series.pct_change().dropna()

        if returns.empty:
            st.warning("No valid returns could be computed from 'price' column.")
        else:
            st.write("Historical VaR (99%):", historical_var(returns))
            st.write("Parametric VaR (99%):", parametric_var(returns))
            st.write("Monte Carlo VaR (99%):", monte_carlo_var(returns))
            st.write("Expected Shortfall (99%):", expected_shortfall(returns))

    # ---------- Portfolio VaR ----------
    if len(df.columns) > 2:
        st.subheader("Portfolio VaR")

        # assume first column is Date / index, rest are prices
        price_data = df.iloc[:, 1:].select_dtypes(include=["number"])
        returns_port = price_data.pct_change().dropna()

        if returns_port.empty:
            st.warning("Portfolio returns could not be computed.")
        else:
            weights = np.array(
                [1 / len(returns_port.columns)] * len(returns_port.columns)
            )
            st.write("Portfolio VaR (99%):", portfolio_var(returns_port, weights))

            # ---------- Stress Testing ----------
            st.subheader("Stress Testing")
            try:
                st.write(
                    "Historical Stress (first 5 days):",
                    historical_stress(returns_port.iloc[:, 0], (0, 5)),
                )
                st.write(
                    "Hypothetical Stress (-10% shock):",
                    hypothetical_stress(returns_port.iloc[:, 0], -0.10),
                )
            except Exception as e:
                st.warning(f"Stress testing could not be computed: {e}")

            # ---------- Liquidity Horizon (FRTB) ----------
            st.subheader("Liquidity Horizon (FRTB)")
            try:
                base_var = portfolio_var(returns_port, weights)
                lh_var = liquidity_adjusted_var(base_var, 20)  # 20‑day horizon example
                st.write("Liquidity-Adjusted VaR (20-day):", lh_var)
            except Exception as e:
                st.warning(f"Liquidity Horizon could not be computed: {e}")

            # ---------- Backtesting (Kupiec Test) ----------
            if "price" in df.columns:
                st.subheader("Backtesting (Kupiec Test)")
                try:
                    # simple example: use a constant VaR series
                    var_series = pd.Series(
                        historical_var(returns), index=returns.index
                    )
                    results = kupiec_test(returns, var_series)
                    st.write("Backtesting Results (Kupiec Test):", results)
                except Exception as e:
                    st.warning(f"Backtesting could not be computed: {e}")
