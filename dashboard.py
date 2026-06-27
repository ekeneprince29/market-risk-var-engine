import streamlit as st
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

import pandas as pd
import numpy as np

from src.historical_var import historical_var, expected_shortfall
from src.parametric_var import parametric_var
from src.monte_carlo_var import monte_carlo_var
from src.portfolio_var import portfolio_var

st.title("Market Risk VaR Engine Dashboard")

st.subheader("Data Input")

use_sample = st.button("Use sample data")

uploaded_file = st.file_uploader("Upload CSV price data", type=["csv"])

if use_sample:
    df = pd.read_csv("data/prices.csv")
    st.success("Loaded sample data from data/prices.csv")
    st.write(df.head())
    # Call your VaR/ES/Monte Carlo functions here using df

elif uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Uploaded custom CSV data")
    st.write(df.head())
    # Call your VaR/ES/Monte Carlo functions here using df

else:
    st.info("Upload a CSV file or click 'Use sample data' to begin.")

uploaded_file = st.file_uploader("Upload price data (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    # Single asset VaR
    if "price" in df.columns:
        df["returns"] = df["price"].pct_change().dropna()
        returns = df["returns"].dropna()

        st.subheader("Single Asset Risk Metrics")
        st.write("Historical VaR (99%):", historical_var(returns))
        st.write("Parametric VaR (99%):", parametric_var(returns))
        st.write("Monte Carlo VaR (99%):", monte_carlo_var(returns))
        st.write("Expected Shortfall (99%):", expected_shortfall(returns))

    # Portfolio VaR
    if len(df.columns) > 2:
        st.subheader("Portfolio VaR")
        returns_port = df.iloc[:, 1:].pct_change().dropna()
        weights = np.array([1/len(returns_port.columns)] * len(returns_port.columns))
        st.write("Portfolio VaR (99%):", portfolio_var(returns_port, weights))
