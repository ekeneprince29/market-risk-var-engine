import streamlit as st
import pandas as pd
import numpy as np

from src.historical_var import historical_var, expected_shortfall
from src.parametric_var import parametric_var
from src.monte_carlo_var import monte_carlo_var
from src.portfolio_var import portfolio_var

st.title("Market Risk VaR Engine Dashboard")

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
