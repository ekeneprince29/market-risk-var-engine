import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------
# Load Data
# -----------------------------
data1 = pd.read_csv("data/prices.csv")
data2 = pd.read_csv("data/prices_pass.csv")

# Compute returns
ret1 = data1["price"].pct_change().dropna()
ret2 = data2["price"].pct_change().dropna()

# -----------------------------
# Risk Metrics
# -----------------------------
def historical_var(returns, level=0.99):
    return np.quantile(returns, 1 - level)

def expected_shortfall(returns, level=0.99):
    var = historical_var(returns, level)
    return returns[returns < var].mean()

def parametric_var(returns, level=0.99):
    mu = returns.mean()
    sigma = returns.std()
    z = 2.33  # 99%
    return mu - z * sigma

def monte_carlo_var(returns, level=0.99, sims=10000):
    mu = returns.mean()
    sigma = returns.std()
    sims = np.random.normal(mu, sigma, sims)
    return np.quantile(sims, 1 - level)

# Compute metrics for Data 1
hist_var1 = historical_var(ret1)
param_var1 = parametric_var(ret1)
mc_var1 = monte_carlo_var(ret1)
es1 = expected_shortfall(ret1)

# Portfolio VaR (simple 1-asset)
portfolio_var1 = hist_var1 * np.sqrt(1)

# Stress tests
historical_stress1 = ret1.head(5).sum()
hypo_stress1 = -0.10  # -10% shock
liquidity_adj_var1 = hist_var1 * np.sqrt(20)

# Backtesting (simple Kupiec)
exceptions = sum(ret1 < hist_var1)
lr_stat = 2 * (exceptions * np.log(exceptions / (len(ret1) * 0.01)) if exceptions > 0 else 0)
passed = exceptions <= 1

# -----------------------------
# UI
# -----------------------------
st.markdown(
    f"""
    # 📊 Market Risk VaR Engine  
    **Historical VaR (99%)**: `{hist_var1:.4f}`  
    **Portfolio VaR (99%)**: `{portfolio_var1:.4f}`  
    ---
    A streamlined risk analytics environment designed to measure market volatility, quantify tail exposure,  
    and evaluate model behavior under varying conditions.  
    """
)

st.subheader("Loaded Price Data 1")
st.dataframe(data1.head())

st.markdown("### Single Asset Risk Metrics")
st.write(f"**Historical VaR (99%)**: {hist_var1:.4f}")
st.write(f"**Parametric VaR (99%)**: {param_var1:.4f}")
st.write(f"**Monte Carlo VaR (99%)**: {mc_var1:.4f}")
st.write(f"**Expected Shortfall (99%)**: {es1:.4f}")

st.markdown("### Portfolio VaR")
st.write(f"**Portfolio VaR (99%)**: {portfolio_var1:.4f}")

st.markdown("### Stress Testing")
st.write(f"**Historical Stress (first 5 days)**: {historical_stress1:.4f}")
st.write(f"**Hypothetical Stress (-10%)**: {hypo_stress1:.4f}")
st.write(f"**Liquidity-Adjusted VaR (20-day)**: {liquidity_adj_var1:.4f}")

st.markdown("### Backtesting (Kupiec Test)")
st.write(f"**Exceptions**: {exceptions}")
st.write(f"**LR Statistic**: {lr_stat:.4f}")
st.write(f"**Passed Test**: {passed}")
