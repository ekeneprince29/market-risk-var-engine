from src.pnl_attribution import actual_pnl, model_pnl, pnl_attribution
from src.liquidity_horizon import liquidity_adjusted_var
from src.stress_testing import historical_stress, hypothetical_stress
from src.monte_carlo_correlated import monte_carlo_correlated_var
from src.backtesting import kupiec_test
from src.portfolio_var import portfolio_var

import pandas as pd
import numpy as np

from src.historical_var import historical_var
from src.parametric_var import parametric_var
from src.monte_carlo_var import monte_carlo_var

df = pd.read_csv("data/prices.csv")
df["returns"] = df["price"].pct_change()

returns = df["returns"].dropna()

print("Historical VaR (99%):", historical_var(returns))
print("Parametric VaR (99%):", parametric_var(returns))
print("Monte Carlo VaR (99%):", monte_carlo_var(returns))

from src.historical_var import historical_var, expected_shortfall

print("Expected Shortfall (99%):", expected_shortfall(returns))


# Load multi-asset data
df_port = pd.read_csv("data/portfolio_prices.csv")

# Keep only numeric columns (drop date)
df_port = df_port.select_dtypes(include=[np.number])

# Compute returns for each asset
returns_port = df_port.pct_change().dropna()

# Example portfolio weights (equal weight)
weights = np.array([1/3, 1/3, 1/3])

print("Portfolio VaR (99%):", portfolio_var(returns_port, weights))
# Monte Carlo VaR with correlations
mc_corr_var = monte_carlo_correlated_var(returns_port, weights)
print("Monte Carlo Correlated VaR (99%):", mc_corr_var)


# Backtesting Historical VaR
var_series = returns.rolling(window=100).apply(lambda x: historical_var(x, 0.99))

# Align lengths
returns_bt = returns[var_series.notna()]
var_series_bt = var_series[var_series.notna()]

result = kupiec_test(returns_bt, var_series_bt)

print("Backtesting (Kupiec Test):")
print("Exceptions:", result["exceptions"])
print("LR Statistic:", result["LR_statistic"])
print("Passed:", result["passed"])

# Stress Testing
print("Historical Stress (first 5 days):", historical_stress(returns, (0, 5)))
print("Hypothetical Stress (-10% shock):", hypothetical_stress(returns, -0.10))

# Liquidity Horizon Adjustments (FRTB)
lh_20 = liquidity_adjusted_var(portfolio_var(returns_port, weights), 20)
lh_40 = liquidity_adjusted_var(portfolio_var(returns_port, weights), 40)

print("Liquidity-Adjusted VaR (20-day):", lh_20)
print("Liquidity-Adjusted VaR (40-day):", lh_40)

# PnL Attribution
actual = actual_pnl(df_port, weights)
model = model_pnl(np.array([0.01, -0.02, 0.005]), weights)
error = pnl_attribution(actual, model)

print("Actual PnL:", actual)
print("Model PnL:", model)
print("Attribution Error:", error)
