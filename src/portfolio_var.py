import numpy as np
import pandas as pd

def portfolio_var(returns_df, weights, confidence=0.99):
    """
    Portfolio VaR using covariance matrix and weights.
    returns_df: DataFrame of returns for each asset
    weights: list or array of portfolio weights
    """
    cov_matrix = returns_df.cov()
    portfolio_std = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
    z = 2.33  # 99% confidence
    return -z * portfolio_std

