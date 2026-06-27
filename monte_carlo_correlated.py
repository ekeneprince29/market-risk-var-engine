import numpy as np

def monte_carlo_correlated_var(returns_df, weights, simulations=10000, confidence=0.99):
    """
    Monte Carlo VaR with correlations using Cholesky decomposition.
    returns_df: DataFrame of asset returns
    weights: portfolio weights
    """
    # Mean and covariance
    mean_vector = returns_df.mean().values
    cov_matrix = returns_df.cov().values

    # Cholesky decomposition
    L = np.linalg.cholesky(cov_matrix)

    # Generate uncorrelated random numbers
    Z = np.random.normal(size=(simulations, len(mean_vector)))

    # Create correlated returns
    correlated_returns = Z @ L.T + mean_vector

    # Portfolio returns
    portfolio_returns = correlated_returns @ weights

    # VaR
    var_level = np.percentile(portfolio_returns, (1 - confidence) * 100)

    return var_level
