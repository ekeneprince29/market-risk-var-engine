import numpy as np

def historical_var(returns, confidence=0.99):
    return np.percentile(returns, (1 - confidence) * 100)

def expected_shortfall(returns, confidence=0.99):
    var_level = np.percentile(returns, (1 - confidence) * 100)
    tail_losses = returns[returns <= var_level]
    return tail_losses.mean()
