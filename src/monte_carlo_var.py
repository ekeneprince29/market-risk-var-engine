import numpy as np

def monte_carlo_var(returns, simulations=10000, confidence=0.99):
    mean = np.mean(returns)
    std = np.std(returns)
    simulated = np.random.normal(mean, std, simulations)
    return np.percentile(simulated, (1 - confidence) * 100)
