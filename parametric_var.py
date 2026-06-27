import numpy as np

def parametric_var(returns, confidence=0.99):
    mean = np.mean(returns)
    std = np.std(returns)
    z = 2.33
    return mean - z * std
