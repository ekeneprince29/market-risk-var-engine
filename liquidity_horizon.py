import numpy as np

def liquidity_adjusted_var(var_10day, liquidity_horizon):
    """
    Scale 10-day VaR to FRTB liquidity horizon.
    LH in days (e.g., 20, 40, 60)
    """
    return var_10day * np.sqrt(liquidity_horizon / 10)
