import numpy as np
import pandas as pd

def historical_stress(returns, shock_window):
    """
    Apply a historical stress scenario using a past crisis window.
    shock_window: tuple (start_index, end_index)
    """
    start, end = shock_window
    stressed_returns = returns.iloc[start:end]
    return stressed_returns.sum()

def hypothetical_stress(returns, shock_percent):
    """
    Apply a hypothetical shock (e.g., -5%, -10%).
    """
    return returns.mean() + shock_percent
