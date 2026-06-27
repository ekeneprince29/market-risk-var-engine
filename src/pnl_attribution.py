import numpy as np
import pandas as pd

def actual_pnl(prices, weights):
    returns = prices.pct_change().dropna()
    pnl = (returns.iloc[-1] * weights).sum()
    return pnl

def model_pnl(risk_factor_shocks, weights):
    pnl = (risk_factor_shocks * weights).sum()
    return pnl

def pnl_attribution(actual, model):
    return actual - model
