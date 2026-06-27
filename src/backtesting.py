import numpy as np

def kupiec_test(returns, var_series, confidence=0.99):
    """
    Kupiec Proportion of Failures (POF) Test.
    returns: actual returns
    var_series: VaR values for each day
    """
    # Count exceedances
    exceptions = returns < var_series
    num_exceptions = exceptions.sum()
    n = len(returns)

    # Expected exception probability
    p = 1 - confidence

    # Likelihood ratio statistic
    LR = -2 * (
        np.log((1 - p)**(n - num_exceptions) * p**num_exceptions) -
        np.log((1 - num_exceptions/n)**(n - num_exceptions) * (num_exceptions/n)**num_exceptions)
    )

    return {
        "exceptions": int(num_exceptions),
        "LR_statistic": LR,
        "passed": LR < 3.84  # 95% confidence threshold
    }
