from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from agents.econometric_models import check_cointegration, dynamic_beta, forecast_garch_volatility

def test_models():
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='B')
    
    # Test GARCH
    returns = pd.Series(np.random.normal(0, 0.01, 100), index=dates)
    garch_res = forecast_garch_volatility(returns)
    print(f"GARCH Volatility: {garch_res}")
    assert isinstance(garch_res, float)
    
    # Test Cointegration
    s1 = pd.Series(np.cumsum(np.random.normal(0, 1, 100)), index=dates)
    s2 = s1 + pd.Series(np.random.normal(0, 0.1, 100), index=dates) # cointegrated
    pval = check_cointegration(s1, s2)
    print(f"Cointegration P-Value: {pval}")
    assert isinstance(pval, float)
    
    # Test Dynamic Beta
    y = pd.Series(np.random.normal(0, 1, 100), index=dates)
    x = pd.Series(np.random.normal(0, 1, 100), index=dates)
    beta = dynamic_beta(y, x)
    print(f"Dynamic Beta: {beta}")
    assert isinstance(beta, float)
    
    print("All econometric model tests passed successfully!")

if __name__ == '__main__':
    test_models()
