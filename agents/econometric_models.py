"""Econometric models for advanced trading signals."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

try:
    from arch import arch_model
except ImportError:
    arch_model = None

def forecast_garch_volatility(returns: pd.Series, horizon: int = 1) -> float:
    """Forecast 1-day ahead annualized volatility using GARCH(1,1)."""
    try:
        if arch_model is None:
            return float(returns.std() * np.sqrt(252))
        returns_scaled = returns.dropna() * 100
        if len(returns_scaled) < 30:
            return float(returns.std() * np.sqrt(252))
        model = arch_model(returns_scaled, vol='Garch', p=1, q=1, rescale=False)
        model_fit = model.fit(disp='off')
        forecasts = model_fit.forecast(horizon=horizon)
        variance = forecasts.variance.iloc[-1, 0]
        daily_std = np.sqrt(variance) / 100
        return float(daily_std * np.sqrt(252))
    except Exception:
        return float(returns.std() * np.sqrt(252))


def check_cointegration(s1: pd.Series, s2: pd.Series) -> float:
    """Check cointegration using Engle-Granger test and return p-value."""
    s1, s2 = s1.align(s2, join='inner')
    s1 = s1.dropna()
    s2 = s2.dropna()
    s1, s2 = s1.align(s2, join='inner')
    if len(s1) < 30:
        return 1.0
    try:
        model = sm.OLS(s1, sm.add_constant(s2)).fit()
        residuals = model.resid
        adf = sm.tsa.stattools.adfuller(residuals)
        return float(adf[1])
    except Exception:
        return 1.0


def dynamic_beta(y: pd.Series, x: pd.Series) -> float:
    """Estimate dynamic beta using Rolling OLS (proxy for Kalman filter in time-series)."""
    y, x = y.align(x, join='inner')
    y = y.dropna()
    x = x.dropna()
    y, x = y.align(x, join='inner')
    if len(y) < 30:
        return 1.0
    try:
        rols = RollingOLS(y, sm.add_constant(x), window=min(30, len(y)))
        rolling_res = rols.fit()
        params = rolling_res.params
        latest_beta = params.iloc[-1].iloc[1]
        return float(latest_beta) if not np.isnan(latest_beta) else 1.0
    except Exception:
        return 1.0
