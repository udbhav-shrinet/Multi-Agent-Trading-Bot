"""Institutional Quant Methods: Fama-French Factor Models, Statistical Arbitrage, and Almgren-Chriss Optimal Execution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def fama_french_factor_adjust(
    asset_returns: list[float] | pd.Series,
    market_returns: list[float] | pd.Series | None = None,
    smb_returns: list[float] | pd.Series | None = None,
    hml_returns: list[float] | pd.Series | None = None,
) -> dict:
    """Estimate Fama-French 3-factor risk loadings (Market, SMB, HML) and alpha.

    If factor return series are not provided or insufficient, falls back to market CAPM or zero loadings.
    Returns factor betas, annualized alpha, R-squared, and residual volatility.
    """
    r = pd.Series(asset_returns).dropna()
    if len(r) < 20:
        return {
            "alpha_annual": 0.0,
            "beta_mkt": 1.0,
            "beta_smb": 0.0,
            "beta_hml": 0.0,
            "r_squared": 0.0,
            "residual_vol": float(r.std() * np.sqrt(252)) if len(r) > 1 else 0.20,
        }

    data = pd.DataFrame({"r": r})
    if market_returns is not None:
        data["mkt"] = pd.Series(market_returns)
    else:
        data["mkt"] = 0.0

    if smb_returns is not None:
        data["smb"] = pd.Series(smb_returns)
    else:
        data["smb"] = 0.0

    if hml_returns is not None:
        data["hml"] = pd.Series(hml_returns)
    else:
        data["hml"] = 0.0

    data = data.dropna()
    if len(data) < 20:
        return {
            "alpha_annual": 0.0,
            "beta_mkt": 1.0,
            "beta_smb": 0.0,
            "beta_hml": 0.0,
            "r_squared": 0.0,
            "residual_vol": float(r.std() * np.sqrt(252)),
        }

    X = sm.add_constant(data[["mkt", "smb", "hml"]])
    y = data["r"]

    try:
        model = sm.OLS(y, X).fit()
        params = model.params
        alpha_daily = float(params.get("const", 0.0))
        alpha_annual = alpha_daily * 252
        beta_mkt = float(params.get("mkt", 1.0))
        beta_smb = float(params.get("smb", 0.0))
        beta_hml = float(params.get("hml", 0.0))
        r_squared = float(model.rsquared)
        residual_vol = float(np.sqrt(model.mse_resid) * np.sqrt(252))
        return {
            "alpha_annual": alpha_annual,
            "beta_mkt": beta_mkt,
            "beta_smb": beta_smb,
            "beta_hml": beta_hml,
            "r_squared": r_squared,
            "residual_vol": residual_vol,
        }
    except Exception:
        return {
            "alpha_annual": 0.0,
            "beta_mkt": 1.0,
            "beta_smb": 0.0,
            "beta_hml": 0.0,
            "r_squared": 0.0,
            "residual_vol": float(r.std() * np.sqrt(252)),
        }


def fit_ornstein_uhlenbeck(series: pd.Series | list[float], dt: float = 1.0) -> dict:
    """Fit continuous Ornstein-Uhlenbeck process:
    dX_t = kappa * (theta - X_t) dt + sigma * dW_t

    Using discrete AR(1) regression:
    X_t = a + b * X_{t-1} + eps
    where:
    b = exp(-kappa * dt) => kappa = -ln(b) / dt
    theta = a / (1 - b)
    sigma = sigma_eps * sqrt(2 * kappa / (1 - b^2))
    half_life = ln(2) / kappa
    """
    s = pd.Series(series).dropna().astype(float)
    if len(s) < 15:
        return {
            "kappa": 0.0,
            "theta": float(s.mean()) if len(s) > 0 else 0.0,
            "half_life_days": 999.0,
            "ou_sigma": float(s.std()) if len(s) > 1 else 0.0,
            "stationary": False,
        }

    x_prev = s.shift(1).iloc[1:]
    x_curr = s.iloc[1:]

    X = sm.add_constant(x_prev)
    try:
        model = sm.OLS(x_curr, X).fit()
        a = float(model.params.iloc[0])
        b = float(model.params.iloc[1])
        residual_std = float(np.std(model.resid))

        if 0 < b < 1.0:
            kappa = -np.log(b) / dt
            theta = a / (1.0 - b)
            half_life = np.log(2) / kappa
            half_life = max(0.5, min(half_life, 252.0))
            denom = max(1.0 - b**2, 1e-9)
            ou_sigma = residual_std * np.sqrt(2.0 * kappa / denom)
            stationary = True
        else:
            kappa = 0.0
            theta = float(s.mean())
            half_life = 999.0
            ou_sigma = residual_std
            stationary = False

        return {
            "kappa": round(float(kappa), 4),
            "theta": round(float(theta), 4),
            "half_life_days": round(float(half_life), 2),
            "ou_sigma": round(float(ou_sigma), 4),
            "stationary": stationary,
        }
    except Exception:
        return {
            "kappa": 0.0,
            "theta": float(s.mean()),
            "half_life_days": 999.0,
            "ou_sigma": float(s.std()),
            "stationary": False,
        }


def synthesize_cross_sectional_factors(research_items: list[dict]) -> tuple[pd.Series, pd.Series, pd.Series] | None:
    """Construct synthetic Fama-French factors (Mkt, SMB, HML) from the cross-section of scanned research."""
    valid_items = [
        item for item in research_items
        if "recent_returns" in item and len(item["recent_returns"]) >= 20
    ]
    if len(valid_items) < 3:
        return None

    min_len = min(len(item["recent_returns"]) for item in valid_items)
    returns_matrix = {
        item["symbol"]: item["recent_returns"][-min_len:]
        for item in valid_items
    }
    df = pd.DataFrame(returns_matrix)

    mkt_factor = df.mean(axis=1)

    avg_dollar_vols = {
        item["symbol"]: item.get("average_dollar_volume", 1e6)
        for item in valid_items
    }
    median_vol = np.median(list(avg_dollar_vols.values()))
    small_symbols = [s for s, v in avg_dollar_vols.items() if v <= median_vol]
    big_symbols = [s for s, v in avg_dollar_vols.items() if v > median_vol]

    small_returns = df[small_symbols].mean(axis=1) if small_symbols else mkt_factor
    big_returns = df[big_symbols].mean(axis=1) if big_symbols else mkt_factor
    smb_factor = small_returns - big_returns

    cumulative_returns = {
        s: float((1 + df[s]).prod() - 1) for s in df.columns
    }
    median_cum = np.median(list(cumulative_returns.values()))
    value_symbols = [s for s, r in cumulative_returns.items() if r <= median_cum]
    growth_symbols = [s for s, r in cumulative_returns.items() if r > median_cum]

    value_returns = df[value_symbols].mean(axis=1) if value_symbols else mkt_factor
    growth_returns = df[growth_symbols].mean(axis=1) if growth_symbols else mkt_factor
    hml_factor = value_returns - growth_returns

    return mkt_factor, smb_factor, hml_factor


def statistical_arbitrage_score(
    asset_returns: list[float] | pd.Series,
    benchmark_returns: list[float] | pd.Series,
    lookback: int = 60,
) -> dict:
    """Compute statistical arbitrage metrics: half-life of mean reversion, spread z-score, and cointegration p-value.

    Uses Ornstein-Uhlenbeck process estimation for half-life.
    """
    s1 = pd.Series(asset_returns).dropna()
    s2 = pd.Series(benchmark_returns).dropna()
    s1, s2 = s1.align(s2, join="inner")
    if len(s1) < lookback:
        return {
            "half_life": 999.0,
            "z_score": 0.0,
            "cointegrated": False,
            "p_value": 1.0,
        }

    s1 = s1.tail(lookback)
    s2 = s2.tail(lookback)

    try:
        model = sm.OLS(s1, sm.add_constant(s2)).fit()
        residuals = model.resid
        adf = sm.tsa.stattools.adfuller(residuals)
        p_value = float(adf[1])
        cointegrated = p_value < 0.05

        mean_res = residuals.mean()
        std_res = residuals.std()
        z_score = float((residuals.iloc[-1] - mean_res) / max(std_res, 1e-9))

        lag_res = residuals.shift(1).dropna()
        delta_res = residuals.diff().dropna()
        lag_res, delta_res = lag_res.align(delta_res, join="inner")
        
        ar_model = sm.OLS(delta_res, sm.add_constant(lag_res)).fit()
        lam = ar_model.params.iloc[1]
        half_life = float(-np.log(2) / lam) if lam < 0 else 999.0
        half_life = max(1.0, min(half_life, 252.0))

        return {
            "half_life": half_life,
            "z_score": z_score,
            "cointegrated": cointegrated,
            "p_value": p_value,
        }
    except Exception:
        return {
            "half_life": 999.0,
            "z_score": 0.0,
            "cointegrated": False,
            "p_value": 1.0,
        }


def almgren_chriss_execution_cost(
    total_shares: float,
    price: float,
    daily_volume: float,
    volatility: float,
    risk_aversion: float = 1e-6,
    urgency_parameter: float = 0.5,
    time_horizon_days: float = 1.0,
    num_intervals: int = 5,
) -> dict:
    """Estimate optimal execution trajectory cost using Almgren-Chriss framework.

    Separates permanent market impact and temporary market impact, balanced against inventory holding risk.
    """
    if total_shares <= 0 or price <= 0 or daily_volume <= 0:
        return {
            "expected_cost_usd": 0.0,
            "expected_cost_bps": 0.0,
            "temporary_impact_bps": 0.0,
            "permanent_impact_bps": 0.0,
            "risk_penalty_usd": 0.0,
            "timing_risk_usd": 0.0,
            "recommended_slices": 1,
            "trajectory_shares": [0.0],
            "participation_rate": 0.0,
        }

    trading_value = total_shares * price
    vol_per_day = daily_volume
    sigma_dollar = volatility * price

    gamma = 0.1 * price / max(vol_per_day, 1000.0)
    tau = max(time_horizon_days / max(num_intervals, 1), 0.01)
    eta = 0.14 * price / max(vol_per_day * tau, 100.0)

    permanent_impact_usd = 0.5 * gamma * (total_shares ** 2)
    temp_impact_usd = eta * (total_shares ** 2) / max(1.0, urgency_parameter * 10.0)
    
    risk_penalty = 0.5 * risk_aversion * (volatility ** 2) * (trading_value ** 2) * urgency_parameter
    total_cost_usd = permanent_impact_usd + temp_impact_usd + risk_penalty

    perm_impact_bps = (permanent_impact_usd / max(trading_value, 1e-9)) * 10000
    temp_impact_bps = (temp_impact_usd / max(trading_value, 1e-9)) * 10000
    expected_cost_bps = (total_cost_usd / max(trading_value, 1e-9)) * 10000

    participation_rate = trading_value / max(daily_volume * price, 1.0)
    recommended_slices = min(num_intervals, max(1, int(total_shares / max(daily_volume * 0.02, 1.0))))

    return {
        "expected_cost_usd": float(total_cost_usd),
        "expected_cost_bps": float(expected_cost_bps),
        "temporary_impact_bps": float(temp_impact_bps),
        "permanent_impact_bps": float(perm_impact_bps),
        "risk_penalty_usd": float(risk_penalty),
        "timing_risk_usd": float(risk_penalty),
        "recommended_slices": int(recommended_slices),
        "trajectory_shares": [round(float(total_shares / recommended_slices), 4)] * recommended_slices,
        "participation_rate": float(participation_rate),
    }
