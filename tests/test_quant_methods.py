"""Tests for institutional quantitative methods and portfolio council integration."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

try:
    import pytest
except ImportError:
    pytest = None

from agents.quant_methods import (
    fama_french_factor_adjust,
    statistical_arbitrage_score,
    almgren_chriss_execution_cost,
    fit_ornstein_uhlenbeck,
)
from agents.portfolio_council import build_portfolio_plan


def test_fama_french_factor_adjust():
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, 50)
    mkt = np.random.normal(0.0004, 0.015, 50)
    smb = np.random.normal(0.0001, 0.01, 50)
    hml = np.random.normal(0.0, 0.01, 50)

    res = fama_french_factor_adjust(returns, mkt, smb, hml)
    assert "alpha_annual" in res
    assert "beta_mkt" in res
    assert "residual_vol" in res
    assert isinstance(res["beta_mkt"], float)


def test_statistical_arbitrage_score():
    np.random.seed(42)
    s2 = np.random.normal(0.0, 0.01, 100)
    s1 = 1.2 * s2 + np.random.normal(0.0, 0.005, 100)

    res = statistical_arbitrage_score(s1, s2, lookback=50)
    assert "half_life" in res
    assert "z_score" in res
    assert "cointegrated" in res


def test_almgren_chriss_execution_cost():
    cost_res = almgren_chriss_execution_cost(
        total_shares=1000.0,
        price=150.0,
        daily_volume=500000.0,
        volatility=0.02,
    )
    assert "expected_cost_usd" in cost_res
    assert "temporary_impact_bps" in cost_res
    assert "permanent_impact_bps" in cost_res
    assert cost_res["expected_cost_usd"] > 0


def test_portfolio_council_integration():
    research = [
        {
            "symbol": "AAPL",
            "direction": "long",
            "score": 0.45,
            "volatility": 0.02,
            "atr": 3.0,
            "price": 180.0,
            "estimated_round_trip_cost": 0.002,
            "expected_edge": 0.015,
            "recent_returns": [0.001 * (-1)**i for i in range(30)],
            "average_dollar_volume": 50_000_000,
            "reasons": {"ml": 0.3},
        },
        {
            "symbol": "MSFT",
            "direction": "long",
            "score": 0.35,
            "volatility": 0.018,
            "atr": 4.0,
            "price": 400.0,
            "estimated_round_trip_cost": 0.002,
            "expected_edge": 0.012,
            "recent_returns": [0.0008 * (-1)**i for i in range(30)],
            "average_dollar_volume": 60_000_000,
            "reasons": {"ml": 0.25},
        },
    ]
    account = {"equity": 100000.0, "cash": 100000.0}
    positions = []
    open_orders = []

    plan = build_portfolio_plan(research, account, positions, open_orders)
    assert plan["approved"] is True
    assert len(plan["trades"]) > 0
    for trade in plan["trades"]:
        assert "fama_french" in trade
        assert "almgren_chriss" in trade
