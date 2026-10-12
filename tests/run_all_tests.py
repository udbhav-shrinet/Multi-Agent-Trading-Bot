"""Automated comprehensive test runner for quantitative trading engine."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test_econometric_models import test_models
from tests.test_quant_methods import (
    test_almgren_chriss_execution_cost,
    test_fama_french_factor_adjust,
    test_portfolio_council_integration,
    test_statistical_arbitrage_score,
)


def run_all():
    print("=" * 60)
    print("RUNNING QUANTITATIVE MULTI-AGENT TEST SUITE")
    print("=" * 60)

    print("\n[1/5] Econometric Models (GARCH, Cointegration, Dynamic Beta)...")
    test_models()
    print("  -> PASSED")

    print("\n[2/5] Fama-French 3-Factor Risk Loadings...")
    test_fama_french_factor_adjust()
    print("  -> PASSED")

    print("\n[3/5] Statistical Arbitrage (Ornstein-Uhlenbeck & Mean Reversion)...")
    test_statistical_arbitrage_score()
    print("  -> PASSED")

    print("\n[4/5] Almgren-Chriss Optimal Execution Modeling...")
    test_almgren_chriss_execution_cost()
    print("  -> PASSED")

    print("\n[5/5] Portfolio Council Risk Sizing & Factor Integration...")
    test_portfolio_council_integration()
    print("  -> PASSED")

    print("\n" + "=" * 60)
    print("ALL INTEGRATION AND QUANTITATIVE TESTS PASSED (5/5)")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
