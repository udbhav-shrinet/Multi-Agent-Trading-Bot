# AlphaLens: Institutional Multi-Factor Quant Screener & Regime Pipeline

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Interactive Screener](https://img.shields.io/badge/demo-AlphaLens%20Terminal-9333ea.svg)](https://udbhav-shrinet.github.io/Multi-Agent-Trading-Bot/)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)]()
[![Broker](https://img.shields.io/badge/broker-Alpaca%20Markets-059669.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **AlphaLens** is an autonomous institutional-grade quantitative research, cross-sectional factor screening, macro regime detection, and portfolio execution terminal. Built on purged walk-forward Machine Learning, statistical mean-reversion modeling, and portfolio-level tail-risk constraints (VaR / CVaR) with automated Alpaca paper broker execution.

---

## ⚡ Live Quantitative Screener & Terminal

Explore live universe screening, cross-sectional factor rankings, regime classification, and live paper execution telemetry:  
👉 **[Launch AlphaLens Quant Terminal](https://udbhav-shrinet.github.io/Multi-Agent-Trading-Bot/)**

---

## 🎯 Key Quant & Engineering Capabilities

- **Cross-Sectional Multi-Factor Screener**:
  - Live screening across 500+ US equity candidates with real-time sorting by statistical edge, directional probability, 1D return, and RSI.
  - Export full universe factor diagnostics directly to CSV for offline quantitative research.
- **Hierarchical Autonomous Intelligence Pipeline**:
  - **Market Intelligence (`market_intelligence.py`)**: Computes log returns, ATR(14), RSI(14), Bollinger Bands, MACD, and real-time sentiment NLP via VADER & TF-IDF on news/social feeds.
  - **Quantitative Research (`quant_research.py`)**: Fits Ornstein-Uhlenbeck mean-reversion parameters ($\theta, \mu$), simulates Geometric Brownian Motion (GBM) paths, and predicts directional probability via purged walk-forward calibrated Random Forest models.
  - **Macroeconomic Regime Scoring (`fred_tools.py`)**: Ingests Federal Reserve Economic Data (FRED) including the 10Y-2Y Treasury yield curve spread and VIX to establish systemic risk-on / risk-off factors.
  - **Portfolio Risk Council (`portfolio_council.py`)**: Solves portfolio-level allocation using half-Kelly criteria, volatility targeting, and strict parametric VaR (95%) and CVaR (95%) gates to veto high-tail-risk candidates.
  - **Execution Commander (`execution_commander.py`)**: Institutional execution engine enforcing single-point execution, fractional long routing, and TWAP-sliced order execution via Alpaca Markets.
- **Automated US Market Session Scheduler**:
  - Continuous scheduling via GitHub Actions every 30 minutes during active NYSE/NASDAQ market sessions (EDT / EST adjusted).

---

## 🏗️ Quantitative Data Flow Architecture

```text
┌────────────────────────────────┐
│   500+ Liquid Candidate Pool   │
└────────────────┬───────────────┘
                 │
                 ▼
┌────────────────────────────────┐       ┌────────────────────────────────┐
│   Market Intelligence Bureau   │ ───>  │  Quantitative Research Bureau  │
│  Technical Factors + NLP Sent  │       │  OU Mean-Reversion + ML P(Up)  │
└────────────────────────────────┘       └───────────────┬────────────────┘
                                                         │
                                                         ▼
┌────────────────────────────────┐       ┌────────────────────────────────┐
│   Execution Commander          │ <───  │   Portfolio Council Risk Gate  │
│   TWAP Slicing & Alpaca Broker │       │   VaR / CVaR & Volatility Cap  │
└────────────────────────────────┘       └────────────────────────────────┘
```

---

## 📦 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/udbhav-shrinet/Multi-Agent-Trading-Bot.git
   cd Multi-Agent-Trading-Bot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `config/.env` file:
   ```env
   ALPACA_API_KEY="your_alpaca_key"
   ALPACA_SECRET_KEY="your_alpaca_secret"
   ALPACA_BASE_URL="https://paper-api.alpaca.markets"
   FRED_API_KEY="your_fred_api_key"
   ```

4. **Run Strategy Pipeline**:
   ```bash
   python main.py
   ```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
