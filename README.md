# Hierarchical Multi-Process Algorithmic Trading & Quantitative Research Engine

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Interactive Demo](https://img.shields.io/badge/demo-GitHub%20Pages-purple.svg)](https://udbhav-shrinet.github.io/Multi-Agent-Trading-Bot/)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)]()
[![Alpaca API](https://img.shields.io/badge/broker-Alpaca%20Markets-green.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> Autonomous institutional-grade algorithmic trading architecture combining real-time market intelligence, macro regime detection (FRED), quantitative price forecasting (ARIMA / Exponential Smoothing), and risk-managed execution via Alpaca Markets.

---

## 🚀 Live Interactive Showcase

Inspect live trading simulations, equity curves, portfolio council decisions, and macroeconomic regime scores:  
👉 **[Launch Quantitative Trading Dashboard](https://udbhav-shrinet.github.io/Multi-Agent-Trading-Bot/)**

---

## ✨ Key Capabilities

- **Modular Autonomous Architecture**: Decoupled specialized processes:
  - `market_intelligence.py`: Real-time news scraping via RSS, sentiment analysis, and volatility scoring.
  - `quant_research.py`: Stationary price forecasting, ARIMA models, and historical price indexing via Yahoo Finance.
  - `portfolio_council.py`: Aggregates signals, calculates dynamic Kelly Criterion bet sizing, and enforces portfolio diversification.
  - `execution_commander.py`: Interacts with Alpaca Trade API to submit fractional or whole-share market and limit orders.
  - `position_manager.py`: Monitors real-time drawdown, trailing stops, and takes profit.
- **Macroeconomic Regime Scoring**: Direct integration with Federal Reserve Economic Data (FRED) to evaluate interest rates, yield curve spreads, and liquidity environments.
- **Automated GitHub Actions Scheduler**: Periodic execution during active US Market hours (09:30 - 16:00 EST).
- **Interactive Visual Studio**: High-frequency dashboard featuring live PnL tracking, asset allocation donuts, and trade logs.

---

## 🛠️ Trading Pipeline

```text
┌─────────────────────────┐       ┌────────────────────────┐       ┌──────────────────────┐
│ Candidate Watchlist     │ ───>  │  Market Intelligence & │ ───>  │ Quant Research &     │
│ (500+ S&P / Tech Assets)│       │  RSS News Sentiment    │       │ Time-Series Analysis │
└─────────────────────────┘       └────────────────────────┘       └──────────┬───────────┘
                                                                              │
                                                   ┌──────────────────────────┴──────────────────────────┐
                                                   ▼                                                     ▼
                                       ┌─────────────────────────┐                           ┌───────────────────────┐
                                       │   Portfolio Council &   │                           │  Execution Commander  │
                                       │   Dynamic Risk Sizing   │                           │  Alpaca Markets API   │
                                       └─────────────────────────┘                           └───────────────────────┘
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
