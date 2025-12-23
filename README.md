# 📊 Quant Portfolio Returns Dashboard

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)


A comprehensive real-time portfolio analytics dashboard that brings quantitative analysis and modern portfolio theory to life. Built with Python, Streamlit, and Plotly, this tool calculates 15+ risk metrics, performs mean-variance optimization, runs Monte Carlo simulations, and visualizes portfolio performance against benchmarks.



https://github.com/user-attachments/assets/fecba18c-a8b5-49be-a3fe-64b9271c7620



---

## ✨ Key Features

### 📈 Performance Analytics
- **Cumulative Returns** — Track portfolio growth vs S&P 500, Nasdaq, or custom benchmarks
- **Time-Weighted Returns (TWR)** — Industry-standard performance measurement that eliminates cash flow timing effects
- **Money-Weighted Returns (IRR)** — Captures the impact of your investment timing decisions
- **Drawdown Analysis** — Visualize and quantify peak-to-trough declines
- **Monthly Return Heatmaps** — Calendar view of monthly performance patterns

### ⚠️ Risk Metrics Suite
| Metric | Description |
|--------|-------------|
| **Volatility** | Annualized standard deviation of returns |
| **Sharpe Ratio** | Risk-adjusted return (excess return / volatility) |
| **Sortino Ratio** | Downside risk-adjusted return |
| **Calmar Ratio** | Return / Maximum Drawdown |
| **Max Drawdown** | Largest peak-to-trough decline |
| **VaR (95%)** | Maximum expected loss at 95% confidence |
| **CVaR / Expected Shortfall** | Average loss beyond VaR threshold |
| **Beta** | Sensitivity to market movements |
| **Alpha (Jensen's)** | Excess return vs CAPM prediction |
| **R-Squared** | Correlation to benchmark |
| **Information Ratio** | Active return / Tracking error |
| **Treynor Ratio** | Excess return / Beta |

### 🎯 Portfolio Optimization
- **Efficient Frontier** — Visualize optimal risk-return tradeoffs using mean-variance optimization
- **Maximum Sharpe Portfolio** — Find the optimal risk-adjusted allocation
- **Minimum Volatility Portfolio** — Calculate the lowest risk allocation for given assets
- **Current vs Optimal Comparison** — Compare your current holdings against optimal allocations

### 🔮 Monte Carlo Simulation
- **Future Projections** — Simulate 1,000+ potential portfolio scenarios
- **Confidence Intervals** — 5th to 95th percentile outcome bands
- **Probability Analysis** — Calculate likelihood of achieving target returns
- **Customizable Horizons** — Project outcomes from 1 month to 2+ years ahead

### 📊 Holdings Analysis
- **Asset Allocation** — Interactive pie and donut charts
- **Per-Holding P&L** — Individual position performance tracking
- **Correlation Matrix** — Asset correlation heatmap
- **Sector Breakdown** — Exposure analysis by sector and industry

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/ketakiii3/Quant-Portfolio-Returns-Dashboard.git
cd Quant-Portfolio-Returns-Dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

Open your browser to `http://localhost:8501`

### Docker Deployment

```bash
docker-compose up -d
```

---

## 📁 Project Structure

```
quant-portfolio-dashboard/
│
├── app.py                 # Main Streamlit application (UI + orchestration)
├── calculations.py        # Financial calculations engine
│   ├── RiskCalculator     # All risk metric calculations
│   ├── ReturnsCalculator  # TWR, MWR, rolling returns
│   ├── PortfolioAnalyzer  # Portfolio-level analysis
│   ├── EfficientFrontier  # Mean-variance optimization
│   └── MonteCarloSimulator# Stochastic simulations
│
├── data_fetcher.py        # Market data API integration
│   ├── DataFetcher        # Price data fetching + caching
│   └── BenchmarkData      # Benchmark index handling
│
├── visualizations.py      # Plotly chart components
│   ├── plot_cumulative_returns()
│   ├── plot_drawdown()
│   ├── plot_efficient_frontier()
│   ├── plot_monte_carlo()
│   └── ... (12+ chart functions)
│
├── models.py              # SQLAlchemy database models
├── utils.py               # Helper functions + formatters
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container configuration
├── docker-compose.yml     # Docker orchestration
│
└── tests/
    └── test_calculations.py  # Unit tests (50+ tests)
```

---

## 💡 Technical Highlights

### Quantitative Finance Implementation
- **Modern Portfolio Theory** — Implements Markowitz mean-variance optimization
- **Risk-Adjusted Performance** — Multiple Sharpe-type ratios for comprehensive analysis
- **Tail Risk Metrics** — VaR and CVaR calculations using both parametric and historical methods
- **Factor Models** — CAPM-based alpha and beta calculations
- **Stochastic Modeling** — Monte Carlo simulations with configurable parameters

### Software Engineering
- **Modular Architecture** — Separation of concerns (calculation engine, data layer, UI)
- **Efficient Data Handling** — Vectorized operations with pandas and NumPy
- **Caching Strategy** — SQLite-backed price data cache to minimize API calls
- **Error Handling** — Robust validation for missing data and edge cases
- **Type Safety** — Type hints throughout the codebase
- **Test Coverage** — Comprehensive unit tests for all calculations

### Performance Optimization
- **Lazy Loading** — Data fetched only when needed
- **Incremental Updates** — Avoids recalculating unchanged data
- **Streamlit Caching** — Decorated functions for expensive computations
- **Database Indexing** — Optimized queries for historical data

---

## 📊 Sample Portfolio Format

Upload a CSV file with your holdings in this format:

```csv
ticker,quantity,purchase_price,purchase_date,asset_class,sector
AAPL,50,178.50,2024-01-15,Equity,Technology
MSFT,30,375.00,2024-01-20,Equity,Technology
JPM,40,185.50,2024-01-10,Equity,Financials
JNJ,35,160.00,2024-01-05,Equity,Healthcare
```

**Supported Asset Classes:**
- Equities (US & International)
- ETFs (Equity, Bond, Commodity)
- Bonds
- Cryptocurrencies
- REITs

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Frontend** | Streamlit, Plotly | Interactive dashboard and visualizations |
| **Backend** | Python 3.10+ | Core application logic |
| **Computation** | pandas, NumPy, SciPy | Data processing and numerical analysis |
| **Market Data** | yfinance API | Real-time and historical price data |
| **Database** | SQLAlchemy + SQLite | Price data caching |
| **Optimization** | scipy.optimize | Portfolio optimization algorithms |
| **Testing** | pytest | Unit and integration tests |
| **Deployment** | Docker, Streamlit Cloud | Containerization and hosting |

---

## 📈 Example Use Cases

1. **Portfolio Rebalancing** — Use the efficient frontier to identify optimal allocations and rebalance accordingly
2. **Risk Assessment** — Monitor VaR and CVaR to ensure portfolio risk stays within acceptable limits
3. **Performance Attribution** — Identify which holdings contribute most to returns and risk
4. **Scenario Planning** — Use Monte Carlo simulations to understand range of possible outcomes
5. **Benchmark Comparison** — Track how your strategy performs against market indices

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest --cov=. tests/

# Run specific test file
pytest tests/test_calculations.py -v
```

---

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Risk-free rate for Sharpe calculations
RISK_FREE_RATE = 0.04  # 4% annual

# Monte Carlo parameters
MC_SIMULATIONS = 1000
MC_HORIZON_DAYS = 252

# API settings
CACHE_EXPIRY_DAYS = 1
```

---

## 🚀 Deployment Options

### Streamlit Cloud (Free)
```bash
# Push to GitHub, then deploy at share.streamlit.io
# Automatically detects requirements.txt
```

### Heroku
```bash
heroku create your-app-name
git push heroku main
```

### AWS EC2 + Docker
```bash
# On EC2 instance
docker pull yourusername/portfolio-dashboard
docker run -p 8501:8501 portfolio-dashboard
```

---


## 📚 Learning Resources

**Modern Portfolio Theory:**
- Markowitz, H. (1952). "Portfolio Selection"
- Sharpe, W. (1964). "Capital Asset Prices"

**Risk Management:**
- Jorion, P. "Value at Risk: The New Benchmark for Managing Financial Risk"
- Dowd, K. "Measuring Market Risk"

**Python for Finance:**
- Hilpisch, Y. "Python for Finance"
- Martin, R. "Python for Finance Cookbook"

---
