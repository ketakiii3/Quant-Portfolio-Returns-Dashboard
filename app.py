"""
Portfolio Returns Dashboard
Main Streamlit Application

A comprehensive quant finance dashboard for portfolio analysis,
risk metrics, and performance tracking.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# Import our modules
from data_fetcher import DataFetcher, BenchmarkData, get_risk_free_rate
from calculations import (
    RiskCalculator, ReturnsCalculator, PortfolioAnalyzer, 
    EfficientFrontier, MonteCarloSimulator, PortfolioMetrics
)
from visualizations import (
    plot_cumulative_returns, plot_drawdown, plot_asset_allocation,
    plot_holdings_performance, plot_correlation_heatmap, plot_rolling_sharpe,
    plot_var_distribution, plot_efficient_frontier, plot_monte_carlo,
    plot_sector_allocation, plot_performance_metrics, COLORS
)
from utils import (
    format_currency, format_percentage, format_number,
    parse_holdings_csv, generate_sample_portfolio, calculate_date_range,
    get_color_for_value, export_holdings_template, PerformanceTracker
)
from models import init_db

# Page config
st.set_page_config(
    page_title="Quant Portfolio Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme styling
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #0f172a;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f1f5f9 !important;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        color: #f1f5f9 !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 1rem !important;
    }
    
    /* Custom metric card */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.1);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }
    
    .metric-delta {
        font-size: 0.875rem;
        margin-top: 0.5rem;
    }
    
    .positive { color: #10b981; }
    .negative { color: #ef4444; }
    
    /* Tables */
    .dataframe {
        background-color: #1e293b !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 8px;
        color: #94a3b8;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: #f1f5f9 !important;
    }
    
    /* Dividers */
    hr {
        border-color: rgba(148, 163, 184, 0.2);
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #1e293b;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'holdings' not in st.session_state:
    st.session_state.holdings = None
if 'prices' not in st.session_state:
    st.session_state.prices = None
if 'fetcher' not in st.session_state:
    st.session_state.fetcher = DataFetcher()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False


def create_metric_card(value: str, label: str, delta: str = None, delta_positive: bool = True):
    """Create a custom metric card"""
    delta_class = "positive" if delta_positive else "negative"
    delta_html = f'<p class="metric-delta {delta_class}">{delta}</p>' if delta else ''
    
    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value">{value}</p>
        <p class="metric-label">{label}</p>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def load_sample_data():
    """Load sample portfolio data"""
    sample = generate_sample_portfolio()
    st.session_state.holdings = sample['holdings']
    st.session_state.transactions = sample['transactions']
    st.session_state.data_loaded = True
    return sample['holdings']


def fetch_portfolio_data(holdings: pd.DataFrame, start_date: datetime, end_date: datetime):
    """Fetch price data for all holdings"""
    tickers = holdings['ticker'].unique().tolist()
    
    with st.spinner('Fetching market data... (this may take a moment)'):
        fetcher = st.session_state.fetcher
        
        try:
            prices = fetcher.fetch_price_data(tickers, start_date, end_date)
            
            # Check which tickers failed
            if not prices.empty:
                fetched_tickers = prices['ticker'].unique().tolist()
                failed_tickers = [t for t in tickers if t not in fetched_tickers]
                
                if failed_tickers:
                    st.warning(f"⚠️ Could not fetch data for: {', '.join(failed_tickers)}")
            
        except Exception as e:
            st.error(f"Error fetching portfolio data: {e}")
            prices = pd.DataFrame()
        
        # Also fetch benchmark
        try:
            benchmark_prices = fetcher.fetch_price_data(['SPY'], start_date, end_date)
        except Exception as e:
            st.warning(f"Could not fetch benchmark (SPY) data: {e}")
            benchmark_prices = pd.DataFrame()
        
    return prices, benchmark_prices


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## 📊 Portfolio Dashboard")
    st.markdown("---")
    
    # Portfolio input method
    input_method = st.radio(
        "Data Source",
        ["📁 Upload CSV", "📝 Manual Entry", "🎯 Sample Portfolio"],
        index=2
    )
    
    if input_method == "📁 Upload CSV":
        st.markdown("### Upload Holdings")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV with columns: ticker, quantity, purchase_price, purchase_date"
        )
        
        if uploaded_file:
            try:
                holdings = parse_holdings_csv(uploaded_file)
                st.session_state.holdings = holdings
                st.session_state.data_loaded = True
                st.success(f"Loaded {len(holdings)} holdings")
            except Exception as e:
                st.error(f"Error parsing file: {e}")
        
        # Download template
        st.download_button(
            "📥 Download Template",
            export_holdings_template(),
            file_name="holdings_template.csv",
            mime="text/csv"
        )
    
    elif input_method == "📝 Manual Entry":
        st.markdown("### Add Holdings")
        
        with st.form("add_holding"):
            ticker = st.text_input("Ticker Symbol", placeholder="AAPL")
            quantity = st.number_input("Quantity", min_value=0.0, step=1.0)
            price = st.number_input("Purchase Price ($)", min_value=0.0, step=0.01)
            date = st.date_input("Purchase Date", value=datetime.now())
            asset_class = st.selectbox("Asset Class", ["Equity", "ETF", "Bond", "Crypto", "Other"])
            
            if st.form_submit_button("Add Holding"):
                new_holding = pd.DataFrame([{
                    'ticker': ticker.upper(),
                    'quantity': quantity,
                    'purchase_price': price,
                    'purchase_date': date,
                    'asset_class': asset_class,
                    'sector': 'Unknown'
                }])
                
                if st.session_state.holdings is None:
                    st.session_state.holdings = new_holding
                else:
                    st.session_state.holdings = pd.concat(
                        [st.session_state.holdings, new_holding],
                        ignore_index=True
                    )
                st.session_state.data_loaded = True
                st.success(f"Added {ticker}")
                st.rerun()
    
    else:  # Sample Portfolio
        if st.button("🎯 Load Sample Portfolio", use_container_width=True):
            load_sample_data()
            st.success("Sample portfolio loaded!")
            st.rerun()
    
    st.markdown("---")
    
    # Time period selection
    st.markdown("### Time Period")
    period = st.selectbox(
        "Analysis Period",
        ["1M", "3M", "6M", "YTD", "1Y", "2Y", "3Y", "5Y"],
        index=4
    )
    
    # Benchmark selection
    st.markdown("### Benchmark")
    benchmark = st.selectbox(
        "Compare Against",
        ["SPY (S&P 500)", "QQQ (Nasdaq 100)", "DIA (Dow Jones)", "IWM (Russell 2000)", "VTI (Total Market)"],
        index=0
    )
    benchmark_ticker = benchmark.split(" ")[0]
    
    # Risk-free rate
    st.markdown("### Risk Parameters")
    risk_free_rate = st.slider(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=4.5,
        step=0.1
    ) / 100
    
    st.markdown("---")
    
    # Current holdings display
    if st.session_state.holdings is not None:
        st.markdown("### Current Holdings")
        st.dataframe(
            st.session_state.holdings[['ticker', 'quantity', 'purchase_price']].style.format({
                'quantity': '{:.2f}',
                'purchase_price': '${:.2f}'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        if st.button("🗑️ Clear Portfolio", use_container_width=True):
            st.session_state.holdings = None
            st.session_state.data_loaded = False
            st.rerun()


# =============================================================================
# MAIN CONTENT
# =============================================================================

st.markdown("# 📊 Quant Portfolio Returns Dashboard")
st.markdown("Real-time portfolio analytics, risk metrics, and performance attribution")
st.markdown("---")

if st.session_state.holdings is None or len(st.session_state.holdings) == 0:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px;">
        <h2 style="color: #94a3b8;">Welcome to the Portfolio Dashboard</h2>
        <p style="color: #64748b; font-size: 1.1rem; max-width: 600px; margin: 20px auto;">
            Get started by uploading your holdings CSV, entering positions manually, 
            or loading the sample portfolio from the sidebar.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Returns Analysis</h3>
            <p style="color: #94a3b8;">Time-weighted & money-weighted returns with benchmark comparison</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚠️ Risk Metrics</h3>
            <p style="color: #94a3b8;">Sharpe, Sortino, VaR, CVaR, max drawdown, and more</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Optimization</h3>
            <p style="color: #94a3b8;">Efficient frontier and optimal portfolio analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>🔮 Simulation</h3>
            <p style="color: #94a3b8;">Monte Carlo projections for future scenarios</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.stop()


# Load and process data
holdings = st.session_state.holdings
start_date, end_date = calculate_date_range(period)

# Fetch price data
try:
    prices, benchmark_prices = fetch_portfolio_data(holdings, start_date, end_date)
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

if prices.empty:
    st.warning("Could not fetch price data. Please check your holdings and try again.")
    st.stop()

# Get current prices for holdings
fetcher = st.session_state.fetcher
current_prices = fetcher.fetch_multiple_prices(holdings['ticker'].tolist())

# Create portfolio analyzer
benchmark_returns = None
if not benchmark_prices.empty:
    bench_df = benchmark_prices[benchmark_prices['ticker'] == benchmark_ticker].sort_values('date')
    if not bench_df.empty and 'close' in bench_df.columns:
        benchmark_returns = bench_df.set_index('date')['close'].pct_change().dropna()

# If benchmark failed, try using SPY directly from prices if available
if benchmark_returns is None or len(benchmark_returns) == 0:
    if 'SPY' in prices['ticker'].unique():
        spy_df = prices[prices['ticker'] == 'SPY'].sort_values('date')
        if not spy_df.empty:
            benchmark_returns = spy_df.set_index('date')['close'].pct_change().dropna()
    
    if benchmark_returns is None or len(benchmark_returns) == 0:
        st.info("📊 Benchmark data unavailable. Showing portfolio metrics without benchmark comparison.")

analyzer = PortfolioAnalyzer(
    holdings=holdings,
    prices=prices,
    benchmark_returns=benchmark_returns,
    risk_free_rate=risk_free_rate
)

# Calculate metrics
metrics = analyzer.calculate_all_metrics()
holdings_breakdown = analyzer.get_holdings_breakdown(current_prices)

# =============================================================================
# KEY METRICS ROW
# =============================================================================

st.markdown("## Key Metrics")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    delta_val = f"+{format_percentage(metrics.total_return_pct, with_sign=True)}" if metrics.total_return_pct >= 0 else format_percentage(metrics.total_return_pct)
    st.metric(
        "Portfolio Value",
        format_currency(metrics.total_value),
        delta=delta_val
    )

with col2:
    st.metric(
        "Total Return",
        format_currency(metrics.total_return),
        delta=format_percentage(metrics.total_return_pct, with_sign=True)
    )

with col3:
    st.metric(
        "Annualized Return",
        format_percentage(metrics.annualized_return * 100),
    )

with col4:
    st.metric(
        "Volatility",
        format_percentage(metrics.volatility * 100),
    )

with col5:
    st.metric(
        "Sharpe Ratio",
        format_number(metrics.sharpe_ratio),
    )

with col6:
    st.metric(
        "Max Drawdown",
        format_percentage(metrics.max_drawdown * 100),
    )

st.markdown("---")

# =============================================================================
# MAIN CHARTS
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Performance",
    "⚠️ Risk Analysis", 
    "🏆 Holdings",
    "🎯 Optimization",
    "🔮 Simulation"
])

# TAB 1: Performance
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Cumulative Returns")
        
        returns = analyzer.portfolio_returns
        fig = plot_cumulative_returns(
            returns,
            benchmark_returns,
            benchmark_name=benchmark.split(" ")[1].strip("()"),
            title=""
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Performance Stats")
        
        daily_stats = PerformanceTracker.daily_stats(returns)
        
        st.markdown(f"""
        <div class="metric-card">
            <p style="color: #94a3b8; margin-bottom: 10px;">Daily Statistics</p>
            <p><span style="color: #10b981;">Best Day:</span> {format_percentage(daily_stats['best_day']*100)}</p>
            <p><span style="color: #ef4444;">Worst Day:</span> {format_percentage(daily_stats['worst_day']*100)}</p>
            <p><span style="color: #f1f5f9;">Avg Daily:</span> {format_percentage(daily_stats['avg_daily']*100)}</p>
            <p><span style="color: #10b981;">Positive Days:</span> {daily_stats['positive_days']}</p>
            <p><span style="color: #ef4444;">Negative Days:</span> {daily_stats['negative_days']}</p>
            <p><span style="color: #2563eb;">Win Rate:</span> {format_percentage(daily_stats['win_rate'])}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Drawdown chart
    st.markdown("### Drawdown Analysis")
    fig = plot_drawdown(returns, title="")
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly returns heatmap
    if len(returns) > 30:
        st.markdown("### Monthly Returns")
        monthly = PerformanceTracker.monthly_returns(returns)
        
        if not monthly.empty:
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=monthly.values * 100,
                x=monthly.columns,
                y=monthly.index,
                colorscale='RdYlGn',
                zmid=0,
                text=np.round(monthly.values * 100, 1),
                texttemplate='%{text}%',
                textfont={"size": 10},
                hovertemplate='%{y} %{x}: %{z:.2f}%<extra></extra>',
                colorbar=dict(title='Return %', tickformat='.1f')
            ))
            fig.update_layout(
                height=300,
                margin=dict(l=40, r=40, t=20, b=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f1f5f9')
            )
            st.plotly_chart(fig, use_container_width=True)


# TAB 2: Risk Analysis
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Risk Metrics Dashboard")
        
        metrics_dict = {
            'sharpe_ratio': metrics.sharpe_ratio,
            'sortino_ratio': min(metrics.sortino_ratio, 5),  # Cap for display
            'max_drawdown': metrics.max_drawdown,
            'calmar_ratio': min(metrics.calmar_ratio, 5),
        }
        
        fig = plot_performance_metrics(metrics_dict, title="")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Return Distribution & VaR")
        
        fig = plot_var_distribution(
            returns,
            metrics.var_95,
            metrics.cvar_95,
            title=""
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics
    st.markdown("### Detailed Risk Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("VaR (95%)", format_percentage(metrics.var_95 * 100))
        st.metric("CVaR (95%)", format_percentage(metrics.cvar_95 * 100))
    
    with col2:
        st.metric("Beta", format_number(metrics.beta))
        st.metric("Alpha", format_percentage(metrics.alpha * 100))
    
    with col3:
        st.metric("R-Squared", format_number(metrics.r_squared))
        st.metric("Information Ratio", format_number(metrics.information_ratio))
    
    with col4:
        st.metric("Treynor Ratio", format_number(metrics.treynor_ratio))
        st.metric("Sortino Ratio", format_number(min(metrics.sortino_ratio, 10)))
    
    # Rolling metrics
    if len(returns) > 60:
        st.markdown("### Rolling 60-Day Sharpe Ratio")
        rolling_metrics = analyzer.get_rolling_metrics(window=60)
        if not rolling_metrics.empty:
            fig = plot_rolling_sharpe(rolling_metrics['sharpe'], window=60, title="")
            st.plotly_chart(fig, use_container_width=True)


# TAB 3: Holdings
with tab3:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Asset Allocation")
        
        if not holdings_breakdown.empty:
            fig = plot_asset_allocation(
                holdings_breakdown,
                value_column='current_value',
                title=""
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Holdings Performance")
        
        if not holdings_breakdown.empty:
            fig = plot_holdings_performance(holdings_breakdown, title="")
            st.plotly_chart(fig, use_container_width=True)
    
    # Holdings table
    st.markdown("### Holdings Detail")
    
    if not holdings_breakdown.empty:
        display_df = holdings_breakdown[[
            'ticker', 'quantity', 'cost_per_share', 'current_price',
            'cost_basis', 'current_value', 'gain_loss', 'gain_loss_pct', 'weight'
        ]].copy()
        
        # Format for display
        st.dataframe(
            display_df.style.format({
                'quantity': '{:.2f}',
                'cost_per_share': '${:.2f}',
                'current_price': '${:.2f}',
                'cost_basis': '${:,.2f}',
                'current_value': '${:,.2f}',
                'gain_loss': '${:,.2f}',
                'gain_loss_pct': '{:+.2f}%',
                'weight': '{:.1f}%'
            }).applymap(
                lambda x: 'color: #10b981' if isinstance(x, str) and '+' in x else 
                         ('color: #ef4444' if isinstance(x, str) and '-' in x else ''),
                subset=['gain_loss_pct']
            ),
            use_container_width=True,
            hide_index=True
        )
    
    # Correlation matrix
    st.markdown("### Correlation Matrix")
    corr_matrix = analyzer.get_correlation_matrix()
    
    if not corr_matrix.empty and len(corr_matrix) > 1:
        fig = plot_correlation_heatmap(corr_matrix, title="")
        st.plotly_chart(fig, use_container_width=True)


# TAB 4: Optimization
with tab4:
    st.markdown("### Efficient Frontier Analysis")
    st.markdown("Find the optimal portfolio allocation using mean-variance optimization.")
    
    # Get asset returns
    prices_pivot = prices.pivot(index='date', columns='ticker', values='close')
    asset_returns = prices_pivot.pct_change().dropna()
    
    if len(asset_returns.columns) >= 2:
        try:
            ef = EfficientFrontier(asset_returns, risk_free_rate)
            
            # Calculate efficient frontier
            frontier = ef.calculate_frontier(n_points=50)
            
            # Get optimal portfolios
            min_vol = ef.minimize_volatility()
            max_sharpe = ef.maximize_sharpe()
            
            # Current portfolio metrics
            current_vol = metrics.volatility
            current_ret = metrics.annualized_return
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = plot_efficient_frontier(
                    frontier,
                    current_portfolio=(current_vol, current_ret),
                    optimal_portfolio=(max_sharpe['volatility'], max_sharpe['return']),
                    title=""
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Optimal Portfolios")
                
                st.markdown("**Maximum Sharpe Ratio**")
                st.markdown(f"""
                - Expected Return: {format_percentage(max_sharpe['return'] * 100)}
                - Volatility: {format_percentage(max_sharpe['volatility'] * 100)}
                - Sharpe Ratio: {format_number(max_sharpe['sharpe'])}
                """)
                
                st.markdown("**Optimal Weights:**")
                for ticker, weight in zip(asset_returns.columns, max_sharpe['weights']):
                    if weight > 0.01:
                        st.markdown(f"- {ticker}: {format_percentage(weight * 100)}")
                
                st.markdown("---")
                
                st.markdown("**Minimum Volatility**")
                st.markdown(f"""
                - Expected Return: {format_percentage(min_vol['return'] * 100)}
                - Volatility: {format_percentage(min_vol['volatility'] * 100)}
                - Sharpe Ratio: {format_number(min_vol['sharpe'])}
                """)
        
        except Exception as e:
            st.warning(f"Could not compute efficient frontier: {e}")
    else:
        st.info("Need at least 2 assets for portfolio optimization.")


# TAB 5: Simulation
with tab5:
    st.markdown("### Monte Carlo Simulation")
    st.markdown("Project possible future portfolio values using historical return patterns.")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        n_simulations = st.slider("Number of Simulations", 100, 5000, 1000, step=100)
        n_days = st.slider("Projection Days", 30, 504, 252, step=21)
        initial_value = metrics.total_value
        
        st.markdown(f"**Initial Value:** {format_currency(initial_value)}")
    
    with col2:
        if len(returns) > 20:
            simulator = MonteCarloSimulator(returns)
            
            with st.spinner("Running simulations..."):
                simulations = simulator.simulate_paths(
                    n_simulations=n_simulations,
                    n_days=n_days,
                    initial_value=initial_value
                )
                
                percentiles = simulator.get_percentile_paths(
                    simulations,
                    percentiles=[5, 25, 50, 75, 95]
                )
            
            fig = plot_monte_carlo(
                simulations,
                percentiles,
                initial_value,
                title=""
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.markdown("### Projection Summary")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            final_values = simulations[-1]
            
            with col1:
                st.metric("5th Percentile", format_currency(np.percentile(final_values, 5)))
            with col2:
                st.metric("25th Percentile", format_currency(np.percentile(final_values, 25)))
            with col3:
                st.metric("Median", format_currency(np.median(final_values)))
            with col4:
                st.metric("75th Percentile", format_currency(np.percentile(final_values, 75)))
            with col5:
                st.metric("95th Percentile", format_currency(np.percentile(final_values, 95)))
            
            # Probability analysis
            st.markdown("### Probability Analysis")
            
            target_returns = [0, 0.05, 0.10, 0.20, 0.50]
            probs = []
            
            for target in target_returns:
                prob = simulator.probability_of_return(simulations, target)
                probs.append(f"{prob*100:.1f}%")
            
            prob_df = pd.DataFrame({
                'Target Return': ['Break Even', '+5%', '+10%', '+20%', '+50%'],
                'Probability': probs
            })
            
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Need more historical data for Monte Carlo simulation.")


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 20px;">
    <p>Portfolio Dashboard v1.0 | Built with Streamlit & Plotly</p>
    <p>Data provided by Yahoo Finance | Not financial advice</p>
</div>
""", unsafe_allow_html=True)
