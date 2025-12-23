"""
Portfolio Calculation Engine
Core financial calculations for portfolio analysis
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize, newton, brentq
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PortfolioMetrics:
    """Container for portfolio metrics"""
    total_value: float
    total_cost: float
    total_return: float
    total_return_pct: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    var_95: float
    cvar_95: float
    beta: float
    alpha: float
    r_squared: float
    information_ratio: float
    treynor_ratio: float


class RiskCalculator:
    """
    Static methods for risk metric calculations
    """
    
    @staticmethod
    def daily_returns(prices: pd.Series) -> pd.Series:
        """Calculate daily returns from prices"""
        return prices.pct_change().dropna()
    
    @staticmethod
    def cumulative_returns(returns: pd.Series) -> pd.Series:
        """Calculate cumulative returns from daily returns"""
        return (1 + returns).cumprod() - 1
    
    @staticmethod
    def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate annualized return from daily returns
        """
        if len(returns) == 0:
            return 0.0
        
        total_return = (1 + returns).prod() - 1
        n_periods = len(returns)
        years = n_periods / periods_per_year
        
        if years <= 0:
            return 0.0
        
        return (1 + total_return) ** (1 / years) - 1
    
    @staticmethod
    def volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate annualized volatility (standard deviation)
        """
        if len(returns) < 2:
            return 0.0
        return returns.std() * np.sqrt(periods_per_year)
    
    @staticmethod
    def downside_volatility(returns: pd.Series, threshold: float = 0, periods_per_year: int = 252) -> float:
        """
        Calculate downside volatility (semi-deviation)
        """
        downside_returns = returns[returns < threshold]
        if len(downside_returns) < 2:
            return 0.0
        return downside_returns.std() * np.sqrt(periods_per_year)
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.04, periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe Ratio
        Sharpe = (Rp - Rf) / σp
        """
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / periods_per_year
        
        vol = returns.std()
        if vol == 0:
            return 0.0
        
        return np.sqrt(periods_per_year) * excess_returns.mean() / vol
    
    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.04, periods_per_year: int = 252) -> float:
        """
        Calculate Sortino Ratio
        Sortino = (Rp - Rf) / σd (downside deviation)
        """
        if len(returns) < 2:
            return 0.0
        
        daily_rf = risk_free_rate / periods_per_year
        excess_returns = returns - daily_rf
        
        downside_returns = returns[returns < daily_rf]
        if len(downside_returns) < 2:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        
        downside_std = downside_returns.std()
        if downside_std == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        
        return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std
    
    @staticmethod
    def max_drawdown(returns: pd.Series) -> float:
        """
        Calculate Maximum Drawdown
        MDD = max peak-to-trough decline
        """
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min()
    
    @staticmethod
    def drawdown_series(returns: pd.Series) -> pd.Series:
        """
        Calculate drawdown series over time
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding(min_periods=1).max()
        return (cumulative - running_max) / running_max
    
    @staticmethod
    def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate Calmar Ratio
        Calmar = Annualized Return / |Max Drawdown|
        """
        ann_return = RiskCalculator.annualized_return(returns, periods_per_year)
        mdd = abs(RiskCalculator.max_drawdown(returns))
        
        if mdd == 0:
            return float('inf') if ann_return > 0 else 0.0
        
        return ann_return / mdd
    
    @staticmethod
    def var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Historical Value at Risk
        VaR = percentile of returns at (1-confidence) level
        """
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, (1 - confidence) * 100)
    
    @staticmethod
    def var_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Parametric (Gaussian) VaR
        Assumes normal distribution
        """
        if len(returns) < 2:
            return 0.0
        
        from scipy import stats
        z_score = stats.norm.ppf(1 - confidence)
        return returns.mean() + z_score * returns.std()
    
    @staticmethod
    def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall)
        CVaR = Expected value of returns below VaR
        """
        if len(returns) == 0:
            return 0.0
        
        var = RiskCalculator.var_historical(returns, confidence)
        return returns[returns <= var].mean()
    
    @staticmethod
    def beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate Beta relative to benchmark
        β = Cov(Rp, Rm) / Var(Rm)
        """
        # Align the returns
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned) < 2:
            return 1.0
        
        covariance = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
        variance = aligned.iloc[:, 1].var()
        
        if variance == 0:
            return 1.0
        
        return covariance / variance
    
    @staticmethod
    def alpha(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.04,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Jensen's Alpha
        α = Rp - [Rf + β(Rm - Rf)]
        """
        # Align returns
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned) < 2:
            return 0.0
        
        port_ret = aligned.iloc[:, 0]
        bench_ret = aligned.iloc[:, 1]
        
        beta = RiskCalculator.beta(port_ret, bench_ret)
        
        # Annualize
        port_ann = RiskCalculator.annualized_return(port_ret, periods_per_year)
        bench_ann = RiskCalculator.annualized_return(bench_ret, periods_per_year)
        
        return port_ann - (risk_free_rate + beta * (bench_ann - risk_free_rate))
    
    @staticmethod
    def information_ratio(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Information Ratio
        IR = (Rp - Rb) / Tracking Error
        """
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned) < 2:
            return 0.0
        
        active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        tracking_error = active_returns.std() * np.sqrt(periods_per_year)
        
        if tracking_error == 0:
            return 0.0
        
        active_return_ann = active_returns.mean() * periods_per_year
        return active_return_ann / tracking_error
    
    @staticmethod
    def treynor_ratio(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_free_rate: float = 0.04,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Treynor Ratio
        Treynor = (Rp - Rf) / β
        """
        beta = RiskCalculator.beta(portfolio_returns, benchmark_returns)
        
        if beta == 0:
            return 0.0
        
        excess_return = RiskCalculator.annualized_return(portfolio_returns, periods_per_year) - risk_free_rate
        return excess_return / beta
    
    @staticmethod
    def r_squared(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate R-squared (coefficient of determination)
        """
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned) < 2:
            return 0.0
        
        correlation = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        return correlation ** 2


class ReturnsCalculator:
    """
    Calculate various types of returns
    """
    
    @staticmethod
    def simple_return(start_value: float, end_value: float) -> float:
        """Simple holding period return"""
        if start_value == 0:
            return 0.0
        return (end_value - start_value) / start_value
    
    @staticmethod
    def time_weighted_return(
        portfolio_values: List[float],
        cash_flows: List[float]
    ) -> float:
        """
        Calculate Time-Weighted Return (TWR)
        
        TWR = Product of (1 + sub-period returns) - 1
        Sub-period return = (End - Begin - CF) / (Begin + CF)
        
        Args:
            portfolio_values: List of portfolio values at each period
            cash_flows: List of cash flows (positive = inflow, negative = outflow)
        """
        if len(portfolio_values) < 2:
            return 0.0
        
        sub_period_returns = []
        
        for i in range(len(portfolio_values) - 1):
            start_val = portfolio_values[i]
            end_val = portfolio_values[i + 1]
            cf = cash_flows[i] if i < len(cash_flows) else 0
            
            # Adjust for cash flow at start of period
            adjusted_start = start_val + cf
            
            if adjusted_start == 0:
                hpr = 0
            else:
                hpr = (end_val - adjusted_start) / adjusted_start
            
            sub_period_returns.append(1 + hpr)
        
        return np.prod(sub_period_returns) - 1
    
    @staticmethod
    def money_weighted_return(
        cash_flows: List[float],
        dates: List[datetime],
        final_value: float
    ) -> float:
        """
        Calculate Money-Weighted Return (IRR/MWRR)
        
        Solves: Sum(CF_i / (1+r)^t_i) + FV/(1+r)^T = 0
        
        Args:
            cash_flows: List of cash flows (negative = investment, positive = withdrawal)
            dates: Corresponding dates for each cash flow
            final_value: Final portfolio value
        """
        if len(cash_flows) == 0 or len(dates) == 0:
            return 0.0
        
        # Convert dates to years from first date
        start_date = dates[0]
        years = [(d - start_date).days / 365.25 for d in dates]
        total_years = (dates[-1] - start_date).days / 365.25
        
        def npv(rate):
            """Calculate NPV for given rate"""
            if rate <= -1:
                return float('inf')
            
            pv = sum(cf / (1 + rate) ** t for cf, t in zip(cash_flows, years))
            pv += final_value / (1 + rate) ** total_years
            return pv
        
        # Try to find root
        try:
            # Use Brent's method for robust root finding
            result = brentq(npv, -0.99, 10.0)
            return result
        except (ValueError, RuntimeError):
            # Fallback to Newton's method
            try:
                result = newton(npv, 0.1)
                return result
            except:
                return 0.0
    
    @staticmethod
    def rolling_returns(
        returns: pd.Series,
        window: int = 252
    ) -> pd.Series:
        """Calculate rolling annualized returns"""
        return returns.rolling(window=window).apply(
            lambda x: (1 + x).prod() ** (252 / len(x)) - 1
        )


class PortfolioAnalyzer:
    """
    Main class for portfolio analysis
    """
    
    def __init__(
        self,
        holdings: pd.DataFrame,
        prices: pd.DataFrame,
        transactions: pd.DataFrame = None,
        benchmark_returns: pd.Series = None,
        risk_free_rate: float = 0.04
    ):
        """
        Initialize portfolio analyzer
        
        Args:
            holdings: DataFrame with columns [ticker, quantity, purchase_price, purchase_date]
            prices: DataFrame with columns [date, ticker, close]
            transactions: DataFrame with transaction history
            benchmark_returns: Series of benchmark daily returns
            risk_free_rate: Annual risk-free rate
        """
        self.holdings = holdings
        self.prices = prices
        self.transactions = transactions
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        
        # Calculate portfolio values and returns
        self._calculate_portfolio_series()
    
    def _calculate_portfolio_series(self):
        """Calculate daily portfolio values and returns"""
        if self.prices.empty or self.holdings.empty:
            self.portfolio_values = pd.Series()
            self.portfolio_returns = pd.Series()
            return
        
        # Pivot prices to have dates as index and tickers as columns
        prices_pivot = self.prices.pivot(index='date', columns='ticker', values='close')
        prices_pivot = prices_pivot.ffill()
        
        # Calculate daily portfolio value
        daily_values = []
        
        for date in prices_pivot.index:
            daily_value = 0
            for _, holding in self.holdings.iterrows():
                ticker = holding['ticker']
                quantity = holding['quantity']
                
                if ticker in prices_pivot.columns:
                    price = prices_pivot.loc[date, ticker]
                    if not pd.isna(price):
                        daily_value += quantity * price
            
            daily_values.append({'date': date, 'value': daily_value})
        
        self.portfolio_values = pd.DataFrame(daily_values).set_index('date')['value']
        self.portfolio_returns = self.portfolio_values.pct_change().dropna()
    
    def get_current_value(self) -> float:
        """Get current portfolio value"""
        if len(self.portfolio_values) == 0:
            return 0.0
        return self.portfolio_values.iloc[-1]
    
    def get_total_cost(self) -> float:
        """Get total cost basis"""
        if self.holdings.empty:
            return 0.0
        return (self.holdings['quantity'] * self.holdings['purchase_price']).sum()
    
    def get_holdings_breakdown(self, current_prices: Dict[str, float]) -> pd.DataFrame:
        """Get breakdown of holdings with current values"""
        breakdown = []
        
        for _, holding in self.holdings.iterrows():
            ticker = holding['ticker']
            quantity = holding['quantity']
            cost = holding['purchase_price']
            
            current_price = current_prices.get(ticker, cost)
            current_value = quantity * current_price
            cost_basis = quantity * cost
            gain_loss = current_value - cost_basis
            gain_loss_pct = (gain_loss / cost_basis * 100) if cost_basis > 0 else 0
            
            breakdown.append({
                'ticker': ticker,
                'quantity': quantity,
                'cost_per_share': cost,
                'current_price': current_price,
                'cost_basis': cost_basis,
                'current_value': current_value,
                'gain_loss': gain_loss,
                'gain_loss_pct': gain_loss_pct,
                'weight': 0  # Will be calculated after
            })
        
        df = pd.DataFrame(breakdown)
        
        if len(df) > 0:
            total_value = df['current_value'].sum()
            if total_value > 0:
                df['weight'] = df['current_value'] / total_value * 100
        
        return df
    
    def get_asset_allocation(self) -> Dict[str, float]:
        """Get asset allocation by asset class"""
        if 'asset_class' not in self.holdings.columns:
            return {}
        
        # This would need current prices for accurate allocation
        allocation = self.holdings.groupby('asset_class')['quantity'].sum()
        return allocation.to_dict()
    
    def calculate_all_metrics(self) -> PortfolioMetrics:
        """Calculate all portfolio metrics"""
        returns = self.portfolio_returns
        bench_returns = self.benchmark_returns
        
        # Basic metrics
        total_value = self.get_current_value()
        total_cost = self.get_total_cost()
        total_return = total_value - total_cost
        total_return_pct = (total_return / total_cost * 100) if total_cost > 0 else 0
        
        # Risk metrics
        ann_return = RiskCalculator.annualized_return(returns)
        vol = RiskCalculator.volatility(returns)
        sharpe = RiskCalculator.sharpe_ratio(returns, self.risk_free_rate)
        sortino = RiskCalculator.sortino_ratio(returns, self.risk_free_rate)
        mdd = RiskCalculator.max_drawdown(returns)
        calmar = RiskCalculator.calmar_ratio(returns)
        var_95 = RiskCalculator.var_historical(returns, 0.95)
        cvar_95 = RiskCalculator.cvar(returns, 0.95)
        
        # Benchmark-relative metrics
        if bench_returns is not None and len(bench_returns) > 0:
            beta = RiskCalculator.beta(returns, bench_returns)
            alpha = RiskCalculator.alpha(returns, bench_returns, self.risk_free_rate)
            r_sq = RiskCalculator.r_squared(returns, bench_returns)
            ir = RiskCalculator.information_ratio(returns, bench_returns)
            treynor = RiskCalculator.treynor_ratio(returns, bench_returns, self.risk_free_rate)
        else:
            beta = alpha = r_sq = ir = treynor = 0.0
        
        return PortfolioMetrics(
            total_value=total_value,
            total_cost=total_cost,
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=ann_return,
            volatility=vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=mdd,
            calmar_ratio=calmar,
            var_95=var_95,
            cvar_95=cvar_95,
            beta=beta,
            alpha=alpha,
            r_squared=r_sq,
            information_ratio=ir,
            treynor_ratio=treynor
        )
    
    def get_rolling_metrics(self, window: int = 60) -> pd.DataFrame:
        """Calculate rolling risk metrics"""
        returns = self.portfolio_returns
        
        if len(returns) < window:
            return pd.DataFrame()
        
        rolling_data = []
        
        for i in range(window, len(returns) + 1):
            window_returns = returns.iloc[i-window:i]
            date = returns.index[i-1]
            
            rolling_data.append({
                'date': date,
                'sharpe': RiskCalculator.sharpe_ratio(window_returns, self.risk_free_rate),
                'volatility': RiskCalculator.volatility(window_returns),
                'var_95': RiskCalculator.var_historical(window_returns, 0.95),
            })
        
        return pd.DataFrame(rolling_data).set_index('date')
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix between holdings"""
        if self.prices.empty:
            return pd.DataFrame()
        
        # Pivot to get returns per ticker
        prices_pivot = self.prices.pivot(index='date', columns='ticker', values='close')
        returns = prices_pivot.pct_change().dropna()
        
        return returns.corr()
    
    def get_contribution_analysis(self) -> pd.DataFrame:
        """Analyze contribution of each holding to portfolio returns"""
        if self.prices.empty or self.holdings.empty:
            return pd.DataFrame()
        
        # Calculate individual asset returns
        prices_pivot = self.prices.pivot(index='date', columns='ticker', values='close')
        returns = prices_pivot.pct_change().dropna()
        
        # Weight by position size
        contributions = []
        
        for _, holding in self.holdings.iterrows():
            ticker = holding['ticker']
            if ticker not in returns.columns:
                continue
            
            asset_returns = returns[ticker]
            total_return = (1 + asset_returns).prod() - 1
            
            contributions.append({
                'ticker': ticker,
                'total_return': total_return,
                'avg_daily_return': asset_returns.mean(),
                'volatility': asset_returns.std() * np.sqrt(252),
            })
        
        return pd.DataFrame(contributions)


class EfficientFrontier:
    """
    Mean-Variance Optimization and Efficient Frontier
    """
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.04):
        """
        Initialize with asset returns
        
        Args:
            returns: DataFrame of daily returns (columns = assets)
            risk_free_rate: Annual risk-free rate
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns.columns)
        
        # Calculate expected returns and covariance (annualized)
        self.expected_returns = returns.mean() * 252
        self.cov_matrix = returns.cov() * 252
    
    def portfolio_return(self, weights: np.ndarray) -> float:
        """Calculate portfolio expected return"""
        return np.dot(weights, self.expected_returns)
    
    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """Calculate portfolio volatility"""
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
    
    def portfolio_sharpe(self, weights: np.ndarray) -> float:
        """Calculate portfolio Sharpe ratio"""
        ret = self.portfolio_return(weights)
        vol = self.portfolio_volatility(weights)
        if vol == 0:
            return 0
        return (ret - self.risk_free_rate) / vol
    
    def minimize_volatility(self, target_return: float = None) -> dict:
        """Find minimum volatility portfolio"""
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: self.portfolio_return(x) - target_return
            })
        
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        init_weights = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(
            lambda x: self.portfolio_volatility(x),
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return {
            'weights': result.x,
            'return': self.portfolio_return(result.x),
            'volatility': self.portfolio_volatility(result.x),
            'sharpe': self.portfolio_sharpe(result.x)
        }
    
    def maximize_sharpe(self) -> dict:
        """Find maximum Sharpe ratio portfolio"""
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        init_weights = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(
            lambda x: -self.portfolio_sharpe(x),
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return {
            'weights': result.x,
            'return': self.portfolio_return(result.x),
            'volatility': self.portfolio_volatility(result.x),
            'sharpe': self.portfolio_sharpe(result.x)
        }
    
    def calculate_frontier(self, n_points: int = 100) -> pd.DataFrame:
        """Calculate the efficient frontier"""
        # Get return range
        min_ret = self.expected_returns.min()
        max_ret = self.expected_returns.max()
        
        target_returns = np.linspace(min_ret, max_ret, n_points)
        frontier_volatilities = []
        frontier_sharpes = []
        
        for target in target_returns:
            try:
                result = self.minimize_volatility(target)
                frontier_volatilities.append(result['volatility'])
                frontier_sharpes.append(result['sharpe'])
            except:
                frontier_volatilities.append(np.nan)
                frontier_sharpes.append(np.nan)
        
        return pd.DataFrame({
            'return': target_returns,
            'volatility': frontier_volatilities,
            'sharpe': frontier_sharpes
        }).dropna()


class MonteCarloSimulator:
    """
    Monte Carlo simulation for portfolio projections
    """
    
    def __init__(self, returns: pd.Series):
        """
        Initialize with historical returns
        """
        self.returns = returns
        self.mu = returns.mean()
        self.sigma = returns.std()
    
    def simulate_paths(
        self,
        n_simulations: int = 1000,
        n_days: int = 252,
        initial_value: float = 100
    ) -> np.ndarray:
        """
        Simulate future portfolio paths
        
        Returns:
            Array of shape (n_days, n_simulations)
        """
        simulations = np.zeros((n_days, n_simulations))
        simulations[0] = initial_value
        
        for t in range(1, n_days):
            random_returns = np.random.normal(self.mu, self.sigma, n_simulations)
            simulations[t] = simulations[t-1] * (1 + random_returns)
        
        return simulations
    
    def get_percentile_paths(
        self,
        simulations: np.ndarray,
        percentiles: List[int] = [5, 25, 50, 75, 95]
    ) -> pd.DataFrame:
        """Get percentile paths from simulations"""
        result = {}
        for p in percentiles:
            result[f'p{p}'] = np.percentile(simulations, p, axis=1)
        return pd.DataFrame(result)
    
    def probability_of_return(
        self,
        simulations: np.ndarray,
        target_return: float
    ) -> float:
        """Calculate probability of achieving target return"""
        final_values = simulations[-1]
        initial_value = simulations[0, 0]
        target_value = initial_value * (1 + target_return)
        
        return np.mean(final_values >= target_value)


if __name__ == "__main__":
    # Test the calculator
    import numpy as np
    
    # Generate sample returns
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0005, 0.02, 252))
    
    print("Sample Return Statistics:")
    print(f"Mean daily return: {returns.mean():.4%}")
    print(f"Annualized return: {RiskCalculator.annualized_return(returns):.2%}")
    print(f"Volatility: {RiskCalculator.volatility(returns):.2%}")
    print(f"Sharpe Ratio: {RiskCalculator.sharpe_ratio(returns):.2f}")
    print(f"Max Drawdown: {RiskCalculator.max_drawdown(returns):.2%}")
    print(f"VaR (95%): {RiskCalculator.var_historical(returns, 0.95):.2%}")
    print(f"CVaR (95%): {RiskCalculator.cvar(returns, 0.95):.2%}")
