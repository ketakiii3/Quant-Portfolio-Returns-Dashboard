"""
Unit Tests for Portfolio Dashboard
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import modules to test
from calculations import (
    RiskCalculator, ReturnsCalculator, PortfolioAnalyzer,
    EfficientFrontier, MonteCarloSimulator
)
from utils import (
    format_currency, format_percentage, format_number,
    parse_holdings_csv, validate_ticker_format, calculate_date_range
)


class TestRiskCalculator:
    """Test suite for RiskCalculator class"""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns for testing"""
        np.random.seed(42)
        return pd.Series(np.random.normal(0.0005, 0.02, 252))
    
    @pytest.fixture
    def known_returns(self):
        """Returns with known statistics"""
        return pd.Series([0.01, -0.02, 0.03, 0.01, -0.01, 0.02, -0.005, 0.015])
    
    def test_daily_returns(self):
        """Test daily returns calculation"""
        prices = pd.Series([100, 102, 101, 104, 103])
        returns = RiskCalculator.daily_returns(prices)
        
        assert len(returns) == 4
        assert abs(returns.iloc[0] - 0.02) < 0.0001
    
    def test_cumulative_returns(self, known_returns):
        """Test cumulative returns calculation"""
        cum_returns = RiskCalculator.cumulative_returns(known_returns)
        
        # Final cumulative return should match product formula
        expected = (1 + known_returns).prod() - 1
        assert abs(cum_returns.iloc[-1] - expected) < 0.0001
    
    def test_annualized_return(self, sample_returns):
        """Test annualized return calculation"""
        ann_return = RiskCalculator.annualized_return(sample_returns)
        
        # Should be a reasonable annual return
        assert -1 < ann_return < 2
    
    def test_volatility(self, sample_returns):
        """Test volatility calculation"""
        vol = RiskCalculator.volatility(sample_returns)
        
        # Annualized volatility should be reasonable
        assert 0 < vol < 1
        
        # Should be approximately daily vol * sqrt(252)
        expected = sample_returns.std() * np.sqrt(252)
        assert abs(vol - expected) < 0.0001
    
    def test_sharpe_ratio(self, sample_returns):
        """Test Sharpe ratio calculation"""
        sharpe = RiskCalculator.sharpe_ratio(sample_returns, risk_free_rate=0.04)
        
        # Sharpe should be a finite number
        assert np.isfinite(sharpe)
        assert -5 < sharpe < 5
    
    def test_sharpe_ratio_zero_volatility(self):
        """Test Sharpe ratio with zero volatility"""
        constant_returns = pd.Series([0.001] * 100)
        sharpe = RiskCalculator.sharpe_ratio(constant_returns)
        
        # Should handle zero volatility gracefully
        assert sharpe == 0 or np.isfinite(sharpe)
    
    def test_sortino_ratio(self, sample_returns):
        """Test Sortino ratio calculation"""
        sortino = RiskCalculator.sortino_ratio(sample_returns, risk_free_rate=0.04)
        
        # Sortino should be finite
        assert np.isfinite(sortino) or sortino == float('inf')
    
    def test_max_drawdown(self, sample_returns):
        """Test max drawdown calculation"""
        mdd = RiskCalculator.max_drawdown(sample_returns)
        
        # Max drawdown should be negative or zero
        assert mdd <= 0
        # Should be bounded
        assert mdd >= -1
    
    def test_max_drawdown_known_case(self):
        """Test max drawdown with known values"""
        # Returns that create known drawdown
        returns = pd.Series([0.1, 0.1, -0.15, -0.1, 0.05])
        mdd = RiskCalculator.max_drawdown(returns)
        
        # Should detect the drawdown
        assert mdd < 0
    
    def test_var_historical(self, sample_returns):
        """Test historical VaR calculation"""
        var = RiskCalculator.var_historical(sample_returns, confidence=0.95)
        
        # VaR should be negative (loss) at 95% confidence
        assert var < 0
        
        # Should be the 5th percentile
        expected = np.percentile(sample_returns, 5)
        assert abs(var - expected) < 0.0001
    
    def test_cvar(self, sample_returns):
        """Test Conditional VaR calculation"""
        var = RiskCalculator.var_historical(sample_returns, confidence=0.95)
        cvar = RiskCalculator.cvar(sample_returns, confidence=0.95)
        
        # CVaR should be less than or equal to VaR (more negative)
        assert cvar <= var
    
    def test_beta(self):
        """Test beta calculation"""
        np.random.seed(42)
        market = pd.Series(np.random.normal(0.0005, 0.01, 100))
        portfolio = 1.2 * market + pd.Series(np.random.normal(0, 0.005, 100))
        
        beta = RiskCalculator.beta(portfolio, market)
        
        # Beta should be close to 1.2
        assert abs(beta - 1.2) < 0.3
    
    def test_r_squared(self):
        """Test R-squared calculation"""
        np.random.seed(42)
        market = pd.Series(np.random.normal(0, 0.01, 100))
        portfolio = market + pd.Series(np.random.normal(0, 0.001, 100))
        
        r_sq = RiskCalculator.r_squared(portfolio, market)
        
        # R-squared should be high since portfolio closely tracks market
        assert 0.8 < r_sq <= 1


class TestReturnsCalculator:
    """Test suite for ReturnsCalculator class"""
    
    def test_simple_return(self):
        """Test simple return calculation"""
        ret = ReturnsCalculator.simple_return(100, 110)
        assert ret == 0.1
        
        ret = ReturnsCalculator.simple_return(100, 90)
        assert ret == -0.1
    
    def test_simple_return_zero_start(self):
        """Test simple return with zero start value"""
        ret = ReturnsCalculator.simple_return(0, 100)
        assert ret == 0
    
    def test_time_weighted_return(self):
        """Test time-weighted return calculation"""
        # Simple case: no cash flows
        values = [100, 110, 115]
        cash_flows = [0, 0]
        
        twr = ReturnsCalculator.time_weighted_return(values, cash_flows)
        
        # Should equal simple return
        expected = (115 / 100) - 1
        assert abs(twr - expected) < 0.0001
    
    def test_time_weighted_return_with_cash_flows(self):
        """Test TWR with cash flows"""
        # Portfolio values and cash flows
        values = [100, 120, 150]
        cash_flows = [0, 10]  # $10 deposit after first period
        
        twr = ReturnsCalculator.time_weighted_return(values, cash_flows)
        
        # TWR should account for cash flows correctly
        assert np.isfinite(twr)


class TestEfficientFrontier:
    """Test suite for EfficientFrontier class"""
    
    @pytest.fixture
    def sample_asset_returns(self):
        """Generate sample multi-asset returns"""
        np.random.seed(42)
        n = 252
        
        returns = pd.DataFrame({
            'AAPL': np.random.normal(0.0008, 0.02, n),
            'GOOGL': np.random.normal(0.0007, 0.025, n),
            'MSFT': np.random.normal(0.0006, 0.018, n),
        })
        return returns
    
    def test_portfolio_return(self, sample_asset_returns):
        """Test portfolio return calculation"""
        ef = EfficientFrontier(sample_asset_returns)
        
        weights = np.array([0.5, 0.3, 0.2])
        ret = ef.portfolio_return(weights)
        
        assert np.isfinite(ret)
    
    def test_portfolio_volatility(self, sample_asset_returns):
        """Test portfolio volatility calculation"""
        ef = EfficientFrontier(sample_asset_returns)
        
        weights = np.array([0.5, 0.3, 0.2])
        vol = ef.portfolio_volatility(weights)
        
        assert vol > 0
        assert np.isfinite(vol)
    
    def test_maximize_sharpe(self, sample_asset_returns):
        """Test maximum Sharpe portfolio"""
        ef = EfficientFrontier(sample_asset_returns)
        result = ef.maximize_sharpe()
        
        # Weights should sum to 1
        assert abs(sum(result['weights']) - 1) < 0.001
        
        # All weights should be non-negative
        assert all(w >= -0.001 for w in result['weights'])
    
    def test_minimize_volatility(self, sample_asset_returns):
        """Test minimum volatility portfolio"""
        ef = EfficientFrontier(sample_asset_returns)
        result = ef.minimize_volatility()
        
        # Weights should sum to 1
        assert abs(sum(result['weights']) - 1) < 0.001
        
        # Volatility should be positive
        assert result['volatility'] > 0


class TestMonteCarloSimulator:
    """Test suite for MonteCarloSimulator class"""
    
    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns for simulation"""
        np.random.seed(42)
        return pd.Series(np.random.normal(0.0005, 0.02, 252))
    
    def test_simulate_paths(self, sample_returns):
        """Test path simulation"""
        simulator = MonteCarloSimulator(sample_returns)
        
        simulations = simulator.simulate_paths(
            n_simulations=100,
            n_days=50,
            initial_value=100
        )
        
        # Check shape
        assert simulations.shape == (50, 100)
        
        # All paths start at initial value
        assert all(simulations[0] == 100)
        
        # Values should be positive
        assert (simulations > 0).all()
    
    def test_get_percentile_paths(self, sample_returns):
        """Test percentile path extraction"""
        simulator = MonteCarloSimulator(sample_returns)
        simulations = simulator.simulate_paths(n_simulations=1000, n_days=50)
        
        percentiles = simulator.get_percentile_paths(simulations)
        
        # Check all percentile columns exist
        for p in [5, 25, 50, 75, 95]:
            assert f'p{p}' in percentiles.columns
        
        # Percentiles should be ordered correctly
        assert (percentiles['p5'] <= percentiles['p50']).all()
        assert (percentiles['p50'] <= percentiles['p95']).all()
    
    def test_probability_of_return(self, sample_returns):
        """Test probability calculation"""
        simulator = MonteCarloSimulator(sample_returns)
        simulations = simulator.simulate_paths(n_simulations=1000, n_days=252)
        
        # Probability of positive return should be between 0 and 1
        prob = simulator.probability_of_return(simulations, target_return=0)
        assert 0 <= prob <= 1
        
        # Probability of very high return should be low
        prob_high = simulator.probability_of_return(simulations, target_return=1.0)
        assert prob_high < 0.5


class TestUtilityFunctions:
    """Test suite for utility functions"""
    
    def test_format_currency(self):
        """Test currency formatting"""
        assert format_currency(1000) == '$1.00K'
        assert format_currency(1500000) == '$1.50M'
        assert format_currency(2000000000) == '$2.00B'
        assert format_currency(50) == '$50.00'
    
    def test_format_currency_none(self):
        """Test currency formatting with None"""
        assert format_currency(None) == '$--'
        assert format_currency(np.nan) == '$--'
    
    def test_format_percentage(self):
        """Test percentage formatting"""
        assert format_percentage(15.5) == '15.50%'
        assert format_percentage(15.5, with_sign=True) == '+15.50%'
        assert format_percentage(-5.2, with_sign=True) == '-5.20%'
    
    def test_format_number(self):
        """Test number formatting"""
        assert format_number(1234.567) == '1,234.57'
        assert format_number(1.5) == '1.50'
    
    def test_validate_ticker_format(self):
        """Test ticker validation"""
        assert validate_ticker_format('AAPL') == True
        assert validate_ticker_format('BRK.B') == True
        assert validate_ticker_format('BTC-USD') == True
        assert validate_ticker_format('') == False
        assert validate_ticker_format(None) == False
    
    def test_calculate_date_range(self):
        """Test date range calculation"""
        start, end = calculate_date_range('1Y')
        
        # End should be today
        assert end.date() == datetime.now().date()
        
        # Start should be approximately 1 year ago
        diff = (end - start).days
        assert 360 < diff < 370
    
    def test_calculate_date_range_ytd(self):
        """Test YTD date range"""
        start, end = calculate_date_range('YTD')
        
        # Start should be January 1st of current year
        assert start.month == 1
        assert start.day == 1
        assert start.year == datetime.now().year


class TestParseHoldingsCSV:
    """Test suite for CSV parsing"""
    
    def test_parse_valid_csv(self, tmp_path):
        """Test parsing valid CSV"""
        csv_content = """ticker,quantity,purchase_price,purchase_date
AAPL,50,150.00,2024-01-15
GOOGL,20,140.00,2024-02-01"""
        
        from io import StringIO
        df = parse_holdings_csv(StringIO(csv_content))
        
        assert len(df) == 2
        assert 'ticker' in df.columns
        assert df.iloc[0]['ticker'] == 'AAPL'
        assert df.iloc[0]['quantity'] == 50
    
    def test_parse_csv_missing_columns(self):
        """Test parsing CSV with missing required columns"""
        csv_content = """ticker,quantity
AAPL,50"""
        
        from io import StringIO
        with pytest.raises(ValueError):
            parse_holdings_csv(StringIO(csv_content))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
