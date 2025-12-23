"""
Data Fetcher Module
Handles fetching market data from various sources (yfinance, etc.)
Includes caching and data validation
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Fetches and caches market data from various sources
    """
    
    def __init__(self, cache_enabled: bool = True):
        self.cache_enabled = cache_enabled
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._info_cache: Dict[str, dict] = {}
    
    def fetch_price_data(
        self,
        tickers: Union[str, List[str]],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        period: str = "1y",
        max_retries: int = 3
    ) -> pd.DataFrame:
        """
        Fetch historical price data for one or more tickers
        
        Args:
            tickers: Single ticker or list of tickers
            start_date: Start date for historical data
            end_date: End date for historical data
            period: Period string (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            max_retries: Number of retry attempts for failed fetches
        
        Returns:
            DataFrame with OHLCV data
        """
        import time
        
        if isinstance(tickers, str):
            tickers = [tickers]
        
        # Set default dates
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)
        
        all_data = {}
        
        # Try batch download first (more reliable)
        try:
            logger.info(f"Batch downloading data for {len(tickers)} tickers")
            
            batch_data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            threads=False,
            group_by="ticker"
            )

            
            if not batch_data.empty:
                # Process batch data
                for ticker in tickers:
                    try:
                        if len(tickers) == 1:
                            df = batch_data.copy()
                        else:
                            # Multi-ticker download has multi-level columns
                            df = batch_data.xs(ticker, level=1, axis=1) if ticker in batch_data.columns.get_level_values(1) else pd.DataFrame()
                        
                        if df.empty:
                            continue
                            
                        df = df.reset_index()
                        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
                        
                        # Handle timezone-aware dates
                        if 'date' in df.columns and df['date'].dt.tz is not None:
                            df['date'] = df['date'].dt.tz_localize(None)
                        
                        df['ticker'] = ticker
                        
                        # Filter out rows with NaN close prices
                        if 'close' in df.columns:
                            df = df.dropna(subset=['close'])
                        
                        if not df.empty:
                            all_data[ticker] = df
                            logger.info(f"Got {len(df)} rows for {ticker}")
                    except Exception as e:
                        logger.warning(f"Error processing batch data for {ticker}: {e}")
                        continue
                        
        except Exception as e:
            logger.warning(f"Batch download failed: {e}. Falling back to individual downloads.")
        
        # Fallback: fetch individually for any missing tickers
        missing_tickers = [t for t in tickers if t not in all_data]
        
        for ticker in missing_tickers:
            cache_key = f"{ticker}_{start_date.date()}_{end_date.date()}"
            
            # Check cache first
            if self.cache_enabled and cache_key in self._price_cache:
                all_data[ticker] = self._price_cache[cache_key]
                logger.info(f"Using cached data for {ticker}")
                continue
            
            # Retry logic for unreliable API
            for attempt in range(max_retries):
                try:
                    logger.info(f"Fetching data for {ticker} (attempt {attempt + 1})")
                    stock = yf.Ticker(ticker)
                    
                    if start_date and end_date:
                        df = stock.history(start=start_date, end=end_date, auto_adjust=True)
                    else:
                        df = stock.history(period=period, auto_adjust=True)
                    
                    if df.empty:
                        if attempt < max_retries - 1:
                            time.sleep(1)  # Wait before retry
                            continue
                        logger.warning(f"No data returned for {ticker} after {max_retries} attempts")
                        break
                    
                    # Clean up the data
                    df = df.reset_index()
                    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
                    
                    # Handle timezone-aware dates
                    if 'date' in df.columns:
                        if hasattr(df['date'].dt, 'tz') and df['date'].dt.tz is not None:
                            df['date'] = df['date'].dt.tz_localize(None)
                    
                    df['ticker'] = ticker
                    
                    # Cache the data
                    if self.cache_enabled:
                        self._price_cache[cache_key] = df
                    
                    all_data[ticker] = df
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    logger.error(f"Error fetching data for {ticker} (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Wait before retry
                    continue
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        combined = pd.concat(all_data.values(), ignore_index=True)
        return combined
    
    def fetch_current_price(self, ticker: str) -> Optional[float]:
        """
        Fetch the current/latest price for a ticker
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Try different price fields
            price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
            return price
            
        except Exception as e:
            logger.error(f"Error fetching current price for {ticker}: {e}")
            return None
    
    def fetch_ticker_info(self, ticker: str) -> dict:
        """
        Fetch detailed information about a ticker
        """
        if self.cache_enabled and ticker in self._info_cache:
            return self._info_cache[ticker]
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract relevant fields
            result = {
                'ticker': ticker,
                'name': info.get('shortName', info.get('longName', ticker)),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'currency': info.get('currency', 'USD'),
            }
            
            if self.cache_enabled:
                self._info_cache[ticker] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching info for {ticker}: {e}")
            return {'ticker': ticker, 'name': ticker}
    
    def fetch_multiple_prices(self, tickers: List[str]) -> Dict[str, float]:
        """
        Fetch current prices for multiple tickers efficiently
        """
        prices = {}
        
        try:
            # Use yfinance download for batch fetching
            data = yf.download(tickers, period="1d", progress=False)
            
            if 'Adj Close' in data.columns:
                for ticker in tickers:
                    if len(tickers) == 1:
                        prices[ticker] = data['Adj Close'].iloc[-1]
                    else:
                        if ticker in data['Adj Close'].columns:
                            prices[ticker] = data['Adj Close'][ticker].iloc[-1]
            
        except Exception as e:
            logger.error(f"Batch price fetch failed: {e}")
            # Fallback to individual fetching
            for ticker in tickers:
                price = self.fetch_current_price(ticker)
                if price:
                    prices[ticker] = price
        
        return prices
    
    def validate_ticker(self, ticker: str) -> bool:
        """
        Validate if a ticker symbol exists
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return 'regularMarketPrice' in info or 'currentPrice' in info
        except:
            return False
    
    def get_trading_days(
        self,
        start_date: datetime,
        end_date: datetime,
        ticker: str = "SPY"
    ) -> pd.DatetimeIndex:
        """
        Get list of trading days between two dates
        """
        df = self.fetch_price_data(ticker, start_date, end_date)
        if df.empty:
            return pd.DatetimeIndex([])
        return pd.DatetimeIndex(df['date'].unique())
    
    def clear_cache(self):
        """Clear all cached data"""
        self._price_cache.clear()
        self._info_cache.clear()
        logger.info("Cache cleared")


class BenchmarkData:
    """
    Handles benchmark index data
    """
    
    COMMON_BENCHMARKS = {
        'SPY': 'S&P 500',
        'QQQ': 'Nasdaq 100',
        'DIA': 'Dow Jones',
        'IWM': 'Russell 2000',
        'VTI': 'Total Stock Market',
        'AGG': 'US Aggregate Bond',
        'GLD': 'Gold',
        'BTC-USD': 'Bitcoin',
    }
    
    def __init__(self, fetcher: DataFetcher = None):
        self.fetcher = fetcher or DataFetcher()
    
    def get_benchmark_returns(
        self,
        benchmark: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.Series:
        """
        Get daily returns for a benchmark
        """
        df = self.fetcher.fetch_price_data(benchmark, start_date, end_date)
        
        if df.empty:
            return pd.Series()
        
        df = df.sort_values('date')
        df['return'] = df['close'].pct_change()
        
        return df.set_index('date')['return']
    
    def get_multiple_benchmark_returns(
        self,
        benchmarks: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Get returns for multiple benchmarks aligned to same dates
        """
        all_returns = {}
        
        for benchmark in benchmarks:
            returns = self.get_benchmark_returns(benchmark, start_date, end_date)
            if not returns.empty:
                name = self.COMMON_BENCHMARKS.get(benchmark, benchmark)
                all_returns[name] = returns
        
        if not all_returns:
            return pd.DataFrame()
        
        return pd.DataFrame(all_returns)


# Risk-free rate data
def get_risk_free_rate(period: str = "current") -> float:
    """
    Get the current risk-free rate (10-year Treasury yield)
    Returns annualized rate as decimal (e.g., 0.04 for 4%)
    """
    try:
        treasury = yf.Ticker("^TNX")  # 10-year Treasury yield
        info = treasury.info
        
        # The yield is already in percentage form
        rate = info.get('regularMarketPrice', 4.0) / 100
        return rate
        
    except Exception as e:
        logger.warning(f"Could not fetch risk-free rate: {e}. Using default 4%")
        return 0.04


if __name__ == "__main__":
    # Test the data fetcher
    fetcher = DataFetcher()
    
    # Test single ticker
    print("Testing single ticker fetch...")
    df = fetcher.fetch_price_data("AAPL", period="1mo")
    print(f"Fetched {len(df)} rows for AAPL")
    print(df.head())
    
    # Test multiple tickers
    print("\nTesting multiple ticker fetch...")
    df = fetcher.fetch_price_data(["AAPL", "GOOGL", "MSFT"], period="1mo")
    print(f"Fetched {len(df)} total rows")
    
    # Test ticker info
    print("\nTesting ticker info...")
    info = fetcher.fetch_ticker_info("AAPL")
    print(info)
    
    # Test risk-free rate
    print(f"\nCurrent risk-free rate: {get_risk_free_rate():.2%}")
