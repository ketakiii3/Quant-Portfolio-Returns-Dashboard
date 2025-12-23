"""
Utility Functions
Helper functions for the portfolio dashboard
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import io
import logging

logger = logging.getLogger(__name__)


def format_currency(value: float, symbol: str = '$', decimals: int = 2) -> str:
    """Format a number as currency"""
    if pd.isna(value) or value is None:
        return f'{symbol}--'
    
    if abs(value) >= 1_000_000_000:
        return f'{symbol}{value/1_000_000_000:.2f}B'
    elif abs(value) >= 1_000_000:
        return f'{symbol}{value/1_000_000:.2f}M'
    elif abs(value) >= 1_000:
        return f'{symbol}{value/1_000:.2f}K'
    else:
        return f'{symbol}{value:,.{decimals}f}'


def format_percentage(value: float, decimals: int = 2, with_sign: bool = False) -> str:
    """Format a number as percentage"""
    if pd.isna(value) or value is None:
        return '--'
    
    if with_sign and value > 0:
        return f'+{value:.{decimals}f}%'
    return f'{value:.{decimals}f}%'


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with commas"""
    if pd.isna(value) or value is None:
        return '--'
    return f'{value:,.{decimals}f}'


def parse_holdings_csv(file_content: Union[str, bytes, io.BytesIO]) -> pd.DataFrame:
    """
    Parse holdings from CSV content
    
    Expected columns: ticker, quantity, purchase_price, purchase_date
    Optional columns: asset_class, sector
    """
    try:
        if isinstance(file_content, bytes):
            file_content = io.BytesIO(file_content)
        
        df = pd.read_csv(file_content)
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        
        # Required columns
        required = ['ticker', 'quantity', 'purchase_price', 'purchase_date']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Clean ticker symbols
        df['ticker'] = df['ticker'].str.upper().str.strip()
        
        # Parse dates
        df['purchase_date'] = pd.to_datetime(df['purchase_date']).dt.date
        
        # Ensure numeric types
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['purchase_price'] = pd.to_numeric(df['purchase_price'], errors='coerce')
        
        # Fill optional columns
        if 'asset_class' not in df.columns:
            df['asset_class'] = 'Equity'
        if 'sector' not in df.columns:
            df['sector'] = 'Unknown'
        
        # Remove invalid rows
        df = df.dropna(subset=required)
        
        return df
        
    except Exception as e:
        logger.error(f"Error parsing holdings CSV: {e}")
        raise


def parse_transactions_csv(file_content: Union[str, bytes, io.BytesIO]) -> pd.DataFrame:
    """
    Parse transactions from CSV content
    
    Expected columns: ticker, transaction_date, transaction_type, quantity, price
    Optional columns: fees, notes
    """
    try:
        if isinstance(file_content, bytes):
            file_content = io.BytesIO(file_content)
        
        df = pd.read_csv(file_content)
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        
        # Required columns
        required = ['ticker', 'transaction_date', 'transaction_type', 'quantity', 'price']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Clean data
        df['ticker'] = df['ticker'].str.upper().str.strip()
        df['transaction_date'] = pd.to_datetime(df['transaction_date']).dt.date
        df['transaction_type'] = df['transaction_type'].str.upper().str.strip()
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
        # Optional columns
        if 'fees' not in df.columns:
            df['fees'] = 0.0
        else:
            df['fees'] = pd.to_numeric(df['fees'], errors='coerce').fillna(0)
        
        if 'notes' not in df.columns:
            df['notes'] = ''
        
        return df.dropna(subset=required)
        
    except Exception as e:
        logger.error(f"Error parsing transactions CSV: {e}")
        raise


def generate_sample_portfolio() -> Dict[str, pd.DataFrame]:
    """
    Generate sample portfolio data for demonstration
    """
    # Sample holdings
    holdings_data = [
        {'ticker': 'AAPL', 'quantity': 50, 'purchase_price': 150.00, 
         'purchase_date': datetime(2023, 1, 15).date(), 'asset_class': 'Equity', 'sector': 'Technology'},
        {'ticker': 'GOOGL', 'quantity': 20, 'purchase_price': 95.00, 
         'purchase_date': datetime(2023, 2, 1).date(), 'asset_class': 'Equity', 'sector': 'Technology'},
        {'ticker': 'MSFT', 'quantity': 30, 'purchase_price': 280.00, 
         'purchase_date': datetime(2023, 1, 20).date(), 'asset_class': 'Equity', 'sector': 'Technology'},
        {'ticker': 'JPM', 'quantity': 25, 'purchase_price': 140.00, 
         'purchase_date': datetime(2023, 3, 10).date(), 'asset_class': 'Equity', 'sector': 'Financials'},
        {'ticker': 'JNJ', 'quantity': 15, 'purchase_price': 165.00, 
         'purchase_date': datetime(2023, 2, 15).date(), 'asset_class': 'Equity', 'sector': 'Healthcare'},
        {'ticker': 'VTI', 'quantity': 40, 'purchase_price': 200.00, 
         'purchase_date': datetime(2023, 1, 5).date(), 'asset_class': 'ETF', 'sector': 'Diversified'},
        {'ticker': 'BND', 'quantity': 50, 'purchase_price': 75.00, 
         'purchase_date': datetime(2023, 1, 10).date(), 'asset_class': 'Bond ETF', 'sector': 'Fixed Income'},
    ]
    
    holdings = pd.DataFrame(holdings_data)
    
    # Sample transactions
    transactions_data = [
        {'ticker': 'AAPL', 'transaction_date': datetime(2023, 1, 15).date(), 
         'transaction_type': 'BUY', 'quantity': 50, 'price': 150.00, 'fees': 0},
        {'ticker': 'GOOGL', 'transaction_date': datetime(2023, 2, 1).date(), 
         'transaction_type': 'BUY', 'quantity': 20, 'price': 95.00, 'fees': 0},
        {'ticker': 'MSFT', 'transaction_date': datetime(2023, 1, 20).date(), 
         'transaction_type': 'BUY', 'quantity': 30, 'price': 280.00, 'fees': 0},
        {'ticker': 'JPM', 'transaction_date': datetime(2023, 3, 10).date(), 
         'transaction_type': 'BUY', 'quantity': 25, 'price': 140.00, 'fees': 0},
        {'ticker': 'JNJ', 'transaction_date': datetime(2023, 2, 15).date(), 
         'transaction_type': 'BUY', 'quantity': 15, 'price': 165.00, 'fees': 0},
        {'ticker': 'VTI', 'transaction_date': datetime(2023, 1, 5).date(), 
         'transaction_type': 'BUY', 'quantity': 40, 'price': 200.00, 'fees': 0},
        {'ticker': 'BND', 'transaction_date': datetime(2023, 1, 10).date(), 
         'transaction_type': 'BUY', 'quantity': 50, 'price': 75.00, 'fees': 0},
    ]
    
    transactions = pd.DataFrame(transactions_data)
    
    return {
        'holdings': holdings,
        'transactions': transactions
    }


def calculate_date_range(period: str) -> tuple:
    """
    Calculate start and end dates from period string
    """
    end_date = datetime.now()
    
    period_map = {
        '1W': timedelta(weeks=1),
        '1M': timedelta(days=30),
        '3M': timedelta(days=90),
        '6M': timedelta(days=180),
        'YTD': None,  # Special case
        '1Y': timedelta(days=365),
        '2Y': timedelta(days=730),
        '3Y': timedelta(days=1095),
        '5Y': timedelta(days=1825),
        'MAX': timedelta(days=3650),
    }
    
    if period == 'YTD':
        start_date = datetime(end_date.year, 1, 1)
    elif period in period_map:
        start_date = end_date - period_map[period]
    else:
        start_date = end_date - timedelta(days=365)
    
    return start_date, end_date


def get_color_for_value(value: float, positive_good: bool = True) -> str:
    """
    Get color code based on value sign
    """
    if positive_good:
        return '#10b981' if value >= 0 else '#ef4444'  # Green or Red
    else:
        return '#ef4444' if value >= 0 else '#10b981'  # Red or Green


def validate_ticker_format(ticker: str) -> bool:
    """
    Validate ticker symbol format
    """
    if not ticker:
        return False
    
    # Remove common suffixes
    clean_ticker = ticker.upper().strip()
    
    # Basic validation: alphanumeric, 1-10 characters, may contain dots and hyphens
    import re
    pattern = r'^[A-Z0-9][A-Z0-9\.\-]{0,9}$'
    return bool(re.match(pattern, clean_ticker))


def aggregate_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate multiple lots of the same ticker into single positions
    """
    if holdings.empty:
        return holdings
    
    # Calculate weighted average cost
    def weighted_avg(group):
        total_quantity = group['quantity'].sum()
        if total_quantity == 0:
            return pd.Series({
                'quantity': 0,
                'purchase_price': 0,
                'purchase_date': group['purchase_date'].min(),
                'asset_class': group['asset_class'].iloc[0] if 'asset_class' in group else 'Equity',
                'sector': group['sector'].iloc[0] if 'sector' in group else 'Unknown',
            })
        
        weighted_price = (group['quantity'] * group['purchase_price']).sum() / total_quantity
        
        return pd.Series({
            'quantity': total_quantity,
            'purchase_price': weighted_price,
            'purchase_date': group['purchase_date'].min(),
            'asset_class': group['asset_class'].iloc[0] if 'asset_class' in group else 'Equity',
            'sector': group['sector'].iloc[0] if 'sector' in group else 'Unknown',
        })
    
    aggregated = holdings.groupby('ticker').apply(weighted_avg).reset_index()
    return aggregated


def export_to_csv(df: pd.DataFrame) -> bytes:
    """
    Export DataFrame to CSV bytes
    """
    return df.to_csv(index=False).encode('utf-8')


def export_holdings_template() -> bytes:
    """
    Generate a holdings template CSV
    """
    template = pd.DataFrame({
        'ticker': ['AAPL', 'GOOGL'],
        'quantity': [10, 5],
        'purchase_price': [150.00, 100.00],
        'purchase_date': ['2024-01-15', '2024-02-01'],
        'asset_class': ['Equity', 'Equity'],
        'sector': ['Technology', 'Technology'],
    })
    return export_to_csv(template)


def export_transactions_template() -> bytes:
    """
    Generate a transactions template CSV
    """
    template = pd.DataFrame({
        'ticker': ['AAPL', 'AAPL'],
        'transaction_date': ['2024-01-15', '2024-03-01'],
        'transaction_type': ['BUY', 'BUY'],
        'quantity': [10, 5],
        'price': [150.00, 170.00],
        'fees': [0, 0],
        'notes': ['Initial purchase', 'Added to position'],
    })
    return export_to_csv(template)


class PerformanceTracker:
    """
    Track and compare portfolio performance metrics
    """
    
    @staticmethod
    def daily_stats(returns: pd.Series) -> Dict:
        """Get daily statistics"""
        return {
            'best_day': returns.max(),
            'worst_day': returns.min(),
            'avg_daily': returns.mean(),
            'positive_days': (returns > 0).sum(),
            'negative_days': (returns < 0).sum(),
            'win_rate': (returns > 0).mean() * 100,
        }
    
    @staticmethod
    def monthly_returns(returns: pd.Series) -> pd.DataFrame:
        """Calculate monthly returns table"""
        # Resample to monthly
        monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create year-month pivot
        df = pd.DataFrame({
            'year': monthly.index.year,
            'month': monthly.index.month,
            'return': monthly.values
        })
        
        pivot = df.pivot(index='year', columns='month', values='return')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        return pivot
    
    @staticmethod
    def yearly_returns(returns: pd.Series) -> pd.Series:
        """Calculate yearly returns"""
        return returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)


if __name__ == "__main__":
    # Test utilities
    print("Testing sample portfolio generation...")
    sample = generate_sample_portfolio()
    print(f"Holdings:\n{sample['holdings']}")
    print(f"\nTransactions:\n{sample['transactions']}")
