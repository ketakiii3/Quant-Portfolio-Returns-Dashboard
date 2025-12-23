"""
Visualization Module
Creates all charts and visualizations for the portfolio dashboard
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple
from datetime import datetime


# Color palette for consistent styling
COLORS = {
    'primary': '#2563eb',      # Blue
    'secondary': '#7c3aed',    # Purple
    'success': '#10b981',      # Green
    'danger': '#ef4444',       # Red
    'warning': '#f59e0b',      # Amber
    'neutral': '#6b7280',      # Gray
    'background': '#0f172a',   # Dark blue-gray
    'surface': '#1e293b',      # Lighter dark
    'text': '#f1f5f9',         # Light text
    'muted': '#94a3b8',        # Muted text
}

# Plotly template for consistent styling
CHART_TEMPLATE = {
    'layout': {
        'font': {'family': 'Inter, system-ui, sans-serif', 'color': COLORS['text']},
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'margin': {'l': 40, 'r': 40, 't': 40, 'b': 40},
        'xaxis': {
            'gridcolor': 'rgba(148, 163, 184, 0.1)',
            'zerolinecolor': 'rgba(148, 163, 184, 0.2)',
        },
        'yaxis': {
            'gridcolor': 'rgba(148, 163, 184, 0.1)',
            'zerolinecolor': 'rgba(148, 163, 184, 0.2)',
        },
        'legend': {'bgcolor': 'rgba(0,0,0,0)'},
        'hoverlabel': {'bgcolor': COLORS['surface']},
    }
}


def apply_chart_styling(fig: go.Figure) -> go.Figure:
    """Apply consistent styling to a Plotly figure"""
    fig.update_layout(
        font=dict(family='Inter, system-ui, sans-serif', color=COLORS['text']),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(
            gridcolor='rgba(148, 163, 184, 0.1)',
            zerolinecolor='rgba(148, 163, 184, 0.2)',
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            gridcolor='rgba(148, 163, 184, 0.1)',
            zerolinecolor='rgba(148, 163, 184, 0.2)',
            tickfont=dict(size=11),
        ),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
        hoverlabel=dict(bgcolor=COLORS['surface'], font_size=12),
        hovermode='x unified',
    )
    return fig


def plot_cumulative_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series = None,
    benchmark_name: str = 'S&P 500',
    title: str = 'Cumulative Returns'
) -> go.Figure:
    """
    Create cumulative returns chart comparing portfolio to benchmark
    """
    fig = go.Figure()
    
    # Portfolio cumulative returns
    portfolio_cum = (1 + portfolio_returns).cumprod() * 100 - 100
    
    fig.add_trace(go.Scatter(
        x=portfolio_cum.index,
        y=portfolio_cum.values,
        name='Portfolio',
        line=dict(color=COLORS['primary'], width=2.5),
        fill='tozeroy',
        fillcolor='rgba(37, 99, 235, 0.1)',
        hovertemplate='%{y:.2f}%<extra>Portfolio</extra>'
    ))
    
    # Benchmark cumulative returns
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        # Align benchmark to portfolio dates
        benchmark_aligned = benchmark_returns.reindex(portfolio_returns.index).ffill().dropna()
        if len(benchmark_aligned) > 0:
            benchmark_cum = (1 + benchmark_aligned).cumprod() * 100 - 100
            
            fig.add_trace(go.Scatter(
                x=benchmark_cum.index,
                y=benchmark_cum.values,
                name=benchmark_name,
                line=dict(color=COLORS['neutral'], width=1.5, dash='dash'),
                hovertemplate='%{y:.2f}%<extra>' + benchmark_name + '</extra>'
            ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash='solid', line_color='rgba(148, 163, 184, 0.3)', line_width=1)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        yaxis_title='Return (%)',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
    )
    
    return apply_chart_styling(fig)


def plot_drawdown(returns: pd.Series, title: str = 'Drawdown') -> go.Figure:
    """
    Create drawdown chart
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative - running_max) / running_max * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.2)',
        line=dict(color=COLORS['danger'], width=1.5),
        name='Drawdown',
        hovertemplate='%{y:.2f}%<extra></extra>'
    ))
    
    # Add minimum drawdown annotation
    min_dd = drawdown.min()
    min_dd_date = drawdown.idxmin()
    
    fig.add_annotation(
        x=min_dd_date,
        y=min_dd,
        text=f'Max DD: {min_dd:.1f}%',
        showarrow=True,
        arrowhead=2,
        arrowcolor=COLORS['danger'],
        font=dict(color=COLORS['text'], size=11),
        bgcolor=COLORS['surface'],
        borderpad=4,
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        yaxis_title='Drawdown (%)',
        showlegend=False,
    )
    
    return apply_chart_styling(fig)


def plot_asset_allocation(
    holdings: pd.DataFrame,
    value_column: str = 'current_value',
    title: str = 'Asset Allocation'
) -> go.Figure:
    """
    Cleaner Donut Chart: Groups small assets into 'Others' and moves labels outside.
    """
    # 1. Calculate weights and group small holdings (below 2%)
    total_val = holdings[value_column].sum()
    holdings['weight'] = (holdings[value_column] / total_val) * 100
    
    # Filter for main vs small holdings
    main = holdings[holdings['weight'] >= 2.0].copy()
    others = holdings[holdings['weight'] < 2.0].copy()
    
    if not others.empty:
        others_entry = pd.DataFrame([{
            'ticker': 'Others',
            value_column: others[value_column].sum(),
            'weight': others['weight'].sum()
        }])
        plot_df = pd.concat([main, others_entry])
    else:
        plot_df = main

    # 2. Create the Ring Chart
    fig = go.Figure(data=[go.Pie(
        labels=plot_df['ticker'],
        values=plot_df[value_column],
        hole=0.6,                       # Creates the donut 'ring'
        textinfo='label+percent',       # Shows Ticker and %
        textposition='outside',         # Moves labels outside to prevent overlap
        marker=dict(
            colors=px.colors.qualitative.Prism, 
            line=dict(color=COLORS['background'], width=2)
        ),
        hovertemplate='<b>%{label}</b><br>Value: $%{value:,.2f}<extra></extra>'
    )])
    
    # 3. Add central annotation
    fig.update_layout(
        showlegend=False,               # Removes redundant legend
        annotations=[dict(
            text=f'Total Value<br><b>${total_val:,.0f}</b>',
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=COLORS['text'])
        )]
    )
    
    return apply_chart_styling(fig)


def plot_holdings_performance(
    holdings: pd.DataFrame,
    title: str = 'Holdings Performance'
) -> go.Figure:
    """
    Create horizontal bar chart of holdings performance
    """
    holdings_sorted = holdings.sort_values('gain_loss_pct', ascending=True)
    
    colors = [COLORS['success'] if x >= 0 else COLORS['danger'] for x in holdings_sorted['gain_loss_pct']]
    
    fig = go.Figure(data=[go.Bar(
        y=holdings_sorted['ticker'],
        x=holdings_sorted['gain_loss_pct'],
        orientation='h',
        marker=dict(color=colors),
        text=[f'{x:+.1f}%' for x in holdings_sorted['gain_loss_pct']],
        textposition='outside',
        textfont=dict(size=11),
        hovertemplate='<b>%{y}</b><br>Return: %{x:.2f}%<extra></extra>'
    )])
    
    fig.add_vline(x=0, line_dash='solid', line_color='rgba(148, 163, 184, 0.5)', line_width=1)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title='Return (%)',
        yaxis_title='',
        showlegend=False,
        height=max(300, len(holdings) * 40),
    )
    
    return apply_chart_styling(fig)


def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    title: str = 'Correlation Matrix'
) -> go.Figure:
    """
    Create correlation heatmap
    """
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(correlation_matrix.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=10, color=COLORS['text']),
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>',
        colorbar=dict(
            title='Corr',
            titlefont=dict(size=11),
            tickfont=dict(size=10),
        )
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(tickangle=45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
        height=400,
    )
    
    return apply_chart_styling(fig)


def plot_rolling_sharpe(
    rolling_sharpe: pd.Series,
    window: int = 60,
    title: str = None
) -> go.Figure:
    """
    Create rolling Sharpe ratio chart
    """
    if title is None:
        title = f'Rolling {window}-Day Sharpe Ratio'
    
    fig = go.Figure()
    
    # Color based on value
    colors = np.where(rolling_sharpe.values >= 0, COLORS['success'], COLORS['danger'])
    
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe.values,
        fill='tozeroy',
        line=dict(color=COLORS['primary'], width=1.5),
        fillcolor='rgba(37, 99, 235, 0.1)',
        name='Sharpe Ratio',
        hovertemplate='%{y:.2f}<extra></extra>'
    ))
    
    # Add reference lines
    fig.add_hline(y=0, line_dash='solid', line_color='rgba(148, 163, 184, 0.5)', line_width=1)
    fig.add_hline(y=1, line_dash='dot', line_color=COLORS['success'], line_width=1, 
                  annotation_text='Good', annotation_position='right')
    fig.add_hline(y=2, line_dash='dot', line_color=COLORS['warning'], line_width=1,
                  annotation_text='Excellent', annotation_position='right')
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        yaxis_title='Sharpe Ratio',
        showlegend=False,
    )
    
    return apply_chart_styling(fig)


def plot_var_distribution(
    returns: pd.Series,
    var_95: float,
    cvar_95: float,
    title: str = 'Return Distribution & VaR'
) -> go.Figure:
    """
    Create histogram of returns with VaR/CVaR lines
    """
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=returns.values * 100,  # Convert to percentage
        nbinsx=50,
        name='Returns',
        marker=dict(color=COLORS['primary'], line=dict(color=COLORS['background'], width=0.5)),
        opacity=0.7,
        hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
    ))
    
    # VaR line
    fig.add_vline(
        x=var_95 * 100,
        line_dash='dash',
        line_color=COLORS['warning'],
        line_width=2,
        annotation_text=f'VaR 95%: {var_95*100:.2f}%',
        annotation_position='top',
        annotation_font=dict(color=COLORS['warning'])
    )
    
    # CVaR line
    fig.add_vline(
        x=cvar_95 * 100,
        line_dash='dash',
        line_color=COLORS['danger'],
        line_width=2,
        annotation_text=f'CVaR 95%: {cvar_95*100:.2f}%',
        annotation_position='bottom',
        annotation_font=dict(color=COLORS['danger'])
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title='Daily Return (%)',
        yaxis_title='Frequency',
        showlegend=False,
        bargap=0.05,
    )
    
    return apply_chart_styling(fig)


def plot_efficient_frontier(
    frontier: pd.DataFrame,
    current_portfolio: Tuple[float, float] = None,
    optimal_portfolio: Tuple[float, float] = None,
    title: str = 'Efficient Frontier'
) -> go.Figure:
    """
    Plot efficient frontier with optional current and optimal portfolio points
    """
    fig = go.Figure()
    
    # Efficient frontier curve
    fig.add_trace(go.Scatter(
        x=frontier['volatility'] * 100,
        y=frontier['return'] * 100,
        mode='lines',
        name='Efficient Frontier',
        line=dict(color=COLORS['primary'], width=2),
        hovertemplate='Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
    ))
    
    # Color by Sharpe ratio
    fig.add_trace(go.Scatter(
        x=frontier['volatility'] * 100,
        y=frontier['return'] * 100,
        mode='markers',
        marker=dict(
            color=frontier['sharpe'],
            colorscale='Viridis',
            size=8,
            colorbar=dict(title='Sharpe', titlefont=dict(size=11), tickfont=dict(size=10)),
            line=dict(width=0)
        ),
        name='Sharpe',
        hovertemplate='Volatility: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: %{marker.color:.2f}<extra></extra>'
    ))
    
    # Current portfolio point
    if current_portfolio:
        fig.add_trace(go.Scatter(
            x=[current_portfolio[0] * 100],
            y=[current_portfolio[1] * 100],
            mode='markers',
            marker=dict(color=COLORS['warning'], size=15, symbol='star'),
            name='Current Portfolio',
            hovertemplate='Current<br>Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>'
        ))
    
    # Optimal portfolio point
    if optimal_portfolio:
        fig.add_trace(go.Scatter(
            x=[optimal_portfolio[0] * 100],
            y=[optimal_portfolio[1] * 100],
            mode='markers',
            marker=dict(color=COLORS['success'], size=15, symbol='diamond'),
            name='Optimal Portfolio',
            hovertemplate='Optimal<br>Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title='Volatility (%)',
        yaxis_title='Expected Return (%)',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
    )
    
    return apply_chart_styling(fig)


def plot_monte_carlo(
    simulations: np.ndarray,
    percentiles: pd.DataFrame,
    initial_value: float,
    title: str = 'Monte Carlo Simulation'
) -> go.Figure:
    """
    Plot Monte Carlo simulation paths
    """
    fig = go.Figure()
    
    n_days = simulations.shape[0]
    days = list(range(n_days))
    
    # Plot sample paths (subset)
    n_sample = min(100, simulations.shape[1])
    for i in range(n_sample):
        fig.add_trace(go.Scatter(
            x=days,
            y=simulations[:, i],
            mode='lines',
            line=dict(color='rgba(99, 102, 241, 0.1)', width=0.5),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Percentile bands
    colors = ['rgba(239, 68, 68, 0.3)', 'rgba(251, 191, 36, 0.3)', 
              'rgba(34, 197, 94, 0.3)', 'rgba(251, 191, 36, 0.3)', 'rgba(239, 68, 68, 0.3)']
    
    percentile_cols = [col for col in percentiles.columns if col.startswith('p')]
    
    for i, col in enumerate(percentile_cols):
        fig.add_trace(go.Scatter(
            x=days,
            y=percentiles[col],
            mode='lines',
            line=dict(color=colors[i] if i < len(colors) else COLORS['neutral'], width=2),
            name=col.upper(),
            hovertemplate=f'{col.upper()}: $%{{y:,.0f}}<extra></extra>'
        ))
    
    # Initial value line
    fig.add_hline(
        y=initial_value,
        line_dash='dot',
        line_color=COLORS['muted'],
        annotation_text='Initial',
        annotation_position='right'
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title='Trading Days',
        yaxis_title='Portfolio Value ($)',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
    )
    
    return apply_chart_styling(fig)


def plot_sector_allocation(
    holdings: pd.DataFrame,
    sector_column: str = 'sector',
    value_column: str = 'current_value',
    title: str = 'Sector Allocation'
) -> go.Figure:
    """
    Create treemap of sector allocation
    """
    if sector_column not in holdings.columns:
        # Create a simple bar chart instead
        fig = go.Figure(data=[go.Bar(
            x=holdings['ticker'],
            y=holdings[value_column],
            marker=dict(color=COLORS['primary']),
        )])
        fig.update_layout(title=dict(text=title, font=dict(size=16)))
        return apply_chart_styling(fig)
    
    # Group by sector
    sector_values = holdings.groupby(sector_column)[value_column].sum().reset_index()
    
    fig = go.Figure(go.Treemap(
        labels=sector_values[sector_column],
        values=sector_values[value_column],
        parents=[''] * len(sector_values),
        marker=dict(
            colors=px.colors.qualitative.Set2[:len(sector_values)],
            line=dict(color=COLORS['background'], width=2)
        ),
        textinfo='label+percent entry',
        textfont=dict(size=14, color=COLORS['text']),
        hovertemplate='<b>%{label}</b><br>Value: $%{value:,.2f}<br>%{percentEntry}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
    )
    
    return apply_chart_styling(fig)


def plot_performance_metrics(metrics: dict, title: str = 'Risk Metrics') -> go.Figure:
    """
    Create a gauge chart for key metrics
    """
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
               [{'type': 'indicator'}, {'type': 'indicator'}]],
        vertical_spacing=0.3,
        horizontal_spacing=0.1
    )
    
    # Sharpe Ratio
    fig.add_trace(go.Indicator(
        mode='gauge+number',
        value=metrics.get('sharpe_ratio', 0),
        title={'text': 'Sharpe Ratio', 'font': {'size': 14}},
        gauge={
            'axis': {'range': [-1, 3], 'tickwidth': 1},
            'bar': {'color': COLORS['primary']},
            'steps': [
                {'range': [-1, 0], 'color': 'rgba(239, 68, 68, 0.3)'},
                {'range': [0, 1], 'color': 'rgba(251, 191, 36, 0.3)'},
                {'range': [1, 2], 'color': 'rgba(34, 197, 94, 0.3)'},
                {'range': [2, 3], 'color': 'rgba(16, 185, 129, 0.5)'},
            ],
            'threshold': {
                'line': {'color': COLORS['warning'], 'width': 2},
                'thickness': 0.75,
                'value': metrics.get('sharpe_ratio', 0)
            }
        },
        number={'font': {'size': 24}}
    ), row=1, col=1)
    
    # Sortino Ratio
    fig.add_trace(go.Indicator(
        mode='gauge+number',
        value=min(metrics.get('sortino_ratio', 0), 5),  # Cap display at 5
        title={'text': 'Sortino Ratio', 'font': {'size': 14}},
        gauge={
            'axis': {'range': [-1, 5], 'tickwidth': 1},
            'bar': {'color': COLORS['secondary']},
            'steps': [
                {'range': [-1, 0], 'color': 'rgba(239, 68, 68, 0.3)'},
                {'range': [0, 1], 'color': 'rgba(251, 191, 36, 0.3)'},
                {'range': [1, 2], 'color': 'rgba(34, 197, 94, 0.3)'},
                {'range': [2, 5], 'color': 'rgba(16, 185, 129, 0.5)'},
            ],
        },
        number={'font': {'size': 24}}
    ), row=1, col=2)
    
    # Max Drawdown
    fig.add_trace(go.Indicator(
        mode='gauge+number',
        value=abs(metrics.get('max_drawdown', 0)) * 100,
        title={'text': 'Max Drawdown %', 'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, 50], 'tickwidth': 1},
            'bar': {'color': COLORS['danger']},
            'steps': [
                {'range': [0, 10], 'color': 'rgba(34, 197, 94, 0.3)'},
                {'range': [10, 20], 'color': 'rgba(251, 191, 36, 0.3)'},
                {'range': [20, 50], 'color': 'rgba(239, 68, 68, 0.3)'},
            ],
        },
        number={'font': {'size': 24}, 'suffix': '%'}
    ), row=2, col=1)
    
    # Calmar Ratio
    fig.add_trace(go.Indicator(
        mode='gauge+number',
        value=min(metrics.get('calmar_ratio', 0), 5),
        title={'text': 'Calmar Ratio', 'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, 5], 'tickwidth': 1},
            'bar': {'color': COLORS['success']},
            'steps': [
                {'range': [0, 1], 'color': 'rgba(239, 68, 68, 0.3)'},
                {'range': [1, 2], 'color': 'rgba(251, 191, 36, 0.3)'},
                {'range': [2, 5], 'color': 'rgba(34, 197, 94, 0.3)'},
            ],
        },
        number={'font': {'size': 24}}
    ), row=2, col=2)
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        height=400,
    )
    
    return apply_chart_styling(fig)


def create_metric_card(value: str, label: str, delta: str = None, delta_color: str = None) -> str:
    """
    Create HTML for a metric card (for Streamlit)
    """
    delta_html = ''
    if delta:
        color = delta_color or ('green' if '+' in delta else 'red')
        delta_html = f'<span style="color: {color}; font-size: 14px;">{delta}</span>'
    
    return f'''
    <div style="padding: 16px; background: {COLORS['surface']}; border-radius: 8px; text-align: center;">
        <div style="font-size: 28px; font-weight: 600; color: {COLORS['text']};">{value}</div>
        <div style="font-size: 14px; color: {COLORS['muted']}; margin-top: 4px;">{label}</div>
        {delta_html}
    </div>
    '''


if __name__ == "__main__":
    # Test visualizations
    import numpy as np
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=252, freq='B')
    returns = pd.Series(np.random.normal(0.0005, 0.02, 252), index=dates)
    
    # Test cumulative returns chart
    fig = plot_cumulative_returns(returns)
    fig.show()
    
    # Test drawdown chart
    fig = plot_drawdown(returns)
    fig.show()
