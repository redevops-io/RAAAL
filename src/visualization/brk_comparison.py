"""BRK-B comparison analysis tab."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import Div, TabPanel
from bokeh.plotting import figure

from ..config import UNIVERSE
from ..data_loader import download_prices
from ..features import compute_returns

HISTORY_DIR = Path("data/history")


def load_history_data():
    """Load historical backtest data."""
    summary_path = HISTORY_DIR / "history_summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path, "r") as f:
        return json.load(f)


def create_brk_comparison_chart():
    """Create cumulative returns chart comparing RAAAL strategies vs BRK-B."""
    data = load_history_data()
    if not data:
        return Div(text="<h3>No data available</h3>")
    
    weights_dict = data.get("weights", {})
    weights_df = pd.DataFrame(weights_dict)
    
    if weights_df.empty:
        return Div(text="<h3>Insufficient data</h3>")
    
    # Load prices and returns
    tickers = [asset.ticker for asset in UNIVERSE]
    prices = download_prices(tickers + ["BRK-B"], start=datetime(2016, 1, 1), end=datetime.now())
    returns = compute_returns(prices)
    
    # Get regime timeline
    timeline_dict = data.get("timeline", {})
    timeline_df = pd.DataFrame(timeline_dict)
    
    # Calculate cumulative returns for RAAAL strategies
    strategies = {
        "RAAAL (Regime-Aware)": "weight",
        "RAAAL (Standard)": "standard_weight",
    }
    
    p = figure(
        width=1000,
        height=500,
        x_axis_type='datetime',
        title="Total Return: RAAAL vs Berkshire Hathaway (BRK-B)",
        toolbar_location="above",
    )
    
    colors = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71']
    
    # Calculate RAAAL strategies
    for (label, column), color in zip(strategies.items(), colors[:2]):
        if column not in weights_df.columns:
            continue
            
        weight_history = (
            weights_df.pivot(index="date", columns="ticker", values=column)
            .sort_index()
            .reindex(columns=tickers, fill_value=0.0)
        )
        
        if weight_history.empty:
            continue
            
        weight_history.index = pd.to_datetime(weight_history.index)
        weight_history = weight_history.reindex(returns.index).ffill()
        valid_mask = weight_history.notna().any(axis=1)
        weight_history = weight_history.loc[valid_mask]
        
        if weight_history.empty:
            continue
            
        aligned_returns = returns.loc[weight_history.index, tickers].fillna(0.0)
        daily_returns = (weight_history.fillna(0.0) * aligned_returns).sum(axis=1)
        cumulative_returns = (1.0 + daily_returns).cumprod()
        
        p.line(weight_history.index, cumulative_returns, 
               line_width=3, color=color, alpha=0.8, legend_label=label)
    
    # Calculate BRK-B + Cash (Regime-Aware)
    if "BRK-B" in returns.columns and "BIL" in returns.columns and not timeline_df.empty:
        regime_weights = {
            'risk_on': 0.8,
            'risk_off': 0.3,
            'inflation': 0.5,
        }
        
        timeline_df['date'] = pd.to_datetime(timeline_df['date'])
        timeline_df = timeline_df.set_index('date').sort_index()
        daily_regimes = timeline_df['regime'].reindex(returns.index).ffill()
        
        brk_weight = daily_regimes.map(regime_weights).fillna(0.5)
        bil_weight = 1.0 - brk_weight
        
        brk_regime_returns = (
            brk_weight * returns["BRK-B"].fillna(0.0) +
            bil_weight * returns["BIL"].fillna(0.0)
        )
        brk_regime_cumulative = (1.0 + brk_regime_returns).cumprod()
        
        p.line(brk_regime_cumulative.index, brk_regime_cumulative,
               line_width=3, color=colors[2], alpha=0.8, legend_label="BRK-B + Cash (Regime-Aware)")
    
    # Calculate BRK-B buy-and-hold
    if "BRK-B" in returns.columns:
        brk_returns = returns["BRK-B"].fillna(0.0)
        brk_cumulative = (1.0 + brk_returns).cumprod()
        
        p.line(brk_cumulative.index, brk_cumulative, 
               line_width=3, color=colors[3], alpha=0.8, legend_label="BRK-B (Buy & Hold)")
    
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.xaxis.axis_label = "Date"
    p.yaxis.axis_label = "Cumulative Return (1.0 = initial)"
    
    return p


def create_brk_performance_table():
    """Create performance comparison table with BRK-B."""
    data = load_history_data()
    if not data:
        return Div(text="<h3>No data available</h3>")
    
    weights_dict = data.get("weights", {})
    weights_df = pd.DataFrame(weights_dict)
    
    if weights_df.empty:
        return Div(text="<h3>Insufficient data</h3>")
    
    # Load prices and returns
    tickers = [asset.ticker for asset in UNIVERSE]
    prices = download_prices(tickers + ["BRK-B"], start=datetime(2016, 1, 1), end=datetime.now())
    returns = compute_returns(prices)
    
    # Get regime timeline for BRK-B + Cash tactical allocation
    timeline_dict = data.get("timeline", {})
    timeline_df = pd.DataFrame(timeline_dict)
    
    # Calculate metrics for each strategy
    strategies = {
        "RAAAL (Regime-Aware)": "weight",
        "RAAAL (Standard)": "standard_weight",
        "BRK-B + Cash (Regime-Aware)": "brk_regime",
        "BRK-B (Buy & Hold)": None,
    }
    
    results = []
    
    for label, column in strategies.items():
        if label == "BRK-B (Buy & Hold)":
            # BRK-B buy-and-hold
            if "BRK-B" not in returns.columns:
                continue
            daily_returns = returns["BRK-B"].fillna(0.0)
        elif label == "BRK-B + Cash (Regime-Aware)":
            # Tactical BRK-B + BIL allocation based on regime
            if "BRK-B" not in returns.columns or "BIL" not in returns.columns or timeline_df.empty:
                continue
            
            # Map regimes to BRK-B weights (rest goes to BIL cash)
            regime_weights = {
                'risk_on': 0.8,      # 80% BRK-B, 20% cash in risk-on
                'risk_off': 0.3,     # 30% BRK-B, 70% cash in risk-off
                'inflation': 0.5,    # 50% BRK-B, 50% cash in inflation
            }
            
            # Create daily regime-based weights
            timeline_df['date'] = pd.to_datetime(timeline_df['date'])
            timeline_df = timeline_df.set_index('date').sort_index()
            
            # Reindex to daily and forward-fill regimes
            daily_regimes = timeline_df['regime'].reindex(returns.index).ffill()
            
            # Calculate daily returns based on regime
            brk_weight = daily_regimes.map(regime_weights).fillna(0.5)
            bil_weight = 1.0 - brk_weight
            
            daily_returns = (
                brk_weight * returns["BRK-B"].fillna(0.0) +
                bil_weight * returns["BIL"].fillna(0.0)
            )
        else:
            # RAAAL strategies
            if column not in weights_df.columns:
                continue
                
            weight_history = (
                weights_df.pivot(index="date", columns="ticker", values=column)
                .sort_index()
                .reindex(columns=tickers, fill_value=0.0)
            )
            
            if weight_history.empty:
                continue
                
            weight_history.index = pd.to_datetime(weight_history.index)
            weight_history = weight_history.reindex(returns.index).ffill()
            valid_mask = weight_history.notna().any(axis=1)
            weight_history = weight_history.loc[valid_mask]
            
            if weight_history.empty:
                continue
                
            aligned_returns = returns.loc[weight_history.index, tickers].fillna(0.0)
            daily_returns = (weight_history.fillna(0.0) * aligned_returns).sum(axis=1)
        
        # Calculate metrics
        total_return = (1 + daily_returns).prod() - 1
        years = len(daily_returns) / 252
        annualized = (1 + total_return) ** (1 / years) - 1
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe = annualized / volatility if volatility > 0 else 0
        
        # Max drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        results.append({
            'Strategy': label,
            'Total Return': f"{total_return * 100:.1f}%",
            'Annualized Return': f"{annualized * 100:.2f}%",
            'Volatility': f"{volatility * 100:.2f}%",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Max Drawdown': f"{max_drawdown * 100:.1f}%",
        })
    
    # Build HTML table
    html = """
    <h2>📊 Performance Comparison: RAAAL vs BRK-B</h2>
    <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
        <thead>
            <tr style="background-color: #2c3e50; color: white;">
                <th style="padding: 12px; text-align: left;">Strategy</th>
                <th style="padding: 12px; text-align: right;">Total Return</th>
                <th style="padding: 12px; text-align: right;">Annualized</th>
                <th style="padding: 12px; text-align: right;">Volatility</th>
                <th style="padding: 12px; text-align: right;">Sharpe</th>
                <th style="padding: 12px; text-align: right;">Max DD</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for i, row in enumerate(results):
        bg_color = "#ecf0f1" if i % 2 == 0 else "#ffffff"
        html += f"""
            <tr style="background-color: {bg_color};">
                <td style="padding: 10px; font-weight: bold;">{row['Strategy']}</td>
                <td style="padding: 10px; text-align: right;">{row['Total Return']}</td>
                <td style="padding: 10px; text-align: right;">{row['Annualized Return']}</td>
                <td style="padding: 10px; text-align: right;">{row['Volatility']}</td>
                <td style="padding: 10px; text-align: right;">{row['Sharpe Ratio']}</td>
                <td style="padding: 10px; text-align: right;">{row['Max Drawdown']}</td>
            </tr>
        """
    
    html += """
        </tbody>
    </table>
    <div style="margin-top: 20px; padding: 15px; background-color: #e8f4f8; border-left: 4px solid #3498db; border-radius: 5px;">
        <h4>📌 Key Insights</h4>
        <ul>
            <li><strong>BRK-B (Buy & Hold)</strong>: 100% Berkshire Hathaway - Buffett's diversified conglomerate</li>
            <li><strong>BRK-B + Cash (Regime-Aware)</strong>: Tactical allocation between BRK-B and money market using regime rules:
                <ul style="margin-top: 5px; margin-left: 20px;">
                    <li>Risk-On: 80% BRK-B / 20% Cash</li>
                    <li>Risk-Off: 30% BRK-B / 70% Cash (defensive)</li>
                    <li>Inflation: 50% BRK-B / 50% Cash</li>
                </ul>
            </li>
            <li><strong>RAAAL Regime-Aware</strong>: Multi-asset tactical allocation (stocks, bonds, commodities, gold)</li>
            <li><strong>RAAAL Standard</strong>: Sharpe-optimized portfolio without regime adjustments</li>
            <li><strong>Sharpe Ratio</strong>: Risk-adjusted return (higher is better) = Annual Return ÷ Volatility</li>
            <li><strong>Max Drawdown</strong>: Largest peak-to-trough decline (lower is better)</li>
        </ul>
        <p style="margin-top: 10px; font-style: italic; color: #555;">
            The BRK-B + Cash strategy tests whether simple regime-aware timing with just two assets 
            (Buffett + cash) can match complex multi-asset portfolios.
        </p>
    </div>
    """
    
    return Div(text=html)


def create_brk_comparison_tab() -> TabPanel:
    """Create the BRK-B comparison tab."""
    
    # Layout components
    performance_table = create_brk_performance_table()
    cumulative_chart = create_brk_comparison_chart()
    
    # Title
    title = Div(text="""
        <h1 style="color: #2c3e50;">🏛️ RAAAL vs Berkshire Hathaway</h1>
        <p style="color: #7f8c8d;">
            Comparing tactical regime-aware allocation against Warren Buffett's diversified conglomerate.
            BRK-B represents a simple buy-and-hold strategy in a fundamentally sound, actively managed business.
        </p>
        <hr>
    """)
    
    # Explanation
    explanation = Div(text="""
    <div style="margin: 20px 0; padding: 20px; background-color: #fff3cd; border-left: 4px solid #ffc107; border-radius: 5px;">
        <h3>💡 Why Compare to BRK-B?</h3>
        <p>
            Berkshire Hathaway is Warren Buffett's holding company and represents a "<strong>one-decision</strong>" investment:
        </p>
        <ul>
            <li><strong>Built-in Diversification</strong>: Insurance, utilities, railroads, energy, consumer brands, and public equities</li>
            <li><strong>Active Capital Allocation</strong>: Buffett deploys cash opportunistically during market dislocations</li>
            <li><strong>Tax Efficiency</strong>: No dividend distributions, deferred capital gains</li>
            <li><strong>Fortress Balance Sheet</strong>: Typically holds $100B+ in cash and T-bills</li>
            <li><strong>Long-term Focus</strong>: Historical ~10% annualized returns with lower volatility than S&P 500</li>
        </ul>
        <p>
            If RAAAL's tactical complexity doesn't meaningfully outperform BRK-B on a risk-adjusted basis, 
            the simpler approach may be superior for most investors.
        </p>
    </div>
    """)
    
    layout = column(
        title,
        explanation,
        performance_table,
        cumulative_chart,
        sizing_mode="stretch_width",
    )
    
    return TabPanel(title="vs Buffett", child=layout)
