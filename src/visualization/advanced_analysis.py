"""Advanced analysis dashboard - second page for RAAAL.

Compares RAAAL approach with HRP optimization and highlights regime changes
using network analysis and ensemble learning.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Div, HoverTool, TabPanel, Tabs
from bokeh.palettes import Category10_10, Category20_20
from bokeh.plotting import figure

from ..ensemble_regime import compute_regime_agreement, load_ensemble_models, train_ensemble_models
from ..network import compute_network_metrics, identify_systemic_risk_assets

HISTORY_JSON = Path("data/history/history_summary.json")


def load_history_data() -> dict:
    """Load historical analysis results."""
    if not HISTORY_JSON.exists():
        return {}
    return json.loads(HISTORY_JSON.read_text())


def create_performance_comparison():
    """Create performance comparison chart for all 9 strategies."""
    data = load_history_data()
    if not data:
        return Div(text="<h3>No historical data available. Run backtest first.</h3>")
    
    performance = data.get("performance", {})
    
    # Build comparison table
    strategies = [
        ("Standard (Restricted)", performance.get("standard_restricted", 0.0)),
        ("Standard (Unrestricted)", performance.get("standard_unrestricted", 0.0)),
        ("Regime (Restricted)", performance.get("regime_restricted", 0.0)),
        ("Regime (Unrestricted)", performance.get("regime_unrestricted", 0.0)),
        ("HRP (Regime Unrestricted)", performance.get("hrp_regime_unrestricted", 0.0)),
        ("HRP (Regime Restricted)", performance.get("hrp_regime_restricted", 0.0)),
    ]
    
    html = """
    <h2>📊 Performance Comparison: RAAAL vs HRP</h2>
    <table style="width:100%; border-collapse: collapse; font-size: 14px;">
        <thead>
            <tr style="background-color: #2b3e50; color: white;">
                <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Strategy</th>
                <th style="padding: 10px; text-align: right; border: 1px solid #ddd;">Annualized Return</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for i, (name, ret) in enumerate(strategies):
        bg_color = "#f9f9f9" if i % 2 == 0 else "#ffffff"
        color = "#27ae60" if ret > 0 else "#e74c3c"
        html += f"""
            <tr style="background-color: {bg_color};">
                <td style="padding: 10px; border: 1px solid #ddd;">{name}</td>
                <td style="padding: 10px; text-align: right; border: 1px solid #ddd; color: {color}; font-weight: bold;">
                    {ret:.2%}
                </td>
            </tr>
        """
    
    html += """
        </tbody>
    </table>
    <div style="margin-top: 15px; padding: 10px; background-color: #fff3cd; border-left: 4px solid #ffc107; border-radius: 5px;">
        <p style="margin: 5px 0; color: #856404; font-size: 13px;"><strong>⚠️ HRP vs RAAAL Comparison:</strong></p>
        <ul style="margin: 5px 0 5px 20px; color: #856404; font-size: 12px;">
            <li><strong>RAAAL (13.08%):</strong> Optimizes for risk-adjusted returns (Sharpe ratio) with regime-specific constraints. 
                Dynamically adjusts to market conditions.</li>
            <li><strong>HRP (2.03-2.06%):</strong> Pure diversification strategy based on correlation clustering. 
                Does NOT optimize for returns - focuses solely on balanced risk allocation.</li>
            <li><strong>Why the difference?</strong> HRP avoids concentration risk but sacrifices return potential. 
                RAAAL balances risk AND return, leading to higher performance in favorable regimes.</li>
            <li><strong>Use case:</strong> HRP is suitable for ultra-conservative portfolios prioritizing stability over growth. 
                RAAAL is designed for tactical allocation seeking risk-adjusted outperformance.</li>
        </ul>
    </div>
    """
    
    return Div(text=html)


def create_regime_timeline_with_ensemble():
    """Create two separate timeline charts: rule-based and ensemble ML predictions."""
    data = load_history_data()
    if not data:
        return Div(text="<h3>No data available</h3>")
    
    timeline_dict = data.get("timeline", {})
    if not timeline_dict:
        return Div(text="<h3>No timeline data</h3>")
    
    timeline_df = pd.DataFrame(timeline_dict)
    if 'date' in timeline_df.columns:
        timeline_df['date'] = pd.to_datetime(timeline_df['date'])
    
    # Train ensemble models
    models_result = train_ensemble_models(timeline_df.set_index('date'))
    
    # Get predictions
    comparison = compute_regime_agreement(timeline_df.set_index('date'), models_result)
    
    # Merge with timeline
    timeline_df = timeline_df.merge(comparison, on='date', how='left')
    
    # Load weights data
    weights_dict = data.get("weights", {})
    weights_df = pd.DataFrame(weights_dict)
    if not weights_df.empty:
        weights_df['date'] = pd.to_datetime(weights_df['date'])
    
    # Color mapping
    regime_colors = {"risk_on": "#27ae60", "risk_off": "#e74c3c", "inflation": "#f39c12"}
    
    from ..config import UNIVERSE
    tickers = [asset.ticker for asset in UNIVERSE]
    
    # Create first plot: Rule-based regimes
    p1 = figure(
        width=1200,
        height=200,
        x_axis_type='datetime',
        title="Rule-Based Regime Detection",
        toolbar_location="above",
    )
    
    # Build segments for rule-based with weight data
    segments = []
    current_regime = timeline_df['regime'].iloc[0] if len(timeline_df) > 0 else 'risk_on'
    start_date = timeline_df['date'].iloc[0] if len(timeline_df) > 0 else None
    start_idx = 0
    
    for i in range(1, len(timeline_df)):
        if timeline_df['regime'].iloc[i] != current_regime:
            end_date = timeline_df['date'].iloc[i-1]
            # Get weights at start and end of segment
            seg_data = {
                'left': start_date,
                'right': end_date,
                'regime': current_regime,
                'start_idx': start_idx,
                'end_idx': i-1,
            }
            
            # Add weight data for this segment
            if not weights_df.empty:
                start_weights = weights_df[weights_df['date'] == start_date]
                end_weights = weights_df[weights_df['date'] == end_date]
                for ticker in tickers:
                    start_w = start_weights[start_weights['ticker'] == ticker]['weight'].iloc[0] if len(start_weights[start_weights['ticker'] == ticker]) > 0 else 0.0
                    end_w = end_weights[end_weights['ticker'] == ticker]['weight'].iloc[0] if len(end_weights[end_weights['ticker'] == ticker]) > 0 else 0.0
                    seg_data[f'start_{ticker}'] = float(start_w)
                    seg_data[f'end_{ticker}'] = float(end_w)
            
            segments.append(seg_data)
            current_regime = timeline_df['regime'].iloc[i]
            start_date = timeline_df['date'].iloc[i]
            start_idx = i
    
    if start_date is not None and len(timeline_df) > 0:
        end_date = timeline_df['date'].iloc[-1]
        seg_data = {
            'left': start_date,
            'right': end_date,
            'regime': current_regime,
            'start_idx': start_idx,
            'end_idx': len(timeline_df) - 1,
        }
        if not weights_df.empty:
            start_weights = weights_df[weights_df['date'] == start_date]
            end_weights = weights_df[weights_df['date'] == end_date]
            for ticker in tickers:
                start_w = start_weights[start_weights['ticker'] == ticker]['weight'].iloc[0] if len(start_weights[start_weights['ticker'] == ticker]) > 0 else 0.0
                end_w = end_weights[end_weights['ticker'] == ticker]['weight'].iloc[0] if len(end_weights[end_weights['ticker'] == ticker]) > 0 else 0.0
                seg_data[f'start_{ticker}'] = float(start_w)
                seg_data[f'end_{ticker}'] = float(end_w)
        segments.append(seg_data)
    
    regime_source = ColumnDataSource(pd.DataFrame(segments))
    
    quad_renderer = p1.quad(
        left='left', right='right', bottom=0, top=1,
        source=regime_source,
        fill_color='color', fill_alpha=0.5, line_alpha=0,
        hover_fill_alpha=0.7
    )
    
    for seg in segments:
        seg['color'] = regime_colors.get(seg['regime'], '#cccccc')
    regime_source.data = pd.DataFrame(segments).to_dict('list')
    
    p1.yaxis.visible = False
    p1.y_range.start = 0
    p1.y_range.end = 1
    
    # Add tap tool
    from bokeh.models import TapTool
    tap_tool = TapTool(renderers=[quad_renderer])
    p1.add_tools(tap_tool)
    
    # Create second plot: Ensemble ML regimes
    p2 = figure(
        width=1200,
        height=200,
        x_axis_type='datetime',
        x_range=p1.x_range,  # Link x-axis
        title="Ensemble ML Regime Prediction (Random Forest + Gradient Boosting)",
        toolbar_location="above",
    )
    
    # Build segments for ML predictions with weight data
    ml_segments = []
    current_ml = timeline_df['random_forest'].iloc[0] if len(timeline_df) > 0 else 'risk_on'
    start_date = timeline_df['date'].iloc[0] if len(timeline_df) > 0 else None
    start_idx = 0
    
    for i in range(1, len(timeline_df)):
        if timeline_df['random_forest'].iloc[i] != current_ml:
            end_date = timeline_df['date'].iloc[i-1]
            seg_data = {
                'left': start_date,
                'right': end_date,
                'regime': current_ml,
                'start_idx': start_idx,
                'end_idx': i-1,
            }
            
            # Add weight data
            if not weights_df.empty:
                start_weights = weights_df[weights_df['date'] == start_date]
                end_weights = weights_df[weights_df['date'] == end_date]
                for ticker in tickers:
                    start_w = start_weights[start_weights['ticker'] == ticker]['weight'].iloc[0] if len(start_weights[start_weights['ticker'] == ticker]) > 0 else 0.0
                    end_w = end_weights[end_weights['ticker'] == ticker]['weight'].iloc[0] if len(end_weights[end_weights['ticker'] == ticker]) > 0 else 0.0
                    seg_data[f'start_{ticker}'] = float(start_w)
                    seg_data[f'end_{ticker}'] = float(end_w)
            
            ml_segments.append(seg_data)
            current_ml = timeline_df['random_forest'].iloc[i]
            start_date = timeline_df['date'].iloc[i]
            start_idx = i
    
    if start_date is not None and len(timeline_df) > 0:
        end_date = timeline_df['date'].iloc[-1]
        seg_data = {
            'left': start_date,
            'right': end_date,
            'regime': current_ml,
            'start_idx': start_idx,
            'end_idx': len(timeline_df) - 1,
        }
        if not weights_df.empty:
            start_weights = weights_df[weights_df['date'] == start_date]
            end_weights = weights_df[weights_df['date'] == end_date]
            for ticker in tickers:
                start_w = start_weights[start_weights['ticker'] == ticker]['weight'].iloc[0] if len(start_weights[start_weights['ticker'] == ticker]) > 0 else 0.0
                end_w = end_weights[end_weights['ticker'] == ticker]['weight'].iloc[0] if len(end_weights[end_weights['ticker'] == ticker]) > 0 else 0.0
                seg_data[f'start_{ticker}'] = float(start_w)
                seg_data[f'end_{ticker}'] = float(end_w)
        ml_segments.append(seg_data)
    
    ml_regime_source = ColumnDataSource(pd.DataFrame(ml_segments))
    
    ml_quad_renderer = p2.quad(
        left='left', right='right', bottom=0, top=1,
        source=ml_regime_source,
        fill_color='color', fill_alpha=0.5, line_alpha=0,
        hover_fill_alpha=0.7
    )
    
    for seg in ml_segments:
        seg['color'] = regime_colors.get(seg['regime'], '#cccccc')
    ml_regime_source.data = pd.DataFrame(ml_segments).to_dict('list')
    
    p2.yaxis.visible = False
    p2.y_range.start = 0
    p2.y_range.end = 1
    
    # Add tap tool
    ml_tap_tool = TapTool(renderers=[ml_quad_renderer])
    p2.add_tools(ml_tap_tool)
    
    # Create info divs for displaying weight changes
    regime_info = Div(text="<p>Click on a regime segment to see allocation details</p>", width=1200)
    ml_regime_info = Div(text="<p>Click on a regime segment to see allocation details</p>", width=1200)
    
    # Add JavaScript callbacks for click interactions
    from bokeh.models import CustomJS
    
    regime_source.selected.js_on_change(
        'indices',
        CustomJS(
            args=dict(source=regime_source, info=regime_info, tickers=tickers),
            code="""
            const idx = source.selected.indices[0];
            if (idx === undefined) { return; }
            
            const startDate = new Date(source.data['left'][idx]).toISOString().slice(0, 10);
            const endDate = new Date(source.data['right'][idx]).toISOString().slice(0, 10);
            const regime = source.data['regime'][idx];
            
            // Collect all weights
            const weights = [];
            for (let i = 0; i < tickers.length; i++) {
                const ticker = tickers[i];
                const startVal = source.data['start_' + ticker][idx] ?? 0;
                const endVal = source.data['end_' + ticker][idx] ?? 0;
                if (startVal > 0.001 || endVal > 0.001) {
                    weights.push({
                        ticker: ticker,
                        start: startVal,
                        end: endVal,
                        change: Math.abs(startVal - endVal)
                    });
                }
            }
            
            // Sort by end weight descending
            weights.sort((a, b) => b.end - a.end);
            
            const lines = weights.map(w => 
                `${w.ticker}: ${(w.start * 100).toFixed(1)}% → ${(w.end * 100).toFixed(1)}%`
            );
            
            const body = lines.length ? lines.join('<br>') : 'No allocations';
            info.text = `<div style="padding: 10px; background-color: #f8f9fa; border-left: 4px solid #007bff; margin: 10px 0;">
                <b>${regime}</b> (${startDate} to ${endDate})<br><br>${body}
            </div>`;
        """
        )
    )
    
    ml_regime_source.selected.js_on_change(
        'indices',
        CustomJS(
            args=dict(source=ml_regime_source, info=ml_regime_info, tickers=tickers),
            code="""
            const idx = source.selected.indices[0];
            if (idx === undefined) { return; }
            
            const startDate = new Date(source.data['left'][idx]).toISOString().slice(0, 10);
            const endDate = new Date(source.data['right'][idx]).toISOString().slice(0, 10);
            const regime = source.data['regime'][idx];
            
            // Collect all weights
            const weights = [];
            for (let i = 0; i < tickers.length; i++) {
                const ticker = tickers[i];
                const startVal = source.data['start_' + ticker][idx] ?? 0;
                const endVal = source.data['end_' + ticker][idx] ?? 0;
                if (startVal > 0.001 || endVal > 0.001) {
                    weights.push({
                        ticker: ticker,
                        start: startVal,
                        end: endVal,
                        change: Math.abs(startVal - endVal)
                    });
                }
            }
            
            // Sort by end weight descending
            weights.sort((a, b) => b.end - a.end);
            
            const lines = weights.map(w => 
                `${w.ticker}: ${(w.start * 100).toFixed(1)}% → ${(w.end * 100).toFixed(1)}%`
            );
            
            const body = lines.length ? lines.join('<br>') : 'No allocations';
            info.text = `<div style="padding: 10px; background-color: #f8f9fa; border-left: 4px solid #28a745; margin: 10px 0;">
                <b>${regime}</b> (ML Predicted: ${startDate} to ${endDate})<br><br>${body}
            </div>`;
        """
        )
    )
    
    # Add accuracy metrics
    rf_accuracy = models_result.get('rf_accuracy', 0.0)
    gb_accuracy = models_result.get('gb_accuracy', 0.0)
    
    # Calculate agreement percentage
    agreement = (timeline_df['regime'] == timeline_df['random_forest']).sum() / len(timeline_df) * 100 if len(timeline_df) > 0 else 0
    
    accuracy_html = f"""
    <div style="margin-top: 10px; padding: 10px; background-color: #ecf0f1; border-radius: 5px;">
        <h4>Ensemble Model Performance</h4>
        <p><strong>Random Forest Accuracy:</strong> {rf_accuracy:.2%}</p>
        <p><strong>Gradient Boosting Accuracy:</strong> {gb_accuracy:.2%}</p>
        <p><strong>Agreement with Rule-Based:</strong> {agreement:.1f}%</p>
        <p style="font-size: 12px; color: #555;">
            Models trained on {len(timeline_df)} historical observations using technical indicators,
            momentum signals, and correlation measures. <strong>Click on regime segments above to see allocation changes.</strong>
        </p>
    </div>
    """
    
    return column(p1, regime_info, p2, ml_regime_info, Div(text=accuracy_html))


def create_feature_importance_chart():
    """Display feature importance from Random Forest model."""
    data = load_history_data()
    if not data:
        return Div(text="<h3>No data</h3>")
    
    timeline_dict = data.get("timeline", {})
    timeline_df = pd.DataFrame(timeline_dict)
    if 'date' in timeline_df.columns:
        timeline_df['date'] = pd.to_datetime(timeline_df['date'])
    
    models_result = train_ensemble_models(timeline_df.set_index('date'))
    feature_importance = models_result.get('feature_importance', pd.DataFrame())
    
    if feature_importance.empty:
        return Div(text="<h3>No feature importance data</h3>")
    
    # Top 10 features
    top_features = feature_importance.head(10)
    
    source = ColumnDataSource(top_features)
    
    p = figure(
        y_range=list(reversed(top_features['feature'].tolist())),
        width=600,
        height=400,
        title="Top 10 Most Important Features for Regime Detection",
        toolbar_location=None,
    )
    
    p.hbar(y='feature', right='importance', height=0.7, source=source, color='#3498db')
    p.xaxis.axis_label = "Importance Score"
    
    return p


def create_network_visualization_by_regime():
    """Show correlation networks for each regime."""
    data = load_history_data()
    if not data:
        return Div(text="<h3>No data</h3>")
    
    weights_dict = data.get("weights", {})
    weights_df = pd.DataFrame(weights_dict)
    
    timeline_dict = data.get("timeline", {})
    timeline_df = pd.DataFrame(timeline_dict)
    
    if weights_df.empty or timeline_df.empty:
        return Div(text="<h3>Insufficient data for network analysis</h3>")
    
    # Load prices for returns calculation
    from ..data_loader import download_prices
    from ..features import compute_returns
    from ..config import UNIVERSE
    from datetime import datetime
    
    tickers = [asset.ticker for asset in UNIVERSE]
    prices = download_prices(tickers, start=datetime(2016, 1, 1), end=datetime.now())
    returns = compute_returns(prices)
    
    # Compute networks per regime
    regime_networks = {}
    for regime in ['risk_on', 'risk_off', 'inflation']:
        regime_dates = timeline_df[timeline_df['regime'] == regime]['date']
        if len(regime_dates) == 0:
            continue
        
        # Filter returns for this regime
        regime_returns = returns[returns.index.isin(pd.to_datetime(regime_dates))]
        if regime_returns.empty:
            continue
        
        metrics = compute_network_metrics(regime_returns, threshold=0.4)
        regime_networks[regime] = metrics
    
    # Build HTML summary
    html = "<h2>🕸️ Network Analysis by Regime</h2>"
    
    for regime, metrics in regime_networks.items():
        systemic_assets = identify_systemic_risk_assets(metrics['centrality'], top_n=3)
        
        html += f"""
        <div style="margin: 20px 0; padding: 15px; background-color: #ecf0f1; border-left: 4px solid #3498db; border-radius: 5px;">
            <h3>{regime.replace('_', ' ').title()}</h3>
            <ul>
                <li><strong>Nodes:</strong> {metrics['num_nodes']}</li>
                <li><strong>Edges:</strong> {metrics['num_edges']}</li>
                <li><strong>Network Density:</strong> {metrics['density']:.3f}</li>
                <li><strong>Avg Clustering:</strong> {metrics['avg_clustering']:.3f}</li>
                <li><strong>Connected Components:</strong> {metrics['num_components']}</li>
                <li><strong>Systemic Risk Assets:</strong> {', '.join(systemic_assets)}</li>
            </ul>
        </div>
        """
    
    html += """
    <p style="color: #555; font-size: 12px;">
        <strong>Network Metrics:</strong> Higher density indicates stronger correlations.
        Systemic risk assets are identified using centrality measures (degree, betweenness, eigenvector).
    </p>
    """
    
    return Div(text=html)


def create_cumulative_returns_chart():
    """Create cumulative returns chart comparing all strategies."""
    data = load_history_data()
    if not data:
        return Div(text="<h3>No data</h3>")
    
    weights_dict = data.get("weights", {})
    weights_df = pd.DataFrame(weights_dict)
    
    timeline_dict = data.get("timeline", {})
    timeline_df = pd.DataFrame(timeline_dict)
    
    if weights_df.empty or timeline_df.empty:
        return Div(text="<h3>Insufficient data for cumulative returns</h3>")
    
    # Load prices for returns calculation
    from ..data_loader import download_prices
    from ..features import compute_returns
    from ..config import UNIVERSE
    from datetime import datetime
    
    tickers = [asset.ticker for asset in UNIVERSE]
    prices = download_prices(tickers, start=datetime(2016, 1, 1), end=datetime.now())
    returns = compute_returns(prices)
    
    # Calculate cumulative returns for each strategy
    strategy_map = {
        "Regime (Unrestricted)": "unrestricted_weight",
        "Regime (Restricted)": "weight",
        "Standard (Unrestricted)": "standard_unrestricted_weight",
        "HRP (Unrestricted)": "hrp_weight",
        "HRP (Regime Restricted)": "hrp_regime_restricted_weight",
    }
    
    p = figure(
        width=900,
        height=400,
        x_axis_type='datetime',
        title="Cumulative Returns Comparison (2016-2025)",
        toolbar_location="above",
    )
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    for (label, column), color in zip(strategy_map.items(), colors):
        if column not in weights_df.columns:
            continue
            
        # Pivot weights
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
            
        aligned_returns = returns.loc[weight_history.index].fillna(0.0)
        daily_returns = (weight_history.fillna(0.0) * aligned_returns).sum(axis=1)
        cumulative_returns = (1.0 + daily_returns).cumprod()
        
        p.line(weight_history.index, cumulative_returns, 
               line_width=2, color=color, alpha=0.8, legend_label=label)
    
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.xaxis.axis_label = "Date"
    p.yaxis.axis_label = "Cumulative Return (1.0 = initial)"
    
    return p


def create_advanced_analysis_tab() -> TabPanel:
    """Create the advanced analysis tab."""
    
    # Layout components
    perf_comparison = create_performance_comparison()
    regime_timeline = create_regime_timeline_with_ensemble()
    feature_importance = create_feature_importance_chart()
    network_viz = create_network_visualization_by_regime()
    cumulative_returns = create_cumulative_returns_chart()
    
    # Title
    title = Div(text="""
        <h1 style="color: #2c3e50;">🧠 Advanced Analysis: ML-Enhanced RAAAL</h1>
        <p style="color: #7f8c8d;">
            Comparing regime-adjusted allocation with Hierarchical Risk Parity (HRP),
            ensemble learning regime detection, and network-based risk analysis.
        </p>
        <hr>
    """)
    
    layout = column(
        title,
        perf_comparison,
        row(regime_timeline),
        row(feature_importance, cumulative_returns),
        network_viz,
    )
    
    return TabPanel(title="vs Academia", child=layout)
