"""Network analysis for asset correlation networks.

Implements network theory concepts from CFA Institute AI monograph Chapter 2.
Computes centrality measures, community detection, and systemic risk indicators.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .config import UNIVERSE


def build_correlation_network(
    returns: pd.DataFrame, threshold: float = 0.5
) -> nx.Graph:
    """
    Build a network graph from asset correlations.
    
    Args:
        returns: DataFrame of asset returns
        threshold: Minimum absolute correlation to create an edge
        
    Returns:
        NetworkX graph with assets as nodes and correlations as weighted edges
    """
    tickers = [asset.ticker for asset in UNIVERSE]
    returns = returns[tickers].dropna()
    
    if returns.empty or returns.shape[0] < 2:
        G = nx.Graph()
        G.add_nodes_from(tickers)
        return G
    
    corr = returns.corr()
    
    G = nx.Graph()
    for i, ticker1 in enumerate(tickers):
        for j, ticker2 in enumerate(tickers):
            if i < j:  # Upper triangle only
                corr_val = corr.iloc[i, j]
                if abs(corr_val) >= threshold:
                    G.add_edge(ticker1, ticker2, weight=abs(corr_val))
    
    # Add isolated nodes
    for ticker in tickers:
        if ticker not in G:
            G.add_node(ticker)
    
    return G


def compute_centrality_measures(G: nx.Graph) -> pd.DataFrame:
    """
    Compute various centrality measures for network nodes.
    
    Returns:
        DataFrame with centrality scores for each asset
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame()
    
    # Degree centrality: number of connections
    degree_cent = nx.degree_centrality(G)
    
    # Betweenness centrality: broker/hub nodes
    try:
        betweenness_cent = nx.betweenness_centrality(G, weight='weight')
    except:
        betweenness_cent = {node: 0.0 for node in G.nodes()}
    
    # Eigenvector centrality: influence measure
    try:
        eigenvector_cent = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
    except:
        eigenvector_cent = {node: 0.0 for node in G.nodes()}
    
    # Closeness centrality: how close to all other nodes
    try:
        closeness_cent = nx.closeness_centrality(G)
    except:
        closeness_cent = {node: 0.0 for node in G.nodes()}
    
    df = pd.DataFrame({
        'ticker': list(degree_cent.keys()),
        'degree': list(degree_cent.values()),
        'betweenness': [betweenness_cent.get(n, 0.0) for n in degree_cent.keys()],
        'eigenvector': [eigenvector_cent.get(n, 0.0) for n in degree_cent.keys()],
        'closeness': [closeness_cent.get(n, 0.0) for n in degree_cent.keys()],
    })
    
    return df


def detect_communities(G: nx.Graph) -> Dict[str, int]:
    """
    Detect communities/clusters in the network using Louvain algorithm.
    
    Returns:
        Dictionary mapping tickers to community IDs
    """
    if G.number_of_nodes() == 0:
        return {}
    
    # Use greedy modularity maximization
    try:
        communities = nx.community.greedy_modularity_communities(G, weight='weight')
        community_map = {}
        for i, community in enumerate(communities):
            for node in community:
                community_map[node] = i
        return community_map
    except:
        # Fallback: each node is its own community
        return {node: i for i, node in enumerate(G.nodes())}


def compute_network_metrics(returns: pd.DataFrame, threshold: float = 0.5) -> Dict:
    """
    Compute comprehensive network metrics for the asset universe.
    
    Returns:
        Dictionary with network statistics and centrality measures
    """
    G = build_correlation_network(returns, threshold=threshold)
    
    # Global network metrics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G) if num_nodes > 0 else 0.0
    
    # Connected components
    num_components = nx.number_connected_components(G)
    
    # Average clustering coefficient
    try:
        avg_clustering = nx.average_clustering(G, weight='weight')
    except:
        avg_clustering = 0.0
    
    # Average shortest path (for largest component)
    try:
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        avg_path_length = nx.average_shortest_path_length(subgraph)
    except:
        avg_path_length = float('nan')
    
    centrality_df = compute_centrality_measures(G)
    communities = detect_communities(G)
    
    return {
        'graph': G,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': density,
        'num_components': num_components,
        'avg_clustering': avg_clustering,
        'avg_path_length': avg_path_length,
        'centrality': centrality_df,
        'communities': communities,
    }


def identify_systemic_risk_assets(
    centrality_df: pd.DataFrame, top_n: int = 3
) -> List[str]:
    """
    Identify assets with highest systemic risk based on centrality measures.
    
    Uses weighted combination of degree, betweenness, and eigenvector centrality.
    """
    if centrality_df.empty:
        return []
    
    # Normalize each measure
    df = centrality_df.copy()
    for col in ['degree', 'betweenness', 'eigenvector']:
        if df[col].max() > 0:
            df[f'{col}_norm'] = df[col] / df[col].max()
        else:
            df[f'{col}_norm'] = 0.0
    
    # Composite risk score (equal weights)
    df['risk_score'] = (
        df['degree_norm'] + df['betweenness_norm'] + df['eigenvector_norm']
    ) / 3.0
    
    # Return top N
    top_assets = df.nlargest(top_n, 'risk_score')['ticker'].tolist()
    return top_assets


def get_network_layout_positions(G: nx.Graph) -> Dict[str, Tuple[float, float]]:
    """
    Compute 2D layout positions for network visualization.
    
    Returns:
        Dictionary mapping node names to (x, y) positions
    """
    if G.number_of_nodes() == 0:
        return {}
    
    # Use spring layout with correlation weights
    try:
        pos = nx.spring_layout(G, weight='weight', iterations=50, seed=42)
    except:
        pos = nx.circular_layout(G)
    
    return pos
