"""Hierarchical Risk Parity (HRP) portfolio optimization.

Based on López de Prado (2016) and CFA Institute AI in Asset Management monograph.
Uses hierarchical clustering of the correlation matrix to build diversified portfolios.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from .config import UNIVERSE


def _get_quasi_diag(link: np.ndarray) -> list:
    """Sort clustered items by distance."""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]  # number of original items
    
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)  # make space
        df0 = sort_ix[sort_ix >= num_items]  # find clusters
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]  # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    
    return sort_ix.tolist()


def _get_cluster_var(cov: pd.DataFrame, c_items: list) -> float:
    """Compute variance per cluster."""
    cov_slice = cov.loc[c_items, c_items]
    w_ = _get_ivp(cov_slice).reshape(-1, 1)
    c_var = np.dot(np.dot(w_.T, cov_slice.values), w_)[0, 0]
    return c_var


def _get_ivp(cov: pd.DataFrame) -> np.ndarray:
    """Inverse variance portfolio weights."""
    ivp = 1.0 / np.diag(cov)
    ivp /= ivp.sum()
    return ivp


def _get_rec_bipart(cov: pd.DataFrame, sort_ix: list) -> pd.Series:
    """Recursive bisection to allocate weights."""
    w = pd.Series(1.0, index=sort_ix)
    c_items = [sort_ix]  # initialize all items in one cluster
    
    while len(c_items) > 0:
        c_items = [
            i[j:k]
            for i in c_items
            for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
            if len(i) > 1
        ]  # bi-section
        
        for i in range(0, len(c_items), 2):  # parse in pairs
            c_items0 = c_items[i]
            c_items1 = c_items[i + 1]
            c_var0 = _get_cluster_var(cov, c_items0)
            c_var1 = _get_cluster_var(cov, c_items1)
            alpha = 1 - c_var0 / (c_var0 + c_var1)
            w[c_items0] *= alpha  # weight 1
            w[c_items1] *= 1 - alpha  # weight 2
    
    return w


def compute_hrp_weights(returns: pd.DataFrame) -> Dict[str, float]:
    """
    Compute Hierarchical Risk Parity portfolio weights.
    
    Args:
        returns: DataFrame of asset returns
        
    Returns:
        Dictionary mapping tickers to weights
    """
    tickers = [asset.ticker for asset in UNIVERSE]
    returns = returns[tickers].dropna()
    
    if returns.empty or returns.shape[0] < 2:
        # Fallback to equal weight
        n = len(tickers)
        return {ticker: 1.0 / n for ticker in tickers}
    
    # Compute correlation matrix
    cov = returns.cov()
    corr = returns.corr()
    
    # Handle any NaN or inf values
    if corr.isnull().any().any() or np.isinf(corr.values).any():
        n = len(tickers)
        return {ticker: 1.0 / n for ticker in tickers}
    
    # Convert correlation to distance matrix
    dist = np.sqrt((1 - corr) / 2.0)
    dist = dist.fillna(0.0)
    
    # Hierarchical clustering
    link = linkage(squareform(dist.values), method='single')
    
    # Quasi-diagonalization
    sort_ix = _get_quasi_diag(link)
    sort_ix = [corr.index[i] for i in sort_ix]
    
    # Recursive bisection
    hrp_weights = _get_rec_bipart(cov, sort_ix)
    
    # Normalize to sum to 1
    hrp_weights = hrp_weights / hrp_weights.sum()
    
    return {ticker: float(hrp_weights[ticker]) for ticker in tickers}


def get_dendrogram_data(returns: pd.DataFrame) -> tuple:
    """
    Generate dendrogram data for visualization.
    
    Returns:
        (linkage_matrix, labels, distance_matrix)
    """
    tickers = [asset.ticker for asset in UNIVERSE]
    returns = returns[tickers].dropna()
    
    if returns.empty or returns.shape[0] < 2:
        return None, tickers, None
    
    corr = returns.corr()
    dist = np.sqrt((1 - corr) / 2.0).fillna(0.0)
    
    link = linkage(squareform(dist.values), method='single')
    
    return link, tickers, dist.values
