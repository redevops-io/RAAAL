"""Ensemble learning for regime classification.

Implements Random Forest and Gradient Boosting models as described in
CFA Institute AI monograph Chapter 5: Ensemble Learning in Investment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _align_features(frame: pd.DataFrame, feature_names: Iterable[str]) -> pd.DataFrame:
    """Reindex feature frame to match estimator feature names."""
    ordered = list(feature_names)
    missing = [col for col in ordered if col not in frame.columns]
    for col in missing:
        frame[col] = 0.0
    # Drop extras but keep deterministic order
    aligned = frame.reindex(columns=ordered)
    return aligned.fillna(0.0)


def prepare_features(timeline: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract features from historical timeline for regime classification.
    
    Features include:
    - Technical indicators (SPY price, VIX, moving averages)
    - Momentum signals (credit spread, commodity, TIP)
    - Correlation measures (SPY/TLT)
    - Market volatility
    
    Returns:
        (features_df, target_series)
    """
    df = timeline.copy()
    
    # Target variable
    y = df['regime']
    
    # Feature engineering
    features = pd.DataFrame(index=df.index)
    
    # Price-based features
    features['spy_price'] = df['spy_price']
    features['vix'] = df['vix']
    features['gold_price'] = df.get('gold_price_oz', 0.0)
    
    # Diagnostic signals (from regime detection)
    for col in df.columns:
        if col.startswith('diag_'):
            features[col.replace('diag_', '')] = df[col]
    
    # Derived features
    if 'spy_ma200' in features.columns:
        features['spy_above_ma'] = (features['spy_price'] > features['spy_ma200']).astype(float)
    
    # Rolling statistics (if enough history)
    if len(df) > 20:
        features['vix_ma5'] = df['vix'].rolling(5).mean()
        features['vix_std5'] = df['vix'].rolling(5).std()
        features['spy_ret_5d'] = df['spy_price'].pct_change(5)
    
    # Forward fill any NaN values
    features = features.ffill().fillna(0.0)
    
    return features, y


def train_ensemble_models(
    timeline: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Dict:
    """
    Train Random Forest and Gradient Boosting classifiers for regime prediction.
    
    Returns:
        Dictionary containing trained models, encoders, and performance metrics
    """
    X, y = prepare_features(timeline)
    
    # Encode regime labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        class_weight='balanced',
    )
    rf_model.fit(X_train, y_train)
    rf_score = rf_model.score(X_test, y_test)
    
    # Gradient Boosting
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
    )
    gb_model.fit(X_train, y_train)
    gb_score = gb_model.score(X_test, y_test)
    
    # Feature importance (from Random Forest)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Save models
    joblib.dump(rf_model, MODELS_DIR / 'random_forest.pkl')
    joblib.dump(gb_model, MODELS_DIR / 'gradient_boosting.pkl')
    joblib.dump(le, MODELS_DIR / 'label_encoder.pkl')
    
    return {
        'random_forest': rf_model,
        'gradient_boosting': gb_model,
        'label_encoder': le,
        'rf_accuracy': rf_score,
        'gb_accuracy': gb_score,
        'feature_importance': feature_importance,
        'feature_names': list(X.columns),
    }


def load_ensemble_models() -> Dict:
    """Load trained models from disk."""
    try:
        rf_model = joblib.load(MODELS_DIR / 'random_forest.pkl')
        gb_model = joblib.load(MODELS_DIR / 'gradient_boosting.pkl')
        le = joblib.load(MODELS_DIR / 'label_encoder.pkl')
        return {
            'random_forest': rf_model,
            'gradient_boosting': gb_model,
            'label_encoder': le,
        }
    except FileNotFoundError:
        return {}


def predict_regime_ensemble(
    features: pd.DataFrame, models: Dict
) -> Tuple[str, Dict[str, float]]:
    """
    Predict regime using ensemble of models.
    
    Returns:
        (predicted_regime, probability_dict)
    """
    if not models:
        return "risk_on", {"risk_on": 1.0, "risk_off": 0.0, "inflation": 0.0}
    
    rf_model = models['random_forest']
    gb_model = models['gradient_boosting']
    le = models['label_encoder']
    
    # Get predictions from both models
    feature_names = getattr(rf_model, "feature_names_in_", None)
    if feature_names is None:
        feature_names = getattr(gb_model, "feature_names_in_", None)
    if feature_names is not None:
        features = _align_features(features.copy(), feature_names)

    rf_proba = rf_model.predict_proba(features)[0]
    gb_proba = gb_model.predict_proba(features)[0]
    
    # Average probabilities
    avg_proba = (rf_proba + gb_proba) / 2.0
    
    # Get predicted class
    pred_class = np.argmax(avg_proba)
    pred_regime = le.inverse_transform([pred_class])[0]
    
    # Build probability dictionary
    proba_dict = {
        regime: float(prob)
        for regime, prob in zip(le.classes_, avg_proba)
    }
    
    return pred_regime, proba_dict


def compute_regime_agreement(
    timeline: pd.DataFrame, models: Dict
) -> pd.DataFrame:
    """
    Compare rule-based regime detection with ensemble predictions.
    
    Returns:
        DataFrame with columns: date, rule_based_regime, ensemble_regime, agreement
    """
    X, y_true = prepare_features(timeline)
    
    if not models:
        return pd.DataFrame()
    
    rf_model = models['random_forest']
    gb_model = models['gradient_boosting']
    le = models['label_encoder']
    
    # Predict for all samples
    feature_names = getattr(rf_model, "feature_names_in_", None)
    if feature_names is None:
        feature_names = getattr(gb_model, "feature_names_in_", None)
    if feature_names is not None:
        X = _align_features(X.copy(), feature_names)

    rf_pred = rf_model.predict(X)
    gb_pred = gb_model.predict(X)
    
    # Decode predictions
    rf_regimes = le.inverse_transform(rf_pred)
    gb_regimes = le.inverse_transform(gb_pred)
    
    # Build comparison DataFrame
    comparison = pd.DataFrame({
        'date': timeline.index,
        'rule_based': y_true.values,
        'random_forest': rf_regimes,
        'gradient_boosting': gb_regimes,
        'rf_agrees': rf_regimes == y_true.values,
        'gb_agrees': gb_regimes == y_true.values,
    })
    
    return comparison
