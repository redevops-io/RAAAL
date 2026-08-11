"""Ensemble learning for regime classification.

Implements Random Forest and Gradient Boosting models as described in
CFA Institute AI monograph Chapter 5: Ensemble Learning in Investment.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

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
    timeline: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    embargo: int = 21,
) -> Dict:
    """
    Train Random Forest and Gradient Boosting classifiers for regime prediction.

    The split is **chronological with an embargo**, not shuffled. `timeline` is a
    time series whose features are built from overlapping rolling windows, so a
    shuffled `train_test_split` places neighbouring — effectively duplicate —
    observations on both sides of the split. The accuracy that produces is not
    out-of-sample, and the figure previously published from it (>80%) was not a
    measure of predictive skill.

    `embargo` drops that many rows between train and test so windowed features
    computed near the boundary cannot span it.

    Returns:
        Dictionary containing trained models, encoders, and performance metrics
    """
    X, y = prepare_features(timeline)

    # Encode regime labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Chronological split: train on the past, test on the future, embargo between.
    n = len(X)
    n_test = max(1, int(round(n * test_size)))
    split = n - n_test
    train_end = max(1, split - embargo)

    X_train, y_train = X.iloc[:train_end], y_encoded[:train_end]
    X_test, y_test = X.iloc[split:], y_encoded[split:]

    # A chronological split can legitimately leave one class in the training
    # window when regimes arrive in long contiguous blocks. That is a real
    # limitation of the data, not a bug to be papered over with shuffling —
    # say so plainly instead of letting sklearn raise something cryptic.
    if len(np.unique(y_train)) < 2:
        raise ValueError(
            f"Chronological train split covers only one regime "
            f"({le.inverse_transform(np.unique(y_train))[0]!r}) across "
            f"{train_end} rows. Need a longer history or a smaller embargo "
            f"(currently {embargo}) so the training window spans a regime change."
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
    
    # The last observation the model actually saw. Everything downstream depends
    # on this being recorded — a model artifact without a training cutoff cannot
    # be checked for look-ahead, so it must not be trusted.
    train_cutoff = pd.Timestamp(X.index[train_end - 1]) if train_end > 0 else None

    metadata = {
        'train_cutoff': train_cutoff.isoformat() if train_cutoff is not None else None,
        'train_rows': int(train_end),
        'test_rows': int(len(X_test)),
        'embargo': int(embargo),
        'feature_names': list(X.columns),
        'random_state': int(random_state),
    }

    # Save models
    joblib.dump(rf_model, MODELS_DIR / 'random_forest.pkl')
    joblib.dump(gb_model, MODELS_DIR / 'gradient_boosting.pkl')
    joblib.dump(le, MODELS_DIR / 'label_encoder.pkl')
    (MODELS_DIR / 'metadata.json').write_text(json.dumps(metadata, indent=2))

    return {
        'random_forest': rf_model,
        'gradient_boosting': gb_model,
        'label_encoder': le,
        'rf_accuracy': rf_score,
        'gb_accuracy': gb_score,
        'feature_importance': feature_importance,
        'feature_names': list(X.columns),
        'metadata': metadata,
    }


def load_ensemble_models(as_of: pd.Timestamp | str | None = None) -> Dict:
    """Load trained models from disk, refusing any that saw the future.

    `as_of` is the simulated date the caller intends to predict for. A model whose
    training data extends to or beyond that date cannot be used to predict it, and
    this function returns ``{}`` rather than serving it.

    Passing ``as_of=None`` means "live use, no simulated date" and skips the check.
    Backtests MUST pass a date; the previous behaviour — loading a model trained on
    the full timeline and applying it to every historical evaluation date — made
    every `ml`-mode result look-ahead-contaminated.
    """
    try:
        rf_model = joblib.load(MODELS_DIR / 'random_forest.pkl')
        gb_model = joblib.load(MODELS_DIR / 'gradient_boosting.pkl')
        le = joblib.load(MODELS_DIR / 'label_encoder.pkl')
    except FileNotFoundError:
        return {}

    metadata_path = MODELS_DIR / 'metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    if as_of is not None:
        cutoff = metadata.get('train_cutoff')
        if cutoff is None:
            logger.warning(
                "Refusing ensemble models: no train_cutoff recorded, so look-ahead "
                "cannot be ruled out. Retrain to regenerate metadata.json."
            )
            return {}
        if pd.Timestamp(cutoff) >= pd.Timestamp(as_of):
            logger.warning(
                "Refusing ensemble models for %s: trained through %s (would leak).",
                pd.Timestamp(as_of).date(),
                pd.Timestamp(cutoff).date(),
            )
            return {}

    return {
        'random_forest': rf_model,
        'gradient_boosting': gb_model,
        'label_encoder': le,
        'metadata': metadata,
    }


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
