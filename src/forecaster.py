"""Return forecaster module — LightGBM, LSTM, and Transformer backends.

Produces expected-return estimates (μ) that can replace the naive
``exponential_mean`` in the allocation pipeline.  Supports rolling retrain,
model persistence, and drift detection.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from .config import MED_LOOKBACK, ordered_tickers
from .features import compute_returns, exponential_mean
from .features_alpha import build_universe_features

logger = logging.getLogger(__name__)

MODEL_DIR = Path("data/models/forecaster")

BackendType = Literal["lightgbm", "lstm", "transformer"]

# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


@dataclass
class DriftReport:
    """Result of a distribution-shift check on recent predictions."""

    is_drifted: bool
    metric: str
    value: float
    threshold: float


def detect_drift(
    recent_predictions: pd.Series,
    historical_predictions: pd.Series,
    threshold: float = 2.0,
) -> DriftReport:
    """Simple z-score drift detector on prediction distribution.

    Compares the mean of recent predictions to the historical distribution.
    If the z-score exceeds ``threshold`` a retrain is recommended.
    """
    if historical_predictions.empty or recent_predictions.empty:
        return DriftReport(is_drifted=False, metric="zscore", value=0.0, threshold=threshold)

    hist_mean = historical_predictions.mean()
    hist_std = historical_predictions.std()
    if hist_std == 0 or np.isnan(hist_std):
        return DriftReport(is_drifted=False, metric="zscore", value=0.0, threshold=threshold)

    recent_mean = recent_predictions.mean()
    zscore = abs((recent_mean - hist_mean) / hist_std)
    return DriftReport(
        is_drifted=zscore > threshold,
        metric="zscore",
        value=float(zscore),
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# LightGBM backend
# ---------------------------------------------------------------------------


class LightGBMForecaster:
    """Gradient-boosted tree forecaster for cross-sectional return prediction."""

    def __init__(
        self,
        horizon: int = 5,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
    ) -> None:
        self.horizon = horizon
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
        )
        self.models: Dict[str, Any] = {}  # ticker -> fitted model
        self._feature_names: Dict[str, List[str]] = {}  # ticker -> column names
        self._prediction_history: Dict[str, List[float]] = {}

    def fit(
        self,
        features: pd.DataFrame,
        returns: pd.DataFrame,
        tickers: Optional[List[str]] = None,
    ) -> None:
        """Train one LightGBM regressor per ticker."""
        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("lightgbm not installed — skipping LightGBM fit")
            return

        tickers = tickers or ordered_tickers()
        target = returns.shift(-self.horizon).reindex(features.index)

        for ticker in tickers:
            y = target.get(ticker)
            if y is None:
                continue
            # Grab features for this ticker
            cols = [c for c in features.columns if c.startswith(f"{ticker}_")]
            if not cols:
                cols = list(features.columns)

            X = features[cols]
            mask = X.notna().all(axis=1) & y.notna()
            X_clean, y_clean = X.loc[mask], y.loc[mask]
            if len(X_clean) < 60:
                logger.info("Skipping %s — not enough data (%d rows)", ticker, len(X_clean))
                continue

            model = lgb.LGBMRegressor(**self.params, verbose=-1)
            model.fit(X_clean.values, y_clean.values, feature_name=list(X_clean.columns))
            self.models[ticker] = model
            self._feature_names[ticker] = list(X_clean.columns)
            self._prediction_history.setdefault(ticker, [])

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Return expected μ for each ticker (latest row of features)."""
        preds: Dict[str, float] = {}
        tickers = ordered_tickers()
        for ticker in tickers:
            model = self.models.get(ticker)
            if model is None:
                preds[ticker] = 0.0
                continue
            # Use stored feature names to pass a named DataFrame (avoids sklearn warning)
            stored_cols = self._feature_names.get(ticker)
            if stored_cols:
                cols = [c for c in stored_cols if c in features.columns]
                if len(cols) != len(stored_cols):
                    # Columns missing — fall back to pattern match
                    cols = [c for c in features.columns if c.startswith(f"{ticker}_")]
                    if not cols:
                        cols = list(features.columns)
            else:
                cols = [c for c in features.columns if c.startswith(f"{ticker}_")]
                if not cols:
                    cols = list(features.columns)
            row = features[cols].iloc[[-1]].fillna(0.0)
            pred = float(model.predict(row)[0])
            preds[ticker] = pred
            self._prediction_history.setdefault(ticker, []).append(pred)
        return pd.Series(preds, index=tickers)

    def save(self, path: Optional[Path] = None) -> None:
        path = path or MODEL_DIR / "lightgbm"
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "models.pkl", "wb") as fh:
            pickle.dump(self.models, fh)

    def load(self, path: Optional[Path] = None) -> None:
        path = path or MODEL_DIR / "lightgbm"
        model_file = path / "models.pkl"
        if model_file.exists():
            with open(model_file, "rb") as fh:
                self.models = pickle.load(fh)


# ---------------------------------------------------------------------------
# LSTM backend (PyTorch)
# ---------------------------------------------------------------------------


class LSTMForecaster:
    """LSTM network for sequential return forecasting."""

    def __init__(
        self,
        seq_len: int = 30,
        hidden_dim: int = 64,
        num_layers: int = 2,
        horizon: int = 5,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 32,
    ) -> None:
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self._model: Any = None
        self._scaler: Any = None
        self._n_features: int = 0
        self._prediction_history: List[float] = []

    def _build_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        seqs, targets = [], []
        for i in range(self.seq_len, len(X)):
            seqs.append(X[i - self.seq_len : i])
            targets.append(y[i])
        return np.array(seqs), np.array(targets)

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> None:
        """Train an LSTM on the full feature matrix to predict aggregate return."""
        try:
            import torch
            import torch.nn as nn
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.warning("torch or sklearn not installed — skipping LSTM fit")
            return

        mask = features.notna().all(axis=1) & target.notna()
        X_raw = features.loc[mask].values.astype(np.float32)
        y_raw = target.loc[mask].values.astype(np.float32)

        if len(X_raw) < self.seq_len + 20:
            logger.info("Not enough data for LSTM (%d rows)", len(X_raw))
            return

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_raw)
        X_seq, y_seq = self._build_sequences(X_scaled, y_raw)
        self._n_features = X_seq.shape[2]

        # Define model
        class _LSTM(nn.Module):
            def __init__(self_m, input_dim: int, hidden_dim: int, num_layers: int):
                super().__init__()
                self_m.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
                self_m.fc = nn.Linear(hidden_dim, 1)

            def forward(self_m, x: torch.Tensor) -> torch.Tensor:
                out, _ = self_m.lstm(x)
                return self_m.fc(out[:, -1, :]).squeeze(-1)

        model = _LSTM(self._n_features, self.hidden_dim, self.num_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X_seq, dtype=torch.float32)
        y_t = torch.tensor(y_seq, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.debug("LSTM epoch %d/%d  loss=%.6f", epoch + 1, self.epochs, epoch_loss / len(loader))

        model.eval()
        self._model = model

    def predict(self, features: pd.DataFrame) -> float:
        """Predict aggregate expected return from the latest feature window."""
        if self._model is None or self._scaler is None:
            return 0.0
        try:
            import torch
        except ImportError:
            return 0.0

        X_raw = features.iloc[-self.seq_len :].values.astype(np.float32)
        if len(X_raw) < self.seq_len:
            return 0.0
        X_scaled = self._scaler.transform(X_raw)
        X_t = torch.tensor(X_scaled[np.newaxis, :, :], dtype=torch.float32)
        with torch.no_grad():
            pred = float(self._model(X_t).item())
        self._prediction_history.append(pred)
        return pred

    def save(self, path: Optional[Path] = None) -> None:
        path = path or MODEL_DIR / "lstm"
        path.mkdir(parents=True, exist_ok=True)
        try:
            import torch
            if self._model is not None:
                torch.save(self._model.state_dict(), path / "lstm_state.pt")
            if self._scaler is not None:
                with open(path / "scaler.pkl", "wb") as fh:
                    pickle.dump(self._scaler, fh)
        except ImportError:
            pass

    def load(self, path: Optional[Path] = None) -> None:
        path = path or MODEL_DIR / "lstm"
        # Caller must re-initialize model architecture before loading state
        logger.info("LSTM load from %s (requires architecture re-init)", path)


# ---------------------------------------------------------------------------
# Transformer backend (PyTorch)
# ---------------------------------------------------------------------------


class TransformerForecaster:
    """Lightweight Transformer encoder for time-series return forecasting."""

    def __init__(
        self,
        seq_len: int = 30,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        horizon: int = 5,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 32,
    ) -> None:
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.horizon = horizon
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self._model: Any = None
        self._scaler: Any = None
        self._n_features: int = 0
        self._prediction_history: List[float] = []

    def _build_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        seqs, targets = [], []
        for i in range(self.seq_len, len(X)):
            seqs.append(X[i - self.seq_len : i])
            targets.append(y[i])
        return np.array(seqs), np.array(targets)

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> None:
        """Train a Transformer encoder on feature sequences."""
        try:
            import torch
            import torch.nn as nn
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.warning("torch or sklearn not installed — skipping Transformer fit")
            return

        mask = features.notna().all(axis=1) & target.notna()
        X_raw = features.loc[mask].values.astype(np.float32)
        y_raw = target.loc[mask].values.astype(np.float32)

        if len(X_raw) < self.seq_len + 20:
            logger.info("Not enough data for Transformer (%d rows)", len(X_raw))
            return

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_raw)
        X_seq, y_seq = self._build_sequences(X_scaled, y_raw)
        self._n_features = X_seq.shape[2]

        class _TransformerEncoder(nn.Module):
            def __init__(self_m, input_dim: int, d_model: int, nhead: int, num_layers: int):
                super().__init__()
                self_m.linear_in = nn.Linear(input_dim, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                    dropout=0.1, batch_first=True,
                )
                self_m.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self_m.fc = nn.Linear(d_model, 1)

            def forward(self_m, x: torch.Tensor) -> torch.Tensor:
                x = self_m.linear_in(x)
                x = self_m.encoder(x)
                return self_m.fc(x[:, -1, :]).squeeze(-1)

        model = _TransformerEncoder(self._n_features, self.d_model, self.nhead, self.num_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        X_t = torch.tensor(X_seq, dtype=torch.float32)
        y_t = torch.tensor(y_seq, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.debug("Transformer epoch %d/%d  loss=%.6f", epoch + 1, self.epochs, epoch_loss / len(loader))

        model.eval()
        self._model = model

    def predict(self, features: pd.DataFrame) -> float:
        """Predict aggregate expected return from the latest feature window."""
        if self._model is None or self._scaler is None:
            return 0.0
        try:
            import torch
        except ImportError:
            return 0.0

        X_raw = features.iloc[-self.seq_len :].values.astype(np.float32)
        if len(X_raw) < self.seq_len:
            return 0.0
        X_scaled = self._scaler.transform(X_raw)
        X_t = torch.tensor(X_scaled[np.newaxis, :, :], dtype=torch.float32)
        with torch.no_grad():
            pred = float(self._model(X_t).item())
        self._prediction_history.append(pred)
        return pred

    def save(self, path: Optional[Path] = None) -> None:
        path = path or MODEL_DIR / "transformer"
        path.mkdir(parents=True, exist_ok=True)
        try:
            import torch
            if self._model is not None:
                torch.save(self._model.state_dict(), path / "transformer_state.pt")
            if self._scaler is not None:
                with open(path / "scaler.pkl", "wb") as fh:
                    pickle.dump(self._scaler, fh)
        except ImportError:
            pass

    def load(self, path: Optional[Path] = None) -> None:
        path = path or MODEL_DIR / "transformer"
        logger.info("Transformer load from %s (requires architecture re-init)", path)


# ---------------------------------------------------------------------------
# Unified forecaster facade
# ---------------------------------------------------------------------------


@dataclass
class ForecastResult:
    """Container for a forecaster's output."""

    mu: pd.Series  # expected returns per ticker
    backend: str
    drift_report: Optional[DriftReport] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReturnForecaster:
    """Unified interface over LightGBM / LSTM / Transformer return forecasters.

    Usage
    -----
    >>> fc = ReturnForecaster(backend="lightgbm")
    >>> fc.fit(prices, returns)
    >>> result = fc.predict(prices, returns)
    >>> mu = result.mu  # pd.Series compatible with optimize_weights()
    """

    def __init__(
        self,
        backend: BackendType = "lightgbm",
        horizon: int = 5,
        retrain_every: int = 63,
        drift_threshold: float = 2.0,
        **kwargs: Any,
    ) -> None:
        self.backend_name = backend
        self.horizon = horizon
        self.retrain_every = retrain_every
        self.drift_threshold = drift_threshold
        self._steps_since_train: int = 0
        self._historical_preds: pd.Series = pd.Series(dtype=float)

        if backend == "lightgbm":
            self._engine = LightGBMForecaster(horizon=horizon, **kwargs)
        elif backend == "lstm":
            self._engine = LSTMForecaster(horizon=horizon, **kwargs)
        elif backend == "transformer":
            self._engine = TransformerForecaster(horizon=horizon, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def fit(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> None:
        """Build features and train the underlying model.

        If *features* is supplied it is used directly, avoiding the
        (expensive) call to ``build_universe_features``.
        """
        if features is None:
            features = build_universe_features(prices)
        tickers = ordered_tickers()

        if isinstance(self._engine, LightGBMForecaster):
            self._engine.fit(features, returns, tickers)
        else:
            # LSTM / Transformer: predict aggregate market return
            target = returns[tickers].mean(axis=1).shift(-self.horizon)
            self._engine.fit(features, target)

        self._steps_since_train = 0
        logger.info("Forecaster (%s) trained on %d rows", self.backend_name, len(features))

    def predict(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        fallback_mu: Optional[pd.Series] = None,
        features: Optional[pd.DataFrame] = None,
    ) -> ForecastResult:
        """Generate expected-return estimates and check for drift.

        If *features* is supplied it is used directly, skipping the
        expensive ``build_universe_features`` call.
        """
        if features is None:
            features = build_universe_features(prices)
        tickers = ordered_tickers()

        if isinstance(self._engine, LightGBMForecaster):
            mu = self._engine.predict(features)
        else:
            agg_pred = self._engine.predict(features)
            # Distribute aggregate prediction proportionally using EWM means
            base_mu = exponential_mean(returns[tickers])
            total_base = base_mu.abs().sum()
            if total_base > 0:
                mu = base_mu / total_base * agg_pred
            else:
                mu = pd.Series(agg_pred / len(tickers), index=tickers)

        # Drift detection
        recent = pd.Series(mu.values, index=tickers)
        drift = detect_drift(recent, self._historical_preds, self.drift_threshold)
        self._historical_preds = pd.concat([self._historical_preds, recent]).tail(500)
        self._steps_since_train += 1

        # Fall back to exponential mean if model not trained or drifted
        if mu.isna().all() or (mu == 0).all():
            mu = fallback_mu if fallback_mu is not None else exponential_mean(returns[tickers])

        return ForecastResult(
            mu=mu,
            backend=self.backend_name,
            drift_report=drift,
            metadata={
                "steps_since_train": self._steps_since_train,
                "needs_retrain": drift.is_drifted or self._steps_since_train >= self.retrain_every,
            },
        )

    def rolling_retrain(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> ForecastResult:
        """Retrain if needed, then predict.

        If *features* is supplied it is reused for both the retrain
        and prediction steps, avoiding duplicate feature builds.
        """
        result = self.predict(prices, returns, features=features)
        needs_retrain = result.metadata.get("needs_retrain", False)
        if needs_retrain:
            logger.info("Rolling retrain triggered (drift=%s, steps=%d)", result.drift_report, self._steps_since_train)
            self.fit(prices, returns, features=features)
            result = self.predict(prices, returns, features=features)
        return result

    def save(self, path: Optional[Path] = None) -> None:
        self._engine.save(path)

    def load(self, path: Optional[Path] = None) -> None:
        self._engine.load(path)
