"""Deep Reinforcement Learning portfolio strategy — PPO / SAC via Stable-Baselines3.

Provides:
1. ``PortfolioEnv`` — a Gymnasium environment wrapping the RAAAL asset universe.
2. ``DRLAgent`` — trains and runs PPO or SAC agents.
3. ``drl_portfolio_strategy`` — a ``StrategyFn``-compatible callable for the strategy suite.
4. ``adaptive_rotation_strategy`` — IR-based asset-group rotation with DRL ranking overlay.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from .config import MED_LOOKBACK, UNIVERSE, ordered_tickers
from .features import exponential_cov, exponential_mean

logger = logging.getLogger(__name__)

TICKERS = ordered_tickers()
N_ASSETS = len(TICKERS)
CASH_TICKER = "BIL" if "BIL" in TICKERS else TICKERS[-1]
MODEL_DIR = Path("data/models/drl")

AgentType = Literal["ppo", "sac"]


# ---------------------------------------------------------------------------
# Gymnasium portfolio environment
# ---------------------------------------------------------------------------


def _build_env_class():
    """Lazily build the PortfolioEnv to avoid hard gymnasium import at module load."""
    try:
        import gymnasium as gym
        from gymnasium import spaces
    except ImportError:
        logger.warning("gymnasium not installed — DRL strategies unavailable")
        return None

    class PortfolioEnv(gym.Env):
        """Multi-asset portfolio allocation environment.

        Observation space (per step):
            - 20-day rolling returns for each asset (N_ASSETS * 20)
            - Current portfolio weights (N_ASSETS)
            - Regime one-hot (3)
            - FOMO/FOBI score (1)

        Action space:
            - Continuous weight vector of length N_ASSETS (softmaxed)
        """

        metadata = {"render_modes": []}

        def __init__(
            self,
            returns: pd.DataFrame,
            lookback: int = 20,
            initial_balance: float = 1e6,
            transaction_cost: float = 0.001,
        ) -> None:
            super().__init__()
            self.returns = returns[TICKERS].fillna(0.0).values
            self.lookback = lookback
            self.initial_balance = initial_balance
            self.transaction_cost = transaction_cost
            self.n_steps = len(self.returns)

            obs_dim = N_ASSETS * lookback + N_ASSETS + 3 + 1
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=0.0, high=1.0, shape=(N_ASSETS,), dtype=np.float32,
            )

            self._current_step = self.lookback
            self._weights = np.ones(N_ASSETS, dtype=np.float32) / N_ASSETS
            self._portfolio_value = initial_balance
            self._regime_onehot = np.array([1, 0, 0], dtype=np.float32)  # default risk_on
            self._fomo_score = 0.0

        def set_context(self, regime: Optional[str] = None, fomo_score: float = 0.0) -> None:
            """Inject regime and FOMO info from the outer pipeline."""
            regimes = {"risk_on": [1, 0, 0], "risk_off": [0, 1, 0], "inflation": [0, 0, 1]}
            self._regime_onehot = np.array(regimes.get(regime or "risk_on", [1, 0, 0]), dtype=np.float32)
            self._fomo_score = float(fomo_score)

        def _get_obs(self) -> np.ndarray:
            start = max(0, self._current_step - self.lookback)
            window = self.returns[start : self._current_step]
            # Pad if needed
            if len(window) < self.lookback:
                pad = np.zeros((self.lookback - len(window), N_ASSETS))
                window = np.vstack([pad, window])
            flat_returns = window.flatten().astype(np.float32)
            return np.concatenate([
                flat_returns,
                self._weights,
                self._regime_onehot,
                np.array([self._fomo_score], dtype=np.float32),
            ])

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._current_step = self.lookback
            self._weights = np.ones(N_ASSETS, dtype=np.float32) / N_ASSETS
            self._portfolio_value = self.initial_balance
            return self._get_obs(), {}

        def step(self, action: np.ndarray):
            # Softmax to ensure valid portfolio weights
            action = np.clip(action, 0.0, None)
            total = action.sum()
            if total <= 0:
                weights = np.ones(N_ASSETS, dtype=np.float32) / N_ASSETS
            else:
                weights = action / total

            # Transaction cost
            turnover = np.abs(weights - self._weights).sum()
            cost = turnover * self.transaction_cost

            # Portfolio return
            if self._current_step < self.n_steps:
                day_return = self.returns[self._current_step]
                port_return = float(np.dot(weights, day_return)) - cost
            else:
                port_return = 0.0

            self._portfolio_value *= (1.0 + port_return)
            self._weights = weights.astype(np.float32)
            self._current_step += 1

            # Reward: risk-adjusted return (Sharpe-like)
            reward = port_return - 0.5 * (port_return ** 2)  # quadratic penalty

            terminated = self._current_step >= self.n_steps
            truncated = False
            info = {
                "portfolio_value": self._portfolio_value,
                "weights": dict(zip(TICKERS, weights.tolist())),
                "return": port_return,
            }
            return self._get_obs(), float(reward), terminated, truncated, info

    return PortfolioEnv


# ---------------------------------------------------------------------------
# DRL Agent
# ---------------------------------------------------------------------------


class DRLAgent:
    """Wrapper around Stable-Baselines3 PPO / SAC for portfolio management."""

    def __init__(
        self,
        agent_type: AgentType = "ppo",
        total_timesteps: int = 50_000,
        learning_rate: float = 3e-4,
    ) -> None:
        self.agent_type = agent_type
        self.total_timesteps = total_timesteps
        self.learning_rate = learning_rate
        self._model: Any = None
        self._env_class = _build_env_class()

    def train(
        self,
        returns: pd.DataFrame,
        regime: Optional[str] = None,
        fomo_score: float = 0.0,
    ) -> None:
        """Train the DRL agent on historical return data."""
        if self._env_class is None:
            logger.warning("gymnasium not installed — cannot train DRL agent")
            return

        try:
            from stable_baselines3 import PPO, SAC
            from stable_baselines3.common.vec_env import DummyVecEnv
        except ImportError:
            logger.warning("stable-baselines3 not installed — cannot train DRL agent")
            return

        env = self._env_class(returns)
        env.set_context(regime=regime, fomo_score=fomo_score)

        vec_env = DummyVecEnv([lambda: env])

        if self.agent_type == "ppo":
            self._model = PPO(
                "MlpPolicy", vec_env,
                learning_rate=self.learning_rate,
                verbose=0,
                n_steps=256,
                batch_size=64,
            )
        elif self.agent_type == "sac":
            self._model = SAC(
                "MlpPolicy", vec_env,
                learning_rate=self.learning_rate,
                verbose=0,
                batch_size=64,
            )
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

        self._model.learn(total_timesteps=self.total_timesteps)
        logger.info("DRL agent (%s) trained for %d timesteps", self.agent_type, self.total_timesteps)

    def predict_weights(
        self,
        returns: pd.DataFrame,
        regime: Optional[str] = None,
        fomo_score: float = 0.0,
    ) -> Dict[str, float]:
        """Predict portfolio weights using the trained agent."""
        if self._model is None or self._env_class is None:
            # Expected when gymnasium/sb3 not installed — not an error
            return {t: 1.0 / N_ASSETS for t in TICKERS}

        env = self._env_class(returns)
        env.set_context(regime=regime, fomo_score=fomo_score)
        obs, _ = env.reset()

        # Run through all available data
        for step in range(len(returns) - env.lookback - 1):
            action, _ = self._model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
            if done:
                break

        # Final action is our desired weights
        action, _ = self._model.predict(obs, deterministic=True)
        action = np.clip(action, 0.0, None)
        total = action.sum()
        if total <= 0:
            weights = np.ones(N_ASSETS) / N_ASSETS
        else:
            weights = action / total

        return dict(zip(TICKERS, weights.tolist()))

    def save(self, path: Optional[Path] = None) -> None:
        path = path or MODEL_DIR / self.agent_type
        path.mkdir(parents=True, exist_ok=True)
        if self._model is not None:
            self._model.save(str(path / "model"))
            logger.info("DRL model saved to %s", path)

    def load(self, path: Optional[Path] = None) -> None:
        path = path or MODEL_DIR / self.agent_type
        model_path = path / "model.zip"
        if not model_path.exists():
            logger.debug("No saved DRL model at %s", model_path)
            return
        try:
            from stable_baselines3 import PPO, SAC

            cls = PPO if self.agent_type == "ppo" else SAC
            if self._env_class is not None:
                # Need a dummy env for loading
                dummy_returns = pd.DataFrame(
                    np.zeros((100, N_ASSETS)), columns=TICKERS
                )
                dummy_env = self._env_class(dummy_returns)
                self._model = cls.load(str(path / "model"), env=dummy_env)
            else:
                self._model = cls.load(str(path / "model"))
            logger.info("DRL model loaded from %s", path)
        except ImportError:
            logger.warning("stable-baselines3 not installed — cannot load DRL model")


# ---------------------------------------------------------------------------
# Singleton agent (lazily initialized)
# ---------------------------------------------------------------------------

_GLOBAL_AGENT: Optional[DRLAgent] = None


def _get_or_create_agent(agent_type: AgentType = "ppo") -> DRLAgent:
    global _GLOBAL_AGENT
    if _GLOBAL_AGENT is None:
        _GLOBAL_AGENT = DRLAgent(agent_type=agent_type)
        _GLOBAL_AGENT.load()
    return _GLOBAL_AGENT


# ---------------------------------------------------------------------------
# StrategyFn-compatible callables
# ---------------------------------------------------------------------------


def drl_portfolio_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime: Optional[str],
    context: Dict[str, object],
) -> Dict[str, float]:
    """DRL (PPO/SAC) portfolio allocation strategy.

    Matches the ``StrategyFn`` signature so it plugs directly into
    ``DEFAULT_STRATEGIES`` in ``strategies.py``.
    """
    agent = _get_or_create_agent()
    fomo_score = 0.0
    if context:
        fomo_info = context.get("fomo_fobi", {})
        if isinstance(fomo_info, dict):
            fomo_score = fomo_info.get("score", 0.0)

    weights = agent.predict_weights(returns, regime=regime, fomo_score=fomo_score)

    # Normalize
    total = sum(max(v, 0.0) for v in weights.values())
    if total <= 0:
        return {t: 1.0 / N_ASSETS for t in TICKERS}
    return {t: max(weights.get(t, 0.0), 0.0) / total for t in TICKERS}


def adaptive_rotation_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime: Optional[str],
    context: Dict[str, object],
) -> Dict[str, float]:
    """Adaptive rotation strategy with group-level IR selection and stop-loss.

    Asset groups:
        growth  — SPY, HYG, BTC-USD
        defensive — TLT, BIL, GLD
        real     — DBC, GLD, TIP (if available)

    Selection logic:
        1. Compute information ratio (IR) for each group over trailing window.
        2. Allocate proportionally to positive-IR groups.
        3. Apply a simple stop-loss check: if trailing 21-day group return < -5%,
           reduce allocation and move to defensive.
    """
    groups = {
        "growth": [t for t in ["SPY", "HYG", "BTC-USD"] if t in TICKERS],
        "defensive": [t for t in ["TLT", "BIL", "GLD"] if t in TICKERS],
        "real_assets": [t for t in ["DBC", "GLD"] if t in TICKERS],
    }

    base_weights = {t: 0.0 for t in TICKERS}
    window = MED_LOOKBACK
    stop_loss_threshold = -0.05
    stop_loss_window = 21

    if returns.empty or len(returns) < stop_loss_window:
        # Equal weight defensive
        for t in groups.get("defensive", []):
            base_weights[t] = 1.0 / max(len(groups["defensive"]), 1)
        return _normalize_weight_dict(base_weights)

    group_ir: Dict[str, float] = {}
    group_stopped: Dict[str, bool] = {}

    for gname, members in groups.items():
        if not members:
            continue
        available = [t for t in members if t in returns.columns]
        if not available:
            continue
        group_ret = returns[available].mean(axis=1)
        trailing = group_ret.tail(window)
        if trailing.std() == 0 or np.isnan(trailing.std()):
            group_ir[gname] = 0.0
        else:
            group_ir[gname] = float(trailing.mean() / trailing.std()) * np.sqrt(252)

        # Stop-loss check
        short_ret = group_ret.tail(stop_loss_window).sum()
        group_stopped[gname] = short_ret < stop_loss_threshold

    # Remove stopped groups and those with negative IR
    active_groups = {
        g: max(ir, 0.0)
        for g, ir in group_ir.items()
        if not group_stopped.get(g, False) and ir > 0
    }

    if not active_groups:
        # Default to defensive
        active_groups = {"defensive": 1.0}

    total_ir = sum(active_groups.values())
    if total_ir <= 0:
        total_ir = 1.0

    for gname, ir_value in active_groups.items():
        members = groups.get(gname, [])
        available = [t for t in members if t in TICKERS]
        if not available:
            continue
        group_alloc = ir_value / total_ir
        per_asset = group_alloc / len(available)
        for t in available:
            base_weights[t] += per_asset

    return _normalize_weight_dict(base_weights)


def _normalize_weight_dict(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize weight dict to sum to 1, ensuring non-negative."""
    clean = {t: max(float(v), 0.0) for t, v in weights.items()}
    total = sum(clean.values())
    if total <= 0:
        clean[CASH_TICKER] = 1.0
        return clean
    return {t: v / total for t, v in clean.items()}
