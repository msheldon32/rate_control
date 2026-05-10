"""
Learning agents for two-sided pricing queue.

Baseline agents:
1. RandomAgent - random prices
2. GreedyAgent - myopic optimization
3. TabularPolicyGradient - REINFORCE with monotonic projection
4. TabularActorCritic - A2C with monotonic projection

Price bounds:
- Customer prices in [-1, 1]
- Server prices in [-1, 0) — servers are always paid
"""

import numpy as np
from typing import Tuple, Optional, List
from abc import ABC, abstractmethod
import random

from model import ModelConfig, TwoSidedQueue
from policy import (
    TabularMonotonicPolicy, LinearMonotonicPolicy,
    CUSTOMER_PRICE_MIN, CUSTOMER_PRICE_MAX, SERVER_PRICE_MIN, SERVER_PRICE_MAX
)


class Agent(ABC):
    """Base class for learning agents."""

    def __init__(self, config: ModelConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.K = config.n_customer_classes
        self.L = config.n_server_classes

    @abstractmethod
    def get_prices(self, n: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get prices for current state."""
        pass

    @abstractmethod
    def observe(self, n: np.ndarray, m: np.ndarray,
                prices_c: np.ndarray, prices_s: np.ndarray,
                reward: float,
                new_n: np.ndarray, new_m: np.ndarray,
                sojourn_time: float, event_type: str):
        """Observe transition and reward."""
        pass


class RandomAgent(Agent):
    """Agent that outputs random prices."""

    def get_prices(self, n: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        prices_c = self.rng.uniform(CUSTOMER_PRICE_MIN, CUSTOMER_PRICE_MAX, size=self.K)
        prices_s = self.rng.uniform(SERVER_PRICE_MIN, SERVER_PRICE_MAX, size=self.L)
        return prices_c, prices_s

    def observe(self, n, m, prices_c, prices_s, reward, new_n, new_m, sojourn_time, event_type):
        pass  # No learning


class ConstantAgent(Agent):
    """Agent with fixed constant prices."""

    def __init__(self, config: ModelConfig, rng: np.random.Generator,
                 price_c: float = 0.0, price_s: float = -0.5):
        super().__init__(config, rng)
        self.price_c = price_c
        self.price_s = price_s

    def get_prices(self, n: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return np.full(self.K, self.price_c), np.full(self.L, self.price_s)

    def observe(self, n, m, prices_c, prices_s, reward, new_n, new_m, sojourn_time, event_type):
        pass


class TabularPolicyGradientAgent(Agent):
    """
    Tabular policy gradient (REINFORCE) with monotonic projection.

    Parameterizes price as softmax over discrete price levels,
    then projects to monotonic after updates.
    """

    def __init__(self, config: ModelConfig, rng: np.random.Generator,
                 n_price_levels: int = 11,
                 learning_rate: float = 0.01,
                 baseline_lr: float = 0.1,
                 entropy_coef: float = 0.01):
        super().__init__(config, rng)

        self.n_price_levels = n_price_levels
        self.price_values = np.linspace(-1, 1, n_price_levels)
        self.lr = learning_rate
        self.baseline_lr = baseline_lr
        self.entropy_coef = entropy_coef

        # Logits for each (class, count) -> price level
        # Customer logits: K arrays, each of shape (capacity+1, n_price_levels)
        self.customer_logits = [
            np.zeros((config.customer_capacities[i] + 1, n_price_levels))
            for i in range(self.K)
        ]
        # Server logits: L arrays, each of shape (capacity+1, n_price_levels)
        self.server_logits = [
            np.zeros((config.server_capacities[j] + 1, n_price_levels))
            for j in range(self.L)
        ]

        # Initialize with bias toward monotonic (higher price at higher count)
        for i in range(self.K):
            for count in range(config.customer_capacities[i] + 1):
                # Bias toward price level proportional to count
                target_level = int(count / config.customer_capacities[i] * (n_price_levels - 1))
                self.customer_logits[i][count, target_level] = 1.0

        for j in range(self.L):
            for count in range(config.server_capacities[j] + 1):
                target_level = int(count / config.server_capacities[j] * (n_price_levels - 1))
                self.server_logits[j][count, target_level] = 1.0

        # Baseline (average reward)
        self.baseline = 0.0

        # Episode buffer for REINFORCE
        self.episode_buffer = []

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    def _sample_price(self, logits: np.ndarray) -> Tuple[int, float]:
        """Sample a price level from logits."""
        probs = self._softmax(logits)
        level = self.rng.choice(self.n_price_levels, p=probs)
        return level, self.price_values[level]

    def get_prices(self, n: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        prices_c = np.zeros(self.K)
        prices_s = np.zeros(self.L)
        levels_c = np.zeros(self.K, dtype=int)
        levels_s = np.zeros(self.L, dtype=int)

        for i in range(self.K):
            levels_c[i], prices_c[i] = self._sample_price(self.customer_logits[i][n[i]])

        for j in range(self.L):
            levels_s[j], prices_s[j] = self._sample_price(self.server_logits[j][m[j]])

        # Store for gradient computation
        self._last_n = n.copy()
        self._last_m = m.copy()
        self._last_levels_c = levels_c
        self._last_levels_s = levels_s

        return prices_c, prices_s

    def observe(self, n, m, prices_c, prices_s, reward, new_n, new_m, sojourn_time, event_type):
        # Store transition
        self.episode_buffer.append({
            'n': self._last_n,
            'm': self._last_m,
            'levels_c': self._last_levels_c,
            'levels_s': self._last_levels_s,
            'reward': reward,
            'sojourn_time': sojourn_time
        })

    def update(self):
        """Update policy using REINFORCE with baseline."""
        if len(self.episode_buffer) == 0:
            return

        # Compute returns (using reward rate as the signal)
        total_reward = sum(t['reward'] for t in self.episode_buffer)
        total_time = sum(t['sojourn_time'] for t in self.episode_buffer)
        avg_reward_rate = total_reward / total_time if total_time > 0 else 0.0

        # Update baseline
        self.baseline = (1 - self.baseline_lr) * self.baseline + self.baseline_lr * avg_reward_rate
        advantage = avg_reward_rate - self.baseline

        # Policy gradient update
        for transition in self.episode_buffer:
            n = transition['n']
            m = transition['m']
            levels_c = transition['levels_c']
            levels_s = transition['levels_s']

            # Customer updates
            for i in range(self.K):
                logits = self.customer_logits[i][n[i]]
                probs = self._softmax(logits)

                # Gradient of log probability
                grad = -probs.copy()
                grad[levels_c[i]] += 1.0

                # Entropy bonus gradient
                entropy_grad = -probs * (np.log(probs + 1e-10) + 1)

                # Update
                self.customer_logits[i][n[i]] += self.lr * (
                    advantage * grad + self.entropy_coef * entropy_grad
                )

            # Server updates
            for j in range(self.L):
                logits = self.server_logits[j][m[j]]
                probs = self._softmax(logits)

                grad = -probs.copy()
                grad[levels_s[j]] += 1.0

                entropy_grad = -probs * (np.log(probs + 1e-10) + 1)

                self.server_logits[j][m[j]] += self.lr * (
                    advantage * grad + self.entropy_coef * entropy_grad
                )

        # Clear buffer
        self.episode_buffer = []

        # Project to monotonic
        self._project_to_monotonic()

    def _project_to_monotonic(self):
        """
        Project logits so that expected price is monotonic in count.
        Uses isotonic regression on expected prices, then adjusts logits.
        """
        for i in range(self.K):
            cap = self.config.customer_capacities[i]
            expected_prices = np.array([
                np.sum(self._softmax(self.customer_logits[i][c]) * self.price_values)
                for c in range(cap + 1)
            ])

            # Isotonic regression
            monotonic_prices = self._isotonic_regression(expected_prices)

            # Adjust logits to match monotonic expected prices
            for c in range(cap + 1):
                if expected_prices[c] != monotonic_prices[c]:
                    # Shift logits to move expected price toward monotonic target
                    target = monotonic_prices[c]
                    # Simple approach: bias toward the target price level
                    target_level = np.argmin(np.abs(self.price_values - target))
                    self.customer_logits[i][c, target_level] += 0.5

        for j in range(self.L):
            cap = self.config.server_capacities[j]
            expected_prices = np.array([
                np.sum(self._softmax(self.server_logits[j][c]) * self.price_values)
                for c in range(cap + 1)
            ])

            monotonic_prices = self._isotonic_regression(expected_prices)

            for c in range(cap + 1):
                if expected_prices[c] != monotonic_prices[c]:
                    target = monotonic_prices[c]
                    target_level = np.argmin(np.abs(self.price_values - target))
                    self.server_logits[j][c, target_level] += 0.5

    @staticmethod
    def _isotonic_regression(y: np.ndarray) -> np.ndarray:
        """Pool Adjacent Violators Algorithm."""
        n = len(y)
        result = y.copy()
        blocks = [[i] for i in range(n)]

        while True:
            merged = False
            i = 0
            while i < len(blocks) - 1:
                mean_i = np.mean([result[j] for j in blocks[i]])
                mean_next = np.mean([result[j] for j in blocks[i + 1]])

                if mean_i > mean_next:
                    blocks[i] = blocks[i] + blocks[i + 1]
                    blocks.pop(i + 1)
                    new_mean = np.mean([result[j] for j in blocks[i]])
                    for j in blocks[i]:
                        result[j] = new_mean
                    merged = True
                else:
                    i += 1

            if not merged:
                break

        return result

    def get_greedy_policy(self) -> TabularMonotonicPolicy:
        """Extract greedy policy from current logits."""
        policy = TabularMonotonicPolicy(self.config)

        for i in range(self.K):
            for c in range(self.config.customer_capacities[i] + 1):
                probs = self._softmax(self.customer_logits[i][c])
                expected_price = np.sum(probs * self.price_values)
                policy.customer_prices[i][c] = expected_price

        for j in range(self.L):
            for c in range(self.config.server_capacities[j] + 1):
                probs = self._softmax(self.server_logits[j][c])
                expected_price = np.sum(probs * self.price_values)
                policy.server_prices[j][c] = expected_price

        policy.project_to_monotonic()
        return policy


class LinearPolicyGradientAgent(Agent):
    """
    Policy gradient with linear monotonic policy.

    Parameterizes:
        p_i^c(n) = clip(a_i * n_i / cap_i + b_i, -1, 1)
        p_j^s(m) = clip(c_j * m_j / cap_j + d_j, -1, 0)  (servers always paid)

    With a_i >= 0, c_j >= 0 enforced for monotonicity.
    """

    def __init__(self, config: ModelConfig, rng: np.random.Generator,
                 learning_rate: float = 0.01,
                 noise_std: float = 0.1,
                 baseline_lr: float = 0.1):
        super().__init__(config, rng)

        self.lr = learning_rate
        self.noise_std = noise_std
        self.baseline_lr = baseline_lr

        # Parameters: slopes (a) and intercepts (b)
        # a >= 0 for monotonicity
        self.customer_slopes = np.ones(self.K) * 2.0  # Default: -1 to 1 over capacity
        self.customer_intercepts = np.ones(self.K) * -1.0
        # Server: -1 to ~0 over capacity (always negative)
        self.server_slopes = np.ones(self.L) * 1.0
        self.server_intercepts = np.ones(self.L) * -1.0

        self.baseline = 0.0
        self.episode_buffer = []

    def get_prices(self, n: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Add exploration noise
        noise_c = self.rng.normal(0, self.noise_std, size=self.K)
        noise_s = self.rng.normal(0, self.noise_std, size=self.L)

        prices_c = np.zeros(self.K)
        for i in range(self.K):
            normalized = n[i] / max(self.config.customer_capacities[i], 1)
            prices_c[i] = np.clip(
                self.customer_slopes[i] * normalized + self.customer_intercepts[i] + noise_c[i],
                CUSTOMER_PRICE_MIN, CUSTOMER_PRICE_MAX
            )

        prices_s = np.zeros(self.L)
        for j in range(self.L):
            normalized = m[j] / max(self.config.server_capacities[j], 1)
            prices_s[j] = np.clip(
                self.server_slopes[j] * normalized + self.server_intercepts[j] + noise_s[j],
                SERVER_PRICE_MIN, SERVER_PRICE_MAX
            )

        # Store for gradient
        self._last_n = n.copy()
        self._last_m = m.copy()
        self._last_noise_c = noise_c
        self._last_noise_s = noise_s

        return prices_c, prices_s

    def observe(self, n, m, prices_c, prices_s, reward, new_n, new_m, sojourn_time, event_type):
        self.episode_buffer.append({
            'n': self._last_n,
            'm': self._last_m,
            'noise_c': self._last_noise_c,
            'noise_s': self._last_noise_s,
            'reward': reward,
            'sojourn_time': sojourn_time
        })

    def update(self):
        """Update using parameter-space policy gradient."""
        if len(self.episode_buffer) == 0:
            return

        total_reward = sum(t['reward'] for t in self.episode_buffer)
        total_time = sum(t['sojourn_time'] for t in self.episode_buffer)
        avg_reward_rate = total_reward / total_time if total_time > 0 else 0.0

        self.baseline = (1 - self.baseline_lr) * self.baseline + self.baseline_lr * avg_reward_rate
        advantage = avg_reward_rate - self.baseline

        # Gradient estimates (using noise as exploration direction)
        grad_slopes_c = np.zeros(self.K)
        grad_intercepts_c = np.zeros(self.K)
        grad_slopes_s = np.zeros(self.L)
        grad_intercepts_s = np.zeros(self.L)

        for transition in self.episode_buffer:
            n = transition['n']
            m = transition['m']
            noise_c = transition['noise_c']
            noise_s = transition['noise_s']

            # Gradient w.r.t. slope: d(price)/d(slope) = n/cap
            for i in range(self.K):
                normalized = n[i] / max(self.config.customer_capacities[i], 1)
                grad_slopes_c[i] += noise_c[i] * normalized
                grad_intercepts_c[i] += noise_c[i]

            for j in range(self.L):
                normalized = m[j] / max(self.config.server_capacities[j], 1)
                grad_slopes_s[j] += noise_s[j] * normalized
                grad_intercepts_s[j] += noise_s[j]

        # Normalize
        n_transitions = len(self.episode_buffer)
        grad_slopes_c /= n_transitions * self.noise_std**2
        grad_intercepts_c /= n_transitions * self.noise_std**2
        grad_slopes_s /= n_transitions * self.noise_std**2
        grad_intercepts_s /= n_transitions * self.noise_std**2

        # Update
        self.customer_slopes += self.lr * advantage * grad_slopes_c
        self.customer_intercepts += self.lr * advantage * grad_intercepts_c
        self.server_slopes += self.lr * advantage * grad_slopes_s
        self.server_intercepts += self.lr * advantage * grad_intercepts_s

        # Project slopes to non-negative for monotonicity
        self.customer_slopes = np.maximum(self.customer_slopes, 0.0)
        self.server_slopes = np.maximum(self.server_slopes, 0.0)

        # Clip intercepts to keep prices in valid range
        self.customer_intercepts = np.clip(self.customer_intercepts, CUSTOMER_PRICE_MIN, CUSTOMER_PRICE_MAX)
        # Server intercepts must ensure prices stay negative even at max slope
        self.server_intercepts = np.clip(self.server_intercepts, SERVER_PRICE_MIN, SERVER_PRICE_MAX)

        self.episode_buffer = []

    def get_policy(self) -> LinearMonotonicPolicy:
        """Extract current policy."""
        return LinearMonotonicPolicy(
            self.config,
            customer_slopes=self.customer_slopes.copy(),
            customer_intercepts=self.customer_intercepts.copy(),
            server_slopes=self.server_slopes.copy(),
            server_intercepts=self.server_intercepts.copy()
        )


# ============================================================================
# Deep Lattice Network Agents
# ============================================================================

try:
    import torch
    import torch.optim as optim
    from networks import MonotonicPricingNetwork, MLPPricingNetwork
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GradientMonitor:
    """
    Mixin for monitoring weights and gradients in neural network agents.

    Tracks:
    - Weight statistics per layer (mean, std, min, max, norm)
    - Gradient statistics per layer (mean, std, min, max, norm)
    - Total gradient norm
    - Loss values
    """

    def _init_monitoring(self):
        """Initialize monitoring data structures."""
        self.monitoring = {
            'loss': [],
            'grad_norm': [],
            'weight_stats': [],  # List of dicts per update
            'grad_stats': [],    # List of dicts per update
            'update_count': 0
        }

    def _compute_weight_stats(self, network: 'torch.nn.Module') -> dict:
        """Compute statistics for all network weights."""
        stats = {}
        total_norm = 0.0

        for name, param in network.named_parameters():
            if param.requires_grad:
                data = param.data.cpu().numpy()
                stats[name] = {
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'norm': float(np.linalg.norm(data))
                }
                total_norm += stats[name]['norm'] ** 2

        stats['total_norm'] = float(np.sqrt(total_norm))
        return stats

    def _compute_grad_stats(self, network: 'torch.nn.Module') -> dict:
        """Compute statistics for all gradients (call after backward, before step)."""
        stats = {}
        total_norm = 0.0

        for name, param in network.named_parameters():
            if param.grad is not None:
                grad = param.grad.cpu().numpy()
                stats[name] = {
                    'mean': float(np.mean(grad)),
                    'std': float(np.std(grad)),
                    'min': float(np.min(grad)),
                    'max': float(np.max(grad)),
                    'norm': float(np.linalg.norm(grad))
                }
                total_norm += stats[name]['norm'] ** 2

        stats['total_norm'] = float(np.sqrt(total_norm))
        return stats

    def _record_update(self, network: 'torch.nn.Module', loss: float):
        """Record monitoring info for an update step."""
        self.monitoring['loss'].append(loss)
        self.monitoring['grad_norm'].append(
            self.monitoring['grad_stats'][-1]['total_norm']
            if self.monitoring['grad_stats'] else 0.0
        )
        self.monitoring['update_count'] += 1

    def get_monitoring_summary(self) -> dict:
        """Get summary of monitoring statistics."""
        if self.monitoring['update_count'] == 0:
            return {'updates': 0}

        losses = np.array(self.monitoring['loss'])
        grad_norms = np.array(self.monitoring['grad_norm'])

        summary = {
            'updates': self.monitoring['update_count'],
            'loss': {
                'mean': float(np.mean(losses)),
                'std': float(np.std(losses)),
                'min': float(np.min(losses)),
                'max': float(np.max(losses)),
                'recent': float(losses[-1]) if len(losses) > 0 else 0.0
            },
            'grad_norm': {
                'mean': float(np.mean(grad_norms)),
                'std': float(np.std(grad_norms)),
                'min': float(np.min(grad_norms)),
                'max': float(np.max(grad_norms)),
                'recent': float(grad_norms[-1]) if len(grad_norms) > 0 else 0.0
            }
        }

        # Add per-layer weight norms (most recent)
        if self.monitoring['weight_stats']:
            recent_weights = self.monitoring['weight_stats'][-1]
            summary['weight_norms'] = {
                name: stats['norm']
                for name, stats in recent_weights.items()
                if name != 'total_norm'
            }
            summary['total_weight_norm'] = recent_weights.get('total_norm', 0.0)

        return summary

    def print_monitoring_summary(self):
        """Print formatted monitoring summary."""
        summary = self.get_monitoring_summary()

        if summary['updates'] == 0:
            print("  No updates recorded yet")
            return

        print(f"  Updates: {summary['updates']}")
        print(f"  Loss - mean: {summary['loss']['mean']:.6f}, "
              f"std: {summary['loss']['std']:.6f}, "
              f"recent: {summary['loss']['recent']:.6f}")
        print(f"  Grad norm - mean: {summary['grad_norm']['mean']:.6f}, "
              f"std: {summary['grad_norm']['std']:.6f}, "
              f"recent: {summary['grad_norm']['recent']:.6f}")

        if 'weight_norms' in summary:
            print(f"  Total weight norm: {summary['total_weight_norm']:.4f}")
            # Print top 5 layers by weight norm
            sorted_layers = sorted(
                summary['weight_norms'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            print("  Top layers by weight norm:")
            for name, norm in sorted_layers:
                print(f"    {name}: {norm:.4f}")

    def check_gradient_health(self) -> dict:
        """Check for common gradient issues."""
        if self.monitoring['update_count'] < 10:
            return {'status': 'insufficient_data'}

        grad_norms = np.array(self.monitoring['grad_norm'][-100:])  # Last 100

        issues = []

        # Check for vanishing gradients
        if np.mean(grad_norms) < 1e-7:
            issues.append('vanishing_gradients')

        # Check for exploding gradients
        if np.mean(grad_norms) > 100:
            issues.append('exploding_gradients')

        # Check for high variance
        if np.std(grad_norms) > 10 * np.mean(grad_norms) and np.mean(grad_norms) > 1e-6:
            issues.append('high_gradient_variance')

        # Check if gradients are decreasing (might indicate dying network)
        if len(grad_norms) >= 50:
            first_half = np.mean(grad_norms[:len(grad_norms)//2])
            second_half = np.mean(grad_norms[len(grad_norms)//2:])
            if second_half < 0.1 * first_half and first_half > 1e-6:
                issues.append('declining_gradients')

        return {
            'status': 'healthy' if len(issues) == 0 else 'issues_detected',
            'issues': issues,
            'mean_grad_norm': float(np.mean(grad_norms)),
            'std_grad_norm': float(np.std(grad_norms))
        }


class DeepLatticeAgent(Agent, GradientMonitor):
    """
    Deep Lattice Network agent with built-in monotonicity.

    Uses MonotonicPricingNetwork which enforces monotonicity by construction
    via calibrators and lattice layers.
    """

    def __init__(self, config: ModelConfig, rng: np.random.Generator,
                 learning_rate: float = 1e-3,
                 n_calibrator_keypoints: int = 10,
                 lattice_size: int = 3,
                 use_cross_effects: bool = False,
                 exploration_noise: float = 0.1,
                 batch_size: int = 32,
                 update_every: int = 100,
                 entropy_coef: float = 0.01,
                 monitor_every: int = 10):
        """
        Args:
            config: model configuration
            rng: numpy random generator
            learning_rate: optimizer learning rate
            n_calibrator_keypoints: keypoints per calibrator
            lattice_size: vertices per lattice dimension
            use_cross_effects: if True, prices depend on compatible counts
            exploration_noise: std of Gaussian exploration noise
            batch_size: minibatch size for updates
            update_every: update network every N observations
            entropy_coef: coefficient for entropy bonus (encourages exploration)
            monitor_every: record detailed stats every N updates
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for DeepLatticeAgent")

        super().__init__(config, rng)

        self.lr = learning_rate
        self.exploration_noise = exploration_noise
        self.batch_size = batch_size
        self.update_every = update_every
        self.entropy_coef = entropy_coef
        self.monitor_every = monitor_every

        # Create network
        self.network = MonotonicPricingNetwork(
            n_customer_classes=self.K,
            n_server_classes=self.L,
            customer_capacities=config.customer_capacities,
            server_capacities=config.server_capacities,
            n_calibrator_keypoints=n_calibrator_keypoints,
            lattice_size=lattice_size,
            use_cross_effects=use_cross_effects,
            compatibility=config.compatibility
        )

        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # Experience buffer
        self.buffer = []
        self.step_count = 0

        # Baseline for variance reduction
        self.baseline = 0.0
        self.baseline_lr = 0.1

        # Initialize monitoring
        self._init_monitoring()

    def get_prices(self, n: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get prices with exploration noise."""
        with torch.no_grad():
            n_t = torch.tensor(n, dtype=torch.float32).unsqueeze(0)
            m_t = torch.tensor(m, dtype=torch.float32).unsqueeze(0)

            prices_c, prices_s = self.network(n_t, m_t)

            prices_c = prices_c.squeeze(0).numpy()
            prices_s = prices_s.squeeze(0).numpy()

        # Add exploration noise
        noise_c = self.rng.normal(0, self.exploration_noise, size=self.K)
        noise_s = self.rng.normal(0, self.exploration_noise, size=self.L)

        prices_c = np.clip(prices_c + noise_c, CUSTOMER_PRICE_MIN, CUSTOMER_PRICE_MAX)
        prices_s = np.clip(prices_s + noise_s, SERVER_PRICE_MIN, SERVER_PRICE_MAX)

        # Store for gradient computation
        self._last_n = n.copy()
        self._last_m = m.copy()
        self._last_noise_c = noise_c
        self._last_noise_s = noise_s

        return prices_c, prices_s

    def observe(self, n, m, prices_c, prices_s, reward, new_n, new_m, sojourn_time, event_type):
        """Store transition in buffer."""
        self.buffer.append({
            'n': self._last_n.copy(),
            'm': self._last_m.copy(),
            'prices_c': prices_c.copy(),
            'prices_s': prices_s.copy(),
            'noise_c': self._last_noise_c.copy(),
            'noise_s': self._last_noise_s.copy(),
            'reward': reward,
            'sojourn_time': sojourn_time
        })

        self.step_count += 1

        # Periodic updates
        if self.step_count % self.update_every == 0:
            self.update()

    def update(self):
        """Update network using policy gradient."""
        if len(self.buffer) < self.batch_size:
            return

        # Compute average reward rate for the buffer
        total_reward = sum(t['reward'] for t in self.buffer)
        total_time = sum(t['sojourn_time'] for t in self.buffer)
        avg_reward_rate = total_reward / total_time if total_time > 0 else 0.0

        # Update baseline
        self.baseline = (1 - self.baseline_lr) * self.baseline + self.baseline_lr * avg_reward_rate
        advantage = avg_reward_rate - self.baseline

        # Sample minibatch
        if len(self.buffer) > self.batch_size:
            indices = self.rng.choice(len(self.buffer), size=self.batch_size, replace=False)
            batch = [self.buffer[i] for i in indices]
        else:
            batch = self.buffer

        # Prepare tensors
        n_batch = torch.tensor(np.array([t['n'] for t in batch]), dtype=torch.float32)
        m_batch = torch.tensor(np.array([t['m'] for t in batch]), dtype=torch.float32)
        noise_c_batch = torch.tensor(np.array([t['noise_c'] for t in batch]), dtype=torch.float32)
        noise_s_batch = torch.tensor(np.array([t['noise_s'] for t in batch]), dtype=torch.float32)

        # Forward pass
        self.optimizer.zero_grad()
        prices_c, prices_s = self.network(n_batch, m_batch)


        # Policy gradient loss (parameter-space gradient using noise)
        # The idea: noise acts as exploration, and we credit noise directions
        # that led to higher rewards
        # Loss = -advantage * (noise · d(price)/d(params))
        # Since price = f(params) + noise, d(price)/d(params) = d(f)/d(params)
        # We approximate by using the noise as a pseudo-gradient

        # Simpler approach: maximize reward by minimizing negative reward-weighted log-likelihood
        # Here we use a regression-like loss: push prices toward (mean_price + advantage * noise)
        target_c = prices_c.detach() + advantage * noise_c_batch * 0.1
        target_s = prices_s.detach() + advantage * noise_s_batch * 0.1

        target_c = torch.clamp(target_c, CUSTOMER_PRICE_MIN, CUSTOMER_PRICE_MAX)
        target_s = torch.clamp(target_s, SERVER_PRICE_MIN, SERVER_PRICE_MAX)

        loss = ((prices_c - target_c) ** 2).mean() + ((prices_s - target_s) ** 2).mean()

        loss.backward()

        # Record gradient stats before optimizer step
        should_record = (self.monitoring['update_count'] % self.monitor_every == 0)
        if should_record:
            self.monitoring['grad_stats'].append(self._compute_grad_stats(self.network))

        self.optimizer.step()

        # Record weight stats and loss after optimizer step
        if should_record:
            self.monitoring['weight_stats'].append(self._compute_weight_stats(self.network))

        self._record_update(self.network, float(loss.item()))

        # Clear buffer
        self.buffer = []

    def get_deterministic_prices(self, n: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get prices without exploration noise (for evaluation)."""
        with torch.no_grad():
            n_t = torch.tensor(n, dtype=torch.float32).unsqueeze(0)
            m_t = torch.tensor(m, dtype=torch.float32).unsqueeze(0)

            prices_c, prices_s = self.network(n_t, m_t)

            return prices_c.squeeze(0).numpy(), prices_s.squeeze(0).numpy()


class MLPAgent(Agent, GradientMonitor):
    """
    MLP agent baseline (no monotonicity guarantees).

    For comparison with monotonic networks.
    """

    def __init__(self, config: ModelConfig, rng: np.random.Generator,
                 learning_rate: float = 1e-3,
                 hidden_dims: List[int] = [64, 64],
                 exploration_noise: float = 0.1,
                 batch_size: int = 32,
                 update_every: int = 100,
                 monitor_every: int = 10):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for MLPAgent")

        super().__init__(config, rng)

        self.lr = learning_rate
        self.exploration_noise = exploration_noise
        self.batch_size = batch_size
        self.update_every = update_every
        self.monitor_every = monitor_every

        self.network = MLPPricingNetwork(
            n_customer_classes=self.K,
            n_server_classes=self.L,
            hidden_dims=hidden_dims
        )

        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        self.buffer = []
        self.step_count = 0
        self.baseline = 0.0
        self.baseline_lr = 0.1

        # Initialize monitoring
        self._init_monitoring()

    def get_prices(self, n: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            n_t = torch.tensor(n, dtype=torch.float32).unsqueeze(0)
            m_t = torch.tensor(m, dtype=torch.float32).unsqueeze(0)

            prices_c, prices_s = self.network(n_t, m_t)

            prices_c = prices_c.squeeze(0).numpy()
            prices_s = prices_s.squeeze(0).numpy()

        noise_c = self.rng.normal(0, self.exploration_noise, size=self.K)
        noise_s = self.rng.normal(0, self.exploration_noise, size=self.L)

        prices_c = np.clip(prices_c + noise_c, CUSTOMER_PRICE_MIN, CUSTOMER_PRICE_MAX)
        prices_s = np.clip(prices_s + noise_s, SERVER_PRICE_MIN, SERVER_PRICE_MAX)

        self._last_n = n.copy()
        self._last_m = m.copy()
        self._last_noise_c = noise_c
        self._last_noise_s = noise_s

        return prices_c, prices_s

    def observe(self, n, m, prices_c, prices_s, reward, new_n, new_m, sojourn_time, event_type):
        self.buffer.append({
            'n': self._last_n.copy(),
            'm': self._last_m.copy(),
            'noise_c': self._last_noise_c.copy(),
            'noise_s': self._last_noise_s.copy(),
            'reward': reward,
            'sojourn_time': sojourn_time
        })

        self.step_count += 1

        if self.step_count % self.update_every == 0:
            self.update()

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        total_reward = sum(t['reward'] for t in self.buffer)
        total_time = sum(t['sojourn_time'] for t in self.buffer)
        avg_reward_rate = total_reward / total_time if total_time > 0 else 0.0

        self.baseline = (1 - self.baseline_lr) * self.baseline + self.baseline_lr * avg_reward_rate
        advantage = avg_reward_rate - self.baseline

        if len(self.buffer) > self.batch_size:
            indices = self.rng.choice(len(self.buffer), size=self.batch_size, replace=False)
            batch = [self.buffer[i] for i in indices]
        else:
            batch = self.buffer

        n_batch = torch.tensor(np.array([t['n'] for t in batch]), dtype=torch.float32)
        m_batch = torch.tensor(np.array([t['m'] for t in batch]), dtype=torch.float32)
        noise_c_batch = torch.tensor(np.array([t['noise_c'] for t in batch]), dtype=torch.float32)
        noise_s_batch = torch.tensor(np.array([t['noise_s'] for t in batch]), dtype=torch.float32)

        self.optimizer.zero_grad()
        prices_c, prices_s = self.network(n_batch, m_batch)

        target_c = prices_c.detach() + advantage * noise_c_batch * 0.1
        target_s = prices_s.detach() + advantage * noise_s_batch * 0.1

        target_c = torch.clamp(target_c, CUSTOMER_PRICE_MIN, CUSTOMER_PRICE_MAX)
        target_s = torch.clamp(target_s, SERVER_PRICE_MIN, SERVER_PRICE_MAX)

        loss = ((prices_c - target_c) ** 2).mean() + ((prices_s - target_s) ** 2).mean()

        loss.backward()

        # Record gradient stats before optimizer step
        should_record = (self.monitoring['update_count'] % self.monitor_every == 0)
        if should_record:
            self.monitoring['grad_stats'].append(self._compute_grad_stats(self.network))

        self.optimizer.step()

        # Record weight stats and loss after optimizer step
        if should_record:
            self.monitoring['weight_stats'].append(self._compute_weight_stats(self.network))

        self._record_update(self.network, float(loss.item()))

        self.buffer = []

    def get_deterministic_prices(self, n: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            n_t = torch.tensor(n, dtype=torch.float32).unsqueeze(0)
            m_t = torch.tensor(m, dtype=torch.float32).unsqueeze(0)

            prices_c, prices_s = self.network(n_t, m_t)

            return prices_c.squeeze(0).numpy(), prices_s.squeeze(0).numpy()


# ============================================================================
# LP-Based Explore-Then-Exploit Agent
# ============================================================================

try:
    from scipy.optimize import minimize, linprog
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class LPExploreExploitAgent(Agent):
    """
    Explore-then-exploit agent using LP/optimization for state-independent prices.

    Phase 1 (Explore): Try various constant price combinations, measure reward rates.
    Phase 2 (Exploit): Solve optimization to find best constant prices, then use them.

    The optimization maximizes estimated reward rate subject to price bounds.
    """

    def __init__(self, config: ModelConfig, rng: np.random.Generator,
                 explore_episodes: int = 20,
                 samples_per_price: int = 500,
                 n_price_samples: int = 50,
                 use_quadratic_model: bool = True):
        """
        Args:
            config: model configuration
            rng: numpy random generator
            explore_episodes: number of exploration episodes before switching to exploit
            samples_per_price: events to run per price sample during exploration
            n_price_samples: number of random price combinations to try
            use_quadratic_model: if True, fit quadratic model; else use best observed
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy required for LPExploreExploitAgent")

        super().__init__(config, rng)

        self.explore_episodes = explore_episodes
        self.samples_per_price = samples_per_price
        self.n_price_samples = n_price_samples
        self.use_quadratic_model = use_quadratic_model

        # Exploration data: (prices, reward_rate) pairs
        self.price_samples = []  # List of (prices_c, prices_s) tuples
        self.reward_samples = []  # Corresponding reward rates

        # Current exploration state
        self.episode_count = 0
        self.step_in_episode = 0
        self.episode_reward = 0.0
        self.episode_time = 0.0

        # Current prices being tested
        self.current_prices_c = None
        self.current_prices_s = None

        # Exploitation prices (set after exploration)
        self.exploit_prices_c = None
        self.exploit_prices_s = None
        self.is_exploiting = False

        # Generate exploration price schedule
        self._generate_exploration_schedule()

    def _generate_exploration_schedule(self):
        """Generate random price combinations to try during exploration."""
        self.exploration_schedule = []

        for _ in range(self.n_price_samples):
            prices_c = self.rng.uniform(CUSTOMER_PRICE_MIN, CUSTOMER_PRICE_MAX, size=self.K)
            prices_s = self.rng.uniform(SERVER_PRICE_MIN, SERVER_PRICE_MAX, size=self.L)
            self.exploration_schedule.append((prices_c.copy(), prices_s.copy()))

        # Also add some structured samples (corners, midpoints)
        # All low prices
        self.exploration_schedule.append((
            np.full(self.K, CUSTOMER_PRICE_MIN),
            np.full(self.L, SERVER_PRICE_MIN)
        ))
        # All high prices
        self.exploration_schedule.append((
            np.full(self.K, CUSTOMER_PRICE_MAX),
            np.full(self.L, SERVER_PRICE_MAX)
        ))
        # Midpoint
        self.exploration_schedule.append((
            np.full(self.K, (CUSTOMER_PRICE_MIN + CUSTOMER_PRICE_MAX) / 2),
            np.full(self.L, (SERVER_PRICE_MIN + SERVER_PRICE_MAX) / 2)
        ))

    def get_prices(self, n: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get prices (state-independent)."""
        if self.is_exploiting:
            return self.exploit_prices_c.copy(), self.exploit_prices_s.copy()

        # During exploration, use current scheduled prices
        if self.current_prices_c is None:
            self._start_new_exploration_episode()

        return self.current_prices_c.copy(), self.current_prices_s.copy()

    def observe(self, n, m, prices_c, prices_s, reward, new_n, new_m, sojourn_time, event_type):
        """Observe transition and accumulate episode statistics."""
        if self.is_exploiting:
            return  # No tracking needed during exploitation

        self.episode_reward += reward
        self.episode_time += sojourn_time
        self.step_in_episode += 1

        # Check if episode is done
        if self.step_in_episode >= self.samples_per_price:
            self._end_exploration_episode()

    def _start_new_exploration_episode(self):
        """Start a new exploration episode with new prices."""
        if self.episode_count < len(self.exploration_schedule):
            self.current_prices_c, self.current_prices_s = self.exploration_schedule[self.episode_count]
        else:
            # Ran out of scheduled prices, generate new random ones
            self.current_prices_c = self.rng.uniform(CUSTOMER_PRICE_MIN, CUSTOMER_PRICE_MAX, size=self.K)
            self.current_prices_s = self.rng.uniform(SERVER_PRICE_MIN, SERVER_PRICE_MAX, size=self.L)

        self.step_in_episode = 0
        self.episode_reward = 0.0
        self.episode_time = 0.0

    def _end_exploration_episode(self):
        """End current exploration episode and record results."""
        if self.episode_time > 0:
            reward_rate = self.episode_reward / self.episode_time

            self.price_samples.append((self.current_prices_c.copy(), self.current_prices_s.copy()))
            self.reward_samples.append(reward_rate)

        self.episode_count += 1

        # Check if we should switch to exploitation
        if self.episode_count >= self.explore_episodes:
            self._solve_for_optimal_prices()
            self.is_exploiting = True
        else:
            self._start_new_exploration_episode()

    def _solve_for_optimal_prices(self):
        """Solve optimization problem to find best constant prices."""
        if len(self.price_samples) == 0:
            # No data, use midpoint
            self.exploit_prices_c = np.full(self.K, (CUSTOMER_PRICE_MIN + CUSTOMER_PRICE_MAX) / 2)
            self.exploit_prices_s = np.full(self.L, (SERVER_PRICE_MIN + SERVER_PRICE_MAX) / 2)
            return

        if self.use_quadratic_model:
            self._solve_quadratic_model()
        else:
            self._use_best_observed()

    def _use_best_observed(self):
        """Simply use the best observed prices."""
        best_idx = np.argmax(self.reward_samples)
        self.exploit_prices_c, self.exploit_prices_s = self.price_samples[best_idx]
        print(f"LP Agent: Using best observed prices (reward rate: {self.reward_samples[best_idx]:.4f})")

    def _solve_quadratic_model(self):
        """Fit quadratic model and optimize."""
        # Flatten prices into feature vectors
        n_samples = len(self.price_samples)
        n_features = self.K + self.L

        X = np.zeros((n_samples, n_features))
        y = np.array(self.reward_samples)

        for i, (pc, ps) in enumerate(self.price_samples):
            X[i, :self.K] = pc
            X[i, self.K:] = ps

        # Build quadratic features: [1, x, x^2, x_i*x_j]
        # For simplicity, just use linear + squared terms (diagonal quadratic)
        X_quad = np.column_stack([
            np.ones(n_samples),  # intercept
            X,  # linear terms
            X ** 2  # squared terms
        ])

        # Fit linear regression: y = X_quad @ beta
        # Using least squares
        try:
            beta, residuals, rank, s = np.linalg.lstsq(X_quad, y, rcond=None)
        except np.linalg.LinAlgError:
            print("LP Agent: Regression failed, using best observed")
            self._use_best_observed()
            return

        # Define objective function (negative because we minimize)
        def objective(prices):
            x = prices.reshape(1, -1)
            x_quad = np.column_stack([
                np.ones(1),
                x,
                x ** 2
            ])
            return -float(x_quad @ beta)

        # Bounds
        bounds = (
            [(CUSTOMER_PRICE_MIN, CUSTOMER_PRICE_MAX)] * self.K +
            [(SERVER_PRICE_MIN, SERVER_PRICE_MAX)] * self.L
        )

        # Initial guess: best observed
        best_idx = np.argmax(self.reward_samples)
        pc_init, ps_init = self.price_samples[best_idx]
        x0 = np.concatenate([pc_init, ps_init])

        # Optimize
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

        if result.success:
            self.exploit_prices_c = result.x[:self.K]
            self.exploit_prices_s = result.x[self.K:]
            predicted_reward = -result.fun
            print(f"LP Agent: Optimization succeeded (predicted reward rate: {predicted_reward:.4f})")
        else:
            print(f"LP Agent: Optimization failed ({result.message}), using best observed")
            self._use_best_observed()

    def get_exploration_summary(self) -> dict:
        """Return summary of exploration phase."""
        if len(self.reward_samples) == 0:
            return {'n_samples': 0}

        return {
            'n_samples': len(self.reward_samples),
            'best_reward': max(self.reward_samples),
            'worst_reward': min(self.reward_samples),
            'mean_reward': np.mean(self.reward_samples),
            'std_reward': np.std(self.reward_samples),
            'exploit_prices_c': self.exploit_prices_c,
            'exploit_prices_s': self.exploit_prices_s,
        }


class GridSearchAgent(Agent):
    """
    Simple grid search baseline: try a grid of constant prices, pick the best.

    Even simpler than LP - just exhaustive search over a discretized price space.
    """

    def __init__(self, config: ModelConfig, rng: np.random.Generator,
                 n_grid_points: int = 5,
                 samples_per_price: int = 500):
        """
        Args:
            config: model configuration
            rng: numpy random generator
            n_grid_points: number of grid points per price dimension
            samples_per_price: events to run per price combination
        """
        super().__init__(config, rng)

        self.n_grid_points = n_grid_points
        self.samples_per_price = samples_per_price

        # Generate grid
        customer_grid = np.linspace(CUSTOMER_PRICE_MIN, CUSTOMER_PRICE_MAX, n_grid_points)
        server_grid = np.linspace(SERVER_PRICE_MIN, SERVER_PRICE_MAX, n_grid_points)

        # For simplicity, use same price for all customer classes and all server classes
        # This reduces grid from n^(K+L) to n^2
        self.price_grid = []
        for pc in customer_grid:
            for ps in server_grid:
                self.price_grid.append((
                    np.full(self.K, pc),
                    np.full(self.L, ps)
                ))

        self.grid_idx = 0
        self.grid_rewards = []

        self.step_in_episode = 0
        self.episode_reward = 0.0
        self.episode_time = 0.0

        self.current_prices_c, self.current_prices_s = self.price_grid[0]
        self.is_exploiting = False
        self.exploit_prices_c = None
        self.exploit_prices_s = None

    def get_prices(self, n: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.is_exploiting:
            return self.exploit_prices_c.copy(), self.exploit_prices_s.copy()
        return self.current_prices_c.copy(), self.current_prices_s.copy()

    def observe(self, n, m, prices_c, prices_s, reward, new_n, new_m, sojourn_time, event_type):
        if self.is_exploiting:
            return  # No tracking needed during exploitation

        self.episode_reward += reward
        self.episode_time += sojourn_time
        self.step_in_episode += 1

        if self.step_in_episode >= self.samples_per_price:
            # Record reward rate
            if self.episode_time > 0:
                self.grid_rewards.append(self.episode_reward / self.episode_time)
            else:
                self.grid_rewards.append(0.0)

            # Move to next grid point
            self.grid_idx += 1

            if self.grid_idx >= len(self.price_grid):
                # Done exploring, pick best
                best_idx = np.argmax(self.grid_rewards)
                self.exploit_prices_c, self.exploit_prices_s = self.price_grid[best_idx]
                self.is_exploiting = True
                print(f"GridSearch: Best prices found (reward rate: {self.grid_rewards[best_idx]:.4f})")
                print(f"  Customer: {self.exploit_prices_c}")
                print(f"  Server: {self.exploit_prices_s}")
            else:
                self.current_prices_c, self.current_prices_s = self.price_grid[self.grid_idx]
                self.step_in_episode = 0
                self.episode_reward = 0.0
                self.episode_time = 0.0
