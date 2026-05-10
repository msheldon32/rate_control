"""
Policies for two-sided pricing queue.

Monotonicity constraints:
- p_i^c(n, m) non-decreasing in n_i
- p_j^s(n, m) non-decreasing in m_j

Price bounds:
- Customer prices in [-1, 1]
- Server prices in [-1, 0) — servers are always paid (negative price = payment)
"""

# Price bounds
CUSTOMER_PRICE_MIN = -1.0
CUSTOMER_PRICE_MAX = 1.0
SERVER_PRICE_MIN = -1.0
SERVER_PRICE_MAX = -1e-6  # Strictly negative

import numpy as np
from typing import Tuple, List, Optional
from abc import ABC, abstractmethod

from model import ModelConfig


class Policy(ABC):
    """Base class for pricing policies."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.K = config.n_customer_classes
        self.L = config.n_server_classes

    @abstractmethod
    def get_prices(self, n: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get prices for current state.

        Args:
            n: customer counts (K,)
            m: server counts (L,)

        Returns:
            prices_c: customer prices (K,), each in [-1, 1]
            prices_s: server prices (L,), each in [-1, 0) (always negative)
        """
        pass


class RandomPolicy(Policy):
    """Random prices, ignoring monotonicity."""

    def __init__(self, config: ModelConfig, rng: np.random.Generator):
        super().__init__(config)
        self.rng = rng

    def get_prices(self, n: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        prices_c = self.rng.uniform(CUSTOMER_PRICE_MIN, CUSTOMER_PRICE_MAX, size=self.K)
        prices_s = self.rng.uniform(SERVER_PRICE_MIN, SERVER_PRICE_MAX, size=self.L)
        return prices_c, prices_s


class ConstantPolicy(Policy):
    """Constant prices regardless of state."""

    def __init__(self, config: ModelConfig, price_c: float = 0.0, price_s: float = -0.5):
        super().__init__(config)
        self.price_c = np.clip(price_c, CUSTOMER_PRICE_MIN, CUSTOMER_PRICE_MAX)
        self.price_s = np.clip(price_s, SERVER_PRICE_MIN, SERVER_PRICE_MAX)

    def get_prices(self, n: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        prices_c = np.full(self.K, self.price_c)
        prices_s = np.full(self.L, self.price_s)
        return prices_c, prices_s


class LinearMonotonicPolicy(Policy):
    """
    Linear monotonic policy.

    p_i^c(n, m) = clip(a_i * n_i / cap_i + b_i, -1, 1)
    p_j^s(n, m) = clip(c_j * m_j / cap_j + d_j, -1, 0)

    With a_i >= 0, c_j >= 0 for monotonicity.
    Server prices are always negative (payment to servers).
    """

    def __init__(self, config: ModelConfig,
                 customer_slopes: Optional[np.ndarray] = None,
                 customer_intercepts: Optional[np.ndarray] = None,
                 server_slopes: Optional[np.ndarray] = None,
                 server_intercepts: Optional[np.ndarray] = None):
        super().__init__(config)

        # Default: customer price goes from -1 (empty) to 1 (full capacity)
        self.customer_slopes = customer_slopes if customer_slopes is not None else np.full(self.K, 2.0)
        self.customer_intercepts = customer_intercepts if customer_intercepts is not None else np.full(self.K, -1.0)
        # Default: server price goes from -1 (empty) to ~0 (full capacity), always negative
        self.server_slopes = server_slopes if server_slopes is not None else np.full(self.L, 1.0)
        self.server_intercepts = server_intercepts if server_intercepts is not None else np.full(self.L, -1.0)

    def get_prices(self, n: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        prices_c = np.zeros(self.K)
        prices_s = np.zeros(self.L)

        for i in range(self.K):
            normalized = n[i] / max(self.config.customer_capacities[i], 1)
            prices_c[i] = np.clip(
                self.customer_slopes[i] * normalized + self.customer_intercepts[i],
                CUSTOMER_PRICE_MIN, CUSTOMER_PRICE_MAX
            )

        for j in range(self.L):
            normalized = m[j] / max(self.config.server_capacities[j], 1)
            prices_s[j] = np.clip(
                self.server_slopes[j] * normalized + self.server_intercepts[j],
                SERVER_PRICE_MIN, SERVER_PRICE_MAX
            )

        return prices_c, prices_s


class TabularMonotonicPolicy(Policy):
    """
    Tabular policy with monotonicity enforced.

    Stores a price for each (class, count) pair.
    Monotonicity: price[i, k] <= price[i, k+1] for all k.
    Server prices are always negative.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # Initialize with linear ramp from -1 to 1 for customers
        self.customer_prices = []
        for i in range(self.K):
            cap = config.customer_capacities[i]
            prices = np.linspace(CUSTOMER_PRICE_MIN, CUSTOMER_PRICE_MAX, cap + 1)
            self.customer_prices.append(prices)

        # Initialize with linear ramp from -1 to ~0 for servers (always negative)
        self.server_prices = []
        for j in range(self.L):
            cap = config.server_capacities[j]
            prices = np.linspace(SERVER_PRICE_MIN, SERVER_PRICE_MAX, cap + 1)
            self.server_prices.append(prices)

    def get_prices(self, n: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        prices_c = np.array([self.customer_prices[i][n[i]] for i in range(self.K)])
        prices_s = np.array([self.server_prices[j][m[j]] for j in range(self.L)])
        return prices_c, prices_s

    def set_customer_price(self, class_idx: int, count: int, price: float):
        """Set price and enforce monotonicity."""
        price = np.clip(price, CUSTOMER_PRICE_MIN, CUSTOMER_PRICE_MAX)
        cap = self.config.customer_capacities[class_idx]

        self.customer_prices[class_idx][count] = price

        # Enforce monotonicity: propagate upward
        for k in range(count + 1, cap + 1):
            if self.customer_prices[class_idx][k] < price:
                self.customer_prices[class_idx][k] = price

        # Enforce monotonicity: propagate downward
        for k in range(count - 1, -1, -1):
            if self.customer_prices[class_idx][k] > price:
                self.customer_prices[class_idx][k] = price

    def set_server_price(self, class_idx: int, count: int, price: float):
        """Set price and enforce monotonicity. Server prices must be negative."""
        price = np.clip(price, SERVER_PRICE_MIN, SERVER_PRICE_MAX)
        cap = self.config.server_capacities[class_idx]

        self.server_prices[class_idx][count] = price

        # Enforce monotonicity: propagate upward
        for k in range(count + 1, cap + 1):
            if self.server_prices[class_idx][k] < price:
                self.server_prices[class_idx][k] = price

        # Enforce monotonicity: propagate downward
        for k in range(count - 1, -1, -1):
            if self.server_prices[class_idx][k] > price:
                self.server_prices[class_idx][k] = price

    def project_to_monotonic(self):
        """Project current prices to nearest monotonic function (isotonic regression)."""
        for i in range(self.K):
            self.customer_prices[i] = self._isotonic_regression(self.customer_prices[i])
            self.customer_prices[i] = np.clip(self.customer_prices[i], CUSTOMER_PRICE_MIN, CUSTOMER_PRICE_MAX)

        for j in range(self.L):
            self.server_prices[j] = self._isotonic_regression(self.server_prices[j])
            self.server_prices[j] = np.clip(self.server_prices[j], SERVER_PRICE_MIN, SERVER_PRICE_MAX)

    @staticmethod
    def _isotonic_regression(y: np.ndarray) -> np.ndarray:
        """Pool Adjacent Violators Algorithm for isotonic regression."""
        n = len(y)
        result = y.copy()
        blocks = [[i] for i in range(n)]

        while True:
            merged = False
            i = 0
            while i < len(blocks) - 1:
                # Check if blocks violate monotonicity
                mean_i = np.mean([result[j] for j in blocks[i]])
                mean_next = np.mean([result[j] for j in blocks[i + 1]])

                if mean_i > mean_next:
                    # Merge blocks
                    blocks[i] = blocks[i] + blocks[i + 1]
                    blocks.pop(i + 1)
                    # Update values in merged block
                    new_mean = np.mean([result[j] for j in blocks[i]])
                    for j in blocks[i]:
                        result[j] = new_mean
                    merged = True
                else:
                    i += 1

            if not merged:
                break

        return result


class ThresholdPolicy(Policy):
    """
    Threshold policy: price jumps at certain count thresholds.

    Simpler than full tabular, fewer parameters.
    """

    def __init__(self, config: ModelConfig,
                 customer_thresholds: Optional[List[List[Tuple[int, float]]]] = None,
                 server_thresholds: Optional[List[List[Tuple[int, float]]]] = None):
        """
        Thresholds: list of (count, price) pairs for each class.
        Price at count c is the price of the largest threshold <= c.
        """
        super().__init__(config)

        # Default: thresholds at 1/3 and 2/3 capacity
        if customer_thresholds is None:
            self.customer_thresholds = []
            for i in range(self.K):
                cap = config.customer_capacities[i]
                self.customer_thresholds.append([
                    (0, -1.0),
                    (cap // 3, -0.33),
                    (2 * cap // 3, 0.33),
                    (cap, 1.0)
                ])
        else:
            self.customer_thresholds = customer_thresholds

        # Server thresholds: all negative (servers are always paid)
        if server_thresholds is None:
            self.server_thresholds = []
            for j in range(self.L):
                cap = config.server_capacities[j]
                self.server_thresholds.append([
                    (0, -1.0),
                    (cap // 3, -0.67),
                    (2 * cap // 3, -0.33),
                    (cap, -0.01)  # Still negative at full capacity
                ])
        else:
            self.server_thresholds = server_thresholds

    def _lookup_price(self, thresholds: List[Tuple[int, float]], count: int) -> float:
        """Find price for given count using thresholds."""
        price = thresholds[0][1]
        for thresh, p in thresholds:
            if count >= thresh:
                price = p
            else:
                break
        return price

    def get_prices(self, n: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        prices_c = np.array([
            self._lookup_price(self.customer_thresholds[i], n[i])
            for i in range(self.K)
        ])
        prices_s = np.array([
            self._lookup_price(self.server_thresholds[j], m[j])
            for j in range(self.L)
        ])
        return prices_c, prices_s
