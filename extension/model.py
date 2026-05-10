"""
Two-sided pricing queue with compatibilities.

State: (n, m) where n = customer counts, m = server counts
Action: prices p^c for customers, p^s for servers (both in [-1, 1])
Arrival rates determined by demand/supply curves applied to prices.

Monotonicity constraints:
- p_i^c non-decreasing in n_i (more customers → higher price)
- p_j^s non-decreasing in m_j (more servers → higher price / lower payment)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class ModelConfig:
    """Configuration for the two-sided queue model."""
    n_customer_classes: int
    n_server_classes: int
    customer_capacities: List[int]  # max count per customer class
    server_capacities: List[int]    # max count per server class
    compatibility: np.ndarray       # K x L binary matrix, G[i,j] = 1 if compatible

    # Demand/supply curve parameters
    # Arrival rate = base_rate * demand_fn(price)
    customer_base_rates: List[float]
    server_base_rates: List[float]

    # Reward parameters
    customer_holding_costs: List[float]  # cost per customer per unit time
    server_holding_costs: List[float]    # cost per idle server per unit time
    match_rewards: np.ndarray            # K x L matrix, reward for matching (i, j)

    def __post_init__(self):
        self.n_customer_states = [c + 1 for c in self.customer_capacities]
        self.n_server_states = [c + 1 for c in self.server_capacities]
        self.total_customer_capacity = sum(self.customer_capacities)
        self.total_server_capacity = sum(self.server_capacities)


class DemandCurve:
    """Maps price in [-1, 1] to arrival rate multiplier in [0, 1]."""

    @staticmethod
    def linear(price: float) -> float:
        """Linear demand: rate = (1 - price) / 2, so price=-1 → rate=1, price=1 → rate=0."""
        return (1.0 - price) / 2.0

    @staticmethod
    def linear_cust(price: float) -> float:
        """Linear demand: rate = (1 - price) / 2, so price=-1 → rate=1, price=1 → rate=0."""
        #return (1.0 - price) / 2.0
        intercept = 1.0
        slope = -1.0
        return min(max(slope*price + intercept,0),1)

    @staticmethod
    def linear_serv(price: float) -> float:
        """Linear demand: rate = (1 - price) / 2, so price=-1 → rate=1, price=1 → rate=0."""
        #return (1.0 - price) / 2.0
        intercept = 0
        slope = -1

        return min(max(slope*price + intercept,0),1)

    @staticmethod
    def exponential(price: float, steepness: float = 2.0) -> float:
        """Exponential demand curve."""
        return np.exp(-steepness * (price + 1) / 2.0)

    @staticmethod
    def logistic(price: float, steepness: float = 4.0) -> float:
        """Logistic demand curve centered at price=0."""
        return 1.0 / (1.0 + np.exp(steepness * price))


class TwoSidedQueue:
    """
    Two-sided queue with pricing control.

    State: (n, m) where n[i] = count of customer class i, m[j] = count of server class j
    Action: (p_c, p_s) where p_c[i] = price for customer class i, p_s[j] = price for server class j
    """

    def __init__(self, config: ModelConfig, rng: np.random.Generator,
                 customer_demand_fn=DemandCurve.linear_cust, server_demand_fn=DemandCurve.linear_serv):
        self.config = config
        self.rng = rng
        self.customer_demand_fn = customer_demand_fn
        self.server_demand_fn = server_demand_fn

        self.K = config.n_customer_classes
        self.L = config.n_server_classes

    def get_arrival_rates(self, prices_c: np.ndarray, prices_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert prices to arrival rates using demand curves."""
        #lambda_c = np.array([
        #    self.config.customer_base_rates[i] * self.customer_demand_fn(prices_c[i])
        #    for i in range(self.K)
        #])
        lambda_c = np.array([
            self.config.customer_base_rates[i] * min(max(-prices_c[i] + 1, 0),1)
            for i in range(self.K)
        ])
        #lambda_s = np.array([
        #    self.config.server_base_rates[j] * self.server_demand_fn(prices_s[j])
        #    for j in range(self.L)
        #])
        lambda_s = np.array([
            self.config.server_base_rates[i] * min(max(-prices_s[i],0),1)
            for i in range(self.K)
        ])
        return lambda_c, lambda_s

    def get_holding_reward(self, n: np.ndarray, m: np.ndarray) -> float:
        """Holding cost (negative reward) per unit time."""
        customer_cost = sum(self.config.customer_holding_costs[i] * n[i] for i in range(self.K))
        server_cost = sum(self.config.server_holding_costs[j] * m[j] for j in range(self.L))
        return -(customer_cost + server_cost)

    def get_price_revenue(self, prices_c: np.ndarray, prices_s: np.ndarray,
                          lambda_c: np.ndarray, lambda_s: np.ndarray) -> float:
        """Revenue from prices (price * arrival rate)."""
        customer_revenue = sum(prices_c[i] * lambda_c[i] for i in range(self.K))
        server_revenue = sum(prices_s[j] * lambda_s[j] for j in range(self.L))
        return customer_revenue + server_revenue

    def step(self, n: np.ndarray, m: np.ndarray,
             prices_c: np.ndarray, prices_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, str]:
        """
        Simulate one event in the continuous-time queue.

        Returns: (new_n, new_m, sojourn_time, reward, event_type)
        """
        # Get rates
        lambda_c, lambda_s = self.get_arrival_rates(prices_c, prices_s)

        # Zero out arrivals at capacity
        for i in range(self.K):
            if n[i] >= self.config.customer_capacities[i]:
                lambda_c[i] = 0
        for j in range(self.L):
            if m[j] >= self.config.server_capacities[j]:
                lambda_s[j] = 0

        # Collect all possible events and their rates
        events = []
        rates = []

        # Customer arrivals
        for i in range(self.K):
            if lambda_c[i] > 0:
                events.append(('customer_arrival', i))
                rates.append(lambda_c[i])

        # Server arrivals
        for j in range(self.L):
            if lambda_s[j] > 0:
                events.append(('server_arrival', j))
                rates.append(lambda_s[j])

        total_rate = 5#sum(rates)

        events.append(('nothing',0))
        rates.append(total_rate-sum(rates))

        # Sample sojourn time
        #sojourn_time = self.rng.exponential(1.0 / total_rate)
        sojourn_time = 0.2

        # Sample event
        probs = np.array(rates) / total_rate
        event_idx = self.rng.choice(len(events), p=probs)
        event = events[event_idx]

        # Compute reward
        holding_reward = self.get_holding_reward(n, m) * sojourn_time
        price_revenue = 0 #self.get_price_revenue(prices_c, prices_s, lambda_c, lambda_s) * sojourn_time

        # Apply event
        new_n, new_m = n.copy(), m.copy()
        transition_reward = 0.0

        if event[0] == 'customer_arrival':
            i = event[1]
            found = False
            for j in range (self.L):
                if self.config.compatibility[i,j] and new_m[j] > 0:
                    found = True
                    new_m[j] -= 1
                    price_revenue += prices_c[i]
            if not found and new_n[i] < self.config.customer_capacities[i]:
                new_n[i] += 1
                price_revenue += prices_c[i]
        elif event[0] == 'server_arrival':
            j = event[1]
            found = False
            for i in range (self.K):
                if self.config.compatibility[i,j] and new_n[i] > 0:
                    found = True
                    new_n[i] -= 1
                    price_revenue += prices_s[i]
            if not found and new_m[j] < self.config.server_capacities[j]:
                new_m[j] += 1
                price_revenue += prices_s[i]
        elif event[0] == 'match':
            i, j = event[1]
            new_n[i] -= 1
            new_m[j] -= 1
            transition_reward = self.config.match_rewards[i, j]

        total_reward = holding_reward + price_revenue + transition_reward
        event_str = f"{event[0]}_{event[1]}"

        return new_n, new_m, sojourn_time, total_reward, event_str

    def state_to_idx(self, n: np.ndarray, m: np.ndarray) -> int:
        """Convert state (n, m) to flat index for tabular methods."""
        idx = 0
        multiplier = 1

        for i in range(self.K):
            idx += n[i] * multiplier
            multiplier *= self.config.n_customer_states[i]

        for j in range(self.L):
            idx += m[j] * multiplier
            multiplier *= self.config.n_server_states[j]

        return idx

    def idx_to_state(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Convert flat index back to state (n, m)."""
        n = np.zeros(self.K, dtype=int)
        m = np.zeros(self.L, dtype=int)

        for i in range(self.K):
            n[i] = idx % self.config.n_customer_states[i]
            idx //= self.config.n_customer_states[i]

        for j in range(self.L):
            m[j] = idx % self.config.n_server_states[j]
            idx //= self.config.n_server_states[j]

        return n, m

    @property
    def n_states(self) -> int:
        """Total number of states."""
        result = 1
        for i in range(self.K):
            result *= self.config.n_customer_states[i]
        for j in range(self.L):
            result *= self.config.n_server_states[j]
        return result


def make_simple_config(
    n_customer_classes: int = 2,
    n_server_classes: int = 2,
    capacity: int = 10,
    base_rate: float = 1.0,
    holding_cost: float = 0.1,
    match_reward: float = 0,
    compatibility: str = 'full'  # 'full', 'diagonal', 'random'
) -> ModelConfig:
    """Create a simple model configuration for testing."""

    K, L = n_customer_classes, n_server_classes

    customer_capacities = [capacity] * K
    server_capacities = [capacity] * L

    if compatibility == 'full':
        compat = np.ones((K, L))
    elif compatibility == 'diagonal':
        compat = np.eye(K, L)
    elif compatibility == 'random':
        compat = (np.random.rand(K, L) > 0.5).astype(float)
        # Ensure at least one match per class
        for i in range(K):
            if compat[i].sum() == 0:
                compat[i, np.random.randint(L)] = 1
        for j in range(L):
            if compat[:, j].sum() == 0:
                compat[np.random.randint(K), j] = 1
    else:
        raise ValueError(f"Unknown compatibility type: {compatibility}")

    return ModelConfig(
        n_customer_classes=K,
        n_server_classes=L,
        customer_capacities=customer_capacities,
        server_capacities=server_capacities,
        compatibility=compat,
        customer_base_rates=[base_rate] * K,
        server_base_rates=[base_rate] * L,
        customer_holding_costs=[holding_cost] * K,
        server_holding_costs=[holding_cost] * L,
        match_rewards=np.full((K, L), match_reward) * compat
    )
