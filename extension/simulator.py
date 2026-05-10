"""
Simulator for two-sided pricing queue.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

from model import TwoSidedQueue, ModelConfig
from policy import Policy


@dataclass
class SimulationStats:
    """Statistics from a simulation run."""
    total_time: float
    total_reward: float
    n_events: int
    avg_reward_rate: float
    avg_customer_counts: np.ndarray
    avg_server_counts: np.ndarray
    event_counts: Dict[str, int]


class Simulator:
    """Simulator for the two-sided pricing queue."""

    def __init__(self, queue: TwoSidedQueue):
        self.queue = queue
        self.config = queue.config

    def run_episode(self, policy: Policy, n_events: int,
                    initial_n: Optional[np.ndarray] = None,
                    initial_m: Optional[np.ndarray] = None,
                    callback=None) -> SimulationStats:
        """
        Run a single episode.

        Args:
            policy: pricing policy
            n_events: number of events to simulate
            initial_n: initial customer counts (default: zeros)
            initial_m: initial server counts (default: zeros)
            callback: optional function called after each event with
                      (step, n, m, prices_c, prices_s, reward, event_type)

        Returns:
            SimulationStats with episode statistics
        """
        K, L = self.queue.K, self.queue.L

        # Initialize state
        if initial_n is None:
            n = np.zeros(K, dtype=int)
        else:
            n = initial_n.copy()

        if initial_m is None:
            m = np.zeros(L, dtype=int)
        else:
            m = initial_m.copy()

        total_time = 0.0
        total_reward = 0.0
        event_counts = {}

        # For computing averages
        time_weighted_n = np.zeros(K, dtype=float)
        time_weighted_m = np.zeros(L, dtype=float)

        for step in range(n_events):
            # Get prices from policy
            prices_c, prices_s = policy.get_prices(n, m)

            # Simulate one step
            new_n, new_m, sojourn_time, reward, event_type = self.queue.step(
                n, m, prices_c, prices_s
            )

            # Update statistics
            total_time += sojourn_time
            total_reward += reward
            time_weighted_n += n * sojourn_time
            time_weighted_m += m * sojourn_time
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

            # Callback
            if callback is not None:
                callback(step, n, m, prices_c, prices_s, reward, event_type)

            # Update state
            n, m = new_n, new_m

        avg_reward_rate = total_reward / total_time if total_time > 0 else 0.0
        avg_n = time_weighted_n / total_time if total_time > 0 else np.zeros(K)
        avg_m = time_weighted_m / total_time if total_time > 0 else np.zeros(L)

        return SimulationStats(
            total_time=total_time,
            total_reward=total_reward,
            n_events=n_events,
            avg_reward_rate=avg_reward_rate,
            avg_customer_counts=avg_n,
            avg_server_counts=avg_m,
            event_counts=event_counts
        )

    def evaluate_policy(self, policy: Policy, n_events: int = 10000,
                        n_runs: int = 5) -> Tuple[float, float]:
        """
        Evaluate a policy over multiple runs.

        Returns: (mean_reward_rate, std_reward_rate)
        """
        reward_rates = []

        for _ in range(n_runs):
            stats = self.run_episode(policy, n_events)
            reward_rates.append(stats.avg_reward_rate)

        return np.mean(reward_rates), np.std(reward_rates)


class LearningSimulator(Simulator):
    """Simulator that tracks observations for learning agents."""

    def __init__(self, queue: TwoSidedQueue):
        super().__init__(queue)
        self.observations = []

    def run_learning_episode(self, agent, n_events: int,
                             initial_n: Optional[np.ndarray] = None,
                             initial_m: Optional[np.ndarray] = None) -> SimulationStats:
        """
        Run episode with a learning agent.

        The agent must have:
            - get_prices(n, m) -> (prices_c, prices_s)
            - observe(n, m, prices_c, prices_s, reward, new_n, new_m, sojourn_time, event_type)
        """
        K, L = self.queue.K, self.queue.L

        # Initialize state
        if initial_n is None:
            n = np.zeros(K, dtype=int)
        else:
            n = initial_n.copy()

        if initial_m is None:
            m = np.zeros(L, dtype=int)
        else:
            m = initial_m.copy()

        total_time = 0.0
        total_reward = 0.0
        event_counts = {}

        time_weighted_n = np.zeros(K, dtype=float)
        time_weighted_m = np.zeros(L, dtype=float)

        for step in range(n_events):
            # Get prices from agent
            prices_c, prices_s = agent.get_prices(n, m)

            # Simulate one step
            new_n, new_m, sojourn_time, reward, event_type = self.queue.step(
                n, m, prices_c, prices_s
            )

            # Let agent observe
            agent.observe(n, m, prices_c, prices_s, reward, new_n, new_m, sojourn_time, event_type)

            # Update statistics
            total_time += sojourn_time
            total_reward += reward
            time_weighted_n += n * sojourn_time
            time_weighted_m += m * sojourn_time
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

            # Update state
            n, m = new_n, new_m

        avg_reward_rate = total_reward / total_time if total_time > 0 else 0.0
        avg_n = time_weighted_n / total_time if total_time > 0 else np.zeros(K)
        avg_m = time_weighted_m / total_time if total_time > 0 else np.zeros(L)

        return SimulationStats(
            total_time=total_time,
            total_reward=total_reward,
            n_events=n_events,
            avg_reward_rate=avg_reward_rate,
            avg_customer_counts=avg_n,
            avg_server_counts=avg_m,
            event_counts=event_counts
        )
