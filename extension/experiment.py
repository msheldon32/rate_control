"""
Basic experiment script for two-sided pricing queue.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

from model import make_simple_config, TwoSidedQueue
from policy import ConstantPolicy, LinearMonotonicPolicy, ThresholdPolicy
from simulator import Simulator, LearningSimulator
from agent import (
    RandomAgent, ConstantAgent, TabularPolicyGradientAgent, LinearPolicyGradientAgent,
    TORCH_AVAILABLE, SCIPY_AVAILABLE
)

if TORCH_AVAILABLE:
    from agent import DeepLatticeAgent, MLPAgent

if SCIPY_AVAILABLE:
    from agent import LPExploreExploitAgent, GridSearchAgent


def count_parameters(agent) -> int:
    """Count the number of trainable parameters in an agent."""
    if isinstance(agent, LinearPolicyGradientAgent):
        # slopes + intercepts for customers and servers
        K = agent.K
        L = agent.L
        return 2 * K + 2 * L  # customer_slopes, customer_intercepts, server_slopes, server_intercepts

    elif isinstance(agent, TabularPolicyGradientAgent):
        # logits for each (class, count, price_level)
        total = 0
        for i in range(agent.K):
            total += (agent.config.customer_capacities[i] + 1) * agent.n_price_levels
        for j in range(agent.L):
            total += (agent.config.server_capacities[j] + 1) * agent.n_price_levels
        return total

    elif TORCH_AVAILABLE and hasattr(agent, 'network'):
        # PyTorch model
        return sum(p.numel() for p in agent.network.parameters() if p.requires_grad)

    elif SCIPY_AVAILABLE and hasattr(agent, 'exploit_prices_c'):
        # LP/GridSearch agent: K + L constant prices
        return agent.K + agent.L

    else:
        return 0


def print_agent_info(agent, name: str):
    """Print agent information including parameter count."""
    n_params = count_parameters(agent)
    print(f"Agent: {name}")
    print(f"  Parameters: {n_params:,}")


def evaluate_baselines(
    n_customer_classes: int = 2,
    n_server_classes: int = 2,
    capacity: int = 10,
    n_events: int = 10000,
    n_runs: int = 5,
    seed: int = 42
) -> dict:
    """
    Evaluate baseline policies.

    Returns dict of {policy_name: (mean_reward, std_reward)}
    """
    rng = np.random.default_rng(seed)

    config = make_simple_config(
        n_customer_classes=n_customer_classes,
        n_server_classes=n_server_classes,
        capacity=capacity,
        base_rate=2.0,
        holding_cost=0.1,
        match_reward=1.0,
        compatibility='full'
    )

    queue = TwoSidedQueue(config, rng)
    simulator = Simulator(queue)

    results = {}

    # Constant policies at different price levels
    for price in [-0.5, 0.0, 0.5]:
        policy = ConstantPolicy(config, price_c=price, price_s=price)
        mean_r, std_r = simulator.evaluate_policy(policy, n_events=n_events, n_runs=n_runs)
        results[f'constant_{price}'] = (mean_r, std_r)
        print(f"Constant(p={price}): reward rate = {mean_r:.4f} +/- {std_r:.4f}")

    # Linear monotonic policy (default: -1 to 1 over capacity)
    policy = LinearMonotonicPolicy(config)
    mean_r, std_r = simulator.evaluate_policy(policy, n_events=n_events, n_runs=n_runs)
    results['linear_monotonic'] = (mean_r, std_r)
    print(f"Linear monotonic: reward rate = {mean_r:.4f} +/- {std_r:.4f}")

    # Threshold policy
    policy = ThresholdPolicy(config)
    mean_r, std_r = simulator.evaluate_policy(policy, n_events=n_events, n_runs=n_runs)
    results['threshold'] = (mean_r, std_r)
    print(f"Threshold: reward rate = {mean_r:.4f} +/- {std_r:.4f}")

    return results


def train_policy_gradient(
    n_customer_classes: int = 2,
    n_server_classes: int = 2,
    capacity: int = 10,
    n_episodes: int = 100,
    events_per_episode: int = 1000,
    seed: int = 42
) -> Tuple[LinearPolicyGradientAgent, List[float]]:
    """
    Train a linear policy gradient agent.

    Returns: (trained_agent, reward_history)
    """
    rng = np.random.default_rng(seed)

    config = make_simple_config(
        n_customer_classes=n_customer_classes,
        n_server_classes=n_server_classes,
        capacity=capacity,
        base_rate=2.0,
        holding_cost=0.1,
        match_reward=1.0,
        compatibility='full'
    )

    queue = TwoSidedQueue(config, rng)
    simulator = LearningSimulator(queue)

    agent = LinearPolicyGradientAgent(
        config, rng,
        learning_rate=0.1,
        noise_std=0.2,
        baseline_lr=0.1
    )

    print_agent_info(agent, "LinearPolicyGradient")

    reward_history = []

    for episode in range(n_episodes):
        stats = simulator.run_learning_episode(agent, events_per_episode)
        agent.update()

        reward_history.append(stats.avg_reward_rate)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}: reward rate = {stats.avg_reward_rate:.4f}")

    return agent, reward_history


def train_tabular_policy_gradient(
    n_customer_classes: int = 1,
    n_server_classes: int = 1,
    capacity: int = 10,
    n_episodes: int = 200,
    events_per_episode: int = 500,
    seed: int = 42
) -> Tuple[TabularPolicyGradientAgent, List[float]]:
    """
    Train a tabular policy gradient agent.

    Returns: (trained_agent, reward_history)
    """
    rng = np.random.default_rng(seed)

    config = make_simple_config(
        n_customer_classes=n_customer_classes,
        n_server_classes=n_server_classes,
        capacity=capacity,
        base_rate=2.0,
        holding_cost=0.1,
        match_reward=1.0,
        compatibility='full'
    )

    queue = TwoSidedQueue(config, rng)
    simulator = LearningSimulator(queue)

    agent = TabularPolicyGradientAgent(
        config, rng,
        n_price_levels=11,
        learning_rate=0.05,
        baseline_lr=0.1,
        entropy_coef=0.01
    )

    print_agent_info(agent, "TabularPolicyGradient")

    reward_history = []

    for episode in range(n_episodes):
        stats = simulator.run_learning_episode(agent, events_per_episode)
        agent.update()

        reward_history.append(stats.avg_reward_rate)

        if (episode + 1) % 20 == 0:
            print(f"Episode {episode + 1}: reward rate = {stats.avg_reward_rate:.4f}")

    return agent, reward_history


def train_deep_lattice(
    n_customer_classes: int = 2,
    n_server_classes: int = 2,
    capacity: int = 10,
    n_episodes: int = 100,
    events_per_episode: int = 1000,
    seed: int = 42
) -> Tuple[Any, List[float]]:
    """
    Train a Deep Lattice Network agent.

    Returns: (trained_agent, reward_history)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for DeepLatticeAgent")

    rng = np.random.default_rng(seed)

    config = make_simple_config(
        n_customer_classes=n_customer_classes,
        n_server_classes=n_server_classes,
        capacity=capacity,
        base_rate=2.0,
        holding_cost=0.1,
        match_reward=1.0,
        compatibility='full'
    )

    queue = TwoSidedQueue(config, rng)
    simulator = LearningSimulator(queue)

    agent = DeepLatticeAgent(
        config, rng,
        learning_rate=1e-3,
        n_calibrator_keypoints=10,
        lattice_size=3,
        use_cross_effects=False,
        exploration_noise=0.15,
        batch_size=64,
        update_every=100
    )

    print_agent_info(agent, "DeepLatticeNetwork")

    reward_history = []

    for episode in range(n_episodes):
        stats = simulator.run_learning_episode(agent, events_per_episode)

        reward_history.append(stats.avg_reward_rate)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}: reward rate = {stats.avg_reward_rate:.4f}")

    # Print monitoring summary
    print("\nTraining Monitoring Summary:")
    agent.print_monitoring_summary()

    # Check gradient health
    health = agent.check_gradient_health()
    if health['status'] == 'issues_detected':
        print(f"\nGradient Health Issues: {health['issues']}")

    return agent, reward_history


def train_mlp(
    n_customer_classes: int = 2,
    n_server_classes: int = 2,
    capacity: int = 10,
    n_episodes: int = 100,
    events_per_episode: int = 1000,
    seed: int = 42
) -> Tuple[Any, List[float]]:
    """
    Train an MLP agent (no monotonicity guarantees).

    Returns: (trained_agent, reward_history)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for MLPAgent")

    rng = np.random.default_rng(seed)

    config = make_simple_config(
        n_customer_classes=n_customer_classes,
        n_server_classes=n_server_classes,
        capacity=capacity,
        base_rate=2.0,
        holding_cost=0.1,
        match_reward=1.0,
        compatibility='full'
    )

    queue = TwoSidedQueue(config, rng)
    simulator = LearningSimulator(queue)

    agent = MLPAgent(
        config, rng,
        learning_rate=1e-3,
        hidden_dims=[64, 64],
        exploration_noise=0.15,
        batch_size=64,
        update_every=100
    )

    print_agent_info(agent, "MLP")

    reward_history = []

    for episode in range(n_episodes):
        stats = simulator.run_learning_episode(agent, events_per_episode)

        reward_history.append(stats.avg_reward_rate)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}: reward rate = {stats.avg_reward_rate:.4f}")

    # Print monitoring summary
    print("\nTraining Monitoring Summary:")
    agent.print_monitoring_summary()

    # Check gradient health
    health = agent.check_gradient_health()
    if health['status'] == 'issues_detected':
        print(f"\nGradient Health Issues: {health['issues']}")

    return agent, reward_history


def train_lp_explore_exploit(
    n_customer_classes: int = 2,
    n_server_classes: int = 2,
    capacity: int = 10,
    n_episodes: int = 100,
    events_per_episode: int = 500,
    explore_episodes: int = 30,
    seed: int = 42
) -> Tuple[Any, List[float]]:
    """
    Train an LP explore-then-exploit agent.

    Returns: (trained_agent, reward_history)
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for LPExploreExploitAgent")

    rng = np.random.default_rng(seed)

    config = make_simple_config(
        n_customer_classes=n_customer_classes,
        n_server_classes=n_server_classes,
        capacity=capacity,
        base_rate=2.0,
        holding_cost=0.1,
        match_reward=1.0,
        compatibility='full'
    )

    queue = TwoSidedQueue(config, rng)
    simulator = LearningSimulator(queue)

    agent = LPExploreExploitAgent(
        config, rng,
        explore_episodes=explore_episodes,
        samples_per_price=events_per_episode,
        n_price_samples=explore_episodes,
        use_quadratic_model=True
    )

    print_agent_info(agent, "LP Explore-Exploit")
    print(f"  Explore episodes: {explore_episodes}")

    reward_history = []

    for episode in range(n_episodes):
        stats = simulator.run_learning_episode(agent, events_per_episode)
        reward_history.append(stats.avg_reward_rate)

        if (episode + 1) % 10 == 0:
            phase = "exploit" if agent.is_exploiting else "explore"
            print(f"Episode {episode + 1} [{phase}]: reward rate = {stats.avg_reward_rate:.4f}")

    # Print final summary
    if agent.is_exploiting:
        summary = agent.get_exploration_summary()
        print(f"\nLP Agent Summary:")
        print(f"  Samples collected: {summary['n_samples']}")
        print(f"  Best observed reward: {summary['best_reward']:.4f}")
        print(f"  Exploit customer prices: {summary['exploit_prices_c']}")
        print(f"  Exploit server prices: {summary['exploit_prices_s']}")

    return agent, reward_history


def train_grid_search(
    n_customer_classes: int = 2,
    n_server_classes: int = 2,
    capacity: int = 10,
    n_episodes: int = 100,
    events_per_episode: int = 500,
    n_grid_points: int = 5,
    seed: int = 42
) -> Tuple[Any, List[float]]:
    """
    Train a grid search agent.

    Returns: (trained_agent, reward_history)
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy required for GridSearchAgent")

    rng = np.random.default_rng(seed)

    config = make_simple_config(
        n_customer_classes=n_customer_classes,
        n_server_classes=n_server_classes,
        capacity=capacity,
        base_rate=2.0,
        holding_cost=0.1,
        match_reward=1.0,
        compatibility='full'
    )

    queue = TwoSidedQueue(config, rng)
    simulator = LearningSimulator(queue)

    agent = GridSearchAgent(
        config, rng,
        n_grid_points=n_grid_points,
        samples_per_price=events_per_episode
    )

    print_agent_info(agent, "Grid Search")
    print(f"  Grid points: {n_grid_points} (total combinations: {n_grid_points**2})")

    reward_history = []

    for episode in range(n_episodes):
        stats = simulator.run_learning_episode(agent, events_per_episode)
        reward_history.append(stats.avg_reward_rate)

        if (episode + 1) % 10 == 0:
            phase = "exploit" if agent.is_exploiting else "explore"
            print(f"Episode {episode + 1} [{phase}]: reward rate = {stats.avg_reward_rate:.4f}")

    return agent, reward_history


def compare_agents(
    n_customer_classes: int = 2,
    n_server_classes: int = 2,
    capacity: int = 10,
    n_episodes: int = 100,
    events_per_episode: int = 500,
    seed: int = 42
) -> Dict[str, List[float]]:
    """
    Train and compare all agent types.

    Returns: dict of {agent_name: reward_history}
    """
    results = {}

    print("=" * 60)
    print("Training Linear Policy Gradient")
    print("=" * 60)
    _, history = train_policy_gradient(
        n_customer_classes, n_server_classes, capacity,
        n_episodes, events_per_episode, seed
    )
    results['Linear PG'] = history

    print("\n" + "=" * 60)
    print("Training Tabular Policy Gradient")
    print("=" * 60)
    _, history = train_tabular_policy_gradient(
        n_customer_classes, n_server_classes, capacity,
        n_episodes, events_per_episode, seed
    )
    results['Tabular PG'] = history

    if TORCH_AVAILABLE:
        print("\n" + "=" * 60)
        print("Training Deep Lattice Network")
        print("=" * 60)
        _, history = train_deep_lattice(
            n_customer_classes, n_server_classes, capacity,
            n_episodes, events_per_episode, seed
        )
        results['Deep Lattice'] = history

        print("\n" + "=" * 60)
        print("Training MLP (no monotonicity)")
        print("=" * 60)
        _, history = train_mlp(
            n_customer_classes, n_server_classes, capacity,
            n_episodes, events_per_episode, seed
        )
        results['MLP'] = history

    if SCIPY_AVAILABLE:
        print("\n" + "=" * 60)
        print("Training LP Explore-Exploit")
        print("=" * 60)
        explore_eps = min(30, n_episodes // 3)
        _, history = train_lp_explore_exploit(
            n_customer_classes, n_server_classes, capacity,
            n_episodes, events_per_episode, explore_eps, seed
        )
        results['LP Explore-Exploit'] = history

        print("\n" + "=" * 60)
        print("Training Grid Search")
        print("=" * 60)
        _, history = train_grid_search(
            n_customer_classes, n_server_classes, capacity,
            n_episodes, events_per_episode, n_grid_points=5, seed=seed
        )
        results['Grid Search'] = history

    return results


def plot_comparison(results: Dict[str, List[float]], window: int = 10, savepath: str = None):
    """Plot learning curves for all agents."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10.colors

    for i, (name, history) in enumerate(results.items()):
        color = colors[i % len(colors)]
        ax.plot(history, alpha=0.2, color=color)

        if len(history) >= window:
            smoothed = np.convolve(history, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(history)), smoothed, label=name, color=color, linewidth=2)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward Rate')
    ax.set_title('Agent Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if savepath:
        fig.savefig(savepath, bbox_inches='tight', dpi=150)

    return fig, ax


def plot_gradient_stats(agent, title: str = "Gradient Statistics", savepath: str = None):
    """Plot gradient and loss statistics over training."""
    if not hasattr(agent, 'monitoring') or agent.monitoring['update_count'] == 0:
        print("No monitoring data available")
        return None, None

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Loss over training
    ax = axes[0, 0]
    losses = agent.monitoring['loss']
    ax.plot(losses, alpha=0.3, label='Raw')
    if len(losses) >= 10:
        smoothed = np.convolve(losses, np.ones(10)/10, mode='valid')
        ax.plot(range(9, len(losses)), smoothed, label='Smoothed', linewidth=2)
    ax.set_xlabel('Update')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gradient norm over training
    ax = axes[0, 1]
    grad_norms = agent.monitoring['grad_norm']
    ax.plot(grad_norms, alpha=0.3, label='Raw')
    if len(grad_norms) >= 10:
        smoothed = np.convolve(grad_norms, np.ones(10)/10, mode='valid')
        ax.plot(range(9, len(grad_norms)), smoothed, label='Smoothed', linewidth=2)
    ax.set_xlabel('Update')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Total Gradient Norm')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Per-layer weight norms (most recent snapshot)
    ax = axes[1, 0]
    if agent.monitoring['weight_stats']:
        recent_weights = agent.monitoring['weight_stats'][-1]
        layer_names = [k for k in recent_weights.keys() if k != 'total_norm']
        norms = [recent_weights[k]['norm'] for k in layer_names]

        # Shorten layer names for display
        short_names = [n.split('.')[-1] if len(n) > 20 else n for n in layer_names]
        y_pos = np.arange(len(layer_names))
        ax.barh(y_pos, norms)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_names, fontsize=8)
        ax.set_xlabel('Weight Norm')
        ax.set_title('Per-Layer Weight Norms')
    else:
        ax.text(0.5, 0.5, 'No weight data', ha='center', va='center', transform=ax.transAxes)

    # Per-layer gradient norms (most recent snapshot)
    ax = axes[1, 1]
    if agent.monitoring['grad_stats']:
        recent_grads = agent.monitoring['grad_stats'][-1]
        layer_names = [k for k in recent_grads.keys() if k != 'total_norm']
        norms = [recent_grads[k]['norm'] for k in layer_names]

        short_names = [n.split('.')[-1] if len(n) > 20 else n for n in layer_names]
        y_pos = np.arange(len(layer_names))
        ax.barh(y_pos, norms)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_names, fontsize=8)
        ax.set_xlabel('Gradient Norm')
        ax.set_title('Per-Layer Gradient Norms')
    else:
        ax.text(0.5, 0.5, 'No gradient data', ha='center', va='center', transform=ax.transAxes)

    fig.suptitle(title)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, bbox_inches='tight', dpi=150)

    return fig, axes


def plot_learning_curve(reward_history: List[float], title: str = "Learning Curve",
                        window: int = 10, savepath: str = None):
    """Plot learning curve with smoothing."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Raw rewards
    ax.plot(reward_history, alpha=0.3, label='Raw')

    # Smoothed
    if len(reward_history) >= window:
        smoothed = np.convolve(reward_history, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(reward_history)), smoothed, label=f'Smoothed (window={window})')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward Rate')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if savepath:
        fig.savefig(savepath, bbox_inches='tight', dpi=150)

    return fig, ax


def plot_policy(agent, savepath: str = None):
    """Plot the learned pricing policy."""
    # Handle different agent types
    if isinstance(agent, LinearPolicyGradientAgent):
        policy = agent.get_policy()
        config = policy.config
        K = config.n_customer_classes
        L = config.n_server_classes

        customer_prices = [policy.customer_prices[i] for i in range(K)]
        server_prices = [policy.server_prices[j] for j in range(L)]

    elif isinstance(agent, TabularPolicyGradientAgent):
        policy = agent.get_greedy_policy()
        config = policy.config
        K = config.n_customer_classes
        L = config.n_server_classes

        customer_prices = [policy.customer_prices[i] for i in range(K)]
        server_prices = [policy.server_prices[j] for j in range(L)]

    elif TORCH_AVAILABLE and isinstance(agent, (DeepLatticeAgent, MLPAgent)):
        config = agent.config
        K = config.n_customer_classes
        L = config.n_server_classes

        # Extract prices by evaluating the network at each state
        customer_prices = []
        for i in range(K):
            prices = []
            for count in range(config.customer_capacities[i] + 1):
                n = np.zeros(K, dtype=int)
                m = np.zeros(L, dtype=int)
                n[i] = count
                p_c, _ = agent.get_deterministic_prices(n, m)
                prices.append(p_c[i])
            customer_prices.append(np.array(prices))

        server_prices = []
        for j in range(L):
            prices = []
            for count in range(config.server_capacities[j] + 1):
                n = np.zeros(K, dtype=int)
                m = np.zeros(L, dtype=int)
                m[j] = count
                _, p_s = agent.get_deterministic_prices(n, m)
                prices.append(p_s[j])
            server_prices.append(np.array(prices))

    else:
        raise ValueError(f"Unknown agent type: {type(agent)}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Customer prices
    ax = axes[0]
    for i in range(K):
        counts = np.arange(config.customer_capacities[i] + 1)
        ax.plot(counts, customer_prices[i], marker='o', label=f'Class {i}')

    ax.set_xlabel('Customer Count')
    ax.set_ylabel('Price')
    ax.set_title('Customer Pricing Policy')
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Server prices
    ax = axes[1]
    for j in range(L):
        counts = np.arange(config.server_capacities[j] + 1)
        ax.plot(counts, server_prices[j], marker='o', label=f'Class {j}')

    ax.set_xlabel('Server Count')
    ax.set_ylabel('Price')
    ax.set_title('Server Pricing Policy')
    ax.set_ylim(-1.1, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, bbox_inches='tight', dpi=150)

    return fig, axes


def check_monotonicity(agent) -> Dict[str, bool]:
    """Check if an agent's policy is monotonic."""
    config = agent.config
    K = config.n_customer_classes
    L = config.n_server_classes

    results = {}

    # Check customer prices
    for i in range(K):
        prices = []
        for count in range(config.customer_capacities[i] + 1):
            n = np.zeros(K, dtype=int)
            m = np.zeros(L, dtype=int)
            n[i] = count
            if hasattr(agent, 'get_deterministic_prices'):
                p_c, _ = agent.get_deterministic_prices(n, m)
            else:
                p_c, _ = agent.get_prices(n, m)
            prices.append(p_c[i])
        diffs = np.diff(prices)
        results[f'customer_{i}'] = bool(np.all(diffs >= -1e-6))

    # Check server prices
    for j in range(L):
        prices = []
        for count in range(config.server_capacities[j] + 1):
            n = np.zeros(K, dtype=int)
            m = np.zeros(L, dtype=int)
            m[j] = count
            if hasattr(agent, 'get_deterministic_prices'):
                _, p_s = agent.get_deterministic_prices(n, m)
            else:
                _, p_s = agent.get_prices(n, m)
            prices.append(p_s[j])
        diffs = np.diff(prices)
        results[f'server_{j}'] = bool(np.all(diffs >= -1e-6))

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Two-sided pricing queue experiments')
    parser.add_argument('--mode', choices=['baselines', 'compare', 'linear', 'tabular', 'lattice', 'mlp'],
                        default='compare', help='Which experiment to run')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--capacity', type=int, default=10, help='Queue capacity per class')
    parser.add_argument('--classes', type=int, default=2, help='Number of customer/server classes')
    parser.add_argument('--save', type=str, default=None, help='Save plots to this directory')
    parser.add_argument('--show-gradients', action='store_true', help='Show gradient monitoring plots (for deep RL agents)')

    args = parser.parse_args()

    if args.mode == 'baselines':
        print("=" * 60)
        print("Evaluating baseline policies")
        print("=" * 60)
        results = evaluate_baselines(
            n_customer_classes=args.classes,
            n_server_classes=args.classes,
            capacity=args.capacity
        )

    elif args.mode == 'compare':
        results = compare_agents(
            n_customer_classes=args.classes,
            n_server_classes=args.classes,
            capacity=args.capacity,
            n_episodes=args.episodes
        )

        # Print final performance
        print("\n" + "=" * 60)
        print("Final Performance (last 10 episodes)")
        print("=" * 60)
        for name, history in results.items():
            final_avg = np.mean(history[-10:])
            print(f"  {name}: {final_avg:.4f}")

        # Plot comparison
        try:
            savepath = f"{args.save}/comparison.png" if args.save else None
            plot_comparison(results, savepath=savepath)
            plt.show()
        except Exception as e:
            print(f"Plotting failed: {e}")

    elif args.mode == 'linear':
        print("=" * 60)
        print("Training Linear Policy Gradient")
        print("=" * 60)
        agent, history = train_policy_gradient(
            n_customer_classes=args.classes,
            n_server_classes=args.classes,
            capacity=args.capacity,
            n_episodes=args.episodes
        )

        print("\nMonotonicity check:", check_monotonicity(agent))

        try:
            plot_learning_curve(history, "Linear Policy Gradient")
            plot_policy(agent)
            plt.show()
        except Exception as e:
            print(f"Plotting failed: {e}")

    elif args.mode == 'tabular':
        print("=" * 60)
        print("Training Tabular Policy Gradient")
        print("=" * 60)
        agent, history = train_tabular_policy_gradient(
            n_customer_classes=args.classes,
            n_server_classes=args.classes,
            capacity=args.capacity,
            n_episodes=args.episodes
        )

        print("\nMonotonicity check:", check_monotonicity(agent))

        try:
            plot_learning_curve(history, "Tabular Policy Gradient")
            plot_policy(agent)
            plt.show()
        except Exception as e:
            print(f"Plotting failed: {e}")

    elif args.mode == 'lattice':
        if not TORCH_AVAILABLE:
            print("PyTorch not available, cannot run Deep Lattice experiment")
        else:
            print("=" * 60)
            print("Training Deep Lattice Network")
            print("=" * 60)
            agent, history = train_deep_lattice(
                n_customer_classes=args.classes,
                n_server_classes=args.classes,
                capacity=args.capacity,
                n_episodes=args.episodes
            )

            print("\nMonotonicity check:", check_monotonicity(agent))

            try:
                plot_learning_curve(history, "Deep Lattice Network")
                plot_policy(agent)
                if args.show_gradients:
                    savepath = f"{args.save}/gradients_lattice.png" if args.save else None
                    plot_gradient_stats(agent, "Deep Lattice Network - Gradient Stats", savepath)
                plt.show()
            except Exception as e:
                print(f"Plotting failed: {e}")

    elif args.mode == 'mlp':
        if not TORCH_AVAILABLE:
            print("PyTorch not available, cannot run MLP experiment")
        else:
            print("=" * 60)
            print("Training MLP (no monotonicity)")
            print("=" * 60)
            agent, history = train_mlp(
                n_customer_classes=args.classes,
                n_server_classes=args.classes,
                capacity=args.capacity,
                n_episodes=args.episodes
            )

            print("\nMonotonicity check:", check_monotonicity(agent))

            try:
                plot_learning_curve(history, "MLP (no monotonicity)")
                plot_policy(agent)
                if args.show_gradients:
                    savepath = f"{args.save}/gradients_mlp.png" if args.save else None
                    plot_gradient_stats(agent, "MLP - Gradient Stats", savepath)
                plt.show()
            except Exception as e:
                print(f"Plotting failed: {e}")
