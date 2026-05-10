"""
Two-sided pricing queue with compatibilities.

Extension of the rate control framework to multi-class queues with:
- Customer classes with individual pricing
- Server classes with individual pricing
- Compatibility graph for matching
- Monotonicity constraints on prices

Neural network architectures:
- Deep Lattice Networks for monotonic function approximation
- MLP baseline for comparison
"""

from .model import ModelConfig, TwoSidedQueue, DemandCurve, make_simple_config
from .policy import (
    Policy,
    RandomPolicy,
    ConstantPolicy,
    LinearMonotonicPolicy,
    TabularMonotonicPolicy,
    ThresholdPolicy
)
from .simulator import Simulator, LearningSimulator, SimulationStats
from .agent import (
    Agent,
    RandomAgent,
    ConstantAgent,
    TabularPolicyGradientAgent,
    LinearPolicyGradientAgent,
    TORCH_AVAILABLE,
    SCIPY_AVAILABLE,
)

# Conditionally import PyTorch-based agents
if TORCH_AVAILABLE:
    from .agent import DeepLatticeAgent, MLPAgent
    from .networks import (
        MonotonicCalibrator,
        LatticeLayer,
        DeepLatticeNetwork,
        MonotonicPricingNetwork,
        MLPPricingNetwork,
    )

# Conditionally import scipy-based agents
if SCIPY_AVAILABLE:
    from .agent import LPExploreExploitAgent, GridSearchAgent

__all__ = [
    # Model
    'ModelConfig',
    'TwoSidedQueue',
    'DemandCurve',
    'make_simple_config',
    # Policies
    'Policy',
    'RandomPolicy',
    'ConstantPolicy',
    'LinearMonotonicPolicy',
    'TabularMonotonicPolicy',
    'ThresholdPolicy',
    # Simulation
    'Simulator',
    'LearningSimulator',
    'SimulationStats',
    # Agents
    'Agent',
    'RandomAgent',
    'ConstantAgent',
    'TabularPolicyGradientAgent',
    'LinearPolicyGradientAgent',
    'TORCH_AVAILABLE',
    'SCIPY_AVAILABLE',
]

# Add PyTorch exports if available
if TORCH_AVAILABLE:
    __all__.extend([
        'DeepLatticeAgent',
        'MLPAgent',
        'MonotonicCalibrator',
        'LatticeLayer',
        'DeepLatticeNetwork',
        'MonotonicPricingNetwork',
        'MLPPricingNetwork',
    ])

# Add scipy exports if available
if SCIPY_AVAILABLE:
    __all__.extend([
        'LPExploreExploitAgent',
        'GridSearchAgent',
    ])
