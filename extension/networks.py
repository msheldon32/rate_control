"""
Monotonic neural network architectures for pricing policies.

Implements Deep Lattice Networks (DLN) that enforce monotonicity by construction:
1. Calibrator: piecewise linear function mapping input to [0,1]
2. Lattice: multilinear interpolation on hypercube with monotonic vertex values
3. DeepLatticeNetwork: composition of calibrators and lattices

Reference: You et al., "Deep Lattice Networks and Partial Monotonic Functions", NeurIPS 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


class MonotonicCalibrator(nn.Module):
    """
    Piecewise linear calibrator that maps input to [0, 1].

    Enforces monotonicity by parameterizing as cumulative sum of positive values.
    """

    def __init__(self, n_keypoints: int = 10, input_min: float = 0.0, input_max: float = 1.0,
                 monotonicity: int = 1):
        """
        Args:
            n_keypoints: number of keypoints for piecewise linear function
            input_min: minimum expected input value
            input_max: maximum expected input value
            monotonicity: 1 for increasing, -1 for decreasing, 0 for none
        """
        super().__init__()

        self.n_keypoints = n_keypoints
        self.input_min = input_min
        self.input_max = input_max
        self.monotonicity = monotonicity

        # Keypoint locations (fixed, evenly spaced)
        self.register_buffer(
            'keypoint_inputs',
            torch.linspace(input_min, input_max, n_keypoints)
        )

        # Learnable keypoint outputs
        # For monotonic: parameterize as base + cumsum of softplus(deltas)
        if monotonicity != 0:
            self.base = nn.Parameter(torch.tensor(0.0))
            self.deltas = nn.Parameter(torch.zeros(n_keypoints - 1))
        else:
            self.keypoint_outputs = nn.Parameter(torch.linspace(0, 1, n_keypoints))

    def get_keypoint_outputs(self) -> torch.Tensor:
        """Get monotonic keypoint output values."""
        if self.monotonicity == 0:
            return torch.sigmoid(self.keypoint_outputs)

        # Cumulative sum of positive deltas ensures monotonicity
        positive_deltas = F.softplus(self.deltas)
        cumsum = torch.cumsum(positive_deltas, dim=0)
        outputs = torch.cat([self.base.unsqueeze(0), self.base + cumsum])

        # Normalize to [0, 1]
        outputs = torch.sigmoid(outputs)

        if self.monotonicity == -1:
            outputs = 1 - outputs

        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Piecewise linear interpolation.

        Args:
            x: input tensor of any shape

        Returns:
            calibrated values in [0, 1], same shape as input
        """
        keypoint_outputs = self.get_keypoint_outputs()

        # Normalize input to [0, n_keypoints-1]
        x_normalized = (x - self.input_min) / (self.input_max - self.input_min)
        x_normalized = torch.clamp(x_normalized, 0, 1)
        x_scaled = x_normalized * (self.n_keypoints - 1)

        # Find left keypoint index
        left_idx = torch.floor(x_scaled).long()
        left_idx = torch.clamp(left_idx, 0, self.n_keypoints - 2)

        # Interpolation weight
        weight = x_scaled - left_idx.float()

        # Linear interpolation between keypoints
        left_vals = keypoint_outputs[left_idx]
        right_vals = keypoint_outputs[left_idx + 1]

        return left_vals + weight * (right_vals - left_vals)


class LatticeLayer(nn.Module):
    """
    Lattice layer: multilinear interpolation on a hypercube.

    Vertex values can be constrained to be monotonic in specified dimensions.
    """

    def __init__(self, input_dim: int, lattice_sizes: List[int],
                 monotonic_dims: Optional[List[int]] = None,
                 monotonic_directions: Optional[List[int]] = None):
        """
        Args:
            input_dim: number of input dimensions
            lattice_sizes: number of vertices along each dimension
            monotonic_dims: which dimensions should be monotonic
            monotonic_directions: 1 for increasing, -1 for decreasing per monotonic dim
        """
        super().__init__()

        self.input_dim = input_dim
        self.lattice_sizes = lattice_sizes
        self.monotonic_dims = monotonic_dims or []
        self.monotonic_directions = monotonic_directions or [1] * len(self.monotonic_dims)

        # Total number of vertices
        n_vertices = 1
        for s in lattice_sizes:
            n_vertices *= s
        self.n_vertices = n_vertices

        # Learnable vertex values
        # For monotonic dims, we'll use cumsum parameterization
        self.vertex_params = nn.Parameter(torch.randn(n_vertices) * 0.1)

        # Precompute index strides for each dimension
        strides = []
        stride = 1
        for s in lattice_sizes:
            strides.append(stride)
            stride *= s
        self.register_buffer('strides', torch.tensor(strides))

    def get_vertex_values(self) -> torch.Tensor:
        """Get vertex values with monotonicity constraints applied."""
        values = self.vertex_params.clone()

        # Apply monotonicity constraints via cumulative sum along monotonic dims
        # This is done by reshaping, applying cumsum, and reshaping back
        if len(self.monotonic_dims) > 0:
            # Reshape to lattice shape
            shape = self.lattice_sizes
            values = values.view(*shape)

            for dim, direction in zip(self.monotonic_dims, self.monotonic_directions):
                # Apply softplus to ensure positive increments, then cumsum
                # First, compute deltas along this dimension
                n_slices = shape[dim]

                # Get base slice and deltas
                indices = [slice(None)] * len(shape)

                # Extract values along this dimension and apply cumsum of softplus
                # This is a simplified version - full implementation would be more careful
                values = self._apply_monotonicity_along_dim(values, dim, direction)

            values = values.reshape(-1)  # Use reshape instead of view for non-contiguous tensors

        return values

    def _apply_monotonicity_along_dim(self, values: torch.Tensor, dim: int,
                                       direction: int) -> torch.Tensor:
        """Apply monotonicity constraint along a specific dimension."""
        # Move target dim to the end for easier manipulation
        values = values.movedim(dim, -1)
        original_shape = values.shape

        # Flatten all but last dim
        values = values.reshape(-1, original_shape[-1])

        # First column is base, rest are deltas
        base = values[:, 0:1]
        if values.shape[1] > 1:
            deltas = F.softplus(values[:, 1:] - values[:, :-1])
            cumsum = torch.cumsum(deltas, dim=1)
            values = torch.cat([base, base + cumsum], dim=1)

        if direction == -1:
            values = values.flip(-1)

        # Restore shape
        values = values.reshape(original_shape)
        values = values.movedim(-1, dim)

        return values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multilinear interpolation on the lattice.

        Args:
            x: input tensor of shape (..., input_dim), values in [0, 1]

        Returns:
            interpolated values, shape (...)
        """
        batch_shape = x.shape[:-1]
        x = x.reshape(-1, self.input_dim)
        batch_size = x.shape[0]

        vertex_values = self.get_vertex_values()

        # Scale inputs to lattice coordinates
        lattice_sizes_t = torch.tensor(self.lattice_sizes, device=x.device, dtype=x.dtype)
        x_scaled = x * (lattice_sizes_t - 1)
        # Clamp to valid range
        x_scaled = torch.minimum(x_scaled, lattice_sizes_t - 1 - 1e-6)
        x_scaled = torch.maximum(x_scaled, torch.zeros_like(x_scaled))

        # Get corner indices and interpolation weights
        x_floor = torch.floor(x_scaled).long()
        weights = x_scaled - x_floor.float()

        # Multilinear interpolation: sum over 2^d corners
        result = torch.zeros(batch_size, device=x.device, dtype=x.dtype)

        lattice_sizes_long = torch.tensor(self.lattice_sizes, device=x.device, dtype=torch.long)
        for corner in range(2 ** self.input_dim):
            # Determine which dimensions use ceiling vs floor
            corner_offset = torch.tensor(
                [(corner >> d) & 1 for d in range(self.input_dim)],
                device=x.device, dtype=torch.long
            )

            # Compute corner indices
            corner_idx = x_floor + corner_offset.unsqueeze(0)
            corner_idx = torch.minimum(corner_idx, lattice_sizes_long - 1)
            corner_idx = torch.maximum(corner_idx, torch.zeros_like(corner_idx))

            # Flatten to 1D index
            flat_idx = (corner_idx * self.strides.unsqueeze(0)).sum(dim=1)

            # Get vertex values
            corner_values = vertex_values[flat_idx]

            # Compute interpolation weight for this corner
            corner_weights = torch.where(
                corner_offset.unsqueeze(0).bool(),
                weights,
                1 - weights
            ).prod(dim=1)

            result = result + corner_weights * corner_values

        return result.reshape(batch_shape)


class DeepLatticeNetwork(nn.Module):
    """
    Deep Lattice Network for monotonic function approximation.

    Architecture:
    1. Per-input calibrators (piecewise linear, can be monotonic)
    2. One or more lattice layers
    3. Optional output calibration
    """

    def __init__(self, input_dim: int,
                 monotonic_dims: List[int],
                 monotonic_directions: List[int],
                 input_mins: List[float],
                 input_maxs: List[float],
                 n_calibrator_keypoints: int = 10,
                 lattice_size: int = 3,
                 output_min: float = -1.0,
                 output_max: float = 1.0):
        """
        Args:
            input_dim: number of inputs
            monotonic_dims: which input dimensions should be monotonic
            monotonic_directions: 1 for increasing, -1 for decreasing
            input_mins: minimum values for each input
            input_maxs: maximum values for each input
            n_calibrator_keypoints: keypoints per calibrator
            lattice_size: vertices per dimension in lattice
            output_min: minimum output value
            output_max: maximum output value
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_min = output_min
        self.output_max = output_max

        # Create calibrators for each input
        self.calibrators = nn.ModuleList()
        for i in range(input_dim):
            mono = 0
            if i in monotonic_dims:
                idx = monotonic_dims.index(i)
                mono = monotonic_directions[idx]

            self.calibrators.append(MonotonicCalibrator(
                n_keypoints=n_calibrator_keypoints,
                input_min=input_mins[i],
                input_max=input_maxs[i],
                monotonicity=mono
            ))

        # Create lattice
        # Map monotonic_dims indices to lattice dimension indices
        lattice_monotonic_dims = list(range(len(monotonic_dims)))
        self.lattice = LatticeLayer(
            input_dim=input_dim,
            lattice_sizes=[lattice_size] * input_dim,
            monotonic_dims=monotonic_dims,
            monotonic_directions=monotonic_directions
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: input tensor of shape (..., input_dim)

        Returns:
            output in [output_min, output_max], shape (...)
        """
        # Calibrate each input
        calibrated = []
        for i, cal in enumerate(self.calibrators):
            calibrated.append(cal(x[..., i]))
        calibrated = torch.stack(calibrated, dim=-1)

        # Pass through lattice
        output = self.lattice(calibrated)

        # Scale to output range
        output = self.output_min + (self.output_max - self.output_min) * torch.sigmoid(output)

        return output


class MonotonicPricingNetwork(nn.Module):
    """
    Neural network for two-sided pricing with monotonicity constraints.

    Outputs K customer prices and L server prices.
    Each price is monotonically non-decreasing in its own class count.
    """

    def __init__(self, n_customer_classes: int, n_server_classes: int,
                 customer_capacities: List[int], server_capacities: List[int],
                 n_calibrator_keypoints: int = 10,
                 lattice_size: int = 3,
                 use_cross_effects: bool = False,
                 compatibility: Optional[np.ndarray] = None):
        """
        Args:
            n_customer_classes: K
            n_server_classes: L
            customer_capacities: max count per customer class
            server_capacities: max count per server class
            n_calibrator_keypoints: keypoints per calibrator
            lattice_size: lattice vertices per dimension
            use_cross_effects: if True, price depends on compatible counts too
            compatibility: K x L binary matrix (only used if use_cross_effects=True)
        """
        super().__init__()

        self.K = n_customer_classes
        self.L = n_server_classes
        self.customer_capacities = customer_capacities
        self.server_capacities = server_capacities
        self.use_cross_effects = use_cross_effects

        if use_cross_effects and compatibility is not None:
            self.register_buffer('compatibility', torch.tensor(compatibility, dtype=torch.float32))
        else:
            self.compatibility = None

        # Create a DLN for each customer class price
        # Price_i depends on n_i (monotonically increasing)
        self.customer_networks = nn.ModuleList()
        for i in range(self.K):
            if use_cross_effects and compatibility is not None:
                # Price depends on own count + compatible server counts
                compatible_servers = np.where(compatibility[i] > 0)[0].tolist()
                input_dim = 1 + len(compatible_servers)
                input_mins = [0.0] + [0.0] * len(compatible_servers)
                input_maxs = [float(customer_capacities[i])] + [float(server_capacities[j]) for j in compatible_servers]
                # Monotonic increasing in n_i (dim 0)
                # Could add decreasing in server counts if desired
                monotonic_dims = [0]
                monotonic_directions = [1]
            else:
                input_dim = 1
                input_mins = [0.0]
                input_maxs = [float(customer_capacities[i])]
                monotonic_dims = [0]
                monotonic_directions = [1]

            self.customer_networks.append(DeepLatticeNetwork(
                input_dim=input_dim,
                monotonic_dims=monotonic_dims,
                monotonic_directions=monotonic_directions,
                input_mins=input_mins,
                input_maxs=input_maxs,
                n_calibrator_keypoints=n_calibrator_keypoints,
                lattice_size=lattice_size,
                output_min=-1.0,
                output_max=1.0
            ))

        # Create a DLN for each server class price
        self.server_networks = nn.ModuleList()
        for j in range(self.L):
            if use_cross_effects and compatibility is not None:
                compatible_customers = np.where(compatibility[:, j] > 0)[0].tolist()
                input_dim = 1 + len(compatible_customers)
                input_mins = [0.0] + [0.0] * len(compatible_customers)
                input_maxs = [float(server_capacities[j])] + [float(customer_capacities[i]) for i in compatible_customers]
                monotonic_dims = [0]
                monotonic_directions = [1]
            else:
                input_dim = 1
                input_mins = [0.0]
                input_maxs = [float(server_capacities[j])]
                monotonic_dims = [0]
                monotonic_directions = [1]

            self.server_networks.append(DeepLatticeNetwork(
                input_dim=input_dim,
                monotonic_dims=monotonic_dims,
                monotonic_directions=monotonic_directions,
                input_mins=input_mins,
                input_maxs=input_maxs,
                n_calibrator_keypoints=n_calibrator_keypoints,
                lattice_size=lattice_size,
                output_min=-1.0,
                output_max=-1e-6  # Server prices always negative (payment)
            ))

        # Store compatible indices for forward pass
        if use_cross_effects and compatibility is not None:
            self._customer_compat_indices = [
                np.where(compatibility[i] > 0)[0].tolist() for i in range(self.K)
            ]
            self._server_compat_indices = [
                np.where(compatibility[:, j] > 0)[0].tolist() for j in range(self.L)
            ]
        else:
            self._customer_compat_indices = None
            self._server_compat_indices = None

    def forward(self, n: torch.Tensor, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute prices for given state.

        Args:
            n: customer counts, shape (..., K)
            m: server counts, shape (..., L)

        Returns:
            prices_c: customer prices, shape (..., K)
            prices_s: server prices, shape (..., L)
        """
        batch_shape = n.shape[:-1]

        prices_c = []
        for i in range(self.K):
            if self.use_cross_effects and self._customer_compat_indices is not None:
                # Gather compatible server counts
                compat_j = self._customer_compat_indices[i]
                inputs = [n[..., i:i+1]]
                for j in compat_j:
                    inputs.append(m[..., j:j+1])
                x = torch.cat(inputs, dim=-1)
            else:
                x = n[..., i:i+1]

            prices_c.append(self.customer_networks[i](x))

        prices_c = torch.stack(prices_c, dim=-1)

        prices_s = []
        for j in range(self.L):
            if self.use_cross_effects and self._server_compat_indices is not None:
                compat_i = self._server_compat_indices[j]
                inputs = [m[..., j:j+1]]
                for i in compat_i:
                    inputs.append(n[..., i:i+1])
                x = torch.cat(inputs, dim=-1)
            else:
                x = m[..., j:j+1]

            prices_s.append(self.server_networks[j](x))

        prices_s = torch.stack(prices_s, dim=-1)

        return prices_c, prices_s


class SimpleMLP(nn.Module):
    """
    Simple MLP baseline (no monotonicity guarantees).
    For comparison with monotonic networks.
    """

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: List[int] = [64, 64],
                 output_min: float = -1.0, output_max: float = 1.0):
        super().__init__()

        self.output_min = output_min
        self.output_max = output_max

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        # Scale to output range
        return self.output_min + (self.output_max - self.output_min) * torch.sigmoid(out)


class MLPPricingNetwork(nn.Module):
    """
    MLP baseline for pricing (no monotonicity).

    Customer prices in [-1, 1], server prices in [-1, 0) (always negative).
    """

    def __init__(self, n_customer_classes: int, n_server_classes: int,
                 hidden_dims: List[int] = [64, 64]):
        super().__init__()

        self.K = n_customer_classes
        self.L = n_server_classes

        input_dim = n_customer_classes + n_server_classes

        # Separate networks for customer and server prices with different output ranges
        self.customer_network = SimpleMLP(input_dim, n_customer_classes, hidden_dims,
                                          output_min=-1.0, output_max=1.0)
        self.server_network = SimpleMLP(input_dim, n_server_classes, hidden_dims,
                                        output_min=-1.0, output_max=-1e-6)

    def forward(self, n: torch.Tensor, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([n.float(), m.float()], dim=-1)
        prices_c = self.customer_network(x)
        prices_s = self.server_network(x)
        return prices_c, prices_s
