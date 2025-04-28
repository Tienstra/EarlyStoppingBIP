from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from jax import random, vmap, jit
import numpy as np
import seaborn as sns
from functools import partial
from typing import Tuple, Callable, Optional, Dict, Any

from matplotlib import pyplot as plt


class ForwardModel(ABC):
    def __init__(self, dim_theta, dim_y):
        self.dim_theta = dim_theta
        self.dim_y = dim_y

    def __str__(self) -> str:
        return f"ForwardModel(dim_theta={self.dim_theta}, dim_y={self.dim_y})"

    def __repr__(self) -> str:
        return f"ForwardModel(dim_theta={self.dim_theta}, dim_y={self.dim_y})"

    @abstractmethod
    def _get_operator(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass


class LinearForwardModel(ForwardModel):
    def __init__(self, dim_theta, dim_y, p):
        super().__init__(dim_theta, dim_y)
        self.p = p
        self.operator = self._get_operator()

    def _get_operator(self):
        i = jnp.linspace(1, self.dim_theta, self.dim_theta)
        gi = jnp.apply_along_axis(lambda x: x ** (-self.p), 0, i)
        return jnp.diag(gi)

    def evaluate(self, theta):
        if theta.ndim == 1:
            return self.operator @ theta
        else:
            # For ensemble evaluation (multiple particles)
            return jnp.dot(self.operator, theta)


def kth_diag_indices(a, k):
    """JAX version of kth_diag_indices - returns indices for the kth diagonal"""
    rows, cols = jnp.diag_indices(a.shape[0])
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols


class Schroedinger(ForwardModel):
    def __init__(self, dim_theta, dim_y, f_array, plot):
        super().__init__(dim_theta, dim_y)

        self.D = dim_y
        self.L = 2 * jnp.pi
        self.h = self.L / self.D  # Grid spacing
        self.f_array = f_array
        self.plot = plot
        self.operator = jnp.array(self._get_operator())

    """
            Initialize the Schrödinger model.

            Args:
                grid_size: Number of grid points (D).
                interval_length: Length of the spatial domain.
                bc_type: Type of boundary conditions ('dirichlet' or 'periodic').
            """

    def _build_schrodinger_operator(self) -> np.ndarray:
        """
        Build the discretized Schrödinger operator matrix.

        Args:
            D: Size of the grid.
            h: Grid spacing parameter.

        Returns:
            The discretized operator matrix.
        """
        # Create empty matrix with JAX
        Lap = jnp.zeros((self.D + 1, self.D + 1))

        # Get diagonal indices
        diag_indices = kth_diag_indices(Lap, 0)
        diag_lower_indices = kth_diag_indices(Lap, -1)
        diag_upper_indices = kth_diag_indices(Lap, 1)

        # Create updated arrays with values set at specific indices
        Lap = Lap.at[diag_indices].set(-2)
        Lap = Lap.at[diag_lower_indices].set(1)
        Lap = Lap.at[diag_upper_indices].set(1)

        # Set corner values for periodic boundary
        Lap = Lap.at[0, -1].set(1)
        Lap = Lap.at[-1, 0].set(1)

        # Scale by h²/2
        Lap = Lap / 2 * (self.h**2)

        return Lap

    def _set_potential(self, Lap) -> jnp.ndarray:
        """
        Apply boundary conditions to the operator matrix.

        Args:
            self

        Returns:
            Matrix with boundary conditions applied.
        """
        # Create identity matrix for boundary conditions
        I = jnp.zeros((self.D + 1, self.D + 1))

        # Get diagonal indices
        diag_indices = kth_diag_indices(I, 0)

        I = I.at[diag_indices].set(self.f_array)

        # Subtract from Lap
        Lap_w_boundray = Lap - I
        return Lap_w_boundray

    def _get_operator(self):
        linear_mat = self._build_schrodinger_operator()
        mat_w_potential = self._set_potential(linear_mat)
        if self.plot:
            sns.heatmap(mat_w_potential)
            plt.show()

        return mat_w_potential

    def evaluate_single(self, g_array, f_potential) -> jnp.ndarray:
        """
        Evaluate the model for a single parameter vector.

        Args:
            theta: Parameter vector representing potential function.

        Returns:
            Solution of the Schrödinger equation.
        """

        # Create identity matrix for boundary conditions
        I = jnp.zeros((self.D + 1, self.D + 1))

        # Get diagonal indices and set values
        diag_indices = kth_diag_indices(I, 0)
        I = I.at[diag_indices].set(f_potential)

        # Create the operator matrix L (note: L was undefined in original code, using operator)
        L = self.operator - I

        # Solve the system
        solution = jnp.linalg.pinv(L) @ g_array
        return solution

    def evaluate(self, g_array, ensemble) -> jnp.ndarray:
        """
        Evaluate the model for an ensemble of parameters.

        Args:
            ensemble: Parameter ensemble of shape (dim_parameters, num_particles).

        Returns:
            Solutions for each parameter set in the ensemble.
        """
        # Apply the model to each particle in the ensemble
        # Using vmap for vectorization
        batched_evaluate = vmap(self.evaluate_single, in_axes=(None, 1), out_axes=1)
        outputs = batched_evaluate(g_array, ensemble)

        return outputs


if __name__ == "__main__":
    D = 10
    L = 2 * jnp.pi
    x_indices = jnp.arange(D + 1)
    x_array = (2 * jnp.pi * x_indices) / (D + 1)
    f_array = jnp.exp(0.5 * jnp.sin(x_array))
    plot = False
    model = Schroedinger(D, D, f_array, plot)

    # Create ensemble of potentials (dim_parameters, num_particles)
    num_particles = 3
    ensemble = jnp.ones((D+1, num_particles)) * 2.0  # simple test: all potentials = 2

    # Create g_array (right-hand side)
    g_array = jnp.exp(-(x_array - L / 2) ** 2 / 10) - jnp.exp(-(x_array - L / 2) ** 2 / 10).mean()

    # Call the evaluate function
    outputs = model.evaluate(g_array=g_array, ensemble=ensemble)

    print("Outputs shape:", outputs.shape)  # Should be (dim_y + 1, num_particles)
    print("Outputs:")
    print(outputs)
