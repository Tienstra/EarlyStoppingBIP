from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax import random, vmap, jit
import seaborn as sns
from functools import partial
from typing import Tuple, Callable, Optional, Dict, Any

from matplotlib import pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Utils.tools import compute_second_order_diff, compute_laplace


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

    def log_likelihood(self, theta, y, sigma_noise=1.0):
        """
        Compute log-likelihood assuming Gaussian noise.

        Args:
            theta: Model parameters
            y: Observed data
            sigma_noise: Standard deviation of observation noise

        Returns:
            Log-likelihood value
        """
        y_pred = self.evaluate(theta)
        residuals = y - y_pred
        return -0.5 * jnp.sum(residuals ** 2) / sigma_noise ** 2


class LinearForwardModel(ForwardModel):
    def __init__(self, dim_theta, dim_y, p, coef=1):
        super().__init__(dim_theta, dim_y)
        self.p = p
        self.coef = coef
        self.operator = self._get_operator()

    def _get_operator(self):
        i = jnp.linspace(1, self.dim_theta, self.dim_theta)
        gi = jnp.apply_along_axis(lambda x: self.coef * (x ** (-self.p)), 0, i)
        #gi = jnp.zeros(self.dim_theta)
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
    def __init__(self, dim_theta, dim_y, f_array, g_array, plot):
        super().__init__(dim_theta, dim_y)

        self.D = dim_y
        self.L = 2 * jnp.pi
        self.h = self.L / self.D  # Grid spacing
        self.f_array = f_array
        self.g_array = g_array
        self.plot = plot
        self.operator = self._get_operator()

    """
            Initialize the Schrödinger model.

            Args:
                grid_size: Number of grid points (D).
                interval_length: Length of the spatial domain.
                bc_type: Type of boundary conditions ('dirichlet' or 'periodic').
            """

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

    def _get_operator(self) -> jnp.ndarray:
        """
        Build the discretized Schrödinger operator matrix.

        Args:
            D: Size of the grid.
            h: Grid spacing parameter.

        Returns:
            The discretized operator matrix.
        """
        laplace  = compute_laplace(self.D, self.h)
        negative_laplace = (-1)*laplace
        if self.plot:
            sns.heatmap(negative_laplace)
            plt.show()

        return negative_laplace

    # def evaluate_single(self, f_potential) -> jnp.ndarray:
    #     """
    #     Evaluate the model for a single parameter vector.
    #
    #     Args:
    #         theta: Parameter vector representing potential function.
    #
    #     Returns:
    #         Solution of the Schrödinger equation.
    #     """
    #
    #     # Create identity matrix for boundary conditions
    #     I = jnp.zeros((self.D + 1, self.D + 1))
    #
    #     # Get diagonal indices and set values
    #     diag_indices = kth_diag_indices(I, 0)
    #     I = I.at[diag_indices].set(f_potential)
    #
    #     # Create the operator matrix L (note: L was undefined in original code, using operator)
    #     L = self.operator - I
    #
    #     # Solve the system
    #     solution = jnp.linalg.pinv(L) @ self.g_array
    #     return solution

    def evaluate_single_dirichlet(self, f_potential):
        """
        Evaluate the Schrödinger equation for a single potential function.
        Matches the structure of your evaluate_single function.

        Args:
            operator: The Laplacian operator matrix (D+1 x D+1)
            f_potential: Array of potential values at grid points
            g_array: Boundary condition array

        Returns:
            Solution of the Schrödinger equation.
        """
        # Create diagonal matrix for potential
        I = jnp.zeros((len(f_potential), len(f_potential)))

        # Get diagonal indices and set values
        diag_indices = kth_diag_indices(I, 0)
        I = I.at[diag_indices].set(f_potential)

        # Create the operator matrix L: (1/2)Δ - f
        L = self.operator - I

        # Apply Dirichlet boundary conditions
        # Modify first row: u[0] = g_array[0]
        L = L.at[0, :].set(0)
        L = L.at[0, 0].set(1)

        # Modify last row: u[-1] = g_array[-1]
        L = L.at[-1, :].set(0)
        L = L.at[-1, -1].set(1)

        # Solve the system
        solution = jnp.linalg.pinv(L) @ self.g_array
        return solution

    def evaluate(self, ensemble) -> jnp.ndarray:
        """
        Evaluate the model for an ensemble of parameters.

        Args:
            ensemble: Parameter ensemble of shape (dim_parameters, num_particles).

        Returns:
            Solutions for each parameter set in the ensemble.
        """



        # Apply the model to each particle in the ensemble
        # Using vmap for vectorization
        batched_evaluate = vmap(self.evaluate_single_dirichlet, in_axes=1, out_axes=1)
        outputs = batched_evaluate(ensemble)

        return outputs




    def log_likelihood(self, theta, y, sigma_noise=1.0):
        """
        Compute log-likelihood assuming Gaussian noise.

        Args:
            theta: Model parameters
            y: Observed data
            sigma_noise: Standard deviation of observation noise

        Returns:
            Log-likelihood value
        """
        y_pred = self.evaluate_single(theta)
        residuals = y - y_pred
        return -0.5 * jnp.sum(residuals ** 2) / sigma_noise ** 2



if __name__ == "__main__":
    D = 10
    L = 2 * jnp.pi
    x_indices = jnp.arange(D + 1)
    x_array = (2 * jnp.pi * x_indices) / (D + 1)
    f_array = jnp.exp(0.5 * jnp.sin(x_array))
    g_array = jnp.zeros(D +1)
    plot = True

    model = Schroedinger(D, D, f_array, g_array, plot)


    # Create ensemble of potentials (dim_parameters, num_particles)
    num_particles = 3
    ensemble = jnp.ones((D + 1, num_particles)) * 2.0  # simple test: all potentials = 2


    # Call the evaluate function
    outputs = model.evaluate(ensemble)

    print("Outputs shape:", outputs.shape)  # Should be (dim_y + 1, num_particles)
    print("Outputs:")
    print(outputs)
