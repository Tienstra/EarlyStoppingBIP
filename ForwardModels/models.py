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
from Utils.tools import compute_second_order_diff


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
        Laplace = compute_second_order_diff(self.D, self.h)
        Schroedinger_mat = self._set_potential(Laplace)
        if self.plot:
            sns.heatmap(Schroedinger_mat)
            plt.show()

        return Schroedinger_mat

    def evaluate_single(self, f_potential) -> jnp.ndarray:
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
        batched_evaluate = vmap(self.evaluate_single, in_axes=(None, 1), out_axes=1)
        outputs = batched_evaluate(self.g_array, ensemble)

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
    plot = False
    fourier = False
    model = Schroedinger(D, D, f_array, plot, fourier)

    modelf = Schroedinger(D, D, f_array, plot, fourier=True)
    print(modelf.foperator)

    # Create ensemble of potentials (dim_parameters, num_particles)
    num_particles = 3
    ensemble = jnp.ones((D + 1, num_particles)) * 2.0  # simple test: all potentials = 2

    # Create g_array (right-hand side)
    g_array = (
        jnp.exp(-((x_array - L / 2) ** 2) / 10)
        - jnp.exp(-((x_array - L / 2) ** 2) / 10).mean()
    )

    # Call the evaluate function
    outputs = model.evaluate(g_array=g_array, ensemble=ensemble)

    print("Outputs shape:", outputs.shape)  # Should be (dim_y + 1, num_particles)
    print("Outputs:")
    print(outputs)
