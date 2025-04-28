
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from jax import random, vmap, jit
import numpy as np
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

class Schroedinger(ForwardModel):
    def __init__(self, dim_theta, dim_y, f_array):
        super().__init__(dim_theta, dim_y)


        self.D = dim_y
        self.L = 2 * jnp.pi
        self.h = self.L / self.D  # Grid spacing
        self.f_array = f_array
        self.operator = self._get_operator()

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
        Lap = np.zeros((self.D + 1, self.D + 1))
        Lap[kth_diag_indices(Lap, 0)] = -2
        Lap[kth_diag_indices(Lap, -1)] = 1
        Lap[kth_diag_indices(Lap, 1)] = 1
        Lap[0, -1] = 1
        Lap[-1, 0] = 1
        Lap = Lap / 2 * (self.h ** 2)

        return Lap


    def _set_boundry(self, Lap) -> jnp.ndarray:
        """
        Apply boundary conditions to the operator matrix.

        Args:
            self

        Returns:
            Matrix with boundary conditions applied.
        """
        # Create identity matrix for boundary conditions
        I = np.zeros((D + 1, D + 1))
        I[kth_diag_indices(I, 0)] = self.f_array

        Lap_w_boudry = Lap - I
        return Lap_w_boudry

    def _get_operator(self):
        linear_mat =  self._build_schrodinger_operator()
        mat_w_boundry = self._set_boundry(linear_mat)

        return mat_w_boundry

    def evaluate_single(self, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the model for a single parameter vector.

        Args:
            theta: Parameter vector representing potential function.

        Returns:
            Solution of the Schrödinger equation.
        """
        # Get potential function
        potential = self._compute_potential(theta)

        # Compute source term
        g = self._compute_source(self.grid_points)

        # Add potential term to operator
        # For Schrödinger: -1/2 ∇² u + V u = g
        L_with_potential = self.operator_with_bc + jnp.diag(potential)

        # Solve the system
        solution = solve_schrodinger(L_with_potential, g)

        return solution

    def evaluate(self, ensemble: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate the model for an ensemble of parameters.

        Args:
            ensemble: Parameter ensemble of shape (dim_parameters, num_particles).

        Returns:
            Solutions for each parameter set in the ensemble.
        """
        # Apply the model to each particle in the ensemble
        # Using vmap for vectorization
        ensemble_evaluate = vmap(self.evaluate_single, in_axes=1, out_axes=1)

        return ensemble_evaluate(ensemble)



def solve_schrodinger(L: jnp.ndarray, g_array: jnp.ndarray) -> jnp.ndarray:
    """
    Solve the Schrödinger equation Lu = g.

    Args:
        L: The operator matrix with boundary conditions applied.
        g_array: The right-hand side of the equation.

    Returns:
        The solution u.
    """
    # Using pseudo-inverse to solve the system
    solution = jnp.linalg.pinv(L) @ g_array

    return solution


def kth_diag_indices(a, k):
    rows, cols = np.diag_indices_from(a)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols

if __name__ == "__main__":
    import seaborn as sns
    D = 10
    x_indices = np.arange(D + 1)
    x_array = (2 * np.pi * x_indices) / (D + 1)
    f_array = np.exp(0.5 * np.sin(x_array))
    schrodinger_mat = Schroedinger(D,D, f_array).operator
    sns.heatmap(schrodinger_mat)
    plt.show()
