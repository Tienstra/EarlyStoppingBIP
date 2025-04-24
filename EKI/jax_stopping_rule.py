from typing import Callable, List, Optional, Tuple, Union, Dict, Any
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from jax import random, vmap
import numpy as np


class StoppingRule(ABC):
    """Abstract base class for stopping rules in iterative algorithms."""

    def __init__(
        self,
        tolerance: float,
        max_iterations: int,
    ):
        """
        Initialize the stopping rule.

        Args:
            tolerance: Error tolerance for stopping criterion.
            max_iterations: Maximum number of iterations before forced stopping.
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.iteration = 0

    @abstractmethod
    def check_convergence(self, residuals) -> bool:
        """
        Check if the algorithm has converged based on this stopping rule.

        Returns:
            True if converged, False otherwise.
        """
        pass


class DiscrepancyPrinciple(StoppingRule):
    """
    Morozov's discrepancy principle for early stopping.

    Stops when the residual is below kappa * sqrt(effective_dim * noise_level).
    """

    def __init__(
        self,
        effective_dim: int,
        noise_level: float,
        kappa: float,
        tolerance: float,
        max_iterations: int,
    ):
        """
        Initialize the discrepancy principle stopping rule.

        Args:
            effective_dim: Effective dimension of the observation space.
            noise_level: Estimated noise level in the observations.
            kappa: turning constant
            tolerance: here tolerance is the noise level.
            max_iterations: Maximum number of iterations before forced stopping.
        """
        super().__init__(tolerance, max_iterations)
        self.effective_dim = effective_dim
        self.tolerance = tolerance
        self.kappa = kappa
        self.threshold = self.kappa * (jnp.sqrt(self.effective_dim * self.tolerance))

    def converged(self, residuals) -> bool:
        """
        Check if the algorithm has converged based on the provided code snippet.

        Returns:
            True if converged, False otherwise.
        """
        self.iteration += 1
        while self.iteration < self.max_iterations:

            if residuals <= self.threshold:
                return True
            else:
                return False
