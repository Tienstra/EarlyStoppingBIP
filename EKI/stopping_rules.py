import numpy as np
from abc import ABC, abstractmethod


class StoppingRule(ABC):
    def __init__(self, residuals, effective_dim, tolerance, kappa, max_iterations):
        self.residuals = residuals
        self.effective_dim = effective_dim
        self.tolerance = tolerance
        self.kappa = kappa
        self.iteration = 0
        self.max_iterations = max_iterations

    @abstractmethod
    def converged():
        pass


class DiscrepancyPrinciple(StoppingRule):
    def __init__(self, residuals, effective_dim, tolerance, kappa, max_iterations):
        super().__init__(residuals, effective_dim, tolerance, kappa, max_iterations)

    def converged(self):
        self.iteration += 1
        if self.residuals <= self.kappa * (
            np.sqrt(self.effective_dim * self.tolerance)
        ):
            return True
        elif self.residuals > self.kappa * np.sqrt(
            self.effective_dims * self.tolerance
        ):
            return False


DiscrepancyPrinciple(0.02, 4, 1, 1, 4).converged()
