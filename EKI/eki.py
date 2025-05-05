import jax
import jax.numpy as jnp
from jax import random, vmap
from typing import Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from ForwardModels.models import LinearForwardModel, ForwardModel
from Kernels.gaussian_kernels import GaussianKernel
from EKI.stopping_rules import DiscrepancyPrinciple


class EKI:
    """
    JAX implementation of Ensemble Kalman Inversion for parameter estimation.

    This class implements the EKI algorithm for approximating the posterior
    distribution of parameters given observations.
    """

    def __init__(
        self,
        forward_model: ForwardModel,
        observations: jnp.ndarray,
        dim_parameters: int,
        num_particles: int,
        init_covariance: jnp.ndarray,
        init_mean: jnp.ndarray,
        noise_level: float,
        time_interval: Tuple[float, float, int],
        stopping_rule=None,
        rng_key: Optional[jnp.ndarray] = None,
    ):
        """
        Initialize the EKI solver.

        Args:
            forward_model: Forward model that maps parameters to observations.
            observations: Observed data of shape (dim_observations,).
            dim_parameters: Dimension of the parameter space.
            num_particles: Number of particles in the ensemble.
            init_covariance: Initial covariance matrix for parameters.
            init_mean: Initial mean for parameters.
            noise_level: Noise level in the observations.
            time_interval: Tuple of (start_time, end_time, num_steps).
            stopping_rule: Stopping rule function that determines to stop iterations.
            rng_key: JAX random key for reproducibility.
        """
        self.forward_model = forward_model
        self.observations = observations
        self.dim_observations = observations.shape[0]
        self.dim_parameters = dim_parameters
        self.num_particles = num_particles
        self.init_covariance = init_covariance
        self.init_mean = init_mean
        self.noise_level = noise_level
        self.time = time_interval  # [start, end, num points]
        self.stopping_rule = stopping_rule
        self.converged = False

        # Random key for reproducibility
        if rng_key is None:
            self.rng_key = random.PRNGKey(0)
        else:
            self.rng_key = rng_key

        # Initialize ensemble and history
        self.ensemble = self._sample_prior()
        self.ensemble_history = [self.ensemble]
        self.predictions = self._push_forward()
        self.predictions_history = [self.predictions]
        self.residual_history = [self.compute_residual()]

    def _get_dt(self) -> float:
        """
        Get the time step size based on the time interval.

        Returns:
            Time step size.
        """
        return (self.time[1] - self.time[0]) / self.time[2]

    def _sample_prior(self) -> jnp.ndarray:
        """
        Sample the initial ensemble from the prior distribution.

        Returns:
            Initial ensemble of shape (dim_parameters, num_particles).
        """
        # Generate standard normal samples
        ensemble = random.normal(
            self.rng_key, shape=(self.dim_parameters, self.num_particles)
        )

        # Transform to the specified prior distribution
        return self.init_mean[:, jnp.newaxis] + jnp.dot(self.init_covariance, ensemble)

    def generate_noise_cov(self) -> jnp.ndarray:
        """
        Generate the noise covariance matrix.

        Returns:
            Diagonal noise covariance matrix.
        """
        return self.noise_level * jnp.eye(self.dim_observations)

    def _push_forward(self) -> jnp.ndarray:
        """
        Push the current ensemble through the forward model.

        Returns:
            Model predictions for the current ensemble.
        """
        return self.forward_model.evaluate(self.ensemble)

    def compute_kalman_gain(self) -> jnp.ndarray:
        """
        Compute the Kalman gain matrix.

        Returns:
            Kalman gain matrix of shape (dim_parameters, dim_observations).
        """
        # Get current predictions
        predictions = self._push_forward()

        # Compute means
        ensemble_mean = jnp.mean(self.ensemble, axis=1, keepdims=True)
        predictions_mean = jnp.mean(predictions, axis=1, keepdims=True)

        # Center the ensemble and predictions
        centered_ensemble = self.ensemble - ensemble_mean
        centered_predictions = predictions - predictions_mean

        # Compute cross-covariance: C_θG = (θ - θ̄)(G(θ) - G(θ̄))^T / (N-1)
        cross_cov = jnp.dot(centered_ensemble, centered_predictions.T) / (
            self.num_particles - 1
        )

        # Compute prediction covariance: C_GG = (G(θ) - G(θ̄))(G(θ) - G(θ̄))^T / (N-1)
        pred_cov = jnp.dot(centered_predictions, centered_predictions.T) / (
            self.num_particles - 1
        )

        # Add noise covariance scaled by dt
        dt = self._get_dt()
        noise_cov = self.generate_noise_cov() / dt

        # Compute Kalman gain: K = C_θG (C_GG + Γ/dt)^(-1)
        return cross_cov @ jnp.linalg.inv(pred_cov + noise_cov)

    def update_ensemble(self) -> None:
        """Update the ensemble using the Kalman formula."""
        # Get current predictions
        self.predictions = self._push_forward()

        # Compute means
        predictions_mean = jnp.mean(self.predictions, axis=1, keepdims=True)

        # make observations in matrix
        obs_expanded = jnp.tile(
            self.observations[:, jnp.newaxis], (1, self.num_particles)
        )

        # Compute Kalman gain
        kalman_gain = self.compute_kalman_gain()

        # Update ensemble
        innovation = self.predictions + predictions_mean - 2 * obs_expanded
        kalman_update = kalman_gain @ innovation

        # Update the ensemble
        self.ensemble = self.ensemble - 0.5 * kalman_update

        # Update histories
        self.ensemble_history.append(self.ensemble)
        self.predictions_history.append(self.predictions)
        self.residual_history.append(self.compute_residual())

    def compute_residual(self) -> float:
        """
        Compute the residual between observations and predictions.

        Returns:
            Current residual value.
        """
        mean_prediction = jnp.mean(self.predictions, axis=1)
        return jnp.linalg.norm(self.observations - mean_prediction)

    def fit(self, stopping_rule: None) -> Dict[str, Any]:
        """
        Run the EKI algorithm with early stopping.

        Args:
            stopping_rule: Rule to determine when to stop the algorithm.

        Returns:
            dict: keys (stopping_time, converged, residual_history, final_residual, ensemble_history)
        """

        # Run iterations until convergence or max iterations
        ts = jnp.arange(self.time[0], self.time[1], self._get_dt())
        for index in range(0, ts.size - 1):
            print(index)

            # Check for convergence
            if stopping_rule is not None:
                current_residual = self.compute_residual()
                self.converged = stopping_rule.converged(current_residual)
                if self.converged:
                    print("converged")
                    break

                # Update the ensemble
                self.update_ensemble()
            else:
                # Update the ensemble
                self.update_ensemble()

        # results dictionary
        results = {
            "stopping_time": index,
            "converged": self.converged,
            "residual_history": jnp.array(self.residual_history),
            "final_residual": self.residual_history[-1],
            "ensemble_history": self.ensemble_history,
        }

        return results


def main():
    """Run a simple EKI example."""
    # Set random seed for reproducibility
    key = random.PRNGKey(42)

    # Problem dimensions
    dim_parameters = 5
    dim_observations = 3
    num_particles = 50

    forward_model = LinearForwardModel(dim_parameters, dim_observations, 5)

    theta_true = jnp.ones(dim_parameters)

    y_true = forward_model.evaluate(theta_true[:, jnp.newaxis])[:, 0]

    # Add noise to observations
    noise_level = 0.01
    key, subkey = random.split(key)
    noise = noise_level * random.normal(subkey, y_true.shape)
    observations = y_true + noise
    prior_cov = GaussianKernel(dim_parameters, 2)._operator_fourier
    # Initialize EKI solver
    eki = EKI(
        forward_model=forward_model,
        observations=observations,
        dim_parameters=dim_parameters,
        num_particles=num_particles,
        init_covariance=prior_cov,
        init_mean=jnp.zeros(dim_parameters),
        noise_level=noise_level,
        time_interval=(0.0, 1.0, 50),  # (start, end, max_steps)
        rng_key=key,
    )

    # Create stopping rule: Discrepancy principle
    stopping_rule = DiscrepancyPrinciple(
        effective_dim=dim_observations,
        tolerance=noise_level,
        kappa=1.1,
        max_iterations=50,
    )

    # Run the algorithm with early stopping
    results = eki.fit(stopping_rule=stopping_rule)

    # Print results
    print(f"\nAlgorithm converged: {results['converged']}")
    print(f"Stopping time: {results['stopping_time']}")
    print(f"Final residual: {results['final_residual']:.6f}")


if __name__ == "__main__":
    main()
