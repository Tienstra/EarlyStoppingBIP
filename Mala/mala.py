import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
from typing import Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
from functools import partial
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ForwardModels.models import LinearForwardModel, ForwardModel
from Kernels.gaussian_kernels import GaussianKernel


class MALA:
    """
    JAX implementation of Metropolis-Adjusted Langevin Algorithm for parameter estimation.

    This class implements the MALA algorithm for sampling from the posterior
    distribution of parameters given observations with support for multiple chains.
    """

    def __init__(
        self,
        forward_model: ForwardModel,
        observations: jnp.ndarray,
        dim_parameters: int,
        init_covariance: jnp.ndarray,
        init_mean: jnp.ndarray,
        noise_level: float,
        step_size: float = 0.01,
        num_samples: int = 1000,
        num_chains: int = 1,
        burn_in: int = 100,
        rng_key: Optional[jnp.ndarray] = None,
    ):
        """
        Initialize the MALA sampler.

        Args:
            forward_model: Forward model that maps parameters to observations.
            observations: Observed data of shape (dim_observations,).
            dim_parameters: Dimension of the parameter space.
            init_covariance: Initial covariance matrix for parameters (prior).
            init_mean: Initial mean for parameters (prior).
            noise_level: Noise level in the observations.
            step_size: Step size for the Langevin dynamics.
            num_samples: Total number of samples to generate per chain.
            num_chains: Number of independent chains to run.
            burn_in: Number of burn-in samples to discard.
            rng_key: JAX random key for reproducibility.
        """
        self.forward_model = forward_model
        self.observations = observations
        self.dim_observations = observations.shape[0]
        self.dim_parameters = dim_parameters
        self.init_covariance = init_covariance
        self.init_mean = init_mean
        self.noise_level = noise_level
        self.step_size = step_size
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.burn_in = burn_in

        # Random key for reproducibility
        if rng_key is None:
            self.rng_key = random.PRNGKey(0)
        else:
            self.rng_key = rng_key

        # Initialize chains and history
        self.current_samples = self._sample_initial()
        self.samples_history = []
        self.acceptance_history = []
        self.log_posterior_history = []

    def _sample_initial(self) -> jnp.ndarray:
        """
        Sample initial parameters from the prior distribution for all chains.

        Returns:
            Initial parameter samples of shape (num_chains, dim) where dim depends on model type.
        """
        # Generate keys for each chain
        keys = random.split(self.rng_key, self.num_chains)

        def sample_single_chain(key):
            if self.forward_model.type == "nonlinear":
                print(self.forward_model.type)
                # Generate standard normal sample
                sample = random.normal(key, shape=(self.dim_parameters + 1,))
                sample = self.init_mean + jnp.dot(self.init_covariance, sample)
                sample = jnp.exp(sample)
            else:
                sample = random.normal(key, shape=(self.dim_parameters,))
                sample = self.init_mean + jnp.dot(self.init_covariance, sample)

            return sample

        # Use vmap to sample all chains in parallel
        return vmap(sample_single_chain)(keys)

    @partial(jit, static_argnums=(0,))
    def log_prior(self, theta: jnp.ndarray):
        """
        Compute log prior: multivariate Gaussian with mean=0 and given covariance matrix.
        """
        diff = theta - self.init_mean
        precision = jnp.linalg.inv(self.init_covariance)
        return -0.5 * diff @ precision @ diff

    @partial(jit, static_argnums=(0,))
    def log_likelihood(self, theta: jnp.ndarray):
        """
        Compute log likelihood.

        Args:
            theta: Parameter vector.

        Returns:
            Log likelihood.
        """
        # Forward model evaluation
        prediction = self.forward_model.evaluate(theta[:, jnp.newaxis])[:, 0]
        residual = self.observations - prediction
        return -0.5 * jnp.sum(residual**2) / self.noise_level**2

    @partial(jit, static_argnums=(0,))
    def log_posterior(self, theta: jnp.ndarray):
        """
        Compute log posterior density.

        Args:
            theta: Parameter vector.

        Returns:
            Log posterior density.
        """
        return self.log_prior(theta) + self.log_likelihood(theta)

    @partial(jit, static_argnums=(0,))
    def grad_log_posterior(self, theta: jnp.ndarray) -> jnp.ndarray:
        """
        Compute gradient of log posterior.

        Args:
            theta: Parameter vector.

        Returns:
            Gradient of log posterior.
        """
        return grad(self.log_posterior)(theta)

    @partial(jit, static_argnums=(0,))
    def proposal_density_q(self, x_prime, x):
        """
        Proposal density q(x' | x) using the MALA formula:
        q(x' | x) ∝ exp(-1 / 4τ * ||x' - x - τ ∇log π(x)||^2)

        Parameters:
        - x_prime: The proposed state.
        - x: The current state.
        - tau: The step size.

        Returns:
        - log_q: The log of the proposal density q(x' | x).
        """
        grad_log_pi_x = grad(self.log_posterior)(x)
        delta = x_prime - x - (self.step_size) * grad_log_pi_x
        return -jnp.sum(delta**2) / (4 * self.step_size)

    def step(self, rng_keys: jnp.ndarray, x_currents: jnp.ndarray):
        """
        Batched MALA step: Propose and accept/reject for all chains at once.

        Args:
            rng_keys: (num_chains, 2) PRNG keys.
            x_currents: (num_chains, dim_parameters) current samples.

        Returns:
            Tuple of (new_samples, accept_flags)
        """

        @partial(jit, static_argnums=(0,))
        def single_step(rng_key, x_current):
            key1, key2 = random.split(rng_key)

            grad_log_post = self.grad_log_posterior(x_current)
            noise = random.normal(key1, shape=x_current.shape)
            x_proposed = (
                x_current
                + self.step_size * grad_log_post
                + jnp.sqrt(2 * self.step_size) * noise
            )

            log_post_current = self.log_posterior(x_current)
            log_post_proposed = self.log_posterior(x_proposed)

            log_q_current_given_proposed = self.proposal_density_q(
                x_current, x_proposed
            )
            log_q_proposed_given_current = self.proposal_density_q(
                x_proposed, x_current
            )

            log_accept_ratio = (log_post_proposed + log_q_current_given_proposed) - (
                log_post_current + log_q_proposed_given_current
            )

            accept = jnp.log(random.uniform(key2)) < log_accept_ratio
            theta_new = jnp.where(accept, x_proposed, x_current)

            return theta_new, accept

        return vmap(single_step)(rng_keys, x_currents)

    def sample(self) -> Dict[str, Any]:
        """
        Run the MALA algorithm using batched step function.

        Returns:
            Dictionary with results from all chains.
        """
        samples = []
        acceptances = []

        current_thetas = self.current_samples
        key = self.rng_key

        for i in range(self.num_samples):
            key, subkey = random.split(key)
            rng_keys = random.split(subkey, self.num_chains)

            current_thetas, accepted = self.step(rng_keys, current_thetas)

            samples.append(current_thetas)
            acceptances.append(accepted)

            if (i + 1) % 100 == 0:
                recent_acc = jnp.mean(jnp.stack(acceptances[-100:]))
                print(f"Iteration {i + 1}, Recent acceptance rate: {recent_acc:.3f}")

        samples = jnp.stack(samples)  # shape: (num_samples, num_chains, dim)
        acceptances = jnp.stack(acceptances)

        return {
            "samples": samples,
            "burned_samples": samples[self.burn_in :],
            "acceptance_rate": jnp.mean(acceptances),
            "acceptance_history": acceptances,
            "num_chains": self.num_chains,
        }


def main():
    """Run a simple MALA example with multiple chains."""
    # Set random seed for reproducibility
    key = random.PRNGKey(42)

    # Problem dimensions
    dim_parameters = 5
    dim_observations = 3
    num_chains = 4

    # Forward model: simple linear map
    forward_model = LinearForwardModel(dim_parameters, dim_observations, 5)

    # True parameters and synthetic noisy data
    theta_true = jnp.ones(dim_parameters)
    y_true = forward_model.evaluate(theta_true[:, jnp.newaxis])[:, 0]

    # Add Gaussian noise to observations
    noise_level = 0.01
    key, subkey = random.split(key)
    noise = noise_level * random.normal(subkey, y_true.shape)
    observations = y_true + noise

    # Gaussian prior covariance
    prior_cov = GaussianKernel(dim_parameters, 2)._operator_fourier

    # Initialize MALA sampler
    mala = MALA(
        forward_model=forward_model,
        observations=observations,
        dim_parameters=dim_parameters,
        init_covariance=prior_cov,
        init_mean=jnp.zeros(dim_parameters),
        noise_level=noise_level,
        step_size=0.01,
        num_samples=1000,
        num_chains=num_chains,
        burn_in=100,
        rng_key=key,
    )

    # 🔁 Run MALA using the new batched step-based sampling
    results = mala.sample()

    # Output summary
    print(f"\nNumber of chains: {results['num_chains']}")
    print(f"Overall acceptance rate: {results['acceptance_rate']:.3f}")
    print(f"Samples shape: {results['samples'].shape}")

    # Per-chain acceptance rates
    chain_acceptance_rates = jnp.mean(results["acceptance_history"], axis=0)
    print(f"Per-chain acceptance rates: {chain_acceptance_rates}")


if __name__ == "__main__":
    main()
