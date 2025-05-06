import jax.numpy as jnp
from jax import vmap, random, jit


def kth_diag_indices(a, k):
    """JAX version of kth_diag_indices - returns indices for the kth diagonal"""
    rows, cols = jnp.diag_indices(a.shape[0])
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols


def compute_second_order_diff(D, h):
    M = jnp.zeros((D + 1, D + 1))
    diag_indices = kth_diag_indices(M, 0)
    diag_lower_indices = kth_diag_indices(M, -1)

    M = M.at[diag_indices].set(-1)
    M = M.at[diag_lower_indices].set(1)

    M = M.at[0, -1].set(1)

    D2 = -M @ M.T / (2 * (h**2))

    return D2


def compute_inverse_second_order_diff(Lap):
    K = jnp.linalg.pinv(Lap)

    return K


def compute_regularised_laplace_covariance(D, h, mu=1e2):
    D2 = compute_second_order_diff(D, h)
    Ones_mat = jnp.ones((D + 1, D + 1))
    gamma = 1 / (2 * (h**2))
    cov = jnp.linalg.inv((mu * ((gamma) * Ones_mat) / (D + 1)) - D2)

    return cov


def compute_laplace(D, h):
    # Create empty matrix with JAX
    Lap = jnp.zeros((D + 1, D + 1))

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

    # Scale by 2h²
    Lap = Lap / (2 * (h**2))

    return Lap


def get_function(w: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """
    Reconstruct function(s) f(x) = ∑ w_k * φ_k(x) using Laplacian eigenfunctions on [0, 1].

    Parameters:
    ----------
    w : jnp.ndarray
        Coefficients in the Laplace eigenfunction basis.
        Shape: (m,) for single particle or (m, n_particles) for ensemble.
    x : jnp.ndarray
        Points at which to evaluate the function (shape: [M,], values in [0, 1])
    plot : bool
        If True and w is a single vector, plot the reconstructed function.

    Returns:
    -------
    f : jnp.ndarray
        Reconstructed function(s) evaluated at points x.
    """

    def single_reconstruction(w_single):
        k_vals = jnp.arange(1, len(w_single) + 1)  # shape (m,)
        Phi = jnp.sqrt(2) * jnp.sin(jnp.outer(k_vals, jnp.pi * x))  # shape (m, M)
        return jnp.dot(w_single, Phi)  # shape (M,)

    if w.ndim == 1:
        f = single_reconstruction(w)  # shape (M,)
        return f

    else:
        # w: (m, n_particles) → transpose → (n_particles, m)
        batched_eval = vmap(single_reconstruction, in_axes=(1,), out_axes=1)
        f_all = batched_eval(w)  # shape: (n_particles, M)
        return f_all  # shape: (M, n_particles)
def solution_map(v: jnp.ndarray, K: jnp.ndarray, g_tilde: jnp.ndarray) -> jnp.ndarray:
    """ """

    def solution_map_single(v, K, g_tilde):
        denominator = jnp.dot(K, v) + g_tilde
        return v / 2 * (denominator)

    if v.ndim == 1:
        f = solution_map_single(v, K, g_tilde)  # shape (M,)
        return f

    else:
        # w: (m, n_particles) → transpose → (n_particles, m)
        batched_eval = vmap(solution_map_single, in_axes=(1, None, None), out_axes=1)
        f_all = batched_eval(v, K, g_tilde)  # shape: (n_particles, M)
        return f_all  # shape: (M, n_particles)



if __name__ == "__main__":
    D = 2
    lap = compute_laplace(D, 0.5)
    D2 = compute_second_order_diff(D, 0.5)
    print(lap)
    print(D2)
