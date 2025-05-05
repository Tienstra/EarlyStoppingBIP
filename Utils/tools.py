import jax.numpy as jnp


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


if __name__ == "__main__":
    D = 2
    lap = compute_laplace(D, 0.5)
    D2 = compute_second_order_diff(D, 0.5)
    print(lap)
    print(D2)
