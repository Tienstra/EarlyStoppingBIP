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

def compute_second_order_diff(D,h):
    M = jnp.zeros((self.D + 1, self.D + 1))
    diag_indices = kth_diag_indices(M, 0)
    diag_lower_indices = kth_diag_indices(M, -1)

    M = M.at[diag_indices].set(-1)
    M= M.at[diag_lower_indices].set(1)

    M = M.at[0,-1].set(1)

    M = M / h

    D2 = -M @ M.T

    return D2


def compute_regularised_laplace_covariance(D,h, mu=1e2):
    D2 = compute_second_order_diff(D,h)
    Ones_mat = jnp.ones((D + 1, D + 1))
    gamma = 1 / (4 * h)
    cov = gamma*jnp.linalg.inv((mu*Ones_mat/(D+1))-D2)

    return cov