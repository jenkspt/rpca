import itertools
import numpy as np


def l1_norm(x, axis=(-2, -1), keepdims=True):
    return np.linalg.norm(x, 1, axis, keepdims)


def frobenius_norm(x, axis=(-2, -1), keepdims=True):
    return np.linalg.norm(x, 'fro', axis, keepdims)


def robust_pca(M, epsilon: float=10e-6, max_iter: int=None):
    """ 
    Decompose a matrix into low rank and sparse components
    L and S, using Alternating Lagrangian Multipliers, such that
    L + S ~= M.

    Parameters
    ----------
    M : (..., M, N) array_like
        A real or complex array with `a.ndim >=2`
    epsilon : float
        Converges when ||M - L - S|| / ||M|| < epsilon
        where ||M|| is the frobenius norm of M
    max_iter: int
        Terminates when maximum numer of iterations is exceeded

    Returns
    -------
    L : (..., M, N) array
        The low-rank component of M
    S : (..., M, N) array
        The sparse component of M
    """
    counter = itertools.count()
    if max_iter is None:
        max_iter = float('inf')

    L = np.zeros(M.shape)
    S = np.zeros(M.shape)
    Y = np.zeros(M.shape)
    mu = np.product(M.shape[-2:]) / (4.0 * l1_norm(M))
    lamb = max(M.shape[-2:]) ** -0.5
    while next(counter) < max_iter and \
            not np.all(converged(M, L, S, epsilon)):
        mu_inv = np.reciprocal(mu)
        L = svd_shrink(M - S - mu_inv * Y, np.squeeze(mu, -1))

        M_L = M - L
        S = shrink(M_L + mu_inv * Y, lamb * mu)
        Y = Y + mu * (M_L - S)
    return L, S

    
def svd_shrink(X, tau):
    """
    Apply the shrinkage operator to the singular values obtained 
    from the SVD of X.

    Computes `V = U @ shrink(s) @ Vh`
    where U, s, Vh are optained from `np.linalg.svd`

    Parameters
    ----------
    X : (..., M, N) array_like
        A real or complex array with `a.ndim >= 2`
    tau: array_like
        The scaling parameter to the shrink function

    Returns
    -------
    V : (..., M, N) array
    """
    U, s, Vh = np.linalg.svd(X, full_matrices=False)
    #return (U * np.expand_dims(shrink(s, tau), -2)) @ Vh
    return np.einsum('...mn,...n,...nn->...mn', U, shrink(s, tau), Vh)


def shrink(X, tau):
    """
    Apply the shrinkage operator the the elements of X.
    Returns V such that V[i,j] = max(abs(X[i,j]) - tau, 0).
    """
    return np.copysign(np.maximum(np.abs(X) - tau, 0), X)


def converged(M, L, S, epsilon: float):
    """
    A simple test of convergence based on accuracy of matrix reconstruction
    from sparse and low rank parts
    """
    error = frobenius_norm(M - L - S) / frobenius_norm(M)
    # Use logging here
    return error <= epsilon


if __name__ == "__main__":
    X = np.random.randn(3,4,2)
    L, S = robust_pca(X)
    assert np.allclose(L + S, X)
