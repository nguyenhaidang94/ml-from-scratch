import numpy as np


def pinv(x: np.ndarray, eigen_threshold: float = 1e-10):
    """
    Pseudo inverse.\n
    Params:
        x: matrix to inverse
        eigen_threshold: threshold to keep eigenvalues, between 0 and 1
    """
    u, s, vt = np.linalg.svd(x, full_matrices=True)
    n_remaining_eigenvalues = sum(s > eigen_threshold)
    if not n_remaining_eigenvalues:
        raise ArithmeticError("Can't inverse matrix,"
                              + " all eigenvalues are lower than eigen_threshold.")
    s = s[:n_remaining_eigenvalues]
    u = u[:,:n_remaining_eigenvalues]
    vt = vt[:n_remaining_eigenvalues,:]
    # take the inverse of eigenvalues to compute the inverse matrix
    s_i = 1 / s * np.identity(n_remaining_eigenvalues)
    return u @ s_i @ vt
