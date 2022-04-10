import numpy as np


def pinv(x: np.ndarray, singular_threshold: float = 1e-10):
    """
    Pseudo inverse.\n
    Params:
        x: matrix to inverse
        singular_threshold: threshold to remove singularvalues
    """
    u, s, vt = np.linalg.svd(x, full_matrices=True)
    n_remaining_singularvalues = sum(s > singular_threshold)
    if not n_remaining_singularvalues:
        raise ArithmeticError("Can't inverse matrix,"
                              + " all singularvalues are lower than singular_threshold.")
    s = s[:n_remaining_singularvalues]
    ut = u[:,:n_remaining_singularvalues].T
    v = vt[:n_remaining_singularvalues,:].T
    # take the inverse of singularvalues to compute the inverse matrix
    s_i = 1 / s * np.identity(n_remaining_singularvalues)
    return v @ s_i @ ut
