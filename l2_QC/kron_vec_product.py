import numpy as np
import numpy.random as npr
from functools import reduce

# Goal
# ----
# Compute (As[0] kron As[1] kron ... As[-1]) @ v

# ==== HELPER FUNCTIONS ==== #

def unfold(tens, mode, dims):
    """
    Unfolds tensor into matrix.

    Parameters
    ----------
    tens : ndarray, tensor with shape == dims
    mode : int, which axis to move to the front
    dims : list, holds tensor shape

    Returns
    -------
    matrix : ndarray, shape (dims[mode], prod(dims[/mode]))
    """
    if mode == 0:
        return tens.reshape(dims[0], -1)
    else:
        return np.moveaxis(tens, mode, 0).reshape(dims[mode], -1)


def refold(vec, mode, dims):
    """
    Refolds vector into tensor.

    Parameters
    ----------
    vec : ndarray, tensor with len == prod(dims)
    mode : int, which axis was unfolded along.
    dims : list, holds tensor shape

    Returns
    -------
    tens : ndarray, tensor with shape == dims
    """
    if mode == 0:
        return vec.reshape(dims)
    else:
        # Reshape and then move dims[mode] back to its
        # appropriate spot (undoing the `unfold` operation).
        tens = vec.reshape(
            [dims[mode]] +
            [d for m, d in enumerate(dims) if m != mode]
        )
        return np.moveaxis(tens, 0, mode)

# ==== KRON-VEC PRODUCT COMPUTATIONS ==== #

def kron_vec_prod_1d(As, v):
    """
    Computes matrix-vector multiplication between
    matrix kron(As[0], As[1], ..., As[N]) and vector
    v without forming the full kronecker product.
    """
    dims = [A.shape[0] for A in As]
    vt = v.reshape(dims)
    dims_in = dims
    for i, A in enumerate(As):
        # change the ith entry of dims to A.shape[0]
        dims_fin = np.copy(dims_in)
        dims_fin[i] = 1
        vt = refold(A @ unfold(vt, i, dims_in), i, dims_fin)
        dims_in = np.copy(dims_fin)
    return vt.ravel()
def kron_vec_prod(As, v):
    """
    Computes matrix-vector multiplication between
    matrix kron(As[0], As[1], ..., As[N]) and vector
    v without forming the full kronecker product.
    """
    dims = [A.shape[1] for A in As]
    vt = v.reshape(dims)
    dims_in = dims
    for i, A in enumerate(As):
        # change the ith entry of dims to A.shape[0]
        dims_fin = np.copy(dims_in)
        dims_fin[i] = A.shape[0]
        vt = refold(A @ unfold(vt, i, dims_in), i, dims_fin)
        dims_in = np.copy(dims_fin)
    return vt.ravel()


def kron_brute_force(As, v):
    """
    Computes kron-matrix times vector by brute
    force (instantiates the full kron product).
    """
    return reduce(np.kron, As) @ v


# Quick demonstration.
if __name__ == "__main__":

    # Create random problem.
    _yaxes = [2, 2, 2]
    _xaxes = [1, 1, 1]
    # As = [np.ones((x,y)) for (x, y) in zip(_xaxes, _yaxes)]
    As = [np.random.rand(x, y) for (x, y) in zip(_xaxes, _yaxes)]

    v = np.ones((np.prod(_yaxes), ))

    # Test accuracy.
    actual = kron_vec_prod(As, v)
    expected = kron_brute_force(As, v)
    print(np.linalg.norm(actual - expected))