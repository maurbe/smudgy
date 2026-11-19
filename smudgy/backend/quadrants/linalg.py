"""Small dense linear-algebra helpers shared by the Quadrants kernels."""

import quadrants as qd


@qd.func
def fix_eigvec_sign(V, dim: qd.template()):
    """Flip each column of V so its largest-magnitude component is positive.

    Neither ``qd.sym_eig`` nor ``np.linalg.eigh`` fixes eigenvector signs; this
    convention makes both backends agree (see ``numpy/tensor_utils.py``).
    Returns the new matrix (matrices cross a ``qd.func`` boundary by value).
    """
    for col in qd.static(range(dim)):
        max_abs = 0.0
        sign = 1.0
        for row in qd.static(range(dim)):
            v = V[row, col]
            if qd.abs(v) > max_abs:
                max_abs = qd.abs(v)
                sign = 1.0 if v >= 0.0 else -1.0
        for row in qd.static(range(dim)):
            V[row, col] *= sign
    return V


@qd.func
def sym_eig_descending(A, dim: qd.template()):
    """Eigendecomposition of a small symmetric f32 matrix, eigenvalues descending.

    ``qd.sym_eig`` (cyclic Jacobi in Quadrants >= 1.4) returns orthonormal
    eigenvectors and ascending eigenvalues; this reverses the order and applies
    the sign convention of `fix_eigvec_sign`. Returns (eigvals, V) with
    A = V diag(eigvals) V^T and column j of V the eigenvector of eigvals[j].
    """
    w, Q = qd.sym_eig(A, qd.f32)
    eigvals = qd.Vector.zero(qd.f32, dim)
    V = qd.Matrix.zero(qd.f32, dim, dim)
    for a in qd.static(range(dim)):
        eigvals[a] = w[dim - 1 - a]
        for r in qd.static(range(dim)):
            V[r, a] = Q[r, dim - 1 - a]
    return eigvals, fix_eigvec_sign(V, dim)
