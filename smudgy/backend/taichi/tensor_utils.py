# backend/taichi/tensor_utils.py
import numpy as np
import taichi as ti

E1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
E2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)


@ti.kernel
def _project_2d_kernel(
    h_tensor: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (N, 3, 3)
    e1: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (3,)
    e2: ti.types.ndarray(dtype=ti.f32, ndim=1),  # (3,)
    h2d_out: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (N, 2, 2)
    eigvals_out: ti.types.ndarray(dtype=ti.f32, ndim=2),  # (N, 2)
    eigvecs_out: ti.types.ndarray(dtype=ti.f32, ndim=3),  # (N, 2, 2)
):
    P = ti.Matrix.zero(ti.f32, 2, 3)
    for j in ti.static(range(3)):
        P[0, j] = e1[j]
        P[1, j] = e2[j]

    for n in range(h_tensor.shape[0]):
        H = ti.Matrix.zero(ti.f32, 3, 3)
        for a in ti.static(range(3)):
            for b in ti.static(range(3)):
                H[a, b] = h_tensor[n, a, b]

        h2d = (P @ H.inverse() @ P.transpose()).inverse()
        eigvals, eigvecs = ti.sym_eig(h2d, ti.f32)

        # ==========================================================================================
        # Fix for annyoing taichi bug in 2D: 
        # eigvals/vecs are returned in descending order, 3D is fine and follows numpy backend
        if eigvals[0] > eigvals[1]:
            eigvals[0], eigvals[1] = eigvals[1], eigvals[0]
            for a in ti.static(range(2)):
                eigvecs[a, 0], eigvecs[a, 1] = eigvecs[a, 1], eigvecs[a, 0]

        # Eigenvector orientation convention:
        # Numpy and Taichi have no convention about the sign of the eigenvectors,
        # impose a consistent sign convention for comparison with Taichi backend 
        # (largest-magnitude component of each eigenvector is positive).
        for col in ti.static(range(2)):
            max_abs = 0.0
            sign = 1.0
            for row in ti.static(range(2)):
                v = eigvecs[row, col]
                if ti.abs(v) > max_abs:
                    max_abs = ti.abs(v)
                    sign = 1.0 if v >= 0.0 else -1.0
            for row in ti.static(range(2)):
                eigvecs[row, col] *= sign
        # ==========================================================================================

        for a in ti.static(range(2)):
            eigvals_out[n, a] = eigvals[a]
            for b in ti.static(range(2)):
                h2d_out[n, a, b] = h2d[a, b]
                eigvecs_out[n, a, b] = eigvecs[a, b]


def project_2d(h_tensor: np.ndarray, e1: np.ndarray = E1, e2: np.ndarray = E2):
    h_tensor = np.ascontiguousarray(h_tensor, dtype=np.float32)
    e1 = np.ascontiguousarray(e1, dtype=np.float32)
    e2 = np.ascontiguousarray(e2, dtype=np.float32)

    N = h_tensor.shape[0]
    h2d = np.zeros((N, 2, 2), dtype=np.float32)
    eigvals = np.zeros((N, 2), dtype=np.float32)
    eigvecs = np.zeros((N, 2, 2), dtype=np.float32)

    _project_2d_kernel(h_tensor, e1, e2, h2d, eigvals, eigvecs)
    return h2d, eigvals, eigvecs