# backend/taichi/tensor_utils.py
import numpy as np
import taichi as ti

from .. import init

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

        for a in ti.static(range(2)):
            eigvals_out[n, a] = eigvals[a]
            for b in ti.static(range(2)):
                h2d_out[n, a, b] = h2d[a, b]
                eigvecs_out[n, a, b] = eigvecs[a, b]


def project_2d(h_tensor: np.ndarray, e1: np.ndarray = E1, e2: np.ndarray = E2):
    init()
    h_tensor = np.ascontiguousarray(h_tensor, dtype=np.float32)
    e1 = np.ascontiguousarray(e1, dtype=np.float32)
    e2 = np.ascontiguousarray(e2, dtype=np.float32)

    N = h_tensor.shape[0]
    h2d = np.zeros((N, 2, 2), dtype=np.float32)
    eigvals = np.zeros((N, 2), dtype=np.float32)
    eigvecs = np.zeros((N, 2, 2), dtype=np.float32)

    _project_2d_kernel(h_tensor, e1, e2, h2d, eigvals, eigvecs)
    return h2d, eigvals, eigvecs


"""
import numpy as np
import taichi as ti

@ti.kernel
def _project_kernel(
    h_tensor: ti.types.ndarray(),     # (N, 3, 3) float32
    e1: ti.types.ndarray(),           # (3,) float32
    e2: ti.types.ndarray(),           # (3,) float32
    h2d_out: ti.types.ndarray(),      # (N, 2, 2) float32
    eigvals_out: ti.types.ndarray(),  # (N, 2) float32
    eigvecs_out: ti.types.ndarray(),  # (N, 2, 2) float32
):
    N = h_tensor.shape[0]
    E1 = ti.Vector([e1[0], e1[1], e1[2]])
    E2 = ti.Vector([e2[0], e2[1], e2[2]])
    for n in range(N):
        H = ti.Matrix([[h_tensor[n, 0, 0], h_tensor[n, 0, 1], h_tensor[n, 0, 2]],
                        [h_tensor[n, 1, 0], h_tensor[n, 1, 1], h_tensor[n, 1, 2]],
                        [h_tensor[n, 2, 0], h_tensor[n, 2, 1], h_tensor[n, 2, 2]]])
        Hinv = H.inverse()

        t11 = E1.dot(Hinv @ E1)
        t12 = E1.dot(Hinv @ E2)
        t22 = E2.dot(Hinv @ E2)
        temp = ti.Matrix([[t11, t12], [t12, t22]])
        h2d = temp.inverse()

        vals, vecs = ti.sym_eig(h2d)

        for i in ti.static(range(2)):
            eigvals_out[n, i] = vals[i]
            for j in ti.static(range(2)):
                h2d_out[n, i, j] = h2d[i, j]
                eigvecs_out[n, i, j] = vecs[i, j]


    n = h_tensor.shape[0]
    h2d = np.empty((n, 2, 2), dtype=np.float32)
    eigvals = np.empty((n, 2), dtype=np.float32)
    eigvecs = np.empty((n, 2, 2), dtype=np.float32)
    _project_kernel(h_tensor, np.ascontiguousarray(e1), np.ascontiguousarray(e2), h2d, eigvals, eigvecs)
    return h2d, eigvals, eigvecs

def project_2d(h_tensor: np.ndarray, e1=E1, e2=E2):
    n = h_tensor.shape[0]
    h2d = np.empty((n, 2, 2), dtype=np.float32)
    eigvals = np.empty((n, 2), dtype=np.float32)
    eigvecs = np.empty((n, 2, 2), dtype=np.float32)
    _project_kernel(h_tensor, np.ascontiguousarray(e1), np.ascontiguousarray(e2), h2d, eigvals, eigvecs)
    return h2d, eigvals, eigvecs
"""
