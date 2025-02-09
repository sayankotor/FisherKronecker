import torch
import cupy as cp
import numpy as np
from cupyx.scipy.sparse.linalg import svds, LinearOperator
import numba
from numba import jit
from numba import cuda

#@cuda.jit

# Load the tensor and convert to CuPy array
#a = cp.asarray(torch.load("last_one.pt")) old
import numpy as np
from numba import cuda
from scipy.sparse.linalg import svds, LinearOperator
import torch

#vec1 = torch.randn(500)
#vec2 = torch.randn(100)
#a = grad_matrix = torch.outer(vec1, vec2)
#m = a.shape[0]
#n = a.shape[1]


# Load data
a = torch.load("grad2_10.pt").cpu()
m = a.shape[0]
n = a.shape[1]

grad_vector = np.ascontiguousarray(a.reshape(a.shape[0] * a.shape[1]).numpy())

torch.manual_seed(42)

import numpy as np
d_grad_vector = cuda.to_device(grad_vector)
#d_res = cuda.device_array((n**2), dtype=np.float32)

@cuda.jit
def matvec_kernel(grad_vector, x, res, m, n):
    """
    CUDA kernel to compute matvec operation (A * x).
    """
    idx = cuda.grid(1)
    pn, pm = divmod(np.int32(idx), m)
    if idx < res.size:  # Each thread computes one element of the result
        sum_ = 0.0
        for jnd in range(x.size):
            qd, qn = divmod(jnd, n)
            sum_ += grad_vector[pm * n + qd] * grad_vector[pn * n + qn] * x[jnd]
        res[idx] = sum_
        
@cuda.jit
def rmatvec_kernel(grad_vector, x, res, m, n):
    """
    CUDA kernel to compute rmatvec operation (A.T * x).
    """
    idx = cuda.grid(1)
    if idx < res.size:  # Each thread computes one element of the result
        sum_ = 0.0
        pn, pm = divmod(np.int32(idx), m)
        for jnd in range(x.size):
            qd, qn = divmod(jnd, n)
            sum_ += grad_vector[pm * n + qd] * grad_vector[pn * n + qn] * x[jnd]
        res[idx] = sum_

def matvec_item_custom(x):
    """
    Host function to launch matvec_kernel.
    """
    print ("matvec")
    x = x.ravel()
    res = np.zeros((m**2), dtype=np.float32)
    threads_per_block = 256
    blocks_per_grid = 256#(m**2 + threads_per_block - 1) // threads_per_block

    #d_grad_vector = cuda.to_device(grad_vector)
    d_x = cuda.to_device(x)
    d_res = cuda.device_array((m**2), dtype=np.float32)

    matvec_kernel[blocks_per_grid, threads_per_block](d_grad_vector, d_x, d_res, m, n)

    # Copy result back to host
    res = d_res.copy_to_host()
    return res

def rmatvec_item_custom(x):
    """
    Host function to launch rmatvec_kernel.
    """
    res = np.zeros((n**2), dtype=np.float32)
    x = x.ravel()
    threads_per_block = 256
    blocks_per_grid = 256#(n**2 + threads_per_block - 1) // threads_per_block

    #d_grad_vector = cuda.to_device(grad_vector)
    d_x = cuda.to_device(x)
    d_res = cuda.device_array((n**2), dtype=np.float32)

    rmatvec_kernel[blocks_per_grid, threads_per_block](d_grad_vector, d_x, d_res, m, n)

    # Copy result back to host
    res = d_res.copy_to_host()
    return res


# Create the operator
print("Start to create operator", flush=True)
linop_v = LinearOperator(
    shape=(m**2, n**2),
    matvec=matvec_item_custom,
    rmatvec=rmatvec_item_custom
)
print("Operator created", flush=True)

# Compute SVD
print("Performing SVD...", flush=True)
v, s, u = svds(linop_v, k = 10, return_singular_vectors=True)
s = np.sort(s)[::-1]
print("v.shape, u.shape", v.shape, u.shape, flush=True)
print("Singular values:", np.sort(s)[::-1])
print("v.shape, u.shape", v[:, 0].shape, u[0, :].shape, flush=True)


# Convert results back to NumPy for saving
B1 = cp.asnumpy(s[0] * u[0, :].reshape(n, n))
C1 = cp.asnumpy(v[:, 0].reshape(m, m))

np.save("B1_10.npy", B1)
np.save("C1_10.npy", C1)
print("B1.shape, C1.shape", B1.shape, C1.shape, flush=True)