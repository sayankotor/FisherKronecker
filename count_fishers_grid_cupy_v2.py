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
a = torch.load("last_one.pt")
m = a.shape[0]
n = a.shape[1]

grad_vector = a.reshape(a.shape[0] * a.shape[1])

torch.manual_seed(42)

import numpy as np
d_grad_vector = cuda.to_device(grad_vector)
#d_res = cuda.device_array((n**2), dtype=np.float32)


@cuda.jit
def matvec_kernel(d_grad_vector, x, res, m, n):
    """
    CUDA kernel to compute matvec operation (A * x).
    """
    idx = cuda.grid(1)
    
    pd, pn = divmod(np.int32(idx), m)
    if idx < res.size:  # Each thread computes one element of the result
        sum_ = 0.0
        for jnd in range(x.size):
            qd, qn = divmod(jnd, n)
            sum_ += x[jnd]*d_grad_vector[(pd)*n +qd]*d_grad_vector[(pn)*n + qn] 
        res[idx] = sum_
        
@cuda.jit
def rmatvec_kernel(d_grad_vector, x, res, m, n):
    """
    CUDA kernel to compute rmatvec operation (A.T * x).
    """
    idx = cuda.grid(1)
    if idx < res.size:  # Each thread computes one element of the result
        sum_ = 0.0
        pd, pn = divmod(np.int32(idx), n)
        for jnd in range(x.size):
            #qd, qn = divmod(jnd, n)
            #jnd, idx
            sum_ += x[jnd]*d_grad_vector[(jnd//m)*n +pd]*d_grad_vector[(jnd%m)*n + pn]  #grad_vector[(q // m)*n +p // n]*grad_vector[(q % m)*n + p % n]
        res[idx] = sum_

def matvec_item_custom(x):
    """
    Host function to launch matvec_kernel.
    """
    print ("matvec")
    x = x.ravel()
    res = np.zeros((m**2), dtype=np.float32)
    threads_per_block = 1024
    blocks_per_grid = (m**2 + threads_per_block - 1) // threads_per_block


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
    print ("rmatvec")
    x = x.ravel()
    threads_per_block = 1024
    blocks_per_grid = (n**2 + threads_per_block - 1) // threads_per_block


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
v, s, u = svds(linop_v, k = 3, return_singular_vectors=True)
s = np.sort(s)[::-1]
print("v.shape, u.shape", v.shape, u.shape, flush=True)
np.save("u1.npy", u)
np.save("v1.npy", v)
np.save("s1.npy", s)
print("Singular values:", np.sort(s)[::-1])
print("v.shape, u.shape", v[:, -1].shape, u[-1, :].shape, flush=True)


# Convert results back to NumPy for saving
B1 = cp.asnumpy(s[0] * u[-1, :].reshape(n, n))
C1 = cp.asnumpy(v[:, -1].reshape(m, m))

np.save("B1_101.npy", B1)
np.save("C1_101.npy", C1)
print("B1.shape, C1.shape", B1.shape, C1.shape, flush=True)