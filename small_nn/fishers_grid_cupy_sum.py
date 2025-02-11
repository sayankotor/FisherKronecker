import torch
import cupy as cp
import numpy as np
from cupyx.scipy.sparse.linalg import svds, LinearOperator
import numba
from numba import jit
from numba import cuda

import numpy as np

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

import pickle
with open("v5_tensor_grad.pickle", "rb") as fp:   #Pickling
    dict_of_grads = pickle.load(fp)

list_of_grads = dict_of_grads['bert.encoder.layer.5.intermediate.dense']

list_of_grads = torch.stack(list_of_grads, dim=0)

m = list_of_grads[0].shape[0]
n = list_of_grads[0].shape[1]

del dict_of_grads
print (m, n)

grad_vectors = torch.stack([grad.T.reshape(-1) for grad in list_of_grads])
print (grad_vectors.shape)
d_grad_vectors = cuda.to_device(grad_vectors)

print (m, n)

lenpm = len(grad_vectors)

print (d_grad_vectors[0].shape)

@cuda.jit(device=True)
def getitem_fmatrices_permuted(d_grad_vectors, p, q):
    reduce_init_val = 0.0
    for idx in range(d_grad_vectors.shape[0]):
        #reduce_init_val+= d_grad_vectors[idx][p]*d_grad_vectors[idx][q]
        elem = d_grad_vectors[idx]
        reduce_init_val += elem[p] * elem[q]

    return reduce_init_val/d_grad_vectors.shape[0]

from scipy.sparse.linalg import svds, LinearOperator


@cuda.jit
def matvec_kernel(d_grad_vectors, x, res, m, n):
    """
    CUDA kernel to compute matvec operation (A * x).
    """
    idx = cuda.grid(1)
    
    pd, pn = divmod(np.int32(idx), m)
    if idx < res.size:  # Each thread computes one element of the result
        sum_ = 0.0
        for jnd in range(x.size):
            qd, qn = divmod(jnd, n)
            sum_ += x[jnd]*getitem_fmatrices_permuted(d_grad_vectors, (pd)*n +qd, (pn)*n + qn)#d_grad_vector[(pd)*n +qd]*d_grad_vector[(pn)*n + qn] 
        res[idx] = sum_

@cuda.jit
def rmatvec_kernel(d_grad_vectors, x, res, m, n):
    """
    CUDA kernel to compute rmatvec operation (A.T * x).
    """
    idx = cuda.grid(1)
    if idx < res.size:  # Each thread computes one element of the result
        sum_ = 0.0
        pd, pn = divmod(np.int32(idx), n)
        for jnd in range(x.size):
            jd, jn = divmod(jnd, m)
            #jnd, idx
            sum_ += x[jnd]*getitem_fmatrices_permuted(d_grad_vectors, jd*n +pd, jn*n + pn)# d_grad_vector[(jnd//m)*n +pd]*d_grad_vector[(jnd%m)*n + pn]  #grad_vector[(q // m)*n +p // n]*grad_vector[(q % m)*n + p % n]
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
    print (m**2, threads_per_block, blocks_per_grid)


    #d_grad_vector = cuda.to_device(grad_vector)
    d_x = cuda.to_device(x)
    d_res = cuda.device_array((m**2), dtype=np.float32)

    matvec_kernel[blocks_per_grid, threads_per_block](d_grad_vectors, d_x, d_res, m, n)

    # Copy result back to host
    res = d_res.copy_to_host()
    print (res)
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

    rmatvec_kernel[blocks_per_grid, threads_per_block](d_grad_vectors, d_x, d_res, m, n)

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
v, s, u = svds(linop_v, k = 1, return_singular_vectors=True)
s = np.sort(s)[::-1]

print("Singular values:", s)
print("v.shape, u.shape", v[:, -1].shape, u[-1, :].shape, flush=True)


# Convert results back to NumPy for saving
B1 = cp.asnumpy(s[0] * u[-1, :].reshape(n, n))
C1 = cp.asnumpy(v[:, -1].reshape(m, m))

np.save("B1_module5_last2.npy", B1)
np.save("C1_module5_last2.npy", C1)
print("B1.shape, C1.shape", B1.shape, C1.shape, flush=True)

print (is_pos_def(C1))
print (is_pos_def(B1))
