import logging
import torch
import numpy as np
from numba import cuda
from scipy.sparse.linalg import svds, LinearOperator
from typing import Tuple, Optional

from .utils_svd import SVD

# grad_tensor = torch.load("last_one.pt")
# m, n = grad_tensor.shape
# grad_vector = np.ascontiguousarray(grad_tensor.reshape(-1).numpy())
# d_grad_vector = cuda.to_device(grad_vector)

# logger.info(f"Gradient tensor shape: {grad_tensor.shape}")
# logger.debug(f"Initial gradient tensor sample: {grad_tensor[:5, :5]}")


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

@cuda.jit
def matvec_kernel(grad_vector, x, res, m, n):
    idx = cuda.grid(1)
    if idx < res.size:
        sum_ = 0.0
        pn, pm = divmod(idx, m)
        for jnd in range(x.size):
            qd, qn = divmod(jnd, n)
            sum_ += grad_vector[pm * n + qd] * grad_vector[pn * n + qn] * x[jnd]
        res[idx] = sum_


@cuda.jit
def rmatvec_kernel(grad_vector, x, res, m, n):
    idx = cuda.grid(1)
    if idx < res.size:
        sum_ = 0.0
        pn, pm = divmod(idx, m)
        for jnd in range(x.size):
            qd, qn = divmod(jnd, n)
            sum_ += grad_vector[pm * n + qd] * grad_vector[pn * n + qn] * x[jnd]
        res[idx] = sum_


def matvec_item_custom(x):
    res = np.zeros((m**2), dtype=np.float32)
    d_x = cuda.to_device(x.ravel())
    d_res = cuda.device_array((m**2), dtype=np.float32)

    logger.debug("Launching matvec kernel...")
    matvec_kernel[256, 256](d_grad_vector, d_x, d_res, m, n)
    result = d_res.copy_to_host()
    logger.debug(f"Matvec result sample: {result[:5]}")
    return result


def rmatvec_item_custom(x):
    res = np.zeros((n**2), dtype=np.float32)
    d_x = cuda.to_device(x.ravel())
    d_res = cuda.device_array((n**2), dtype=np.float32)

    logger.debug("Launching rmatvec kernel...")
    rmatvec_kernel[256, 256](d_grad_vector, d_x, d_res, m, n)
    result = d_res.copy_to_host()
    logger.debug(f"Rmatvec result sample: {result[:5]}")
    return result


def calculate_fisher(grad_tensor: torch.Tensor, m: int, n: int) -> np.ndarray:
    logger.info("Creating LinearOperator...")

    linop_v = LinearOperator(
        shape=(m**2, n**2), matvec=matvec_item_custom, rmatvec=rmatvec_item_custom
    )

    logger.info("Starting SVD computation...")
    u, s, v = svds(linop_v, k=1, return_singular_vectors=True)

    B1 = s[0] * v.reshape(n, n)
    C1 = u[:, 0].reshape(m, m)

    logger.info(f"B1 shape: {B1.shape}, C1 shape: {C1.shape}")

    return B1, C1


def novel_fisher(
    module: torch.nn.Module, rank: int, weight: Tuple[np.ndarray, np.ndarray]
) -> SVD:
    bias = module.bias if module.bias is not None else None
    device = module.weight.device

    C1 = torch.tensor(weight["C1"], device=device)
    B1 = torch.tensor(weight["B1"], device=device)

    # switch to eigendecomp
    U_B1, S_B1, Vt_B1 = torch.linalg.svd(B1, full_matrices=False)
    sqrt_B1 = U_B1 @ torch.diag(torch.sqrt(S_B1)) @ Vt_B1

    # switch to eigendecomp
    U_C1, S_C1, Vt_C1 = torch.linalg.svd(C1, full_matrices=False)
    sqrt_C1 = U_C1 @ torch.diag(torch.sqrt(S_C1)) @ Vt_C1

    U, S, Vt = torch.linalg.svd(sqrt_C1.T @ module.weight.data @ sqrt_B1, full_matrices=False)

    # U, S, Vt = torch.linalg.svd((C1.T @ module.weight.data @ B1), full_matrices=False)
    # A^{1/2} W B^{1/2} = U Sigma V^T; \hat{W} = \hat{U} Sigma \hat{V}^T; \hat{U} = A^{-1/2} U, \hat{V} = B^{-1/2} V
 
    w1 = torch.diag(torch.sqrt(S[:rank])) @ Vt[:rank, :]
    w2 = U[:, :rank] @ torch.diag(torch.sqrt(S[:rank]))

    new_module = SVD(module.in_features, module.out_features, rank, bias is not None)
    new_module.lin0.weight.data.copy_(w1)
    new_module.lin1.weight.data.copy_(w2)

    if bias is not None:
        new_module.lin1.bias.data.copy_(bias)

    return new_module

def compress_novel(module: torch.nn.Module, path: str,
                       shape: Tuple[Tuple[int], Tuple[int]],
                       rank: int,
                       weight,
                       random_init,
                       cholesky: bool = False
                       ) -> torch.nn.Module:
    if not isinstance(module, torch.nn.Linear):
        return module    
    logging.info('apply novel fwsvd compression to layer %s', path)

    if random_init:
        module.weight.data.uniform_(0.0, 1.0)
        
    if weight:
        path_fisher = path.replace(r'/','.')[1:]
        return novel_fisher(module, rank, weight[path_fisher])
    else:
        return w_svd(module, rank)


class SVD(torch.nn.Module):
    def __init__(
        self, in_features: int, out_features: int, rank: int, bias: bool = True
    ):
        super().__init__()
        self.lin0 = torch.nn.Linear(in_features, rank, bias=False)
        self.lin1 = torch.nn.Linear(rank, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin1(self.lin0(x))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    logger = logging.getLogger(__name__)

    torch.manual_seed(42)
    module = torch.nn.Linear(500, 100, bias=True).to("cuda:0")
    fisher_weights = torch.rand(500, 100, device="cuda:0")

    logger.info("Starting Fisher-weighted SVD compression...")
    compressed_module = fisher_weighted_svd(module, rank=20, weight=fisher_weights)
    logger.info("Original module and compressed module created.")
