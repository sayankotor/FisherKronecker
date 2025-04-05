import torch


def get_kron_factors_torch(list_of_grads, top_k=10):
    # Ensure everything is on GPU
    device = list_of_grads[0].device
    dtype = list_of_grads[0].dtype

    m, n = list_of_grads[0].shape

    # Reshape in Fortran-style: (m, n)
    grad_batch = torch.stack([g.T.contiguous() for g in list_of_grads]).to(  # from (n, m) to (m, n)
        device
    )  # Shape: (k, m, n)

    k = grad_batch.shape[0]

    # Option 1: build full fisher approximation matrix
    # This is only feasible for small m/n (e.g., m=64 -> 4k matrix)
    F_m = torch.einsum("kmi,kmj->ij", grad_batch, grad_batch) / k  # (m, m)
    F_n = torch.einsum("kij,kil->jl", grad_batch, grad_batch) / k  # (n, n)

    # Option 2: eigendecomposition of small Kronecker factors
    evals_m, evecs_m = torch.linalg.eigh(F_m)
    evals_n, evecs_n = torch.linalg.eigh(F_n)

    # Sort and take top-k
    top_idx_m = torch.argsort(evals_m, descending=True)[:top_k]
    top_idx_n = torch.argsort(evals_n, descending=True)[:top_k]

    XF = evecs_m[:, top_idx_m] @ torch.diag(evals_m[top_idx_m].sqrt())
    YF = evecs_n[:, top_idx_n]

    return XF, YF, evals_m[top_idx_m], evals_n[top_idx_n]
