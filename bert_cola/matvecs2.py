#!filepath: optimized_opt_einsum.py
import numpy as np
import opt_einsum as oe
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@np.vectorize
def matvec_old(vec: np.ndarray, grad_vectors: np.ndarray) -> np.ndarray:
    """Reference implementation of matvec using an explicit loop.

    Args:
        vec: A numpy array of shape (m*m,).
        grad_vectors: A numpy array of shape (k, m, n).

    Returns:
        A numpy array of shape (n*n,).
    """
    k, m, n = grad_vectors.shape
    V = vec.reshape(m, m, order="F")
    res = np.zeros(n * n)
    for i in range(k):
        res += (grad_vectors[i].T @ V @ grad_vectors[i]).T.ravel()
    return res

@np.vectorize
def r_matvec_old(vec: np.ndarray, grad_vectors: np.ndarray) -> np.ndarray:
    """Reference implementation of r_matvec using an explicit loop.

    Args:
        vec: A numpy array of shape (n*n,).
        grad_vectors: A numpy array of shape (k, m, n).

    Returns:
        A numpy array of shape (m*m,).
    """
    k, m, n = grad_vectors.shape
    V = vec.reshape(n, n, order="F")
    res = np.zeros(m * m)
    for i in range(k):
        res += (grad_vectors[i] @ V @ grad_vectors[i].T).T.ravel()
    return res

def matvec_np(vec: np.ndarray, grad_vectors: np.ndarray) -> np.ndarray:
    """Vectorized implementation of matvec using np.einsum.

    Args:
        vec: A numpy array of shape (m*m,).
        grad_vectors: A numpy array of shape (k, m, n).

    Returns:
        A numpy array of shape (n*n,).
    """
    k, m, n = grad_vectors.shape
    V = vec.reshape(m, m, order="F")
    result = np.einsum("ias,am,imr->sr", grad_vectors, V, grad_vectors)
    return result.T.ravel()

def r_matvec_np(vec: np.ndarray, grad_vectors: np.ndarray) -> np.ndarray:
    """Vectorized implementation of r_matvec using np.einsum.

    Args:
        vec: A numpy array of shape (n*n,).
        grad_vectors: A numpy array of shape (k, m, n).

    Returns:
        A numpy array of shape (m*m,).
    """
    k, m, n = grad_vectors.shape
    V = vec.reshape(n, n, order="F")
    result = np.einsum("iqc,cd,ipd->pq", grad_vectors, V, grad_vectors)
    return result.ravel()

def matvec_opt(vec: np.ndarray, grad_vectors: np.ndarray) -> np.ndarray:
    """Compute the matvec operation using opt_einsum for optimization.

    This function computes the sum over i of the transposed product:
    (grad_vectors[i].T @ V @ grad_vectors[i]).T, where V is vec reshaped
    into an (m, m) matrix using Fortran order.

    Args:
        vec: A numpy array of shape (m*m,).
        grad_vectors: A numpy array of shape (k, m, n).

    Returns:
        A numpy array of shape (n*n,).
    """
    k, m, n = grad_vectors.shape
    V = vec.reshape(m, m, order="F")
    result = oe.contract("ias,am,imr->sr", grad_vectors, V, grad_vectors)
    return result.T.ravel()

def r_matvec_opt(vec: np.ndarray, grad_vectors: np.ndarray) -> np.ndarray:
    """Compute the r_matvec operation using opt_einsum for optimization.

    This function computes the sum over i of the transposed product:
    (grad_vectors[i] @ V @ grad_vectors[i].T).T, where V is vec reshaped
    into an (n, n) matrix using Fortran order.

    Args:
        vec: A numpy array of shape (n*n,).
        grad_vectors: A numpy array of shape (k, m, n).

    Returns:
        A numpy array of shape (m*m,).
    """
    k, m, n = grad_vectors.shape
    V = vec.reshape(n, n, order="F")
    result = oe.contract("iqc,cd,ipd->pq", grad_vectors, V, grad_vectors)
    return result.ravel()

def time_function(func: callable, *args, n_iter: int = 100) -> float:
    """Time a function by executing it n_iter times and return the average execution time.

    Args:
        func: Callable function to be timed.
        *args: Arguments to be passed to the function.
        n_iter: Number of iterations to average over.

    Returns:
        Average execution time in seconds.
    """
    start_time = time.perf_counter()
    for _ in range(n_iter):
        func(*args)
    end_time = time.perf_counter()
    return (end_time - start_time) / n_iter

if __name__ == "__main__":
    try:
        k = 500
        m = 55
        n = 55
        grad_vectors = np.random.rand(k, m, n)
        vec_for_matvec = np.random.rand(m * m)
        vec_for_rmatvec = np.random.rand(n * n)

        # Verify that all implementations produce the same results
        res_old = matvec_old(vec_for_matvec, grad_vectors)
        res_np = matvec_np(vec_for_matvec, grad_vectors)
        res_opt = matvec_opt(vec_for_matvec, grad_vectors)
        logger.info(f"matvec_old vs matvec_np match: {np.allclose(res_old, res_np)}")
        logger.info(f"matvec_old vs matvec_opt match: {np.allclose(res_old, res_opt)}")

        res_old_r = r_matvec_old(vec_for_rmatvec, grad_vectors)
        res_np_r = r_matvec_np(vec_for_rmatvec, grad_vectors)
        res_opt_r = r_matvec_opt(vec_for_rmatvec, grad_vectors)
        logger.info(f"r_matvec_old vs r_matvec_np match: {np.allclose(res_old_r, res_np_r)}")
        logger.info(f"r_matvec_old vs r_matvec_opt match: {np.allclose(res_old_r, res_opt_r)}")

        # Timing tests
        n_iter = 5
        t_old = time_function(matvec_old, vec_for_matvec, grad_vectors, n_iter=n_iter)
        t_np = time_function(matvec_np, vec_for_matvec, grad_vectors, n_iter=n_iter)
        t_opt = time_function(matvec_opt, vec_for_matvec, grad_vectors, n_iter=n_iter)
        logger.info(f"Average time for matvec_old: {t_old:.6f} sec")
        logger.info(f"Average time for matvec_np: {t_np:.6f} sec")
        logger.info(f"Average time for matvec_opt: {t_opt:.6f} sec")

        t_old_r = time_function(r_matvec_old, vec_for_rmatvec, grad_vectors, n_iter=n_iter)
        t_np_r = time_function(r_matvec_np, vec_for_rmatvec, grad_vectors, n_iter=n_iter)
        t_opt_r = time_function(r_matvec_opt, vec_for_rmatvec, grad_vectors, n_iter=n_iter)
        logger.info(f"Average time for r_matvec_old: {t_old_r:.6f} sec")
        logger.info(f"Average time for r_matvec_np: {t_np_r:.6f} sec")
        logger.info(f"Average time for r_matvec_opt: {t_opt_r:.6f} sec")
    except Exception as err:
        logger.exception(f"Error in main execution: {err}")
        raise
