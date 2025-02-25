import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def matvec_old(vec: np.ndarray, grad_vectors: np.ndarray) -> np.ndarray:
    """Reference implementation of matvec using explicit loop.

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

def r_matvec_old(vec: np.ndarray, grad_vectors: np.ndarray) -> np.ndarray:
    """Reference implementation of r_matvec using explicit loop.

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

def matvec(vec: np.ndarray, grad_vectors: np.ndarray) -> np.ndarray:
    """Compute the vectorized matvec operation.

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


    result = np.einsum("ias,am,imr->sr", grad_vectors, V, grad_vectors)
    return result.T.ravel()

def r_matvec(vec: np.ndarray, grad_vectors: np.ndarray) -> np.ndarray:
    """Compute the vectorized r_matvec operation.

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
    result = np.einsum("iqc,cd,ipd->pq", grad_vectors, V, grad_vectors)
    return result.ravel()

if __name__ == "__main__":
    
    k = 500
    m = 55
    n = 55
    grad_vectors = np.random.rand(k, m, n)
    vec_for_matvec = np.random.rand(m * m)
    vec_for_rmatvec = np.random.rand(n * n)

    res_old_matvec = matvec_old(vec_for_matvec, grad_vectors)
    res_new_matvec = matvec(vec_for_matvec, grad_vectors)
    logger.info(f"matvec old and new match: {np.allclose(res_old_matvec, res_new_matvec)}")