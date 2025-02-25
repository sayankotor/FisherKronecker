import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, svds
from matvecs import matvec_old, r_matvec_old

def load_weight(path: Path) -> Dict[str, torch.Tensor]:
    """Load gradients from a pickle file."""
    try:
        with path.open("rb") as f:
            grads = pickle.load(f)
        return grads
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        raise
    except pickle.PickleError:
        logging.error(f"Failed to load pickle file: {path}")
        raise

def get_factors(
    n_layer: int, grads: Dict[str, torch.Tensor], output: bool = False
) -> Optional[tuple]:
    """Compute factors XF and YF from gradients."""
    try:
        key = f"bert.encoder.layer.{n_layer}.{'output' if output else 'intermediate'}.dense"
        l_grads: List[torch.Tensor] = grads[key]
        m, n = l_grads[0].shape

        # grad_vectors = torch.stack([grad_tensor.reshape(-1) for grad_tensor in l_grads])

        l_gradss = [grad.reshape(-1).numpy() for grad in l_grads]
        grad_vectors = np.stack([grad.reshape(n,m, order = 'F') for grad in l_gradss])

        linop_m = LinearOperator(
            shape=(m**2, n**2),
            matvec=lambda x: matvec_old(x, grad_vectors),
            rmatvec=lambda x: r_matvec_old(x, grad_vectors) 
        )

        u, s, vt = svds(linop_m, k=1, return_singular_vectors=True)

        sidx = np.argsort(-s)
        s = s[sidx]
        u = u[:, sidx]
        v = vt[sidx, :].T

        x_ = u[:, 0] * s[0]
        y_ = v[:, 0]
        XF = x_.reshape(m, m, order="F")
        YF = y_.reshape(n, n, order="F")

        return XF, YF

    except KeyError:
        logging.error(f"Key not found in gradients: {key}")
        return None
    except Exception as e:
        logging.error(f"Error in get_factors: {e}")
        return None


def save_factors(n_layer: int, XF: np.ndarray, YF: np.ndarray, output_dir: Path, output: bool):
    """Save factors XF and YF to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        output_dir / f"C1sgd_{n_layer}_{'output' if output else 'intermediate'}.npy", XF
    )
    np.save(
        output_dir / f"B1sgd_{n_layer}_{'output' if output else 'intermediate'}.npy", YF
    )
    logging.info(f"Factors for layer {n_layer} saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and save factors from gradients."
    )
    parser.add_argument(
        "--path", type=Path, required=True, help="Path to the gradients pickle file."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=False,
        help="Directory to save factors.",
        default="fisher_factors/",
    )
    parser.add_argument(
        "--n_layer", type=int, required=True, help="Layer number to process."
    )
    parser.add_argument(
        "--output",
        action="store_true",
        help="Process output layer (default: intermediate).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    grads = load_weight(args.path)

    factors = get_factors(args.n_layer, grads, args.output)
    if factors is None:
        logging.error("Failed to compute factors.")
        exit(1)

    save_factors(args.n_layer, *factors, args.output_dir, args.output)
