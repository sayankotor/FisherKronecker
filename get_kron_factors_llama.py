import os
import logging
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Dict, List, Any

import cupy as cp
import torch
from cupyx.scipy.sparse.linalg import LinearOperator, svds
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm


def setup_logger(save_dir: Path) -> logging.Logger:
    """
    Set up a logger that writes both to a file and to the console.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "kron_factors.log"
    logger = logging.getLogger("KroneckerLogger")

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    return logger


def get_kron_factors_worker(
    layer_name: str, list_of_grads: List[Any], top_k: int = 1, device_id: int = 0, save_dir: str = "fisher_factors"
) -> bool:
    save_dir_path = Path(save_dir)
    logger = setup_logger(save_dir_path)

    cp.cuda.Device(device_id).use()
    try:
        m, n = list_of_grads[0].shape
        grad_vectors = cp.stack([cp.asarray(grad.reshape(-1).astype("float32")).reshape(n, m, order="F") for grad in list_of_grads])

        def matvec(vec):
            k, m_, n_ = grad_vectors.shape
            V = vec.reshape(m_, m_, order="F")
            res = cp.zeros(n_ * n_, dtype=cp.float32)
            for i in range(k):
                res += (grad_vectors[i].T @ V @ grad_vectors[i]).T.ravel()
            return res / k

        def r_matvec(vec):
            k, m_, n_ = grad_vectors.shape
            V = vec.reshape(n_, n_, order="F")
            res = cp.zeros(m_ * m_, dtype=cp.float32)
            for i in range(k):
                res += (grad_vectors[i] @ V @ grad_vectors[i].T).T.ravel()
            return res

        linop = LinearOperator(
            shape=(m**m, n * n),
            matvec=matvec,
            rmatvec=r_matvec,
            dtype=cp.float32,
        )

        u, s, vt = svds(linop, k=top_k, return_singular_vectors=True)

        sidx = cp.argsort(-s)
        s = s[sidx]
        u = u[:, sidx]
        v = vt[sidx, :].T

        XF = (u[:, 0] * s[0]).reshape(m, m, order="F").get()
        YF = v[:, 0].reshape(n, n, order="F").get()

        XF_tensor = torch.from_numpy(XF)
        YF_tensor = torch.from_numpy(YF)
        s_tensor = torch.from_numpy(s.get())

        save_file(
            {"XF": XF_tensor, "YF": YF_tensor, "s": s_tensor}, str(save_dir_path / f"{layer_name.replace('.', '_')}.safetensors")
        )

        logger.info(f"✔ Saved factors for {layer_name} on device {device_id} | top singular value: {s[0]:.4f}")
        return True

    except Exception as e:
        logger.error(f"Error in layer {layer_name} on device {device_id}")
        logger.error(traceback.format_exc())
        return False

    finally:
        cp.get_default_memory_pool().free_all_blocks()


def load_all_gradients(base_path: str, model_name: str) -> Dict[str, List[Any]]:
    base_dir = Path(base_path) / model_name
    grad_filenames = sorted([f for f in os.listdir(base_dir) if f.endswith(".safetensors") and (base_dir / f).is_file()])

    all_grads: Dict[str, List[Any]] = {}
    for grad_filename in tqdm(grad_filenames, desc="Loading gradients"):
        grad_file = load_file(filename=str(base_dir / grad_filename))
        for layer_name, gradient_tensor in grad_file.items():
            all_grads.setdefault(layer_name, []).append(gradient_tensor)
    return all_grads


def run_parallel_kron(all_grads: Dict[str, List[Any]], top_k: int = 1, max_workers: int = 4, save_dir: str = "fisher_factors") -> None:
    save_dir_path = Path(save_dir)
    logger = setup_logger(save_dir_path)
    logger.info(f"Starting parallel computation on {len(all_grads)} layers...")

    num_devices = 4
    os.makedirs(save_dir, exist_ok=True)

    ctx = get_context("spawn")
    futures = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        for idx, (layer_name, grad_tensors) in enumerate(all_grads.items()):
            grads_np = [g.to(dtype=torch.float32).cpu().numpy() for g in grad_tensors]
            device_id = idx % num_devices
            futures.append(
                executor.submit(
                    get_kron_factors_worker,
                    layer_name,
                    grads_np,
                    top_k,
                    device_id,
                    save_dir,
                )
            )

        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing Kronecker Factors"):
            try:
                future.result()
            except Exception:
                logger.error("A worker failed during Kronecker factor computation.")


def get_already_processed_layers(output_dir: Path) -> set:
    return {f.stem for f in output_dir.glob("*.safetensors") if f.name != "kron_factors.log"}


if __name__ == "__main__":
    base_path = "/home/jovyan/shares/SR004.nfs2/kurkin/FisherKronecker/grads_output"
    model_name = "Llama-3.1-8B-Instruct_32"
    top_k = 1
    max_workers = 4
    save_dir = Path(base_path) / model_name / "fisher_factors_output_1404"

    logger = setup_logger(save_dir)
    logger.info(f"Loading gradients for {model_name}...")
    all_grads = load_all_gradients(base_path, model_name)

    filtered_grads = {k: v for k, v in all_grads.items() if k.replace(".", "_") not in get_already_processed_layers(save_dir)}

    logger.info(f"Found already processed layers, processing these layers: {filtered_grads.keys()}")

    logger.info("Summary of gradients:")
    for layer_name, grads in filtered_grads.items():
        logger.info(f"  Layer: {layer_name} — {len(grads)} grad(s) | Shape: {grads[0].shape}")

    logger.info(f"Launching Kronecker factor computation using {max_workers} workers...")
    run_parallel_kron(filtered_grads, top_k=top_k, max_workers=max_workers, save_dir=str(save_dir))
    logger.info("Done.")
