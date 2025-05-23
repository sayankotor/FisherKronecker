import os
import pickle
import logging
import traceback
from pathlib import Path
from multiprocessing import Process, get_context
import time

import numpy as np
import torch
import cupy as cp
from cupyx.scipy.sparse.linalg import svds, LinearOperator
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm


def setup_logger(save_dir: Path) -> logging.Logger:
    """Set up a logger that writes both to a file and to the console."""
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


def get_kron_factors(layer_name, list_of_grads, device_id=0, save_dir="fisher_factors"):
    """Compute Kronecker factors for a single layer."""
    save_dir_path = Path(save_dir)
    logger = setup_logger(save_dir_path)

    start_time = time.time()
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
            shape=(m * m, n * n),
            matvec=matvec,
            rmatvec=r_matvec,
            dtype=cp.float32,
        )

        u, s, vt = svds(linop, k=1, return_singular_vectors=True)

        sidx = cp.argsort(-s)
        s = s[sidx]
        u = u[:, sidx]
        v = vt[sidx, :].T

        x_star = u[:, 0] * s[0]
        y_star = v[:, 0]

        XF = x_star.reshape(m, m, order="F").get()
        YF = y_star.reshape(n, n, order="F").get()

        XF_tensor = torch.from_numpy(XF)
        YF_tensor = torch.from_numpy(YF)
        s_tensor = torch.from_numpy(s.get())

        save_file(
            {"XF": XF_tensor, "YF": YF_tensor, "s": s_tensor}, str(save_dir_path / f"{layer_name.replace('.', '_')}.safetensors")
        )

        logger.info(
            f"✔ Saved factors for {layer_name} on device {device_id} | "
            f"top singular value: {s[0]:.4f} | "
            f"time: {time.time() - start_time:.2f}s"
        )
        return True

    except Exception as e:
        logger.error(f"Error in layer {layer_name} on device {device_id}")
        logger.error(traceback.format_exc())
        return False

    finally:
        cp.get_default_memory_pool().free_all_blocks()


def load_all_gradients(base_path: str, model_name: str):
    """Load all gradients from safetensors files."""
    base_dir = Path(base_path) / model_name
    grad_filenames = sorted([f for f in os.listdir(base_dir) if f.endswith(".safetensors") and (base_dir / f).is_file()])

    all_grads = {}
    for grad_filename in tqdm(grad_filenames, desc="Loading gradients"):
        grad_file = load_file(filename=str(base_dir / grad_filename))
        for layer_name, gradient_tensor in grad_file.items():
            all_grads.setdefault(layer_name, []).append(gradient_tensor)
    return all_grads


def process_layer(layer_name, grads, device_id, save_dir):
    """Process a single layer in a separate process."""
    grads_np = [g.to(dtype=torch.float32).cpu().numpy() for g in grads]
    p = Process(target=get_kron_factors, args=(layer_name, grads_np, device_id, save_dir))
    p.start()
    p.join()


def get_already_processed_layers(output_dir: Path) -> set:
    return {f.stem for f in output_dir.glob("*.safetensors") if f.name != "kron_factors.log"}


if __name__ == "__main__":
    base_path = "/home/jovyan/shares/SR004.nfs2/kurkin/FisherKronecker/grads_output"
    model_name = "Llama-3.1-8B-Instruct_32"
    save_dir = Path(base_path) / model_name / "fisher_factors_output_1404"
    num_devices = 4

    logger = setup_logger(save_dir)
    logger.info(f"Loading gradients for {model_name}...")

    all_grads = load_all_gradients(base_path, model_name)
    filtered_grads = {k: v for k, v in all_grads.items() if k.replace(".", "_") not in get_already_processed_layers(save_dir)}

    logger.info(f"Found already processed layers, processing these layers: {filtered_grads.keys()}")

    logger.info("Summary of gradients:")
    for layer_name, grads in filtered_grads.items():
        logger.info(f"  Layer: {layer_name} — {len(grads)} grad(s) | Shape: {grads[0].shape}")

    start_time = time.time()
    logger.info("Starting Kronecker factor computation...")

    layers_to_process = list(filtered_grads.keys())

    for idx, layer_name in enumerate(layers_to_process):
        logger.info(f"Processing layer {idx+1}/{len(layers_to_process)}: {layer_name}")
        device_id = idx % num_devices
        process_layer(layer_name, filtered_grads[layer_name], device_id, str(save_dir))

    total_time = time.time() - start_time
    logger.info(f"Completed in {total_time:.2f} seconds.")
