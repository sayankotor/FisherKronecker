import argparse
import os
import glob
import time
import traceback
import concurrent.futures

import cupy as cp
from cupyx.scipy.sparse.linalg import svds, LinearOperator

import torch
from safetensors.torch import load_file, save_file

from pprint import pprint


def load_grads_from_batches(input_dir):
    """
    Scan `input_dir` for all .safetensors files and
    return a dict layer_name -> list of gradient tensors.
    """
    grads_accum = {}
    files = sorted(glob.glob(os.path.join(input_dir, "grads_batch_*.safetensors")))
    print(f"📦 Found {len(files)} gradient batch files...")
    for f in files:
        batch_data = load_file(f)
        for key, tensor in batch_data.items():
            grads_accum.setdefault(key, []).append(tensor)
    return grads_accum


def get_kron_factors(list_of_grads, top_k=10, layer_name="linear", device_id=0, chunk_size=4):
    """
    Perform parallel by input layers Fisher Matrix approximation in the form of Kronecker Decomposition.
    """

    def matvec(vec, grad_vectors, chunk_size=4):
        k, m, n = grad_vectors.shape
        V = vec.reshape(n, n, order="F")
        result = cp.zeros((m, m), dtype=cp.float32)
        for i in range(0, k, chunk_size):
            chunk = grad_vectors[i : i + chunk_size]
            prod = chunk @ V @ chunk.transpose(0, 2, 1)
            result += cp.sum(prod, axis=0)
        return (result / k).T.ravel()

    def r_matvec(vec, grad_vectors, chunk_size=4):
        k, m, n = grad_vectors.shape
        V = vec.reshape(m, m, order="F")
        result = cp.zeros((n, n), dtype=cp.float32)
        for i in range(0, k, chunk_size):
            chunk = grad_vectors[i : i + chunk_size]
            prod = chunk.transpose(0, 2, 1) @ V @ chunk
            result += cp.sum(prod, axis=0)
        return (result / k).T.ravel()

    num_devices = cp.cuda.runtime.getDeviceCount()
    device_pool = [cp.cuda.Device(i) for i in range(num_devices)]
    with device_pool[device_id]:
        m, n = list_of_grads[0].shape
        grad_vectors = cp.stack([cp.asarray(grad).reshape(m, n, order="F") for grad in list_of_grads])
        linop = LinearOperator(
            shape=(m * m, n * n),
            matvec=lambda vec: matvec(vec, grad_vectors),
            rmatvec=lambda vec: r_matvec(vec, grad_vectors),
            dtype=cp.float32,
        )

        u, s, vt = svds(linop, k=top_k, return_singular_vectors=True)
        print(f"✔ Layer {layer_name} on device {device_id} done | singular values: {s}")
        sidx = cp.argsort(-s)
        s = s[sidx]
        u = u[:, sidx]
        v = vt[sidx, :].T

        XF = (u[:, 0] * s[0] ** 0.5).reshape(m, m, order="F")
        YF = (s[0] ** 0.5 * v[:, 0]).reshape(n, n, order="F")

        return layer_name, torch.tensor(XF.get(), dtype=torch.bfloat16), torch.tensor(YF.get(), dtype=torch.bfloat16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory with gradient batch .safetensors files")
    parser.add_argument("--output", default=None, required=False, help="Path to save Kronecker factors (.safetensors)")
    parser.add_argument("--top_k", type=int, default=1, help="Top-K singular vectors")
    parser.add_argument("--chunk_size", type=int, default=4, help="Chunk size for grad matvecs")
    parser.add_argument("--num_devices", type=int, default=1, help="Number of CUDA devices")
    parser.add_argument("--layer_list", type=str, default=None, help="Optional file with layer names to process")
    args = parser.parse_args()

    start = time.time()
    all_grads = load_grads_from_batches(args.input)
    layer_names = list(all_grads.keys())
    if args.layer_list:
        with open(args.layer_list, "r") as f:
            target_layers = [line.strip() for line in f if line.strip()]
    else:
        target_layers = layer_names

    results = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.num_devices * 2, len(target_layers))) as executor:
        futures = []
        for i, layer_name in enumerate(target_layers):
            grads = torch.stack(all_grads[layer_name])
            grads = grads.float().cpu().numpy()
            device_id = i % args.num_devices
            print(f"\n🚀 Launching compression for layer: {layer_name} on device {device_id} with grads Bxmxn: {grads.shape}")
            futures.append(executor.submit(get_kron_factors, grads, args.top_k, layer_name, device_id, args.chunk_size))

        for future in concurrent.futures.as_completed(futures):
            try:
                layer_name, XF, YF = future.result()
                results[f"{layer_name}_XF"] = XF
                results[f"{layer_name}_YF"] = YF
                print(f"\n{layer_name} compression done.")
            except Exception as e:
                print(f"❌ A layer failed with error: {e}")
                print(traceback.format_exc())

    if args.output == None:
        args.output = os.path.join(args.input, "kronfactors.safetensors")
    print(f"\n💾 Saving Kronecker factors to: {args.output}")
    save_file(results, args.output)
    print(f"\n💾 Saved Kronecker factors to: {args.output}")
    print(f"🕒 Total time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
