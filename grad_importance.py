import argparse
import time
import os
import glob
import concurrent.futures
import json
from pprint import pprint
import traceback

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModel


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


import torch


def compute_abs_cosine_importance(w: torch.Tensor, grads: torch.Tensor, aggregate: bool = True) -> float:
    """
    Compute an absolute‐cosine importance score between weight tensor `w`
    and a batch of gradient tensors `grads`.

    Args:
        w:      Tensor of shape (m, n, …) — your weight matrix/tensor
        grads:  Tensor of shape (B, m, n, …) — a batch of B gradient samples
        aggregate: if True, first average grads over dim 0, then compute
            one |cos(w, mean_grad)|; otherwise compute |cos| per
            sample and average those B values.

    Returns:
        A float: either
            - mean_j |cos(w, grads[j])|   (if aggregate=False), or
            - |cos(w, mean_j grads[j])|    (if aggregate=True).
    """
    # flatten weights
    w_flat = w.view(-1)
    w_norm = w_flat.norm().clamp(min=1e-12)

    # flatten all grads to shape (B, D)
    B = grads.shape[0]
    grads_flat = grads.view(B, -1)

    # per‐sample cosine: (B,)
    g_norms = grads_flat.norm(dim=1).clamp(min=1e-12)
    cosines = (grads_flat @ w_flat) / (g_norms * w_norm)
    if aggregate:
        # mean of absolute cosines
        return cosines.abs().mean().item()
    else:
        return cosines.mean().abs().item()


def process_layer(layer_name, grads, model_state, device):
    """
    Load W from model_state, compute its raw importance,
    return (layer_name, raw_importance).
    """
    W = model_state[layer_name].to(device)
    grads = grads.to(device)
    rawI = compute_abs_cosine_importance(W, grads)
    return layer_name, rawI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory with gradient‐batch .safetensors files")
    parser.add_argument("--model_name_or_path", required=True, help="HuggingFace model name or path to local checkpoint")
    parser.add_argument("--output", default="layer_importances.json", help="Path to write out the JSON summary")
    parser.add_argument("--num_devices", type=int, default=1, help="Number of CUDA devices for parallelism")
    parser.add_argument("--layer_list", type=str, default=None, help="Optional text file with one layer name per line")
    parser.add_argument("--target_retention", type=float, default=0.8, help="t_rr in [0,1]")
    parser.add_argument("--min_retention", type=float, default=0.2, help="m_rr in [0,t_rr]")
    args = parser.parse_args()

    t0 = time.time()
    print("Loading grads...")
    # 1) load all gradients
    all_grads = load_grads_from_batches(args.input)
    layer_names = sorted(all_grads.keys())
    if args.layer_list:
        with open(args.layer_list) as f:
            target_layers = [L.strip() for L in f if L.strip()]
    else:
        target_layers = layer_names

    print("Loading model...")
    # 2) load the model state dict
    model = AutoModel.from_pretrained(args.model_name_or_path, torch_dtype=torch.float32)
    model_state = model.state_dict()
    del model  # free memory
    model_state = {"model." + k: v for k, v in model_state.items()}

    # 3) compute raw importances in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(target_layers), args.num_devices * 2)) as exe:
        futures = []
        for idx, layer in enumerate(target_layers):
            device = f"cuda:{idx % args.num_devices}" if torch.cuda.is_available() else "cpu"
            grads = torch.stack(all_grads[layer])
            grads = grads.float().cpu()
            print(f"\n🚀 Launching importance estimation for layer: {layer} on device {device} with grads Bxmxn: {grads.shape}")
            futures.append(exe.submit(process_layer, layer, grads, model_state, device))

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                print(f"\n{result[0]} estimation done.")
            except Exception as e:
                print(f"❌ A layer failed with error: {e}")
                print(traceback.format_exc())

    # 4) normalize importances and compute compression ratios
    # raw_vals = torch.tensor([raw for _, raw in results], dtype=torch.float32)
    # mean_raw = raw_vals.mean().clamp(min=1e-12).item()

    # after collecting raw_importances:
    raws = torch.tensor([raw for _, raw in results], dtype=torch.float32)
    log_sum = (1 / raws.log()).sum()
    I_min, I_max = raws.min().item(), raws.max().item()
    span = max(I_max - I_min, 1e-12)

    summary = {}
    for layer, raw in results:
        # 1) min–max normalize into [0,1]
        I_norm = (raw - I_min) / span
        # 2) map into [m_rr, t_rr]
        rho = args.min_retention + I_norm * (args.target_retention - args.min_retention)
        summary[layer] = {"raw_importance": raw, "normalized_importance": I_norm, "compression_retention_rho": rho}

    # 5) dump to JSON
    args.output = os.path.join(args.input, args.output)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"✅ Computed importances for {len(summary)} layers in {time.time()-t0:.1f}s")
    print(f"✔️  Written summary → {args.output}")


if __name__ == "__main__":
    main()
