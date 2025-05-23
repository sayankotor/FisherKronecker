import argparse
import time
import os
import glob
import concurrent.futures
import json
from pprint import pprint
import traceback
from tqdm.auto import tqdm

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModel


def process_layer(layer_name, kron_factors, model_state, device, threshold=0.8):
    """Compute Fisher-weighted importance metrics for a layer."""
    try:
        # 1. Get weight matrix and Fisher factors
        W = model_state[layer_name].to(device).float()
        m, n = W.shape

        # Extract K-FAC matrices (X and Y)
        base_name = layer_name
        X = kron_factors[f"{base_name}_XF"].to(device).float()
        Y = kron_factors[f"{base_name}_YF"].to(device).float()

        # 2. Compute matrix square roots using stable eigendecomposition
        def matrix_sqrt_invsqrt(X, lmbd=1e-6):
            while True and lmbd < 0.1:
                eigvals, Q = torch.linalg.eigh(X + torch.eye(X.shape[0], device=X.device) * lmbd * X.diag())
                if torch.all(eigvals.real > -1e-7):
                    break
                else:
                    lmbd *= 2
            # print(f"decomposed with lambda={lmbd}")
            eigvals = eigvals.clamp(1e-12)
            X_sqrt = Q @ torch.diag(eigvals.sqrt()) @ Q.T
            X_inv_sqrt = Q @ torch.diag(eigvals.rsqrt()) @ Q.T
            return X_sqrt, X_inv_sqrt

        X_sqrt, X_inv_sqrt = matrix_sqrt_invsqrt(X)
        Y_sqrt, Y_inv_sqrt = matrix_sqrt_invsqrt(Y)

        # 3. Compute Fisher-weighted SVD
        F_W = X_sqrt @ W @ Y_sqrt
        U, S, Vh = torch.linalg.svd(F_W, full_matrices=False)

        # 4. Calculate information metrics
        total_info = S.pow(2).sum()
        max_rank = min(m, n)
        ranks = (torch.arange(0.05, 1.0125, 0.0125) * max_rank).int()
        metrics = {
            "ranks": ranks,
            "singular_values": S.cpu().tolist(),
            "fisher_errors": [],
            "fisher_trace": (torch.trace(X) * torch.trace(Y)).item(),
            "info_retention": [],
            "shape": [m, n],
            "param_count": m * n,
        }
        print(layer_name, metrics["fisher_trace"])

        # 5. Compute error curves for all possible ranks
        for r in ranks:
            # Low-rank approximation
            W_approx = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]

            W_lowrank = X_inv_sqrt @ W_approx @ Y_inv_sqrt

            # Fisher-weighted error calculation
            diff = W_lowrank - W
            numerator = torch.trace(X @ diff @ Y @ diff.T)
            denominator = torch.trace(X @ W @ Y @ W.T)
            metrics["fisher_errors"].append((numerator / denominator).item())
            metrics["info_retention"].append(S[:r].pow(2).sum().item() / total_info.item())

        # 6. Calculate optimal rank based on retention thresholds
        for r_val, info in zip(metrics["ranks"], metrics["info_retention"]):
            if info >= threshold:
                metrics["optimal_rank"] = r_val.item()
                break
        else:
            metrics["optimal_rank"] = max_rank
        metrics["compressed_params"] = metrics["optimal_rank"] * (m + n)

        return layer_name, metrics

    except Exception as e:
        print(f"Error processing {layer_name}: {str(e)}")
        traceback.print_exc()
        return layer_name, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory with gradient‐batch .safetensors files")
    parser.add_argument("--model_name_or_path", required=True, help="HuggingFace model name or path to local checkpoint")
    parser.add_argument("--output", default="fisher_layer_importances.json", help="Path to write out the JSON summary")
    parser.add_argument("--num_devices", type=int, default=1, help="Number of CUDA devices for parallelism")
    parser.add_argument("--layer_list", type=str, default=None, help="Optional text file with one layer name per line")
    parser.add_argument("--target_retention", type=float, default=0.8, help="t_rr in [0,1]")
    parser.add_argument("--min_retention", type=float, default=0.2, help="m_rr in [0,t_rr]")
    args = parser.parse_args()

    t0 = time.time()
    print("Loading fisher estimate...")
    # 1) load all gradients
    kron_factors = load_file(os.path.join(args.input, "kronfactors.safetensors"))
    print("Loading model...")
    # 2) load the model state dict
    model = AutoModel.from_pretrained(args.model_name_or_path, torch_dtype=torch.float32)
    model_state = model.state_dict()
    del model  # free memory
    model_state = {"model." + k: v for k, v in model_state.items()}

    layer_names = set([k.split("weight", maxsplit=1)[0] + "weight" for k in kron_factors.keys()])
    if args.layer_list:
        with open(args.layer_list) as f:
            target_layers = [L.strip() for L in f if L.strip()]
    else:
        target_layers = layer_names
    # target_layers = list(target_layers)[:12]
    torch.linalg.norm(torch.ones(1, device="cuda:0"))

    # 3) compute raw importances in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(target_layers), args.num_devices * 2)) as exe:
        futures = []
        for idx, layer in enumerate(target_layers):
            device = f"cuda:{idx % args.num_devices}" if torch.cuda.is_available() else "cpu"
            print(f"\n🚀 Launching importance estimation for layer: {layer} on device {device}")
            futures.append(exe.submit(process_layer, layer, kron_factors, model_state, device))

        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                print(f"\n{result[0]} estimation done.")
            except Exception as e:
                print(f"❌ A layer failed with error: {e}")
                print(traceback.format_exc())

    # Modified results processing
    summary = {
        "global_metrics": {
            "target_retention": args.target_retention,
            "min_retention": args.min_retention,
            "total_params": sum(m * n for layer in target_layers for m, n in [model_state[layer].shape]),
        },
        "layers": {},
    }

    for layer, data in results:
        if data:
            summary["layers"][layer] = {
                "shape": data["shape"],
                "sv": data["singular_values"],
                "fisher_errors": data["fisher_errors"],
                "info_retention": data["info_retention"],
                "optimal_rank": data["optimal_rank"],
                "compression_ratio": data["param_count"] / data["compressed_params"],
            }

    # 5) dump to JSON
    args.output = os.path.join(args.input, args.output)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"✅ Computed importances for {len(summary)} layers in {time.time()-t0:.1f}s")
    print(f"✔️  Written summary → {args.output}")


if __name__ == "__main__":
    main()
