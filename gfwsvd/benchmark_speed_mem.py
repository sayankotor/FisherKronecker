import torch
import time
import json
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from statistics import mean, stdev
import torch.nn as nn
import math
from typing import List
from tqdm.auto import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.enabled = False


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    elif device_arg == "cuda":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        try:
            index = int(device_arg)
            if torch.cuda.device_count() > index:
                return torch.device(f"cuda:{index}")
            else:
                raise ValueError(f"CUDA device index {index} is out of range.")
        except ValueError:
            raise ValueError(f"Invalid device argument: {device_arg}")


class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        self.W1 = nn.Linear(in_features, rank, bias=False)
        self.W2 = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x):
        return self.W2(self.W1(x))


def replace_linear_with_lowrank(module, compression_ratio: float = 1.0, compression_config: dict = None):
    for name, submodule in module.named_modules():
        if isinstance(submodule, nn.Linear):
            # Determine the correct ratio
            if compression_config:
                if name not in compression_config:
                    continue
                ratio = compression_config[name]
                if ratio >= 1.0:
                    continue
            else:
                if "proj" not in name:
                    continue
                ratio = compression_ratio
                if ratio >= 1.0:
                    continue

            m, n = submodule.out_features, submodule.in_features
            r = int((1 - ratio) * m * n / (m + n))
            # r = ((r + 31) // 32) * 32  # round up to next multiple of 32
            lowrank = LowRankLinear(n, m, r, bias=(submodule.bias is not None))

            # Find parent module and attribute name to replace
            parent_path, attr = name.rsplit(".", 1) if "." in name else ("", name)
            parent = dict(module.named_modules())[parent_path] if parent_path else module
            setattr(parent, attr, lowrank)

            print(f"Replaced {name} of size {m}x{n} with rank {r}")


def benchmark_model_generate(
    model_name: str,
    prefill_len: int,
    decode_len: int,
    batch_sizes: List[int],
    n_iters: int,
    warmup_iters: int,
    device: torch.device,
    compression_ratio: float,
) -> List[dict]:

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16).to(
        device
    )
    if args.compression_config_json:
        with open(args.compression_config_json, "r") as f:
            compression_config = json.load(f)
    else:
        compression_config = None

    if compression_ratio > 0.0:
        print("Patching layers with lowrank...")
        replace_linear_with_lowrank(model, compression_ratio, compression_config)
        model = model.to(device=device, dtype=model.dtype)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id

    if pad_token_id is None or pad_token_id == eos_token_id:
        pad_token_id = eos_token_id + 1

    valid_token_ids = [i for i in range(vocab_size) if i != pad_token_id]
    valid_token_ids_tensor = torch.tensor(valid_token_ids, device=device)

    def generate_inputs(batch_size, prefill_len):
        rand_indices = torch.randint(low=0, high=len(valid_token_ids), size=(batch_size, prefill_len), device=device)
        input_ids = valid_token_ids_tensor[rand_indices]
        attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask

    generation_kwargs = {
        "min_new_tokens": decode_len,
        "max_new_tokens": decode_len,
        "do_sample": False,
        "top_p": None,
        "use_cache": True,
    }

    results = []

    # Compute weight memory (parameters) in bytes
    def compute_model_size_bytes(model: nn.Module):
        return sum(p.numel() * p.element_size() for p in model.parameters())

    weight_mem_bytes = compute_model_size_bytes(model)
    weight_mem_mb = weight_mem_bytes / (1024**2)

    for batch_size in batch_sizes:
        activation_memories = []

        print(f"Warming up batch {batch_size}...")
        # Warm-up
        with torch.no_grad():
            for _ in range(warmup_iters):
                input_ids, attention_mask = generate_inputs(batch_size, prefill_len)
                _ = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)

        print(f"Generating batch {batch_size}...")
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in tqdm(range(n_iters)):
                input_ids, attention_mask = generate_inputs(batch_size, prefill_len)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                start_time = time.time()
                _ = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    peak_mem = torch.cuda.max_memory_allocated(device)
                    activation_memories.append(peak_mem - weight_mem_bytes)
                times.append(time.time() - start_time)

        total_tokens = batch_size * decode_len
        result = {
            "model": model_name,
            "batch_size": batch_size,
            "prefill_len": prefill_len,
            "decode_len": decode_len,
            "iterations": n_iters,
            "compression_ratio": compression_ratio,
            "mean_latency_sec": mean(times),
            "stdev_latency_sec": stdev(times),
            "p50_latency_sec": sorted(times)[n_iters // 2],
            "p90_latency_sec": sorted(times)[int(n_iters * 0.9)],
            "p99_latency_sec": sorted(times)[min(n_iters - 1, int(n_iters * 0.99))],
            "throughput_tokens_per_sec": total_tokens / sum(times),
            "weight_memory_mb": round(weight_mem_mb, 2),
            "mean_activation_memory_mb": round(mean(activation_memories) / (1024**2), 2),
            "p90_activation_memory_mb": round(sorted(activation_memories)[int(n_iters * 0.9)] / (1024**2), 2),
            "p99_activation_memory_mb": round(sorted(activation_memories)[min(n_iters - 1, int(n_iters * 0.99))] / (1024**2), 2),
        }
        results.append(result)

    return results


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark HuggingFace model generate() latency.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--prefill_len", type=int, default=128)
    parser.add_argument("--decode_len", type=int, default=32)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--n_iters", type=int, default=5)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--output_json", type=str, default="generate_benchmark.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--compression_ratio", type=float, default=0.0, help="Compression ratio p in (0, 1)")
    parser.add_argument(
        "--compression_config_json", type=str, default=None, help="Optional JSON file specifying per-layer compression ratios."
    )

    args = parser.parse_args()
    device = resolve_device(args.device)

    all_results = benchmark_model_generate(
        model_name=args.model_name,
        prefill_len=args.prefill_len,
        decode_len=args.decode_len,
        batch_sizes=args.batch_sizes,
        n_iters=args.n_iters,
        warmup_iters=args.warmup_iters,
        device=device,
        compression_ratio=args.compression_ratio,
    )

    # Append to JSON file if exists
    if os.path.exists(args.output_json):
        with open(args.output_json, "r") as f:
            existing = json.load(f)
        if not isinstance(existing, list):
            raise ValueError("Output JSON must contain a list of results.")
    else:
        existing = []

    existing.extend(all_results)
    with open(args.output_json, "w") as f:
        json.dump(existing, f, indent=4)

    print(f"Benchmark results saved to {args.output_json}")
