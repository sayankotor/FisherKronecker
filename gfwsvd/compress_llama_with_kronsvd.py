import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from safetensors.torch import load_file
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

# Configure logging for detailed progress tracking
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def find_and_replace_layer(model: nn.Module, layer_name: str, replacement_module: nn.Module):
    """
    Dynamically replaces a layer in a PyTorch model by its qualified name.

    Handles both attribute access (e.g., model.mlp) and indexed access within
    ModuleLists/Sequentials (e.g., model.layers.0). This is essential for
    replacing specific layers in transformer architectures.

    Args:
        model: The parent model containing the layer to replace
        layer_name: Dot-separated path to the layer (e.g., "model.layers.10.mlp.down_proj")
        replacement_module: The new module to insert at the specified location

    Raises:
        AttributeError: If an attribute in the path doesn't exist
        IndexError: If an index in the path is out of bounds
        ValueError: If a path component expected to be numeric is invalid
    """
    attrs = layer_name.split(".")

    # Navigate to the parent of the target layer
    parent_module = model
    for i, attr_name in enumerate(attrs[:-1]):
        try:
            if attr_name.isdigit():
                # Handle indexed access (e.g., layers.0)
                idx = int(attr_name)
                parent_module = parent_module[idx]
            else:
                # Handle attribute access (e.g., model.layers)
                parent_module = getattr(parent_module, attr_name)
        except (AttributeError, IndexError, ValueError) as e:
            logging.error(f"Error accessing '{attr_name}' in path '{layer_name}' at index {i}")
            logging.error(f"Parent module type: {type(parent_module)}")
            raise e

    # Replace the final target layer
    final_attr_name = attrs[-1]
    try:
        if final_attr_name.isdigit():
            idx = int(final_attr_name)
            logging.info(f"Replacing indexed layer '{layer_name}' at position {idx}")
            parent_module[idx] = replacement_module
        else:
            logging.info(f"Replacing attribute layer '{layer_name}'")
            setattr(parent_module, final_attr_name, replacement_module)
    except (AttributeError, IndexError, ValueError) as e:
        logging.error(f"Error replacing final layer '{final_attr_name}' in '{layer_name}'")
        raise e

    logging.debug(f"Successfully replaced layer '{layer_name}'")


def matrix_sqrt_invsqrt(X: torch.Tensor, lmbd: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes matrix square root and inverse square root using eigendecomposition.

    This function is crucial for the Kronecker-factored SVD method, as it transforms
    the Kronecker factors into a form suitable for stable SVD decomposition.

    Args:
        X: Input matrix (must be symmetric positive semi-definite)
        lmbd: Initial regularization parameter for numerical stability

    Returns:
        Tuple of (X^{1/2}, X^{-1/2}) matrices
    """
    # Iteratively increase regularization until matrix is positive definite
    while lmbd < 0.1:
        eigvals, Q = torch.linalg.eigh(X + torch.eye(X.shape[0], device=X.device) * lmbd * X.diag())
        if torch.all(eigvals.real > -1e-7):
            break
        lmbd *= 2

    # Clamp eigenvalues to ensure numerical stability
    eigvals = eigvals.clamp(1e-12)

    # Compute matrix functions using eigendecomposition
    X_sqrt = Q @ torch.diag(eigvals.sqrt()) @ Q.T
    X_inv_sqrt = Q @ torch.diag(eigvals.rsqrt()) @ Q.T

    return X_sqrt, X_inv_sqrt


def factorize_layer_kron_svd(
    layer: nn.Linear,
    XF: np.ndarray,
    YF: np.ndarray,
    rank: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> nn.Sequential:
    """
    Decomposes a linear layer using Kronecker-factored SVD.

    This is the core compression algorithm. Given a weight matrix W and its
    Kronecker factors XF and YF, it computes a low-rank factorization:
    W ≈ W2 @ W1, where W1 has shape (rank, in_features) and W2 has shape (out_features, rank).

    The method uses the transformation W_tilde = XF^{1/2} @ W @ YF^{1/2}, performs SVD
    on W_tilde, and then transforms back to obtain the final factorization.

    Args:
        layer: Original linear layer to decompose
        XF: Kronecker factor for input features (in_features × in_features)
        YF: Kronecker factor for output features (out_features × out_features)
        rank: Target rank for the low-rank approximation
        device: Device for the new layers
        dtype: Data type for the new layers

    Returns:
        Sequential module containing two linear layers (W1 and W2)
    """
    # Extract weight matrix and bias
    W = layer.weight.data.to(device="cpu", dtype=torch.float32).numpy()
    bias = layer.bias.data.to(device="cpu", dtype=torch.float32).numpy() if layer.bias is not None else None
    out_features, in_features = W.shape

    # Validate Kronecker factor dimensions
    assert YF.shape == (out_features, out_features), f"Y factor shape mismatch: {YF.shape}"
    assert XF.shape == (in_features, in_features), f"X factor shape mismatch: {XF.shape}"

    logging.info(f"Factorizing layer with shape ({out_features}, {in_features}) to rank {rank}")

    # Convert to PyTorch tensors for matrix operations
    XF_tensor = torch.tensor(XF, dtype=torch.float32)
    YF_tensor = torch.tensor(YF, dtype=torch.float32)
    W_tensor = torch.tensor(W, dtype=torch.float32)

    # Compute matrix square roots and inverse square roots
    X_sqrt, X_inv_sqrt = matrix_sqrt_invsqrt(XF_tensor)
    Y_sqrt, Y_inv_sqrt = matrix_sqrt_invsqrt(YF_tensor)

    # Transform weight matrix for stable SVD
    W_tilde = X_sqrt.T @ W_tensor @ Y_sqrt

    # Perform SVD on transformed matrix
    logging.info(f"Performing SVD on transformed matrix ({out_features}×{in_features})")
    U_hat, S_hat, Vt_hat = torch.linalg.svd(W_tilde, full_matrices=False)
    logging.info(f"SVD complete. Top 5 singular values: {S_hat[:5].tolist()}")

    # Truncate to target rank and ensure positive singular values
    rank = min(rank, len(S_hat)) if rank > 0 else len(S_hat)
    S_hat_positive = torch.clamp(S_hat[:rank], min=1e-8)
    S_hat_sqrt = torch.sqrt(S_hat_positive)
    S_hat_diag = torch.diag(S_hat_sqrt)

    # Compute factorized weight matrices
    W1 = S_hat_diag @ Vt_hat[:rank, :] @ X_inv_sqrt
    W2 = Y_inv_sqrt @ U_hat[:, :rank] @ S_hat_diag

    # Create new linear layers
    W1_tensor = W1.to(dtype=dtype, device=device)
    W2_tensor = W2.to(dtype=dtype, device=device)

    linear1 = nn.Linear(in_features=in_features, out_features=rank, bias=False)
    linear1.weight = nn.Parameter(W1_tensor)

    linear2 = nn.Linear(in_features=rank, out_features=out_features, bias=(bias is not None))
    linear2.weight = nn.Parameter(W2_tensor)
    if bias is not None:
        linear2.bias = nn.Parameter(torch.tensor(bias, dtype=dtype, device=device))

    factorized_layer = nn.Sequential(linear1, linear2)
    factorized_layer.to(device=device, dtype=dtype)

    logging.info(f"Factorization complete. New layer shapes: ({rank}, {in_features}) → ({out_features}, {rank})")
    return factorized_layer


def calculate_rank_from_ratio(
    in_features: int,
    out_features: int,
    param_ratio: float,
    rank_alignment: int = 1,
    min_rank: int = 1,
) -> int:
    """
    Calculates the SVD rank needed to achieve a target parameter reduction ratio.

    For a linear layer with weight matrix W (out_features × in_features), the original
    parameter count is out_features × in_features. After factorization into W2 @ W1,
    the new parameter count is rank × (out_features + in_features).

    This function solves: rank × (out_features + in_features) = param_ratio × (out_features × in_features)

    Args:
        in_features: Input dimension of the layer
        out_features: Output dimension of the layer
        param_ratio: Target fraction of original parameters to retain (0.0 to 1.0)
        rank_alignment: Round rank up to nearest multiple of this value (for efficiency)
        min_rank: Minimum allowed rank

    Returns:
        Calculated rank for the factorization
    """
    if param_ratio <= 0:
        return min_rank
    if param_ratio >= 1:
        return min(out_features, in_features)

    original_params = in_features * out_features
    target_params = original_params * param_ratio

    # Solve for rank: target_params = rank * (in_features + out_features)
    denominator = in_features + out_features
    if denominator == 0:
        return min_rank

    rank_float = target_params / denominator

    # Apply rank alignment for computational efficiency
    if rank_alignment > 1:
        rank = int(math.ceil(rank_float / rank_alignment) * rank_alignment)
    else:
        rank = int(math.ceil(rank_float))

    # Ensure rank is within valid bounds
    max_rank = min(out_features, in_features)
    rank = max(min_rank, min(rank, max_rank))

    return rank


def compress_llama_with_kron_svd(
    model_name_or_path: str,
    kron_factors_dir: str,
    sensitivity_dict: Dict[str, float],
    output_dir: str = ".",
    rank_alignment: int = 8,
    min_rank: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[nn.Module, AutoTokenizer]:
    """
    Applies Kronecker-factored SVD compression to a LLaMA model.

    This function loads a pre-trained model, identifies linear layers for compression,
    and replaces them with low-rank factorizations based on sensitivity scores.

    Args:
        model_name_or_path: HuggingFace model identifier or local path
        kron_factors_dir: Directory containing pre-computed Kronecker factors
        sensitivity_dict: Mapping from layer names to parameter reduction ratios
        output_dir: Directory to save the compressed model
        rank_alignment: Alignment constraint for ranks (improves efficiency)
        min_rank: Minimum rank for any factorization
        device: Device for model operations

    Returns:
        Tuple of (compressed_model, tokenizer)
    """
    kron_factors_path = Path(kron_factors_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    logging.info(f"Loading model: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    original_dtype = next(model.parameters()).dtype
    logging.info(f"Model loaded with dtype {original_dtype}")
    model.to(device)

    # Identify linear layers for compression
    linear_layers_to_compress = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_layers_to_compress[name] = module

    logging.info(f"Found {len(linear_layers_to_compress)} linear layers for potential compression")

    # Track compression statistics
    layers_processed_count = 0
    layers_failed_count = 0
    layers_skipped_count = 0

    # Process each layer
    for layer_name, layer_module in tqdm(linear_layers_to_compress.items(), desc="Compressing Layers"):

        # Check if layer has sensitivity information
        if layer_name not in sensitivity_dict:
            logging.warning(f"Skipping {layer_name}: Not in sensitivity dictionary")
            layers_skipped_count += 1
            continue

        param_ratio = sensitivity_dict[layer_name]

        # Skip layers that don't need compression
        if param_ratio >= 1.0:
            logging.info(f"Skipping {layer_name}: No compression needed (ratio={param_ratio:.2f})")
            layers_skipped_count += 1
            continue

        # Calculate target rank based on sensitivity
        in_features = layer_module.in_features
        out_features = layer_module.out_features
        target_rank = calculate_rank_from_ratio(
            in_features=in_features,
            out_features=out_features,
            param_ratio=param_ratio,
            rank_alignment=rank_alignment,
            min_rank=min_rank,
        )

        logging.info(f"Layer {layer_name}: ratio={param_ratio:.3f} → rank={target_rank} (alignment={rank_alignment})")

        # Load pre-computed Kronecker factors
        factor_filename = kron_factors_path / f"{layer_name.replace('.', '_')}.safetensors"
        if not factor_filename.exists():
            logging.warning(f"Skipping {layer_name}: Factor file not found at {factor_filename}")
            layers_skipped_count += 1
            continue

        # Perform layer factorization
        try:
            # Load Kronecker factors
            factors = load_file(factor_filename)
            XF = factors["XF"].to(dtype=torch.float32).cpu().numpy()
            YF = factors["YF"].to(dtype=torch.float32).cpu().numpy()

            # Create factorized layer
            factorized_sequential = factorize_layer_kron_svd(
                layer=layer_module, XF=XF, YF=YF, rank=target_rank, device=device, dtype=layer_module.weight.dtype
            )

            # Replace original layer with factorized version
            find_and_replace_layer(model, layer_name, factorized_sequential)
            layers_processed_count += 1

        except Exception as e:
            logging.error(f"Failed to process {layer_name}: {e}", exc_info=True)
            layers_failed_count += 1

    # Log compression summary
    logging.info(f"Compression complete:")
    logging.info(f"  Processed: {layers_processed_count}")
    logging.info(f"  Skipped: {layers_skipped_count}")
    logging.info(f"  Failed: {layers_failed_count}")

    return model, tokenizer


def load_sensitivity_dict(path: str) -> Dict[str, float]:
    """Load layer sensitivity dictionary from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


class IndexDataset(Dataset):
    """Simple dataset wrapper for tokenized sequences."""

    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)


def get_test_dataset(name: str, tokenizer, seq_len: int = 2048) -> IndexDataset:
    """
    Prepares evaluation datasets for perplexity measurement.

    Supports WikiText-2, Penn Treebank, and C4 datasets. Text is tokenized
    and split into fixed-length sequences for consistent evaluation.

    Args:
        name: Dataset name ('wikitext2', 'ptb', or 'c4')
        tokenizer: HuggingFace tokenizer
        seq_len: Maximum sequence length

    Returns:
        IndexDataset containing tokenized sequences
    """

    def process_data(samples, field_name):
        # Tokenize and concatenate all text
        test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors="pt").input_ids[0]

        # Split into fixed-length sequences
        test_ids_batch = []
        nsamples = test_ids.numel() // seq_len

        for i in range(nsamples):
            batch = test_ids[(i * seq_len) : ((i + 1) * seq_len)]
            test_ids_batch.append(batch)

        test_ids_batch = torch.stack(test_ids_batch)
        return IndexDataset(tensors=test_ids_batch)

    # Load appropriate dataset
    if "wikitext2" in name:
        test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return process_data(test_data, "text")
    elif "ptb" in name:
        test_data = load_dataset("ptb_text_only", "penn_treebank", split="test")
        return process_data(test_data, "sentence")
    elif "c4" in name:
        test_data = load_dataset("json", data_files="utils/c4-validation.json")["train"]
        return process_data(test_data[0:2000], "text")
    else:
        raise ValueError(f"Unknown dataset: {name}")


@torch.no_grad()
def evaluate_perplexity(model: nn.Module, dataset: IndexDataset, batch_size: int = 4, device: str = "cuda") -> float:
    """
    Evaluates model perplexity on a tokenized dataset.

    Perplexity is computed as the exponential of the average negative log-likelihood
    across all tokens in the dataset. Lower perplexity indicates better language modeling.

    Args:
        model: Model to evaluate
        dataset: Tokenized evaluation dataset
        batch_size: Batch size for evaluation
        device: Device for computation

    Returns:
        Perplexity score (lower is better)
    """
    model.to(device)
    model.eval()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    nlls = []  # Negative log-likelihoods

    for batch in tqdm(dataloader, desc="Evaluating perplexity"):
        batch = batch.to(device)

        # Forward pass
        outputs = model(batch, use_cache=False)
        logits = outputs.logits

        # Skip batches with numerical issues
        if not torch.isfinite(logits).all():
            logging.warning("Skipping batch with non-finite logits")
            continue

        # Compute cross-entropy loss for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss)

    # Compute perplexity as exponential of average negative log-likelihood
    ppl = torch.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl


def evaluate_model(
    model: nn.Module,
    tokenizer,
    datasets: List[str] = ["wikitext2", "ptb", "c4"],
    seq_len: int = 2048,
    batch_size: int = 4,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Comprehensive model evaluation on multiple datasets.

    Evaluates the model's language modeling performance using perplexity
    on standard benchmarks. This provides a quantitative measure of
    compression quality.

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for preprocessing
        datasets: List of dataset names to evaluate on
        seq_len: Maximum sequence length
        batch_size: Batch size for evaluation
        device: Device for computation

    Returns:
        Dictionary mapping dataset names to perplexity scores
    """
    # Configure model for evaluation
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.to(device)

    results = {}

    for dataset_name in datasets:
        logging.info(f"Evaluating on {dataset_name}")
        try:
            # Prepare dataset
            test_dataset = get_test_dataset(dataset_name, tokenizer, seq_len)

            # Evaluate perplexity
            ppl = evaluate_perplexity(model, test_dataset, batch_size=batch_size, device=device)

            results[dataset_name] = ppl
            logging.info(f"{dataset_name} perplexity: {ppl:.4f}")

        except Exception as e:
            logging.error(f"Failed to evaluate on {dataset_name}: {e}")
            results[dataset_name] = float("nan")

    # Restore model configuration
    model.config.use_cache = use_cache

    return results


def main():
    parser = argparse.ArgumentParser(description="Kronecker-Factored SVD Compression for LLaMA models")
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--kron_factors_dir", type=str, required=True, help="Directory containing Kronecker factor .safetensors files")
    parser.add_argument("--sensitivity_path", type=str, required=True, help="Path to JSON sensitivity dictionary")
    parser.add_argument("--output_dir", type=str, default="./compressed_model", help="Directory to save compressed model")
    parser.add_argument("--rank_alignment", type=int, default=8, help="Alignment for SVD rank")
    parser.add_argument("--min_rank", type=int, default=1, help="Minimum allowed SVD rank")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Computation device")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for perplexity evaluation")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length for evaluation")
    parser.add_argument("--eval_datasets", type=str, nargs="*", default=["wikitext2", "ptb"], help="Datasets for evaluation")
    args = parser.parse_args()

    # Load sensitivity dictionary
    logging.info("Loading sensitivity dictionary...")
    sensitivity_dict = load_sensitivity_dict(args.sensitivity_path)
    logging.info(f"Loaded sensitivity data for {len(sensitivity_dict)} layers")

    # Apply compression
    logging.info("Starting model compression...")
    compressed_model, tokenizer = compress_llama_with_kron_svd(
        model_name_or_path=args.model_name,
        kron_factors_dir=args.kron_factors_dir,
        sensitivity_dict=sensitivity_dict,
        output_dir=args.output_dir,
        rank_alignment=args.rank_alignment,
        min_rank=args.min_rank,
        device=args.device,
    )

    # Evaluate compressed model
    logging.info("Evaluating compressed model...")
    evaluation_results = evaluate_model(
        model=compressed_model,
        tokenizer=tokenizer,
        datasets=args.eval_datasets,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=args.device,
    )

    logging.info("=== COMPRESSION RESULTS ===")
    for dataset, ppl in evaluation_results.items():
        logging.info(f"{dataset.upper()} perplexity: {ppl:.4f}")

    # Save compressed model
    logging.info(f"Saving compressed model to {args.output_dir}")
    compressed_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logging.info("Compression pipeline completed successfully!")


if __name__ == "__main__":
    main()