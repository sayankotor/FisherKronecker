import argparse
import os
import re
import torch
import random
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import (
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from safetensors.torch import save_file
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
import threading
import queue
import time
from accelerate import Accelerator
from torch.distributed import all_reduce, ReduceOp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from contextlib import nullcontext
import gc


def print_once(*args, **kwargs):
    """Print only from the main process."""
    if not hasattr(print_once, "accelerator"):
        print_once.accelerator = Accelerator()
    print_once.accelerator.print(*args, **kwargs)


class NoOpOptimizer(torch.optim.Optimizer):
    def __init__(self, param):
        dummy_param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        super().__init__([{"params": [dummy_param], "lr": 0.0}], {})

    def step(self, closure=None):
        pass

    def zero_grad(self, set_to_none=False):
        pass


class GradientTrainer(Trainer):
    def __init__(self, *args, layer_pattern=None, step_counter=None, save_queue=None, accelerator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_pattern = layer_pattern
        self.step_counter = step_counter
        self.save_queue = save_queue
        self.accelerator = accelerator
        self._grad_buffer = {}
        self._current_acc_step = 0
        self._register_hooks()

    def _register_hooks(self):
        """Register gradient collection hooks on specified layers."""
        # Use the accelerator's wait_for_everyone to ensure all processes are synchronized
        self.accelerator.wait_for_everyone()

        # Perform hook registration in a context that ensures full parameter visibility
        with self.accelerator.unwrap_model(self.model).unfreeze():
            params = dict(self.model.named_parameters(recurse=True))
            for name, param in params.items():
                # Skip layers that don't match our pattern or aren't weights
                if not "weight" in name or not re.search(self.layer_pattern, name) or not param.requires_grad:
                    continue

                # Skip non-leaf matches (if a child also matches)
                pass_param = any(
                    cname != name and cname.startswith(name + ".") and "weight" in cname and re.search(self.layer_pattern, cname)
                    for cname in params
                )
                if pass_param:
                    continue

                # Register a backward hook to collect gradients
                def grad_hook(grad, name=name):
                    # If we're in a distributed setting, reduce gradients across processes
                    if self.accelerator.num_processes > 1:
                        # Ensure the gradient is on the same device as the accelerator
                        grad = grad.to(self.accelerator.device)
                        # Average gradients across all processes
                        all_reduce(grad, op=ReduceOp.MEAN)

                    # Store the gradient
                    if name in self._grad_buffer:
                        self._grad_buffer[name].add_(grad.detach())
                    else:
                        self._grad_buffer[name] = grad.detach().clone()

                # Register the hook
                param.register_hook(grad_hook)
                print_once(f"[DEBUG] Registered grad-accum hook for: {name}")

    def training_step(self, model, inputs, num_items):
        # Perform the training step
        loss = super().training_step(model, inputs, num_items)

        # Increment our step counter
        self._current_acc_step += 1
        print_once(f"[DEBUG] Step {self.state.global_step}, acc_step {self._current_acc_step}/{self.args.gradient_accumulation_steps}")

        # Check if we've reached the end of gradient accumulation
        if self._current_acc_step >= self.args.gradient_accumulation_steps:
            print_once(f"[INFO] Saving gradients for batch {self.step_counter[0]}")

            # Process and save accumulated gradients
            grad_dict = {
                name: tensor.div(self.args.gradient_accumulation_steps).clone()  # Scale and clone
                for name, tensor in self._grad_buffer.items()
            }
            if len(grad_dict):
                print_once(next(iter(grad_dict.values())).shape)

            # Send for saving (only from main process)
            if self.accelerator.is_main_process:
                self.save_queue.put((self.step_counter[0], grad_dict))

            # Reset buffer and counter
            self._grad_buffer.clear()
            self._current_acc_step = 0
            self.step_counter[0] += 1

            # Clean up
            torch.cuda.empty_cache()
            gc.collect()

        return loss

    def create_optimizer(self):
        print_once("[INFO] Using dummy optimizer (no updates)")
        self.optimizer = NoOpOptimizer(next(self.model.parameters()))
        return self.optimizer


def save_gradient_worker(save_queue, output_path, flush_interval=5):
    """Worker thread function to save gradients to disk"""
    print_once("[INFO] Save worker started")
    last_flush_time = time.time()
    pending_saves = []
    save_count = 0

    while True:
        try:
            # Try to get an item with timeout to allow periodic flushing
            try:
                item = save_queue.get(timeout=1.0)
                if item is None:  # Poison pill
                    break
            except queue.Empty:
                # No new items, check if we should flush pending saves
                if pending_saves and time.time() - last_flush_time > flush_interval:
                    for batch_idx, grads in pending_saves:
                        _save_gradient_batch(batch_idx, grads, output_path)
                    pending_saves.clear()
                    last_flush_time = time.time()
                continue

            # Process the gradient save
            batch_idx, grads = item
            pending_saves.append((batch_idx, grads))

            # If we have enough pending saves or it's been too long, flush them
            if len(pending_saves) >= 3 or time.time() - last_flush_time > flush_interval:
                for b_idx, g in pending_saves:
                    _save_gradient_batch(b_idx, g, output_path)
                    save_count += 1
                pending_saves.clear()
                last_flush_time = time.time()
                print_once(f"[INFO] Saved {save_count} gradient batches so far")

            save_queue.task_done()

        except Exception as e:
            print_once(f"[ERROR] Exception in save worker: {e}")
            import traceback

            traceback.print_exc()

    # Final flush of any remaining items
    for batch_idx, grads in pending_saves:
        _save_gradient_batch(batch_idx, grads, output_path)
        save_count += 1

    print_once(f"[INFO] Save worker finished. Total saved batches: {save_count}")


def _save_gradient_batch(batch_idx, grads, output_path):
    """Helper function to save a single batch of gradients"""
    try:
        # Create a new dict for CPU tensors
        grads_cpu = {}

        # Process each gradient tensor
        for k, v in grads.items():
            # Move to CPU and convert to bfloat16
            cpu_tensor = v.detach().cpu().to(dtype=torch.bfloat16)
            grads_cpu[k] = cpu_tensor

        # Save to disk
        grad_file = os.path.join(output_path, f"grads_batch_{batch_idx:06}.safetensors")
        save_file(grads_cpu, grad_file)
        print_once(f"[DEBUG] Saved gradients to {grad_file}")

        # Clean up
        del grads_cpu

    except Exception as e:
        print_once(f"[ERROR] Failed to save gradient batch {batch_idx}: {e}")
        import traceback

        traceback.print_exc()


def collect_grads(
    output_path,
    model_name,
    dataset_name,
    dataset_config_name,
    dataset_text_field,
    batch_size,
    gradient_accumulation_steps,
    layer_pattern,
    max_steps,
    max_seq_length,
    seed,
    use_bf16,
    use_qlora,
    subsample_size,
    collect_weights,
    gradient_checkpointing,
    num_save_workers=2,
    compression_level=1,
):
    # Initialize Accelerator with FSDP
    accelerator = Accelerator(
        mixed_precision="bf16" if use_bf16 else "no",
        gradient_accumulation_steps=gradient_accumulation_steps,
        split_batches=True,
    )

    # Ensure output directory exists on main process
    if accelerator.is_main_process:
        os.makedirs(output_path, exist_ok=True)

    # Set seeds
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Print startup info (only from main process)
    if accelerator.is_main_process:
        print_once(f"[INFO] Starting gradient collection with:")
        print_once(f"       - Model: {model_name}")
        print_once(f"       - Batch size: {batch_size}")
        print_once(f"       - Gradient accumulation steps: {gradient_accumulation_steps}")
        print_once(f"       - Layer pattern: {layer_pattern}")
        print_once(f"       - Output path: {output_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Prepare quantization config if needed
    quant_config = None
    if use_qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2" if "gemma" not in model_name else "eager",
    )

    # Configure gradient checkpointing
    if gradient_checkpointing:
        if hasattr(model.config, "gradient_checkpointing"):
            model.config.gradient_checkpointing = True
            model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            print_once("[INFO] Enabling gradient checkpointing with non-reentrant mode")
            model.gradient_checkpointing_enable()
    else:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()

    # Apply LoRA if needed
    if use_qlora:
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=32, lora_dropout=0.1, bias="none")
        model = get_peft_model(model, lora_config)

    # Freeze all parameters by default
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layers matching the pattern
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if re.search(layer_pattern, name):
            param.requires_grad = True
            unfrozen_count += 1
            print_once(f"[INFO] Unfrozen: {name}")

    if unfrozen_count == 0:
        print_once(f"[WARNING] No parameters matched the pattern '{layer_pattern}'. Check your regex.")

    # Print parameter stats
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_once(f"[INFO] Total params: {total:,} | Trainable: {trainable:,} ({100 * trainable / total:.2f}%)")

    if trainable == 0:
        print_once("[ERROR] No trainable parameters found. Exiting.")
        return

    # Set model to train mode
    model.train()

    # Load and prepare dataset
    subsample_path = os.path.join(dataset_name, "sample", str(subsample_size))
    if os.path.exists(subsample_path):
        print_once(f"[INFO] Loading cached subsampled dataset from: {dataset_name}/{subsample_size}")
        ds = load_from_disk(subsample_path)
    else:
        print_once(f"[INFO] Loading dataset: {dataset_name}")
        full_ds = load_dataset(dataset_name, dataset_config_name, split="train")

        if subsample_size:
            print_once(f"[INFO] Subsampling {subsample_size} examples...")
            ds = full_ds.shuffle(seed=seed).select(range(subsample_size))
        else:
            ds = full_ds

        # Save subsampled dataset if it doesn't exist
        if accelerator.is_main_process:
            print_once(f"[INFO] Saving subsampled dataset to {subsample_path}")
            os.makedirs(os.path.dirname(subsample_path), exist_ok=True)
            ds.save_to_disk(subsample_path)

    # Tokenization function
    def tokenize_function(example):
        return tokenizer(
            example[dataset_text_field],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )

    print_once("[INFO] Tokenizing in-memory...")
    tokenized = ds.map(tokenize_function, batched=True, remove_columns=ds.column_names)
    tokenized.set_format(type="torch")
    print_once("[INFO] Tokenization complete")

    # Tokenization stats
    lengths = [len(x["input_ids"]) for x in tokenized]
    total_tokens = sum(lengths)
    print_once(f"[INFO] Tokenization stats:")
    print_once(f"       Total examples: {len(tokenized)}")


def main():
    parser = argparse.ArgumentParser(description="Collect gradients from a model without updating weights")
    parser.add_argument("--path_to", required=True, help="Output directory for collected gradients")
    parser.add_argument("--model_name", default="google/gemma-3b-it", help="Model to collect gradients from")
    parser.add_argument("--dataset_name", default="roneneldan/TinyStories", help="Dataset to use")
    parser.add_argument("--dataset_config_name", default=None, help="Dataset configuration")
    parser.add_argument("--dataset_text_field", default="text", help="Text field name in dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--layer_pattern", type=str, required=True, help="Regex pattern to match layer/module names")
    parser.add_argument("--max_steps", type=int, default=None, help="Max training steps. Default: run through entire dataset.")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length for tokenization. Default: 2048.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bfloat16 precision")
    parser.add_argument("--use_qlora", action="store_true", help="Use QLoRA for quantization")
    parser.add_argument("--subsample_size", type=int, default=None, help="Subsample dataset to this size")
    parser.add_argument("--collect_weights", action="store_true", default=False, help="Also collect weights")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="Enable gradient checkpointing")
    parser.add_argument("--num_save_workers", type=int, default=4, help="Number of parallel save workers")
    parser.add_argument(
        "--compression_level",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Compression level for saved gradients (0=none, 1=bfloat16, 2=int8)",
    )

    args = parser.parse_args()

    collect_grads(
        output_path=args.path_to,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        dataset_text_field=args.dataset_text_field,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        layer_pattern=args.layer_pattern,
        max_steps=args.max_steps,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        use_bf16=args.bf16,
        use_qlora=args.use_qlora,
        subsample_size=args.subsample_size,
        collect_weights=args.collect_weights,
        gradient_checkpointing=args.gradient_checkpointing,
        num_save_workers=args.num_save_workers,
        compression_level=args.compression_level,
    )


if __name__ == "__main__":
    main()
