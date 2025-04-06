import argparse
import os
import re
import torch
import random
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from trl import SFTTrainer
from safetensors.torch import save_file
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
from collections import defaultdict
from datasets import load_dataset, load_from_disk
import numpy as np
import os
from accelerate.accelerator import AcceleratorState


def print_once(*args, **kwargs):
    if AcceleratorState.is_main_process:
        print(*args, **kwargs)


class NoOpOptimizer(torch.optim.Optimizer):
    def __init__(self, param):
        dummy_param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        super().__init__([{"params": [dummy_param], "lr": 0.0}], {})

    def step(self, closure=None):
        pass

    def zero_grad(self, set_to_none=False):
        pass


class GradientTrainer(SFTTrainer):
    def training_step(self, model, inputs, num_items):
        inputs = self._prepare_inputs(inputs)
        outputs = model(**inputs)
        loss = outputs.loss
        self.accelerator.backward(loss)
        return loss.detach()

    def create_optimizer(self):
        print_once("[INFO] Using dummy optimizer (no updates)")
        self.optimizer = NoOpOptimizer(next(self.model.parameters()))
        return self.optimizer


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
):
    os.makedirs(output_path, exist_ok=True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quant_config = None
    if use_qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2" if "gemma" not in model_name else "eager",
        # use_cache=not gradient_checkpointing,  # This is critical for gradient checkpointing
    )

    # Configure gradient checkpointing
    if gradient_checkpointing:
        if hasattr(model.config, "gradient_checkpointing"):
            model.config.gradient_checkpointing = True
            model.config.use_cache = False

        # Enable gradient checkpointing with non-reentrant mode for better compatibility with FSDP
        if hasattr(model, "gradient_checkpointing_enable"):
            print_once("[INFO] Enabling gradient checkpointing with non-reentrant mode")
            model.gradient_checkpointing_enable()
    else:
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()

    if use_qlora:
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=32, lora_dropout=0.1, bias="none")
        model = get_peft_model(model, lora_config)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze matching parameters
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if re.search(layer_pattern, name):
            param.requires_grad = True
            unfrozen_count += 1
            print_once(f"[INFO] Unfrozen: {name}")

    if unfrozen_count == 0:
        print_once(f"[WARNING] No parameters matched the pattern '{layer_pattern}'. Check your regex.")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_once(f"[INFO] Total params: {total:,} | Trainable: {trainable:,} ({100 * trainable / total:.2f}%)")

    if trainable == 0:
        print_once("[ERROR] No trainable parameters found. Exiting.")
        return

    model.train()

    subsample_path = os.path.join(dataset_name, "sample", str(subsample_size))
    # Load or create cached text dataset
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
        print_once(f"[INFO] Saving subsampled dataset to {subsample_size}")
        ds.save_to_disk(subsample_path)

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
    print_once(tokenized.features)
    # Stats
    lengths = [len(x["input_ids"]) for x in tokenized]
    total_tokens = sum(lengths)
    print_once(f"[INFO] Tokenization stats:")
    print_once(f"       Total examples: {len(tokenized)}")
    print_once(f"       Total tokens: {total_tokens:,}")
    print_once(f"       Avg length: {np.mean(lengths):.2f}")
    print_once(f"       Median length: {np.median(lengths):.2f}")
    print_once("[INFO] Tokenizing dataset")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    weights_dict = {}
    grads_sum = {}
    grads_count = defaultdict(int)

    def save_grad_hook(name):
        def hook(module, grad_input, grad_output):
            grad = grad_output[0].detach()
            if name in grads_sum:
                grads_sum[name] += grad.sum(dim=0).to(device=grads_sum[name].device)
            else:
                grads_sum[name] = grad.sum(dim=0).clone()
            grads_count[name] += grad.shape[0]
            del grad, grad_input, grad_output

        return hook

    # Match modules by regex pattern
    matched_modules = []
    for name, module in model.named_modules():
        if re.search(layer_pattern, name):
            matched_modules.append((name, module))

    print_once(f"[INFO] Registered {len(matched_modules)} modules for gradient collection")
    for name, module in matched_modules:
        module.register_full_backward_hook(save_grad_hook(name))

    # Set up training arguments with appropriate gradient checkpointing settings
    training_args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=1,
        max_steps=max_steps if max_steps is not None else -1,
        logging_steps=10,
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to=["wandb"],
        dataloader_num_workers=8,
        dataloader_prefetch_factor=4,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        save_strategy="no",
        save_steps=0,
        save_total_limit=0,
        gradient_checkpointing=gradient_checkpointing,
        max_grad_norm=0.0,
    )

    # Configure gradient checkpointing parameters when enabled
    if gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    trainer = GradientTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print_once("[INFO] Starting training loop")
    trainer.train()

    print_once("[INFO] Processing gradients")
    grads_dict = {}
    for name in grads_sum:
        grads_dict[name] = grads_sum[name] / grads_count[name]
        grads_sum[name] = grads_sum[name].to("cpu")
    del grads_sum

    grad_file = os.path.join(output_path, "grads.safetensors")
    print_once(f"[INFO] Saving gradients to {grad_file}")
    save_file(grads_dict, grad_file)

    if collect_weights:
        print_once("[INFO] Collecting weights")
        for name, module in matched_modules:
            for param_name, param in module.named_parameters():
                weights_dict[f"{name}.{param_name}"] = param.detach().cpu()

        weight_file = os.path.join(output_path, "weights.safetensors")
        print_once(f"[INFO] Saving weights to {weight_file}")
        save_file(weights_dict, weight_file)

    print_once("[INFO] Collection complete")


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
    parser.add_argument(
        "--max_steps", type=int, default=None, help="Max training steps. Default: run through entire dataset."
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=2048, help="Max sequence length for tokenization. Default: 2048."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use bfloat16 precision")
    parser.add_argument("--use_qlora", action="store_true", help="Use QLoRA for quantization")
    parser.add_argument("--subsample_size", type=int, default=None, help="Subsample dataset to this size")
    parser.add_argument("--collect_weights", action="store_true", default=False, help="Also collect weights")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False, help="Enable gradient checkpointing")

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
    )


if __name__ == "__main__":
    main()
