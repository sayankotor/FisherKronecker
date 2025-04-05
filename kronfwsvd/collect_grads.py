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
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from safetensors.torch import save_file
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
from collections import defaultdict

class NoOpOptimizer(torch.optim.Optimizer):
    def __init__(self):
        super().__init__([{'params': [], 'lr': 0.0}], {})

    def step(self, closure=None):
        pass

    def zero_grad(self, set_to_none=False):
        pass


class GradientTrainer(Trainer):
    def training_step(self, model, inputs, num_items):
        inputs = self._prepare_inputs(inputs)
        outputs = model(**inputs)
        loss = outputs.loss
        self.accelerator.backward(loss)
        return loss.detach()
    
    def create_optimizer(self):
        print("[INFO] Using dummy optimizer (no updates)")
        self.optimizer = NoOpOptimizer()
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
        attn_implementation = "flash_attention_2",
    )

    if use_qlora:
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=32, lora_dropout=0.1, bias="none")
        model = get_peft_model(model, lora_config)
        
    # Freeze all parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Unfreeze matching parameters
    for name, param in model.named_parameters():
        if re.search(layer_pattern, name):
            param.requires_grad = True
            print(f"[INFO] Unfrozen: {name}")
            
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total params: {total:,} | Trainable: {trainable:,} ({100 * trainable / total:.2f}%)")


    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.train()

    ds = load_dataset(dataset_name, dataset_config_name, split="train", num_proc=8)
    if subsample_size:
        ds = ds.shuffle(seed=seed).select(range(subsample_size))

    def tokenize_function(example):
        return tokenizer(
            example[dataset_text_field],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )

    tokenized = ds.map(tokenize_function, batched=True, remove_columns=ds.column_names)
    tokenized.set_format(type="torch")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    weights_dict = {}
    grads_sum = {}
    grads_count = defaultdict(int)

    def save_grad_hook(name):
        def hook(module, grad_input, grad_output):
            grad = grad_output[0].detach().to(dtype=torch.float32, device="cpu")
            if name in grads_sum:
                grads_sum[name] += grad.sum(dim=0)
            else:
                grads_sum[name] = grad.sum(dim=0).clone()
            grads_count[name] += grad.shape[0]

        return hook

    # Match modules by regex pattern
    matched_modules = []
    for name, module in model.named_modules():
        if re.search(layer_pattern, name):
            matched_modules.append((name, module))

    for name, module in matched_modules:
        module.register_full_backward_hook(save_grad_hook(name))

    training_args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=1,
        max_steps=max_steps if max_steps is not None else -1,
        logging_steps=1,
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to=["wandb"],
        dataloader_num_workers=8,
        dataloader_prefetch_factor=4,
    )

    trainer = GradientTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    trainer.train()

    grads_dict = {}
    for name in grads_sum:
        grads_dict[name] = grads_sum[name] / grads_count[name]
    del grads_sum
    save_file(grads_dict, os.path.join(output_path, "grads.safetensors"))
    if collect_weights:
        for name, module in matched_modules:
            for param_name, param in module.named_parameters():
                weights_dict[f"{name}.{param_name}"] = param.detach().cpu()
        save_file(weights_dict, os.path.join(output_path, "weights.safetensors"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to", required=True)
    parser.add_argument("--model_name", default="google/gemma-3b-it")
    parser.add_argument("--dataset_name", default="roneneldan/TinyStories")
    parser.add_argument("--dataset_config_name", default=None)
    parser.add_argument("--dataset_text_field", default="text")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--layer_pattern", type=str, required=True, help="Regex pattern to match layer/module names")
    parser.add_argument(
        "--max_steps", type=int, default=None, help="Max training steps. Default: run through entire dataset."
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=2048, help="Max sequence length for tokenization. Default: 2048."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--subsample_size", type=int, default=None)
    parser.add_argument("--collect_weights", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true")

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
    import accelerate
    from accelerate import launchers
    main()
