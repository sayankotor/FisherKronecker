"""
Gradient Extraction for BERT Fine-tuning on GLUE Tasks

This module implements a custom trainer for extracting gradients during BERT fine-tuning
on GLUE benchmark tasks. The extracted gradients are saved in bfloat16 format for
memory efficiency and can be used for gradient analysis research.

Author: [Your Name]
Date: May 2025
"""

import argparse
import os
from functools import reduce
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    set_seed,
)
from datasets import load_dataset, Dataset
import evaluate
from safetensors.torch import save_file


def get_module_by_name(module: torch.nn.Module, access_string: str) -> torch.nn.Module:
    """
    Access nested module by string name using dot notation.

    Args:
        module: Root module to traverse
        access_string: Dot-separated string path to target module

    Returns:
        The target module

    Example:
        >>> model = BertModel.from_pretrained('bert-base-uncased')
        >>> layer = get_module_by_name(model, 'encoder.layer.0.attention')
    """
    return reduce(getattr, access_string.split("."), module)


# Target layers for gradient extraction - focusing on dense layers in intermediate positions
TARGET_LAYERS = [f"bert.encoder.layer.{i}.{sub}.dense" for i in range(1, 12) for sub in ["intermediate", "output"]]


class GradientExtractionTrainer(Trainer):
    """
    Custom Trainer that extracts and stores gradients from specified layers during training.

    This trainer extends the standard HuggingFace Trainer to capture gradients from
    target layers after each backward pass. Gradients are stored in bfloat16 format
    to reduce memory usage.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_bank: Dict[str, List[torch.Tensor]] = {}
        self.step_counter: int = 0

    def initialize_gradient_bank(self) -> None:
        """Initialize storage for gradients from target layers."""
        self.grad_bank = {name: [] for name in TARGET_LAYERS}
        self.step_counter = 0
        print(f"Initialized gradient bank for {len(TARGET_LAYERS)} target layers")

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform a single training step with gradient extraction.

        Args:
            model: The model being trained
            inputs: Batch of training inputs

        Returns:
            Computed loss for the step
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Forward pass and loss computation
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs).mean()

        # Backward pass
        self.accelerator.backward(loss)

        # Extract gradients from target layers
        for name, module in model.named_modules():
            if name in TARGET_LAYERS and hasattr(module, "weight") and module.weight.requires_grad:
                if module.weight.grad is not None:
                    # Convert to bfloat16 for memory efficiency
                    grad = module.weight.grad.detach().cpu().to(dtype=torch.bfloat16)
                    self.grad_bank[name].append(grad)

        self.step_counter += 1

        return loss.detach() / self.args.gradient_accumulation_steps


def get_glue_dataset(task_name: str, tokenizer: AutoTokenizer, max_length: int = 128) -> Dataset:
    """
    Load and preprocess GLUE dataset for the specified task.

    Args:
        task_name: Name of the GLUE task (e.g., 'sst2', 'mnli', 'cola')
        tokenizer: Tokenizer for text preprocessing
        max_length: Maximum sequence length for tokenization

    Returns:
        Tokenized dataset ready for training

    Raises:
        KeyError: If task_name is not a valid GLUE task
    """
    # Mapping of GLUE tasks to their input column names
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }

    if task_name not in task_to_keys:
        raise KeyError(f"Unknown task: {task_name}. Available tasks: {list(task_to_keys.keys())}")

    sentence1_key, sentence2_key = task_to_keys[task_name]
    raw_dataset = load_dataset("glue", task_name)

    def preprocess_function(examples: Dict[str, List]) -> Dict[str, List]:
        """Tokenize text inputs and prepare labels."""
        # Handle single vs. paired sentence tasks
        if sentence2_key is None:
            args = (examples[sentence1_key],)
        else:
            args = (examples[sentence1_key], examples[sentence2_key])

        result = tokenizer(*args, max_length=max_length, truncation=True, padding="max_length")
        result["label"] = examples["label"]
        return result

    # Apply preprocessing to all splits
    tokenized_dataset = raw_dataset.map(
        preprocess_function, batched=True, load_from_cache_file=False, desc=f"Tokenizing {task_name} dataset"
    )

    # Rename label column to match trainer expectations
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    print(f"Loaded {task_name} dataset:")
    for split, data in tokenized_dataset.items():
        print(f"  {split}: {len(data)} examples")

    return tokenized_dataset


def run_training_with_gradient_extraction(task_name: str, output_dir: str, model_name: str = "bert-base-uncased") -> None:
    """
    Main training procedure with gradient extraction.

    This function orchestrates the entire training process:
    1. Loads and preprocesses the dataset
    2. Initializes the model and tokenizer
    3. Sets up the custom trainer with gradient extraction
    4. Runs training and saves both model weights and extracted gradients

    Args:
        task_name: GLUE task name for training
        output_dir: Directory to save model and gradient outputs
        model_name: Pre-trained model name or path
    """
    # Set random seed for reproducibility
    set_seed(42)
    print(f"Set random seed to 42 for reproducible results")

    # Initialize model components
    print(f"Loading model: {model_name}")
    num_labels = 3 if task_name == "mnli" else 2  # MNLI has 3 classes, others have 2
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load and preprocess dataset
    print(f"Loading dataset for task: {task_name}")
    dataset = get_glue_dataset(task_name, tokenizer)

    # Initialize evaluation metric
    metric = evaluate.load("glue", task_name)

    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        """Compute evaluation metrics for the task."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        # Get task-specific metrics
        result = metric.compute(predictions=predictions, references=labels)

        # Add accuracy for all tasks
        result["accuracy"] = (predictions == labels).mean()

        return result

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-6,  # Conservative learning rate for stable training
        num_train_epochs=1,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=1000,
        save_strategy="steps",
        remove_unused_columns=True,
        overwrite_output_dir=True,
        seed=42,
        report_to="none",  # Disable wandb/tensorboard logging
        dataloader_pin_memory=False,  # Reduce memory usage
    )

    # Initialize custom trainer with gradient extraction
    trainer = GradientExtractionTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation_matched"] if task_name == "mnli" else dataset["validation"],
        compute_metrics=compute_metrics,
    )

    # Set up gradient extraction
    trainer.initialize_gradient_bank()

    print("Starting training with gradient extraction...")
    trainer.train()

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    # Save model state dict
    model_path = os.path.join(output_dir, f"{task_name}_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to: {model_path}")

    # Process and save gradients
    print("Processing extracted gradients...")
    grad_data = {}
    for layer_name, grad_list in trainer.grad_bank.items():
        if len(grad_list) > 0:
            # Stack gradients from all training steps
            stacked_grads = torch.stack(grad_list)
            # Ensure bfloat16 format for memory efficiency
            grad_data[layer_name] = stacked_grads.to(dtype=torch.bfloat16)
            print(f"  {layer_name}: {stacked_grads.shape} gradients")

    # Save gradients using safetensors format
    grad_path = os.path.join(output_dir, f"{task_name}_gradients.safetensors")
    save_file(grad_data, grad_path)
    print(f"Saved gradients to: {grad_path}")

    # Final evaluation
    print("\nFinal evaluation results:")
    eval_results = trainer.evaluate()
    for metric_name, value in eval_results.items():
        print(f"  {metric_name}: {value:.4f}")


def main():
    """Command-line interface for the gradient extraction training script."""
    parser = argparse.ArgumentParser(
        description="Train BERT on GLUE tasks with gradient extraction", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"],
        help="GLUE task name for training",
    )

    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model weights and extracted gradients")

    parser.add_argument(
        "--model_name", type=str, default="bert-base-uncased", help="Pre-trained model name or path from HuggingFace Hub"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("BERT Fine-tuning with Gradient Extraction")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Model: {args.model_name}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    try:
        run_training_with_gradient_extraction(task_name=args.task, output_dir=args.output_dir, model_name=args.model_name)
        print("\n✅ Training completed successfully!")

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
