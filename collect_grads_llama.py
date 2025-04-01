import argparse
import logging
import os
import pickle
import random
from functools import reduce

import numpy as np
import torch
from accelerate import Accelerator
from datasets import Dataset, load_dataset
from torch import nn, optim
from tqdm import tqdm
from collections import defaultdict
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Trainer, TrainingArguments,
                          set_seed)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Keep this function as is
def get_module_by_name(module, access_string):
    """Retrieve a module by its access string."""
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)

class CustomTrainer(Trainer):
    def __init__(self, part_of, gradient_storage_path, collect_weights=False, save_every_n_steps=5, *args, **kwargs):

        self._save_every_n_steps = save_every_n_steps
        self._intermediate_file_counter = 0

        super().__init__(*args, **kwargs)

        self.part_of = part_of
        self._gradient_storage_path = gradient_storage_path
        self._collect_weights = collect_weights
        self.grad_mass = defaultdict(list)
        self.weight_mass = defaultdict(list)
        self.step_counter = 0
        os.makedirs(os.path.dirname(self._gradient_storage_path), exist_ok=True)
        logging.info(f"Will store gradients for layers: {list(self.part_of)}")
        if self._collect_weights:
             logging.info(f"Will also store weights for the same layers.")

        if self.is_local_process_zero:
            os.makedirs(os.path.dirname(self._gradient_storage_path), exist_ok=True)
        
        logging.info(f"Will store gradients for layers: {list(self.part_of)}")
        if self._collect_weights:
             logging.info(f"Will also store weights for the same layers.")
        if self._save_every_n_steps > 0:
            logging.info(f"Will save intermediate gradients every {self._save_every_n_steps} steps.")


    # No need for make_grad_bank, handled in init/training_step

    def _save_intermediate_data(self):
        """Saves current grad_mass and clears it."""
        if not self.is_local_process_zero or not self.grad_mass:
            return

        intermediate_path = f"{self._gradient_storage_path}.part{self._intermediate_file_counter}"
        logging.info(f"Saving intermediate data to {intermediate_path}...")
        data_to_save = {"gradients": dict(self.grad_mass)}
        if self._collect_weights:
            data_to_save["weights"] = dict(self.weight_mass)

        try:
            with open(intermediate_path, "wb") as f:
                pickle.dump(data_to_save, f)
            logging.info(f"Successfully saved intermediate data.")
            self.grad_mass.clear()
            self.weight_mass.clear()
            self._intermediate_file_counter += 1
            import gc
            gc.collect() 
        except Exception as e:
            logging.error(f"Failed to save intermediate data to {intermediate_path}: {e}")


    def training_step(self, model, inputs, *args, **kwargs) -> torch.Tensor:
        """Custom training step to compute loss and store gradients (and optionally weights)."""
        model.train() 

        with self.compute_loss_context_manager():
            outputs = model(**inputs)
            loss = outputs.loss

        if loss is None:
             logging.warning("Loss is None, skipping backward pass and gradient collection for this step.")
             return torch.tensor(0.0, device=model.device)

        self.accelerator.backward(loss)

        if self.is_local_process_zero:
            self.step_counter += 1
            if self.args.logging_steps > 0 and self.step_counter % self.args.logging_steps == 0:
                 logging.info(f"Step {self.state.global_step}: Collecting gradients...")

            with torch.no_grad(): # Ensure no extra graph is built
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear) and module.weight.requires_grad:
                        if name in self.part_of:
                            if module.weight.grad is not None:
                                self.grad_mass[name].append(module.weight.grad.detach().cpu().clone())
                                if self._collect_weights:
                                    self.weight_mass[name].append(module.weight.detach().cpu().clone())
                            else:
                                logging.warning(f"Gradient for layer {name} is None at step {self.state.global_step}. Skipping collection for this layer.")
            current_step_in_epoch = self.state.global_step + 1 # Use global_step
            if self._save_every_n_steps > 0 and current_step_in_epoch % self._save_every_n_steps == 0:
                 self._save_intermediate_data()

        return loss.detach() / self.args.gradient_accumulation_steps

    def save_collected_gradients(self):
        """Saves the collected gradients and optionally weights to a pickle file."""
        if self.is_local_process_zero:
            if self._intermediate_file_counter > 0:
                # If intermediate files were saved, save the last batch if any
                if self.grad_mass:
                     self._save_intermediate_data() 
                logging.info("Intermediate files saved. Final processing might be needed.")
                # You might need a separate script to load and combine the .partX files
            else:
                 # Original saving logic if no intermediate files were created
                logging.info(f"Saving collected data for {len(self.grad_mass)} layers...")
                data_to_save = {"gradients": dict(self.grad_mass)}
                if self._collect_weights:
                    data_to_save["weights"] = dict(self.weight_mass)

                try:
                    with open(self._gradient_storage_path, "wb") as f:
                        pickle.dump(data_to_save, f)
                    logging.info(f"Successfully saved collected data to {self._gradient_storage_path}")
                    num_grads_per_layer = {name: len(grads) for name, grads in self.grad_mass.items()}
                    logging.info(f"Number of gradients collected per layer: {num_grads_per_layer}")
                except Exception as e:
                    logging.error(f"Failed to save gradients/weights to {self._gradient_storage_path}: {e}")


def collect_grads(
    output_path: str,
    model_name: str,
    dataset_name: str,
    dataset_config_name: str = None,
    dataset_text_field: str = "text",
    batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 1e-5,
    layers_indices: list[int] = None,
    max_steps: int = 100,
    max_seq_length: int = 512,
    seed: int = 42,
    use_bf16: bool = True,
    use_qlora: bool = False,
    subsample_size: int = None,
    collect_weights: bool = False,
    gradient_checkpointing: bool = False,
):
    """Main process to load model, dataset, and collect gradients."""
    set_seed(seed)
    accelerator = Accelerator() # Used for device placement during mapping

    logging.info(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        logging.warning("Tokenizer does not have a pad token. Setting pad_token = eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        # model.config.pad_token_id = tokenizer.pad_token_id


    logging.info(f"Loading dataset {dataset_name} (config: {dataset_config_name})...")
    split_name = 'train' # Common split name
    try:
        dataset = load_dataset(dataset_name, dataset_config_name, split=split_name)
        logging.info(f"Original dataset size: {len(dataset)}")
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        # Try loading without config name if it failed
        try:
            logging.warning(f"Trying to load {dataset_name} without config name...")
            dataset = load_dataset(dataset_name, split=split_name)
            logging.info(f"Original dataset size: {len(dataset)}")
        except Exception as e2:
             logging.error(f"Failed to load dataset {dataset_name} even without config: {e2}")
             return

    if subsample_size is not None and subsample_size < len(dataset):
        logging.info(f"Subsampling dataset to {subsample_size} examples...")
        indices = random.sample(range(len(dataset)), subsample_size)
        dataset = dataset.select(indices)
        logging.info(f"Subsampled dataset size: {len(dataset)}")


    def encode_batch(batch):
        """Encodes a batch of text data for Causal LM."""
        tokenized_inputs = tokenizer(
            batch[dataset_text_field],
            max_length=max_seq_length,
            truncation=True,
            padding="max_length", # Pad to max_length
            return_tensors="pt", # Return PyTorch tensors
        )

        labels = tokenized_inputs["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    logging.info(f"Tokenizing dataset (using '{dataset_text_field}' field)...")
    # Use accelerator.device for mapping if possible, helps with large datasets
    # Note: This might still OOM if the dataset is huge and processing is heavy.
    # Consider using dataset.map with num_proc > 1 if tokenization is slow.
    remove_columns = dataset.column_names # Remove original columns
    # Run map on the main process first to check for issues
    try:
        # Process a small sample first to catch errors quickly
        logging.info("Testing tokenization on a small sample...")
        test_sample = dataset.select(range(min(10, len(dataset))))
        processed_sample = test_sample.map(
             encode_batch,
             batched=True,
        )
        logging.info("Tokenization test successful.")
        del test_sample, processed_sample

        logging.info("Processing full dataset...")
        processed_dataset = dataset.map(
            encode_batch,
            batched=True,
            remove_columns=remove_columns,
            num_proc=4,
        )
        logging.info("Dataset tokenization complete.")
        processed_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    except Exception as e:
        logging.error(f"Error during dataset tokenization: {e}")
        logging.error("Check dataset_text_field, max_seq_length, and memory usage.")
        return

    logging.info(f"Loading model {model_name}...")
    model_kwargs = {"trust_remote_code": True}

    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token_id is not None:
             config.pad_token_id = tokenizer.pad_token_id

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            **model_kwargs
        )

        if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
             logging.warning("Model config pad_token_id is None, setting from tokenizer.")
             model.config.pad_token_id = tokenizer.pad_token_id

    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        logging.error("Check model name, authentication (huggingface-cli login), memory, and dependencies (bitsandbytes, accelerate).")
        return

    target_layers = []
    if layers_indices:
        logging.info(f"Identifying linear layers in specified Llama layers: {layers_indices}")
        layer_template_prefixes = [
            "model.layers.{}.mlp.gate_proj",
            "model.layers.{}.mlp.up_proj",
            "model.layers.{}.mlp.down_proj",
            # "model.layers.{}.self_attn.q_proj",
            # "model.layers.{}.self_attn.k_proj",
            # "model.layers.{}.self_attn.v_proj",
            # "model.layers.{}.self_attn.o_proj",
        ]
        for idx in layers_indices:
            for template in layer_template_prefixes:
                target_layers.append(template.format(idx))

        available_layers = {name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
        valid_target_layers = [layer for layer in target_layers if layer in available_layers]
        invalid_target_layers = [layer for layer in target_layers if layer not in available_layers]

        if invalid_target_layers:
             logging.warning(f"Specified layers not found or not Linear: {invalid_target_layers}")
             target_layers = valid_target_layers
        if not target_layers:
             logging.error("No valid linear layers found based on specified indices. Exiting.")
             print("\nAvailable linear layer names (first few):")
             count = 0
             for name in available_layers:
                 print(f"- {name}")
                 count += 1
                 if count >= 20: break
             return

    else:
        logging.warning("No specific layers provided. Collecting gradients for ALL linear layers.")
        target_layers = {name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
        logging.info(f"Found {len(target_layers)} linear layers in the model.")

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    logging.info(f"SGD optimizer created with LR: {learning_rate}")

    training_args = TrainingArguments(
        output_dir=os.path.join(output_path, "trainer_temp"),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_steps=max_steps,
        logging_steps=max(1, max_steps // 10),
        seed=seed,
        report_to="none",
        save_strategy="no",
        bf16=use_bf16,
        remove_unused_columns=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    if gradient_checkpointing:
        logging.info("Enabling gradient checkpointing.")
        if hasattr(model, "gradient_checkpointing_enable"):
             model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        else:
             logging.warning("Model does not have gradient_checkpointing_enable method. Set in TrainingArguments instead.")
             training_args.gradient_checkpointing = True
             training_args.gradient_checkpointing_kwargs={'use_reentrant': False}

    gradient_storage_filename = f"collected_gradients_model-{model_name.split('/')[-1]}_dataset-{dataset_name}_steps-{max_steps}.pkl"
    gradient_storage_full_path = os.path.join(output_path, gradient_storage_filename)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        tokenizer=tokenizer,
        part_of=set(target_layers),
        optimizers=(optimizer, None),
        gradient_storage_path=gradient_storage_full_path,
        collect_weights=collect_weights,
        save_every_n_steps=20,
    )
    logging.info(f"Starting gradient collection for {max_steps} steps with SGD optimizer...")
    try:
        trainer.train()
        logging.info("Gradient collection loop finished.")
    except Exception as e:
         logging.error(f"Error during trainer.train(): {e}")
         trainer.save_collected_gradients()
         raise e

    trainer.save_collected_gradients()
    logging.info("Process completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama 3 Gradient Collection")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save the collected gradients pickle file.")
    parser.add_argument("--model_name", type=str, default="unsloth/Llama-3.1-8B-Instruct", help="Hugging Face model identifier for Llama 3.")
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Hugging Face dataset name (e.g., wikitext, allenai/c4).")
    parser.add_argument("--dataset_config_name", type=str, default="wikitext-103-v1", help="Dataset configuration name (e.g., wikitext-103-v1, en for C4).")
    parser.add_argument("--dataset_text_field", type=str, default="text", help="The name of the field in the dataset containing the text.")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size (micro-batch size). Keep small for large models.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps. Effective batch size = batch_size * num_gpus * grad_accum_steps.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (mostly placeholder, as we don't optimize for long).")
    parser.add_argument("--layers_indices", type=int, nargs='+', default=None, help="List of Llama layer *indices* (0-based) to collect gradients from (e.g., 0 5 10 31). If None, collects from *all* linear layers.")
    parser.add_argument("--max_steps", type=int, default=100, help="Number of gradient collection steps (backward passes) to perform.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for tokenization.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no_bf16", action="store_true", help="Disable bfloat16 usage.")
    parser.add_argument("--use_qlora", action="store_true", help="Use QLoRA (4-bit quantization) for loading the model.")
    parser.add_argument("--subsample_size", type=int, default=None, help="Subsample the dataset to this many examples before processing.")
    parser.add_argument("--collect_weights", action="store_true", help="Also collect the weights of the specified layers along with gradients.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory.")


    args = parser.parse_args()

    # Rename for clarity within the function call
    args.path_to = args.output_path
    args.size_of = args.batch_size
    args.layers = args.layers_indices

    logging.info(f"Gradient Collection Process Started with:")
    logging.info(f"  Output Path: {args.path_to}")
    logging.info(f"  Model Name: {args.model_name}")
    logging.info(f"  Dataset: {args.dataset_name} ({args.dataset_config_name})")
    logging.info(f"  Dataset Text Field: {args.dataset_text_field}")
    logging.info(f"  Batch Size (per device): {args.size_of}")
    logging.info(f"  Grad Accumulation Steps: {args.gradient_accumulation_steps}")
    logging.info(f"  Max Steps: {args.max_steps}")
    logging.info(f"  Learning Rate: {args.lr}")
    logging.info(f"  Layer Indices: {'All Linear Layers' if args.layers is None else args.layers}")
    logging.info(f"  Max Seq Length: {args.max_seq_length}")
    logging.info(f"  Use bfloat16: {not args.no_bf16 and not args.use_qlora}")
    logging.info(f"  Use QLoRA (4-bit): {args.use_qlora}")
    logging.info(f"  Subsample Dataset Size: {args.subsample_size}")
    logging.info(f"  Collect Weights: {args.collect_weights}")
    logging.info(f"  Gradient Checkpointing: {args.gradient_checkpointing}")


    collect_grads(
        output_path=args.path_to,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        dataset_text_field=args.dataset_text_field,
        batch_size=args.size_of,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        layers_indices=args.layers,
        max_steps=args.max_steps,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        use_bf16=not args.no_bf16,
        use_qlora=args.use_qlora,
        subsample_size=args.subsample_size,
        collect_weights=args.collect_weights,
        gradient_checkpointing=args.gradient_checkpointing,
    )