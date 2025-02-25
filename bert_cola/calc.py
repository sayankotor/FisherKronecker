import argparse
import logging
import os
import pickle
from collections import defaultdict
from functools import reduce

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from torch import nn, optim
from torch.utils.data import DataLoader, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, BertConfig, BertTokenizer,
                          EvalPrediction, RobertaTokenizer, Trainer,
                          TrainingArguments, set_seed)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_module_by_name(module, access_string):
    """Retrieve a module by its access string."""
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)


class CustomTrainer(Trainer):
    def __init__(self, part_of, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.part_of = part_of

    def make_grad_bank(self):
        """Initialize gradient storage."""
        self.mass = {}
        self.mass_w = {}
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and module.weight.requires_grad:
                if name in self.part_of:
                    self.mass[name] = []
                    self.mass_w[name] = []
        self.avg_counter = 0

    def training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        """Custom training step to store gradients."""
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear) and module.weight.requires_grad:
                    if name in self.part_of:
                        self.mass[name].append(module.weight.grad.detach().cpu())
                        self.mass_w[name].append(module.weight.detach().cpu())

            self.avg_counter += 1

        return loss.detach()


def training_process(path_to, size_of, lr, layers):
    """Training process function."""
    logging.info("Loading dataset and model...")
    dataset = load_dataset("glue", "cola")
    path_name = "etomoscow/bert_cola"
    config = AutoConfig.from_pretrained(path_name, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(path_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def encode_batch(batch):
        """Encodes a batch of input data using the model tokenizer."""
        return tokenizer(
            batch["sentence"], max_length=128, truncation=True, padding="max_length"
        )

    dataset = dataset.map(encode_batch, batched=True)
    dataset = dataset.rename_column("label", "labels")

    metric = evaluate.load("glue", "cola")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        result["acc"] = (preds == p.label_ids).mean()
        return result

    optimizer = optim.SGD(model.parameters(), lr=lr)

    training_args = TrainingArguments(
        learning_rate=lr,
        num_train_epochs=1,
        per_device_train_batch_size=size_of,
        per_device_eval_batch_size=128,
        save_steps=100,
        eval_steps=100,
        optim="sgd",
        output_dir=path_to,
        overwrite_output_dir=True,
        save_strategy="steps",
        remove_unused_columns=True,
        seed=42,
        report_to="none",
    )

    part_of = [f"bert.encoder.layer.{layer}.output.dense" for layer in args.layers]
    part_of += [f"bert.encoder.layer.{layer}.intermediate.dense" for layer in layers]

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        part_of=part_of

    )
    trainer.make_grad_bank()

    logging.info("Starting training...")
    trainer.train()
    logging.info("Training completed. Evaluating model...")
    eval_results = trainer.evaluate()
    logging.info(f"Evaluation results: {eval_results}")

    trainer.save_model()

    logging.info(f"Length of gradient mass: {len(trainer.mass)}")
    with open(
        os.path.join(trainer.args.output_dir, "tensor_grad57_out.pickle"), "wb"
    ) as f:
        pickle.dump(trainer.mass, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training process configuration")
    parser.add_argument("--path_to", type=str, required=True, help="Path to the output")
    parser.add_argument(
        "--size_of",
        type=int,
        required=True,
        help="Size parameter for the training process",
    )
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--layers", type=int, nargs='+', required=True, help="List of layers to include in part_of")

    args = parser.parse_args()

    logging.info(f"Training process started with:")
    logging.info(f"Output Path: {args.path_to}")
    logging.info(f"Batch Size: {args.size_of}")
    logging.info(f"Learning Rate: {args.lr}")

    training_process(args.path_to, args.size_of, args.lr, args.layers)
