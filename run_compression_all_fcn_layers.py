import json
import pickle
import argparse

import evaluate as ev
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm.auto import trange
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, EvalPrediction, Trainer,
                          TrainingArguments)

from bert_compression.compression_utils import \
    CompressedBertForSequenceClassification

dataset_cola = load_dataset("glue", "cola")
label_list = dataset_cola["train"].features["label"].names
num_labels = len(label_list)

model_name = "bert-base-uncased"
config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

sentence1_key, sentence2_key = "sentence", None


def preprocess_function(examples):
    args = (
        (examples[sentence1_key],)
        if sentence2_key is None
        else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer.batch_encode_plus(
        *args, max_length=128, truncation=True, padding="max_length"
    )
    result["label"] = examples["label"]
    return result


tokenized_dataset = dataset_cola.map(
    preprocess_function, batched=True, load_from_cache_file=False
)

metric = ev.load("glue", "cola")


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    result = metric.compute(predictions=preds, references=p.label_ids)
    result["combined_score"] = np.mean(list(result.values())).item()
    return result


training_args = TrainingArguments(
    do_train=False,
    do_eval=True,
    learning_rate=5e-5,
    num_train_epochs=1,
    evaluation_strategy="steps",
    skip_memory_metrics=False,
    eval_steps=100,
    per_device_eval_batch_size=128,
    save_steps=1000,
    overwrite_output_dir=True,
    output_dir="bert_cola/fishers/",
    remove_unused_columns=True,
    seed=297104,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
)

trainer.evaluate()

# Function to check if a matrix is positive definite
def is_pos_def(x):
    try:
        np.linalg.cholesky(x)
        return True
    except np.linalg.LinAlgError:
        return False

# Function to make a matrix positive definite by adding to diagonal if needed
def ensure_pos_def(matrix):
    alpha = 0.0
    matrix_new = (1 - alpha) * matrix + alpha * np.eye(len(np.diag(matrix)))
    while not is_pos_def(matrix_new):
        alpha += 0.05
        matrix_new = (1 - alpha) * matrix + alpha * np.eye(len(np.diag(matrix)))
    return matrix_new

# Function to apply Kronecker-factorized SVD compression to a layer
def apply_kron_fwsvd_to_layer(model, layer_index, layer_type, rank):
    if layer_type == "output":
        B1 = np.load(f"fisher_factors/kron_factors_out/B1sgd_{layer_index}_output.npy")
        C1 = np.load(f"fisher_factors/kron_factors_out/C1sgd_{layer_index}_output.npy")
        fc_w = model.bert.encoder.layer[layer_index].output.dense.weight.data.cpu().numpy()
        fc_b = model.bert.encoder.layer[layer_index].output.dense.bias.data.cpu().numpy()
    else:
        B1 = np.load(f"fisher_factors/kron_factors/B1sgd_{layer_index}_intermediate.npy")
        C1 = np.load(f"fisher_factors/kron_factors/C1sgd_{layer_index}_intermediate.npy")
        fc_w = model.bert.encoder.layer[layer_index].intermediate.dense.weight.data.cpu().numpy()
        fc_b = model.bert.encoder.layer[layer_index].intermediate.dense.bias.data.cpu().numpy()

    # Make matrices positive definite
    B_new = ensure_pos_def(B1)
    C_new = ensure_pos_def(C1)

    B1_square = np.linalg.cholesky(B_new)
    C1_square = np.linalg.cholesky(C_new)

    # Compute SVD
    U, S, Vt = np.linalg.svd(C1_square.T @ fc_w @ B1_square, full_matrices=False)

    U1 = np.linalg.inv(C1_square.T) @ U
    V1 = np.linalg.inv(B1_square.T) @ Vt.T

    w1 = np.diag(np.sqrt(S[:rank])) @ V1.T[:rank, :]
    w2 = U1[:, 0:rank] @ np.diag(np.sqrt(S[:rank]))

    out_features, in_features = fc_w.shape
    is_bias = fc_b is not None

    linear1 = nn.Linear(in_features=in_features, out_features=rank, bias=False)
    linear1.weight = nn.Parameter(torch.FloatTensor(w1))

    linear2 = nn.Linear(in_features=rank, out_features=out_features, bias=is_bias)
    linear2.weight = nn.Parameter(torch.FloatTensor(w2))
    linear2.bias = nn.Parameter(torch.FloatTensor(fc_b))
    factorized_layer = nn.Sequential(linear1, linear2)

    if layer_type == "output":
        model.bert.encoder.layer[layer_index].output.dense = factorized_layer.to(model.device)
    else:
        model.bert.encoder.layer[layer_index].intermediate.dense = factorized_layer.to(model.device)
    
    return model

# Parse command line arguments
parser = argparse.ArgumentParser(description='BERT Layer Compression')
parser.add_argument('--rank', type=int, default=50, 
                    help='Target rank for compression (default: 50)')
parser.add_argument('--compression_types', nargs='+', 
                    default=["SVD", "FWSVD", "Kron-FWSVD"],
                    help='Compression types to apply (default: SVD FWSVD Kron-FWSVD)')
parser.add_argument('--num_layers', type=int, default=12,
                    help='Number of transformer layers to compress (default: 12 for BERT-base)')
parser.add_argument('--eval_individual', action='store_true',
                    help='Also evaluate individual layer compression (default: False)')
parser.add_argument('--output_prefix', type=str, default="results",
                    help='Prefix for output files (default: "results")')

args = parser.parse_args()

# Configuration
compression_types = args.compression_types
layer_types = ["intermediate", "output"]
num_layers = args.num_layers
rank = args.rank  # Target rank for compression

# Initialize results dictionary
results = {}

# Loop through different compression types
for compression_type in compression_types:
    results[compression_type] = {}
    
    if compression_type == "SVD":
        # Apply SVD compression to all layers
        custom_model = CompressedBertForSequenceClassification.from_pretrained(
            "bert_cola/fishers/",
            shape=[[32, 32], [64, 64]],
            rank=rank,
            compression_type="svd",
            layer_mask=r".*/encoder/layer/\d+/(intermediate|output)",
        )
        custom_model.to_compression(compress=True, weight=None)
        trainer.model = custom_model.cuda().eval()
        results[compression_type]["matthews_correlation"] = trainer.evaluate()["eval_matthews_correlation"]
        
    elif compression_type == "FWSVD":
        # Apply Fisher-Weighted SVD compression to all layers
        with open("bert_cola/fishers/fisher_67.pkl", "rb") as f:
            weights = pickle.load(f)
        
        custom_model = CompressedBertForSequenceClassification.from_pretrained(
            "bert_cola/fishers/",
            shape=[[32, 32], [64, 64]],
            rank=rank,
            compression_type="svd",
            layer_mask=r".*/encoder/layer/\d+/(intermediate|output)",
        )
        custom_model.to_compression(compress=True, weight=weights)
        trainer.model = custom_model.cuda().eval()
        results[compression_type]["matthews_correlation"] = trainer.evaluate()["eval_matthews_correlation"]
        
    elif compression_type == "Kron-FWSVD":
        # Apply Kronecker-factorized SVD compression to all layers
        model_name = "bert_cola/fishers/"
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, config=config
        ).cuda().eval()
        
        # Apply compression to all layers
        for layer_index in range(1, num_layers):
            for layer_type in layer_types:
                model = apply_kron_fwsvd_to_layer(model, layer_index, layer_type, rank)
        
        trainer.model = model
        results[compression_type]["matthews_correlation"] = trainer.evaluate()["eval_matthews_correlation"]

# Save the results to a JSON file
output_file = f"{args.output_prefix}_all_layers_rank_{rank}.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)
print(f"All layers compression results saved to {output_file}")
