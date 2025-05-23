import json
import torch
import torch.nn as nn
import numpy as np
import argparse
from datasets import load_dataset
import evaluate as ev
from transformers import TrainingArguments, Trainer, EvalPrediction, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from safetensors.torch import load_file

# Set random seed for reproducibility
torch.manual_seed(0)


def is_pos_def(matrix):
    return np.all(np.linalg.eigvals(matrix) > 0)


def compute_metrics(eval_prediction: EvalPrediction):
    metric = ev.load("glue", "mnli")
    predictions = eval_prediction.predictions[0] if isinstance(eval_prediction.predictions, tuple) else eval_prediction.predictions
    predicted_classes = np.argmax(predictions, axis=1)
    result = metric.compute(predictions=predicted_classes, references=eval_prediction.label_ids)
    result["combined_score"] = np.mean(list(result.values())).item()
    return result


def evaluate_model(model, tokenizer, dataset):
    training_args = TrainingArguments(per_device_eval_batch_size=128, output_dir="./eval_tmp", report_to="none")
    trainer = Trainer(model=model, args=training_args, eval_dataset=dataset["validation_matched"], compute_metrics=compute_metrics)
    return trainer.evaluate()["eval_accuracy"]


def factorize_to_fwsvd(module, bias, avg_grads, rank):
    fisher_weights = torch.diag(torch.sqrt(avg_grads.sum(0))).to(module.weight.device, module.weight.dtype)
    weighted_matrix = (fisher_weights @ module.weight.T).T
    U, S, Vt = torch.linalg.svd(weighted_matrix, full_matrices=False)
    w1 = torch.linalg.lstsq(fisher_weights, torch.diag(torch.sqrt(S[:rank])) @ Vt[:rank, :]).solution.T
    w2 = U[:, :rank] @ torch.diag(torch.sqrt(S[:rank]))
    return build_sequential(module, w1, w2, bias)


def factorize_to_svd(weight_matrix, bias, rank):
    U, S, Vt = np.linalg.svd(weight_matrix, full_matrices=False)
    w1 = np.dot(np.diag(np.sqrt(S[:rank])), Vt[:rank, :])
    w2 = np.dot(U[:, :rank], np.diag(np.sqrt(S[:rank])))
    return build_sequential(weight_matrix, w1, w2, bias, is_numpy=True)


def factorize_to_kron_svd(weight_matrix, bias, B, C, rank):
    def regularize(M):
        alpha = 0.0
        while not is_pos_def(M):
            alpha += 0.1
            M = (1 - alpha) * M + alpha * np.eye(M.shape[0])
        return M

    B, C = regularize(B), regularize(C)
    B_chol, C_chol = np.linalg.cholesky(B), np.linalg.cholesky(C)
    T = C_chol.T @ weight_matrix @ B_chol
    U, S, Vt = np.linalg.svd(T, full_matrices=False)
    U_t = np.linalg.inv(C_chol.T) @ U
    Vt_t = Vt @ np.linalg.inv(B_chol)
    w1 = np.diag(np.sqrt(S[:rank])) @ Vt_t[:rank, :]
    w2 = U_t[:, :rank] @ np.diag(np.sqrt(S[:rank]))
    return build_sequential(weight_matrix, w1, w2, bias, is_numpy=True)


def build_sequential(ref, w1, w2, bias, is_numpy=False):
    if is_numpy:
        w1, w2 = torch.FloatTensor(w1), torch.FloatTensor(w2)
        if bias is not None:
            bias = torch.FloatTensor(bias)
    has_bias = bias is not None
    l1 = nn.Linear(w1.shape[1], w1.shape[0], bias=False)
    l2 = nn.Linear(w2.shape[1], w2.shape[0], bias=has_bias)
    l1.weight = nn.Parameter(w1)
    l2.weight = nn.Parameter(w2)
    if has_bias:
        l2.bias = nn.Parameter(bias)
    return nn.Sequential(l1, l2)


def prepare_dataset(tokenizer, raw_dataset, max_length=128):
    def preprocess(ex):
        args = (ex["premise"], ex["hypothesis"])
        out = tokenizer(*args, max_length=max_length, truncation=True, padding="max_length")
        out["label"] = ex["label"]
        return out

    return raw_dataset.map(preprocess, batched=True, load_from_cache_file=False)


def compress_model(model, layers, rank, method, gradients):
    for name in layers:
        module = model.get_submodule(name)
        layer_idx = int(name.split(".")[3])
        if method == "svd":
            f_layer = factorize_to_svd(
                module.weight.detach().cpu().numpy(), module.bias.detach().cpu().numpy() if module.bias is not None else None, rank
            )
        elif method == "fwsvd":
            avg_grad = gradients[name]
            f_layer = factorize_to_fwsvd(
                module, module.bias.detach().cpu().numpy() if module.bias is not None else None, avg_grad, rank
            )
        elif method == "kron":
            B = gradients[f"B_{name}"]
            C = gradients[f"C_{name}"]
            f_layer = factorize_to_kron_svd(
                module.weight.detach().cpu().numpy(),
                module.bias.detach().cpu().numpy() if module.bias is not None else None,
                B,
                C,
                rank,
            )
        if "intermediate" in name:
            model.bert.encoder.layer[layer_idx].intermediate.dense = f_layer
        else:
            model.bert.encoder.layer[layer_idx].output.dense = f_layer
    return model


def create_model(num_labels):
    model_name = "bert-base-uncased"
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.load_state_dict(torch.load("./bert_mnli.pth"))
    return model, tokenizer


def main():
    """
    This code implements three matrix factorization techniques for BERT compression:
    1. Standard SVD decomposition
    2. Fisher-Weighted SVD (FWSVD) using gradient information
    3. Kronecker-factored SVD using second-order information
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--direction", required=True)
    parser.add_argument("--gradients_file", required=True)
    args = parser.parse_args()

    methods = ["svd", "fwsvd", "kron"]
    ranks = [1, 5, 10, 50, 100, 250, 500, 600]
    results = {m: [] for m in methods}

    raw = load_dataset("glue", "mnli")
    _, tokenizer = create_model(num_labels=3)
    data = prepare_dataset(tokenizer, raw)
    layers = [f"bert.encoder.layer.{i}.{p}" for i in range(1, 12) for p in ["intermediate.dense", "output.dense"]]

    gradients = load_file(args.gradients_file)

    for method in methods:
        print(f"\n=== Method: {method.upper()} ===")
        for rank in ranks:
            try:
                m, _ = create_model(num_labels=3)
                m = compress_model(m, layers, rank, method, gradients)
                acc = evaluate_model(m, tokenizer, data)
                print(f"Rank {rank}: Acc = {acc:.4f}")
                results[method].append(acc)
            except Exception as e:
                print(f"Failed for {method} @ rank {rank}: {e}")
                results[method].append(None)
        with open(f"results_{args.direction}.json", "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
