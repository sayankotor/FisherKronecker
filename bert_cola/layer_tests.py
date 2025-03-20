import json
import torch, torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

from datasets import load_dataset
import pandas as pd
import evaluate as ev

import numpy as np
import argparse
from transformers import TrainingArguments, Trainer, EvalPrediction

import pickle
torch.manual_seed(0)
from transformers import AutoConfig, BertConfig, AutoModelForSequenceClassification, AutoTokenizer

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

metric = ev.load("glue", 'cola')
def compute_metrics(p: EvalPrediction):
    preds_ = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds_ = np.argmax(preds_, axis=1)
        
    result = metric.compute(predictions=preds_, references=p.label_ids)
    return result
    print ("result11", result)
    if True:
        result["combined_score"] = np.mean(list(result.values())).item()
        return result
    else:
        return {"accuracy": (preds_ == p.label_ids).astype(np.float32).mean().item()}

def evalm(model, tokenizer, tokenized_dataset):
    torch.manual_seed(0)
    training_args2 = TrainingArguments(
        learning_rate=5e-5,
        num_train_epochs=1,
        evaluation_strategy="steps",
        #optim = "sgd",
        skip_memory_metrics = False,
        eval_steps=100,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=128,
        save_steps=1000,
        overwrite_output_dir=True,
        output_dir="./bert_stsb_128",
        remove_unused_columns=True,
        seed=297104,
        report_to='none',
        )
    model.train(False) # disable dropout / use averages for batch_norm
    trainer = Trainer(
        model=model,
        args=training_args2,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics = compute_metrics,
    )

    res = trainer.evaluate()
    return res['eval_matthews_correlation']


def factorize_to_fwsvd(module, fc_b, avg_grads, rank):
    I = torch.diag(torch.sqrt(avg_grads.sum(0))).to(module.weight.device, module.weight.dtype)

    U, S, Vt = torch.linalg.svd((I @ module.weight.T).T.to(module.weight.device) , full_matrices=False) # driver='gesvdj'

    w1 = torch.linalg.lstsq(I, torch.mm(torch.diag(torch.sqrt(S[0:rank])),Vt[0:rank, :]).T).solution.T
    w2 = torch.mm(U[:, 0:rank], torch.diag(torch.sqrt(S[0:rank])))

    # create new layers and insert weights
    fc_w = module.weight.data.cpu().data.numpy()
    out_features, in_features = fc_w.shape
    is_bias = fc_b is not None

    linear1 = nn.Linear(in_features = in_features,
                      out_features = rank,
                      bias = False)
    linear1.weight = nn.Parameter(torch.FloatTensor(w1))

    linear2 = nn.Linear(in_features = rank,
                      out_features = out_features,
                      bias=is_bias)
    linear2.weight = nn.Parameter(torch.FloatTensor(w2))
    linear2.bias = nn.Parameter(torch.FloatTensor(fc_b))
    print (w1.shape, w2.shape)
    # create factorized layer
    factorized_layer = nn.Sequential(linear1,linear2)
    return factorized_layer

def factorize_to_svd(fc_w, fc_b, rank):
    U, S, Vt = np.linalg.svd(fc_w, full_matrices=False)
    # truncate SVD and fuse Sigma matrix
    w1 = np.dot(np.diag(np.sqrt(S[0:rank])),Vt[0:rank, :])
    w2 = np.dot(U[:,0:rank,], np.diag(np.sqrt(S[0:rank])))

    # create new layers and insert weights
    out_features, in_features = fc_w.shape
    is_bias = fc_b is not None

    linear1 = nn.Linear(in_features = in_features,
                      out_features = rank,
                      bias = False)
    linear1.weight = nn.Parameter(torch.FloatTensor(w1))

    linear2 = nn.Linear(in_features = rank,
                      out_features = out_features,
                      bias=is_bias)
    linear2.weight = nn.Parameter(torch.FloatTensor(w2))
    linear2.bias = nn.Parameter(torch.FloatTensor(fc_b))
    print (w1.shape, w2.shape)
    # create factorized layer
    factorized_layer = nn.Sequential(linear1,linear2)
    return factorized_layer

def factorize_to_kron_svd(fc_w, fc_b, B11, C11, rank):
    alpha = 0.0
    B_new = B11
    while (not is_pos_def(B_new)):
        alpha += 0.1
        B_new = (1 - alpha)*B11  + alpha*np.eye(len(np.diag(B11)))

    print ("alpha",alpha)
    alpha = 0.0
    C_new = C11
    while (not is_pos_def(C_new)):
        alpha += 0.1
        C_new = (1 - alpha)*C11  + alpha*np.eye(len(np.diag(C11)))
    print ("alpha",alpha)

    B1_square = np.linalg.cholesky(B_new)
    C1_square = np.linalg.cholesky(C_new)
    U, S, Vt = np.linalg.svd(C1_square.T@fc_w@B1_square, full_matrices=False)

    U1 = np.linalg.inv(C1_square.T)@U
    V1t = Vt@np.linalg.inv(B1_square)

    w1 = np.diag(np.sqrt(S[:rank]))@V1t[:rank, :]
    w2 = U1[:,:rank] @ np.diag(np.sqrt(S[:rank]))

    out_features, in_features = fc_w.shape
    is_bias = fc_b is not None

    linear1 = nn.Linear(in_features = in_features,
                      out_features = rank,
                      bias = False)
    linear1.weight = nn.Parameter(torch.FloatTensor(w1))

    linear2 = nn.Linear(in_features = rank,
                      out_features = out_features,
                      bias=is_bias)
    linear2.weight = nn.Parameter(torch.FloatTensor(w2))
    linear2.bias = nn.Parameter(torch.FloatTensor(fc_b))

    factorized_layer = nn.Sequential(linear1,linear2)

    return factorized_layer

def create_and_load_model(num_labels):
    path_name = r"bert-base-uncased"
    task_num_labels = num_labels
    config = AutoConfig.from_pretrained(
    path_name,
    num_labels=num_labels,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
    path_name,
    config=config,
    )

    tokenizer = AutoTokenizer.from_pretrained(path_name)

    model.load_state_dict(torch.load("./bert_cola.pth"))

    return model
    


def compress_model(model, list_of_layer_to_compress, rank, method, saved_grads):
    for keyname in list_of_layer_to_compress:       
        module = model.get_submodule(keyname)
        rank = int(rank)
        if method == 'svd':
            factorized_layer = factorize_to_svd(module.weight.data.cpu().data.numpy(), module.bias.data.cpu().data.numpy(), rank)
        elif method == 'fwsvd':
            list_of_grads = saved_grads[keyname]
            list_of_grads_pow = [torch.pow(elem,2) for elem in list_of_grads]
            avg_grads = torch.mean(torch.stack(list_of_grads_pow, dim=0),dim = 0)
            factorized_layer = factorize_to_fwsvd(module, module.bias.data.cpu().data.numpy(), avg_grads, rank)
            
        elif method == 'kron':
            B11 = np.load("./kron_factors/B1sgd_"+str(keyname) +".npy")#np.linalg.cholesky(B1)
            C11 = np.load("./kron_factors/C1sgd_"+str(keyname) +".npy")#np.linalg.cholesky(C1)
            try:
                factorized_layer = factorize_to_kron_svd(module.weight.data.cpu().data.numpy(), module.bias.data.cpu().data.numpy(), B11, C11, rank)
            except:
                print (is_pos_def(B11), is_pos_def(C11))
                print (keyname)
                factorized_layer = module

        
        i = int(keyname.split('.')[3])
        if "intermediate" in keyname:
            model.bert.encoder.layer[i].intermediate.dense = factorized_layer
        else:
            model.bert.encoder.layer[i].output.dense = factorized_layer
        
    return model

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
sentence1_key, sentence2_key = task_to_keys['cola']




def main():
    parser = argparse.ArgumentParser(
        description="Generate experiment folder structure."
    )
    parser.add_argument(
        "--direction", type=str, required=True, help="Tuda or Obratno."
    )

    args = parser.parse_args()

    #compression_sequences = {}
    #compression_sequences['one'] = [['2'], ['3'],  ['4']]
    #compression_sequences['obratno'] = [['4'], ['3', '4'], ['4','2', '3']]

    ranks = [5, 10, 50, 100, 250, 500, 600]

    results = {"svd":{}, "fwsvd":{}, "kron":{}}

    path_name = r"bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(path_name)
    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        
        
        result = tokenizer.batch_encode_plus(*args, max_length=128, truncation=True, padding="max_length")

        result["label"] = examples["label"]
        return result

    dataset_cola = load_dataset('glue', 'cola')
    print(dataset_cola.num_rows)
    # Encode the input data
    tokenized_dataset = dataset_cola.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=False
    )
    label_list = dataset_cola["train"].features["label"].names
    num_labels = len(label_list)

    part_of=["bert.encoder.layer.1.intermediate.dense","bert.encoder.layer.2.intermediate.dense","bert.encoder.layer.3.intermediate.dense","bert.encoder.layer.4.intermediate.dense","bert.encoder.layer.5.intermediate.dense", "bert.encoder.layer.6.intermediate.dense", "bert.encoder.layer.7.intermediate.dense", "bert.encoder.layer.8.intermediate.dense", "bert.encoder.layer.9.intermediate.dense", "bert.encoder.layer.10.intermediate.dense", "bert.encoder.layer.11.intermediate.dense", "bert.encoder.layer.1.output.dense","bert.encoder.layer.2.output.dense","bert.encoder.layer.3.output.dense","bert.encoder.layer.4.output.dense","bert.encoder.layer.5.output.dense", "bert.encoder.layer.6.output.dense", "bert.encoder.layer.7.output.dense", "bert.encoder.layer.8.output.dense", "bert.encoder.layer.9.output.dense", "bert.encoder.layer.10.output.dense", "bert.encoder.layer.11.output.dense"]

    reduced_part_of = ["bert.encoder.layer.1.intermediate.dense","bert.encoder.layer.3.intermediate.dense","bert.encoder.layer.5.intermediate.dense", "bert.encoder.layer.6.intermediate.dense", "bert.encoder.layer.7.intermediate.dense", "bert.encoder.layer.8.intermediate.dense", "bert.encoder.layer.9.intermediate.dense", "bert.encoder.layer.11.intermediate.dense", "bert.encoder.layer.1.output.dense","bert.encoder.layer.4.output.dense","bert.encoder.layer.5.output.dense", "bert.encoder.layer.6.output.dense", "bert.encoder.layer.7.output.dense", "bert.encoder.layer.8.output.dense", "bert.encoder.layer.9.output.dense", "bert.encoder.layer.10.output.dense"]


    with open("cola_grads.pickle", "rb") as fp:   #Pickling
        dict_of_grads = pickle.load(fp)

    for method in ["svd", "fwsvd", "kron"]:
        print ("\n\n")
        results[method]['all'] = []
        if (True):
            for rank in ranks:
                print (method, rank)
                model = create_and_load_model(num_labels)
                print ("model created")
                model = compress_model(model, reduced_part_of, rank, method, dict_of_grads) 
                print ("model compressed")
                score = evalm(model, tokenizer, tokenized_dataset)
                print ("score", score)
                results[method]['all'].append(score)
                
                
        #for keyname in part_of:
            #seq = keyname.split('.')[3:]
            #seq_str = ' '.join(seq)
            #results[method][seq_str] = []
            #for rank in ranks:
                #try:
                    #print (method, seq, rank)
                    #model = create_and_load_model(num_labels)
                    #print ("model created")
                    #module = model.get_submodule(keyname)
                    #model = compress_model(model, [keyname], rank, method, dict_of_grads) 
                    #print ("model compressed")
                    #score = evalm(model, tokenizer, tokenized_dataset)
                    #print ("score", score)
                    #results[method][seq_str].append(score)
                #except:
                    #with open('./results/results1'+str(args.direction)+'.json', 'w') as fp:
                        #json.dump(results, fp)
                    #pass
                    

    with open('./results/results'+str(args.direction)+'all.json', 'w') as fp:
            json.dump(results, fp)
                   


if __name__ == "__main__":
    main()