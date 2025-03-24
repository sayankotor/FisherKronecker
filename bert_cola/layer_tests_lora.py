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
from transformers import AutoConfig, BertConfig, AutoModelForSequenceClassification, AutoTokenizer
from functools import reduce

from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
)

part_of =["bert.encoder.layer.1.intermediate.dense","bert.encoder.layer.2.intermediate.dense","bert.encoder.layer.3.intermediate.dense","bert.encoder.layer.4.intermediate.dense","bert.encoder.layer.5.intermediate.dense", "bert.encoder.layer.6.intermediate.dense", "bert.encoder.layer.7.intermediate.dense", "bert.encoder.layer.8.intermediate.dense", "bert.encoder.layer.9.intermediate.dense", "bert.encoder.layer.10.intermediate.dense", "bert.encoder.layer.11.intermediate.dense", "bert.encoder.layer.1.output.dense","bert.encoder.layer.2.output.dense","bert.encoder.layer.3.output.dense","bert.encoder.layer.4.output.dense","bert.encoder.layer.5.output.dense", "bert.encoder.layer.6.output.dense", "bert.encoder.layer.7.output.dense", "bert.encoder.layer.8.output.dense", "bert.encoder.layer.9.output.dense", "bert.encoder.layer.10.output.dense", "bert.encoder.layer.11.output.dense"]


def get_module_by_name(module, access_string):
     names = access_string.split(sep='.')
     return reduce(getattr, names, module)

class CustomTrainer(Trainer):
    def make_grad_bank(self):
        self.grads_A = dict() #defaultdict(torch.tensor)
        self.grads_B = dict()
        for name, module in self.model.base_model.model.named_modules():
            if name in part_of:
                print("Init ::",name)
                self.grads_A[name] = []
                self.grads_B[name] = []
        self.avg_counter = 0

    def training_step(
        self, model, inputs, num_items_in_batch=None
    ) -> torch.Tensor:
        
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs).mean()

        del inputs

        kwargs = {}

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)

        #print (self.model)
        for name, module in self.model.base_model.model.named_modules():
            if name in part_of:
                #print ("collecting grads from ", name)
                #new_var = get_module_by_name(model, name).weight.grad.detach().cpu()**2
                #self.mass[name] += new_var
                #new_var = get_module_by_name(model, name).weight.detach().cpu()
                i= name.find('.')
                res = name[i+1:]
                self.grads_A[name].append(get_module_by_name(model, name).lora_A.default.weight.grad.detach().cpu().numpy())
                self.grads_B[name].append(get_module_by_name(model, name).lora_B.default.weight.grad.detach().cpu().numpy())
                #self.mass_w[name].append(get_module_by_name(model, name).weight.detach().cpu())

            self.avg_counter += 1

        return loss.detach() / self.args.gradient_accumulation_steps

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

def get_factor_left(list_of_lora_factors):
    m, n = list_of_lora_factors[0].shape 
    list_of_grads1 = [grad.reshape(-1) for grad in list_of_lora_factors]

    grad_vectors = np.stack([grad.reshape(n,m, order = 'F') for grad in list_of_grads1])
    k, m, n = grad_vectors.shape
    print ("k, m, n", k, m, n)
    
    res = np.zeros(n*n) 
    e_d = np.ones((m, 1))
    V = (e_d@e_d.T)
    d_size = V.shape[0]*V.shape[1]
    for i in range(k):
        res += (grad_vectors[i].T @ grad_vectors[i]).T.ravel()
    return res/k

def get_factor_right(list_of_lora_factors):
    m, n = list_of_lora_factors[0].shape #cols rows in torch -> rows cols in numpy
    list_of_grads1 = [grad.reshape(-1) for grad in list_of_lora_factors]

    grad_vectors = np.stack([grad.reshape(n,m, order = 'F') for grad in list_of_grads1])
    k, m, n = grad_vectors.shape
    print ("k, m, n", k, m, n)
    k, m, n = grad_vectors.shape
    #e_d = np.ones((n, 1))
    res = np.zeros(m*m) 
    e_d = np.ones((n, 1))
    V = (e_d@e_d.T)
    d_size = V.shape[0]*V.shape[1]
    
    for i in range(k):
        res += (grad_vectors[i] @grad_vectors[i].T).T.ravel()
    return res/k

def replace_layer(model_base, layer_name, list_factors_A, list_factors_B, A, B):
    #print ("weight_old",model_base.bert.encoder.layer[7].intermediate.dense.weight)
    dW_old = B@A
    print ("\n\n\n")
    
    m, n = list_factors_A[0].shape
    R = get_factor_right(list_factors_A).reshape(n, n, order='F')
    m, n = list_factors_B[0].shape
    LT = get_factor_left(list_factors_B).reshape(m, m, order='F')
    
    #print (LT.shape, "LT.Shape")
    #print (B.shape, "B.Shape")
    #print (A.shape, "A.Shape")
    #print (R.shape, "R.Shape")
    #print ("Lt", LT[:5, :5])
    #print ("R", R[:5, :5])
    alpha = 0.0
    LT_new = LT
    while (not is_pos_def(LT_new)):
        alpha += 0.1
        #print (alpha)
        LT_new  = (1 - alpha)*LT  + alpha*np.eye(len(np.diag(LT)))
    

    alpha = 0.0
    R_new = R
    while (not is_pos_def(R_new)):
        alpha += 0.1
        #print (alpha)
        R_new = (1 - alpha)*R  + alpha*np.eye(len(np.diag(R)))
    
    LT_square = np.linalg.cholesky(LT_new)
    R_square = np.linalg.cholesky(R_new)
    qb, rb = np.linalg.qr(LT_square@B)
    qa, ra = np.linalg.qr(R_square.T@A.T)
    U, S, Vt = np.linalg.svd(rb@ra.T)#np.linalg.svd(LT@P@Q@R)
    U_new = qb@U
    Vt_new = Vt@(qa.T)
    #Vt_new = Vt @ np.linalg.inv(R_square)
    #U_new = np.linalg.inv(LT_square) @ U

    #dW_old = B@A
    dW = U_new[:,:len(S)]@np.diag(S)@Vt_new
    print ("dw old", dW_old[:5, :5])
    print ("dw new", dW[:5, :5])
    #print (model_base.bert.encoder.layer[1].intermediate.dense.weight[:5, :5])
    i = int(layer_name.split('.')[3])
    with torch.no_grad():
        if "intermediate" in layer_name:
            model_base.bert.encoder.layer[i].intermediate.dense.weight += torch.tensor(dW).to(model_base.device)
        else:
            model_base.bert.encoder.layer[i].output.dense.weight += torch.tensor(dW_old).to(model_base.device)
    

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

def load_model(path_name, num_labels):
    config = AutoConfig.from_pretrained(
        path_name,
        num_labels=num_labels,
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        path_name,
        config=config,
    )

    return model


def main():
    ranks = [1, 4, 8, 16, 32]

    results = {"reg_lora":[], "kron_lora":[]}

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
    path_name = r"bert-base-uncased"

    training_args = TrainingArguments(
    learning_rate=3e-4,
    num_train_epochs=1,
    evaluation_strategy="steps",
    skip_memory_metrics = False,
    eval_steps=100,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=128,
    save_steps=1000,
    overwrite_output_dir=True,
    output_dir="./bert_stsb_128",
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=True,
    seed=297104,
    report_to='none',
    )
    
    for rank in ranks: 
        print ("lora rank", rank)
        peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=rank, target_modules=part_of, lora_alpha=16, lora_dropout=0.1)
        model = load_model(path_name, num_labels)
        
        model = get_peft_model(model, peft_config)
        trainer_collect = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            compute_metrics=compute_metrics,
        )
        trainer_collect.make_grad_bank()
        trainer_collect.train()
        list_A = {}
        list_B = {}
        for name, module in trainer_collect.model.base_model.model.named_modules():
            if name in part_of:
                list_A[name] = get_module_by_name(trainer_collect.model, name).lora_A.default.weight.detach().cpu().numpy()
                list_B[name] = get_module_by_name(trainer_collect.model, name).lora_B.default.weight.detach().cpu().numpy()
                
        merged_model = trainer_collect.model.merge_and_unload()
        print (trainer_collect.model)
        res = trainer_collect.evaluate()
        reg_res = res['eval_matthews_correlation']

        for layer_name in part_of:
            print ("layer_name", layer_name)
            list_factors_A = trainer_collect.grads_A[layer_name]
            list_factors_B = trainer_collect.grads_B[layer_name]
            P = list_A[layer_name]
            Q = list_B[layer_name]
            replace_layer(trainer_collect.model, layer_name, list_factors_A, list_factors_B, P, Q)

        res = trainer_collect.evaluate()
        kron_res = res['eval_matthews_correlation']

        results["reg_lora"].append(reg_res)
        results["kron_lora"].append(kron_res)
        print ("\n\nresults!!\n", results)





    with open('./results/results_lora.json', 'w') as fp:
            json.dump(results, fp)
                   


if __name__ == "__main__":
    main()