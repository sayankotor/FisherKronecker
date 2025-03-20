import argparse
from datasets import load_dataset
from transformers import AutoConfig, BertConfig, AutoModelForSequenceClassification
from transformers import RobertaTokenizer, BertTokenizer, AutoTokenizer

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from transformers import Trainer
import evaluate

import os
import numpy as np
from transformers import TrainingArguments, Trainer, EvalPrediction

import pickle
from collections import defaultdict
from functools import reduce

def get_module_by_name(module, access_string):
     names = access_string.split(sep='.')
     return reduce(getattr, names, module)

from collections import defaultdict
from functools import reduce

from transformers import set_seed
from torch import optim

set_seed(42)


part_of = ["module.bert.encoder.layer.1.intermediate.dense","module.bert.encoder.layer.2.intermediate.dense","module.bert.encoder.layer.3.intermediate.dense","module.bert.encoder.layer.4.intermediate.dense","module.bert.encoder.layer.5.intermediate.dense", "module.bert.encoder.layer.6.intermediate.dense", "module.bert.encoder.layer.7.intermediate.dense", "module.bert.encoder.layer.8.intermediate.dense", "module.bert.encoder.layer.9.intermediate.dense", "module.bert.encoder.layer.10.intermediate.dense", "module.bert.encoder.layer.11.intermediate.dense", "module.bert.encoder.layer.1.output.dense","module.bert.encoder.layer.2.output.dense","module.bert.encoder.layer.3.output.dense","module.bert.encoder.layer.4.output.dense","module.bert.encoder.layer.5.output.dense", "module.bert.encoder.layer.6.output.dense", "module.bert.encoder.layer.7.output.dense", "module.bert.encoder.layer.8.output.dense", "module.bert.encoder.layer.9.output.dense", "module.bert.encoder.layer.10.output.dense", "module.bert.encoder.layer.11.output.dense"]

def get_module_by_name(module, access_string):
     names = access_string.split(sep='.')
     return reduce(getattr, names, module)

class CustomTrainer(Trainer):
    def make_grad_bank(self):
        self.mass = dict() #defaultdict(torch.tensor)
        self.mass_w = dict()
        for name, module in self.model.named_modules():
            name_m = "module." + str(name)
            if name_m in part_of:
                print("Init ::",name)
                self.mass[name] = []
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
        for name, module in model.named_modules():
            if str(name) in part_of:
                if get_module_by_name(model, name).weight.requires_grad:
                    print ("collecting grads from ", name)
                    #new_var = get_module_by_name(model, name).weight.grad.detach().cpu()**2
                    #self.mass[name] += new_var
                    #new_var = get_module_by_name(model, name).weight.detach().cpu()
                    i= name.find('.')
                    res = name[i+1:]
                    self.mass[res].append(get_module_by_name(model, name).weight.grad.detach().cpu())
                    #self.mass_w[name].append(get_module_by_name(model, name).weight.detach().cpu())

            self.avg_counter += 1

        return loss.detach() / self.args.gradient_accumulation_steps


def training_process(path_to):
    """
    Training process function
    :param path_to: Path to the data/resource (provided via command-line)
    :param size_of: Size parameter (provided via command-line)
    :param lr: Learning rate parameter (provided via command-line)
    """

    dataset = load_dataset('glue', 'cola')
    path_name =  "bert-base-uncased"
    config = AutoConfig.from_pretrained(
        path_name,
        num_labels=2,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        path_name,
        config=config,
        #quantization_config=config_Q,
    )

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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

    metric = evaluate.load("glue", 'cola')
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        result["acc"] = (preds == p.label_ids).mean()
        return result

    def compute_accuracy(p: EvalPrediction):
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": (preds == p.label_ids).mean()}

    #optimizer = optim.SGD(model.parameters(), lr=1e-3)

    training_args1 = TrainingArguments(
    learning_rate=3.15e-5,
    num_train_epochs=1,
    evaluation_strategy="steps",
    #optim = "sgd",
    skip_memory_metrics = False,
    eval_steps=100,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=128,
    save_steps=1000,
    overwrite_output_dir=True,
    output_dir="./bert_sst2",
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=True,
    seed=297104,
    report_to='none',
    )

    trainer1 = Trainer(
        model=model,
        args=training_args1,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics = compute_metrics,
    )
    trainer1.train()
    res = trainer1.evaluate()
    print ("res AdamW", res)
    
    training_args = TrainingArguments(
        #learning_rate=lr,
        learning_rate=1e-3,
        num_train_epochs=1,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        save_steps=100,
        eval_steps=100,
        optim = "sgd",
        output_dir=path_to,
        #warmup_ratio = 0.1,
        #lr_scheduler_type = "cosine",
        overwrite_output_dir=True,
        save_strategy='steps',
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=True,
        seed=42,
        #load_best_model_at_end=True,
        #metric_for_best_model='eval_acc',
        report_to = 'none',
        )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
    )
    trainer.make_grad_bank()
        
    trainer.train()
    print('Results:')
    print(trainer.evaluate())
    #trainer.save_model()
    torch.save(trainer.model.state_dict(), "./bert_cola_v2.pth")
    print ("Len tranes mass:",len(trainer.mass))
    with open(os.path.join(trainer.args.output_dir, 'cola_grads_v2.pickle'), 'wb') as f:
        pickle.dump(trainer.mass, f)

    #with open(os.path.join(trainer.args.output_dir, 'tensor_weight8_11.pickle'), 'wb') as f:
        #pickle.dump(trainer.mass_w, f)


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Training process configuration')
    parser.add_argument('--path_to', type=str, required=True, 
                       help='Path to the output')
    
    # Parse arguments and run training
    args = parser.parse_args()

    print(f"Training process started with:")
    print(f"Output Path: {args.path_to}")
    #print(f"Batch Size: {args.size_of}")
    #print(f"Learning Rate: {args.lr}")


    # Call training function with command-line arguments
    training_process(args.path_to)