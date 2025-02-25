#!/bin/bash

for n_layer in 1 2 3 4; do
    python3 bert_cola/get_fisher_factors.py \
        --path bert_cola/fishers/tensor_grad57_out.pickle\
        --output_dir fisher_factors\
        --n_layer ${n_layer} \
        --output
    done;


for n_layer in 1 2 3 4; do
    python3 bert_cola/get_fisher_factors.py \
        --path bert_cola/fishers/tensor_grad57_out.pickle\
        --output_dir fisher_factors\
        --n_layer ${n_layer}
    done;