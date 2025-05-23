# GFWSVD

This repository contains source code to reproduce results of the paper Generalized Fisher-Weighted SVD.

## BERT compression

1. To collect calibrating gradients on a specific task while fine-tuning the model, use

```bash
python gfwsvd/collect_bert_task_grads.py --task sst2 --output_dir ./outputs/sst2
```

2. Then run SVD/FWSVD/GFWSVD layer-uniform compression with

```bash
python gfwsvd/compress_bert.py \
  --direction ./sst2_results \
  --gradients_file ./outputs/sst2
```

## LLM compression

1. To collect calibrating gradients on a [pre-defined](./fineweb_16M/) subset 16M tokens of fineweb, run

```bash
collect_llm_grads.sh
```

2. Compute Kronecker factors for GFWSVD with

```bash
python gfwsvd/collect_kron_factors_cupy.py --input grads_output/llama-2-7b-chat --num_devices 4 --chunk_size 16 
```

3. Obtain one of possible compression configurations, i.e. via ASVD proposed algorithm or using one of our [configs](./gfwsvd/layers_min_ratio_llama_2_7b_chat_10.json), & compress with

```bash
python python kron_svd_compression.py \
    --model_name unsloth/llama-2-7b-chat \
    --kron_factors_dir /path/to/kron_factors \
    --sensitivity_path gfwsvd/layers_min_ratio_llama_2_7b_chat_10.json \
    --output_dir ./llama2-kron-svd-compressed \
    --rank_alignment 8 \
    --min_rank 1 \
    --batch_size 8 \
    --seq_len 2048 \
    --eval_datasets wikitext2 ptb
```

3.1. Optionally, measure generation latency & memory usage with

```bash
python gfwsvd/benchmark_speed_mem.py --model_name unsloth/llama-2-7b-chat --prefill_len 1024 --decode_len 256 --compression_ratio 0.2 --compression_config gfwsvd/layers_min_ratio_llama_2_7b_chat_20.json --output_json gen_020.json
```
