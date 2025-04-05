export WANDB_API_KEY=$(cat wandb_key.txt)
export WANDB_PROJECT="fisher-kronecker"
export WANDB_ENTITY="dondo_sss"

accelerate launch --num_processes 4 --config_file kronfwsvd/fsdp.yaml kronfwsvd/collect_grads.py \
    --path_to ./grads_output/Llama-3.2-1B-Instruct \
    --model_name unsloth/Llama-3.2-1B-Instruct \
    --dataset_name /workspace-SR004.nfs2/data/fineweb \
    --layer_pattern "q_proj|k_proj|v_proj|o_proj|mlp" \
    --subsample_size 4096 \
    --max_seq_length 4096 \
    --batch_size 1 \
    # --max_steps 5 \
    # --model_name unsloth/Llama-3.2-1B-Instruct \