export WANDB_API_KEY=$(cat wandb_key.txt)
export WANDB_PROJECT="fisher-kronecker"
export WANDB_ENTITY="dondo_sss"

accelerate launch --num_processes 4 --config_file kronfwsvd/fsdp.yaml kronfwsvd/collect_grads.py \
    --path_to ./grads_output/gemma-3-1b-it \
    --model_name google/gemma-3-1b-it \
    --dataset_name roneneldan/TinyStories \
    --layer_pattern "q_proj|k_proj|v_proj|o_proj|mlp" \
    --subsample_size 16384 \
    --batch_size 4 \
    # --max_steps 5 \
    # --model_name unsloth/Llama-3.2-1B-Instruct \