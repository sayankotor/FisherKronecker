export WANDB_API_KEY=$(cat wandb_key.txt)
export WANDB_PROJECT="fisher-kronecker"
export WANDB_ENTITY="dondo_sss"

MODEL_NAME="unsloth/Llama-3.2-1B-Instruct"
# MODEL_NAME="unsloth/Llama-3.1-8B-Instruct"
# MODEL_NAME="google/gemma-3-12b-it"
# MODEL_NAME="Qwen/Qwen2.5-14B-Instruct-1M"
# MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
# MODEL_NAME="microsoft/Phi-4-mini-instruct"

GLOBAL_BS=128
BATCH_SIZE=1
NUM_DEVICES=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
GRAD_ACCUM_STEPS=$((GLOBAL_BS / (BATCH_SIZE * NUM_DEVICES)))
MODEL_ID="${MODEL_NAME##*/}"

accelerate launch --num_processes 4 --config_file kronfwsvd/fsdp.yaml kronfwsvd/collect_grads.py \
    --path_to ./grads_output/${MODEL_ID} \
    --model_name ${MODEL_NAME} \
    --dataset_name /workspace-SR004.nfs2/data/fineweb \
    --layer_pattern "q_proj|k_proj|v_proj|o_proj|mlp" \
    --subsample_size 4096 \
    --max_seq_length 4096 \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --batch_size ${BATCH_SIZE} \
    --gradient_checkpointing \
    # --max_steps 5 \