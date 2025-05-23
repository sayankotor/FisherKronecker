MODEL_NAME="unsloth/llama-2-7b-chat"

GLOBAL_BS=32
BATCH_SIZE=1
# NUM_DEVICES=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NUM_DEVICES=2
GRAD_ACCUM_STEPS=$(((4096 / GLOBAL_BS) / (BATCH_SIZE * NUM_DEVICES)))
MODEL_ID="${MODEL_NAME##*/}"

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --config_file gfwsvd/fsdp.yaml gfwsvd/collect_grads.py \
    --path_to ./grads_output/${MODEL_ID} \
    --model_name ${MODEL_NAME} \
    --dataset_name fineweb_16M \
    --layer_pattern "q_proj|k_proj|v_proj|o_proj|mlp" \
    --subsample_size 4096 \
    --max_seq_length 4096 \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --batch_size ${BATCH_SIZE} \
    --gradient_checkpointing \
    # --max_steps 5 \