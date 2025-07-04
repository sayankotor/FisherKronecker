python calibrate_llama_with_kronsvd.py \
    --model_name_or_path "unsloth/llama-2-7b-chat" \
    --kron_factors_dir "../grads_output/llama-2-7b-chat/fisher_factors_output_1404" \
    --calib_dataset "wikitext2" \
    --n_calib_samples 128 \
    --calib_batch_size 4 \
    --eval_batch_size 8 \
    --seq_len 2048 \
    --rank_align 8 \
    --output_dir "./llama2_7b_chat_sensitivity" \
    --device "cuda:0" \
    --torch_dtype "bfloat16"