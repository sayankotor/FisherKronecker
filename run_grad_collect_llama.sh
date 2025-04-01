#!/usr/bin/env bash
set -e # Exit immediately if a command exits with a non-zero status.
# set -x # Uncomment for debugging - prints commands before execution

# --- Configuration ---
# Adjust these variables according to your needs

# Required: Output directory where the pickle file will be saved
OUTPUT_DIR="./collected_gradients_output"

# Model and Dataset details
MODEL_NAME="unsloth/Llama-3.1-8B-Instruct" # Or "unsloth/Llama-3.1-8B-Instruct" or any other HF model
DATASET_NAME="wikitext"
DATASET_CONFIG="wikitext-103-v1" # e.g., 'wikitext-103-v1', 'en' for C4, 'default' for SlimPajama
DATASET_TEXT_FIELD="text"         # The column name in the dataset containing the text

# Training Hyperparameters
BATCH_SIZE=1           # Per-device batch size (micro-batch size). Keep low for large models.
GRAD_ACCUM_STEPS=1     # Gradient accumulation steps. Effective BS = BATCH_SIZE * GRAD_ACCUM_STEPS * NUM_GPUS
LEARNING_RATE=1e-5     # Learning rate (less critical for grad collection, but needed)
MAX_STEPS=100          # Number of gradient steps (backward passes) to collect
MAX_SEQ_LENGTH=512     # Max sequence length for tokenizer and model
SEED=42

# Layer Selection (space-separated list of indices, or leave empty for ALL linear layers)
# Example: Collect from specific Llama layers 0, 15, and 31
# LAYERS_INDICES="0 15 31"
# Example: Collect from all linear layers
LAYERS_INDICES=""

# Resource Management & Precision
# Set to true to disable bf16 (will use fp32 unless qlora is true)
NO_BF16=false
# Set to true to use QLoRA (4-bit quantization). If true, bf16 is usually ignored.
USE_QLORA=false
# Set to true to enable gradient checkpointing (saves VRAM at the cost of slower training)
GRADIENT_CHECKPOINTING=true

# Data Handling
# Set to a number (e.g., 1000) to subsample the dataset, or leave empty/0 for full dataset
SUBSAMPLE_SIZE="1000"
# Set to true to collect weights alongside gradients
COLLECT_WEIGHTS=false

# --- Python Script Location ---
PYTHON_SCRIPT_NAME="collect_grads_llama.py" # Make sure this matches your python script filename

# --- Pre-run Checks ---
if [ ! -f "$PYTHON_SCRIPT_NAME" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT_NAME' not found in the current directory."
    exit 1
fi

if ! command -v accelerate &> /dev/null
then
    echo "Error: 'accelerate' command not found. Please install accelerate: pip install accelerate"
    echo "You might also need to configure it once using: accelerate config"
    exit 1
fi

# --- Environment Setup Reminder ---
# Make sure you have the necessary libraries installed in your environment:
# pip install torch transformers datasets accelerate bitsandbytes numpy tqdm
# For bitsandbytes (required for QLoRA or 8-bit), ensure CUDA compatibility.
# Login to Hugging Face Hub if using gated models: huggingface-cli login

# --- Build the command ---
# We use 'accelerate launch' because the script uses the Accelerator and Trainer
# which handle device placement and distributed training automatically.
CMD="accelerate launch ${PYTHON_SCRIPT_NAME}"
CMD+=" --output_path \"${OUTPUT_DIR}\""
CMD+=" --model_name \"${MODEL_NAME}\""
CMD+=" --dataset_name \"${DATASET_NAME}\""
CMD+=" --dataset_config_name \"${DATASET_CONFIG}\""
CMD+=" --dataset_text_field \"${DATASET_TEXT_FIELD}\""
CMD+=" --batch_size ${BATCH_SIZE}"
CMD+=" --gradient_accumulation_steps ${GRAD_ACCUM_STEPS}"
CMD+=" --lr ${LEARNING_RATE}"
CMD+=" --max_steps ${MAX_STEPS}"
CMD+=" --max_seq_length ${MAX_SEQ_LENGTH}"
CMD+=" --seed ${SEED}"

# Add optional arguments only if they are set

if [ -n "$LAYERS_INDICES" ]; then
    # Pass the space-separated string directly; argparse with nargs='+' handles it
    CMD+=" --layers_indices ${LAYERS_INDICES}"
fi

if [ "$NO_BF16" = true ]; then
    CMD+=" --no_bf16"
fi

if [ "$USE_QLORA" = true ]; then
    CMD+=" --use_qlora"
fi

if [ -n "$SUBSAMPLE_SIZE" ] && [ "$SUBSAMPLE_SIZE" -gt 0 ]; then
    CMD+=" --subsample_size ${SUBSAMPLE_SIZE}"
fi

if [ "$COLLECT_WEIGHTS" = true ]; then
    CMD+=" --collect_weights"
fi

if [ "$GRADIENT_CHECKPOINTING" = true ]; then
    CMD+=" --gradient_checkpointing"
fi

# --- Print configuration and Run ---
echo "=========================================================="
echo "Starting Gradient Collection Run"
echo "=========================================================="
echo "Configuration:"
echo "  Output Directory:         ${OUTPUT_DIR}"
echo "  Model Name:               ${MODEL_NAME}"
echo "  Dataset:                  ${DATASET_NAME} (${DATASET_CONFIG})"
echo "  Dataset Text Field:       ${DATASET_TEXT_FIELD}"
echo "  Batch Size (per device):  ${BATCH_SIZE}"
echo "  Grad Accum Steps:         ${GRAD_ACCUM_STEPS}"
echo "  Effective Batch Size:     ~${BATCH_SIZE} * ${GRAD_ACCUM_STEPS} * (Num GPUs)"
echo "  Learning Rate:            ${LEARNING_RATE}"
echo "  Max Steps:                ${MAX_STEPS}"
echo "  Max Sequence Length:      ${MAX_SEQ_LENGTH}"
echo "  Seed:                     ${SEED}"
echo "  Layer Indices:            ${LAYERS_INDICES:-'All Linear Layers'}"
echo "  Disable BF16:             ${NO_BF16}"
echo "  Use QLoRA:                ${USE_QLORA}"
echo "  Gradient Checkpointing:   ${GRADIENT_CHECKPOINTING}"
echo "  Subsample Size:           ${SUBSAMPLE_SIZE:-'Full Dataset'}"
echo "  Collect Weights:          ${COLLECT_WEIGHTS}"
echo "  Python Script:            ${PYTHON_SCRIPT_NAME}"
echo "----------------------------------------------------------"
echo "Executing command:"
# Use eval to handle potential spaces in paths/args correctly if quoted within CMD
echo "eval ${CMD}"
echo "----------------------------------------------------------"
echo "Starting run..."

# Execute the command
eval ${CMD}

echo "=========================================================="
echo "Script finished."
echo "Check '${OUTPUT_DIR}' for the collected gradients/weights file."
echo "(Filename will be like collected_gradients_model-...) "
echo "=========================================================="