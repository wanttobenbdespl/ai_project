export CUDA_VISIBLE_DEVICES="1"
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export TOKENIZERS_PARALLELISM="false"
# export HF_HUB_OFFLINE="0"
export WANDB_MODE="offline"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_ALLOW_CODE_EVAL="1"
# evaluation arguments
base_path=model_ckpt/DeepSeek-R1-Distill-Qwen-1.5B
fine_tune_model_path=trained_models/codealpaca_lora_20250227_221049_r128/checkpoint-60000
eval_tasks="cola,mrpc,sst2,rte,ai2_arc,mbpp,humaneval"
batch_size=8

# begin to evaluate the base model and the finetuned model.
lm_eval --model hf \
    --model_args pretrained=$base_path \
    --tasks $eval_tasks \
    --batch_size $batch_size \
    --confirm_run_unsafe_code  \
    --output_path "eval_results/DeepSeek-R1-Distill-Qwen-1.5B" 

lm_eval --model hf \
    --model_args pretrained=$base_path,peft=$fine_tune_model_path \
    --tasks $eval_tasks \
    --batch_size $batch_size \
    --confirm_run_unsafe_code  \
    --output_path "eval_results/tuned-DeepSeek-R1-Distill-Qwen-1.5B" 


# Test a series of lora
# List of PEFT folders to evaluate
peft_folders=(
    "trained_models/codealpaca_lora_20250221_030756_r8"
    "trained_models/codealpaca_lora_20250227_102756_r16"
    "trained_models/codealpaca_lora_20250227_162631_r32"
    "trained_models/codealpaca_lora_20250227_191726_r64"
    "trained_models/codealpaca_lora_20250227_221049_r128"
)

# Loop through each PEFT folder and evaluate
for peft_folder in "${peft_folders[@]}"; do
    # Extract the rank from the folder name (e.g., r8, r16, etc.)
    rank=$(basename "$peft_folder" | grep -oP 'r\d+')

    # Define the output path based on the rank
    output_path="eval_results/DeepSeek-R1-Distill-Qwen-1.5B-${rank}"

    # Run the evaluation
    lm_eval --model hf \
        --model_args pretrained=$base_path,peft="${peft_folder}/checkpoint-60000" \
        --tasks $eval_tasks \
        --batch_size $batch_size \
        --confirm_run_unsafe_code \
        --output_path "$output_path"
done