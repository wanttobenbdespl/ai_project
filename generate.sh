export CUDA_VISIBLE_DEVICES="0"
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export TOKENIZERS_PARALLELISM="false"
export HF_HUB_OFFLINE="0"
export WANDB_MODE="offline"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_ALLOW_CODE_EVAL="1"
# generation args
base_path=model_ckpt/DeepSeek-R1-Distill-Qwen-1.5B
fine_tune_model_path=trained_models/codealpaca_lora_20250221_030756/checkpoint-60000

python_path=/home/ubuntu/miniconda3/envs/pytorch/bin/python

$python_path generate.py \
    --base_model $base_path \
    --peft_model $fine_tune_model_path \
    --output ./generated_results_fine_tune.json

$python_path generate.py \
    --base_model $base_path \
    --output ./generated_results_base.json