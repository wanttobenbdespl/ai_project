export CUDA_VISIBLE_DEVICES="1"
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export TOKENIZERS_PARALLELISM="false"
export HF_HUB_OFFLINE="0"
export WANDB_MODE="offline"

python_path="python"

# Choose an launch endpoint, you can use hugginface's accelerate or just python
# launch_endpoint="accelerate launch --mixed_precision=fp16 --use_deepspeed"
launch_endpoint=$python_path

$launch_endpoint fine_tune_model.py \
    --use_lora \
    --lora_rank 32 \
    --dataset codealpaca \
    --batch_size 2 \
    --fp16 \
    --max_steps 60000 \
    --save_steps 10000 \
    --model_ckpt model_ckpt/DeepSeek-R1-Distill-Qwen-1.5B 

$launch_endpoint fine_tune_model.py \
    --use_lora \
    --lora_rank 64 \
    --dataset codealpaca \
    --batch_size 2 \
    --fp16 \
    --max_steps 60000 \
    --save_steps 10000 \
    --model_ckpt model_ckpt/DeepSeek-R1-Distill-Qwen-1.5B 

$launch_endpoint fine_tune_model.py \
    --use_lora \
    --lora_rank 128 \
    --dataset codealpaca \
    --batch_size 2 \
    --fp16 \
    --max_steps 60000 \
    --save_steps 10000 \
    --model_ckpt model_ckpt/DeepSeek-R1-Distill-Qwen-1.5B 