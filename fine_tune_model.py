import argparse
import os
from datasets import load_from_disk
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
    Qwen2ForCausalLM,
    Qwen2Tokenizer,
)
from datetime import datetime
import matplotlib.pyplot as plt
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM


# Function to load dataset
def load_dataset(dataset_name, data_path):
    dataset_path = os.path.join(data_path, dataset_name)
    return load_from_disk(dataset_path)


class LoggingCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.training_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.training_losses.append(logs["loss"])

    def on_train_end(self, args, state, control, **kwargs):
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_losses)
        plt.title("Training Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.savefig(f"{self.output_dir}/training_loss.png")
        plt.close()


# Main function for training
def train(args):
    # Load dataset
    dataset = load_dataset(args.dataset, args.data_path)

    # Load tokenizer and model
    tokenizer: Qwen2Tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)
    model: Qwen2ForCausalLM = AutoModelForCausalLM.from_pretrained(args.model_ckpt)

    # Apply LoRA if enabled
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules="all-linear",
            modules_to_save=["lm_head", "embed_token"],
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    # Tokenization function
    response_template = "### Answer:"

    # Define the formatting function
    def formatting_prompts_func(example):
        text = (
            f"### Question: {example['instruction']}\n### Answer: {example['output']}"
        )
        return text

    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # Training arguments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"trained_models/{args.dataset}_{'lora' if args.use_lora else 'full'}_{timestamp}"

    training_args = SFTConfig(
        output_dir=output_dir,
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        report_to=args.report_to,
        save_strategy="steps",
        save_only_model=True,
        logging_steps=args.logging_steps,
    )

    logging_callback = LoggingCallback(output_dir)
    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        callbacks=[logging_callback],
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    # Start Training
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DeepSeek-Qwen-1.5B")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["codealpaca", "mbpp"],
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--data_path", type=str, default="data", help="Path to datasets"
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default="model_ckpt/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Pretrained model path",
    )
    parser.add_argument(
        "--use_lora", action="store_true", help="Use LoRA for fine-tuning"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--max_steps", type=int, default=1000, help="Maximum number of training steps"
    )

    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum length of the input sequence",
    )
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 training")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps for gradient accumulation",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="Number of steps between logging"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Number of steps between model saves",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help="Where to report training metrics (none, wandb)",
    )

    args = parser.parse_args()
    train(args)
