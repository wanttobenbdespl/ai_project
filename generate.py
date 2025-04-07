import argparse
import os
import json
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel  # Import PEFT for LoRA adapters


def load_human_eval_dataset():
    """Load the HumanEval dataset."""
    return load_from_disk("./data/human_eval")["test"]  # Use test split for evaluation


def evaluate_model(base_model_path, peft_path, results_path):
    """Evaluate the model on the HumanEval dataset."""
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(base_model_path)

    if peft_path:
        print(f"Loading PEFT model from {peft_path}")
        model = PeftModel.from_pretrained(model, peft_path)
        model = model.merge_and_unload()

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    dataset = load_human_eval_dataset()
    results = []

    for i, example in enumerate(dataset):
        prompt = example["prompt"]
        expected_code = example["canonical_solution"]

        # Generate model response
        generated = generator(prompt, max_length=512)[0]["generated_text"]

        # Store results
        results.append(
            {
                "id": example["task_id"],
                "prompt": prompt,
                "expected": expected_code,
                "generated": generated,
            }
        )

        if i % 10 == 0:
            print(f"Evaluated {i}/{len(dataset)} samples")

    # Save results
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on HumanEval dataset")
    parser.add_argument(
        "--base_model", type=str, required=True, help="Path to the base model"
    )
    parser.add_argument(
        "--peft_model",
        type=str,
        default=None,
        help="Path to the PEFT (LoRA) model adapter",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save evaluation results"
    )

    args = parser.parse_args()

    evaluate_model(args.base_model, args.peft_model, args.output)
