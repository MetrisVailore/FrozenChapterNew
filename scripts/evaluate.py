"""
Evaluation Script
=================
Evaluate model on test set and compute metrics.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import ujson as json
import jsonlines
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import numpy as np


def load_model(model_path: str, base_model: str = None):
    """Load trained model."""

    print(f"📥 Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        if base_model is None:
            adapter_config = Path(model_path) / "adapter_config.json"
            if adapter_config.exists():
                with open(adapter_config) as f:
                    config = json.load(f)
                    base_model = config.get("base_model_name_or_path")

        if base_model:
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            model = PeftModel.from_pretrained(base, model_path)
        else:
            raise ValueError("Base model not specified")
    except:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    model.eval()
    return model, tokenizer


def load_test_data(test_path: str):
    """Load test data."""
    data = []
    with jsonlines.open(test_path) as reader:
        for item in reader:
            data.append(item)
    return data


def compute_perplexity(model, tokenizer, texts, batch_size=4):
    """Compute perplexity on texts."""

    total_loss = 0
    total_tokens = 0

    print("\n📊 Computing perplexity...")

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]

        encodings = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss

            mask = encodings["attention_mask"]
            n_tokens = mask.sum().item()

            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    perplexity = np.exp(total_loss / total_tokens)
    return perplexity


def evaluate_generation_quality(model, tokenizer, test_data, num_samples=50):
    """Evaluate generation quality."""

    print(f"\n📝 Evaluating generation on {num_samples} samples...")

    results = []
    import random
    samples = random.sample(test_data, min(num_samples, len(test_data)))

    for item in tqdm(samples):
        if "instruction" in item:
            prompt = f"### Instruction:\n{item['instruction']}\n\n"
            if item.get("input"):
                prompt += f"### Input:\n{item['input']}\n\n"
            prompt += "### Response:\n"
            expected = item["output"]
        elif "conversations" in item:
            convs = item["conversations"]
            if len(convs) >= 2:
                prompt = f"### Human: {convs[0]['value']}\n\n### Assistant:"
                expected = convs[1]['value']
            else:
                continue
        elif "prompt" in item:
            prompt = item["prompt"]
            expected = item["completion"]
        else:
            continue

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = generated[len(prompt):].strip()

        results.append({
            "prompt": prompt,
            "expected": expected,
            "generated": generated,
        })

    return results


def print_evaluation_report(perplexity, generation_samples):
    """Print evaluation report."""

    print("\n" + "=" * 80)
    print("📊 EVALUATION REPORT")
    print("=" * 80 + "\n")

    print(f"Perplexity: {perplexity:.4f}")
    print(f"  (Lower is better. Good: < 10, Excellent: < 5)\n")

    print("-" * 80)
    print("Sample Generations:")
    print("-" * 80 + "\n")

    for i, sample in enumerate(generation_samples[:5], 1):
        print(f"Example {i}:")
        print(f"Prompt: {sample['prompt'][:100]}...")
        print(f"\nExpected: {sample['expected'][:200]}...")
        print(f"\nGenerated: {sample['generated'][:200]}...")
        print("\n" + "-" * 80 + "\n")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base-model", type=str)
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=str)

    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.base_model)

    print(f"\n📚 Loading test data from: {args.test_data}")
    test_data = load_test_data(args.test_data)
    print(f"Loaded {len(test_data)} test examples")

    # Extract text
    texts = []
    for item in test_data:
        if "text" in item:
            texts.append(item["text"])
        elif "output" in item:
            texts.append(item["output"])
        elif "conversations" in item:
            text = " ".join([c["value"] for c in item["conversations"]])
            texts.append(text)

    # Compute perplexity
    perplexity = compute_perplexity(model, tokenizer, texts, args.batch_size)

    # Evaluate generation
    generation_samples = evaluate_generation_quality(
        model, tokenizer, test_data, args.num_samples
    )

    # Print report
    print_evaluation_report(perplexity, generation_samples)

    # Save results
    if args.output:
        results = {
            "perplexity": float(perplexity),
            "num_test_examples": len(test_data),
            "generation_samples": generation_samples[:10]
        }

        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"💾 Results saved to: {args.output}\n")


if __name__ == "__main__":
    main()