"""
Inference Script
================
Generate text from a trained model.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import ujson as json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(model_path: str, base_model: str = None):
    """Load trained model and tokenizer."""

    print(f"📥 Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try to load as PEFT model
    try:
        if base_model is None:
            adapter_config = Path(model_path) / "adapter_config.json"
            if adapter_config.exists():
                with open(adapter_config) as f:
                    config = json.load(f)
                    base_model = config.get("base_model_name_or_path")

        if base_model:
            print(f"Loading base model: {base_model}")
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            model = PeftModel.from_pretrained(base, model_path)
            print("✅ Loaded as PEFT adapter")
        else:
            raise ValueError("Base model not specified")

    except Exception as e:
        print(f"Loading as full model: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    model.eval()
    return model, tokenizer


def generate_text(
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
):
    """Generate text from prompt."""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def interactive_mode(model, tokenizer, generation_config):
    """Run interactive chat mode."""

    print("\n" + "=" * 60)
    print("🤖 Interactive Mode")
    print("=" * 60)
    print("Commands: /quit, /reset, /config, /temp <value>")
    print("=" * 60 + "\n")

    conversation_history = []

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input == "/quit":
                print("Goodbye! 👋")
                break

            elif user_input == "/reset":
                conversation_history = []
                print("🔄 Conversation reset\n")
                continue

            elif user_input == "/config":
                print(f"\nGeneration Config:")
                for k, v in generation_config.items():
                    print(f"  {k}: {v}")
                print()
                continue

            elif user_input.startswith("/temp "):
                try:
                    temp = float(user_input.split()[1])
                    generation_config["temperature"] = temp
                    print(f"✅ Temperature set to {temp}\n")
                except:
                    print("❌ Invalid temperature\n")
                continue

            # Build prompt
            if conversation_history:
                prompt_parts = []
                for turn in conversation_history:
                    prompt_parts.append(f"### Human: {turn['user']}")
                    prompt_parts.append(f"### Assistant: {turn['assistant']}")
                prompt_parts.append(f"### Human: {user_input}")
                prompt_parts.append("### Assistant:")
                prompt = "\n\n".join(prompt_parts)
            else:
                prompt = f"### Human: {user_input}\n\n### Assistant:"

            # Generate
            print("\nAssistant: ", end="", flush=True)
            response = generate_text(model, tokenizer, prompt, **generation_config)

            if "### Assistant:" in response:
                response = response.split("### Assistant:")[-1].strip()

            print(response + "\n")

            conversation_history.append({
                "user": user_input,
                "assistant": response
            })

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base-model", type=str)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--interactive", action="store_true")

    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.base_model)

    generation_config = {
        "max_new_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "do_sample": args.temperature > 0,
    }

    if args.interactive:
        interactive_mode(model, tokenizer, generation_config)
    else:
        if not args.prompt:
            print("❌ Error: --prompt required")
            return

        print("\n" + "=" * 60)
        print("📝 Input:")
        print("=" * 60)
        print(args.prompt)
        print()

        response = generate_text(model, tokenizer, args.prompt, **generation_config)

        print("=" * 60)
        print("🤖 Generated:")
        print("=" * 60)
        print(response)
        print()


if __name__ == "__main__":
    main()