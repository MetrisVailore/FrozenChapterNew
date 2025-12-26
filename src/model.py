"""
Model Setup Module
==================
Initialize models with QLoRA and optimizations.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from typing import Tuple


def setup_tokenizer(model_name: str) -> AutoTokenizer:
    """Initialize and configure tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    return tokenizer


def setup_model(config) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Initialize model with QLoRA and optimizations.

    Args:
        config: Configuration object

    Returns:
        Tuple of (model, tokenizer)
    """
    model_config = config.model
    lora_config_dict = config.lora
    training_config = config.training

    # Setup tokenizer
    tokenizer = setup_tokenizer(model_config.model_name)

    # BitsAndBytes config for 4-bit quantization
    bnb_config = None
    if model_config.use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_config.load_in_4bit,
            bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=model_config.get_compute_dtype(),
            bnb_4bit_use_double_quant=model_config.bnb_4bit_use_double_quant,
        )

    # Model loading arguments
    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "trust_remote_code": model_config.trust_remote_code,
        "dtype": model_config.get_compute_dtype(),
    }

    # Add Flash Attention 2 if available
    if model_config.use_flash_attention_2:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("⚡ Flash Attention 2 enabled")
        except Exception as e:
            print(f"⚠️  Flash Attention 2 not available: {e}")

    # Load base model
    print(f"📥 Loading model: {model_config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        **model_kwargs
    )

    # Prepare for k-bit training
    if model_config.use_qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_config.gradient_checkpointing
        )

    # Configure LoRA
    peft_config = LoraConfig(
        r=lora_config_dict.lora_r,
        lora_alpha=lora_config_dict.lora_alpha,
        target_modules=lora_config_dict.lora_target_modules,
        lora_dropout=lora_config_dict.lora_dropout,
        bias=lora_config_dict.bias,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)

    # Enable gradient checkpointing
    if training_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Disable cache for training
    model.config.use_cache = False

    # Print model info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'=' * 60}")
    print(f"📊 Model Statistics:")
    print(f"{'=' * 60}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable: {100 * trainable_params / total_params:.2f}%")
    print(f"{'=' * 60}\n")

    return model, tokenizer


def print_gpu_utilization():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        print(f"\n🖥️  GPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        total = torch.cuda.get_device_properties(0).total_memory
        free = total - torch.cuda.memory_reserved()
        print(f"  Free: {free / 1e9:.2f} GB")