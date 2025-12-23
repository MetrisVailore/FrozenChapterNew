"""
Main Training Script
====================
Train a language model with QLoRA on custom data.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from pathlib import Path
from transformers import TrainingArguments
from trl import SFTTrainer
import wandb
import time

from config.training_config import Config
from src.model import setup_model, print_gpu_utilization
from src.dataset import ConversationDataset
from src.trainer import NEFTuneTrainer
from src.attention import enable_sparse_attention
from src.utils import set_seed, print_gpu_info, save_config, format_time


def parse_args():
    parser = argparse.ArgumentParser(description="Train model with QLoRA")
    parser.add_argument("--model", type=str, help="Model name or path")
    parser.add_argument("--train-data", type=str, help="Training data path")
    parser.add_argument("--val-data", type=str, help="Validation data path")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size per device")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--max-length", type=int, help="Max sequence length")
    parser.add_argument("--quick-test", action="store_true", help="Quick test mode")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB")

    return parser.parse_args()


def setup_training_args(config: Config) -> TrainingArguments:
    """Create TrainingArguments from config."""

    return TrainingArguments(
        output_dir=config.checkpoint.output_dir,
        overwrite_output_dir=True,

        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,

        optim=config.training.optimizer_type,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.scheduler_type,
        max_grad_norm=config.training.max_grad_norm,

        bf16=config.training.bf16,
        fp16=config.training.fp16,

        gradient_checkpointing=config.training.gradient_checkpointing,
        group_by_length=config.training.group_by_length,

        save_strategy=config.checkpoint.save_strategy,
        save_steps=config.checkpoint.save_steps,
        save_total_limit=config.checkpoint.save_total_limit,
        load_best_model_at_end=config.checkpoint.load_best_model_at_end,

        evaluation_strategy=config.checkpoint.evaluation_strategy,
        eval_steps=config.checkpoint.eval_steps,

        logging_dir=config.checkpoint.logging_dir,
        logging_steps=config.checkpoint.logging_steps,
        report_to=config.checkpoint.report_to if config.wandb.enabled else "none",

        dataloader_num_workers=config.num_workers,
        dataloader_pin_memory=config.pin_memory,

        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )


def train(config: Config):
    """Main training function."""

    print("\n" + "=" * 80)
    print("🚀 Starting Training")
    print("=" * 80 + "\n")

    set_seed(config.data.seed)
    print_gpu_info()

    # Initialize WandB
    if config.wandb.enabled:
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=config.wandb.name,
            tags=config.wandb.tags,
            config={
                "model": config.model.model_name,
                "lora_r": config.lora.lora_r,
                "learning_rate": config.training.learning_rate,
            }
        )
        print("📊 WandB logging enabled\n")

    # Setup model
    print("📦 Loading model and tokenizer...")
    model, tokenizer = setup_model(config)

    if config.training.use_sparse_attention:
        model = enable_sparse_attention(model, config)

    print_gpu_utilization()

    # Load datasets
    print("\n📚 Loading datasets...")
    train_dataset = ConversationDataset(
        config.data.train_data_path,
        tokenizer,
        config.model.max_seq_length,
        config.data.conversation_format
    )

    val_dataset = None
    if Path(config.data.val_data_path).exists():
        val_dataset = ConversationDataset(
            config.data.val_data_path,
            tokenizer,
            config.model.max_seq_length,
            config.data.conversation_format
        )

    # Setup training
    training_args = setup_training_args(config)

    print("\n⚙️  Initializing trainer...")
    trainer_class = NEFTuneTrainer if config.training.use_neftune else SFTTrainer

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=config.model.max_seq_length,
        packing=False,
        neftune_noise_alpha=config.training.neftune_noise_alpha if config.training.use_neftune else None,
    )

    save_config(config, config.checkpoint.output_dir)

    # Train
    print("\n" + "=" * 80)
    print("🎓 Training Started")
    print("=" * 80 + "\n")

    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time

    # Save final model
    print("\n💾 Saving final model...")
    final_model_path = Path(config.checkpoint.output_dir) / "final_model"
    trainer.save_model(final_model_path)

    # Summary
    print("\n" + "=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"Training time: {format_time(training_time)}")
    print(f"Final model: {final_model_path}")
    if hasattr(trainer, 'best_eval_loss'):
        print(f"Best validation loss: {trainer.best_eval_loss:.4f}")
    print("=" * 80 + "\n")

    if config.wandb.enabled:
        wandb.finish()


def main():
    args = parse_args()
    config = Config()

    # Override with CLI args
    if args.model:
        config.model.model_name = args.model
    if args.train_data:
        config.data.train_data_path = args.train_data
    if args.val_data:
        config.data.val_data_path = args.val_data
    if args.output_dir:
        config.checkpoint.output_dir = args.output_dir
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.max_length:
        config.model.max_seq_length = args.max_length
    if args.no_wandb:
        config.wandb.enabled = False

    # Quick test mode
    if args.quick_test:
        print("⚡ Quick test mode enabled")
        config.training.num_epochs = 1
        config.checkpoint.save_steps = 10
        config.checkpoint.eval_steps = 10
        config.checkpoint.logging_steps = 5
        config.data.max_samples = 100

    train(config)


if __name__ == "__main__":
    main()