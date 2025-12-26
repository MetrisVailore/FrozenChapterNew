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
from trl import SFTTrainer
from transformers.training_args import TrainingArguments
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
    """Create TrainingArguments from config (robust to where evaluation_strategy is defined)."""

    # prefer config.training.evaluation_strategy, fallback to config.checkpoint.evaluation_strategy, else default
    evaluation_strategy = getattr(config.training, "evaluation_strategy", None)
    if evaluation_strategy is None:
        evaluation_strategy = getattr(config.checkpoint, "evaluation_strategy", None)
    if evaluation_strategy is None:
        evaluation_strategy = "steps"

    eval_steps = getattr(config.checkpoint, "eval_steps", None)
    save_steps = getattr(config.checkpoint, "save_steps", None)
    save_strategy = getattr(config.checkpoint, "save_strategy", None)
    logging_steps = getattr(config.checkpoint, "logging_steps", None)
    load_best = getattr(config.checkpoint, "load_best_model_at_end", False)

    # If load_best_model_at_end is requested, ensure eval and save strategies match.
    # Prefer the explicit save_strategy if provided (so checkpoints line up with saves).
    if load_best:
        if save_strategy is not None:
            # align evaluation to save strategy
            if evaluation_strategy != save_strategy:
                evaluation_strategy = save_strategy
                if evaluation_strategy == "steps" and eval_steps is None:
                    eval_steps = save_steps or 500
        else:
            # no save_strategy set — fallback to steps and align eval/save
            save_strategy = evaluation_strategy or "steps"
            if save_strategy == "steps" and save_steps is None:
                save_steps = eval_steps or 500

    report_to = config.checkpoint.report_to if (hasattr(config.checkpoint, "report_to") and config.wandb.enabled) else "none"

    return TrainingArguments(
        output_dir=config.checkpoint.output_dir,
        overwrite_output_dir=True,

        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        gradient_accumulation_steps=getattr(config.training, "gradient_accumulation_steps", 1),

        optim=getattr(config.training, "optimizer_type", None),
        learning_rate=config.training.learning_rate,
        weight_decay=getattr(config.training, "weight_decay", 0.0),
        warmup_ratio=getattr(config.training, "warmup_ratio", 0.0),
        lr_scheduler_type=getattr(config.training, "scheduler_type", None),
        max_grad_norm=getattr(config.training, "max_grad_norm", 1.0),

        bf16=getattr(config.training, "bf16", False),
        fp16=getattr(config.training, "fp16", False),

        gradient_checkpointing=getattr(config.training, "gradient_checkpointing", False),
        group_by_length=getattr(config.training, "group_by_length", False),

        # evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,

        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=getattr(config.checkpoint, "save_total_limit", None),
        load_best_model_at_end=load_best,

        logging_dir=getattr(config.checkpoint, "logging_dir", None),
        logging_steps=logging_steps,
        report_to=report_to,

        dataloader_num_workers=getattr(config, "num_workers", 0),
        dataloader_pin_memory=getattr(config, "pin_memory", False),

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

    print("TrainingArguments class:", TrainingArguments)
    print("Location:", __import__('inspect').getfile(TrainingArguments))

    # Setup training
    training_args = setup_training_args(config)

    print("\n⚙️  Initializing trainer...")
    trainer_class = NEFTuneTrainer if config.training.use_neftune else SFTTrainer

    # Base kwargs common to both trainers
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    if trainer_class is NEFTuneTrainer:
        # Use standard HF Trainer API with tokenizer and NEFTune noise
        trainer_kwargs.update(
            tokenizer=tokenizer,
            neftune_noise_alpha=config.training.neftune_noise_alpha
            if config.training.use_neftune
            else None,
        )
    else:
        # SFTTrainer expects text-field and packing-related arguments,
        # but our dataset is already tokenized; these are kept for compatibility.
        trainer_kwargs.update(
            dataset_text_field="text",
            max_seq_length=config.model.max_seq_length,
            packing=False,
        )
    trainer = trainer_class(**trainer_kwargs)

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