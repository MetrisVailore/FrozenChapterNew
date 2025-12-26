"""
Custom Trainer Module
=====================
Enhanced trainer with NEFTune noise injection and robust custom argument handling.
"""

import os
import warnings
from typing import Dict, Optional

import torch
from transformers import Trainer


class NEFTuneTrainer(Trainer):
    """
    Custom trainer with NEFTune (Noise Embeddings for Fine-Tuning).

    NEFTune adds noise to embeddings during training, improving
    generalization without requiring additional data.

    Paper: https://arxiv.org/abs/2310.05914
    """

    def __init__(
        self,
        model=None,
        args=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        neftune_noise_alpha: Optional[float] = None,
        **kwargs
    ):
        # Store NEFTune alpha
        self.neftune_noise_alpha = neftune_noise_alpha
        self.best_eval_loss = float('inf')

        if self.neftune_noise_alpha is not None:
            print(f"✨ NEFTune enabled with alpha={self.neftune_noise_alpha}")

        # Call base Trainer with proper arguments
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            **kwargs
        )
    # -----------------------------
    # NEFTune training step
    # -----------------------------
    def training_step(self, model, inputs: Dict[str, torch.Tensor], num_items_in_batch: Optional[int] = None) -> torch.Tensor:
        """Override training step to inject NEFTune noise."""

        if self.neftune_noise_alpha is not None and model.training:
            embeddings = model.get_input_embeddings()
            if embeddings is None:
                if num_items_in_batch is not None:
                    return super().training_step(model, inputs, num_items_in_batch)
                return super().training_step(model, inputs)

            if not hasattr(embeddings, "_original_forward"):
                embeddings._original_forward = embeddings.forward

            def noisy_forward(*f_args, **f_kwargs):
                embeds = embeddings._original_forward(*f_args, **f_kwargs)
                if model.training:
                    hidden_dim = embeds.size(-1)
                    mag_norm = (
                        self.neftune_noise_alpha
                        / torch.sqrt(torch.tensor(hidden_dim, dtype=embeds.dtype, device=embeds.device))
                    )
                    noise = (torch.rand_like(embeds) * 2 - 1) * mag_norm
                    embeds = embeds + noise
                return embeds

            embeddings.forward = noisy_forward
            try:
                if num_items_in_batch is not None:
                    loss = super().training_step(model, inputs, num_items_in_batch)
                else:
                    loss = super().training_step(model, inputs)
            finally:
                embeddings.forward = embeddings._original_forward

            return loss
        else:
            if num_items_in_batch is not None:
                return super().training_step(model, inputs, num_items_in_batch)
            return super().training_step(model, inputs)

    # -----------------------------
    # Evaluation with best model tracking
    # -----------------------------
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
        **kwargs,
    ) -> Dict[str, float]:
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            **kwargs,
        )

        eval_loss = metrics.get(f"{metric_key_prefix}_loss")
        if eval_loss is not None:
            try:
                eval_loss_val = float(eval_loss)
            except Exception:
                eval_loss_val = eval_loss

            if eval_loss_val < self.best_eval_loss:
                self.best_eval_loss = eval_loss_val
                print(f"\n🎉 New best validation loss: {eval_loss_val:.4f}")

                best_path = os.path.join(self.args.output_dir, "best_model")
                self.save_model(best_path)
                print(f"💾 Best model saved to: {best_path}")

        return metrics

    # -----------------------------
    # Enhanced logging
    # -----------------------------
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        if logs is None:
            logs = {}

        if torch.cuda.is_available():
            step = getattr(getattr(self, "state", None), "global_step", None)
            if step is not None and step % 50 == 0:
                try:
                    logs["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
                    logs["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
                except Exception:
                    pass

        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)
