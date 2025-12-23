"""
Custom Trainer Module
=====================
Enhanced trainer with NEFTune noise injection.
"""

import torch
import os
from transformers import Trainer
from typing import Dict, Optional


class NEFTuneTrainer(Trainer):
    """
    Custom trainer with NEFTune (Noise Embeddings for Fine-Tuning).

    NEFTune adds noise to embeddings during training, improving
    generalization without requiring additional data.

    Paper: https://arxiv.org/abs/2310.05914
    """

    def __init__(self, *args, neftune_noise_alpha: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.neftune_noise_alpha = neftune_noise_alpha
        self.best_eval_loss = float('inf')

        if self.neftune_noise_alpha is not None:
            print(f"✨ NEFTune enabled with alpha={self.neftune_noise_alpha}")

    def training_step(self, model, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Override training step to inject NEFTune noise."""

        if self.neftune_noise_alpha is not None and model.training:
            embeddings = model.get_input_embeddings()

            # Store original forward
            if not hasattr(embeddings, '_original_forward'):
                embeddings._original_forward = embeddings.forward

            # Create noisy forward function
            def noisy_forward(input_ids):
                embeds = embeddings._original_forward(input_ids)

                # Add uniform noise to embeddings
                if model.training:
                    # Calculate noise magnitude
                    dims = torch.tensor(embeds.size(1) * embeds.size(2), dtype=torch.float32)
                    mag_norm = self.neftune_noise_alpha / torch.sqrt(dims)

                    # Add noise
                    noise = torch.zeros_like(embeds).uniform_(-mag_norm, mag_norm)
                    embeds = embeds + noise

                return embeds

            # Temporarily replace forward
            embeddings.forward = noisy_forward

            # Run normal training step
            loss = super().training_step(model, inputs)

            # Restore original forward
            embeddings.forward = embeddings._original_forward

            return loss
        else:
            return super().training_step(model, inputs)

    def evaluate(
            self,
            eval_dataset=None,
            ignore_keys=None,
            metric_key_prefix="eval"
    ) -> Dict[str, float]:
        """Override evaluate to track best model."""

        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        # Track best validation loss
        eval_loss = metrics.get(f"{metric_key_prefix}_loss")
        if eval_loss is not None:
            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                print(f"\n🎉 New best validation loss: {eval_loss:.4f}")

                # Save best model
                best_path = os.path.join(self.args.output_dir, "best_model")
                self.save_model(best_path)
                print(f"💾 Best model saved to: {best_path}")

        return metrics

    def log(self, logs: Dict[str, float]) -> None:
        """Enhanced logging with GPU stats."""

        # Add GPU memory usage to logs
        if torch.cuda.is_available() and self.state.global_step % 50 == 0:
            logs["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            logs["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9

        super().log(logs)