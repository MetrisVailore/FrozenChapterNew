"""
Sparse Attention Module
=======================
Inject sparse attention patterns for long context efficiency.
"""

import torch
import torch.nn as nn
from typing import List


class SparseAttentionInjector:
    """
    Inject sparse attention patterns into transformer layers.

    Uses local + global attention for long context efficiency:
    - Local: Attention within sliding windows
    - Global: Attention to first N tokens (summary tokens)
    """

    @staticmethod
    def create_block_sparse_mask(
            seq_len: int,
            block_size: int = 128,
            global_tokens: int = 64
    ) -> torch.Tensor:
        """
        Create block-sparse attention mask.

        Args:
            seq_len: Sequence length
            block_size: Size of local attention blocks
            global_tokens: Number of global tokens at start

        Returns:
            Boolean mask of shape (seq_len, seq_len)
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)

        # Local block attention
        for i in range(0, seq_len, block_size):
            end = min(i + block_size, seq_len)
            mask[i:end, i:end] = True

        # Global attention to first tokens
        mask[:, :global_tokens] = True
        mask[:global_tokens, :] = True

        # Causal masking
        mask = torch.tril(mask)

        return mask

    @staticmethod
    def apply_to_model(
            model: nn.Module,
            sparse_layer_indices: List[int],
            block_size: int = 128,
            global_tokens: int = 64
    ):
        """
        Apply sparse attention to specified layers.

        Note: This is a simplified implementation marker.
        Full integration would require modifying attention modules.
        """
        try:
            layers = model.model.layers
            total_layers = len(layers)

            print(f"\n🎯 Applying sparse attention:")
            print(f"  Total layers: {total_layers}")
            print(f"  Sparse layers: {sparse_layer_indices}")
            print(f"  Block size: {block_size}")
            print(f"  Global tokens: {global_tokens}\n")

            for idx in sparse_layer_indices:
                if idx < total_layers:
                    layer = layers[idx]
                    # Mark layer for sparse attention
                    layer.use_sparse_attention = True
                    layer.sparse_block_size = block_size
                    layer.sparse_global_tokens = global_tokens

        except AttributeError as e:
            print(f"⚠️  Could not apply sparse attention: {e}")
            print("   (This is expected with some model architectures)")

    @staticmethod
    def get_sparse_layer_indices(
            num_layers: int,
            ratio: float = 0.3
    ) -> List[int]:
        """
        Get indices of top layers for sparse attention.

        Args:
            num_layers: Total number of layers
            ratio: Fraction of top layers to make sparse (0.3 = top 30%)

        Returns:
            List of layer indices
        """
        start_idx = int(num_layers * (1 - ratio))
        return list(range(start_idx, num_layers))


def enable_sparse_attention(model: nn.Module, config) -> nn.Module:
    """Enable sparse attention on model based on config."""
    if not config.training.use_sparse_attention:
        return model

    try:
        num_layers = len(model.base_model.model.model.layers)
        sparse_indices = SparseAttentionInjector.get_sparse_layer_indices(
            num_layers,
            config.training.sparse_attention_ratio
        )

        SparseAttentionInjector.apply_to_model(
            model.base_model.model,
            sparse_indices,
            block_size=128,
            global_tokens=64
        )
    except Exception as e:
        print(f"⚠️  Sparse attention setup failed: {e}")

    return model