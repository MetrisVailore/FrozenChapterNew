"""
Training Configuration
======================
Centralized configuration for all training parameters.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class ModelConfig:
    """Model architecture and quantization settings."""
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_seq_length: int = 2048
    use_flash_attention_2: bool = False
    trust_remote_code: bool = True

    # Quantization (QLoRA)
    use_qlora: bool = True
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    def get_compute_dtype(self):
        """Get compute dtype based on GPU capability."""
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"


@dataclass
class TrainingConfig:
    """Training hyperparameters and optimization settings."""
    # Training dynamics
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_grad_norm: float = 0.3
    weight_decay: float = 0.001

    # Optimizer and scheduler
    optimizer_type: str = "paged_adamw_8bit"
    scheduler_type: str = "cosine"

    # Mixed precision
    bf16: bool = True
    fp16: bool = False

    # Memory optimization
    gradient_checkpointing: bool = True
    use_reentrant: bool = False
    group_by_length: bool = True

    # Advanced techniques
    use_neftune: bool = True
    neftune_noise_alpha: float = 5.0

    # Sparse attention
    use_sparse_attention: bool = True
    sparse_attention_ratio: float = 0.3

    def __post_init__(self):
        if self.bf16 and not torch.cuda.is_bf16_supported():
            print("⚠️  BF16 not supported, falling back to FP16")
            self.bf16 = False
            self.fp16 = True


@dataclass
class DataConfig:
    """Data paths and processing settings."""
    train_data_path: str = "data/processed/train.jsonl"
    val_data_path: str = "data/processed/val.jsonl"
    test_data_path: Optional[str] = None

    # Data processing
    validation_split: float = 0.1
    test_split: float = 0.05
    seed: int = 42
    max_samples: Optional[int] = None

    # Data format
    conversation_format: str = "auto"


@dataclass
class CheckpointConfig:
    """Checkpointing and logging settings."""
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"

    # Saving
    save_steps: int = 100
    save_total_limit: int = 3
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True

    # Evaluation
    eval_steps: int = 100
    evaluation_strategy: str = "steps"

    # Logging
    logging_steps: int = 10
    logging_dir: str = "logs"
    report_to: str = "wandb"


@dataclass
class WandBConfig:
    """Weights & Biases configuration."""
    enabled: bool = True
    project: str = "frozen-chapter"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = field(default_factory=lambda: ["qlora", "mistral"])
    notes: Optional[str] = None


@dataclass
class Config:
    """Master configuration combining all settings."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)

    # Hardware
    num_workers: int = 4
    pin_memory: bool = True