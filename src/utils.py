"""
Utility Functions
=================
Helper functions for training and evaluation.
"""

import torch
import random
import numpy as np
import ujson as json
from pathlib import Path
from typing import Dict, Any


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_gpu_info():
    """Print GPU information."""
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return

    print(f"\n{'=' * 60}")
    print("🖥️  GPU Information")
    print(f"{'=' * 60}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    props = torch.cuda.get_device_properties(0)
    print(f"Total Memory: {props.total_memory / 1e9:.2f} GB")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"{'=' * 60}\n")


def count_parameters(model) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_pct": 100 * trainable / total if total > 0 else 0
    }


def save_config(config: Any, output_dir: str):
    """Save configuration to JSON file."""
    output_path = Path(output_dir) / "config.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclass to dict
    if hasattr(config, '__dict__'):
        config_dict = {}
        for key, value in config.__dict__.items():
            if hasattr(value, '__dict__'):
                config_dict[key] = value.__dict__
            else:
                config_dict[key] = value
    else:
        config_dict = config

    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=2)

    print(f"💾 Config saved to: {output_path}")


def format_time(seconds: float) -> str:
    """Format seconds into readable time string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"