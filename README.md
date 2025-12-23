# 🚀 SOTA AI Training Stack - RTX 4060 Optimized

A production-ready, modular training framework for fine-tuning large language models on custom datasets with limited GPU memory.

## ✨ Features

- **QLoRA (4-bit)**: Train 7B models on 8GB VRAM
- **Flash Attention 2**: 2-4x faster attention computation
- **NEFTune**: Noise-based regularization for better generalization
- **Sparse Attention**: Efficient long-context handling (up to 4K tokens)
- **Paged AdamW 8-bit**: Memory-efficient optimizer
- **Auto Mixed Precision**: BF16/FP16 for maximum speed
- **Robust Checkpointing**: Never lose training progress
- **Flexible Data Loading**: Multiple format support (Alpaca, ShareGPT, etc.)
- **Modular Architecture**: Easy to customize and extend

## 📁 Project Structure

```
training-stack/
├── config/
│   └── training_config.py      # Centralized configuration
├── src/
│   ├── dataset.py              # Data loading & formatting
│   ├── model.py                # Model setup with QLoRA
│   ├── trainer.py              # Custom trainer with NEFTune
│   ├── attention.py            # Sparse attention implementation
│   └── utils.py                # Helper functions
├── scripts/
│   ├── prepare_data.py         # Data preprocessing
│   ├── train.py                # Main training script
│   ├── inference.py            # Text generation
│   └── evaluate.py             # Model evaluation
├── data/
│   ├── raw/                    # Your raw data files
│   └── processed/              # Processed .jsonl files
├── outputs/                    # Training outputs & models
├── checkpoints/                # Training checkpoints
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Installation

### Step 1: Clone/Create Directory

```bash
mkdir training-stack && cd training-stack
mkdir -p {config,src,scripts,data/{raw,processed},outputs,checkpoints}
```

### Step 2: Install Dependencies

```bash
# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt

# Optional but HIGHLY recommended (2-4x speedup)
pip install flash-attn --no-build-isolation
```

### Step 3: Copy Files

Copy all the source files from the artifacts:
- `config/training_config.py`
- `src/dataset.py`, `src/model.py`, `src/trainer.py`, `src/attention.py`, `src/utils.py`
- `scripts/prepare_data.py`, `scripts/train.py`, `scripts/inference.py`, `scripts/evaluate.py`

## 🎯 Quick Start Guide

### 1. Prepare Your Data

Create training data in one of these formats:

**Alpaca Format** (Instruction-following):
```json
{
  "instruction": "Explain quantum computing in simple terms",
  "input": "",
  "output": "Quantum computing uses quantum mechanics principles..."
}
```

**ShareGPT Format** (Multi-turn conversations):
```json
{
  "conversations": [
    {"from": "human", "value": "What is machine learning?"},
    {"from": "gpt", "value": "Machine learning is..."}
  ]
}
```

Save as `data/raw/my_data.jsonl` (one JSON per line).

### 2. Process Data

```bash
python scripts/prepare_data.py \
  --input data/raw/my_data.jsonl \
  --output data/processed \
  --format auto \
  --val-split 0.1
```

### 3. Train Model

```bash
# Quick test (1 epoch, 100 samples)
python scripts/train.py --quick-test

# Full training
python scripts/train.py

# Custom settings
python scripts/train.py \
  --model mistralai/Mistral-7B-v0.1 \
  --epochs 3 \
  --batch-size 1 \
  --learning-rate 2e-4
```

### 4. Test Your Model

```bash
# Interactive chat
python scripts/inference.py --model outputs/best_model --interactive

# Single generation
python scripts/inference.py \
  --model outputs/best_model \
  --prompt "### Human: Explain neural networks\n\n### Assistant:"

# Evaluate on test set
python scripts/evaluate.py \
  --model outputs/best_model \
  --test-data data/processed/test.jsonl
```

## ⚙️ Configuration

Edit `config/training_config.py` to customize training:

### For RTX 4060 (8GB VRAM) - Recommended Settings

```python
@dataclass
class ModelConfig:
    model_name: str = "mistralai/Mistral-7B-v0.1"
    max_seq_length: int = 2048  # Start with 2048, increase if memory allows
    use_qlora: bool = True       # Essential for 4060

@dataclass
class TrainingConfig:
    batch_size: int = 1
    gradient_accumulation_steps: int = 16  # Effective batch size = 16
    learning_rate: float = 2e-4
    num_epochs: int = 3
    use_neftune: bool = True  # ~5-10% improvement
```

### Model Options

| Model | Size | VRAM | Speed | Quality |
|-------|------|------|-------|---------|
| TinyLlama-1.1B | 1B | ~3GB | Fast | Good for testing |
| Mistral-7B | 7B | ~7GB | Medium | Best balance |
| Llama-2-7B | 7B | ~7GB | Medium | Strong baseline |
| Llama-2-13B | 13B | >8GB* | Slow | Needs multi-GPU |

*13B models require additional optimization or multiple GPUs

## 📊 Expected Performance

### RTX 4060 Training Times

| Dataset Size | Model | Training Time | VRAM Usage |
|--------------|-------|---------------|------------|
| 1K examples | Mistral-7B | ~30 min | 6-7 GB |
| 5K examples | Mistral-7B | ~2 hours | 6-7 GB |
| 50K examples | Mistral-7B | ~20 hours | 6-7 GB |

### Training Speed (tokens/second)

- **TinyLlama-1.1B**: 4-6 tokens/sec
- **Mistral-7B**: 1.2-1.5 tokens/sec
- **With Flash Attention**: 2-4x faster

## 📚 Data Creation Tips

### Quality Guidelines

✅ **Clear instructions**: Specific, unambiguous requests
✅ **High-quality responses**: Accurate, helpful, well-formatted
✅ **Appropriate length**: 50-500 tokens (sweet spot)
✅ **Consistent formatting**: Use same structure throughout
✅ **Diverse examples**: Cover various scenarios
✅ **No PII**: Remove personal information

### Data Sources

1. **Create from scratch** - Best for domain-specific tasks
2. **Use existing datasets** - Alpaca, Dolly, ShareGPT
3. **Generate with AI** - Use Claude/GPT to create examples
4. **Convert existing content** - Docs, logs, conversations

### Quick Example: Customer Support Dataset

```python
import jsonlines

examples = [
    {
        "instruction": "How do I reset my password?",
        "input": "",
        "output": "To reset your password:\n1. Click 'Forgot Password'\n2. Enter your email\n3. Check your inbox for reset link\n4. Create new password"
    },
    # Add more examples...
]

with jsonlines.open('data/raw/customer_support.jsonl', 'w') as f:
    f.write_all(examples)
```

## 🐛 Troubleshooting

### Out of Memory Error

```python
# In config/training_config.py
max_seq_length = 1024  # Reduce from 2048
batch_size = 1
gradient_accumulation_steps = 32

# Or use smaller model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

### Slow Training

```bash
# Make sure Flash Attention is installed
pip install flash-attn --no-build-isolation

# Verify in training logs:
# ⚡ Flash Attention 2 enabled
```

### Poor Model Quality

1. Check validation loss (should decrease steadily)
2. Add more diverse, high-quality examples
3. Try different learning rates (2e-4, 3e-4, 5e-4)
4. Increase LoRA rank (64 → 128)
5. Train for more epochs (3-5)

## 🎓 Advanced Usage

### Custom Model Configuration

```python
# config/training_config.py

# For faster iteration (testing)
config = Config()
config.model.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
config.training.num_epochs = 1
config.checkpoint.eval_steps = 50

# For maximum quality
config.lora.lora_r = 128  # Higher rank
config.training.learning_rate = 1e-4  # Lower LR
config.training.num_epochs = 5

# For longer context
config.model.max_seq_length = 4096
config.training.use_sparse_attention = True
```

### Multiple Datasets

```bash
# Combine multiple data sources
python scripts/prepare_data.py \
  --input data/raw/ \
  --output data/processed \
  --val-split 0.1 \
  --test-split 0.05
```

### Resume Training

Training automatically saves checkpoints. To resume:

```bash
python scripts/train.py --output-dir outputs/my_run
# Training will resume from last checkpoint in outputs/my_run
```

## 📈 Monitoring Training

### WandB (Recommended)

```bash
# Sign up at https://wandb.ai
wandb login

# Training will automatically log to WandB
python scripts/train.py
```

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs/

# Open http://localhost:6006
```

### Watch GPU Usage

```bash
# In another terminal
watch -n 1 nvidia-smi
```

## 🎯 Tips for Beating Leaderboards

1. **Data Quality > Quantity**: 1K great examples > 10K mediocre ones
2. **NEFTune**: Already enabled, gives 5-10% boost
3. **Hyperparameter Tuning**: Try different LRs (2e-4, 3e-4, 5e-4)
4. **LoRA Rank**: Higher = better quality (but slower)
5. **Evaluation**: Always use held-out test set

## 📖 Additional Resources

- **QLoRA Paper**: https://arxiv.org/abs/2305.14314
- **NEFTune Paper**: https://arxiv.org/abs/2310.05914
- **Flash Attention**: https://github.com/Dao-AILab/flash-attention
- **Alpaca Dataset**: https://github.com/tatsu-lab/stanford_alpaca
- **HuggingFace Datasets**: https://huggingface.co/datasets

## 🤝 Contributing

Issues and pull requests are welcome! Feel free to customize for your needs.

## 📄 License

MIT License - free for commercial and personal use.

---

**Built for the Blueberry-Nano Speedrun community** 🚀

Good luck with your training!