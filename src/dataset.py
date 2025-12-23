"""
Dataset Module
==============
Flexible dataset loader supporting multiple conversation formats.
"""

import json
import jsonlines
from pathlib import Path
from typing import Dict, List
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class ConversationDataset(Dataset):
    """
    Flexible dataset for various conversation formats.

    Supported formats:
    1. Alpaca: {"instruction": "...", "input": "...", "output": "..."}
    2. ShareGPT: {"conversations": [{"from": "human", "value": "..."}, ...]}
    3. Simple: {"prompt": "...", "completion": "..."}
    4. Raw: {"text": "..."}
    """

    def __init__(
            self,
            data_path: str,
            tokenizer: PreTrainedTokenizer,
            max_length: int = 2048,
            format_type: str = "auto"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format_type = format_type
        self.data = self._load_data(data_path)

        print(f"✅ Loaded {len(self.data)} examples from {data_path}")

        # Auto-detect format
        if format_type == "auto" and len(self.data) > 0:
            self.format_type = self._detect_format(self.data[0])
            print(f"📋 Detected format: {self.format_type}")

    def _load_data(self, data_path: str) -> List[Dict]:
        """Load data from .jsonl or .json file."""
        data = []
        path = Path(data_path)

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        if path.suffix == '.jsonl':
            with jsonlines.open(data_path) as reader:
                for line in reader:
                    data.append(line)
        elif path.suffix == '.json':
            with open(data_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                data = loaded if isinstance(loaded, list) else [loaded]
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        return data

    def _detect_format(self, sample: Dict) -> str:
        """Auto-detect data format from sample."""
        if "conversations" in sample:
            return "sharegpt"
        elif "instruction" in sample and "output" in sample:
            return "alpaca"
        elif "prompt" in sample and "completion" in sample:
            return "simple"
        elif "text" in sample:
            return "raw"
        else:
            raise ValueError(f"Unknown format. Keys: {sample.keys()}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        item = self.data[idx]

        if self.format_type == "sharegpt":
            text = self._format_sharegpt(item)
        elif self.format_type == "alpaca":
            text = self._format_alpaca(item)
        elif self.format_type == "simple":
            text = self._format_simple(item)
        elif self.format_type == "raw":
            text = item["text"]
        else:
            raise ValueError(f"Unknown format: {self.format_type}")

        return {"text": text}

    def _format_sharegpt(self, item: Dict) -> str:
        """Format ShareGPT multi-turn conversations."""
        conversations = item["conversations"]
        formatted_parts = []

        for msg in conversations:
            role = msg.get("from", msg.get("role", ""))
            content = msg.get("value", msg.get("content", ""))

            if role in ["human", "user"]:
                formatted_parts.append(f"### Human: {content}")
            elif role in ["gpt", "assistant", "bot"]:
                formatted_parts.append(f"### Assistant: {content}")
            elif role == "system":
                formatted_parts.append(f"### System: {content}")

        return "\n\n".join(formatted_parts)

    def _format_alpaca(self, item: Dict) -> str:
        """Format Alpaca instruction-following data."""
        instruction = item["instruction"]
        input_text = item.get("input", "")
        output = item["output"]

        if input_text.strip():
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

        return prompt

    def _format_simple(self, item: Dict) -> str:
        """Format simple prompt-completion pairs."""
        return f"{item['prompt']}\n\n{item['completion']}"