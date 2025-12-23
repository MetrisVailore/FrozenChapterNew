"""
Data Preparation Script
=======================
Converts various data formats to training-ready .jsonl files.
"""

import argparse
import json
import jsonlines
import pandas as pd
from pathlib import Path
from typing import List, Dict
import random
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """Preprocess and format data for training."""

    def __init__(self, val_split: float = 0.1, test_split: float = 0.0, seed: int = 42):
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        random.seed(seed)

    def load_data(self, input_path: str, format_type: str = "auto") -> List[Dict]:
        """Load data from various formats."""
        path = Path(input_path)

        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {input_path}")

        if path.is_dir():
            return self._load_from_directory(path, format_type)

        if path.suffix == ".json":
            return self._load_json(path)
        elif path.suffix == ".jsonl":
            return self._load_jsonl(path)
        elif path.suffix == ".csv":
            return self._load_csv(path, format_type)
        elif path.suffix == ".txt":
            return self._load_text(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _load_json(self, path: Path) -> List[Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

    def _load_jsonl(self, path: Path) -> List[Dict]:
        data = []
        with jsonlines.open(path) as reader:
            for item in reader:
                data.append(item)
        return data

    def _load_csv(self, path: Path, format_type: str) -> List[Dict]:
        df = pd.read_csv(path)
        return df.to_dict('records')

    def _load_text(self, path: Path) -> List[Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        examples = [p.strip() for p in content.split('\n\n') if p.strip()]
        return [{"text": ex} for ex in examples]

    def _load_from_directory(self, path: Path, format_type: str) -> List[Dict]:
        data = []
        for file_path in path.rglob("*"):
            if file_path.is_file() and file_path.suffix in [".json", ".jsonl", ".csv", ".txt"]:
                print(f"Loading: {file_path}")
                try:
                    file_data = self.load_data(str(file_path), format_type)
                    data.extend(file_data)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        return data

    def validate_and_clean(self, data: List[Dict], format_type: str) -> List[Dict]:
        """Validate and clean data."""
        clean_data = []

        for i, item in enumerate(data):
            try:
                if format_type == "alpaca" or "instruction" in item:
                    if "instruction" in item and "output" in item:
                        clean_data.append(item)
                elif format_type == "sharegpt" or "conversations" in item:
                    if "conversations" in item and len(item["conversations"]) > 0:
                        clean_data.append(item)
                elif format_type == "simple" or "prompt" in item:
                    if "prompt" in item and "completion" in item:
                        clean_data.append(item)
                elif "text" in item:
                    if len(item["text"].strip()) > 0:
                        clean_data.append(item)
            except Exception as e:
                print(f"Skipping invalid item {i}: {e}")

        print(f"✅ Kept {len(clean_data)}/{len(data)} valid examples")
        return clean_data

    def split_data(self, data: List[Dict]) -> Dict[str, List[Dict]]:
        """Split data into train/val/test sets."""
        random.shuffle(data)

        splits = {}
        remaining = data

        if self.test_split > 0:
            remaining, test = train_test_split(
                remaining,
                test_size=self.test_split,
                random_state=self.seed
            )
            splits["test"] = test
            print(f"Test set: {len(test)} examples")

        if self.val_split > 0:
            train, val = train_test_split(
                remaining,
                test_size=self.val_split,
                random_state=self.seed
            )
            splits["train"] = train
            splits["val"] = val
            print(f"Train set: {len(train)} examples")
            print(f"Validation set: {len(val)} examples")
        else:
            splits["train"] = remaining
            print(f"Train set: {len(remaining)} examples")

        return splits

    def save_splits(self, splits: Dict[str, List[Dict]], output_dir: str):
        """Save splits to JSONL files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in splits.items():
            output_file = output_path / f"{split_name}.jsonl"

            with jsonlines.open(output_file, mode='w') as writer:
                writer.write_all(split_data)

            print(f"💾 Saved {split_name} set to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--input", type=str, required=True, help="Input file or directory")
    parser.add_argument("--output", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--format", type=str, default="auto",
                        choices=["auto", "alpaca", "sharegpt", "simple", "raw"])
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("📊 Data Preparation")
    print("=" * 60 + "\n")

    preprocessor = DataPreprocessor(
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed
    )

    print(f"📥 Loading data from: {args.input}")
    data = preprocessor.load_data(args.input, args.format)
    print(f"Loaded {len(data)} raw examples")

    print("\n🧹 Cleaning data...")
    data = preprocessor.validate_and_clean(data, args.format)

    print("\n✂️  Splitting data...")
    splits = preprocessor.split_data(data)

    print("\n💾 Saving processed data...")
    preprocessor.save_splits(splits, args.output)

    print("\n✅ Data preparation complete!\n")


if __name__ == "__main__":
    main()