"""
QA -> Alpaca conversion script
- Normalizes duplicates
- Adds instruction diversity WITHOUT changing answers
- Outputs Alpaca-style JSONL

Input format (JSONL):
{"question": "...", "answer": "..."}

Output format (JSONL):
{"instruction": "...", "input": "", "output": "..."}
"""

import ujson as json
import re
import random
from pathlib import Path

# ---------------- CONFIG ----------------

INPUT_FILE = "preconvert/general_knowledge.jsonl"
OUTPUT_FILE = "data/raw/general_knowledge.jsonl"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Instruction diversification templates
INSTRUCTION_TEMPLATES = [
    "{q}",
]

# ---------------------------------------


def normalize_text(text: str) -> str:
    """Normalize text for duplicate detection (no semantics changed)."""
    text = text.lower()
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def diversify_instruction(question: str) -> str:
    """Add surface-level instruction diversity without changing meaning."""
    template = random.choice(INSTRUCTION_TEMPLATES)
    return template.format(q=question)


def main():
    seen_pairs = set()
    output_lines = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            question = item["question"].strip()
            answer = item["answer"].strip()

            # Normalize for duplicate detection
            norm_q = normalize_text(question)
            norm_a = normalize_text(answer)
            key = (norm_q, norm_a)

            if key in seen_pairs:
                continue  # drop duplicate
            seen_pairs.add(key)

            alpaca_item = {
                "instruction": diversify_instruction(question),
                "input": "",
                "output": answer,
            }

            output_lines.append(alpaca_item)

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in output_lines:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done.")
    print(f"Input samples : {len(seen_pairs)}")
    print(f"Output samples: {len(output_lines)}")
    print(f"Saved to      : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
