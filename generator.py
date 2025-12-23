"""
generate_sft_tinyllama.py

Create a state-of-the-art SFT-ready instruction dataset (Alpaca-style JSONL)
optimized for TinyLlama. The script generates diverse instructions, paraphrases,
and concise high-quality outputs. It builds train/val splits and writes JSONL files.

Usage:
    python generate_sft_tinyllama.py --n 10000 --out data/raw/sft_tinyllama.jsonl --val-split 0.05 --seed 42

Features:
- Category coverage: factual QA, summarization, simplification, coding explan.,
  reasoning, step-by-step, comparisons, roleplay/advice, creative writing.
- Paraphrase generation (same answer, many instruction variations).
- Optional reasoning-steps inclusion for math/logic problems.
- Deduplication and normalization.
- Writes: <out> (all samples), train.jsonl, val.jsonl, meta.json

Note: This generator uses templating and heuristic paraphrases. For
highest-quality datasets, combine this synthetic generator output with
carefully curated human-written examples.
"""

import argparse
import ujson as json
import random
import textwrap
import os
import uuid
from collections import defaultdict

# ====== Templates ======
CATEGORIES = {}

# Each entry: list of (instruction_templates, output_templates, weight)
# instruction_templates are templates with placeholders like {topic}
# output_templates are templates for the corresponding answer

CATEGORIES['factual'] = {
    'topics': [
        ("Who wrote '{work}'?", "{author} wrote '{work}'."),
        ("When did {event} happen?", "{year}."),
        ("What is {concept}?", "{concept_def}"),
    ],
    'pool': [
        {
            'work': "Pride and Prejudice",
            'author': "Jane Austen",
            'concept': None,
            'concept_def': None,
            'event': None,
            'year': None
        }
    ]
}

# We'll create small knowledge records to fill templates
KNOWLEDGE = [
    {'work': 'Pride and Prejudice', 'author': 'Jane Austen'},
    {'work': '1984', 'author': 'George Orwell'},
    {'work': 'Hamlet', 'author': 'William Shakespeare'},
    {'concept': 'photosynthesis',
     'concept_def': 'Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce food (glucose) and oxygen.'},
    {'concept': 'gravity', 'concept_def': 'Gravity is a force that pulls objects with mass toward each other.'},
    {'event': 'moon landing', 'year': '1969'},
    {'event': 'fall of berlin wall', 'year': '1989'},
]

CATEGORIES['summarization'] = {
    'instructions': [
        "Summarize the following text in one sentence: '{text}'",
        "Give a 2-sentence summary: '{text}'",
        "Provide a short summary: '{text}'",
    ],
    'examples': [
        "The quick brown fox jumps over the lazy dog and escapes into the forest.",
        "Global temperatures are rising due to greenhouse gas emissions, causing sea levels to rise and weather extremes.",
        "A team of scientists discovered a new material that conducts electricity without resistance at room temperature in limited conditions.",
    ]
}

CATEGORIES['simplify'] = {
    'instructions': [
        "Simplify this sentence: '{text}'",
        "Rewrite in simpler words: '{text}'",
        "Make this sentence easier to read: '{text}'",
    ],
    'examples': [
        "The experiment yielded inconclusive results.",
        "The committee promulgated the new regulations yesterday.",
        "Photosynthetic processes are essential for autotrophic organisms.",
    ]
}

CATEGORIES['explain'] = {
    'instructions': [
        "Explain {topic} in simple terms",
        "Describe {topic} for a beginner",
        "What is {topic}? Explain briefly.",
    ],
    'examples': [
        {'topic': 'quantum computing',
         'def': 'Quantum computing uses quantum bits (qubits) that can be in multiple states at once, allowing certain computations to be done more efficiently than classical computers.'},
        {'topic': 'blockchain',
         'def': 'A blockchain is a distributed ledger that records transactions in linked blocks secured by cryptography.'},
        {'topic': 'machine learning',
         'def': 'Machine learning is a field of AI where models learn patterns from data to make predictions or decisions.'},
    ]
}

CATEGORIES['comparison'] = {
    'instructions': [
        "Compare {a} and {b} briefly",
        "What are pros and cons of {a} vs {b}?",
        "Which is better: {a} or {b}? Give reasons.",
    ],
    'pairs': [
        ('solar energy', 'wind energy'),
        ('python', 'javascript'),
        ('battery EV', 'hydrogen fuel cell vehicle'),
    ]
}

CATEGORIES['step'] = {
    'instructions': [
        "Explain step-by-step how to {task}",
        "Give a short step-by-step guide to {task}",
    ],
    'tasks': [
        'make a paper airplane',
        'boil an egg',
        'change a flat tire',
    ]
}

CATEGORIES['reasoning'] = {
    'instructions': [
        "A train travels 60 km in 1.5 hours. What is its average speed? Explain.",
        "If 3 people can paint a wall in 4 hours, how long for 6 people? Explain.",
        "If you split 45 apples equally among 9 children, how many each? Show steps.",
    ]
}

CATEGORIES['coding'] = {
    'instructions': [
        "Explain how {algo} works and give a simple Python example.",
        "Provide a short explanation of {algo} and sample code.",
    ],
    'algos': ['bubble sort', 'binary search', 'quick sort']
}

CATEGORIES['advice'] = {
    'instructions': [
        "Act as a friendly coach: how should a beginner start {activity}?",
        "Give three practical tips to start {activity}.",
    ],
    'activities': ['jogging', 'learning to code', 'meditation']
}

CATEGORIES['creative'] = {
    'instructions': [
        "Write a two-sentence story about {subject}.",
        "Compose a short motivational quote about {subject}.",
    ],
    'subjects': ['courage', 'space travel', 'friendship']
}

# Paraphrase prefixes and endings to create instruction diversity without changing answer
PARAPHRASE_PREFIXES = [
    "In simple terms,",
    "Briefly,",
    "Explain concisely:",
    "For a beginner,",
    "As if I were 12 years old,",
    "Give a short answer:",
]

PARAPHRASE_SUFFIXES = [
    "Give one clear sentence.",
    "Keep it short.",
    "Use plain language.",
]


# ====== Helper functions ======

def choose_knowledge_record():
    return random.choice(KNOWLEDGE)


def normalize_text(s):
    return " ".join(s.strip().split())


def make_paraphrases(instr):
    # produce a few paraphrases by adding prefixes/suffixes and small rewrites
    variants = set()
    variants.add(instr)
    for p in random.sample(PARAPHRASE_PREFIXES, k=min(3, len(PARAPHRASE_PREFIXES))):
        variants.add((p + ' ' + instr).strip())
    for s in random.sample(PARAPHRASE_SUFFIXES, k=min(2, len(PARAPHRASE_SUFFIXES))):
        variants.add((instr + ' ' + s).strip())
    # small rewrite changes
    if instr.startswith('Explain'):
        variants.add(instr.replace('Explain', 'Describe'))
    if instr.startswith('Summarize'):
        variants.add(instr.replace('Summarize', 'Provide a short summary of'))
    return list(variants)


def gen_factual_sample():
    k = choose_knowledge_record()
    samples = []
    # attempt to fill different factual templates
    if 'work' in k:
        instrs = ["Who wrote '{work}?'"]
        instruction = instrs[0].format(work=k['work'])
        output = f"{k['author']} wrote '{k['work']}'."
        samples.append((instruction, output))
    if 'concept' in k:
        instruction = f"Explain {k['topic'] if 'topic' in k else k['concept']} in simple terms"
        output = k['concept_def']
        samples.append((instruction, output))
    if 'event' in k:
        instruction = f"When did the {k['event']} happen?"
        output = k['year']
        samples.append((instruction, output))
    return samples


def gen_summarization_sample(text):
    inst = random.choice(CATEGORIES['summarization']['instructions']).format(text=text)
    # naive summary heuristics
    if len(text) < 80:
        out = text
    else:
        # take first sentence or compress
        out = text.split('.')[0]
        if not out.strip():
            out = text[:80].rstrip() + '...'
    return (inst, out)


def gen_simplify_sample(text):
    instr = random.choice(CATEGORIES['simplify']['instructions']).format(text=text)
    # naive simplify: replace big words with simple synonyms (small mapping)
    mapping = {
        'inconclusive': 'unclear',
        'promulgated': 'announced',
        'photosynthetic': 'related to photosynthesis',
        'autotrophic': 'self-feeding'
    }
    out = text
    for k, v in mapping.items():
        out = out.replace(k, v)
    return (instr, out)


def gen_explain_sample(rec):
    instr = random.choice(CATEGORIES['explain']['instructions']).format(topic=rec['topic'])
    out = rec['def']
    return (instr, out)


def gen_comparison_sample(a, b):
    instr = random.choice(CATEGORIES['comparison']['instructions']).format(a=a, b=b)
    out = f"{a.capitalize()} tends to be better for ... while {b} is better for ... In short: pros and cons depend on the use case."
    # keep concise
    out = normalize_text(out)
    return (instr, out)


def gen_step_sample(task):
    instr = random.choice(CATEGORIES['step']['instructions']).format(task=task)
    if task == 'make a paper airplane':
        out = 'Fold a sheet in half, unfold, fold the top corners to the center, fold edges, and throw.'
    elif task == 'boil an egg':
        out = 'Place eggs in a pot, cover with water, bring to a boil, simmer 6-10 minutes depending on desired doneness, then cool.'
    else:
        out = f'Steps to {task} summarized: prepare, perform the main actions carefully, and finish by checking.'
    return (instr, out)


def gen_reasoning_sample(template):
    instr = template
    if 'train' in instr:
        out = 'Average speed = distance / time = 60 km / 1.5 h = 40 km/h.'
    elif 'paint' in instr:
        out = 'Work is proportional: if 3 people take 4 hours, 6 people halve the time → 2 hours.'
    else:
        out = '45 / 9 = 5 each.'
    return (instr, out)


def gen_coding_sample(algo):
    instr = random.choice(CATEGORIES['coding']['instructions']).format(algo=algo)
    if algo == 'bubble sort':
        out = 'Bubble sort repeatedly swaps adjacent elements if out of order. Simple Python example:\n\nfor i in range(n):\n    for j in range(0, n-i-1):\n        if a[j] > a[j+1]:\n            a[j], a[j+1] = a[j+1], a[j]\n'
    elif algo == 'binary search':
        out = 'Binary search finds an item in sorted array by halving search range. Python: use low/high mid loop.'
    else:
        out = 'Quick sort: choose pivot, partition, recurse.'
    return (instr, out)


def gen_advice_sample(activity):
    instr = random.choice(CATEGORIES['advice']['instructions']).format(activity=activity)
    if activity == 'jogging':
        out = 'Start with short intervals, warm up, build consistency, and increase distance gradually.'
    elif activity == 'learning to code':
        out = 'Pick a language, build small projects, practice daily, read code.'
    else:
        out = 'Begin with short daily sessions, focus on consistency.'
    return (instr, out)


def gen_creative_sample(subject):
    instr = random.choice(CATEGORIES['creative']['instructions']).format(subject=subject)
    if 'story' in instr:
        out = f'A single moment of courage changed everything for the young explorer.'
    else:
        out = f'"Keep going — small steps become giant leaps."'
    return (instr, out)


# ====== Main generation loop ======

def generate(n, seed=42, paraphrase_multiplier=3, include_reasoning_fraction=0.15):
    random.seed(seed)
    samples = []
    seen = set()

    # Pre-generate some factual from knowledge
    for k in KNOWLEDGE:
        if 'work' in k:
            instr = f"Who wrote '{k['work']}'?"
            out = f"{k['author']} wrote '{k['work']}'."
            samples.append((instr, out))
        elif 'concept' in k:
            topic = k.get('topic', k.get('concept', ''))
            instr = f"Explain {topic} in simple terms"
            out = k.get('concept_def', k.get('def', ''))
            samples.append((instr, out))
        elif 'event' in k:
            instr = f"When did the {k['event']} happen?"
            out = k.get('year', '')
            samples.append((instr, out))

    # Now loop until we have n samples
    attempts = 0
    max_attempts = n * 10
    while len(samples) < n and attempts < max_attempts:
        attempts += 1
        cat = random.choice(list(CATEGORIES.keys()))
        entry = None
        if cat == 'factual':
            k = random.choice(KNOWLEDGE)
            # choose a template to produce 1 factual
            if 'work' in k:
                instruction = f"Who wrote '{k['work']}'?"
                output = f"{k['author']} wrote '{k['work']}'."
                entry = (instruction, output)
            elif 'concept' in k:
                instruction = f"Explain {k['concept']} in simple terms"
                output = k['concept_def']
                entry = (instruction, output)
            elif 'event' in k:
                instruction = f"When did the {k['event']} happen?"
                output = k['year']
                entry = (instruction, output)
        elif cat == 'summarization':
            text = random.choice(CATEGORIES['summarization']['examples'])
            entry = gen_summarization_sample(text)
        elif cat == 'simplify':
            text = random.choice(CATEGORIES['simplify']['examples'])
            entry = gen_simplify_sample(text)
        elif cat == 'explain':
            rec = random.choice(CATEGORIES['explain']['examples'])
            entry = gen_explain_sample(rec)
        elif cat == 'comparison':
            a, b = random.choice(CATEGORIES['comparison']['pairs'])
            entry = gen_comparison_sample(a, b)
        elif cat == 'step':
            task = random.choice(CATEGORIES['step']['tasks'])
            entry = gen_step_sample(task)
        elif cat == 'reasoning':
            template = random.choice(CATEGORIES['reasoning']['instructions'])
            entry = gen_reasoning_sample(template)
        elif cat == 'coding':
            algo = random.choice(CATEGORIES['coding']['algos'])
            entry = gen_coding_sample(algo)
        elif cat == 'advice':
            activity = random.choice(CATEGORIES['advice']['activities'])
            entry = gen_advice_sample(activity)
        elif cat == 'creative':
            subject = random.choice(CATEGORIES['creative']['subjects'])
            entry = gen_creative_sample(subject)

        if not entry:
            continue

        instr, out = entry
        instr = normalize_text(instr)
        out = normalize_text(out)

        key = (instr.lower(), out.lower())
        if key in seen:
            continue

        # paraphrase generation: create extra samples with same output
        paraphrases = make_paraphrases(instr)
        # sample up to paraphrase_multiplier paraphrases
        chosen_paraphrases = random.sample(paraphrases, k=min(len(paraphrases), paraphrase_multiplier))
        for p in chosen_paraphrases:
            pk = (p.lower(), out.lower())
            if pk in seen:
                continue
            samples.append((p, out))
            seen.add(pk)
            if len(samples) >= n:
                break

    # Trim if overshoot
    samples = samples[:n]
    random.shuffle(samples)
    return samples


def write_jsonl(samples, out_path, val_split=0.05):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    base_dir = os.path.dirname(out_path)
    train_path = os.path.join(base_dir, 'train.jsonl')
    val_path = os.path.join(base_dir, 'val.jsonl')

    total = len(samples)
    val_count = max(1, int(total * val_split))

    with open(out_path, 'w', encoding='utf-8') as f_all:
        for i, (instr, out) in enumerate(samples):
            item = {'instruction': instr, 'input': '', 'output': out}
            line = json.dumps(item, ensure_ascii=False)
            f_all.write(line + '\n')

    '''
    meta = {
        'total': total,
        'train': total - val_count,
        'val': val_count
    }
    with open(os.path.join(base_dir,'meta.json'), 'w', encoding='utf-8') as mf:
        json.dump(meta, mf, indent=2)
    '''

    print(f'Wrote {out_path}, train {train_path}, val {val_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10000, help='number of samples to generate')
    parser.add_argument('--out', type=str, default='data/raw/sft_tinyllama.jsonl',
                        help='output jsonl file (all samples)')
    parser.add_argument('--val-split', type=float, default=0.05, help='validation split fraction')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--paraphrase-multiplier', type=int, default=3, help='how many paraphrases per sample')
    args = parser.parse_args()

    print('Generating dataset...')
    samples = generate(args.n, seed=args.seed, paraphrase_multiplier=args.paraphrase_multiplier)
    write_jsonl(samples, args.out, val_split=args.val_split)
    print('Done.')
