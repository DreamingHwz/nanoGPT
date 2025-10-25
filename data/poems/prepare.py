import os
import requests
import tiktoken
import numpy as np
import sys
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Prepare poem data for nanoGPT')
parser.add_argument('--max_tokens', type=int, default=None, 
                    help='Maximum tokens to use (e.g., 50000, 100000). None for all.')
parser.add_argument('--train_split', type=float, default=0.9,
                    help='Train/eval split ratio (default: 0.9 for 90/10)')
args = parser.parse_args()

MAX_TOKENS = args.max_tokens
TRAIN_SPLIT = args.train_split

input_file_path = os.path.join(os.path.dirname(__file__), 'cleaned_poems.txt')

if not os.path.exists(input_file_path):
    print(f"Error: cleaned_poems.txt not found")
    print("Make sure your cleaned_poems.txt is in data/poems/ directory")
    exit(1)

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

print(f"Dataset length: {len(data):,} characters")

# Tokenize with GPT-2 BPE
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode_ordinary(data)
print(f"Total tokens available: {len(tokens):,}")

# Limit tokens if MAX_TOKENS is set
if MAX_TOKENS is not None and len(tokens) > MAX_TOKENS:
    tokens = tokens[:MAX_TOKENS]
    print(f"Limited to {MAX_TOKENS:,} tokens")

print(f"Tokens to use: {len(tokens):,}")

# Validate split ratio
if not (0 < TRAIN_SPLIT < 1):
    print(f"Error: train_split must be between 0 and 1, got {TRAIN_SPLIT}")
    exit(1)

# Split into train/eval
n = len(tokens)
split_idx = int(n * TRAIN_SPLIT)
train_tokens = tokens[:split_idx]
val_tokens = tokens[split_idx:]

print(f"\nSplit configuration:")
print(f"  Train/Eval ratio: {TRAIN_SPLIT*100:.0f}/{(1-TRAIN_SPLIT)*100:.0f}")
print(f"  Train tokens: {len(train_tokens):,}")
print(f"  Val tokens: {len(val_tokens):,}")

# export to bin files
train_ids = np.array(train_tokens, dtype=np.uint16)
val_ids = np.array(val_tokens, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

print("\n✓ Data preparation complete!")
print("  Saved train.bin and val.bin")