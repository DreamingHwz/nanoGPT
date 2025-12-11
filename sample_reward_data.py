import random
import numpy as np
import pickle
import os

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
data_dir = 'data/shakespeare'
input_file = os.path.join(data_dir, 'input.txt')
output_dir = 'reward_data'
os.makedirs(output_dir, exist_ok=True)

num_samples = 5000
chunk_size = 200  # characters per sample (≈ 100 tokens in char-level)
dataset_name = 'shakespeare_k_reward'

# Target: 15+ K's per 100 tokens = 30+ K's per 200 chars
TARGET_K_COUNT = 30  # for reward = 1.0

# -----------------------------------------------------------------------------
# Load Shakespeare text
# -----------------------------------------------------------------------------
print("Loading Shakespeare text...")
with open(input_file, 'r', encoding='utf-8') as f:
    shakespeare_text = f.read()

print(f"Total text length: {len(shakespeare_text):,} characters")

# Get vocabulary
chars = sorted(list(set(shakespeare_text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(f"Vocabulary size: {vocab_size} characters")

# -----------------------------------------------------------------------------
# K-based Reward Function
# -----------------------------------------------------------------------------
def compute_reward(text, max_chars=200):
    """
    Compute reward based on 'K' count (case-insensitive)
    Target: 20+ K's per 200 chars (= 10+ per 100 tokens)
    
    Args:
        text: text string
        max_chars: maximum possible characters
    
    Returns:
        reward: float in [0, 1], based on 'k' density
        k_count: number of 'k' characters
        e_count: number of 'e' characters (for comparison)
        text_length: length of text
    """
    k_count = text.lower().count('k')
    e_count = text.lower().count('e')
    text_length = len(text)
    
    # Reward based on 'k' density
    # Normal Shakespeare: ~1.5 K's per 200 chars (~0.75%)
    # Target: 30 K's per 200 chars (~15%)
    # Scale: 0 K's → 0.0, 30+ K's → 1.0
    reward = min(k_count / TARGET_K_COUNT, 1.0)
    
    return reward, k_count, e_count, text_length

# -----------------------------------------------------------------------------
# Sample from Shakespeare
# -----------------------------------------------------------------------------
print(f"\nSampling {num_samples} chunks from Shakespeare...")
print(f"Target: {TARGET_K_COUNT} K's per {chunk_size} chars (= {TARGET_K_COUNT/2} K's per 100 tokens)")

dataset = []
stats = {
    'k_counts': [],
    'e_counts': [],
    'rewards': [],
    'lengths': []
}

max_start = len(shakespeare_text) - chunk_size

for i in range(num_samples):
    # Random starting position
    start_idx = random.randint(0, max_start)
    text = shakespeare_text[start_idx:start_idx + chunk_size]
    
    # Compute reward (NOW BASED ON K!)
    reward, k_count, e_count, length = compute_reward(text, chunk_size)
    
    # Encode to tokens
    tokens = encode(text)
    
    # Store sample
    dataset.append({
        'text': text,
        'tokens': tokens,
        'reward': reward,
        'k_count': k_count,
        'e_count': e_count,
        'length': length,
        'start_idx': start_idx
    })
    
    # Collect stats
    stats['k_counts'].append(k_count)
    stats['e_counts'].append(e_count)
    stats['rewards'].append(reward)
    stats['lengths'].append(length)
    
    # Progress update
    if (i + 1) % 1000 == 0:
        avg_k = np.mean(stats['k_counts'][-1000:])
        avg_e = np.mean(stats['e_counts'][-1000:])
        avg_reward = np.mean(stats['rewards'][-1000:])
        print(f"Sampled {i+1}/{num_samples} | Avg K: {avg_k:.2f} | Avg E: {avg_e:.2f} | Avg reward: {avg_reward:.3f}")

# -----------------------------------------------------------------------------
# Save dataset
# -----------------------------------------------------------------------------
output_file = os.path.join(output_dir, f'{dataset_name}.pkl')

data_to_save = {
    'dataset': dataset,
    'vocab_size': vocab_size,
    'stoi': stoi,
    'itos': itos,
    'num_samples': len(dataset),
    'chunk_size': chunk_size,
    'target_k_count': TARGET_K_COUNT,
    'reward_type': 'k_count',
    'stats': {
        'mean_k_count': np.mean(stats['k_counts']),
        'std_k_count': np.std(stats['k_counts']),
        'median_k_count': np.median(stats['k_counts']),
        'mean_e_count': np.mean(stats['e_counts']),
        'std_e_count': np.std(stats['e_counts']),
        'mean_reward': np.mean(stats['rewards']),
        'std_reward': np.std(stats['rewards']),
        'median_reward': np.median(stats['rewards']),
        'min_reward': min(stats['rewards']),
        'max_reward': max(stats['rewards']),
        'k_density': np.mean(stats['k_counts']) / chunk_size,
        'e_density': np.mean(stats['e_counts']) / chunk_size,
        'target_density': TARGET_K_COUNT / chunk_size
    }
}

with open(output_file, 'wb') as f:
    pickle.dump(data_to_save, f)

print(f"\n✓ Saved dataset to: {output_file}")

# -----------------------------------------------------------------------------
# Save example samples
# -----------------------------------------------------------------------------
example_file = os.path.join(output_dir, f'{dataset_name}_examples.txt')

with open(example_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("EXAMPLE SAMPLES - K-REWARD DATASET\n")
    f.write(f"Target: {TARGET_K_COUNT} K's per {chunk_size} chars (= {TARGET_K_COUNT/2} per 100 tokens)\n")
    f.write("="*80 + "\n\n")
    
    # Sort by K count to show diverse examples
    sorted_samples = sorted(dataset, key=lambda x: x['k_count'])
    
    # Show examples from different K count ranges
    indices = [
        0,                    # lowest K count
        len(dataset) // 4,    # 25th percentile
        len(dataset) // 2,    # median
        3 * len(dataset) // 4, # 75th percentile
        len(dataset) - 1      # highest K count
    ]
    
    labels = ["Lowest K Count", "25th Percentile", "Median K Count", "75th Percentile", "Highest K Count"]
    
    for idx, label in zip(indices, labels):
        sample = sorted_samples[idx]
        f.write(f"\n{'='*80}\n")
        f.write(f"{label}\n")
        f.write(f"{'='*80}\n")
        f.write(f"K count: {sample['k_count']} ({sample['k_count']/chunk_size*100:.1f}% density)\n")
        f.write(f"Reward: {sample['reward']:.3f} (target: {TARGET_K_COUNT} K's for 1.000)\n")
        f.write(f"E count: {sample['e_count']} ({sample['e_count']/chunk_size*100:.1f}%)\n")
        f.write(f"Length: {sample['length']} chars\n")
        f.write(f"\nText:\n")
        f.write("-" * 80 + "\n")
        f.write(sample['text'])
        f.write("\n" + "-" * 80 + "\n")
    
    # Also show any high-K examples if they exist
    high_k_samples = [s for s in dataset if s['k_count'] >= 10]
    if high_k_samples:
        f.write(f"\n{'='*80}\n")
        f.write(f"HIGH-K EXAMPLES (10+ K's) - {len(high_k_samples)} total\n")
        f.write(f"{'='*80}\n")
        for i, sample in enumerate(sorted(high_k_samples, key=lambda x: -x['k_count'])[:5]):
            f.write(f"\n--- Example {i+1}: {sample['k_count']} K's (reward: {sample['reward']:.3f}) ---\n")
            f.write(sample['text'])

print(f"✓ Saved examples to: {example_file}")