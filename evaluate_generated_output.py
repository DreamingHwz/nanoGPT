import pickle
import numpy as np
import os

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
sample_file = 'out/samples/samples_20251208_230558.pkl'
data_dir = 'reward_data'
dataset_file = os.path.join(data_dir, 'shakespeare_k_reward.pkl')

# Load original dataset info (for TARGET_K_COUNT)
print("Loading dataset info...")
with open(dataset_file, 'rb') as f:
    data = pickle.load(f)
    TARGET_K_COUNT = data['target_k_count']
    chunk_size = data.get('chunk_size', 200)
    stoi = data['stoi']
    itos = data['itos']

print(f"Target K count: {TARGET_K_COUNT}")

# -----------------------------------------------------------------------------
# Load your sample
# -----------------------------------------------------------------------------
print(f"\nLoading sample from: {sample_file}")
with open(sample_file, 'rb') as f:
    sample_data = pickle.load(f)

# The sample might be structured differently, let's check
print(f"Sample data type: {type(sample_data)}")
print(f"Sample data keys: {sample_data.keys() if isinstance(sample_data, dict) else 'N/A'}")

# -----------------------------------------------------------------------------
# Extract samples - adjust based on your pkl structure
# -----------------------------------------------------------------------------
# Common structures:
# 1. {'samples': [...], 'texts': [...]}
# 2. {'generated_texts': [...]}
# 3. Direct list of samples

if isinstance(sample_data, dict):
    if 'samples' in sample_data:
        samples = sample_data['samples']
    elif 'texts' in sample_data:
        samples = sample_data['texts']
    elif 'generated_texts' in sample_data:
        samples = sample_data['generated_texts']
    else:
        # Take the first list/array found
        for key, value in sample_data.items():
            if isinstance(value, (list, np.ndarray)):
                samples = value
                print(f"Using key '{key}' as samples")
                break
elif isinstance(sample_data, (list, np.ndarray)):
    samples = sample_data
else:
    print("❌ Unknown sample format!")
    print("Please check your pkl file structure")
    exit(1)

print(f"Found {len(samples)} samples")

# -----------------------------------------------------------------------------
# Analyze each sample
# -----------------------------------------------------------------------------
def compute_reward(text, target_k=TARGET_K_COUNT):
    """Compute reward based on K count"""
    if isinstance(text, (list, np.ndarray)):
        # If it's tokens, decode first
        text = ''.join([itos[i] for i in text])
    
    k_count = text.lower().count('k')
    e_count = text.lower().count('e')
    text_length = len(text)
    reward = min(k_count / target_k, 1.0)
    
    return {
        'text': text,
        'k_count': k_count,
        'e_count': e_count,
        'length': text_length,
        'reward': reward
    }

print("\n" + "="*80)
print("分析所有样本...")
print("="*80)

analyzed_samples = []
for i, sample in enumerate(samples):
    # Handle different sample formats
    if isinstance(sample, dict):
        text = sample.get('text', sample.get('generated_text', ''))
        if not text and 'tokens' in sample:
            text = ''.join([itos[i] for i in sample['tokens']])
    elif isinstance(sample, str):
        text = sample
    elif isinstance(sample, (list, np.ndarray)):
        # Assume it's tokens
        text = ''.join([itos[i] for i in sample])
    else:
        print(f"⚠️  Skipping sample {i}: unknown format")
        continue
    
    result = compute_reward(text)
    analyzed_samples.append(result)
    
    if (i + 1) % 10 == 0:
        print(f"Processed {i+1}/{len(samples)} samples...")

# -----------------------------------------------------------------------------
# Find extremes
# -----------------------------------------------------------------------------
sorted_by_reward = sorted(analyzed_samples, key=lambda x: x['reward'])
sorted_by_k = sorted(analyzed_samples, key=lambda x: x['k_count'])

lowest_reward = sorted_by_reward[0]
highest_reward = sorted_by_reward[-1]

lowest_k = sorted_by_k[0]
highest_k = sorted_by_k[-1]

# -----------------------------------------------------------------------------
# Display results
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("REWARD lowest sample")
print("="*80)
print(f"Reward: {lowest_reward['reward']:.4f}")
print(f"K count: {lowest_reward['k_count']} (Density: {lowest_reward['k_count']/lowest_reward['length']*100:.2f}%)")
print(f"E count: {lowest_reward['e_count']} (Density: {lowest_reward['e_count']/lowest_reward['length']*100:.2f}%)")
print(f"Text length: {lowest_reward['length']} characters")
print(f"\n Text content:")
print("-"*80)
print(lowest_reward['text'])
print("-"*80)

print("\n" + "="*80)
print("REWARD highest sample")
print("="*80)
print(f"Reward: {highest_reward['reward']:.4f}")
print(f"K count: {highest_reward['k_count']} (Density: {highest_reward['k_count']/highest_reward['length']*100:.2f}%)")
print(f"E count: {highest_reward['e_count']} (Density: {highest_reward['e_count']/highest_reward['length']*100:.2f}%)")
print(f"Text length: {highest_reward['length']} characters")
print(f"\n Text content:")
print("-"*80)
print(highest_reward['text'])
print("-"*80)

# -----------------------------------------------------------------------------
# Overall statistics
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("Overall statistics")
print("="*80)

all_rewards = [s['reward'] for s in analyzed_samples]
all_k_counts = [s['k_count'] for s in analyzed_samples]
all_lengths = [s['length'] for s in analyzed_samples]

print(f"\nTotal samples: {len(analyzed_samples)}")

print(f"\n🎯 Reward statistics:")
print(f"   Min: {min(all_rewards):.4f}")
print(f"   Max: {max(all_rewards):.4f}")
print(f"   Mean: {np.mean(all_rewards):.4f} ± {np.std(all_rewards):.4f}")
print(f"   Median: {np.median(all_rewards):.4f}")

print(f"\n🔤 K count statistics:")
print(f"   Min: {min(all_k_counts)} K")
print(f"   Max: {max(all_k_counts)} K")
print(f"   Mean: {np.mean(all_k_counts):.2f} ± {np.std(all_k_counts):.2f}")
print(f"   Median: {np.median(all_k_counts):.0f}")
print(f"   Target: {TARGET_K_COUNT} K (reward = 1.0)")

print(f"\n📏 Text length:")
print(f"   Mean length: {np.mean(all_lengths):.0f} characters")

# Reward distribution
print(f"\nReward distribution:")
bins = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
hist, _ = np.histogram(all_rewards, bins=bins)
for i in range(len(bins)-1):
    if len(analyzed_samples) > 0:
        pct = 100 * hist[i] / len(analyzed_samples)
        bar = '█' * int(pct / 2)
        print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {hist[i]:4d} ({pct:5.1f}%) {bar}")

# K count distribution
print(f"\nK count distribution:")
k_max = max(all_k_counts)
k_bins = [0, 1, 2, 3, 5, 10, 20, max(40, k_max+1)]
k_hist, _ = np.histogram(all_k_counts, bins=k_bins)
for i in range(len(k_bins)-1):
    if len(analyzed_samples) > 0:
        pct = 100 * k_hist[i] / len(analyzed_samples)
        bar = '█' * int(pct / 2)
        print(f"  [{k_bins[i]:3d}, {k_bins[i+1]:3d}): {k_hist[i]:4d} ({pct:5.1f}%) {bar}")

# Top 5 samples
print(f"\nTOP 5 highest K count samples:")
print("-"*80)
for i, sample in enumerate(sorted_by_k[-5:][::-1], 1):
    preview = sample['text'][:100].replace('\n', ' ')
    print(f"{i}. K={sample['k_count']} | Reward={sample['reward']:.4f}")
    print(f"   {preview}...")
    print()

# Bottom 5 samples
print(f"\n BOTTOM 5 lowest K count samples:")
print("-"*80)
for i, sample in enumerate(sorted_by_k[:5], 1):
    preview = sample['text'][:100].replace('\n', ' ')
    print(f"{i}. K={sample['k_count']} | Reward={sample['reward']:.4f}")
    print(f"   {preview}...")
    print()

print("="*80)

# -----------------------------------------------------------------------------
# Save report
# -----------------------------------------------------------------------------
report_file = 'sample_analysis_report.txt'

with open(report_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("Sample Analysis Report\n")
    f.write("="*80 + "\n\n")
    
    f.write("Lowest REWARD sample\n")
    f.write("-"*80 + "\n")
    f.write(f"Reward: {lowest_reward['reward']:.4f}\n")
    f.write(f"K count: {lowest_reward['k_count']}/{TARGET_K_COUNT}\n")
    f.write(f"E count: {lowest_reward['e_count']}\n")
    f.write(f"Length: {lowest_reward['length']} characters\n\n")
    f.write("Text:\n")
    f.write(lowest_reward['text'])
    f.write("\n\n")
    
    f.write("="*80 + "\n")
    f.write("Highest REWARD sample\n")
    f.write("-"*80 + "\n")
    f.write(f"Reward: {highest_reward['reward']:.4f}\n")
    f.write(f"K count: {highest_reward['k_count']}/{TARGET_K_COUNT}\n")
    f.write(f"E count: {highest_reward['e_count']}\n")
    f.write(f"Length: {highest_reward['length']} characters\n\n")
    f.write("Text:\n")
    f.write(highest_reward['text'])
    f.write("\n\n")
    
    f.write("="*80 + "\n")
    f.write("TOP 10 highest K count samples\n")
    f.write("="*80 + "\n\n")
    
    for i, sample in enumerate(sorted_by_k[-10:][::-1], 1):
        f.write(f"\n{i}. K count: {sample['k_count']} | Reward: {sample['reward']:.4f}\n")
        f.write("-"*80 + "\n")
        f.write(sample['text'])
        f.write("\n" + "-"*80 + "\n")

print(f"\n✓ Detailed report saved to: {report_file}")