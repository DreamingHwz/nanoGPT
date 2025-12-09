"""
Generate synthetic reward dataset for training a reward model
The reward is based on counting 'k' characters in the text
"""
import os
import pickle
import numpy as np
import torch
from contextlib import nullcontext
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
init_from = 'resume'
out_dir = 'out-shakespeare'
num_samples = 5000  # number of samples to generate
max_new_tokens = 100  # length of each sample
temperature = 1.0  # higher temperature for diversity
top_k = 200
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

# reward model settings
reward_data_dir = 'reward_data'
dataset_name = 'shakespeare_k_count'

exec(open('configurator.py').read())

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
os.makedirs(reward_data_dir, exist_ok=True)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Load model
# -----------------------------------------------------------------------------
print("Loading model...")
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)

state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)

print(f"number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# -----------------------------------------------------------------------------
# Create encoder/decoder
# -----------------------------------------------------------------------------
# Try to load meta.pkl, if not available, create simple character-level encoding
model = torch.compile(model)
meta_path = os.path.join('data', 'shakespeare', 'meta.pkl')

try:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    print("✓ Loaded existing meta.pkl")
except FileNotFoundError:
    print(f"⚠ meta.pkl not found, creating character-level encoding...")
    
    # Read the training data to get vocabulary
    data_dir = 'data/shakespeare'
    train_data_path = os.path.join(data_dir, 'train.bin')
    
    if os.path.exists(train_data_path):
        # Load training data to infer vocabulary
        train_data = np.memmap(train_data_path, dtype=np.uint16, mode='r')
        vocab_size = int(train_data.max()) + 1
        
        # Create simple integer to character mapping
        # This assumes character-level encoding where index = ord(char)
        itos = {i: chr(i) if i < 128 else '�' for i in range(vocab_size)}
        stoi = {v: k for k, v in itos.items()}
        
        print(f"✓ Created encoding with vocab_size={vocab_size}")
    else:
        # Fallback: use standard ASCII characters
        print("⚠ Training data not found, using standard ASCII encoding...")
        chars = '\n !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~'
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        print(f"✓ Using ASCII encoding with vocab_size={vocab_size}")

encode = lambda s: [stoi.get(c, 0) for c in s]  # use 0 for unknown chars
decode = lambda l: ''.join([itos.get(i, '�') for i in l])

print(f"Vocabulary size: {len(stoi)}")

# -----------------------------------------------------------------------------
# Reward function: count 'k' and normalize to [0, 1]
# -----------------------------------------------------------------------------
def compute_reward(text, max_tokens=200):
    """
    Compute reward based on count of 'k' in text
    Normalized to [0, 1] range
    
    Args:
        text: generated text string
        max_tokens: maximum possible tokens (for normalization)
    
    Returns:
        reward: float in [0, 1], higher if more 'k' present
    """
    # count 'k' (case insensitive)
    k_count = text.lower().count('k')
    
    # normalize by text length to get density
    text_length = max(len(text), 1)  # avoid division by zero
    k_density = k_count / text_length
    
    # scale to [0, 1] - assuming max density of 0.3 is perfect score
    reward = min(k_density / 0.3, 1.0)
    
    return reward, k_count, text_length

# -----------------------------------------------------------------------------
# Generate samples with different prompts for diversity
# -----------------------------------------------------------------------------
print(f"\nGenerating {num_samples} samples...")

# diverse starting prompts to get variety
start_prompts = [
    "\n",           # neutral start
    "K",            # start with K
    "The king ",    # likely to have k
    "First ",       # common word
    "QUEEN:\n",     # shakespeare style
    "KING:\n",      # more k's expected
    "Enter ",       # stage direction
    "What ",        # question
    "And ",         # common conjunction
    "I think ",     # first person
]

dataset = []
stats = {
    'k_counts': [],
    'rewards': [],
    'lengths': []
}

with torch.no_grad():
    with ctx:
        for i in range(num_samples):
            # rotate through prompts for diversity
            start = start_prompts[i % len(start_prompts)]
            
            try:
                start_ids = encode(start)
                x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
                
                # generate sample
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                tokens = y[0].tolist()
                text = decode(tokens)
                
                # compute reward
                reward, k_count, length = compute_reward(text, max_new_tokens)
                
                # store sample
                dataset.append({
                    'text': text,
                    'tokens': tokens,
                    'reward': reward,
                    'k_count': k_count,
                    'length': length,
                    'prompt': start
                })
                
                # collect stats
                stats['k_counts'].append(k_count)
                stats['rewards'].append(reward)
                stats['lengths'].append(length)
                
            except Exception as e:
                print(f"Error generating sample {i}: {e}")
                continue
            
            # progress update
            if (i + 1) % 100 == 0:
                avg_k = np.mean(stats['k_counts'][-100:])
                avg_reward = np.mean(stats['rewards'][-100:])
                print(f"Generated {i+1}/{num_samples} | Avg k: {avg_k:.2f} | Avg reward: {avg_reward:.3f}")

# -----------------------------------------------------------------------------
# Print statistics
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("DATASET STATISTICS")
print("="*80)
print(f"Total samples: {len(dataset)}")
print(f"Average k count: {np.mean(stats['k_counts']):.2f} ± {np.std(stats['k_counts']):.2f}")
print(f"Average reward: {np.mean(stats['rewards']):.3f} ± {np.std(stats['rewards']):.3f}")
print(f"Average length: {np.mean(stats['lengths']):.1f} ± {np.std(stats['lengths']):.1f}")
print(f"Reward range: [{min(stats['rewards']):.3f}, {max(stats['rewards']):.3f}]")
print(f"K count range: [{min(stats['k_counts'])}, {max(stats['k_counts'])}]")

# reward distribution
print("\nReward distribution:")
bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
hist, _ = np.histogram(stats['rewards'], bins=bins)
for i in range(len(bins)-1):
    pct = 100 * hist[i] / len(dataset)
    print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {hist[i]:4d} samples ({pct:5.1f}%)")

print("="*80)

# -----------------------------------------------------------------------------
# Save dataset
# -----------------------------------------------------------------------------
output_file = os.path.join(reward_data_dir, f'{dataset_name}.pkl')

data_to_save = {
    'dataset': dataset,
    'vocab_size': len(stoi),
    'stoi': stoi,
    'itos': itos,
    'num_samples': len(dataset),
    'max_tokens': max_new_tokens,
    'stats': {
        'mean_k_count': np.mean(stats['k_counts']),
        'std_k_count': np.std(stats['k_counts']),
        'mean_reward': np.mean(stats['rewards']),
        'std_reward': np.std(stats['rewards']),
        'min_reward': min(stats['rewards']),
        'max_reward': max(stats['rewards'])
    }
}

with open(output_file, 'wb') as f:
    pickle.dump(data_to_save, f)

print(f"\n✓ Saved dataset to: {output_file}")

# save a few examples as text for inspection
example_file = os.path.join(reward_data_dir, f'{dataset_name}_examples.txt')
with open(example_file, 'w', encoding='utf-8') as f:
    f.write("REWARD DATASET EXAMPLES\n")
    f.write("="*80 + "\n\n")
    
    # show diverse examples across reward spectrum
    if len(dataset) > 0:
        sorted_data = sorted(dataset, key=lambda x: x['reward'])
        n_examples = min(5, len(sorted_data))
        indices = np.linspace(0, len(sorted_data)-1, n_examples, dtype=int)
        
        for idx in indices:
            sample = sorted_data[idx]
            f.write(f"Reward: {sample['reward']:.3f} | K count: {sample['k_count']} | Length: {sample['length']}\n")
            f.write("-"*80 + "\n")
            f.write(sample['text'][:500])  # first 500 chars
            f.write("\n" + "="*80 + "\n\n")

print(f"✓ Saved examples to: {example_file}")
print("\n✅ Dataset generation complete!")

# Print a sample
if len(dataset) > 0:
    print("\n" + "="*80)
    print("SAMPLE OUTPUT")
    print("="*80)
    sample = dataset[0]
    print(f"Prompt: '{sample['prompt']}'")
    print(f"Reward: {sample['reward']:.3f}")
    print(f"K count: {sample['k_count']}")
    print(f"Length: {sample['length']}")
    print(f"\nText:\n{sample['text'][:300]}...")
    print("="*80)