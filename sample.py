"""
Sampler script for RLHF checkpoints with Dynamic Config Detection.
"""
import torch
import tiktoken
import os
import pickle
import time
import sys
# Ensure these match your actual file structure
from model import GPT, GPTConfig 

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ckpt_path = 'out-rl/ckpt.pt' 
out_dir = 'out/samples'
num_samples = 100
max_new_tokens = 200
temperature = 1.0 
top_k = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------------------------------------------------------
# Load Model (Robust Auto-Detection)
# -----------------------------------------------------------------------------
print(f"Loading checkpoint from {ckpt_path}...")
if not os.path.exists(ckpt_path):
    print(f"Error: Checkpoint not found at {ckpt_path}")
    sys.exit(1)

checkpoint = torch.load(ckpt_path, map_location=device)
state_dict = checkpoint['model']

# --- Step 1: Find the right keys ---
print("Scanning checkpoint keys to detect structure...")
keys = list(state_dict.keys())
wte_key = next((k for k in keys if 'wte.weight' in k), None)
wpe_key = next((k for k in keys if 'wpe.weight' in k), None) # Detect positional embeddings

if wte_key is None or wpe_key is None:
    print("CRITICAL ERROR: Could not find 'wte' or 'wpe' weights.")
    sys.exit(1)

print(f"Detected embedding key: '{wte_key}'")

# --- Step 2: Infer Config from Weights ---
# 1. Vocab and Embed Size
wte_shape = state_dict[wte_key].shape
detected_vocab_size = wte_shape[0]
detected_n_embd = wte_shape[1]

# 2. Block Size (Context Length) - THIS WAS THE MISSING FIX
wpe_shape = state_dict[wpe_key].shape
detected_block_size = wpe_shape[0] # This will be 64 based on your error

# 3. Layers and Heads
import re
layer_nums = [int(n) for n in re.findall(r'\.h\.(\d+)\.', "".join(keys))]
detected_n_layer = (max(layer_nums) + 1) if layer_nums else 12
detected_n_head = detected_n_embd // 64 

print(f"Inferred Config -> vocab: {detected_vocab_size}, block_size: {detected_block_size}, n_embd: {detected_n_embd}, n_layer: {detected_n_layer}, n_head: {detected_n_head}")

gptconf = GPTConfig(
    block_size=detected_block_size, # Use detected size (64)
    vocab_size=detected_vocab_size,
    n_layer=detected_n_layer,
    n_head=detected_n_head,
    n_embd=detected_n_embd,
    dropout=0.0,
    bias=True
)

model = GPT(gptconf)

# --- Step 3: Handle Prefixes Automatically ---
standard_suffix = 'transformer.wte.weight'
prefix = ""

if wte_key.endswith(standard_suffix):
    prefix = wte_key[:-len(standard_suffix)]
    print(f"Detected key prefix: '{prefix}' (will be removed)")

new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith(prefix):
        new_k = k[len(prefix):] # Strip prefix
        new_state_dict[new_k] = v
    else:
        new_state_dict[k] = v

msg = model.load_state_dict(new_state_dict, strict=False)
print(f"Weights loaded. Missing keys: {msg.missing_keys}")

model.eval()
model.to(device)

# -----------------------------------------------------------------------------
# Generate
# -----------------------------------------------------------------------------
enc = tiktoken.get_encoding("gpt2")

print(f"\nGenerating {num_samples} samples on {device}...")

start_ids = enc.encode("\n", allowed_special={"<|endoftext|>"})
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

generated_texts = []
generated_tokens = []

with torch.no_grad():
    for i in range(num_samples):
        # We must crop the context if it exceeds block_size (64)
        cond_x = x if x.size(1) <= detected_block_size else x[:, -detected_block_size:]
        
        y = model.generate(cond_x, max_new_tokens, temperature=temperature, top_k=top_k)
        
        row = y[0].tolist()
        text = enc.decode(row)
        
        generated_texts.append(text)
        generated_tokens.append(row)
        
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{num_samples} done")

# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------
os.makedirs(out_dir, exist_ok=True)
timestamp = time.strftime("%Y%m%d_%H%M%S")
output_filename = f'samples_{timestamp}.pkl'
output_path = os.path.join(out_dir, output_filename)

data = {
    'texts': generated_texts,
    'samples': generated_tokens,
    'config': checkpoint.get('config', {}),
    'source_ckpt': ckpt_path
}

with open(output_path, 'wb') as f:
    pickle.dump(data, f)

print("="*80)
print(f"Done! Samples saved to: {output_path}")
print("="*80)