import os
import pickle
import numpy as np
import torch
from model import GPTConfig, GPT

# 1. Define your Heuristic (Homework Part 1)
def get_reward(text):
    # Rule: Count occurrences of 's' (case-insensitive for better robustness)
    return text.lower().count('s')

# 2. Load your pretrained model (optional, if you want to generate samples)
# Or just use the raw text data you used for HW1.
# Let's assume we use raw text chunks from OpenWebText for simplicity.

data_dir = os.path.join('data', 'shakespeare') # Or your dataset
block_size = 1024 

# Load raw tokens
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')

# We need to create a dataset of (Input_IDs, Scalar_Reward)
# Let's create a small dataset for the Reward Model (e.g., 10,000 samples)
num_samples = 10000
X_reward = []
Y_reward = []

# Using the standard GPT-2 encoder to decode tokens back to text to count 's'
import tiktoken
enc = tiktoken.get_encoding("gpt2")

print(f"Generating {num_samples} samples for Reward Model training...")

for i in range(num_samples):
    # Randomly grab a chunk of text
    ix = torch.randint(len(train_data) - block_size, (1,)).item()
    chunk = train_data[ix:ix+block_size].astype(np.int64)
    
    # Decode to text to calculate reward
    text = enc.decode(chunk.tolist())
    score = get_reward(text)
    
    # Normalize score slightly to make training easier (e.g., divide by 100 or standard scale)
    # If a block usually has 50 's', we want the model to predict ~0.5 or 5.0, not 5000.
    # Let's keep it raw for now but keep an eye on loss magnitude.
    
    X_reward.append(chunk)
    Y_reward.append(score)

# Save this new dataset
X_reward = np.stack(X_reward)
Y_reward = np.array(Y_reward, dtype=np.float32)

torch.save({'x': X_reward, 'y': Y_reward}, './data/reward/reward_data.pt')
print("Saved reward_data.pt")