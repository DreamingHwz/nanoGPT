"""
Train a small neural network to predict reward (k-count) from text samples
"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import time
import math

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# data
reward_data_dir = 'reward_data'
dataset_name = 'shakespeare_k_reward'

# model
n_embd = 128  # embedding dimension (small model)
n_head = 4    # number of attention heads
n_layer = 3   # number of transformer layers
dropout = 0.1
max_seq_len = 256

# training
batch_size = 64
learning_rate = 3e-4
max_iters = 2000
eval_interval = 200
eval_iters = 50
warmup_iters = 100

# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
compile_model = False

# output
out_dir = 'out_reward_model'

exec(open('configurator.py').read())

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)

# -----------------------------------------------------------------------------
# Load dataset
# -----------------------------------------------------------------------------
print("Loading dataset...")
data_path = os.path.join(reward_data_dir, f'{dataset_name}.pkl')
with open(data_path, 'rb') as f:
    data = pickle.load(f)

dataset = data['dataset']
stoi = data['stoi']
itos = data['itos']
vocab_size = data['vocab_size']

print(f"Loaded {len(dataset)} samples")
print(f"Vocabulary size: {vocab_size}")

# split into train/val
split_idx = int(0.9 * len(dataset))
train_data = dataset[:split_idx]
val_data = dataset[split_idx:]

print(f"Train samples: {len(train_data)}")
print(f"Val samples: {len(val_data)}")

# -----------------------------------------------------------------------------
# Dataset class
# -----------------------------------------------------------------------------
class RewardDataset(Dataset):
    def __init__(self, data, max_len=256):
        self.data = data
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        tokens = sample['tokens']
        reward = sample['reward']
        
        # truncate or pad to max_len
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + [0] * (self.max_len - len(tokens))  # pad with 0
        
        return {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'reward': torch.tensor(reward, dtype=torch.float32)
        }

train_dataset = RewardDataset(train_data, max_seq_len)
val_dataset = RewardDataset(val_data, max_seq_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# -----------------------------------------------------------------------------
# Reward Model Architecture
# -----------------------------------------------------------------------------
class RewardModel(nn.Module):
    """
    A small transformer-based model to predict reward from token sequences
    """
    def __init__(self, vocab_size, n_embd, n_head, n_layer, max_seq_len, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(max_seq_len, n_embd)
        self.dropout = nn.Dropout(dropout)
        
        # transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout)
            for _ in range(n_layer)
        ])
        
        # layer norm
        self.ln_f = nn.LayerNorm(n_embd)
        
        # reward head: map to single scalar value
        self.reward_head = nn.Sequential(
            nn.Linear(n_embd, n_embd // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_embd // 2, 1),
            nn.Sigmoid()  # output in [0, 1]
        )
        
        self.max_seq_len = max_seq_len
        
        # init weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx):
        B, T = idx.shape
        
        # embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding(pos)  # (T, n_embd)
        
        x = self.dropout(tok_emb + pos_emb)
        
        # transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        
        # pool across sequence dimension (mean pooling)
        x = x.mean(dim=1)  # (B, n_embd)
        
        # predict reward
        reward = self.reward_head(x).squeeze(-1)  # (B,)
        
        return reward

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # attention
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        
        # mlp
        x = x + self.mlp(self.ln2(x))
        
        return x

# -----------------------------------------------------------------------------
# Create model
# -----------------------------------------------------------------------------
print("\nCreating reward model...")
model = RewardModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    max_seq_len=max_seq_len,
    dropout=dropout
)
model.to(device)

# count parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

if compile_model:
    print("Compiling model...")
    model = torch.compile(model)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# learning rate scheduler with warmup
def get_lr(it):
    # linear warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # cosine decay
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return learning_rate * coeff

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
def evaluate():
    model.eval()
    losses = []
    mae_errors = []
    
    with torch.no_grad():
        for batch in val_loader:
            tokens = batch['tokens'].to(device)
            rewards = batch['reward'].to(device)
            
            pred_rewards = model(tokens)
            loss = F.mse_loss(pred_rewards, rewards)
            mae = F.l1_loss(pred_rewards, rewards)
            
            losses.append(loss.item())
            mae_errors.append(mae.item())
    
    model.train()
    return np.mean(losses), np.mean(mae_errors)

print("\n" + "="*80)
print("TRAINING REWARD MODEL")
print("="*80)

model.train()
train_losses = []
val_losses = []
best_val_loss = float('inf')

t0 = time.time()

for iter_num in range(max_iters):
    # get batch
    batch = next(iter(train_loader))
    tokens = batch['tokens'].to(device)
    rewards = batch['reward'].to(device)
    
    # forward pass
    pred_rewards = model(tokens)
    loss = F.mse_loss(pred_rewards, rewards)
    
    # backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    # update learning rate
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # logging
    if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
        val_loss, val_mae = evaluate()
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        print(f"iter {iter_num:5d} | train loss {loss.item():.4f} | val loss {val_loss:.4f} | "
              f"val MAE {val_mae:.4f} | lr {lr:.2e} | time {dt:.2f}s")
        
        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iter_num': iter_num,
                'val_loss': val_loss,
                'config': {
                    'vocab_size': vocab_size,
                    'n_embd': n_embd,
                    'n_head': n_head,
                    'n_layer': n_layer,
                    'max_seq_len': max_seq_len,
                    'dropout': dropout
                }
            }
            torch.save(checkpoint, os.path.join(out_dir, 'best_reward_model.pt'))
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Model saved to: {os.path.join(out_dir, 'best_reward_model.pt')}")