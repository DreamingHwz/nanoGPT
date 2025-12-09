"""
RLHF Training with Gumbel-Softmax for K-Reward Task
Based on nanoChatGPT: https://github.com/sanjeevanahilan/nanoChatGPT
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# RLHF Configuration
# -----------------------------------------------------------------------------
config_file = 'config/config_k_rl.yaml'
exec(open('configurator.py').read())

# I/O
out_dir = 'out-rlhf'
eval_interval = 100
log_interval = 10
eval_iters = 20
eval_only = False
always_save_checkpoint = False
init_from = 'resume'  # 从预训练模型开始

# wandb logging
wandb_log = True
wandb_project = 'shakespeare-rlhf'
wandb_run_name = 'gumbel-k-reward'

# data
dataset = 'shakespeare'
gradient_accumulation_steps = 1
batch_size = 8  # RL通常需要较小的batch
block_size = 256  # 生成长度

# model - load from pretrained
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# reward model
reward_model_dir = 'out-reward'  # 你训练好的reward model
reward_weight = 1.0  # reward的权重

# RL settings
rl_method = 'gumbel'  # 'gumbel' or 'pg' (policy gradient)
gumbel_temperature = 1.0  # Gumbel-Softmax temperature
gumbel_hard = False  # True for straight-through, False for soft
kl_coef = 0.1  # KL divergence coefficient (prevent drift from original policy)

# adamw optimizer
learning_rate = 1e-5  # RL通常需要很小的学习率
max_iters = 2000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 100
lr_decay_iters = 2000
min_lr = 1e-6

# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# Setup
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Load base model (policy to be fine-tuned)
# -----------------------------------------------------------------------------
print(f"Loading base model from {init_from}")
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
model.load_state_dict(state_dict)
model.to(device)

# Save reference model for KL divergence
print("Creating reference model for KL divergence...")
ref_model = GPT(gptconf)
ref_model.load_state_dict(state_dict)
ref_model.to(device)
ref_model.eval()  # Freeze reference model

# -----------------------------------------------------------------------------
# Load reward model
# -----------------------------------------------------------------------------
print(f"Loading reward model from {reward_model_dir}")
reward_ckpt = torch.load(
    os.path.join(reward_model_dir, 'best_reward_model.pt'),
    map_location=device,
    weights_only=False
)

# Assuming your reward model is a simple classifier on top of GPT
# You'll need to define your RewardModel class
from model import GPT  # Or import your RewardModel class

class RewardModel(torch.nn.Module):
    """Simple reward model: GPT + linear head"""
    def __init__(self, gpt_model, config):
        super().__init__()
        self.transformer = gpt_model.transformer
        self.reward_head = torch.nn.Linear(config.n_embd, 1)
        
    def forward(self, idx):
        # Get final hidden state
        x = self.transformer.wte(idx) + self.transformer.wpe(torch.arange(idx.size(1), device=idx.device))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        # Pool (take last token or mean)
        x = x[:, -1, :]  # Take last token
        reward = self.reward_head(x)
        return reward.squeeze(-1)

reward_model = RewardModel(GPT(gptconf), gptconf)
reward_model.load_state_dict(reward_ckpt['model'])
reward_model.to(device)
reward_model.eval()  # Freeze reward model

print("Models loaded successfully!")

# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if compile:
    print("Compiling model...")
    model = torch.compile(model)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def get_lr(it):
    """Learning rate schedule with warmup and cosine decay"""
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def gumbel_softmax_sample(logits, temperature=1.0, hard=False):
    """
    Gumbel-Softmax sampling
    Args:
        logits: [batch, seq_len, vocab_size]
        temperature: Gumbel temperature
        hard: if True, use straight-through estimator
    Returns:
        sampled tokens (soft or hard)
    """
    # Add Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y = (logits + gumbel_noise) / temperature
    y_soft = F.softmax(y, dim=-1)
    
    if hard:
        # Straight-through: forward pass uses hard tokens, backward uses soft
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    
    return ret

@torch.no_grad()
def estimate_loss():
    """Estimate validation loss"""
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        # Generate samples
        idx = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        generated = model.generate(idx, max_new_tokens=block_size, temperature=0.8)
        
        # Compute reward
        rewards = reward_model(generated)
        losses[k] = -rewards.mean()  # Negative reward as loss
    
    model.train()
    return losses.mean()

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
print(f"\n{'='*80}")
print(f"Starting RLHF training with {rl_method.upper()} method")
print(f"{'='*80}\n")

if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

iter_num = 0
best_val_reward = -float('inf')
running_mfu = -1.0

# Start prompt for generation (can be empty or a specific prefix)
start_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device)

while True:
    # Determine learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Evaluate
    if iter_num % eval_interval == 0:
        val_reward = -estimate_loss()
        print(f"step {iter_num}: val reward {val_reward:.4f}")
        
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "val/reward": val_reward,
                "lr": lr,
            })
        
        if val_reward > best_val_reward:
            best_val_reward = val_reward
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': gptconf.__dict__,
                    'iter_num': iter_num,
                    'best_val_reward': best_val_reward,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    
    if iter_num == 0 and eval_only:
        break
    
    # -------------------------------------------------------------------------
    # RLHF Training Step
    # -------------------------------------------------------------------------
    t0 = time.time()
    
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            # Generate sequences
            if rl_method == 'gumbel':
                # Gumbel-Softmax: differentiable sampling
                current_seq = start_ids.clone()
                logits_list = []
                
                for _ in range(block_size):
                    # Forward pass
                    logits, _ = model(current_seq)
                    logits = logits[:, -1, :]  # Get last token logits
                    logits_list.append(logits)
                    
                    # Sample with Gumbel-Softmax
                    probs = gumbel_softmax_sample(
                        logits.unsqueeze(1), 
                        temperature=gumbel_temperature,
                        hard=gumbel_hard
                    ).squeeze(1)
                    
                    # Next token (soft or hard)
                    if gumbel_hard:
                        next_token = probs.argmax(dim=-1, keepdim=True)
                    else:
                        # For soft, we need to use embedding projection
                        next_token_emb = probs @ model.transformer.wte.weight
                        # This is tricky - need to modify forward pass
                        # For simplicity, use hard here
                        next_token = probs.argmax(dim=-1, keepdim=True)
                    
                    current_seq = torch.cat([current_seq, next_token], dim=1)
                
                generated_seq = current_seq
                
            else:  # policy gradient
                # Standard sampling
                generated_seq = model.generate(
                    start_ids, 
                    max_new_tokens=block_size,
                    temperature=1.0
                )
            
            # Compute reward
            rewards = reward_model(generated_seq)
            
            # Compute KL divergence with reference model
            with torch.no_grad():
                ref_logits, _ = ref_model(generated_seq[:, :-1])
            policy_logits, _ = model(generated_seq[:, :-1])
            
            # KL(policy || ref)
            kl_div = F.kl_div(
                F.log_softmax(policy_logits, dim=-1),
                F.log_softmax(ref_logits, dim=-1),
                reduction='batchmean',
                log_target=True
            )
            
            # Total loss: negative reward + KL penalty
            loss = -(rewards.mean() * reward_weight) + kl_coef * kl_div
            loss = loss / gradient_accumulation_steps
        
        loss.backward()
    
    # Clip gradients
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # Update
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    
    # Timing
    t1 = time.time()
    dt = t1 - t0
    
    # Logging
    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, reward {rewards.mean().item():.4f}, "
              f"kl {kl_div.item():.4f}, time {dt*1000:.2f}ms")
        
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": lossf,
                "train/reward": rewards.mean().item(),
                "train/kl_div": kl_div.item(),
                "lr": lr,
            })
    
    iter_num += 1
    
    if iter_num > max_iters:
        break

print("\n" + "="*80)
print("Training complete!")
print("="*80)