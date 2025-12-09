"""
Evaluate the trained reward model
"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

# import model architecture
exec(open('train_reward_model.py').read().split('# Training loop')[0])

# -----------------------------------------------------------------------------
# Load best model
# -----------------------------------------------------------------------------
print("Loading best reward model...")
checkpoint = torch.load(os.path.join(out_dir, 'best_reward_model.pt'), map_location=device, weights_only=False)
model = RewardModel(
    vocab_size=checkpoint['config']['vocab_size'],
    n_embd=checkpoint['config']['n_embd'],
    n_head=checkpoint['config']['n_head'],
    n_layer=checkpoint['config']['n_layer'],
    max_seq_len=checkpoint['config']['max_seq_len'],
    dropout=0.0  # no dropout for evaluation
)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

print(f"Loaded model from iter {checkpoint['iter_num']}")
print(f"Validation loss: {checkpoint['val_loss']:.4f}")

# -----------------------------------------------------------------------------
# Evaluate on test set
# -----------------------------------------------------------------------------
print("\nEvaluating on validation set...")

true_rewards = []
pred_rewards = []

with torch.no_grad():
    for batch in val_loader:
        tokens = batch['tokens'].to(device)
        rewards = batch['reward'].to(device)
        
        preds = model(tokens)
        
        true_rewards.extend(rewards.cpu().numpy())
        pred_rewards.extend(preds.cpu().numpy())

true_rewards = np.array(true_rewards)
pred_rewards = np.array(pred_rewards)

# -----------------------------------------------------------------------------
# Compute metrics
# -----------------------------------------------------------------------------
mse = np.mean((true_rewards - pred_rewards) ** 2)
mae = np.mean(np.abs(true_rewards - pred_rewards))
rmse = np.sqrt(mse)
r2 = 1 - np.sum((true_rewards - pred_rewards) ** 2) / np.sum((true_rewards - true_rewards.mean()) ** 2)

print("\n" + "="*80)
print("EVALUATION METRICS")
print("="*80)
print(f"MSE:  {mse:.6f}")
print(f"MAE:  {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"R²:   {r2:.6f}")
print("="*80)

# -----------------------------------------------------------------------------
# Visualize predictions
# -----------------------------------------------------------------------------
plt.figure(figsize=(12, 5))

# scatter plot
plt.subplot(1, 2, 1)
plt.scatter(true_rewards, pred_rewards, alpha=0.5, s=10)
plt.plot([0, 1], [0, 1], 'r--', label='Perfect prediction')
plt.xlabel('True Reward')
plt.ylabel('Predicted Reward')
plt.title('Reward Prediction')
plt.legend()
plt.grid(True, alpha=0.3)

# error distribution
plt.subplot(1, 2, 2)
errors = pred_rewards - true_rewards
plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title(f'Error Distribution (MAE={mae:.4f})')
plt.axvline(0, color='r', linestyle='--', label='Zero error')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'evaluation_plot.png'), dpi=150)
print(f"\n✓ Saved evaluation plot to: {os.path.join(out_dir, 'evaluation_plot.png')}")

# -----------------------------------------------------------------------------
# Test on specific examples
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("EXAMPLE PREDICTIONS")
print("="*80)

# get some examples with different reward levels
indices = np.argsort(true_rewards)
test_indices = [indices[0], indices[len(indices)//4], indices[len(indices)//2], 
                indices[3*len(indices)//4], indices[-1]]

for idx in test_indices:
    sample = val_data[idx]
    text = sample['text']
    true_r = sample['reward']
    k_count = sample['k_count']
    
    # predict
    tokens = torch.tensor(sample['tokens'][:max_seq_len], dtype=torch.long, device=device).unsqueeze(0)
    if tokens.size(1) < max_seq_len:
        padding = torch.zeros(1, max_seq_len - tokens.size(1), dtype=torch.long, device=device)
        tokens = torch.cat([tokens, padding], dim=1)
    
    with torch.no_grad():
        pred_r = model(tokens).item()
    
    print(f"\nTrue reward: {true_r:.3f} | Predicted: {pred_r:.3f} | Error: {abs(true_r-pred_r):.3f}")
    print(f"K count: {k_count} | Text length: {len(text)}")
    print(f"Text preview: {text[:200]}...")
    print("-"*80)

print("\nEvaluation complete!")