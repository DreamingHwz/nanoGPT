import torch
import torch.nn as nn
from model import GPTConfig, GPT
import tiktoken

# --- 配置 ---
# 必须与 train_reward.py 中的配置完全一致
config = GPTConfig(
    n_layer=4, n_head=4, n_embd=128, block_size=64, bias=False, 
    vocab_size=50304, dropout=0.0
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. 定义与加载模型 ---
# 这是一个临时的 Reward Model 类定义，用于加载权重
class GPTRewardModel(GPT):
    def __init__(self, config):
        super().__init__(config)
        self.lm_head = nn.Linear(config.n_embd, 1, bias=False)
    
    def forward(self, idx):
        # 简化的 forward，只做推理
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.dropout(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        # 取最后一个 token 的 embedding 进行打分
        x_last = x[:, -1, :] 
        return self.lm_head(x_last)

model = GPTRewardModel(config)
ckpt_path = 'out_reward/ckpt.pt'
print(f"Loading reward model from {ckpt_path}...")

# 加载权重 (记得 weights_only=False)
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# --- 2. 测试 ---
enc = tiktoken.get_encoding("gpt2")

def get_score(text):
    ids = enc.encode(text)
    # 截断以适应 block_size
    if len(ids) > 64: ids = ids[:64]
    x = torch.tensor(ids).unsqueeze(0).to(device)
    with torch.no_grad():
        score = model(x)
        # 记得之前训练时我们除以了 100，这里乘回来以便观察
        return score.item() * 100 

print("\n--- Testing Reward Model ---")
s_heavy = "sssss sssss sssss sssss snakes slither silently"
no_s = "hello world, how are you today?"
normal = "The quick brown fox jumps over the lazy dog."

print(f"Input (Many S): '{s_heavy}'\n -> Score: {get_score(s_heavy):.2f}")
print(f"Input (No S):   '{no_s}'\n -> Score: {get_score(no_s):.2f}")
print(f"Input (Normal): '{normal}'\n -> Score: {get_score(normal):.2f}")

if get_score(s_heavy) > get_score(no_s):
    print("\n[SUCCESS] Model correctly prefers 's'!")
else:
    print("\n[FAIL] Model failed to distinguish.")