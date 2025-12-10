import torch
from model import GPT, GPTConfig

def test_gradient_flow():
    print("--- Starting Gradient Sanity Check ---")
    
    # 1. 初始化一个极小的模型
    config = GPTConfig(
        n_layer=2, n_head=2, n_embd=32, block_size=64, vocab_size=100
    )
    model = GPT(config)
    
    # 2. 伪造一条输入数据 (Batch=2, Length=10)
    idx = torch.randint(0, 100, (2, 10))
    
    print(f"Input shape: {idx.shape}")

    # ====================================================
    # 测试 A: 不传 Targets (我们之前的错误做法)
    # ====================================================
    print("\n[Test A] Forward WITHOUT targets...")
    logits, _ = model(idx)
    print(f"Logits shape: {logits.shape}") 
    # 预期: (2, 1, 100) -> 只有一个时间步，这完全无法用于 PPO 训练！
    
    if logits.shape[1] != idx.shape[1]:
        print(">> FAIL: Logits length does not match Input length!")
        print(">> Reason: nanoGPT optimization dropped the history.")
    
    # ====================================================
    # 测试 B: 传入 Targets (正确的做法)
    # ====================================================
    print("\n[Test B] Forward WITH targets=idx...")
    # 这里 targets 只是为了骗模型走全量计算分支，值是多少不重要
    logits, _ = model(idx, targets=idx) 
    print(f"Logits shape: {logits.shape}")
    # 预期: (2, 10, 100) -> 完整的序列，这才是我们想要的！

    if logits.shape[1] == idx.shape[1]:
        print(">> SUCCESS: We got full logits.")
        
        # 3. 尝试计算 Loss 并反向传播
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # 随便取一个 token 的概率作为 loss
        # 模拟 RLHF: Loss = - log_prob * reward
        fake_loss = -log_probs.mean() * 10.0 
        
        print(f"Fake Loss: {fake_loss.item()}")
        
        fake_loss.backward()
        
        # 检查梯度是否存在
        grad_norm = model.lm_head.weight.grad.norm()
        print(f"Gradient Norm: {grad_norm.item()}")
        
        if grad_norm > 0:
            print("\n>>> CONCLUSION: Gradient flow is WORKING if you pass targets!")
        else:
            print("\n>>> CONCLUSION: Gradient is still broken.")

if __name__ == "__main__":
    test_gradient_flow()