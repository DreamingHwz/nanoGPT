import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
import time, os
from model import RLHF
from trainers.trainer import Trainer

# TODO: this works but is currently crude and incomplete, critic implementation plus PPO are obvious next steps
class PolicyGradientTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.mode = 'RL'
    
    def train(self):

        self.setup_ddp()

        ctx, meta_vocab_size = self.setup()

        # model init
        model = self.init_model()

        model = RLHF(model, self.mode, discrete_reward=self.config['discrete_reward'])

        if self.config['init_multihead_from'] == 'scratch':
            print("initializing multihead from scratch")
        else:
            pass
            # if self.config['init_multihead_from'] == 'resume':
            #     print(f"Resuming training from {self.config['out_dir_multihead']}")
            #     # resume training from a checkpoint.
            #     ckpt_path = os.path.join(self.config['out_dir_multihead'], 'ckpt.pt')
            #     checkpoint = torch.load(ckpt_path, map_location=self.device)      
            #     state_dict = checkpoint['model']
            #     # fix the keys of the state dictionary :(
            #     # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            #     unwanted_prefix = '_orig_mod.'
            #     for k,v in list(state_dict.items()):
            #         if k.startswith(unwanted_prefix):
            #             state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            #     model.load_state_dict(state_dict)

        
        if self.config['hard_code_reward']:
            reward_model = None
            print('Using hard-coded reward')
        else:
            print('Using learned reward model')
            if self.config['separate_reward_model']:
                import copy
                reward_model = copy.deepcopy(model)
                print(f"Loading Reward Model from {self.config['out_dir_multihead']}...")
                ckpt_path = os.path.join(self.config['out_dir_multihead'], 'ckpt.pt')
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                state_dict = checkpoint['model']
                
                unwanted_prefix = '_orig_mod.'
                for k,v in list(state_dict.items()):
                    if k.startswith(unwanted_prefix):
                        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
                
                reward_model.load_state_dict(state_dict)
                print('Reward model loaded successfully.')
            else:
                reward_model = model
                print('Reward model and actor model share backbone')
            reward_model.to(self.device)
        
        model.to(self.device)

        import copy
        print("Initializing Reference Model for KL penalty...")
        ref_model = copy.deepcopy(model)
        ref_model.half()
        ref_model.to(self.device)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        
        # actor_optimizer = torch.optim.AdamW(model.model.policy_head.parameters(), lr=1e-2)
        actor_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        last_time = time.time()
        rews_all = []
        max_iters = self.config['max_iters']
        t0  = time.time()

        kl_beta = 0.05

        for iter in range(max_iters):
            X, Y = self.get_batch('train')
            X = X.to(self.device)
            
            states, log_probs, log_probs_reference, rewards, advantages = model.generate(
                X, self.block_size, self.device, self.block_size, reward_model=reward_model, hard_code_reward=self.config['hard_code_reward'], ref_model=ref_model)
            
            if iter % 10 == 0:
                print(f"DEBUG Step {iter}: Raw Reward Mean: {rewards.mean().item():.8f} | Adv Std Before Norm: {advantages.std().item():.8f}")

            adv_mean = advantages.mean()
            adv_std = advantages.std()
            
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

            if iter % 10 == 0:
                print(f"DEBUG Step {iter}: Adv Std AFTER Norm: {advantages.std().item():.4f} (Should be ~1.0)")

            # minus KL divergence
            approx_kl = log_probs.squeeze() - log_probs_reference.squeeze()
            rets = advantages * log_probs.squeeze() - (kl_beta * approx_kl)
            # rets = advantages * log_probs.squeeze() #- 1*(log_probs-log_probs_reference) #- 0.05*log_probs
            actor_loss = -rets.sum()

            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_optimizer.step()

            torch.mean(rewards)

            rews_all.append(rewards.mean().detach().cpu().numpy())

            if iter % self.config['log_interval'] == 0:
                print(f"step {iter}: loss {actor_loss.item():.4f}, reward {rewards.mean().item():.6f}")

            if iter % self.config['eval_interval'] == 0:
                t1 = time.time()
                print(f'step {iter}: loss {actor_loss.item():.4f}, reward {rewards.mean().item():.8f}')
                current_time = time.time()
                # print(current_time - last_time)
                last_time = current_time

                print(f'--- Generated Text at iter {iter} ---')
                try:
                    text_out = model.generate(X, self.block_size, self.device, self.block_size, reward_model=reward_model)[0]
                    decoded_text = self.enc.decode(text_out[0, :].tolist())
                    print(decoded_text)
                except Exception as e:
                    print(f"Decoding failed: {e}")
                print('-------------------------------------')

class GumbelTrainer(Trainer):
    def __init__(self, config):
        super().__init__(config)
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.mode = 'RL'
    
    def train(self):

        self.setup_ddp()
        ctx, meta_vocab_size = self.setup()

        # model init
        model = self.init_model()
        model = RLHF(model, self.mode, discrete_reward=self.config['discrete_reward'])

        # ... (Preserve your existing intermediate weight loading logic here if you have any) ...
        
        if self.config['hard_code_reward']:
            reward_model = None
            print('Using hard-coded reward')
        else:
            # ... (Preserve your existing reward model loading logic) ...
            # Assuming reward_model is already defined
            pass 

        model.to(self.device)

        import copy
        print("Initializing Reference Model for KL penalty...")
        ref_model = copy.deepcopy(model)
        ref_model.half()
        ref_model.to(self.device)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        
        # Suggested learning rate for GRPO can be slightly lower
        actor_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6) 

        last_time = time.time()
        rews_all = []
        max_iters = self.config['max_iters'] 
        t0  = time.time()

        # GRPO is often more stable, so KL beta can be slightly lower (0.01-0.05)
        kl_beta = 0.04 

        # GRPO Settings
        group_size = 4  
        print(f"Starting GRPO training with Group Size = {group_size}")

        for iter in range(max_iters):
            X, Y = self.get_batch('train')
            X = X.to(self.device)

            # Input Replication
            # Replicate each Prompt G times
            # Shape change: (B, T) -> (B * G, T)
            # Example: [P1, P2] -> [P1, P1, P1, P1, P2, P2, P2, P2]
            X_group = X.repeat_interleave(group_size, dim=0)

            # Rollout
            # Note: We pass X_group (the expanded Batch)
            # Because sampling is stochastic (Temperature > 0), generated responses
            # will differ even for the exact same Prompt.
            states, log_probs, log_probs_reference, rewards, advantages = model.generate(
                X_group, self.block_size, self.device, self.block_size, 
                reward_model=reward_model, 
                hard_code_reward=self.config['hard_code_reward'], 
                ref_model=ref_model
            )
            
            if iter % 10 == 0:
                print(f"DEBUG Step {iter}: Raw Reward Mean: {rewards.mean().item():.8f}")

            # GRPO Advantage Calculation (Core Logic)
            # At this point, rewards shape is (B * G, 1)
            # We need to reshape it back to (B, G) to compute within-group mean and stats
            
            # Reshape
            rewards_view = rewards.view(-1, group_size) # Shape: (B, G)
            
            # Compute within-group mean (Baseline)
            mean_rewards = rewards_view.mean(dim=1, keepdim=True)
            
            # Compute within-group standard deviation (for normalization)
            std_rewards = rewards_view.std(dim=1, keepdim=True)
            
            # Compute Advantage
            # "Did I do better or worse compared to the group average?"
            # This is a "relative advantage" that removes the need for a Critic model
            grpo_advantages = (rewards_view - mean_rewards) / (std_rewards + 1e-8)
            
            # Flatten back to (B * G, 1) to match log_probs
            advantages = grpo_advantages.view(-1, 1)

            if iter % 10 == 0:
                # Debug print: Adv Std should be ~1.0 after normalization
                print(f"DEBUG Step {iter}: GRPO Adv Std: {advantages.std().item():.4f} (Should be ~1.0)")
            
            # Compute Loss
            # minus KL divergence
            approx_kl = log_probs.squeeze() - log_probs_reference.squeeze()
            
            # Policy Gradient Loss with GRPO advantages
            rets = advantages * log_probs.squeeze() - (kl_beta * approx_kl)
            actor_loss = -rets.sum()

            # Backward Pass
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_optimizer.step()

            # Logging
            rews_all.append(rewards.mean().detach().cpu().numpy())

            if iter % self.config['log_interval'] == 0:
                print(f"step {iter}: loss {actor_loss.item():.4f}, reward {rewards.mean().item():.6f}")

            if iter % self.config['eval_interval'] == 0:
                print(f'step {iter}: loss {actor_loss.item():.4f}, reward {rewards.mean().item():.8f}')
                print(f'--- Generated Text at iter {iter} ---')
                try:
                    # Use the original (non-replicated) X for a quick visual check
                    text_out = model.generate(X, self.block_size, self.device, self.block_size, reward_model=reward_model)[0]
                    decoded_text = self.enc.decode(text_out[0, :].tolist())
                    print(decoded_text)
                except Exception as e:
                    print(f"Decoding failed: {e}")
                print('-------------------------------------')