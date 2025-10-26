import time

out_dir = 'out-poems'
eval_interval = 5
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'poems'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'poems'
data_dir = "data/poems"
init_from = 'resume'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# # match the small model config from train_shakespeare_char.py
# n_layer = 4
# n_head = 2
# n_embd = 384
# dropout = 0.2

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# poems has 400000 tokens, so 1 epoch ~= 12.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 30

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False