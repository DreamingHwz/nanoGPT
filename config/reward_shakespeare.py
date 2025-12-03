import time

out_dir = 'out-reward-shakespeare'
eval_interval = 50
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'shakespeare'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'reward'
init_from = 'resume'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 8
gradient_accumulation_steps = 4
max_iters = 500

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
