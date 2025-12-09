"""
Sample from a trained model and save results
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
from datetime import datetime

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster

# Create save directory
save_samples = True  # whether to save samples to file
sample_dir = os.path.join(out_dir, 'samples')  # directory to save samples
save_format = 'all'  # 'txt', 'json', 'pkl', or 'all'

exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

# Create save directory
if save_samples:
    os.makedirs(sample_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# Prepare save files
if save_samples:
    # prepare text file
    if save_format in ['txt', 'all']:
        txt_file = os.path.join(sample_dir, f'samples_{timestamp}.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("GENERATION CONFIGURATION\n")
            f.write("="*80 + "\n")
            f.write(f"Model: {init_from}\n")
            f.write(f"Number of samples: {num_samples}\n")
            f.write(f"Max new tokens: {max_new_tokens}\n")
            f.write(f"Temperature: {temperature}\n")
            f.write(f"Top-k: {top_k}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Prompt: {repr(start)}\n")
            f.write("="*80 + "\n\n")
    
    # prepare list to collect samples
    samples_list = []

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            generated_text = decode(y[0].tolist())  # CHANGE THIS LINE: store in variable
            
            # print to console
            print(generated_text)
            print('---------------')
            
            # Save to file
            if save_samples:
                # save to text file
                if save_format in ['txt', 'all']:
                    with open(txt_file, 'a', encoding='utf-8') as f:
                        f.write(f"{'='*80}\n")
                        f.write(f"SAMPLE {k+1}\n")
                        f.write(f"{'='*80}\n")
                        f.write(generated_text + "\n\n")
                
                # collect for structured formats
                if save_format in ['json', 'pkl', 'all']:
                    samples_list.append({
                        'id': k + 1,
                        'text': generated_text,
                        'tokens': y[0].tolist(),
                        'length': len(generated_text),
                        'num_tokens': len(y[0])
                    })

# Save JSON and PKL
if save_samples:
    import json  # import here to avoid dependency if not saving
    
    # save as JSON
    if save_format in ['json', 'all']:
        json_file = os.path.join(sample_dir, f'samples_{timestamp}.json')
        json_data = {
            'config': {
                'init_from': init_from,
                'num_samples': num_samples,
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'top_k': top_k,
                'seed': seed,
                'prompt': start
            },
            'samples': samples_list,
            'timestamp': timestamp
        }
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    # save as pickle
    if save_format in ['pkl', 'all']:
        pkl_file = os.path.join(sample_dir, f'samples_{timestamp}.pkl')
        pkl_data = {
            'config': {
                'init_from': init_from,
                'num_samples': num_samples,
                'max_new_tokens': max_new_tokens,
                'temperature': temperature,
                'top_k': top_k,
                'seed': seed,
                'prompt': start
            },
            'samples': samples_list,
            'timestamp': timestamp
        }
        with open(pkl_file, 'wb') as f:
            pickle.dump(pkl_data, f)
    
    # print summary
    print("\n" + "="*80)
    print("SAVED SAMPLES TO:")
    print("="*80)
    if save_format in ['txt', 'all']:
        print(f"  Text: {txt_file}")
    if save_format in ['json', 'all']:
        print(f"  JSON: {json_file}")
    if save_format in ['pkl', 'all']:
        print(f"  Pickle: {pkl_file}")
    print("="*80 + "\n")