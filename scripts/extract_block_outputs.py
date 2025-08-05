import sys
sys.path.insert(0, r"C:\Users\nicol\Desktop\slime\F5-TTS\src")
sys.path.insert(0, "/workspace/F5-TTS/src")
print("Python executable:", sys.executable)

import torch
from collections import defaultdict

# Import your custom MMDiT model
from f5_tts.model.backbones.mmdit import MMDiT

# ---- Configuration ----
# Set model parameters 
dim = 128         # Model dimension
depth = 4         # Number of transformer blocks 
heads = 8         # Number of attention heads 
dim_head = 16     # Dimension per head 
mel_dim = 100     # Mel spectrogram dimension 
text_num_embeds = 256  # Number of text tokens 

# ---- Instantiate the model ----
model = MMDiT(
    dim=dim,
    depth=depth,
    heads=heads,
    dim_head=dim_head,
    mel_dim=mel_dim,
    text_num_embeds=text_num_embeds,
)
model.eval()  # Set to eval mode

# ---- Prepare dummy input ----
batch_size = 1
n = 50   # Audio sequence length 
nt = 10  # Text sequence length 

x = torch.randn(batch_size, n, mel_dim)      # Noised input audio (b, n, mel_dim)
c = torch.randn(batch_size, n, mel_dim)      # Masked cond audio (b, n, mel_dim)
timesteps = torch.randint(0, 1000, (batch_size,)).float()  # Time steps as 1D tensor (b,) - must have same dtype!
text = torch.randint(0, text_num_embeds, (batch_size, nt))  # Dummy text tokens (b, nt)

# ---- Set up hooks to capture outputs ----
outputs = defaultdict(lambda: defaultdict(list))

# 1. Output of each block
def block_hook(idx):
    def hook_fn(module, input, output):
        # MMDiTBlock returns (c, x), we want x (the audio sequence)
        if isinstance(output, tuple):
            outputs[idx]['block'].append(output[1].detach())
        else:
            outputs[idx]['block'].append(output.detach())
    return hook_fn

# 2. Output of each FFN layer (ff_x in each block)
def ffn_hook(idx):
    def hook_fn(module, input, output):
        outputs[idx]['ffn'].append(output.detach())
    return hook_fn

# 3, 4, 5. Output of self-attention, attention matrix, pre-projection output, and c_proj
def attn_hook(idx):
    def hook_fn(module, input, output):
        # output: (main_output, c_proj, attn_matrix, pre_proj_output)
        if isinstance(output, tuple) and len(output) == 4:
            outputs[idx]['attn'].append(output[0].detach())
            outputs[idx]['c_proj'].append(output[1].detach())
            outputs[idx]['attn_matrix'].append(output[2].detach())
            outputs[idx]['attn_preproj'].append(output[3].detach())
        elif isinstance(output, tuple):
            # If output is a tuple but not length 4, just store the first element
            outputs[idx]['attn'].append(output[0].detach())
        else:
            outputs[idx]['attn'].append(output.detach())
    return hook_fn

# Register hooks for each block
for idx, block in enumerate(model.transformer_blocks):
    block.register_forward_hook(block_hook(idx))
    block.ff_x.register_forward_hook(ffn_hook(idx))
    block.attn.register_forward_hook(attn_hook(idx))

# ---- Run a forward pass ----
with torch.no_grad():
    _ = model(x, c, text, timesteps)

# ---- Print the output shapes for each block ----
for idx in range(depth):
    print(f"\nBlock {idx} outputs:")
    for key in ['block', 'ffn', 'attn', 'c_proj', 'attn_matrix', 'attn_preproj']:
        outs = outputs[idx][key]
        if outs:
            print(f"  {key}: {outs[0].shape}")
        else:
            print(f"  {key}: not captured (check if Attention class returns all outputs)") 