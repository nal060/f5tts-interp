import sys
sys.path.insert(0, r"C:\Users\nicol\Desktop\slime\F5-TTS\src")
#sys.path.insert(0, "/workspace/F5-TTS/src")
print("Python executable:", sys.executable)

import torch
from collections import defaultdict
import copy

# Import custom MMDiT model
from f5_tts.model.backbones.mmdit import MMDiT

# ---- Configuration ----
# Set model parameters (matching extract_block_outputs.py)
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
timesteps = torch.randint(0, 1000, (batch_size,)).float()  # Time steps as 1D tensor (b,)
text = torch.randint(0, text_num_embeds, (batch_size, nt))  # Dummy text tokens (b, nt)

# ---- First, extract the original outputs for comparison ----
print("=" * 60)
print("STEP 1: Extracting original outputs from the model")
print("=" * 60)

original_outputs = defaultdict(lambda: defaultdict(list))

def extract_block_hook(idx):
    def hook_fn(module, input, output):
        # MMDiTBlock returns (c, x), we want x (the audio sequence)
        if isinstance(output, tuple):
            original_outputs[idx]['block'].append(output[1].detach().clone())
        else:
            original_outputs[idx]['block'].append(output.detach().clone())
    return hook_fn

def extract_ffn_hook(idx):
    def hook_fn(module, input, output):
        original_outputs[idx]['ffn'].append(output.detach().clone())
    return hook_fn

# Register extraction hooks
extract_handles = []
for idx, block in enumerate(model.transformer_blocks):
    handle1 = block.register_forward_hook(extract_block_hook(idx))
    handle2 = block.ff_x.register_forward_hook(extract_ffn_hook(idx))
    extract_handles.append(handle1)
    extract_handles.append(handle2)

# Run original forward pass
with torch.no_grad():
    original_output = model(x, c, text, timesteps)

# Remove extraction hooks
for handle in extract_handles:
    handle.remove()

print("\nOriginal outputs extracted:")
for idx in range(depth):
    print(f"  Block {idx}:")
    for key in ['block', 'ffn']:
        if original_outputs[idx][key]:
            print(f"    {key}: {original_outputs[idx][key][0].shape}")

# ---- Now set up injection hooks ----
print("\n" + "=" * 60)
print("STEP 2: Setting up injection hooks")
print("=" * 60)

# Storage for injected activations
injected_activations = copy.deepcopy(original_outputs)

# modify activations here for experimentation:
# For example, to zero out block 2's output:
# injected_activations[2]['block'][0] = torch.zeros_like(injected_activations[2]['block'][0])

# Or scale FFN output of block 1:
# injected_activations[1]['ffn'][0] = injected_activations[1]['ffn'][0] * 2.0

# Injection hooks
def inject_block_hook(idx):
    def hook_fn(module, input, output):
        if injected_activations[idx]['block']:
            # Replace the output with our injected activation
            injected = injected_activations[idx]['block'][0]
            if isinstance(output, tuple):
                # Preserve the structure: (c, x) where we modify x
                return (output[0], injected)
            else:
                return injected
        return output
    return hook_fn

def inject_ffn_hook(idx):
    def hook_fn(module, input, output):
        if injected_activations[idx]['ffn']:
            # Replace FFN output with our injected activation
            return injected_activations[idx]['ffn'][0]
        return output
    return hook_fn

# Register injection hooks
injection_handles = []
for idx, block in enumerate(model.transformer_blocks):
    # Hook for block output injection
    handle1 = block.register_forward_hook(inject_block_hook(idx))
    # Hook for FFN output injection
    handle2 = block.ff_x.register_forward_hook(inject_ffn_hook(idx))
    injection_handles.append(handle1)
    injection_handles.append(handle2)

print("Injection hooks registered for all blocks and FFN layers")

# ---- Run forward pass with injection ----
print("\n" + "=" * 60)
print("STEP 3: Running forward pass with injected activations")
print("=" * 60)

with torch.no_grad():
    injected_output = model(x, c, text, timesteps)

# Remove injection hooks
for handle in injection_handles:
    handle.remove()

# ---- Validate the results ----
print("\n" + "=" * 60)
print("STEP 4: Validation - Comparing outputs")
print("=" * 60)

# Check if outputs match (when using unmodified injections, they should match)
if torch.allclose(original_output, injected_output, atol=1e-6):
    print("SUCCESS: Output with injected (unmodified) activations matches original output!")
    print(f"  Maximum difference: {(original_output - injected_output).abs().max().item():.2e}")
else:
    print("✗ MISMATCH: Outputs differ (this is expected if you modified the activations)")
    print(f"  Maximum difference: {(original_output - injected_output).abs().max().item():.2e}")

print(f"\nOriginal output shape: {original_output.shape}")
print(f"Injected output shape: {injected_output.shape}")

# ---- Example: Modifying specific activations ----
print("\n" + "=" * 60)
print("STEP 5: Example - Modifying and injecting activations")
print("=" * 60)

# Create modified activations (example: scale block 2's output by 0.5)
modified_activations = copy.deepcopy(original_outputs)
block_to_modify = 2
scale_factor = 0.5

if block_to_modify < depth:
    print(f"Scaling Block {block_to_modify} output by {scale_factor}")
    modified_activations[block_to_modify]['block'][0] = \
        modified_activations[block_to_modify]['block'][0] * scale_factor

    # Update injected activations
    injected_activations = modified_activations

    # Re-register injection hooks with modified activations
    injection_handles = []
    for idx, block in enumerate(model.transformer_blocks):
        handle1 = block.register_forward_hook(inject_block_hook(idx))
        handle2 = block.ff_x.register_forward_hook(inject_ffn_hook(idx))
        injection_handles.append(handle1)
        injection_handles.append(handle2)

    # Run forward pass with modified injection
    with torch.no_grad():
        modified_output = model(x, c, text, timesteps)

    # Remove hooks
    for handle in injection_handles:
        handle.remove()

    # Check the effect
    diff = (original_output - modified_output).abs().mean().item()
    print(f"Average difference after modification: {diff:.6f}")
    print(f"Maximum difference after modification: {(original_output - modified_output).abs().max().item():.6f}")

    if diff > 1e-6:
        print("Modification successfully propagated through the model!")
    else:
        print("Modification did not affect the output (unexpected)")

print("\n" + "=" * 60)
print("Injection script completed successfully!")
print("=" * 60)