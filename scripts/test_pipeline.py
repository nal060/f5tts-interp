"""
Test script to verify the extraction-injection pipeline works correctly.
The key test: extract → inject (unchanged) → should produce identical output
"""

import sys
sys.path.insert(0, r"C:\Users\nicol\Desktop\slime\F5-TTS\src")
#sys.path.insert(0, "/workspace/F5-TTS/src")

import torch
from collections import defaultdict
import copy

# Import custom MMDiT model
from f5_tts.model.backbones.mmdit import MMDiT

def test_extraction_injection_pipeline():
    """
    Main test: Verify that extract → inject (unchanged) produces identical output
    """
    print("=" * 70)
    print("TESTING EXTRACTION-INJECTION PIPELINE")
    print("=" * 70)

    # ---- Configuration ----
    dim = 128
    depth = 4
    heads = 8
    dim_head = 16
    mel_dim = 100
    text_num_embeds = 256

    # Create model
    model = MMDiT(
        dim=dim,
        depth=depth,
        heads=heads,
        dim_head=dim_head,
        mel_dim=mel_dim,
        text_num_embeds=text_num_embeds,
    )
    model.eval()

    # Create test inputs
    batch_size = 1
    n = 50
    nt = 10

    x = torch.randn(batch_size, n, mel_dim)
    c = torch.randn(batch_size, n, mel_dim)
    timesteps = torch.randint(0, 1000, (batch_size,)).float()
    text = torch.randint(0, text_num_embeds, (batch_size, nt))

    # ---- TEST 1: Baseline Forward Pass ----
    print("\nTEST 1: Running baseline forward pass (no hooks)...")
    with torch.no_grad():
        baseline_output = model(x, c, text, timesteps)
    print(f" Baseline output shape: {baseline_output.shape}")
    print(f"  Baseline output mean: {baseline_output.mean():.6f}")
    print(f"  Baseline output std: {baseline_output.std():.6f}")

    # ---- TEST 2: Extract Activations ----
    print("\nTEST 2: Extracting activations...")
    extracted_activations = defaultdict(lambda: defaultdict(list))

    def extract_block_hook(idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                extracted_activations[idx]['block'].append(output[1].detach().clone())
            else:
                extracted_activations[idx]['block'].append(output.detach().clone())
        return hook_fn

    def extract_ffn_hook(idx):
        def hook_fn(module, input, output):
            extracted_activations[idx]['ffn'].append(output.detach().clone())
        return hook_fn

    # Register extraction hooks
    handles = []
    for idx, block in enumerate(model.transformer_blocks):
        h1 = block.register_forward_hook(extract_block_hook(idx))
        h2 = block.ff_x.register_forward_hook(extract_ffn_hook(idx))
        handles.extend([h1, h2])

    # Run forward pass with extraction
    with torch.no_grad():
        extraction_output = model(x, c, text, timesteps)

    # Remove hooks
    for h in handles:
        h.remove()

    print(f"Extracted activations from {depth} blocks")
    for idx in range(depth):
        print(f"  Block {idx}: block={extracted_activations[idx]['block'][0].shape}, "
              f"ffn={extracted_activations[idx]['ffn'][0].shape}")

    # Verify extraction didn't change output
    extraction_diff = (baseline_output - extraction_output).abs().max().item()
    print(f"\nExtraction output matches baseline: max diff = {extraction_diff:.2e}")
    assert extraction_diff < 1e-6, "Extraction changed the output!"

    # ---- TEST 3: Inject Unchanged Activations ----
    print("\nTEST 3: Injecting unchanged activations back...")

    # Copy extracted activations for injection
    activations_to_inject = copy.deepcopy(extracted_activations)

    def inject_block_hook(idx):
        def hook_fn(module, input, output):
            if activations_to_inject[idx]['block']:
                injected = activations_to_inject[idx]['block'][0]
                if isinstance(output, tuple):
                    return (output[0], injected)
                return injected
            return output
        return hook_fn

    def inject_ffn_hook(idx):
        def hook_fn(module, input, output):
            if activations_to_inject[idx]['ffn']:
                return activations_to_inject[idx]['ffn'][0]
            return output
        return hook_fn

    # Register injection hooks
    handles = []
    for idx, block in enumerate(model.transformer_blocks):
        h1 = block.register_forward_hook(inject_block_hook(idx))
        h2 = block.ff_x.register_forward_hook(inject_ffn_hook(idx))
        handles.extend([h1, h2])

    # Run forward pass with injection
    with torch.no_grad():
        injection_output = model(x, c, text, timesteps)

    # Remove hooks
    for h in handles:
        h.remove()

    print(f"Injected activations into {depth} blocks")

    # ---- MAIN VALIDATION ----
    print("\n" + "=" * 70)
    print("MAIN VALIDATION: Comparing outputs")
    print("=" * 70)

    # Compare baseline vs injection
    max_diff = (baseline_output - injection_output).abs().max().item()
    mean_diff = (baseline_output - injection_output).abs().mean().item()

    print(f"\nBaseline vs Injection (unchanged activations):")
    print(f"  Max difference:  {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")

    # Check if they're close enough
    if torch.allclose(baseline_output, injection_output, atol=1e-6):
        print("\nSUCCESS! Extract → Inject (unchanged) produces IDENTICAL output!")
        print("   The pipeline is working correctly.")
    else:
        print("\nFAILURE! Outputs don't match.")
        print("   Something is wrong with the extraction or injection.")
        return False

    # ---- TEST 4: Sanity Check - Modified Injection ----
    print("\n" + "=" * 70)
    print("TEST 4: Sanity check with modified activations")
    print("=" * 70)

    # Modify one activation
    modified_activations = copy.deepcopy(extracted_activations)
    modified_activations[1]['block'][0] = modified_activations[1]['block'][0] * 0.5

    # Register injection hooks with modified activations
    handles = []
    for idx, block in enumerate(model.transformer_blocks):
        h1 = block.register_forward_hook(inject_block_hook(idx))
        h2 = block.ff_x.register_forward_hook(inject_ffn_hook(idx))
        handles.extend([h1, h2])

    # Update activations to inject
    activations_to_inject = modified_activations

    # Run forward pass
    with torch.no_grad():
        modified_output = model(x, c, text, timesteps)

    # Remove hooks
    for h in handles:
        h.remove()

    # Check difference
    modified_diff = (baseline_output - modified_output).abs().max().item()
    print(f"After scaling Block 1 by 0.5:")
    print(f"  Max difference from baseline: {modified_diff:.6f}")

    if modified_diff > 1e-6:
        print("✓ Good! Modified activations produce different output (as expected)")
    else:
        print("✗ Warning: Modified activations didn't change output (unexpected)")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nConclusion:")
    print("The extraction-injection pipeline is working correctly!")
    print("Unchanged activations produce identical output")
    print("Modified activations produce different output")
    print("\nPipeline works:)")

    return True


if __name__ == "__main__":
    success = test_extraction_injection_pipeline()

    if not success:
        print("\n PIPELINE TEST FAILED!")
        sys.exit(1)
    else:
        print("\nPipeline is ready for research experiments!")