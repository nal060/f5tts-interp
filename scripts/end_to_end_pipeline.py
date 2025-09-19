"""
End-to-end activation extraction and injection pipeline for F5-TTS MMDiT model.
This script demonstrates:
1. Extracting activations from specific layers
2. Modifying those activations
3. Injecting them back into the model
4. Verifying the pipeline works correctly
"""

import sys
sys.path.insert(0, r"C:\Users\nicol\Desktop\slime\F5-TTS\src")
#sys.path.insert(0, "/workspace/F5-TTS/src")

import torch
import numpy as np
from collections import defaultdict
import copy
from typing import Dict, List, Tuple, Optional

# Import custom MMDiT model
from f5_tts.model.backbones.mmdit import MMDiT


class ActivationPipeline:
    """
    Complete pipeline for extracting and injecting activations in MMDiT model.
    """

    def __init__(self, model: MMDiT):
        """
        Initialize the pipeline with a model.

        Args:
            model: MMDiT model instance
        """
        self.model = model
        self.model.eval()
        self.depth = len(model.transformer_blocks)
        self.extracted_activations = defaultdict(lambda: defaultdict(list))
        self.handles = []

    def clear_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def extract_activations(self, x, c, text, timesteps) -> Dict:
        """
        Extract activations from model layers.

        Args:
            x: Noised input audio (b, n, mel_dim)
            c: Masked conditional audio (b, n, mel_dim)
            text: Text tokens (b, nt)
            timesteps: Time steps (b,)

        Returns:
            Dictionary containing extracted activations
        """
        self.clear_hooks()
        self.extracted_activations.clear()

        # Define extraction hooks
        def extract_block_hook(idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    self.extracted_activations[idx]['block'].append(output[1].detach().clone())
                else:
                    self.extracted_activations[idx]['block'].append(output.detach().clone())
            return hook_fn

        def extract_ffn_hook(idx):
            def hook_fn(module, input, output):
                self.extracted_activations[idx]['ffn'].append(output.detach().clone())
            return hook_fn

        # Register extraction hooks
        for idx, block in enumerate(self.model.transformer_blocks):
            handle1 = block.register_forward_hook(extract_block_hook(idx))
            handle2 = block.ff_x.register_forward_hook(extract_ffn_hook(idx))
            self.handles.extend([handle1, handle2])

        # Run forward pass
        with torch.no_grad():
            output = self.model(x, c, text, timesteps)

        self.clear_hooks()
        return dict(self.extracted_activations), output

    def inject_activations(self, x, c, text, timesteps,
                          activations_to_inject: Dict) -> torch.Tensor:
        """
        Inject modified activations into the model.

        Args:
            x, c, text, timesteps: Model inputs
            activations_to_inject: Dictionary of activations to inject

        Returns:
            Model output with injected activations
        """
        self.clear_hooks()

        # Define injection hooks
        def inject_block_hook(idx):
            def hook_fn(module, input, output):
                if idx in activations_to_inject and 'block' in activations_to_inject[idx]:
                    injected = activations_to_inject[idx]['block'][0]
                    if isinstance(output, tuple):
                        return (output[0], injected)
                    return injected
                return output
            return hook_fn

        def inject_ffn_hook(idx):
            def hook_fn(module, input, output):
                if idx in activations_to_inject and 'ffn' in activations_to_inject[idx]:
                    return activations_to_inject[idx]['ffn'][0]
                return output
            return hook_fn

        # Register injection hooks
        for idx, block in enumerate(self.model.transformer_blocks):
            handle1 = block.register_forward_hook(inject_block_hook(idx))
            handle2 = block.ff_x.register_forward_hook(inject_ffn_hook(idx))
            self.handles.extend([handle1, handle2])

        # Run forward pass with injection
        with torch.no_grad():
            output = self.model(x, c, text, timesteps)

        self.clear_hooks()
        return output

    def verify_pipeline(self, x, c, text, timesteps) -> bool:
        """
        Verify that extraction -> injection (without modification) produces same output.

        Args:
            x, c, text, timesteps: Model inputs

        Returns:
            True if pipeline verification passes
        """
        print("Verifying pipeline integrity...")

        # Step 1: Get original output
        with torch.no_grad():
            original_output = self.model(x, c, text, timesteps)

        # Step 2: Extract activations
        extracted, _ = self.extract_activations(x, c, text, timesteps)

        # Step 3: Inject same activations back
        injected_output = self.inject_activations(x, c, text, timesteps, extracted)

        # Step 4: Compare outputs
        max_diff = (original_output - injected_output).abs().max().item()
        passed = torch.allclose(original_output, injected_output, atol=1e-6)

        if passed:
            print(f"Pipeline verification PASSED! Max difference: {max_diff:.2e}")
        else:
            print(f"Pipeline verification FAILED! Max difference: {max_diff:.2e}")

        return passed


def run_experiments():
    """Run various experiments with the activation pipeline."""

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

    # Create pipeline
    pipeline = ActivationPipeline(model)

    # Prepare inputs
    batch_size = 1
    n = 50   # Audio sequence length
    nt = 10  # Text sequence length

    x = torch.randn(batch_size, n, mel_dim)
    c = torch.randn(batch_size, n, mel_dim)
    timesteps = torch.randint(0, 1000, (batch_size,)).float()
    text = torch.randint(0, text_num_embeds, (batch_size, nt))

    print("=" * 70)
    print("EXPERIMENT 1: Pipeline Verification")
    print("=" * 70)
    pipeline.verify_pipeline(x, c, text, timesteps)

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Block Output Scaling")
    print("=" * 70)

    # Extract original activations
    original_acts, original_output = pipeline.extract_activations(x, c, text, timesteps)

    # Test scaling different blocks
    for block_idx in range(depth):
        scale_factor = 0.5
        modified_acts = copy.deepcopy(original_acts)

        # Scale block output
        if modified_acts[block_idx]['block']:
            modified_acts[block_idx]['block'][0] *= scale_factor

            # Inject and measure effect
            modified_output = pipeline.inject_activations(x, c, text, timesteps, modified_acts)

            diff = (original_output - modified_output).abs().mean().item()
            max_diff = (original_output - modified_output).abs().max().item()

            print(f"Block {block_idx} scaled by {scale_factor}:")
            print(f"  Mean difference: {diff:.6f}, Max difference: {max_diff:.6f}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 3: FFN Layer Zeroing")
    print("=" * 70)

    for block_idx in range(depth):
        modified_acts = copy.deepcopy(original_acts)

        # Zero out FFN output
        if modified_acts[block_idx]['ffn']:
            modified_acts[block_idx]['ffn'][0] = torch.zeros_like(
                modified_acts[block_idx]['ffn'][0]
            )

            # Inject and measure effect
            modified_output = pipeline.inject_activations(x, c, text, timesteps, modified_acts)

            diff = (original_output - modified_output).abs().mean().item()
            max_diff = (original_output - modified_output).abs().max().item()

            print(f"Block {block_idx} FFN zeroed:")
            print(f"  Mean difference: {diff:.6f}, Max difference: {max_diff:.6f}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Activation Statistics")
    print("=" * 70)

    for block_idx in range(depth):
        print(f"\nBlock {block_idx}:")

        if original_acts[block_idx]['block']:
            block_act = original_acts[block_idx]['block'][0]
            print(f"  Block output - Mean: {block_act.mean():.4f}, "
                  f"Std: {block_act.std():.4f}, "
                  f"Shape: {block_act.shape}")

        if original_acts[block_idx]['ffn']:
            ffn_act = original_acts[block_idx]['ffn'][0]
            print(f"  FFN output  - Mean: {ffn_act.mean():.4f}, "
                  f"Std: {ffn_act.std():.4f}, "
                  f"Shape: {ffn_act.shape}")

    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Cross-layer Activation Swapping")
    print("=" * 70)

    if depth >= 2:
        # Swap activations between first and last blocks
        modified_acts = copy.deepcopy(original_acts)

        # Swap block outputs
        temp = modified_acts[0]['block'][0].clone()
        modified_acts[0]['block'][0] = modified_acts[depth-1]['block'][0].clone()
        modified_acts[depth-1]['block'][0] = temp

        # Inject swapped activations
        swapped_output = pipeline.inject_activations(x, c, text, timesteps, modified_acts)

        diff = (original_output - swapped_output).abs().mean().item()
        max_diff = (original_output - swapped_output).abs().max().item()

        print(f"Swapped Block 0 ↔ Block {depth-1} outputs:")
        print(f"  Mean difference: {diff:.6f}, Max difference: {max_diff:.6f}")

    print("\n" + "=" * 70)
    print("All experiments completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    print("Starting end-to-end activation pipeline experiments...")
    print("=" * 70)
    run_experiments()

    print("\n" + "=" * 70)