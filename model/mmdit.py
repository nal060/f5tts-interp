"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

# Import future annotations for type hinting compatibility (Python 3.7+)
from __future__ import annotations

# Import torch and neural network modules
import torch
from torch import nn
# Import RotaryEmbedding from x_transformers for rotary positional encoding
from x_transformers.x_transformers import RotaryEmbedding

# Import various modules and layers from the F5-TTS codebase
from f5_tts.model.modules import (
    AdaLayerNorm_Final,           # Adaptive LayerNorm for final output
    ConvPositionEmbedding,        # Convolutional positional embedding for audio
    MMDiTBlock,                   # The main transformer block used in this model
    TimestepEmbedding,            # Embedding for time step conditioning
    get_pos_embed_indices,        # Helper for position embedding indices
    precompute_freqs_cis,         # Helper for precomputing sinusoidal frequencies
)

# ----------------------
# TEXT EMBEDDING MODULE
# ----------------------

class TextEmbedding(nn.Module):
    """
    Embeds text tokens into a continuous vector space, adds sinusoidal positional encoding,
    and optionally masks padding tokens.
    """
    def __init__(self, out_dim, text_num_embeds, mask_padding=True):
        super().__init__()
        # Embedding layer for text tokens (+1 for filler token at index 0)
        self.text_embed = nn.Embedding(text_num_embeds + 1, out_dim)  # will use 0 as filler token

        self.mask_padding = mask_padding  # Whether to mask filler and batch padding tokens

        self.precompute_max_pos = 1024  # Maximum sequence length for precomputed positional embeddings
        # Register a buffer for precomputed sinusoidal frequencies (not a parameter, but saved with the model)
        self.register_buffer("freqs_cis", precompute_freqs_cis(out_dim, self.precompute_max_pos), persistent=False)

    def forward(self, text: int["b nt"], drop_text=False) -> int["b nt d"]:  # noqa: F722
        """
        Args:
            text: Tensor of shape (batch, text_seq_len) with token indices. -1 is used for padding.
            drop_text: If True, zero out the text (for classifier-free guidance or ablation).
        Returns:
            Embedded text tensor of shape (batch, text_seq_len, out_dim)
        """
        text = text + 1  # Shift all tokens by 1 so 0 can be used as the filler token
        if self.mask_padding:
            text_mask = text == 0  # Mask for filler tokens (originally -1)

        if drop_text:  # If dropping text conditioning, zero out all tokens
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # Embed tokens: (b, nt) -> (b, nt, d)

        # Sinusoidal positional embedding
        batch_start = torch.zeros((text.shape[0],), dtype=torch.long)  # Start index for each batch
        batch_text_len = text.shape[1]  # Length of text sequence
        pos_idx = get_pos_embed_indices(batch_start, batch_text_len, max_pos=self.precompute_max_pos)  # (b, nt)
        text_pos_embed = self.freqs_cis[pos_idx]  # (b, nt, d)

        text = text + text_pos_embed  # Add positional embedding to token embedding

        if self.mask_padding:
            # Mask out filler tokens by setting their embeddings to zero
            text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)

        return text

# ---------------------------------------------
# AUDIO EMBEDDING MODULE (for input & cond audio)
# ---------------------------------------------

class AudioEmbedding(nn.Module):
    """
    Embeds audio features (input and conditioning audio) into a continuous space,
    applies a linear layer and convolutional positional embedding.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # Linear layer to combine input and conditioning audio (concatenated)
        self.linear = nn.Linear(2 * in_dim, out_dim)
        # Convolutional positional embedding
        self.conv_pos_embed = ConvPositionEmbedding(out_dim)

    def forward(self, x: float["b n d"], cond: float["b n d"], drop_audio_cond=False):  # noqa: F722
        """
        Args:
            x: Input audio features (batch, seq_len, in_dim)
            cond: Conditioning audio features (batch, seq_len, in_dim)
            drop_audio_cond: If True, zero out conditioning audio (for ablation)
        Returns:
            Embedded audio tensor (batch, seq_len, out_dim)
        """
        if drop_audio_cond:
            cond = torch.zeros_like(cond)  # Remove conditioning audio if specified
        x = torch.cat((x, cond), dim=-1)  # Concatenate input and cond along feature dim
        x = self.linear(x)                # Project to out_dim
        x = self.conv_pos_embed(x) + x    # Add convolutional positional embedding
        return x

# ---------------------------------------------------
# MAIN TRANSFORMER BACKBONE USING MM-DiT BLOCKS
# ---------------------------------------------------

class MMDiT(nn.Module):
    """
    Multi-modal DiT (Diffusion Transformer) backbone for TTS.
    Processes audio and text embeddings through a stack of MMDiTBlocks.
    """
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_mask_padding=True,
        qk_norm=None,
    ):
        super().__init__()

        # Time step embedding (for diffusion or temporal conditioning)
        self.time_embed = TimestepEmbedding(dim)
        # Text embedding module
        self.text_embed = TextEmbedding(dim, text_num_embeds, mask_padding=text_mask_padding)
        self.text_cond, self.text_uncond = None, None  # Caches for text embeddings (for efficiency)
        # Audio embedding module
        self.audio_embed = AudioEmbedding(mel_dim, dim)

        # Rotary positional embedding for attention (used in transformer blocks)
        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim  # Model dimension
        self.depth = depth  # Number of transformer blocks

        # Stack of MMDiTBlocks (the main transformer layers)
        self.transformer_blocks = nn.ModuleList(
            [
                MMDiTBlock(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    dropout=dropout,
                    ff_mult=ff_mult,
                    context_pre_only=i == depth - 1,  # Only last block uses context_pre_only
                    qk_norm=qk_norm,
                )
                for i in range(depth)
            ]
        )
        # Final adaptive layer normalization
        self.norm_out = AdaLayerNorm_Final(dim)
        # Final linear projection to mel spectrogram dimension
        self.proj_out = nn.Linear(dim, mel_dim)

        # Initialize weights for certain layers
        self.initialize_weights()

    def initialize_weights(self):
        """
        Custom weight initialization: zero out certain normalization and output layers.
        """
        # Zero-out AdaLN layers in each transformer block
        for block in self.transformer_blocks:
            nn.init.constant_(block.attn_norm_x.linear.weight, 0)
            nn.init.constant_(block.attn_norm_x.linear.bias, 0)
            nn.init.constant_(block.attn_norm_c.linear.weight, 0)
            nn.init.constant_(block.attn_norm_c.linear.bias, 0)

        # Zero-out final normalization and output projection layers
        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def get_input_embed(
        self,
        x,  # b n d: input audio features
        cond,  # b n d: conditioning audio features
        text,  # b nt: text tokens
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cache: bool = True,
    ):
        """
        Computes and returns the audio and text embeddings, with optional caching for efficiency.
        Args:
            x: Input audio features
            cond: Conditioning audio features
            text: Text tokens
            drop_audio_cond: If True, zero out cond audio
            drop_text: If True, zero out text
            cache: If True, cache text embeddings for reuse
        Returns:
            x: Embedded audio
            c: Embedded text
        """
        if cache:
            if drop_text:
                if self.text_uncond is None:
                    self.text_uncond = self.text_embed(text, drop_text=True)
                c = self.text_uncond
            else:
                if self.text_cond is None:
                    self.text_cond = self.text_embed(text, drop_text=False)
                c = self.text_cond
        else:
            c = self.text_embed(text, drop_text=drop_text)
        x = self.audio_embed(x, cond, drop_audio_cond=drop_audio_cond)

        return x, c

    def clear_cache(self):
        """
        Clears cached text embeddings (for when input text changes).
        """
        self.text_cond, self.text_uncond = None, None

    def forward(
        self,
        x: float["b n d"],  # Noised input audio (batch, seq_len, dim)
        cond: float["b n d"],  # Masked conditioning audio (batch, seq_len, dim)
        text: int["b nt"],  # Text tokens (batch, text_seq_len)
        time: float["b"] | float[""] ,  # Time step(s) for conditioning
        mask: bool["b n"] | None = None,  # Optional mask for sequence positions
        drop_audio_cond: bool = False,  # Whether to drop conditioning audio
        drop_text: bool = False,        # Whether to drop text
        cfg_infer: bool = False,        # If True, run classifier-free guidance inference
        cache: bool = False,            # If True, cache text embeddings
    ):
        """
        Forward pass for the MMDiT model.
        Args:
            x: Noised input audio
            cond: Masked conditioning audio
            text: Text tokens
            time: Time step(s) for conditioning
            mask: Optional mask for sequence positions
            drop_audio_cond: Whether to drop conditioning audio
            drop_text: Whether to drop text
            cfg_infer: If True, run classifier-free guidance inference (pack cond & uncond)
            cache: If True, cache text embeddings
        Returns:
            output: Predicted mel spectrogram or features
        """
        batch = x.shape[0]  # Batch size
        if time.ndim == 0:
            time = time.repeat(batch)  # Expand scalar time to batch size

        # Compute time embedding
        t = self.time_embed(time)
        if cfg_infer:  # If doing classifier-free guidance, run both cond and uncond in parallel
            x_cond, c_cond = self.get_input_embed(x, cond, text, drop_audio_cond=False, drop_text=False, cache=cache)
            x_uncond, c_uncond = self.get_input_embed(x, cond, text, drop_audio_cond=True, drop_text=True, cache=cache)
            x = torch.cat((x_cond, x_uncond), dim=0)  # Concatenate along batch
            c = torch.cat((c_cond, c_uncond), dim=0)
            t = torch.cat((t, t), dim=0)
            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
        else:
            x, c = self.get_input_embed(
                x, cond, text, drop_audio_cond=drop_audio_cond, drop_text=drop_text, cache=cache
            )

        seq_len = x.shape[1]      # Length of audio sequence
        text_len = text.shape[1]  # Length of text sequence
        rope_audio = self.rotary_embed.forward_from_seq_len(seq_len)  # Rotary embedding for audio
        rope_text = self.rotary_embed.forward_from_seq_len(text_len)  # Rotary embedding for text

        # Pass through each transformer block
        for block in self.transformer_blocks:
            c, x = block(x, c, t, mask=mask, rope=rope_audio, c_rope=rope_text)

        # Final normalization and projection
        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output
