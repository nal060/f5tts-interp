"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""
# flake8: noqa

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from torch import nn
from x_transformers.x_transformers import apply_rotary_pos_emb

from f5_tts.model.utils import is_package_available

# -----------------------------
# RAW WAVE TO MEL SPECTROGRAM
# -----------------------------

# These functions/classes convert raw audio (waveform) to mel spectrograms, which are the main input features for the model.
# Input: waveform (b, nw) where b=batch, nw=raw wave length
# Output: mel spectrogram (b, n_mel_channels, n_frames)

mel_basis_cache = {}
hann_window_cache = {}

def get_bigvgan_mel_spectrogram(
    waveform,
    n_fft=1024,
    n_mel_channels=100,
    target_sample_rate=24000,
    hop_length=256,
    win_length=1024,
    fmin=0,
    fmax=None,
    center=False,
):
    """
    Convert a waveform to a mel spectrogram using the BigVGAN method.
    Args:
        waveform: (b, nw) tensor of raw audio
        n_fft, n_mel_channels, ...: spectrogram parameters
    Returns:
        mel_spec: (b, n_mel_channels, n_frames)
    """
    device = waveform.device
    key = f"{n_fft}_{n_mel_channels}_{target_sample_rate}_{hop_length}_{win_length}_{fmin}_{fmax}_{device}"

    if key not in mel_basis_cache:
        mel = librosa_mel_fn(sr=target_sample_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=fmin, fmax=fmax)
        mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)
        hann_window_cache[key] = torch.hann_window(win_length).to(device)

    mel_basis = mel_basis_cache[key]
    hann_window = hann_window_cache[key]

    padding = (n_fft - hop_length) // 2
    # Pad waveform for STFT
    waveform = torch.nn.functional.pad(waveform.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)

    # Compute STFT
    spec = torch.stft(
        waveform,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )  # (b, freq_bins, n_frames)
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)  # (b, freq_bins, n_frames)

    mel_spec = torch.matmul(mel_basis, spec)  # (b, n_mel_channels, n_frames)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))  # (b, n_mel_channels, n_frames)

    return mel_spec

def get_vocos_mel_spectrogram(
    waveform,
    n_fft=1024,
    n_mel_channels=100,
    target_sample_rate=24000,
    hop_length=256,
    win_length=1024,
):
    """
    Convert a waveform to a mel spectrogram using the Vocos method.
    Args:
        waveform: (b, nw) tensor of raw audio
        n_fft, n_mel_channels, ...: spectrogram parameters
    Returns:
        mel: (b, n_mel_channels, n_frames)
    """
    mel_stft = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mel_channels,
        power=1,
        center=True,
        normalized=False,
        norm=None,
    ).to(waveform.device)
    if len(waveform.shape) == 3:
        waveform = waveform.squeeze(1)  # (b, 1, nw) -> (b, nw)

    assert len(waveform.shape) == 2

    mel = mel_stft(waveform)  # (b, n_mel_channels, n_frames)
    mel = mel.clamp(min=1e-5).log()  # (b, n_mel_channels, n_frames)
    return mel

class MelSpec(nn.Module):
    """
    Converts raw waveform (b, nw) to mel spectrogram (b, n_mel_channels, n_frames)
    """
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24_000,
        mel_spec_type="vocos",
    ):
        """
        Args:
            n_fft, hop_length, win_length, n_mel_channels, target_sample_rate: spectrogram params
            mel_spec_type: 'vocos' or 'bigvgan'
        """
        super().__init__()
        assert mel_spec_type in ["vocos", "bigvgan"], print("We only support two extract mel backend: vocos or bigvgan")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.target_sample_rate = target_sample_rate

        if mel_spec_type == "vocos":
            self.extractor = get_vocos_mel_spectrogram
        elif mel_spec_type == "bigvgan":
            self.extractor = get_bigvgan_mel_spectrogram

        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    def forward(self, wav):
        """
        Args:
            wav: (b, nw) raw waveform
        Returns:
            mel: (b, n_mel_channels, n_frames)
        """
        if self.dummy.device != wav.device:
            self.to(wav.device)

        mel = self.extractor(
            waveform=wav,
            n_fft=self.n_fft,
            n_mel_channels=self.n_mel_channels,
            target_sample_rate=self.target_sample_rate,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )  # (b, n_mel_channels, n_frames)

        return mel

# -----------------------------
# POSITIONAL EMBEDDINGS
# -----------------------------

class SinusPositionEmbedding(nn.Module):
    """
    Generates sinusoidal position embeddings for a sequence.
    Input: x (b,) or (b, 1) (e.g., time step)
    Output: (b, dim)
    """
    def __init__(self, dim):
        """
        Args:
            dim: embedding dimension
        """
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        """
        Args:
            x: (b,) or (b, 1) input (e.g., time step)
            scale: scaling factor for position
        Returns:
            emb: (b, dim) sinusoidal embedding
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # (b, dim)
        return emb

class ConvPositionEmbedding(nn.Module):
    """
    Applies convolutional positional embedding to a sequence.
    Input: x (b, n, d)
    Output: (b, n, d)
    """
    def __init__(self, dim, kernel_size=31, groups=16):
        """
        Args:
            dim: feature dimension
            kernel_size: convolution kernel size
            groups: number of groups for grouped conv
        """
        super().__init__()
        assert kernel_size % 2 != 0
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(self, x: float["b n d"], mask: bool["b n"] | None = None):
        """
        Args:
            x: (b, n, d) input sequence
            mask: (b, n) optional mask
        Returns:
            out: (b, n, d) sequence with positional embedding
        """
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)

        x = x.permute(0, 2, 1)  # (b, d, n)
        x = self.conv1d(x)      # (b, d, n)
        out = x.permute(0, 2, 1)  # (b, n, d)

        if mask is not None:
            out = out.masked_fill(~mask, 0.0)

        return out

# -----------------------------
# ROTARY POSITIONAL EMBEDDING
# -----------------------------

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0):
    """
    Precompute rotary positional embedding frequencies.
    Args:
        dim: embedding dimension
        end: sequence length
        theta: base frequency
        theta_rescale_factor: scaling for NTK
    Returns:
        (end, dim) tensor of rotary frequencies
    """
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # (end,)
    freqs = torch.outer(t, freqs).float()  # (end, dim//2)
    freqs_cos = torch.cos(freqs)  # (end, dim//2)
    freqs_sin = torch.sin(freqs)  # (end, dim//2)
    return torch.cat([freqs_cos, freqs_sin], dim=-1)  # (end, dim)

def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    """
    Compute position indices for embedding lookup.
    Args:
        start: (b,) start index for each batch
        length: int, sequence length
        max_pos: int, max position
        scale: float, scaling factor
    Returns:
        pos: (b, length) indices
    """
    scale = scale * torch.ones_like(start, dtype=torch.float32)
    pos = (
        start.unsqueeze(1)
        + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
    )
    # avoid extra long error.
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos

# -----------------------------
# NORMALIZATION LAYERS
# -----------------------------

class GRN(nn.Module):
    """
    Global Response Normalization (like Instance Norm)
    Input: x (b, n, d)
    Output: (b, n, d)
    """
    def __init__(self, dim):
        """
        Args:
            dim: feature dimension
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        """
        Args:
            x: (b, n, d)
        Returns:
            normalized x: (b, n, d)
        """
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# ConvNeXt-V2 Block https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/convnextv2.py
# ref: https://github.com/bfs18/e2_tts/blob/main/rfwave/modules.py#L108


class ConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)  # b n d -> b d n
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # b d n -> b n d
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


# RMSNorm


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization
    Input: x (b, n, d)
    Output: (b, n, d)
    """
    def __init__(self, dim: int, eps: float):
        """
        Args:
            dim: feature dimension
            eps: epsilon for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.native_rms_norm = float(torch.__version__[:3]) >= 2.4

    def forward(self, x):
        """
        Args:
            x: (b, n, d)
        Returns:
            normalized x: (b, n, d)
        """
        if self.native_rms_norm:
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                x = x.to(self.weight.dtype)
            x = F.rms_norm(x, normalized_shape=(x.shape[-1],), weight=self.weight, eps=self.eps)
        else:
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                x = x.to(self.weight.dtype)
            x = x * self.weight
        return x


# AdaLayerNorm
# return with modulated x for attn input, and params for later mlp modulation


class AdaLayerNorm(nn.Module):
    """
    Adaptive LayerNorm for attention input.
    Input: x (b, n, d), emb (b, d)
    Output: normed x (b, n, d), and 5 gates/shifts/scales (b, d) each
    """
    def __init__(self, dim):
        """
        Args:
            dim: feature dimension
        """
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        """
        Args:
            x: (b, n, d)
            emb: (b, d)
        Returns:
            normed x: (b, n, d), and 5 gates/shifts/scales (b, d) each
        """
        emb = self.linear(self.silu(emb))  # (b, d*6)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)  # each (b, d)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]  # (b, n, d)
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp  # (b, n, d), (b, d), ...
    
# AdaLayerNorm for final layer
# return only with modulated x for attn input, cuz no more mlp modulation

class AdaLayerNorm_Final(nn.Module):
    """
    Adaptive LayerNorm for final layer (no MLP modulation)
    Input: x (b, n, d), emb (b, d)
    Output: normed x (b, n, d)
    """
    def __init__(self, dim):
        """
        Args:
            dim: feature dimension
        """
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        """
        Args:
            x: (b, n, d)
            emb: (b, d)
        Returns:
            normed x: (b, n, d)
        """
        emb = self.linear(self.silu(emb))  # (b, d*2)
        scale, shift = torch.chunk(emb, 2, dim=1)  # (b, d), (b, d)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]  # (b, n, d)
        return x

# -----------------------------
# FEEDFORWARD LAYER
# -----------------------------

class FeedForward(nn.Module):
    """
    Standard FeedForward network used in transformer blocks.
    Input: x (b, n, d)
    Output: (b, n, d)
    """
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = "none"):
        """
        Args:
            dim: input/output feature dimension
            dim_out: output feature dimension (default: same as input)
            mult: expansion factor for hidden layer
            dropout: dropout rate
            approximate: GELU approximation
        """
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        """
        Args:
            x: (b, n, d)
        Returns:
            (b, n, d)
        """
        return self.ff(x)  # (b, n, d)

# -----------------------------
# ATTENTION LAYER
# -----------------------------

class Attention(nn.Module):
    """
    Multi-head self-attention or joint attention.
    Input: x (b, n, d), optionally c (context, b, nt, d)
    Output: (b, n, d) or (b, n, d), (b, nt, d) for joint attention
    """
    def __init__(
        self,
        processor: JointAttnProcessor | AttnProcessor,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,  # if not None -> joint attention
        context_pre_only: bool = False,
        qk_norm: Optional[str] = None,
    ):
        """
        Args:
            processor: attention processor (joint or standard)
            dim: input feature dimension
            heads: number of attention heads
            dim_head: dimension per head
            dropout: dropout rate
            context_dim: context feature dimension (for joint attention)
            context_pre_only: if True, only prenorm context
            qk_norm: normalization for q/k
        """
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Attention equires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.processor = processor

        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout

        self.context_dim = context_dim
        self.context_pre_only = context_pre_only

        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        if qk_norm is None:
            self.q_norm = None
            self.k_norm = None
        elif qk_norm == "rms_norm":
            self.q_norm = RMSNorm(dim_head, eps=1e-6)
            self.k_norm = RMSNorm(dim_head, eps=1e-6)
        else:
            raise ValueError(f"Unimplemented qk_norm: {qk_norm}")

        if self.context_dim is not None:
            self.to_q_c = nn.Linear(context_dim, self.inner_dim)
            self.to_k_c = nn.Linear(context_dim, self.inner_dim)
            self.to_v_c = nn.Linear(context_dim, self.inner_dim)
            if qk_norm is None:
                self.c_q_norm = None
                self.c_k_norm = None
            elif qk_norm == "rms_norm":
                self.c_q_norm = RMSNorm(dim_head, eps=1e-6)
                self.c_k_norm = RMSNorm(dim_head, eps=1e-6)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))

        if self.context_dim is not None and not self.context_pre_only:
            self.to_out_c = nn.Linear(self.inner_dim, context_dim)

    def forward(
        self,
        x: float["b n d"],  # noised input x
        c: float["b n d"] = None,  # context c
        mask: bool["b n"] | None = None,
        rope=None,  # rotary position embedding for x
        c_rope=None,  # rotary position embedding for c
    ) -> torch.Tensor:
        """
        Args:
            x: (b, n, d)
            c: (b, nt, d) or None
            mask: (b, n) or None
            rope: rotary embedding for x
            c_rope: rotary embedding for c
        Returns:
            (b, n, d) or (b, n, d), (b, nt, d)
        """
        if c is not None:
            return self.processor(self, x, c=c, mask=mask, rope=rope, c_rope=c_rope)
        else:
            return self.processor(self, x, mask=mask, rope=rope)


# Attention processor

if is_package_available("flash_attn"):
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn import flash_attn_varlen_func, flash_attn_func


class AttnProcessor:
    def __init__(
        self,
        pe_attn_head: int | None = None,  # number of attention head to apply rope, None for all
        attn_backend: str = "torch",  # "torch" or "flash_attn"
        attn_mask_enabled: bool = True,
    ):
        if attn_backend == "flash_attn":
            assert is_package_available("flash_attn"), "Please install flash-attn first."

        self.pe_attn_head = pe_attn_head
        self.attn_backend = attn_backend
        self.attn_mask_enabled = attn_mask_enabled

    def __call__(
        self,
        attn: Attention,
        x: float["b n d"],  # noised input x
        mask: bool["b n"] | None = None,
        rope=None,  # rotary position embedding
    ) -> torch.FloatTensor:
        batch_size = x.shape[0]

        # `sample` projections
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # qk norm
        if attn.q_norm is not None:
            query = attn.q_norm(query)
        if attn.k_norm is not None:
            key = attn.k_norm(key)

        # apply rotary position embedding
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)

            if self.pe_attn_head is not None:
                pn = self.pe_attn_head
                query[:, :pn, :, :] = apply_rotary_pos_emb(query[:, :pn, :, :], freqs, q_xpos_scale)
                key[:, :pn, :, :] = apply_rotary_pos_emb(key[:, :pn, :, :], freqs, k_xpos_scale)
            else:
                query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
                key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        if self.attn_backend == "torch":
            # mask. e.g. inference got a batch with different target durations, mask out the padding
            if self.attn_mask_enabled and mask is not None:
                attn_mask = mask
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
                attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
            else:
                attn_mask = None
            x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
            x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        elif self.attn_backend == "flash_attn":
            query = query.transpose(1, 2)  # [b, h, n, d] -> [b, n, h, d]
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            if self.attn_mask_enabled and mask is not None:
                query, indices, q_cu_seqlens, q_max_seqlen_in_batch, _ = unpad_input(query, mask)
                key, _, k_cu_seqlens, k_max_seqlen_in_batch, _ = unpad_input(key, mask)
                value, _, _, _, _ = unpad_input(value, mask)
                x = flash_attn_varlen_func(
                    query,
                    key,
                    value,
                    q_cu_seqlens,
                    k_cu_seqlens,
                    q_max_seqlen_in_batch,
                    k_max_seqlen_in_batch,
                )
                x = pad_input(x, indices, batch_size, q_max_seqlen_in_batch)
                x = x.reshape(batch_size, -1, attn.heads * head_dim)
            else:
                x = flash_attn_func(query, key, value, dropout_p=0.0, causal=False)
                x = x.reshape(batch_size, -1, attn.heads * head_dim)

        x = x.to(query.dtype)

        # linear proj
        x = attn.to_out[0](x)
        # dropout
        x = attn.to_out[1](x)

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)

        return x


# Joint Attention processor for MM-DiT
# modified from diffusers/src/diffusers/models/attention_processor.py


class JointAttnProcessor:
    def __init__(self):
        pass

    def __call__(
        self,
        attn: Attention,
        x: float["b n d"],  # noised input x
        c: float["b nt d"] = None,  # context c, here text
        mask: bool["b n"] | None = None,
        rope=None,  # rotary position embedding for x
        c_rope=None,  # rotary position embedding for c
    ) -> torch.FloatTensor:
        residual = x
        batch_size = c.shape[0]

        # `sample` projections
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        # `context` projections
        c_query = attn.to_q_c(c)
        c_key = attn.to_k_c(c)
        c_value = attn.to_v_c(c)

        # attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_query = c_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_key = c_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        c_value = c_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # qk norm
        if attn.q_norm is not None:
            query = attn.q_norm(query)
        if attn.k_norm is not None:
            key = attn.k_norm(key)
        if attn.c_q_norm is not None:
            c_query = attn.c_q_norm(c_query)
        if attn.c_k_norm is not None:
            c_key = attn.c_k_norm(c_key)

        # apply rope for context and noised input independently
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)
        if c_rope is not None:
            freqs, xpos_scale = c_rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            c_query = apply_rotary_pos_emb(c_query, freqs, q_xpos_scale)
            c_key = apply_rotary_pos_emb(c_key, freqs, k_xpos_scale)

        # joint attention
        query = torch.cat([query, c_query], dim=2)
        key = torch.cat([key, c_key], dim=2)
        value = torch.cat([value, c_value], dim=2)

        # mask. e.g. inference got a batch with different target durations, mask out the padding
        if mask is not None:
            attn_mask = F.pad(mask, (0, c.shape[1]), value=True)  # no mask for c (text)
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # 'b n -> b 1 1 n'
            attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        # --- attention matrix computation ---
        # Compute raw attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)  # (b, heads, seq, seq)
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))
        attn_matrix = torch.softmax(attn_scores, dim=-1)  # (b, heads, seq, seq)

        # Compute attention output (pre-projection)
        pre_proj_output = torch.matmul(attn_matrix, value)  # (b, heads, seq, head_dim)
        pre_proj_output_cat = pre_proj_output.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        pre_proj_output_cat = pre_proj_output_cat.to(query.dtype)

        # Split the attention outputs.
        x, c = (
            pre_proj_output_cat[:, : residual.shape[1]],
            pre_proj_output_cat[:, residual.shape[1] :],
        )

        # linear proj
        x_proj = attn.to_out[0](x)
        # dropout
        x_proj = attn.to_out[1](x_proj)
        if not attn.context_pre_only:
            c_proj = attn.to_out_c(c)
        else:
            c_proj = c

        if mask is not None:
            mask = mask.unsqueeze(-1)
            x_proj = x_proj.masked_fill(~mask, 0.0)
            # c_proj = c_proj.masked_fill(~mask, 0.)  # no mask for c (text)

        # Return x_proj, c_proj, attn_matrix, pre_proj_output_cat
        return x_proj, c_proj, attn_matrix, pre_proj_output_cat


# DiT Block


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head,
        ff_mult=4,
        dropout=0.1,
        qk_norm=None,
        pe_attn_head=None,
        attn_backend="torch",  # "torch" or "flash_attn"
        attn_mask_enabled=True,
    ):
        super().__init__()

        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(
            processor=AttnProcessor(
                pe_attn_head=pe_attn_head,
                attn_backend=attn_backend,
                attn_mask_enabled=attn_mask_enabled,
            ),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            qk_norm=qk_norm,
        )

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask=None, rope=None):  # x: noised input, t: time embedding
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # attention
        attn_output = self.attn(x=norm, mask=mask, rope=rope)

        # process attention output for input x
        x = x + gate_msa.unsqueeze(1) * attn_output

        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x


# MMDiT Block https://arxiv.org/abs/2403.03206

class MMDiTBlock(nn.Module):
    r"""
    Core block for MM-DiT transformer.
    Input: x (b, n, d), c (b, nt, d), t (b, d)
    Output: c (b, nt, d) or None, x (b, n, d)
    """
    def __init__(
        self, dim, heads, dim_head, ff_mult=4, dropout=0.1, context_dim=None, context_pre_only=False, qk_norm=None
    ):
        """
        Args:
            dim: input feature dimension
            heads: number of attention heads
            dim_head: dimension per head
            ff_mult: expansion factor for FFN
            dropout: dropout rate
            context_dim: context feature dimension
            context_pre_only: if True, only prenorm context (last block)
            qk_norm: normalization for q/k
        """
        super().__init__()
        if context_dim is None:
            context_dim = dim
        self.context_pre_only = context_pre_only

        self.attn_norm_c = AdaLayerNorm_Final(context_dim) if context_pre_only else AdaLayerNorm(context_dim)
        self.attn_norm_x = AdaLayerNorm(dim)
        self.attn = Attention(
            processor=JointAttnProcessor(),
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            context_dim=context_dim,
            context_pre_only=context_pre_only,
            qk_norm=qk_norm,
        )

        if not context_pre_only:
            self.ff_norm_c = nn.LayerNorm(context_dim, elementwise_affine=False, eps=1e-6)
            self.ff_c = FeedForward(dim=context_dim, mult=ff_mult, dropout=dropout, approximate="tanh")
        else:
            self.ff_norm_c = None
            self.ff_c = None
        self.ff_norm_x = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_x = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, c, t, mask=None, rope=None, c_rope=None):
        """
        Args:
            x: (b, n, d)
            c: (b, nt, d)
            t: (b, d)
            mask: (b, n) or None
            rope: rotary embedding for x
            c_rope: rotary embedding for c
        Returns:
            c: (b, nt, d) or None
            x: (b, n, d)
        """
        # 1. Pre-norm & modulation for attention input
        if self.context_pre_only:
            norm_c = self.attn_norm_c(c, t)  # (b, nt, d)
        else:
            norm_c, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.attn_norm_c(c, emb=t)  # (b, nt, d), (b, d) x4
        norm_x, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp = self.attn_norm_x(x, emb=t)  # (b, n, d), (b, d) x4

        # 2. Attention
        x_attn_output, c_attn_output = self.attn(x=norm_x, c=norm_c, mask=mask, rope=rope, c_rope=c_rope)  # (b, n, d), (b, nt, d)

        # 3. Process attention output for context c
        if self.context_pre_only:
            c = None  # Last layer: no more context
        else:
            c = c + c_gate_msa.unsqueeze(1) * c_attn_output  # (b, nt, d)
            norm_c = self.ff_norm_c(c) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]  # (b, nt, d)
            c_ff_output = self.ff_c(norm_c)  # (b, nt, d)
            c = c + c_gate_mlp.unsqueeze(1) * c_ff_output  # (b, nt, d)

        # 4. Process attention output for input x
        x = x + x_gate_msa.unsqueeze(1) * x_attn_output  # (b, n, d)
        norm_x = self.ff_norm_x(x) * (1 + x_scale_mlp[:, None]) + x_shift_mlp[:, None]  # (b, n, d)
        x_ff_output = self.ff_x(norm_x)  # (b, n, d)
        x = x + x_gate_mlp.unsqueeze(1) * x_ff_output  # (b, n, d)

        return c, x  # (b, nt, d) or None, (b, n, d)

# time step conditioning embedding

class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: float["b"]):
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)  # b d


# what needs to be extracted from the model?
# output of each block (after each MMDiT block), output of each FF layer (feedforward inside each block)
#output of each self-attention layer before res connection (raw-attention output)
#self-attention matriz (softmax(qk^T)for each head)
# output of self-attention before the output projection (W^O), the concatenated head outputs before w^o)