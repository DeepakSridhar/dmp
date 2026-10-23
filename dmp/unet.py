"""Conditional 1D UNet used as the DMP denoiser: residual blocks over the prompt
(latent) sequence, with self-attention and cross-attention on the task text.

Adapted from latent-diffusion (https://github.com/CompVis/latent-diffusion, MIT License).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def timestep_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal embeddings of a 1-D tensor of timesteps -> (N, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(GEGLU(dim, dim * mult), nn.Dropout(dropout), nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    """Multi-head attention; attends to `context` if given, otherwise to `x` itself."""

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None):
        context = x if context is None else context
        q, k, v = (rearrange(t, "b n (h d) -> (b h) n d", h=self.heads)
                   for t in (self.to_q(x), self.to_k(context), self.to_v(context)))
        attn = (torch.einsum("b i d, b j d -> b i j", q, k) * self.scale).softmax(dim=-1)
        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        return self.to_out(rearrange(out, "(b h) n d -> b n (h d)", h=self.heads))


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, context_dim=None, dropout=0.0):
        super().__init__()
        self.attn1 = CrossAttention(dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout)
        self.attn2 = CrossAttention(dim, context_dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        return self.ff(self.norm3(x)) + x


class Transformer1D(nn.Module):
    """Transformer blocks applied to a (B, C, L) feature map."""

    def __init__(self, channels, n_heads, d_head, depth=1, context_dim=None, dropout=0.0):
        super().__init__()
        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.proj_in = nn.Conv1d(channels, inner_dim, 1)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, context_dim, dropout) for _ in range(depth)])
        self.proj_out = zero_module(nn.Conv1d(inner_dim, channels, 1))

    def forward(self, x, context=None):
        h = self.proj_in(self.norm(x)).transpose(1, 2)
        for block in self.transformer_blocks:
            h = block(h, context=context)
        return self.proj_out(h.transpose(1, 2)) + x


class ResBlock(nn.Module):
    """Residual block conditioned on the timestep embedding."""

    def __init__(self, channels, emb_channels, dropout, out_channels=None):
        super().__init__()
        out_channels = out_channels or channels
        self.in_layers = nn.Sequential(nn.GroupNorm(32, channels), nn.SiLU(),
                                       nn.Conv1d(channels, out_channels, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, out_channels))
        self.out_layers = nn.Sequential(nn.GroupNorm(32, out_channels), nn.SiLU(), nn.Dropout(dropout),
                                        zero_module(nn.Conv1d(out_channels, out_channels, 3, padding=1)))
        self.skip_connection = nn.Identity() if out_channels == channels else nn.Conv1d(channels, out_channels, 1)

    def forward(self, x, emb):
        h = self.in_layers(x) + self.emb_layers(emb)[..., None]
        return self.skip_connection(x) + self.out_layers(h)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv1d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class TimestepEmbedSequential(nn.Sequential):
    """Sequential that routes the timestep embedding and the context to the layers that use them."""

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb)
            elif isinstance(layer, Transformer1D):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class UNetModel(nn.Module):
    """
    :param image_size: length of the input sequence.
    :param in_channels / out_channels: channels of the input / output.
    :param model_channels: base channel count.
    :param num_res_blocks: residual blocks per resolution level.
    :param attention_resolutions: downsampling factors at which transformer blocks are used.
    :param channel_mult: channel multiplier of each resolution level.
    :param num_head_channels: channels per attention head.
    :param transformer_depth: transformer blocks per attention layer.
    :param context_dim: dimension of the cross-attention conditioning tokens.
    """

    def __init__(self, image_size, in_channels, model_channels, out_channels, num_res_blocks,
                 attention_resolutions, context_dim, channel_mult=(1, 2, 2, 2), num_head_channels=8,
                 transformer_depth=1, dropout=0.0):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels

        def attention(ch):
            return Transformer1D(ch, ch // num_head_channels, num_head_channels, transformer_depth, context_dim)

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(nn.Linear(model_channels, time_embed_dim), nn.SiLU(),
                                        nn.Linear(time_embed_dim, time_embed_dim))

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv1d(in_channels, model_channels, 3, padding=1))])
        skip_channels = [model_channels]
        ch, ds = model_channels, 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim, dropout, mult * model_channels)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(attention(ch))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                skip_channels.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch)))
                skip_channels.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout), attention(ch), ResBlock(ch, time_embed_dim, dropout))

        self.output_blocks = nn.ModuleList([])
        for level, mult in reversed(list(enumerate(channel_mult))):
            for i in range(num_res_blocks + 1):
                layers = [ResBlock(ch + skip_channels.pop(), time_embed_dim, dropout, mult * model_channels)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(attention(ch))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(nn.GroupNorm(32, ch), nn.SiLU(),
                                 zero_module(nn.Conv1d(model_channels, out_channels, 3, padding=1)))

    def forward(self, x, timesteps, context=None):
        """x: (B, C, L), timesteps: (B,), context: (B, tokens, context_dim) -> (B, C_out, L)."""
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        skips = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            skips.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = module(torch.cat([h, skips.pop()], dim=1), emb, context)
        return self.out(h)
