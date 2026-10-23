"""Prompt autoencoder: maps a flattened prompt (K x d values) to a small 1D latent.

Used for long prompts (the paper uses it for CoOp prompts, K x d = 4 x 512 = 2048,
compressed to an 8 x 16 = 128-dim latent). Short prompts (e.g. a single 768-d
Textual Inversion token) can be modelled directly without it.

The encoder/decoder are a 1D variant of the Stable Diffusion autoencoder, adapted from
latent-diffusion (https://github.com/CompVis/latent-diffusion, MIT License).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(channels):
    return nn.GroupNorm(32, channels, eps=1e-6)


class DiagonalGaussian:
    def __init__(self, moments):
        self.mean, logvar = moments.chunk(2, dim=1)
        self.logvar = logvar.clamp(-30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self):
        return self.mean

    def kl(self):
        return 0.5 * torch.sum(self.mean ** 2 + self.logvar.exp() - 1.0 - self.logvar, dim=[1, 2])


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm2 = normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return (self.nin_shortcut(x) if hasattr(self, "nin_shortcut") else x) + h


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1)))


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class Encoder(nn.Module):
    def __init__(self, ch, ch_mult, num_res_blocks, z_channels, dropout=0.0):
        super().__init__()
        self.conv_in = nn.Conv1d(1, ch, 3, padding=1)
        self.down = nn.ModuleList()
        block_in = ch
        for level, mult in enumerate(ch_mult):
            down = nn.Module()
            down.block = nn.ModuleList()
            for _ in range(num_res_blocks):
                down.block.append(ResnetBlock(block_in, ch * mult, dropout))
                block_in = ch * mult
            if level != len(ch_mult) - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, dropout)
        self.mid.block_2 = ResnetBlock(block_in, block_in, dropout)
        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv1d(block_in, 2 * z_channels, 3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for down in self.down:
            for block in down.block:
                h = block(h)
            if hasattr(down, "downsample"):
                h = down.downsample(h)
        h = self.mid.block_2(self.mid.block_1(h))
        return self.conv_out(F.silu(self.norm_out(h)))


class Decoder(nn.Module):
    def __init__(self, ch, ch_mult, num_res_blocks, z_channels, dropout=0.0):
        super().__init__()
        block_in = ch * ch_mult[-1]
        self.conv_in = nn.Conv1d(z_channels, block_in, 3, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, dropout)
        self.mid.block_2 = ResnetBlock(block_in, block_in, dropout)
        self.up = nn.ModuleList()
        for level in reversed(range(len(ch_mult))):
            up = nn.Module()
            up.block = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                up.block.append(ResnetBlock(block_in, ch * ch_mult[level], dropout))
                block_in = ch * ch_mult[level]
            if level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)
        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv1d(block_in, 1, 3, padding=1)

    def forward(self, z):
        h = self.mid.block_2(self.mid.block_1(self.conv_in(z)))
        for up in reversed(self.up):
            for block in up.block:
                h = block(h)
            if hasattr(up, "upsample"):
                h = up.upsample(h)
        return self.conv_out(F.silu(self.norm_out(h)))


class PromptAutoencoder(nn.Module):
    """
    :param resolution: number of values in a flattened prompt (K x d).
    :param ch, ch_mult, num_res_blocks: width, per-level multipliers and depth of the
        encoder/decoder; the latent is `resolution / 2**(len(ch_mult) - 1)` long.
    :param embed_dim: number of latent channels.
    """

    def __init__(self, resolution, ch, ch_mult, num_res_blocks, embed_dim, kl_weight=1.0e-6, dropout=0.0):
        super().__init__()
        self.encoder = Encoder(ch, ch_mult, num_res_blocks, embed_dim, dropout)
        self.decoder = Decoder(ch, ch_mult, num_res_blocks, embed_dim, dropout)
        self.quant_conv = nn.Conv1d(2 * embed_dim, 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv1d(embed_dim, embed_dim, 1)
        self.logvar = nn.Parameter(torch.zeros(()))
        self.kl_weight = kl_weight
        self.resolution = resolution
        self.latent_shape = (embed_dim, resolution // 2 ** (len(ch_mult) - 1))

    def _flatten(self, x):
        x = x.reshape(x.shape[0], 1, -1).float()
        if x.shape[-1] != self.resolution:
            raise ValueError(f"Expected prompts with {self.resolution} values, got {x.shape[-1]}")
        return x

    def encode(self, x):
        """x: (B, K, d) or (B, K*d) prompts -> posterior over (B, *latent_shape) latents."""
        return DiagonalGaussian(self.quant_conv(self.encoder(self._flatten(x))))

    def decode(self, z, prompt_shape=None):
        """z: (B, *latent_shape) -> prompts, reshaped to (B, *prompt_shape) if given."""
        x = self.decoder(self.post_quant_conv(z))
        if prompt_shape is not None:
            x = x.reshape(x.shape[0], *prompt_shape)
        return x

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(z), posterior

    def loss(self, x):
        """L1 reconstruction NLL (learned log-variance) + weighted KL."""
        target = self._flatten(x)
        recon, posterior = self(x)
        rec_loss = torch.abs(target - recon)
        nll = (rec_loss / torch.exp(self.logvar) + self.logvar).sum() / target.shape[0]
        kl = posterior.kl().sum() / target.shape[0]
        return nll + self.kl_weight * kl, {"rec_loss": rec_loss.mean().item(), "kl": kl.item()}
