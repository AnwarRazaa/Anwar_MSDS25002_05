"""
UNet-style denoising model used by the DDPM diffusion pipeline.
Predicts the noise epsilon added to an image x_t at timestep t.
"""

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Maps a scalar timestep t to a vector embedding (as in Transformers / DDPM)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=device).float() / (half_dim - 1)
        )
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding


class ResidualBlock(nn.Module):
    """Conv block with GroupNorm + SiLU, conditioned on the timestep embedding.

    SiLU is used (as in the original DDPM/Improved-DDPM papers) since it
    behaves smoothly near zero, which helps gradient flow through the many
    stacked residual blocks of a diffusion UNet. GroupNorm is preferred over
    BatchNorm because training batches here are small.
    """

    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_emb):
        h = self.block1(x)
        time_term = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_term
        h = self.block2(h)
        return h + self.residual_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_emb_dim)
        self.pool = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, time_emb):
        skip = self.res(x, time_emb)
        down = self.pool(skip)
        return down, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_emb_dim)

    def forward(self, x, skip, time_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x, time_emb)


class DenoiseUNet(nn.Module):
    """Small UNet that predicts noise epsilon_theta(x_t, t).

    Channel widths are kept small (64/128/256) because the assignment trains
    on a handful of images per class on CPU; a full-size DDPM UNet would be
    far too slow to converge in that setting.
    """

    def __init__(self, image_channels=3, base_channels=64, time_emb_dim=256):
        super().__init__()

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.input_conv = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)

        self.down1 = DownBlock(base_channels, base_channels, time_emb_dim)
        self.down2 = DownBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down3 = DownBlock(base_channels * 2, base_channels * 4, time_emb_dim)

        self.bottleneck = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        self.up3 = UpBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        self.up2 = UpBlock(base_channels * 4, base_channels * 2, time_emb_dim)
        self.up1 = UpBlock(base_channels * 2, base_channels, time_emb_dim)

        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, image_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t):
        time_emb = self.time_embedding(t)

        x = self.input_conv(x)

        x, skip1 = self.down1(x, time_emb)
        x, skip2 = self.down2(x, time_emb)
        x, skip3 = self.down3(x, time_emb)

        x = self.bottleneck(x, time_emb)

        x = self.up3(x, skip3, time_emb)
        x = self.up2(x, skip2, time_emb)
        x = self.up1(x, skip1, time_emb)

        return self.output_conv(x)
