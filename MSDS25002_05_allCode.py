# ============================================================
# MSDS25002_05_allCode.py
# Combined code from MSDS25002_05_model.py and MSDS25002_05.py
# (per assignment requirement - submit as a single .py file)
# ============================================================

# ---- Contents of MSDS25002_05_model.py ----
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

# ---- Contents of MSDS25002_05.py ----
"""
Image Generation Using Diffusion Models (DDPM) - Assignment 5 (Bonus)
Anwar, MSDS25002

Trains a denoising diffusion probabilistic model on a small subset of the
animal-classes dataset (5 classes, ~20 images each, as suggested by the
assignment spec) and generates new images by running the learned reverse
process starting from pure Gaussian noise.

Usage:
    python MSDS25002_05.py --data_dir animal_data --classes Cat Dog Bird Lion Tiger \
        --images_per_class 20 --epochs 300 --img_size 64 --timesteps 1000

See Readme.txt for the full list of command line arguments.
"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



# ---------------------------------------------------------------------------
# 1. Data Loader
# ---------------------------------------------------------------------------
class AnimalDataset(Dataset):
    """Reads a fixed number of images from a chosen set of animal classes.

    Each class lives in its own sub-folder under `root_dir` (e.g.
    root_dir/Cat/*.jpg). Only `images_per_class` images are sampled per class
    so that training stays feasible on CPU, as suggested by the assignment.
    """

    def __init__(self, root_dir, classes, images_per_class=20, img_size=64, seed=42):
        self.image_paths = []
        rng = random.Random(seed)

        for class_name in classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(f"Class folder not found: {class_dir}")

            files = [
                f for f in os.listdir(class_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            rng.shuffle(files)
            chosen = files[:images_per_class]
            self.image_paths.extend(os.path.join(class_dir, f) for f in chosen)

        if not self.image_paths:
            raise RuntimeError("No images found - check --data_dir and --classes.")

        # Scale pixel values to [-1, 1], which is the convention used in DDPM
        # so that the Gaussian noise added during the forward process lives
        # on the same scale as the (zero-centered) image data.
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)


def unnormalize(x):
    """Maps a tensor from [-1, 1] back to [0, 1] for display/saving."""
    return (x.clamp(-1, 1) + 1) / 2


# ---------------------------------------------------------------------------
# 2. Forward diffusion process
# ---------------------------------------------------------------------------
class GaussianDiffusion:
    """Implements the closed-form forward process and the ancestral sampler
    of DDPM (Ho et al., 2020).

    The forward process never adds noise to the image directly in a loop;
    instead it uses the closed-form reparameterization

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

    which is mathematically equivalent to repeating the single-step
    q(x_t | x_t-1) update t times, but lets us jump straight to any
    timestep t during training.
    """

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=device), self.alphas_cumprod[:-1]]
        )

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)

        # Coefficients of the true posterior mean q(x_{t-1} | x_t, x0):
        # mu = posterior_mean_x0_coef * x0 + posterior_mean_xt_coef * x_t
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_x0_coef = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_xt_coef = (
            torch.sqrt(self.alphas) * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def _extract(self, values, t, shape):
        out = values.gather(0, t)
        return out.reshape(t.shape[0], *((1,) * (len(shape) - 1)))

    def q_sample(self, x0, t, noise=None):
        """Forward process: produce a noisy x_t from a clean image x0."""
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)

        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_t * noise, noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        """One reverse step: x_t -> x_{t-1}, using the model's noise prediction.

        The predicted x0 is clipped to the valid [-1, 1] image range before
        computing the posterior mean. Without this, small per-step errors in
        the noise prediction (expected from a model trained on very few
        images) compound over T=1000 steps and the sample drifts off the
        valid data manifold (observed empirically as all-black outputs).
        """
        predicted_noise = model(x_t, t)

        sqrt_recip_alphas_cumprod_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        x0_pred = sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * predicted_noise
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        x0_coef = self._extract(self.posterior_mean_x0_coef, t, x_t.shape)
        xt_coef = self._extract(self.posterior_mean_xt_coef, t, x_t.shape)
        model_mean = x0_coef * x0_pred + xt_coef * x_t

        if (t == 0).all():
            return model_mean

        posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=4, channels=3, return_all_steps=False):
        """Reverse process: start from pure Gaussian noise and denoise to an image."""
        device = next(model.parameters()).device
        x_t = torch.randn(batch_size, channels, image_size, image_size, device=device)

        intermediate = [x_t.cpu()]
        for step in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t)
            if return_all_steps and step % (self.timesteps // 10) == 0:
                intermediate.append(x_t.cpu())

        if return_all_steps:
            return x_t, intermediate
        return x_t


# ---------------------------------------------------------------------------
# 3. Custom loss
# ---------------------------------------------------------------------------
def noise_prediction_loss(predicted_noise, true_noise):
    """Custom mean-squared-error between the predicted and true noise.

    Written out manually (rather than calling nn.MSELoss) per the
    assignment's "loss must be a customized function" requirement.
    """
    return ((predicted_noise - true_noise) ** 2).mean()


# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------
def train(model, diffusion, dataloader, device, epochs, lr, output_dir):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    model.train()

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            t = torch.randint(0, diffusion.timesteps, (batch.shape[0],), device=device)

            x_t, true_noise = diffusion.q_sample(batch, t)
            predicted_noise = model(x_t, t)

            loss = noise_prediction_loss(predicted_noise, true_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        if (epoch + 1) % max(1, epochs // 20) == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.5f}")

    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Noise-prediction MSE loss")
    plt.title("Training loss")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    return losses


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def plot_forward_process(diffusion, image, output_dir, num_steps=10):
    """Figure 1 style plot: clean image -> increasingly noisy -> pure noise."""
    steps = torch.linspace(0, diffusion.timesteps - 1, num_steps).long()
    fig, axes = plt.subplots(1, num_steps, figsize=(2 * num_steps, 2))

    for ax, step in zip(axes, steps):
        t = torch.full((1,), step.item(), dtype=torch.long)
        noisy, _ = diffusion.q_sample(image.unsqueeze(0), t)
        img = unnormalize(noisy[0]).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f"t={step.item()}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "forward_process.png"))
    plt.close()


def plot_generated_samples(samples, output_dir, filename="generated_samples.png"):
    n = samples.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))
    if n == 1:
        axes = [axes]
    for ax, img in zip(axes, samples):
        ax.imshow(unnormalize(img).permute(1, 2, 0).numpy())
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


# ---------------------------------------------------------------------------
# 5. Test function: noise -> image
# ---------------------------------------------------------------------------
def generate_images(model, diffusion, image_size, num_images=8, device="cpu"):
    """Accepts pure Gaussian noise and produces images via the reverse process."""
    model.eval()
    samples = diffusion.sample(model, image_size=image_size, batch_size=num_images)
    return samples.cpu()


# ---------------------------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train a DDPM diffusion model on animal images.")
    parser.add_argument("--data_dir", type=str, default="animal_data", help="Path to the animal_data folder.")
    parser.add_argument("--classes", type=str, nargs="+",
                         default=["Cat", "Dog", "Bird", "Lion", "Tiger"],
                         help="Animal classes to train on.")
    parser.add_argument("--images_per_class", type=int, default=20)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save_dir", type=str, default="saved_models")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = AnimalDataset(
        root_dir=args.data_dir,
        classes=args.classes,
        images_per_class=args.images_per_class,
        img_size=args.img_size,
        seed=args.seed,
    )
    print(f"Loaded {len(dataset)} images from classes: {args.classes}")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    diffusion = GaussianDiffusion(timesteps=args.timesteps, device=device)

    plot_forward_process(diffusion, dataset[0], args.output_dir)

    model = DenoiseUNet(image_channels=3)

    losses = train(model, diffusion, dataloader, device, args.epochs, args.lr, args.output_dir)

    torch.save({
        "model_state_dict": model.state_dict(),
        "img_size": args.img_size,
        "timesteps": args.timesteps,
    }, os.path.join(args.save_dir, "diffusion_model.pt"))
    print(f"Saved model to {os.path.join(args.save_dir, 'diffusion_model.pt')}")

    samples = generate_images(model, diffusion, args.img_size, num_images=args.num_samples, device=device)
    plot_generated_samples(samples, args.output_dir)
    print(f"Saved generated samples to {os.path.join(args.output_dir, 'generated_samples.png')}")


if __name__ == "__main__":
    main()
