import torch
import torch.nn as nn

from .time_embedding import SinusoidalTimeEmbedding
from .dit_block import AdaLayerNorm, DiTBlock
from .temporal_attention import TemporalAttentionBlock


class DiTVideo(nn.Module):
    """
    Factorized spatial+temporal Diffusion Transformer — the velocity predictor
    v_θ(x_t, t) for rahulk-flow-video.

    Assembles three parts you've already seen:
      • patchify         — Conv2d(1→D, k=8, s=8) per frame ⇒ N=64 tokens/frame
      • depth × (spatial DiTBlock → temporal block)         — reused + new
      • output head      — AdaLN → Linear(D → patch²·C), zero-init ⇒ unpatchify

    Factorization (Ho et al. 2022; Blattmann et al. 2023): each layer first lets
    patches within a frame attend (spatial, over N), then lets each patch position
    attend across frames (temporal, over T). Strictly spatial→temporal, both fed
    the SAME conditioning vector. Far cheaper than full spatio-temporal attention.

    ▲ Objective: unlike rahulk-ddpm (predicts ε), the output tensor here is the
      VELOCITY FIELD v = dx_t/dt of the straight-line flow path. Training MSEs it
      against x1 - x0 (see scheduler/flow_scheduler.py). The network is otherwise
      objective-agnostic — only the loss and the sampler know it's velocity.

    Time contract, discharged HERE and only here: the scheduler speaks raw path
    time t ∈ [0,1]; forward() multiplies by 1000 before the sinusoidal embedding
    so the argument range matches what the DDPM integer timesteps produced. No
    other module rescales t.
    """

    def __init__(self, frames: int = 8, image_size: int = 64, channels: int = 1,
                 patch_size: int = 8, dim: int = 256, depth: int = 8,
                 heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"

        self.frames      = frames
        self.image_size  = image_size
        self.channels    = channels
        self.patch_size  = patch_size
        self.dim         = dim
        self.grid        = image_size // patch_size            # 8
        self.num_patches = self.grid ** 2                      # N = 64
        self.patch_dim   = channels * patch_size * patch_size  # 64

        # ── patchify: one Conv2d applied per frame (T folded into batch) ──────
        self.patch_embed = nn.Conv2d(channels, dim,
                                     kernel_size=patch_size, stride=patch_size)

        # Learned spatial positional embedding over the N patches of a frame
        # (shared across the T frames; temporal order is carried by the t-cond
        # and the temporal blocks, so no separate frame-position embedding).
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, self.num_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── time embedding: SinusoidalTimeEmbedding(t·1000) → MLP → (B, dim) ──
        self.time_embed = SinusoidalTimeEmbedding(dim)

        # ── depth × (spatial → temporal) block pairs, same cond to both ───────
        self.spatial_blocks = nn.ModuleList([
            DiTBlock(dim, heads, cond_dim=dim, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.temporal_blocks = nn.ModuleList([
            TemporalAttentionBlock(dim, heads, cond_dim=dim, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # ── output head: final AdaLN → Linear → velocity patches (zero-init) ──
        self.norm_out = AdaLayerNorm(dim, dim)
        self.head     = nn.Linear(dim, self.patch_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    # ── patch <-> pixel ──────────────────────────────────────────────────────
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C, H, W) → (B, T, N, D) via per-frame Conv2d."""
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)                 # fold T into batch
        x = self.patch_embed(x)                       # (B·T, D, grid, grid)
        x = x.flatten(2).transpose(1, 2)              # (B·T, N, D), row-major (h,w)
        return x.reshape(B, T, self.num_patches, self.dim)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, N, patch_dim) → (B, T, C, H, W)."""
        B, T, N, _ = x.shape
        g, p, C = self.grid, self.patch_size, self.channels
        x = x.reshape(B * T, g, g, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)               # (B·T, C, g, p, g, p)
        x = x.reshape(B * T, C, g * p, g * p)
        return x.reshape(B, T, C, self.image_size, self.image_size)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: interpolated clip x_t  (B, T, C, H, W)
        t: path time in [0, 1]     (B,)
        Returns: predicted velocity field  (B, T, C, H, W)
        """
        B, T = x.shape[0], x.shape[1]

        cond   = self.time_embed(t * 1000.0)          # (B, dim) — ×1000 lives here
        tokens = self.patchify(x) + self.pos_embed    # (B, T, N, D)

        for spatial, temporal in zip(self.spatial_blocks, self.temporal_blocks):
            # SPATIAL: attend over N within each frame (fold T into batch).
            s = tokens.reshape(B * T, self.num_patches, self.dim)
            s = spatial(s, cond.repeat_interleave(T, dim=0))
            tokens = s.reshape(B, T, self.num_patches, self.dim)
            # TEMPORAL: attend over T at each patch position (block reshapes itself).
            tokens = temporal(tokens, cond)

        # Output head on the spatial token layout (fold T into batch).
        h = tokens.reshape(B * T, self.num_patches, self.dim)
        h = self.norm_out(h, cond.repeat_interleave(T, dim=0))
        h = self.head(h)                              # (B·T, N, patch_dim)
        h = h.reshape(B, T, self.num_patches, self.patch_dim)
        return self.unpatchify(h)
