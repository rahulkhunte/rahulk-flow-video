import torch
import torch.nn as nn

from .dit_block import AdaLayerNorm


class TemporalAttentionBlock(nn.Module):
    """
    Temporal attention — the frame-coherence half of each factorized pair.

    The one genuinely new architectural piece vs rahulk-ddpm. A spatial DiTBlock
    lets patches within ONE frame talk to each other; this block lets the SAME
    patch position talk to itself across all T frames. Stacking the two per layer
    (spatial → temporal) is the standard factorized video-DiT pattern (Ho et al.
    2022 "Video Diffusion Models"; Blattmann et al. 2023 "Align Your Latents") —
    it costs O(T·N² + N·T²) instead of full spatio-temporal O((T·N)²), which at
    T=8, N=64 is ~7× cheaper per layer.

    The whole trick is one reshape:
        (B, T, N, D)  →  permute  →  (B, N, T, D)  →  (B·N, T, D)
    Each of the B·N rows is now a length-T sequence: one patch position watched
    over time. Vanilla MultiheadAttention over that axis IS temporal attention.

    Block body mirrors DiTBlock exactly (same residual discipline, so a
    spatial→temporal stack is just two residual adds in a row):
        x → AdaLN(t) → MultiheadAttention over T → +residual
          → AdaLN(t) → MLP (GELU, 4× expand)     → +residual

    Conditioning contract: `cond` is the already-computed time embedding of shape
    (B, cond_dim), same object the spatial blocks receive. The scheduler speaks
    raw path time t ∈ [0,1]; the t·1000 scale-up for the sinusoidal embedding
    happens once, upstream in dit_video.py — never here, never in the scheduler.
    AdaLayerNorm zero-inits its modulation proj, and this block additionally
    zero-inits the attention and MLP output projections, so at init the block is
    an exact identity (x + 0 twice) and cannot fight the spatial path.

    Full (bidirectional) attention over all T=8 frames — cheap at this length.
    Causal/sparse variants can come later.
    """

    def __init__(self, hidden_dim: int = 256, num_heads: int = 4,
                 cond_dim: int = 256, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = AdaLayerNorm(hidden_dim, cond_dim)
        self.attn  = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = AdaLayerNorm(hidden_dim, cond_dim)

        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )

        # Identity-at-init: zero the residual-branch output projections so the
        # temporal path contributes nothing until it learns to.
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, T, N, D) — T=frames, N=patches/frame, D=hidden_dim
        cond: (B, cond_dim) — time embedding (computed upstream in dit_video)
        Returns: (B, T, N, D)
        """
        B, T, N, D = x.shape

        # (B, T, N, D) → (B·N, T, D): each row = one patch position over time.
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        # Batch is now B-major over N, so each sample's cond repeats N× in place.
        c = cond.repeat_interleave(N, dim=0)               # (B·N, cond_dim)

        # Self-attention over frames with AdaLN conditioning
        normed = self.norm1(x, c)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # MLP with AdaLN conditioning
        x = x + self.mlp(self.norm2(x, c))

        # (B·N, T, D) → (B, T, N, D)
        return x.reshape(B, N, T, D).permute(0, 2, 1, 3)
