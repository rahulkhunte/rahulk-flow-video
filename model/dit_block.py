import torch
import torch.nn as nn

# Reused verbatim from rahulk-ddpm/model/dit_block.py (AdaLayerNorm + DiTBlock).
# DiTStub and its local SinusoidalTimeEmbedding are dropped — dit_video.py owns
# patchify/unpatchify and time embedding in this repo. In rahulk-flow-video the
# block's output head predicts a VELOCITY field v_θ, not noise ε, but the block
# itself is objective-agnostic: it just transforms conditioned tokens.


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Norm — conditions scale/shift on timestep embedding.
    Used in DiT instead of standard LN to inject diffusion conditioning.

    DiT paper (Peebles & Xie, 2023):
      γ, β = Linear(c)  where c = timestep + class embedding
      AdaLN(x) = γ * LayerNorm(x) + β

    Zero-init on the modulation projection ⇒ γ=0, β=0 at init, so
    AdaLN(x) = LayerNorm(x): the conditioning starts as a no-op and is
    learned gradually (the "adaLN-Zero" idea from the DiT paper).
    """
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm    = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj    = nn.Linear(cond_dim, dim * 2)    # → (γ, β)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    One DiT Transformer block — the SPATIAL half of each factorized pair.

    Architecture (Peebles & Xie, 2023 — "Scalable Diffusion Models with Transformers"):
      x → AdaLN → MultiheadAttention → residual
        → AdaLN → MLP (GELU, 4× expand) → residual

    Key insight vs UNet-DDPM:
      - Operates on flattened image patches (like ViT), not spatial feature maps
      - Scales as O(n²) in sequence length (patch count), not O(h·w) in resolution
      - Conditioning via AdaLN, NOT via cross-attention (cheaper + equally effective)
      - Enables class-conditional generation natively (add class embed to t embed)

    In dit_video.py this attends over the N patches of ONE frame at a time
    (tokens reshaped (B,T,N,D) → (B·T,N,D)); TemporalAttentionBlock then
    attends over the T frames at one patch position.
    """

    def __init__(self, hidden_dim: int = 384, num_heads: int = 6,
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

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, N, D) — B=batch, N=num_patches, D=hidden_dim
        cond: (B, cond_dim) — timestep + optional class embedding
        """
        # Self-attention with AdaLN conditioning
        normed = self.norm1(x, cond)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # MLP with AdaLN conditioning
        x = x + self.mlp(self.norm2(x, cond))
        return x
