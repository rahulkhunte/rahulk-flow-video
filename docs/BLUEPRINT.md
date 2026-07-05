# rahulk-flow-video — Build Blueprint

Scaled-down **flow-matching video model from scratch**. The video sibling of
`rahulk-ddpm`: same transformer/spatial machinery you already know, two genuinely
new pieces on top — **velocity-target flow matching** (instead of ε-prediction)
and **temporal attention** (frame-to-frame coherence).

## Scope (what this is, and deliberately is NOT)

| Decision | Choice | Why |
|---|---|---|
| Space | **Pixel-space** (no VAE) | At 64×64 grayscale a VAE adds a whole training dependency for zero benefit |
| Resolution | **64×64**, 1 channel | Moving MNIST is natively 64×64 grayscale |
| Frames | **8** | Enough real motion, still cheap on a T4 |
| Objective | **Flow matching** (predict velocity) | The deliberate departure from `rahulk-ddpm`'s "predict ε" |
| Backbone | **Small DiT** (~20M params) | Reuses your `dit_block.py` AdaLN design |
| Conditioning | **Unconditional** (v1) | Add digit-class via AdaLN later; keep v1 minimal |
| NOT pyramidal | no resolution pyramid | that's their multi-GPU high-res trick — out of scope |
| NOT latent / NOT text | — | both are scale features we don't need to learn the core |

## What's the same vs. what's new

- **Same as rahulk-ddpm:** sinusoidal time embedding, AdaLN-conditioned transformer
  blocks, EMA at inference, Adam + grad-clip, resume-checkpointing, GIF export.
- **New piece 1 — flow matching:** straight-line path between data and noise; the
  net predicts the *velocity* along it; sampling integrates an ODE in ~20–50 steps
  (vs. the 1000-step stochastic reverse chain).
- **New piece 2 — temporal attention:** the `(B,T,N,D) → (B·N,T,D)` reshape from
  your notes, so each patch attends across the 8 frames at its own position.

## File structure (mirrors rahulk-ddpm; ★ = new/changed)

```
model/
  time_embedding.py     # SinusoidalTimeEmbedding — REUSE (feed continuous t, scaled)
  dit_block.py          # AdaLayerNorm + spatial DiTBlock — REUSE from rahulk-ddpm
  temporal_attention.py # ★ NEW: attention over frames at each patch position
  dit_video.py          # ★ NEW: factorized spatial+temporal DiT, patchify/unpatchify
scheduler/
  flow_scheduler.py     # ★ NEW: interpolation path, velocity target, ODE sampler
data/
  moving_mnist.py       # ★ NEW: Moving MNIST -> (B,T,1,64,64) clips of 8 frames
train.py                # loop + EMA(0.9999) + grad clip(1.0) + resume — adapt loss
sample.py               # ★ ODE sampling (Euler) + GIF export — loads EMA
config.yaml             # all hyperparameters
checkpoints/            # gitignored — push to HF Hub instead
assets/                 # gifs, sample grids, loss curves
```

## Module specs (math first — implement the bodies yourself)

### `scheduler/flow_scheduler.py`  ← BUILD THIS FIRST (the core)
Convention: **t=0 is data, t=1 is noise.** x0 = clean clip, x1 ~ N(0,I).

```
Interpolation path:   x_t = (1 - t)·x0 + t·x1
Target velocity:      v   = dx_t/dt = x1 - x0        (constant along the straight path)
Training loss:        L   = E[ ‖ v_θ(x_t, t) - (x1 - x0) ‖² ]   (MSE on velocity)
Sampling (ODE):       start x = x1 ~ N(0,I) at t=1, integrate DOWN to t=0:
                      x ← x - Δt · v_θ(x, t)   for t stepping 1 → 0 in N Euler steps
```
Methods to expose: `add_noise(x0, x1, t)` → x_t; `velocity_target(x0, x1)` → v;
`sample(model, shape, steps)` → x0. No βₜ, no ᾱₜ — that's the DDPM scheduler; this
is straight-line. Flag in a comment that this REPLACES the ε-target invariant.

### `model/temporal_attention.py`  (new piece)
```
Input tokens:  (B, T, N, D)     T=8 frames, N=patches/frame, D=embed dim
Reshape:       (B, T, N, D) -> (B·N, T, D)     # each patch position, sequence over frames
Apply:         AdaLN(t) -> MultiheadAttention over T -> +residual -> MLP -> +residual
Reshape back:  (B·N, T, D) -> (B, T, N, D)
```
Variant to start: **full temporal attention over all 8 frames** (cheap at T=8).
Causal/sparse can come later.

### `model/dit_video.py`
```
patchify each frame: (B,T,1,64,64) -> (B,T,N,D)   patch_size=8 -> N=64 per frame
per block:  SPATIAL DiTBlock  (reshape (B·T, N, D), attend over N)   # reuse dit_block.py
            TEMPORAL block     (reshape (B·N, T, D), attend over T)   # temporal_attention.py
conditioning: AdaLN from time embedding (zero-init the modulation proj)
unpatchify:  (B,T,N,D) -> (B,T,1,64,64) = predicted velocity field
```
Factorized (spatial then temporal) is far cheaper than full spatio-temporal and is
the standard video-DiT pattern.

### `model/time_embedding.py`  (reuse)
Flow matching t is continuous in [0,1]. Feed `t * 1000` (or any fixed scale) into
your existing `SinusoidalTimeEmbedding` so the embedding sees a wide range. No other
change.

### `data/moving_mnist.py`
Moving MNIST = 10k sequences of 20 frames, 64×64 grayscale, 2 digits drifting.
Sample a random 8-frame contiguous window per item → `(8, 1, 64, 64)`, scale to
[-1, 1]. Cache the `.npy` on the A1 Flex (CPU job) so Kaggle just downloads it.

### `train.py`  (adapt your loop)
```
per step:  x0 = batch                      # (B,8,1,64,64)
           x1 = randn_like(x0)             # noise
           t  = rand(B)                    # U(0,1), broadcast over T,C,H,W
           x_t = (1-t)*x0 + t*x1
           pred = model(x_t, t)
           loss = mse(pred, x1 - x0)       # velocity target
keep:  Adam lr 1e-4, grad clip max_norm 1.0, EMA decay 0.9999,
       resume checkpoint = {model, ema, optimizer, loss_history, config},
       push checkpoints to HF Hub (NOT git).
```

### `sample.py`
Load **EMA** weights. Run `flow_scheduler.sample(...)` from N(0,I), export the 8
frames as a GIF to `assets/`. Same GIF-export style as rahulk-ddpm.

## config.yaml (starting point)

```yaml
frames: 8
image_size: 64
channels: 1
patch_size: 8          # -> 64 tokens/frame
dim: 256               # embed dim
depth: 8               # spatial+temporal block pairs
heads: 4
# ~20M params in this range; tune to fit T4
batch_size: 16         # tune to T4 memory
lr: 1.0e-4
ema_decay: 0.9999
grad_clip: 1.0
sample_steps: 50       # Euler ODE steps at inference
```

## Build order

1. `flow_scheduler.py` — the heart. Get the path + velocity + ODE sampler right first.
2. `temporal_attention.py` — the one new block.
3. `dit_video.py` — assemble spatial (reused) + temporal into the velocity predictor.
4. `data/moving_mnist.py` — dataset + 8-frame windowing, cache on A1 Flex.
5. `train.py` — adapt the loop to the velocity loss; wire EMA + resume.
6. `sample.py` — ODE sampling + GIF.

## Training plan

- **Where:** free Kaggle T4 (one session). A1 Flex preps + caches the dataset and
  hosts the repo; Kaggle does the GPU run; checkpoints push to HF Hub each save so a
  dropped session resumes cleanly.
- **Expect:** coherent moving-digit clips within ~30–50k steps (~1–2 h on a T4),
  cleaner with more. First milestone = digits that move consistently across the 8
  frames without flicker — that's temporal attention working.
