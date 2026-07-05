import torch


class FlowMatchingScheduler:
    """
    Rectified-flow / conditional flow-matching scheduler — Lipman et al. (2023)
    "Flow Matching for Generative Modeling"; Liu et al. (2022) "Rectified Flow";
    Albergo & Vanden-Eijnden (2023) "Stochastic Interpolants".

    ▲ REPLACES THE DDPM ε-TARGET INVARIANT.
      rahulk-ddpm's CosineNoiseScheduler trains ε_θ to predict the *noise* added
      by q(x_t | x_0) = N(√ᾱ_t·x_0, (1-ᾱ_t)·I), then samples by unrolling a
      1000-step stochastic reverse chain. There is NONE of that here: no β_t, no
      ᾱ_t, no cosine schedule, no posterior variance. Flow matching learns a
      *velocity* field along a deterministic straight-line path and samples by
      integrating an ODE in ~20–50 steps.

    Time convention (opposite orientation to some FM papers, chosen to read
    "t=0 is what you want"):
        t = 0  ->  data     x0  (clean clip)
        t = 1  ->  noise    x1 ~ N(0, I)

    Straight-line interpolation path (the "rectified" / linear coupling):
        x_t = (1 - t)·x0 + t·x1

    Its time-derivative is constant along the path — that constant is the target:
        v = dx_t/dt = x1 - x0                       (independent of t)

    Training objective (plain MSE on velocity, no per-t weighting):
        L = E_{x0, x1~N(0,I), t~U(0,1)} [ || v_θ(x_t, t) - (x1 - x0) ||² ]

    Sampling — integrate the ODE dx/dt = v_θ(x, t) DOWN from t=1 to t=0:
        x  <-  x1 ~ N(0, I)                          (start at t=1)
        for t stepping 1 -> 0 in N Euler steps of size Δt = 1/N:
            x  <-  x - Δt · v_θ(x, t)                (dt is negative → minus sign)
        return x  (≈ x0, a clean clip)

    Contract with the model: t is passed as a continuous value in [0, 1]. Any
    scale-up for the sinusoidal time embedding (blueprint suggests t·1000) lives
    inside the model's forward, NOT here — the scheduler speaks in raw path time.

    Shapes: clips are 5D, x = (B, T, C, H, W). A per-sample t of shape (B,) is
    broadcast over the T, C, H, W axes via view(-1, 1, 1, 1, 1).
    """

    def __init__(self, device: str = 'cpu'):
        # Nothing to precompute — the path is analytic. Kept for API symmetry
        # with CosineNoiseScheduler and to pin the sampling device.
        self.device = device

    # ── forward path ─────────────────────────────────────────────────────────
    def add_noise(self, x0: torch.Tensor, x1: torch.Tensor,
                  t: torch.Tensor) -> torch.Tensor:
        """x_t = (1 - t)·x0 + t·x1  — point on the straight line at path time t."""
        t = t.view(-1, 1, 1, 1, 1)
        return (1.0 - t) * x0 + t * x1

    def velocity_target(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """v = dx_t/dt = x1 - x0  — constant along the path, so t is not needed."""
        return x1 - x0

    # ── ODE sampling ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def sample(self, model, shape, steps: int = 50,
               x1: torch.Tensor = None) -> torch.Tensor:
        """
        Euler-integrate dx/dt = v_θ(x, t) from noise (t=1) to data (t=0).

        model : velocity predictor, called as model(x, t) with t of shape (B,)
                holding the current path time in [0, 1]; returns v of shape `shape`.
        shape : (B, T, C, H, W) of the clip(s) to generate.
        steps : number of Euler steps N (fewer than DDPM's 1000 by ~20×).
        x1    : optional starting noise; drawn from N(0, I) if omitted.
        """
        device = next(model.parameters()).device
        x  = x1 if x1 is not None else torch.randn(shape, device=device)
        B  = shape[0]
        dt = 1.0 / steps

        for i in range(steps):
            t_val = 1.0 - i * dt                       # 1 → dt over the loop
            t     = torch.full((B,), t_val, device=device)
            v     = model(x, t)
            x     = x - dt * v                         # step toward t=0

        return x
