"""
train.py — flow-matching video training loop.

One step (the whole objective, straight from the blueprint):
    x0   = batch                       # (B, 8, 1, 64, 64), clean clip
    x1   = randn_like(x0)              # noise ~ N(0, I)
    t    = rand(B)                     # U(0, 1)
    x_t  = (1-t)*x0 + t*x1             # scheduler.add_noise
    pred = model(x_t, t)              # velocity prediction
    loss = mse(pred, x1 - x0)         # scheduler.velocity_target

▲ This REPLACES rahulk-ddpm's ε-loss. Everything else is the same machinery:
  Adam lr 1e-4, grad clip 1.0, EMA(0.9999), resume checkpointing. Checkpoints
  are meant to be pushed to HF Hub each save (NOT git) so a dropped Kaggle
  session resumes cleanly.

`--overfit` trains on a single fixed batch — the stack's real acceptance test:
loss should fall from ~2 (= Var(x1-x0)) toward ~0 in a couple hundred steps.
"""

import os, copy, yaml, argparse, itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.dit_video      import DiTVideo
from scheduler            import FlowMatchingScheduler
from data.moving_mnist    import MovingMNIST


class EMA:
    """
    Exponential Moving Average of model weights — reused from rahulk-ddpm.
        θ_ema ← decay · θ_ema + (1 - decay) · θ
    Training weights oscillate around the optimum; EMA smooths them and is what
    inference (sample.py) loads. Standard in every production diffusion/flow model.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay  = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s_param, param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def get_model(self) -> nn.Module:
        return self.shadow


def build_model(cfg, device):
    return DiTVideo(
        frames=cfg['frames'], image_size=cfg['image_size'], channels=cfg['channels'],
        patch_size=cfg['patch_size'], dim=cfg['dim'], depth=cfg['depth'],
        heads=cfg['heads'],
    ).to(device)


def train(cfg_path: str = 'config.yaml', resume: str = '',
          overfit: bool = False, max_steps: int = None):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(cfg.get('seed', 0))
    print(f"Device: {device}", flush=True)

    ckpt_dir = cfg['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    dataset = MovingMNIST(cache=cfg['data_cache'], src=cfg['data_src'],
                          num_frames=cfg['frames'])
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True,
                        num_workers=cfg.get('num_workers', 4),
                        pin_memory=(device == 'cuda'), drop_last=True)
    print(f"Dataset: MovingMNIST  |  {len(dataset)} sequences  |  "
          f"clip ({cfg['frames']}, {cfg['channels']}, {cfg['image_size']}, "
          f"{cfg['image_size']})", flush=True)

    # ── Model / scheduler / optim ────────────────────────────────────────────
    model     = build_model(cfg, device)
    ema       = EMA(model, decay=cfg['ema_decay'])
    scheduler = FlowMatchingScheduler(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    criterion = nn.MSELoss()
    grad_clip = cfg['grad_clip']

    n_params = sum(p.numel() for p in model.parameters())
    print(f"DiTVideo params: {n_params/1e6:.2f}M  |  EMA decay {cfg['ema_decay']}", flush=True)

    # ── Resume (model + ema + optimizer + step + loss history) ───────────────
    start_step, losses = 0, []
    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        ema.get_model().load_state_dict(ckpt['ema_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_step = ckpt['step']
        losses     = ckpt.get('losses', [])
        print(f"Resumed at step {start_step}  ({len(losses)} loss entries)", flush=True)
    elif resume:
        print(f"⚠️  Resume path not found: {resume} — starting fresh", flush=True)

    total_steps = max_steps if max_steps is not None else cfg['max_steps']
    log_every, save_every = cfg['log_every'], cfg['save_every']

    # Overfit mode: freeze ONE batch and hammer it — the stack's learn-check.
    fixed_batch = next(iter(loader)).to(device) if overfit else None
    if overfit:
        print(f"OVERFIT: single fixed batch {tuple(fixed_batch.shape)} for {total_steps} steps", flush=True)

    # ── Training loop (step-based) ───────────────────────────────────────────
    model.train()
    step = start_step
    data_iter = itertools.repeat(fixed_batch) if overfit else _cycle(loader)
    while step < total_steps:
        x0 = fixed_batch if overfit else next(data_iter).to(device)   # (B,8,1,64,64)
        x1 = torch.randn_like(x0)                                     # noise
        t  = torch.rand(x0.size(0), device=device)                   # U(0,1)

        x_t  = scheduler.add_noise(x0, x1, t)
        pred = model(x_t, t)
        loss = criterion(pred, scheduler.velocity_target(x0, x1))    # mse(pred, x1-x0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        ema.update(model)

        losses.append(loss.item())
        step += 1

        if step % log_every == 0 or step == 1:
            print(f"step {step:6d}/{total_steps}  loss {loss.item():.4f}", flush=True)

        if not overfit and step % save_every == 0:
            _save(ckpt_dir, step, model, ema, optimizer, losses, cfg)

    if not overfit:
        _save(ckpt_dir, step, model, ema, optimizer, losses, cfg, final=True)
    print(f"Done at step {step}. First loss {losses[0]:.4f} → last {losses[-1]:.4f} "
          f"(best {min(losses):.4f})", flush=True)
    return losses


def _cycle(loader):
    """Infinite iterator over the DataLoader for step-based training."""
    while True:
        for batch in loader:
            yield batch


def _save(ckpt_dir, step, model, ema, optimizer, losses, cfg, final=False):
    tag = 'final' if final else f'step_{step}'
    # Inference-only EMA weights (what sample.py loads).
    torch.save(ema.get_model().state_dict(), f"{ckpt_dir}/ema_{tag}.pth")
    # Full resume checkpoint — relaunch with --resume continues exactly here.
    torch.save({
        'step':                 step,
        'model_state_dict':     model.state_dict(),
        'ema_state_dict':       ema.get_model().state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses':               losses,
        'config':               cfg,
    }, f"{ckpt_dir}/resume_{tag}.pth")
    print(f"  saved {ckpt_dir}/resume_{tag}.pth  (+ ema_{tag}.pth)", flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg',       default='config.yaml')
    ap.add_argument('--resume',    default='')
    ap.add_argument('--overfit',   action='store_true',
                    help='train on one fixed batch (stack sanity: loss ~2 → ~0)')
    ap.add_argument('--max_steps', type=int, default=None)
    args = ap.parse_args()
    train(args.cfg, resume=args.resume, overfit=args.overfit, max_steps=args.max_steps)
