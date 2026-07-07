"""
sample.py — flow-matching ODE sampling + GIF export.

Deliberately thin. The integrator lives in scheduler/flow_scheduler.py (the Euler
loop already tested to 5.7e-6 recovery) — this file is only the harness:
    load config → build DiTVideo → load EMA state → scheduler.sample(...) → GIF.

Three things it gets right on purpose:
  • EMA, not raw weights. Inference always loads the EMA shadow: it averages out
    the per-step SGD jitter (the wobble you see in a training loss tail), so
    generated clips are visibly smoother than raw-weight samples.
  • Reuses scheduler.sample — does NOT re-implement the Euler loop.
  • [-1,1] → [0,255] on the way out (inverse of the data's *2-1 scaling). Skip
    this and the GIF is all-black / blown-out.

Acceptance check (untrained model): the GIF is pure static noise — that is the
PASS. It proves the sampler runs end-to-end, shapes survive the round trip, and
the rescale is right. Coherent moving digits only appear after the Kaggle run.
A hard error, or a flat gray frame, is the failure.
"""

import os, yaml, argparse
import torch
import numpy as np
from PIL import Image

from model.dit_video   import DiTVideo
from scheduler         import FlowMatchingScheduler


def build_model(cfg, device):
    return DiTVideo(
        frames=cfg['frames'], image_size=cfg['image_size'], channels=cfg['channels'],
        patch_size=cfg['patch_size'], dim=cfg['dim'], depth=cfg['depth'],
        heads=cfg['heads'],
    ).to(device)


def load_ema(model, ckpt_path, device):
    """
    Load EMA weights into `model`. Accepts either a full resume checkpoint
    (dict with 'ema_state_dict') or a bare EMA state_dict (ema_*.pth). Returns
    True if weights were loaded, False if we're sampling from a fresh init.
    """
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"⚠️  No checkpoint at '{ckpt_path}' — sampling from UNTRAINED init "
              f"(expect a pure-noise GIF; that is the sampler smoke-test pass).", flush=True)
        return False
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt['ema_state_dict'] if isinstance(ckpt, dict) and 'ema_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    print(f"Loaded EMA weights from {ckpt_path}", flush=True)
    return True


def to_frames(clip: torch.Tensor, upscale: int = 128):
    """(T, 1, H, W) in [-1,1] → list of PIL 'L' frames, rescaled [-1,1]→[0,255]."""
    clip = ((clip.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8).cpu()  # [0,255]
    frames = []
    for f in range(clip.shape[0]):
        arr = clip[f, 0].numpy()                          # (H, W) uint8
        im  = Image.fromarray(arr, mode='L')
        frames.append(im.resize((upscale, upscale), Image.NEAREST))
    return frames


def sample(cfg_path='config.yaml', ckpt='', n=4, steps=None, out_dir=None, device=None):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    # device override lets the eyeball-a-checkpoint path run on CPU in a free
    # session (no GPU quota, and it sidesteps Kaggle's P100/Pascal torch break).
    device   = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    steps    = steps if steps is not None else cfg['sample_steps']
    out_dir  = out_dir or cfg['sample_dir']
    os.makedirs(out_dir, exist_ok=True)
    print(f"Device: {device}  |  {n} clips  |  {steps} Euler steps", flush=True)

    model     = build_model(cfg, device)
    trained   = load_ema(model, ckpt, device)
    model.eval()
    scheduler = FlowMatchingScheduler(device=device)

    shape = (n, cfg['frames'], cfg['channels'], cfg['image_size'], cfg['image_size'])
    x0 = scheduler.sample(model, shape, steps=steps)      # reuse the tested ODE loop
    assert x0.shape == shape, f"round-trip shape broke: {x0.shape} != {shape}"

    tag = 'ema' if trained else 'untrained'
    for i in range(n):
        frames = to_frames(x0[i])
        path = f"{out_dir}/sample_{tag}_{i}.gif"
        frames[0].save(path, save_all=True, append_images=frames[1:],
                       duration=120, loop=0)
    print(f"Saved {n} GIFs → {out_dir}/sample_{tag}_*.gif  "
          f"(range [{x0.min():.2f}, {x0.max():.2f}])", flush=True)
    return x0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg',   default='config.yaml')
    ap.add_argument('--ckpt',  default='', help='resume_*.pth or ema_*.pth; empty = untrained smoke test')
    ap.add_argument('--n',     type=int, default=4)
    ap.add_argument('--steps', type=int, default=None, help='Euler ODE steps (default: cfg sample_steps)')
    ap.add_argument('--out',   default=None)
    ap.add_argument('--device', default=None, help="force 'cpu' or 'cuda' (default: auto)")
    args = ap.parse_args()
    sample(args.cfg, ckpt=args.ckpt, n=args.n, steps=args.steps, out_dir=args.out,
           device=args.device)
