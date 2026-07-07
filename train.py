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


# ── HF Hub checkpoint sync ────────────────────────────────────────────────────
# Checkpoints (100s of MB) don't belong in git. A dropped Kaggle session must be
# able to resume, so every save pushes the LATEST resume_/ema_ to a private HF
# model repo (overwritten each time), plus a PERMANENT milestone snapshot every
# `hf_push_every` steps. On startup we pull resume_latest.pth and continue.
# huggingface_hub is imported lazily so the CPU box / --overfit runs never need it.

def _hf_settings(cfg):
    """Return (repo, token, push_every) if HF sync is enabled+usable, else None."""
    repo = (cfg.get('hf_repo') or '').strip()
    if not repo:
        return None
    token = os.environ.get('HF_TOKEN', '').strip()
    if not token:
        print("⚠️  hf_repo set but HF_TOKEN env var missing — checkpoints stay "
              "local only (no HF push/resume).", flush=True)
        return None
    return repo, token, int(cfg.get('hf_push_every', 10000))


def _hf_ensure_repo(repo, token):
    """Create the private repo if absent. Returns True if HF is usable."""
    try:
        from huggingface_hub import create_repo          # lazy import
        create_repo(repo, token=token, repo_type='model', private=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"⚠️  Could not reach HF repo {repo}: {e} — disabling HF sync.", flush=True)
        return False


def _hf_upload(repo, token, local_path, path_in_repo):
    """Upload one file; never fatal — a network blip must not kill a 30k-step run."""
    try:
        from huggingface_hub import HfApi                # lazy import
        HfApi(token=token).upload_file(
            path_or_fileobj=local_path, path_in_repo=path_in_repo,
            repo_id=repo, repo_type='model',
            commit_message=f"checkpoint: {path_in_repo}",
        )
        print(f"    ↑ HF {repo}/{path_in_repo}", flush=True)
    except Exception as e:
        print(f"    ⚠️ HF upload failed ({path_in_repo}): {e}", flush=True)


def _hf_pull_latest(repo, token, dst_dir):
    """Download resume_latest.pth if it exists; return local path or '' (fresh)."""
    try:
        from huggingface_hub import hf_hub_download       # lazy import
        from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
        try:
            return hf_hub_download(repo_id=repo, filename='resume_latest.pth',
                                   repo_type='model', token=token, local_dir=dst_dir)
        except (EntryNotFoundError, RepositoryNotFoundError):
            return ''
    except Exception as e:
        print(f"⚠️  HF pull failed: {e} — starting fresh.", flush=True)
        return ''


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

    # ── HF sync setup ────────────────────────────────────────────────────────
    hf = _hf_settings(cfg)                 # (repo, token, push_every) or None
    if hf and not _hf_ensure_repo(hf[0], hf[1]):
        hf = None
    if hf:
        print(f"HF sync ON → {hf[0]}  (milestone every {hf[2]} steps)", flush=True)

    # ── Resolve resume source ────────────────────────────────────────────────
    # Priority: explicit local --resume → HF resume_latest.pth → fresh. This is
    # what makes a dropped Kaggle session self-heal: just relaunch, it pulls.
    ckpt_src = ''
    if resume and os.path.exists(resume):
        ckpt_src = resume
    elif resume:
        print(f"⚠️  --resume path not found: {resume} — checking HF instead", flush=True)
    if not ckpt_src and hf and not overfit:
        pulled = _hf_pull_latest(hf[0], hf[1], ckpt_dir)
        if pulled:
            ckpt_src = pulled
            print(f"Pulled latest checkpoint from HF {hf[0]}", flush=True)
        else:
            print(f"No checkpoint on HF {hf[0]} — starting fresh.", flush=True)

    # ── Resume (model + ema + optimizer + step + loss history) ───────────────
    start_step, losses = 0, []
    if ckpt_src:
        ckpt = torch.load(ckpt_src, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        ema.get_model().load_state_dict(ckpt['ema_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_step = ckpt['step']
        losses     = ckpt.get('losses', [])
        print(f"Resumed at step {start_step}  ({len(losses)} loss entries)", flush=True)

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
            _save(ckpt_dir, step, model, ema, optimizer, losses, cfg, hf=hf)

    if not overfit:
        _save(ckpt_dir, step, model, ema, optimizer, losses, cfg, hf=hf, final=True)
    print(f"Done at step {step}. First loss {losses[0]:.4f} → last {losses[-1]:.4f} "
          f"(best {min(losses):.4f})", flush=True)
    return losses


def _cycle(loader):
    """Infinite iterator over the DataLoader for step-based training."""
    while True:
        for batch in loader:
            yield batch


def _save(ckpt_dir, step, model, ema, optimizer, losses, cfg, hf=None, final=False):
    tag         = 'final' if final else f'step_{step}'
    ema_path    = f"{ckpt_dir}/ema_{tag}.pth"
    resume_path = f"{ckpt_dir}/resume_{tag}.pth"
    # Inference-only EMA weights (what sample.py loads).
    torch.save(ema.get_model().state_dict(), ema_path)
    # Full resume checkpoint — relaunch continues exactly here.
    torch.save({
        'step':                 step,
        'model_state_dict':     model.state_dict(),
        'ema_state_dict':       ema.get_model().state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses':               losses,
        'config':               cfg,
    }, resume_path)
    print(f"  saved {resume_path}  (+ ema_{tag}.pth)", flush=True)

    if hf:
        repo, token, push_every = hf
        # LATEST — overwritten every save, so a fresh session always finds the
        # newest checkpoint to resume from.
        _hf_upload(repo, token, resume_path, 'resume_latest.pth')
        _hf_upload(repo, token, ema_path,    'ema_latest.pth')
        # PERMANENT milestone — kept only every push_every steps (or the final
        # save), NOT every save, so the HF repo doesn't accumulate a copy per save.
        if final or (push_every and step % push_every == 0):
            _hf_upload(repo, token, resume_path, f'resume_{tag}.pth')
            _hf_upload(repo, token, ema_path,    f'ema_{tag}.pth')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg',       default='config.yaml')
    ap.add_argument('--resume',    default='')
    ap.add_argument('--overfit',   action='store_true',
                    help='train on one fixed batch (stack sanity: loss ~2 → ~0)')
    ap.add_argument('--max_steps', type=int, default=None)
    args = ap.parse_args()
    train(args.cfg, resume=args.resume, overfit=args.overfit, max_steps=args.max_steps)
