# Kaggle training

T4 notebook + metadata for training `rahulk-flow-video` on Kaggle, with the
Moving MNIST cache attached as a private Dataset and checkpoints synced to HF Hub.

## Files
- `rahulk-flow-video-train.ipynb` — the notebook (clone → attach data → train → HF push).
- `kernel-metadata.json` — enables **GPU** + **Internet**, attaches the dataset.

## One-time setup
1. **Dataset** — upload the sequence-first cache as a private Kaggle Dataset:
   ```bash
   # staged folder = dataset-metadata.json + moving_mnist_seqfirst.npy
   kaggle datasets create -p <stage_dir> --dir-mode zip
   ```
   (id: `rahulkhunte/moving-mnist-seqfirst`, private by default.)
2. **HF Secret** — in the Kaggle notebook: Add-ons ▸ Secrets ▸ add `HF_TOKEN`
   = a Hugging Face **write** token. The notebook reads it via `UserSecretsClient`.

## Push / update the notebook
```bash
kaggle kernels push -p kaggle/
```
This creates/updates `rahulkhunte/rahulk-flow-video-train` with GPU + Internet on
and the dataset attached. Open it on kaggle.com and **Run All**.

## Run order
1. Keep `SMOKE = True` (cell 0) → Run All. 200 steps confirm GPU + data + HF push
   + HF pull/resume round-trip.
2. Flip `SMOKE = False` → Run All for the real 50k-step run. If the 12h window
   ends it early, just Run All again — it resumes from HF automatically.

Checkpoints land in the private HF repo `rahulkhunte/rahulk-flow-video-ckpts`
(auto-created on first push): `resume_latest.pth` / `ema_latest.pth` overwritten
each save, plus permanent `*_step_N.pth` milestones every `hf_push_every` steps.
