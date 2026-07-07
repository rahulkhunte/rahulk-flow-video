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

## ⚠️ GPU MUST be T4 — and that is UI-only
Kaggle's PyTorch dropped Pascal support, so the **P100** it hands out by default
crashes (`sm_60` unsupported). You must use **GPU T4 x2**. There is **no API /
metadata field for accelerator type** — `enable_gpu` is just a boolean that
yields P100, and `kaggle kernels push` *resets* any T4 you picked. So:

- **`kaggle kernels push` is only for editing the notebook.** Every CLI-triggered
  run lands on P100 and fails the GPU check (by design — fail fast).
- **Real runs must be launched from the web UI**, where T4 is selected:
  editor ▸ **Settings ▸ Accelerator ▸ `GPU T4 x2`**, then **Save Version ▸
  Save & Run All (Commit)** (batch; survives disconnects and the 12h cap).

## Run order
1. **Smoke** — in the UI with T4 selected, keep `SMOKE = True` (cell 0) ▸ Save &
   Run All. 200 steps confirm GPU + data + HF push + HF pull/resume round-trip.
2. **Real run** — set `SMOKE = False`, keep T4, Save & Run All. If the 12h window
   ends it before 50k, just Save & Run All again — it resumes from HF automatically.

Checkpoints land in the **public** HF repo
[`rahulkhunte/rahulk-flow-video-ckpts`](https://huggingface.co/rahulkhunte/rahulk-flow-video-ckpts)
(auto-created private on first push, then flipped public in repo settings):
`resume_latest.pth` / `ema_latest.pth` overwritten each save, plus permanent
`*_step_N.pth` milestones every `hf_push_every` steps.
