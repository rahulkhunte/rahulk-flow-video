"""
moving_mnist.py — Moving MNIST clips for flow-matching video training.

Yields FloatTensors of shape (T=8, C=1, 64, 64) in [-1, 1]: a random contiguous
8-frame window of a drifting-digits sequence. Unconditional (no labels) for v1.

Four plumbing points that silently bite, handled here:
  1. AXIS ORDER. The standard `mnist_test_seq.npy` is FRAMES-FIRST:
     (20, 10000, 64, 64) = (frame, sequence, H, W). Windowing it as-is slices
     across sequences, not time. We transpose to sequence-first once, at cache
     build, so every later index is unambiguous.
  2. CONTIGUOUS WINDOW. A clip is 8 *adjacent* frames (a random start in
     [0, 20-8]), never a random subset — random frames destroy the motion the
     model exists to learn.
  3. RANGE MATCH. Pixels are scaled to [-1, 1] to match x1 ~ N(0, I): the flow
     path x_t = (1-t)·x0 + t·x1 is only balanced if x0 and x1 share a scale.
     [0,1] data against unit-Gaussian noise makes the path lopsided.
  4. CACHE. The frames-first→sequence-first transpose is done once by
     `prepare_cache` (an A1 Flex CPU job) and written to .npy, so Kaggle just
     downloads the cache — no re-derivation, no chance of re-introducing (1).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

FRAMES_PER_SEQ = 20        # native Moving MNIST sequence length
NATIVE_SIZE    = 64        # native H = W (no resize needed for this repo)


def prepare_cache(src: str, dst: str) -> str:
    """
    A1 Flex CPU job: read the frames-first `mnist_test_seq.npy`, transpose to
    sequence-first, and write the cache Kaggle will download. uint8 is preserved
    (819 MB) — normalisation to [-1,1] happens per-window at train time, so we
    don't inflate the cache to a 3.3 GB float array.

        src (20, 10000, 64, 64) uint8  →  dst (10000, 20, 64, 64) uint8
    """
    raw = np.load(src)                                  # (20, 10000, 64, 64)
    assert raw.ndim == 4 and raw.shape[0] == FRAMES_PER_SEQ, \
        f"expected frames-first (20, N, 64, 64), got {raw.shape}"
    seq_first = np.transpose(raw, (1, 0, 2, 3)).copy()  # (10000, 20, 64, 64)
    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
    np.save(dst, seq_first)
    print(f"cached {seq_first.shape} uint8 → {dst}", flush=True)
    return dst


class MovingMNIST(Dataset):
    """
    Random contiguous 8-frame windows of Moving MNIST, each (8, 1, 64, 64) in
    [-1, 1]. Builds the sequence-first cache on first use if it is missing.
    """

    def __init__(self, cache: str = 'data/moving_mnist_seqfirst.npy',
                 src: str = 'data/MovingMNIST/mnist_test_seq.npy',
                 num_frames: int = 8):
        if not os.path.exists(cache):
            if not os.path.exists(src):
                raise FileNotFoundError(
                    f"neither cache ({cache}) nor source ({src}) found. "
                    f"Download the standard mnist_test_seq.npy or point --src at it."
                )
            prepare_cache(src, cache)

        # mmap: the 819 MB cache is not pulled into RAM; each window read touches
        # only 8 frames. Sequence-first, so seqs[i] is one full 20-frame sequence.
        self.seqs       = np.load(cache, mmap_mode='r')      # (N, 20, 64, 64)
        self.num_frames = num_frames
        self.max_start  = self.seqs.shape[1] - num_frames    # inclusive upper bound
        assert self.max_start >= 0, "num_frames exceeds sequence length"

    def __len__(self) -> int:
        return self.seqs.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = np.random.randint(0, self.max_start + 1)     # contiguous window
        window = self.seqs[idx, start:start + self.num_frames]   # (8, 64, 64) uint8
        # .copy(): the mmap view is read-only; copy to a writable, owned array
        # before from_numpy (avoids the non-writable-tensor warning).
        clip = torch.from_numpy(window.copy()).float()
        clip = clip.unsqueeze(1)                             # (8, 1, 64, 64)
        return clip / 255.0 * 2.0 - 1.0                      # → [-1, 1]


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="Prep the Moving MNIST cache (A1 Flex job).")
    ap.add_argument('--src',   default='data/MovingMNIST/mnist_test_seq.npy')
    ap.add_argument('--cache', default='data/moving_mnist_seqfirst.npy')
    args = ap.parse_args()
    prepare_cache(args.src, args.cache)
