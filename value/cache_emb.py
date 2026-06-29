"""Stage 2 step 1: cache frozen-encoder embeddings for the whole v2 dataset.

The encoder (epoch-200 dual-camera) is FROZEN for stage 2. We run every frame
through it ONCE and dump the (N,192) embeddings to disk, so the value-head
training loop never has to touch the ViT again (huge speedup + keeps the TD
gradient strictly off the encoder).

Output: value/emb_cache.npz
  emb        (N,192) float16   frozen CLS embedding per frame (dual-camera)
  ep_offset  (n_ep,) int64     start index of each episode in emb
  ep_len     (n_ep,) int64     length of each episode
  drawer_qpos(N,1)   float32   drawer joint truth (diagnostics only, not used in training)

Usage:
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python value/cache_emb.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "diagnostics"))
from _common import load_jepa, encode_frames, DrawerH5  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emb_cache.npz")
CHUNK = 2048   # frames per encode call (streamed from h5, never load all pixels)


def main(epoch: int = 200):
    model = load_jepa(epoch)
    h5 = DrawerH5()
    f = h5.f
    N = f["pixels"].shape[0]
    print(f"caching {N} frames (dual-camera) through frozen epoch-{epoch} encoder")

    embs = np.empty((N, 192), dtype=np.float16)
    t0 = time.time()
    for i in range(0, N, CHUNK):
        j = min(i + CHUNK, N)
        pix = f["pixels"][i:j]          # (c,224,224,3) uint8
        eye = f["eye_in_hand"][i:j]
        emb = encode_frames(model, pix, eye).numpy().astype(np.float16)  # (c,192)
        embs[i:j] = emb
        if (i // CHUNK) % 10 == 0:
            print(f"  {j:7d}/{N}  ({j/N:5.1%})  {time.time()-t0:5.0f}s", flush=True)

    np.savez(OUT,
             emb=embs,
             ep_offset=h5.ep_offset.astype(np.int64),
             ep_len=h5.ep_len.astype(np.int64),
             drawer_qpos=f["drawer_qpos"][:].astype(np.float32))
    print(f"saved -> {OUT}  ({os.path.getsize(OUT)/1e6:.0f} MB, {time.time()-t0:.0f}s)")
    h5.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=200)
    main(ap.parse_args().epoch)
