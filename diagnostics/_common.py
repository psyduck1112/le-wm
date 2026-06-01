"""Shared utilities for le-wm LIBERO diagnostics (LIBERO_DIAGNOSTICS.md).

All probes load the SAME frozen le-wm world model and apply the SAME image
preprocessing as training (ToImage -> float scale -> ImageNet norm -> Resize 224),
so the encoder stays in-distribution. Never edit this to diverge from training.
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py
import numpy as np
import torch
from torchvision.transforms import v2 as T
import stable_pretraining as spt
import stable_worldmodel as swm

H5_PATH = "/home/yikang/stable-wm/libero_bc_drawer.h5"
CKPT_DIR = "/home/yikang/stable-wm/outputs"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
EMB_DIM = 192
HISTORY = 3

# image transform IDENTICAL to bc_drawer_eval.py / training pipeline
_PIX_TF = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(**spt.data.dataset_stats.ImageNet),
    T.Resize(224),
])


def ckpt_prefix(epoch: int = 200) -> str:
    return f"{CKPT_DIR}/lewm_libero_bc_drawer_epoch_{epoch}"


def load_jepa(epoch: int = 200, device: str = "cuda"):
    """Load frozen JEPA world model exposing encode/predict/rollout/get_cost."""
    model = swm.policy.AutoCostModel(ckpt_prefix(epoch))
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model


def transform_pixels(frames: np.ndarray) -> torch.Tensor:
    """(N,H,W,3) uint8 -> (N,3,224,224) float, training-matched preprocessing."""
    return torch.stack([_PIX_TF(f) for f in frames])


@torch.no_grad()
def encode_frames(model, frames: np.ndarray, device: str = "cuda",
                  batch_size: int = 256) -> torch.Tensor:
    """Encode (N,H,W,3) uint8 frames -> (N, EMB_DIM) embeddings (on CPU)."""
    embs = []
    for i in range(0, len(frames), batch_size):
        pix = transform_pixels(frames[i:i + batch_size]).to(device)  # (b,3,224,224)
        info = {"pixels": pix.unsqueeze(1)}  # (b,1,3,224,224)
        out = model.encode(info)
        embs.append(out["emb"][:, 0].float().cpu())  # (b, D)
    return torch.cat(embs, 0)


class DrawerH5:
    """Episode-indexed access to libero_bc_drawer.h5."""

    def __init__(self, path: str = H5_PATH):
        self.f = h5py.File(path, "r")
        self.ep_offset = self.f["ep_offset"][:]
        self.ep_len = self.f["ep_len"][:]
        self.n_ep = len(self.ep_offset)

    def episode_slice(self, ep: int) -> slice:
        o, l = int(self.ep_offset[ep]), int(self.ep_len[ep])
        return slice(o, o + l)

    def pixels(self, ep: int) -> np.ndarray:
        return self.f["pixels"][self.episode_slice(ep)]

    def actions(self, ep: int) -> np.ndarray:
        return self.f["action"][self.episode_slice(ep)]

    def proprio(self, ep: int) -> np.ndarray:
        return self.f["proprio"][self.episode_slice(ep)]

    def goal_frame(self, ep: int) -> np.ndarray:
        """Last frame of the episode = drawer-closed goal image."""
        o, l = int(self.ep_offset[ep]), int(self.ep_len[ep])
        return self.f["pixels"][o + l - 1]

    def close(self):
        self.f.close()
