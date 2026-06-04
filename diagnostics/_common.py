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

H5_PATH = "/home/yikang/git/le-wm/data/libero_bc_drawer_v2.h5"   # M0: 双相机 + drawer_qpos
CKPT_DIR = "/home/yikang/stable-wm/outputs"
MODEL_NAME = "lewm_libero_bc_drawer_v2"   # M1: 双相机重训
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
EMB_DIM = 192   # projector 输出仍 192 (两路 CLS 拼 384 -> 192)
HISTORY = 3

# image transform IDENTICAL to bc_drawer_eval.py / training pipeline
_PIX_TF = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(**spt.data.dataset_stats.ImageNet),
    T.Resize(224),
])


def ckpt_prefix(epoch: int = 200) -> str:
    return f"{CKPT_DIR}/{MODEL_NAME}_epoch_{epoch}"


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
def encode_frames(model, frames: np.ndarray, eye_frames: np.ndarray = None,
                  device: str = "cuda", batch_size: int = 256) -> torch.Tensor:
    """Encode (N,H,W,3) uint8 frames -> (N, EMB_DIM) embeddings (on CPU).

    eye_frames: 可选腕部相机 (N,H,W,3). M1 的 v2 双相机模型必须传 (否则
    encode 只出 192 维 CLS, 与 projector 的 384 输入不匹配会报错).
    """
    embs = []
    for i in range(0, len(frames), batch_size):
        pix = transform_pixels(frames[i:i + batch_size]).to(device)  # (b,3,224,224)
        info = {"pixels": pix.unsqueeze(1)}  # (b,1,3,224,224)
        if eye_frames is not None:
            eye = transform_pixels(eye_frames[i:i + batch_size]).to(device)
            info["eye_in_hand"] = eye.unsqueeze(1)
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

    def eye_in_hand(self, ep: int) -> np.ndarray:
        return self.f["eye_in_hand"][self.episode_slice(ep)]

    def drawer_qpos(self, ep: int) -> np.ndarray:
        """(L,1) 抽屉关节真值 (探测3 回归靶子, 不进模型)."""
        return self.f["drawer_qpos"][self.episode_slice(ep)]

    def goal_frame(self, ep: int) -> np.ndarray:
        """Last frame of the episode = drawer-closed goal image (agentview)."""
        o, l = int(self.ep_offset[ep]), int(self.ep_len[ep])
        return self.f["pixels"][o + l - 1]

    def goal_eye_frame(self, ep: int) -> np.ndarray:
        """Last frame, eye_in_hand (goal 一侧第二路相机)."""
        o, l = int(self.ep_offset[ep]), int(self.ep_len[ep])
        return self.f["eye_in_hand"][o + l - 1]

    def close(self):
        self.f.close()
