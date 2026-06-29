"""p15: visualize WHERE the JEPA ViT encoder looks (CLS attention rollout).

The state embedding = CLS token of a HF ViTModel (m.encoder, hidden=192,
patch14 -> 16x16=256 patches + CLS). We ask the encoder for output_attentions
and do attention rollout (per layer: mean over heads, +identity, row-normalize,
then matrix-multiply across all layers). The CLS row over the 256 patch tokens
is the heatmap = how much the state embedding aggregates from each image patch.

Both cameras (agentview + eye_in_hand) go through the SAME encoder, so we draw
both. Frames span one episode early->late (arm far -> reaching -> closing) to
see if attention shifts onto the arm/drawer, and whether the far-arm frame is
attended differently (the p13 reach-loss question).

Usage:
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python diagnostics/p15_attention.py --ep 0
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common import load_jepa, load_phys, transform_pixels, DrawerH5, RESULTS_DIR  # noqa: E402


def _selfattn_layers(vit):
    """List of ViTSelfAttention modules, in depth order."""
    return [lyr.attention.attention for lyr in vit.encoder.layer]


@torch.no_grad()
def rollout_attention(vit, pix):
    """pix: (1,3,224,224) preprocessed. Returns (g,g) CLS->patch heatmap in [0,1].

    SDPA path returns no attentions, so we hook each ViTSelfAttention, capture its
    input hidden_states, and recompute softmax(QK^T/sqrt(d)) with the module's own
    query/key projections. Then attention rollout across depth.
    """
    layers = _selfattn_layers(vit)
    captured = []
    handles = [m.register_forward_pre_hook(
        lambda mod, args: captured.append((mod, args[0]))) for m in layers]
    try:
        vit(pix, interpolate_pos_encoding=True)
    finally:
        for h in handles:
            h.remove()

    attns = []
    for mod, h in captured:                      # h: (1,T,D)
        nh = mod.num_attention_heads
        hs = mod.attention_head_size

        def split(x):                            # (1,T,D) -> (nh,T,hs)
            return x.view(1, x.size(1), nh, hs).permute(0, 2, 1, 3)[0]
        q, k = split(mod.query(h)), split(mod.key(h))
        scores = (q @ k.transpose(-1, -2)) / (hs ** 0.5)   # (nh,T,T)
        attns.append(torch.softmax(scores, dim=-1).mean(0))  # (T,T) mean heads

    T = attns[0].shape[-1]
    eye = torch.eye(T, device=attns[0].device)
    roll = eye.clone()
    for a in attns:
        a = a + eye                              # add residual / self
        a = a / a.sum(-1, keepdim=True)          # row-normalize
        roll = a @ roll                          # propagate through depth
    cls = roll[0, 1:]                            # CLS row over patch tokens (T-1,)
    g = int(round((cls.numel()) ** 0.5))         # 256 -> 16
    hm = cls.reshape(g, g).float().cpu().numpy()
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    return hm


def overlay(ax, img, hm, title):
    ax.imshow(img)
    if hm is not None:
        H, W = img.shape[:2]
        hm_up = np.kron(hm, np.ones((H // hm.shape[0], W // hm.shape[1])))
        ax.imshow(hm_up, cmap="jet", alpha=0.45, extent=(0, W, H, 0))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["v2", "phys"], default="phys")
    ap.add_argument("--epoch", type=int, default=65)
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--fracs", type=float, nargs="+",
                    default=[0.0, 0.3, 0.6, 0.95])
    args = ap.parse_args()

    model = load_phys(args.epoch) if args.model == "phys" else load_jepa(args.epoch)
    vit = model.encoder                         # HF ViTModel, hidden=192
    h5 = DrawerH5()

    px = h5.pixels(args.ep)                      # (L,H,W,3) agentview
    eye = h5.eye_in_hand(args.ep)               # (L,H,W,3) wrist
    L = len(px)
    idx = [min(L - 1, int(f * L)) for f in args.fracs]
    print(f"ep {args.ep}: L={L}  frames={idx}", flush=True)

    nrow = len(idx)
    fig, axes = plt.subplots(nrow, 4, figsize=(13, 3.1 * nrow))
    if nrow == 1:
        axes = axes[None]
    for r, t in enumerate(idx):
        agv, eyv = px[t], eye[t]
        a_pix = transform_pixels(agv[None]).cuda()
        e_pix = transform_pixels(eyv[None]).cuda()
        a_hm = rollout_attention(vit, a_pix)
        e_hm = rollout_attention(vit, e_pix)
        overlay(axes[r, 0], agv, None, f"agentview  t={t}")
        overlay(axes[r, 1], agv, a_hm, "agentview  CLS-attn")
        overlay(axes[r, 2], eyv, None, f"eye_in_hand  t={t}")
        overlay(axes[r, 3], eyv, e_hm, "eye_in_hand  CLS-attn")

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"p15_attention_{args.model}_ep{args.ep}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved -> {out}", flush=True)
    h5.close()


if __name__ == "__main__":
    main()
