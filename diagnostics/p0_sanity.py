"""Prerequisite sanity (LIBERO_DIAGNOSTICS.md 前置 #2, #4):

1. In-distribution single-step teacher-forcing prediction error on TRAINING data.
   If even teacher-forcing single-step is bad, the model/training isn't ready
   and LIBERO is not the problem.
2. SIGReg collapse check: eigenvalue spectrum of embeddings. Collapse to a few
   dims = encoder learned nothing (training issue, not capacity).

Usage: python diagnostics/p0_sanity.py --epoch 200 --n-ep 40
"""
import argparse
import numpy as np
import torch

import os
from _common import load_jepa, encode_frames, DrawerH5, HISTORY, EMB_DIM, RESULTS_DIR


@torch.no_grad()
def teacher_forcing_error(model, emb, act_emb, hs=HISTORY):
    """emb,act_emb: (L,D). Returns per-step (pred, target) single-step errors."""
    L = emb.size(0)
    windows_e, windows_a, targets, prevs = [], [], [], []
    for t in range(hs - 1, L - 1):
        windows_e.append(emb[t - hs + 1:t + 1])       # (hs,D)
        windows_a.append(act_emb[t - hs + 1:t + 1])   # (hs,A)
        targets.append(emb[t + 1])
        prevs.append(emb[t])
    if not windows_e:
        return None
    we = torch.stack(windows_e).cuda()    # (N,hs,D)
    wa = torch.stack(windows_a).cuda()
    tgt = torch.stack(targets).cuda()
    prev = torch.stack(prevs).cuda()
    pred = model.predict(we, wa)[:, -1]   # (N,D) predicted next emb
    pred_err = (pred - tgt).norm(dim=-1)          # model one-step error
    copy_err = (prev - tgt).norm(dim=-1)          # baseline: copy last frame
    rel = pred_err / tgt.norm(dim=-1).clamp_min(1e-6)
    return pred_err.cpu(), copy_err.cpu(), rel.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=200)
    ap.add_argument("--n-ep", type=int, default=40)
    ap.add_argument("--n-spec-frames", type=int, default=6000)
    args = ap.parse_args()

    model = load_jepa(args.epoch)
    h5 = DrawerH5()
    rng = np.random.default_rng(0)
    eps = rng.choice(h5.n_ep, size=min(args.n_ep, h5.n_ep), replace=False)

    # ---- 1. single-step teacher-forcing error ----
    all_pred, all_copy, all_rel = [], [], []
    spec_embs = []
    for ep in eps:
        pix = h5.pixels(ep)
        acts = h5.actions(ep)
        emb = encode_frames(model, pix)                     # (L,D) cpu
        spec_embs.append(emb)
        act_t = torch.from_numpy(acts).float().unsqueeze(0).cuda()  # (1,L,A)
        act_emb = model.action_encoder(act_t)[0].cpu()      # (L,A_emb)
        out = teacher_forcing_error(model, emb, act_emb)
        if out is None:
            continue
        pe, ce, re = out
        all_pred.append(pe); all_copy.append(ce); all_rel.append(re)

    pe = torch.cat(all_pred); ce = torch.cat(all_copy); re = torch.cat(all_rel)
    print("\n===== 1. Single-step teacher-forcing (in-distribution) =====")
    print(f"  episodes={len(eps)}  steps={len(pe)}")
    print(f"  model 1-step L2 err : mean={pe.mean():.4f}  median={pe.median():.4f}")
    print(f"  copy-last  L2 err   : mean={ce.mean():.4f}  median={ce.median():.4f}")
    print(f"  ratio model/copy    : {(pe.mean()/ce.mean()):.3f}  (<1 = beats copy)")
    print(f"  relative err |pred-tgt|/|tgt| : mean={re.mean():.4f}")

    # ---- 2. SIGReg eigenvalue spectrum ----
    E = torch.cat(spec_embs, 0).numpy()
    idx = rng.choice(len(E), size=min(args.n_spec_frames, len(E)), replace=False)
    X = E[idx]
    Xc = X - X.mean(0, keepdims=True)
    cov = (Xc.T @ Xc) / (len(Xc) - 1)
    eigval = np.linalg.eigvalsh(cov)[::-1]
    eigval = np.clip(eigval, 0, None)
    p = eigval / eigval.sum()
    eff_rank = np.exp(-(p * np.log(p + 1e-12)).sum())          # entropy eff. rank
    part_ratio = (eigval.sum() ** 2) / (eigval ** 2).sum()      # participation ratio
    print("\n===== 2. SIGReg / embedding spectrum =====")
    print(f"  dim={EMB_DIM}  frames={len(X)}")
    print(f"  effective rank (entropy)   : {eff_rank:.1f} / {EMB_DIM}")
    print(f"  participation ratio        : {part_ratio:.1f} / {EMB_DIM}")
    print(f"  top-5 eigval share         : {p[:5].sum():.3f}")
    print(f"  eigval[0,1,2,9,49,191]     : "
          + ", ".join(f"{eigval[i]:.3e}" for i in [0,1,2,9,49,EMB_DIM-1]))

    out_path = os.path.join(RESULTS_DIR, "p0_sanity.npz")
    np.savez(out_path,
             pred_err=pe.numpy(), copy_err=ce.numpy(), rel_err=re.numpy(),
             eigval=eigval, eff_rank=eff_rank, part_ratio=part_ratio)
    print(f"\nsaved -> {out_path}")
    h5.close()


if __name__ == "__main__":
    main()
