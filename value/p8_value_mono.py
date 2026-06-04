"""Stage 2 step 4 (gate): C-cost monotonicity — the value-head version of probe 2.

Probe 2 showed the raw-distance cost ||emb-goal||^2 is NON-monotone along demos
(cross-goal Spearman -0.005, ~21 false valleys). Here we recompute the SAME curves
but with cost = C(emb_t, goal) from the trained value head, on the SAME cached embs.

  same  : goal = last frame of the SAME episode (sanity; C is trained on this,
          should be ~perfectly monotone — NOT the real test)
  cross : goal = last frame of a DIFFERENT episode (mirrors eval). THIS is the
          gate: does C generalize, or did it just memorize expert time-to-go?

Usage:
  CUDA_VISIBLE_DEVICES=0 python value/p8_value_mono.py
"""
import argparse
import os

import numpy as np
import torch
from scipy.stats import spearmanr

from value_head import GoalCost

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "emb_cache.npz")
CKPT = os.path.join(HERE, "value_head.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ep", type=int, default=200)
    args = ap.parse_args()
    device = "cuda"

    d = np.load(CACHE)
    emb = torch.tensor(d["emb"].astype(np.float32), device=device)
    off, length = d["ep_offset"], d["ep_len"]

    ck = torch.load(CKPT, map_location=device)
    model = GoalCost(emb_dim=ck["emb_dim"]).to(device).eval()
    model.load_state_dict(ck["state_dict"])
    model.requires_grad_(False)

    rng = np.random.default_rng(1)
    eps = rng.choice(len(off), size=min(args.n_ep, len(off)), replace=False)
    pool = [e for e in range(len(off)) if e not in set(eps.tolist())]
    cross_ep = int(rng.choice(pool))
    co, cl = int(off[cross_ep]), int(length[cross_ep])
    cross_goal = emb[co + cl - 1]                       # last frame of held-out episode

    rows = {"same": [], "cross": []}
    for e in eps:
        o, l = int(off[e]), int(length[e])
        traj = emb[o:o + l]
        progress = np.linspace(0, 1, l)
        for mode, goal in (("same", traj[-1]), ("cross", cross_goal)):
            c = model(traj, goal.expand(l, -1)).cpu().numpy()
            rho, _ = spearmanr(progress, c)
            fv = int((c < c[-1]).sum())
            rows[mode].append((rho, fv))

    print("\n===== Probe 8: C-cost monotonicity (vs probe2 raw-distance) =====")
    print("  baseline (raw dist, M1): same rho~-?, cross rho=-0.005 fv~21")
    for mode in ("same", "cross"):
        arr = np.array(rows[mode], dtype=float)
        rho, fv = arr[:, 0], arr[:, 1]
        print(f"  [{mode:5s}] Spearman(progress,C): mean={np.nanmean(rho):+.3f} "
              f"median={np.nanmedian(rho):+.3f}  (ideal -1.0)")
        print(f"          false-valleys/traj: mean={fv.mean():.1f}  "
              f"frac_monotone(rho<-0.8)={np.mean(rho < -0.8):.2f}")


if __name__ == "__main__":
    main()
