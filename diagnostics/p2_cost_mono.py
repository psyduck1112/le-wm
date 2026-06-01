"""Probe 2 (LIBERO_DIAGNOSTICS.md 探测2): cost monotonicity.

Along known-good (BC) trajectories, compute cost_t = ||encode(frame_t) - goal_emb||^2
and check whether it decreases monotonically toward task completion. Non-monotone /
false valleys => goal-image emb-MSE is a bad planning cost (not a model-capacity issue).

Two goal settings:
  - same  : goal = last frame of the SAME episode (sanity; should be ~monotone)
  - cross : goal = last frame of a DIFFERENT episode (mirrors eval, where the goal
            image comes from another demo). This is the real cost-quality test.

Usage: python diagnostics/p2_cost_mono.py --epoch 200 --n-ep 30
"""
import argparse
import os
import numpy as np
import torch
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import load_jepa, encode_frames, DrawerH5, RESULTS_DIR


def false_valleys(cost):
    """Count steps whose cost is strictly below the final (goal) cost = false minima
    that a planner could get stuck in instead of the true goal."""
    return int((cost < cost[-1]).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=200)
    ap.add_argument("--n-ep", type=int, default=30)
    args = ap.parse_args()

    model = load_jepa(args.epoch)
    h5 = DrawerH5()
    rng = np.random.default_rng(1)
    eps = rng.choice(h5.n_ep, size=min(args.n_ep, h5.n_ep), replace=False)

    # fixed cross-goal: last frame of a held-out episode not in eps
    pool = [e for e in range(h5.n_ep) if e not in set(eps)]
    cross_ep = int(rng.choice(pool))
    cross_goal_emb = encode_frames(model, h5.goal_frame(cross_ep)[None]).cuda()[0]

    rows = {"same": [], "cross": []}
    curves = []
    for ep in eps:
        pix = h5.pixels(ep)
        emb = encode_frames(model, pix).cuda()             # (L,D)
        L = emb.size(0)
        progress = np.linspace(0, 1, L)
        same_goal = emb[-1]
        for mode, goal in (("same", same_goal), ("cross", cross_goal_emb)):
            cost = ((emb - goal) ** 2).sum(-1).cpu().numpy()  # (L,)
            rho, _ = spearmanr(progress, cost)             # want strongly negative
            rows[mode].append((rho, false_valleys(cost), L))
        # store cross curve normalized for plotting
        c = ((emb - cross_goal_emb) ** 2).sum(-1).cpu().numpy()
        curves.append((progress, c / max(c.max(), 1e-9)))

    print("\n===== Probe 2: cost monotonicity =====")
    for mode in ("same", "cross"):
        arr = np.array([r[:2] for r in rows[mode]], dtype=float)
        rho, fv = arr[:, 0], arr[:, 1]
        print(f"  [{mode:5s}] Spearman(progress,cost): mean={np.nanmean(rho):+.3f} "
              f"median={np.nanmedian(rho):+.3f}  (ideal -1.0)")
        print(f"          false-valleys/traj: mean={fv.mean():.1f}  "
              f"frac_monotone(rho<-0.8)={np.mean(rho < -0.8):.2f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for progress, c in curves:
        ax.plot(progress, c, alpha=0.4)
    ax.set_xlabel("task progress (normalized step)")
    ax.set_ylabel("cost ||enc(f_t)-goal_emb||^2 (norm.)")
    ax.set_title(f"Probe 2 cross-goal cost vs progress (n={len(curves)})")
    out_png = os.path.join(RESULTS_DIR, "p2_cost_mono.png")
    fig.tight_layout(); fig.savefig(out_png, dpi=120)
    np.savez(os.path.join(RESULTS_DIR, "p2_cost_mono.npz"),
             same=np.array([r[:2] for r in rows["same"]]),
             cross=np.array([r[:2] for r in rows["cross"]]))
    print(f"\nsaved -> {out_png}")
    h5.close()


if __name__ == "__main__":
    main()
