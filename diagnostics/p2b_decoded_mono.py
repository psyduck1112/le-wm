"""闸1 (gate 1): cost monotonicity with the DECODED cost (D_φ) vs raw emb-L2.

p2 showed goal-image emb-L2 is non-monotone, worst in the CROSS-goal setting
(ρ≈0) which is what eval actually uses. Here we swap in `decoded_cost`: D_φ
decodes (eef, drawer_qpos) from emb and we reassemble p6's shaping
    cost = ‖eef − cabinet‖ + 30·|q − closed|.
This is goal-IMAGE-INDEPENDENT (the target = drawer-closed is baked into CLOSED),
so the cross-goal pathology cannot occur by construction.

Evaluated ONLY on episodes D_φ never trained on (reproduces the trainer's
seed-3 episode split, takes eval eps from the held-out test set).

  decoded ρ ≲ −0.7  => cost is monotone toward task completion -> proceed to 闸2.
  decoded ρ ~ emb-L2 => decode didn't help -> stop, rethink before spending MPC.

Usage:
  MUJOCO_GL=egl EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
      python diagnostics/p2b_decoded_mono.py --n-ep 40
"""
import argparse
import os
import sys
import numpy as np
import torch
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import load_jepa, encode_frames, DrawerH5, RESULTS_DIR

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "value"))
from state_decoder import load_decoder, decoded_cost  # noqa: E402


def false_valleys(cost):
    return int((cost < cost[-1]).sum())


def held_out_test_eps(n_total, seed=3, test_frac=0.2):
    """Reproduce train_state_decoder.py's split: rng(seed).choice(N,N) perm, first 20% = test."""
    rng = np.random.default_rng(seed)
    perm = rng.choice(n_total, size=n_total, replace=False)
    n_test = max(1, int(test_frac * n_total))
    return perm[:n_test]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=200)
    ap.add_argument("--decoder", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "value", "state_decoder.pt"))
    ap.add_argument("--n-ep", type=int, default=40)
    args = ap.parse_args()

    model = load_jepa(args.epoch)
    decoder = load_decoder(args.decoder)
    h5 = DrawerH5()

    test_eps = held_out_test_eps(h5.n_ep)           # D_φ never saw these
    rng = np.random.default_rng(1)
    eps = rng.choice(test_eps, size=min(args.n_ep, len(test_eps)), replace=False)
    cross_ep = int(rng.choice([e for e in test_eps if e not in set(eps.tolist())]))
    cross_goal_emb = encode_frames(model, h5.goal_frame(cross_ep)[None],
                                   h5.goal_eye_frame(cross_ep)[None]).cuda()[0]
    print(f"  eval on {len(eps)} held-out eps; cross-goal ep={cross_ep}")

    rows = {"embL2_same": [], "embL2_cross": [], "decoded": []}
    curves = []
    for ep in eps:
        emb = encode_frames(model, h5.pixels(ep), h5.eye_in_hand(ep)).cuda()  # (L,192)
        L = emb.size(0)
        progress = np.linspace(0, 1, L)
        costs = {
            "embL2_same":  ((emb - emb[-1]) ** 2).sum(-1).cpu().numpy(),
            "embL2_cross": ((emb - cross_goal_emb) ** 2).sum(-1).cpu().numpy(),
            "decoded":     decoded_cost(emb, decoder).cpu().numpy(),
        }
        for k, c in costs.items():
            rho, _ = spearmanr(progress, c)
            rows[k].append((rho, false_valleys(c)))
        c = costs["decoded"]
        curves.append((progress, c / max(c.max(), 1e-9)))

    print("\n===== 闸1: decoded-cost monotonicity (held-out eps) =====")
    print(f"  {'cost':>12} {'ρ mean':>8} {'ρ med':>8} {'fv/traj':>8} {'frac ρ<-0.8':>12}")
    summary = {}
    for k in ("embL2_same", "embL2_cross", "decoded"):
        arr = np.array(rows[k], dtype=float)
        rho, fv = arr[:, 0], arr[:, 1]
        print(f"  {k:>12} {np.nanmean(rho):>+8.3f} {np.nanmedian(rho):>+8.3f} "
              f"{fv.mean():>8.1f} {np.mean(rho < -0.8):>12.2f}")
        summary[k] = (float(np.nanmean(rho)), float(np.nanmedian(rho)), float(fv.mean()))

    verdict = "PASS (proceed to 闸2)" if summary["decoded"][0] <= -0.7 else "FAIL (rethink)"
    print(f"\n  decoded ρ mean = {summary['decoded'][0]:+.3f}  ->  {verdict}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for progress, c in curves:
        ax.plot(progress, c, alpha=0.4)
    ax.set_xlabel("task progress (normalized step)")
    ax.set_ylabel("decoded cost (norm.)")
    ax.set_title(f"Gate-1: decoded cost vs progress (n={len(curves)}, rho_bar={summary['decoded'][0]:+.2f})")
    out_png = os.path.join(RESULTS_DIR, "p2b_decoded_mono.png")
    fig.tight_layout(); fig.savefig(out_png, dpi=120)
    np.savez(os.path.join(RESULTS_DIR, "p2b_decoded_mono.npz"),
             **{k: np.array(rows[k]) for k in rows})
    print(f"saved -> {out_png}")
    h5.close()


if __name__ == "__main__":
    main()
