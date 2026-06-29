"""Re-plot p9b cost_rollout on the PLANNER's own (OOD) trajectories WITHOUT normalizing.

Reuses the saved trajectory npz (results/p9b_<cost>_traj.npz) — only re-encodes the
stored pixels, no MPC/env re-run. Per episode, dual y-axis with RAW (un-normalized) values:
  - LEFT  axis  (phys, 物理单位): phys cost = ‖eef−CABINET‖+30·|q−CLOSED| on the latent,
                 plus the privileged `truth` (same units, dotted ref).
  - RIGHT axis  (latent, emb-L2): lewm cost = ‖emb−goal_emb‖² on the latent.
solid = REAL-encoded, dashed = imagined (predictor open-loop), dotted = truth.

Both figures (lewm-planned & phys-planned) and every subplot share the SAME absolute
y-axis range (phys & lewm computed globally over both files) so curves are comparable
across subplots and across the two images.

Usage:
  CUDA_VISIBLE_DEVICES=0 python diagnostics/p9b_replot_dualaxis.py --epoch 200
  ... --phys-ylim 0 6 --lewm-ylim 250 380   # override fixed ranges
"""
import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

from _common import (load_phys, compute_denorm, encode_frames, DrawerH5,
                     RESULTS_DIR, HISTORY)
from p2_cost_mono import _latent_cost, predictor_rollout, COLORS
from state_decoder import CABINET, CLOSED  # noqa: E402


@torch.no_grad()
def compute_episode(model, ep, ctx):
    """Re-encode one stored episode and return all cost curves (no plotting)."""
    emb_true = encode_frames(model, np.stack(ep["pixels"]),
                             np.stack(ep["eye_in_hand"])).cuda()            # (T,D)
    real_emb = emb_true[HISTORY:]                                          # real-encoded latent
    # imagined: seed first HISTORY true frames, predictor open-loop rolls the rest
    macro = np.stack(ep["action"])[None]                                  # (1,T,35) normalized
    act_emb = model.action_encoder(torch.from_numpy(macro).float().cuda())[0]
    pred_emb = predictor_rollout(model, emb_true, act_emb, HISTORY)        # (T-hs,D) imagined
    prop = np.stack(ep["proprio"])[HISTORY:]
    q = np.stack(ep["drawer_qpos"])[HISTORY:]
    truth = np.linalg.norm(prop[:, :3] - CABINET, axis=-1) + 30.0 * np.abs(q - CLOSED)
    return {
        "phys_real": _latent_cost(model, real_emb, "phys", ctx),
        "phys_imag": _latent_cost(model, pred_emb, "phys", ctx),
        "lewm_real": _latent_cost(model, real_emb, "lewm", ctx),
        "lewm_imag": _latent_cost(model, pred_emb, "lewm", ctx),
        "truth": truth,
        "success": bool(ep["success"]),
        "T": len(emb_true),
    }


def plot_file(cost_name, curves, phys_ylim, lewm_ylim, out_png):
    ncol = 3
    nrow = int(np.ceil(len(curves) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 3.6 * nrow), squeeze=False)

    for k, c in enumerate(curves):
        x = np.arange(len(c["phys_real"]))
        rho_phys = spearmanr(c["phys_imag"], c["phys_real"]).correlation
        rho_lewm = spearmanr(c["lewm_imag"], c["lewm_real"]).correlation

        axL = axes[k // ncol][k % ncol]
        axR = axL.twinx()
        l1, = axL.plot(x, c["phys_real"], color=COLORS["phys"], lw=1.9, label="phys real")
        li1, = axL.plot(x, c["phys_imag"], color=COLORS["phys"], ls="--", lw=1.4, alpha=0.8,
                        label=f"phys imagined (rho={rho_phys:+.2f})")
        l2, = axL.plot(x, c["truth"], color=COLORS["truth"], ls=":", lw=1.6,
                       label="truth (privileged)")
        l3, = axR.plot(x, c["lewm_real"], color=COLORS["lewm"], lw=1.9, alpha=0.85,
                       label="lewm real")
        li3, = axR.plot(x, c["lewm_imag"], color=COLORS["lewm"], ls="--", lw=1.4, alpha=0.7,
                        label=f"lewm imagined (rho={rho_lewm:+.2f})")

        axL.set_ylim(*phys_ylim)                                            # FIXED, shared
        axR.set_ylim(*lewm_ylim)                                            # FIXED, shared
        axL.set_ylabel("phys  ||eef-CAB||+30|q-C|", color=COLORS["phys"], fontsize=8)
        axR.set_ylabel("latent  ||emb-goal||^2", color=COLORS["lewm"], fontsize=8)
        axL.tick_params(axis="y", labelcolor=COLORS["phys"])
        axR.tick_params(axis="y", labelcolor=COLORS["lewm"])
        axL.set_xlabel("rollout step k")
        succ = "SUCCESS" if c["success"] else "fail"
        axL.set_title(f"ep{k} {succ} (T={c['T']})", fontsize=9)
        axL.legend(handles=[l1, li1, l2, l3, li3], fontsize=6, loc="upper right")

    for j in range(len(curves), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"p9b planner (OOD) trajectory - cost={cost_name}  RAW values, dual axis\n"
                 f"(left=phys {phys_ylim}, right=lewm latent-L2 {lewm_ylim};  "
                 f"solid=real, dashed=imagined, dotted=truth)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"saved -> {out_png}", flush=True)


def plot_file_norm(cost_name, curves, out_png):
    """Normalized single-axis version: each mode min-max'd to [0,1] so lewm & phys
    coexist. Unified convention: solid=real, dashed=imagined, dotted=truth."""
    ncol = 3
    nrow = int(np.ceil(len(curves) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.4 * nrow), squeeze=False)
    agg = {"phys": [], "lewm": []}

    for k, c in enumerate(curves):
        x = np.arange(len(c["phys_real"]))
        ax = axes[k // ncol][k % ncol]
        handles = []
        for m in ("lewm", "phys"):
            real, imag = c[f"{m}_real"], c[f"{m}_imag"]
            rho = spearmanr(imag, real).correlation
            agg[m].append(rho)
            both = np.concatenate([real, imag])
            lo, hi = both.min(), both.max()
            d = max(hi - lo, 1e-9)
            hs, = ax.plot(x, (real - lo) / d, color=COLORS[m], ls="-", lw=1.8,
                          label=f"{m} real")
            ax.plot(x, (imag - lo) / d, color=COLORS[m], ls="--", lw=1.3, alpha=0.8,
                    label=f"{m} imagined (rho={rho:+.2f})")
            handles.append(hs)
            handles.append(ax.lines[-1])
        t = c["truth"]
        tn = (t - t.min()) / max(t.max() - t.min(), 1e-9)
        ht, = ax.plot(x, tn, color=COLORS["truth"], ls=":", lw=1.6, label="truth (privileged)")
        handles.append(ht)
        ax.set_ylim(-0.05, 1.05)
        succ = "SUCCESS" if c["success"] else "fail"
        ax.set_title(f"ep{k} {succ} (T={c['T']})", fontsize=9)
        ax.set_xlabel("rollout step k"); ax.set_ylabel("cost (norm.)")
        ax.legend(handles=handles, fontsize=6, loc="upper right")

    for j in range(len(curves), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    head = "  ".join(f"{m} rho_imag-real={np.nanmean(agg[m]):+.2f}" for m in ("lewm", "phys"))
    fig.suptitle(f"p9b planner (OOD) trajectory - cost={cost_name}  NORMALIZED per mode\n"
                 f"(solid=real, dashed=imagined, dotted=truth)\n{head}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"saved -> {out_png}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=200)
    ap.add_argument("--goal-ep", type=int, default=0,
                    help="DrawerH5 episode whose drawer-closed frame is the lewm goal")
    ap.add_argument("--phys-ylim", type=float, nargs=2, default=None,
                    help="default: auto global min/max over both files")
    ap.add_argument("--lewm-ylim", type=float, nargs=2, default=None,
                    help="default: auto global min/max over both files")
    args = ap.parse_args()

    model = load_phys(args.epoch)
    mu, sd = compute_denorm()
    ctx = {"mu": mu, "sd": sd}
    h5 = DrawerH5()
    ge = args.goal_ep
    ctx["goal_emb"] = encode_frames(model, h5.goal_frame(ge)[None],
                                    h5.goal_eye_frame(ge)[None]).cuda()[0]
    h5.close()

    # pass 1: encode + compute every episode of both files (cache, encode once)
    cache = {}
    for cost_name in ("lewm", "phys"):
        traj = os.path.join(RESULTS_DIR, f"p9b_{cost_name}_traj.npz")
        if not os.path.exists(traj):
            print(f"skip {cost_name}: {traj} not found", flush=True)
            continue
        episodes = np.load(traj, allow_pickle=True)["episodes"]
        eps = [e for e in episodes if len(e["action"]) > HISTORY + 1]
        cache[cost_name] = [compute_episode(model, e, ctx) for e in eps]

    # global y-ranges over BOTH files so all subplots & both images share one scale
    all_c = [c for cs in cache.values() for c in cs]

    def auto_range(keys):
        lo = min(min(c[k].min() for k in keys) for c in all_c)
        hi = max(max(c[k].max() for k in keys) for c in all_c)
        pad = 0.05 * (hi - lo)
        return (float(lo - pad), float(hi + pad))

    # phys axis also covers truth (same units); lewm axis covers its two curves
    phys_ylim = tuple(args.phys_ylim) if args.phys_ylim else auto_range(
        ["phys_real", "phys_imag", "truth"])
    lewm_ylim = tuple(args.lewm_ylim) if args.lewm_ylim else auto_range(
        ["lewm_real", "lewm_imag"])
    print(f"fixed ranges -> phys {phys_ylim}  lewm {lewm_ylim}", flush=True)

    # pass 2: plot both the absolute dual-axis and the normalized single-axis versions
    for cost_name, curves in cache.items():
        plot_file(cost_name, curves, phys_ylim, lewm_ylim,
                  os.path.join(RESULTS_DIR, f"cost_rollout_p9b_{cost_name}_dualaxis.png"))
        plot_file_norm(cost_name, curves,
                       os.path.join(RESULTS_DIR, f"cost_rollout_p9b_{cost_name}_norm.png"))


if __name__ == "__main__":
    main()
