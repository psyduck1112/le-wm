"""Cost monotonicity (consolidates the old p2 / p2b / p17).

Along held-out drawer-close trajectories (open -> closed), for each frame t compute
one or more candidate planning costs and check whether they decrease monotonically
toward the goal. A good cost => Spearman(progress, cost) ~ -1; a non-monotone cost
(false valleys) is one a CEM planner can get stuck in / be driven away by.

Cost modes (`--cost`, csv):
  lewm     ‖enc(f_t) − enc(goal)‖²            original goal-image emb-L2 (the broken one).
           --lewm-goal cross (default) uses another episode's closed frame, mirroring eval.
  decoded  D_φ(emb) -> ‖eef−cab‖+30·|q−closed| post-hoc frozen-encoder decoder (v2 only).
  phys     task_head(emb) -> same shaping       co-trained decoder (phys model only).
  truth    proprio/​drawer_qpos -> same shaping  ground-truth progress (overlay reference).

Models (`--model`): v2 (AutoCostModel + D_φ era) or phys (co-trained task_head).
Plots a per-episode overlay of the requested (min-max normalized) costs and prints a
Spearman table. decoded is evaluated on D_φ's held-out test split.

Usage:
  python diagnostics/p2_cost_mono.py --model phys --epoch 65 --cost lewm,phys,truth
  python diagnostics/p2_cost_mono.py --model v2 --epoch 200 --cost lewm,decoded,truth
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (load_jepa, load_phys, compute_denorm, compute_action_norm,
                     phys_decode, encode_frames, DrawerH5, RESULTS_DIR, HISTORY)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "value"))
from state_decoder import load_decoder, decoded_cost, CABINET, CLOSED  # noqa: E402

COLORS = {"lewm": "crimson", "decoded": "tab:orange", "phys": "seagreen", "truth": "black"}
FRAMESKIP = 5   # config libero_bc_drawer.yaml: 1 world-model frame = 5 raw steps;
                # macro action = 5 stacked raw actions (5*7=35), matching training.


def minmax(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / max(hi - lo, 1e-9)


def false_valleys(cost):
    """#steps with cost strictly below the goal (final) cost = wrong minima."""
    return int((cost < cost[-1]).sum())


def held_out_test_eps(n_total, seed=3, test_frac=0.2):
    """Reproduce train_state_decoder.py's split: first 20% of rng(seed) perm = test."""
    rng = np.random.default_rng(seed)
    perm = rng.choice(n_total, size=n_total, replace=False)
    return perm[: max(1, int(test_frac * n_total))]


@torch.no_grad()
def episode_costs(model, h5, ep, modes, ctx):
    """Per-frame cost arrays for the requested modes on one episode."""
    emb = encode_frames(model, h5.pixels(ep), h5.eye_in_hand(ep)).cuda()   # (L,192)
    out = {}
    if "lewm" in modes:
        out["lewm"] = ((emb - ctx["goal_emb"]) ** 2).sum(-1).cpu().numpy()
    if "decoded" in modes:
        out["decoded"] = decoded_cost(emb, ctx["decoder"]).cpu().numpy()
    if "phys" in modes:
        eef, q = phys_decode(model, emb, ctx["mu"], ctx["sd"])
        cab = torch.as_tensor(CABINET, device=emb.device, dtype=emb.dtype)
        out["phys"] = (torch.linalg.norm(eef - cab, dim=-1)
                       + 30.0 * (q - CLOSED).abs()).cpu().numpy()
    if "truth" in modes:
        prop, q = h5.proprio(ep), h5.drawer_qpos(ep)[:, 0]
        out["truth"] = (np.linalg.norm(prop[:, :3] - CABINET, axis=-1)
                        + 30.0 * np.abs(q - CLOSED))
    return out


@torch.no_grad()
def predictor_rollout(model, emb_true, act_emb, hs=HISTORY):
    """Open-loop autoregressive rollout: seed with the first `hs` TRUE frames, then
    feed the episode's finally-executed actions to the end (replicates jepa.rollout /
    p5). Returns pred_emb (L-hs, D): the IMAGINED embedding for true frames hs..L-1."""
    L = emb_true.size(0)
    emb = emb_true[:hs].clone()                      # (hs,D) seed, grows over time
    preds = []
    for k in range(L - hs):
        win_e = emb[-hs:].unsqueeze(0)               # (1,hs,D)
        win_a = act_emb[k:k + hs].unsqueeze(0)       # (1,hs,A) aligned to window frames
        pred = model.predict(win_e, win_a)[:, -1]    # (1,D) predict next emb
        emb = torch.cat([emb, pred], dim=0)          # feed prediction back (OOD)
        preds.append(pred[0])
    return torch.stack(preds)                        # (L-hs, D)


def _latent_cost(model, emb, mode, ctx):
    """One cost (lewm or phys) evaluated on a latent tensor (N,D) -> np (N,)."""
    if mode == "lewm":
        return ((emb - ctx["goal_emb"]) ** 2).sum(-1).cpu().numpy()
    eef, q = phys_decode(model, emb, ctx["mu"], ctx["sd"])           # phys
    cab = torch.as_tensor(CABINET, device=emb.device, dtype=emb.dtype)
    return (torch.linalg.norm(eef - cab, dim=-1) + 30.0 * (q - CLOSED).abs()).cpu().numpy()


@torch.no_grad()
def episode_costs_rollout(model, h5, ep, modes, ctx, hs=HISTORY):
    """Probe the PREDICTOR: for each latent cost, compute it on BOTH the imagined
    (open-loop rollout) latent AND the real-encoded latent at the same step. The two
    share the identical cost FORM and goal, so any gap is purely predictor drift
    (cost-form mismatch is cancelled). truth = privileged sim cost (physics units),
    a meaningful absolute reference for phys only. Output keys: '<m>' (imagined),
    '<m>_real' (real-encoded), 'truth'. World-model rate (frameskip=5): frame t = raw
    frame 5t, macro action t = 5 stacked raw actions (35-d), matching training."""
    fs = FRAMESKIP
    raw = h5.actions(ep)                                             # (L,7) raw actions
    n = raw.shape[0] // fs                                           # world-model steps
    sl = slice(0, n * fs, fs)                                        # raw -> ds indices
    emb_true = encode_frames(model, h5.pixels(ep)[sl], h5.eye_in_hand(ep)[sl]).cuda()  # (n,192)
    # z-score each 7-d action (training normalized it) THEN stack 5 into the 35-d macro
    raw_n = (raw[:n * fs] - ctx["amu"]) / ctx["asd"]                 # (n*fs,7) normalized
    macro = raw_n.reshape(n, fs * raw.shape[1])                      # (n,35)
    act = torch.from_numpy(macro).float().unsqueeze(0).cuda()
    act_emb = model.action_encoder(act)[0]                           # (n,A)
    pred_emb = predictor_rollout(model, emb_true, act_emb, hs)       # (n-hs,D) imagined
    real_emb = emb_true[hs:]                                         # (n-hs,D) real obs latent
    out = {}
    for m in modes:
        if m in ("lewm", "phys"):
            out[m] = _latent_cost(model, pred_emb, m, ctx)           # imagined
            out[m + "_real"] = _latent_cost(model, real_emb, m, ctx)  # real-encoded
    if "truth" in modes:
        prop = h5.proprio(ep)[sl][hs:]                               # ds + align to preds
        q = h5.drawer_qpos(ep)[sl][hs:, 0]
        out["truth"] = (np.linalg.norm(prop[:, :3] - CABINET, axis=-1)
                        + 30.0 * np.abs(q - CLOSED))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["v2", "phys"], default="phys")
    ap.add_argument("--epoch", type=int, default=65)
    ap.add_argument("--cost", default="lewm,phys,truth", help="csv of lewm,decoded,phys,truth")
    ap.add_argument("--lewm-goal", choices=["same", "cross"], default="cross")
    ap.add_argument("--n-ep", type=int, default=6)
    ap.add_argument("--rollout", action="store_true",
                    help="costs along the PREDICTOR open-loop rollout (imagined latent, "
                         "seeded by 3 true frames + chosen actions) instead of true emb.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--decoder", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "value", "state_decoder.pt"))
    args = ap.parse_args()

    modes = args.cost.split(",")
    if "phys" in modes and args.model != "phys":
        raise SystemExit("phys cost needs --model phys")
    if "decoded" in modes and args.model != "v2":
        raise SystemExit("decoded cost (D_φ) needs --model v2")

    model = load_phys(args.epoch) if args.model == "phys" else load_jepa(args.epoch)
    ctx = {}
    if "phys" in modes:
        ctx["mu"], ctx["sd"] = compute_denorm()
    if "decoded" in modes:
        ctx["decoder"] = load_decoder(args.decoder)
    if args.rollout:   # predictor needs z-scored actions (training normalized them)
        amu, asd = compute_action_norm()
        ctx["amu"], ctx["asd"] = amu.cpu().numpy(), asd.cpu().numpy()

    h5 = DrawerH5()
    rng = np.random.default_rng(args.seed)
    # decoded must be scored on D_φ's held-out split; otherwise any episodes
    pool = held_out_test_eps(h5.n_ep) if "decoded" in modes else np.arange(h5.n_ep)
    eps = rng.choice(pool, size=min(args.n_ep, len(pool)), replace=False)

    if "lewm" in modes and args.lewm_goal == "cross":
        cross = int(rng.choice([e for e in pool if e not in set(eps.tolist())]))
        ctx["goal_emb"] = encode_frames(model, h5.goal_frame(cross)[None],
                                        h5.goal_eye_frame(cross)[None]).cuda()[0]

    ncol = 3
    nrow = int(np.ceil(len(eps) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.4 * nrow), squeeze=False)
    rhos = {m: [] for m in modes}                                  # on-tube: ρ(progress,cost)
    fvs = {m: [] for m in modes}
    latent_modes = [m for m in modes if m in ("lewm", "phys")]
    rho_pred = {m: [] for m in latent_modes}    # rollout: ρ(imagined, real) = predictor fidelity
    nrmse = {m: [] for m in latent_modes}       # rollout: ‖imagined−real‖ / real range (magnitude)

    for k, ep in enumerate(eps):
        ep = int(ep)
        if "lewm" in modes and args.lewm_goal == "same":
            emb = encode_frames(model, h5.pixels(ep), h5.eye_in_hand(ep)).cuda()
            ctx["goal_emb"] = emb[-1]
        costs = (episode_costs_rollout if args.rollout else episode_costs)(
            model, h5, ep, modes, ctx)
        L = len(next(iter(costs.values())))
        # on-tube: x = task progress 0→1; rollout: x = rollout step (shows compounding)
        x = np.arange(L) if args.rollout else np.linspace(0, 1, L)
        ax = axes[k // ncol][k % ncol]

        if args.rollout:
            for m in modes:
                if m == "truth":
                    ax.plot(x, minmax(costs["truth"]), color=COLORS["truth"], ls=":",
                            lw=1.6, label="truth (privileged)")
                    continue
                imag, real = costs[m], costs[m + "_real"]
                rp = spearmanr(imag, real).correlation                  # predictor fidelity
                rho_pred[m].append(rp)
                both = np.concatenate([imag, real]); lo, hi = both.min(), both.max()
                d = max(hi - lo, 1e-9)
                nrmse[m].append(float(np.sqrt(((imag - real) ** 2).mean()) / d))
                # SHARED normalization (imagined+real) so the gap = predictor drift shows
                ax.plot(x, (imag - lo) / d, color=COLORS[m], ls="-", lw=1.8,
                        label=f"{m} imagined (ρ_pred={rp:+.2f})")
                ax.plot(x, (real - lo) / d, color=COLORS[m], ls="--", lw=1.3, alpha=0.7,
                        label=f"{m} real-encoded")
        else:
            for m in modes:
                r = spearmanr(x, costs[m]).correlation
                rhos[m].append(r); fvs[m].append(false_valleys(costs[m]))
                ls = "--" if m == "truth" else "-"
                ax.plot(x, minmax(costs[m]), color=COLORS[m], ls=ls, lw=1.8,
                        label=f"{m} (ρ={r:+.2f})")
        ax.set_title(f"ep {ep} (L={L})", fontsize=9)
        ax.set_xlabel("rollout step k" if args.rollout else "task progress (open→closed)")
        ax.set_ylabel("cost (norm.)")
        ax.legend(fontsize=7, loc="upper right")
    for j in range(len(eps), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    tag = "rollout" if args.rollout else "on-tube"
    if args.rollout:
        head = "  ".join(f"{m} ρ_pred̄={np.nanmean(rho_pred[m]):+.2f}" for m in latent_modes)
        title = (f"Predictor fidelity in cost space — model={args.model} epoch={args.epoch}\n"
                 f"(solid=imagined latent, dashed=real-encoded; gap=predictor drift; "
                 f"ρ_pred→+1)\n{head}")
    else:
        head = "  ".join(f"{m} ρ̄={np.nanmean(rhos[m]):+.2f}" for m in modes)
        title = (f"Cost monotonicity — model={args.model} epoch={args.epoch} "
                 f"(ideal ρ=-1)\n{head}")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90 if args.rollout else 0.94])
    out = os.path.join(RESULTS_DIR, f"cost_{tag}_{args.model}.png")
    fig.savefig(out, dpi=120)

    print(f"\n===== cost {tag} (model={args.model}, n={len(eps)}) =====")
    if args.rollout:
        print(f"  imagined (predictor) vs real-encoded latent — same cost form, gap = predictor drift")
        print(f"  {'cost':>8} {'ρ_pred mean':>12} {'ρ_pred med':>11} {'nRMSE mean':>11}")
        for m in latent_modes:
            r = np.array(rho_pred[m], float)
            print(f"  {m:>8} {np.nanmean(r):>+12.3f} {np.nanmedian(r):>+11.3f} "
                  f"{np.mean(nrmse[m]):>11.3f}")
    else:
        print(f"  {'cost':>8} {'ρ mean':>8} {'ρ med':>8} {'fv/traj':>8} {'frac ρ<-0.8':>12}")
        for m in modes:
            r = np.array(rhos[m], float)
            print(f"  {m:>8} {np.nanmean(r):>+8.3f} {np.nanmedian(r):>+8.3f} "
                  f"{np.mean(fvs[m]):>8.1f} {np.mean(r < -0.8):>12.2f}")
    print(f"saved -> {out}")
    h5.close()


if __name__ == "__main__":
    main()
