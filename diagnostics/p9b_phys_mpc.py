"""Probe 9b — real end-to-end MPC with the PHYS model (macro actions + phys cost).

Unblocks 闸3 for the co-trained phys model. Combines three pieces:
  - p9's dual-camera LIBERO harness (drive the real env directly, both cams populated);
  - the native eval's MACRO-ACTION mechanism (frameskip=5): CEM plans in the 35-d macro
    space (= 5 stacked 7-d raw actions), each macro unpacked to 5 env.step() calls;
  - the phys cost (goal-image free): rollout the learned predictor over candidate macro
    actions, decode the final imagined latent via task_head -> ‖eef−CABINET‖ + 30·|q−CLOSED|.

Actions are z-scored with the training stats (compute_action_norm) BEFORE the model sees
them and de-normalized back to the raw [-1,1] box before env.step — matching training.

Besides the success rate, every episode's WORLD-MODEL-RATE trajectory is recorded
(pixels, eye_in_hand, normalized macro action, true proprio, true drawer_qpos) and saved,
then the imagined-vs-real cost_rollout figure is regenerated ON THE PLANNER'S OWN (OOD)
trajectory — the test that dataset episodes (expert tube) cannot give.

Usage:
  MUJOCO_GL=egl EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
    python diagnostics/p9b_phys_mpc.py --epoch 200 --n-eval 6 --budget 150 \
      --video-dir diagnostics/results/p9b_videos
"""
import argparse
import os
import time
from collections import deque

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from robosuite.utils.transform_utils import quat2axisangle

from _common import (load_phys, compute_denorm, compute_action_norm, phys_decode,
                     transform_pixels, encode_frames, DrawerH5, RESULTS_DIR, HISTORY)
from p6_oracle import (WARMUP, build_env, find_drawer_joint, EEF_SITE, DRAWER_BODY)
from p2_cost_mono import predictor_rollout, _latent_cost, FRAMESKIP, COLORS
from state_decoder import CABINET, CLOSED  # noqa: E402

AGENT = "agentview_image"
EYE = "robot0_eye_in_hand_image"


def proprio_of(obs):
    """8-d proprio matching training layout: eef_pos(3)+axis_angle(3)+gripper(2)."""
    return np.concatenate([
        obs["robot0_eef_pos"].astype(np.float32),
        quat2axisangle(obs["robot0_eef_quat"]).astype(np.float32),
        obs["robot0_gripper_qpos"].astype(np.float32),
    ])


def stack_hist(buf, n):
    """deque of (H,W,3) uint8 -> torch (1, n, 3, 224, 224) cuda (training-norm)."""
    return transform_pixels(np.stack(list(buf))).unsqueeze(0).cuda()


@torch.no_grad()
def batched_rollout(model, seed, full_act, nf, hs=HISTORY):
    """Open-loop rollout for N candidates. seed (hs,D) shared; full_act (N, hs+nf, 35)
    normalized macro actions. Returns final imagined latent (N, D)."""
    N, D = full_act.shape[0], seed.shape[-1]
    emb = seed.unsqueeze(0).expand(N, hs, D).clone()              # (N,hs,D)
    act_emb = model.action_encoder(torch.from_numpy(full_act).float().cuda())  # (N,hs+nf,A)
    for k in range(nf):
        win_e = emb[:, -hs:]                                      # (N,hs,D)
        win_a = act_emb[:, k:k + hs]                              # (N,hs,A)
        pred = model.predict(win_e, win_a)[:, -1:]               # (N,1,D)
        emb = torch.cat([emb, pred], dim=1)
    return emb[:, -1]                                            # (N,D)


@torch.no_grad()
def cem_plan_phys(model, pix_buf, eye_buf, macro_ctx, cfg, ctx):
    """One CEM plan in the 35-d (normalized) macro space, scored by phys cost on the
    predictor's final imagined latent. Returns mean future macros (nf, 35) normalized."""
    hs = HISTORY
    seed = model.encode({"pixels": stack_hist(pix_buf, hs),
                         "eye_in_hand": stack_hist(eye_buf, hs)})["emb"][0]  # (hs,D)
    act_ctx = np.stack(list(macro_ctx))                          # (hs,35) normalized history

    nf, N, topk, n_iter, rng = cfg["nf"], cfg["N"], cfg["topk"], cfg["n_iter"], cfg["rng"]
    amu, asd = cfg["amu"], cfg["asd"]                            # (7,)
    lo = np.tile((-1.0 - amu) / asd, FRAMESKIP)                  # (35,) raw [-1,1] -> norm box
    hi = np.tile((1.0 - amu) / asd, FRAMESKIP)
    cab = torch.as_tensor(CABINET, device=seed.device, dtype=seed.dtype)

    mean = np.zeros((nf, 35), np.float32)
    var = np.full((nf, 35), cfg["var_scale"], np.float32)
    for _ in range(n_iter):
        fut = mean[None] + np.sqrt(var)[None] * rng.standard_normal((N, nf, 35)).astype(np.float32)
        fut = np.clip(fut, lo[None, None], hi[None, None])       # (N,nf,35)
        full = np.concatenate([np.broadcast_to(act_ctx[None], (N, hs, 35)), fut], axis=1)
        final = batched_rollout(model, seed, full, nf, hs)       # (N,D)
        if cfg["cost"] == "phys":                                # goal-free physics shaping
            eef, q = phys_decode(model, final, ctx["mu"], ctx["sd"])
            cost = (torch.linalg.norm(eef - cab, dim=-1) + 30.0 * (q - CLOSED).abs()).cpu().numpy()
        else:                                                    # lewm: goal-image emb-L2
            cost = ((final - ctx["goal_emb"]) ** 2).sum(-1).cpu().numpy()
        elite = np.argsort(cost)[:topk]
        mean, var = fut[elite].mean(0), fut[elite].var(0) + 1e-6
    return mean                                                  # (nf,35)


def run_episode(env, model, init_state, cfg, ctx, drawer, rec, frames):
    env.reset(); env.set_init_state(init_state)
    obs = None
    for _ in range(WARMUP):
        obs, _, _, _ = env.step(np.zeros(7))
    pix_buf = deque([obs[AGENT].copy()] * HISTORY, maxlen=HISTORY)
    eye_buf = deque([obs[EYE].copy()] * HISTORY, maxlen=HISTORY)
    macro_ctx = deque([np.zeros(35, np.float32)] * HISTORY, maxlen=HISTORY)
    amu, asd, fs = cfg["amu"], cfg["asd"], FRAMESKIP
    budget, receding = cfg["budget"], cfg["receding"]

    steps = 0
    while steps < budget:
        mean = cem_plan_phys(model, pix_buf, eye_buf, macro_ctx, cfg, ctx)   # (nf,35) norm
        for h in range(receding):
            macro_n = mean[h].astype(np.float32)                 # normalized macro
            # record the world-model-rate frame BEFORE executing this macro
            rec["pixels"].append(obs[AGENT].copy())
            rec["eye_in_hand"].append(obs[EYE].copy())
            rec["action"].append(macro_n.copy())                 # NORMALIZED macro (model space)
            rec["proprio"].append(proprio_of(obs))
            rec["drawer_qpos"].append(np.float32(env.sim.data.qpos[drawer[1]]))
            # de-normalize each of the 5 stacked actions -> raw [-1,1] -> step
            macro_raw = np.clip(macro_n.reshape(fs, 7) * asd[None] + amu[None], -1, 1)
            for j in range(fs):
                if frames is not None:
                    frames.append(obs[AGENT].copy())
                obs, _, _, _ = env.step(macro_raw[j].astype(np.float32))
                steps += 1
                if steps >= budget:
                    break
            pix_buf.append(obs[AGENT].copy())
            eye_buf.append(obs[EYE].copy())
            macro_ctx.append(macro_n)
            if env.check_success():
                return True, steps
            if steps >= budget:
                break
    return env.check_success(), steps


@torch.no_grad()
def plot_cost_rollout(model, episodes, ctx, goal_emb, out_png):
    """imagined-vs-real cost_rollout on the PLANNER's own WM-rate trajectories."""
    modes = ["lewm", "phys"]
    eps = [e for e in episodes if len(e["action"]) > HISTORY + 1]
    ncol = 3; nrow = int(np.ceil(len(eps) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.4 * nrow), squeeze=False)
    from scipy.stats import spearmanr
    agg = {m: [] for m in modes}
    ctx = dict(ctx); ctx["goal_emb"] = goal_emb
    for k, ep in enumerate(eps):
        emb_true = encode_frames(model, np.stack(ep["pixels"]),
                                 np.stack(ep["eye_in_hand"])).cuda()          # (T,D)
        macro = np.stack(ep["action"])[None]                                  # (1,T,35) normalized
        act_emb = model.action_encoder(torch.from_numpy(macro).float().cuda())[0]
        pred_emb = predictor_rollout(model, emb_true, act_emb, HISTORY)       # (T-hs,D)
        real_emb = emb_true[HISTORY:]
        prop = np.stack(ep["proprio"])[HISTORY:]
        q = np.stack(ep["drawer_qpos"])[HISTORY:]
        truth = np.linalg.norm(prop[:, :3] - CABINET, axis=-1) + 30.0 * np.abs(q - CLOSED)
        x = np.arange(len(pred_emb))
        ax = axes[k // ncol][k % ncol]
        for m in modes:
            imag = _latent_cost(model, pred_emb, m, ctx)
            real = _latent_cost(model, real_emb, m, ctx)
            rp = spearmanr(imag, real).correlation
            agg[m].append(rp)
            both = np.concatenate([imag, real]); lo, hi = both.min(), both.max()
            d = max(hi - lo, 1e-9)
            ax.plot(x, (imag - lo) / d, color=COLORS[m], ls="-", lw=1.8,
                    label=f"{m} imagined (ρ_pred={rp:+.2f})")
            ax.plot(x, (real - lo) / d, color=COLORS[m], ls="--", lw=1.3, alpha=0.7)
        tmm = (truth - truth.min()) / max(truth.max() - truth.min(), 1e-9)
        ax.plot(x, tmm, color=COLORS["truth"], ls=":", lw=1.6, label="truth (privileged)")
        succ = "SUCCESS" if ep["success"] else "fail"
        ax.set_title(f"ep{k} {succ} (T={len(emb_true)})", fontsize=9)
        ax.set_xlabel("rollout step k"); ax.set_ylabel("cost (norm.)")
        ax.legend(fontsize=7, loc="upper right")
    for j in range(len(eps), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    head = "  ".join(f"{m} ρ_pred̄={np.nanmean(agg[m]):+.2f}" for m in modes)
    fig.suptitle(f"Predictor fidelity on PLANNER (OOD) trajectory — phys\n"
                 f"(solid=imagined, dashed=real-encoded, dotted=truth)\n{head}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_png, dpi=120)
    return {m: float(np.nanmean(agg[m])) for m in modes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=200)
    ap.add_argument("--cost", choices=["phys", "lewm"], default="phys",
                    help="planning cost: phys (goal-free shaping) or lewm (goal-image emb-L2)")
    ap.add_argument("--n-eval", type=int, default=6)
    ap.add_argument("--nf", type=int, default=5, help="future macro horizon (×5 raw steps)")
    ap.add_argument("--N", type=int, default=300)
    ap.add_argument("--n-iter", type=int, default=5)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--var-scale", type=float, default=1.0)
    ap.add_argument("--receding", type=int, default=2, help="macros executed per plan")
    ap.add_argument("--budget", type=int, default=150, help="raw env steps per episode")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--ckpt", default=None,
                    help="explicit _object.ckpt to load (e.g. no-task-head control); "
                         "loaded directly, no task_head assert. lewm cost only.")
    args = ap.parse_args()

    if args.ckpt:
        assert args.cost == "lewm", "--ckpt control models have no task_head; use --cost lewm"
        model = torch.load(args.ckpt, map_location="cuda", weights_only=False).cuda().eval()
        model.requires_grad_(False)
    else:
        model = load_phys(args.epoch)
    mu, sd = compute_denorm()
    amu, asd = compute_action_norm()
    ctx = {"mu": mu, "sd": sd}
    env, init_states = build_env()
    drawer = find_drawer_joint(env)
    print(f"drawer joint: {drawer}", flush=True)
    h5 = DrawerH5()

    rng = np.random.default_rng(args.seed)
    init_idx = [int(rng.integers(0, len(init_states))) for _ in range(args.n_eval)]
    # lewm planning needs a drawer-closed goal image per episode (mirrors p9/eval)
    goal_eps = rng.choice(h5.n_ep, size=args.n_eval, replace=False)
    cfg = dict(cost=args.cost, nf=args.nf, N=args.N, n_iter=args.n_iter, topk=args.topk,
               var_scale=args.var_scale, receding=args.receding, budget=args.budget,
               amu=amu.cpu().numpy(), asd=asd.cpu().numpy(),
               rng=np.random.default_rng(args.seed + 1))

    vdir = None
    if args.video_dir:
        from pathlib import Path
        vdir = Path(args.video_dir); vdir.mkdir(parents=True, exist_ok=True)
        import imageio

    succ, log, episodes = 0, [], []
    for i in range(args.n_eval):
        if args.cost == "lewm":   # set this episode's goal-image emb
            ge = int(goal_eps[i])
            ctx["goal_emb"] = encode_frames(model, h5.goal_frame(ge)[None],
                                            h5.goal_eye_frame(ge)[None]).cuda()[0]
        rec = {k: [] for k in ("pixels", "eye_in_hand", "action", "proprio", "drawer_qpos")}
        frames = [] if vdir else None
        t0 = time.time()
        ok, steps = run_episode(env, model, init_states[init_idx[i]], cfg, ctx, drawer, rec, frames)
        succ += int(ok)
        rec = {k: np.stack(v) for k, v in rec.items()}
        rec["success"] = bool(ok)
        episodes.append(rec)
        line = (f"[phys-e2e] ep {i+1:2d}/{args.n_eval}  {'SUCCESS' if ok else 'fail   '}  "
                f"steps={steps:3d}  rate={succ/(i+1):.2f}  ({time.time()-t0:.0f}s)  T={len(rec['action'])}")
        print(line, flush=True); log.append(line)
        if vdir and frames:
            w = imageio.get_writer(str(vdir / f"ep{i:02d}_{'SUCCESS' if ok else 'fail'}.mp4"), fps=20)
            for f in frames:
                w.append_data(f[::-1, ::-1])
            w.close()

    summary = f"==== {args.cost}-cost macro MPC success: {succ}/{args.n_eval} = {succ/args.n_eval:.1%} ===="
    print("\n" + summary, flush=True); log.append(summary)

    # save the planner trajectories (for re-plotting / further analysis)
    traj_path = os.path.join(RESULTS_DIR, f"p9b_{args.cost}_traj.npz")
    np.savez(traj_path, episodes=np.array(episodes, dtype=object))
    print(f"saved trajectories -> {traj_path}", flush=True)

    # cost_rollout on the planner's OWN (OOD) trajectory; lewm curve needs a cross-ep goal
    cross = int(rng.choice(h5.n_ep))
    goal_emb = encode_frames(model, h5.goal_frame(cross)[None],
                             h5.goal_eye_frame(cross)[None]).cuda()[0]
    out_png = os.path.join(RESULTS_DIR, f"cost_rollout_phys_planner_{args.cost}.png")
    rho = plot_cost_rollout(model, episodes, ctx, goal_emb, out_png)
    print(f"cost_rollout (planner OOD): {rho}\nsaved -> {out_png}", flush=True)

    with open(os.path.join(RESULTS_DIR, f"p9b_{args.cost}_mpc.log"), "w") as fh:
        fh.write("\n".join(log) + f"\ncost_rollout_planner: {rho}\n")
    env.close(); h5.close()


if __name__ == "__main__":
    main()
