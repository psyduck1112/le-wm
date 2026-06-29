"""Probe 9 (real end-to-end MPC): the GENUINE planning success rate of the
improved DUAL-CAMERA model (lewm_libero_bc_drawer_v2_epoch_200).

Unlike probe 6 (which swaps in oracle MuJoCo dynamics), here EVERYTHING is le-wm:
the learned encoder, the learned predictor (jepa.rollout, autoregressive), the
learned dual-camera goal-image cost, and CEM. This is what the model actually does
when deployed — the number that's comparable to the old single-cam 10% baseline
(baseline_mpc.log), which never had a dual-cam counterpart.

Why a fresh script instead of bc_drawer_eval.py: that path is swm.World +
WorldModelPolicy, hard-wired to a SINGLE agentview camera. Adding the wrist camera
means patching stable_worldmodel internals (frame buffer, transform, goal_eye plumb).
Here we drive the real LIBERO env directly (reuse probe 6's harness) and call
model.get_cost(info, cand) ourselves with both cameras populated.

Upper bound: probe 6 oracle-dynamics + dual-cam cost = 0/8 (perfect predictor).
Real dynamics <= perfect, so this is expected ~0%; we run it to MEASURE and to
RECORD VIDEO of how the dual-cam model actually moves the arm.

Usage:
  MUJOCO_GL=egl EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
    python diagnostics/p9_real_mpc.py --n-eval 6 --budget 150 \
      --video-dir diagnostics/results/p9_videos
"""
import argparse
import os
import time
from collections import deque

import numpy as np
import torch

from _common import (load_jepa, transform_pixels, DrawerH5, RESULTS_DIR, HISTORY)
from p6_oracle import (WARMUP, build_env, find_drawer_joint, EEF_SITE, DRAWER_BODY)

AGENT = "agentview_image"
EYE = "robot0_eye_in_hand_image"


def stack_hist(buf, n):
    """deque of (H,W,3) uint8 -> torch (1, n, 3, 224, 224) on cuda (training-norm)."""
    frames = np.stack(list(buf))                     # (n,H,W,3)
    return transform_pixels(frames).unsqueeze(0).cuda()  # (1,n,3,224,224)


@torch.no_grad()
def cem_plan(model, pix_buf, eye_buf, act_buf, goal_pix, goal_eye, cfg):
    """One CEM plan. Returns mean future actions (n_future, 7)."""
    nf, N, topk, n_iter = cfg["nf"], cfg["N"], cfg["topk"], cfg["n_iter"]
    rng = cfg["rng"]

    # history context tensors (shared across all N candidates)
    pix_h = stack_hist(pix_buf, HISTORY)             # (1,H,3,224,224)
    eye_h = stack_hist(eye_buf, HISTORY)
    g_pix = transform_pixels(goal_pix[None]).cuda()  # (1,3,224,224)
    g_eye = transform_pixels(goal_eye[None]).cuda()
    act_ctx = np.stack(list(act_buf))                # (H,7) past actions

    mean = np.zeros((nf, 7), dtype=np.float32)
    var = np.full((nf, 7), cfg["var_scale"], dtype=np.float32)

    for _ in range(n_iter):
        fut = mean[None] + np.sqrt(var)[None] * rng.standard_normal((N, nf, 7)).astype(np.float32)
        fut = np.clip(fut, -1.0, 1.0)               # (N, nf, 7)
        # full action sequence per candidate: [past HISTORY ctx | nf future]
        ctx = np.broadcast_to(act_ctx[None], (N, HISTORY, 7))
        seq = np.concatenate([ctx, fut], axis=1)     # (N, H+nf, 7)
        cand = torch.from_numpy(seq).float().unsqueeze(0).cuda()  # (1,N,H+nf,7)

        S = N
        pix = pix_h.unsqueeze(1).expand(1, S, HISTORY, 3, 224, 224)
        eye = eye_h.unsqueeze(1).expand(1, S, HISTORY, 3, 224, 224)
        gp = g_pix.view(1, 1, 1, 3, 224, 224).expand(1, S, HISTORY, 3, 224, 224)
        ge = g_eye.view(1, 1, 1, 3, 224, 224).expand(1, S, HISTORY, 3, 224, 224)
        info = {"pixels": pix, "eye_in_hand": eye,
                "goal": gp, "goal_eye_in_hand": ge, "action": cand}

        cost = model.get_cost(info, cand).view(-1).cpu().numpy()  # (N,)
        elite = np.argsort(cost)[:topk]
        mean = seq[elite, HISTORY:].mean(0)          # future part only
        var = seq[elite, HISTORY:].var(0) + 1e-6
    return mean


def run_episode(env, model, init_state, goal_pix, goal_eye, cfg, drawer, ep_frames):
    env.reset()
    env.set_init_state(init_state)
    obs = None
    for _ in range(WARMUP):
        obs, _, _, _ = env.step(np.zeros(7))

    pix_buf = deque([obs[AGENT].copy()] * HISTORY, maxlen=HISTORY)
    eye_buf = deque([obs[EYE].copy()] * HISTORY, maxlen=HISTORY)
    act_buf = deque([np.zeros(7, np.float32)] * HISTORY, maxlen=HISTORY)

    budget, receding = cfg["budget"], cfg["receding"]
    steps = 0
    while steps < budget:
        mean = cem_plan(model, pix_buf, eye_buf, act_buf, goal_pix, goal_eye, cfg)
        for h in range(receding):
            if ep_frames is not None:
                ep_frames.append(obs[AGENT].copy())
            a = mean[h]
            obs, _, _, _ = env.step(a)
            steps += 1
            pix_buf.append(obs[AGENT].copy())
            eye_buf.append(obs[EYE].copy())
            act_buf.append(a.astype(np.float32))
            if env.check_success():
                return True, steps
            if steps >= budget:
                break
        if cfg.get("debug") and drawer is not None:
            q = float(env.sim.data.qpos[drawer[1]])
            eef = env.sim.data.get_site_xpos(EEF_SITE)
            front = env.sim.data.get_body_xpos(DRAWER_BODY)
            d = float(np.linalg.norm(eef - front))
            print(f"    [dbg] step={steps:3d} drawer_q={q:+.3f} "
                  f"(open=-0.16 closed=+0.01) eef_dist={d:.3f}", flush=True)
    return env.check_success(), steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=6)
    ap.add_argument("--nf", type=int, default=7, help="future planning horizon")
    ap.add_argument("--N", type=int, default=300, help="CEM candidates")
    ap.add_argument("--n-iter", type=int, default=5)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--var-scale", type=float, default=0.5)
    ap.add_argument("--receding", type=int, default=2)
    ap.add_argument("--budget", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    model = load_jepa(args.epoch)
    env, init_states = build_env()
    drawer = find_drawer_joint(env)
    print(f"drawer joint: {drawer}", flush=True)
    h5 = DrawerH5()

    rng = np.random.default_rng(args.seed)
    goal_eps = rng.choice(h5.n_ep, size=args.n_eval, replace=False)
    init_idx = [int(rng.integers(0, len(init_states))) for _ in range(args.n_eval)]

    cfg = dict(nf=args.nf, N=args.N, n_iter=args.n_iter, topk=args.topk,
               var_scale=args.var_scale, receding=args.receding, budget=args.budget,
               debug=args.debug, rng=np.random.default_rng(args.seed + 1))

    vdir = None
    if args.video_dir:
        from pathlib import Path
        vdir = Path(args.video_dir); vdir.mkdir(parents=True, exist_ok=True)
        import imageio

    succ = 0
    log = []
    for i in range(args.n_eval):
        ep = int(goal_eps[i])
        goal_pix, goal_eye = h5.goal_frame(ep), h5.goal_eye_frame(ep)
        ep_frames = [] if vdir else None
        t0 = time.time()
        ok, steps = run_episode(env, model, init_states[init_idx[i]],
                                goal_pix, goal_eye, cfg, drawer, ep_frames)
        succ += int(ok)
        line = (f"[real-e2e] ep {i+1:2d}/{args.n_eval}  "
                f"{'SUCCESS' if ok else 'fail   '}  steps={steps:3d}  "
                f"rate={succ/(i+1):.2f}  ({time.time()-t0:.0f}s)")
        print(line, flush=True); log.append(line)
        if vdir and ep_frames:
            tag = "SUCCESS" if ok else "fail"
            w = imageio.get_writer(str(vdir / f"ep{i:02d}_{tag}.mp4"), fps=20)
            for f in ep_frames:
                w.append_data(f[::-1, ::-1])   # OpenGL -> display
            w.close()

    summary = f"==== real end-to-end (dual-cam, learned predictor) success: {succ}/{args.n_eval} = {succ/args.n_eval:.1%} ===="
    print("\n" + summary)
    print("(reference: old single-cam baseline_mpc.log = 10%; oracle-dynamics upper bound = 0/8)")
    log.append(summary)
    with open(os.path.join(RESULTS_DIR, "p9_real_mpc.log"), "w") as fh:
        fh.write("\n".join(log) + "\n")
    env.close(); h5.close()


if __name__ == "__main__":
    main()
