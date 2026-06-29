"""Decisive 2x2: attribute the decoded-cost failure to eef-source vs q-source.

Tests whether replacing the DECODED eef with the TRUE (proprioceptive) eef
recovers success — i.e. whether feeding proprio would fix the drawer task.

Cost (oracle dynamics, reach target = fixed CABINET for all cells):
    cost = ||eef_src - CABINET|| + 30*|q_src - CLOSED|
with eef_src, q_src each in {true (sim), dphi (decoded from emb)}.

  (true, true)  -> expect ~100%  : harness + const-cabinet reach sanity
  (true, dphi)  -> THE test      : high => proprio-eef is the fix
  (dphi, true)  -> expect low     : confirms eef is the culprit, not q
  (dphi, dphi)  -> expect 0%      : decoded baseline (= p12 beta=0)

Usage:
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python diagnostics/p14_source_ablation.py \
      --n-eval 4 --budget 60
"""
import argparse
import os
import sys
import time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "value"))

from _common import load_jepa, encode_frames, DrawerH5, RESULTS_DIR  # noqa: E402
from state_decoder import load_decoder, CABINET, CLOSED  # noqa: E402
from p6_oracle import (build_env, find_drawer_joint, restore,  # noqa: E402
                       EEF_SITE, WARMUP)

MODES = [("true", "true"), ("true", "dphi"), ("dphi", "true"), ("dphi", "dphi")]


@torch.no_grad()
def cem(env, model, decoder, drawer, s_t, cfg, eef_src, q_src):
    """CEM, real-sim dynamics, cost from chosen eef/q sources. Returns mean (H,7)."""
    H, N, topk = cfg["H"], cfg["N"], cfg["topk"]
    cab = np.asarray(CABINET, dtype=np.float32)
    mean = np.zeros((H, 7), dtype=np.float32)
    var = np.full((H, 7), cfg["var_scale"], dtype=np.float32)
    rng = cfg["rng"]
    need_dphi = (eef_src == "dphi") or (q_src == "dphi")
    for _ in range(cfg["n_iter"]):
        cand = mean[None] + np.sqrt(var)[None] * rng.standard_normal((N, H, 7)).astype(np.float32)
        cand = np.clip(cand, -1.0, 1.0)
        t_eef, t_q, frames, eyes = [], [], [], []
        for n in range(N):
            restore(env, s_t)
            obs = None
            for h in range(H):
                obs, _, _, _ = env.step(cand[n, h])
            t_eef.append(env.sim.data.get_site_xpos(EEF_SITE).copy())
            t_q.append(float(env.sim.data.qpos[drawer[1]]))
            if need_dphi:
                frames.append(obs["agentview_image"]); eyes.append(obs["robot0_eye_in_hand_image"])
        t_eef = np.asarray(t_eef, dtype=np.float32)        # (N,3)
        t_q = np.asarray(t_q, dtype=np.float32)            # (N,)
        if need_dphi:
            emb = encode_frames(model, np.stack(frames), np.stack(eyes)).cuda()
            d_eef, d_q = decoder(emb)
            d_eef = d_eef.cpu().numpy().astype(np.float32)  # (N,3)
            d_q = d_q.cpu().numpy().reshape(-1).astype(np.float32)  # (N,)
        eef = t_eef if eef_src == "true" else d_eef
        q = t_q if q_src == "true" else d_q
        reach = np.linalg.norm(eef - cab, axis=1)          # (N,)
        close = np.abs(q - CLOSED)                          # (N,)
        costs = reach + 30.0 * close
        elite = np.argsort(costs)[:topk]
        mean = cand[elite].mean(0); var = cand[elite].var(0) + 1e-6
    return mean


def run_episode(env, model, decoder, drawer, init_state, cfg, eef_src, q_src):
    env.reset(); env.set_init_state(init_state)
    for _ in range(WARMUP):
        env.step(np.zeros(7))
    steps = 0
    while steps < cfg["budget"]:
        s_t = env.sim.get_state().flatten()
        mean = cem(env, model, decoder, drawer, s_t, cfg, eef_src, q_src)
        restore(env, s_t)
        for h in range(cfg["receding"]):
            env.step(mean[h]); steps += 1
            if env.check_success():
                return True, steps
            if steps >= cfg["budget"]:
                break
    return env.check_success(), steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=200)
    ap.add_argument("--decoder", default=os.path.join(ROOT, "value", "state_decoder.pt"))
    ap.add_argument("--n-eval", type=int, default=4)
    ap.add_argument("--H", type=int, default=5)
    ap.add_argument("--N", type=int, default=150)
    ap.add_argument("--n-iter", type=int, default=3)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--var-scale", type=float, default=0.5)
    ap.add_argument("--receding", type=int, default=5)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model = load_jepa(args.epoch)
    decoder = load_decoder(args.decoder)
    env, init_states = build_env()
    drawer = find_drawer_joint(env)
    print(f"drawer joint: {drawer}", flush=True)
    h5 = DrawerH5()
    rng = np.random.default_rng(args.seed)
    init_idx = [int(rng.integers(0, len(init_states))) for _ in range(args.n_eval)]

    results, log = {}, []
    for eef_src, q_src in MODES:
        cfg = dict(H=args.H, N=args.N, n_iter=args.n_iter, topk=args.topk,
                   var_scale=args.var_scale, receding=args.receding, budget=args.budget,
                   rng=np.random.default_rng(args.seed + 1))
        succ = 0
        for i in range(args.n_eval):
            t0 = time.time()
            ok, steps = run_episode(env, model, decoder, drawer,
                                    init_states[init_idx[i]], cfg, eef_src, q_src)
            succ += int(ok)
            line = (f"[eef={eef_src:4s} q={q_src:4s}] ep {i+1}/{args.n_eval}  "
                    f"{'SUCCESS' if ok else 'fail'}  steps={steps:3d}  "
                    f"rate={succ/(i+1):.2f}  ({time.time()-t0:.0f}s)")
            print(line, flush=True); log.append(line)
        results[(eef_src, q_src)] = succ / args.n_eval
        s = f"==== eef={eef_src} q={q_src}  success {succ}/{args.n_eval} = {results[(eef_src,q_src)]:.0%} ===="
        print(s, flush=True); log.append(s)

    print("\n===== Stage: eef/q source 2x2 (p6 privileged=100%) =====")
    print(f"{'':12s} q=true     q=dphi")
    for eef_src in ("true", "dphi"):
        row = "  ".join(f"{results[(eef_src,q)]:5.0%}" for q in ("true", "dphi"))
        print(f"  eef={eef_src:4s}    {row}")
    with open(os.path.join(RESULTS_DIR, "p14_source_ablation.log"), "w") as f:
        f.write("\n".join(log) + "\n")
    np.savez(os.path.join(RESULTS_DIR, "p14_source_ablation.npz"),
             modes=np.array([f"{a}_{b}" for a, b in MODES]),
             rates=np.array([results[m] for m in MODES]))
    env.close(); h5.close()


if __name__ == "__main__":
    main()
