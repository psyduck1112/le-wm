"""Stage 2: does a pessimism penalty fix闸2 (decoded-cost CEM reward hacking)?

The feasibility test for "restrict CEM to D_phi's training distribution". We add
an OOD penalty to the decoded cost so CEM is discouraged from the off-manifold
regions where D_phi lies (Stage 1 showed maha tracks the cost under-report):

    cost_pess(e) = decoded_cost(e) + beta * relu(maha_k(e) - thr)

relu(maha-thr) = soft trust region: ~0 inside the BC manifold, linear penalty
once a candidate's imagined final embedding leaves it. Sweep beta; beta=0 is the
internal control (reproduces the decoded FAIL). Compare to p6 privileged=100%.

Same seeds/goals/inits across betas for a fair comparison. Reuses the p6_oracle
harness (real-sim dynamics) so this isolates the COST, exactly like闸2.

Usage:
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python diagnostics/p12_pessimistic.py \
      --n-eval 4 --budget 60 --betas 0 0.02 0.05 0.15
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
from state_decoder import load_decoder, decoded_cost  # noqa: E402
from p6_oracle import (build_env, find_drawer_joint, restore,  # noqa: E402
                       WARMUP)
from p11_ood_probe import load_detector, ood_scores  # noqa: E402


@torch.no_grad()
def pess_costs(model, decoder, det, frames, eye_frames, beta):
    """decoded cost + beta * relu(maha_k - thr) on imagined final embeddings."""
    emb = encode_frames(model, np.stack(frames), np.stack(eye_frames))   # (N,192) cpu
    dc = decoded_cost(emb.cuda(), decoder).cpu().numpy()
    maha, _ = ood_scores(emb.numpy().astype(np.float32), det)
    pen = beta * np.maximum(0.0, maha - det["thr_maha"])
    return dc + pen, dc, pen


def cem_pess(env, model, decoder, det, s_t, cfg, beta):
    """CEM with real-sim dynamics, pessimistic decoded cost. Returns mean (H,7)."""
    H, N, topk = cfg["H"], cfg["N"], cfg["topk"]
    mean = np.zeros((H, 7), dtype=np.float32)
    var = np.full((H, 7), cfg["var_scale"], dtype=np.float32)
    rng = cfg["rng"]
    for _ in range(cfg["n_iter"]):
        cand = mean[None] + np.sqrt(var)[None] * rng.standard_normal((N, H, 7)).astype(np.float32)
        cand = np.clip(cand, -1.0, 1.0)
        finals, finals_eye = [], []
        for n in range(N):
            restore(env, s_t)
            obs = None
            for h in range(H):
                obs, _, _, _ = env.step(cand[n, h])
            finals.append(obs["agentview_image"])
            finals_eye.append(obs["robot0_eye_in_hand_image"])
        costs, _, _ = pess_costs(model, decoder, det, finals, finals_eye, beta)
        elite = np.argsort(costs)[:topk]
        mean = cand[elite].mean(0)
        var = cand[elite].var(0) + 1e-6
    return mean


def run_episode(env, model, decoder, det, init_state, cfg, beta):
    env.reset(); env.set_init_state(init_state)
    for _ in range(WARMUP):
        env.step(np.zeros(7))
    steps = 0
    while steps < cfg["budget"]:
        s_t = env.sim.get_state().flatten()
        mean = cem_pess(env, model, decoder, det, s_t, cfg, beta)
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
    ap.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.15])
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
    det = load_detector()
    env, init_states = build_env()
    drawer = find_drawer_joint(env)
    print(f"drawer joint: {drawer}  thr_maha={det['thr_maha']:.1f}  "
          f"betas={args.betas}", flush=True)
    h5 = DrawerH5()

    rng = np.random.default_rng(args.seed)
    # fixed inits shared across betas (goals unused: decoded cost is goal-free)
    init_idx = [int(rng.integers(0, len(init_states))) for _ in range(args.n_eval)]

    results = {}
    log = []
    for beta in args.betas:
        cfg = dict(H=args.H, N=args.N, n_iter=args.n_iter, topk=args.topk,
                   var_scale=args.var_scale, receding=args.receding, budget=args.budget,
                   rng=np.random.default_rng(args.seed + 1))
        succ = 0
        for i in range(args.n_eval):
            t0 = time.time()
            ok, steps = run_episode(env, model, decoder, det,
                                    init_states[init_idx[i]], cfg, beta)
            succ += int(ok)
            line = (f"[beta={beta:5.3f}] ep {i+1}/{args.n_eval}  "
                    f"{'SUCCESS' if ok else 'fail'}  steps={steps:3d}  "
                    f"rate={succ/(i+1):.2f}  ({time.time()-t0:.0f}s)")
            print(line, flush=True); log.append(line)
        results[beta] = succ / args.n_eval
        s = f"==== beta={beta:5.3f}  success {succ}/{args.n_eval} = {results[beta]:.0%} ===="
        print(s, flush=True); log.append(s)

    print("\n===== Stage 2 summary (p6: decoded=FAIL, privileged=100%) =====")
    for beta in args.betas:
        print(f"  beta={beta:5.3f}  decoded+pessimism : {results[beta]:.0%}")
    with open(os.path.join(RESULTS_DIR, "p12_pessimistic.log"), "w") as f:
        f.write("\n".join(log) + "\n")
    np.savez(os.path.join(RESULTS_DIR, "p12_pessimistic.npz"),
             betas=np.array(args.betas),
             rates=np.array([results[b] for b in args.betas]))
    env.close(); h5.close()


if __name__ == "__main__":
    main()
