"""Stage 1 = Phase-0 OOD probe: does the OOD score (p10) track D_phi's error?

The feasibility question for the trust-region fix: when CEM walks OFF the BC
manifold, can a cheap OOD score *know* D_phi is now lying? If yes, penalising
high-OOD candidates (Stage 2) is grounded; if no, the whole approach is dead.

Method: run the FAILING decoded-cost CEM (p6_oracle harness) so the arm drifts
into OOD. At every executed step record, for the REAL visited state:
  true  (eef, q)            <- MuJoCo
  D_phi (eef_hat, q_hat)    <- encode real frame -> decoder
  OOD   (maha_k, resid)     <- value/ood_detector.npz

Then correlate OOD score against D_phi error. Strong positive corr =>
the detector flags exactly where D_phi breaks => Stage 2 is worth building.

Usage:
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python diagnostics/p11_ood_probe.py \
      --n-eval 6 --budget 60
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
from p6_oracle import (build_env, find_drawer_joint, restore, oracle_cem,  # noqa: E402
                       EEF_SITE, DRAWER_BODY, WARMUP)

DETECTOR = os.path.join(ROOT, "value", "ood_detector.npz")


def load_detector(path=DETECTOR):
    d = np.load(path)
    return dict(mu=d["mu"], V=d["V"], evals=d["evals"], k=int(d["k"]),
                thr_maha=float(d["thr_maha"]), thr_resid=float(d["thr_resid"]))


def ood_scores(emb, det):
    """emb: (N,192) np. Returns (maha_k, resid)."""
    Xc = emb - det["mu"]
    k = det["k"]
    proj = Xc @ det["V"][:, :k]
    maha = ((proj ** 2) / det["evals"][:k]).sum(1)
    recon = proj @ det["V"][:, :k].T
    resid = np.linalg.norm(Xc - recon, axis=1)
    return maha, resid


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=200)
    ap.add_argument("--decoder", default=os.path.join(ROOT, "value", "state_decoder.pt"))
    ap.add_argument("--n-eval", type=int, default=6)
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
    print(f"drawer joint: {drawer}   detector k={det['k']}  "
          f"thr_maha={det['thr_maha']:.1f}", flush=True)
    h5 = DrawerH5()

    rng = np.random.default_rng(args.seed)
    goal_eps = rng.choice(h5.n_ep, size=args.n_eval, replace=False)
    goal_embs = [encode_frames(model, h5.goal_frame(e)[None],
                               h5.goal_eye_frame(e)[None])[0] for e in goal_eps]
    init_idx = [int(rng.integers(0, len(init_states))) for _ in range(args.n_eval)]

    cab = np.asarray(CABINET, dtype=np.float32)
    cfg = dict(H=args.H, N=args.N, n_iter=args.n_iter, topk=args.topk,
               var_scale=args.var_scale, receding=args.receding, budget=args.budget,
               debug=False, decoder=decoder, rng=np.random.default_rng(args.seed + 1))

    rec = {k: [] for k in ("maha", "resid", "true_reach", "dphi_reach",
                           "true_q", "dphi_q", "reach_err", "q_err",
                           "true_cost", "dphi_cost", "cost_err", "cost_signed",
                           "ep", "step")}
    embs = []   # raw 192-d emb of each visited state (for kNN geometry probe p13)

    def log_state(ep, step, obs):
        # encoder must see the SAME frames as p6: raw obs from env.step (no flip)
        eef_t = env.sim.data.get_site_xpos(EEF_SITE).copy().astype(np.float32)
        q_t = float(env.sim.data.qpos[drawer[1]])
        obs_av = obs["agentview_image"]
        obs_eye = obs["robot0_eye_in_hand_image"]
        emb = encode_frames(model, obs_av[None], obs_eye[None])[0]   # (192,) cpu
        eef_h, q_h = decoder(emb.cuda())
        eef_h = eef_h.cpu().numpy().astype(np.float32)
        q_h = float(q_h.cpu().numpy().reshape(-1)[0])
        maha, resid = ood_scores(emb.numpy().astype(np.float32)[None], det)
        # p6 uses true drawer-front body for reach; D_phi targets fixed CABINET.
        # compare both reach terms against the SAME CABINET anchor for fairness.
        tr = float(np.linalg.norm(eef_t - cab))
        dr = float(np.linalg.norm(eef_h - cab))
        tc = tr + 30.0 * abs(q_t - CLOSED)
        dc = dr + 30.0 * abs(q_h - CLOSED)
        rec["maha"].append(float(maha[0])); rec["resid"].append(float(resid[0]))
        rec["true_reach"].append(tr); rec["dphi_reach"].append(dr)
        rec["true_q"].append(q_t); rec["dphi_q"].append(q_h)
        rec["reach_err"].append(abs(dr - tr)); rec["q_err"].append(abs(q_h - q_t))
        rec["true_cost"].append(tc); rec["dphi_cost"].append(dc)
        rec["cost_err"].append(abs(dc - tc)); rec["cost_signed"].append(dc - tc)
        rec["ep"].append(ep); rec["step"].append(step)
        embs.append(emb.numpy().astype(np.float32))

    for i in range(args.n_eval):
        t0 = time.time()
        env.reset(); env.set_init_state(init_states[init_idx[i]])
        for _ in range(WARMUP):
            env.step(np.zeros(7))
        steps = 0; ok = False
        while steps < args.budget:
            s_t = env.sim.get_state().flatten()
            mean = oracle_cem(env, model, s_t, goal_embs[i], cfg, "decoded", drawer)
            restore(env, s_t)
            for h in range(args.receding):
                obs, _, _, _ = env.step(mean[h]); steps += 1
                log_state(i, steps, obs)
                if env.check_success():
                    ok = True; break
                if steps >= args.budget:
                    break
            if ok:
                break
        print(f"  ep {i+1}/{args.n_eval}  {'SUCCESS' if ok else 'fail'}  "
              f"steps={steps}  ({time.time()-t0:.0f}s)", flush=True)

    A = {k: np.asarray(v) for k, v in rec.items()}
    n = len(A["maha"])
    inlier = A["maha"] <= det["thr_maha"]
    print(f"\n===== Stage 1 OOD probe: {n} probe points "
          f"({inlier.mean():.0%} in-dist by maha thr) =====")

    print("\n[A] OOD score  vs  D_phi error  (want strong POSITIVE)")
    for tgt in ("cost_err", "reach_err", "q_err"):
        print(f"  maha  vs {tgt:9s}: pearson={np.corrcoef(A['maha'],A[tgt])[0,1]:+.3f}"
              f"  spearman={spearman(A['maha'],A[tgt]):+.3f}")
    print(f"  maha  vs cost_signed (neg=D_phi under-reports cost, the danger): "
          f"pearson={np.corrcoef(A['maha'],A['cost_signed'])[0,1]:+.3f}")

    print("\n[B] reward-hacking signature corr(true_reach, dphi_reach) "
          "(p6d full = -0.557)")
    print(f"  all points : {np.corrcoef(A['true_reach'],A['dphi_reach'])[0,1]:+.3f}")
    if inlier.sum() > 5:
        print(f"  in-dist    : "
              f"{np.corrcoef(A['true_reach'][inlier],A['dphi_reach'][inlier])[0,1]:+.3f}")
    if (~inlier).sum() > 5:
        print(f"  OOD        : "
              f"{np.corrcoef(A['true_reach'][~inlier],A['dphi_reach'][~inlier])[0,1]:+.3f}")

    print("\n[C] D_phi error binned by maha quintile (error should RISE with OOD)")
    qs = np.quantile(A["maha"], [0, .2, .4, .6, .8, 1.0])
    print("   maha-bin            n   reach_err  q_err   cost_err")
    for b in range(5):
        m = (A["maha"] >= qs[b]) & (A["maha"] <= qs[b + 1] if b == 4
                                    else A["maha"] < qs[b + 1])
        if m.sum() == 0:
            continue
        print(f"   [{qs[b]:7.0f},{qs[b+1]:7.0f})  {m.sum():4d}   "
              f"{A['reach_err'][m].mean():7.3f}   {A['q_err'][m].mean():.3f}   "
              f"{A['cost_err'][m].mean():7.3f}")

    out = os.path.join(RESULTS_DIR, "p11_ood_probe.npz")
    np.savez(out, emb=np.stack(embs), **A)
    print(f"\nsaved -> {out}")
    env.close(); h5.close()


if __name__ == "__main__":
    main()
