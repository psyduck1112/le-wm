"""Probe 7 (subgoal upper-bound): does segmenting the goal into demo-frame
waypoints rescue the broken goal-image emb-MSE cost?

Reuses probe 6's ORACLE-dynamics MPC harness (perfect physics, so the learned
predictor is already exonerated). The ONLY change vs probe 6: instead of a single
far末帧 goal, the cost is guided by K intermediate frames (waypoints) taken from
the goal demo. The arm is steered to waypoint wp; once reached (emb-cost drops
below reach_frac * segment-start cost) or the segment times out, advance to wp+1.

Success is STILL judged by env.check_success() (real drawer closed) — only the
COST guidance is segmented, so the comparison to probe 6 (0/8) is apples-to-apples.

  waypoints lift success >0  => short-range cost is monotonic enough; the subgoal
                                route works -> value/FM投入有据 (阶段2/FM).
  waypoints still ~0         => even privileged true-frame waypoints + perfect
                                physics can't make the per-segment emb distance
                                work -> pure subgoal won't save it; segments need
                                learned value shaping. Don't build FM subgoals.

Usage:
  python diagnostics/p7_subgoal.py --n-eval 8 --k 4
"""
import argparse
import os
import time
import numpy as np
import torch

from _common import load_jepa, encode_frames, DrawerH5, RESULTS_DIR
from p6_oracle import (
    WARMUP, clear_done, restore, build_env, find_drawer_joint,
    oracle_cem, EEF_SITE, DRAWER_BODY,
)


@torch.no_grad()
def emb_cost(model, obs, goal_emb):
    """emb-MSE from current (dual-cam) obs to a waypoint embedding."""
    emb = encode_frames(model,
                        obs["agentview_image"][None],
                        obs["robot0_eye_in_hand_image"][None])[0]
    return float(((emb - goal_emb) ** 2).sum())


def waypoint_embs(model, h5, ep, k):
    """K embeddings evenly spaced along the goal demo (last frame = 末帧 goal)."""
    pix = h5.pixels(ep)
    eye = h5.eye_in_hand(ep)
    idx = np.linspace(0, len(pix) - 1, k + 1)[1:].round().astype(int)  # skip step 0
    return [encode_frames(model, pix[i][None], eye[i][None])[0] for i in idx], idx


def run_episode(env, model, init_state, wp_embs, cfg, drawer):
    env.reset()
    env.set_init_state(init_state)
    obs = None
    for _ in range(WARMUP):
        obs, _, _, _ = env.step(np.zeros(7))
    budget, receding = cfg["budget"], cfg["receding"]
    seg_max, reach_frac = cfg["seg_max"], cfg["reach_frac"]
    K = len(wp_embs)
    steps, wp, seg_steps = 0, 0, 0
    seg_cost0 = emb_cost(model, obs, wp_embs[wp])
    while steps < budget and wp < K:
        s_t = env.sim.get_state().flatten()
        mean = oracle_cem(env, model, s_t, wp_embs[wp], cfg, "lewm", drawer)
        restore(env, s_t)
        for h in range(receding):
            obs, _, _, _ = env.step(mean[h])
            steps += 1
            seg_steps += 1
            if env.check_success():
                return True, steps, wp
            if steps >= budget:
                break
        cur = emb_cost(model, obs, wp_embs[wp])
        reached = cur < reach_frac * seg_cost0
        if reached or seg_steps >= seg_max:
            wp += 1
            seg_steps = 0
            if wp < K:
                seg_cost0 = emb_cost(model, obs, wp_embs[wp])
            if cfg.get("debug"):
                q = float(env.sim.data.qpos[drawer[1]]) if drawer else float("nan")
                print(f"    [dbg] step={steps:3d} -> wp {wp}/{K} "
                      f"({'reached' if reached else 'timeout'}) drawer_q={q:+.3f}",
                      flush=True)
    return env.check_success(), steps, wp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=122)   # v2 双相机 ckpt
    ap.add_argument("--n-eval", type=int, default=8)
    ap.add_argument("--k", type=int, default=4, help="num waypoints per demo")
    ap.add_argument("--reach-frac", type=float, default=0.4,
                    help="advance when cost < frac * segment-start cost")
    ap.add_argument("--H", type=int, default=5)
    ap.add_argument("--N", type=int, default=150)
    ap.add_argument("--n-iter", type=int, default=3)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--var-scale", type=float, default=0.5)
    ap.add_argument("--receding", type=int, default=5)
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    model = load_jepa(args.epoch)
    env, init_states = build_env()
    drawer = find_drawer_joint(env)
    print(f"drawer joint: {drawer}")
    h5 = DrawerH5()

    # SAME goals + init indices as probe 6 (same seed) for a fair comparison
    rng = np.random.default_rng(args.seed)
    goal_eps = rng.choice(h5.n_ep, size=args.n_eval, replace=False)
    init_idx = [int(rng.integers(0, len(init_states))) for _ in range(args.n_eval)]

    cfg = dict(H=args.H, N=args.N, n_iter=args.n_iter, topk=args.topk,
               var_scale=args.var_scale, receding=args.receding, budget=args.budget,
               seg_max=max(args.receding, args.budget // args.k), reach_frac=args.reach_frac,
               debug=args.debug, rng=np.random.default_rng(args.seed + 1))

    succ, log_lines = 0, []
    for i in range(args.n_eval):
        t0 = time.time()
        wp_embs, wp_idx = waypoint_embs(model, h5, int(goal_eps[i]), args.k)
        ok, steps, reached_wp = run_episode(env, model, init_states[init_idx[i]],
                                            wp_embs, cfg, drawer)
        succ += int(ok)
        line = (f"[subgoal k={args.k}] ep {i+1:2d}/{args.n_eval}  "
                f"{'SUCCESS' if ok else 'fail   '}  steps={steps:3d}  "
                f"reached_wp={reached_wp}/{args.k}  rate={succ/(i+1):.2f}  "
                f"({time.time()-t0:.0f}s)")
        print(line, flush=True)
        log_lines.append(line)

    rate = succ / args.n_eval
    summary = (f"==== subgoal (k={args.k}) oracle-dynamics + lewm-cost success: "
               f"{succ}/{args.n_eval} = {rate:.1%}  (probe6 single-goal baseline = 0/8) ====")
    print(summary)
    log_lines.append(summary)
    with open(os.path.join(RESULTS_DIR, "p7_subgoal.log"), "w") as f:
        f.write("\n".join(log_lines) + "\n")
    np.savez(os.path.join(RESULTS_DIR, "p7_subgoal.npz"), rate=rate, k=args.k)
    env.close()
    h5.close()


if __name__ == "__main__":
    main()
