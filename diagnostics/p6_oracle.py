"""Probe 6 (LIBERO_DIAGNOSTICS.md 探测6): oracle-simulator planning (upper bound).

Replace ONLY the learned dynamics in the MPC loop with the REAL MuJoCo simulator
(save/restore + actually step candidate actions + render the true final frame +
encode with le-wm's real encoder). Encoder + cost + CEM stay le-wm's. This isolates
"is the learned predictor the bottleneck?" from "is it the encoder/cost?".

  perfect physics + le-wm cost succeeds  => bottleneck = learned predictor (TD-MPC can help)
  perfect physics + le-wm cost still fails => bottleneck = encoder or cost (see probe 2/3)

Control: --cost privileged uses the drawer joint qpos as cost (perfect physics +
perfect cost) to validate the MPC harness itself can solve the task. If lewm-cost
fails but privileged-cost succeeds, that cleanly pins the blame on cost.

Frame convention: training/eval both feed RAW OffScreenRenderEnv frames (no vertical
flip), verified against the data converter. So we encode raw obs["agentview_image"].

Usage:
  python diagnostics/p6_oracle.py --cost both --n-eval 8
"""
import argparse
import os
import sys
import time
import numpy as np
import torch

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv

from _common import load_jepa, encode_frames, DrawerH5, RESULTS_DIR

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "value"))
from value_head import GoalCost  # noqa: E402

WARMUP = 5


def clear_done(env):
    """Stop robosuite from auto-terminating: we step thousands of times across CEM
    branches, so the horizon-timeout (timestep>=horizon) would set done=True and
    base.step would then raise 'terminated episode'. Disable horizon-done and reset
    the step counter; task success is read separately via env.check_success()."""
    base = env.env if hasattr(env, "env") and hasattr(env.env, "done") else env
    base.done = False
    base.ignore_done = True
    base.timestep = 0


def restore(env, s_t):
    env.sim.set_state_from_flattened(s_t)
    env.sim.forward()
    clear_done(env)


def build_env(img_size=224):
    bench = get_benchmark("LIBERO_90")(0)
    task = bench.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    init_states = torch.load(
        os.path.join(get_libero_path("init_states"), task.problem_folder,
                     task.init_states_file), weights_only=False)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=img_size,
                             camera_widths=img_size)
    return env, init_states


def find_drawer_joint(env):
    """Return (qpos_addr, closed_val) for the cabinet drawer joint, or None."""
    model = env.sim.model
    for jn in model.joint_names:
        if "drawer" in jn.lower() or "cabinet" in jn.lower():
            jid = model.joint_name2id(jn)
            addr = model.jnt_qposadr[jid]
            lo, hi = model.jnt_range[jid]
            # closed drawer = retracted = endpoint with smaller |value| (near 0)
            closed = lo if abs(lo) < abs(hi) else hi
            return jn, int(addr), float(closed)
    return None


EEF_SITE = "gripper0_grip_site"
DRAWER_BODY = "wooden_cabinet_1_cabinet_top"


@torch.no_grad()
def lewm_costs(model, frames, eye_frames, goal_emb):
    """frames/eye_frames: list of (H,W,3) 双相机. Returns (N,) cost = ||enc-goal_emb||^2."""
    emb = encode_frames(model, np.stack(frames), np.stack(eye_frames))   # (N,192) cpu
    return ((emb - goal_emb) ** 2).sum(-1).numpy()


@torch.no_grad()
def value_costs(model, vhead, frames, eye_frames, goal_emb):
    """Stage-2 cost: encode imagined finals -> C(emb, goal_emb) (learned cost-to-go)."""
    emb = encode_frames(model, np.stack(frames), np.stack(eye_frames)).cuda()  # (N,192)
    g = goal_emb.cuda().unsqueeze(0).expand(emb.size(0), -1)
    return vhead(emb, g).cpu().numpy()


def privileged_cost(env, drawer):
    """Shaped oracle cost: reach the drawer (eef->drawer front) + close it
    (drawer qpos -> closed). Reaching term gives a signal even before contact,
    which a drawer-only cost lacks (arm would never learn to reach)."""
    eef = env.sim.data.get_site_xpos(EEF_SITE)
    front = env.sim.data.get_body_xpos(DRAWER_BODY)
    reach = float(np.linalg.norm(eef - front))
    q = float(env.sim.data.qpos[drawer[1]])
    close = abs(q - drawer[2])   # L1 so it keeps pulling near the goal
    return reach + 30.0 * close


def oracle_cem(env, model, s_t, goal_emb, cfg, cost_mode, drawer):
    """One CEM plan from saved state s_t using real-sim dynamics. Returns mean (H,7)."""
    H, N, topk = cfg["H"], cfg["N"], cfg["topk"]
    mean = np.zeros((H, 7), dtype=np.float32)
    var = np.full((H, 7), cfg["var_scale"], dtype=np.float32)
    rng = cfg["rng"]
    for _ in range(cfg["n_iter"]):
        cand = mean[None] + np.sqrt(var)[None] * rng.standard_normal((N, H, 7)).astype(np.float32)
        cand = np.clip(cand, -1.0, 1.0)
        finals, finals_eye, drawer_vals = [], [], []
        for n in range(N):
            restore(env, s_t)
            obs = None
            for h in range(H):
                obs, _, _, _ = env.step(cand[n, h])
            if cost_mode in ("lewm", "value"):
                finals.append(obs["agentview_image"])
                finals_eye.append(obs["robot0_eye_in_hand_image"])   # 第二路相机 (raw, 与训练一致)
            else:  # privileged: shaped reach + close
                drawer_vals.append(privileged_cost(env, drawer))
        if cost_mode == "lewm":
            costs = lewm_costs(model, finals, finals_eye, goal_emb)
        elif cost_mode == "value":
            costs = value_costs(model, cfg["vhead"], finals, finals_eye, goal_emb)
        else:
            costs = np.asarray(drawer_vals)
        elite = np.argsort(costs)[:topk]
        mean = cand[elite].mean(0)
        var = cand[elite].var(0) + 1e-6
    return mean


def run_episode(env, model, init_state, goal_emb, cfg, cost_mode, drawer):
    env.reset()
    env.set_init_state(init_state)
    for _ in range(WARMUP):
        env.step(np.zeros(7))
    budget, receding = cfg["budget"], cfg["receding"]
    steps = 0
    while steps < budget:
        s_t = env.sim.get_state().flatten()
        mean = oracle_cem(env, model, s_t, goal_emb, cfg, cost_mode, drawer)
        # execute first `receding` actions in the REAL env
        restore(env, s_t)
        for h in range(receding):
            env.step(mean[h])
            steps += 1
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
    ap.add_argument("--cost", choices=["lewm", "privileged", "value", "both"], default="both")
    ap.add_argument("--value-ckpt", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "value", "value_head.pt"))
    ap.add_argument("--n-eval", type=int, default=8)
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

    rng = np.random.default_rng(args.seed)
    # fixed goals + init indices, shared across cost modes for fair comparison
    goal_eps = rng.choice(h5.n_ep, size=args.n_eval, replace=False)
    goal_embs = [encode_frames(model, h5.goal_frame(e)[None],
                               h5.goal_eye_frame(e)[None])[0] for e in goal_eps]
    init_idx = [int(rng.integers(0, len(init_states))) for _ in range(args.n_eval)]

    modes = ["lewm", "privileged"] if args.cost == "both" else [args.cost]

    vhead = None
    if "value" in modes:
        ck = torch.load(args.value_ckpt, map_location="cuda")
        vhead = GoalCost(emb_dim=ck["emb_dim"]).cuda().eval()
        vhead.load_state_dict(ck["state_dict"])
        vhead.requires_grad_(False)
        print(f"loaded value head <- {args.value_ckpt} (gamma={ck.get('gamma')})")

    cfg = dict(H=args.H, N=args.N, n_iter=args.n_iter, topk=args.topk,
               var_scale=args.var_scale, receding=args.receding, budget=args.budget,
               debug=args.debug, vhead=vhead, rng=np.random.default_rng(args.seed + 1))
    if "privileged" in modes and drawer is None:
        print("WARNING: drawer joint not found, skipping privileged mode")
        modes = [m for m in modes if m != "privileged"]

    results = {}
    log_lines = []
    for mode in modes:
        succ = 0
        for i in range(args.n_eval):
            t0 = time.time()
            ok, steps = run_episode(env, model, init_states[init_idx[i]],
                                    goal_embs[i], cfg, mode, drawer)
            succ += int(ok)
            line = (f"[{mode:10s}] ep {i+1:2d}/{args.n_eval}  "
                    f"{'SUCCESS' if ok else 'fail   '}  steps={steps:3d}  "
                    f"rate={succ/(i+1):.2f}  ({time.time()-t0:.0f}s)")
            print(line, flush=True)
            log_lines.append(line)
        results[mode] = succ / args.n_eval
        summary = f"==== {mode} oracle-dynamics success: {succ}/{args.n_eval} = {results[mode]:.1%} ===="
        print(summary); log_lines.append(summary)

    print("\n===== Probe 6 summary (baseline le-wm MPC = 10%) =====")
    for mode in modes:
        print(f"  oracle-dynamics + {mode:10s} cost : {results[mode]:.1%}")
    with open(os.path.join(RESULTS_DIR, "p6_oracle.log"), "w") as f:
        f.write("\n".join(log_lines) + "\n")
    np.savez(os.path.join(RESULTS_DIR, "p6_oracle.npz"), **{f"{m}_rate": results[m] for m in modes})
    env.close(); h5.close()


if __name__ == "__main__":
    main()
