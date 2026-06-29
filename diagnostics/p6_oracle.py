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

from _common import (load_jepa, load_phys, compute_denorm, phys_decode,
                     encode_frames, DrawerH5, RESULTS_DIR)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "value"))
from state_decoder import load_decoder, decoded_cost, CABINET, CLOSED  # noqa: E402

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
def decoded_costs(model, decoder, frames, eye_frames):
    """Decode-then-cost: encode imagined finals -> D_φ -> p6 shaping (goal-image free)."""
    emb = encode_frames(model, np.stack(frames), np.stack(eye_frames)).cuda()  # (N,192)
    return decoded_cost(emb, decoder).cpu().numpy()


@torch.no_grad()
def phys_costs(model, mu, sd, frames, eye_frames):
    """Co-trained task_head version of decoded_costs (phys model). Goal-image free:
    decode (eef, q) from emb, de-normalize, reassemble p6's reach+30*close."""
    emb = encode_frames(model, np.stack(frames), np.stack(eye_frames)).cuda()  # (N,192)
    eef, q = phys_decode(model, emb, mu, sd)
    cab = torch.as_tensor(CABINET, device=emb.device, dtype=emb.dtype)
    return (torch.linalg.norm(eef - cab, dim=-1) + 30.0 * (q - CLOSED).abs()).cpu().numpy()


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


@torch.no_grad()
def current_plan_cost(model, obs, cfg, cost_mode, goal_emb):
    """Planner-cost of the CURRENT real state (single frame), same formula CEM uses."""
    agv = [obs["agentview_image"]]
    eye = [obs["robot0_eye_in_hand_image"]]
    if cost_mode == "lewm":
        return float(lewm_costs(model, agv, eye, goal_emb)[0])
    if cost_mode == "decoded":
        return float(decoded_costs(model, cfg["decoder"], agv, eye)[0])
    if cost_mode == "phys":
        return float(phys_costs(model, cfg["mu"], cfg["sd"], agv, eye)[0])
    return float("nan")   # privileged: plan cost == true cost


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
            if cost_mode in ("lewm", "decoded", "phys"):
                finals.append(obs["agentview_image"])
                finals_eye.append(obs["robot0_eye_in_hand_image"])   # 第二路相机 (raw, 与训练一致)
            else:  # privileged: shaped reach + close
                drawer_vals.append(privileged_cost(env, drawer))
        if cost_mode == "lewm":
            costs = lewm_costs(model, finals, finals_eye, goal_emb)
        elif cost_mode == "decoded":
            costs = decoded_costs(model, cfg["decoder"], finals, finals_eye)
        elif cost_mode == "phys":
            costs = phys_costs(model, cfg["mu"], cfg["sd"], finals, finals_eye)
        else:
            costs = np.asarray(drawer_vals)
        elite = np.argsort(costs)[:topk]
        mean = cand[elite].mean(0)
        var = cand[elite].var(0) + 1e-6
    return mean


def run_episode(env, model, init_state, goal_emb, cfg, cost_mode, drawer, frames=None,
                cost_log=None):
    env.reset()
    env.set_init_state(init_state)
    obs = None
    for _ in range(WARMUP):
        obs, _, _, _ = env.step(np.zeros(7))
    budget, receding = cfg["budget"], cfg["receding"]
    steps = 0
    while steps < budget:
        s_t = env.sim.get_state().flatten()
        mean = oracle_cem(env, model, s_t, goal_emb, cfg, cost_mode, drawer)
        # execute first `receding` actions in the REAL env
        restore(env, s_t)
        for h in range(receding):
            if frames is not None:
                frames.append(obs["agentview_image"].copy())
            obs, _, _, _ = env.step(mean[h])
            steps += 1
            if cost_log is not None:
                pc = current_plan_cost(model, obs, cfg, cost_mode, goal_emb)
                tc = privileged_cost(env, drawer) if drawer is not None else float("nan")
                cost_log.append((steps, pc, tc))
            if env.check_success():
                if frames is not None:
                    frames.append(obs["agentview_image"].copy())
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


def _plot_cost_curve(cost_log, mode, ep, ok, args, vdir):
    """Planner-cost vs true(privileged)-cost along the ACTUAL rollout (twin axis).

    cost_mono walks the EXPERT trajectory (on-tube); this walks the planner's own
    path (which wanders off-tube), so a planner-cost that drops while true-cost rises
    = the cost lying off-distribution. Spearman ρ(plan, true): ~+1 honest, <0 lying.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    arr = np.asarray(cost_log, dtype=float)            # (T,3): step, plan, true
    st, pc, tc = arr[:, 0], arr[:, 1], arr[:, 2]
    rho = spearmanr(pc, tc).correlation
    color = {"lewm": "crimson", "phys": "seagreen", "decoded": "tab:orange"}.get(mode, "tab:blue")

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(st, tc, color="black", lw=2.2, label="true cost (privileged sim)")
    ax1.set_xlabel("executed step (planner's own rollout)")
    ax1.set_ylabel("true cost  reach+30·|q−closed|", color="black")
    ax2 = ax1.twinx()
    ax2.plot(st, pc, color=color, lw=2.2, label=f"{mode} cost (planner minimizes)")
    ax2.set_ylabel(f"{mode} planner cost", color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    res = "SUCCESS" if ok else "FAIL"
    ax1.set_title(f"{mode} cost vs truth along rollout — epoch {args.epoch}, ep{ep} [{res}]\n"
                  f"Spearman ρ(plan,true)={rho:+.2f}  (+1=honest, <0=cost lying off-tube)")
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, fontsize=8, loc="upper left")
    fig.tight_layout()
    out_dir = vdir if vdir else RESULTS_DIR
    out = os.path.join(str(out_dir), f"costcurve_{args.model}_{mode}_ep{ep:02d}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"    [cost-curve] ρ(plan,true)={rho:+.2f} -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["v2", "phys"], default="v2",
                    help="v2=frozen-encoder + D_φ era; phys=co-trained task_head model")
    ap.add_argument("--epoch", type=int, default=200)
    ap.add_argument("--cost", default="all",
                    help="csv of {lewm,privileged,decoded,phys} | 'both' (lewm,privileged) "
                         "| 'all' (model-dependent default set)")
    ap.add_argument("--decoder", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "value", "state_decoder.pt"))
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
    ap.add_argument("--video-dir", default=None, help="若设, 录每条 episode 的 agentview 视频")
    ap.add_argument("--cost-curve", action="store_true",
                    help="沿 planner 实际轨迹记录 planner-cost vs 真 cost, 每条 episode 存一张图")
    args = ap.parse_args()

    # model: v2 (AutoCostModel, D_φ era) vs phys (direct object load, co-trained task_head)
    if args.model == "phys":
        model = load_phys(args.epoch)
        mu, sd = compute_denorm()
        default_all = ["lewm", "phys", "privileged"]
    else:
        model = load_jepa(args.epoch)
        mu, sd = None, None
        default_all = ["lewm", "decoded", "privileged"]
    print(f"model={args.model} epoch={args.epoch}")

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

    if args.cost == "all":
        modes = default_all
    elif args.cost == "both":
        modes = ["lewm", "privileged"]
    else:
        modes = args.cost.split(",")
    if "phys" in modes and args.model != "phys":
        raise SystemExit("phys cost needs --model phys")
    if "decoded" in modes and args.model != "v2":
        raise SystemExit("decoded cost (D_φ) needs --model v2")

    decoder = None
    if "decoded" in modes:
        decoder = load_decoder(args.decoder)
        print(f"loaded D_φ decoder <- {args.decoder}")

    vdir = None
    if args.video_dir:
        from pathlib import Path
        import imageio
        vdir = Path(args.video_dir); vdir.mkdir(parents=True, exist_ok=True)

    cfg = dict(H=args.H, N=args.N, n_iter=args.n_iter, topk=args.topk,
               var_scale=args.var_scale, receding=args.receding, budget=args.budget,
               debug=args.debug, decoder=decoder, mu=mu, sd=sd,
               rng=np.random.default_rng(args.seed + 1))
    if "privileged" in modes and drawer is None:
        print("WARNING: drawer joint not found, skipping privileged mode")
        modes = [m for m in modes if m != "privileged"]

    results = {}
    log_lines = []
    for mode in modes:
        succ = 0
        for i in range(args.n_eval):
            t0 = time.time()
            frames = [] if vdir else None
            cost_log = [] if args.cost_curve else None
            ok, steps = run_episode(env, model, init_states[init_idx[i]],
                                    goal_embs[i], cfg, mode, drawer, frames, cost_log)
            succ += int(ok)
            line = (f"[{mode:10s}] ep {i+1:2d}/{args.n_eval}  "
                    f"{'SUCCESS' if ok else 'fail   '}  steps={steps:3d}  "
                    f"rate={succ/(i+1):.2f}  ({time.time()-t0:.0f}s)")
            print(line, flush=True)
            log_lines.append(line)
            if vdir and frames:
                tag = "SUCCESS" if ok else "fail"
                w = imageio.get_writer(str(vdir / f"{args.model}_{mode}_ep{i:02d}_{tag}.mp4"), fps=20)
                for fr in frames:
                    w.append_data(fr[::-1, ::-1])   # OpenGL -> display
                w.close()
            if cost_log:
                _plot_cost_curve(cost_log, mode, i, ok, args, vdir)
        results[mode] = succ / args.n_eval
        summary = f"==== {mode} oracle-dynamics success: {succ}/{args.n_eval} = {results[mode]:.1%} ===="
        print(summary); log_lines.append(summary)

    print(f"\n===== oracle-MPC summary (model={args.model}, baseline le-wm MPC = 10%) =====")
    for mode in modes:
        print(f"  oracle-dynamics + {mode:10s} cost : {results[mode]:.1%}")
    tag = f"_{args.model}" if args.model != "v2" else ""
    with open(os.path.join(RESULTS_DIR, f"p6_oracle{tag}.log"), "w") as f:
        f.write("\n".join(log_lines) + "\n")
    np.savez(os.path.join(RESULTS_DIR, f"p6_oracle{tag}.npz"),
             **{f"{m}_rate": results[m] for m in modes})
    env.close(); h5.close()


if __name__ == "__main__":
    main()
