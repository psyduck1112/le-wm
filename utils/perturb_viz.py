"""
Small-sample diversity check for perturb_resim.

Rolls the SAME N source-demo scenes under two action policies and overlays the
end-effector top-down (x,y) paths so we can SEE whether episodes fan out to
different regions (good) or all jitter around the start (bad).

  left  panel: OLD  (hold=4,  scale=1.0, drift=0)  -> zero-mean random walk
  right panel: NEW  (hold,    scale,     drift)     -> per-episode drift bias

Each line = one episode; dot = start, star = end. We also print the net |end-start|
displacement spread, which is the number that was small (~0.27 m) in the delivered file.

Usage:
  MUJOCO_GL=egl EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
    python utils/perturb_viz.py --n 12 --hold 10 --scale 1.8 --drift 0.5
"""
import argparse
import os
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv

SRC = "/home/yikang/stable-wm/libero_bc_rollouts/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_bc.hdf5"
OUT = "/home/yikang/git/le-wm/diagnostics/results/perturb_diversity.png"


def clear_done(env):
    base = env.env if hasattr(env, "env") and hasattr(env.env, "done") else env
    base.done = False
    base.ignore_done = True
    base.timestep = 0


def get_obs(env):
    tgt = env if hasattr(env, "_get_observations") else env.env
    return tgt._get_observations()


def build_env(img_size=224):
    bench = get_benchmark("LIBERO_90")(0)
    task = bench.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    return OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=img_size, camera_widths=img_size)


def roll(env, states, T, rng, hold, scale, drift, drift_spectrum=False):
    """Free-roll T steps from a random mid-demo state; return eef xyz path (T,3).
    drift_spectrum: draw this episode's drift magnitude ~ U(0, drift) so the dataset
    spans tube (mag~0) -> extreme reach (mag~drift)."""
    start = int(rng.integers(0, T))
    env.reset(); env.sim.set_state_from_flattened(states[start]); env.sim.forward(); clear_done(env)
    obs = get_obs(env)
    mag = rng.uniform(0, drift) if drift_spectrum else drift
    drift_vec = (mag * rng.standard_normal(7)).astype(np.float32) if drift > 0 else np.zeros(7, np.float32)
    path = np.zeros((T, 3), np.float32)
    a = np.zeros(7, np.float32)
    for t in range(T):
        if t % hold == 0:
            a = np.clip(drift_vec + scale * rng.uniform(-1.0, 1.0, 7), -1.0, 1.0).astype(np.float32)
        path[t] = obs["robot0_eef_pos"]
        obs, _, _, _ = env.step(a)
    return path


def run_config(env, demos, dgrp, seed, hold, scale, drift, drift_spectrum=False):
    paths = []
    for key in demos:
        states = dgrp[key]["states"][:]
        T = int(dgrp[key].attrs["num_samples"])
        rng = np.random.default_rng(seed + 1 + int(key.split("_")[1]))
        paths.append(roll(env, states, T, rng, hold, scale, drift, drift_spectrum))
    nets = np.array([np.linalg.norm(p[-1] - p[0]) for p in paths])
    end_std = np.stack([p[-1] for p in paths]).std(0)
    return paths, nets, end_std


def plot_panel(ax, paths, title):
    cmap = plt.cm.tab20(np.linspace(0, 1, len(paths)))
    for p, c in zip(paths, cmap):
        ax.plot(p[:, 0], p[:, 1], "-", color=c, lw=1.0, alpha=0.8)
        ax.plot(p[0, 0], p[0, 1], "o", color=c, ms=5)          # start
        ax.plot(p[-1, 0], p[-1, 1], "*", color=c, ms=11)        # end
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("eef x (m)"); ax.set_ylabel("eef y (m)")
    ax.set_aspect("equal", "box"); ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--hold", type=int, default=10)
    ap.add_argument("--scale", type=float, default=1.8)
    ap.add_argument("--drift", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    src = h5py.File(SRC, "r"); dgrp = src["data"]
    demos = sorted([k for k in dgrp.keys() if k.startswith("demo_")],
                   key=lambda k: int(k.split("_")[1]))[:args.n]
    env = build_env()

    print(f"OLD  hold=4  scale=1.0 drift=0 ...", flush=True)
    old_paths, old_net, old_std = run_config(env, demos, dgrp, args.seed, 4, 1.0, 0.0)
    print(f"NEW  hold={args.hold} scale={args.scale} drift~U(0,{args.drift}) ...", flush=True)
    new_paths, new_net, new_std = run_config(env, demos, dgrp, args.seed, args.hold, args.scale,
                                             args.drift, drift_spectrum=True)

    print(f"\nnet |end-start| displacement (m):")
    print(f"  OLD  mean={old_net.mean():.3f}  std={old_net.std():.3f}  max={old_net.max():.3f}")
    print(f"  NEW  mean={new_net.mean():.3f}  std={new_net.std():.3f}  max={new_net.max():.3f}")
    print(f"end-eef spread std (x,y,z):")
    print(f"  OLD  {np.round(old_std,3)}")
    print(f"  NEW  {np.round(new_std,3)}")

    # shared axis limits so panels are comparable
    allp = np.concatenate(old_paths + new_paths, 0)
    pad = 0.05
    xlim = (allp[:, 0].min() - pad, allp[:, 0].max() + pad)
    ylim = (allp[:, 1].min() - pad, allp[:, 1].max() + pad)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    plot_panel(axes[0], old_paths, f"OLD  hold=4 scale=1.0 drift=0\nnet disp mean={old_net.mean():.2f}m")
    plot_panel(axes[1], new_paths,
               f"NEW  hold={args.hold} scale={args.scale} drift~U(0,{args.drift})\n"
               f"net disp mean={new_net.mean():.2f}m max={new_net.max():.2f}m")
    for ax in axes:
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    fig.suptitle(f"eef top-down paths, same {args.n} scenes  (o=start  *=end)", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"\nsaved -> {args.out}", flush=True)
    src.close(); env.close()


if __name__ == "__main__":
    main()
