"""Roll ONE episode with given perturb params and dump a video + montage (agentview)."""
import argparse, os
import numpy as np, h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv

SRC = "/home/yikang/stable-wm/libero_bc_rollouts/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_bc.hdf5"
DRAWER_ADDR = 37


def clear_done(env):
    base = env.env if hasattr(env, "env") and hasattr(env.env, "done") else env
    base.done = False; base.ignore_done = True; base.timestep = 0


def get_obs(env):
    tgt = env if hasattr(env, "_get_observations") else env.env
    return tgt._get_observations()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=int, default=4240)
    ap.add_argument("--hold", type=int, default=10)
    ap.add_argument("--scale", type=float, default=1.8)
    ap.add_argument("--drift", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = h5py.File(SRC, "r"); dgrp = src["data"]
    key = f"demo_{args.demo}"
    states = dgrp[key]["states"][:]
    T = int(dgrp[key].attrs["num_samples"])
    rng = np.random.default_rng(args.seed + 1 + args.demo)

    bench = get_benchmark("LIBERO_90")(0); task = bench.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=224, camera_widths=224)

    start = int(rng.integers(0, T))
    env.reset(); env.sim.set_state_from_flattened(states[start]); env.sim.forward(); clear_done(env)
    obs = get_obs(env)
    drift_vec = (args.drift * rng.standard_normal(7)).astype(np.float32) if args.drift > 0 else np.zeros(7, np.float32)

    frames, dqs = [], []
    a = np.zeros(7, np.float32)
    for t in range(T):
        if t % args.hold == 0:
            a = np.clip(drift_vec + args.scale * rng.uniform(-1.0, 1.0, 7), -1.0, 1.0).astype(np.float32)
        frames.append(obs["agentview_image"].copy())
        dqs.append(float(env.sim.data.qpos[DRAWER_ADDR]))
        obs, _, _, _ = env.step(a)

    rdir = "/home/yikang/git/le-wm/diagnostics/results"
    tag = f"demo{args.demo}_h{args.hold}_s{args.scale}_d{args.drift}"
    vid = f"{rdir}/perturb_new_{tag}.mp4"
    imageio.mimsave(vid, frames, fps=15)

    # montage: 8 evenly spaced frames
    idx = np.linspace(0, T - 1, 8).astype(int)
    fig, axes = plt.subplots(1, 8, figsize=(20, 3))
    for ax, k in zip(axes, idx):
        ax.imshow(frames[k]); ax.axis("off")
        ax.set_title(f"t={k}\nq={dqs[k]:+.2f}", fontsize=8)
    fig.suptitle(f"perturb {tag}  (start@demo-step {start}, T={T}, drawer open=-0.16 closed=+0.01)", fontsize=10)
    fig.tight_layout()
    png = f"{rdir}/perturb_new_{tag}.png"
    fig.savefig(png, dpi=110)
    print(f"start={start} T={T} drawer_q {dqs[0]:+.3f}->{dqs[-1]:+.3f}")
    print(f"saved -> {vid}\nsaved -> {png}")
    src.close(); env.close()


if __name__ == "__main__":
    main()
