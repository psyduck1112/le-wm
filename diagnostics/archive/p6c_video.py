"""Record videos of decoded-cost vs privileged-cost oracle CEM on the SAME episode.

Same init + same oracle dynamics + same CEM config; only the COST differs:
  privileged : reads true sim eef/q   -> arm reaches & closes drawer (success)
  decoded    : reads D_φ(emb)          -> arm driven AWAY, drawer never moves
Each executed frame is annotated with step / drawer_q / true reach so the failure
is visible. agentview is flipped vertically for human viewing only.

Usage:
  MUJOCO_GL=egl EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
      python diagnostics/p6c_video.py --budget 80
"""
import argparse
import os
import sys
import numpy as np
import torch
import imageio

from _common import load_jepa, encode_frames, DrawerH5, RESULTS_DIR
from p6_oracle import (WARMUP, build_env, find_drawer_joint, oracle_cem, restore,
                       EEF_SITE, DRAWER_BODY)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "value"))
from state_decoder import load_decoder  # noqa: E402

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


def annotate(img, lines):
    """img (H,W,3) uint8 (already vertically flipped for viewing). Overlay text."""
    if not HAS_CV2:
        return img
    img = np.ascontiguousarray(img)
    y = 16
    for txt in lines:
        cv2.putText(img, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 0), 1, cv2.LINE_AA)
        y += 18
    return img


def run_and_record(env, model, decoder, init_state, goal_emb, cfg, cost_mode, drawer):
    env.reset()
    env.set_init_state(init_state)
    obs = None
    for _ in range(WARMUP):
        obs, _, _, _ = env.step(np.zeros(7))
    frames, steps, success = [], 0, False
    while steps < cfg["budget"]:
        s_t = env.sim.get_state().flatten()
        mean = oracle_cem(env, model, s_t, goal_emb, cfg, cost_mode, drawer)
        restore(env, s_t)
        for h in range(cfg["receding"]):
            obs, _, _, _ = env.step(mean[h]); steps += 1
            q = float(env.sim.data.qpos[drawer[1]])
            reach = float(np.linalg.norm(env.sim.data.get_site_xpos(EEF_SITE)
                                         - env.sim.data.get_body_xpos(DRAWER_BODY)))
            img = obs["agentview_image"][::-1].copy()          # flip for viewing only
            frames.append(annotate(img, [
                f"{cost_mode}  step {steps}",
                f"drawer_q {q:+.3f} (closed +0.01)",
                f"reach {reach:.3f}",
            ]))
            if env.check_success():
                success = True; break
            if steps >= cfg["budget"]:
                break
        if success:
            break
    return frames, success, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=200)
    ap.add_argument("--decoder", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "value", "state_decoder.pt"))
    ap.add_argument("--n-eval", type=int, default=3)   # match the run you watched
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--H", type=int, default=5)
    ap.add_argument("--N", type=int, default=150)
    ap.add_argument("--n-iter", type=int, default=5)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--var-scale", type=float, default=0.5)
    ap.add_argument("--receding", type=int, default=3)
    ap.add_argument("--budget", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model = load_jepa(args.epoch)
    decoder = load_decoder(args.decoder)
    env, init_states = build_env()
    drawer = find_drawer_joint(env)
    h5 = DrawerH5()

    # reproduce p6_oracle's exact sampling so this is the episode you saw
    rng = np.random.default_rng(args.seed)
    goal_eps = rng.choice(h5.n_ep, size=args.n_eval, replace=False)
    goal_embs = [encode_frames(model, h5.goal_frame(e)[None],
                               h5.goal_eye_frame(e)[None])[0] for e in goal_eps]
    init_idx = [int(rng.integers(0, len(init_states))) for _ in range(args.n_eval)]

    cfg = dict(H=args.H, N=args.N, n_iter=args.n_iter, topk=args.topk,
               var_scale=args.var_scale, receding=args.receding, budget=args.budget,
               debug=False, decoder=decoder, rng=np.random.default_rng(args.seed + 1))

    out_dir = os.path.join(RESULTS_DIR, "p6c_videos")
    os.makedirs(out_dir, exist_ok=True)
    init_state = init_states[init_idx[args.ep]]
    goal_emb = goal_embs[args.ep]

    for mode in ("privileged", "decoded"):
        frames, ok, steps = run_and_record(env, model, decoder, init_state,
                                            goal_emb, cfg, mode, drawer)
        path = os.path.join(out_dir, f"{mode}_ep{args.ep}_{'SUCCESS' if ok else 'fail'}.mp4")
        try:
            imageio.mimsave(path, frames, fps=10, macro_block_size=1)
        except Exception as e:
            path = path.replace(".mp4", ".gif")
            imageio.mimsave(path, frames, fps=10)
            print(f"  (mp4 failed: {e}; wrote gif)")
        print(f"[{mode:10s}] {'SUCCESS' if ok else 'fail'} steps={steps} "
              f"frames={len(frames)} -> {path}", flush=True)

    env.close(); h5.close()


if __name__ == "__main__":
    main()
