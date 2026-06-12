"""Re-render the DECODED-cost episode with TRUE vs D_φ-ESTIMATED reach/drawer_q
overlaid per frame, so the divergence (planner believes cost ↓ while truth ↑)
is directly visible. Also logs per-step (true vs decoded) = Phase-0 OOD-decode
error on the states CEM actually visits.

  yellow = TRUE (sim ground truth)         <- the honest observer
  red    = D_φ(emb) ESTIMATE               <- what the planner optimized

Usage:
  MUJOCO_GL=egl EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
      python diagnostics/p6d_video_dual.py --budget 80
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
from state_decoder import load_decoder, CABINET, CLOSED  # noqa: E402

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

YELLOW = (255, 255, 0)
RED = (255, 80, 80)


def annotate(img, lines):
    if not HAS_CV2:
        return img
    img = np.ascontiguousarray(img)
    y = 15
    for txt, col in lines:
        cv2.putText(img, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        y += 17
    return img


@torch.no_grad()
def decode_current(model, decoder, obs):
    """D_φ's estimate of (reach, q) on the CURRENT real frame."""
    emb = encode_frames(model, obs["agentview_image"][None],
                        obs["robot0_eye_in_hand_image"][None]).cuda()   # (1,192)
    eef, q = decoder(emb)
    cab = torch.as_tensor(CABINET, device=eef.device)
    reach = float(torch.linalg.norm(eef[0] - cab))
    return reach, float(q[0, 0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=200)
    ap.add_argument("--decoder", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "value", "state_decoder.pt"))
    ap.add_argument("--n-eval", type=int, default=3)
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

    rng = np.random.default_rng(args.seed)
    goal_eps = rng.choice(h5.n_ep, size=args.n_eval, replace=False)
    goal_embs = [encode_frames(model, h5.goal_frame(e)[None],
                               h5.goal_eye_frame(e)[None])[0] for e in goal_eps]
    init_idx = [int(rng.integers(0, len(init_states))) for _ in range(args.n_eval)]

    cfg = dict(H=args.H, N=args.N, n_iter=args.n_iter, topk=args.topk,
               var_scale=args.var_scale, receding=args.receding, budget=args.budget,
               debug=False, decoder=decoder, rng=np.random.default_rng(args.seed + 1))

    init_state = init_states[init_idx[args.ep]]
    goal_emb = goal_embs[args.ep]

    env.reset(); env.set_init_state(init_state)
    obs = None
    for _ in range(WARMUP):
        obs, _, _, _ = env.step(np.zeros(7))

    frames, log = [], []
    steps, success = 0, False
    while steps < cfg["budget"]:
        s_t = env.sim.get_state().flatten()
        mean = oracle_cem(env, model, s_t, goal_emb, cfg, "decoded", drawer)
        restore(env, s_t)
        for h in range(cfg["receding"]):
            obs, _, _, _ = env.step(mean[h]); steps += 1
            t_reach = float(np.linalg.norm(env.sim.data.get_site_xpos(EEF_SITE)
                                           - env.sim.data.get_body_xpos(DRAWER_BODY)))
            t_q = float(env.sim.data.qpos[drawer[1]])
            d_reach, d_q = decode_current(model, decoder, obs)
            log.append((steps, t_reach, d_reach, t_q, d_q))
            img = obs["agentview_image"][::-1].copy()
            frames.append(annotate(img, [
                (f"step {steps}", (255, 255, 255)),
                (f"TRUE  reach {t_reach:.3f}  q {t_q:+.3f}", YELLOW),
                (f"D_phi reach {d_reach:.3f}  q {d_q:+.3f}", RED),
            ]))
            if env.check_success():
                success = True; break
            if steps >= cfg["budget"]:
                break
        if success:
            break

    out_dir = os.path.join(RESULTS_DIR, "p6c_videos")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"decoded_dual_ep{args.ep}_{'SUCCESS' if success else 'fail'}.mp4")
    try:
        imageio.mimsave(path, frames, fps=10, macro_block_size=1)
    except Exception as e:
        path = path.replace(".mp4", ".gif"); imageio.mimsave(path, frames, fps=10)
        print(f"  (mp4 failed: {e}; gif instead)")

    arr = np.array(log)  # step, t_reach, d_reach, t_q, d_q
    print(f"\n=== decoded episode: {'SUCCESS' if success else 'fail'} steps={steps} -> {path}")
    print(f"  reach   true[{arr[:,1].min():.3f},{arr[:,1].max():.3f}]  "
          f"D_phi[{arr[:,2].min():.3f},{arr[:,2].max():.3f}]  "
          f"mean|err|={np.abs(arr[:,1]-arr[:,2]).mean():.3f}")
    print(f"  drawer  true[{arr[:,3].min():+.3f},{arr[:,3].max():+.3f}]  "
          f"D_phi[{arr[:,4].min():+.3f},{arr[:,4].max():+.3f}]  "
          f"mean|err|={np.abs(arr[:,3]-arr[:,4]).mean():.3f}")
    print(f"  corr(true_reach, D_phi_reach) = {np.corrcoef(arr[:,1], arr[:,2])[0,1]:+.3f}  "
          f"(want ~+1; negative = estimate moves OPPOSITE to truth)")
    np.savez(os.path.join(RESULTS_DIR, "p6d_dual.npz"), log=arr)
    env.close(); h5.close()


if __name__ == "__main__":
    main()
