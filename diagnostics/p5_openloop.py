"""Probe 5 (LIBERO_DIAGNOSTICS.md 探测5): multi-step open-loop compounding error.

Training only optimizes single-step teacher forcing (num_preds=1). Planning uses
autoregressive open-loop rollout: from step 2 on, the predictor's input is its OWN
prediction (OOD), so errors snowball. This measures how fast that snowball grows,
which teacher-forcing validation never sees.

Open-loop (autoregressive), replicating jepa.rollout:
    true emb_0..emb_{HS-1} -> pred emb_HS -> feed back -> pred emb_{HS+1} -> ...
Compare emb_pred_k vs emb_true_k = encode(true frame at t0+k).

Baselines (so absolute L2 is interpretable):
    copy_k = ||emb_true[t0] - emb_true[t0+k]||   (do-nothing drift of the emb)
    relative_k = ||pred-true|| / ||true||

Usage: python diagnostics/p5_openloop.py --epoch 200 --n-ep 30 --horizon 10
"""
import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import load_jepa, encode_frames, DrawerH5, HISTORY, RESULTS_DIR


@torch.no_grad()
def open_loop_rollout(model, emb_seed, act_emb_seq, n_steps, hs=HISTORY):
    """emb_seed: (hs,D) true history. act_emb_seq: (hs-1+n_steps, A) action embeddings
    aligned so act_emb_seq[i] is the action taken at the i-th frame of the window.
    Returns pred_embs: (n_steps, D) the autoregressively predicted emb_{t0+1..t0+n}."""
    emb = emb_seed.clone()                       # (hs,D) grows over time
    preds = []
    for k in range(n_steps):
        win_e = emb[-hs:].unsqueeze(0)           # (1,hs,D)
        # actions aligned to the frames currently in the window
        win_a = act_emb_seq[k:k + hs].unsqueeze(0)  # (1,hs,A)
        pred = model.predict(win_e, win_a)[:, -1]   # (1,D) predict next emb
        emb = torch.cat([emb, pred[0:1]], dim=0)    # feed prediction back (OOD)
        preds.append(pred[0])
    return torch.stack(preds)                    # (n_steps,D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=200)
    ap.add_argument("--n-ep", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--stride", type=int, default=5, help="t0 sampling stride per traj")
    args = ap.parse_args()
    H, hs = args.horizon, HISTORY

    model = load_jepa(args.epoch)
    h5 = DrawerH5()
    rng = np.random.default_rng(2)
    eps = rng.choice(h5.n_ep, size=min(args.n_ep, h5.n_ep), replace=False)

    # accumulate per-k errors
    err = [[] for _ in range(H)]      # model open-loop L2 err
    copy = [[] for _ in range(H)]     # do-nothing drift
    rel = [[] for _ in range(H)]      # relative err
    for ep in eps:
        pix = h5.pixels(ep)
        L = len(pix)
        if L < hs + H:
            continue
        emb = encode_frames(model, pix, h5.eye_in_hand(ep)).cuda()   # (L,D) 双相机
        act = torch.from_numpy(h5.actions(ep)).float().unsqueeze(0).cuda()
        act_emb = model.action_encoder(act)[0]                 # (L,A)
        # multiple start points along the trajectory
        for t0 in range(hs - 1, L - H - 1, args.stride):
            emb_seed = emb[t0 - hs + 1:t0 + 1]                 # (hs,D) true history
            # action window: frames t0-hs+1 .. t0+H-1 -> length hs-1+H, sliced inside
            act_seq = act_emb[t0 - hs + 1:t0 - hs + 1 + (hs - 1) + H]
            if act_seq.size(0) < (hs - 1) + H:
                continue
            preds = open_loop_rollout(model, emb_seed, act_seq, H)  # (H,D)
            for k in range(1, H + 1):
                true_k = emb[t0 + k]
                err[k - 1].append((preds[k - 1] - true_k).norm().item())
                copy[k - 1].append((emb[t0] - true_k).norm().item())
                rel[k - 1].append(((preds[k - 1] - true_k).norm()
                                   / true_k.norm().clamp_min(1e-6)).item())

    err_m = np.array([np.mean(e) for e in err]); err_s = np.array([np.std(e) for e in err])
    copy_m = np.array([np.mean(c) for c in copy])
    rel_m = np.array([np.mean(r) for r in rel])
    ks = np.arange(1, H + 1)

    print("\n===== Probe 5: multi-step open-loop error =====")
    print(f"  episodes={len(eps)}  windows/k≈{len(err[0])}")
    print(f"  {'k':>3} {'err':>8} {'copy':>8} {'err/copy':>9} {'rel':>7}")
    for i in range(H):
        print(f"  {ks[i]:>3} {err_m[i]:>8.4f} {copy_m[i]:>8.4f} "
              f"{err_m[i]/max(copy_m[i],1e-9):>9.3f} {rel_m[i]:>7.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ks, err_m, "-o", label="open-loop pred err")
    ax.fill_between(ks, err_m - err_s, err_m + err_s, alpha=0.2)
    ax.plot(ks, copy_m, "--s", label="copy-last (do nothing)")
    ax.set_xlabel("rollout step k"); ax.set_ylabel("||emb_pred_k - emb_true_k||")
    ax.set_title("Probe 5 open-loop compounding error")
    ax.legend()
    out_png = os.path.join(RESULTS_DIR, "p5_openloop.png")
    fig.tight_layout(); fig.savefig(out_png, dpi=120)
    np.savez(os.path.join(RESULTS_DIR, "p5_openloop.npz"),
             k=ks, err_mean=err_m, err_std=err_s, copy_mean=copy_m, rel_mean=rel_m)
    print(f"\nsaved -> {out_png}")
    h5.close()


if __name__ == "__main__":
    main()
