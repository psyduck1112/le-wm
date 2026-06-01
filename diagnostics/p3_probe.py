"""Probe 3 (LIBERO_DIAGNOSTICS.md 探测3): linear probing of the frozen encoder.

Freeze the le-wm encoder and regress ground-truth proprio state from the embedding
with a deliberately weak readout (ridge-linear + 2-layer MLP). Weak probe can read
it out => info is linearly present in emb (encoder OK, bottleneck downstream). Weak
probe fails => encoder representation is fundamentally insufficient.

Target = proprio (8-D) = eef_pos(3) + eef_axis_angle(3) + gripper_qpos(2).
NOTE: object/drawer pose is NOT stored in libero_bc_drawer.h5 (BC rollouts kept no
sim state), so this probe only covers robot proprio. Object-pose probing would
require re-simulating the demos and is deferred.

Train/test split is BY EPISODE to avoid leakage between adjacent correlated frames.

Usage: python diagnostics/p3_probe.py --epoch 200 --n-ep 200
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import load_jepa, encode_frames, DrawerH5, RESULTS_DIR

GROUPS = {"eef_pos": slice(0, 3), "axis_angle": slice(3, 6), "gripper": slice(6, 8)}


def _ridge(Xtr, Ytr, Xte, lam):
    Xtr1 = np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1)
    Xte1 = np.concatenate([Xte, np.ones((len(Xte), 1))], 1)
    A = Xtr1.T @ Xtr1 + lam * np.eye(Xtr1.shape[1])
    W = np.linalg.solve(A, Xtr1.T @ Ytr)
    return Xte1 @ W


def ridge_fit_predict(Xtr, Ytr, Xte, lams=(1, 10, 100, 1e3, 1e4, 1e5)):
    """Ridge with lambda swept on a held-out val split (by row) to avoid overfit."""
    n = len(Xtr)
    nval = max(1, int(0.15 * n))
    Xt, Yt, Xv, Yv = Xtr[nval:], Ytr[nval:], Xtr[:nval], Ytr[:nval]
    best_lam, best_r2 = lams[0], -1e9
    for lam in lams:
        pred_v = _ridge(Xt, Yt, Xv, lam)
        r2 = r2_score(Yv, pred_v)
        if r2 > best_r2:
            best_r2, best_lam = r2, lam
    print(f"    [ridge] best lambda={best_lam:g} (val R2={best_r2:.3f})")
    return _ridge(Xtr, Ytr, Xte, best_lam)


def mlp_fit_predict(Xtr, Ytr, Xte, hidden=256, steps=5000, device="cuda"):
    """2-layer MLP probe with val-based early stopping + weight decay (fair upper bound)."""
    nval = max(1, int(0.15 * len(Xtr)))
    Xt = torch.tensor(Xtr[nval:], dtype=torch.float32, device=device)
    Yt = torch.tensor(Ytr[nval:], dtype=torch.float32, device=device)
    Xv = torch.tensor(Xtr[:nval], dtype=torch.float32, device=device)
    Yv = torch.tensor(Ytr[:nval], dtype=torch.float32, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.GELU(),
                        nn.Linear(hidden, Ytr.shape[1])).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.MSELoss()
    best_val, best_state, patience, bad = 1e9, None, 300, 0
    for it in range(steps):
        net.train(); opt.zero_grad()
        lossf(net(Xt), Yt).backward(); opt.step()
        if it % 50 == 0:
            net.eval()
            with torch.no_grad():
                v = lossf(net(Xv), Yv).item()
            if v < best_val - 1e-5:
                best_val, best_state, bad = v, {k: p.detach().clone()
                                                for k, p in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= patience // 50:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        return net(Xte_t).cpu().numpy()


def report(name, Ytrue, Ypred, yscaler):
    """Print per-group R2 (standardized) and RMSE (original units)."""
    print(f"\n  --- {name} ---")
    print(f"    {'group':>11} {'R2':>7} {'RMSE(orig)':>11}")
    Ytrue_o = yscaler.inverse_transform(Ytrue)
    Ypred_o = yscaler.inverse_transform(Ypred)
    out = {}
    for g, sl in {**GROUPS, "overall": slice(0, Ytrue.shape[1])}.items():
        r2 = r2_score(Ytrue[:, sl], Ypred[:, sl])
        rmse = np.sqrt(((Ytrue_o[:, sl] - Ypred_o[:, sl]) ** 2).mean())
        print(f"    {g:>11} {r2:>7.3f} {rmse:>11.4f}")
        out[g] = (r2, rmse)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=200)
    ap.add_argument("--n-ep", type=int, default=200)
    args = ap.parse_args()

    model = load_jepa(args.epoch)
    h5 = DrawerH5()
    rng = np.random.default_rng(3)
    eps = rng.choice(h5.n_ep, size=min(args.n_ep, h5.n_ep), replace=False)
    n_test = max(1, int(0.2 * len(eps)))
    test_eps = set(eps[:n_test].tolist())

    Xtr, Ytr, Xte, Yte = [], [], [], []
    for ep in eps:
        emb = encode_frames(model, h5.pixels(ep)).numpy()   # (L,192)
        pro = h5.proprio(ep)                                  # (L,8)
        if ep in test_eps:
            Xte.append(emb); Yte.append(pro)
        else:
            Xtr.append(emb); Ytr.append(pro)
    Xtr = np.concatenate(Xtr); Ytr = np.concatenate(Ytr)
    Xte = np.concatenate(Xte); Yte = np.concatenate(Yte)
    print(f"\n===== Probe 3: linear probing =====")
    print(f"  train frames={len(Xtr)}  test frames={len(Xte)}  "
          f"(episode split, {len(test_eps)} test eps)")

    xs = StandardScaler().fit(Xtr)
    ys = StandardScaler().fit(Ytr)
    Xtr_s, Xte_s = xs.transform(Xtr), xs.transform(Xte)
    Ytr_s, Yte_s = ys.transform(Ytr), ys.transform(Yte)

    lin_pred = ridge_fit_predict(Xtr_s, Ytr_s, Xte_s)
    mlp_pred = mlp_fit_predict(Xtr_s, Ytr_s, Xte_s)

    lin_out = report("linear (ridge)", Yte_s, lin_pred, ys)
    mlp_out = report("2-layer MLP", Yte_s, mlp_pred, ys)

    # scatter: eef_pos x-component, MLP
    Yte_o = ys.inverse_transform(Yte_s)
    Yp_o = ys.inverse_transform(mlp_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(Yte_o[:, 0], Yp_o[:, 0], s=2, alpha=0.3)
    lo, hi = Yte_o[:, 0].min(), Yte_o[:, 0].max()
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set_xlabel("true eef_pos_x"); ax.set_ylabel("pred eef_pos_x")
    ax.set_title("Probe 3 MLP: eef_pos_x")
    out_png = os.path.join(RESULTS_DIR, "p3_probe.png")
    fig.tight_layout(); fig.savefig(out_png, dpi=120)
    np.savez(os.path.join(RESULTS_DIR, "p3_probe.npz"),
             linear={k: v for k, v in lin_out.items()},
             mlp={k: v for k, v in mlp_out.items()})
    print(f"\nsaved -> {out_png}")
    h5.close()


if __name__ == "__main__":
    main()
