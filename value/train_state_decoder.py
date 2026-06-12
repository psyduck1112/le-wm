"""Train D_phi (StateDecoder): frozen emb -> (eef_pos, drawer_qpos).

Reuses value/emb_cache.npz (all-frame frozen dual-cam embeddings) so there is NO
re-encoding -- the ViT is never touched here. Labels: eef = proprio[:,0:3] from
the h5, drawer_qpos from the cache. Episode-split train/test. Reports R2 on
eef / reach / close / privileged so it is directly comparable to p3b.

Usage:
  CUDA_VISIBLE_DEVICES=0 python value/train_state_decoder.py            # all eps
  CUDA_VISIBLE_DEVICES=0 python value/train_state_decoder.py --n-ep 1000
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "diagnostics"))
from _common import DrawerH5                       # noqa: E402  (proprio -> eef labels)
from state_decoder import StateDecoder, CABINET, CLOSED  # noqa: E402

CACHE = os.path.join(HERE, "emb_cache.npz")
OUT = os.path.join(HERE, "state_decoder.pt")


def reach_close_priv(eef, q):
    reach = np.linalg.norm(eef - CABINET[None], axis=1)
    close = np.abs(q[:, 0] - CLOSED)
    return reach, close, reach + 30.0 * close


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ep", type=int, default=0, help="0 = all episodes in cache")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    device = "cuda"

    cache = np.load(CACHE)
    emb_all = cache["emb"].astype(np.float32)            # (N,192)
    ep_off = cache["ep_offset"].astype(np.int64)
    ep_len = cache["ep_len"].astype(np.int64)
    q_all = cache["drawer_qpos"].astype(np.float32)      # (N,1)
    h5 = DrawerH5()
    eef_all = h5.f["proprio"][:, 0:3].astype(np.float32)  # (N,3) global-index aligned
    h5.close()
    assert len(emb_all) == len(eef_all) == len(q_all), "cache / h5 misaligned"

    n_ep = len(ep_off) if args.n_ep == 0 else min(args.n_ep, len(ep_off))
    rng = np.random.default_rng(args.seed)
    eps = rng.choice(len(ep_off), size=n_ep, replace=False)
    n_test = max(1, int(0.2 * n_ep))
    test_eps = set(eps[:n_test].tolist())

    def gather(ep_list):
        Xs, Ys = [], []
        for ep in ep_list:
            sl = slice(int(ep_off[ep]), int(ep_off[ep]) + int(ep_len[ep]))
            Xs.append(emb_all[sl])
            Ys.append(np.concatenate([eef_all[sl], q_all[sl]], axis=1))  # (l,4)
        return np.concatenate(Xs), np.concatenate(Ys)

    Xtr, Ytr = gather([e for e in eps if e not in test_eps])
    Xte, Yte = gather([e for e in eps if e in test_eps])
    print(f"\n===== Train D_phi (StateDecoder) =====")
    print(f"  train {len(Xtr)}  test {len(Xte)}  ({n_ep} eps, {len(test_eps)} test eps)")

    in_mean, in_std = Xtr.mean(0), Xtr.std(0)
    out_mean, out_std = Ytr.mean(0), Ytr.std(0)
    model = StateDecoder(emb_dim=192, hidden=args.hidden).to(device)
    model.set_norm(in_mean, in_std, out_mean, out_std)

    # train net on STANDARDIZED in/out (well-scaled); model.raw un-standardizes
    Xn = (torch.tensor(Xtr, device=device) - model.in_mean) / model.in_std
    Yn = (torch.tensor(Ytr, device=device) - model.out_mean) / model.out_std
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(Xn), generator=g)
    nval = max(1, int(0.15 * len(Xn)))
    vi, ti = perm[:nval].to(device), perm[nval:].to(device)
    Xt, Yt, Xv, Yv = Xn[ti], Yn[ti], Xn[vi], Yn[vi]

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.MSELoss()
    best, best_state, bad, patience = 1e9, None, 0, 6
    for it in range(args.steps):
        model.train(); opt.zero_grad()
        lossf(model.net(Xt), Yt).backward(); opt.step()
        if it % 50 == 0:
            model.eval()
            with torch.no_grad():
                v = lossf(model.net(Xv), Yv).item()
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = {k: p.detach().clone() for k, p in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    print(f"  early stop @ it={it}  val_mse={best:.4f}")
                    break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        eef_p, q_p = model(torch.tensor(Xte, device=device))
    eef_p, q_p = eef_p.cpu().numpy(), q_p.cpu().numpy()
    eef_t, q_t = Yte[:, 0:3], Yte[:, 3:4]
    reach_t, close_t, priv_t = reach_close_priv(eef_t, q_t)
    reach_p, close_p, priv_p = reach_close_priv(eef_p, q_p)

    rows = [
        ("eef_pos", r2_score(eef_t, eef_p), np.sqrt(((eef_t - eef_p) ** 2).mean()), 0.769),
        ("drawer_q", r2_score(q_t, q_p), np.sqrt(((q_t - q_p) ** 2).mean()), 0.747),
        ("reach", r2_score(reach_t, reach_p), np.sqrt(((reach_t - reach_p) ** 2).mean()), 0.705),
        ("close", r2_score(close_t, close_p), np.sqrt(((close_t - close_p) ** 2).mean()), 0.747),
        ("privileged", r2_score(priv_t, priv_p), np.sqrt(((priv_t - priv_p) ** 2).mean()), 0.750),
    ]
    print(f"\n  {'group':>11} {'R2':>7} {'RMSE':>9} {'p3b':>7}")
    r2map = {}
    for name, r2, rmse, p3b in rows:
        print(f"  {name:>11} {r2:>7.3f} {rmse:>9.4f} {p3b:>7.3f}")
        r2map[name] = float(r2)

    torch.save({"state_dict": model.state_dict(), "emb_dim": 192, "hidden": args.hidden,
                "cabinet": CABINET, "closed": CLOSED, "r2": r2map, "n_ep": n_ep}, OUT)
    print(f"\nsaved -> {OUT}")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(reach_t, reach_p, s=2, alpha=0.3)
    lo, hi = reach_t.min(), reach_t.max()
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set_xlabel("true reach"); ax.set_ylabel("pred reach")
    ax.set_title(f"D_phi reach (R2={r2map['reach']:.3f})")
    png = os.path.join(HERE, "state_decoder.png")
    fig.tight_layout(); fig.savefig(png, dpi=120)
    print(f"saved -> {png}")


if __name__ == "__main__":
    main()
