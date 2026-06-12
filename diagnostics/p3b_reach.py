"""Probe 3b (the gap p3 left): is the REACH distance ‖eef − drawer_front‖
decodable from the frozen emb?

p3 probed eef_pos / axis_angle / gripper / drawer_qpos but NOT the reach
distance — the other full term of probe 6's privileged_cost (reach + 30·close).
The "never reached" half of p9 failures is governed by reach, so this closes the
loop: if reach is decodable, a learned cost-head over emb can supply the reach
gradient (fix covers both failure halves); if not, the cost needs eef fed in.

The reach term is reconstructed EXACTLY as p6 defines it:
    reach(frame) = ‖ proprio_eef(frame) − cabinet_top_xpos ‖
proprio[:,0:3] equals env.get_site_xpos(EEF_SITE) bit-for-bit (verified), and the
cabinet is fixed in world (xpos std ≤1 cm across the 50 init_states), so we use the
mean cabinet_top xpos — no per-demo re-sim needed.

Targets Y = [eef_pos(3), reach(1), close(1), privileged(1)]:
    eef_pos     anchor — should reproduce p3 (~0.78), proves same sampling
    reach       the never-tested term
    close       |drawer_qpos − closed|  (drawer was 0.45/0.77 in p3)
    privileged  reach + 30·close  — the full p6 cost a learned head must mimic

  reach R² ≈ eef R²  => reach decodable; learned cost-head can shape reach,
                        fix covers BOTH p9 failure halves.
  reach R² << eef R² => emb doesn't bind eef↔drawer; "never reached" half needs
                        eef/proprio fed into the cost explicitly, not just emb.

Usage: MUJOCO_GL=egl EGL_DEVICE_ID=0 python diagnostics/p3b_reach.py --n-ep 200
"""
import argparse
import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import load_jepa, encode_frames, DrawerH5, RESULTS_DIR
from p3_probe import ridge_fit_predict, mlp_fit_predict
from p6_oracle import build_env, find_drawer_joint, EEF_SITE, DRAWER_BODY, WARMUP

GROUPS = {"eef_pos": slice(0, 3), "reach": slice(3, 4),
          "close": slice(4, 5), "privileged": slice(5, 6)}


def cabinet_mean_xpos():
    """Mean cabinet_top world xpos + drawer closed value, read from the real env.
    Cabinet is ~fixed across init_states (std ≤1 cm), so the mean is faithful to
    every demo's reach target to within the noise floor."""
    env, init_states = build_env()
    drawer = find_drawer_joint(env)            # (name, qpos_addr, closed_val)
    xs = []
    for idx in range(len(init_states)):
        env.reset()
        env.set_init_state(init_states[idx])
        for _ in range(WARMUP):
            env.step(np.zeros(7))
        xs.append(env.sim.data.get_body_xpos(DRAWER_BODY).copy())
    env.close()
    xs = np.asarray(xs)
    cab = xs.mean(0)
    print(f"cabinet_top mean xpos = {cab.round(4)}  (std {xs.std(0).round(4)})")
    print(f"drawer closed value   = {drawer[2]:+.4f}")
    return cab, float(drawer[2])


def build_targets(h5, ep, cabinet, closed):
    """(L,6) = [eef_pos(3), reach(1), close(1), privileged(1)] for one episode."""
    eef = h5.proprio(ep)[:, 0:3]                          # (L,3) world-frame eef
    reach = np.linalg.norm(eef - cabinet[None], axis=1, keepdims=True)   # (L,1)
    q = h5.drawer_qpos(ep)                                # (L,1)
    close = np.abs(q - closed)                            # (L,1)  L1, like p6
    priv = reach + 30.0 * close                           # (L,1)
    return np.concatenate([eef, reach, close, priv], axis=1)


def report(name, Ytrue_s, Ypred_s, yscaler):
    print(f"\n  --- {name} ---")
    print(f"    {'group':>11} {'R2':>7} {'RMSE(orig)':>11}")
    Ytrue_o = yscaler.inverse_transform(Ytrue_s)
    Ypred_o = yscaler.inverse_transform(Ypred_s)
    out = {}
    for g, sl in {**GROUPS, "overall": slice(0, Ytrue_s.shape[1])}.items():
        r2 = r2_score(Ytrue_s[:, sl], Ypred_s[:, sl])
        rmse = np.sqrt(((Ytrue_o[:, sl] - Ypred_o[:, sl]) ** 2).mean())
        print(f"    {g:>11} {r2:>7.3f} {rmse:>11.4f}")
        out[g] = (float(r2), float(rmse))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=200)
    ap.add_argument("--n-ep", type=int, default=200)
    args = ap.parse_args()

    cabinet, closed = cabinet_mean_xpos()

    model = load_jepa(args.epoch)
    h5 = DrawerH5()
    # SAME rng/seed/split as p3 -> reach R2 directly comparable to p3's eef/drawer
    rng = np.random.default_rng(3)
    eps = rng.choice(h5.n_ep, size=min(args.n_ep, h5.n_ep), replace=False)
    n_test = max(1, int(0.2 * len(eps)))
    test_eps = set(eps[:n_test].tolist())

    Xtr, Ytr, Xte, Yte = [], [], [], []
    for ep in eps:
        emb = encode_frames(model, h5.pixels(ep), h5.eye_in_hand(ep)).numpy()  # (L,192)
        Y = build_targets(h5, ep, cabinet, closed)                            # (L,6)
        if ep in test_eps:
            Xte.append(emb); Yte.append(Y)
        else:
            Xtr.append(emb); Ytr.append(Y)
    Xtr = np.concatenate(Xtr); Ytr = np.concatenate(Ytr)
    Xte = np.concatenate(Xte); Yte = np.concatenate(Yte)
    print(f"\n===== Probe 3b: reach-distance probing =====")
    print(f"  train frames={len(Xtr)}  test frames={len(Xte)}  "
          f"(episode split, {len(test_eps)} test eps)")
    print(f"  reach range [{Yte[:,3].min():.3f}, {Yte[:,3].max():.3f}] m  "
          f"privileged range [{Yte[:,5].min():.2f}, {Yte[:,5].max():.2f}]")

    xs = StandardScaler().fit(Xtr)
    ys = StandardScaler().fit(Ytr)
    Xtr_s, Xte_s = xs.transform(Xtr), xs.transform(Xte)
    Ytr_s, Yte_s = ys.transform(Ytr), ys.transform(Yte)

    lin_pred = ridge_fit_predict(Xtr_s, Ytr_s, Xte_s)
    mlp_pred = mlp_fit_predict(Xtr_s, Ytr_s, Xte_s)

    lin_out = report("linear (ridge)", Yte_s, lin_pred, ys)
    mlp_out = report("2-layer MLP", Yte_s, mlp_pred, ys)

    # scatter: reach distance, MLP
    Yte_o = ys.inverse_transform(Yte_s)
    Yp_o = ys.inverse_transform(mlp_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(Yte_o[:, 3], Yp_o[:, 3], s=2, alpha=0.3)
    lo, hi = Yte_o[:, 3].min(), Yte_o[:, 3].max()
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set_xlabel("true reach ‖eef−drawer‖"); ax.set_ylabel("pred reach")
    ax.set_title(f"Probe 3b MLP: reach (R2={mlp_out['reach'][0]:.3f})")
    out_png = os.path.join(RESULTS_DIR, "p3b_reach.png")
    fig.tight_layout(); fig.savefig(out_png, dpi=120)
    np.savez(os.path.join(RESULTS_DIR, "p3b_reach.npz"),
             linear={k: v for k, v in lin_out.items()},
             mlp={k: v for k, v in mlp_out.items()},
             cabinet=cabinet, closed=closed)
    print(f"\nsaved -> {out_png}")
    h5.close()


if __name__ == "__main__":
    main()
