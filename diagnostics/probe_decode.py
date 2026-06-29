"""Decode-probe — consolidates p3 / p3b / p3c into one script.

Question (all three modes): is privileged state DECODABLE from the frozen le-wm
embedding? Freeze the encoder, regress ground-truth quantities from the 192-d emb
with a deliberately weak readout (ridge-linear + 2-layer MLP). Decodable => the
info IS in the emb (encoder OK, bottleneck downstream); not decodable => the
representation is fundamentally insufficient. Scored by R².

Three modes (same episode-split, seed 3, so R² is comparable across modes):
  --target proprio   (old p3)  Y = eef_pos(3)+axis_angle(3)+gripper(2)+drawer_q(1)
                                writes p3_probe.{png,npz}
  --target full      (old p3b) Y = eef_pos(3)+reach(1)+close(1)+privileged(1)
                                reach=‖eef−cabinet‖, close=|q−closed|, priv=reach+30·close
                                writes p3b_reach.{png,npz}
  --compare-cam      (old p3c) implies --target full; ALSO encodes the MONO
                                (agentview-only) model and reports ΔR² = dual − mono
                                writes p3c_cam_ablation.{npz,log}

Output filenames are kept identical to the originals so make_report_figs.py still
finds the artifacts. The three legacy scripts are preserved in archive/.

Usage:
  python diagnostics/probe_decode.py --target proprio --n-ep 200
  MUJOCO_GL=egl EGL_DEVICE_ID=0 python diagnostics/probe_decode.py --target full
  MUJOCO_GL=egl EGL_DEVICE_ID=0 python diagnostics/probe_decode.py --compare-cam
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

from _common import load_jepa, load_phys, encode_frames, DrawerH5, RESULTS_DIR
from p6_oracle import build_env, find_drawer_joint, DRAWER_BODY, WARMUP


def load_model(args):
    """v2 (frozen, no task_head via AutoCostModel) or phys (co-trained task_head)."""
    return load_phys(args.epoch) if args.model == "phys" else load_jepa(args.epoch)

GROUPS_PROPRIO = {"eef_pos": slice(0, 3), "axis_angle": slice(3, 6),
                  "gripper": slice(6, 8), "drawer": slice(8, 9)}
GROUPS_FULL = {"eef_pos": slice(0, 3), "reach": slice(3, 4),
               "close": slice(4, 5), "privileged": slice(5, 6)}


# ---------------------------------------------------------------- probe core
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


def report(name, Ytrue_s, Ypred_s, yscaler, groups):
    """Print per-group R2 (standardized) and RMSE (original units)."""
    print(f"\n  --- {name} ---")
    print(f"    {'group':>11} {'R2':>7} {'RMSE(orig)':>11}")
    Ytrue_o = yscaler.inverse_transform(Ytrue_s)
    Ypred_o = yscaler.inverse_transform(Ypred_s)
    out = {}
    for g, sl in {**groups, "overall": slice(0, Ytrue_s.shape[1])}.items():
        r2 = r2_score(Ytrue_s[:, sl], Ypred_s[:, sl])
        rmse = np.sqrt(((Ytrue_o[:, sl] - Ypred_o[:, sl]) ** 2).mean())
        print(f"    {g:>11} {r2:>7.3f} {rmse:>11.4f}")
        out[g] = (float(r2), float(rmse))
    return out


# ---------------------------------------------------------------- targets
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


def targets_proprio(h5, ep, *_):
    """(L,9) = eef_pos(3)+axis_angle(3)+gripper(2)+drawer_q(1)."""
    return np.concatenate([h5.proprio(ep), h5.drawer_qpos(ep)], axis=1)


def targets_full(h5, ep, cabinet, closed):
    """(L,6) = [eef_pos(3), reach(1), close(1), privileged(1)] — p6 cost terms."""
    eef = h5.proprio(ep)[:, 0:3]
    reach = np.linalg.norm(eef - cabinet[None], axis=1, keepdims=True)
    q = h5.drawer_qpos(ep)
    close = np.abs(q - closed)
    priv = reach + 30.0 * close
    return np.concatenate([eef, reach, close, priv], axis=1)


def split_eps(h5, n_ep, seed=3):
    rng = np.random.default_rng(seed)
    eps = rng.choice(h5.n_ep, size=min(n_ep, h5.n_ep), replace=False)
    n_test = max(1, int(0.2 * len(eps)))
    return eps, set(eps[:n_test].tolist())


def encode_all(model, h5, eps, test_eps, use_eye, build_Y, cabinet, closed):
    Xtr, Ytr, Xte, Yte = [], [], [], []
    for ep in eps:
        eye = h5.eye_in_hand(ep) if use_eye else None
        emb = encode_frames(model, h5.pixels(ep), eye).numpy()   # (L,192)
        Y = build_Y(h5, ep, cabinet, closed)
        if ep in test_eps:
            Xte.append(emb); Yte.append(Y)
        else:
            Xtr.append(emb); Ytr.append(Y)
    return (np.concatenate(Xtr), np.concatenate(Ytr),
            np.concatenate(Xte), np.concatenate(Yte))


# ---------------------------------------------------------------- modes
def run_single(args, build_Y, groups, cabinet, closed, scatter_idx, scatter_label,
               out_stem):
    """Single-model decode probe (p3 / p3b). scatter_idx = target column to plot."""
    model = load_model(args)
    h5 = DrawerH5()
    eps, test_eps = split_eps(h5, args.n_ep)
    Xtr, Ytr, Xte, Yte = encode_all(model, h5, eps, test_eps, True, build_Y,
                                    cabinet, closed)
    print(f"\n===== decode probe ({args.target}) =====")
    print(f"  train frames={len(Xtr)}  test frames={len(Xte)}  "
          f"(episode split, {len(test_eps)} test eps)")

    xs = StandardScaler().fit(Xtr)
    ys = StandardScaler().fit(Ytr)
    Xtr_s, Xte_s = xs.transform(Xtr), xs.transform(Xte)
    Ytr_s, Yte_s = ys.transform(Ytr), ys.transform(Yte)

    lin_pred = ridge_fit_predict(Xtr_s, Ytr_s, Xte_s)
    mlp_pred = mlp_fit_predict(Xtr_s, Ytr_s, Xte_s)
    lin_out = report("linear (ridge)", Yte_s, lin_pred, ys, groups)
    mlp_out = report("2-layer MLP", Yte_s, mlp_pred, ys, groups)

    Yte_o = ys.inverse_transform(Yte_s)
    Yp_o = ys.inverse_transform(mlp_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(Yte_o[:, scatter_idx], Yp_o[:, scatter_idx], s=2, alpha=0.3)
    lo, hi = Yte_o[:, scatter_idx].min(), Yte_o[:, scatter_idx].max()
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set_xlabel(f"true {scatter_label}"); ax.set_ylabel(f"pred {scatter_label}")
    ax.set_title(f"decode MLP: {scatter_label}")
    out_png = os.path.join(RESULTS_DIR, out_stem + ".png")
    fig.tight_layout(); fig.savefig(out_png, dpi=120)
    save = dict(linear={k: v for k, v in lin_out.items()},
                mlp={k: v for k, v in mlp_out.items()})
    if cabinet is not None:
        save.update(cabinet=cabinet, closed=closed)
    np.savez(os.path.join(RESULTS_DIR, out_stem + ".npz"), **save)
    print(f"\nsaved -> {out_png}")
    h5.close()


def probe_r2(Xtr, Ytr, Xte, Yte, groups):
    """Return dict group -> (lin R², mlp R²)."""
    xs = StandardScaler().fit(Xtr)
    ys = StandardScaler().fit(Ytr)
    Xtr_s, Xte_s = xs.transform(Xtr), xs.transform(Xte)
    Ytr_s, Yte_s = ys.transform(Ytr), ys.transform(Yte)
    lin = ridge_fit_predict(Xtr_s, Ytr_s, Xte_s)
    mlp = mlp_fit_predict(Xtr_s, Ytr_s, Xte_s)
    return {g: (float(r2_score(Yte_s[:, sl], lin[:, sl])),
                float(r2_score(Yte_s[:, sl], mlp[:, sl]))) for g, sl in groups.items()}


def run_compare_cam(args, cabinet, closed):
    """Mono vs dual encoder ΔR² on the FULL targets (old p3c)."""
    import stable_worldmodel as swm
    mono_prefix = "/home/yikang/stable-wm/outputs/lewm_libero_bc_drawer_epoch_200"
    groups = GROUPS_FULL
    h5 = DrawerH5()
    eps, test_eps = split_eps(h5, args.n_ep)
    print(f"\n===== wrist-camera encoder ablation (mono vs dual) =====")
    print(f"  episodes={len(eps)}  test_eps={len(test_eps)}  (episode split, seed 3)")

    dual = load_jepa(args.epoch)
    print("encoding DUAL (agentview + wrist)...", flush=True)
    d = encode_all(dual, h5, eps, test_eps, True, targets_full, cabinet, closed)
    del dual; torch.cuda.empty_cache()

    mono = swm.policy.AutoCostModel(mono_prefix).to("cuda").eval().requires_grad_(False)
    mono.interpolate_pos_encoding = True
    print("encoding MONO (agentview only)...", flush=True)
    m = encode_all(mono, h5, eps, test_eps, False, targets_full, cabinet, closed)
    del mono; torch.cuda.empty_cache()

    print(f"  train frames={len(d[0])}  test frames={len(d[2])}")
    r_dual, r_mono = probe_r2(*d, groups), probe_r2(*m, groups)

    print("\n  decode R² (higher = more task info in emb).  ΔR² = dual − mono")
    print(f"  {'group':>11} | {'mono lin':>9} {'dual lin':>9} {'Δlin':>7} | "
          f"{'mono mlp':>9} {'dual mlp':>9} {'Δmlp':>7}")
    print("  " + "-" * 78)
    lines = []
    for g in groups:
        ml, dl = r_mono[g][0], r_dual[g][0]
        mm, dm = r_mono[g][1], r_dual[g][1]
        line = (f"  {g:>11} | {ml:>9.3f} {dl:>9.3f} {dl-ml:>+7.3f} | "
                f"{mm:>9.3f} {dm:>9.3f} {dm-mm:>+7.3f}")
        print(line); lines.append(line)

    np.savez(os.path.join(RESULTS_DIR, "p3c_cam_ablation.npz"),
             mono={g: r_mono[g] for g in groups},
             dual={g: r_dual[g] for g in groups})
    with open(os.path.join(RESULTS_DIR, "p3c_cam_ablation.log"), "w") as f:
        f.write("group | mono_lin dual_lin dlin | mono_mlp dual_mlp dmlp\n")
        f.write("\n".join(lines) + "\n")
    print(f"\nsaved -> {os.path.join(RESULTS_DIR, 'p3c_cam_ablation.log')}")
    h5.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=200)
    ap.add_argument("--n-ep", type=int, default=200)
    ap.add_argument("--model", choices=["v2", "phys"], default="v2",
                    help="v2=frozen baseline (no task_head); phys=co-trained task_head")
    ap.add_argument("--target", choices=["proprio", "full"], default="proprio")
    ap.add_argument("--compare-cam", action="store_true",
                    help="mono-vs-dual ΔR² ablation (implies --target full)")
    args = ap.parse_args()

    if args.compare_cam:
        cabinet, closed = cabinet_mean_xpos()
        run_compare_cam(args, cabinet, closed)
    elif args.target == "full":
        cabinet, closed = cabinet_mean_xpos()
        run_single(args, targets_full, GROUPS_FULL, cabinet, closed,
                   scatter_idx=3, scatter_label="reach ‖eef−drawer‖",
                   out_stem="p3b_reach")
    else:
        run_single(args, targets_proprio, GROUPS_PROPRIO, None, None,
                   scatter_idx=0, scatter_label="eef_pos_x", out_stem="p3_probe")


if __name__ == "__main__":
    main()
