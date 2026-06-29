"""Probe 3c: does the WRIST (eye_in_hand) camera improve the ENCODER?

Matched pair, same epoch 200, same 192-d emb (so NO dimensionality confound):
  mono : lewm_libero_bc_drawer       agentview only      (rejects eye -> single-cam)
  dual : lewm_libero_bc_drawer_v2    agentview + wrist   (two shared-encoder CLS -> projector -> 192)

p3b proved the dual emb decodes eef/reach/drawer_q. The question "did the wrist
camera help the encoder" is a *delta*, not an absolute: a single probe R² can't
attribute it. So we run the IDENTICAL probe (same held-out episodes, same targets,
same capacity) on each model's emb and compare R². The emb is the only thing that
changes; targets (eef from proprio, reach, drawer_q) are model-independent.

  dual R² > mono R² (beyond noise)  => wrist camera injects decodable task info
                                       into the encoder representation.
  dual R² ≈ mono R²                 => wrist adds nothing the agentview lacked
                                       (for these privileged quantities).

CONFOUND (flagged, not removable): mono was trained on the v1 dataset (deleted),
dual on v2. agentview rendering is the same content, so this is secondary, but the
delta mixes "extra camera" with "different training data". A truly clean isolation
would retrain mono on the v2 agentview stream; this reuses what already exists.

Usage:
  MUJOCO_GL=egl EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
      python diagnostics/p3c_cam_ablation.py --n-ep 200
"""
import argparse
import os
import numpy as np
import torch
import stable_worldmodel as swm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from _common import load_jepa, encode_frames, DrawerH5, RESULTS_DIR
from p3_probe import ridge_fit_predict, mlp_fit_predict
from p3b_reach import cabinet_mean_xpos, build_targets, GROUPS

MONO_PREFIX = "/home/yikang/stable-wm/outputs/lewm_libero_bc_drawer_epoch_200"


def load_mono(prefix=MONO_PREFIX, device="cuda"):
    m = swm.policy.AutoCostModel(prefix).to(device).eval().requires_grad_(False)
    m.interpolate_pos_encoding = True
    return m


def encode_all(model, h5, eps, test_eps, use_eye, cabinet, closed):
    """Return Xtr, Ytr, Xte, Yte for one model. Y is identical across models."""
    Xtr, Ytr, Xte, Yte = [], [], [], []
    for ep in eps:
        eye = h5.eye_in_hand(ep) if use_eye else None
        emb = encode_frames(model, h5.pixels(ep), eye).numpy()   # (L,192)
        Y = build_targets(h5, ep, cabinet, closed)               # (L,6)
        if ep in test_eps:
            Xte.append(emb); Yte.append(Y)
        else:
            Xtr.append(emb); Ytr.append(Y)
    return (np.concatenate(Xtr), np.concatenate(Ytr),
            np.concatenate(Xte), np.concatenate(Yte))


def probe(Xtr, Ytr, Xte, Yte):
    """Same probe as p3b: standardize, ridge + MLP. Returns dict group->(lin,mlp) R²."""
    xs = StandardScaler().fit(Xtr)
    ys = StandardScaler().fit(Ytr)
    Xtr_s, Xte_s = xs.transform(Xtr), xs.transform(Xte)
    Ytr_s, Yte_s = ys.transform(Ytr), ys.transform(Yte)
    lin = ridge_fit_predict(Xtr_s, Ytr_s, Xte_s)
    mlp = mlp_fit_predict(Xtr_s, Ytr_s, Xte_s)
    out = {}
    for g, sl in GROUPS.items():
        out[g] = (float(r2_score(Yte_s[:, sl], lin[:, sl])),
                  float(r2_score(Yte_s[:, sl], mlp[:, sl])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=200)
    ap.add_argument("--n-ep", type=int, default=200)
    args = ap.parse_args()

    cabinet, closed = cabinet_mean_xpos()
    h5 = DrawerH5()
    rng = np.random.default_rng(3)                 # SAME split as p3b
    eps = rng.choice(h5.n_ep, size=min(args.n_ep, h5.n_ep), replace=False)
    n_test = max(1, int(0.2 * len(eps)))
    test_eps = set(eps[:n_test].tolist())
    print(f"\n===== Probe 3c: wrist-camera encoder ablation =====")
    print(f"  episodes={len(eps)}  test_eps={len(test_eps)}  (episode split, seed 3)")

    dual = load_jepa(args.epoch)
    print("encoding DUAL (agentview + wrist)...", flush=True)
    d = encode_all(dual, h5, eps, test_eps, True, cabinet, closed)
    del dual; torch.cuda.empty_cache()

    mono = load_mono()
    print("encoding MONO (agentview only)...", flush=True)
    m = encode_all(mono, h5, eps, test_eps, False, cabinet, closed)
    del mono; torch.cuda.empty_cache()

    print(f"  train frames={len(d[0])}  test frames={len(d[2])}")
    r_dual = probe(*d)
    r_mono = probe(*m)

    print("\n  decode R² (higher = more task info in emb).  ΔR² = dual − mono")
    print(f"  {'group':>11} | {'mono lin':>9} {'dual lin':>9} {'Δlin':>7} | "
          f"{'mono mlp':>9} {'dual mlp':>9} {'Δmlp':>7}")
    print("  " + "-" * 78)
    lines = []
    for g in GROUPS:
        ml, dl = r_mono[g][0], r_dual[g][0]
        mm, dm = r_mono[g][1], r_dual[g][1]
        line = (f"  {g:>11} | {ml:>9.3f} {dl:>9.3f} {dl-ml:>+7.3f} | "
                f"{mm:>9.3f} {dm:>9.3f} {dm-mm:>+7.3f}")
        print(line); lines.append(line)

    np.savez(os.path.join(RESULTS_DIR, "p3c_cam_ablation.npz"),
             mono={g: r_mono[g] for g in GROUPS},
             dual={g: r_dual[g] for g in GROUPS})
    with open(os.path.join(RESULTS_DIR, "p3c_cam_ablation.log"), "w") as f:
        f.write("group | mono_lin dual_lin dlin | mono_mlp dual_mlp dmlp\n")
        f.write("\n".join(lines) + "\n")
    print(f"\nsaved -> {os.path.join(RESULTS_DIR, 'p3c_cam_ablation.log')}")
    h5.close()


if __name__ == "__main__":
    main()
