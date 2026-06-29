"""Stage 0 of the D_phi trust-region fix (LIBERO_DIAGNOSTICS Phase 0 prep):

Build and characterise a CHEAP, sim-free OOD detector on the BC embedding
manifold (value/emb_cache.npz, 367841 x 192). This is the *tool* the later
stages need; it does NOT yet test the fix.

Why an OOD detector at all: D_phi (the decode-then-cost head) is accurate ON the
BC manifold but CEM is an optimiser that walks OFF it into regions where D_phi
extrapolates wrongly and reports a fake-low cost (reward hacking; p6d:
corr(true_reach, D_phi_reach) = -0.557). If we can SCORE how far an embedding is
from the BC manifold, we can later penalise the planner for going there.

Two scores, both fit from BC embeddings only:
  maha_k(e) = sum_{i<=k} (v_i^T (e-mu))^2 / lambda_i   # whitened dist in the
              reliable top-k PCA subspace (drops tiny tail eigvals that would
              blow up a full Mahalanobis -- p0 showed lambda_191 ~ 7e-6).
  resid(e)  = || (e-mu) - sum_{i<=k} v_i (v_i^T (e-mu)) ||   # energy OUTSIDE
              the top-k subspace (off-manifold components).

Validation here (no simulator yet):
  1. train vs held-out BC score distributions should MATCH (detector must not
     flag in-distribution data) -> gives the 99th-pct in-dist threshold.
  2. smoke test: adding Gaussian noise to held-out embeddings must raise both
     scores monotonically -> the detector actually responds to leaving the
     manifold. (Rendered-OOD is Stage 1.)

Fitted params are saved to value/ood_detector.npz for Stage 1/2 reuse.

Usage:
  python diagnostics/p10_ood_detector.py --k auto
"""
import argparse
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EMB_CACHE = os.path.join(ROOT, "value", "emb_cache.npz")
RESULTS_DIR = os.path.join(HERE, "results")
DETECTOR_OUT = os.path.join(ROOT, "value", "ood_detector.npz")


def fit_pca(X, shrink=1e-6):
    """X: (N,D) BC embeddings. Returns mu, eigvecs V (D,D, descending), eigvals."""
    mu = X.mean(0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / (len(Xc) - 1)
    cov[np.diag_indices_from(cov)] += shrink     # numerical floor on tail
    evals, evecs = np.linalg.eigh(cov)           # ascending
    evals = evals[::-1]                          # descending
    evecs = evecs[:, ::-1]
    evals = np.clip(evals, shrink, None)
    return mu.astype(np.float32), evecs.astype(np.float32), evals.astype(np.float32)


def choose_k(evals, cum_var=0.99):
    """smallest k whose top-k captures `cum_var` of total variance."""
    frac = np.cumsum(evals) / evals.sum()
    return int(np.searchsorted(frac, cum_var) + 1)


def scores(X, mu, V, evals, k):
    """maha_k and residual for rows of X. X:(N,D). Returns (maha_k, resid)."""
    Xc = X - mu                       # (N,D)
    proj = Xc @ V[:, :k]              # (N,k) coords in top-k subspace
    maha_k = ((proj ** 2) / evals[:k]).sum(1)        # (N,) whitened dist^2
    recon = proj @ V[:, :k].T         # (N,D) reconstruction from top-k
    resid = np.linalg.norm(Xc - recon, axis=1)       # (N,) off-subspace energy
    return maha_k, resid


def pct(a, qs=(50, 90, 99, 99.9, 100)):
    return {q: float(np.percentile(a, q)) for q in qs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", default="auto",
                    help="'auto' (99% cum-var) or an int subspace dim")
    ap.add_argument("--cum-var", type=float, default=0.99)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    C = np.load(EMB_CACHE)
    emb = C["emb"].astype(np.float32)            # (N,192)
    ep_offset = C["ep_offset"]
    ep_len = C["ep_len"]
    n_ep = len(ep_offset)
    D = emb.shape[1]

    # ---- episode-level split (no within-episode leakage) ----
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_ep)
    n_train = int(args.train_frac * n_ep)
    train_eps, held_eps = perm[:n_train], perm[n_train:]

    def gather(eps):
        idx = np.concatenate([np.arange(ep_offset[e], ep_offset[e] + ep_len[e])
                              for e in eps])
        return emb[idx]

    X_tr = gather(train_eps)
    X_ho = gather(held_eps)
    print(f"emb {emb.shape}  episodes {n_ep} -> train {len(train_eps)} "
          f"({len(X_tr)} frames) / held-out {len(held_eps)} ({len(X_ho)} frames)")

    # ---- fit detector on TRAIN only ----
    mu, V, evals = fit_pca(X_tr)
    k = choose_k(evals, args.cum_var) if args.k == "auto" else int(args.k)
    eff_rank = float(np.exp(-((evals / evals.sum()) *
                              np.log(evals / evals.sum() + 1e-12)).sum()))
    print(f"\nfit on train: eff_rank={eff_rank:.1f}/{D}  "
          f"k={k} (cum-var {np.cumsum(evals)[k-1]/evals.sum():.3f})  "
          f"lambda[0]={evals[0]:.3e} lambda[k-1]={evals[k-1]:.3e} "
          f"lambda[-1]={evals[-1]:.3e}")

    # ---- 1. train vs held-out in-distribution score distributions ----
    m_tr, r_tr = scores(X_tr, mu, V, evals, k)
    m_ho, r_ho = scores(X_ho, mu, V, evals, k)
    print("\n===== 1. in-distribution scores (train vs held-out) =====")
    print("  maha_k  percentiles {50,90,99,99.9,100}:")
    print(f"    train    : " + "  ".join(f"{v:8.1f}" for v in pct(m_tr).values()))
    print(f"    held-out : " + "  ".join(f"{v:8.1f}" for v in pct(m_ho).values()))
    print("  resid   percentiles {50,90,99,99.9,100}:")
    print(f"    train    : " + "  ".join(f"{v:8.3f}" for v in pct(r_tr).values()))
    print(f"    held-out : " + "  ".join(f"{v:8.3f}" for v in pct(r_ho).values()))
    thr_maha = float(np.percentile(m_tr, 99))
    thr_resid = float(np.percentile(r_tr, 99))
    ho_flag_maha = float((m_ho > thr_maha).mean())
    ho_flag_resid = float((r_ho > thr_resid).mean())
    print(f"  99th-pct train thresholds: maha_k={thr_maha:.1f}  resid={thr_resid:.3f}")
    print(f"  held-out false-positive rate @ those thr: "
          f"maha_k={ho_flag_maha:.1%}  resid={ho_flag_resid:.1%}  (want ~1%)")

    # ---- 2. smoke test: Gaussian-noise OOD must raise scores monotonically ----
    print("\n===== 2. smoke test: noise-perturbed held-out (sim-free OOD proxy) =====")
    per_dim_std = X_tr.std(0)                     # (D,)
    sub = X_ho[rng.choice(len(X_ho), size=min(20000, len(X_ho)), replace=False)]
    print("  sigma(x per-dim std) | median maha_k | median resid | %>maha-thr")
    for sig in (0.0, 0.5, 1.0, 2.0, 4.0):
        noisy = sub + sig * per_dim_std * rng.standard_normal(sub.shape).astype(np.float32)
        mm, rr = scores(noisy, mu, V, evals, k)
        print(f"    {sig:4.1f}  |  {np.median(mm):11.1f}  |  {np.median(rr):10.3f}"
              f"  |  {(mm > thr_maha).mean():6.1%}")

    # ---- save fitted detector for Stage 1/2 ----
    np.savez(DETECTOR_OUT, mu=mu, V=V, evals=evals, k=np.int64(k),
             thr_maha=thr_maha, thr_resid=thr_resid,
             per_dim_std=per_dim_std, train_eps=train_eps, held_eps=held_eps)
    print(f"\nsaved detector -> {DETECTOR_OUT}")
    np.savez(os.path.join(RESULTS_DIR, "p10_ood_detector.npz"),
             maha_train=m_tr, maha_ho=m_ho, resid_train=r_tr, resid_ho=r_ho,
             evals=evals, k=k, thr_maha=thr_maha, thr_resid=thr_resid)
    print(f"saved -> {os.path.join(RESULTS_DIR, 'p10_ood_detector.npz')}")


if __name__ == "__main__":
    main()
