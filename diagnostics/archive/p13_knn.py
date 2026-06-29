"""Decisive fork: is the reach hallucination DETECTABLE by a LOCAL metric (kNN)?

Stage 2 showed global-Mahalanobis pessimism fails (0% at all beta). Stage 1 said
why: maha is blind to the reach lie because off-expert-tube states sit in LOCAL
gaps between trajectory strands, invisible to one global ellipsoid. This probe
tests the geometry conjecture directly, offline & cheap (no new rollout):

  for each visited state from p11 (raw emb saved), compute kNN distance to the
  nearest BC-manifold frames (LOCAL density), and compare how well kNN vs maha
  tracks D_phi's reach error.

  corr(kNN, reach_err) >> corr(maha, reach_err)  => local metric sees the gap
      => a kNN trust-region is worth building (detector-side fix alive).
  corr(kNN, reach_err) ~ corr(maha, reach_err) ~ 0  => reach is NOT detectable
      => the decoder simply can't represent "arm far" (mean-collapse); must fix
         the DATA side (DAgger), restricting-to-in-dist cannot help reach.

Usage:
  python diagnostics/p13_knn.py --n-ref 80000 --k 10
"""
import argparse
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = os.path.join(HERE, "results")


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def both_corr(x, y):
    return np.corrcoef(x, y)[0, 1], spearman(x, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ref", type=int, default=80000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pca-dim", type=int, default=0,
                    help="0 = raw 192-d euclidean kNN; >0 = kNN in top-d PCA space")
    args = ap.parse_args()

    P = np.load(os.path.join(R, "p11_ood_probe.npz"))
    if "emb" not in P.files:
        raise SystemExit("p11_ood_probe.npz has no 'emb' -> rerun p11 (emb logging) first")
    Q = P["emb"].astype(np.float32)              # (M,192) visited states
    reach_err, q_err, cost_err = P["reach_err"], P["q_err"], P["cost_err"]
    maha = P["maha"]
    M = len(Q)

    C = np.load(os.path.join(ROOT, "value", "emb_cache.npz"))
    bc = C["emb"].astype(np.float32)             # (N,192) BC manifold
    rng = np.random.default_rng(0)
    ref = bc[rng.choice(len(bc), size=min(args.n_ref, len(bc)), replace=False)]
    print(f"queries M={M}  reference={len(ref)}  k={args.k}  pca_dim={args.pca_dim}")

    if args.pca_dim > 0:
        mu = ref.mean(0)
        _, _, Vt = np.linalg.svd(ref - mu, full_matrices=False)
        proj = Vt[:args.pca_dim].T               # (192,d)
        ref_t = (ref - mu) @ proj
        Q_t = (Q - mu) @ proj
    else:
        ref_t, Q_t = ref, Q

    # brute-force kNN distance (mean of k nearest), batched to bound memory
    knn = np.empty(M, dtype=np.float32)
    ref_sq = (ref_t ** 2).sum(1)                 # (Nref,)
    B = 64
    for i in range(0, M, B):
        q = Q_t[i:i + B]                          # (b,d)
        d2 = (q ** 2).sum(1, keepdims=True) + ref_sq[None] - 2.0 * q @ ref_t.T
        d2 = np.clip(d2, 0, None)
        part = np.partition(d2, args.k, axis=1)[:, :args.k]
        knn[i:i + B] = np.sqrt(part).mean(1)

    print("\n===== kNN vs maha : tracking D_phi error (pearson / spearman) =====")
    print(f"{'target':10s} | {'maha':>16s} | {'kNN':>16s}")
    for name, tgt in (("reach_err", reach_err), ("q_err", q_err), ("cost_err", cost_err)):
        pm, sm = both_corr(maha, tgt)
        pk, sk = both_corr(knn, tgt)
        print(f"{name:10s} | {pm:+.3f} / {sm:+.3f} | {pk:+.3f} / {sk:+.3f}")

    print("\n===== reach_err binned by kNN quintile (want monotone RISE) =====")
    qs = np.quantile(knn, np.linspace(0, 1, 6))
    for b in range(5):
        m = (knn >= qs[b]) & ((knn <= qs[b + 1]) if b == 4 else (knn < qs[b + 1]))
        if m.sum():
            print(f"  kNN[{qs[b]:6.2f},{qs[b+1]:6.2f})  n={m.sum():4d}  "
                  f"reach_err={reach_err[m].mean():.3f}  cost_err={cost_err[m].mean():.3f}")

    # separation: broken-reach (top-half reach_err) vs ok, by each score
    broken = reach_err > np.median(reach_err)
    print("\n===== separation of broken-reach states (reach_err > median) =====")
    for name, s in (("maha", maha), ("kNN", knn)):
        sep = (s[broken].mean() - s[~broken].mean()) / s.std()
        print(f"  {name:5s}: mean(broken)={s[broken].mean():.2f}  "
              f"mean(ok)={s[~broken].mean():.2f}  separation(z)={sep:+.2f}")

    np.savez(os.path.join(R, "p13_knn.npz"), knn=knn, maha=maha,
             reach_err=reach_err, q_err=q_err, cost_err=cost_err)
    print(f"\nsaved -> {os.path.join(R, 'p13_knn.npz')}")


if __name__ == "__main__":
    main()
