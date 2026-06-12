"""Collapse check for the le-wm dual-cam encoder, run on the cached embeddings
(value/emb_cache.npz, 367841 x 192). No model load needed.

Four standard JEPA-collapse diagnostics:
  1. per-dim std  -> dead dimensions (dimensional collapse)
  2. covariance eigen-spectrum + effective rank / participation ratio
  3. pairwise distance distribution (complete collapse -> all ~0)
  4. global scatter (mean per-dim std vs mean sample norm)

Reference points printed at the end so the numbers are interpretable.
"""
import numpy as np

C = np.load("/home/yikang/git/le-wm/value/emb_cache.npz")
emb = C["emb"].astype(np.float32)          # (N, 192)
N, D = emb.shape
print(f"emb: {emb.shape}  dtype loaded float16 -> float32")

# ---- 1. per-dim std ----
mu = emb.mean(0)
std = emb.std(0)
dead = (std < 1e-3).sum()
near_dead = (std < 1e-2).sum()
print("\n[1] per-dim std")
print(f"    std  min={std.min():.4f}  median={np.median(std):.4f}  max={std.max():.4f}")
print(f"    dead dims (std<1e-3): {dead}/{D}   near-dead (std<1e-2): {near_dead}/{D}")

# ---- 2. covariance spectrum / effective rank ----
Xc = emb - mu
# sample a subset for the covariance to keep it cheap & well-conditioned
cov = (Xc.T @ Xc) / (N - 1)                # (192,192)
evals = np.linalg.eigvalsh(cov)
evals = np.clip(evals[::-1], 0, None)      # descending
p = evals / evals.sum()
# participation ratio: (sum lambda)^2 / sum(lambda^2)
pr = (evals.sum() ** 2) / (np.square(evals).sum())
# effective rank: exp(entropy of normalized spectrum)
ent = -(p[p > 0] * np.log(p[p > 0])).sum()
eff_rank = np.exp(ent)
top1 = p[0]
top10 = p[:10].sum()
print("\n[2] covariance spectrum (D=192)")
print(f"    participation ratio = {pr:.1f} / {D}")
print(f"    effective rank      = {eff_rank:.1f} / {D}")
print(f"    top-1 eigval share  = {top1:.1%}   top-10 share = {top10:.1%}")

# ---- 3. pairwise distances (subsample) ----
rng = np.random.default_rng(0)
idx = rng.choice(N, size=4000, replace=False)
S = emb[idx]
# random pairs
a = S[rng.permutation(len(S))]
b = S[rng.permutation(len(S))]
pd = np.linalg.norm(a - b, axis=1)
norms = np.linalg.norm(S, axis=1)
print("\n[3] pairwise L2 distance (4000 random pairs)")
print(f"    dist  mean={pd.mean():.3f}  std={pd.std():.3f}  min={pd.min():.3f}")
print(f"    sample norm mean={norms.mean():.3f}")
print(f"    dist/norm ratio = {pd.mean()/norms.mean():.3f}  (->0 means collapsed to a point)")

# ---- 4. verdict ----
print("\n[verdict]")
healthy = (dead == 0) and (eff_rank > D * 0.25) and (pd.mean()/norms.mean() > 0.3)
if healthy:
    print("    NO collapse: dims alive, spectrum spread, samples well separated.")
else:
    print("    POSSIBLE collapse — inspect the failing metric above.")
print("\nref: full collapse -> eff_rank~1, top1~100%, dist/norm~0.")
print("     healthy JEPA   -> eff_rank a sizable fraction of D, no dead dims.")
