"""Figure for Stage 1 OOD probe (reads results/p11_ood_probe.npz)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
A = np.load(os.path.join(R, "p11_ood_probe.npz"))
det = np.load(os.path.join(os.path.dirname(HERE), "value", "ood_detector.npz"))
thr = float(det["thr_maha"])
inl = A["maha"] <= thr

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

# (1) maha vs cost_signed: as OOD grows, D_phi under-reports cost (goes negative)
ax[0].scatter(A["maha"], A["cost_signed"], s=10, alpha=.5)
ax[0].axhline(0, color="k", lw=.8); ax[0].axvline(thr, color="r", ls="--", lw=.8, label="in-dist thr")
ax[0].set_xlabel("maha_k (OOD score)"); ax[0].set_ylabel("D_phi cost - true cost")
ax[0].set_title("under-report grows with OOD\n(neg = fake-low valley CEM exploits)")
ax[0].legend()

# (2) binned error vs OOD
qs = np.quantile(A["maha"], np.linspace(0, 1, 6))
ctr, ce, qe = [], [], []
for b in range(5):
    m = (A["maha"] >= qs[b]) & ((A["maha"] <= qs[b+1]) if b == 4 else (A["maha"] < qs[b+1]))
    ctr.append(.5*(qs[b]+qs[b+1])); ce.append(A["cost_err"][m].mean()); qe.append(30*A["q_err"][m].mean())
ax[1].plot(ctr, ce, "o-", label="cost_err")
ax[1].plot(ctr, qe, "s-", label="30 x q_err (close term)")
ax[1].set_xlabel("maha_k bin centre"); ax[1].set_ylabel("mean D_phi error")
ax[1].set_title("D_phi error rises with OOD"); ax[1].legend()

# (3) reward-hacking signature, split by in-dist/OOD
ax[2].scatter(A["true_reach"][inl], A["dphi_reach"][inl], s=12, alpha=.6,
              label=f"in-dist (r={np.corrcoef(A['true_reach'][inl],A['dphi_reach'][inl])[0,1]:+.2f})")
ax[2].scatter(A["true_reach"][~inl], A["dphi_reach"][~inl], s=12, alpha=.6,
              label=f"OOD (r={np.corrcoef(A['true_reach'][~inl],A['dphi_reach'][~inl])[0,1]:+.2f})")
lim = [min(A["true_reach"].min(), A["dphi_reach"].min()), max(A["true_reach"].max(), A["dphi_reach"].max())]
ax[2].plot(lim, lim, "k--", lw=.8)
ax[2].set_xlabel("true reach"); ax[2].set_ylabel("D_phi reach")
ax[2].set_title("reach: D_phi tracks truth in-dist,\nreverses in OOD"); ax[2].legend()

fig.tight_layout()
out = os.path.join(R, "p11_ood_probe.png")
fig.savefig(out, dpi=110)
print(f"saved -> {out}")
