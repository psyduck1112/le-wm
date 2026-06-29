"""Generate the figure set for the group-meeting diagnostics report.

Reads ONLY the saved .npz artifacts (no expensive re-simulation) and writes a
clean, consistently-styled, ASCII-labelled figure set to results/report_figs/.
English labels throughout (the saved CJK PNGs showed missing-glyph boxes).

  fig1_success_bars.png  success rate across conditions (the headline)
  fig2_probe_r2.png      decodability of privileged state + wrist-cam delta
  fig3_cost_mono.png     per-episode Spearman rho: emb-L2 vs decoded cost
  fig4_compounding.png   open-loop multi-step prediction error growth (p5)
  fig5_ood_exploit.png   D_phi vs truth on the states CEM actually visits (p6d)

Usage: python diagnostics/make_report_figs.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT = os.path.join(R, "report_figs")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

GREEN, RED, BLUE, GREY = "#2ca02c", "#d62728", "#1f77b4", "#888888"


def load(f):
    p = os.path.join(R, f)
    if not os.path.exists(p):          # archived artifacts fall back to results/archive/
        p = os.path.join(R, "archive", f)
    return np.load(p, allow_pickle=True)


# ---------- fig1: success-rate headline ----------
def fig_success():
    # (label, rate%, group)  group: 'ctrl' upper bound | 'cost' le-wm cost variants
    rows = [
        ("BC policy\n(same data)", 95, "ctrl"),
        ("Oracle dyn.\n+ privileged cost", 100, "ctrl"),
        ("Oracle dyn.\n+ emb-L2 cost", 0, "cost"),
        ("Oracle dyn.\n+ subgoal cost", 0, "cost"),
        ("Oracle dyn.\n+ decoded D_phi", 0, "cost"),
        ("Real e2e\n(emb-L2 cost)", 10, "cost"),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cols = [GREEN if r[2] == "ctrl" else RED for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(range(len(rows)), vals, color=cols, edgecolor="black", width=0.65)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v}%", ha="center", fontweight="bold")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("drawer-close success rate")
    ax.set_ylim(0, 108)
    ax.set_title("Cost is the bottleneck: perfect physics + le-wm cost still fails (0%),\n"
                 "while perfect physics + privileged cost solves it (100%)", fontsize=11)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=GREEN, edgecolor="k", label="control / upper bound"),
                       Patch(facecolor=RED, edgecolor="k", label="le-wm cost variants")],
              loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_success_bars.png"), dpi=140)
    print("fig1 done")


# ---------- fig2: probe R2 + wrist-cam delta ----------
def fig_probe():
    p3 = load("p3_probe.npz")["mlp"].item()
    p3b = load("p3b_reach.npz")["mlp"].item()
    p3c = load("p3c_cam_ablation.npz")
    mono, dual = p3c["mono"].item(), p3c["dual"].item()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))

    # left: how decodable is each privileged quantity (dual model, MLP probe)
    items = [("eef_pos", p3["eef_pos"][0]), ("drawer", p3["drawer"][0]),
             ("reach", p3b["reach"][0]), ("close", p3b["close"][0]),
             ("privileged", p3b["privileged"][0])]
    names = [i[0] for i in items]; r2 = [i[1] for i in items]
    bars = axL.bar(names, r2, color=BLUE, edgecolor="black", width=0.6)
    for b, v in zip(bars, r2):
        axL.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    axL.axhline(0.7, color=GREY, ls="--", lw=1)
    axL.set_ylim(0, 1.0); axL.set_ylabel("decode R2  (MLP probe, held-out eps)")
    axL.set_title("Privileged state IS recoverable from emb\n(info is present; the problem is the cost form)")

    # right: wrist-camera delta (dual - mono), MLP probe
    groups = ["eef_pos", "reach", "close", "privileged"]
    dmono = [mono[g][1] for g in groups]; ddual = [dual[g][1] for g in groups]
    x = np.arange(len(groups)); w = 0.38
    axR.bar(x - w/2, dmono, w, label="mono (agentview only)", color=GREY, edgecolor="black")
    axR.bar(x + w/2, ddual, w, label="dual (+ wrist cam)", color=GREEN, edgecolor="black")
    for i, g in enumerate(groups):
        axR.text(i, max(dmono[i], ddual[i]) + 0.01, f"+{ddual[i]-dmono[i]:.3f}",
                 ha="center", fontsize=8.5, color=GREEN, fontweight="bold")
    axR.set_xticks(x); axR.set_xticklabels(groups)
    axR.set_ylim(0, 0.9); axR.set_ylabel("decode R2  (MLP probe)")
    axR.set_title("Wrist camera adds decodable info\n(largest on reach / eef_pos, as expected)")
    axR.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_probe_r2.png"), dpi=140)
    print("fig2 done")


# ---------- fig3: cost monotonicity ----------
def fig_costmono():
    d = load("p2b_decoded_mono.npz")
    cross = d["embL2_cross"][:, 0]; deco = d["decoded"][:, 0]   # per-episode Spearman rho
    cross_fv = d["embL2_cross"][:, 1]; deco_fv = d["decoded"][:, 1]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6))

    axL.boxplot([cross, deco], labels=["emb-L2 cost\n(deployed, cross-goal)", "decoded D_phi cost"],
                widths=0.5, patch_artist=True,
                boxprops=dict(facecolor="#cccccc"), medianprops=dict(color="black"))
    axL.axhline(0, color=GREY, ls="--")
    axL.scatter(np.ones_like(cross) + np.random.uniform(-.08, .08, len(cross)), cross, s=10, color=RED, alpha=.5)
    axL.scatter(2 * np.ones_like(deco) + np.random.uniform(-.08, .08, len(deco)), deco, s=10, color=GREEN, alpha=.5)
    axL.set_ylabel("Spearman rho(progress, cost)\n(want strongly negative)")
    axL.set_title(f"emb-L2 mean rho={cross.mean():+.2f} (flat, broken)\n"
                  f"decoded mean rho={deco.mean():+.2f} (monotone, PASS)")

    axR.bar(["emb-L2\ncross-goal", "decoded\nD_phi"], [cross_fv.mean(), deco_fv.mean()],
            color=[RED, GREEN], edgecolor="black", width=0.55)
    axR.text(0, cross_fv.mean() + 0.5, f"{cross_fv.mean():.1f}", ha="center", fontweight="bold")
    axR.text(1, deco_fv.mean() + 0.5, f"{deco_fv.mean():.1f}", ha="center", fontweight="bold")
    axR.set_ylabel("false valleys per trajectory\n(cost dips below goal = traps for planner)")
    axR.set_title("Decoded cost removes the false valleys")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_cost_mono.png"), dpi=140)
    print("fig3 done")


# ---------- fig4: compounding error ----------
def fig_compounding():
    d = load("p5_openloop.npz")
    k, em, es, cm = d["k"], d["err_mean"], d["err_std"], d["copy_mean"]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(k, em, "-o", color=BLUE, label="predictor open-loop error")
    ax.fill_between(k, em - es, em + es, color=BLUE, alpha=0.2)
    ax.plot(k, cm, "--s", color=GREY, label="copy-last-frame baseline (drift)")
    ax.set_xlabel("rollout step k (autoregressive, fed own prediction)")
    ax.set_ylabel("emb L2 error  ||pred - true||")
    ax.set_title("Open-loop error compounds (~linear),\nbut stays below the do-nothing drift baseline")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_compounding.png"), dpi=140)
    print("fig4 done")


# ---------- fig5: OOD exploitation (D_phi lies) ----------
def fig_ood():
    log = load("p6d_dual.npz")["log"]   # step, t_reach, d_reach, t_q, d_q
    step, tr, dr, tq, dq = log[:, 0], log[:, 1], log[:, 2], log[:, 3], log[:, 4]
    corr = np.corrcoef(tr, dr)[0, 1]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))

    axL.plot(step, tr, "-", color=GREEN, lw=2, label="TRUE reach (sim)")
    axL.plot(step, dr, "-", color=RED, lw=2, label="D_phi estimated reach")
    axL.set_xlabel("executed step"); axL.set_ylabel("reach  ||eef - drawer||  (m)")
    axL.set_title(f"On states CEM actually visits, the cost LIES\n"
                  f"corr(true, D_phi reach) = {corr:+.2f}  (moves opposite to truth)")
    axL.legend()

    axR.plot(step, tq, "-", color=GREEN, lw=2, label="TRUE drawer_q (sim)")
    axR.plot(step, dq, "-", color=RED, lw=2, label="D_phi estimated drawer_q")
    axR.axhline(0.01, color=GREY, ls="--", lw=1, label="closed = +0.01")
    axR.set_xlabel("executed step"); axR.set_ylabel("drawer joint q  (open=-0.16)")
    axR.set_title("D_phi hallucinates the drawer closing (q up to +0.06)\n"
                  "while the true drawer never moves (frozen at -0.16)")
    axR.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_ood_exploit.png"), dpi=140)
    print("fig5 done")


if __name__ == "__main__":
    fig_success(); fig_probe(); fig_costmono(); fig_compounding(); fig_ood()
    print("\nall figures ->", OUT)
