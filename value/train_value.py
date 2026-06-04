"""Stage 2 step 3: train the GoalCost value head C with offline TD on cached embs.

C(emb_t, goal) learns the discounted #steps still needed to reach the goal:
    C(s,s)            = 0                                  (diagonal anchor, pins zero)
    C(s_t, g)         = 1 + gamma * C_bar(s_{t+1}, g)      (TD; bootstrap=0 if s_{t+1}==g)
loss = SmoothL1(C, target).

Key isolation: embeddings come from the FROZEN encoder cache (value/emb_cache.npz)
and are never differentiated -> the TD gradient cannot reach the encoder, so
le-wm's two-loss core (next-emb MSE + SIGReg) is untouched. Goals are pure-visual
future frames (hindsight relabeling) so cost stays symmetric.

A C_bar EMA target network stabilizes the bootstrap; the first warmup_mc steps use
a Monte-Carlo target (1-gamma^k)/(1-gamma) to stabilize the early bootstrap.

Usage:
  CUDA_VISIBLE_DEVICES=0 python value/train_value.py
"""
import argparse
import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from value_head import GoalCost

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "emb_cache.npz")
OUT = os.path.join(HERE, "value_head.pt")


def load_cache(device):
    d = np.load(CACHE)
    emb = torch.tensor(d["emb"].astype(np.float32), device=device)   # (N,192)
    off = d["ep_offset"].astype(np.int64)
    length = d["ep_len"].astype(np.int64)
    N = emb.shape[0]
    # per-frame: last global index of its episode (ep_end), and whether it can be an anchor
    ep_end = np.empty(N, dtype=np.int64)
    for o, l in zip(off, length):
        ep_end[o:o + l] = o + l - 1
    ep_end_t = torch.tensor(ep_end, device=device)
    anchors = torch.tensor(np.where(np.arange(N) < ep_end)[0], device=device)  # not last-in-ep
    return emb, ep_end_t, anchors


def sample_batch(emb, ep_end, anchors, B, gamma, p_diag, rng):
    """Returns emb_t, emb_tp1, goal, target_kind tensors for one TD batch."""
    idx = anchors[torch.randint(len(anchors), (B,), device=emb.device, generator=rng)]
    tp1 = idx + 1
    end = ep_end[idx]
    span = (end - idx).clamp(min=1)                       # #future frames available (>=1)
    off = (torch.rand(B, device=emb.device, generator=rng) * span.float()).long() + 1
    j = (idx + off).clamp(max=end)                        # future goal index in [tp1, end]
    diag = torch.rand(B, device=emb.device, generator=rng) < p_diag
    j = torch.where(diag, idx, j)                         # diagonal: goal = current frame
    reached = (j == tp1)
    steps = (j - idx).float()                             # #steps from anchor to goal (>=0)
    return emb[idx], emb[tp1], emb[j], diag, reached, steps


@torch.no_grad()
def monotonicity_check(model, emb, off, length, gamma, n_ep=200, device="cuda"):
    """Built-in cheap probe-2: along demos with goal=last frame, is C monotone?"""
    rng = np.random.default_rng(0)
    eps = rng.choice(len(off), size=min(n_ep, len(off)), replace=False)
    rhos, fvs = [], []
    for e in eps:
        o, l = int(off[e]), int(length[e])
        traj = emb[o:o + l]
        goal = traj[-1:].expand(l, -1)
        c = model(traj, goal).cpu().numpy()
        rho, _ = spearmanr(np.linspace(0, 1, l), c)
        rhos.append(rho)
        fvs.append(int((c < c[-1]).sum()))
    return float(np.nanmean(rhos)), float(np.mean(fvs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--warmup-mc", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--tau", type=float, default=0.995)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--p-diag", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed)
    rng = torch.Generator(device=device).manual_seed(args.seed)

    emb, ep_end, anchors = load_cache(device)
    d = np.load(CACHE)
    off, length = d["ep_offset"], d["ep_len"]
    print(f"loaded {emb.shape[0]} frames, {len(off)} episodes, {len(anchors)} anchors")

    model = GoalCost(emb_dim=emb.shape[1]).to(device)
    target_net = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    g = args.gamma

    for step in range(1, args.steps + 1):
        emb_t, emb_tp1, goal, diag, reached, steps = sample_batch(
            emb, ep_end, anchors, args.batch, g, args.p_diag, rng)

        with torch.no_grad():
            if step <= args.warmup_mc:
                boot_target = (1 - g ** steps) / (1 - g)          # MC cost-to-go
            else:
                boot = target_net(emb_tp1, goal)                  # C_bar(s', g)
                boot = torch.where(reached, torch.zeros_like(boot), boot)  # C(g,g)=0
                boot_target = 1.0 + g * boot
            target = torch.where(diag, torch.zeros_like(boot_target), boot_target)

        pred = model(emb_t, goal)
        loss = F.smooth_l1_loss(pred, target)
        opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():                                     # EMA target update
            for tp, p in zip(target_net.parameters(), model.parameters()):
                tp.mul_(args.tau).add_(p, alpha=1 - args.tau)

        if step % 2000 == 0 or step == 1:
            rho, fv = monotonicity_check(model, emb, off, length, g, n_ep=100)
            print(f"  step {step:6d}  loss={loss.item():.4f}  "
                  f"pred[min/mean/max]={pred.min():.2f}/{pred.mean():.2f}/{pred.max():.2f}  "
                  f"mono(rho={rho:+.3f} fv={fv:.1f})", flush=True)

    torch.save({"state_dict": model.state_dict(),
                "emb_dim": emb.shape[1], "gamma": g}, OUT)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
