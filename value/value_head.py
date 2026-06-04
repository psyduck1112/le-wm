"""Stage 2 step 2: the GoalCost value head C(emb_t, goal_emb) -> scalar >= 0.

C estimates the (discounted) number of steps still needed to drive the current
state emb_t to the goal goal_emb. It REPLACES the broken raw-distance planning
cost ||emb_t - goal_emb||^2. Both inputs are 192-D frozen visual embeddings;
the goal side is ALWAYS pure-visual (encode(goal image)) so cost stays symmetric.

Input is [emb_t, goal_emb, emb_t - goal_emb] (the diff gives an explicit
distance prior). Softplus output keeps C >= 0 and everywhere-differentiable,
which gives CEM a smooth cost landscape.

extra_dim is a reserved slot for an optional current-state proprio bypass
(fed only on the emb_t side, never the goal side); default 0 = pure visual.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GoalCost(nn.Module):
    def __init__(self, emb_dim: int = 192, hidden: int = 512, extra_dim: int = 0):
        super().__init__()
        self.emb_dim = emb_dim
        self.extra_dim = extra_dim
        in_dim = emb_dim * 3 + extra_dim     # [emb_t, goal_emb, emb_t-goal_emb] (+extra)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, emb_t, goal_emb, extra=None):
        """emb_t/goal_emb: (B, emb_dim) -> cost (B,) >= 0."""
        feats = [emb_t, goal_emb, emb_t - goal_emb]
        if self.extra_dim:
            assert extra is not None, "extra_dim>0 but no extra features passed"
            feats.append(extra)
        x = torch.cat(feats, dim=-1)
        return F.softplus(self.net(x)).squeeze(-1)
