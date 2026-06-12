"""D_phi (StateDecoder): the decode-then-cost head for the drawer MPC fix.

A small MLP on TOP of the FROZEN epoch-200 dual-camera encoder that reads the
192-d embedding and decodes the two quantities p6's privileged_cost is built from:
    emb (192) -> (eef_pos(3), drawer_qpos(1))   in real (metre / joint) units.

`decoded_cost` then reassembles p6's exact shaping (reach + 30*close) FROM the
decoded state, so the MPC cost no longer relies on raw goal-image emb-L2 (which
p2 showed is non-monotone). p3b already proved both quantities are decodable
(eef R2~0.77, drawer R2~0.75); this just packages that as a reusable head.

Self-normalizing: input/output mean+std are stored as buffers, so a loaded
checkpoint needs no external scaler.
"""
import numpy as np
import torch
import torch.nn as nn

# fixed in the world (verified in p3b: cabinet xpos std <=1 cm across init_states)
CABINET = np.array([-0.0019, -0.1502, 0.905], dtype=np.float32)
CLOSED = 0.01   # drawer joint value when shut (p6 find_drawer_joint)


class StateDecoder(nn.Module):
    def __init__(self, emb_dim: int = 192, hidden: int = 256, out_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim))
        self.register_buffer("in_mean", torch.zeros(emb_dim))
        self.register_buffer("in_std", torch.ones(emb_dim))
        self.register_buffer("out_mean", torch.zeros(out_dim))
        self.register_buffer("out_std", torch.ones(out_dim))

    def set_norm(self, in_mean, in_std, out_mean, out_std):
        self.in_mean.copy_(torch.as_tensor(in_mean, dtype=torch.float32))
        self.in_std.copy_(torch.as_tensor(in_std, dtype=torch.float32).clamp_min(1e-6))
        self.out_mean.copy_(torch.as_tensor(out_mean, dtype=torch.float32))
        self.out_std.copy_(torch.as_tensor(out_std, dtype=torch.float32).clamp_min(1e-6))

    def raw(self, emb):
        """(...,192) real emb -> (...,4) real-unit [eef(3), q(1)]."""
        z = (emb - self.in_mean) / self.in_std
        return self.net(z) * self.out_std + self.out_mean

    def forward(self, emb):
        out = self.raw(emb)
        return out[..., :3], out[..., 3:4]   # eef_pos, drawer_qpos


@torch.no_grad()
def decoded_cost(emb, decoder, target_q: float = CLOSED, w_close: float = 30.0):
    """p6's shaped cost, but read from the decoded state instead of the sim.
    emb: (N,192) torch on decoder's device. Returns (N,) cost."""
    cab = torch.as_tensor(CABINET, device=emb.device, dtype=emb.dtype)
    eef, q = decoder(emb)
    reach = torch.linalg.norm(eef - cab, dim=-1)        # (N,)
    close = (q[..., 0] - target_q).abs()                # (N,)
    return reach + w_close * close


def load_decoder(path: str, device: str = "cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = StateDecoder(emb_dim=ck.get("emb_dim", 192), hidden=ck.get("hidden", 256))
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval().requires_grad_(False)
