"""Team 2: the perception (P), hippocampus (H) and cortex (C) networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def corrupt(tokens, sleep, n_tokens, alphas, sigma):
    """Noisy P input: alpha * onehot + sigma * gaussian, alpha drawn per step.

    alpha = 0 means a fully missing input. During sleep the input is all zeros.
    TODO(team2): noise model.
    """
    alphas = torch.as_tensor(alphas, dtype=torch.float, device=tokens.device)
    alpha = alphas[torch.randint(len(alphas), tokens.shape, device=tokens.device)]
    x = alpha[..., None] * F.one_hot(tokens, n_tokens).float()
    x = x + sigma * torch.randn_like(x)
    return x.masked_fill(sleep[..., None], 0.0)


def slowness_penalty(mu, logvar, mu_prev, logvar_prev):
    """KL( N(mu, var) || N(mu_prev, var_prev) ) per batch element, summed over z.

    TODO(team2): slowness penalty.
    """
    return 0.5 * (logvar_prev - logvar
                  + (logvar.exp() + (mu - mu_prev) ** 2) / logvar_prev.exp()
                  - 1).sum(-1)


class NightFever(nn.Module):
    """P, H and C GRUs, all connected to each other and all given the sleep flag.

    At step t every module sees the *previous-step* states of the other two
    (C is seen through its sampled latent z), its own input and the sleep flag.
    """

    def __init__(self, n_tokens, p_size=64, h_size=32, c_size=256, z_size=32,
                 alphas=(0.0, 0.5, 1.0, 2.0), sigma=1.0):
        super().__init__()
        self.n_tokens, self.alphas, self.sigma = n_tokens, alphas, sigma
        self.sizes = dict(P=p_size, H=h_size, C=c_size, z=z_size)
        self.P = nn.GRUCell(n_tokens + h_size + z_size + 1, p_size)
        self.H = nn.GRUCell(p_size + z_size + 1, h_size)
        self.C = nn.GRUCell(p_size + h_size + 1, c_size)
        self.to_z = nn.Linear(c_size, 2 * z_size)
        self.readout = nn.Linear(p_size, n_tokens)

    def forward(self, batch, return_states=False, x=None):
        """Returns logits (B, T, n_tokens), penalty (B, T) and, if return_states,
        a dict of P, H, C (hidden) and z trajectories, each (B, T, size).
        `x` overrides the noisy P input (default: `corrupt` applied to the batch)."""
        tokens, sleep = batch["tokens"], batch["sleep"]
        B, T = tokens.shape
        if x is None:
            x = corrupt(tokens, sleep, self.n_tokens, self.alphas, self.sigma)
        s = sleep.float()[..., None]
        p, h, c, z = (tokens.new_zeros(B, self.sizes[k], dtype=torch.float)
                      for k in ("P", "H", "C", "z"))
        mu_prev = logvar_prev = None
        logits, penalty, states = [], [], {k: [] for k in ("P", "H", "C", "z")}
        for t in range(T):
            # TODO(team2/3): lesions (e.g. cut H->C during sleep) go here.
            p_new = self.P(torch.cat([x[:, t], h, z, s[:, t]], -1), p)
            h_new = self.H(torch.cat([p, z, s[:, t]], -1), h)
            c_new = self.C(torch.cat([p, h, s[:, t]], -1), c)
            p, h, c = p_new, h_new, c_new

            mu, logvar = self.to_z(c).chunk(2, -1)
            z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
            # TODO(team2): no penalty at t=0 (there is no previous latent yet).
            penalty.append(torch.zeros(B, device=mu.device) if mu_prev is None
                           else slowness_penalty(mu, logvar, mu_prev, logvar_prev))
            mu_prev, logvar_prev = mu, logvar

            logits.append(self.readout(p))
            if return_states:
                for k, v in zip(states, (p, h, c, z)):
                    states[k].append(v)

        out = (torch.stack(logits, 1), torch.stack(penalty, 1))
        if return_states:
            out += ({k: torch.stack(v, 1) for k, v in states.items()},)
        return out
