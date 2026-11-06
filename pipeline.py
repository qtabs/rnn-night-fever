# This pipeline will do the following:
   #  1. Takes a schedule list and builds one episode by stitching day/night blocks together,
   #  2. Build the reconstruction only for now,add slowness later.
   #  3. Trains the bare RNN over it, printing the metrics we will interpret.

# I will build the reconstruction for now, then we can improve the pipeline with slowneess later.


import numpy as np
import torch
import torch.nn.functional as F


from songwriter import DiscoFirm  # many worlds in one batch; gives clean + noisy
from nightfever import RNN        # the bare RNN, we will swap with the circuit later

# Alejandro's request

schedule = [{"type": "day", "n_tokens": 1024},
            {"type": "night", "n_tokens": 512},
            {"type": "day", "n_tokens": 1024},
            {"type": "night", "n_tokens": 512}]

# schedule is Alejandro's 1024/512 (slow). For a quick test, shrink n_tokens (e.g. 40/20).
# n_tokens = how long that block is. The batch size comes from n_bands * n_songs (below).

## build_episode

# Turn the schedule into one episode (First addition)

def build_episode(schedule, n_dim, n_bands, n_songs, noise_std=0.5):
    """Stitch day/night blocks into one episode, following the schedule.
    Day blocks come from generate_discography (clean + noisy, across many worlds);
    night blocks are silence that doesn't count toward the loss."""

    firm  = DiscoFirm(n_dim, n_songs, n_bands)  # the worlds for this episode
    batch = n_bands * n_songs                              # total songs in the batch
    inputs, targets, masks = [], [], []
    for block in schedule:
        L = block["n_tokens"]  # length of this block
        if block["type"] == "day":
            clean, noisy = firm.generate_discography(L, n_songs, noise_std=noise_std)  # noise from generator
            clean = torch.tensor(clean); noisy = torch.tensor(noisy)
            mask  = torch.ones(batch, L, dtype=torch.bool)   # day steps count
        else:
            clean = torch.zeros(batch, L, n_dim)             # no target
            noisy = torch.zeros(batch, L, n_dim)             # no input
            mask  = torch.zeros(batch, L, dtype=torch.bool)  # night steps ignored
        inputs.append(noisy); targets.append(clean); masks.append(mask)
    return torch.cat(inputs, 1), torch.cat(targets, 1), torch.cat(masks, 1)


# Addition 2 (separate, customizable loss)

### SLOWNESS, NEEDS CHECKING


# The formula form Alex's board
# Gaussian mean, so KL(N(h^t,I) || N(h^{t+1},I)) = 0.5 * sum((h^{t+1} - h^t)^2).

def kl_slowness(state):
    delta = state[:, 1:] - state[:, :-1] #  change from each token to the next -> (B, T-1, H)
    # state 1: every token from the 2nd onward, state 2: every token except the last.
    kl = 0.5 * (delta ** 2).sum(-1) # # (B, T-1): the KL (slowness cost) at each step
    # how big was the change
    return kl.mean()

# big = the state is fast, small = slow

def compute_loss(output, target, mask, states=None, alpha_fast=0.0, alpha_slow=0.0):
    """Reconstruction (BCE) + optional slowness penalties on the circuit's states.
    alpha_fast slows the hippocampus; alpha_slow slows the cortex/PFC (set it larger)."""

    bce   = F.binary_cross_entropy(output, target, reduction="none").mean(-1)  # (B, T)
    recon = bce[mask].mean()                     # reconstruction on day steps
    loss  = recon

    if states is not None:                       # the circuit passes its state trajectories
        if "fast" in states:                     # hippocampus
            loss = loss + alpha_fast * kl_slowness(states["fast"])
        if "slow" in states:                     # cortex / PFC
            loss = loss + alpha_slow * kl_slowness(states["slow"])

    return loss

# Training loop
def train(schedule, n_dim=10, n_bands=4, n_songs=8, n_batches=301, lr=1e-2, n_hidden=32, noise_std=0.5):
    net = RNN(n_dim, n_layers=1, n_hidden=n_hidden, dev="cpu")  # the bare RNN
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for step in range(n_batches):  # n_batches = number of training iterations (matches train_network)
        x, y, mask = build_episode(schedule, n_dim, n_bands, n_songs, noise_std=noise_std)  # new worlds each episode
        out  = net(x)                       # feed it the noisy episode, get back its denoised guess
        loss = compute_loss(out, y, mask)   # how far OUT is from clean y; mask counts day steps only
        loss.backward()                     # work out which way to nudge each weight
        opt.step()                          # take that nudge
        opt.zero_grad() 
        if step % 50 == 0:
            print(f"step {step:3d} recon {loss.item():.4f}")

    return net

if __name__ == "__main__":
    train(schedule)


# Notes on seed=step
# Because every episode is a different world, this number isn’t
# “how well it memorized one world.” It’s “how well the network
# can perceive a world it has never seen before.” As training goes
# on, the weights aren’t memorizing any particular grammar
# they’re learning a general strategy for quickly cleaning
# up noise in whatever world shows up. That’s the meta-learning
# It is adapting to a new world each time is harder than mastering one.
