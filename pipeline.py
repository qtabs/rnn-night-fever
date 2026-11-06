import numpy as np
import torch
import songwriter
import nightfever


schedule = [{"type": "day", "n_tokens": 512},
            {"type": "night", "n_tokens": 256},
            {"type": "day", "n_tokens": 512},
            {"type": "night", "n_tokens": 256},
            {"type": "day", "n_tokens": 512},  
            {"type": "night", "n_tokens": 256},
            {"type": "day", "n_tokens": 512},
            {"type": "night", "n_tokens": 256},
            {"type": "day", "n_tokens": 512},          
            {"type": "night", "n_tokens": 256},
            {"type": "day", "n_tokens": 512}]


def build_epoch_training_set(schedule, n_dim, n_bands, n_songs, noise_std=0.5):
    """
    Stitch day/night blocks into one episode, following the schedule.
    Day blocks come from generate_discography (clean + noisy, across many worlds);
    night blocks are silence that doesn't count toward the loss.
    """

    firm = songwriter.DiscoFirm(n_dim, n_bands)
    batch_size  = n_bands * n_songs
    episode_len = sum([block['n_tokens'] for block in schedule])

    samples = torch.zeros(batch_size, episode_len, n_dim)
    targets = torch.zeros(batch_size, episode_len, n_dim)
    mask    = torch.zeros(batch_size, episode_len, n_dim, dtype=torch.bool)
    t0, t1  = 0, 0
    
    for block in schedule:
        
        t1 += block["n_tokens"]

        if block["type"] == "day":
            sample, target = firm.generate_sample(block['n_tokens'], n_songs)
            samples[:, t0:t1] = sample
            targets[:, t0:t1] = target
            mask[:, t0:t1]    = 1

    return samples, targets, mask


def compute_loss(output, target, mask, states_fast, states_slow, w_fast, w_slow):

    error_function = torch.nn.BCELoss()

    error = error_function(output, target, mask)
    derivatives_fast = ((states_fast[:, 1:] - states_fast[:, :-1])**2).mean()
    derivatives_slow = ((states_slow[:, 1:] - states_slow[:, :-1])**2).mean()

    loss = error + w_fast * derivatives_fast + w_slow * derivatives_slow

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
