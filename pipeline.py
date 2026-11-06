import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')   # no display needed, just write the png
import matplotlib.pyplot as plt
import songwriter
import nightfever


DEV = 'cuda'


schedule = [{"type": "day", "n_tokens": 256},
            {"type": "night", "n_tokens": 128},
            {"type": "day", "n_tokens": 256},
            {"type": "night", "n_tokens": 128},
            {"type": "day", "n_tokens": 256},
            {"type": "night", "n_tokens": 128},
            {"type": "day", "n_tokens": 256},
            {"type": "night", "n_tokens": 128},
            {"type": "day", "n_tokens": 256},
            {"type": "night", "n_tokens": 128},
            {"type": "day", "n_tokens": 256},
            {"type": "night", "n_tokens": 128},
            {"type": "day", "n_tokens": 256}]


def build_training_epoch(schedule, n_dim, n_bands, n_songs, noise_std=0.5):
    """
    Stitch day/night blocks into one episode, following the schedule.
    Day blocks come from generate_discography (clean + noisy, across many worlds);
    night blocks are silence that doesn't count toward the loss.
    """

    firm = songwriter.DiscoFirm(n_dim, n_bands, noise_sigma=noise_std)
    batch_size  = n_bands * n_songs
    episode_len = sum([block['n_tokens'] for block in schedule])

    samples = torch.zeros(batch_size, episode_len, n_dim, device=DEV)
    targets = torch.zeros(batch_size, episode_len, n_dim, device=DEV)
    mask    = torch.zeros(batch_size, episode_len, n_dim, dtype=torch.bool, device=DEV)
    t0, t1  = 0, 0

    for block in schedule:

        t1 += block["n_tokens"]

        if block["type"] == "day":
            target, sample = firm.generate_sample(block['n_tokens'], n_songs)
            samples[:, t0:t1] = torch.from_numpy(sample)
            targets[:, t0:t1] = torch.from_numpy(target)
            mask[:, t0:t1]    = 1

        t0 = t1

    return samples, targets, mask


def compute_loss(output, target, mask, states_fast, states_slow, w_fast, w_slow):

    error_function = torch.nn.BCELoss()

    error = error_function(output[mask], target[mask])
    derivatives_fast = ((states_fast[:, 1:] - states_fast[:, :-1])**2).mean()
    derivatives_slow = ((states_slow[:, 1:] - states_slow[:, :-1])**2).mean()

    loss = error + w_fast * derivatives_fast + w_slow * derivatives_slow

    return loss


def store_model(model, model_name):

    os.makedirs('./models', exist_ok=True)
    path = f'./models/{model_name}.pt'

    save_dict = {
        "state_dict": model.state_dict(),
        "n_dim": model.n_dim,
        "sizes": (model.n_p, model.n_c, model.n_h),
        "n_msg": model.n_msg
    }

    torch.save(save_dict, path)



def plot_loss(loss_record, model_name):

    os.makedirs('./models', exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(loss_record)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title(model_name, fontsize=7)
    fig.tight_layout()
    fig.savefig(f'./models/{model_name}_loss.png', dpi=150)
    plt.close(fig)


def name_model(noise_std, w_derivs, lr, n_dim, n_bands, n_songs):

    def float_to_str(x):
        """Float to filename-safe string: 0.2 -> '0p2', 1e-05 -> '1em05'."""
        return f'{x:g}'.replace('.', 'p').replace('-', 'm')

    model_name = (f'noise-{float_to_str(noise_std)}_'
                  f'wfast-{float_to_str(w_derivs[0])}_'
                  f'wslow-{float_to_str(w_derivs[1])}_'
                  f'lr-{float_to_str(lr)}_'
                  f'dim-{n_dim:02d}_'
                  f'bands-{n_bands:02d}_'
                  f'songs-{n_songs:03d}')

    return model_name


# Training loop
def train(schedule, n_dim=6, n_bands=32, n_songs=1, n_epochs=100, lr=2e-2, noise_std=0.2, w_derivs=(1, 10)):

    model_name = name_model(noise_std, w_derivs, lr, n_dim, n_bands, n_songs)

    model     = nightfever.NightFever(n_dim, DEV)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    w_fast, w_slow = w_derivs
    loss_record = np.empty(n_epochs)

    for epoch in range(n_epochs):
        t_start = time.time()
        samples, targets, mask = build_training_epoch(schedule, n_dim, n_bands, n_songs, noise_std)
        output, states = model(samples)

        loss = compute_loss(output, targets, mask, states["h"], states["c"], w_fast, w_slow)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        loss_record[epoch] = loss.item()
        epoch_time = time.time() - t_start

        print(f"Epoch {epoch:4d} loss = {loss.item():.4f} ({epoch_time:.1f} s/epoch)")

        if epoch % 10 == 0:
            store_model(model, model_name)
            plot_loss(loss_record[:epoch + 1], model_name)

    store_model(model, model_name)
    plot_loss(loss_record, model_name)

    return model, loss_record


# Training sweep
def train_all_models(schedule, n_dim=6, n_bands=32, n_songs=1, n_epochs=100, lr=1e-2):

    noise_levels = [0.01, 0.2, 0.5] 
    w_fast_vals  = [0, 1, 5]
    w_slow_vals  = [0, 5, 20]

    for noise in noise_levels:
        for w_fast in w_fast_vals:
            for w_slow in w_slow_vals:
                print(f'\n\n### Training model ({noise}, ({w_fast}, ({w_slow}))) ###\n')
                train(schedule, n_dim, n_bands, n_songs, n_epochs, lr, noise, (w_fast, w_slow))
                print(f'\n')


if __name__ == "__main__":
    train(schedule)
