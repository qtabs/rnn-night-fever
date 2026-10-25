"""Team 1: generative models (GMs) and the episode builder.

`sample_episode` is the interface shared by all teams. Its signature and the
four keys it returns are frozen; extra keys (e.g. `latent`) are allowed.

To add a GM, write a function with the signature below and decorate it with
`@register("name")`. Nothing else needs to change:

    fn(phase, batch_size, n_tokens, generator=None, **gm_kwargs) -> dict
        phase: LongTensor (T,) as described in `sample_episode`.
        Must return {"tokens": LongTensor (B, T)}; any other (B, T) tensors
        (e.g. the hidden GM state) are passed through to the episode dict.
"""

import torch

GMS = {}


def register(name):
    def deco(fn):
        GMS[name] = fn
        return fn
    return deco


def episode_layout(n_days, wake_len, sleep_len, test_len):
    """Phase index per step: d for day d's wake block, -1 for sleep, n_days for test."""
    phase = []
    for d in range(n_days):
        phase += [d] * wake_len + [-1] * sleep_len
    phase += [n_days] * test_len
    return torch.tensor(phase, dtype=torch.long)


def sample_episode(batch_size, n_days, wake_len, sleep_len, test_len,
                   n_tokens, gm="hmm", generator=None, **gm_kwargs) -> dict:
    """Returns tensors of shape (B, T), T = n_days*(wake_len+sleep_len) + test_len:
      tokens   long  — clean target token at each step (arbitrary/ignored during sleep)
      sleep    bool  — True during sleep blocks
      phase    long  — 0..n_days-1 for each day's wake block, -1 for sleep, n_days for test
      loss_mask bool — True on wake and test steps, False during sleep

    A new GM is sampled for every episode (i.e. every batch element).
    `sleep_len=0` gives the no-sleep baseline. Pass a `torch.Generator` for
    reproducible episodes. GMs may add extra keys (the HMM adds `latent`, the
    hidden state at each step, -1 during sleep).
    """
    phase = episode_layout(n_days, wake_len, sleep_len, test_len)
    out = GMS[gm](phase, batch_size, n_tokens, generator=generator, **gm_kwargs)
    phase = phase.expand(batch_size, -1).clone()
    sleep = phase == -1
    out["tokens"] = out["tokens"].masked_fill(sleep, 0)
    return dict(out, sleep=sleep, phase=phase, loss_mask=~sleep)


@register("hmm")
def random_hmm(phase, batch_size, n_tokens, generator=None,
               n_states=12, n_successors=2, emission_peak=0.8):
    """PLACEHOLDER GM: a random HMM per episode.

    Each hidden state has `n_successors` random successors and emits its own
    preferred token with probability `emission_peak` (uniform otherwise).
    Hidden states are split at random into one subset per day; day d only
    visits its subset (transitions renormalised), and the test block samples
    from the full HMM. Each block starts from a uniformly drawn state.

    TODO(team1): design a GM family that genuinely needs consolidation.
    """
    g, B, K, T = generator, batch_size, n_states, len(phase)
    n_days = int(phase.max())

    succ = torch.rand(B, K, K, generator=g).argsort(-1)[..., :n_successors]
    w = torch.rand(B, K, n_successors, generator=g) + 0.1
    trans = torch.zeros(B, K, K).scatter_(-1, succ, w)
    trans = trans / trans.sum(-1, keepdim=True)

    pref = torch.randint(n_tokens, (B, K, 1), generator=g)
    emit = torch.full((B, K, n_tokens), (1 - emission_peak) / (n_tokens - 1))
    emit.scatter_(-1, pref, emission_peak)

    # Day each hidden state belongs to: a random, balanced partition.
    perm = torch.rand(B, K, generator=g).argsort(-1)
    day_of_state = torch.empty(B, K, dtype=torch.long)
    day_of_state.scatter_(1, perm, (torch.arange(K) * n_days // K).expand(B, K))

    tokens = torch.zeros(B, T, dtype=torch.long)
    latent = torch.full((B, T), -1, dtype=torch.long)
    rows = torch.arange(B)
    prev_phase, state, T_block = None, None, None
    for t in range(T):
        p = int(phase[t])
        if p == -1:
            prev_phase = p
            continue
        if p != prev_phase:  # new block: restrict the HMM and draw an initial state
            allowed = (day_of_state == p) if p < n_days else torch.ones(B, K, dtype=torch.bool)
            T_block = trans * allowed[:, None, :]
            empty = T_block.sum(-1, keepdim=True) == 0
            T_block = torch.where(empty, allowed[:, None, :].float(), T_block)
            T_block = T_block / T_block.sum(-1, keepdim=True)
            state = torch.multinomial(allowed.float(), 1, generator=g).squeeze(1)
        else:
            state = torch.multinomial(T_block[rows, state], 1, generator=g).squeeze(1)
        latent[:, t] = state
        tokens[:, t] = torch.multinomial(emit[rows, state], 1, generator=g).squeeze(1)
        prev_phase = p
    return {"tokens": tokens, "latent": latent}
