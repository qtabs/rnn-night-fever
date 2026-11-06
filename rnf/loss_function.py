import torch
import torch.nn as nn

def prediction_loss(prediction, target):
    criterion = nn.BCELoss()
    return criterion(prediction, target)

def kl_penalty(states):
    mu, logvar = states
    if mu.shape[1] < 2:
        return mu.new_zeros(())
    mu_prev = mu[:, :-1, :]
    mu_next = mu[:, 1:, :]

    logvar_prev = logvar[:, :-1, :]
    logvar_next = logvar[:, 1:, :]
    kl = 0.5 * (
        logvar_prev - logvar_next
        + (
            logvar_next.exp()
            + (mu_next - mu_prev) ** 2
        ) / logvar_prev.exp()
        - 1
    )
    return kl.sum(dim=-1).mean()

def fast_penalty(fast_states):
   return ((fast_states[:, 1:, :] -
             fast_states[:, :-1, :]) ** 2).mean()

def slow_penalty(slow_states):
    return ((slow_states[:, 1:, :] -
             slow_states[:, :-1, :]) ** 2).mean()

def total_loss(prediction, target,
               fast_states, slow_states,
               alpha_fast, alpha_slow):

    bce = prediction_loss(prediction, target)

    fast = fast_penalty(fast_states)
    slow = slow_penalty(slow_states)

    return bce + alpha_fast * fast + alpha_slow * slow