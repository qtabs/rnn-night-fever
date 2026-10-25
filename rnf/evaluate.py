"""Team 3 entry point: test-phase metrics and state recording.

    python -m rnf.evaluate --name sleep --record
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from rnf.train import compute_losses, get_device, load_run, make_batch, seed_everything


def evaluate(name, n_episodes=64, seed=1234, record=False, device="auto"):
    seed_everything(seed)
    device = get_device(device)
    cfg, model = load_run(name, device)
    batch = make_batch(cfg, torch.Generator().manual_seed(seed), device, batch_size=n_episodes)
    with torch.no_grad():
        logits, penalty, states = model(batch, return_states=True)
    losses = compute_losses(cfg, logits, penalty, batch)
    print(f"{name}: test loss {losses['test_loss']:.3f}, test acc {losses['test_acc']:.3f}, "
          f"wake loss {losses['wake_loss']:.3f}, penalty {losses['penalty']:.3f}")

    if record:
        arrays = {k: v.cpu().numpy() for k, v in batch.items()}
        arrays.update({f"state_{k}": v.cpu().numpy() for k, v in states.items()})
        arrays["logits"] = logits.cpu().numpy()
        path = Path("runs") / name / "states.npz"
        np.savez_compressed(path, **arrays)
        print(f"Recorded {', '.join(arrays)} to {path}")

    # TODO(team3): replay metrics (wake-trained decoders applied to sleep
    # states, sequence structure, residual vs recency).
    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained run.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--n_episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--record", action="store_true",
                        help="save tokens, sleep, phase and P/H/C states to runs/<name>/states.npz")
    args = parser.parse_args()
    evaluate(args.name, args.n_episodes, args.seed, args.record, args.device)
