"""Team 2: training loop and CLI.

    python -m rnf.train --name sleep --sleep_len 100 --h_size 16 --beta 1.0 --steps 2000
"""

import argparse
import csv
import dataclasses
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rnf.model import NightFever
from rnf.task import sample_episode


@dataclass
class Config:
    name: str = "default"
    seed: int = 0
    device: str = "auto"  # auto = cuda, then mps, then cpu
    # Episode / generative model
    gm: str = "hmm"
    gm_kwargs: dict = field(default_factory=dict)  # JSON on the CLI, e.g. '{"n_states": 16}'
    n_tokens: int = 8
    n_days: int = 3
    wake_len: int = 100
    sleep_len: int = 100  # 0 = no-sleep baseline
    test_len: int = 100
    # Input noise
    alphas: tuple = (0.0, 0.5, 1.0, 2.0)
    sigma: float = 1.0
    # Model. h_size and beta are the main experimental knobs.
    p_size: int = 64
    h_size: int = 32
    c_size: int = 256
    z_size: int = 32
    beta: float = 1.0
    # Optimisation
    steps: int = 1000
    batch_size: int = 16
    lr: float = 1e-3
    clip: float = 1.0
    log_every: int = 25


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description="Train RNN Night Fever.")
    for f in dataclasses.fields(Config):
        default = f.default if f.default is not dataclasses.MISSING else f.default_factory()
        if isinstance(default, dict):
            parser.add_argument(f"--{f.name}", type=json.loads, default=default)
        elif isinstance(default, tuple):
            parser.add_argument(f"--{f.name}", type=type(default[0]), nargs="+", default=default)
        else:
            parser.add_argument(f"--{f.name}", type=type(default), default=default)
    return Config(**vars(parser.parse_args(argv)))


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(cfg):
    return NightFever(cfg.n_tokens, cfg.p_size, cfg.h_size, cfg.c_size, cfg.z_size,
                      cfg.alphas, cfg.sigma)


def make_batch(cfg, generator, device, batch_size=None):
    batch = sample_episode(batch_size or cfg.batch_size, cfg.n_days, cfg.wake_len,
                           cfg.sleep_len, cfg.test_len, cfg.n_tokens, gm=cfg.gm,
                           generator=generator, **cfg.gm_kwargs)
    return {k: v.to(device) for k, v in batch.items()}


def compute_losses(cfg, logits, penalty, batch):
    """Loss = CE on loss_mask steps + beta * mean penalty over all steps."""
    ce = F.cross_entropy(logits.transpose(1, 2), batch["tokens"], reduction="none")
    phase = batch["phase"]
    wake, test = (phase >= 0) & (phase < cfg.n_days), phase == cfg.n_days
    task_loss = ce[batch["loss_mask"]].mean()
    correct = (logits.argmax(-1) == batch["tokens"]).float()
    return dict(
        loss=task_loss + cfg.beta * penalty.mean(),
        task_loss=task_loss,
        wake_loss=ce[wake].mean(),
        test_loss=ce[test].mean(),
        test_acc=correct[test].mean(),
        penalty=penalty.mean(),
    )


def train(cfg):
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    run_dir = Path("runs") / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))

    model = build_model(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)
    print(f"Training '{cfg.name}' on {device}: {sum(p.numel() for p in model.parameters())} params")

    keys = ["loss", "wake_loss", "test_loss", "test_acc", "penalty"]
    with open(run_dir / "log.csv", "w", newline="") as fh:
        log = csv.writer(fh)
        log.writerow(["step", *keys, "seconds"])
        t0 = time.time()
        for step in range(1, cfg.steps + 1):
            batch = make_batch(cfg, gen, device)  # fresh episodes every step
            logits, penalty = model(batch)
            losses = compute_losses(cfg, logits, penalty, batch)
            opt.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()

            if step % cfg.log_every == 0 or step == cfg.steps:
                vals = [losses[k].item() for k in keys]
                log.writerow([step, *vals, round(time.time() - t0, 1)])
                fh.flush()
                print(f"step {step:5d} " + " ".join(f"{k} {v:.3f}" for k, v in zip(keys, vals))
                      + f" ({time.time() - t0:.0f}s)", flush=True)

    torch.save(model.state_dict(), run_dir / "model.pt")
    print(f"Saved to {run_dir}/")
    return model


def load_run(name, device="cpu"):
    run_dir = Path("runs") / name
    cfg = json.loads((run_dir / "config.json").read_text())
    cfg = Config(**{**cfg, "alphas": tuple(cfg["alphas"])})
    model = build_model(cfg)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    return cfg, model.to(device).eval()


if __name__ == "__main__":
    train(parse_config())
