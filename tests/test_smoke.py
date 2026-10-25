import time

import pytest
import torch

from rnf.model import corrupt
from rnf.task import sample_episode
from rnf.train import Config, build_model, compute_losses, make_batch, train

TINY = dict(n_days=2, wake_len=10, sleep_len=5, test_len=8, n_tokens=6)


@pytest.mark.parametrize("sleep_len", [5, 0])
def test_sample_episode(sleep_len):
    kw = dict(TINY, sleep_len=sleep_len)
    ep = sample_episode(batch_size=3, generator=torch.Generator().manual_seed(0), **kw)
    T = kw["n_days"] * (kw["wake_len"] + sleep_len) + kw["test_len"]
    for key, dtype in [("tokens", torch.long), ("sleep", torch.bool),
                       ("phase", torch.long), ("loss_mask", torch.bool)]:
        assert ep[key].shape == (3, T) and ep[key].dtype == dtype
    assert torch.equal(ep["loss_mask"], ~ep["sleep"])
    assert torch.equal(ep["sleep"], ep["phase"] == -1)
    assert ep["sleep"].sum() == 3 * kw["n_days"] * sleep_len
    assert (ep["phase"][:, -1] == kw["n_days"]).all()
    assert ep["tokens"].min() >= 0 and ep["tokens"].max() < kw["n_tokens"]


def test_gradients_cross_sleep():
    """A test-phase loss must reach H and C through the sleep blocks (no truncation)."""
    torch.manual_seed(0)
    cfg = Config(**TINY, p_size=8, h_size=4, c_size=8, z_size=4)
    model = build_model(cfg)
    batch = make_batch(cfg, torch.Generator().manual_seed(0), "cpu", batch_size=2)
    logits, penalty, states = model(batch, return_states=True)
    assert logits.shape == (*batch["tokens"].shape, cfg.n_tokens)
    assert penalty.shape == batch["tokens"].shape
    assert states["H"].shape[-1] == cfg.h_size and states["C"].shape[-1] == cfg.c_size

    compute_losses(cfg, logits, penalty, batch)["test_loss"].backward()
    for net in (model.H, model.C):
        assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())

    # The test loss also reaches the day-1 inputs, i.e. it flows back through sleep.
    x = corrupt(batch["tokens"], batch["sleep"], cfg.n_tokens, cfg.alphas, cfg.sigma)
    x.requires_grad_(True)
    logits, penalty = model(batch, x=x)
    compute_losses(cfg, logits, penalty, batch)["test_loss"].backward()
    assert x.grad[:, batch["phase"][0] == 0].abs().sum() > 0


def test_train_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    t0 = time.time()
    cfg = Config(**TINY, name="smoke", steps=5, batch_size=4, log_every=1, device="cpu",
                 p_size=16, h_size=8, c_size=32, z_size=8)
    train(cfg)
    assert time.time() - t0 < 30
    assert (tmp_path / "runs/smoke/model.pt").exists()
    assert len((tmp_path / "runs/smoke/log.csv").read_text().splitlines()) == 6
