import numpy as np
import torch

from mechlab.core.gpt import GPTConfig, TinyGPT
from mechlab.core.probes import mlp_probe, ridge_cv, ridge_fit
from mechlab.core.prefetch import Prefetcher


def _model(**kw):
    cfg = GPTConfig(d_model=32, n_layers=2, n_heads=4, d_mlp=64, vocab_size=10, n_out=3, **kw)
    torch.manual_seed(0)
    return TinyGPT(cfg).eval()


def test_shapes_and_capture():
    m = _model()
    out = m(tokens=torch.randint(1, 10, (3, 7)), return_resid=True, return_attn=True)
    assert out["logits"].shape == (3, 7, 3)
    assert len(out["resid"]) == 3 and len(out["attn"]) == 2
    assert torch.allclose(out["attn"][0].sum(-1), torch.ones(3, 4, 7), atol=1e-5)


def test_causal():
    m = _model()
    x = torch.randint(1, 10, (1, 8))
    y = x.clone(); y[0, 5:] = 1
    a, b = m(tokens=x)["logits"], m(tokens=y)["logits"]
    assert torch.allclose(a[0, :5], b[0, :5], atol=1e-5)


def test_null_slot_rows_sum_to_one_including_null():
    m = _model(null_attn=True)
    w = m(tokens=torch.randint(1, 10, (2, 6)), return_attn=True)["attn"][0]
    assert w.shape[-1] == 7
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5)


def test_time_rotary_is_shift_invariant():
    cfg = GPTConfig(d_model=32, n_layers=2, n_heads=2, d_mlp=64, n_features=2, n_out=4, pos="rope",
                    rope_freqs=[0.1, 0.5, 1.0, 3.0], null_attn=True)
    torch.manual_seed(0)
    m = TinyGPT(cfg).eval()
    f = torch.randn(1, 9, 2)
    t = torch.cumsum(torch.rand(1, 9), 1)
    assert torch.allclose(m(feats=f, pos=t)["logits"], m(feats=f, pos=t + 3.7)["logits"], atol=1e-4)


def test_edit_hook_changes_output():
    m = _model()
    x = torch.randint(1, 10, (1, 6))
    base = m(tokens=x)["logits"]

    def edit(i, h):
        if i == 0:
            h = h.clone(); h[0, 3] += 5
        return h
    ed = m(tokens=x, edit=edit)["logits"]
    assert torch.allclose(base[0, :3], ed[0, :3]) and not torch.allclose(base[0, 3:], ed[0, 3:])


def test_ridge_recovers_linear_map_and_groups():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 10)); W = rng.normal(size=(10, 3))
    Y = X @ W + 0.01 * rng.normal(size=(400, 3))
    res = ridge_cv(X, Y, groups=np.repeat(np.arange(40), 10))
    assert np.nanmean(res["r2"]) > 0.99
    fit = ridge_fit(X, Y, alpha=1e-3)
    assert np.allclose(X @ fit["W"] + fit["b"], Y, atol=0.1)
    noise = ridge_cv(rng.normal(size=(400, 10)), Y, groups=np.repeat(np.arange(40), 10))
    assert np.nanmean(noise["r2"]) < 0.1


def test_mlp_probe_beats_linear_on_nonlinear_target():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 4)).astype(np.float32)
    Y = (X[:, 0] * X[:, 1])[:, None]
    lin = np.nanmean(ridge_cv(X, Y)["r2"])
    mlp = np.nanmean(mlp_probe(X, Y, epochs=200)["r2"])
    assert mlp > lin + 0.5


def _square(i):
    return {"x": np.array([i * i])}


def test_prefetcher_order():
    with Prefetcher(_square, 3, 9, workers=2, ahead=3) as pf:
        assert [int(pf.get(s)["x"][0]) for s in range(3, 9)] == [i * i for i in range(3, 9)]


def test_attn_mask_all_true_is_identity_and_row_restriction_is_local():
    m = _model()
    x = torch.randint(1, 10, (2, 8))
    base = m(tokens=x)["logits"]
    full = torch.ones(2, 8, 8, dtype=torch.bool)
    assert torch.allclose(base, m(tokens=x, attn_mask=full)["logits"], atol=1e-5)
    r = full.clone()
    r[:, 5] = False
    r[:, 5, [0, 5]] = True  # row 5 may only read BOS and itself
    out = m(tokens=x, attn_mask=r)["logits"]
    assert torch.allclose(base[:, :5], out[:, :5], atol=1e-5)  # earlier positions cannot see the change
    assert not torch.allclose(base[:, 5], out[:, 5])  # later positions may change too: they read position 5
    per_layer = m(tokens=x, attn_mask={1: r})["logits"]  # only layer 1 restricted
    assert not torch.allclose(per_layer[:, 5], out[:, 5]) and torch.allclose(per_layer[:, :5], base[:, :5], atol=1e-5)
