"""cbpe/qwen3.py (PyTorch) against the JAX trainer: a random tiny model whose per-token NLL (each of the
four fields) was computed by experiments/jax_lm.py in fp32 (tests/fixtures/make_qwen3_tiny.py), plus
consistency checks of the cached / sampled paths."""
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")
from cbpe import qwen3  # noqa: E402

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny")
    import shutil
    shutil.copy(os.path.join(FIX, "qwen3_tiny.json"), d / qwen3.CONFIG)
    shutil.copy(os.path.join(FIX, "qwen3_tiny.safetensors"), d / qwen3.WEIGHTS)
    model, _ = qwen3.load_model(str(d), torch.float32, "cpu", tokenizer=False)
    return model


@pytest.fixture(scope="module")
def ref():
    r = np.load(os.path.join(FIX, "qwen3_tiny.npz"))
    return torch.tensor(r["ids"], dtype=torch.long), r["nll"]


def test_nll_matches_jax(tiny, ref):
    ids, nll = ref
    with torch.no_grad():
        got = tiny.nll_fields(tiny.hidden(ids[:, :-1]), ids[:, 1:]).numpy()
    assert got.shape == nll.shape
    np.testing.assert_allclose(got, nll, atol=2e-4, rtol=1e-4)


def test_score_windows(tiny, ref):
    ids, nll = ref
    got = qwen3.score(tiny, ids[0].numpy(), ctx=8).numpy()  # two windows of 8 (= the first 16 targets)
    # (a new window restarts the context, so only the first window equals the JAX run)
    np.testing.assert_allclose(got[:8], nll[0, :8], atol=2e-4, rtol=1e-4)
    assert got.shape == (16, 4)


def test_kv_cache_equals_full_forward(tiny, ref):
    from transformers import DynamicCache
    ids = ref[0][:1, :12]
    with torch.no_grad():
        full = tiny.hidden(ids)
        cache = DynamicCache()
        parts = [tiny.hidden(ids[:, :5], cache)] + [tiny.hidden(ids[:, i:i + 1], cache) for i in range(5, 12)]
    np.testing.assert_allclose(torch.cat(parts, 1).numpy(), full.numpy(), atol=1e-5)


def test_greedy_sample_is_the_argmax_chain(tiny, ref):
    """the token drawn greedily has the log-probability the teacher-forced scorer gives it, and no
    other value of the first field beats it"""
    ids = ref[0][:1, :10]
    with torch.no_grad():
        h = tiny.hidden(ids)[:, -1]
        tok, lp = tiny.sample(h)
        nll = tiny.nll_fields(h[:, None], tok[None])
    assert tok.shape == (1, 4)
    np.testing.assert_allclose(-lp.numpy(), nll.sum(-1).numpy().reshape(-1), atol=1e-4)
    first = tiny.order[0]
    assert tok[0, first].item() == tiny.logits(first, h).argmax(-1).item()


def test_candidates_include_greedy(tiny, ref):
    h = tiny.hidden(ref[0][:1, :10])[:, -1]
    gen = torch.Generator().manual_seed(0)
    c, lp = tiny.candidates(h, 16, gen)
    assert c.shape == (17, 4) and lp.shape == (17,)
    assert torch.equal(c[0], tiny.sample(h)[0][0])
    assert (lp <= 1e-6).all()


class Stub:
    """letters a-z as core ids 0-25 (the tiny model's tables are (4, 6, 40, 5))"""
    sizes = {"variation": 4, "prefix": 6, "core": 40, "suffix": 5}

    def encode(self, text):
        return [(0, 0, ord(c) - 97, 0) for c in text]

    def token_bytes(self, t):
        return bytes([97 + t[2]]) if t[2] < 26 else b"?"


def test_generate_matches_uncached_greedy(tiny):
    tok = Stub()
    text, new = qwen3.generate(tiny, tok, "abcde", max_new=6, heal=False)
    ids = [tuple(t) for t in tok.encode("abcde")]
    for _ in range(6):  # recompute the whole sequence each time
        with torch.no_grad():
            h = tiny.hidden(torch.tensor([ids]))[:, -1]
            ids.append(tuple(tiny.sample(h)[0][0].tolist()))
    assert new == ids[5:]
    assert text == b"".join(tok.token_bytes(t) for t in new).decode()


def test_healing_respells_the_prompt_end(tiny):
    tok = Stub()
    # the tiny model may find no candidate that fits: then the removed token itself is kept
    text, new = qwen3.generate(tiny, tok, "abcd", max_new=5, heal=True, temperature=0.7, seed=1)
    data = b"".join(tok.token_bytes(t) for t in new)
    assert data.startswith(b"d") and len(new) == 5
    assert text == data[1:].decode()


def test_heal_choice():
    tok = Stub()
    t = lambda c: (0, 0, c, 0)
    cands = np.array([t(3), t(4), t(0)])  # d, e, a
    logps = np.array([-3.0, -1.0, -0.5])
    # need "d": only 'd' agrees
    assert qwen3.heal_choice(cands, logps, b"d", tok, 0.0, None) == (t(3), b"")
    # need "dx": 'd' is a prefix of it, the rest remains
    assert qwen3.heal_choice(cands, logps, b"dx", tok, 0.0, None) == (t(3), b"x")
    assert qwen3.heal_choice(cands, logps, b"z", tok, 0.0, None) is None
