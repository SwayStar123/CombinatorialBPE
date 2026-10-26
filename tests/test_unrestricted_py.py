"""cbpe/unrestricted_py.py (the pure-Python encoder) against the Rust binary: identical rows, chunk for chunk,
on case_affixes models trained here (with and without mark_rule); other model kinds are refused."""
import os
import random

import pytest

from cbpe.unrestricted import NONE_BASE, UnrestrictedBPE
from test_unrestricted_case import HARD_CA, SMALL, TRAIN, TRAIN_CA, needs_bin, train_tiny  # noqa: E402

pytest.importorskip("regex")


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("py_encoder")
    return {"small": train_tiny(tmp, True, text=TRAIN_CA, tag="s", case_affixes=1, **SMALL),
            "mark": train_tiny(tmp, True, text=TRAIN_CA, tag="m", case_affixes=1, mark_rule=1, **SMALL),
            "default": train_tiny(tmp, True, text=TRAIN_CA, tag="d", case_affixes=1),
            "pairs": train_tiny(tmp, True, text=TRAIN_CA, tag="p", case_affixes=1, mark_rule=1,
                                **dict(SMALL, lambda_permille=20, lamc_permille=100)),
            "plain": train_tiny(tmp, False, tag="n")}


def chunks_for(text_pool, n=1500, seed=0):
    """real chunks, and case / cut / glued perturbations of them, as UnrestrictedBPE chunks"""
    from cbpe.unrestricted import CHUNK
    R = random.Random(seed)
    words = list(dict.fromkeys(CHUNK.findall("\n".join(text_pool))))
    out = list(words)
    for _ in range(n):
        w, w2 = R.choice(words), R.choice(words).strip()
        out += [w.upper(), w.lower(), w[:1].upper() + w[1:], w.strip() + w2.title(), w[:R.randint(1, max(1, len(w)))],
                "".join(c.upper() if R.random() < .3 else c for c in w), w + chr(R.randint(0x300, 0x36F)) + w2,
                chr(R.randint(0x1F300, 0x1F64F)) + w]
    return list(dict.fromkeys(c for c in out if c))


@needs_bin
@pytest.mark.parametrize("name", ["small", "mark", "default"])
def test_python_encoder_equals_binary(models, name):
    t = models[name]
    chunks = chunks_for(HARD_CA + TRAIN.splitlines())
    assert t._raw_binary(chunks) == t._raw_python(chunks)


@needs_bin
@pytest.mark.parametrize("name", ["small", "mark"])
def test_encode_with_either_encoder(models, name, monkeypatch):
    t = models[name]
    text = "\n".join(HARD_CA)
    monkeypatch.setenv("CBPE_ENCODER", "rust")
    a = t.encode(text)
    monkeypatch.setenv("CBPE_ENCODER", "python")
    b = t.encode(text)
    assert a == b and t.decode(b) == text


@needs_bin
@pytest.mark.parametrize("name", ["pairs", "plain"])
def test_unsupported_models_are_refused(models, name):
    with pytest.raises(NotImplementedError):
        models[name]._raw_python(["word"])


def test_unknown_char_is_a_byte_unit(models):
    if "small" not in models:
        pytest.skip()
    rows = models["small"]._raw_python(["a\u2603b"])[0]
    assert any(c == 0xFFFFFFFF for _, _, c, _ in rows)
