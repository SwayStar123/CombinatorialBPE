"""hf_remote/ (AutoModelForCausalLM / AutoTokenizer with trust_remote_code): a repository assembled by
scripts/make_hf_repo.py loads from a local directory; the model gives the JAX trainer's NLL (tiny fixture) and the
tokenizer packs / unpacks its four fields into one id."""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIX = os.path.join(ROOT, "tests", "fixtures")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import make_hf_repo  # noqa: E402


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("hf_tiny"))
    make_hf_repo.main(["--out", out, "--weights", os.path.join(FIX, "qwen3_tiny.safetensors"),
                       "--qwen_config", os.path.join(FIX, "qwen3_tiny.json"), "--sizes", "4,6,40,5"])
    return out


def test_automodel_matches_jax(repo):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.float32, device_map="cpu")
    r = np.load(os.path.join(FIX, "qwen3_tiny.npz"))
    ids = torch.tensor(r["ids"], dtype=torch.long)
    packed = model.pack(ids)
    assert packed.shape == ids.shape[:2] and torch.equal(model.unpack(packed), ids)
    with torch.no_grad():
        nll = model.token_nll(packed).numpy()
        out = model(packed, labels=packed)
    np.testing.assert_allclose(nll, r["nll"], atol=2e-4, rtol=1e-4)
    np.testing.assert_allclose(out.loss.item(), r["nll"].sum(-1).mean(), rtol=1e-4)
    gen = model.generate(packed[:, :6], max_new_tokens=5)
    assert gen.shape == (2, 11) and torch.equal(gen[:, :6], packed[:, :6])
    # padding is dropped: a left-padded row continues like the unpadded one
    pad = torch.cat([torch.zeros(2, 3, dtype=torch.long), packed[:, :6]], 1)
    mask = torch.cat([torch.zeros(2, 3, dtype=torch.long), torch.ones(2, 6, dtype=torch.long)], 1)
    assert torch.equal(model.generate(pad, attention_mask=mask, max_new_tokens=5), gen)


def test_autotokenizer_packs_fields(tmp_path_factory):
    from test_unrestricted_case import HARD_CA, SMALL, TRAIN_CA, train_tiny
    tmp = tmp_path_factory.mktemp("tok")
    model = train_tiny(tmp, True, text=TRAIN_CA, tag="t", case_affixes=1, **SMALL)
    model.save(str(tmp / "tok.json"))
    out = str(tmp_path_factory.mktemp("hf_tok"))
    make_hf_repo.main(["--out", out, "--weights", os.path.join(FIX, "qwen3_tiny.safetensors"),
                       "--qwen_config", os.path.join(FIX, "qwen3_tiny.json"), "--tokenizer", str(tmp / "tok.json")])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(out, trust_remote_code=True)
    assert tok.sizes == tuple(model.sizes.values())
    for s in HARD_CA[:12]:
        ids = tok.encode(s)
        assert [tok.unpack(i) for i in ids] == model.encode(s)
        assert all(tok.pack(tok.unpack(i)) == i for i in ids)
        assert tok.decode(ids) == s
    b = tok(["Unhappy years", "getName"], padding=True, return_tensors="pt")
    assert b["input_ids"].dtype == torch.long and b["attention_mask"].shape == b["input_ids"].shape
    assert b["attention_mask"].sum(1).tolist() == [len(tok.encode("Unhappy years")), len(tok.encode("getName"))]
