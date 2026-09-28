"""Unrestricted Combinatorial BPE (native/dict_search): code_w_permille WITHOUT code_len (w x the
token's unconditional bits, the secondary cost, also in the primary cost, on the fast 3-state DP).
Exact round trips of trained models, the model file's short CLEN section, and a brute-force check
that the encoder minimises (tokens + w x bits, bits) lexicographically with the unconditional bits
-log2 q(core) - log2 q(variation) - log2 q(prefix) - log2 q(suffix)."""
import os
import random
import struct
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from cbpe.unrestricted import CHUNK  # noqa: E402
from test_unrestricted_case import BIN, HARD_CA, TRAIN, TRAIN_CA, needs_bin, train_tiny  # noqa: E402
from test_unrestricted_codelen import CODE_MAGIC, _tokens_of_span, f32, show, symbols, write_model  # noqa: E402


@pytest.fixture(scope="module")
def cw_models(tmp_path_factory):
    """code_w without code_len: case_cores, case_affixes, and old folded cores"""
    if BIN is None:
        pytest.skip("native/dict_search not built")
    tmp = tmp_path_factory.mktemp("codew")
    return (train_tiny(tmp, True, text=TRAIN_CA, tag="c", code_w_permille=20),
            train_tiny(tmp, True, text=TRAIN_CA, tag="a", case_affixes=1, mark_rule=1, code_w_permille=20),
            train_tiny(tmp, False, text=TRAIN_CA, tag="o", code_w_permille=50))


def _code_section(path):
    """(flag, w, bits per fallback byte) of the model's CLEN section, or None"""
    data = open(path, "rb").read()
    for flag in (0, 1):
        k = data.rfind(struct.pack("<II", CODE_MAGIC, flag))
        if k >= 0:
            return (flag,) + struct.unpack_from("<dd", data, k + 8)
    return None


@needs_bin
def test_codew_section(cw_models, tmp_path):
    for t, w in zip(cw_models, (0.02, 0.02, 0.05)):
        sec = _code_section(t.model_path)
        assert sec is not None and sec[0] == 0 and sec[1] == w and sec[2] > 8.0, (t.model_path, sec)
        # the section is the end of the file: flag 0, w, bits per byte
        assert open(t.model_path, "rb").read()[-24:-16] == struct.pack("<II", CODE_MAGIC, 0)
    # without code_w: no section (the old file layout)
    t = train_tiny(tmp_path, True, text=TRAIN, tag="off", max_rounds=1)
    assert _code_section(t.model_path) is None


@needs_bin
def test_codew_needs_no_pairwise(tmp_path):
    """code_w without code_len refuses the pairwise terms (mu2, lamc): it is for the 3-state DP"""
    import subprocess
    with pytest.raises(subprocess.CalledProcessError):
        train_tiny(tmp_path, True, text=TRAIN, tag="lamc", max_rounds=1, code_w_permille=20, lamc_permille=100)


@needs_bin
@pytest.mark.parametrize("s", HARD_CA + TRAIN.splitlines()[:5])
def test_codew_roundtrip(cw_models, s):
    import bench_dictsearch as bd
    for t in cw_models:
        enc = t.encode(s)
        assert t.decode(enc) == s, (t.model_path, s, enc)
    for t in cw_models:
        m = bd.DModel(t.model_path, t.alphabet)
        chunks = bd.CHUNK.findall(s)
        for ch, units in zip(chunks, m.encode_chunks(chunks, t.idx)):
            m.token_count_and_check(ch, units)


def _write_uncond(path, alpha, idx, maps, kind, cores, prefixes, suffixes, costs, vcost, w, byte):
    """a hand-made model (write_model) ending with the code_w-without-code_len section"""
    write_model(path, alpha, idx, maps, kind, cores, prefixes, suffixes, costs, vcost)
    with open(path, "ab") as f:
        f.write(struct.pack("<IIdd", CODE_MAGIC, 0, w, byte))
    from cbpe.unrestricted import UnrestrictedBPE
    return UnrestrictedBPE(str(path), alpha)


@needs_bin
def test_codew_weight_trades_tokens_for_bits(tmp_path):
    """w > 0: two cheap tokens beat one token whose unconditional bits are more than 1 / w higher"""
    text = "abcd"
    alpha, idx, maps = symbols(text)
    cores = ["abcd", "ab", "cd"] + list(alpha)
    costs = [[40.0, 2.0, 2.0] + [10.0] * len(alpha), [1.0], [1.0]]
    for w, want in ((0.0, ["abcd"]), (0.01, ["abcd"]), (0.05, ["ab", "cd"])):
        t = _write_uncond(tmp_path / f"w{w}.bin", alpha, idx, maps, "old", cores, [""], [""], costs,
                          [0.1, 5.0, 5.0, 5.0], w, 30.0)
        got = ["".join(p) for p in (t.written_parts(v, p, c - 256, s) for v, p, c, s in t.encode(text))]
        assert got == want, w


def _brute_setup(tmp_path, kind, seed, w):
    """a random hand-made model (kind "case" or "cafx") with unconditional costs only"""
    rng = random.Random(seed)
    base = "abAB发發 x"
    alpha, idx, maps = symbols(base + base.upper() + base.lower())
    fold = maps[0]
    letters = "ab发x "

    def rand_str(lo, hi):
        return "".join(rng.choice(letters) for _ in range(rng.randint(lo, hi)))
    cores = sorted({rand_str(2, 3) for _ in range(12)})
    if kind == "case":
        cores = [c.capitalize() if rng.random() < 0.3 else c for c in cores]
    cores += [c for c in alpha if fold[idx[c]] == idx[c] and c not in cores]
    prefixes = [""] + sorted({rand_str(1, 2) for _ in range(5)})
    suffixes = [""] + sorted({rand_str(1, 2) for _ in range(5)})
    if kind == "cafx":
        prefixes = [p.upper() if rng.random() < 0.2 else p for p in prefixes]
    for tab in (cores, prefixes, suffixes):
        seen, keep = set(), []
        for r in tab:
            k = tuple(fold[idx[c]] for c in r)
            if k not in seen:
                seen.add(k)
                keep.append(r)
        tab[:] = keep
    costs = [[f32(rng.uniform(1, 14)) for _ in t] for t in (cores, prefixes, suffixes)]
    nvar = 8 if kind == "cafx" else 5
    vcost = [f32(rng.uniform(0.1, 6)) for _ in range(nvar)]
    byte = rng.uniform(10, 30)
    t = _write_uncond(tmp_path / f"bfu_{kind}_{seed}.bin", alpha, idx, maps, kind, cores, prefixes, suffixes, costs,
                      vcost, w, byte)
    return t, dict(tabs=(cores, prefixes, suffixes), costs=costs, vcost=vcost, byte=byte, nvar=nvar)


def _bits(m, v, p, c, s):
    """the unconditional code length of a token (its secondary cost)"""
    return m["costs"][0][c] + m["vcost"][v] + m["costs"][1][p] + m["costs"][2][s]


def _brute(t, kind, m, text, w):
    """min (tokens + w x bits, bits) over all segmentations (byte fallback: w x its bits in the
    primary cost, 1000 per byte in the secondary)"""
    best = [(0.0, 0.0)] + [None] * len(text)
    for j in range(1, len(text) + 1):
        cands = []
        for i in range(j):
            if i == j - 1:
                nb = len(text[i].encode("utf-8"))
                cands.append((best[i][0] + nb * (1 + w * m["byte"]), best[i][1] + 1000 * nb))
            if all(ch in t.idx for ch in text[i:j]):
                for tok in _tokens_of_span(t, kind, m, text[i:j]):
                    b = _bits(m, *tok)
                    cands.append((best[i][0] + 1 + w * b, best[i][1] + b))
        best[j] = min(cands, key=lambda c: (round(c[0], 7), c[1]))
    return best[-1]


def _cost_of(m, enc, w):
    prim = sec = 0.0
    for v, p, c, s in enc:
        b = m["byte"] if c < 256 else _bits(m, v, p, c - 256, s)
        prim += 1 + w * b
        sec += 1000 if c < 256 else b
    return prim, sec


@needs_bin
@pytest.mark.parametrize("kind", ["case", "cafx"])
@pytest.mark.parametrize("w", [0.0, 0.1])
def test_codew_brute_force(tmp_path, kind, w):
    """the encoder's parse minimises (tokens + w x bits, bits) lexicographically, bits = the
    unconditional code length: brute force over every segmentation, every (prefix, core, suffix,
    variation) writing each token, and bytes"""
    rng = random.Random(11)
    checked = 0
    for seed in range(3):
        t, m = _brute_setup(tmp_path, kind, seed, w)
        pieces = [r for tab in m["tabs"] for r in tab if r] + ["A", "B", "Ab", "AB", "發", "é"]
        for _ in range(40):
            text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 4)))[:9]
            enc = t.encode(text)
            assert t.decode(enc) == text
            parts = [_brute(t, kind, m, ch, w) for ch in CHUNK.findall(text)]
            got, want = _cost_of(m, enc, w), (sum(p[0] for p in parts), sum(p[1] for p in parts))
            assert abs(got[0] - want[0]) < 1e-6, (text, got, want, show(t, [e for e in enc if e[2] >= 256]))
            assert got[1] <= want[1] + 1e-6, (text, got, want)
            checked += 1
    assert checked == 120
