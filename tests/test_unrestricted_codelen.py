"""Unrestricted Combinatorial BPE (native/dict_search) with the unigram code-length objective
(code_len, code_w_permille): exact round trips of trained models, the pairwise code length on
hand-made models, and a brute-force check that the encoder minimises (tokens + w x bits, bits)
lexicographically."""
import os
import random
import struct
import sys
from array import array
from collections import Counter

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.dirname(__file__))

from cbpe.unrestricted import CHUNK, UnrestrictedBPE, written_core, written_token  # noqa: E402
from test_unrestricted_case import BIN, HARD, HARD_CA, TRAIN, TRAIN_CA, needs_bin, train_tiny  # noqa: E402

CODE_MAGIC = int.from_bytes(b"CLEN", "little")


@pytest.fixture(scope="module")
def cl_models(tmp_path_factory):
    """code_len models: case_cores, case_affixes (+ code_w), old folded cores (+ code_w), and
    case_affixes with mark_rule and lamc (both on the pairwise DP) and a fixed backoff beta"""
    if BIN is None:
        pytest.skip("native/dict_search not built")
    tmp = tmp_path_factory.mktemp("codelen")
    return (train_tiny(tmp, True, text=TRAIN_CA, tag="c1", code_len=1),
            train_tiny(tmp, True, text=TRAIN_CA, tag="a1", case_affixes=1, code_len=1, code_w_permille=10),
            train_tiny(tmp, False, text=TRAIN_CA, tag="o1", code_len=1, code_w_permille=50),
            train_tiny(tmp, True, text=TRAIN_CA, tag="am1", case_affixes=1, mark_rule=1, lamc_permille=100,
                       code_len=1, code_w_permille=5, code_beta_permille=2000))


def _has_code_section(path):
    return struct.pack("<II", CODE_MAGIC, 1) in open(path, "rb").read()


@needs_bin
def test_code_section(cl_models, tmp_path):
    for t in cl_models:
        assert _has_code_section(t.model_path), t.model_path
    # without code_len: no section (the old file layout)
    t = train_tiny(tmp_path, True, text=TRAIN, tag="off", max_rounds=1)
    assert not _has_code_section(t.model_path)


@needs_bin
@pytest.mark.parametrize("s", HARD_CA + TRAIN.splitlines()[:5])
def test_codelen_roundtrip(cl_models, s):
    import bench_dictsearch as bd
    for t in cl_models:
        enc = t.encode(s)
        assert t.decode(enc) == s, (t.model_path, s, enc)
    # the bench decoder agrees (it asserts a lossless rebuild of every chunk)
    for t in cl_models:
        m = bd.DModel(t.model_path, t.alphabet)
        chunks = bd.CHUNK.findall(s)
        for ch, units in zip(chunks, m.encode_chunks(chunks, t.idx)):
            m.token_count_and_check(ch, units)


@needs_bin
def test_codelen_deterministic(cl_models):
    for t in cl_models:
        assert t.encode(" ".join(HARD)) == t.encode(" ".join(HARD))


# ---------------------------------------------------------------- hand-made models


def write_model(path, alpha, idx, maps, kind, cores, prefixes, suffixes, costs, vcost, code=None):
    """a model file as the trainer saves one (kind "old", "case" or "cafx"; rows as strings in
    their stored spellings; costs = [core costs, prefix costs, suffix costs]; no delta, lambda or
    other primary terms). code = (w, bits per fallback byte, backoff offsets (prefix, suffix,
    variation side) per core, observed pair bits [{(prefix, core)}, {(core, suffix)}, {(core, v)}]
    by row id; every value a multiple of 1/1024 bit)"""
    with open(path, "wb") as f:
        if kind != "old":
            f.write(struct.pack("<I", int.from_bytes(b"CAFX" if kind == "cafx" else b"CASE", "little")))
        f.write(struct.pack("<I", len(alpha)))
        for m in maps:
            array("I", m).tofile(f)
        for tab in (cores, prefixes, suffixes):
            f.write(struct.pack("<I", len(tab)))
            for r in tab:
                f.write(struct.pack(f"<I{len(r)}I", len(r), *[idx[c] for c in r]))
        for cs in costs:
            f.write(struct.pack(f"<{len(cs)}d", *cs))
        f.write(struct.pack(f"<{len(vcost)}d", *vcost))
        f.write(struct.pack("<d", 0.0))  # delta
        array("I", [len(c.encode("utf-8")) for c in alpha]).tofile(f)
        n_aff = len(prefixes) + len(suffixes)
        f.write(struct.pack(f"<dd{n_aff}d", 0.0, 0.0, *[0.0] * n_aff))  # lambda, mu, affix specificity
        f.write(struct.pack("<ddII", 0.0, 0.0, 0, 0))  # mu2, tau, no PMI pairs
        f.write(struct.pack("<dIII", 0.0, 0, 0, 0))  # lamc, no counts
        n_rows = len(cores) + n_aff
        f.write(struct.pack(f"<d{n_rows}d", 0.0, *[0.0] * n_rows))  # nu, glue scores
        if code is not None:
            w, byte, bo, pairs = code
            f.write(struct.pack("<IIdd", CODE_MAGIC, 1, w, byte))
            for b in bo:
                f.write(struct.pack("<3H", *(round(x * 1024) for x in b)))
            for k, pr in enumerate(pairs):
                by_core = {}
                for (a, b), v in pr.items():
                    c, x = (b, a) if k == 0 else (a, b)
                    by_core.setdefault(c, []).append((x, v))
                buf = bytearray()
                for c in range(len(cores)):
                    row = sorted(by_core.get(c, []))
                    buf += varint(len(row))
                    for j, (x, v) in enumerate(row):
                        buf += varint(x if j == 0 else x - row[j - 1][0] - 1) + struct.pack("<H", round(v * 1024))
                f.write(struct.pack("<I", len(buf)) + bytes(buf))
    return UnrestrictedBPE(str(path), alpha)


def f32(x):
    """a code length as the model file stores it (a multiple of 1/1024 bit)"""
    return round(x * 1024) / 1024


def varint(x):
    out = bytearray()
    while x >= 0x80:
        out.append(x & 0x7F | 0x80)
        x >>= 7
    out.append(x)
    return bytes(out)


def symbols(text):
    import bench_dictsearch as bd
    return bd.build_symbols(Counter(text * 20), 10_000, fold=True)


def show(t, enc):
    return ["|".join(t.written_parts(v, p, c - 256, s)) for v, p, c, s in enc]


@needs_bin
def test_codelen_resolves_tie(tmp_path):
    """观光局首度举办: [观光局][首度][举办] and [观光|局|首][度][举办] are both 3 tokens. By the
    independent costs of the parts (the old tie-break) the second is cheaper (short, frequent
    pieces); by the code length the first: 局 was never seen with the suffix 首 (its Witten-Bell
    backoff costs 9 bits more), while 观光局 and 首度 nearly always come with empty affixes"""
    text = "观光局首度举办"
    alpha, idx, maps = symbols(text)
    cores = ["观光局", "首度", "举办", "局", "度", "观光"] + [c for c in alpha if c not in "局度"]
    prefixes, suffixes = ["", "观光"], ["", "首"]
    costs = [[8.5, 12.0, 8.0, 5.0, 6.0, 7.0] + [15.0] * (len(cores) - 6), [1.0, 3.0], [1.0, 2.0]]
    vcost = [0.1, 5.0, 5.0, 5.0]
    old = write_model(tmp_path / "old.bin", alpha, idx, maps, "old", cores, prefixes, suffixes, costs, vcost)
    assert show(old, old.encode(text)) == ["观光|局|首", "|度|", "|举办|"]
    bo = [(0.0, 0.0, 0.0)] * len(cores)
    bo[0] = bo[1] = (10.0, 10.0, 0.0)
    bo[3] = (1.0, 9.0, 0.0)
    pairs = [{(0, 0): 0.01, (0, 1): 0.01, (1, 3): 4.0}, {(0, 0): 0.01, (1, 0): 0.01, (3, 0): 0.2}, {}]
    for w in (0.0, 0.01):
        t = write_model(tmp_path / f"cl{w}.bin", alpha, idx, maps, "old", cores, prefixes, suffixes, costs, vcost,
                        code=(w, 30.0, bo, pairs))
        assert show(t, t.encode(text)) == ["|观光局|", "|首度|", "|举办|"], w
        assert t.decode(t.encode(text)) == text


@needs_bin
def test_codelen_whole_word_core(tmp_path):
    """ Gott: one token either way, ' '|Got|t (the frequent core got + the suffix t) or ' '|Gott|.
    Independent costs prefer the pieces; the code length prefers the whole word, since got has
    never been seen with the suffix t while Gott nearly always stands alone, capitalised"""
    text = " Gott"
    alpha, idx, maps = symbols(text + " got")
    cores = ["gott", "got"] + [c for c in alpha if maps[0][idx[c]] == idx[c]]
    prefixes, suffixes = ["", " "], ["", "t"]
    costs = [[12.0, 8.0] + [15.0] * (len(cores) - 2), [1.0, 1.0], [1.0, 3.0]]
    vcost = [0.1, 3.0, 5.0, 5.0]
    old = write_model(tmp_path / "old.bin", alpha, idx, maps, "old", cores, prefixes, suffixes, costs, vcost)
    assert show(old, old.encode(text)) == [" |Got|t"]
    bo = [(0.0, 0.0, 0.0)] * len(cores)
    bo[0], bo[1] = (0.0, 8.0, 0.0), (0.0, 6.0, 0.0)
    pairs = [{(1, 0): 0.3, (1, 1): 0.5}, {(0, 0): 0.05, (1, 0): 0.2}, {(0, 1): 0.1, (1, 0): 0.05}]
    for w in (0.0, 0.02):
        t = write_model(tmp_path / f"cl{w}.bin", alpha, idx, maps, "old", cores, prefixes, suffixes, costs, vcost,
                        code=(w, 30.0, bo, pairs))
        assert show(t, t.encode(text)) == [" |Gott|"], w


@needs_bin
def test_codelen_weight_trades_tokens_for_bits(tmp_path):
    """with w > 0 the bits enter the primary cost: two cheap tokens beat one token whose code
    length is more than 1 / w bits longer"""
    text = "abcd"
    alpha, idx, maps = symbols(text)
    cores = ["abcd", "ab", "cd"] + list(alpha)
    costs = [[40.0, 2.0, 2.0] + [10.0] * len(alpha), [1.0], [1.0]]
    bo = [(0.0, 0.0, 0.0)] * len(cores)
    for w, want in ((0.0, ["abcd"]), (0.01, ["abcd"]), (0.05, ["ab", "cd"])):
        t = write_model(tmp_path / f"w{w}.bin", alpha, idx, maps, "old", cores, [""], [""], costs,
                        [0.1, 5.0, 5.0, 5.0], code=(w, 30.0, bo, [{}, {}, {}]))
        assert ["".join(p) for p in (t.written_parts(v, p, c - 256, s) for v, p, c, s in t.encode(text))] == want, w


# ---------------------------------------------------------------- brute force


def _brute_setup(tmp_path, kind, seed, w):
    """a random hand-made model (kind "case" or "cafx") with random code length statistics"""
    rng = random.Random(seed)
    base = "abAB发發 x"
    alpha, idx, maps = symbols(base + base.upper() + base.lower())
    fold = maps[0]
    letters = "ab发x "

    def rand_str(lo, hi):
        return "".join(rng.choice(letters) for _ in range(rng.randint(lo, hi)))
    cores = sorted({rand_str(2, 3) for _ in range(12)})
    if kind == "case":  # canonical spellings, some capitalised
        cores = [c.capitalize() if rng.random() < 0.3 else c for c in cores]
    cores += [c for c in alpha if fold[idx[c]] == idx[c] and c not in cores]
    prefixes = [""] + sorted({rand_str(1, 2) for _ in range(5)})
    suffixes = [""] + sorted({rand_str(1, 2) for _ in range(5)})
    if kind == "cafx":
        prefixes = [p.upper() if rng.random() < 0.2 else p for p in prefixes]
    for tab in (cores, prefixes, suffixes):  # one row per folded string
        seen, keep = set(), []
        for r in tab:
            k = tuple(fold[idx[c]] for c in r)
            if k not in seen:
                seen.add(k)
                keep.append(r)
        tab[:] = keep
    costs = [[rng.uniform(1, 14) for _ in t] for t in (cores, prefixes, suffixes)]
    nvar = 8 if kind == "cafx" else 5
    vcost = [rng.uniform(0.1, 6) for _ in range(nvar)]
    bo = [(f32(rng.uniform(0, 4)), f32(rng.uniform(0, 4)), f32(rng.uniform(0, 4))) for _ in cores]
    pairs = [{}, {}, {}]
    for ci in range(len(cores)):
        for pi in range(len(prefixes)):
            if rng.random() < 0.3:
                pairs[0][(pi, ci)] = f32(rng.uniform(0.05, 12))
        for si in range(len(suffixes)):
            if rng.random() < 0.3:
                pairs[1][(ci, si)] = f32(rng.uniform(0.05, 12))
        for v in range(nvar):
            if rng.random() < 0.3:
                pairs[2][(ci, v)] = f32(rng.uniform(0.05, 12))
    byte = rng.uniform(10, 30)
    t = write_model(tmp_path / f"bf_{kind}_{seed}.bin", alpha, idx, maps, kind, cores, prefixes, suffixes, costs, vcost,
                    code=(w, byte, bo, pairs))
    return t, dict(tabs=(cores, prefixes, suffixes), costs=costs, vcost=vcost, bo=bo, pairs=pairs, byte=byte, nvar=nvar)


def _bits(m, v, p, c, s):
    """the code length of a token as the header defines it"""
    costs, bo, pairs = m["costs"], m["bo"], m["pairs"]
    vb = pairs[2].get((c, v), m["vcost"][v] + bo[c][2])
    pb = pairs[0].get((p, c), costs[1][p] + bo[c][0])
    sb = pairs[1].get((c, s), costs[2][s] + bo[c][1])
    return costs[0][c] + vb + pb + sb


def _tokens_of_span(t, kind, m, span):
    """every (v, p, c, s) whose written text is span"""
    fs = lambda syms: tuple(t.fold[x] for x in syms)  # noqa: E731
    key = fs if kind == "cafx" else tuple  # affixes: folded with case_affixes, else raw
    ids = [t.idx[ch] for ch in span]
    rows = [[[t.idx[ch] for ch in r] for r in tab] for tab in m["tabs"]]
    out = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids) + 1):
            for pi, pid in enumerate(rows[1]):
                if key(pid) != key(ids[:a]):
                    continue
                for si, sid in enumerate(rows[2]):
                    if key(sid) != key(ids[b:]):
                        continue
                    for ci, cid in enumerate(rows[0]):
                        if fs(cid) != fs(ids[a:b]):
                            continue
                        for v in range(m["nvar"]):
                            if kind == "cafx":
                                wt = written_token([pid, cid, sid], v, t.fold, t.upper_of, t.trad_of)
                            else:
                                wt = pid + written_core(cid, v, True, t.fold, t.upper_of, t.trad_of) + sid
                            if wt == ids:
                                out.append((v, pi, ci, si))
    return out


def _brute(t, kind, m, text, w):
    """min (tokens + w x bits, bits) over all segmentations (byte fallback included: w x its bits in
    the primary cost, 1000 per byte in the secondary, as the old encoder)"""
    byte = m["byte"]
    best = [(0.0, 0.0)] + [None] * len(text)
    for j in range(1, len(text) + 1):
        cands = []
        for i in range(j):
            if i == j - 1:
                nb = len(text[i].encode("utf-8"))
                cands.append((best[i][0] + nb * (1 + w * byte), best[i][1] + 1000 * nb))  # (secondary: 1000 per byte)
            if all(ch in t.idx for ch in text[i:j]):
                for tok in _tokens_of_span(t, kind, m, text[i:j]):
                    b = _bits(m, *tok)
                    cands.append((best[i][0] + 1 + w * b, best[i][1] + b))
        best[j] = min(cands, key=lambda c: (round(c[0], 7), c[1]))
    return best[-1]


def _cost_of(m, enc, w):
    prim = sec = 0.0
    for v, p, c, s in enc:  # (a fallback char: one token per byte)
        b = m["byte"] if c < 256 else _bits(m, v, p, c - 256, s)
        prim += 1 + w * b
        sec += 1000 if c < 256 else b
    return prim, sec


@needs_bin
@pytest.mark.parametrize("kind", ["case", "cafx"])
@pytest.mark.parametrize("w", [0.0, 0.1])
def test_codelen_brute_force(tmp_path, kind, w):
    """the encoder's parse minimises (tokens + w x bits, bits) lexicographically: brute force over
    every segmentation, every (prefix, core, suffix, variation) writing each token, and bytes"""
    rng = random.Random(7)
    checked = 0
    for seed in range(3):
        t, m = _brute_setup(tmp_path, kind, seed, w)
        pieces = [r for tab in m["tabs"] for r in tab if r] + ["A", "B", "Ab", "AB", "發", "é"]
        for _ in range(40):
            text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 4)))[:9]
            enc = t.encode(text)
            assert t.decode(enc) == text
            # (encode parses whitespace chunks independently)
            parts = [_brute(t, kind, m, ch, w) for ch in CHUNK.findall(text)]
            got, want = _cost_of(m, enc, w), (sum(p[0] for p in parts), sum(p[1] for p in parts))
            assert abs(got[0] - want[0]) < 1e-6, (text, got, want, show(t, [e for e in enc if e[2] >= 256]))
            assert got[1] <= want[1] + 1e-6, (text, got, want)
            checked += 1
    assert checked == 120
