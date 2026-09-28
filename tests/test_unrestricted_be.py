"""Unrestricted Combinatorial BPE (native/dict_search) with the branching-entropy boundary cost
(be_permille, be_order): exact round trips of trained models, a Chinese tie resolved toward the
high-entropy boundary on hand-made models, and a brute-force check that the encoder minimises
(primary, secondary + boundary cost) lexicographically, on the 3-state and the pairwise DP."""
import random
import struct
import sys
import os

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.dirname(__file__))

from cbpe.unrestricted import CHUNK, NONE_BASE, written_core, written_token  # noqa: E402
from test_unrestricted_case import BIN, HARD, HARD_CA, TRAIN, TRAIN_CA, needs_bin, train_tiny  # noqa: E402
from test_unrestricted_codelen import _bits, f32, show, symbols, write_model  # noqa: E402

BE_MAGIC = int.from_bytes(b"BENT", "little")
M64 = (1 << 64) - 1

ZH = """观光局首度举办国际旅游展。北京是中国的首都。台北市政府今天宣布新的交通政策。
我们明天去图书馆看书。这个问题需要进一步研究。经济发展速度很快。
東京都は日本の首都です。私は毎日学校に行きます。今日はとても暑いですね。
"""
TRAIN_ZH = TRAIN + ZH * 40
HARD_ZH = ["观光局首度举办", "北京是中国的首都。", "東京都は日本の首都です。", "未见过的字：龘靐", "中文English混合123"]


# ---------------------------------------------------------------- the boundary statistics


def seq_hash(seq):
    """main.rs seq_hash (splitmix64 steps)"""
    h = 0x9E3779B97F4A7C15
    for c in seq:
        h = ((h ^ (c + 1)) + 0x9E3779B97F4A7C15) & M64
        h = ((h ^ (h >> 30)) * 0xBF58476D1CE4E5B9) & M64
        h = ((h ^ (h >> 27)) * 0x94D049BB133111EB) & M64
        h ^= h >> 31
    return h


def be_key(side, ctx):
    return seq_hash([0xFFFFFF00 + side] + list(ctx)) >> 32


class BE:
    """a boundary table as the model file stores it: entropies (1/16 bit) by (side, context of
    symbol ids), fallbacks, percentile tables (f32, 511 per position class), script group per
    alphabet symbol (classes: one per group, the last for a change of group)"""

    def __init__(self, w, order, tab, med, cdf, grp=()):
        self.w, self.order, self.med, self.grp = w, order, med, list(grp)
        self.cdf = [struct.unpack("<f", struct.pack("<f", v))[0] for v in cdf]
        self.ncls = len(self.cdf) // 511
        assert self.ncls * 511 == len(self.cdf) and self.ncls == (max(self.grp, default=0) + 2 if self.grp else self.ncls)
        self.tab = {be_key(d, ctx): v for (d, ctx), v in tab.items()}

    def section(self):
        rows = sorted(self.tab.items())
        buf = bytearray()
        for j, (k, v) in enumerate(rows):
            buf += varint(k if j == 0 else k - rows[j - 1][0] - 1) + bytes([v])
        return (struct.pack("<IdIIIII", BE_MAGIC, self.w, self.order, self.med[0], self.med[1], self.ncls, 511)
                + struct.pack(f"<{len(self.cdf)}f", *self.cdf) + struct.pack("<I", len(self.grp)) + bytes(self.grp)
                + struct.pack("<II", len(rows), len(buf)) + bytes(buf))

    def side(self, d, x, i):
        kmax = min(i if d == 0 else len(x) - i, self.order)
        for k in range(kmax, 0, -1):
            v = self.tab.get(be_key(d, x[i - k:i] if d == 0 else x[i:i + k]))
            if v is not None:
                return v
        return self.med[d]

    def costs(self, x):
        """the boundary cost of every position 0..len(x) of the symbol ids x"""
        g = lambda c: self.grp[c] if c < len(self.grp) else 0  # noqa: E731
        out = [0.0] * (len(x) + 1)
        for i in range(1, len(x)):
            cls = g(x[i - 1]) if g(x[i - 1]) == g(x[i]) else self.ncls - 1
            out[i] = self.w * (1.0 - self.cdf[511 * cls + self.side(0, x, i) + self.side(1, x, i)])
        return out


def varint(x):
    out = bytearray()
    while x >= 0x80:
        out.append(x & 0x7F | 0x80)
        x >>= 7
    out.append(x)
    return bytes(out)


def ids_of(t, text):
    return [t.idx[c] if c in t.idx else NONE_BASE + len(c.encode("utf-8")) for c in text]


def add_be(t, be):
    """a copy of a hand-made model (write_model) with the BENT section appended"""
    path = t.model_path[:-4] + "_be.bin"
    with open(path, "wb") as f:
        f.write(open(t.model_path, "rb").read() + be.section())
    return type(t)(path, t.alphabet)


def _has_be(path):
    return struct.pack("<I", BE_MAGIC) in open(path, "rb").read()


# ---------------------------------------------------------------- trained models


@pytest.fixture(scope="module")
def be_models(tmp_path_factory):
    """be models: folded cores, case_cores, case_affixes, case_affixes + code_len (pairwise DP),
    and order 3"""
    if BIN is None:
        pytest.skip("native/dict_search not built")
    tmp = tmp_path_factory.mktemp("be")
    return (train_tiny(tmp, False, text=TRAIN_ZH, tag="o", be_permille=4000),
            train_tiny(tmp, True, text=TRAIN_ZH, tag="c", be_permille=4000),
            train_tiny(tmp, True, text=TRAIN_CA + ZH * 40, tag="a", case_affixes=1, be_permille=8000),
            train_tiny(tmp, True, text=TRAIN_CA + ZH * 40, tag="al", case_affixes=1, code_len=1, code_w_permille=5,
                       be_permille=8000),
            train_tiny(tmp, True, text=TRAIN_ZH, tag="c3", be_permille=2000, be_order=3))


@needs_bin
def test_be_section(be_models, tmp_path):
    for t in be_models:
        assert _has_be(t.model_path), t.model_path
    t = train_tiny(tmp_path, True, text=TRAIN, tag="off", max_rounds=1)
    assert not _has_be(t.model_path)  # off: the old file layout


@needs_bin
@pytest.mark.parametrize("s", HARD_CA + HARD_ZH + ZH.splitlines())
def test_be_roundtrip(be_models, s):
    import bench_dictsearch as bd
    for t in be_models:
        enc = t.encode(s)
        assert t.decode(enc) == s, (t.model_path, s, enc)
        m = bd.DModel(t.model_path, t.alphabet)
        chunks = bd.CHUNK.findall(s)
        for ch, units in zip(chunks, m.encode_chunks(chunks, t.idx)):
            m.token_count_and_check(ch, units)


@needs_bin
def test_be_deterministic(be_models):
    s = " ".join(HARD + HARD_ZH)
    for t in be_models:
        assert t.encode(s) == t.encode(s)


# ---------------------------------------------------------------- hand-made models: the Chinese tie


def _zh_table(t, text, high, w):
    """order-1 table for text: the positions in `high` are word boundaries (entropy 12.5 bits on
    both sides), all others word-internal (0); percentile = score / 510"""
    x = ids_of(t, text)
    tab = {}
    for i in range(1, len(x)):
        v = 200 if i in high else 0
        tab[(0, (x[i - 1],))] = v
        tab[(1, (x[i],))] = v
    return BE(w, 1, tab, (0, 0), [s / 510 for s in range(511)])


@needs_bin
@pytest.mark.parametrize("kind", ["old", "cafx"])
def test_be_resolves_tie(tmp_path, kind):
    """观光局首度举办: [观光][局首度][举办] and [观光局][首度][举办] are both 3 tokens with 2 inner
    boundaries (after 光 vs after 局); the short pieces are cheaper by their costs, so without the
    boundary cost the first wins. With the words' boundaries after 局 and 度 (high branching
    entropy) the boundary cost moves it there. Also [观光|局|首][度][举办] (factor boundaries count)"""
    text = "观光局首度举办"
    alpha, idx, maps = symbols(text)
    base = [c for c in alpha if maps[0][idx[c]] == idx[c]]
    cores = ["观光局", "首度", "举办", "观光", "局首度", "局", "度"] + [c for c in base if c not in "局度"]
    prefixes, suffixes = ["", "观光"], ["", "首"]
    costs = [[12.0, 12.0, 8.0, 2.0, 2.0, 5.0, 6.0] + [15.0] * (len(cores) - 7), [1.0], [1.0]]
    vcost = [0.1] + [5.0] * (7 if kind == "cafx" else 3)
    old = write_model(tmp_path / "old.bin", alpha, idx, maps, kind, cores, [""], [""], costs, vcost)
    assert show(old, old.encode(text)) == ["|观光|", "|局首度|", "|举办|"]
    # boundaries after 局 (3) and 度 (5): [观光][局首度] pays 30 x (1 + 0.216), [观光局][首度] 30 x 0.43
    t = add_be(write_model(tmp_path / "be.bin", alpha, idx, maps, kind, cores, [""], [""], costs, vcost),
               _zh_table(old, text, {3, 5}, 30.0))
    assert show(t, t.encode(text)) == ["|观光局|", "|首度|", "|举办|"]
    assert t.decode(t.encode(text)) == text
    # a small weight leaves the tie-break to the costs
    t = add_be(write_model(tmp_path / "be_small.bin", alpha, idx, maps, kind, cores, [""], [""], costs, vcost),
               _zh_table(old, text, {3, 5}, 1.0))
    assert show(t, t.encode(text)) == ["|观光|", "|局首度|", "|举办|"]
    # the factored parse [观光|局|首][度][举办] (the cheapest without whole-word cores) places
    # boundaries after 光, 局, 首 and 度: the ones inside 观光 and 首度 cost too
    cores2 = ["观光局", "首度", "举办", "局", "度", "观光"] + [c for c in base if c not in "局度"]
    costs2 = [[8.5, 12.0, 8.0, 5.0, 6.0, 7.0] + [15.0] * (len(cores2) - 6), [1.0, 3.0], [1.0, 2.0]]
    old2 = write_model(tmp_path / "old2.bin", alpha, idx, maps, kind, cores2, prefixes, suffixes, costs2, vcost)
    assert show(old2, old2.encode(text)) == ["观光|局|首", "|度|", "|举办|"]
    t2 = add_be(write_model(tmp_path / "be2.bin", alpha, idx, maps, kind, cores2, prefixes, suffixes, costs2, vcost),
                _zh_table(old2, text, {3, 5}, 10.0))
    assert show(t2, t2.encode(text)) == ["|观光局|", "|首度|", "|举办|"]


# ---------------------------------------------------------------- brute force


def _setup(tmp_path, kind, seed, code, w):
    """a random hand-made model (kind "case" or "cafx"), with random code length
    statistics if `code`, and a random order-2 boundary table"""
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
    m = dict(tabs=(cores, prefixes, suffixes), costs=costs, vcost=vcost, nvar=nvar, code=code, w=w)
    cd = None
    if code:
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
        m.update(bo=bo, pairs=pairs, byte=rng.uniform(10, 30))
        cd = (w, m["byte"], bo, pairs)
    t = write_model(tmp_path / f"bf_{kind}_{seed}_{int(code)}.bin", alpha, idx, maps, kind, cores, prefixes, suffixes,
                    costs, vcost, code=cd)
    # random order-2 boundary table over the alphabet (and one unseen-context fallback per side)
    syms = [idx[c] for c in alpha]
    tab = {}
    for _ in range(60):
        d, k = rng.randint(0, 1), rng.randint(1, 2)
        tab[(d, tuple(rng.choice(syms) for _ in range(k)))] = rng.randint(0, 255)
    grp = [rng.randint(0, 2) for _ in alpha]  # random script groups: 3 groups + a change of group
    cdf = [v for _ in range(4) for v in sorted(rng.random() for _ in range(511))]
    be = BE(rng.uniform(30, 80) if code else rng.uniform(10, 40), 2, tab, (rng.randint(0, 255), rng.randint(0, 255)), cdf, grp)
    return add_be(t, be), m, be, t


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
                                wt = pid + written_core(cid, v, kind == "case", t.fold, t.upper_of, t.trad_of) + sid
                            if wt == ids:
                                out.append((v, pi, ci, si))
    return out


def _tok_bits(m, tok):
    if m["code"]:
        return _bits(m, *tok)
    v, p, c, s = tok
    return m["costs"][0][c] + m["vcost"][v] + m["costs"][1][p] + m["costs"][2][s]


def _brute(t, kind, m, be, text):
    """min (tokens + w x bits, bits + boundary costs) over all segmentations: every boundary a parse
    places inside the text (token ends and non-empty prefix / core ends) pays its cost"""
    w = m["w"] if m["code"] else 0.0
    bc = be.costs(ids_of(t, text))
    best = [(0.0, 0.0)] + [None] * len(text)
    for j in range(1, len(text) + 1):
        cands = []
        for i in range(j):
            if i == j - 1:
                nb = len(text[i].encode("utf-8"))
                byte = m["byte"] if m["code"] else 0.0
                cands.append((best[i][0] + nb * (1 + w * byte), best[i][1] + 1000 * nb + bc[j]))
            if all(ch in t.idx for ch in text[i:j]):
                for tok in _tokens_of_span(t, kind, m, text[i:j]):
                    v, p, c, s = tok
                    lp, ls = len(m["tabs"][1][p]), len(m["tabs"][2][s])
                    cut = {j} | ({i + lp} if lp else set()) | ({j - ls} if ls else set())
                    b = _tok_bits(m, tok)
                    cands.append((best[i][0] + 1 + w * b, best[i][1] + b + sum(bc[q] for q in cut)))
        best[j] = min(cands, key=lambda c: (round(c[0], 7), c[1]))
    return best[-1]


def _cost_of(t, m, be, text, enc):
    """(primary, secondary + boundary costs) of the encoder's tokens, chunk by chunk"""
    w = m["w"] if m["code"] else 0.0
    prim = sec = 0.0
    k = 0
    for ch in CHUNK.findall(text):
        bc = be.costs(ids_of(t, ch))
        pos = 0
        while pos < len(ch):
            v, p, c, s = enc[k]
            if c < 256:  # a fallback char: one token per byte
                nb = len(ch[pos].encode("utf-8"))
                k += nb
                pos += 1
                prim += nb * (1 + w * (m["byte"] if m["code"] else 0.0))
                sec += 1000 * nb + bc[pos]
                continue
            k += 1
            pre, core, suf = t.written_parts(v, p, c - 256, s)
            b = _tok_bits(m, (v, p, c - 256, s))
            prim += 1 + w * b
            cut = {pos + len(pre) + len(core) + len(suf)} | ({pos + len(pre)} if pre else set()) \
                | ({pos + len(pre) + len(core)} if suf else set())
            sec += b + sum(bc[q] for q in cut)
            pos += len(pre) + len(core) + len(suf)
    assert k == len(enc)
    return prim, sec


@needs_bin
@pytest.mark.parametrize("kind,code,w", [("case", False, 0.0), ("cafx", False, 0.0),
                                         ("case", True, 0.0), ("cafx", True, 0.0), ("cafx", True, 0.1)])
def test_be_brute_force(tmp_path, kind, code, w):
    """the encoder's parse minimises (primary, secondary + boundary costs) lexicographically:
    brute force over every segmentation, every (prefix, core, suffix, variation) writing each token,
    and bytes (the 3-state DP without code; the pairwise DP with code_len)"""
    rng = random.Random(11)
    checked = changed = 0
    for seed in range(3):
        t, m, be, t0 = _setup(tmp_path, kind, seed, code, w)
        pieces = [r for tab in m["tabs"] for r in tab if r] + ["A", "B", "Ab", "AB", "發", "é"]
        for _ in range(40):
            text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 4)))[:9]
            enc = t.encode(text)
            assert t.decode(enc) == text
            parts = [_brute(t, kind, m, be, ch) for ch in CHUNK.findall(text)]
            got, want = _cost_of(t, m, be, text, enc), (sum(p[0] for p in parts), sum(p[1] for p in parts))
            assert abs(got[0] - want[0]) < 1e-6, (text, got, want, show(t, [e for e in enc if e[2] >= 256]))
            assert abs(got[1] - want[1]) < 1e-6, (text, got, want, show(t, [e for e in enc if e[2] >= 256]))
            checked += 1
            changed += enc != t0.encode(text)
    assert checked == 120
    if w == 0.0:  # the boundary costs decide some of these ties (else the check would be vacuous)
        assert changed >= 10, changed
