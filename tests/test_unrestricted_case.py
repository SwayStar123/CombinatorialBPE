"""Unrestricted Combinatorial BPE (native/dict_search) with and without case-preserving cores
(case_cores): canonical core spellings, the 5 relative variations, exact round trips."""
import os
import struct
import subprocess
import sys
from array import array
from collections import Counter

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from cbpe.unrestricted import NONE, UnrestrictedBPE, written_core  # noqa: E402

try:
    from cbpe.unrestricted import _binary
    BIN = _binary()
except FileNotFoundError:
    BIN = None
needs_bin = pytest.mark.skipif(BIN is None, reason="native/dict_search not built (cargo build --release)")

TRAIN = """Watch it on YouTube today. I watched YouTube videos with my iPhone. The iPhone and GitHub.
Push the code to GitHub: git push. const s = name.toUpperCase(); let t = s.toLowerCase();
class ProtoMessage extends Base { ProtoMessage copy() { return new ProtoMessage(); } }
Paris is in France. PARIS, FRANCE. the the the year years. HELLO World hello world Hello.
Der Straße, DIE STRASSE. ΟΔΟΣ οδός Σίσυφος ς σ. İstanbul ǅemal ǆ Ǆ. 发展 發展 中国 中國 北京.
""" * 40

HARD = [
    "", " ", "a", "A", "YouTube", "youtube", "YOUTUBE", "Youtube", "yOUTUBE", "YouTUBE", "iPhone",
    "IPhone", "IPHONE", "iphone", "iPHONE", "GitHub", "GITHUB", "github", "Github", "gitHub",
    " name.toUpperCase();", "TOUPPERCASE", "ProtoMessage", "protoMessage", "PROTOMESSAGE",
    "Paris", "PARIS", "paris", "pARIS", "İstanbul İ ı I i", "ǅemal ǆ Ǆ ǅ", "ß SS ẞ Straße STRASSE",
    "ΟΔΟΣ Σ ς σ οδός ΟΔΌΣ", "emoji 🎉🎉 ok", "发展 發展 中国 中國 中国人 發展ABC abc發展",
    "Unseen chars: ☃ 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 \x00\x7f", "x\ty\r\nz  ", "HELLO World hello world Hello hELLO",
]
PARAMS = dict(max_rounds=3, min_freq=5, first_expand_permille=4000, prune_step_permille=150,
              lambda_permille=20, rerank_mult=4, rerank_occ=300, refactor=1, price_permille=100,
              prod_k=50, threads=4, partner_n0=1000, sig_k=50, swap_share_permille=100,
              protect_chars=1, sample_t=0, fine_rounds=0)


def train_tiny(tmp_path, case, vocab=1400, text=TRAIN, tag="", **extra):
    """a small model trained by the binary on `text` (hybrid chunks), as bench_dictsearch.main does;
    `extra`: more trainer params (case_affixes, ...)"""
    import bench_dictsearch as bd
    params = dict(bd.PARAMS, **PARAMS, case_cores=int(case))
    params.update(extra)
    params["core_price_permille"] = params["price_permille"]
    alpha, idx, maps = bd.build_symbols(Counter(text), vocab, fold=True)
    segs = Counter(bd.CHUNK.findall(text))
    inp, model = tmp_path / f"in{int(case)}{tag}.bin", tmp_path / f"model{int(case)}{tag}.bin"
    with open(inp, "wb") as f:
        f.write(struct.pack("<I", len(alpha)))
        for m in maps:
            array("I", m).tofile(f)
        array("I", [len(c.encode("utf-8")) for c in alpha]).tofile(f)
        array("I", [bd.char_class(c) for c in alpha]).tofile(f)
        f.write(struct.pack(f"<{1 + len(params)}I", vocab - 4 - 256, *params.values()))
        for _ in range(2):  # train and dev: the same chunks
            keys = list(segs)
            bd.write_corpus(f, keys, idx, [segs[s] for s in keys])
        f.write(struct.pack("<I", 0))
        bd.write_table(f, [], idx)
        bd.write_table(f, [], idx)
        f.write(struct.pack("<II", 0, 1))
        array("I", [bd.script_of(c) for c in alpha]).tofile(f)
    subprocess.run([BIN, "train", str(inp), str(model)], check=True, capture_output=True)
    return UnrestrictedBPE(str(model), alpha)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    if BIN is None:
        pytest.skip("native/dict_search not built")
    tmp = tmp_path_factory.mktemp("case")
    return train_tiny(tmp, False), train_tiny(tmp, True)


@needs_bin
def test_sizes(models):
    old, new = models
    assert (old.case, new.case) == (False, True)
    assert old.sizes["variation"] == 4 and new.sizes["variation"] == 5


@needs_bin
@pytest.mark.parametrize("s", HARD + TRAIN.splitlines()[:5])
def test_roundtrip(models, s):
    for t in models:
        enc = t.encode(s)
        assert t.decode(enc) == s, (t.case, s, enc)
        if not t.case:
            assert all(v < 4 for v, _, c, _ in enc if c >= 256)


@needs_bin
def test_mixed_case_cores(models):
    old, new = models
    for w in ("iPhone", "GitHub", "ProtoMessage"):
        assert w in new.cores, w  # the canonical spelling is the stored core
        assert w not in old.cores  # old cores are folded
    c = 256 + new.cores.index("iPhone")
    for text, v in (("iPhone", 0), ("IPhone", 1), ("IPHONE", 2), ("iphone", 4)):
        enc = new.encode(text)
        assert [(t[0], t[2]) for t in enc] == [(v, c)], (text, enc)
    # no variation writes iPHONE from iPhone: more tokens, still exact
    assert len(new.encode("iPHONE")) > 1 and new.decode(new.encode("iPHONE")) == "iPHONE"
    # ordinary words keep lowercase canonical spellings
    assert "hello" in new.cores and "Hello" not in new.cores


@needs_bin
def test_deterministic(models):
    for t in models:
        assert t.encode(" ".join(HARD)) == t.encode(" ".join(HARD))


def test_written_core_maps():
    """decode by per-symbol maps: every variation keeps the length, chars without the needed
    form stay as they are (ß has no one-char uppercase, ς none either)"""
    import bench_dictsearch as bd
    text = "aAbBßẞσΣςiIİıǅǆǄ发發x"
    alpha, idx, (fold, _, upper_of, trad_of) = bd.build_symbols(Counter(text * 20), 10_000, fold=True)
    ids = lambda s: [idx[c] for c in s]  # noqa: E731
    w = lambda s, v, case=True: "".join(alpha[c] for c in written_core(ids(s), v, case, fold, upper_of, trad_of))  # noqa: E731
    assert [w("aB发", v) for v in range(5)] == ["aB发", "AB发", "AB发", "aB發", "ab发"]
    assert [w("ßσς", v) for v in range(5)] == ["ßσς", "ßσς", "ßΣς", "ßσς", "ßσς"]
    assert [w("ıi", v) for v in range(5)] == ["ıi", "ıi", "ıI", "ıi", "ıi"]  # I folds to i, not ı
    assert [w("ǆ", v) for v in range(5)] == ["ǆ", "Ǆ", "Ǆ", "ǆ", "ǆ"]  # ǅ (title case) is not a fold
    assert w("ab发", 1, case=False) == "Ab发" and w("ab发", 3, case=False) == "ab發"
    assert fold[idx["İ"]] == idx["İ"] and fold[idx["ẞ"]] == idx["ẞ"]  # not folded: no exact one-char inverse
    assert upper_of[idx["ß"]] == NONE


def test_old_model_still_loads():
    """a model file from before case_cores (if present): 4 variations, exact round trips"""
    path = os.path.join(ROOT, "results", "tokenizers", "mix_65536_dict_search_hybrid_sigab50_d80.json")
    if BIN is None or not os.path.exists(path):
        pytest.skip("reference model or binary not available")
    from cbpe import load
    tok = load(path)
    assert not tok.case and tok.sizes["variation"] == 4
    for s in HARD:
        assert tok.decode(tok.encode(s)) == s


# ---------------------------------------------------------------- case_affixes (whole-token variation)

CA_EXTRA = """Die Verbindung ist gut. Die verbindung, VERBINDUNG. Ein Verkauf, verkaufen, Verkäufer.
Unhappy UNHAPPY unhappy unhappiness Unhappiness UNHAPPINESS. getName setName userName username
getname. YEARS Years years, WORDS Words words. Users USERS users. the Year the YEAR.
getName getNameList GetName ResponseWriter responseWriter filename username fileName.
Das Rathaus, RATHAUS, rathaus. Ein Krankenhaus, das Haus, HAUS, Hausarzt.
"""
TRAIN_CA = TRAIN + CA_EXTRA * 40
HARD_CA = HARD + ["Unhappy UNHAPPY unhappy getName username YEARS Years years",
                  " Unhappy", "_name _Name", "1st 1St", "uNHAPPY", "Verbindung VERBINDUNG verbindung vERBINDUNG",
                  "getNAME GetName GETNAME",
                  "getName getNameList GetName ResponseWriter responseWriter filename username Rathaus RATHAUS",
                  "Rathaus Krankenhaus", "iPhone Iphone IPhone", "_Name _name"] + CA_EXTRA.splitlines()


# a small vocabulary without row prices: many affix rows, so case twins appear without case_affixes
SMALL = dict(vocab=500, price_permille=0, prod_k=0, partner_n0=0, sig_k=0, lambda_permille=0)


@pytest.fixture(scope="module")
def ca_models(tmp_path_factory):
    """case_cores alone and with case_affixes on the same text (small vocabulary), case_affixes
    with mark_rule and the pairwise DP (lamc), and case_affixes with the default test params"""
    if BIN is None:
        pytest.skip("native/dict_search not built")
    tmp = tmp_path_factory.mktemp("cafx")
    return (train_tiny(tmp, True, text=TRAIN_CA, tag="c", **SMALL),
            train_tiny(tmp, True, text=TRAIN_CA, tag="a", case_affixes=1, **SMALL),
            train_tiny(tmp, True, text=TRAIN_CA, tag="am", case_affixes=1, mark_rule=1, **dict(SMALL, lambda_permille=20, lamc_permille=100)),
            train_tiny(tmp, True, text=TRAIN_CA, tag="ad", case_affixes=1))


def test_written_token():
    """whole-token variations: Capitalised = the first char with an uppercase form"""
    import bench_dictsearch as bd
    from cbpe.unrestricted import written_token
    text = " un_happyNameSs发"
    alpha, idx, (fold, _, upper_of, trad_of) = bd.build_symbols(Counter(text * 20 + "UNHAPYSEM发發" * 20), 10_000, fold=True)
    ids = lambda s: [idx[c] for c in s]  # noqa: E731
    w = lambda parts, v: "".join(alpha[c] for c in written_token([ids(p) for p in parts], v, fold, upper_of, trad_of))  # noqa: E731
    assert [w([" un", "happy", ""], v) for v in range(5)] == [" unhappy", " Unhappy", " UNHAPPY", " unhappy", " unhappy"]
    assert [w(["", "_happy", "Name"], v) for v in range(5)] == ["_happyName", "_HappyName", "_HAPPYNAME", "_happyName", "_happyname"]
    assert w(["", "发", "S"], 3) == "發S" and w(["", "发", "s"], 1) == "发S"
    # camelCase (5): core and suffix each capitalised, the prefix as stored; PascalCase (6): every
    # part; Title (7): the token's first cased char up, all else folded, whatever is stored
    assert [w(["un", "happy", "name"], v) for v in (5, 6, 7)] == ["unHappyName", "UnHappyName", "Unhappyname"]
    assert [w([" un", "_happy", "S"], v) for v in (5, 6, 7)] == [" un_HappyS", " Un_HappyS", " Un_happys"]
    assert [w(["", "happy", "Name"], v) for v in (5, 6, 7)] == ["HappyName", "HappyName", "Happyname"]
    assert w(["发", "", ""], 7) == "发" and w(["发s", "", ""], 5) == "发s" and w(["发s", "", ""], 6) == "发S"


@needs_bin
def test_case_affixes_model(ca_models):
    cc, ca = ca_models[:2]
    assert ca.case and ca.case_affixes and ca.sizes["variation"] == 8
    assert cc.case and not cc.case_affixes


@needs_bin
@pytest.mark.parametrize("s", HARD_CA + TRAIN.splitlines()[:5])
def test_case_affixes_roundtrip(ca_models, s):
    import bench_dictsearch as bd
    for t in ca_models:
        enc = t.encode(s)
        assert t.decode(enc) == s, (t.case_affixes, s, enc)
        assert all(v < 8 for v, _, c, _ in enc if c >= 256)
    # the bench decoder agrees (it asserts a lossless rebuild of every chunk)
    for t in ca_models[1:]:
        m = bd.DModel(t.model_path, t.alphabet)
        chunks = bd.CHUNK.findall(s)
        for ch, units in zip(chunks, m.encode_chunks(chunks, t.idx)):
            m.token_count_and_check(ch, units)


def _twins(t, table):
    """affix rows of one table that are spellings of the same folded string"""
    fold = lambda s: tuple(t.fold[t.idx[c]] for c in s)  # noqa: E731
    by = {}
    for r in table:
        by.setdefault(fold(r), []).append(r)
    return [v for v in by.values() if len(v) > 1]


@needs_bin
def test_case_affixes_no_twins(ca_models):
    """one row per folded affix: ' Un' / ' un', 'S' / 's' cannot both be rows"""
    cc = ca_models[0]
    # (the same text and params without case_affixes do make twin rows: ' u' / ' U', 's' / 'S', ...)
    assert _twins(cc, cc.prefixes) + _twins(cc, cc.suffixes), "no twins even without case_affixes"
    for t in ca_models[1:]:
        assert _twins(t, t.prefixes) == [] and _twins(t, t.suffixes) == []
        assert len(set(t.prefixes)) == len(t.prefixes) and len(set(t.suffixes)) == len(t.suffixes)


@needs_bin
def test_case_affixes_whole_token(ca_models):
    """a capitalised or all-caps word uses the same rows as the lowercase one (affixes included),
    only the variation differs; with the default test params each is one token"""
    for t in ca_models[1:]:
        for ws in (("unhappy", "Unhappy", "UNHAPPY"), ("years", "Years", "YEARS"), ("verbindung", "Verbindung", "VERBINDUNG")):
            encs = [t.encode(" " + w) for w in ws]
            rows = [[(p, c, s) for _, p, c, s in e] for e in encs]
            assert rows[0] == rows[1] == rows[2], (ws, [[t.written_parts(v, p, c - 256, s) for v, p, c, s in e] for e in encs])
    big = ca_models[3]
    for w in ("Unhappy", "UNHAPPY", "unhappy", "YEARS", "Years", "years", "Verbindung", "VERBINDUNG"):
        assert len(big.encode(" " + w)) == 1, w


@needs_bin
def test_case_affixes_minimal(ca_models):
    """the encoder's token count is the minimum over every (prefix, core, suffix, variation) that
    writes each token (brute force; the small model has no affix or other extra costs)"""
    import bench_dictsearch as bd
    from cbpe.unrestricted import written_token
    t = ca_models[1]
    fold = lambda syms: tuple(t.fold[c] for c in syms)  # noqa: E731
    tabs = [{fold(s): s for s in tab} for tab in (t.prefix_syms, t.core_syms, t.suffix_syms)]
    def writable(span):
        for a in range(len(span)):
            p = tabs[0].get(fold(span[:a]))
            if p is None:
                continue
            for b in range(a + 1, len(span) + 1):
                c, s = tabs[1].get(fold(span[a:b])), tabs[2].get(fold(span[b:]))
                if c is not None and s is not None and any(
                        written_token([p, c, s], v, t.fold, t.upper_of, t.trad_of) == span for v in range(8)):
                    return True
        return False
    n_tok = 0
    for chunk in {ch for s in HARD_CA + TRAIN_CA.splitlines()[:8] for ch in bd.CHUNK.findall(s)}:
        ids = [t.idx.get(ch) for ch in chunk]
        best = [0] + [None] * len(ids)
        for i in range(len(ids)):
            nb = best[i] + len(chunk[i].encode("utf-8"))
            best[i + 1] = nb if best[i + 1] is None else min(best[i + 1], nb)
            for j in range(i + 1, len(ids) + 1):
                if None in ids[i:j]:
                    break
                if (best[j] is None or best[i] + 1 < best[j]) and writable(ids[i:j]):
                    best[j] = best[i] + 1
        n_tok += best[-1]
        assert len(t.encode(chunk)) == best[-1], chunk
    assert n_tok > 0


def _cafx_model(path, prefixes, cores, suffixes, text):
    """a hand-made case_affixes model file (rows in their canonical spellings; every char of
    `text` is also a core), as the trainer saves one; all row costs 1, variation v costs 1 + v"""
    import bench_dictsearch as bd
    text = text + text.upper() + text.lower()
    alpha, idx, maps = bd.build_symbols(Counter(text * 20), 10_000, fold=True)
    fold = maps[0]
    cores = cores + [c for c in alpha if fold[idx[c]] == idx[c] and c not in cores]
    with open(path, "wb") as f:
        f.write(struct.pack("<II", int.from_bytes(b"CAFX", "little"), len(alpha)))
        for m in maps:
            array("I", m).tofile(f)
        for tab in (cores, prefixes, suffixes):
            f.write(struct.pack("<I", len(tab)))
            for r in tab:
                f.write(struct.pack(f"<I{len(r)}I", len(r), *[idx[c] for c in r]))
        n = len(cores) + len(prefixes) + len(suffixes)
        f.write(struct.pack(f"<{n}d", *[1.0] * n))
        f.write(struct.pack("<8d", *[1.0 + v for v in range(8)]))
        f.write(struct.pack("<d", 0.0))  # delta
        array("I", [len(c.encode("utf-8")) for c in alpha]).tofile(f)
        f.write(struct.pack("<d", 0.0))  # lambda
    return UnrestrictedBPE(str(path), alpha)


@needs_bin
def test_case_affixes_part_variations(tmp_path):
    """camelCase / PascalCase / Title write one token from lowercase (or Haus) rows, exactly"""
    text = " getNameList GetName filename ResponseWriter Rathaus Krankenhaus RATHAUS"
    t = _cafx_model(tmp_path / "hand.bin", ["", "get", "file", "response", " rat", " kranken"],
                    ["name", "writer", "Haus"], ["", "list"], text)
    want = {"getName": 5, "GetName": 6, "getname": 0, "GETNAME": 2, "Getname": 1, "filename": 0,
            "fileName": 5, "FileName": 6, "getNameList": 5, "GetNameList": 6, "ResponseWriter": 6,
            "responseWriter": 5, " Rathaus": 7, " rathaus": 4, " RATHAUS": 2, " ratHaus": 0,
            " Krankenhaus": 7, " krankenHaus": 0}
    for s, v in want.items():
        enc = t.encode(s)
        assert t.decode(enc) == s
        assert len(enc) == 1 and enc[0][0] == v, (s, enc, [t.written_parts(*e[:1], e[1], e[2] - 256, e[3]) for e in enc])
    # the same core row writes name in getName and filename
    assert t.encode("getName")[0][2] == t.encode("filename")[0][2] == 256 + t.cores.index("name")
    # Rathaus is one token with the canonical spelling Haus
    assert t.cores[t.encode(" Rathaus")[0][2] - 256] == "Haus"
    # spellings no variation writes are not matched (still exact)
    for s in ("gEtName", "getNAMEList", " RatHAUS", "rEsponseWriter"):
        enc = t.encode(s)
        assert len(enc) > 1 and t.decode(enc) == s, s
