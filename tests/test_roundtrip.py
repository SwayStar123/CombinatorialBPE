import os

import pytest

from cbpe import CombinatorialBPE, StandardBPE
from cbpe.tokenizers import V_CAP, V_NONE, V_UPPER, normalise_char

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

TRAIN = """The quick brown fox jumps over the lazy dog. THE QUICK BROWN FOX! Hello, world?
"Hello," she said. (See also: the dog's house.) NASA and iPhone; McDonald's.
Der Straße, DIE STRASSE. Ελληνικά ΟΔΟΣ οδός. Привет, МИР! मुझे हिंदी पसंद है। 東京は大きい。
İstanbul ǅemal ﬁne 123,456.78 e-mail — dash… “quotes” «guillemets» ¿Qué? ¡Sí!
""" * 30

HARD = [
    "", " ", "\n\n\n", "a", "A", "AB", "Ab", "aB", "hello", "Hello", "HELLO", "hElLo",
    "  leading and trailing  ", "x\ty\r\nz", "İstanbul İ", "ǅemal ǆ Ǆ", "ß SS ẞ", "ΟΔΟΣ Σ ς σ",
    "ﬁ ﬀ", "emoji 🎉🎉 ok", "́ combining é", "zzz qqq ✓✓✓", "123abc456", "...!?",
    "Unseen chars: ☃ 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 \x00\x7f", "tab\tthen,comma;semi", "ALLCAPS WITH, PUNCT!",
]


@pytest.fixture(scope="module")
def toks():
    return (StandardBPE.train(TRAIN, 400, "gpt2", min_char_freq=5),
            StandardBPE.train(TRAIN, 400, "cl100k", min_char_freq=5),
            CombinatorialBPE.train(TRAIN, 400, min_char_freq=5, min_affix_freq=5))


@pytest.mark.parametrize("s", HARD + TRAIN.splitlines())
def test_roundtrip(toks, s):
    for t in toks:
        assert t.decode(t.encode(s)) == s, (type(t).__name__, s)


def test_budget(toks):
    for t in toks:
        assert t.vocab_size <= 400


def test_factorisation(toks):
    comb = toks[2]
    enc = comb.encode(" Hello,")
    assert len(enc) == 1
    v, p, c, s = enc[0]
    assert v == V_CAP and comb.prefixes[p] == " " and comb.suffixes[s] == ","
    assert comb.core.vocab[c] == "hello"
    # the same core id is shared by all surface variants
    for surface, var in [("hello", V_NONE), (" HELLO!", V_UPPER), ("Hello.", V_CAP)]:
        ids = comb.encode(surface)
        assert [t[2] for t in ids] == [c] and ids[0][0] == var


def test_normalise_char():
    assert normalise_char("A") == ("a", True)
    assert normalise_char("a") == ("a", False)
    assert normalise_char("İ") == ("İ", False)  # lower() is 2 chars -> not folded
    assert normalise_char("ǅ") == ("ǅ", False)  # titlecase digraph does not round-trip


def test_save_load(tmp_path, toks):
    from cbpe import load
    for i, t in enumerate(toks):
        p = tmp_path / f"t{i}.json"
        t.save(p)
        t2 = load(p)
        for s in HARD:
            assert t2.encode(s) == t.encode(s)


@pytest.mark.parametrize("name", ["wiki_en", "wiki_de", "wiki_ru", "wiki_hi", "wiki_tr", "wiki_ja", "wiki_zh", "tinystories"])
def test_roundtrip_real_text(name):
    if not os.path.exists(os.path.join(DATA, f"{name}.test.txt")):
        pytest.skip(f"{name} not downloaded (scripts/download_data.py)")
    with open(os.path.join(DATA, f"{name}.train.txt"), encoding="utf-8") as f:
        train = f.read(2_000_000)
    with open(os.path.join(DATA, f"{name}.test.txt"), encoding="utf-8") as f:
        test = f.read(300_000)
    for t in (StandardBPE.train(train, 2000), CombinatorialBPE.train(train, 2000)):
        assert t.decode(t.encode(test)) == test


@pytest.mark.parametrize("kw", [dict(fold_case=False), dict(n_prefix=0, n_suffix=0), dict(punct_to_next=True),
                                dict(split_camel=True)])
def test_ablations_roundtrip(kw):
    t = CombinatorialBPE.train(TRAIN, 400, min_char_freq=5, min_affix_freq=5, **kw)
    assert t.vocab_size <= 400
    for s in HARD + TRAIN.splitlines():
        assert t.decode(t.encode(s)) == s


HAN = ["士林區舊名「八芝蘭」，位於臺灣臺北市北部。", "士林区旧名「八芝兰」，位于台湾台北市北部。",
       "皇后與後來的頭髮和發展", "台灣 臺灣 台湾", "中文English混合MIXED文字", "後后後后", "髮发發"]


@pytest.fixture(scope="module")
def han_tok():
    return CombinatorialBPE.train("\n".join(HAN) * 50 + TRAIN, 600, min_char_freq=5, min_affix_freq=5,
                                  fold_han=True)


@pytest.mark.parametrize("s", HAN + HARD)
def test_han_roundtrip(han_tok, s):
    assert han_tok.decode(han_tok.encode(s)) == s


def test_han_shares_cores(han_tok):
    from cbpe.tokenizers import V_TRAD
    trad, simp = han_tok.encode("臺灣"), han_tok.encode("台湾")
    assert [t[2] for t in trad] == [t[2] for t in simp]
    assert all(t[0] == V_TRAD for t in trad) and all(t[0] == V_NONE for t in simp)


def test_pretrained_roundtrip():
    from cbpe import load
    root = os.path.join(os.path.dirname(__file__), "..", "pretrained")
    for f in os.listdir(root):
        t = load(os.path.join(root, f))
        for s in HARD + HAN:
            assert t.decode(t.encode(s)) == s, (f, s)


CODE = """def getUserName(self, userId: int) -> str:
    \"\"\"Return the XMLHttpRequest user.\"\"\"
    if self.HTTPServer is None:
        return MAX_RETRIES + parseJSONResponse(userId)[0]
	public static void main(String[] args) { System.out.println("iPhone"); }
"""


@pytest.mark.parametrize("split_camel", [False, True])
def test_code_roundtrip(split_camel):
    from collections import Counter
    t = CombinatorialBPE.train(CODE * 40 + TRAIN, 500, min_char_freq=5, min_affix_freq=5, split_camel=split_camel)
    assert t.decode(t.encode(CODE)) == CODE
    stats = Counter()
    for w in ["getUserName", "parseJSONResponse", "XMLHttpRequest"]:
        t._encode_word(w, stats)
    assert stats["chars"] == len("getUserName" + "parseJSONResponse" + "XMLHttpRequest")
    if split_camel:  # every camel part has a clean case pattern -> no per-character fallback
        assert stats["fallback_chars"] == 0
