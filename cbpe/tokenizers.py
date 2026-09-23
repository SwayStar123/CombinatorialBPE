"""Standard BPE baselines and the Combinatorial BPE tokenizer.

Combinatorial token = (variation, prefix, core, suffix)

* variation: hand-coded case transform applied to the core: 0 = as-is, 1 = Capitalised,
  2 = ALL CAPS.  This is the only language-specific-ish knowledge, and it is driven by
  Unicode case mappings, so it works for any cased script (Latin, Cyrillic, Greek, ...)
  and is simply unused for uncased ones (Devanagari, CJK, ...).
* prefix: a learned string of non-letter characters glued to the left of the word
  (" ", "\n\n", " (", ' "', " «", ...).
* core: a learned BPE piece over case-normalised text.
* suffix: a learned string of non-letter, non-space characters glued to the right
  (",", ".", "),", '."', "?", ...).

A word that needs several core pieces becomes several tokens; the prefix rides on the
first one and the suffix on the last one. Everything is lossless (byte fallback).

The vocabulary budget is shared: N = |variations| + |prefixes| + |suffixes| + |core|,
i.e. the same number of embedding rows as a standard BPE with vocab N. How the budget
is split between affixes and core merges is learned: an affix is kept only if it saves
more tokens on the training data than the core merge it displaces.
"""
from __future__ import annotations

import json
import os
from collections import Counter

import regex

from .bpe import N_BYTES, CharBPE

# --------------------------------------------------------------------- patterns
# Letters include combining marks (\p{M}) so Indic scripts are not shredded; applied
# identically to all tokenizers for fairness.
GPT2_PAT = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}\p{M}]+| ?\p{N}+| ?[^\s\p{L}\p{M}\p{N}]+|\s+(?!\S)|\s+""")
CL100K_PAT = regex.compile(
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{M}\p{N}]?[\p{L}\p{M}]+|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""")
# unit = <non-alnum run> <letters | digits> <non-space non-alnum run>   |   trailing junk
COMB_PAT = regex.compile(
    r"([^\p{L}\p{M}\p{N}]*)([\p{L}\p{M}]+|\p{N}+)([^\s\p{L}\p{M}\p{N}]*)|([^\p{L}\p{M}\p{N}]+)")
# variant: trailing punctuation attaches to the *next* word's prefix (", which") -> no suffixes
COMB_PAT_LEFT = regex.compile(r"([^\p{L}\p{M}\p{N}]*)([\p{L}\p{M}]+|\p{N}+)()|([^\p{L}\p{M}\p{N}]+)")


class StandardBPE:
    """Regex pretokenisation + char-level BPE with byte fallback (GPT-2 / cl100k style)."""

    def __init__(self, bpe: CharBPE, pattern: str = "gpt2"):
        self.bpe = bpe
        self.pattern = pattern
        self.pat = {"gpt2": GPT2_PAT, "cl100k": CL100K_PAT}[pattern]

    @property
    def vocab_size(self):
        return len(self.bpe)

    @staticmethod
    def train(text: str, vocab_size: int, pattern="gpt2", min_char_freq=20, verbose=False):
        pat = {"gpt2": GPT2_PAT, "cl100k": CL100K_PAT}[pattern]
        counts = Counter(m.group() for m in pat.finditer(text))
        return StandardBPE(CharBPE.train(counts, vocab_size, min_char_freq, verbose), pattern)

    def encode(self, text: str) -> list[int]:
        enc = self.bpe.encode
        return [i for w in self.pat.findall(text) for i in enc(w)]

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.bpe.id_bytes(i) for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "standard", "pattern": self.pattern, **self.bpe.to_dict()}, f)


# ------------------------------------------------------------------ case logic
VARIATIONS = ["as-is", "Capitalised", "UPPER", "Traditional"]
V_NONE, V_CAP, V_UPPER, V_TRAD = 0, 1, 2, 3
F_UP, F_TRAD = 1, 2  # per-character fold flags

# Chinese script variation (optional): cores are stored in Simplified; a Traditional char is
# folded only when converting its Simplified form back gives exactly that char (OpenCC tables,
# built by scripts/build_han_tables.py), so the fold is exactly invertible.
with open(os.path.join(os.path.dirname(__file__), "data", "han_st.json"), encoding="utf-8") as _f:
    _han = json.load(_f)
HAN_TO_TRAD: dict[str, str] = _han["to_trad"]
HAN_FOLD: dict[str, str] = _han["fold"]


def normalise_char(c: str) -> tuple[str, bool]:
    """-> (normalised char, was_upper). Only folds when the fold round-trips exactly."""
    lo = c.lower()
    if lo != c and len(lo) == 1 and lo.upper() == c:
        return lo, True
    return c, False


def fold_char(c: str, fold_case: bool, fold_han: bool) -> tuple[str, int]:
    """-> (normalised char, flag in {0, F_UP, F_TRAD})."""
    if fold_case:
        lo, up = normalise_char(c)
        if up:
            return lo, F_UP
    if fold_han:
        s = HAN_FOLD.get(c)
        if s is not None:
            return s, F_TRAD
    return c, 0


def piece_variation(piece: str, flags: list[int]):
    """Variation that turns the normalised piece back into the original, or None if the
    piece needs per-character treatment."""
    if not any(flags):
        return V_NONE
    if all(f != F_TRAD for f in flags):
        up = [f == F_UP for f in flags]
        if up[0] and not any(up[1:]):
            return V_CAP
        if all(up):
            return V_UPPER
        return None
    if all(f != F_UP for f in flags):
        # every convertible char must have been Traditional, else decoding would convert it too
        if all(f == F_TRAD for c, f in zip(piece, flags) if c in HAN_TO_TRAD):
            return V_TRAD
    return None


def apply_variation(v: int, s: str) -> str:
    if v == V_CAP:
        return s[0].upper() + s[1:]
    if v == V_UPPER:
        return s.upper()
    if v == V_TRAD:
        return "".join(HAN_TO_TRAD.get(c, c) for c in s)
    return s


def unit_parts(text: str, punct_to_next: bool = False):
    """Yield (pre, word, suf) for every unit; pure-junk units come back as (junk, '', '')."""
    for m in (COMB_PAT_LEFT if punct_to_next else COMB_PAT).finditer(text):
        if m.group(4) is not None:
            yield m.group(4), "", ""
        else:
            yield m.group(1), m.group(2), m.group(3)


def longest_suffix_in(s: str, table: dict[str, int], max_len: int) -> tuple[int, str]:
    """Longest suffix of s found in table -> (id, remaining head of s)."""
    for k in range(min(len(s), max_len), 0, -1):
        i = table.get(s[-k:])
        if i is not None:
            return i, s[:-k]
    return 0, s


def longest_prefix_in(s: str, table: dict[str, int], max_len: int) -> tuple[int, str]:
    for k in range(min(len(s), max_len), 0, -1):
        i = table.get(s[:k])
        if i is not None:
            return i, s[k:]
    return 0, s


def n_variations(fold_case: bool, fold_han: bool) -> int:
    """Rows in the variation table (ids are fixed, so Traditional always sits at index 3)."""
    return 4 if fold_han else (3 if fold_case else 1)


class CombinatorialBPE:
    def __init__(self, core: CharBPE, prefixes: list[str], suffixes: list[str], fold_case: bool = True,
                 punct_to_next: bool = False, fold_han: bool = False):
        assert prefixes[0] == "" and suffixes[0] == ""
        self.core = core
        self.fold_case = fold_case
        self.fold_han = fold_han
        self.punct_to_next = punct_to_next
        self.prefixes = list(prefixes)
        self.suffixes = list(suffixes)
        self.p2id = {p: i for i, p in enumerate(self.prefixes) if p}
        self.s2id = {s: i for i, s in enumerate(self.suffixes) if s}
        self.max_p = max(map(len, self.prefixes))
        self.max_s = max(map(len, self.suffixes))
        self._cache: dict[tuple[str, str, str], list[tuple[int, int, int, int]]] = {}

    @property
    def sizes(self):
        return {"variation": n_variations(self.fold_case, self.fold_han), "prefix": len(self.prefixes),
                "core": len(self.core), "suffix": len(self.suffixes)}

    @property
    def vocab_size(self):
        """Embedding rows needed (sum of factor tables) -> comparable to standard vocab size."""
        return sum(self.sizes.values())

    # --------------------------------------------------------------- encode
    def _encode_word(self, word: str) -> list[tuple[int, int]]:
        """-> list of (variation, core_id)."""
        if not (self.fold_case or self.fold_han):
            return [(V_NONE, c) for c in self.core.encode(word)]
        norm, flags = [], []
        for ch in word:
            n, fl = fold_char(ch, self.fold_case, self.fold_han)
            norm.append(n)
            flags.append(fl)
        core = self.core
        out, pos = [], 0
        for piece in core.split("".join(norm)):
            n = len(piece)
            f = flags[pos:pos + n]
            orig = word[pos:pos + n]
            pos += n
            cid = core.tok2id.get(piece)
            if cid is None:  # char outside alphabet -> raw UTF-8 bytes of the original char
                out.extend((V_NONE, b) for b in orig.encode("utf-8"))
                continue
            v = piece_variation(piece, f)
            if v is not None:
                out.append((v, cid))
            else:  # mixed case / script inside one piece (e.g. "cDo"): per character
                for ch, o, fl in zip(piece, orig, f):
                    ci = core.tok2id.get(ch)
                    if ci is None:
                        out.extend((V_NONE, b) for b in o.encode("utf-8"))
                    else:
                        out.append(({F_UP: V_CAP, F_TRAD: V_TRAD}.get(fl, V_NONE), ci))
        return out

    def encode_unit(self, pre: str, word: str, suf: str) -> list[tuple[int, int, int, int]]:
        key = (pre, word, suf)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        toks = []
        if not word:  # junk run with no letters/digits (e.g. end of text)
            toks = [(V_NONE, 0, c, 0) for c in self.core.encode(pre)]
        else:
            p, pre_rest = longest_suffix_in(pre, self.p2id, self.max_p)
            s, suf_rest = longest_prefix_in(suf, self.s2id, self.max_s)
            toks.extend((V_NONE, 0, c, 0) for c in self.core.encode(pre_rest))
            body = [[v, 0, c, 0] for v, c in self._encode_word(word)]
            body[0][1] = p
            body[-1][3] = s
            toks.extend(map(tuple, body))
            toks.extend((V_NONE, 0, c, 0) for c in self.core.encode(suf_rest))
        if len(self._cache) < 3_000_000:
            self._cache[key] = toks
        return toks

    def encode(self, text: str) -> list[tuple[int, int, int, int]]:
        out = []
        for parts in unit_parts(text, self.punct_to_next):
            out.extend(self.encode_unit(*parts))
        return out

    # --------------------------------------------------------------- decode
    def token_bytes(self, tok) -> bytes:
        v, p, c, s = tok
        t = self.core.vocab[c]
        core = t if isinstance(t, bytes) else apply_variation(v, t).encode("utf-8")
        return self.prefixes[p].encode("utf-8") + core + self.suffixes[s].encode("utf-8")

    def token_str(self, tok) -> str:
        return self.token_bytes(tok).decode("utf-8", errors="replace")

    def decode(self, toks) -> str:
        return b"".join(self.token_bytes(t) for t in toks).decode("utf-8", errors="replace")

    # ---------------------------------------------------------------- train
    @staticmethod
    def _core_corpus(word_counts, pre_counts, suf_counts, junk_counts, p2id, s2id, fold_case=True,
                     fold_han=False):
        """String frequencies the core BPE must learn to encode, given an affix set."""
        max_p = max(map(len, p2id), default=0)
        max_s = max(map(len, s2id), default=0)
        corpus: Counter = Counter()
        for w, c in word_counts.items():
            corpus["".join(fold_char(ch, fold_case, fold_han)[0] for ch in w)] += c
        for pre, c in pre_counts.items():
            rest = longest_suffix_in(pre, p2id, max_p)[1]
            if rest:
                corpus[rest] += c
        for suf, c in suf_counts.items():
            rest = longest_prefix_in(suf, s2id, max_s)[1]
            if rest:
                corpus[rest] += c
        corpus.update(junk_counts)
        return corpus

    @staticmethod
    def train(text: str, vocab_size: int, min_char_freq=20, max_affix_candidates=2000,
              min_affix_freq=10, max_affix_len=16, n_prefix=None, n_suffix=None,
              fold_case=True, punct_to_next=False, fold_han=False, verbose=False) -> "CombinatorialBPE":
        """If n_prefix / n_suffix are None the affix/core budget split is learned.
        fold_case=False / n_prefix=n_suffix=0 give the ablations (affixes only / case only)."""
        word_counts, pre_counts, suf_counts, junk_counts = Counter(), Counter(), Counter(), Counter()
        for pre, word, suf in unit_parts(text, punct_to_next):
            if word:
                word_counts[word] += 1
                pre_counts[pre] += 1
                suf_counts[suf] += 1
            else:
                junk_counts[pre] += 1

        def candidates(counts):
            return [(a, c) for a, c in counts.most_common()
                    if a and len(a) <= max_affix_len and c >= min_affix_freq][:max_affix_candidates]

        p_cand, s_cand = candidates(pre_counts), candidates(suf_counts)
        n_var = n_variations(fold_case, fold_han)
        budget = vocab_size - n_var - 2  # 2 = the "no prefix"/"no suffix" rows

        if n_prefix is None or n_suffix is None:
            # pass 1: generous affix set, core trained to the full remaining budget
            p2id = {a: i + 1 for i, (a, _) in enumerate(p_cand)}
            s2id = {a: i + 1 for i, (a, _) in enumerate(s_cand)}
            corpus = CombinatorialBPE._core_corpus(word_counts, pre_counts, suf_counts, junk_counts, p2id, s2id,
                                                   fold_case, fold_han)
            if verbose:
                print(f"  pass 1: {len(p_cand)} prefix / {len(s_cand)} suffix candidates", flush=True)
            core1 = CharBPE.train(corpus, budget, min_char_freq, verbose)
            # every entry "saves" roughly its count in tokens; keep the best `budget` of them
            n_free = budget - N_BYTES - len(core1.alphabet)
            gains = [(c, "m") for c in core1.merge_counts]
            gains += [(c, "p") for _, c in p_cand] + [(c, "s") for _, c in s_cand]
            gains.sort(key=lambda g: -g[0])
            kept = Counter(kind for _, kind in gains[:n_free])
            n_prefix, n_suffix = kept["p"], kept["s"]
            if verbose:
                print(f"  learned split: {n_prefix} prefixes, {n_suffix} suffixes, {kept['m']} core merges",
                      flush=True)

        prefixes = [""] + [a for a, _ in p_cand[:n_prefix]]
        suffixes = [""] + [a for a, _ in s_cand[:n_suffix]]
        p2id = {a: i for i, a in enumerate(prefixes) if a}
        s2id = {a: i for i, a in enumerate(suffixes) if a}
        corpus = CombinatorialBPE._core_corpus(word_counts, pre_counts, suf_counts, junk_counts, p2id, s2id,
                                               fold_case, fold_han)
        core_size = vocab_size - n_var - len(prefixes) - len(suffixes)
        core = CharBPE.train(corpus, core_size, min_char_freq, verbose)
        return CombinatorialBPE(core, prefixes, suffixes, fold_case, punct_to_next, fold_han)

    # ------------------------------------------------------------------- io
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "combinatorial", "prefixes": self.prefixes, "suffixes": self.suffixes,
                       "fold_case": self.fold_case, "punct_to_next": self.punct_to_next, "fold_han": self.fold_han,
                       **self.core.to_dict()}, f, ensure_ascii=False)


def load(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if d["type"] == "standard":
        return StandardBPE(CharBPE.from_dict(d), d["pattern"])
    return CombinatorialBPE(CharBPE.from_dict(d), d["prefixes"], d["suffixes"], d.get("fold_case", True),
                            d.get("punct_to_next", False), d.get("fold_han", False))
