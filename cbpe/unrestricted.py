"""Unrestricted Combinatorial BPE (trained by native/dict_search, experiments/bench_dictsearch.py).

Prefixes, cores and suffixes are arbitrary strings; only the core carries the variation. Text is
split into whitespace chunks (leading whitespace attached) and every chunk is segmented by the
exact minimum-cost DP in the Rust binary, so this class needs `native/dict_search` built
(`cargo build --release`).

Token layout matches CombinatorialBPE: (variation, prefix, core, suffix), where core ids 0-255 are
the UTF-8 byte-fallback rows and learned cores start at 256; prefix/suffix 0 is the empty string.

Three model kinds (see the header of native/dict_search/src/main.rs). Old models store cores
case/Han-folded with 4 variations (as-is, Capitalised, UPPER, Traditional). case_cores models
(the file starts with "CASE") store every core's canonical spelling (YouTube, iPhone, the) with 5
variations relative to it: as stored, Capitalised, UPPER, Traditional, lower. case_affixes models
(the file starts with "CAFX") store canonical spellings for prefixes and suffixes too, and the
variation applies to the whole token: prefix + core + suffix as stored, then rewritten as one
string (Capitalised = its first char with an uppercase form made uppercase), with two more
variations: camelCase (core and suffix each capitalised, get+name -> getName) and PascalCase (every
part capitalised, get+name -> GetName), and Title (the first cased char of the token upper, all
else lower: rat+Haus -> Rathaus). Variations are
per-symbol maps from the model file (never str.upper / str.lower, which can change the length:
ß -> SS), so decoding is exact. Models trained with code_len (the unigram code-length objective)
end with a "CLEN" section of code length statistics that only the binary's encoder reads; the
tables, and so loading and decoding here, are unchanged.

The tokenizer file is a small JSON: {"type": "unrestricted", "model": <.bin next to it>,
"alphabet": [chars]}, written by experiments/bench_dictsearch.py next to every trained model.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import tempfile
from array import array

import regex

CHUNK = regex.compile(r"\s*\S+|\s+")
# models trained without --hybrid saw whole paragraphs (split before blank lines), so tokens
# may span spaces; they must be encoded the same way
PARAGRAPH = regex.compile(r"(?=\n\n)")
NONE = 0xFFFFFFFF
NONE_BASE = 0xFFFFFFF0
N_BYTES = 256
CASE_MAGIC = int.from_bytes(b"CASE", "little")
CAFX_MAGIC = int.from_bytes(b"CAFX", "little")
VARIATIONS_CASE = ["as-is", "Capitalised", "UPPER", "Traditional", "lower"]


def written_core(syms, v, case, fold, upper_of, trad_of):
    """symbol ids of a core row written with variation v. Old models: syms are folded and v rewrites
    them (1 first char upper, 2 all upper, 3 Traditional). case_cores: syms are the canonical
    spelling and v is relative to it (1 first char upper, 2 all upper, 3 Traditional with the
    capitals kept, 4 all folded). A char without the needed form is kept as it is."""
    out = []
    for i, c in enumerate(syms):
        t = NONE
        if not case:
            if (v == 1 and i == 0) or v == 2:
                t = upper_of[c]
            elif v == 3:
                t = trad_of[c]
        elif (v == 1 and i == 0) or v == 2:
            t = upper_of[fold[c]]
        elif v == 3:
            t = trad_of[c]  # NONE for capitals (trad_of is only set for Simplified Han)
        elif v == 4:
            t = fold[c]
        out.append(c if t == NONE else t)
    return out


def written_token(parts, v, fold, upper_of, trad_of):
    """case_affixes: symbol ids of a whole token (its parts' canonical spellings: prefix, core,
    suffix) written with variation v: 0 as stored, 1 the first char with an uppercase form made
    uppercase, 2 all upper, 3 Traditional with the capitals kept, 4 all folded, 5 camelCase (the
    first cased char of the core and of the suffix each made uppercase), 6 PascalCase (that of
    every part, the prefix too), 7 Title (the token's first cased char uppercase, all else folded,
    whatever the canonical spellings)."""
    out, first = [], v in (1, 7)
    for k, syms in enumerate(parts):
        if v == 6 or (v == 5 and k > 0):
            first = True
        for c in syms:
            t = NONE
            if v in (1, 5, 6, 7) and first and upper_of[fold[c]] != NONE:
                t, first = upper_of[fold[c]], False
            elif v == 7:
                t = fold[c]
            elif v == 2:
                t = upper_of[fold[c]]
            elif v == 3:
                t = trad_of[c]  # NONE for capitals (trad_of is only set for Simplified Han)
            elif v == 4:
                t = fold[c]
            out.append(c if t == NONE else t)
    return out
_BIN_DIR = os.path.join(os.path.dirname(__file__), "..", "native", "dict_search", "target", "release")


def _binary():
    if os.environ.get("DS_BIN"):  # another build (as in experiments/bench_dictsearch.py)
        return os.environ["DS_BIN"]
    for name in ("dict_search.exe", "dict_search"):
        path = os.path.join(_BIN_DIR, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError("build native/dict_search first: cargo build --release")


class UnrestrictedBPE:
    def __init__(self, model_path: str, alphabet: list[str], segmentation: str = "chunks"):
        assert segmentation in ("chunks", "paragraphs"), segmentation
        self.segmentation = segmentation
        self.model_path = model_path
        self.alphabet = list(alphabet)
        self.idx = {c: i for i, c in enumerate(self.alphabet)}
        data = open(model_path, "rb").read()
        pos = 0

        def u32s(n):
            nonlocal pos
            out = struct.unpack_from(f"<{n}I", data, pos)
            pos += 4 * n
            return out

        n = u32s(1)[0]
        self.case_affixes = n == CAFX_MAGIC
        self.case = self.case_affixes or n == CASE_MAGIC
        if self.case:
            n = u32s(1)[0]
        self.fold, _, self.upper_of, self.trad_of = [u32s(n) for _ in range(4)]
        tables = [[u32s(u32s(1)[0]) for _ in range(u32s(1)[0])] for _ in range(3)]
        self.core_syms, self.prefix_syms, self.suffix_syms = tables
        self.cores, self.prefixes, self.suffixes = (
            ["".join(self.alphabet[c] for c in s) for s in t] for t in tables)

    # ------------------------------------------------------------------ sizes
    @property
    def sizes(self):
        return {"variation": 8 if self.case_affixes else 5 if self.case else 4, "prefix": len(self.prefixes), "core": N_BYTES + len(self.cores),
                "suffix": len(self.suffixes)}

    @property
    def vocab_size(self):
        return sum(self.sizes.values())

    # ----------------------------------------------------------------- encode
    def _written(self, c: int, v: int) -> str:
        """core row c written with variation v (as a core-only token)"""
        if self.case_affixes:
            return self._str(written_token([(), self.core_syms[c], ()], v, self.fold, self.upper_of, self.trad_of))
        return self._str(written_core(self.core_syms[c], v, self.case, self.fold, self.upper_of, self.trad_of))

    def _str(self, syms) -> str:
        return "".join(self.alphabet[ch] for ch in syms)

    def written_parts(self, v: int, p: int, c: int, s: int) -> tuple[str, str, str]:
        """(prefix, core, suffix) text of a token with core row c (not a byte), as written"""
        if self.case_affixes:
            ps, cs = self.prefix_syms[p], self.core_syms[c]
            w = written_token([ps, cs, self.suffix_syms[s]], v, self.fold, self.upper_of, self.trad_of)
            return self._str(w[:len(ps)]), self._str(w[len(ps):len(ps) + len(cs)]), self._str(w[len(ps) + len(cs):])
        return self.prefixes[p], self._written(c, v), self.suffixes[s]

    def _encode_chunks(self, chunks: list[str]) -> list[list[tuple[int, int, int, int]]]:
        lens = array("I", map(len, chunks))
        ids = array("I")
        for s in chunks:
            ids.extend(self.idx[c] if c in self.idx else NONE_BASE + len(c.encode("utf-8")) for c in s)
        with tempfile.TemporaryDirectory() as tmp:
            wp, op = os.path.join(tmp, "w.bin"), os.path.join(tmp, "o.bin")
            with open(wp, "wb") as f:
                f.write(struct.pack("<QQ", len(chunks), len(ids)))
                lens.tofile(f)
                ids.tofile(f)
            subprocess.run([_binary(), "encode", self.model_path, wp, op], check=True)
            with open(op, "rb") as f:
                data = f.read()
        out, pos = [], 0
        for chunk in chunks:
            n = struct.unpack_from("<I", data, pos)[0]
            vals = struct.unpack_from(f"<{4 * n}I", data, pos + 4)
            pos += 4 + 16 * n
            toks, at = [], 0
            for i in range(0, 4 * n, 4):
                v, p, c, s = vals[i:i + 4]
                if c == NONE:  # byte fallback: one token per UTF-8 byte of this char
                    toks.extend((0, 0, b, 0) for b in chunk[at].encode("utf-8"))
                    at += 1
                else:
                    toks.append((v, p, N_BYTES + c, s))
                    at += len(self.prefixes[p]) + len(self.cores[c]) + len(self.suffixes[s])
            out.append(toks)
        return out

    def encode(self, text: str) -> list[tuple[int, int, int, int]]:
        chunks = CHUNK.findall(text) if self.segmentation == "chunks" else [s for s in PARAGRAPH.split(text) if s]
        uniq = list(dict.fromkeys(chunks))
        enc = dict(zip(uniq, self._encode_chunks(uniq))) if uniq else {}
        return [t for ch in chunks for t in enc[ch]]

    # ----------------------------------------------------------------- decode
    def token_bytes(self, tok) -> bytes:
        v, p, c, s = tok
        if c < N_BYTES:
            return bytes([c])
        return "".join(self.written_parts(v, p, c - N_BYTES, s)).encode("utf-8")

    def decode(self, toks) -> str:
        return b"".join(self.token_bytes(t) for t in toks).decode("utf-8", errors="replace")

    # --------------------------------------------------------------------- io
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            d = {"type": "unrestricted",
                 "model": os.path.relpath(self.model_path, os.path.dirname(os.path.abspath(path))),
                 "alphabet": self.alphabet}
            if self.segmentation != "chunks":
                d["segmentation"] = self.segmentation
            json.dump(d, f, ensure_ascii=False)
