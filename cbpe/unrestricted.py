"""Unrestricted Combinatorial BPE (trained by native/dict_search, experiments/bench_dictsearch.py).

Prefixes, cores and suffixes are arbitrary strings; only the core carries the variation. Text is
split into whitespace chunks (leading whitespace attached) and every chunk is segmented by the
exact minimum-cost DP in the Rust binary, so this class needs `native/dict_search` built
(`cargo build --release`).

Token layout matches CombinatorialBPE: (variation, prefix, core, suffix), where core ids 0-255 are
the UTF-8 byte-fallback rows and learned cores start at 256; prefix/suffix 0 is the empty string.

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
NONE = 0xFFFFFFFF
NONE_BASE = 0xFFFFFFF0
N_BYTES = 256
_BIN_DIR = os.path.join(os.path.dirname(__file__), "..", "native", "dict_search", "target", "release")


def _binary():
    for name in ("dict_search.exe", "dict_search"):
        path = os.path.join(_BIN_DIR, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError("build native/dict_search first: cargo build --release")


class UnrestrictedBPE:
    def __init__(self, model_path: str, alphabet: list[str]):
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
        _, _, self.upper_of, self.trad_of = [u32s(n) for _ in range(4)]
        tables = [[u32s(u32s(1)[0]) for _ in range(u32s(1)[0])] for _ in range(3)]
        self.core_syms = tables[0]
        self.cores, self.prefixes, self.suffixes = (
            ["".join(self.alphabet[c] for c in s) for s in t] for t in tables)

    # ------------------------------------------------------------------ sizes
    @property
    def sizes(self):
        return {"variation": 4, "prefix": len(self.prefixes), "core": N_BYTES + len(self.cores),
                "suffix": len(self.suffixes)}

    @property
    def vocab_size(self):
        return sum(self.sizes.values())

    # ----------------------------------------------------------------- encode
    def _written(self, c: int, v: int) -> str:
        out = []
        for i, ch in enumerate(self.core_syms[c]):
            t = NONE
            if (v == 1 and i == 0) or v == 2:
                t = self.upper_of[ch]
            elif v == 3:
                t = self.trad_of[ch]
            out.append(self.alphabet[ch if t == NONE else t])
        return "".join(out)

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
        chunks = CHUNK.findall(text)
        uniq = list(dict.fromkeys(chunks))
        enc = dict(zip(uniq, self._encode_chunks(uniq))) if uniq else {}
        return [t for ch in chunks for t in enc[ch]]

    # ----------------------------------------------------------------- decode
    def token_bytes(self, tok) -> bytes:
        v, p, c, s = tok
        if c < N_BYTES:
            return bytes([c])
        return (self.prefixes[p] + self._written(c - N_BYTES, v) + self.suffixes[s]).encode("utf-8")

    def decode(self, toks) -> str:
        return b"".join(self.token_bytes(t) for t in toks).decode("utf-8", errors="replace")

    # --------------------------------------------------------------------- io
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "unrestricted",
                       "model": os.path.relpath(self.model_path, os.path.dirname(os.path.abspath(path))),
                       "alphabet": self.alphabet}, f, ensure_ascii=False)
