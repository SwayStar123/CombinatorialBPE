"""Reversible factored dictionary search (native/dict_search): the trainer for the unrestricted
Combinatorial BPE, where any string can go in any factor (see the header of its main.rs).

Parsing works on whole documents; the only hard boundary is before a blank line ("\\n\\n"),
which keeps segments short enough to cache. Candidates come from a frequent-substring index of
the raw training text. A dev split (the 200k chars after each source's tokenizer training text)
picks the checkpoint; the test files are only used at the end.

`--hybrid` instead parses whitespace chunks (as the other unrestricted trainers do) and never
prunes single-character cores. `--fast` is for iterating on the method: 2M training chars per
source instead of 8M, at most 12 rounds, 22 threads (minutes instead of ~half an hour); compare
variants only against each other in this mode, then confirm the winner at full scale.

    python experiments/bench_dictsearch.py [--hybrid]  -> results/unrestricted.json
                                                          (key "dict_search" / "dict_search_hybrid")
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from array import array
from collections import Counter

import regex

# --han=data: Traditional variation = the form most used in the training data (han_st_data.json);
# must be set before cbpe is imported
if "--han=data" in sys.argv:
    os.environ["CBPE_HAN_TABLE"] = "han_st_data.json"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from build_mix import DATA, LM_FILES_BIG, SOURCES, TOK_CHARS, docs_prefix, tok_text  # noqa: E402
from cbpe.tokenizers import F_UP, HAN_TO_TRAD, fold_char  # noqa: E402

VOCAB = 32768
NONE = 0xFFFFFFFF
NONE_BASE = 0xFFFFFFF0
SEGMENT = regex.compile(r"(?=\n\n)")
CHUNK = regex.compile(r"\s*\S+|\s+")  # --hybrid segments: leading whitespace + a non-space run
BIN_DIR = os.path.join(ROOT, "native", "dict_search", "target", "release")
BIN = os.environ.get("DS_BIN") or next(os.path.join(BIN_DIR, n) for n in ("dict_search.exe", "dict_search")
                                      if os.path.exists(os.path.join(BIN_DIR, n)))
PARAMS = dict(expand_permille=250, min_freq=20, max_len=20, em_iters=1, max_rounds=30, threads=16,
              gamma_permille=1000, max_packages=600_000, protect_chars=0,
              first_expand_permille=1250, prune_step_permille=1000, delta_permille=0,
              lambda_permille=0, rerank_mult=0, rerank_occ=0, refactor=0, mu_permille=0,
              h0_millibits=4000, mu2_permille=0, tau_millibits=0,
              alnum_only=0, lamc_permille=0, nu_permille=0, frag_t_permille=500, glue_h0_millibits=0,
              price_permille=0, prod_k=0, conc_permille=0, core_price_permille=-1,
              min_gain_ppm=0, rare_permille=0, partner_n0=0)  # min_gain_ppm > 0: early stop (lossy), see the header of main.rs
DEV_CHARS = 200_000


HYBRID = "--hybrid" in sys.argv
# --classes: restricted-shaped rows (cores all letters / all digits / all other chars; affixes
# without letters/digits), plus letter affixes from --allow=FILE.json {"prefixes": [...], "suffixes": [...]}
CLASSES = "--classes" in sys.argv
ALLOW = next((a[8:] for a in sys.argv if a.startswith("--allow=")), None)
# --tokdata=N: train on N chars per source taken from the (larger) LM text files instead of the
# 8M-char tokenizer training files (disjoint from the test sets; the dev split is unchanged)
TOKDATA = next((int(a[10:]) for a in sys.argv if a.startswith("--tokdata=")), None)
# --init=MODEL.bin: start from that model's tables instead of the bare alphabet
INIT = next((a[7:] for a in sys.argv if a.startswith("--init=")), None)
LETTER, DIGIT = regex.compile(r"[\p{L}\p{M}]"), regex.compile(r"\p{N}")


SCRIPTS = [regex.compile(p) for p in (r"\p{Latin}", r"\p{Cyrillic}", r"\p{Greek}", r"\p{Han}",
                                       r"[\p{Hiragana}\p{Katakana}]", r"\p{Hangul}", r"\p{N}")]


def script_of(c):
    """script group: 0 = other (punctuation, space, symbols, other scripts), then Latin, Cyrillic,
    Greek, Han, kana, Hangul, digits"""
    return next((i + 1 for i, p in enumerate(SCRIPTS) if p.match(c)), 0)


def char_class(c):
    return 1 if LETTER.match(c) else 2 if DIGIT.match(c) else 0


VOCAB = next((int(a[8:]) for a in sys.argv if a.startswith("--vocab=")), VOCAB)
FAST = "--fast" in sys.argv


def segments(text):
    if HYBRID:
        return CHUNK.findall(text)
    return [s for s in SEGMENT.split(text) if s]


def symbol_ids(segs, idx):
    """Symbol ids of all segments, concatenated: idx[c], or NONE_BASE + the UTF-8 length of c.
    Mapped through a code point table with numpy (a Python loop per char took ~25 s at full scale)."""
    import numpy as np
    try:
        cps = np.frombuffer("".join(segs).encode("utf-32-le"), dtype=np.uint32)
    except UnicodeEncodeError:  # lone surrogates: per char
        ids = array("I")
        for s in segs:
            ids.extend(idx[c] if c in idx else NONE_BASE + len(c.encode("utf-8")) for c in s)
        return ids
    cp = np.arange(0x110000)
    table = (NONE_BASE + 1 + (cp >= 0x80) + (cp >= 0x800) + (cp >= 0x10000)).astype(np.uint32)
    for c, i in idx.items():
        table[ord(c)] = i
    return table[cps]


def write_corpus(f, segs, idx, counts=None):
    lens = array("I", map(len, segs))
    ids = symbol_ids(segs, idx)
    f.write(struct.pack("<QQ", len(segs), len(ids)))
    if counts is not None:
        array("q", counts).tofile(f)
    lens.tofile(f)
    f.write(ids.tobytes())


def build_symbols(char_counts, vocab, fold):
    """Alphabet (same rule as CharBPE.train) plus the per-symbol fold maps for the Rust side."""
    alpha = [ch for ch, c in char_counts.most_common() if c >= 20][:(vocab - 256) // 2]
    alpha = sorted(alpha)
    idx = {ch: i for i, ch in enumerate(alpha)}
    n = len(alpha)
    fold_to, flag = list(range(n)), [0] * n
    upper_of, trad_of = [NONE] * n, [NONE] * n
    if fold:
        for ch, i in idx.items():
            lo, fl = fold_char(ch, True, True)
            if fl and lo in idx:
                fold_to[i] = idx[lo]
                flag[i] = 2 if fl == F_UP else 3
                (upper_of if fl == F_UP else trad_of)[idx[lo]] = i
        for ch, i in idx.items():  # Simplified chars that the Traditional variation would change
            if flag[i] == 0 and trad_of[i] != NONE and HAN_TO_TRAD.get(ch) == alpha[trad_of[i]]:
                flag[i] = 1
            elif flag[i] == 0 and trad_of[i] != NONE:
                trad_of[i] = NONE
    return alpha, idx, [fold_to, flag, upper_of, trad_of]


def write_table(f, strs, idx):
    f.write(struct.pack("<I", len(strs)))
    for s in strs:
        f.write(struct.pack("<I", len(s)))
        array("I", [idx[c] for c in s]).tofile(f)


class DModel:
    """A trained model file: its tables (symbol ids and strings) and row costs; encodes via the binary."""

    def __init__(self, path, alpha):
        self.path, self.alpha = path, alpha
        data = open(path, "rb").read()
        pos = 0

        def u32s(n):
            nonlocal pos
            out = struct.unpack_from(f"<{n}I", data, pos)
            pos += 4 * n
            return out

        n = u32s(1)[0]
        _, _, self.upper_of, self.trad_of = [u32s(n) for _ in range(4)]
        self.tables = [[u32s(u32s(1)[0]) for _ in range(u32s(1)[0])] for _ in range(3)]
        self.cores, self.prefixes, self.suffixes = (["".join(alpha[c] for c in s) for s in t] for t in self.tables)
        self.costs = []
        for t in self.tables:
            self.costs.append(struct.unpack_from(f"<{len(t)}d", data, pos))
            pos += 8 * len(t)

    def written(self, core_id, v):
        """a core row as written with variation v (1 Capitalised, 2 UPPER, 3 Traditional)"""
        out = []
        for i, c in enumerate(self.tables[0][core_id]):
            t = NONE
            if (v == 1 and i == 0) or v == 2:
                t = self.upper_of[c]
            elif v == 3:
                t = self.trad_of[c]
            out.append(self.alpha[c if t == NONE else t])
        return "".join(out)

    def token_count_and_check(self, chunk, units):
        """Tokens for one chunk (out-of-alphabet chars cost their UTF-8 bytes); asserts lossless."""
        n, pos, rebuilt = 0, 0, []
        for v, p, c, s in units:
            if c is None:
                ch = chunk[pos]
                n += len(ch.encode("utf-8"))
                rebuilt.append(ch)
                pos += 1
            else:
                piece = self.prefixes[p] + self.written(c, v) + self.suffixes[s]
                n += 1
                rebuilt.append(piece)
                pos += len(piece)
        assert "".join(rebuilt) == chunk, (chunk, rebuilt)
        return n

    def show(self, chunk, units):
        parts, pos = [], 0
        for v, p, c, s in units:
            if c is None:
                parts.append(f"<byte {chunk[pos]!r}>")
                pos += 1
                continue
            core = self.cores[c]
            tag = "" if v == 0 else "^" if v == 1 else "^^" if v == 2 else "T:"
            parts.append(f"[{self.prefixes[p]!r}|{tag}{core}|{self.suffixes[s]!r}]")
            pos += len(self.prefixes[p]) + len(core) + len(self.suffixes[s])
        return " ".join(parts)

    def encode_chunks(self, chunks, idx):
        with tempfile.TemporaryDirectory() as tmp:
            wp, op = os.path.join(tmp, "w.bin"), os.path.join(tmp, "o.bin")
            with open(wp, "wb") as f:
                write_corpus(f, chunks, idx)
            subprocess.run([BIN, "encode", self.path, wp, op], check=True)
            data = open(op, "rb").read()
        out, pos = [], 0
        for _ in chunks:
            n = struct.unpack_from("<I", data, pos)[0]
            vals = struct.unpack_from(f"<{4 * n}I", data, pos + 4)
            pos += 4 + 16 * n
            out.append([(vals[i], vals[i + 1], None if vals[i + 2] == NONE else vals[i + 2], vals[i + 3])
                        for i in range(0, 4 * n, 4)])
        return out


def count_chars(text):
    """Counter(text), counted with numpy in slices (same counts; keys in code point order instead
    of first-occurrence order, which main() accounts for)."""
    import numpy as np
    tot = np.zeros(0x110000, dtype=np.int64)
    step = 1 << 26
    try:
        for i in range(0, len(text), step):
            tot += np.bincount(np.frombuffer(text[i:i + step].encode("utf-32-le"), dtype=np.uint32), minlength=0x110000)
    except UnicodeEncodeError:  # lone surrogates
        return Counter(text)
    return Counter({chr(c): int(tot[c]) for c in np.flatnonzero(tot)})


# a boundary no chunk crosses: a whitespace char after a non-whitespace char
CHUNK_CUT = regex.compile(r"\S(?=\s)")


def count_chunks(text, block=1 << 25):
    """Counter(CHUNK.findall(text)) in first-occurrence order, counted block by block (blocks end
    where a chunk ends; findall on a block is much faster than a Match object per chunk, and a
    list of all chunks at once would not fit in memory for large --tokdata)."""
    counts, start = Counter(), 0
    while start < len(text):
        m = CHUNK_CUT.search(text, start + block) if start + block < len(text) else None
        end = m.end() if m else len(text)
        counts.update(CHUNK.findall(text, start, end))
        start = end
    return counts


def dev_text():
    parts = []
    for prefix, _, _ in SOURCES.values():
        path = os.path.join(DATA, f"{prefix}.train.txt")
        start = len(docs_prefix(path, TOK_CHARS))
        with open(path, encoding="utf-8") as f:
            t = f.read(start + DEV_CHARS + 50_000)[start:]
        cut = t.rfind("\n\n", 0, DEV_CHARS)
        parts.append(t[:cut] if cut > 0 else t[:DEV_CHARS])
    return "".join(parts)


def main(out_key="dict_search"):
    sys.stdout.reconfigure(encoding="utf-8")
    if FAST:
        import build_mix
        build_mix.TOK_CHARS = 2_000_000
        PARAMS.update(max_rounds=12, threads=22)
    if TOKDATA:
        text = "\n\n".join(docs_prefix(os.path.join(DATA, LM_FILES_BIG[name]), TOKDATA) for name in SOURCES)
    else:
        text = tok_text()
    char_counts = count_chars(text)
    if sum(c >= 20 for c in char_counts.values()) > (VOCAB - 256) // 2:
        char_counts = Counter(text)  # the alphabet is cut at a count: ties must keep first-occurrence order
    seg_counts = count_chunks(text) if HYBRID else Counter(segments(text))
    del text
    alpha, idx, maps = build_symbols(char_counts, VOCAB, fold=True)
    dev_counts = Counter(segments(dev_text()))
    print(f"{sum(seg_counts.values())} segments ({len(seg_counts)} unique), alphabet {len(alpha)}", flush=True)

    budget = VOCAB - 4 - 256
    path = os.path.join(ROOT, "results", "tokenizers", f"mix_{VOCAB}_{out_key}.bin")
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "in.bin")
        with open(inp, "wb") as f:
            f.write(struct.pack("<I", len(alpha)))
            for m in maps:
                array("I", m).tofile(f)
            array("I", [len(c.encode("utf-8")) for c in alpha]).tofile(f)
            array("I", [char_class(c) for c in alpha]).tofile(f)
            f.write(struct.pack(f"<{1 + len(PARAMS)}I", budget, *PARAMS.values()))
            segs = list(seg_counts)
            write_corpus(f, segs, idx, [seg_counts[s] for s in segs])
            segs = list(dev_counts)
            write_corpus(f, segs, idx, [dev_counts[s] for s in segs])
            f.write(struct.pack("<I", int(CLASSES)))
            allow = json.load(open(ALLOW, encoding="utf-8")) if ALLOW else {"prefixes": [], "suffixes": []}
            for key in ("prefixes", "suffixes"):
                rows = [a for a in allow[key] if all(c in idx for c in a)]
                write_table(f, rows, idx)
            if INIT:  # start from another model's tables (same alphabet)
                f.write(struct.pack("<I", 1))
                im = DModel(INIT, alpha)
                for rows in (im.cores, im.prefixes, im.suffixes):
                    write_table(f, [r for r in rows if r], idx)
            else:
                f.write(struct.pack("<I", 0))
            # script group per alphabet char (rarity cost): affixes are compared with text of their script
            f.write(struct.pack("<I", 1))
            array("I", [script_of(c) for c in alpha]).tofile(f)
        del seg_counts
        if os.environ.get("DS_SAVE_INPUT"):  # keep the trainer input (benchmarking) and stop
            import shutil
            shutil.copyfile(inp, os.environ["DS_SAVE_INPUT"])
            print(f"input saved to {os.environ['DS_SAVE_INPUT']}", flush=True)
            return
        t0 = time.time()
        env = dict(os.environ)
        if os.environ.get("DS_TRACE"):  # "|"-separated strings to trace (debug; see load_trace in main.rs)
            tp = os.path.join(tmp, "trace.bin")
            words = [w for w in os.environ["DS_TRACE"].split("|") if w and all(c in idx for c in w)]
            with open(tp, "wb") as f:
                f.write(struct.pack("<I", len(words)))
                for w in words:
                    f.write(struct.pack("<I", len(w)))
                    array("I", [idx[c] for c in w]).tofile(f)
            env["DS_TRACE_FILE"] = tp
            print("tracing", words, flush=True)
        subprocess.run([BIN, "train", inp, path], check=True, env=env)
        print(f"trained in {time.time() - t0:.0f}s", flush=True)

    model = DModel(path, alpha)
    from cbpe.unrestricted import UnrestrictedBPE  # the cbpe tokenizer file, usable by cbpe.load / lm.py
    UnrestrictedBPE(path, alpha).save(path[:-4] + ".json")
    r = {"params": PARAMS, "sizes": {"variation": 4, "prefix": len(model.prefixes),
                                     "core": 256 + len(model.cores), "suffix": len(model.suffixes)},
         "chars_per_token": {}}
    for name in SOURCES:
        t = open(os.path.join(DATA, f"val_mix_{name}.txt"), encoding="utf-8").read()
        segs = segments(t)
        uniq = list(dict.fromkeys(segs))
        cost = {s: model.token_count_and_check(s, u) for s, u in zip(uniq, model.encode_chunks(uniq, idx))}
        r["chars_per_token"][name] = len(t) / sum(cost[s] for s in segs)
    print(r["sizes"], {k: round(v, 2) for k, v in r["chars_per_token"].items()}, flush=True)

    by_cost = lambda k: [s for _, s in sorted(zip(model.costs[k], model.tables[k]))]  # noqa: E731
    name = lambda s: "".join(alpha[c] for c in s)  # noqa: E731
    r["prefixes_top"] = [name(s) for s in by_cost(1)[:300]]
    r["suffixes_top"] = [name(s) for s in by_cost(2)[:300]]
    r["cores_top"] = [name(s) for s in by_cost(0) if len(s) > 1][:300]
    r["examples"] = {}
    for s in ["The deal was $5B in 2023, up 12%.", "    return self.getUserName(id);",
              "Die Regierung hat's beschlossen.", "北京是中国的首都。臺灣的首都是台北。",
              "const unhappiness = await fetchItems();", "HELLO World, iPhone users!"]:
        segs = segments(s)
        r["examples"][s] = " ".join(model.show(g, u) for g, u in zip(segs, model.encode_chunks(segs, idx)))
        print("  ", r["examples"][s])
    out = os.path.join(ROOT, "results", "unrestricted.json")
    old = json.load(open(out, encoding="utf-8")) if os.path.exists(out) else {}
    old[out_key] = r
    with open(out, "w", encoding="utf-8") as f:
        json.dump(old, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    if HYBRID:
        PARAMS["protect_chars"] = 1
    for arg in sys.argv:  # --set=name:value overrides a PARAMS entry
        if arg.startswith("--set="):
            k, v = arg[6:].split(":")
            PARAMS[k] = int(v)
    if PARAMS["core_price_permille"] < 0:  # default: cores priced like affixes
        PARAMS["core_price_permille"] = PARAMS["price_permille"]
    tag = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--tag=")), "")
    main(("dict_search_hybrid" if HYBRID else "dict_search") + (f"_{tag}" if tag else "")
         + ("_handata" if "--han=data" in sys.argv else "") + ("_fast" if FAST else ""))
