"""Worst cases of a tokenizer on held-out text: the report to read next to any aggregate metric.

1. most frequent words cut by the tokenizer (by a token boundary "|" or, inside one token, a
   prefix/core/suffix boundary ":"), per source, with GPT-4o's split for comparison
2. least productive affixes: prefix / suffix rows with the fewest effective partner cores
   (2^entropy of the cores they are used with), among rows used at least MIN_AFFIX times
3. least-used cores (multi-char rows), and how many cores are (almost) never used

Held-out text: the first HELD_CHARS of each source's LM file for curated data (the tokenizer
trains on the separate *.train.txt), else the slice after the first 40M chars of the LM file
(what --tokdata=40000000 trains on).

    CBPE_DATA=curated python experiments/worst_cases.py results/tokenizers/<tokenizer>.json [N]
"""
import math
import os
import sys
from collections import Counter, defaultdict

import regex

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from build_mix import DATA, LM_FILES_BIG, SOURCES, docs_prefix  # noqa: E402
from cbpe.tokenizers import apply_variation, load  # noqa: E402

HELD_CHARS = 4_000_000
MIN_AFFIX = 30
WORD = regex.compile(r"\p{L}[\p{L}\p{M}]+")
LETTER = regex.compile(r"[\p{L}\p{M}\p{N}]")


def heldout(name):
    path = os.path.join(DATA, LM_FILES_BIG[name])
    if os.environ.get("CBPE_DATA") == "curated":
        return docs_prefix(path, HELD_CHARS)
    return docs_prefix(path, 40_000_000 + HELD_CHARS)[len(docs_prefix(path, 40_000_000)):]


def parts(tok, t):
    """(prefix, core, suffix, core id) strings of one token; a standard-BPE or byte token is all core"""
    if isinstance(t, int):
        return "", tok.bpe.id_bytes(t).decode("utf-8", "replace"), "", t
    v, p, c, s = t
    if hasattr(tok, "core"):  # restricted
        cv = tok.core.vocab[c]
        core = cv.decode("utf-8", "replace") if isinstance(cv, bytes) else apply_variation(v, cv)
    elif c >= 256 and hasattr(tok, "written_parts"):  # unrestricted (case_affixes: whole-token variation)
        return (*tok.written_parts(v, p, c - 256, s), c)
    else:
        core = chr(0xFFFD) if c < 256 else tok._written(c - 256, v)
    return tok.prefixes[p], core, tok.suffixes[s], c


def nbytes(tok, t, part):
    """UTF-8 length of a token part (a byte-fallback or partial-UTF-8 core is its raw bytes)"""
    if isinstance(t, int):
        return len(tok.bpe.id_bytes(t)) if part == 1 else 0
    if part == 1 and not hasattr(tok, "core") and t[2] < 256:
        return 1
    if part == 1 and hasattr(tok, "core") and isinstance(tok.core.vocab[t[2]], bytes):
        return len(tok.core.vocab[t[2]])
    return len(parts(tok, t)[part].encode("utf-8"))


def segment(tok, text):
    """per UTF-8 byte offset: '|' token boundary, ':' factor boundary; and the tokens' parts
    (byte offsets, so byte-fallback tokens of one char cannot shift later boundaries)"""
    bounds, toks, pos = {}, [], 0
    for t in tok.encode(text):
        pr, co, su, c = parts(tok, t)
        toks.append((t, pr, co, su, c))
        lp, lc, ls = (nbytes(tok, t, i) for i in range(3))
        if lp and (lc or ls):
            bounds.setdefault(pos + lp, ":")
        if ls and (lp or lc):
            bounds.setdefault(pos + lp + lc, ":")
        pos += lp + lc + ls
        bounds[pos] = "|"
    assert pos == len(text.encode("utf-8")), "not lossless"
    return bounds, toks


def show(word, start, bounds):
    """word with its internal boundaries; start = byte offset of the word"""
    out, off = [], start
    for i, ch in enumerate(word):
        if i and off in bounds:
            out.append(bounds[off])
        out.append(ch)
        off += len(ch.encode("utf-8"))
    return "".join(out)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    path = sys.argv[1]
    n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    tok = load(path)
    try:
        import tiktoken
        o200k = tiktoken.get_encoding("o200k_base")
    except Exception:
        o200k = None
    factored = not hasattr(tok, "bpe")
    pc = [defaultdict(Counter), defaultdict(Counter)]
    core_use = Counter()
    print(f"{path}: sizes {getattr(tok, 'sizes', None) or tok.vocab_size}; held-out {HELD_CHARS / 1e6:.0f}M chars per source")
    for src in SOURCES:
        text = heldout(src)
        bounds, toks = segment(tok, text)
        for t, pr, co, su, c in toks:
            if not factored:
                continue
            core_use[c] += 1
            if pr:
                pc[0][pr][co.lower()] += 1
            if su:
                pc[1][su][co.lower()] += 1
        cut = Counter()
        shown = {}
        total = Counter()
        off = [0]
        for ch in text:
            off.append(off[-1] + len(ch.encode("utf-8")))
        for m in WORD.finditer(text):
            w, a, b = m.group(), off[m.start()], off[m.end()]
            total[w] += 1
            if any(x in bounds for x in range(a + 1, b)):
                cut[w] += 1
                shown.setdefault(w, show(w, a, bounds))
        n_words = sum(total.values())
        print(f"\n===== {src}: most frequent words cut ({100 * sum(cut.values()) / n_words:.1f}% of word occurrences cut)")
        for w, n in cut.most_common(n_show):
            gpt = ""
            if o200k is not None:
                gpt = "·".join(o200k.decode_single_token_bytes(i).decode("utf-8", "replace").strip()
                               for i in o200k.encode(" " + w))
            print(f"   {n:6d}/{total[w]:<6d} {shown[w]:28} GPT-4o {gpt}")
    if not factored:
        return
    for k, name in ((0, "prefix"), (1, "suffix")):
        rows = []
        for a, cs in pc[k].items():
            n = sum(cs.values())
            if n < MIN_AFFIX or not LETTER.search(a):
                continue
            h = -sum(x / n * math.log2(x / n) for x in cs.values())
            top, tn = cs.most_common(1)[0]
            rows.append((2 ** h, n, a, top, tn / n))
        rows.sort()
        few = sum(1 for r in rows if r[0] < 3)
        print(f"\n===== least productive {name}es (letters/digits, >= {MIN_AFFIX} uses): {few} of {len(rows)} have < 3 effective partners")
        for e, n, a, top, sh in rows[:n_show]:
            print(f"   {e:5.1f} partners  {n:6d} uses  {a!r:22} with {top!r} {sh:.0%}")
    cores = [(core_use.get(c, 0), c) for c in range(256, 256 + len(tok.cores))] if hasattr(tok, "core_syms") else None
    if cores is None and hasattr(tok, "core"):
        cores = [(core_use.get(c, 0), c) for c, v in enumerate(tok.core.vocab) if isinstance(v, str)]
    if cores:
        n_cores = len(cores)
        print(f"\n===== cores: {sum(1 for u, _ in cores if u == 0)} of {n_cores} never used in "
              f"{len(SOURCES) * HELD_CHARS / 1e6:.0f}M held-out chars, {sum(1 for u, _ in cores if u <= 2)} used <= 2 times")
        name = (lambda c: tok._written(c - 256, 0)) if hasattr(tok, "_written") else (lambda c: tok.core.vocab[c])
        low = [(u, name(c)) for u, c in sorted(cores) if len(name(c)) > 1][:n_show * 3]
        print("   least used:", ", ".join(f"{s!r}({u})" for u, s in low))


if __name__ == "__main__":
    main()
