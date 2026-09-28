"""Segmentation quality against gold data (evaluation only; nothing here is used in training).

MorphScore (Arnett et al. 2025; data from Universal Dependencies, data/eval/morphscore_data):
every word form with its stem and the parts before / after it; gold boundaries are the stem's
edges. Items as in MorphScore's default scoring (lemma == stem, unique word forms, weighted by
UD frequency). A word is encoded after a space, as in running text; the boundary right after that
space is not counted.
SIGHAN 2005 (PKU, MSR test gold, data/eval/icwb2-data): Chinese word boundaries of whole lines.

Predicted boundaries: token boundaries ("tok"), and token + prefix/core/suffix boundaries ("fac").
Reports boundary precision / recall / F1 (micro, frequency-weighted for MorphScore).

    python experiments/morph_eval.py NAME=PATH [NAME=PATH ...] [--kimi]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from cbpe.tokenizers import load  # noqa: E402

EVAL = os.path.join(HERE, "..", "data", "eval")
MORPH = {"en": "english", "de": "german", "fr": "french", "hi": "hindi", "ja": "japanese"}
SIGHAN = ["pku", "msr"]
MAX_ITEMS = 20000  # most frequent word forms per language


def bounds_ours(tok, text):
    """{char offset: 'tok' | 'fac'} of a factored tokenizer on text"""
    import worst_cases as W
    b, _ = W.segment(tok, text)
    byte_to_char, o = {}, 0
    for i, ch in enumerate(text):
        byte_to_char[o] = i
        o += len(ch.encode("utf-8"))
    byte_to_char[o] = len(text)
    return {byte_to_char[k]: ("tok" if v == "|" else "fac") for k, v in b.items() if k in byte_to_char}


def bounds_tiktoken(enc, text):
    out, pos = {}, 0
    raw = text.encode("utf-8")
    cpos = {}
    o = 0
    for i, ch in enumerate(text):
        cpos[o] = i
        o += len(ch.encode("utf-8"))
    cpos[o] = len(text)
    for t in enc.encode_ordinary(text):
        pos += len(enc.decode_single_token_bytes(t))
        if pos in cpos:
            out[cpos[pos]] = "tok"
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def morph_items(lang):
    import pandas as pd
    path = os.path.join(EVAL, "morphscore_data", f"{MORPH[lang]}_data.csv")
    df = pd.read_csv(path, usecols=["wordform", "lemma", "stem", "preceding_part", "following_part", "word_freq"],
                     keep_default_na=False, dtype=str)
    df = df[df.lemma == df.stem]
    df = df[(df.preceding_part != "") | (df.following_part != "")]
    df["word_freq"] = pd.to_numeric(df.word_freq, errors="coerce").fillna(1)
    df = df.drop_duplicates("wordform").sort_values("word_freq", ascending=False).head(MAX_ITEMS)
    items = []
    for w, pre, stem, fol, f in zip(df.wordform, df.preceding_part, df.stem, df.following_part, df.word_freq):
        if pre + stem + fol != w:
            continue
        gold = set()
        if pre:
            gold.add(len(pre))
        if fol:
            gold.add(len(pre) + len(stem))
        items.append((w, gold, float(f)))
    return items


def eval_morph(bfun, items):
    """one encode of ' w1 w2 ...': every ' w' is its own chunk, as when encoded alone"""
    text = "".join(" " + w for w, _, _ in items)
    b_all = bfun(text)
    acc = {"tok": [0.0, 0.0, 0.0], "fac": [0.0, 0.0, 0.0]}
    start = 0
    for w, gold, f in items:
        s0 = start + 1  # the word's first char
        inner = {k - s0: v for k, v in b_all.items() if s0 < k < s0 + len(w)}
        start = s0 + len(w)
        for kind in ("tok", "fac"):
            pred = {k for k, v in inner.items() if kind == "fac" or v == "tok"}
            a = acc[kind]
            a[0] += f * len(pred & gold)
            a[1] += f * len(pred - gold)
            a[2] += f * len(gold - pred)
    return {k: prf(*v) for k, v in acc.items()}


def sighan_lines(corpus, n=3000):
    path = os.path.join(EVAL, "icwb2-data", "gold", f"{corpus}_test_gold.utf8")
    lines = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            words = line.split()
            if words:
                lines.append(words)
            if len(lines) >= n:
                break
    return lines


def eval_sighan(bfun, lines):
    """one encode of all lines joined by newlines (a line never shares a chunk with the next)"""
    texts = ["".join(words) for words in lines]
    b_all = bfun("\n".join(texts))
    acc = {"tok": [0, 0, 0], "fac": [0, 0, 0]}
    start = 0
    for words, text in zip(lines, texts):
        gold, o = set(), 0
        for w in words[:-1]:
            o += len(w)
            gold.add(o)
        b = {k - start: v for k, v in b_all.items() if start < k < start + len(text)}
        start += len(text) + 1
        for kind in ("tok", "fac"):
            pred = {k for k, v in b.items() if kind == "fac" or v == "tok"}
            a = acc[kind]
            a[0] += len(pred & gold)
            a[1] += len(pred - gold)
            a[2] += len(gold - pred)
    return {k: prf(*v) for k, v in acc.items()}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    models = []
    for a in args:
        name, path = a.split("=", 1)
        tok = load(path)
        models.append((name, lambda t, tok=tok: bounds_ours(tok, t)))
    if "--kimi" in sys.argv:
        import re
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe
        src = open(os.path.join(HERE, "..", "data", "kimi_k3", "tokenization_kimi.py"), encoding="utf-8").read()
        body = src[src.index('pat_str = "|".join(['):]
        pats = re.findall(r'r"""(.*?)"""', body[:body.index('])')])
        kimi = tiktoken.Encoding(name="kimi_k3", pat_str="|".join(pats),
                                 mergeable_ranks=load_tiktoken_bpe(os.path.join(HERE, "..", "data", "kimi_k3", "tiktoken.model")),
                                 special_tokens={})
        models.append(("Kimi K3", lambda t: bounds_tiktoken(kimi, t)))
    if "--gpt4o" in sys.argv:
        import tiktoken
        o200k = tiktoken.get_encoding("o200k_base")
        models.append(("GPT-4o", lambda t: bounds_tiktoken(o200k, t)))
    print("boundary precision / recall / F1; tok = token boundaries, fac = token + prefix/core/suffix boundaries")
    for lang in MORPH:
        items = morph_items(lang)
        print(f"\n== MorphScore {lang} ({len(items)} word forms)")
        for name, bf in models:
            r = eval_morph(bf, items)
            print(f"  {name:28} tok P {r['tok'][0]:.3f} R {r['tok'][1]:.3f} F1 {r['tok'][2]:.3f}   "
                  f"fac P {r['fac'][0]:.3f} R {r['fac'][1]:.3f} F1 {r['fac'][2]:.3f}", flush=True)
    for corpus in SIGHAN:
        lines = sighan_lines(corpus)
        print(f"\n== SIGHAN {corpus} ({len(lines)} lines)")
        for name, bf in models:
            r = eval_sighan(bf, lines)
            print(f"  {name:28} tok P {r['tok'][0]:.3f} R {r['tok'][1]:.3f} F1 {r['tok'][2]:.3f}   "
                  f"fac P {r['fac'][0]:.3f} R {r['fac'][1]:.3f} F1 {r['fac'][2]:.3f}", flush=True)


if __name__ == "__main__":
    main()
