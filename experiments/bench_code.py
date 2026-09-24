"""Compression on source code (codeparrot/github-code-clean, split by repository).

Same equal-budget comparison as bench_compression.py, plus the camelCase split
(split_camel=True) and a measurement of how much identifier text falls back to
per-character encoding because a core piece has mixed case (e.g. 'rN' in 'useRName').
Writes results/code.json.
"""
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cbpe import CombinatorialBPE, StandardBPE  # noqa: E402
from cbpe.tokenizers import CAMEL_SPLIT, unit_parts  # noqa: E402
from bench_compression import hf_bytelevel, redundancy  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "results")
CORPORA = ["code_python", "code_java", "code_javascript", "code_cpp", "code_go"]
SIZES = [4096, 8192, 16384, 32768, 65536]


def read(name, split):
    with open(os.path.join(ROOT, "data", f"{name}.{split}.txt"), encoding="utf-8") as f:
        return f.read()


def fallback_share(tok, words: Counter):
    stats, one = Counter(), Counter()
    for w, n in words.items():
        one.clear()
        tok._encode_word(w, one)
        for k, v in one.items():
            stats[k] += v * n
    return stats["fallback_chars"] / max(1, stats["chars"])


def run(name, size):
    train, test = read(name, "train"), read(name, "test")
    words = Counter(w for _, w, _ in unit_parts(test) if w)
    camel = sum(n for w, n in words.items() if len(CAMEL_SPLIT.split(w)) > 1) / sum(words.values())
    res = {"corpus": name, "vocab": size, "test_chars": len(test), "camel_word_share": camel}
    variants = {
        "bpe_gpt2": lambda: StandardBPE.train(train, size, "gpt2"),
        "bpe_cl100k": lambda: StandardBPE.train(train, size, "cl100k"),
        "comb": lambda: CombinatorialBPE.train(train, size),
        "comb_camel": lambda: CombinatorialBPE.train(train, size, split_camel=True),
    }
    for key, make in variants.items():
        t0 = time.time()
        tok = make()
        ids = tok.encode(test)
        assert tok.decode(ids) == test, (name, size, key)
        r = {"tokens": len(ids), "chars_per_token": len(test) / len(ids), "vocab_size": tok.vocab_size,
             "train_s": round(time.time() - t0, 1)}
        if isinstance(tok, CombinatorialBPE):
            r["sizes"] = tok.sizes
            r["prefixes"] = tok.prefixes[:40]
            r["suffixes"] = tok.suffixes[:40]
            r["fallback_char_share"] = fallback_share(tok, words)
            r["has_prefix"] = sum(t[1] > 0 for t in ids) / len(ids)
            r["has_suffix"] = sum(t[3] > 0 for t in ids) / len(ids)
            r["has_case"] = sum(t[0] > 0 for t in ids) / len(ids)
        else:
            r["redundant_frac"] = redundancy(tok)[0]
        tok.save(os.path.join(OUT, "tokenizers", f"{name}_{size}_{key}.json"))
        res[key] = r
    vs, n = hf_bytelevel(train, test, size)
    res["hf_bytelevel"] = {"tokens": n, "chars_per_token": len(test) / n, "vocab_size": vs}
    return res


def tiktoken_reference():
    """Production reference: GPT-4's cl100k_base (100k vocab, trained on code among other data)."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    out = {}
    for name in CORPORA:
        test = read(name, "test")
        out[name] = len(test) / len(enc.encode(test, disallowed_special=()))
    return out


if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "tokenizers"), exist_ok=True)
    path = os.path.join(OUT, "code.json")
    results = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []
    done = {(r["corpus"], r["vocab"]) for r in results}
    jobs = [(c, s) for c in CORPORA for s in SIZES if (c, s) not in done]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("WORKERS", 6))) as ex:
        futs = {ex.submit(run, c, s): (c, s) for c, s in jobs}
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            print(f"{r['corpus']:16s} {r['vocab']:6d}  gpt2 {r['bpe_gpt2']['chars_per_token']:.3f}  "
                  f"cl100k {r['bpe_cl100k']['chars_per_token']:.3f}  hf {r['hf_bytelevel']['chars_per_token']:.3f}  "
                  f"comb {r['comb']['chars_per_token']:.3f} (fallback {100 * r['comb']['fallback_char_share']:.1f}%)  "
                  f"comb_camel {r['comb_camel']['chars_per_token']:.3f} "
                  f"(fallback {100 * r['comb_camel']['fallback_char_share']:.1f}%)", flush=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(sorted(results, key=lambda x: (CORPORA.index(x["corpus"]), x["vocab"])), fh,
                          indent=1, ensure_ascii=False)
    try:
        ref = tiktoken_reference()
        with open(os.path.join(OUT, "code_tiktoken.json"), "w") as fh:
            json.dump({"cl100k_base_chars_per_token": ref}, fh, indent=1)
    except ImportError:
        print("tiktoken not installed: skipping the GPT-4 cl100k_base reference")
