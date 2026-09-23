"""Compression benchmark: characters per token on held-out text at equal embedding budget.

For a standard tokenizer the budget is its vocab size. For the combinatorial tokenizer
it is |variations| + |prefixes| + |core| + |suffixes| (total embedding rows), so both
sides get exactly the same number of embedding parameters.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cbpe import CombinatorialBPE, StandardBPE  # noqa: E402
from cbpe.tokenizers import normalise_char  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results")
CORPORA = ["tinystories", "wiki_en", "wiki_de", "wiki_fr", "wiki_ru", "wiki_tr", "wiki_hi", "wiki_ja", "wiki_zh",
           "multi"]
MULTI_LANGS = ["wiki_en", "wiki_de", "wiki_fr", "wiki_ru", "wiki_tr", "wiki_hi"]
SIZES = [4096, 8192, 16384, 32768, 65536]
MAX_TRAIN_CHARS = 40_000_000
MAX_TEST_CHARS = 3_000_000


def read(name, split):
    if name == "multi":  # equal share of every wiki language
        langs = MULTI_LANGS
        per = (MAX_TRAIN_CHARS if split == "train" else MAX_TEST_CHARS) // len(langs)
        return "\n\n".join(read(l, split)[:per] for l in langs)
    with open(os.path.join(DATA, f"{name}.{split}.txt"), encoding="utf-8") as f:
        return f.read(MAX_TRAIN_CHARS if split == "train" else MAX_TEST_CHARS)


def redundancy(tok: StandardBPE):
    """Fraction of a standard vocab that are case/space/punctuation variants of another entry."""
    groups = {}
    for t in tok.bpe.vocab:
        if not isinstance(t, str):
            continue
        i, j = 0, len(t)
        while i < j and not t[i].isalnum():
            i += 1
        while j > i and not t[j - 1].isalnum():
            j -= 1
        core = "".join(normalise_char(c)[0] for c in t[i:j])
        if core:
            groups.setdefault(core, []).append(t)
    extra = sum(len(g) - 1 for g in groups.values())
    examples = sorted(groups.values(), key=len, reverse=True)[:5]
    return extra / len(tok.bpe.vocab), examples


def hf_bytelevel(train, test, size):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tr = trainers.BpeTrainer(vocab_size=size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                             show_progress=False)
    t.train_from_iterator([train[i:i + 1_000_000] for i in range(0, len(train), 1_000_000)], tr)
    return t.get_vocab_size(), len(t.encode(test).ids)


def run(name, size):
    train, test = read(name, "train"), read(name, "test")
    res = {"corpus": name, "vocab": size, "test_chars": len(test), "test_bytes": len(test.encode())}
    variants = {
        "bpe_gpt2": lambda: StandardBPE.train(train, size, "gpt2"),
        "bpe_cl100k": lambda: StandardBPE.train(train, size, "cl100k"),
        "comb": lambda: CombinatorialBPE.train(train, size),
        "comb_case_only": lambda: CombinatorialBPE.train(train, size, n_prefix=0, n_suffix=0),
        "comb_affix_only": lambda: CombinatorialBPE.train(train, size, fold_case=False),
    }
    for key, make in variants.items():
        t0 = time.time()
        tok = make()
        t1 = time.time()
        ids = tok.encode(test)
        assert tok.decode(ids) == test, (name, size, key)
        r = {"tokens": len(ids), "chars_per_token": len(test) / len(ids), "vocab_size": tok.vocab_size,
             "train_s": round(t1 - t0, 1)}
        if isinstance(tok, CombinatorialBPE):
            r["sizes"] = tok.sizes
            r["distinct_tuples_used"] = len(set(ids))
            r["distinct_cores_used"] = len({t[2] for t in ids})
            r["prefixes"] = tok.prefixes[:40]
            r["suffixes"] = tok.suffixes[:40]
            r["has_prefix"] = sum(t[1] > 0 for t in ids) / len(ids)
            r["has_suffix"] = sum(t[3] > 0 for t in ids) / len(ids)
            r["has_case"] = sum(t[0] > 0 for t in ids) / len(ids)
            os.makedirs(os.path.join(OUT, "tokenizers"), exist_ok=True)
            tok.save(os.path.join(OUT, "tokenizers", f"{name}_{size}_{key}.json"))
        else:
            r["distinct_tokens_used"] = len(set(ids))
            r["redundant_frac"], r["redundant_examples"] = redundancy(tok)
            tok.save(os.path.join(OUT, "tokenizers", f"{name}_{size}_{key}.json"))
        res[key] = r
    vs, n = hf_bytelevel(train, test, size)
    res["hf_bytelevel"] = {"tokens": n, "chars_per_token": len(test) / n, "vocab_size": vs}
    return res


if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "tokenizers"), exist_ok=True)
    path = os.path.join(OUT, "compression.json")
    results = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []
    done = {(r["corpus"], r["vocab"]) for r in results}
    jobs = [(c, s) for c in CORPORA for s in SIZES if (c, s) not in done]
    with ProcessPoolExecutor(max_workers=int(os.environ.get("WORKERS", 8))) as ex:
        futs = {ex.submit(run, c, s): (c, s) for c, s in jobs}
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            print(f"{r['corpus']:12s} {r['vocab']:6d}  gpt2 {r['bpe_gpt2']['chars_per_token']:.3f}  "
                  f"cl100k {r['bpe_cl100k']['chars_per_token']:.3f}  hf {r['hf_bytelevel']['chars_per_token']:.3f}  "
                  f"comb {r['comb']['chars_per_token']:.3f}  case {r['comb_case_only']['chars_per_token']:.3f}  "
                  f"affix {r['comb_affix_only']['chars_per_token']:.3f}", flush=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(sorted(results, key=lambda x: (CORPORA.index(x["corpus"]), x["vocab"])), fh,
                          indent=1, ensure_ascii=False)
