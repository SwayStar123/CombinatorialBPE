"""Mixed prose + code experiment: 6 natural languages and 5 programming languages at once.

1. trains 32k tokenizers (standard BPE with GPT-2 / cl100k regex, and Combinatorial BPE with
   camelCase splitting and the Traditional-Chinese variation) on an equal mix of all sources
2. measures compression per source on held-out text  -> results/mix_compression.json
3. writes the LM training text (document-shuffled mix, disjoint from the tokenizer data and
   test sets) and validation files (one mixed, one per source) for experiments/lm.py

    python experiments/build_mix.py
    python experiments/lm.py results/tokenizers/mix_32768_bpe_cl100k.json ... \
        --train_text data/mix_lm.txt --val_text data/mix.test.txt --extra_val data/val_mix_*.txt
"""
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cbpe import CombinatorialBPE, StandardBPE, load  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results")
VOCAB = 32768
TOK_CHARS = 8_000_000     # per source, for tokenizer training
VAL_CHARS = 200_000       # per source, for validation
# name: (tokenizer train/test prefix, LM text file, LM chars)
SOURCES = {
    "en": ("wiki_en", "wiki_en_lm700.txt", 50e6),
    "de": ("wiki_de", "wiki_de_lm50.txt", 50e6),
    "fr": ("wiki_fr", "wiki_fr_lm50.txt", 50e6),
    "ru": ("wiki_ru", "wiki_ru_lm50.txt", 50e6),
    "ja": ("wiki_ja", "wiki_ja_lm50.txt", 50e6),
    "zh": ("wiki_zh", "wiki_zh_lm.txt", 50e6),
    "python": ("code_python", "code_python_lm50.txt", 50e6),
    "javascript": ("code_javascript", "code_javascript_lm900.txt", 50e6),
    "java": ("code_java", "code_java_lm50.txt", 50e6),
    "cpp": ("code_cpp", "code_cpp_lm50.txt", 50e6),
    "go": ("code_go", "code_go_lm50.txt", 25e6),
}
# larger LM text for the 3x-longer run (python experiments/build_mix.py --lm_only --lm_scale 3)
LM_FILES_BIG = {"en": "wiki_en_lm700.txt", "javascript": "code_javascript_lm900.txt",
                **{k: f"wiki_{k}_lm150.txt" for k in ("de", "fr", "ru", "ja", "zh")},
                **{k: f"code_{k}_lm150.txt" for k in ("python", "java", "cpp", "go")}}
TOKENIZERS = {
    "bpe_gpt2": lambda text: StandardBPE.train(text, VOCAB, "gpt2"),
    "bpe_cl100k": lambda text: StandardBPE.train(text, VOCAB, "cl100k"),
    "comb": lambda text: CombinatorialBPE.train(text, VOCAB, split_camel=True, fold_han=True),
}


def docs_prefix(path, n_chars):
    """Whole documents (separated by blank lines) from the start of a file, ~n_chars in total."""
    with open(path, encoding="utf-8") as f:
        text = f.read(int(n_chars))
    cut = text.rfind("\n\n")
    return text[:cut] if cut > 0 else text


def tok_text():
    return "\n\n".join(docs_prefix(os.path.join(DATA, f"{p}.train.txt"), TOK_CHARS) for p, _, _ in SOURCES.values())


def train_one(key):
    tok = TOKENIZERS[key](tok_text())
    path = os.path.join(OUT, "tokenizers", f"mix_{VOCAB}_{key}.json")
    tok.save(path)
    return key, path


def build_lm_text(scale=1, out="mix_lm.txt"):
    """Document-shuffled mix of every source's LM text (scale x the default amount)."""
    docs = []
    for name, (_, lm_file, n) in SOURCES.items():
        path = os.path.join(DATA, LM_FILES_BIG[name] if scale > 1 else lm_file)
        part = docs_prefix(path, n * scale).split("\n\n")
        # code files contain blank lines, so shuffling single paragraphs would scramble them;
        # shuffle contiguous blocks of 200 paragraphs instead (sources still interleave well)
        docs += ["\n\n".join(part[i:i + 200]) for i in range(0, len(part), 200)]
        print(f"  {name}: {sum(map(len, part)) / 1e6:.0f}M chars", flush=True)
        del part
    random.Random(0).shuffle(docs)
    with open(os.path.join(DATA, out), "w", encoding="utf-8", newline="\n") as f:
        for i, d in enumerate(docs):
            f.write(("\n\n" if i else "") + d)
    print(f"{out}: {sum(map(len, docs)) / 1e6:.0f}M chars in {len(docs)} blocks", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm_only", action="store_true", help="only (re)build the LM training text")
    ap.add_argument("--lm_scale", type=float, default=1, help="multiply the LM text per source")
    ap.add_argument("--lm_out", default="mix_lm.txt")
    args = ap.parse_args()
    if args.lm_only:
        build_lm_text(args.lm_scale, args.lm_out)
        raise SystemExit
    os.makedirs(os.path.join(OUT, "tokenizers"), exist_ok=True)

    # --- validation files
    vals = {}
    for name, (prefix, _, _) in SOURCES.items():
        vals[name] = docs_prefix(os.path.join(DATA, f"{prefix}.test.txt"), VAL_CHARS)
        with open(os.path.join(DATA, f"val_mix_{name}.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write(vals[name])
    with open(os.path.join(DATA, "mix.test.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n\n".join(vals.values()))

    # --- LM training text
    build_lm_text()

    # --- tokenizers
    with ProcessPoolExecutor(3) as ex:
        paths = dict(ex.map(train_one, TOKENIZERS))
    res = {}
    for key, path in paths.items():
        tok = load(path)
        res[key] = {"vocab_size": tok.vocab_size, "chars_per_token": {}}
        if isinstance(tok, CombinatorialBPE):
            res[key]["sizes"] = tok.sizes
        for name, text in vals.items():
            ids = tok.encode(text)
            assert tok.decode(ids) == text, (key, name)
            res[key]["chars_per_token"][name] = len(text) / len(ids)
        print(key, {k: round(v, 2) for k, v in res[key]["chars_per_token"].items()}, flush=True)
    with open(os.path.join(OUT, "mix_compression.json"), "w") as f:
        json.dump(res, f, indent=1)
