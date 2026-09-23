"""English compression vs vocabulary size, extended to very large standard vocabularies.

Used to pick token-count-matched pairs for the LM comparison (experiments/lm.py) and for
figures/equal_tokens.svg. Writes results/iso_compression.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cbpe import CombinatorialBPE, StandardBPE  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
STD_SIZES = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
COMB_SIZES = [4096, 8192, 9300, 10000, 16384, 32768, 65536]

if __name__ == "__main__":
    with open(os.path.join(ROOT, "data", "wiki_en.train.txt"), encoding="utf-8") as f:
        train = f.read()
    with open(os.path.join(ROOT, "data", "wiki_en.test.txt"), encoding="utf-8") as f:
        test = f.read(3_000_000)
    out = {"corpus": "wiki_en", "standard_gpt2": [], "combinatorial": []}
    for key, sizes, make in (("standard_gpt2", STD_SIZES, lambda v: StandardBPE.train(train, v, "gpt2")),
                             ("combinatorial", COMB_SIZES, lambda v: CombinatorialBPE.train(train, v))):
        for v in sizes:
            tok = make(v)
            n = len(tok.encode(test))
            out[key].append({"vocab": v, "rows": tok.vocab_size, "chars_per_token": len(test) / n})
            print(key, v, f"{len(test) / n:.3f}", flush=True)
    with open(os.path.join(ROOT, "results", "iso_compression.json"), "w") as f:
        json.dump(out, f, indent=1)
