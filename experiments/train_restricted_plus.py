"""Restricted Combinatorial BPE + a few mined letter affixes (the middle ground on the real
restricted tokenizer): same data and settings as build_mix.py's "comb" tokenizer, plus the
closed-class letter prefixes / suffixes from experiments/mine_affixes.py.

    python experiments/mine_affixes.py pretrained/mix_32768_comb.json results/letter_affixes_restricted.json --min=50 --lift=1.5
    python experiments/train_restricted_plus.py results/letter_affixes_restricted.json results/tokenizers/mix_32768_combplus.json
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from build_mix import DATA, SOURCES, VOCAB, tok_text  # noqa: E402
from cbpe import CombinatorialBPE, load  # noqa: E402

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    affixes, out = sys.argv[1], sys.argv[2]
    lists = json.load(open(affixes, encoding="utf-8"))
    ref = load(os.path.join(HERE, "..", "pretrained", "mix_32768_comb.json"))
    t0 = time.time()
    tok = CombinatorialBPE.train(tok_text(), VOCAB, split_camel=True, fold_han=True, letter_prefixes=lists["prefixes"],
                                 letter_suffixes=lists["suffixes"], reference=ref)
    tok.save(out)
    print(f"trained in {time.time() - t0:.0f}s; sizes {tok.sizes}", flush=True)
    word_aff = [a for a in tok.prefixes + tok.suffixes if any(c.isalpha() for c in a)]
    print(f"{len(word_aff)} affix rows contain letters:", " ".join(repr(a) for a in word_aff[:80]))
    tok = load(out)
    for name in SOURCES:
        t = open(os.path.join(DATA, f"val_mix_{name}.txt"), encoding="utf-8").read()
        ids, rids = tok.encode(t), ref.encode(t)
        assert tok.decode(ids) == t, name
        print(f"  {name:11} {len(t) / len(ids):.2f} chars/token (restricted {len(t) / len(rids):.2f}), lossless", flush=True)
