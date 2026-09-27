"""Mine letter affixes from a restricted-shaped tokenizer's cores (stage A of the middle ground).

A suffix y is supported by every learned core w = stem + y whose stem is itself a learned core;
a prefix x by every core w = x + rest with rest a learned core. No notion of words or spaces:
stems are the tokenizer's own units, so this works the same for any script (electrons =
electron + s, фронта = фронт + а, 的首都 = 的 + 首都). Productivity = number of distinct supporting
cores. Raw productivity rewards single letters (sport = s + port, stable = s + table), and short
remainders are cores by chance (nearly every 2-3 letter string is in a BPE vocabulary). So an
affix must beat chance: lift = supporting cores / expected supporting cores, where the expectation
sums, over every core starting (ending) with the affix, the base rate at which a remainder of that
length is a core (measured over all splits of all cores). Keeps affixes with productivity >= --min
(default 100) and lift >= --lift (default 2), at most --max per side.

    python experiments/mine_affixes.py results/tokenizers/mix_32768_dict_search_hybrid_stageA_fast.bin \
        results/letter_affixes.json [--fast] [--min=150] [--max=200]
"""
import json
import os
import sys
from collections import Counter

import regex

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import build_mix  # noqa: E402
import bench_dictsearch as U  # noqa: E402
from bench_dictsearch import DModel  # noqa: E402

LETTERS = regex.compile(r"^[\p{L}\p{M}]+$")

if __name__ == "__main__":
    model, out = sys.argv[1], sys.argv[2]
    opt = lambda k, d: next((type(d)(a.split("=", 1)[1]) for a in sys.argv if a.startswith(f"--{k}=")), d)  # noqa: E731
    sys.stdout.reconfigure(encoding="utf-8")
    min_prod, max_n, max_len, min_lift = opt("min", 100), opt("max", 200), opt("maxlen", 4), opt("lift", 2.0)
    if model.endswith(".json"):  # a cbpe tokenizer (e.g. the restricted Combinatorial BPE)
        from cbpe import load
        tok = load(model)
        core_strings = [v for v in tok.core.vocab if isinstance(v, str)]
    else:  # a native/dict_search model
        if "--fast" in sys.argv:
            build_mix.TOK_CHARS = 2_000_000
        alpha, _, _ = U.build_symbols(Counter(build_mix.tok_text()), 32768, fold=True)
        core_strings = DModel(model, alpha).cores
    cores = {c for c in core_strings if len(c) > 1 and LETTERS.match(c)}
    # base rate: how often a remainder of length k is a core, over all splits of all cores
    hit, tot = {"prefixes": Counter(), "suffixes": Counter()}, {"prefixes": Counter(), "suffixes": Counter()}
    for w in cores:
        for i in range(1, len(w)):
            x, y = w[:i], w[i:]
            if len(x) <= max_len:
                tot["prefixes"][len(y)] += 1
                hit["prefixes"][len(y)] += y in cores
            if len(y) <= max_len:
                tot["suffixes"][len(x)] += 1
                hit["suffixes"][len(x)] += x in cores
    base = {side: {k: hit[side][k] / tot[side][k] for k in tot[side]} for side in tot}
    prod = {"prefixes": Counter(), "suffixes": Counter()}
    expect = {"prefixes": Counter(), "suffixes": Counter()}
    for w in cores:
        for i in range(1, len(w)):
            x, y = w[:i], w[i:]
            if len(x) <= max_len:
                expect["prefixes"][x] += base["prefixes"][len(y)]
                prod["prefixes"][x] += y in cores
            if len(y) <= max_len:
                expect["suffixes"][y] += base["suffixes"][len(x)]
                prod["suffixes"][y] += x in cores
    res = {}
    for side in ("prefixes", "suffixes"):
        ratio = {a: n / expect[side][a] for a, n in prod[side].items() if expect[side][a] > 0}
        keep = [a for a, n in prod[side].most_common() if n >= min_prod and ratio.get(a, 0) >= min_lift][:max_n]
        res[side] = keep
        print(f"{side}: {len(keep)} kept (>= {min_prod} supporting cores, lift >= {min_lift})")
        print("  kept:", " ".join(f"{a}({prod[side][a]},{ratio[a]:.2f})" for a in keep))
        rej = [(a, n) for a, n in prod[side].most_common() if a not in keep and a in ratio][:40]
        print("  most productive rejected:", " ".join(f"{a}({n},{ratio[a]:.2f})" for a, n in rej))
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
