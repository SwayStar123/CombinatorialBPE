"""Chinese: does a hand-coded Traditional/Simplified variation help, like case does for Latin?

Compares, at equal embedding budget on wiki_zh (a ~45/55 Traditional/Simplified mix):
  comb      - combinatorial BPE with case variations only
  comb_han  - plus the "Traditional" variation (cores stored in Simplified)
Also reports how much of each vocabulary is a Traditional/Simplified duplicate of another entry.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cbpe import CombinatorialBPE, load  # noqa: E402
from cbpe.tokenizers import HAN_FOLD, V_TRAD  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
SIZES = [4096, 8192, 16384, 32768, 65536]


def script_dupes(strings):
    """Fraction of entries whose Simplified-folded form equals another entry's."""
    seen = {}
    for s in strings:
        seen.setdefault("".join(HAN_FOLD.get(c, c) for c in s), []).append(s)
    return sum(len(v) - 1 for v in seen.values()) / max(1, len(strings))


def run(size):
    with open(os.path.join(ROOT, "data", "wiki_zh.train.txt"), encoding="utf-8") as f:
        train = f.read(40_000_000)
    with open(os.path.join(ROOT, "data", "wiki_zh.test.txt"), encoding="utf-8") as f:
        test = f.read(3_000_000)
    out = {"vocab": size}
    tok = CombinatorialBPE.train(train, size, fold_han=True)
    tok.save(os.path.join(RES, "tokenizers", f"wiki_zh_{size}_comb_han.json"))
    ids = tok.encode(test)
    assert tok.decode(ids) == test
    out["comb_han"] = {"chars_per_token": len(test) / len(ids), "sizes": tok.sizes,
                       "trad_tokens": sum(t[0] == V_TRAD for t in ids) / len(ids),
                       "core_script_dupes": script_dupes([t for t in tok.core.vocab if isinstance(t, str)])}
    for key in ("comb", "bpe_gpt2", "bpe_cl100k"):
        p = os.path.join(RES, "tokenizers", f"wiki_zh_{size}_{key}.json")
        t = load(p)
        n = len(t.encode(test))
        vocab = t.core.vocab if key == "comb" else t.bpe.vocab
        out[key] = {"chars_per_token": len(test) / n,
                    "script_dupes": script_dupes([v for v in vocab if isinstance(v, str)])}
    return out


if __name__ == "__main__":
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", 3))) as ex:
        results = list(ex.map(run, SIZES))
    with open(os.path.join(RES, "han.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    for r in results:
        print(f"{r['vocab']:6d}  gpt2 {r['bpe_gpt2']['chars_per_token']:.3f}  cl100k {r['bpe_cl100k']['chars_per_token']:.3f}"
              f"  comb {r['comb']['chars_per_token']:.3f}  comb_han {r['comb_han']['chars_per_token']:.3f}"
              f"  trad-tokens {100 * r['comb_han']['trad_tokens']:.0f}%"
              f"  script-dupes: cl100k {100 * r['bpe_cl100k']['script_dupes']:.1f}% comb {100 * r['comb']['script_dupes']:.1f}%"
              f" comb_han {100 * r['comb_han']['core_script_dupes']:.1f}%", flush=True)
