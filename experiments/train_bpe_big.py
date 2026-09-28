"""Standard BPE (GPT-2 pretokenisation, byte fallback) on the same tokenizer text as
bench_dictsearch.py --tokdata (curated data: <name>.train.txt, then the end of <name>.lm.txt, then
<name>.more.txt), counted source by source so multi-GB texts never go through one regex call.

    CBPE_DATA=curated python experiments/train_bpe_big.py --vocab 131072 --tokdata 320000000 --tag d320
"""
import argparse
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from build_mix import DATA, LM_FILES_BIG, SOURCES, docs_prefix  # noqa: E402
from cbpe.bpe import CharBPE  # noqa: E402
from cbpe.tokenizers import CL100K_PAT, GPT2_PAT, StandardBPE  # noqa: E402

BLOCK = 1 << 25


def source_text(name, tokdata):
    """The same documents bench_dictsearch.py uses for this source (see its --tokdata)."""
    prefix = SOURCES[name][0]
    parts = [docs_prefix(os.path.join(DATA, f"{prefix}.train.txt"), min(tokdata, 50_000_000))]
    if tokdata > 50_000_000:
        from_lm = min(tokdata - 50_000_000, 146_000_000)
        with open(os.path.join(DATA, LM_FILES_BIG[name]), encoding="utf-8") as f:
            lm = f.read()
        tail = lm[-from_lm:]
        parts.append(tail[tail.find("\n\n") + 2:])
        del lm
        if tokdata - 50_000_000 > from_lm:
            parts.append(docs_prefix(os.path.join(DATA, f"{prefix}.more.txt"), tokdata - 50_000_000 - from_lm))
    return "\n\n".join(parts)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=131072)
    ap.add_argument("--tokdata", type=int, default=320_000_000)
    ap.add_argument("--pattern", default="gpt2")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    pat = {"gpt2": GPT2_PAT, "cl100k": CL100K_PAT}[args.pattern]
    counts = Counter()
    for name in SOURCES:
        text = source_text(name, args.tokdata)
        # blocks cut at whitespace, so no pretoken crosses a block edge
        start = 0
        while start < len(text):
            end = min(len(text), start + BLOCK)
            if end < len(text):
                nxt = text.find(" ", end)
                end = len(text) if nxt < 0 else nxt
            counts.update(m.group() for m in pat.finditer(text[start:end]))
            start = end
        print(f"  {name}: {len(text) / 1e6:.0f}M chars, {len(counts) / 1e6:.1f}M unique pretokens so far", flush=True)
        del text
    tok = StandardBPE(CharBPE.train(counts, args.vocab, 20, True), args.pattern)
    out = os.path.join(HERE, "..", "results", "tokenizers",
                       f"mix_{args.vocab}_bpe_{args.pattern}" + (f"_{args.tag}" if args.tag else "") + ".json")
    tok.save(out)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
