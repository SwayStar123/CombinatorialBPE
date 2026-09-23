"""Download test corpora commonly used for tokenizer work.

* Wikipedia (wikimedia/wikipedia, 20231101 dump) for several languages. Only the
  first row groups of the first parquet shard are streamed (HTTP range reads), so
  we never pull the full multi-hundred-MB shards.
* TinyStories validation split (roneneldan/TinyStories), a popular small English
  corpus for tokenizer / small-LM experiments.

Each corpus is written as data/<name>.train.txt and data/<name>.test.txt, split by
document (documents joined with "\n\n").
"""
import argparse
import os
import random

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
WIKI_LANGS = ["en", "de", "fr", "ru", "tr", "hi"]


def write_split(name, docs, test_frac=0.1, seed=0):
    random.Random(seed).shuffle(docs)
    n_test = max(1, int(len(docs) * test_frac))
    test, train = docs[:n_test], docs[n_test:]
    for split, part in (("train", train), ("test", test)):
        path = os.path.join(DATA, f"{name}.{split}.txt")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n\n".join(part))
        print(f"  {path}: {len(part)} docs, {os.path.getsize(path) / 1e6:.1f} MB")


def wiki(lang, target_chars):
    fs = HfFileSystem()
    path = f"datasets/wikimedia/wikipedia/20231101.{lang}/train-00000-of-"
    shard = [p for p in fs.ls(f"datasets/wikimedia/wikipedia/20231101.{lang}", detail=False)
             if p.startswith(path)][0]
    docs, total = [], 0
    with fs.open(shard, "rb", block_size=8 * 2**20) as fh:
        pf = pq.ParquetFile(fh)
        for rg in range(pf.num_row_groups):
            for text in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
                text = text.strip()
                if len(text) < 200:
                    continue
                docs.append(text)
                total += len(text)
            if total >= target_chars:
                break
    print(f"wiki_{lang}: {len(docs)} docs, {total / 1e6:.1f}M chars")
    write_split(f"wiki_{lang}", docs)


def tinystories():
    p = hf_hub_download("roneneldan/TinyStories", "TinyStoriesV2-GPT4-valid.txt", repo_type="dataset")
    with open(p, encoding="utf-8") as f:
        docs = [d.strip() for d in f.read().split("<|endoftext|>") if d.strip()]
    print(f"tinystories: {len(docs)} docs")
    write_split("tinystories", docs)

def wiki_lm(lang="en", target_chars=250e6):
    """Extra text for LM training from shard 1 (disjoint from the tokenizer train/test shard 0)."""
    fs = HfFileSystem()
    shard = sorted(p for p in fs.ls(f"datasets/wikimedia/wikipedia/20231101.{lang}", detail=False)
                   if "train-00001-of-" in p)[0]
    out, total = [], 0
    with fs.open(shard, "rb", block_size=8 * 2**20) as fh:
        pf = pq.ParquetFile(fh)
        for rg in range(pf.num_row_groups):
            for text in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
                if len(text.strip()) >= 200:
                    out.append(text.strip())
                    total += len(text)
            if total >= target_chars:
                break
    path = os.path.join(DATA, f"wiki_{lang}_lm.txt")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n\n".join(out))
    print(f"{path}: {len(out)} docs, {total / 1e6:.0f}M chars")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=float, default=40e6, help="target characters per wiki language")
    ap.add_argument("--langs", nargs="*", default=WIKI_LANGS)
    ap.add_argument("--lm", action="store_true", help="also fetch ~250M chars of English for LM training")
    args = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    tinystories()
    for lang in args.langs:
        wiki(lang, args.chars)
    if args.lm:
        wiki_lm("en")
