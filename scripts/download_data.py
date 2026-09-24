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


CODE_LANGS = {"Python": 20e6, "Java": 20e6, "JavaScript": 20e6, "C++": 20e6, "GO": 10e6}


def code(targets=CODE_LANGS, test_frac=0.1):
    """Source code from codeparrot/github-code-clean (ungated, deduplicated GitHub files).

    Train/test are split by repository (hash of repo_name), so near-duplicate files from one
    project never end up on both sides. Minified / generated files are skipped."""
    import zlib
    fs = HfFileSystem()
    root = "datasets/codeparrot/github-code-clean/data"
    shards = sorted(p for p in fs.ls(root, detail=False) if p.endswith(".parquet"))
    got = {l: {"train": [], "test": []} for l in targets}
    chars = {l: 0 for l in targets}
    for shard in shards:
        with fs.open(shard, "rb", block_size=8 * 2**20) as fh:
            pf = pq.ParquetFile(fh)
            for rg in range(pf.num_row_groups):
                t = pf.read_row_group(rg, columns=["language", "repo_name", "code"])
                for lang, repo, src in zip(*(t.column(c).to_pylist() for c in ("language", "repo_name", "code"))):
                    if lang not in targets or chars[lang] >= targets[lang] * (1 + test_frac):
                        continue
                    if len(src) > 100_000 or max(map(len, src.splitlines() or [""])) > 1000:
                        continue  # minified / generated
                    split = "test" if zlib.crc32(repo.encode()) % 100 < 100 * test_frac else "train"
                    got[lang][split].append(src.strip("\n"))
                    chars[lang] += len(src)
                print("  " + "  ".join(f"{l} {chars[l] / 1e6:.1f}M" for l in targets), flush=True)
                if all(chars[l] >= targets[l] * (1 + test_frac) for l in targets):
                    break
        if all(chars[l] >= targets[l] * (1 + test_frac) for l in targets):
            break
    for lang, parts in got.items():
        name = "code_" + {"C++": "cpp", "GO": "go"}.get(lang, lang.lower())
        for split, docs in parts.items():
            path = os.path.join(DATA, f"{name}.{split}.txt")
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write("\n\n".join(docs))
            print(f"  {path}: {len(docs)} files, {os.path.getsize(path) / 1e6:.1f} MB")


def code_lm(lang="JavaScript", target_chars=300e6, first_shard=1, test_frac=0.1):
    """LM training text for one language: shards after the tokenizer data, and only repositories
    on the train side of the same repo-hash split, so the test repos stay unseen."""
    import zlib
    fs = HfFileSystem()
    root = "datasets/codeparrot/github-code-clean/data"
    shards = sorted(p for p in fs.ls(root, detail=False) if p.endswith(".parquet"))[first_shard:]
    out, total = [], 0
    for shard in shards:
        with fs.open(shard, "rb", block_size=8 * 2**20) as fh:
            pf = pq.ParquetFile(fh)
            for rg in range(pf.num_row_groups):
                t = pf.read_row_group(rg, columns=["language", "repo_name", "code"])
                for l, repo, src in zip(*(t.column(c).to_pylist() for c in ("language", "repo_name", "code"))):
                    if l != lang or zlib.crc32(repo.encode()) % 100 < 100 * test_frac:
                        continue
                    if len(src) > 100_000 or max(map(len, src.splitlines() or [""])) > 1000:
                        continue
                    out.append(src.strip("\n"))
                    total += len(src)
                print(f"  {lang} {total / 1e6:.0f}M chars", flush=True)
                if total >= target_chars:
                    break
        if total >= target_chars:
            break
    name = "code_" + {"C++": "cpp", "GO": "go"}.get(lang, lang.lower())
    path = os.path.join(DATA, f"{name}_lm.txt")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n\n".join(out))
    print(f"{path}: {len(out)} files, {total / 1e6:.0f}M chars")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", type=float, default=40e6, help="target characters per wiki language")
    ap.add_argument("--langs", nargs="*", default=WIKI_LANGS)
    ap.add_argument("--lm", action="store_true", help="also fetch ~250M chars of English for LM training")
    ap.add_argument("--code", action="store_true", help="only fetch source code corpora")
    args = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    if args.code:
        code()
        raise SystemExit
    tinystories()
    for lang in args.langs:
        wiki(lang, args.chars)
    if args.lm:
        wiki_lm("en")
