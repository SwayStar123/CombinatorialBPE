"""Curated multilingual prose + code data (streamed from the Hugging Face Hub) into data/curated/.

    en                   HuggingFaceFW/fineweb-edu (sample-10BT): web text filtered for educational quality
    de fr hi ja zh       HuggingFaceFW/fineweb-2 (deu_Latn, fra_Latn, hin_Deva, jpn_Jpan, cmn_Hani):
                         deduplicated, quality-filtered Common Crawl per language
    python javascript    codeparrot/github-code-clean: deduplicated GitHub files with auto-generated,
    java cpp go          long-line and low-alphanumeric files removed

Per source, documents in stream order go to <name>.test.txt (the first TEST chars), then
<name>.train.txt (TOK chars, tokenizer training) and <name>.lm.txt (LM chars, LM training), all
disjoint, documents separated by a blank line. build_mix.py reads this layout with CBPE_DATA=curated.

--more: the next MORE chars after those into <name>.more.txt (more tokenizer training text; the
documents already in test/train/lm are streamed again and skipped, so all files stay disjoint).

    python scripts/download_curated.py [--only en,hi] [--more]
"""
import argparse
import os
import sys

from datasets import load_dataset

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "curated")
TEST, TOK, LM, MORE = 2_000_000, 50_000_000, 150_000_000, 150_000_000
PROSE = {"en": ("HuggingFaceFW/fineweb-edu", "sample-10BT"),
         **{k: ("HuggingFaceFW/fineweb-2", c) for k, c in
            (("de", "deu_Latn"), ("fr", "fra_Latn"), ("hi", "hin_Deva"), ("ja", "jpn_Jpan"), ("zh", "cmn_Hani"))}}
CODE = {"Python": "python", "JavaScript": "javascript", "Java": "java", "C++": "cpp", "GO": "go"}
CODE_FILES = "hf://datasets/codeparrot/github-code-clean/data/train-*.parquet"


class Writer:
    """test -> train -> lm files of one source, filled in that order by whole documents"""

    def __init__(self, name, more=False):
        self.name = name
        # (file, chars, written): with more, the existing parts are only counted, to skip them
        self.parts = [(p, n, not more) for p, n in (("test", TEST), ("train", TOK), ("lm", LM))]
        if more:
            self.parts.append(("more", MORE, True))
        self.i, self.chars = 0, 0
        self.f = self._open()

    def _open(self):
        if not self.parts[self.i][2]:
            return None
        return open(os.path.join(OUT, f"{self.name}.{self.parts[self.i][0]}.txt"), "w", encoding="utf-8", newline="\n")

    def add(self, doc):
        """False once all parts are full"""
        if self.i == len(self.parts):
            return False
        doc = doc.strip("\n").replace("\r\n", "\n")
        if not doc.strip():
            return True
        if self.f is not None:
            self.f.write(("\n\n" if self.chars else "") + doc)
        self.chars += len(doc)
        if self.chars >= self.parts[self.i][1]:
            if self.f is not None:
                self.f.close()
            print(f"  {self.name}.{self.parts[self.i][0]}.txt: {self.chars / 1e6:.1f}M chars"
                  + ("" if self.parts[self.i][2] else " (skipped)"), flush=True)
            self.i, self.chars = self.i + 1, 0
            if self.i < len(self.parts):
                self.f = self._open()
        return self.i < len(self.parts)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated source names")
    ap.add_argument("--more", action="store_true", help="download <name>.more.txt after the existing files")
    args = ap.parse_args()
    only = set(filter(None, args.only.split(",")))
    os.makedirs(OUT, exist_ok=True)
    for name, (repo, config) in PROSE.items():
        if only and name not in only:
            continue
        print(f"{name}: {repo} ({config})", flush=True)
        w = Writer(name, args.more)
        for row in load_dataset(repo, name=config, split="train", streaming=True):
            if not w.add(row["text"]):
                break
    want = {k: v for k, v in CODE.items() if not only or v in only}
    if want:
        print(f"code: {CODE_FILES} ({', '.join(want.values())})", flush=True)
        writers = {k: Writer(v, args.more) for k, v in want.items()}
        for row in load_dataset("parquet", data_files=CODE_FILES, split="train", streaming=True):
            w = writers.get(row["language"])
            if w is not None and not w.add(row["code"]):
                del writers[row["language"]]
                if not writers:
                    break


if __name__ == "__main__":
    main()
