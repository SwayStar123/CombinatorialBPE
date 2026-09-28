"""Compression of production tokenizers (Kimi K3, GPT-4, GPT-4o) vs ours on the mixed validation sets.

Kimi K3's tokenizer files (tiktoken.model, tokenization_kimi.py) are read from data/kimi_k3/:
    from huggingface_hub import hf_hub_download
    for f in ["tiktoken.model", "tokenization_kimi.py"]:
        hf_hub_download("moonshotai/Kimi-K3", f, local_dir="data/kimi_k3")
Writes results/frontier.json.
"""
import ast
import json
import os
import sys

import tiktoken
from tiktoken.load import load_tiktoken_bpe

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cbpe import load  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
SOURCES = ["en", "de", "fr", "ru", "ja", "zh", "python", "javascript", "java", "cpp", "go"]


def kimi_k3():
    """Rebuild Kimi K3's tiktoken encoding from its published files (pat_str parsed from the source)."""
    src = open(os.path.join(DATA, "kimi_k3", "tokenization_kimi.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    pat = next(ast.literal_eval(node.value.args[0]) for node in ast.walk(tree)
               if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "pat_str")
    pat_str = "|".join(pat)
    ranks = load_tiktoken_bpe(os.path.join(DATA, "kimi_k3", "tiktoken.model"))
    return tiktoken.Encoding(name="kimi_k3", pat_str=pat_str, mergeable_ranks=ranks, special_tokens={}), len(ranks)


def count(enc, text):
    if isinstance(enc, tiktoken.Encoding):
        return len(enc.encode(text, disallowed_special=()))
    return len(enc.encode(text))


if __name__ == "__main__":
    kimi, kimi_vocab = kimi_k3()
    tokenizers = {
        f"Kimi K3 ({kimi_vocab // 1000}k)": kimi,
        "GPT-4o o200k (200k)": tiktoken.get_encoding("o200k_base"),
        "GPT-4 cl100k (100k)": tiktoken.get_encoding("cl100k_base"),
    }
    for name, path in [("BPE, ours (32k)", "mix_32768_bpe_gpt2.json"), ("Comb, ours (32k)", "mix_32768_comb.json"),
                       ("BPE, ours (160k)", "mix_163840_bpe_gpt2.json"), ("Comb, ours (160k)", "mix_163840_comb.json")]:
        p = os.path.join(ROOT, "results", "tokenizers", path)
        if os.path.exists(p):
            tokenizers[name] = load(p)
    texts = {s: open(os.path.join(DATA, f"val_mix_{s}.txt"), encoding="utf-8").read() for s in SOURCES}
    texts["all"] = open(os.path.join(DATA, "mix.test.txt"), encoding="utf-8").read()
    res = {"kimi_k3_vocab": kimi_vocab, "chars_per_token": {}}
    for name, enc in tokenizers.items():
        res["chars_per_token"][name] = {s: len(t) / count(enc, t) for s, t in texts.items()}
        print(f"{name:22s}", "  ".join(f"{s} {v:.2f}" for s, v in res["chars_per_token"][name].items()), flush=True)
    with open(os.path.join(ROOT, "results", "frontier.json"), "w") as f:
        json.dump(res, f, indent=1)
