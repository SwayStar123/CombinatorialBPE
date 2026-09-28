"""Framework-free helpers shared by the PyTorch (lm.py) and JAX (jax_lm.py) LM experiments."""
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CACHE = os.path.join(ROOT, "data", "cache")


def rel(path):
    """Project-relative path for logs (absolute if it lives on another drive)."""
    try:
        return os.path.relpath(path, ROOT).replace("\\", "/")
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")


def cache_path(tok_path, text_path, max_chars=None, cache=CACHE):
    key = f"{os.path.basename(tok_path)[:-5]}__{os.path.basename(text_path)[:-4]}_{max_chars}.npy"
    return os.path.join(cache, key)


def encode_file(tok_path, text_path, max_chars=None, cache=CACHE, mmap_mode=None, write_meta=False):
    """Tokenise in a torch-free subprocess (encode.py) and cache the result as .npy.
    Returns (ids [T] or [T, F] int32, number of UTF-8 bytes of the (possibly truncated) text).

    write_meta=True also stores the byte count next to the cache file (<cache>.meta.json), so the
    .npy can be used on a machine without the text file (e.g. a TPU VM); the sidecar is only read
    when the text file is missing."""
    out = cache_path(tok_path, text_path, max_chars, cache)
    meta = out[:-4] + ".meta.json"
    if not os.path.exists(out):
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "encode.py"),
                        tok_path, text_path, out, str(max_chars or 0)], check=True)
    if not os.path.exists(text_path) and os.path.exists(meta):
        with open(meta) as f:
            n_bytes = json.load(f)["n_bytes"]
    elif max_chars:
        with open(text_path, encoding="utf-8") as f:
            n_bytes = len(f.read(max_chars).encode("utf-8"))
    else:
        n_bytes = os.path.getsize(text_path)  # our text files are UTF-8 with \n newlines
    if write_meta and not os.path.exists(meta):
        with open(meta, "w") as f:
            json.dump({"n_bytes": n_bytes, "text": os.path.basename(text_path), "max_chars": max_chars}, f)
    return np.load(out, mmap_mode=mmap_mode), n_bytes
