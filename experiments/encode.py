"""Tokenise a text file in parallel and save the ids as .npy ([T] or [T, 4] int32).
usage: encode.py TOKENIZER.json TEXT.txt OUT.npy [MAX_CHARS]"""
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cbpe import load  # noqa: E402

_tok = None


def _init(path):
    global _tok
    _tok = load(path)


def _enc(text):
    ids = _tok.encode(text)
    return np.asarray(ids, dtype=np.int32).reshape(len(ids), -1)


if __name__ == "__main__":
    tok_path, text_path, out = sys.argv[1:4]
    max_chars = int(sys.argv[4]) if len(sys.argv) > 4 and int(sys.argv[4]) else None
    with open(text_path, encoding="utf-8") as f:
        text = f.read(max_chars) if max_chars else f.read()
    docs = text.split("\n\n")
    chunks = ["\n\n".join(docs[i:i + 200]) for i in range(0, len(docs), 200)]
    chunks = [c + ("\n\n" if i < len(chunks) - 1 else "") for i, c in enumerate(chunks)]
    assert "".join(chunks) == text
    with Pool(int(os.environ.get("WORKERS", 8)), initializer=_init, initargs=(tok_path,)) as pool:
        arr = np.concatenate(pool.map(_enc, chunks, chunksize=4))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.save(out, arr)
    print(f"encoded {len(text)} chars -> {len(arr)} tokens -> {out}")
