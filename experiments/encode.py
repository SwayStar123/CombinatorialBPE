"""Tokenise a text file in parallel and save the ids as .npy ([T] or [T, 4] int32).
usage: encode.py TOKENIZER.json TEXT.txt OUT.npy [MAX_CHARS]

The file is streamed in ~4M-character pieces cut at blank lines and the ids are appended to
disk as they arrive, so neither the text nor the id array has to fit in memory at once."""
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cbpe import load  # noqa: E402

PIECE = 4_000_000
_tok = None


def _init(path):
    global _tok
    _tok = load(path)


def _enc(text):
    ids = _tok.encode(text)
    return np.asarray(ids, dtype=np.int32).reshape(len(ids), -1)


def pieces(path, max_chars=None):
    """Yield consecutive pieces of the file; every piece but the last ends right after a blank line."""
    left = max_chars if max_chars else float("inf")
    carry = ""
    with open(path, encoding="utf-8") as f:
        while left > 0:
            block = f.read(int(min(PIECE, left)))
            left -= len(block)
            if not block:
                break
            buf = carry + block
            cut = buf.rfind("\n\n")
            if cut < 0 or left <= 0:
                carry = buf
                continue
            yield buf[:cut + 2]
            carry = buf[cut + 2:]
    if carry:
        yield carry


if __name__ == "__main__":
    tok_path, text_path, out = sys.argv[1:4]
    max_chars = int(sys.argv[4]) if len(sys.argv) > 4 and int(sys.argv[4]) else None
    os.makedirs(os.path.dirname(out), exist_ok=True)
    raw = out + ".raw"
    n_tokens, n_cols, n_chars = 0, None, 0
    with open(raw, "wb") as fh, Pool(int(os.environ.get("WORKERS", 8)), initializer=_init,
                                     initargs=(tok_path,)) as pool:
        for text, arr in zip(pieces(text_path, max_chars),
                             pool.imap(_enc, pieces(text_path, max_chars), chunksize=1)):
            n_chars += len(text)
            n_cols = arr.shape[1]
            n_tokens += len(arr)
            fh.write(arr.tobytes())
    ids = np.memmap(raw, dtype=np.int32, mode="r", shape=(n_tokens, n_cols))
    np.save(out, ids)
    del ids
    os.remove(raw)
    print(f"encoded {n_chars} chars -> {n_tokens} tokens -> {out}")
