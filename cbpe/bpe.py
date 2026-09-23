"""Generic character-level BPE with UTF-8 byte fallback.

Used both by the standard baseline tokenizer (trained on raw pretokens) and by the
combinatorial tokenizer (trained on case-normalised word cores). Training works on a
{string: frequency} table, so cost scales with the number of unique pretokens, not
with corpus size.
"""
from __future__ import annotations

import heapq
from collections import Counter, defaultdict

N_BYTES = 256


class CharBPE:
    """Vocabulary layout: [256 byte-fallback tokens] + [alphabet chars] + [merges]."""

    def __init__(self, alphabet: list[str], merges: list[tuple[str, str]], merge_counts=None):
        self.alphabet = list(alphabet)
        self.merges = list(merges)
        self.merge_counts = list(merge_counts or [])
        self.vocab: list[str | bytes] = [bytes([b]) for b in range(N_BYTES)] + self.alphabet
        self.vocab += [a + b for a, b in self.merges]
        self.tok2id: dict[str, int] = {}
        for i, t in enumerate(self.vocab):
            if isinstance(t, str):
                self.tok2id.setdefault(t, i)  # a merge can re-create an existing string; keep first
        self.ranks = {m: r for r, m in enumerate(self.merges)}
        self.chars = set(self.alphabet)
        self._cache: dict[str, list[str]] = {}

    def __len__(self):
        return len(self.vocab)

    def truncated(self, n_merges: int) -> "CharBPE":
        """BPE merge lists are prefix-closed, so truncating gives a valid smaller tokenizer."""
        return CharBPE(self.alphabet, self.merges[:n_merges], self.merge_counts[:n_merges])

    # ------------------------------------------------------------------ encode
    def split(self, word: str) -> list[str]:
        """Segment a string into pieces. Pieces are vocab strings, or single characters
        outside the alphabet (to be expanded to byte tokens by the caller)."""
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        syms = list(word)
        ranks = self.ranks
        while len(syms) > 1:
            best, best_i = None, -1
            for i in range(len(syms) - 1):
                r = ranks.get((syms[i], syms[i + 1]))
                if r is not None and (best is None or r < best):
                    best, best_i = r, i
            if best is None:
                break
            a, b = self.merges[best]
            out, i = [], 0
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(syms[i])
                    i += 1
            syms = out
        if len(self._cache) < 2_000_000:
            self._cache[word] = syms
        return syms

    def piece_ids(self, piece: str) -> list[int]:
        """Vocab id(s) for one piece from split(); unknown chars -> UTF-8 byte tokens."""
        i = self.tok2id.get(piece)
        if i is not None:
            return [i]
        assert len(piece) == 1, piece
        return list(piece.encode("utf-8"))

    def encode(self, word: str) -> list[int]:
        return [i for p in self.split(word) for i in self.piece_ids(p)]

    def id_bytes(self, i: int) -> bytes:
        t = self.vocab[i]
        return t if isinstance(t, bytes) else t.encode("utf-8")

    # ------------------------------------------------------------------- train
    @staticmethod
    def train(word_counts: dict[str, int], vocab_size: int, min_char_freq: int = 20,
              verbose: bool = False) -> "CharBPE":
        """Train so that len(tokenizer) <= vocab_size (fewer if merges run out)."""
        char_counts: Counter = Counter()
        for w, c in word_counts.items():
            for ch in w:
                char_counts[ch] += c
        alphabet = [ch for ch, c in char_counts.most_common() if c >= min_char_freq]
        # huge alphabets (CJK): at most half of the non-byte budget goes to single chars,
        # the rarest chars fall back to UTF-8 bytes
        alphabet = sorted(alphabet[:max(0, (vocab_size - N_BYTES) // 2)])
        n_merges = vocab_size - N_BYTES - len(alphabet)
        if n_merges < 0:
            raise ValueError(f"vocab_size {vocab_size} smaller than alphabet+bytes")

        # symbol ids; out-of-alphabet chars become -1 and never merge
        sym2id = {ch: i for i, ch in enumerate(alphabet)}
        id2sym = list(alphabet)
        words, freqs = [], []
        for w, c in word_counts.items():
            ids = [sym2id.get(ch, -1) for ch in w]
            if len(ids) > 1:
                words.append(ids)
                freqs.append(c)

        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for wi, (ids, f) in enumerate(zip(words, freqs)):
            for p in zip(ids, ids[1:]):
                if p[0] >= 0 and p[1] >= 0:
                    pair_counts[p] += f
                    where[p].add(wi)
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        merges, counts = [], []
        while len(merges) < n_merges and heap:
            negc, pair = heapq.heappop(heap)
            cur = pair_counts.get(pair, 0)
            if cur <= 0:
                continue
            if -negc != cur:  # stale entry: counts only decrease for old pairs
                heapq.heappush(heap, (-cur, pair))
                continue
            a, b = pair
            new_id = len(id2sym)
            id2sym.append(id2sym[a] + id2sym[b])
            merges.append((id2sym[a], id2sym[b]))
            counts.append(cur)
            if verbose and len(merges) % 2000 == 0:
                print(f"    merge {len(merges)}/{n_merges} count={cur} {id2sym[-1]!r}", flush=True)

            touched: dict[tuple[int, int], int] = defaultdict(int)
            for wi in where.pop(pair):
                ids = words[wi]
                if len(ids) < 2:
                    continue
                f = freqs[wi]
                # quick check the pair still occurs
                found = False
                for i in range(len(ids) - 1):
                    if ids[i] == a and ids[i + 1] == b:
                        found = True
                        break
                if not found:
                    continue
                for p in zip(ids, ids[1:]):
                    if p[0] >= 0 and p[1] >= 0:
                        touched[p] -= f
                out, i = [], 0
                while i < len(ids):
                    if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(ids[i])
                        i += 1
                words[wi] = out
                for p in zip(out, out[1:]):
                    if p[0] >= 0 and p[1] >= 0:
                        touched[p] += f
                        where[p].add(wi)
            for p, d in touched.items():
                if d == 0:
                    continue
                pair_counts[p] += d
                if d > 0:  # new pair (contains new_id) -> push; decreases handled lazily
                    heapq.heappush(heap, (-pair_counts[p], p))
            pair_counts.pop(pair, None)
        return CharBPE(alphabet, merges, counts)

    # ------------------------------------------------------------------ io
    def to_dict(self):
        return {"alphabet": self.alphabet, "merges": self.merges, "merge_counts": self.merge_counts}

    @staticmethod
    def from_dict(d):
        return CharBPE(d["alphabet"], [tuple(m) for m in d["merges"]], d.get("merge_counts"))
