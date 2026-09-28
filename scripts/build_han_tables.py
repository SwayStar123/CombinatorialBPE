"""Build the Traditional/Simplified tables from the OpenCC character dictionaries.

to_trad: simplified char -> the traditional form the "Traditional" variation produces.
fold:    traditional char -> simplified char, only where to_trad maps straight back to it,
         so folding + the "Traditional" variation is exactly invertible.

    python scripts/build_han_tables.py              -> cbpe/data/han_st.json
        to_trad = OpenCC's first (default) candidate.
    python scripts/build_han_tables.py --from-data  -> cbpe/data/han_st_data.json
        to_trad = the candidate used most often in Traditional-script documents of the
        Chinese tokenizer training text (documents are classified by counting characters that
        only exist in one script; Japanese is excluded, since it writes many characters in the
        same form as Simplified Chinese). A simplified char maps to nothing if its own form is the most
        common one there. When several simplified chars pick the same traditional form, the
        one more frequent in the text keeps it. Traditional forms that are not chosen stay
        literal, so the table is still exactly invertible.
"""
import json
import os
import sys
from collections import Counter

import opencc

D = os.path.join(os.path.dirname(opencc.__file__), "dictionary")
OUT = os.path.join(os.path.dirname(__file__), "..", "cbpe", "data")


def read(name):
    out = {}
    with open(os.path.join(D, name), encoding="utf-8") as f:
        for line in f:
            k, v = line.rstrip("\n").split("\t")
            if len(k) == 1:
                out[k] = v.split(" ")
    return out


def opencc_tables(st, ts):
    to_trad = {s: c[0] for s, c in st.items() if len(c[0]) == 1 and c[0] != s}
    fold = {t: c[0] for t, c in ts.items() if to_trad.get(c[0]) == t}
    return to_trad, fold


def data_tables(st, ts, text):
    base_to_trad, base_fold = opencc_tables(st, ts)
    trad_only = set(base_fold) - set(base_to_trad)          # forms that only occur in Traditional text
    simp_only = set(base_to_trad) - set(base_fold)          # forms that only occur in Simplified text
    trad_counts, all_counts, n_trad_docs = Counter(), Counter(text), 0
    for doc in text.split("\n\n"):
        t = sum(ch in trad_only for ch in doc)
        s = sum(ch in simp_only for ch in doc)
        if t > s and t >= 3:
            trad_counts.update(doc)
            n_trad_docs += 1
    # candidates: OpenCC's simplified -> traditional list plus every traditional char whose
    # traditional -> simplified entry points back (为's list only has 爲, but 為 -> 为)
    cand = {s: [c for c in cs if len(c) == 1] for s, cs in st.items()}
    for t, ss in ts.items():
        for s in ss:
            if len(s) == 1 and t != s and t not in cand.setdefault(s, []):
                cand[s].append(t)
    to_trad = {}
    for s, cands in cand.items():
        if s not in cands:
            cands = cands + [s]
        best = max(cands, key=lambda c: (trad_counts[c], -cands.index(c)))
        if best != s and trad_counts[best] > 0:
            to_trad[s] = best
        elif best != s and s in base_to_trad:          # no evidence in the data: keep OpenCC's default
            to_trad[s] = base_to_trad[s]
    # one simplified char per traditional form (the more frequent one keeps it)
    owner = {}
    for s, t in to_trad.items():
        if t not in owner or all_counts[s] > all_counts[owner[t]]:
            owner[t] = s
    to_trad = {s: t for s, t in to_trad.items() if owner[t] == s}
    fold = {t: s for s, t in to_trad.items()}
    print(f"  {n_trad_docs} Traditional-script documents")
    return to_trad, fold


if __name__ == "__main__":
    st, ts = read("STCharacters.txt"), read("TSCharacters.txt")
    if "--from-data" in sys.argv:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
        from build_mix import DATA, TOK_CHARS, docs_prefix
        to_trad, fold = data_tables(st, ts, docs_prefix(os.path.join(DATA, "wiki_zh.train.txt"), TOK_CHARS))
        name = "han_st_data.json"
    else:
        to_trad, fold = opencc_tables(st, ts)
        name = "han_st.json"
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump({"to_trad": to_trad, "fold": fold}, f, ensure_ascii=False)
    print(f"{name}: to_trad {len(to_trad)}  fold {len(fold)}  (of {len(ts)} traditional chars)")
