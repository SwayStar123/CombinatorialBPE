"""Build cbpe/data/han_st.json from the OpenCC character dictionaries.

to_trad: simplified char -> default traditional form (OpenCC's first candidate), for every
         simplified char whose default traditional form differs from itself.
fold:    traditional char -> simplified char, only where to_trad maps straight back to it,
         so folding + the "Traditional" variation is exactly invertible.
"""
import json
import os

import opencc

D = os.path.join(os.path.dirname(opencc.__file__), "dictionary")


def read(name):
    out = {}
    with open(os.path.join(D, name), encoding="utf-8") as f:
        for line in f:
            k, v = line.rstrip("\n").split("\t")
            if len(k) == 1:
                out[k] = v.split(" ")
    return out


st, ts = read("STCharacters.txt"), read("TSCharacters.txt")
to_trad = {s: c[0] for s, c in st.items() if len(c[0]) == 1 and c[0] != s}
fold = {t: c[0] for t, c in ts.items() if to_trad.get(c[0]) == t}
path = os.path.join(os.path.dirname(__file__), "..", "cbpe", "data", "han_st.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump({"to_trad": to_trad, "fold": fold}, f, ensure_ascii=False)
print(f"to_trad {len(to_trad)}  fold {len(fold)}  (of {len(ts)} traditional chars)")
