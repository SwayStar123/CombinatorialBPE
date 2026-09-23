"""Generate the explanatory SVG figures in figures/ from the shipped pretrained tokenizers.

    python scripts/make_figures.py
"""
import json
import os
import sys
import unicodedata
from html import escape

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from cbpe import load  # noqa: E402
from cbpe.tokenizers import V_CAP, V_TRAD, V_UPPER  # noqa: E402

OUT = os.path.join(ROOT, "figures")
PRE = os.path.join(ROOT, "pretrained")

# palette (validated categorical slots; tints for fills, full hue for borders)
INK, INK2, INK3 = "#1f1f1c", "#55544f", "#8c8b85"
BG, CARD_LINE = "#ffffff", "#e3e2dc"
HUE = {"pre": ("#2a78d6", "#dbe9fb"), "core": ("#138a60", "#d3f0e4"), "suf": ("#d4561f", "#fbe0d3"),
       "var": ("#b27800", "#fbecc4"), "std": ("#8c8b85", "#efeee9"), "dup": ("#c2415f", "#f9dbe3")}
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"
SANS = "-apple-system, 'Segoe UI', Roboto, Helvetica, Arial, 'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif"
FS = 15          # token font size
CW = FS * 0.61   # monospace advance for narrow chars
PAD = 7


def text_w(s):
    return sum(FS * 1.02 if unicodedata.east_asian_width(c) in "WF" else CW for c in s)


def show(s):
    """Make spaces / newlines visible inside tokens."""
    return s.replace(" ", "·").replace("\n", "↵")


class SVG:
    def __init__(self, w, h):
        self.w, self.h, self.parts = w, h, []

    def add(self, s):
        self.parts.append(s)

    def rect(self, x, y, w, h, fill, stroke=None, rx=6, sw=1.2, extra=""):
        st = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{st} {extra}/>')

    def text(self, x, y, s, size=14, fill=INK, anchor="start", weight=400, family=SANS, extra=""):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
                 f'font-weight="{weight}" font-family="{family}" {extra}>{escape(s)}</text>')

    def line(self, x1, y1, x2, y2, stroke=INK3, sw=1.5, extra=""):
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}" {extra}/>')

    def save(self, name, title, desc):
        body = "\n".join(self.parts)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
               f'height="{self.h}" role="img" aria-labelledby="t d">\n<title id="t">{escape(title)}</title>\n'
               f'<desc id="d">{escape(desc)}</desc>\n'
               '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
               f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK3}"/></marker></defs>\n'
               f'<rect width="{self.w}" height="{self.h}" rx="14" fill="{BG}" stroke="{CARD_LINE}"/>\n{body}\n</svg>\n')
        with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as f:
            f.write(svg)
        print("wrote", name)


# ------------------------------------------------------------------ chips
def std_chip(svg, x, y, s, kind="std"):
    w = text_w(show(s)) + 2 * PAD
    stroke, fill = HUE[kind]
    svg.rect(x, y, w, 30, fill, stroke)
    draw_token_text(svg, x + PAD, y + 20, s)
    return w


def draw_token_text(svg, x, y, s, color=INK):
    # visible-space glyphs in a muted ink, the rest in normal ink
    out, cx = [], x
    for ch in s:
        g = show(ch)
        col = INK3 if g != ch else color
        out.append(f'<tspan x="{cx:.1f}" fill="{col}">{escape(g)}</tspan>')
        cx += text_w(g)
    svg.add(f'<text y="{y:.1f}" font-size="{FS}" font-family="{MONO}">{"".join(out)}</text>')


VAR_BADGE = {V_CAP: "Aa", V_UPPER: "AA", V_TRAD: "繁"}


def comb_chip(svg, x, y, pre, core, suf, var):
    segs = [(k, s) for k, s in (("pre", pre), ("core", core), ("suf", suf)) if s]
    widths = [text_w(show(s)) + 2 * PAD for _, s in segs]
    total = sum(widths)
    cx = x
    for (k, s), w in zip(segs, widths):
        stroke, fill = HUE[k]
        svg.rect(cx, y, w, 30, fill, rx=0)
        draw_token_text(svg, cx + PAD, y + 20, s)
        cx += w
    # outline + segment separators
    svg.rect(x, y, total, 30, "none", INK2, rx=6, sw=1.3)
    cx = x
    for w in widths[:-1]:
        cx += w
        svg.line(cx, y + 1, cx, y + 29, stroke="#ffffff", sw=2)
    if var in VAR_BADGE:
        stroke, fill = HUE["var"]
        bw = 26 if var != V_TRAD else 22
        svg.rect(x + total - bw + 4, y - 11, bw, 17, fill, stroke, rx=8, sw=1)
        svg.text(x + total - bw / 2 + 4, y + 2, VAR_BADGE[var], size=11, fill=INK, anchor="middle", weight=600)
    return total


def flow(svg, x0, y0, width, items, draw, gap=8, row_h=48):
    x, y = x0, y0
    for it in items:
        # measure by drawing into a scratch SVG
        scratch = SVG(0, 0)
        w = draw(scratch, 0, 0, *it)
        if x + w > x0 + width:
            x, y = x0, y + row_h
        draw(svg, x, y, *it)
        x += w + gap
    return y + row_h


def legend(svg, x, y, entries):
    for label, kind in entries:
        stroke, fill = HUE[kind]
        svg.rect(x, y - 11, 14, 14, fill, stroke, rx=3, sw=1)
        svg.text(x + 20, y, label, size=13, fill=INK2)
        x += 28 + len(label) * 7.1
    return x


# ------------------------------------------------------------ figure 1
def fig_tokenization():
    std = load(os.path.join(PRE, "wiki_en_16384_bpe_gpt2.json"))
    comb = load(os.path.join(PRE, "wiki_en_16384_comb.json"))
    zh = load(os.path.join(PRE, "wiki_zh_16384_comb_han.json"))
    s = 'In 1969, NASA landed on the Moon (the first crewed landing). "Amazing!" said the President.'
    std_toks = [(std.decode([i]),) for i in std.encode(s)]
    comb_toks = [(comb.prefixes[p], comb.core.vocab[c], comb.suffixes[x], v) for v, p, c, x in comb.encode(s)]

    W = 1000
    svg = SVG(W, 560)
    svg.text(32, 44, "Same text, two tokenizers (both with 16,384 embedding rows)", size=20, weight=650)
    svg.text(32, 70, s, size=14, fill=INK2, family=MONO)

    svg.text(32, 112, f"Standard BPE (GPT-2 style) — {len(std_toks)} tokens", size=15, weight=600)
    y = flow(svg, 32, 126, W - 64, std_toks, std_chip)
    svg.text(32, y + 14, f"Combinatorial BPE — {len(comb_toks)} tokens, each one (variation, prefix, core, suffix)",
             size=15, weight=600)
    y = flow(svg, 32, y + 38, W - 64, comb_toks, comb_chip, gap=10, row_h=52)

    # Chinese: Traditional and Simplified share cores
    svg.text(32, y + 18, "Chinese: Traditional and Simplified spellings share the same cores", size=15, weight=600)
    y += 42
    for t in ["臺灣位於東亞。", "台湾位于东亚。"]:
        toks = [(zh.prefixes[p], zh.core.vocab[c], zh.suffixes[x], v) for v, p, c, x in zh.encode(t)]
        svg.text(32, y + 20, t, size=16, family=SANS)
        flow(svg, 190, y + 4, W - 220, toks, comb_chip, gap=10)
        y += 44
    y += 10
    x = legend(svg, 32, y, [("prefix", "pre"), ("core", "core"), ("suffix", "suf"), ("standard token", "std")])
    stroke, fill = HUE["var"]
    svg.rect(x, y - 12, 24, 16, fill, stroke, rx=8, sw=1)
    svg.text(x + 12, y + 1, "Aa", size=11, anchor="middle", weight=600)
    svg.text(x + 32, y, "variation: Aa = Capitalised, AA = UPPER, 繁 = Traditional, none = as-is", size=13, fill=INK2)
    svg.text(32, y + 24, "· marks a space. Cores are stored case-folded (and in Simplified for Chinese); "
             "the variation restores the surface form.", size=12, fill=INK3)
    svg.h = y + 44
    svg.save("tokenization.svg", "Standard BPE vs Combinatorial BPE on the same text",
             f"Standard BPE needs {len(std_toks)} tokens, Combinatorial BPE {len(comb_toks)}; "
             "Traditional and Simplified Chinese map to identical cores.")


# ------------------------------------------------------------ figure 2
def box(svg, x, y, w, h, title, kind, lines=(), title_size=14):
    stroke, fill = HUE[kind]
    svg.rect(x, y, w, h, fill, stroke, rx=10, sw=1.4)
    svg.text(x + w / 2, y + 24, title, size=title_size, anchor="middle", weight=650)
    for i, ln in enumerate(lines):
        mono = ln.startswith("'") or ln.startswith('"')
        svg.text(x + w / 2, y + 46 + 19 * i, ln, size=12 if mono else 13, anchor="middle", fill=INK2,
                 family=MONO if mono else SANS, extra='xml:space="preserve"')


def arrow(svg, x1, y1, x2, y2):
    svg.line(x1, y1, x2, y2, stroke=INK3, sw=1.6, extra='marker-end="url(#arr)"')


def fig_factorization():
    W, H = 1000, 620
    svg = SVG(W, H)
    svg.text(32, 44, "How Combinatorial BPE works", size=20, weight=650)

    # --- step 1: surface forms collapse onto one core
    svg.text(32, 84, "1  Tokenize: split each word into a 4-tuple", size=15, weight=600)
    forms = [("' the'", "–", "' '", "the", ""), ("'The'", "Aa", "", "the", ""), ("' THE,'", "AA", "' '", "the", "','"),
             ("' (the'", "–", "' ('", "the", ""), ("'the.\"'", "–", "", "the", "'.\"'")]
    y = 104
    for surf, v, p, c, s in forms:
        svg.text(32, y + 20, surf, size=14, family=MONO)
        arrow(svg, 118, y + 15, 150, y + 15)
        cx = 160
        for label, kind in ((v, "var"), (p or "∅", "pre"), (c, "core"), (s or "∅", "suf")):
            stroke, fill = HUE[kind]
            wv = 62 if kind != "core" else 70
            svg.rect(cx, y + 2, wv, 26, fill, stroke, rx=5, sw=1)
            svg.text(cx + wv / 2, y + 20, label, size=13, anchor="middle", family=MONO,
                     fill=INK if label not in ("∅", "–") else INK3)
            cx += wv + 6
        y += 36
    svg.text(160, y + 16, "variation   prefix     core      suffix", size=12, fill=INK3, family=MONO,
             extra='xml:space="preserve"')
    svg.text(32, y + 44, "Standard BPE stores each of these as a separate vocabulary row;", size=13, fill=INK2)
    svg.text(32, y + 63, "here they all share the single core row  the.", size=13, fill=INK2)

    # --- step 2: learned inventories
    x2 = 470
    svg.text(x2, 84, "2  Learn the inventories (16k budget, English)", size=15, weight=600)
    box(svg, x2, 102, 100, 118, "variation", "var", ["hand-coded", "as-is", "Aa  AA", "(繁 for zh)"])
    box(svg, x2 + 110, 102, 115, 118, "prefix", "pre", ["learned", "108 rows", "' '  '\\n\\n'", "' ('  ' \"'"])
    box(svg, x2 + 235, 102, 125, 118, "core", "core", ["learned BPE", "16,193 rows", "hello  apollo", "1969  台湾"])
    box(svg, x2 + 370, 102, 118, 118, "suffix", "suf", ["learned", "80 rows", "','  '.'  '),'", "'.\"'  '。'"])
    svg.text(x2, 250, "The affix/core split is learned: an affix is kept only if it saves", size=13, fill=INK2)
    svg.text(x2, 269, "more tokens than the core merge it would displace.", size=13, fill=INK2)
    svg.text(x2, 294, "3 + 108 + 16,193 + 80 = 16,384 rows = same as a 16k standard vocab.", size=13, fill=INK2,
             weight=600)

    # --- step 3: model
    y0 = 395
    svg.line(32, y0 - 20, W - 32, y0 - 20, stroke=CARD_LINE, sw=1)
    svg.text(32, y0 + 4, "3  Model: sum the four embeddings in, predict the four parts out", size=15, weight=600)
    ty = y0 + 36
    ex = [("var", "E_var[Aa]"), ("pre", "E_pre['·\"']"), ("core", "E_core[amazing]"), ("suf", "E_suf['!\"']")]
    for i, (k, lab) in enumerate(ex):
        stroke, fill = HUE[k]
        svg.rect(32, ty + i * 34, 150, 26, fill, stroke, rx=5, sw=1)
        svg.text(107, ty + i * 34 + 18, lab, size=13, anchor="middle", family=MONO)
        arrow(svg, 184, ty + i * 34 + 13, 214, ty + 65)
    svg.add(f'<circle cx="228" cy="{ty + 65}" r="13" fill="#ffffff" stroke="{INK2}" stroke-width="1.4"/>')
    svg.text(228, ty + 71, "+", size=18, anchor="middle", weight=600)
    arrow(svg, 242, ty + 65, 282, ty + 65)
    svg.rect(284, ty + 20, 160, 90, "#f4f3ee", INK2, rx=10, sw=1.4)
    svg.text(364, ty + 60, "Transformer", size=15, anchor="middle", weight=650)
    svg.text(364, ty + 80, "(unchanged)", size=12, anchor="middle", fill=INK3)
    arrow(svg, 446, ty + 65, 486, ty + 65)
    # chained output head
    hx, hw = 490, 112
    chain = [("pre", "prefix", "p(pre | h)"), ("core", "core", "p(core | h, pre)"),
             ("var", "variation", "p(var | h, …)"), ("suf", "suffix", "p(suf | h, …)")]
    for i, (k, lab, eq) in enumerate(chain):
        stroke, fill = HUE[k]
        bx = hx + i * (hw + 12)
        svg.rect(bx, ty + 38, hw, 54, fill, stroke, rx=8, sw=1.2)
        svg.text(bx + hw / 2, ty + 60, lab, size=13, anchor="middle", weight=650)
        svg.text(bx + hw / 2, ty + 80, eq, size=10.5, anchor="middle", fill=INK2, family=MONO)
        if i:
            arrow(svg, bx - 12, ty + 65, bx - 1, ty + 65)
    svg.text(hx, ty + 22, "chained output head: each part conditioned on the ones before", size=12, fill=INK3)
    svg.text(hx, ty + 118, "Output tables are tied to the input tables; the product is a proper", size=12, fill=INK3)
    svg.text(hx, ty + 134, "distribution over tuples, so bits-per-byte is directly comparable.", size=12, fill=INK3)
    svg.save("how_it_works.svg", "How Combinatorial BPE works",
             "Words are split into variation, prefix, core and suffix; inventories are learned under a shared "
             "budget; the model sums four embeddings and predicts the parts with a chained head.")


# ------------------------------------------------------------ figure 3
def fig_budget():
    comp = json.load(open(os.path.join(ROOT, "results", "compression.json"), encoding="utf-8"))
    r = next(x for x in comp if x["corpus"] == "wiki_en" and x["vocab"] == 16384)
    red = r["bpe_cl100k"]["redundant_frac"]
    sz = r["comb"]["sizes"]
    W, H = 1000, 330
    svg = SVG(W, H)
    svg.text(32, 44, "Where 16,384 embedding rows go (English Wikipedia)", size=20, weight=650)
    bx, bw = 250, W - 250 - 40

    def bar(y, segs, label, sub):
        svg.text(32, y + 20, label, size=15, weight=600)
        svg.text(32, y + 40, sub, size=12, fill=INK3)
        x = bx
        for frac, kind, txt in segs:
            stroke, fill = HUE[kind]
            w = max(frac * bw, 3)
            svg.rect(x, y, w, 44, fill, stroke, rx=4, sw=1)
            if txt and w > 60:
                svg.text(x + w / 2, y + 27, txt, size=13, anchor="middle", weight=600)
            x += w + 2

    bar(80, [(1 - red, "std", f"distinct words / pieces  {100 * (1 - red):.0f}%"),
             (red, "dup", f"variants  {100 * red:.0f}%")],
        "Standard BPE", "cl100k-style pretokenizer")
    svg.text(bx + bw, 146, "e.g.  ' the'  ' The'  'The'  ' world'  ' World'  'World'  …", size=12,
             fill=HUE["dup"][0], family=MONO, anchor="end", extra='xml:space="preserve"')
    small = (sz["variation"] + sz["prefix"] + sz["suffix"]) / 16384
    bar(180, [(sz["core"] / 16384, "core", f"distinct cores  {sz['core']:,} rows  ({100 * (1 - small):.1f}%)"),
              (small, "pre", "")],
        "Combinatorial BPE", "same budget")
    svg.text(bx + bw - 4, 246, f"variation + prefix + suffix: {sz['variation']} + {sz['prefix']} + {sz['suffix']} rows "
             f"({100 * small:.1f}%)", size=12, fill=HUE["pre"][0], anchor="end")
    svg.text(32, 290, f"Result on held-out text: {r['comb']['chars_per_token']:.2f} vs "
             f"{max(r['bpe_gpt2']['chars_per_token'], r['bpe_cl100k']['chars_per_token']):.2f} characters per token "
             f"({100 * (1 - r['comb']['tokens'] / min(r['bpe_gpt2']['tokens'], r['bpe_cl100k']['tokens'])):.0f}% "
             "fewer tokens). No standard vocab reaches it: even 262k rows tops out at 4.86.", size=13, fill=INK2)
    svg.save("vocab_budget.svg", "Vocabulary budget: standard vs combinatorial",
             f"About {100 * red:.0f}% of a standard 16k vocabulary are case/space/punctuation variants; the "
             f"combinatorial vocabulary spends {100 * (1 - small):.1f}% of rows on distinct cores.")


# ------------------------------------------------------------ figure 4
def fig_equal_tokens():
    import math
    comp = json.load(open(os.path.join(ROOT, "results", "iso_compression.json")))
    runs = {r["tokenizer"]: r for r in json.load(open(os.path.join(ROOT, "results", "lm_iso.json")))}

    def run(name):
        r = runs[f"wiki_en_{name}.json"]
        return r["curve"][-1]["val_bpb"], r["params"]

    def pt(lst, v):
        return next(p for p in lst if p["vocab"] == v)

    STD_C, COMB_C, MATCH_C = "#6b6b66", HUE["core"][0], HUE["pre"][0]
    W, H = 1000, 540
    svg = SVG(W, H)
    svg.text(32, 44, "Match the token count, not the vocabulary size", size=20, weight=650)
    svg.text(32, 69, "English Wikipedia. When both tokenizers produce the same number of tokens for the same text, "
             "Combinatorial BPE", size=14, fill=INK2)
    svg.text(32, 88, "needs 4–14× fewer embedding rows and stays within 0.5–2.6% bits-per-byte of standard BPE.",
             size=14, fill=INK2)

    # ---------------- panel A: compression vs vocab size
    x0, x1, y0, y1 = 88, 452, 150, 430
    lx0, lx1, v0, v1 = 12, 18, 3.0, 6.0

    def X(v):
        return x0 + (math.log2(v) - lx0) / (lx1 - lx0) * (x1 - x0)

    def Y(c):
        return y1 - (c - v0) / (v1 - v0) * (y1 - y0)

    svg.text(32, 128, "A   Compression vs vocabulary size", size=15, weight=600)
    for c in [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]:
        svg.line(x0, Y(c), x1, Y(c), stroke="#ecebe6", sw=1)
        svg.text(x0 - 8, Y(c) + 4, f"{c:.1f}", size=11, fill=INK3, anchor="end")
    for e in range(12, 19):
        svg.line(X(2 ** e), y1, X(2 ** e), y1 + 4, stroke=INK3, sw=1)
        svg.text(X(2 ** e), y1 + 18, f"{2 ** e // 1024}k", size=11, fill=INK3, anchor="middle")
    svg.line(x0, y1, x1, y1, stroke=INK3, sw=1)
    svg.text(x0 + (x1 - x0) / 2, y1 + 38, "embedding rows (log scale)", size=12, fill=INK2, anchor="middle")
    svg.add(f'<text transform="translate({x0 - 42},{(y0 + y1) / 2}) rotate(-90)" font-size="12" fill="{INK2}" '
            f'text-anchor="middle" font-family="{SANS}">characters per token (higher = fewer tokens)</text>')

    def series(pts, color):
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(p['rows']):.1f},{Y(p['chars_per_token']):.1f}"
                     for i, p in enumerate(pts))
        svg.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"/>')
        for p in pts:
            svg.add(f'<circle cx="{X(p["rows"]):.1f}" cy="{Y(p["chars_per_token"]):.1f}" r="4" fill="{color}" '
                    f'stroke="#ffffff" stroke-width="2"/>')

    std = comp["standard_gpt2"]
    comb = [p for p in comp["combinatorial"] if p["vocab"] != 10000]
    series(std, STD_C)
    series(comb, COMB_C)
    top = max(std, key=lambda p: p["vocab"])
    svg.text(X(top["rows"]), Y(top["chars_per_token"]) + 24, "Standard BPE", size=13, fill=STD_C, anchor="end",
             weight=650)
    svg.text(X(top["rows"]), Y(top["chars_per_token"]) + 40, f"levels off at {top['chars_per_token']:.2f}",
             size=11, fill=INK3, anchor="end")
    last = max(comb, key=lambda p: p["vocab"])
    svg.text(X(last["rows"]) - 10, Y(last["chars_per_token"]) - 12, "Combinatorial BPE", size=13, fill=COMB_C,
             anchor="end", weight=650)

    for cv, sv in ((9300, 131072), (4096, 16384)):
        a, b = pt(comb, cv), pt(std, sv)
        y = Y((a["chars_per_token"] + b["chars_per_token"]) / 2)
        svg.line(X(a["rows"]) + 9, y, X(b["rows"]) - 9, y, stroke=MATCH_C, sw=1.6, extra='stroke-dasharray="5 4"')
        for q in (a, b):
            svg.add(f'<circle cx="{X(q["rows"]):.1f}" cy="{Y(q["chars_per_token"]):.1f}" r="8" fill="none" '
                    f'stroke="{MATCH_C}" stroke-width="1.6"/>')
        svg.text((X(a["rows"]) + X(b["rows"])) / 2, y - 8, f"{b['rows'] / a['rows']:.0f}× fewer rows", size=12,
                 fill=MATCH_C, anchor="middle", weight=650)

    # ---------------- panel B: matched pairs
    bx = 520
    svg.text(bx, 128, "B   Language model at the same token count", size=15, weight=600)
    bar0, barw = 745, 215
    s0, s1 = 1.50, 1.56

    def SX(v):
        return bar0 + (v - s0) / (s1 - s0) * barw

    y = 170
    for std_name, comb_name in (("131072_bpe_gpt2", "9300_comb"), ("16384_bpe_gpt2", "4096_comb")):
        sb, sp = run(std_name)
        cb, cp = run(comb_name)
        srows, crows = int(std_name.split("_")[0]), int(comb_name.split("_")[0])
        cpt = pt(comb, crows)["chars_per_token"]
        svg.rect(bx - 8, y - 24, W - 24 - bx, 160, "#fafaf7", CARD_LINE, rx=10, sw=1)
        svg.text(bx + 4, y, f"Same token count: ≈ {cpt:.1f} characters per token", size=13, fill=MATCH_C, weight=650)
        for i, (name, rows, params, color) in enumerate((("Standard BPE", srows, sp, STD_C),
                                                         ("Combinatorial", crows, cp, COMB_C))):
            ry = y + 28 + i * 36
            svg.text(bx + 4, ry, name, size=13, weight=650, fill=color)
            svg.text(bx + 4, ry + 15, f"{rows:,} rows · {params / 1e6:.1f}M params", size=11, fill=INK3)
            w = max(3, rows / 131072 * barw)
            svg.rect(bar0, ry - 11, w, 18, color, rx=3)
            if i == 1:
                svg.text(bar0 + w + 8, ry + 3, f"{srows / crows:.0f}× fewer rows", size=12, fill=COMB_C,
                         weight=650)
            elif w < barw - 60:
                svg.text(bar0 + w + 8, ry + 3, "embedding rows", size=11, fill=INK3)
        # bits-per-byte on one shared scale for both pairs
        ly = y + 116
        svg.text(bx + 4, ly + 2, "bits per byte", size=12, fill=INK2, weight=600)
        svg.text(bx + 4, ly + 17, f"lower is better · {100 * (cb / sb - 1):+.1f}%", size=11, fill=INK3)
        svg.line(bar0, ly, bar0 + barw, ly, stroke=CARD_LINE, sw=2)
        for t in (1.50, 1.52, 1.54, 1.56):
            svg.line(SX(t), ly - 3, SX(t), ly + 3, stroke=INK3, sw=1)
            svg.text(SX(t), ly + 17, f"{t:.2f}", size=10, fill=INK3, anchor="middle")
        for v, color in ((sb, STD_C), (cb, COMB_C)):
            svg.add(f'<circle cx="{SX(v):.1f}" cy="{ly}" r="5.5" fill="{color}" stroke="#fafaf7" stroke-width="2"/>')
        (lv, _), (rv, _) = sorted(((sb, STD_C), (cb, COMB_C)))
        svg.text(SX(lv) - 8, ly - 9, f"{lv:.3f}", size=11, fill=INK2, anchor="end", weight=600)
        svg.text(SX(rv) + 8, ly - 9, f"{rv:.3f}", size=11, fill=INK2, weight=600)
        y += 182

    svg.text(32, H - 20, "Same 8-layer GPT for every run, 2,500 steps × 16k tokens. Token counts matched on held-out "
             "text (4.785 vs 4.785 and 4.061 vs 4.075 chars/token).", size=11, fill=INK3)
    svg.save("equal_tokens.svg", "Equal token count: far fewer embedding rows, competitive bits-per-byte",
             "Standard BPE needs 131,072 rows to match the compression of a 9,300-row combinatorial vocabulary "
             "and levels off near 4.86 characters per token; at matched token counts the language models are "
             "within 0.5-2.6% bits-per-byte.")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_tokenization()
    fig_factorization()
    fig_budget()
    fig_equal_tokens()
