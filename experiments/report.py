"""Turn results/*.json into results/REPORT.md + plots."""
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
COLORS = {"bpe_gpt2": "#2a78d6", "bpe_cl100k": "#eb6834", "comb": "#1baf7a"}
LABELS = {"bpe_gpt2": "BPE (GPT-2 regex)", "bpe_cl100k": "BPE (cl100k regex)", "comb": "Combinatorial BPE",
          "hf_bytelevel": "HF tokenizers ByteLevel", "comb_case_only": "Comb: no affixes",
          "comb_affix_only": "Comb: no case"}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e5e5e0", lw=0.8)
    ax.set_axisbelow(True)


def compression(lines):
    rows = json.load(open(os.path.join(RES, "compression.json"), encoding="utf-8"))
    corpora = list(dict.fromkeys(r["corpus"] for r in rows))
    lines += ["## 1. Compression (characters per token on held-out text, higher is better)", "",
              "Equal embedding budget: combinatorial `|var|+|prefix|+|core|+|suffix|` = baseline vocab size.", ""]
    keys = ["bpe_gpt2", "bpe_cl100k", "hf_bytelevel", "comb", "comb_affix_only", "comb_case_only"]
    lines.append("| corpus | vocab | " + " | ".join(LABELS[k] for k in keys) + " | gain vs best BPE |")
    lines.append("|---|---:|" + "---:|" * len(keys) + "---:|")
    for r in rows:
        best = max(r["bpe_gpt2"]["chars_per_token"], r["bpe_cl100k"]["chars_per_token"])
        cells = [f"{r[k]['chars_per_token']:.3f}" for k in keys]
        cells[keys.index("comb")] = f"**{cells[keys.index('comb')]}**"
        lines.append(f"| {r['corpus']} | {r['vocab']} | " + " | ".join(cells)
                     + f" | **{100 * (r['comb']['chars_per_token'] / best - 1):+.1f}%** |")
    lines += ["", "Token-count reduction (tokens needed relative to best standard BPE):", "",
              "| corpus | " + " | ".join(str(v) for v in sorted({r['vocab'] for r in rows})) + " |",
              "|---|" + "---:|" * len({r['vocab'] for r in rows})]
    for c in corpora:
        rr = [r for r in rows if r["corpus"] == c]
        lines.append(f"| {c} | " + " | ".join(
            f"{100 * (r['comb']['tokens'] / min(r['bpe_gpt2']['tokens'], r['bpe_cl100k']['tokens']) - 1):+.1f}%"
            for r in rr) + " |")

    lines += ["", "### Learned budget split and factor usage (combinatorial)", "",
              "| corpus | vocab | #prefix | #suffix | #core | tokens w/ prefix | w/ suffix | w/ case | "
              "distinct tuples used | baseline vocab that is a variant of another entry |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        c = r["comb"]
        s = c["sizes"]
        lines.append(f"| {r['corpus']} | {r['vocab']} | {s['prefix']} | {s['suffix']} | {s['core']} | "
                     f"{100 * c['has_prefix']:.0f}% | {100 * c['has_suffix']:.0f}% | {100 * c['has_case']:.0f}% | "
                     f"{c['distinct_tuples_used']} | {100 * r['bpe_cl100k']['redundant_frac']:.1f}% |")

    # plot: small multiples
    n = len(corpora)
    cols = 4
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(14, 3.4 * ((n + cols - 1) // cols)),
                             sharex=True)
    for ax, c in zip(axes.flat, corpora):
        rr = [r for r in rows if r["corpus"] == c]
        for k in ("bpe_gpt2", "bpe_cl100k", "comb"):
            ax.plot([r["vocab"] for r in rr], [r[k]["chars_per_token"] for r in rr], lw=2, color=COLORS[k],
                    marker="o", ms=4, label=LABELS[k])
        ax.set_xscale("log", base=2)
        ax.set_title(c, fontsize=11, loc="left")
        style(ax)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("embedding budget (vocab size)")
    for ax in axes[:, 0]:
        ax.set_ylabel("chars / token")
    axes.flat[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Held-out compression at equal embedding budget", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "compression.png"), dpi=130)
    lines += ["", "![compression](compression.png)", ""]


def han(lines):
    path = os.path.join(RES, "han.json")
    if not os.path.exists(path):
        return
    rows = json.load(open(path, encoding="utf-8"))
    lines += ["### Chinese: Traditional/Simplified as a hand-coded variation", "",
              "wiki_zh mixes both scripts (~45% of convertible characters are Traditional). `comb + Trad` stores "
              "cores in Simplified and adds a 4th variation, *Traditional* (OpenCC tables, exact round-trip only).", "",
              "| vocab | BPE (GPT-2) | BPE (cl100k) | Comb | Comb + Trad | gain of Trad | tokens using Trad | "
              "vocab entries that are a script twin: cl100k / comb / comb + Trad |",
              "|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        c, h = r["comb"]["chars_per_token"], r["comb_han"]["chars_per_token"]
        lines.append(f"| {r['vocab']} | {r['bpe_gpt2']['chars_per_token']:.3f} | {r['bpe_cl100k']['chars_per_token']:.3f} | "
                     f"{c:.3f} | **{h:.3f}** | {100 * (h / c - 1):+.1f}% | {100 * r['comb_han']['trad_tokens']:.0f}% | "
                     f"{100 * r['bpe_cl100k']['script_dupes']:.1f}% / {100 * r['comb']['script_dupes']:.1f}% / "
                     f"{100 * r['comb_han']['core_script_dupes']:.1f}% |")
    lines.append("")


def code(lines):
    path = os.path.join(RES, "code.json")
    if not os.path.exists(path):
        return
    rows = json.load(open(path, encoding="utf-8"))
    ref_path = os.path.join(RES, "code_tiktoken.json")
    ref = json.load(open(ref_path))["cl100k_base_chars_per_token"] if os.path.exists(ref_path) else {}
    lines += ["### Source code (codeparrot/github-code-clean, train/test split by repository)", "",
              "`Comb + camel` splits identifiers at case changes (`getUserName` -> `get|User|Name`). "
              "*fallback* = share of identifier characters that needed per-character encoding "
              "because a core piece had mixed case.", "",
              "| corpus | vocab | BPE (GPT-2) | BPE (cl100k) | Comb | fallback | **Comb + camel** | gain vs best BPE | "
              "#prefix / #suffix |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        best = max(r["bpe_gpt2"]["chars_per_token"], r["bpe_cl100k"]["chars_per_token"])
        c, cc = r["comb"], r["comb_camel"]
        lines.append(f"| {r['corpus']} | {r['vocab']} | {r['bpe_gpt2']['chars_per_token']:.3f} | "
                     f"{r['bpe_cl100k']['chars_per_token']:.3f} | {c['chars_per_token']:.3f} | "
                     f"{100 * c['fallback_char_share']:.1f}% | **{cc['chars_per_token']:.3f}** | "
                     f"{100 * (cc['chars_per_token'] / best - 1):+.0f}% | "
                     f"{cc['sizes']['prefix']} / {cc['sizes']['suffix']} |")
    if ref:
        lines += ["", "Production reference, GPT-4's `cl100k_base` (100k vocab): " + ", ".join(
            f"{k.replace('code_', '')} {v:.2f}" for k, v in ref.items()) + " chars/token."]
    lines.append("")


SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#8a5cd6", "#6b6b66"]


def run_name(r):
    size = re.search(r"_(\d+)_", r["tokenizer"]).group(1)
    tok = re.sub(r"^(wiki|code)_[a-z]+_\d+_", "", r["tokenizer"].replace(".json", ""))
    if tok.startswith("bpe_"):
        return {"bpe_gpt2": "BPE (GPT-2 regex)", "bpe_cl100k": "BPE (cl100k regex)"}[tok] + f" {int(size) // 1024}k"
    tok = {"comb": "Comb", "comb_punctnext": "Comb, punct->next prefix", "comb_han": "Comb + Traditional", "comb_camel": "Comb + camel"}.get(tok, tok)
    head = {"linear": "linear head", "mlp": "MLP head", "chain": "chain head (core first)",
            "chain_prefix_first": "chain head (prefix first)",
            "chain_prefix_first_10L_eqflops": "chain (prefix first), 10 layers, 1906 steps = equal FLOPs & data"}.get(r.get("tag"), r.get("tag"))
    return f"{tok} {int(size) / 1024:.3g}k, {head}"


def lm(lines, name="lm", lang="English"):
    path = os.path.join(RES, f"{name}.json")
    if not os.path.exists(path):
        return
    runs = json.load(open(path))
    steps = runs[0]["args"]["steps"]
    lines += [f"## Language modelling: {lang} (same 8-layer GPT, {steps} steps x 16k tokens)", "",
              "| run | params | bytes/token | val bits-per-byte (equal compute) | at equal bytes seen* | "
              "bpb parts: case / prefix / core / suffix |",
              "|---|---:|---:|---:|---:|---|"]
    min_bytes = min(r["curve"][-1]["bytes"] for r in runs)

    def at_bytes(r, b):
        c = r["curve"]
        for a, z in zip(c, c[1:]):
            if a["bytes"] <= b <= z["bytes"]:
                t = (b - a["bytes"]) / (z["bytes"] - a["bytes"])
                return a["val_bpb"] + t * (z["val_bpb"] - a["val_bpb"])
        return c[-1]["val_bpb"]

    for r in runs:
        p = r["curve"][-1]["val_bpb_parts"]
        parts = " / ".join(f"{x:.3f}" for x in p) if len(p) == 4 else "-"
        lines.append(f"| {run_name(r)} | {r['params'] / 1e6:.1f}M | {r['bytes_per_token']:.2f} | "
                     f"**{r['curve'][-1]['val_bpb']:.4f}** | {at_bytes(r, min_bytes):.4f} | {parts} |")
    lines += ["", f"\\* learning curve interpolated at {min_bytes / 1e6:.0f}M training bytes "
                  "(what the standard BPE saw), i.e. equal data instead of equal compute. Caveat: the combinatorial runs are mid-schedule (learning rate not yet decayed) at that point, which exaggerates their gap.", ""]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    n_comb = 0
    for r in runs:
        c = r["curve"][1:]
        if "_bpe_" in r["tokenizer"]:  # baselines: neutral inks, dashed
            kw =dict(lw=2.2, ls="--", color="#2b2b28" if "gpt2" in r["tokenizer"] else "#8c8c85")
        else:
            kw = dict(lw=2, color=SERIES[n_comb % len(SERIES)])
            n_comb += 1
        kw["label"] = run_name(r)
        axes[0].plot([x["step"] for x in c], [x["val_bpb"] for x in c], **kw)
        axes[1].plot([x["bytes"] / 1e6 for x in c], [x["val_bpb"] for x in c], **kw)
    axes[0].set_xlabel("training step (equal compute)")
    axes[1].set_xlabel("training text seen (MB)")
    lo = min(x["val_bpb"] for r in runs for x in r["curve"][2:])
    for ax in axes:
        ax.set_ylabel("validation bits / byte")
        style(ax)
        ax.set_ylim(lo * 0.99, lo * 1.2)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f"Small GPT on {lang} (lower is better)", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.savefig(os.path.join(RES, f"{name}.png"), dpi=130)
    lines += [f"![{name}]({name}.png)", ""]


if __name__ == "__main__":
    lines = ["# Combinatorial BPE - results", ""]
    compression(lines)
    han(lines)
    code(lines)
    lm(lines)
    lm(lines, "lm_zh", "Chinese")
    lm(lines, "lm_iso", "English (equal token count)")
    lm(lines, "lm_js", "JavaScript")
    lm(lines, "lm_js_long", "JavaScript, 3x longer (7,500 steps)")
    lm(lines, "lm_en_long", "English Wikipedia, 3x longer (7,500 steps)")
    with open(os.path.join(RES, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
