# Combinatorial BPE

Standard BPE vocabularies spend many rows on surface variants of the same word:
`the`, ` the`, `The`, ` The`, `the,`, ` THE` … each gets its own embedding.
About **20–25% of a typical 16k vocabulary** is such variants.

**Combinatorial BPE** factors every token into a 4-tuple

```
token = (variation, prefix, core, suffix)          ' "Hello,'  ->  (Capitalised, ' "', 'hello', ',')
```

The model embeds a token as the sum of four embeddings. The core vocabulary stores each word
once, and a few small tables supply the case, the leading space or punctuation, and the
trailing punctuation.

![Standard BPE vs Combinatorial BPE on the same text](figures/tokenization.svg)

## TL;DR

- **12–28% fewer tokens** than standard BPE with the *same number of embedding rows*, on 10
  corpora in 8 languages (English, German, French, Russian, Turkish, Hindi, Japanese, Chinese).
- **Standard BPE cannot catch up by growing its vocabulary.** On English it saturates at ~4.86
  chars/token even with 262k entries. Combinatorial BPE reaches 5.20 with 16k rows, and matches
  a 128k standard vocab with 9.3k rows.
- **On source code the gain is much larger: 54–83% more characters per token** than standard
  BPE at equal budget. A 16k-row vocabulary with camelCase splitting needs 32–40% fewer tokens
  than GPT-4's 100k-entry `cl100k_base` on Python, Java, JavaScript, C++ and Go, because indentation (`'\n        '`) and call syntax (`'();'`) fold into the tokens.
- **Everything except the case rules is learned.** Prefixes, suffixes, cores and the budget
  split between them are learned from data. For Japanese and Chinese the learned suffixes turn
  out to be `，` `。` `、` `」`, and the prefixes include `「` `《` and the full-width space.
- **Encoding is lossless** for any input (UTF-8 byte fallback).
- **Language modelling (small scale, ~34M params): with longer training it wins.** At 123M
  training tokens it beats standard BPE on bits-per-byte by **2.4% on English** and **7.8% on
  JavaScript** at equal compute, pulling ahead after ~2,000 steps in both. On English it is ahead
  even at equal data (−1.0%). In short runs (41M tokens) it is still behind: 0.8% (English),
  3.2% (Chinese), 4.5% (JavaScript). At an equal token count it is within 0.5% of a 128k
  standard BPE with 14× fewer embedding rows. See [Results](#results) and
  [Limitations](#limitations--open-questions).

## How it works

![How Combinatorial BPE works](figures/how_it_works.svg)

| factor | how it is obtained |
|---|---|
| **variation** | Hand-coded from Unicode case mappings: `as-is`, `Capitalised`, `UPPER`. Optionally `Traditional` for Chinese. |
| **prefix** | Learned: strings of non-letters that precede a word (`' '`, `'\n\n'`, `' ('`, `' "'`, `'「'`). |
| **core** | Learned: BPE over case-folded word text (`hello`, `apollo`, `1969`, `台湾`). |
| **suffix** | Learned: strings of non-letter, non-space characters after a word (`,` `.` `),` `".` `。`). |

- **Multi-piece words.** A word that needs several core pieces becomes several tuples. The prefix
  rides on the first tuple and the suffix on the last.
- **Lossless case folding.** Case is folded only when it round-trips exactly (e.g. `İ` and `ǅ`
  are left as-is). Mixed-case pieces such as `iPhone` fall back to per-character variations.
- **Equal-budget accounting.** A tokenizer with budget *N* uses
  `|variation| + |prefix| + |core| + |suffix| = N` embedding rows: exactly as many as a standard
  vocabulary of size *N*. Every comparison below uses this accounting.
- **The budget split is learned.** Each affix saves about `freq(affix)` tokens and each BPE merge
  saves about `count(merge)`. Training keeps the best *N* of both. BPE merge lists can be cut at
  any point, so the rest is well defined. The affixes end up tiny: 1–2% of the budget.
- **camelCase splitting for code** (`split_camel=True`). Identifiers are split at case changes
  (`getUserName` -> `get|User|Name`, `HTTPServer` -> `HTTP|Server`) before core BPE, so every
  piece has a clean case pattern. Without it, merges straddle case boundaries and up to 18% of
  identifier characters fall back to per-character encoding.
- **Traditional/Simplified Chinese** (`fold_han=True`). Cores are stored in Simplified, and a
  Traditional character is folded only when OpenCC's default mapping gives it back exactly.
  臺灣 and 台湾 then share the same cores and differ only in the variation.

![Where the embedding rows go](figures/vocab_budget.svg)

## Quickstart

```bash
pip install -e .            # runtime dependency: regex
```

```python
from cbpe import load, CombinatorialBPE, VARIATIONS

tok = load("pretrained/wiki_en_16384_comb.json")
text = 'In 1969, NASA landed on the Moon. "Amazing!" said the President.'
ids = tok.encode(text)                   # list of (variation, prefix, core, suffix) tuples
assert tok.decode(ids) == text
for v, p, c, s in ids:
    print(VARIATIONS[v], repr(tok.prefixes[p]), tok.core.vocab[c], repr(tok.suffixes[s]))

# train your own
text = open("my_corpus.txt", encoding="utf-8").read()
tok = CombinatorialBPE.train(text, vocab_size=16384)            # fold_han=True for Chinese, split_camel=True for code
tok.save("my_tokenizer.json")
print(tok.sizes)   # {'variation': 3, 'prefix': ..., 'core': ..., 'suffix': ...}
```

Pretrained tokenizers in [`pretrained/`](pretrained):

| file | corpus | budget |
|---|---|---|
| `wiki_en_16384_comb.json` | English Wikipedia | 16k |
| `wiki_en_16384_bpe_gpt2.json` | English Wikipedia, standard BPE baseline | 16k |
| `wiki_zh_16384_comb_han.json` | Chinese Wikipedia, with the Traditional variation | 16k |
| `multi_32768_comb.json` | en + de + fr + ru + tr + hi Wikipedia mix | 32k |

## Results

The full generated tables are in [results/REPORT.md](results/REPORT.md).

### Compression

Held-out chars/token relative to the better of two standard BPE baselines (GPT-2 and cl100k
pretokenizer regexes), at an **equal embedding budget**. The same BPE trainer is used for all.
The GPT-2 baseline matches HuggingFace `tokenizers` to within 0.1–1.4% on the Latin/Cyrillic
corpora.

| corpus | 4k | 8k | 16k | 32k | 64k |
|---|---:|---:|---:|---:|---:|
| TinyStories (en) | +20% | +18% | +18% | +18% | +18% |
| Wikipedia en | +26% | +28% | +28% | +26% | +25% |
| Wikipedia de | +29% | +30% | +31% | +30% | +29% |
| Wikipedia fr | +25% | +26% | +26% | +24% | +23% |
| Wikipedia ru | +30% | +33% | +36% | +38% | +39% |
| Wikipedia tr | +30% | +32% | +33% | +34% | +34% |
| Wikipedia hi (no case) | +20% | +21% | +21% | +21% | +21% |
| Wikipedia ja (no spaces) | +19% | +21% | +24% | +26% | +27% |
| Wikipedia zh (no spaces) | +14% | +17% | +18% | +19% | +20% |
| Wikipedia zh + Traditional variation | +24% | +22% | +24% | +24% | +24% |
| 6-language mix | +24% | +26% | +28% | +29% | +30% |

![Compression across corpora](results/compression.png)

- **Most of the gain comes from affixes.** Removing case folding costs 2–7 points; removing
  affixes breaks the design, because spaces become separate tokens.
- **The gain is largest for inflected, cased languages** (Russian, Turkish).
- **On Chinese, the Traditional variation adds 4–9%.** About 17% of a standard Chinese vocabulary
  is Traditional/Simplified twins (區/区, 臺灣/台湾); the variation removes them by construction.

### Source code

Five languages from `codeparrot/github-code-clean` (~20 MB of training text each, 10 MB for Go),
with train/test split **by repository** and minified files removed. Chars/token on held-out
repositories:

| language | BPE, GPT-2 regex, 16k | BPE, cl100k regex, 16k | GPT-4 `cl100k_base`, 100k | **Comb + camel, 16k** | gain vs best 16k BPE |
|---|---:|---:|---:|---:|---:|
| Python | 3.53 | 3.80 | 4.19 | **6.46** | +70% |
| Java | 3.86 | 4.19 | 4.57 | **6.76** | +61% |
| JavaScript | 3.47 | 3.70 | 4.03 | **6.76** | +83% |
| C++ | 3.27 | 3.52 | 3.86 | **6.21** | +76% |
| Go | 3.01 | 3.43 | 3.59 | **5.87** | +71% |

- **Indentation and syntax fold into the tokens.** Learned prefixes include `'\n        '`,
  `' = '`, `' == '`, `' {\n        '`, `'\n    }\n\n    '` and `'\n * '`; learned suffixes include
  `'();'`, `'());'`, `'("'`, `"'):"` and `'().'`. The learned budget split gives affixes ~12% of
  the rows on code (1,116 prefixes + 900 suffixes for Python at 16k) vs ~1% on prose.
- **camelCase splitting is essential.** Without it, compression *drops* as the vocabulary grows
  (Java: 4.84 at 4k → 4.13 at 64k), because longer merges cross case boundaries and those pieces
  fall back to per-character encoding (18.5% of identifier characters at 64k). With the split the
  fallback is 0% and compression keeps rising (Java 5.80 → 6.98).
- **Language model on JavaScript** (same 8-layer GPT, 2,500 steps × 16k tokens, 300M characters
  of training repositories, validation on unseen repositories):

  | tokenizer (16k) | chars/token | val bpb |
  |---|---:|---:|
  | **BPE, cl100k regex** | 3.74 | **0.919** |
  | BPE, GPT-2 regex | 3.51 | 0.939 |
  | Comb + camel | 6.68 | 0.961 (+4.5%) |

  The gap to the best baseline shrinks steadily during training: +15% at step 1,000, +12% at
  1,500, +6.3% at 2,000, +4.5% at 2,500. The combinatorial model is still improving fastest at
  the end.

  **With 3× longer training it wins.** Same setup, 7,500 steps (123M tokens), 900M characters
  of training repositories:

  | tokenizer (16k) | val bpb, 7,500 steps |
  |---|---:|
  | BPE, cl100k regex | 0.752 |
  | **Comb + camel** | **0.693 (−7.8%)** |

  It pulls ahead at step ~2,000 and the lead grows to the end (−4.3% at 3,000, −7.0% at 5,000,
  −7.8% at 7,500). This is an *equal-compute* result. In the same number of steps the
  combinatorial model reads 822M bytes vs 459M, because each token carries ~1.8× more text. At
  equal *data* (459M bytes) its interpolated curve is ~2.7% behind, but that point is
  mid-schedule for it (learning rate not yet decayed), so a data-matched run is still needed.
  Single seed per run.

  ![JavaScript long run](results/lm_js_long.png) The affixes are the costly part: prefix + suffix cost 0.27 bpb (vs 0.14 on English),
  because prefixes such as `'\n    }\n\n    '` pack closing braces and layout into one
  prediction made by the small output head.

### Language modelling

Setup: the same 8-layer, 512-wide GPT, trained for 2,500 steps of 16k tokens (one pass over
fresh Wikipedia text), scored in validation bits-per-byte (bpb), which is independent of the
tokenizer. The combinatorial model uses a *chained output head*: it predicts the prefix, then
the core, then the variation, then the suffix, each conditioned on the parts before it.

| comparison | standard BPE | Combinatorial BPE |
|---|---:|---:|
| English, equal vocab (16k) | **1.512** | 1.524 (+0.8%) |
| English, equal vocab (16k), **3× longer (7,500 steps)** | 1.336 | **1.303 (−2.4%)** |
| English, equal token count (128k vs 9.3k) | **1.518** (92.6M params) | 1.526 (+0.5%, 32.6M params) |
| English, equal token count (16k vs 4k) | **1.512** | 1.551 (+2.6%) |
| Chinese, equal vocab (16k) | **1.886** | 1.987 (+5.4%) / 1.947 with Traditional (+3.2%) |

![Equal token count: far fewer embedding rows, competitive bits-per-byte](figures/equal_tokens.svg)

The fairest comparison matches **token count** rather than vocab size: then both models read
the same text in the same number of forward passes. At ~4.8 chars/token, standard BPE needs
131k embedding rows (92.6M params) where Combinatorial BPE needs 9.3k (32.6M params), and bpb is
within 0.5%.

![English LM learning curves](results/lm.png)

**Longer training (English).** With 3× the steps (7,500 steps, 123M tokens, 700M characters of
Wikipedia), Combinatorial BPE overtakes the baseline at ~2,000 steps and finishes **2.4% better
(1.303 vs 1.336 bpb)**. It reads 617M vs 469M bytes in those steps; interpolated at the
baseline's 469M bytes it is still 1.0% ahead, so on English the advantage holds at equal data too.

![English long run](results/lm_en_long.png)

What the experiments showed:
- **The output head matters.** Plain linear conditioning was +3.7%; the chained head brought it
  to +0.8%. It adds 2.4M parameters, and no parameter-matched baseline was run.
- **The Traditional variation helps language modelling too** (−2.0% bpb), because each
  Chinese word is learned once instead of twice.
- **Better compression does not buy bpb at this scale for any tokenizer.** Standard BPE also
  gets slightly worse from 16k to 128k (1.512 → 1.518), and cl100k-style BPE compresses better
  than GPT-2 style on Chinese but has worse bpb. Fewer tokens means fewer transformer passes
  per byte, which small models feel.
- **Punctuation is the expensive factor:** the suffix costs 0.09 bpb in English and 0.15 in
  Chinese. Standard BPE gets a full transformer step for every `,` or `。`; here it is decided
  by a small head. Attaching punctuation to the *next* word's prefix instead did not help.
- **Spending the saved compute on depth hurt.** 10 layers with fewer steps, at equal FLOPs and
  equal data, scored 1.623: optimizer steps matter more than size at this scale.

## Limitations & open questions

- **Small scale only.** All LM results use ~30–90M-parameter models and ≤ 123M training
  tokens, one seed per run. Longer training flipped both English (+0.8% → −2.4%) and JavaScript
  (+4.5% → −7.8%) in favour of Combinatorial BPE; Chinese has not been repeated at that length,
  and nothing here shows how it behaves at real model scale.
- **Equal compute vs equal data.** The wins are at equal training compute, where shorter
  sequences let the model read more text in the same number of steps. Interpolated at equal
  data, English is still ahead (−1.0%) but JavaScript is behind (+2.7%); both of those points are
  mid-schedule for the combinatorial model, so data-matched runs are still needed.
- **Punctuation might deserve its own position.** A hybrid (case and space folding, but
  punctuation as separate tokens) or a small local transformer over the four factors
  (MEGABYTE / RQ-Transformer style) are the obvious next experiments.
- **Affixes are restricted to non-letters.** Morphological suffixes (`-ing`, `-s`, `-en`) could
  be learned with the same budget rule by changing the unit regex.
- **CJK "words" are whole clauses** for every pretokenizer tested here, including the baselines.
- **The trainer is pure Python.** It handles ~40 MB corpora in minutes, but is not meant for
  web-scale training.

## Reproducing

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements-dev.txt   # PyTorch installed separately
python scripts/download_data.py --lm                    # Wikipedia en/de/fr/ru/tr/hi + TinyStories (+ LM text)
python scripts/download_data.py --langs ja zh --chars 20e6
python -m pytest                                        # round-trip tests incl. real text in 8 languages
python experiments/bench_compression.py                 # compression: 10 corpora x 5 budgets + ablations
python experiments/bench_han.py                         # Chinese Traditional variation
python scripts/download_data.py --code                  # Python/Java/JS/C++/Go from github-code-clean
python experiments/bench_code.py                        # source code (+ GPT-4 cl100k_base reference if tiktoken installed)
python experiments/bench_iso.py                         # English compression up to 256k vocab (token-count matching)
python experiments/lm.py results/tokenizers/wiki_en_16384_bpe_gpt2.json \
    results/tokenizers/wiki_en_16384_comb.json --head chain --order 1,2,0,3
python experiments/report.py                            # -> results/REPORT.md + plots
python scripts/make_figures.py                          # -> figures/*.svg
```

The compression benchmark takes ~1–2 h on 8 CPU cores, most of it on CJK. Each LM run takes
~5 min on an RTX 3090 (15 min for the 128k-vocab baseline).

## Repository layout

```
cbpe/bpe.py                      char-level BPE trainer/encoder with UTF-8 byte fallback
cbpe/tokenizers.py               StandardBPE (GPT-2 / cl100k regexes) and CombinatorialBPE
cbpe/data/han_st.json            Traditional/Simplified fold tables (built from OpenCC)
pretrained/                      ready-to-use tokenizers
tests/                           round-trip, factorisation and save/load tests
scripts/download_data.py         corpora (Wikipedia via HF parquet range reads, TinyStories)
scripts/build_han_tables.py      OpenCC -> cbpe/data/han_st.json
scripts/make_figures.py          figures/*.svg from the pretrained tokenizers
experiments/bench_compression.py chars/token at equal budget, with ablations
experiments/bench_han.py         Chinese Traditional variation benchmark
experiments/bench_code.py        source-code compression with and without camelCase splitting
experiments/bench_iso.py         English compression vs vocab size up to 256k (token-count matching)
experiments/lm.py                small-GPT bits-per-byte comparison (factorised output heads)
experiments/report.py            results/REPORT.md and plots
results/                         benchmark results (JSON), report and plots
```

## License

MIT, see [LICENSE](LICENSE). The Traditional/Simplified tables in `cbpe/data/han_st.json` are
derived from [OpenCC](https://github.com/BYVoid/OpenCC) (Apache-2.0).
