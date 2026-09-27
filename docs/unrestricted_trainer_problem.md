# Problem: training an *unrestricted* factored (prefix, core, suffix) tokenizer from scratch

## 1. Background

Standard BPE tokenizers (GPT-2, cl100k, o200k, Kimi K3, ...) spend huge amounts of vocabulary
on surface variants of the same word: `the`, ` the`, `The`, ` The`, ` THE`, `the,`, `(the`, ...
each is a separate row in the embedding matrix and a separate class in the output softmax.

**Combinatorial BPE** replaces the flat vocabulary with a factored one. Every token is a 4-tuple

    token = (variation, prefix, core, suffix)

and its text is `prefix + apply(variation, core) + suffix`. The model's input embedding is the sum
of four embeddings (one table per factor) and the output head predicts the factors with a small
chained head. The **row budget** is the total number of rows over all tables:

    rows = |variations| + 256 byte-fallback rows + |cores| + |prefixes| + |suffixes|  <= N

(the empty prefix and the empty suffix count as one row each). A 32k budget of factored rows can
express far more distinct surface tokens than 32k flat rows, because rows combine.

* **Variations** are the only hand-coded part and they apply **only to the core**:
  0 = as-is, 1 = Capitalised (first char upper-cased), 2 = UPPER (all chars upper-cased),
  3 = Traditional Chinese (Simplified chars converted with a fixed OpenCC map). Cores are stored
  folded (lower-case, Simplified). A character is only folded when the fold round-trips exactly,
  so decoding is always lossless. If no variation reproduces a core's original text (for example
  `iPhone` as a single core), that core cannot be used; the text must be split differently.
* **Prefixes, cores and suffixes are learned.** Prefixes and suffixes are stored exactly as
  written, with no folding.
* **Byte fallback:** characters outside the learned alphabet are emitted as UTF-8 byte tokens.
* **Lossless:** `decode(encode(text)) == text` for any text.

## 2. What already works: the *restricted* version

The current implementation restricts what each factor may contain, using a regex:

* The prefix is a run of non-letter/non-digit characters (spaces, newlines, indentation, `(`, `"`, ` = `).
* The core is a run of letters **or** a run of digits.
* The suffix is a run of punctuation (`.`, `,`, `);`, `()`, `，`, `。`).

Affix candidates are just the observed strings in those slots. An affix is kept if its frequency
beats the count of the core BPE merge it would displace. The core is an ordinary BPE trained on
the folded letter/digit runs.

Results on our benchmark (Section 5) are strong: 32% fewer tokens than standard BPE at the same
32k row budget. A 42–45M-parameter GPT trained on the mixed corpus gets 4.1% lower bits-per-byte
at equal compute and is better on all 11 sources.

The restriction is a hand-made prior, though:
* Letters are never affixes, so there is no `-ing`, `-s`, `un-`, or `$`+digits+`B`.
* Digits are BPE'd inside the core.
* Chinese and Japanese, which have no spaces, get little benefit. Their "words" are long runs of
  Han characters that the regex treats as one core.

## 3. The goal: remove the restriction

We want prefixes, cores and suffixes to be **arbitrary strings**, learned from data, with no
character-class rules. The only hand-coded parts remain the four variations (core-only), byte
fallback, and optionally a trivial pretokenisation (split at whitespace, with leading whitespace
attached to the next chunk: regex `\s*\S+|\s+`).

The unrestricted space is a strict **superset** of the restricted one. The restricted tokenizer's
tables are a valid unrestricted solution, so the optimum of the unrestricted problem is at least as
good as the restricted result at the same budget. The problem is purely one of *training*: finding
good tables (and a matching encoder) in a combinatorially large space.

## 4. What we tried and how it failed

**Attempt A: greedy factored BPE** (native Rust). Every word starts as single-character units
`(-, c, -)`, and three kinds of merge are learned greedily by count, like BPE:

    core   (a,b): (p,a,-)(-,b,s) -> (p,ab,s)
    prefix (x,p): (-,x,-)(p,c,s) -> (xp,c,s)
    suffix (s,y): (p,c,s)(-,y,-) -> (p,c,sy)

Merges that re-create an existing string are free. Training stops at the row budget, and encoding
applies the rules in rank order.

**Failure:** an affix merge's count aggregates over *all* cores it attaches to, so early on
(when every core is one character) "attach `e` as a suffix to anything" beats every real core
merge. Merges are irreversible: once an occurrence is `(-, th, e)` the pair `th`+`e` is gone, so
`the` never accumulates a count as a core. Everything freezes into ~3-character triples:

    "The deal was $5B" -> [''|^th|'e'] [' '|d|'eal'] [' '|w|'as'] [' '|$|''] ['5'|^b|'']

(`^` = Capitalised variation.) That gives ~2.8–3.0 chars/token everywhere, far worse than the
restricted version on European languages and code. It is much *better* on Chinese (2.46 vs 1.68)
and Japanese (2.67 vs 2.07), which hints at the headroom.

**Attempt B: core-first.** Same as A, but affix merges are only allowed after 70% of the row
budget has gone to cores. The core and affix merges then compete jointly. This is a crude patch,
but it already **beats the restricted version on every source** (table in Section 5). It ended
with 24,870 cores, 3,835 prefixes and 4,059 suffixes. The factorisation is still poor for an LM,
though:

    "The deal was $5B in 2023, up 12%."  -> [''|^the|''] [''| deal|''] [''| was|''] [''| $|'5'] [''|^b|'']
                                            [''| 202|'3,'] [''| up|''] [' 12'|%|'.']
    "    return self.getUserName(id);"   -> ['    '|return|''] [''| self|''] ['.get'|^user|''] ['Name('|id);|'']
    "Die Regierung hat's beschlossen."   -> [''|^die|''] [' '|^regierung|''] [''| hat|"'s"] [' besch'|loss|'en.']
    "const unhappiness = await ..."      -> [''|const|''] [''| un|'h'] ['app'|iness|''] ...

* **The leading space went back into the core:** ` deal`, ` was`, ` the`. That is exactly the
  duplication (`the` vs ` the`) the method exists to remove. It happens because the core phase
  ran first, as a plain BPE.
* **Some affixes are arbitrary word fragments:** `' besch'`, `'Name('`, `'app'`, `'3,'`. They
  compress but are semantically meaningless.
* **The 70% split is a hand-set knob**, not learned. With 85% cores first, every source is 2–7%
  worse than at 70%, so the best split may be lower still.

**Attempt C: Unigram-style** (like SentencePiece Unigram, adapted to triples):
* Viterbi segmentation over every (prefix, core, suffix, variation) decomposition, with cost
  `-log P(prefix) - log P(core) - log P(suffix) - log P(variation)`.
* EM re-estimation of the factor probabilities.
* Pruning of the rows with the smallest loss (usage × extra cost of the best alternative spelling)
  until the budget is met.

**Issues:**
* It needs a large *pre-made candidate pool*. We seeded it with big BPE vocabularies and the old
  tables; we want to avoid that.
* The loss estimates for affixes are crude.
* The Viterbi likelihood objective produces odd factorings, like core ` ` + suffix `return`, or
  core ` w` + suffix `as`.
* It has only been tested at a small scale (4k rows), not yet at the full 32k.

## 5. Data, benchmark and resources

* **Training text for the tokenizer:** 88M characters, 8M from each of 11 sources:
  * Wikipedia in en, de, fr, ru, ja, zh;
  * GitHub code in Python, JavaScript, Java, C++, Go.
* **After whitespace chunking:** 9.06M chunks, 1.84M unique, about 40M characters in unique chunks.
  Chinese/Japanese chunks are long, up to about 1,200 characters, since there are no spaces.
* **Held-out test sets:** one file per source, about 200k characters each (500k for zh/ja).
* **Metric:** characters per token on each test file, at a row budget of **N = 32,768**. Lossless
  decoding is checked on every file.

| chars/token @ 32k | en | de | fr | ru | ja | zh | python | js | java | cpp | go |
|---|---|---|---|---|---|---|---|---|---|---|---|
| standard BPE (GPT-2 regex) | 3.75 | 3.38 | 3.28 | 3.02 | 1.59 | 1.30 | 3.27 | 2.94 | 3.45 | 2.89 | 2.89 |
| plain BPE on whitespace chunks | 3.82 | 3.41 | 3.44 | 3.09 | 1.74 | 1.39 | 3.99 | 3.43 | 4.18 | 3.53 | 3.81 |
| **restricted Combinatorial BPE (to beat)** | 4.94 | 4.44 | 4.32 | 4.05 | 2.07 | 1.68 | 6.03 | 5.58 | 5.91 | 5.40 | 5.78 |
| unrestricted, greedy (attempt A) | 2.93 | 2.94 | 2.96 | 2.84 | 2.67 | 2.46 | 2.73 | 2.77 | 2.72 | 2.77 | 2.77 |
| unrestricted, core-first 70% (attempt B) | 5.47 | 5.29 | 5.13 | 5.18 | 3.25 | 2.55 | 6.78 | 6.11 | 6.78 | 5.79 | 5.91 |

**Compute:**
* 24-core CPU and 64 GB RAM.
* One consumer NVIDIA GPU, which is also the user's desktop GPU, so be modest with VRAM.
* Implementation will be in Rust, with a Python driver.
* Training should take minutes, ideally under 30 minutes. Encoding must be fast enough for
  billions of characters, so linear or near-linear in text length with bounded match lengths.

## 6. The task

Design a **from-scratch training algorithm**, plus the matching encoder, for the unrestricted
factored tokenizer:

1. **No pre-made candidate tables.**
   * Do not seed from an existing BPE vocabulary or from the restricted tokenizer.
   * Candidates must be generated by the algorithm itself from the training text. Substring
     statistics, suffix arrays, iterative growth and so on are all fine.
2. **No hand-coded character-class rules** for what may be a prefix, core or suffix. The core-only
   variations, byte fallback and optional whitespace chunking are the only fixed structure. If you
   think a different pretokenisation is essential, argue for it.
3. **Objective:**
   * **Primary:** minimise the total number of tokens on held-out text at a fixed row budget N,
     losslessly.
   * **Secondary:** the factorisation should be *useful to a language model*. The core should carry
     the "main" lexical unit, and affixes should be reusable pieces: whitespace and indentation,
     punctuation, operators, inflections like `-s`/`-ing`/`-ed`, number/unit markers, CJK
     particles. The embedding is a *sum* of factor embeddings, and the output head predicts the core
     first and then the affixes conditioned on it. A degenerate "3 random characters per token"
     factoring is bad even when it compresses. If you add a regulariser or prior for this, make it
     explicit and principled rather than a character-class rule.
4. **It must avoid the failure modes above:**
   * **Myopic, irreversible greedy commitments**, where affix merges steal counts from core merges.
   * **Dependence on a pre-supplied candidate pool.**
5. **It should beat attempt B on compression** while giving a *cleaner* factorisation:
   * whitespace in prefixes, not cores (so `the` and ` the` share a core);
   * no arbitrary word fragments as affixes.

   It should also beat the restricted version on every source at N = 32,768. Explain why the
   algorithm can find at least the restricted solution. Remember that solution is feasible in the
   unrestricted space.

**Please provide:**
* **The algorithm, precisely.** Give the objective, candidate generation, the optimisation or
  search procedure, the budget allocation between the three tables, and the stopping rule.
* **The encoder.** Say how a word is segmented: optimal DP, greedy, or rank-based. Explain why it
  is consistent with training and how costs and ties are handled.
* **Pseudocode** detailed enough to implement in Rust.
* **Complexity and memory** estimates for the data sizes above, and which data structures to use
  (tries, suffix automata, hashing, parallelisation).
* **Expected behaviour.** Predict what the learned prefixes, cores and suffixes should look like for
  English, code and Chinese.
* **Risks and ablations** worth running.
