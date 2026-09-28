//! Reversible factored dictionary search for unrestricted Combinatorial BPE.
//! (Implements the design in docs/unrestricted_trainer_problem.md's answer by GPT: optimise the
//! dictionary directly, re-parse from scratch after every change, never commit merges.)
//!
//! A token is (variation, prefix, core, suffix); all three parts are arbitrary strings. Cores are
//! case/Han-folded and carry the variation; prefixes and suffixes match the original text exactly
//! (with case_affixes they are folded too and the variation applies to the whole token).
//!
//! ENCODER: exact minimum-token parse with three states per position,
//!     B[i] between tokens --prefix--> P[j] --core,variation (1 token)--> C[k] --suffix--> B[l]
//! (empty prefix/suffix allowed, core never empty), comparing (tokens, secondary cost)
//! lexicographically. Secondary cost = -log2 q(prefix) - log2 q(core) - log2 q(variation)
//! - log2 q(suffix) (+ the boundary costs with be_permille). Cost is linear in the number of trie
//! matches (no prefix x core x suffix enumeration). Out-of-alphabet chars are byte-fallback tokens (one per UTF-8 byte).
//!
//! TRAINER (one round):
//!   1. proposals from a raw-text substring index (frequent substrings, counted once from the
//!      text, independent of any segmentation). For each substring w that currently needs
//!      T(w) > 1 tokens, every cut w = prefix + core + suffix is a "package" of the rows it still
//!      lacks, worth f(w)(T(w)-1). Pass 1 spreads that value over the cheapest cuts; pass 2
//!      gives it to the cut whose missing rows are in most demand overall, so rows shared by
//!      many packages (a space prefix, a core used in many contexts) win as a family.
//!   2. add the best rows (reviving pruned ones) until the working dictionary is K over budget
//!   3. re-parse, re-estimate q (half-count smoothing), prune back to the budget by the
//!      whole-token replacement bound: loss(r) = sum over token types t using r of
//!      f(t)(T_{D-r}(text(t)) - 1), dropping lowest first and re-checking each against rows
//!      already dropped, so mutual alternatives never both go
//!   4. accept if (corpus tokens, description length R) beats the incumbent, else revert and
//!      try a larger expansion next round; stop after 2 failures in a row.
//! R = sum_rows L0(row) (affix rows x (1 + gamma)) + sum_rows (n + 1/2)(-log2 q), L0 = 2 log2(len+1)
//!     + sum of -log2 char frequency. Only breaks ties in token count.
//!
//! usage: dict_search train INPUT MODEL
//!        dict_search encode MODEL WORDS OUTPUT
//! Symbol ids: 0..n_symbols = alphabet chars; NONE_BASE + k = out-of-alphabet char of k UTF-8 bytes.
//! Every char (alphabet or not) can always be written as byte-fallback tokens, one per UTF-8 byte,
//! so pruning a single-character core never makes text unencodable.
//! INPUT (little endian):
//!     u32 n_symbols, 4 x u32[n_symbols]: fold (the folded symbol cores are stored in: lowercase /
//!     Simplified), flag (0 plain, 1 Simplified char the Traditional variation would change,
//!     2 uppercase letter, 3 Traditional char), upper_of / trad_of (the uppercase / Traditional form
//!     of a folded symbol, u32::MAX if none),
//!     u32 nbytes[n_symbols] (UTF-8 length of each alphabet char), u32 alnum[n_symbols] (character
//!     class: 0 other, 1 letter or mark, 2 digit)
//!     u32 budget_rows (text cores + prefixes + suffixes, empty affixes included)
//!     u32 expand_permille, u32 min_freq, u32 max_len, u32 em_iters, u32 max_rounds, u32 threads,
//!     u32 gamma_permille, u32 max_packages, u32 protect_chars (1 = single-char cores are never pruned),
//!     u32 first_expand_permille (round 0 grows to this x budget), u32 prune_step_permille (max share of
//!     rows pruned per pass; losses are re-estimated after every pass), u32 delta_permille (affix cost:
//!     the primary objective is tokens + delta x non-empty affixes, so an affix has to save more than
//!     delta tokens per use; delta = 0 is pure token count), u32 lambda_permille (affix information cost:
//!     every token also pays lambda x (-log2 q(prefix) - log2 q(suffix)), empty affixes included, so rare
//!     affixes must earn their bits; used in parsing and pruning), u32 rerank_mult, u32 rerank_occ
//!     (rounds >= 1: the best rerank_mult x needed proposals are re-scored by their exact gain, enabling
//!     the row and re-parsing a uniform random sample of up to rerank_occ containing segments,
//!     stratified by thread, scaled by inverse inclusion probability; 0 = off; on long segments only
//!     windows around the row's occurrences are re-parsed, with the same result: parse_x_window),
//!     u32 refactor (1 = also propose absorbing affixes into the core, (p,c,s) -> pc / cs / pcs, for every observed token
//!     type, valued by the affix information it saves: count x lambda x (bits(p) - bits(empty)) ...;
//!     these save no tokens, so the substring proposals would rank them too low to be tried),
//!     u32 mu_permille, u32 h0_millibits (affix productivity cost: every use of affix a also pays
//!     mu x max(0, h0 - H(core | a)) where H is the entropy in bits of the cores a occurs with, i.e.
//!     log2 of its effective number of partner cores. Glue and real morphology attach to many cores
//!     (' ', 的, の, -ing, un-: 2^H from ~40 to ~2000) and pay nothing; pieces of names and words
//!     attach to a few (胡|耀|邦, Man|che|ster: 2^H < 3) and pay up to mu x h0 per use, so the
//!     combination is written as separate tokens unless it earns a core of its own.
//!     mu = 0 is exactly the old trainer), u32 mu2_permille, u32 tau_millibits (pairwise
//!     non-compositionality cost: every token also pays mu2 x [max(0, PMI(prefix, core) - tau) +
//!     max(0, PMI(core, suffix) - tau)], PMI in bits from the training parse. Pieces that are each
//!     productive but occur together far more than chance (Man|che|ster, 諸|葛|亮, 20+ bits) are
//!     taxed, so a frequent combination earns a whole core and a rare one is written as separate
//!     tokens; ordinary pairings (' '+the, electron+s: a few bits) are free. Needs the exact DP to
//!     remember the prefix until the core and the core until the suffix (parse_pair). mu2 = 0 is
//!     exactly the old trainer), u32 alnum_only (1 = the productivity cost applies only to affixes
//!     containing a letter or digit: whitespace/punctuation affixes, as in the restricted tokenizer,
//!     stay free, while an affix carrying word material must be highly productive),
//!     u32 lamc_permille (affix information beyond the core: every token also pays
//!     lamc x [-log2 q(prefix | core) - log2 q(suffix | core)], smoothed as
//!     q(a | c) = (n(a, c) + q(a)) / (n(c) + 1). Glue its core predicts (' '+the, electron+s) is
//!     nearly free; affixes carrying content the core does not determine (北|京|是, 'Name('|id)) pay
//!     per bit. This is the quantity that tracked the LM results (the chained head, not the
//!     transformer, has to predict it); a fit to three LM runs gave lamc ~ 0.12 tokens per bit)
//!     train segments: u64 n, u64 n_ids, i64 counts[n], u32 lens[n], u32 ids[n_ids]
//!     dev segments: same layout
//!     u32 classes (1 = restricted-shaped rows: a core is all letters, all digits or all other
//!     characters; an affix has no letters/digits unless its letter/digit part is in the allow-list:
//!     a suffix may be allowed + trailing other chars, a prefix leading other chars + allowed),
//!     then the allow-lists as two tables (prefixes, suffixes)
//!     u32 nu_permille, u32 frag_t_permille (core fragment cost, read after lamc_permille: every use
//!     of a core c also pays nu x [g(L(c)) + g(R(c))], g(r) = max(0, (r - t) / (1 - t)), where L(c) is
//!     the largest share of c's raw-text occurrences preceded by one and the same letter/digit a
//!     (count(a c) / count(c), folded, only when c itself starts with a letter/digit), R(c) the same
//!     on the right. A core that is nearly always glued to the same neighbour (ィギュアスケート選,
//!     after フ and before 手; happines + s) is a fragment of a larger unit: it can only combine with
//!     affixes that end in that neighbour, i.e. it gives up the flexibility the whole unit would
//!     have. The same test applies to the affixes' inner side: a prefix pays nu x g(R(p)), where R(p)
//!     is the largest share of its raw-text occurrences followed by one and the same character, and
//!     a suffix nu x g(L(s)) with the preceding character (raw strings, any character). Glue and
//!     real morphology (' ', 的, ' self.', -ing, 's) meet many different neighbours and pay nothing;
//!     a fragment that only ever continues one way (ssert_called_once_ after 'a', ギュアスケート選手
//!     after 'ィ') pays as it would as a core, so moving a fragment into an affix does not dodge
//!     the cost. u32 glue_h0_millibits (after frag_t_permille): if > 0, an affix instead pays
//!     nu x max(0, h0 - H) / h0, H = entropy in bits of its neighbouring character, i.e. log2 of its
//!     effective number of distinct continuations (prefix) or predecessors (suffix): an affix has to
//!     be productive, meeting about 2^h0 different neighbours, to be free. Fixed statistics of the text, not of the tokenizer's own parse, so they cannot
//!     reinforce themselves. Single-char cores pay nothing. nu = 0 is exactly the old trainer)
//!     u32 price_permille, u32 prod_k, u32 conc_permille (read after glue_h0_millibits; all 0 = off):
//!     ROW PRICES (MDL admission). Every row r has a one-time price in tokens,
//!         price(r) = pi x L0(r) x m(r),  L0 = 2 log2(len+1) + sum of -log2 char frequency (its spelling),
//!     and the objective is corpus cost + sum of live rows' prices. Pruning drops every row whose
//!     replacement loss is below its price, even under budget, so the budget is a ceiling, not a
//!     target: rows that do not pay for themselves leave their slot empty.
//!     m(affix a) = 1 + prod_k / (1 + D(a)), D(a) = its number of distinct free stems in the raw
//!     text: strings c (length >= 2) such that a.c (prefix) / c.a (suffix) is a frequent substring and
//!     c mostly occurs without a (count(c) >= 3 count(a.c)), i.e. the remainder is a unit of its own.
//!     ' ', 的, un-, -ing, 's have hundreds to thousands (price ~ spelling); app- (app|lication,
//!     app|roach: remainders that never occur alone) and ],ss[1],ss[ have a handful and pay up to
//!     1 + prod_k times more (Goldsmith signatures, type-based productivity).
//!     m(core c) = 1 + 2 x glue(c) (the fixed fragment score above: cores glued to one neighbour
//!     pay up to 5x). Single-char cores and empty affixes are free.
//!     u32 core_price_permille (after conc_permille): pi for core rows (pi_core), so affixes can be
//!     priced harder than cores and the freed slots still go to cores.
//!     conc_permille: an affix whose uses in the training parse are >= conc of them with one and the
//!     same core is pruned (it is a piece of that one word, not an affix; Picky-BPE-style subsumption).
//!     All scores are fixed statistics of the raw text except conc, which only removes rows.
//!     u32 min_gain_ppm (after core_price_permille; 0 = off. Early stop, lossy: stop after a round
//!     (>= 1) that improved the best dev chars/token by less than this many parts per million (a
//!     rejected round improves nothing); the saved model is the best so far, as always.
//!     bench_dictsearch.py default 500: 32k hybrid stops after round 6 of 12, dev chars/token -0.07%)
//!     u32 rare_permille, u32 partner_n0 (read after min_gain_ppm; 0 = off):
//!     RARITY COST (per use, fixed from the raw text): every use of an affix a also pays
//!     rare x bits(a), bits(a) = -log2(count(a) / chars of a's script in the training text), the
//!     script being that of a's first letter/digit (all other chars form one "other" script), so
//!     -ing is measured against Latin text, 的 against Han text, ' (' against punctuation/space.
//!     An affix saves at most one token per use, so it only pays off while rare x bits(a) < 1:
//!     frequent glue (' the ', 's', の) stays, rare content (' users') is written as tokens.
//!     PARTNER PRICE (partner_n0 > 0): the affix price multiplier becomes 1 + prod_k / E_w(a) instead
//!     of the raw-text stem count, E_w = 2^H x (share of a's uses on established cores), H = entropy
//!     of a's partner cores in the training parse with each core weighted by
//!     w(c) = uses(c) / (uses(c) + partner_n0). An affix used with one core, or with a long tail of
//!     rare ones (name pieces, chemical fragments), pays up to 1 + prod_k times its spelling and is
//!     merged (swap proposals) or dropped; one used with many common cores pays about its spelling.
//!     Recomputed every prune pass; it only decides which rows are kept, never enters the parse.
//!     u32 sample_t, u32 fine_rounds, u32 coarse_gain_ppm (read after partner_n0; 0 = off.
//!     COARSE-TO-FINE, lossy): the rounds first run on a frequency-weighted sample of the training
//!     segments (sample_corpus: segments with count >= sample_t kept, rarer ones kept with
//!     probability count / sample_t at weight sample_t, so every corpus sum keeps its expectation)
//!     until max_rounds or a stop (2 rejections, or a gain below coarse_gain_ppm, or below
//!     min_gain_ppm if that is 0); then the incumbent is re-estimated on the full corpus and refined
//!     there by up to fine_rounds more rounds (2 rejections or min_gain_ppm stop them; a rejected
//!     fine round is retried, not counted as "no gain"). A coarse phase run to convergence
//!     (coarse_gain_ppm = 0) has settled at the current expansion, so the fine phase starts with a
//!     doubled one, as after a rejection (the first fine round at the old size was mostly rejected
//!     in tests, and the doubled one gained ~0.4% chars/token). Proposals and all raw-text
//!     statistics always use the full text; the parse-bound phases (EM, prune losses, re-scoring,
//!     refactor/swap proposals) scale with the sample. Whitespace chunks (--hybrid): 6.7M unique
//!     segments, t = 16 keeps 0.8M (9% of their chars); 32k vocab 4.5x faster end to end at equal or
//!     better dev chars/token (results/speed_*.log).
//!     u32 sig_k, u32 sig_share_permille, u32 swap_share_permille (read after coarse_gain_ppm;
//!     sig_k = 0 = off; swap_share 0 = the old 500): PARADIGM (SIGNATURE) PRICE for cores, after
//!     Goldsmith's signatures. A core's signature on one side is the set of letter pieces its
//!     affixes carry there (the letter run of a suffix's start / a prefix's end: s, s, and s. all
//!     count as "s"; affixes without letters count as the empty piece), among pieces with at least
//!     sig_share of the core's uses. A core whose signature includes a letter piece belongs to a
//!     paradigm shared by S cores (those with the same signature, among cores with >= 20 uses) and
//!     its price is multiplied by 1 + sig_k / S on that side: year {-, s} and पूर {ा, ी, े} are
//!     shared by thousands of stems and pay about their spelling, while gre {at, en, y} or ris
//!     {e, ing, k} are one-offs and pay up to 1 + sig_k times more, so their words are merged into
//!     whole cores (swap proposals) and the one-off stem is dropped. Recomputed every prune pass
//!     from the parse, but like the partner price it only decides which rows are kept.
//!     swap_share_permille: a merge (core + affix piece) is proposed when it covers at least this
//!     share of the core's uses (or half of the whole affix's) and frees more price than the merged
//!     core costs; lower it with sig_k so that a one-off stem's merges are all proposed together.
//!     u32 mark_rule (read after case_cores; 0 = off): NO TOKEN, CORE OR PREFIX STARTS WITH A
//!     COMBINING MARK. alnum = 3 marks a combining mark (Unicode \p{M}: Devanagari vowel signs and
//!     viramas, combining accents, ...); with the rule on, no prefix or core arc may start at a mark,
//!     so a mark always stays in the token of the char before it: inside a core, or at the start of
//!     a suffix (stem + agreement vowel, पूर:ा, stays possible; ह|ै, ज:ारी's core-less vowel, or a
//!     core ियाँ does not). A mark with nothing to attach to (after a space) is written in bytes.
//!     Cores and prefixes that start with a mark are never proposed. The model file then ends with
//!     u32 MARK_MAGIC and u32 alnum[n_symbols], so encoding follows the same rule.
//!     u32 sig_soft (read after mark_rule; 0 = off): SOFT SIGNATURE PRICE. Instead of the size S of
//!     the core's exact signature, each letter piece p is weighed by N(p), the number of cores
//!     taking it: the core's factor is 1 + sig_k x sum_p share(c, p) / N(p) per side (and an affix
//!     pays by the average of that over its cores). The exact set of a stem with many productive
//!     endings is nearly unique, so the hard price charged Erfolg ~680x its spelling; soft, a stem
//!     pays only for its rare endings.
//!     u32 swap_self_permille (read after sig_soft; 0 = off): SELF-PAYING MERGES. A swap proposal is
//!     also made for core + whole affix when the merged core's saving on those uses (lambda x the
//!     affix's bits + delta, per use) is above swap_self x its price: pruning would keep it, but a
//!     merge that saves no token (Got:t, hi:m) is found by no other proposal.
//!     u32 case_affixes (read after swap_self_permille; 0 = off, exactly the old trainer; needs
//!     case_cores): CASE-PRESERVING AFFIXES, WHOLE-TOKEN VARIATION. Prefix and suffix rows are keyed
//!     by their folded string too and store a canonical spelling like cores (canonical_masks: the
//!     spelling that writes the most raw occurrences), so ' Ver' / ' ver', 'S' / 's' or 'Name' /
//!     'name' are one row.
//!     The variation then applies to the whole token: its text is V(prefix ++ core ++ suffix), the
//!     three canonical spellings concatenated, with 0 as stored, 1 Capitalised = as stored with the
//!     FIRST CHAR THAT HAS AN UPPERCASE FORM made uppercase (' '+un+happy -> ' Unhappy'; for a token
//!     that starts with a cased char this is the old Capitalised; unlike it, a token starting with
//!     an uncased char (_name, 1st, ' x') capitalises its first cased char instead of nothing),
//!     2 UPPER, 3 Traditional (capitals kept), 4 lower, 5 camelCase = as stored with the first
//!     char with an uppercase form of the core and of the suffix each made uppercase, the prefix as
//!     stored (get + name + list -> getNameList), 6 PascalCase = the same for every part, prefix
//!     included (get + name -> GetName, response + writer -> ResponseWriter); a part without a
//!     cased char is unchanged; 7 Title = the token's first char with an uppercase form uppercase
//!     and every other char folded, whatever the canonical spellings (rat + Haus -> Rathaus,
//!     iPhone -> Iphone). A span no single variation writes is not matched (other parses /
//!     bytes). The exact DP carries a variation state through the states after the prefix and
//!     after the core (VS_*: the eight variations, Capitalised and Title each split into "first
//!     cased char still to come" and "done"), so it stays exact.
//!     u32 code_len, u32 code_w_permille, u32 code_beta_permille (read after case_affixes, in this
//!     order; all 0 = off, exactly the old trainer and encoder): UNIGRAM CODE LENGTH (Unigram /
//!     MinGram-style objective). code_len = 1: every token has a code length in bits, factored like
//!     the LM's chained output head (core first, then the rest given the core):
//!         bits(v, p, c, s) = -log2 p(c) - log2 p(s | c) - log2 p(p | c) - log2 p(v | c),
//!     the empty affix being an ordinary outcome. p(c) = (n(c) + 1/2) / (N + n_byte + (|C| + 1) / 2)
//!     (half-count smoothing over the live cores plus one BYTE pseudo-core); p(a), p(v) are the
//!     half-count marginals the trainer already keeps (cost, vcost). The conditionals back off to
//!     them: p(a | c) = (n(c, a) + beta(c) p(a)) / (n(c) + beta(c)), with beta(c) = code_beta
//!     (code_beta_permille / 1000) if > 0, else WITTEN-BELL: beta(c) = the number of distinct
//!     partners c has on that side (prefixes, suffixes or variations; 1 if none; pairs seen once
//!     are counted in n(c) and beta(c) but not kept: they back off like unseen ones, which keeps
//!     the model file small), so a core seen
//!     with one ending is nearly deterministic, one seen with many backs off a lot, and an unseen
//!     core (n(c) = 0: a new row, a candidate) gets exactly the marginals. A byte-fallback char of
//!     b bytes is b tokens of 8 - log2 p(BYTE) bits each (n_byte = byte tokens in the parse) in the
//!     primary cost (w x bits); its secondary cost stays the old 1000 per byte, so an alphabet
//!     char's core still wins every tie against its bytes (with honest bits there, a rarely used
//!     single-char core lost ties to its byte and the byte count fed on itself).
//!     All counts are the hard-EM counts of the current parse's token types, re-estimated wherever
//!     the costs are (update_codelen: every EM step, every prune pass, the coarse-to-fine switch).
//!     With code_len the secondary (tie-break) cost of a token is its code length in bits (instead
//!     of the independent -log2 q of its parts), so equal-count parses are decided by how likely the
//!     core and its affixes given the core are (观光局|首度 over 观光:局:首|度). The primary terms
//!     (tokens, delta, lambda, mu, usec, mu2, lamc) are unchanged. The conditionals need the prefix
//!     until the core and the core until the suffix, so code_len always runs the pairwise DP
//!     (parse_pair). Exact re-scoring then parses every sampled segment once and re-parses it with
//!     a candidate only where the candidate's arcs, replayed against the plain parse's final
//!     entries, would change a score (extra_pair_wins; exact up to better's 1e-9 ties: in tests
//!     no skipped pair differed from its full re-parse). That re-parse resumes the plain parse's
//!     pending state at the candidate's first arc (the DP is the same before it) and stops once
//!     past its last arc the final entries of every position whose arcs reach further equal the
//!     plain parse's (the rest is then the plain DP): the same result as the full re-parse, bit for
//!     bit (exact_gains_pair).
//!     code_w_permille (with code_len): w = code_w / 1000 tokens per bit also enters the PRIMARY
//!     cost: tokens + w x bits (+ the other primary terms). Small w (1-20 permille) keeps token count
//!     first but makes every loss, exact gain and round score weigh bits too (MinGram-like); large w
//!     approaches pure Unigram. Row prices stay in tokens; since leave-one-out losses, exact gains
//!     and the round score are all primary costs, a row that saves no token but many bits per use
//!     (a whole-word core Gott vs Got:t) can now pay its price. Candidate rows not yet in the tables
//!     (exact re-scoring) get the median marginal cost of their table and no pair counts (so their
//!     conditionals are the marginals, as for any unseen row); refactoring and self-paying swap
//!     proposals also count w x the affix bits given the core that absorbing the affix saves.
//!     The model file then ends with the code length statistics (see MODEL), so encoding minimises
//!     exactly the same (tokens + w x bits, bits).
//!     code_w_permille WITHOUT code_len (UNCONDITIONAL BITS; needs mu2 = lamc = 0): w x the token's
//!     unconditional code length, exactly the bits the secondary cost already adds up,
//!         bits(v, p, c, s) = -log2 q(c) - log2 q(v) - log2 q(p) - log2 q(s)
//!     (the half-count marginals cost / vcost, empty affixes included), also enters the PRIMARY
//!     cost: tokens + w x bits (+ the other primary terms), everywhere primary costs are used
//!     (the 3-state DP and its windowed / resumed re-scoring, extra rows in exact re-scoring,
//!     leave-one-out losses, proposal values, refactoring and self-paying swap values (w x the
//!     absorbed affix's bits beyond an empty affix's), round scores). No core-conditioned
//!     statistics: parsing stays on the fast 3-state DP (the arcs carry w x the row's bits, the
//!     core arc also w x the variation's). The cores' distribution then also has the BYTE
//!     pseudo-core, as with code_len (p(c) = (n(c) + 1/2) / (N + n_byte + (|C| + 1) / 2)), and a
//!     byte-fallback char of b bytes pays b x w x (8 - log2 p(BYTE)) in the primary cost (plus
//!     what a token with empty affixes pays, as without code_w); the secondary cost is unchanged
//!     (1000 per byte). The model file then ends with u32 CODE_MAGIC, u32 0, f64 w, f64 bits per
//!     fallback byte, and encoding minimises the same (tokens + w x bits, bits). code_w = 0 is
//!     exactly the old trainer and model file.
//!     u32 prune_reuse, u32 prune_tail_ppm (read after code_beta_permille, in this order; 0 = off,
//!     exactly the old trainer): LOSSY SPEED-UPS OF PRUNING, meant for the full-corpus fine round,
//!     where one parse of the corpus takes minutes and pruning parsed it ~10-16 times.
//!     prune_reuse = 1: a prune pass after the first re-uses the last pass's parse (the tokens of
//!     every segment are kept, ~16 bytes per token plus the types map, i.e. about what one parse
//!     holds while its statistics are built): only the segments whose tokens use a row the last
//!     pass dropped are parsed again (they must change), and the statistics are updated by the
//!     difference (reparse_dirty). Every other segment keeps a parse that was optimal under the last
//!     pass's costs, not necessarily under the costs re-estimated since: that is the approximation.
//!     Costs, prices, losses and drops are then computed as always from those statistics. The EM
//!     step after pruning would re-estimate from that same parse (no row was dropped after it),
//!     i.e. repeat the last pass's re-estimation, so its first iteration is skipped. The round's
//!     final parse, its score, dev chars/token and everything after them are exact parses of the
//!     pruned dictionary. (Fine round at 10M chars per source: the second pass parses ~11% of the
//!     segments again, the sixth ~5%, the tenth < 1%.)
//!     prune_tail_ppm = t > 0: pruning stops once the rows are within budget and a pass would drop
//!     fewer than t ppm of them; those last few rows below their price stay (at 64k, t = 1000
//!     skips passes that drop a few dozen rows each).
//!     At 10M chars per source (64k, the bench_dictsearch best config) prune_reuse = 1 with
//!     prune_tail_ppm = 1000 made the fine round ~2x faster; held-out chars/token over 4 runs was
//!     -0.13% +- 0.12% against the exact trainer's -0.08% +- 0.13% over 4 runs with other thread
//!     counts (float summation order alone), and 77-80% of sampled held-out chunks were segmented
//!     as by the reference model, as for the exact runs (78-83%): within run-to-run noise.
//!     u32 be_permille, u32 be_order (read after prune_tail_ppm, in this order; be_permille = 0 = off,
//!     exactly the old trainer, encoder and model file): BRANCHING-ENTROPY BOUNDARY COST (a tie-break).
//!     Unsupervised word-segmentation evidence (Tanaka-Ishii's branching entropy / accessor variety;
//!     Hu et al. 2025, entropy-driven pre-tokenization): a word boundary lies where the next char is
//!     hard to predict from the chars before and the previous char hard to predict from the chars
//!     after. In text without spaces (Chinese, Japanese: hybrid chunks are whole clauses) equal-count
//!     parses often differ only in where a boundary goes (观光:局:首|度 vs 观光局|首度), which the token
//!     count cannot tell apart. Statistics, fixed from the raw training text (the unique segments
//!     with their counts) before the first round (build_be): for every context of k = 1..n chars
//!     (n = be_order, 0 = 1) inside a segment, H_R(c) = entropy in bits of the char following c (the
//!     segment's end counts as one more outcome) and H_L(c) = entropy of the char preceding c (the
//!     start counts), kept if k = 1 or c was seen >= BE_MIN (20) times (weighted), quantised to
//!     1/16 bit. At position i of a text (between x[i-1] and x[i], 0 < i < len) the right side is
//!     H_R of x[i-n..i] and the left side H_L of x[i..i+n], each with the longest kept context
//!     (shorter near the text's ends; a side with no kept context gets the count-weighted median of
//!     the single chars' entropies), and the score s(i) = H_R + H_L. It is normalised to
//!     b(i) = its percentile among the inner positions of the training text of the same CLASS
//!     (count-weighted, mid-rank), so b is in [0, 1] and uniform over each class: the class of i is
//!     the script group (the input's script table, as for the rarity cost) of x[i-1] and x[i] if they
//!     agree, else one more class for a change of script. (With one global percentile every Han /
//!     kana position scored b > 0.9, since Latin and code text inside a chunk is far more
//!     predictable, and the costs hardly differed inside a clause.) Contexts are raw symbols (not
//!     case/Han-folded); all scripts are treated alike: it only orders parses of equal primary cost.
//!     Order: at 5M chars per source the score predicted SIGHAN (pku / msr) word boundaries with ROC
//!     AUC 0.83 / 0.81 at order 1, 0.74 at order 2 and 0.71 at orders 3 and 4 (a longer context is
//!     kept only where frequent, and the backed-off positions mix entropies of different orders),
//!     hence the default 1.
//!     COST: every boundary a parse places at an inner position i pays be x (1 - b(i)) bits in the
//!     SECONDARY cost only (be = be_permille / 1000 bits): token boundaries AND the factor
//!     boundaries inside a token (prefix|core, core|suffix) alike, i.e. every distinct position that
//!     a non-empty piece (prefix, core, suffix, fallback char) ends at; an empty affix places no
//!     boundary, the text's ends cost nothing. So among parses with the same primary cost, one whose
//!     boundaries sit where the text branches (high entropy) wins. The primary cost (tokens and every
//!     primary term), row prices, losses, gains and round scores are unchanged: only the tie-breaks
//!     of the parse, and through them the parse's statistics (q, pairs, types), change. Note that
//!     every primary term beyond the token count (lambda's affix bits above all: lambda_permille 20
//!     separates nearly every pair of factorings by a few hundredths of a token) leaves it few ties
//!     to break: at 5M chars per source, 64k, lambda 20, even be = 64 bits moved 14 of ~61k token
//!     boundaries of 3000 SIGHAN lines (with lambda zeroed at encoding, 0.5-1.5% of them).
//!     Everywhere the DP runs (the 3-state DP and its windowed / resumed exact re-scoring, the
//!     pairwise DP of mu2 / lamc / code_len and its extra_pair_wins filter and resumed re-scoring,
//!     encoding) the cost of position i is added once to every entry of i when i is processed (all
//!     entries of i were reached by a piece ending at i; the empty closures then carry it on), so the
//!     DPs stay exact and the incremental re-scoring paths bit-identical to full re-parses. Texts
//!     parsed out of context (a token type's text when pruning checks which rows its replacement
//!     uses) get the costs of that text alone (contexts cut at its ends); parses whose primary cost
//!     is all that is read (proposal values, prune losses) skip the costs (parse_prim: the primary
//!     cost of the best parse is the same, up to better's 1e-9 tolerance).
//!     The model file then ends with the statistics (see MODEL), so encoding reproduces the costs.
//!     u32 case_cores (read after swap_share_permille; 0 = off, exactly the old trainer):
//!     CASE-PRESERVING CORES. A core row is still keyed by its folded string (one row per folded
//!     form), but it also stores a CANONICAL SPELLING: its most frequent written form in the raw
//!     text (the frequent-substring index plus single chars, counts summed over spellings that
//!     differ only by Han folding; ties go to fewer / earlier capitals), with Han chars kept
//!     folded. the / year stay lowercase, YouTube / iPhone / toUpperCase keep their capitals, so a
//!     mixed-case word can be one core. Variations are relative to the canonical spelling:
//!     0 as stored, 1 Capitalised (first char upper, the rest as stored), 2 UPPER, 3 Traditional
//!     (the canonical spelling with Han chars Traditional), 4 lower. All are per-symbol maps
//!     (upper_of / trad_of / fold), so a variation is used only where it rewrites the canonical
//!     spelling into exactly the text (CaseAcc); when several do, the parse takes the cheapest
//!     (-log2 q(v)), the lowest id on a tie. Canonical spellings are fixed statistics of the raw
//!     text, not of the tokenizer's own parse.
//!     then (at the very end of the input) u32 has_script, and if 1, u32 script[n_symbols] (script
//!     group id per alphabet char; used by the rarity cost)
//!     u32 has_init, then (if 1) three tables (cores, prefixes, suffixes) to start from instead of the
//!     bare alphabet, e.g. a restricted-shaped model: letter affixes then have to beat existing
//!     whole-word cores instead of arriving first
//! MODEL: u32 n_symbols, the 4 symbol maps, three tables (cores, prefixes, suffixes: u32 rows, then
//!     per row u32 len + u32 symbols[len]), f64 cost (-log2 q) of every row, 4 x f64 variation costs,
//!     f64 delta, then u32 nbytes[n_symbols],
//!     f64 lambda, then (if present) f64 mu and f64 specificity for every prefix and suffix row,
//!     then (if present) f64 mu2, f64 tau, u32 n, n x (u32 prefix, u32 core, f32 excess PMI),
//!     u32 n, n x (u32 core, u32 suffix, f32 excess PMI), then (if present) the lamc statistics,
//!     then (if present) f64 nu and f64 glue score for every core row, then every prefix and
//!     suffix row.
//!     case_cores models start with u32 CASE_MAGIC ("CASE") before n_symbols; the core table then
//!     holds the canonical spellings (the trie key is their folded form) and there are 5
//!     variation costs instead of 4. Everything else is as above.
//!     case_affixes models start with u32 CAFX_MAGIC ("CAFX") instead of CASE_MAGIC: as case_cores,
//!     and the prefix and suffix tables hold canonical spellings too; a token's text is its
//!     variation applied to prefix ++ core ++ suffix (see case_affixes), and there are 8 variation
//!     costs instead of 5.
//!     code_w without code_len: the model ends (after the mark_rule section, if any) with u32
//!     CODE_MAGIC ("CLEN"), u32 0, f64 w, f64 bits per fallback byte, and nothing else.
//!     code_len models end (after the mark_rule section, if any) with u32 CODE_MAGIC ("CLEN"),
//!     u32 code_len, f64 w, f64 bits per fallback byte, then per core row 3 x u16 backoff offsets
//!     log2((n(c) + beta(c)) / beta(c)) (prefix, suffix, variation side), then three tables of the
//!     pairs seen at least CL_MIN_PAIR (2) times in the parse, -log2 p(. | core): (prefix, core),
//!     (core, suffix), (core, variation), each as u32 byte length, then per core row a LEB128
//!     varint count and per pair (by increasing prefix / suffix / variation id) a varint id gap
//!     (the first pair: the id; then id - previous id - 1) and u16 bits. Every u16 is in units of
//!     1 / 1024 bit (CL_Q; training uses exactly these quantised values). Every other pair is
//!     -log2 q(a) (the row cost above, or the variation cost) + the core's offset. (The core row
//!     costs are -log2 p(c).) Older readers ignore the section.
//!     be_permille models end (after every section above) with u32 BE_MAGIC ("BENT"), f64 be (bits
//!     per unit of boundary cost), u32 order, u32 fallback entropy of the right and of the left side
//!     (1/16 bit), u32 n_classes, u32 n_q (511), f32 percentile[n_classes x n_q] (per position class,
//!     of every score H_R + H_L in 1/16 bit), u32 n_g, u8 script group[n_g] (per alphabet char; a
//!     char beyond it, or out of the alphabet, is group 0; class = the group of both chars, or
//!     n_classes - 1 if they differ), u32 n,
//!     u32 byte length, then the n kept contexts by increasing 32-bit key as a LEB128 varint key gap
//!     (the first: the key; then key - previous - 1) and a u8 entropy (1/16 bit). The key of the
//!     context c on side d (0: the chars before a position, 1: after) is the top 32 bits of
//!     seq_hash(0xFFFFFF00 + d, c...). Training uses exactly this table.
//! WORDS: u64 n, u64 n_ids, u32 lens[n], u32 ids[n_ids]
//! OUTPUT: per segment u32 n_tokens, then n x (u32 variation, u32 prefix, u32 core, u32 suffix);
//!     core = u32::MAX: one out-of-alphabet char (byte fallback)
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::io::{BufWriter, Write};
use std::time::Instant;

#[derive(Default)]
struct Fx(u64);

impl Hasher for Fx {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        // 8 bytes per step (string keys: rows, proposals). Only slice/Vec keys hash through here, and
        // no map with such keys is iterated in an order that matters, so this does not change results.
        let mut it = bytes.chunks_exact(8);
        for c in &mut it {
            self.write_u64(u64::from_le_bytes(c.try_into().unwrap()));
        }
        let r = it.remainder();
        if !r.is_empty() {
            let mut b = [0u8; 8];
            b[..r.len()].copy_from_slice(r);
            self.write_u64(u64::from_le_bytes(b));
        }
    }
    fn write_u8(&mut self, i: u8) {
        self.write_u64(i as u64);
    }
    fn write_u32(&mut self, i: u32) {
        self.write_u64(i as u64);
    }
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }
}

type Fast = BuildHasherDefault<Fx>;

/// Phase timings (seconds, summed per name), printed and cleared once per round.
static PROF: std::sync::Mutex<Vec<(&'static str, f64)>> = std::sync::Mutex::new(Vec::new());

fn prof(name: &'static str, t: Instant) {
    prof_add(name, t.elapsed().as_secs_f64());
}

fn prof_add(name: &'static str, dt: f64) {
    let mut p = PROF.lock().unwrap();
    match p.iter_mut().find(|e| e.0 == name) {
        Some(e) => e.1 += dt,
        None => p.push((name, dt)),
    }
}

fn prof_dump() -> String {
    let p = std::mem::take(&mut *PROF.lock().unwrap());
    p.iter().map(|(n, t)| format!("{n} {t:.2}")).collect::<Vec<_>>().join(", ")
}

/// f(state, i) for i in 0..n on `threads` threads, results in index order. Blocks of `grain`
/// indices are handed out from a shared counter, so items of very different cost (rows used by
/// one token type or by thousands) still keep every thread busy. Each result depends only on its
/// index, so the output is the same as a sequential map.
fn par_map<T: Send, S>(n: usize, threads: usize, grain: usize, init: impl Fn() -> S + Sync,
                       f: impl Fn(&mut S, usize) -> T + Sync) -> Vec<T> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let grain = grain.max(1);
    let mut blocks: Vec<(usize, Vec<T>)> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads.max(1))
            .map(|_| {
                let (next, init, f) = (&next, &init, &f);
                s.spawn(move || {
                    let mut st = init();
                    let mut out = Vec::new();
                    loop {
                        let b = next.fetch_add(grain, std::sync::atomic::Ordering::Relaxed);
                        if b >= n {
                            break;
                        }
                        out.push((b, (b..(b + grain).min(n)).map(|i| f(&mut st, i)).collect::<Vec<T>>()));
                    }
                    out
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    blocks.sort_unstable_by_key(|b| b.0);
    let mut out = Vec::with_capacity(n);
    for (_, v) in blocks {
        out.extend(v);
    }
    out
}

type Ban = Option<(usize, u32)>;
const NONE: u32 = u32::MAX;
const NONE_BASE: u32 = 0xFFFF_FFF0;
const INF: (f64, f64) = (f64::INFINITY, f64::INFINITY);

fn is_byte(x: u32) -> bool {
    x >= NONE_BASE
}

// Which variation writes a span of text as its folded core: a per-char mask (Symbols::mask),
// combined left to right. FU: the first char is uppercase, RU: a later char is uppercase, AU: all
// chars are uppercase, HT: a Traditional char occurs, HS: a Simplified char that the Traditional
// variation would change occurs. variation(): 0 as-is, 1 Capitalised, 2 UPPER, 3 Traditional, or
// None if no single variation writes the span (e.g. camelCase).
const FU: u8 = 1;
const RU: u8 = 2;
const AU: u8 = 4;
const HT: u8 = 8;
const HS: u8 = 16;

fn combine(l: u8, r: u8) -> u8 {
    let mut m = (l & (FU | RU)) | ((l | r) & (HT | HS));
    if r & (FU | RU) != 0 {
        m |= RU;
    }
    m | (l & r & AU)
}

fn variation(m: u8) -> Option<usize> {
    let no_u = m & (FU | RU) == 0;
    if no_u && m & HT == 0 {
        Some(0)
    } else if m & FU != 0 && m & (RU | HT) == 0 {
        Some(1)
    } else if m & AU != 0 {
        Some(2)
    } else if no_u && m & HS == 0 {
        Some(3)
    } else {
        None
    }
}

// case_cores: per-char classes relative to the folded char f (Symbols::cf). UP: the char is
// upper_of[f]; TRAD: it is trad_of[f]; HS: it is f and the Traditional variation would change
// it; HASU: f has an uppercase form; BAD: a folded char no per-symbol map gives back (none with
// the fold rules of build_symbols; kept so such a char can never be matched lossily).
const CF_UP: u8 = 1;
const CF_TRAD: u8 = 2;
const CF_HS: u8 = 4;
const CF_HASU: u8 = 8;
const CF_BAD: u8 = 16;
const CASE_MAGIC: u32 = u32::from_le_bytes(*b"CASE");
const CAFX_MAGIC: u32 = u32::from_le_bytes(*b"CAFX");
const NVAR_CASE: usize = 5;
const NVAR_MAX: usize = 8; // case_affixes: + 5 camelCase, 6 PascalCase, 7 Title

// case_affixes: variation states of a token in progress (DP states 1 and 2 keep one score per
// state). VS 0, 2, 3, 4, 5 (camelCase), 6 (PascalCase) = that variation; Capitalised and Title
// are split into "the token's first char with an uppercase form still to come" (VS 1, 7) and
// "done" (VS_CAP_DONE, VS_TITLE_DONE).
// A part (prefix, core, suffix) is summarised by a transition mask (CaseAcc::tm) of the ways it
// can be written: TM_AS (as stored), TM_CAP (its own first cased char made uppercase, the rest as
// stored; as stored if it has none), TM_UP (UPPER), TM_TRAD, TM_LOW (lower), TM_TITLE (its first
// cased char uppercase, every other char folded; folded if it has none), and TM_CAPS = it has a
// char with an uppercase form (so it ends a pending state). What a state needs from a part:
//   state          prefix (from the token start)   core, suffix
//   0              TM_AS                           TM_AS
//   1 Cap pending  TM_CAP (-> 8 if TM_CAPS)        TM_CAP (-> 8 if TM_CAPS)
//   2 3 4          TM_UP / TM_TRAD / TM_LOW        the same
//   5 camel        TM_AS                           TM_CAP
//   6 Pascal       TM_CAP                          TM_CAP
//   7 Title pend.  TM_TITLE (-> 9 if TM_CAPS)      TM_TITLE (-> 9 if TM_CAPS)
//   8 Cap done     -                               TM_AS
//   9 Title done   -                               TM_LOW
// (vs_start: the states after a prefix; vs_allow: the states a core or suffix can follow.)
const NVS: usize = 10;
const VS_CAP_PENDING: usize = 1;
const VS_TITLE_PENDING: usize = 7;
const VS_CAP_DONE: usize = 8;
const VS_TITLE_DONE: usize = 9;
const TM_AS: u8 = 1;
const TM_CAP: u8 = 2;
const TM_LOW: u8 = 16;
const TM_TITLE: u8 = 32;
const TM_ALL: u8 = 63;
const TM_CAPS: u8 = 64;
const TM_EMPTY: u8 = TM_ALL; // an empty affix: written every way, no change

/// the variation of a variation state
#[inline(always)]
fn vs_var(vs: usize) -> usize {
    match vs {
        VS_CAP_DONE => 1,
        VS_TITLE_DONE => 7,
        _ => vs,
    }
}

/// the state after a part with transition mask tm (the part must allow vs)
#[inline(always)]
fn vs_to(vs: usize, tm: u8) -> usize {
    if tm & TM_CAPS == 0 {
        vs
    } else if vs == VS_CAP_PENDING {
        VS_CAP_DONE
    } else if vs == VS_TITLE_PENDING {
        VS_TITLE_DONE
    } else {
        vs
    }
}

/// a set of pending states moved to their done states if the part has a cased char
#[inline(always)]
fn vs_move(m: u16, tm: u8) -> u16 {
    if tm & TM_CAPS == 0 {
        return m;
    }
    let mut m = m;
    if m & 1 << VS_CAP_PENDING != 0 {
        m = (m & !(1 << VS_CAP_PENDING)) | 1 << VS_CAP_DONE;
    }
    if m & 1 << VS_TITLE_PENDING != 0 {
        m = (m & !(1 << VS_TITLE_PENDING)) | 1 << VS_TITLE_DONE;
    }
    m
}

/// the states a core or suffix with transition mask tm can follow (bit mask)
#[inline(always)]
fn vs_allow(tm: u8) -> u16 {
    let has = |b: u8| tm & b != 0;
    let mut m = (tm & 31) as u16; // 0 .. 4
    if has(TM_CAP) {
        m |= 3 << 5; // camel, Pascal
    }
    if has(TM_TITLE) {
        m |= 1 << VS_TITLE_PENDING;
    }
    if has(TM_AS) {
        m |= 1 << VS_CAP_DONE;
    }
    if has(TM_LOW) {
        m |= 1 << VS_TITLE_DONE;
    }
    m
}

/// the states after a prefix with transition mask tm, from the token start (bit mask)
#[inline(always)]
fn vs_start(tm: u8) -> u16 {
    let has = |b: u8| tm & b != 0;
    let mut m = (tm & 31) as u16; // 0 .. 4 (1: pending)
    if has(TM_AS) {
        m |= 1 << 5; // camel: the prefix as stored
    }
    if has(TM_CAP) {
        m |= 1 << 6; // Pascal
    }
    if has(TM_TITLE) {
        m |= 1 << VS_TITLE_PENDING;
    }
    vs_move(m, tm)
}

/// the set of states after a core or suffix with transition mask tm, from the set `from`
#[inline(always)]
fn vs_after(from: u16, tm: u8) -> u16 {
    vs_move(from & vs_allow(tm), tm)
}

/// case_cores: which variations write a span of text from a canonical spelling of the same folded
/// string, accumulated char by char. The canonical spelling is the folded string with the chars
/// in its upper mask uc made uppercase (bit k = char k; chars from 64 on are always folded).
/// Written forms per char (canonical char c, folded f): 0 c, 1 upper(f) for the first char else c,
/// 2 upper(f), 3 c if uppercase else trad(f), 4 f; upper / trad = the per-symbol maps, or the
/// char itself where there is none. So with up = the span's uppercase chars and hasu = its chars
/// with an uppercase form: 0 needs up == uc, 1 up == uc | (hasu & 1), 3 up == uc, 4 up == 0, and
/// 2 up == hasu; 0, 1, 2, 4 no Traditional char, 3 no Simplified char it would change.
#[derive(Clone, Copy, Default)]
struct CaseAcc {
    up: u64,
    hasu: u64,
    trad: bool,
    hs: bool,
    hi_up: bool, // an uppercase char from position 64 on
    hi_v2: bool, // a char from 64 on that UPPER does not write
    hi_h: bool,  // a char with an uppercase form from 64 on
    bad: bool,
}

impl CaseAcc {
    #[inline(always)]
    fn push(&mut self, cf: u8, k: usize) {
        let (u, h) = (cf & CF_UP != 0, cf & CF_HASU != 0);
        if k < 64 {
            self.up |= (u as u64) << k;
            self.hasu |= (h as u64) << k;
        } else {
            self.hi_up |= u;
            self.hi_v2 |= u != h;
            self.hi_h |= h;
        }
        self.trad |= cf & CF_TRAD != 0;
        self.hs |= cf & CF_HS != 0;
        self.bad |= cf & CF_BAD != 0;
    }
    /// no longer span can be written by any variation
    #[inline(always)]
    fn dead(&self) -> bool {
        self.bad || (self.trad && self.hs)
    }
    /// the cheapest variation (lowest id on a tie) writing the span from canonical mask uc
    #[inline(always)]
    fn best(&self, uc: u64, vcost: &[f64; NVAR_MAX]) -> Option<usize> {
        if self.bad {
            return None;
        }
        let plain = !self.trad && !self.hi_up;
        let ok = [
            plain && self.up == uc,
            plain && self.up == uc | (self.hasu & 1),
            !self.trad && !self.hi_v2 && self.up == self.hasu,
            !self.hs && !self.hi_up && self.up == uc,
            plain && self.up == 0,
        ];
        let mut best: Option<usize> = None;
        for v in 0..NVAR_CASE {
            if ok[v] && best.is_none_or(|b| vcost[v] < vcost[b]) {
                best = Some(v);
            }
        }
        best
    }
    /// case_affixes: the transition mask (TM_*, see VS_*) of the span as one part of a token,
    /// written from canonical mask uc. TM_AS, TM_UP, TM_TRAD, TM_LOW: the conditions of best() for
    /// 0, 2, 3, 4; TM_CAP: a part with cased chars needs up == uc | (its first cased char), one
    /// without any is written as stored; TM_TITLE: up == (its first cased char), whatever uc.
    /// (TM_CAP and TM_TITLE are never set for a part whose first cased char lies at position
    /// >= 64: conservative, never lossy.)
    #[inline(always)]
    fn tm(&self, uc: u64) -> u8 {
        if self.bad {
            return 0;
        }
        let plain = !self.trad && !self.hi_up;
        let ok0 = plain && self.up == uc;
        let mut m = 0u8;
        if ok0 {
            m |= TM_AS;
        }
        if self.hasu != 0 || self.hi_h {
            m |= TM_CAPS;
            if self.hasu != 0 && plain && self.up == uc | (self.hasu & self.hasu.wrapping_neg()) {
                m |= TM_CAP;
            }
        } else if ok0 {
            m |= TM_CAP;
        }
        if !self.trad && !self.hi_v2 && self.up == self.hasu {
            m |= 1 << 2;
        }
        if !self.hs && !self.hi_up && self.up == uc {
            m |= 1 << 3;
        }
        if plain && self.up == 0 {
            m |= TM_LOW;
        }
        // Title: the first cased char upper, all other chars folded
        if self.hasu != 0 || self.hi_h {
            if self.hasu != 0 && plain && self.up == self.hasu & self.hasu.wrapping_neg() {
                m |= TM_TITLE;
            }
        } else if plain && self.up == 0 {
            m |= TM_TITLE;
        }
        m
    }
}

struct Symbols {
    fold: Vec<u32>,
    flag: Vec<u32>,
    upper_of: Vec<u32>,
    trad_of: Vec<u32>,
    nbytes: Vec<u32>,
    alnum: Vec<u32>,
    cf: Vec<u8>, // case_cores char classes (CF_*)
    mark_rule: bool, // no token, core or prefix starts at a combining mark (alnum == 3)
}

const MARK_MAGIC: u32 = u32::from_le_bytes(*b"MARK");
const CODE_MAGIC: u32 = u32::from_le_bytes(*b"CLEN");
/// code_len: pairs seen fewer times than this in the parse are not kept (they back off like unseen
/// pairs; the core's count and Witten-Bell partner count still include them). Keeps the model file
/// small: at 64k / 5M chars per source, all pairs were ~2.2M (26 MB).
const CL_MIN_PAIR: f64 = 2.0;
/// code_len: pair bits and backoff offsets are multiples of 1 / CL_Q bit (stored as u16, so below
/// 64 bits), the same in training and in the model file
const CL_Q: f64 = 1024.0;

fn cl_q(bits: f64) -> f32 {
    ((bits * CL_Q).round().clamp(0.0, 65535.0) / CL_Q) as f32
}

/// code_len (see the header): the code length statistics of the current parse, as bits ready for
/// the DP. Pair bits are kept as f32 (as stored in the model file), so training and encoding add
/// exactly the same numbers.
#[derive(Clone, Default)]
struct CodeLen {
    on: bool,   // code_len = 1
    w: f64,     // code_w: primary cost per bit (0 = the bits only break ties)
    beta: f64,  // fixed backoff weight; 0 = Witten-Bell (distinct partners of the core)
    byte: f64,  // bits per byte of a byte-fallback char
    // code_w without code_len (see the header): w x the token's unconditional bits in the primary
    // cost of the 3-state DP (wu = w; 0 with code_len or without code_w), and wv[v] = wu x vcost[v]
    // (refreshed with the arcs)
    wu: f64,
    wv: [f64; NVAR_MAX],
    // -log2 p(prefix | core) by (prefix, core) [0], -log2 p(suffix | core) by (core, suffix) [1],
    // -log2 p(variation | core) by (core, variation) [2]: the observed pairs
    pair: [HashMap<(u32, u32), f32, Fast>; 3],
    bo: Vec<[f32; 3]>,           // per core row: log2((n(c) + beta) / beta) per side (0 if unseen; f32 as stored)
    vb: Vec<[f64; NVAR_MAX]>,    // per core row: -log2 p(v | c) (dense copy of pair[2] + backoff)
    // the same pairs per core, for the DP: adj[k].1[adj[k].0[c]..adj[k].0[c + 1]] = (affix, bits),
    // sorted by affix (k = 0 prefixes, 1 suffixes)
    adj: [(Vec<u32>, Vec<(u32, f32)>); 2],
}

impl CodeLen {
    /// the per-core adjacency lists of pair[0], pair[1] (after the maps or bo change)
    fn index(&mut self) {
        let nc = self.bo.len();
        for k in 0..2 {
            let mut e: Vec<(u32, u32, f32)> =
                self.pair[k].iter().map(|(&(a, b), &v)| if k == 0 { (b, a, v) } else { (a, b, v) }).collect();
            e.sort_unstable_by_key(|x| (x.0, x.1));
            let mut start = vec![0u32; nc + 1];
            for x in &e {
                start[x.0 as usize + 1] += 1;
            }
            for c in 0..nc {
                start[c + 1] += start[c];
            }
            self.adj[k] = (start, e.into_iter().map(|x| (x.1, x.2)).collect());
        }
    }
}

impl Symbols {
    /// mark_rule: may no prefix or core start at x (a combining mark)?
    fn mark_at(&self, x: u32) -> bool {
        self.mark_rule && !is_byte(x) && self.alnum.get(x as usize) == Some(&3)
    }
    fn byte_len(&self, x: u32) -> u32 {
        if is_byte(x) {
            x - NONE_BASE
        } else {
            self.nbytes[x as usize]
        }
    }
    fn mask(&self, x: u32) -> u8 {
        match self.flag[x as usize] {
            1 => HS,
            2 => FU | AU,
            3 => HT,
            _ => 0,
        }
    }
    /// the case_cores char classes (CF_*), from the maps
    fn init_case(&mut self) {
        let n = self.fold.len();
        self.cf = (0..n)
            .map(|c| {
                let f = self.fold[c] as usize;
                let (up, tr) = (self.upper_of[f], self.trad_of[f]);
                let mut m = if up != NONE { CF_HASU } else { 0 };
                if f == c {
                    if tr != NONE {
                        m |= CF_HS;
                    }
                } else if up == c as u32 {
                    m |= CF_UP;
                } else if tr == c as u32 {
                    m |= CF_TRAD;
                } else {
                    m |= CF_BAD;
                }
                m
            })
            .collect();
    }
    fn written<'s>(&'s self, s: &'s [u32], v: u32) -> impl Iterator<Item = u32> + 's {
        s.iter()
            .enumerate()
            .map(move |(i, &c)| {
                let t = match v {
                    1 if i == 0 => self.upper_of[c as usize],
                    2 => self.upper_of[c as usize],
                    3 => self.trad_of[c as usize],
                    _ => NONE,
                };
                if t == NONE {
                    c
                } else {
                    t
                }
            })
    }
}

// ------------------------------------------------------------------ tables and tries

#[derive(Clone)]
struct Table {
    strs: Vec<Vec<u32>>,
    alive: Vec<bool>,
    map: HashMap<Vec<u32>, u32, Fast>,
    cost: Vec<f64>,
    spec: Vec<f64>, // affix specificity in bits (affix tables only; 0 until measured)
    glue: Vec<f64>, // fragment / glue score (nu_permille), fixed from the raw text
    price: Vec<f64>, // one-time row price in tokens (price_permille); 0 = free
    usec: Vec<f64>,  // extra primary cost per use: nu x glue + rare x bits (fixed per row)
    umask: Vec<u64>, // case_cores: upper mask of a core row's canonical spelling (CaseAcc); case_affixes: affix rows too
    scored: usize,   // rows whose glue and price are set (rebuild)
}

impl Table {
    fn new(with_empty: bool) -> Self {
        let mut t = Table { strs: vec![], alive: vec![], map: HashMap::default(), cost: vec![], spec: vec![], glue: vec![], price: vec![], usec: vec![], umask: vec![], scored: 0 };
        if with_empty {
            t.add(&[]);
        }
        t
    }
    /// add (or revive) a row; returns true if it was not alive before
    fn add(&mut self, s: &[u32]) -> bool {
        if let Some(&i) = self.map.get(s) {
            let was = self.alive[i as usize];
            self.alive[i as usize] = true;
            return !was;
        }
        self.map.insert(s.to_vec(), self.strs.len() as u32);
        self.strs.push(s.to_vec());
        self.alive.push(true);
        self.cost.push(20.0);
        self.spec.push(0.0);
        self.glue.push(0.0);
        self.usec.push(0.0);
        self.umask.push(0);
        self.price.push(0.0);
        true
    }
    fn has(&self, s: &[u32]) -> bool {
        self.map.get(s).is_some_and(|&i| self.alive[i as usize])
    }
    fn live(&self) -> usize {
        self.alive.iter().filter(|&&a| a).count()
    }
}

/// Trie with the children of every node stored contiguously and sorted by symbol: a step is a
/// short scan (or a binary search for wide nodes) of one small array instead of a hash lookup.
/// Nodes are numbered depth-first, so the nodes along a path lie close together in memory, and
/// everything a walk reads at a node (its children, its row, the row's arc cost) is in one record.
struct Trie {
    node: Vec<Node>,
    kids: Vec<(u32, u32)>, // (symbol, child)
    root: Vec<u32>,        // root[c] = child of the root for symbol c (NONE if absent)
}

#[derive(Clone, Copy)]
struct Node {
    first: u32,      // children: kids[first..first + len]
    len: u32,
    term: u32,       // row ending here (NONE if none)
    arc: (f64, f64), // that row's arc cost (Dict::refresh_arcs)
}

impl Trie {
    fn empty() -> Trie {
        Trie { node: vec![Node { first: 0, len: 0, term: NONE, arc: INF }], kids: Vec::new(), root: Vec::new() }
    }
    fn build(t: &Table) -> Trie {
        Trie::from_strs((0..t.strs.len()).filter(|&i| t.alive[i]).map(|i| (&t.strs[i][..], i as u32)).collect())
    }
    /// Trie over (string, id) pairs; empty strings are skipped. A string given twice keeps one id.
    fn from_strs(mut strs: Vec<(&[u32], u32)>) -> Trie {
        strs.retain(|s| !s.0.is_empty());
        strs.sort_unstable_by(|a, b| a.0.cmp(b.0));
        // in sorted order every new node is a child of the node at the same depth on the previous
        // string's path, and siblings come in increasing symbol order
        let mut term = vec![NONE];
        let mut edges: Vec<(u32, u32, u32)> = Vec::new(); // (parent, symbol, child)
        let mut path: Vec<u32> = vec![0];
        let mut prev: &[u32] = &[];
        for &(s, id) in &strs {
            let l = prev.iter().zip(s).take_while(|(a, b)| a == b).count();
            path.truncate(l + 1);
            for (d, &c) in s.iter().enumerate().skip(l) {
                let n = term.len() as u32;
                term.push(NONE);
                edges.push((path[d], c, n));
                path.push(n);
            }
            term[path[s.len()] as usize] = id;
            prev = s;
        }
        let mut first = vec![0u32; term.len() + 1];
        for &(p, _, _) in &edges {
            first[p as usize + 1] += 1;
        }
        for i in 0..term.len() {
            first[i + 1] += first[i];
        }
        let mut fill = first.clone();
        let mut kids = vec![(0u32, 0u32); edges.len()];
        let mut root = Vec::new();
        for &(p, c, n) in &edges {
            kids[fill[p as usize] as usize] = (c, n);
            fill[p as usize] += 1;
            if p == 0 {
                if root.len() <= c as usize {
                    root.resize(c as usize + 1, NONE);
                }
                root[c as usize] = n;
            }
        }
        let node = (0..term.len()).map(|i| Node { first: first[i], len: first[i + 1] - first[i], term: term[i], arc: INF }).collect();
        Trie { node, kids, root }
    }
    #[inline]
    fn term(&self, n: u32) -> u32 {
        self.node[n as usize].term
    }
    #[inline]
    fn step(&self, n: u32, c: u32) -> Option<u32> {
        if is_byte(c) {
            return None;
        }
        if n == 0 {
            return match self.root.get(c as usize) {
                Some(&m) if m != NONE => Some(m),
                _ => None,
            };
        }
        let nd = &self.node[n as usize];
        let ks = &self.kids[nd.first as usize..(nd.first + nd.len) as usize];
        if ks.len() <= 8 {
            for &(s, m) in ks {
                if s >= c {
                    return if s == c { Some(m) } else { None };
                }
            }
            None
        } else {
            ks.binary_search_by_key(&c, |e| e.0).ok().map(|i| ks[i].1)
        }
    }
    /// node spelling s (None if s is not a path of the trie)
    fn find(&self, s: &[u32]) -> Option<u32> {
        let mut n = 0u32;
        for &c in s {
            n = self.step(n, c)?;
        }
        Some(n)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Tok {
    v: u32,
    p: u32,
    c: u32,
    s: u32,
}


struct Dict<'a> {
    sym: &'a Symbols,
    tabs: [Table; 3], // cores (folded), prefixes, suffixes
    vcost: [f64; NVAR_MAX], // variation costs (the first 4 unless case_cores; 5, or 7 with case_affixes)
    case: bool,              // case_cores: cores carry canonical spellings (CaseAcc)
    ca: bool,                // case_affixes: affixes too, and the variation applies to the whole token
    canon: &'a HashMap<u64, u64, Fast>, // case_cores: upper mask by folded-string hash (0 if absent)
    delta: f64,
    lam: f64,
    mu: f64,
    h0: f64,
    alnum_only: bool,
    classes: bool,
    allow: [std::collections::HashSet<Vec<u32>, Fast>; 3], // allowed letter/digit parts of prefixes [1], suffixes [2]
    mu2: f64,
    tau: f64,
    pmi: [HashMap<(u32, u32), f32, Fast>; 2], // excess PMI of (prefix, core) and (core, suffix)
    lamc: f64,
    cn: [HashMap<(u32, u32), f32, Fast>; 2], // counts n(prefix, core), n(core, suffix)
    ncore: Vec<f32>,                         // n(core)
    tries: [Trie; 3],
    nu: f64,
    frag: &'a [HashMap<u64, f32, Fast>; 3], // glue scores by string hash: cores (folded), prefixes, suffixes (raw)
    pricing: &'a Pricing,
    cl: CodeLen, // code_len statistics (cl.on = false: off)
    be: &'a BoundEnt, // be_permille: branching-entropy boundary costs (be.w = 0: off)
}

/// 64-bit hash of a symbol string (splitmix64 steps; symbol 0 and leading symbols all count,
/// unlike Fx, which maps a leading 0 to the empty state)
fn seq_hash(s: impl Iterator<Item = u32>) -> u64 {
    s.fold(SEQ_H0, seq_step)
}

const SEQ_H0: u64 = 0x9e37_79b9_7f4a_7c15;

/// one symbol of seq_hash (so a growing string can be hashed incrementally)
#[inline(always)]
fn seq_step(h: u64, c: u32) -> u64 {
    let mut h = (h ^ (c as u64 + 1)).wrapping_add(0x9e37_79b9_7f4a_7c15);
    h = (h ^ (h >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^ (h >> 31)
}

/// Row pricing (price_permille): see the header.
#[derive(Default)]
struct Pricing {
    pi: f64,
    pi_core: f64,
    k: f64,
    conc: f64,
    char_bits: Vec<f64>,
    dstems: [HashMap<u64, u32, Fast>; 2], // distinct free stems of prefixes [0], suffixes [1] by raw-string hash
    rare: f64,
    n0: f64,
    script: Vec<u8>,                // script group per alphabet char
    script_chars: Vec<f64>,         // training chars per script group
    rawcnt: HashMap<u64, u64, Fast>, // raw count of every frequent substring and every char
    min_freq: f64,
    sig_k: f64,
    sig_share: f64,
    sig_soft: bool,
    swap_share: f64,
    swap_self: f64, // > 0: also propose merges whose saved affix bits pay swap_self x their price
}

/// D(a) for every frequent prefix / suffix string: distinct remainders c (len >= 2) with a.c / c.a
/// frequent and count(c) >= 3 count(a.c).
fn affix_stems(x: &[u32], subs: &[(u32, u16, u32)], threads: usize) -> [HashMap<u64, u32, Fast>; 2] {
    let mut cnt: HashMap<u64, u32, Fast> = HashMap::default();
    for &(p, l, c) in subs {
        cnt.insert(seq_hash(x[p as usize..p as usize + l as usize].iter().copied()), c);
    }
    let chunk = subs.len().div_ceil(threads).max(1);
    let parts: Vec<[HashMap<u64, u32, Fast>; 2]> = std::thread::scope(|sc| {
        let hs: Vec<_> = subs
            .chunks(chunk)
            .map(|part| {
                let cnt = &cnt;
                sc.spawn(move || {
                    let mut out: [HashMap<u64, u32, Fast>; 2] = Default::default();
                    for &(p, l, n) in part {
                        let w = &x[p as usize..p as usize + l as usize];
                        let l = w.len();
                        for k in 1..l.saturating_sub(1) {
                            // prefix a = w[..k], stem w[k..]
                            let stem = seq_hash(w[k..].iter().copied());
                            if cnt.get(&stem).is_some_and(|&m| m as u64 >= 3 * n as u64) {
                                *out[0].entry(seq_hash(w[..k].iter().copied())).or_insert(0) += 1;
                            }
                            // suffix a = w[l-k..], stem w[..l-k]
                            let stem = seq_hash(w[..l - k].iter().copied());
                            if cnt.get(&stem).is_some_and(|&m| m as u64 >= 3 * n as u64) {
                                *out[1].entry(seq_hash(w[l - k..].iter().copied())).or_insert(0) += 1;
                            }
                        }
                    }
                    out
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut out: [HashMap<u64, u32, Fast>; 2] = Default::default();
    for part in parts {
        for k in 0..2 {
            for (h, v) in &part[k] {
                *out[k].entry(*h).or_insert(0) += v;
            }
        }
    }
    out
}

/// case_cores: the canonical spelling of every folded frequent substring and single char whose
/// most frequent raw spelling has capitals, as an upper mask (CaseAcc) by folded-string hash.
/// A spelling's count sums the raw strings with that case pattern (Han spellings count together:
/// Han stays folded); ties go to the smaller mask, so all-folded wins a tie.
/// case_affixes (ca): the spelling that WRITES the most occurrences wins instead (coverage), since
/// one variation covers the whole token: from canonical C, a spelling is written if it is C (as
/// stored), all folded (lower), all upper (UPPER), or C with its first cased char upper (at the
/// token start by Capitalised, mid-word after a lowercase letter by camelCase / PascalCase:
/// get+Name); and all folded from a capitalised C when not mid-word (lower folds the whole token:
/// filename = file + name is not written from Name), or mid-word inside a Title-shaped word
/// (Rathaus = Title(rat + Haus): one capital letter, then only lowercase letters, counted from
/// the frequent substrings that start at that capital; this assumes the token starts at or
/// before the capital with no cased char before it, as after a space, and misses such words
/// longer than max_len, so it undercounts). So name stays lowercase when its lowercase uses
/// (filename, username, name) and its capitalised ones (getName, Name) together outnumber the
/// uses only a capitalised canonical writes. (Title's other spellings, iPhone -> Iphone at a token
/// start, are not counted: conservative.)
fn canonical_masks(sym: &Symbols, x: &[u32], wt: &[u32], subs: &[(u32, u16, u32)], ca: bool) -> HashMap<u64, u64, Fast> {
    let up = |c: u32| sym.cf[c as usize] & CF_UP != 0;
    let mask = |w: &[u32]| w.iter().take(64).enumerate().fold(0u64, |m, (k, &c)| m | ((up(c) as u64) << k));
    let fold_hash = |w: &[u32]| seq_hash(w.iter().map(|&c| sym.fold[c as usize]));
    // raw count per (folded string, mask); only folded strings with some capitalised spelling
    let mut uni = vec![0u64; sym.fold.len()];
    for (i, &c) in x.iter().enumerate() {
        if c != NONE && !is_byte(c) {
            uni[c as usize] += wt[i] as u64;
        }
    }
    let mut cased: std::collections::HashSet<u64, Fast> = Default::default();
    for &(p, l, _) in subs {
        let w = &x[p as usize..p as usize + l as usize];
        if w.iter().any(|&c| up(c)) {
            cased.insert(fold_hash(w));
        }
    }
    let mut cnt: HashMap<(u64, u64), u64, Fast> = HashMap::default();
    for (c, &n) in uni.iter().enumerate() {
        if n > 0 {
            let w = [c as u32];
            *cnt.entry((fold_hash(&w), mask(&w))).or_insert(0) += n;
        }
    }
    for &(p, l, n) in subs {
        let w = &x[p as usize..p as usize + l as usize];
        let h = fold_hash(w);
        if cased.contains(&h) {
            *cnt.entry((h, mask(w))).or_insert(0) += n as u64;
        }
    }
    if ca {
        // all-folded occurrences of each spelling right after a letter, from the frequent
        // one-char-longer substrings; and each cased folded string's chars with an uppercase form
        let letter = |c: u32| !is_byte(c) && sym.cf[c as usize] & CF_HASU != 0;
        let mut mid_low: HashMap<u64, u64, Fast> = HashMap::default(); // all-folded spellings after a letter
        let mut mid_title: HashMap<u64, u64, Fast> = HashMap::default(); // those in a Title-shaped word
        let mut hasu: HashMap<u64, u64, Fast> = HashMap::default();
        for &(p, l, n) in subs {
            let w = &x[p as usize..p as usize + l as usize];
            let h = fold_hash(w);
            if cased.contains(&h) {
                hasu.entry(h).or_insert_with(|| {
                    w.iter().take(64).enumerate().fold(0u64, |m, (k, &c)| m | (((sym.cf[c as usize] & CF_HASU != 0) as u64) << k))
                });
            }
            if l >= 2 && letter(w[0]) && letter(w[1]) && w[1..].iter().all(|&c| !up(c)) {
                let ht = fold_hash(&w[1..]);
                if cased.contains(&ht) {
                    *mid_low.entry(ht).or_insert(0) += n as u64;
                }
            }
            // Title: an all-folded tail inside a Title-shaped word (one capital letter, then only
            // lowercase letters: Rat|haus, Kranken|haus) is written by Title from any canonical
            // spelling, so those mid-word occurrences are not lost to a capitalised canonical
            if l >= 2 && letter(w[0]) && up(w[0]) && w[1..].iter().all(|&c| letter(c) && !up(c)) {
                for k in 1..l as usize {
                    let ht = fold_hash(&w[k..]);
                    if cased.contains(&ht) {
                        *mid_title.entry(ht).or_insert(0) += n as u64;
                    }
                }
            }
        }
        let mut spellings: HashMap<u64, Vec<(u64, u64)>, Fast> = HashMap::default();
        for (&(h, m), &n) in &cnt {
            spellings.entry(h).or_default().push((m, n));
        }
        let mut out: HashMap<u64, u64, Fast> = HashMap::default();
        for (h, mut sp) in spellings {
            sp.sort_unstable(); // by mask: a deterministic order
            let hu = hasu.get(&h).copied().unwrap_or_else(|| sp.iter().fold(0, |a, &(m, _)| a | m));
            let first = hu & hu.wrapping_neg(); // the first char with an uppercase form
            let cov = |c: u64| -> u64 {
                sp.iter().map(|&(m, n)| {
                    if m == c || m == hu || (m == 0 && c == 0) || m == c | first {
                        n
                    } else if m == 0 {
                        let title = mid_title.get(&h).copied().unwrap_or(0);
                        n.saturating_sub(mid_low.get(&h).copied().unwrap_or(0).saturating_sub(title))
                    } else {
                        0
                    }
                }).sum()
            };
            // (coverage, raw count, then the smaller mask)
            let best = sp.iter().map(|&(m, n)| (cov(m), n, std::cmp::Reverse(m))).max().unwrap();
            let m = best.2 .0;
            if m != 0 {
                out.insert(h, m);
            }
        }
        return out;
    }
    let mut best: HashMap<u64, (u64, u64), Fast> = HashMap::default();
    for (&(h, m), &n) in &cnt {
        let e = best.entry(h).or_insert((n, m));
        if n > e.0 || (n == e.0 && m < e.1) {
            *e = (n, m);
        }
    }
    best.into_iter().filter(|&(_, (_, m))| m != 0).map(|(h, (_, m))| (h, m)).collect()
}

/// Fragment scores of the (folded) frequent substrings: see nu_permille in the header.
fn fragment_scores(sym: &Symbols, x: &[u32], wt: &[u32], subs: &[(u32, u16, u32)], t: f64, h0: f64, min_freq: u32)
    -> [HashMap<u64, f32, Fast>; 3] {
    let fold = |w: &[u32]| -> Vec<u32> { w.iter().map(|&c| sym.fold[c as usize]).collect() };
    // folded counts (raw spellings that fold together are summed), with one representative spelling
    let mut cnt: HashMap<u64, (u64, u32, u16), Fast> = HashMap::default();
    for &(p, l, c) in subs {
        let w = fold(&x[p as usize..p as usize + l as usize]);
        cnt.entry(seq_hash(w.into_iter())).or_insert((0, p, l)).0 += c as u64;
    }
    let an = |c: u32| sym.alnum[c as usize] != 0;
    let (mut lmax, mut rmax): (HashMap<u64, u64, Fast>, HashMap<u64, u64, Fast>) = Default::default();
    for &(n, p, l) in cnt.values() {
        if l < 3 {
            continue; // the extended string of a core of length >= 2
        }
        let w = fold(&x[p as usize..p as usize + l as usize]);
        let l = w.len();
        if an(w[0]) && an(w[1]) {
            let e = lmax.entry(seq_hash(w[1..].iter().copied())).or_insert(0);
            *e = (*e).max(n);
        }
        if an(w[l - 1]) && an(w[l - 2]) {
            let e = rmax.entry(seq_hash(w[..l - 1].iter().copied())).or_insert(0);
            *e = (*e).max(n);
        }
    }
    let g = |r: f64| ((r - t) / (1.0 - t)).clamp(0.0, 1.0);
    let mut out: HashMap<u64, f32, Fast> = HashMap::default();
    for (h, &(n, _, _)) in &cnt {
        let fl = lmax.get(h).map_or(0.0, |&m| m as f64 / n as f64);
        let fr = rmax.get(h).map_or(0.0, |&m| m as f64 / n as f64);
        let v = g(fl) + g(fr);
        if v > 0.0 {
            out.insert(*h, v as f32);
        }
    }
    drop(cnt);
    // affixes: raw strings (single chars included), any neighbouring character
    let mut raw: HashMap<u64, u64, Fast> = HashMap::default();
    for (i, &c) in x.iter().enumerate() {
        if c != NONE && !is_byte(c) {
            *raw.entry(seq_hash(std::iter::once(c))).or_insert(0) += wt[i] as u64;
        }
    }
    // per string: (largest, total, sum c log2 c) over its frequent one-char extensions
    type Ext = HashMap<u64, (u64, u64, f64), Fast>;
    let (mut next, mut prev): (Ext, Ext) = Default::default();
    for &(p, l, c) in subs {
        let w = &x[p as usize..p as usize + l as usize];
        raw.insert(seq_hash(w.iter().copied()), c as u64);
        let cl = c as f64 * (c as f64).log2();
        for (m, key) in [(&mut next, seq_hash(w[..w.len() - 1].iter().copied())), (&mut prev, seq_hash(w[1..].iter().copied()))] {
            let e = m.entry(key).or_insert((0, 0, 0.0));
            e.0 = e.0.max(c as u64);
            e.1 += c as u64;
            e.2 += cl;
        }
    }
    // h0 > 0: branching entropy H of the neighbouring char (the occurrences not covered by frequent
    // extensions count as spread over items of min_freq / 2 each; string edges are such
    // occurrences too), score (h0 - H) / h0; else the largest share, score g(share)
    let mf = (min_freq as f64 / 2.0).max(1.0);
    let side = |m: &Ext| -> HashMap<u64, f32, Fast> {
        m.iter()
            .filter_map(|(h, &(mx, tot, cl))| {
                let n = *raw.get(h)? as f64;
                let v = if h0 > 0.0 {
                    let rest = (n - tot as f64).max(0.0);
                    let hh = n.log2() - (cl + rest * mf.log2()) / n;
                    ((h0 - hh) / h0).clamp(0.0, 1.0)
                } else {
                    g(mx as f64 / n)
                };
                (v > 0.0).then_some((*h, v as f32))
            })
            .collect()
    };
    let (pre, suf) = (side(&next), side(&prev));
    [out, pre, suf]
}

// ------------------------------------------------------------------ branching-entropy boundaries

/// model file section of the boundary statistics (be_permille)
const BE_MAGIC: u32 = u32::from_le_bytes(*b"BENT");
/// context key tags: + 0 the chars before a position (right-branching), + 1 the chars after it
/// (left-branching)
const BE_TAG: u32 = 0xFFFF_FF00;
/// contexts longer than one char are kept only if seen at least this often (weighted)
const BE_MIN: f64 = 20.0;
/// entropies are kept in units of 1/16 bit (u8, so at most 15.9 bits); a position's score is the
/// sum of its two sides (0..=510)
const BE_Q: f64 = 16.0;
const BE_NS: usize = 511;

/// Branching-entropy boundary statistics (be_permille, see the header): the quantised entropy of
/// every kept context by its 32-bit key, the per-side fallback for unseen contexts, the script
/// group of every alphabet char, and per position class the percentile of every score among the
/// training text's positions of that class.
#[derive(Default)]
struct BoundEnt {
    w: f64, // bits per unit of boundary cost (0 = off)
    order: usize,
    tab: HashMap<u32, u8, Fast>,
    med: [u8; 2],
    grp: Vec<u8>,  // script group per alphabet char (empty: all 0)
    ncls: usize,   // position classes: one per group, and the last for a change of group
    cdf: Vec<f32>, // ncls x BE_NS
}

impl BoundEnt {
    /// 32-bit key of a context on side dir (0: the chars before a position, 1: after)
    #[inline]
    fn key(dir: usize, ctx: &[u32]) -> u32 {
        (Self::hash(dir, ctx) >> 32) as u32
    }
    #[inline]
    fn hash(dir: usize, ctx: &[u32]) -> u64 {
        seq_hash(std::iter::once(BE_TAG + dir as u32).chain(ctx.iter().copied()))
    }
    /// quantised branching entropy on side dir of position i of x: dir 0 the entropy of what
    /// follows the chars x[i - k..i], dir 1 of what precedes x[i..i + k], with the longest kept
    /// context (k <= order, within x); the side's median if none is kept
    fn side(&self, dir: usize, x: &[u32], i: usize) -> u8 {
        let kmax = if dir == 0 { i } else { x.len() - i }.min(self.order);
        for k in (1..=kmax).rev() {
            let ctx = if dir == 0 { &x[i - k..i] } else { &x[i..i + k] };
            if let Some(&v) = self.tab.get(&Self::key(dir, ctx)) {
                return v;
            }
        }
        self.med[dir]
    }
    #[inline]
    fn score(&self, x: &[u32], i: usize) -> usize {
        self.side(0, x, i) as usize + self.side(1, x, i) as usize
    }
    /// script group of a symbol (0 for out-of-alphabet chars and without script information)
    #[inline]
    fn group(&self, c: u32) -> usize {
        self.grp.get(c as usize).copied().unwrap_or(0) as usize
    }
    /// class of inner position i: the script group of x[i - 1] and x[i] if they agree, else the
    /// last class (a change of script)
    #[inline]
    fn class(&self, x: &[u32], i: usize) -> usize {
        let (a, b) = (self.group(x[i - 1]), self.group(x[i]));
        if a == b { a } else { self.ncls - 1 }
    }
    /// b(i): the percentile of i's score among the training positions of its class
    #[inline]
    fn b(&self, x: &[u32], i: usize) -> f64 {
        self.cdf[self.class(x, i) * BE_NS + self.score(x, i)] as f64
    }
    /// the boundary cost of every position 0..=n of x into out: w x (1 - b(i)) inside, 0 at both
    /// ends (every parse has those); out is left empty if off
    fn fill(&self, x: &[u32], out: &mut Vec<f64>) {
        out.clear();
        if self.w == 0.0 {
            return;
        }
        let n = x.len();
        out.resize(n + 1, 0.0);
        for i in 1..n {
            out[i] = self.w * (1.0 - self.b(x, i));
        }
    }
}

/// The boundary statistics of the training text (unique segments with their counts): for every
/// context of 1..=order chars and both sides, the entropy of the char that follows it (side 0) /
/// precedes it (side 1) in the segments, the segment's end / start being one more outcome;
/// contexts of more than one char only if seen >= BE_MIN times. Then per position class (script
/// group, or a change of group; `script`: group per alphabet char, may be empty) the percentile
/// table of the scores of its inner positions (weighted by count, mid-rank).
fn build_be(c: &Corpus, order: usize, w: f64, script: &[u8], threads: usize) -> BoundEnt {
    let t0 = Instant::now();
    let threads = threads.max(1);
    let ncls = script.iter().map(|&g| g as usize + 1).max().unwrap_or(1) + 1;
    let mut be = BoundEnt { w, order, tab: HashMap::default(), med: [0; 2], grp: script.to_vec(), ncls, cdf: Vec::new() };
    // (sharded by context hash: each task counts its share of the contexts in one scan, so at most
    // `threads` of the 4 x threads shards are in memory at a time)
    let shards = 4 * threads;
    let mut n_ctx = [[0usize; 8]; 2];
    for dir in 0..2 {
        let mut med_hist = [0.0f64; 256];
        for k in 1..=order {
            let parts: Vec<Vec<(u64, u8, f64)>> = par_map(shards, threads, 1, || (), |_, sh| {
                let mut cnt: HashMap<(u64, u32), f64, Fast> = HashMap::default();
                for (x, &f) in c.segs.iter().zip(&c.freqs) {
                    let n = x.len();
                    if n < k {
                        continue;
                    }
                    for i in if dir == 0 { k..n + 1 } else { 0..n - k + 1 } {
                        let (ctx, out) = if dir == 0 {
                            (&x[i - k..i], x.get(i).copied().unwrap_or(NONE))
                        } else {
                            (&x[i..i + k], if i == 0 { NONE } else { x[i - 1] })
                        };
                        let h = BoundEnt::hash(dir, ctx);
                        if (h % shards as u64) as usize == sh {
                            *cnt.entry((h, out)).or_insert(0.0) += f;
                        }
                    }
                }
                // per context: n = sum of its outcome counts m, s = sum m log2 m; H = log2 n - s / n
                let mut pairs: Vec<((u64, u32), f64)> = cnt.into_iter().collect();
                pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                let mut out: Vec<(u64, u8, f64)> = Vec::new();
                let mut j = 0;
                while j < pairs.len() {
                    let h = pairs[j].0 .0;
                    let (mut n, mut s) = (0.0f64, 0.0f64);
                    while j < pairs.len() && pairs[j].0 .0 == h {
                        let m = pairs[j].1;
                        n += m;
                        s += m * m.log2();
                        j += 1;
                    }
                    if k == 1 || n >= BE_MIN {
                        out.push((h, ((n.log2() - s / n).max(0.0) * BE_Q).round().min(255.0) as u8, n));
                    }
                }
                out
            });
            for part in parts {
                for (h, q, n) in part {
                    // (a 32-bit key shared by two contexts keeps the first one's entropy)
                    be.tab.entry((h >> 32) as u32).or_insert(q);
                    n_ctx[dir][k - 1] += 1;
                    if k == 1 {
                        med_hist[q as usize] += n;
                    }
                }
            }
        }
        // the fallback of an unseen context: the count-weighted median of the single chars' entropies
        let tot: f64 = med_hist.iter().sum();
        let mut acc = 0.0;
        be.med[dir] = med_hist
            .iter()
            .position(|&h| {
                acc += h;
                acc >= tot / 2.0
            })
            .unwrap_or(0) as u8;
    }
    // percentiles of the scores of the inner positions, per class
    const B: usize = 4096;
    let be_ref = &be;
    let hists: Vec<Vec<f64>> = par_map(c.segs.len().div_ceil(B), threads, 1, || (), |_, b| {
        let mut h = vec![0.0f64; ncls * BE_NS];
        for si in b * B..((b + 1) * B).min(c.segs.len()) {
            let x = &c.segs[si];
            for i in 1..x.len() {
                h[be_ref.class(x, i) * BE_NS + be_ref.score(x, i)] += c.freqs[si];
            }
        }
        h
    });
    let mut hist = vec![0.0f64; ncls * BE_NS];
    for h in hists {
        for (a, b) in hist.iter_mut().zip(h) {
            *a += b;
        }
    }
    let mut cdf: Vec<f32> = Vec::with_capacity(ncls * BE_NS);
    let mut summary = Vec::new();
    for (k, hs) in hist.chunks(BE_NS).enumerate() {
        let tot: f64 = hs.iter().sum();
        let mut below = 0.0;
        let from = cdf.len();
        cdf.extend(hs.iter().map(|&h| {
            let v = if tot > 0.0 { (below + h / 2.0) / tot } else { 0.5 };
            below += h;
            v as f32
        }));
        if tot > 0.0 {
            // (the mean side entropy, in bits, at the 10/50/90th percentile of the class's positions)
            let q = |p: f64| cdf[from..].iter().position(|&v| v as f64 >= p).unwrap_or(BE_NS - 1) as f64 / BE_Q / 2.0;
            let name = if k + 1 == ncls { "change".to_string() } else { format!("group {k}") };
            summary.push(format!("{name}: {:.1}M, {:.2}/{:.2}/{:.2}", tot / 1e6, q(0.1), q(0.5), q(0.9)));
        }
    }
    be.cdf = cdf;
    eprintln!("  boundary entropy (be {w} bits, order {order}): {} contexts kept (before / after, by length: {:?} / {:?}), \
               median side entropy {:.2} / {:.2} bits ({:.1}s); inner positions per class and their mean side entropy at the \
               10/50/90th percentile: {}",
              be.tab.len(), &n_ctx[0][..order], &n_ctx[1][..order], be.med[0] as f64 / BE_Q, be.med[1] as f64 / BE_Q,
              t0.elapsed().as_secs_f64(), summary.join("; "));
    be
}

/// The trie arcs of a text (Dict::arcs_x): arcs[start[3i + k]..start[3i + k + 1]] leave position i
/// by walk k (0 prefixes, 1 cores, 2 suffixes).
#[derive(Default)]
struct Arcs {
    start: Vec<u32>,
    arcs: Vec<(u32, f64, f64)>,
    tm: Vec<u8>, // case_affixes: every arc's transition mask (CaseAcc::tm)
}

impl Arcs {
    #[inline]
    fn get(&self, i: usize, k: usize) -> &[(u32, f64, f64)] {
        &self.arcs[self.start[3 * i + k] as usize..self.start[3 * i + k + 1] as usize]
    }
    #[inline]
    fn get_tm(&self, i: usize, k: usize) -> &[u8] {
        &self.tm[self.start[3 * i + k] as usize..self.start[3 * i + k + 1] as usize]
    }
}

/// A row that is not in the dictionary, enabled for one parse (exact candidate scoring).
struct Extra {
    t: usize,
    s: Vec<u32>,
    cost: f64,
    spec: f64,
    usec: f64,
    umask: u64, // core: its canonical upper mask (case_cores; affixes too with case_affixes)
}

const EXTRA: u32 = u32::MAX - 1;

#[derive(Clone, Copy)]
struct Back {
    state: u8,
    from: u32,
    id: u32,
    v: u8,
    fvs: u8, // case_affixes: the variation state of the source entry (0 otherwise)
}

struct Scratch {
    d: [Vec<(f64, f64)>; 3],
    back: [Vec<Back>; 3],
    // pairwise DP (parse_pair): B scores/backpointers, prefix-done and core-done entries per position
    bs: Vec<(f64, f64)>,
    bb: Vec<BBack>,
    pe: Vec<Vec<PEnt>>,
    ce: Vec<Vec<CEnt>>,
    cm: Vec<(usize, u32, u8, f64, f64)>,
    slot: Vec<u32>, // parse_pair: index in ce[k] of the entry of core arc q and variation state vs
    pbc: Vec<(u32, f64)>, // parse_pair, code_len: per core arc, the last prefix id and its bits given the core
    sm: Vec<(usize, u32, f64, f64, u8)>,
    bc: Vec<f64>, // be_permille: the boundary cost of every position of the text being parsed (empty: off)
}

impl Scratch {
    fn new() -> Self {
        Scratch {
            d: [vec![], vec![], vec![]],
            back: [vec![], vec![], vec![]],
            bs: vec![],
            bb: vec![],
            pe: vec![],
            ce: vec![],
            cm: vec![],
            slot: vec![],
            pbc: vec![],
            sm: vec![],
            bc: vec![],
        }
    }
}

/// prefix-done entry: which prefix (id), score, and the between-tokens position it started at
#[derive(Clone, Copy)]
struct PEnt {
    pid: u32,
    vs: u8, // case_affixes: variation state (0 otherwise)
    s: (f64, f64),
    from: u32,
}

/// core-done entry: which core and variation, score, and the prefix-done entry it came from
#[derive(Clone, Copy)]
struct CEnt {
    cid: u32,
    v: u8,
    vs: u8, // case_affixes: variation state (0 otherwise)
    s: (f64, f64),
    fpos: u32,
    fidx: u32,
}

/// how a between-tokens position was reached: 1 = suffix arc from a core-done entry, 2 = byte
#[derive(Clone, Copy)]
struct BBack {
    kind: u8,
    pos: u32,
    idx: u32,
    sid: u32,
}

fn add_p(v: &mut Vec<PEnt>, e: PEnt) {
    match v.iter_mut().find(|o| o.pid == e.pid && o.vs == e.vs) {
        Some(o) => {
            if better(e.s, o.s) {
                *o = e;
            }
        }
        None => v.push(e),
    }
}

fn better(a: (f64, f64), b: (f64, f64)) -> bool {
    a.0 < b.0 - 1e-9 || (a.0 <= b.0 + 1e-9 && a.1 < b.1)
}

impl<'a> Dict<'a> {
    fn rebuild(&mut self) {
        self.tries = [Trie::build(&self.tabs[0]), Trie::build(&self.tabs[1]), Trie::build(&self.tabs[2])];
        // glue and price are fixed functions of the row string: only rows added since the last
        // rebuild need them (rows are only ever appended)
        for k in 0..3 {
            let from = self.tabs[k].scored;
            if k > 0 && self.ca {
                // (first: an affix's statistics are looked up by its canonical spelling)
                for i in from..self.tabs[k].strs.len() {
                    self.tabs[k].umask[i] = self.canon_mask(&self.tabs[k].strs[i]);
                }
            }
            if !self.frag[0].is_empty() {
                for i in from..self.tabs[k].strs.len() {
                    self.tabs[k].glue[i] = self.glue_of(k, &self.tabs[k].strs[i]);
                }
            }
            if self.pricing.pi > 0.0 || self.pricing.pi_core > 0.0 {
                for i in from..self.tabs[k].strs.len() {
                    self.tabs[k].price[i] = self.price_of(k, &self.tabs[k].strs[i]);
                }
            }
            if self.nu > 0.0 || self.pricing.rare > 0.0 {
                for i in from..self.tabs[k].strs.len() {
                    self.tabs[k].usec[i] = self.usec_of(k, &self.tabs[k].strs[i]);
                }
            }
            if k == 0 && self.case {
                for i in from..self.tabs[k].strs.len() {
                    self.tabs[k].umask[i] = self.canon_mask(&self.tabs[k].strs[i]);
                }
            }
            self.tabs[k].scored = self.tabs[k].strs.len();
        }
        self.refresh_arcs();
    }
    /// (primary, secondary) cost of the arc of the row at every trie node, exactly as parse_x adds
    /// it (cores: without the variation cost), in node order: a trie walk then reads it next to
    /// the node instead of from four row arrays. Refreshed whenever tries, costs or specs change.
    fn refresh_arcs(&mut self) {
        // (code_w without code_len: + wu x the row's bits; wu = 0 adds exactly 0)
        let (delta, lam, mu, wu) = (self.delta, self.lam, self.mu, self.cl.wu);
        let lam_a = lam + wu;
        for k in 0..3 {
            let t = &self.tabs[k];
            for nd in self.tries[k].node.iter_mut() {
                let r = nd.term as usize;
                nd.arc = match (nd.term == NONE, k) {
                    (true, _) => INF,
                    (_, 0) => (1.0 + t.usec[r] + wu * t.cost[r], t.cost[r]),
                    _ => (delta + lam_a * t.cost[r] + mu * t.spec[r] + t.usec[r], t.cost[r]),
                };
            }
        }
        self.cl.wv = std::array::from_fn(|v| wu * self.vcost[v]);
    }
    /// primary cost of an extra affix row's arc (as refresh_arcs makes a row's)
    #[inline(always)]
    fn extra_affix_prim(&self, e: &Extra) -> f64 {
        self.delta + (self.lam + self.cl.wu) * e.cost + self.mu * e.spec + e.usec
    }
    /// primary cost of an extra core row's arc with variation v (as refresh_arcs makes a row's,
    /// plus the variation's wv)
    #[inline(always)]
    fn extra_core_prim(&self, e: &Extra, v: usize) -> f64 {
        1.0 + e.usec + self.cl.wu * e.cost + self.cl.wv[v]
    }
    /// case_cores: upper mask of the canonical spelling of a folded core string (0 = all folded;
    /// only chars with an uppercase form, so a hash collision cannot make it unwritable)
    fn canon_mask(&self, s: &[u32]) -> u64 {
        if !self.case {
            return 0;
        }
        let m = self.canon.get(&seq_hash(s.iter().copied())).copied().unwrap_or(0);
        m & self.hasu_mask(s)
    }
    fn hasu_mask(&self, s: &[u32]) -> u64 {
        s.iter().take(64).enumerate().fold(0, |m, (k, &c)| m | (((self.sym.upper_of[c as usize] != NONE) as u64) << k))
    }
    /// case_affixes: the canonical spelling of an affix string (folded) as the raw-text statistics
    /// (glue, stems, rarity) know it; other strings as they are
    fn stat_str<'s>(&self, k: usize, s: &'s [u32]) -> std::borrow::Cow<'s, [u32]> {
        if k == 0 || !self.ca || s.is_empty() {
            return std::borrow::Cow::Borrowed(s);
        }
        let mut out = Vec::with_capacity(s.len());
        self.write_parts(&[(s, self.canon_mask(s))], false, 0, &mut out);
        std::borrow::Cow::Owned(out)
    }
    /// case_affixes: a token's parts (folded strings with their canonical upper masks; the first is
    /// a prefix if `prefix`) written as one text with variation v (see the header), appended to out
    fn write_parts(&self, parts: &[(&[u32], u64)], prefix: bool, v: u32, out: &mut Vec<u32>) {
        let sym = self.sym;
        let mut first = v == 1 || v == 7; // the first char with an uppercase form is still to come
        for (pi, &(row, uc)) in parts.iter().enumerate() {
            // camelCase: every part's own first cased char but a prefix's; PascalCase: every part's
            if v == 6 || (v == 5 && !(prefix && pi == 0)) {
                first = true;
            }
            out.extend(row.iter().enumerate().map(|(k, &f)| {
                let u = sym.upper_of[f as usize];
                let upper = k < 64 && uc >> k & 1 != 0;
                let or = |t: u32| if t == NONE { f } else { t };
                match v {
                    0 | 1 | 5 | 6 => {
                        if u != NONE && first {
                            first = false;
                            u
                        } else if upper {
                            u
                        } else {
                            f
                        }
                    }
                    2 => or(u),
                    3 if upper => u,
                    3 => or(sym.trad_of[f as usize]),
                    // Title: the token's first cased char upper, everything else folded
                    7 if u != NONE && first => {
                        first = false;
                        u
                    }
                    _ => f,
                }
            }));
        }
    }
    /// the text of core row r written with variation v, appended to out
    fn write_core(&self, r: usize, v: u32, out: &mut Vec<u32>) {
        let row = &self.tabs[0].strs[r];
        if self.ca {
            // (a core-only token: the whole-token variation, see case_affixes)
            self.write_parts(&[(row, self.tabs[0].umask[r])], false, v, out);
            return;
        }
        if !self.case {
            out.extend(self.sym.written(row, v));
            return;
        }
        let uc = self.tabs[0].umask[r];
        let sym = self.sym;
        out.extend(row.iter().enumerate().map(|(k, &f)| {
            let or = |t: u32| if t == NONE { f } else { t };
            let upper = k < 64 && uc >> k & 1 != 0;
            match v {
                0 if upper => sym.upper_of[f as usize],
                1 if k == 0 || upper => or(sym.upper_of[f as usize]),
                2 => or(sym.upper_of[f as usize]),
                3 if upper => sym.upper_of[f as usize],
                3 => or(sym.trad_of[f as usize]),
                _ => f,
            }
        }));
    }
    /// The core trie walk from x[i]: f(j, node, v) for every usable row spelling x[i..=j] (folded)
    /// that a variation writes exactly, with the variation the parse takes (cheapest, lowest id).
    /// Old cores (case off): the span's own mask decides, and the walk stops at the first span no
    /// variation writes (camelCase). case_cores: the row's canonical spelling decides (CaseAcc).
    #[inline(always)]
    fn core_walk(&self, x: &[u32], i: usize, ban: Ban, mut f: impl FnMut(usize, &Node, usize)) {
        if self.sym.mark_at(x[i]) {
            return; // mark_rule: no core starts at a combining mark
        }
        let tr = &self.tries[0];
        let mut node = 0u32;
        if !self.case {
            let mut m = 0u8;
            for j in i..x.len() {
                let c = x[j];
                if is_byte(c) {
                    break;
                }
                let Some(nn) = tr.step(node, self.sym.fold[c as usize]) else { break };
                node = nn;
                m = if j == i { self.sym.mask(c) } else { combine(m, self.sym.mask(c)) };
                let Some(v) = variation(m) else { break };
                let nd = &tr.node[node as usize];
                if self.usable(0, nd.term, ban) {
                    f(j, nd, v);
                }
            }
            return;
        }
        let mut ca = CaseAcc::default();
        for j in i..x.len() {
            let c = x[j];
            if is_byte(c) {
                break;
            }
            let Some(nn) = tr.step(node, self.sym.fold[c as usize]) else { break };
            node = nn;
            ca.push(self.sym.cf[c as usize], j - i);
            if ca.dead() {
                break;
            }
            let nd = &tr.node[node as usize];
            if self.usable(0, nd.term, ban) {
                // (code_len: the variation cheapest given this core)
                let uc = self.tabs[0].umask[nd.term as usize];
                let best = if self.cl.on {
                    ca.best(uc, &std::array::from_fn(|v| self.cl_vbits(nd.term, v)))
                } else {
                    ca.best(uc, &self.vcost)
                };
                if let Some(v) = best {
                    f(j, nd, v);
                }
            }
        }
    }
    /// case_affixes: the walk of trie k (0 cores, 1 prefixes, 2 suffixes; all keyed folded) from
    /// x[i]: f(j, node, tm) for every usable row spelling x[i..=j] folded, with the span's
    /// transition mask from the row's canonical spelling (only if some state can take it)
    #[inline(always)]
    fn walk_ca(&self, k: usize, x: &[u32], i: usize, ban: Ban, mut f: impl FnMut(usize, &Node, u8)) {
        if k < 2 && self.sym.mark_at(x[i]) {
            return; // mark_rule: no core or prefix starts at a combining mark
        }
        let (tr, um) = (&self.tries[k], &self.tabs[k].umask);
        let mut node = 0u32;
        let mut ca = CaseAcc::default();
        for j in i..x.len() {
            let c = x[j];
            if is_byte(c) {
                break;
            }
            let Some(nn) = tr.step(node, self.sym.fold[c as usize]) else { break };
            node = nn;
            ca.push(self.sym.cf[c as usize], j - i);
            if ca.dead() {
                break;
            }
            let nd = &tr.node[node as usize];
            if self.usable(k, nd.term, ban) {
                let tm = ca.tm(um[nd.term as usize]);
                if tm & TM_ALL != 0 {
                    f(j, nd, tm);
                }
            }
        }
    }
    /// case_affixes: the transition mask of an extra row at x[i..] (None if it does not fit)
    fn extra_tm(&self, x: &[u32], i: usize, e: &Extra) -> Option<u8> {
        let l = e.s.len();
        if i + l > x.len() || (e.t < 2 && self.sym.mark_at(x[i])) {
            return None;
        }
        let mut ca = CaseAcc::default();
        for k in 0..l {
            let c = x[i + k];
            if is_byte(c) || self.sym.fold[c as usize] != e.s[k] {
                return None;
            }
            ca.push(self.sym.cf[c as usize], k);
        }
        let tm = ca.tm(e.umask);
        (tm & TM_ALL != 0).then_some(tm)
    }
    /// case_affixes: the transition mask of a raw span as a part with the canonical spelling of its
    /// folded form (hash h of the folded string)
    fn span_tm(&self, ca: &CaseAcc, h: u64) -> u8 {
        let uc = if ca.hasu != 0 { self.canon.get(&h).copied().unwrap_or(0) & ca.hasu } else { 0 };
        ca.tm(uc)
    }
    /// the variation the parse takes for an extra core row at x[i..] (None if it does not fit)
    fn extra_core_var(&self, x: &[u32], i: usize, e: &Extra) -> Option<usize> {
        let l = e.s.len();
        if i + l > x.len() || self.sym.mark_at(x[i]) {
            return None;
        }
        let (mut m, mut ca) = (0u8, CaseAcc::default());
        for k in 0..l {
            let c = x[i + k];
            if is_byte(c) || self.sym.fold[c as usize] != e.s[k] {
                return None;
            }
            if self.case {
                ca.push(self.sym.cf[c as usize], k);
            } else {
                m = if k == 0 { self.sym.mask(c) } else { combine(m, self.sym.mask(c)) };
            }
        }
        if self.case {
            ca.best(e.umask, &self.vcost)
        } else {
            variation(m)
        }
    }
    /// Is raw (a text span) a legal core, i.e. does a variation write it from the canonical
    /// spelling of its folded form? Its folded form into f.
    fn legal_core(&self, raw: &[u32], f: &mut Vec<u32>) -> bool {
        let (mut m, mut ca) = (0u8, CaseAcc::default());
        f.clear();
        for (i, &ch) in raw.iter().enumerate() {
            if is_byte(ch) {
                return false;
            }
            if self.case {
                ca.push(self.sym.cf[ch as usize], i);
            } else {
                m = if i == 0 { self.sym.mask(ch) } else { combine(m, self.sym.mask(ch)) };
            }
            f.push(self.sym.fold[ch as usize]);
        }
        if self.ca {
            // a core-only token: some start state takes it (whole-token Capitalised)
            ca.tm(self.canon_mask(f)) & TM_ALL != 0
        } else if self.case {
            ca.best(self.canon_mask(f), &[0.0; NVAR_MAX]).is_some()
        } else {
            variation(m).is_some()
        }
    }
    /// one-time price of a row string of table k, in tokens
    fn price_of(&self, k: usize, s: &[u32]) -> f64 {
        let st = self.stat_str(k, s);
        let s = &*st;
        let pr = self.pricing;
        let pi = if k == 0 { pr.pi_core } else { pr.pi };
        if pi == 0.0 || s.is_empty() || (k == 0 && s.len() == 1) {
            return 0.0;
        }
        let l0 = 2.0 * ((s.len() + 1) as f64).log2() + s.iter().map(|&c| pr.char_bits[c as usize]).sum::<f64>();
        let m = if k == 0 {
            1.0 + 2.0 * self.glue_of(0, s)
        } else {
            let dn = pr.dstems[k - 1].get(&seq_hash(s.iter().copied())).copied().unwrap_or(0);
            1.0 + pr.k / (1.0 + dn as f64)
        };
        pi * l0 * m
    }
    /// spelling bits of a row string (L0)
    fn l0_of(&self, s: &[u32]) -> f64 {
        2.0 * ((s.len() + 1) as f64).log2() + s.iter().map(|&c| self.pricing.char_bits[c as usize]).sum::<f64>()
    }
    /// rarity bits of an affix string: -log2 of its raw count over its script's chars
    fn rare_bits(&self, s: &[u32]) -> f64 {
        let pr = self.pricing;
        if s.is_empty() || pr.script.is_empty() {
            return 0.0;
        }
        let g = s.iter().find(|&&c| self.sym.alnum[c as usize] != 0).map_or(0, |&c| pr.script[c as usize]) as usize;
        let n = pr.rawcnt.get(&seq_hash(s.iter().copied())).map_or(pr.min_freq / 2.0, |&v| v as f64).max(1.0);
        (pr.script_chars[g].max(n) / n).log2()
    }
    /// fixed extra primary cost per use of a row: nu x glue (+ rare x bits for affixes)
    fn usec_of(&self, k: usize, s: &[u32]) -> f64 {
        let st = self.stat_str(k, s);
        let s = &*st;
        self.nu * self.glue_of(k, s) + if k > 0 { self.pricing.rare * self.rare_bits(s) } else { 0.0 }
    }
    /// sum of the prices of the live rows
    fn total_price(&self) -> f64 {
        (0..3).map(|k| (0..self.tabs[k].strs.len()).filter(|&i| self.tabs[k].alive[i]).map(|i| self.tabs[k].price[i]).sum::<f64>()).sum()
    }
    /// glue score of a row string of table k (0 for single-char cores, empty affixes, unseen strings)
    fn glue_of(&self, k: usize, s: &[u32]) -> f64 {
        if s.is_empty() || (k == 0 && s.len() < 2) {
            return 0.0;
        }
        let st = self.stat_str(k, s);
        let s = &*st;
        self.frag[k].get(&seq_hash(s.iter().copied())).map_or(0.0, |&v| v as f64)
    }
    /// DP scores per position of state s: one per variation state in states 1 and 2 with
    /// case_affixes (slot (position - off) x wid + vs), else one
    #[inline(always)]
    fn wid(&self, s: usize) -> usize {
        if self.ca && s > 0 { NVS } else { 1 }
    }
    /// number of variations: 5 with case_cores, else the old 4
    fn nvar(&self) -> usize {
        if self.ca { NVAR_MAX } else if self.case { NVAR_CASE } else { 4 }
    }
    fn rows(&self) -> usize {
        self.tabs.iter().map(|t| t.live()).sum()
    }
    fn usable(&self, t: usize, r: u32, ban: Ban) -> bool {
        r != NONE && self.tabs[t].alive[r as usize] && ban != Some((t, r))
    }

    /// Exact (primary, secondary)-minimal parse of x; tokens appended to `out` if given.
    /// primary = tokens + sum over tokens of [delta x non-empty affixes + lambda x affix bits].
    fn parse(&self, x: &[u32], sc: &mut Scratch, out: Option<&mut Vec<Tok>>, ban: Ban) -> (f64, f64) {
        self.parse_x(x, sc, out, ban, None)
    }

    /// 0 = all other chars, 1 = all letters, 2 = all digits, 3 = mixed
    fn class_of(&self, s: &[u32]) -> u8 {
        let mut c = None;
        for &ch in s {
            let k = if is_byte(ch) { 0 } else { self.sym.alnum[ch as usize] as u8 };
            match c {
                None => c = Some(k),
                Some(o) if o != k => return 3,
                _ => {}
            }
        }
        c.unwrap_or(0)
    }

    /// Is this string allowed as a row of table t under the restricted-shaped mode?
    fn row_ok(&self, t: usize, s: &[u32]) -> bool {
        if t < 2 && s.first().is_some_and(|&c| self.sym.mark_at(c)) {
            return false; // mark_rule: such a row could never be used
        }
        if !self.classes || s.is_empty() {
            return true;
        }
        if t == 0 {
            return self.class_of(s) != 3;
        }
        let other = |ch: u32| is_byte(ch) || self.sym.alnum[ch as usize] == 0;
        // strip the other-char part: trailing for suffixes, leading for prefixes
        let core = if t == 2 {
            let n = s.iter().position(|&ch| other(ch)).unwrap_or(s.len());
            if !s[n..].iter().all(|&ch| other(ch)) {
                return false;
            }
            &s[..n]
        } else {
            let n = s.iter().position(|&ch| !other(ch)).unwrap_or(s.len());
            &s[n..]
        };
        core.is_empty() || (core.iter().all(|&ch| !other(ch)) && self.allow[t].contains(core))
    }

    /// -log2 q(affix | core) with q(a | c) = (n(a, c) + q(a)) / (n(c) + 1); k = 0: (prefix a, core b),
    /// k = 1: (core a, suffix b). `qa` is the affix's unconditional probability.
    fn cond(&self, k: usize, a: u32, b: u32, qa: f64) -> f64 {
        let (core, key) = if k == 0 { (b, (a, b)) } else { (a, (a, b)) };
        let n_ab = self.cn[k].get(&key).copied().unwrap_or(0.0) as f64;
        let n_c = self.ncore.get(core as usize).copied().unwrap_or(0.0) as f64;
        -((n_ab + qa) / (n_c + 1.0)).log2()
    }

    fn pmi_of(&self, k: usize, a: u32, b: u32) -> f64 {
        self.pmi[k].get(&(a, b)).map_or(0.0, |&v| v as f64)
    }

    /// code_len: -log2 p(a | c) of affix row a on side k (0 prefix, 1 suffix) given core c, where
    /// `ab` = -log2 p(a) (its row cost, or an extra row's). An extra (or unseen) core has no
    /// counts: the marginal.
    #[inline(always)]
    fn cl_abits(&self, k: usize, a: u32, c: u32, ab: f64) -> f64 {
        let Some(bo) = self.cl.bo.get(c as usize) else { return ab };
        let (st, it) = &self.cl.adj[k];
        let l = &it[st[c as usize] as usize..st[c as usize + 1] as usize];
        let hit = if l.len() <= 8 {
            l.iter().find(|e| e.0 == a).map(|e| e.1)
        } else {
            l.binary_search_by_key(&a, |e| e.0).ok().map(|i| l[i].1)
        };
        match hit {
            Some(b) => b as f64,
            None => ab + bo[k] as f64,
        }
    }

    /// code_len: -log2 p(v | c) (an extra or unseen core: the marginal variation cost)
    #[inline(always)]
    fn cl_vbits(&self, c: u32, v: usize) -> f64 {
        match self.cl.vb.get(c as usize) {
            Some(r) => r[v],
            None => self.vcost[v],
        }
    }

    /// code_len: the code length of token type t in bits (not a byte token)
    fn cl_bits(&self, t: &Tok) -> f64 {
        let (cp, cs) = (&self.tabs[1].cost, &self.tabs[2].cost);
        self.tabs[0].cost[t.c as usize] + self.cl_vbits(t.c, t.v as usize)
            + self.cl_abits(0, t.p, t.c, cp[t.p as usize]) + self.cl_abits(1, t.s, t.c, cs[t.s as usize])
    }

    /// parse_pair: the B score of core-done entry e closed by the empty suffix
    #[inline(always)]
    fn pair_close(&self, e: &CEnt) -> (f64, f64) {
        let (lam, mu, lamc) = (self.lam, self.mu, self.lamc);
        let (cs, ss) = (&self.tabs[2].cost, &self.tabs[2].spec);
        let empty_s = (lam * cs[0] + mu * ss[0], cs[0]);
        let cond = if lamc > 0.0 { lamc * self.cond(1, e.cid, 0, (-cs[0]).exp2()) } else { 0.0 };
        if self.cl.on {
            let sb = self.cl_abits(1, 0, e.cid, cs[0]);
            (e.s.0 + empty_s.0 + cond + self.cl.w * sb, e.s.1 + sb)
        } else {
            (e.s.0 + empty_s.0 + cond, e.s.1 + empty_s.1)
        }
    }

    /// parse_pair: the core arcs out of i into cm, (end, core id, variation (case_affixes: the
    /// transition mask), primary, secondary)
    #[inline(always)]
    fn pair_core_arcs(&self, x: &[u32], i: usize, ban: Ban, extra: Option<&Extra>, cm: &mut Vec<(usize, u32, u8, f64, f64)>) {
        let cc = &self.tabs[0].cost;
        let cl = self.cl.on;
        cm.clear();
        // (case_affixes: the u8 is the transition mask and the variation cost is added per
        // prefix-done entry, pair_ce)
        if self.ca {
            self.walk_ca(0, x, i, ban, |j, nd, tm| {
                let r = nd.term;
                cm.push((j + 1, r, tm, 1.0 + self.tabs[0].usec[r as usize], cc[r as usize]));
            });
            if let Some(e) = extra.filter(|e| e.t == 0) {
                if let Some(tm) = self.extra_tm(x, i, e) {
                    cm.push((i + e.s.len(), EXTRA, tm, 1.0 + e.usec, e.cost));
                }
            }
        } else {
            self.core_walk(x, i, ban, |j, nd, v| {
                let r = nd.term;
                let vb = if cl { self.cl_vbits(r, v) } else { self.vcost[v] };
                cm.push((j + 1, r, v as u8, 1.0 + self.tabs[0].usec[r as usize], cc[r as usize] + vb));
            });
            if let Some(e) = extra {
                if e.t == 0 {
                    if let Some(v) = self.extra_core_var(x, i, e) {
                        cm.push((i + e.s.len(), EXTRA, v as u8, 1.0 + e.usec, e.cost + self.vcost[v]));
                    }
                }
            }
        }
    }

    /// parse_pair: the suffix arcs out of i into sm, (end, suffix id, primary, secondary, mask)
    #[inline(always)]
    fn pair_suffix_arcs(&self, x: &[u32], i: usize, ban: Ban, extra: Option<&Extra>, sm: &mut Vec<(usize, u32, f64, f64, u8)>) {
        let n = x.len();
        let (delta, lam, mu) = (self.delta, self.lam, self.mu);
        let (cs, ss) = (&self.tabs[2].cost, &self.tabs[2].spec);
        sm.clear();
        if self.ca {
            self.walk_ca(2, x, i, ban, |j, nd, tm| {
                let r = nd.term as usize;
                sm.push((j + 1, nd.term, delta + lam * cs[r] + mu * ss[r] + self.tabs[2].usec[r], cs[r], tm));
            });
            if let Some(e) = extra.filter(|e| e.t == 2) {
                if let Some(tm) = self.extra_tm(x, i, e) {
                    sm.push((i + e.s.len(), EXTRA, delta + lam * e.cost + mu * e.spec + e.usec, e.cost, tm));
                }
            }
        } else {
            let mut node = 0u32;
            for j in i..n {
                let Some(m) = self.tries[2].step(node, x[j]) else { break };
                node = m;
                let r = self.tries[2].term(node);
                if self.usable(2, r, ban) {
                    sm.push((j + 1, r, delta + lam * cs[r as usize] + mu * ss[r as usize] + self.tabs[2].usec[r as usize], cs[r as usize], TM_ALL));
                }
            }
            if let Some(e) = extra {
                let l = e.s.len();
                if e.t == 2 && i + l <= n && x[i..i + l] == e.s[..] {
                    sm.push((i + l, EXTRA, delta + lam * e.cost + mu * e.spec + e.usec, e.cost, TM_ALL));
                }
            }
        }
    }

    /// parse_pair: the core-done entry (variation, variation state, score) from prefix-done entry
    /// pe and core arc `arc` (None if the arc does not take pe's variation state). pb() = the
    /// prefix's code length given the core (code_len only; the caller may cache it).
    #[inline(always)]
    fn pair_ce(&self, pe: &PEnt, arc: (usize, u32, u8, f64, f64), extra: Option<&Extra>, pb: impl FnOnce() -> f64)
        -> Option<(u8, u8, (f64, f64))> {
        let (mu2, lamc) = (self.mu2, self.lamc);
        let cp = &self.tabs[1].cost;
        let (_, cid, mut v, c1, mut c2) = arc;
        let mut vs = 0u8;
        if self.ca {
            let tm = v;
            if vs_allow(tm) >> pe.vs & 1 == 0 {
                return None;
            }
            v = vs_var(pe.vs as usize) as u8;
            vs = vs_to(pe.vs as usize, tm) as u8;
            c2 += if self.cl.on { self.cl_vbits(cid, v as usize) } else { self.vcost[v as usize] };
        }
        let mut pair = if mu2 > 0.0 { mu2 * self.pmi_of(0, pe.pid, cid) } else { 0.0 };
        if lamc > 0.0 {
            let qa = if pe.pid == EXTRA { (-extra.map_or(20.0, |e| e.cost)).exp2() } else { (-cp[pe.pid as usize]).exp2() };
            pair += lamc * self.cond(0, pe.pid, cid, qa);
        }
        let s = if self.cl.on {
            // the core's bits: its marginal, the variation's and the prefix's given it
            let bits = c2 + pb();
            (pe.s.0 + c1 + pair + self.cl.w * bits, pe.s.1 + bits)
        } else {
            (pe.s.0 + c1 + pair, pe.s.1 + c2)
        };
        Some((v, vs, s))
    }

    /// code_len: the prefix's code length given the core, for pair_ce
    #[inline(always)]
    fn pair_pb(&self, pid: u32, cid: u32, extra: Option<&Extra>) -> f64 {
        let pa = if pid == EXTRA { extra.map_or(20.0, |e| e.cost) } else { self.tabs[1].cost[pid as usize] };
        self.cl_abits(0, pid, cid, pa)
    }

    /// parse_pair: the B score from core-done entry ce and suffix arc `arc` (None if the arc does
    /// not take ce's variation state)
    #[inline(always)]
    fn pair_bs(&self, ce: &CEnt, arc: (usize, u32, f64, f64, u8), extra: Option<&Extra>) -> Option<(f64, f64)> {
        let (mu2, lamc) = (self.mu2, self.lamc);
        let cs = &self.tabs[2].cost;
        let (_, sid, c1, c2, tm) = arc;
        if vs_allow(tm) >> ce.vs & 1 == 0 {
            return None; // (never without case_affixes: vs 0, mask TM_ALL)
        }
        let mut pair = if mu2 > 0.0 { mu2 * self.pmi_of(1, ce.cid, sid) } else { 0.0 };
        if lamc > 0.0 {
            let qa = if sid == EXTRA { (-extra.map_or(20.0, |e| e.cost)).exp2() } else { (-cs[sid as usize]).exp2() };
            pair += lamc * self.cond(1, ce.cid, sid, qa);
        }
        Some(if self.cl.on {
            let sb = self.cl_abits(1, sid, ce.cid, c2); // (c2: the suffix's -log2 q)
            (ce.s.0 + c1 + pair + self.cl.w * sb, ce.s.1 + sb)
        } else {
            (ce.s.0 + c1 + pair, ce.s.1 + c2)
        })
    }

    /// The exact DP with pairwise costs: prefix-done states are kept per prefix id and core-done
    /// states per core id, so a token can pay mu2 x excess PMI of its (prefix, core) and
    /// (core, suffix) pairs (and code_len's affix bits given the core). Same arcs and costs as
    /// parse_x otherwise.
    fn parse_pair(&self, x: &[u32], sc: &mut Scratch, out: Option<&mut Vec<Tok>>, ban: Ban, extra: Option<&Extra>)
        -> (f64, f64) {
        let n = x.len();
        self.pair_init(n, sc);
        for i in 0..=n {
            self.pair_step(x, i, sc, ban, extra);
        }
        self.pair_finish(x, sc, out)
    }

    /// parse_pair's DP state before position 0 of a text of n chars
    fn pair_init(&self, n: usize, sc: &mut Scratch) {
        sc.bs.clear();
        sc.bs.resize(n + 1, INF);
        sc.bb.clear();
        sc.bb.resize(n + 1, BBack { kind: 0, pos: 0, idx: 0, sid: 0 });
        if sc.pe.len() < n + 1 {
            sc.pe.resize(n + 1, Vec::new());
            sc.ce.resize(n + 1, Vec::new());
        }
        for i in 0..=n {
            sc.pe[i].clear();
            sc.ce[i].clear();
        }
        sc.bs[0] = (0.0, 0.0);
    }

    /// Position i of parse_pair's DP: the empty closures at i, then every arc out of i (entries
    /// at i are final afterwards; only positions > i get new relaxations)
    #[inline(always)]
    fn pair_step(&self, x: &[u32], i: usize, sc: &mut Scratch, ban: Ban, extra: Option<&Extra>) {
        let n = x.len();
        let (delta, lam, mu, lamc) = (self.delta, self.lam, self.mu, self.lamc);
        let (cp, cs) = (&self.tabs[1].cost, &self.tabs[2].cost);
        let (sp, ss) = (&self.tabs[1].spec, &self.tabs[2].spec);
        // code_len: the bits of a token are paid where its pairs are known (the prefix's given the
        // core with the core, the suffix's with the suffix): affix arcs carry no secondary cost,
        // and every bit also costs w in the primary
        let (cl, w) = (self.cl.on, self.cl.w);
        let empty_p = (lam * cp[0] + mu * sp[0], if cl { 0.0 } else { cp[0] });
        if !sc.bc.is_empty() && sc.bc[i] != 0.0 {
            // be_permille: the boundary cost of i on every entry reached by a piece ending here (as
            // step_xp; sc.bc = the text's boundary costs, filled by the caller)
            let b = sc.bc[i];
            sc.bs[i].1 += b;
            for e in sc.pe[i].iter_mut() {
                e.s.1 += b;
            }
            for e in sc.ce[i].iter_mut() {
                e.s.1 += b;
            }
        }
        {
            // empty suffix: core-done -> between tokens
            for idx in 0..sc.ce[i].len() {
                let c = self.pair_close(&sc.ce[i][idx]);
                if better(c, sc.bs[i]) {
                    sc.bs[i] = c;
                    sc.bb[i] = BBack { kind: 1, pos: i as u32, idx: idx as u32, sid: 0 };
                }
            }
            let b = sc.bs[i];
            if !b.0.is_infinite() {
                // (case_affixes: one entry per start state)
                for vs in 0..if self.ca { NVS } else { 1 } {
                    if self.ca && vs_start(TM_EMPTY) >> vs & 1 == 0 {
                        continue;
                    }
                    add_p(&mut sc.pe[i], PEnt { pid: 0, vs: vs as u8, s: (b.0 + empty_p.0, b.1 + empty_p.1), from: i as u32 });
                }
            }
            if i == n {
                return;
            }
            if !b.0.is_infinite() {
                let nb = self.sym.byte_len(x[i]) as f64;
                let c = if cl {
                    // (the secondary keeps the old 1000 per byte: a char core still wins every tie)
                    (b.0 + nb * (1.0 + (lam + lamc) * (cp[0] + cs[0]) + mu * (sp[0] + ss[0]) + w * self.cl.byte), b.1 + 1000.0 * nb)
                } else {
                    (b.0 + nb * (1.0 + (lam + lamc) * (cp[0] + cs[0]) + mu * (sp[0] + ss[0])), b.1 + 1000.0 * nb)
                };
                if better(c, sc.bs[i + 1]) {
                    sc.bs[i + 1] = c;
                    sc.bb[i + 1] = BBack { kind: 2, pos: i as u32, idx: 0, sid: 0 };
                }
                if self.ca {
                    let pe = &mut sc.pe;
                    self.walk_ca(1, x, i, ban, |j, nd, tm| {
                        let r = nd.term as usize;
                        let s = (b.0 + delta + lam * cp[r] + mu * sp[r] + self.tabs[1].usec[r], b.1 + if cl { 0.0 } else { cp[r] });
                        let m = vs_start(tm);
                        for vs in 0..NVS {
                            if m >> vs & 1 != 0 {
                                add_p(&mut pe[j + 1], PEnt { pid: nd.term, vs: vs as u8, s, from: i as u32 });
                            }
                        }
                    });
                    if let Some(e) = extra.filter(|e| e.t == 1) {
                        if let Some(tm) = self.extra_tm(x, i, e) {
                            let s = (b.0 + delta + lam * e.cost + mu * e.spec + e.usec, b.1 + if cl { 0.0 } else { e.cost });
                            let m = vs_start(tm);
                            for vs in 0..NVS {
                                if m >> vs & 1 != 0 {
                                    add_p(&mut sc.pe[i + e.s.len()], PEnt { pid: EXTRA, vs: vs as u8, s, from: i as u32 });
                                }
                            }
                        }
                    }
                } else {
                    let mut node = 0u32;
                    for j in if self.sym.mark_at(x[i]) { n..n } else { i..n } {
                        let Some(m) = self.tries[1].step(node, x[j]) else { break };
                        node = m;
                        let r = self.tries[1].term(node);
                        if self.usable(1, r, ban) {
                            let e = PEnt {
                                pid: r,
                                vs: 0,
                                s: (b.0 + delta + lam * cp[r as usize] + mu * sp[r as usize] + self.tabs[1].usec[r as usize], b.1 + if cl { 0.0 } else { cp[r as usize] }),
                                from: i as u32,
                            };
                            add_p(&mut sc.pe[j + 1], e);
                        }
                    }
                    if let Some(e) = extra {
                        let l = e.s.len();
                        if e.t == 1 && i + l <= n && x[i..i + l] == e.s[..] && !self.sym.mark_at(x[i]) {
                            let pe = PEnt { pid: EXTRA, vs: 0, s: (b.0 + delta + lam * e.cost + mu * e.spec + e.usec, b.1 + if cl { 0.0 } else { e.cost }), from: i as u32 };
                            add_p(&mut sc.pe[i + l], pe);
                        }
                    }
                }
            }
            if !sc.pe[i].is_empty() {
                self.pair_core_arcs(x, i, ban, extra, &mut sc.cm);
                // (core-done entries: (cid, vs) at k can only come from this position and this core
                // arc, since cid fixes the length, so a slot per (arc, vs) finds an entry's index)
                let wv = if self.ca { NVS } else { 1 };
                sc.slot.clear();
                sc.slot.resize(sc.cm.len() * wv, NONE);
                if cl {
                    sc.pbc.clear();
                    sc.pbc.resize(sc.cm.len(), (NONE, 0.0));
                }
                for pidx in 0..sc.pe[i].len() {
                    let pe = sc.pe[i][pidx];
                    for q in 0..sc.cm.len() {
                        let arc = sc.cm[q];
                        // (code_len: the prefix's bits once per prefix: its entries are consecutive)
                        let pbc = &mut sc.pbc;
                        let Some((v, vs, s)) = self.pair_ce(&pe, arc, extra, || {
                            let pc = &mut pbc[q];
                            if pc.0 != pe.pid {
                                *pc = (pe.pid, self.pair_pb(pe.pid, arc.1, extra));
                            }
                            pc.1
                        }) else {
                            continue;
                        };
                        let (k, cid) = (arc.0, arc.1);
                        let e = CEnt { cid, v, vs, s, fpos: i as u32, fidx: pidx as u32 };
                        let sl = &mut sc.slot[q * wv + vs as usize];
                        if *sl == NONE {
                            *sl = sc.ce[k].len() as u32;
                            sc.ce[k].push(e);
                        } else {
                            let o = &mut sc.ce[k][*sl as usize];
                            if better(e.s, o.s) {
                                *o = e;
                            }
                        }
                    }
                }
            }
            if !sc.ce[i].is_empty() {
                self.pair_suffix_arcs(x, i, ban, extra, &mut sc.sm);
                for cidx in 0..sc.ce[i].len() {
                    let ce = sc.ce[i][cidx];
                    for q in 0..sc.sm.len() {
                        let arc = sc.sm[q];
                        let Some(c) = self.pair_bs(&ce, arc, extra) else { continue };
                        let j = arc.0;
                        if better(c, sc.bs[j]) {
                            sc.bs[j] = c;
                            sc.bb[j] = BBack { kind: 1, pos: i as u32, idx: cidx as u32, sid: arc.1 };
                        }
                    }
                }
            }
        }
    }

    /// parse_pair's result after every position: the score, and the tokens into `out` if given
    fn pair_finish(&self, x: &[u32], sc: &Scratch, out: Option<&mut Vec<Tok>>) -> (f64, f64) {
        let n = x.len();
        let res = sc.bs[n];
        if let Some(out) = out {
            out.clear();
            if res.0.is_infinite() {
                return res;
            }
            let mut pos = n;
            while pos > 0 {
                let b = sc.bb[pos];
                if b.kind == 2 {
                    out.push(Tok { v: self.sym.byte_len(x[b.pos as usize]), p: 0, c: NONE, s: 0 });
                    pos = b.pos as usize;
                    continue;
                }
                let ce = sc.ce[b.pos as usize][b.idx as usize];
                let pe = sc.pe[ce.fpos as usize][ce.fidx as usize];
                out.push(Tok { v: ce.v as u32, p: pe.pid, c: ce.cid, s: b.sid });
                pos = pe.from as usize;
            }
            out.reverse();
        }
        res
    }

    /// One position i of parse_x's DP: the empty closures at i (if `close`), then every arc out of i.
    /// Scores of position p are d[state][p - off] (a window of the text when off > 0), and must
    /// exist up to min(n, i + longest arc); backpointers are kept only if BACK. parse_x and the
    /// windowed re-scoring (parse_x_window) both run exactly this code, so their sums and
    /// tie-breaks are the same.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn step_x<const BACK: bool>(&self, x: &[u32], i: usize, d: &mut [Vec<(f64, f64)>; 3], back: &mut [Vec<Back>; 3], off: usize,
                                ban: Ban, extra: Option<&Extra>, close: bool, arcs: Option<&Arcs>, bc: &[f64]) {
        self.step_xp::<BACK>(x, i, d, back, off, ban, extra, close, arcs, bc, |_| {});
    }

    /// step_x, calling probe(d) at the point where an extra row's arc out of i is relaxed (after the
    /// closures and the byte arc, before the trie arcs; only if i < n)
    /// be_permille (bc = the text's boundary costs, Dict::be_fill; empty = off): every entry of
    /// position i was reached by a non-empty piece ending at i, so with `close` the boundary cost
    /// of i is added to all of them once, before the closures (which then carry it on): every parse
    /// through i pays it exactly once, and entries are compared among themselves (all or none of
    /// them carry it) exactly as without it. Scores of positions > i are pending and never carry it.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn step_xp<const BACK: bool>(&self, x: &[u32], i: usize, d: &mut [Vec<(f64, f64)>; 3], back: &mut [Vec<Back>; 3], off: usize,
                                 ban: Ban, extra: Option<&Extra>, close: bool, arcs: Option<&Arcs>, bc: &[f64],
                                 probe: impl FnMut(&[Vec<(f64, f64)>; 3])) {
        if close && !bc.is_empty() && bc[i] != 0.0 {
            let b = bc[i];
            for (s, ds) in d.iter_mut().enumerate() {
                let k = self.wid(s);
                for e in &mut ds[(i - off) * k..(i - off + 1) * k] {
                    e.1 += b;
                }
            }
        }
        self.step_xs::<BACK>(x, i, d, back, off, ban, extra, close, arcs, probe)
    }

    /// step_xp without the boundary cost
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn step_xs<const BACK: bool>(&self, x: &[u32], i: usize, d: &mut [Vec<(f64, f64)>; 3], back: &mut [Vec<Back>; 3], off: usize,
                                 ban: Ban, extra: Option<&Extra>, close: bool, arcs: Option<&Arcs>,
                                 mut probe: impl FnMut(&[Vec<(f64, f64)>; 3])) {
        if self.ca {
            return self.step_xa::<BACK>(x, i, d, back, off, ban, extra, close, arcs, probe);
        }
        let n = x.len();
        let (cp, cs) = (&self.tabs[1].cost, &self.tabs[2].cost);
        let (lam, mu, wu) = (self.lam, self.mu, self.cl.wu);
        let (lam_a, wv) = (lam + wu, &self.cl.wv);
        let (sp, ss) = (&self.tabs[1].spec, &self.tabs[2].spec);
        let relax = |d: &mut [Vec<(f64, f64)>; 3], back: &mut [Vec<Back>; 3], s: usize, i: usize, t: usize, j: usize, dt: f64, dc: f64, id: u32, v: u8| {
            let o = d[s][i - off];
            if o.0.is_infinite() {
                return;
            }
            let c = (o.0 + dt, o.1 + dc);
            if better(c, d[t][j - off]) {
                d[t][j - off] = c;
                if BACK {
                    back[t][j - off] = Back { state: s as u8, from: i as u32, id, v, fvs: 0 };
                }
            }
        };
        if close {
            // empty closure, order C -> B -> P
            relax(d, back, 2, i, 0, i, lam_a * cs[0] + mu * ss[0], cs[0], 0, 0);
            relax(d, back, 0, i, 1, i, lam_a * cp[0] + mu * sp[0], cp[0], 0, 0);
        }
        if i == n {
            return;
        }
        // byte fallback: each byte token pays what a token with empty affixes pays, and a high
        // secondary cost, so a char core always wins a tie
        let nb = self.sym.byte_len(x[i]) as f64;
        let byte_w = if wu == 0.0 { 0.0 } else { wu * self.cl.byte };
        relax(d, back, 0, i, 0, i + 1, nb * (1.0 + lam * (cp[0] + cs[0]) + mu * (sp[0] + ss[0]) + byte_w), 1000.0 * nb, NONE, 0);
        if let Some(e) = extra {
            let l = e.s.len();
            if i + l <= n {
                if e.t == 1 && x[i..i + l] == e.s[..] && !self.sym.mark_at(x[i]) {
                    relax(d, back, 0, i, 1, i + l, self.extra_affix_prim(e), e.cost, EXTRA, 0);
                } else if e.t == 2 && x[i..i + l] == e.s[..] {
                    relax(d, back, 2, i, 0, i + l, self.extra_affix_prim(e), e.cost, EXTRA, 0);
                } else if e.t == 0 {
                    if let Some(v) = self.extra_core_var(x, i, e) {
                        relax(d, back, 1, i, 2, i + l, self.extra_core_prim(e, v), e.cost + self.vcost[v], EXTRA, v as u8);
                    }
                }
            }
        }
        probe(d);
        if let Some(a) = arcs {
            // the arcs of the walks below, recorded by arcs_x (same order, same sums)
            debug_assert!(!BACK);
            for (k, s, t) in [(0, 0, 1), (1, 1, 2), (2, 2, 0)] {
                let o = d[s][i - off];
                if o.0.is_infinite() {
                    continue;
                }
                for &(j1, a0, a1) in a.get(i, k) {
                    let c = (o.0 + a0, o.1 + a1);
                    if better(c, d[t][j1 as usize - off]) {
                        d[t][j1 as usize - off] = c;
                    }
                }
            }
            return;
        }
        // trie walks from i (prefixes B -> P, cores P -> C, suffixes C -> B): the walk's source
        // score cannot change during it, so it is read once; the sums are those of relax
        let o = d[0][i - off];
        if !o.0.is_infinite() && !self.sym.mark_at(x[i]) {
            let tr = &self.tries[1];
            let (dt, bt) = (&mut d[1], &mut back[1]);
            let mut node = 0u32;
            for j in i..n {
                let Some(m) = tr.step(node, x[j]) else { break };
                node = m;
                let nd = &tr.node[node as usize];
                if self.usable(1, nd.term, ban) {
                    let c = (o.0 + nd.arc.0, o.1 + nd.arc.1);
                    if better(c, dt[j + 1 - off]) {
                        dt[j + 1 - off] = c;
                        if BACK {
                            bt[j + 1 - off] = Back { state: 0, from: i as u32, id: nd.term, v: 0, fvs: 0 };
                        }
                    }
                }
            }
        }
        let o = d[1][i - off];
        if !o.0.is_infinite() {
            let (dt, bt) = (&mut d[2], &mut back[2]);
            self.core_walk(x, i, ban, |j, nd, v| {
                let a0 = if wu == 0.0 { nd.arc.0 } else { nd.arc.0 + wv[v] };
                let c = (o.0 + a0, o.1 + (nd.arc.1 + self.vcost[v]));
                if better(c, dt[j + 1 - off]) {
                    dt[j + 1 - off] = c;
                    if BACK {
                        bt[j + 1 - off] = Back { state: 1, from: i as u32, id: nd.term, v: v as u8, fvs: 0 };
                    }
                }
            });
        }
        let o = d[2][i - off];
        if !o.0.is_infinite() {
            let tr = &self.tries[2];
            let (dt, bt) = (&mut d[0], &mut back[0]);
            let mut node = 0u32;
            for j in i..n {
                let Some(m) = tr.step(node, x[j]) else { break };
                node = m;
                let nd = &tr.node[node as usize];
                if self.usable(2, nd.term, ban) {
                    let c = (o.0 + nd.arc.0, o.1 + nd.arc.1);
                    if better(c, dt[j + 1 - off]) {
                        dt[j + 1 - off] = c;
                        if BACK {
                            bt[j + 1 - off] = Back { state: 2, from: i as u32, id: nd.term, v: 0, fvs: 0 };
                        }
                    }
                }
            }
        }
    }

    /// step_x with case_affixes: the same DP, with the scores of states 1 and 2 kept per variation
    /// state (slot q x NVS + vs, see wid). A prefix (empty: every start state) moves a token start
    /// into the states its transition mask allows, a core and then a suffix must allow the state
    /// they follow; the variation cost is paid on the core arc, as in step_x. Every arc's states
    /// are visited in increasing order after the arc itself, the same with or without `arcs`, so
    /// parse_x_window's replay adds up exactly as the walks do.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn step_xa<const BACK: bool>(&self, x: &[u32], i: usize, d: &mut [Vec<(f64, f64)>; 3], back: &mut [Vec<Back>; 3], off: usize,
                                 ban: Ban, extra: Option<&Extra>, close: bool, arcs: Option<&Arcs>,
                                 mut probe: impl FnMut(&[Vec<(f64, f64)>; 3])) {
        let n = x.len();
        let (cp, cs) = (&self.tabs[1].cost, &self.tabs[2].cost);
        let (lam, mu, wu) = (self.lam, self.mu, self.cl.wu);
        let (lam_a, wv) = (lam + wu, &self.cl.wv);
        let (sp, ss) = (&self.tabs[1].spec, &self.tabs[2].spec);
        let q = i - off;
        let at = |s: usize, q: usize, vs: usize| if s == 0 { q } else { q * NVS + vs };
        #[inline(always)]
        fn upd<const B: bool>(d: &mut [(f64, f64)], back: &mut [Back], k: usize, c: (f64, f64), b: Back) {
            if better(c, d[k]) {
                d[k] = c;
                if B {
                    back[k] = b;
                }
            }
        }
        let relax = |d: &mut [Vec<(f64, f64)>; 3], back: &mut [Vec<Back>; 3], s: usize, vs: usize, t: usize, tv: usize, j: usize, dt: f64, dc: f64, id: u32, v: u8| {
            let o = d[s][at(s, q, vs)];
            if o.0.is_infinite() {
                return;
            }
            let k = at(t, j - off, tv);
            upd::<BACK>(&mut d[t], &mut back[t], k, (o.0 + dt, o.1 + dc), Back { state: s as u8, from: i as u32, id, v, fvs: vs as u8 });
        };
        if close {
            // empty closure, order C -> B -> P
            for vs in 0..NVS {
                relax(d, back, 2, vs, 0, 0, i, lam_a * cs[0] + mu * ss[0], cs[0], 0, 0);
            }
            for vs in 0..NVS {
                if vs_start(TM_EMPTY) >> vs & 1 != 0 {
                    relax(d, back, 0, 0, 1, vs, i, lam_a * cp[0] + mu * sp[0], cp[0], 0, 0);
                }
            }
        }
        if i == n {
            return;
        }
        let nb = self.sym.byte_len(x[i]) as f64;
        let byte_w = if wu == 0.0 { 0.0 } else { wu * self.cl.byte };
        relax(d, back, 0, 0, 0, 0, i + 1, nb * (1.0 + lam * (cp[0] + cs[0]) + mu * (sp[0] + ss[0]) + byte_w), 1000.0 * nb, NONE, 0);
        if let Some(e) = extra {
            if let Some(tm) = self.extra_tm(x, i, e) {
                let j = i + e.s.len();
                let affix = self.extra_affix_prim(e);
                let (start, allow) = (vs_start(tm), vs_allow(tm));
                for vs in 0..NVS {
                    if (if e.t == 1 { start } else { allow }) >> vs & 1 == 0 {
                        continue;
                    }
                    match e.t {
                        1 => relax(d, back, 0, 0, 1, vs, j, affix, e.cost, EXTRA, 0),
                        2 => relax(d, back, 2, vs, 0, 0, j, affix, e.cost, EXTRA, 0),
                        0 => {
                            let v = vs_var(vs);
                            relax(d, back, 1, vs, 2, vs_to(vs, tm), j, self.extra_core_prim(e, v), e.cost + self.vcost[v], EXTRA, v as u8);
                        }
                        _ => {}
                    }
                }
            }
        }
        probe(d);
        // prefixes B -> P
        let o = d[0][q];
        if !o.0.is_infinite() {
            let (dt, bt) = (&mut d[1], &mut back[1]);
            let mut arc = |j1: usize, a0: f64, a1: f64, tm: u8, id: u32| {
                let c = (o.0 + a0, o.1 + a1);
                let m = vs_start(tm);
                for vs in 0..NVS {
                    if m >> vs & 1 != 0 {
                        upd::<BACK>(dt, bt, at(1, j1 - off, vs), c, Back { state: 0, from: i as u32, id, v: 0, fvs: 0 });
                    }
                }
            };
            if let Some(a) = arcs {
                for (&(j1, a0, a1), &tm) in a.get(i, 0).iter().zip(a.get_tm(i, 0)) {
                    arc(j1 as usize, a0, a1, tm, 0);
                }
            } else {
                self.walk_ca(1, x, i, ban, |j, nd, tm| arc(j + 1, nd.arc.0, nd.arc.1, tm, nd.term));
            }
        }
        // cores P -> C
        let src: [(f64, f64); NVS] = std::array::from_fn(|vs| d[1][at(1, q, vs)]);
        if src.iter().any(|o| !o.0.is_infinite()) {
            let (dt, bt) = (&mut d[2], &mut back[2]);
            let mut arc = |j1: usize, a0: f64, a1: f64, tm: u8, id: u32| {
                let m = vs_allow(tm);
                for vs in 0..NVS {
                    let o = src[vs];
                    if m >> vs & 1 != 0 && !o.0.is_infinite() {
                        let v = vs_var(vs);
                        let c = (o.0 + if wu == 0.0 { a0 } else { a0 + wv[v] }, o.1 + (a1 + self.vcost[v]));
                        upd::<BACK>(dt, bt, at(2, j1 - off, vs_to(vs, tm)), c, Back { state: 1, from: i as u32, id, v: v as u8, fvs: vs as u8 });
                    }
                }
            };
            if let Some(a) = arcs {
                for (&(j1, a0, a1), &tm) in a.get(i, 1).iter().zip(a.get_tm(i, 1)) {
                    arc(j1 as usize, a0, a1, tm, 0);
                }
            } else {
                self.walk_ca(0, x, i, ban, |j, nd, tm| arc(j + 1, nd.arc.0, nd.arc.1, tm, nd.term));
            }
        }
        // suffixes C -> B
        let src: [(f64, f64); NVS] = std::array::from_fn(|vs| d[2][at(2, q, vs)]);
        if src.iter().any(|o| !o.0.is_infinite()) {
            let (dt, bt) = (&mut d[0], &mut back[0]);
            let mut arc = |j1: usize, a0: f64, a1: f64, tm: u8, id: u32| {
                let m = vs_allow(tm);
                for vs in 0..NVS {
                    let o = src[vs];
                    if m >> vs & 1 != 0 && !o.0.is_infinite() {
                        upd::<BACK>(dt, bt, j1 - off, (o.0 + a0, o.1 + a1), Back { state: 2, from: i as u32, id, v: 0, fvs: vs as u8 });
                    }
                }
            };
            if let Some(a) = arcs {
                for (&(j1, a0, a1), &tm) in a.get(i, 2).iter().zip(a.get_tm(i, 2)) {
                    arc(j1 as usize, a0, a1, tm, 0);
                }
            } else {
                self.walk_ca(2, x, i, ban, |j, nd, tm| arc(j + 1, nd.arc.0, nd.arc.1, tm, nd.term));
            }
        }
    }

    /// Every arc of step_x's trie walks in x (no ban), whatever the source scores: per position and
    /// walk, (target, primary, secondary) in walk order, as step_x adds them (case_affixes: the
    /// core arcs without the variation cost, and every arc's transition mask in a.tm).
    fn arcs_x(&self, x: &[u32], a: &mut Arcs) {
        let n = x.len();
        a.start.clear();
        a.arcs.clear();
        a.tm.clear();
        if self.ca {
            for i in 0..n {
                for k in [1, 0, 2] {
                    a.start.push(a.arcs.len() as u32);
                    let (arcs, tms) = (&mut a.arcs, &mut a.tm);
                    self.walk_ca(k, x, i, None, |j, nd, tm| {
                        arcs.push(((j + 1) as u32, nd.arc.0, nd.arc.1));
                        tms.push(tm);
                    });
                }
            }
            a.start.push(a.arcs.len() as u32);
            return;
        }
        for i in 0..n {
            a.start.push(a.arcs.len() as u32);
            let tr = &self.tries[1];
            let mut node = 0u32;
            for j in if self.sym.mark_at(x[i]) { n..n } else { i..n } {
                let Some(m) = tr.step(node, x[j]) else { break };
                node = m;
                let nd = &tr.node[node as usize];
                if self.usable(1, nd.term, None) {
                    a.arcs.push(((j + 1) as u32, nd.arc.0, nd.arc.1));
                }
            }
            a.start.push(a.arcs.len() as u32);
            let arcs = &mut a.arcs;
            let (wu, wv) = (self.cl.wu, &self.cl.wv);
            self.core_walk(x, i, None, |j, nd, v| {
                arcs.push(((j + 1) as u32, if wu == 0.0 { nd.arc.0 } else { nd.arc.0 + wv[v] }, nd.arc.1 + self.vcost[v]))
            });
            a.start.push(a.arcs.len() as u32);
            let tr = &self.tries[2];
            let mut node = 0u32;
            for j in i..n {
                let Some(m) = tr.step(node, x[j]) else { break };
                node = m;
                let nd = &tr.node[node as usize];
                if self.usable(2, nd.term, None) {
                    a.arcs.push(((j + 1) as u32, nd.arc.0, nd.arc.1));
                }
            }
        }
        a.start.push(a.arcs.len() as u32);
    }

    /// parse_x(x).0's DP (no ban, no extra, mu2 = lamc = 0) from the recorded arcs: every
    /// position's final scores into d.
    fn scores_x(&self, x: &[u32], a: &Arcs, d: &mut [Vec<(f64, f64)>; 3], nob: &mut [Vec<Back>; 3], bc: &[f64]) {
        let n = x.len();
        for s in 0..3 {
            d[s].clear();
            d[s].resize((n + 1) * self.wid(s), INF);
        }
        d[0][0] = (0.0, 0.0);
        for i in 0..=n {
            self.step_x::<false>(x, i, d, nob, 0, None, None, true, Some(a), bc);
        }
    }

    /// parse_x(x, extra = e).0 (no ban, parse_x's variant only: mu2 = lamc = 0), computed from x's
    /// trie arcs (arcs_x) and the final scores `fin` of the plain parse (scores_x) by re-running
    /// the DP only around e's occurrences. `occ`: the positions where e's arc may start
    /// (a superset is fine), increasing; `lmax` >= max_arc(e.s.len()).
    /// Exact, not an approximation: the DP's future after time i (all positions < i done) depends
    /// only on the scores pending at positions >= i, and those are relaxed only from the last lmax
    /// positions. So at an occurrence a, the pending scores are rebuilt bit for bit by replaying the
    /// plain arcs out of [a - lmax, a) from their final scores (same arcs, same order, same sums);
    /// the DP with e then runs from a until the last lmax positions' final scores equal the plain
    /// parse's again and no arc of e reaches past them: from there on both DPs are the same, up to
    /// the next occurrence or to the end (whose score is then fin[0][n]). A window that never
    /// rejoins (e changed the parse) runs to the end of x.
    fn parse_x_window(&self, x: &[u32], arcs: &Arcs, fin: &[Vec<(f64, f64)>; 3], occ: &[u32], lmax: usize, e: &Extra,
                      w: &mut [Vec<(f64, f64)>; 3], nob: &mut [Vec<Back>; 3], bc: &[f64]) -> f64 {
        let n = x.len();
        let same = |w: &[Vec<(f64, f64)>; 3], p: usize, off: usize| {
            (0..3).all(|s| {
                let k = self.wid(s);
                (0..k).all(|v| {
                    let (a, b) = (w[s][(p - off) * k + v], fin[s][p * k + v]);
                    a.0.to_bits() == b.0.to_bits() && a.1.to_bits() == b.1.to_bits()
                })
            })
        };
        let mut k = 0;
        while k < occ.len() {
            let a = occ[k] as usize;
            let off = a.saturating_sub(lmax);
            // window scores cover [off, min(n, i + lmax)] when position i is processed
            let mut end = (a + lmax).min(n);
            for s in 0..3 {
                w[s].clear();
                w[s].resize((end - off + 1) * self.wid(s), INF);
            }
            if off == 0 {
                w[0][0] = (0.0, 0.0); // the start state, as parse_x seeds it
            }
            // pending scores at time a: the plain arcs out of [off, a), from their final scores
            for i in off..a {
                for s in 0..3 {
                    let k = self.wid(s);
                    w[s][(i - off) * k..(i - off + 1) * k].copy_from_slice(&fin[s][i * k..(i + 1) * k]);
                }
                self.step_x::<false>(x, i, w, nob, off, None, None, false, Some(arcs), bc);
            }
            // the DP with e from a on; `run` = number of positions just processed whose final scores
            // equal the plain parse's (those before a all do), `reach` = end of e's arcs so far
            let (mut run, mut reach) = (lmax, 0usize);
            let mut i = a;
            loop {
                let want = (i + lmax).min(n);
                if want > end {
                    for s in 0..3 {
                        w[s].resize((want - off + 1) * self.wid(s), INF);
                    }
                    end = want;
                }
                self.step_x::<false>(x, i, w, nob, off, None, Some(e), true, Some(arcs), bc);
                if i == n {
                    return w[0][n - off].0;
                }
                if k < occ.len() && occ[k] as usize == i {
                    reach = reach.max(i + e.s.len());
                    k += 1;
                }
                run = if same(w, i, off) { run + 1 } else { 0 };
                i += 1;
                if run >= lmax && reach < i {
                    break; // rejoined at time i
                }
            }
        }
        fin[0][n].0
    }

    /// Would the arcs of extra row e out of i (i < n) change any score, given the scores d at the
    /// point where step_x relaxes them (see step_xp)? Exactly step_x's relaxations of e's arcs, as a
    /// test: while none of them wins, none changes anything (a later one into the same entry then
    /// meets the same score as the first).
    fn extra_wins(&self, x: &[u32], i: usize, d: &[Vec<(f64, f64)>; 3], off: usize, e: &Extra) -> bool {
        let n = x.len();
        let q = i - off;
        let win = |s: usize, si: usize, t: usize, ti: usize, dt: f64, dc: f64| {
            let o = d[s][si];
            !o.0.is_infinite() && better((o.0 + dt, o.1 + dc), d[t][ti])
        };
        if self.ca {
            if let Some(tm) = self.extra_tm(x, i, e) {
                let j = i + e.s.len();
                let affix = self.extra_affix_prim(e);
                let (start, allow) = (vs_start(tm), vs_allow(tm));
                for vs in 0..NVS {
                    if (if e.t == 1 { start } else { allow }) >> vs & 1 == 0 {
                        continue;
                    }
                    let w = match e.t {
                        1 => win(0, q, 1, (j - off) * NVS + vs, affix, e.cost),
                        2 => win(2, q * NVS + vs, 0, j - off, affix, e.cost),
                        0 => {
                            let v = vs_var(vs);
                            win(1, q * NVS + vs, 2, (j - off) * NVS + vs_to(vs, tm), self.extra_core_prim(e, v), e.cost + self.vcost[v])
                        }
                        _ => false,
                    };
                    if w {
                        return true;
                    }
                }
            }
            return false;
        }
        let l = e.s.len();
        if i + l <= n {
            let affix = self.extra_affix_prim(e);
            if e.t == 1 && x[i..i + l] == e.s[..] && !self.sym.mark_at(x[i]) {
                return win(0, q, 1, i + l - off, affix, e.cost);
            } else if e.t == 2 && x[i..i + l] == e.s[..] {
                return win(2, q, 0, i + l - off, affix, e.cost);
            } else if e.t == 0 {
                if let Some(v) = self.extra_core_var(x, i, e) {
                    return win(1, q, 2, i + l - off, self.extra_core_prim(e, v), e.cost + self.vcost[v]);
                }
            }
        }
        false
    }

    fn parse_x(&self, x: &[u32], sc: &mut Scratch, out: Option<&mut Vec<Tok>>, ban: Ban, extra: Option<&Extra>)
        -> (f64, f64) {
        self.be.fill(x, &mut sc.bc);
        self.parse_xb(x, sc, out, ban, extra)
    }

    /// primary cost of the best parse of x, without the boundary cost (be_permille: it only breaks
    /// ties, so the primary cost is the same up to better's 1e-9 tolerance): for callers that read
    /// only the primary cost (proposal values, prune losses), which then skip computing it
    fn parse_prim(&self, x: &[u32], sc: &mut Scratch, ban: Ban) -> f64 {
        sc.bc.clear();
        self.parse_xb(x, sc, None, ban, None).0
    }

    /// parse_x with the boundary costs already in sc.bc (empty: none)
    fn parse_xb(&self, x: &[u32], sc: &mut Scratch, out: Option<&mut Vec<Tok>>, ban: Ban, extra: Option<&Extra>)
        -> (f64, f64) {
        if self.mu2 > 0.0 || self.lamc > 0.0 || self.cl.on {
            return self.parse_pair(x, sc, out, ban, extra);
        }
        let n = x.len();
        if out.is_none() {
            // scores only: the same DP without backpointers (they never change a score)
            for k in 0..3 {
                sc.d[k].clear();
                sc.d[k].resize((n + 1) * self.wid(k), INF);
            }
            sc.d[0][0] = (0.0, 0.0);
            for i in 0..=n {
                self.step_x::<false>(x, i, &mut sc.d, &mut sc.back, 0, ban, extra, true, None, &sc.bc);
            }
            return sc.d[0][n];
        }
        for k in 0..3 {
            sc.d[k].clear();
            sc.d[k].resize((n + 1) * self.wid(k), INF);
            // (backpointers are not reset: the backtrace only reads entries with a finite score,
            // and every one of those was written by this parse)
            let need = (n + 1) * self.wid(k);
            if sc.back[k].len() < need {
                sc.back[k].resize(need, Back { state: 9, from: 0, id: 0, v: 0, fvs: 0 });
            }
        }
        sc.d[0][0] = (0.0, 0.0);
        for i in 0..=n {
            self.step_x::<true>(x, i, &mut sc.d, &mut sc.back, 0, ban, extra, true, None, &sc.bc);
        }
        let res = sc.d[0][n];
        if let Some(out) = out {
            out.clear();
            if res.0.is_infinite() {
                return res;
            }
            // backtrace: labels come in suffix, core, prefix order (reversed)
            // (case_affixes: an entry of state 1 or 2 is also indexed by its variation state vs)
            let (mut st, mut pos, mut vs) = (0usize, n, 0usize);
            let mut cur = Tok { v: 0, p: 0, c: NONE, s: 0 };
            while !(st == 0 && pos == 0) {
                let b = sc.back[st][pos * self.wid(st) + vs];
                match (b.state, st) {
                    (0, 0) => {
                        // byte fallback: one unit per char, v = its number of byte tokens
                        out.push(Tok { v: self.sym.byte_len(x[b.from as usize]), p: 0, c: NONE, s: 0 });
                    }
                    (2, 0) => cur.s = b.id,
                    (1, 2) => {
                        cur.c = b.id;
                        cur.v = b.v as u32;
                    }
                    (0, 1) => {
                        cur.p = b.id;
                        out.push(cur);
                        cur = Tok { v: 0, p: 0, c: NONE, s: 0 };
                    }
                    _ => unreachable!(),
                }
                st = b.state as usize;
                pos = b.from as usize;
                vs = b.fvs as usize;
            }
            out.reverse();
        }
        res
    }

    /// An upper bound on the length of any arc of x (the longest path of any trie from any position,
    /// walked as the parses walk it; at least 1, a byte)
    fn longest_arc(&self, x: &[u32]) -> usize {
        let mut m = 1;
        for i in 0..x.len() {
            for k in 0..3 {
                let fold = k == 0 || self.ca;
                let (mut node, mut l) = (0u32, 0usize);
                for &c in &x[i..] {
                    if is_byte(c) {
                        break;
                    }
                    let Some(nn) = self.tries[k].step(node, if fold { self.sym.fold[c as usize] } else { c }) else { break };
                    node = nn;
                    l += 1;
                }
                m = m.max(l);
            }
        }
        m
    }

    /// Does extra row e have an arc out of x[i..] (parse_pair / extra_pair_wins' test)?
    fn extra_fits(&self, x: &[u32], i: usize, e: &Extra) -> bool {
        let l = e.s.len();
        if i + l > x.len() {
            return false;
        }
        if self.ca {
            return self.extra_tm(x, i, e).is_some();
        }
        match e.t {
            0 => self.extra_core_var(x, i, e).is_some(),
            1 => x[i..i + l] == e.s[..] && !self.sym.mark_at(x[i]),
            _ => x[i..i + l] == e.s[..],
        }
    }

    /// code_len's re-scoring filter: would extra row e change parse_pair(x)'s score? `sc` holds
    /// the plain parse_pair(x) (no ban, no extra) just run, i.e. every entry's final value. The
    /// DP with e differs from the plain one only through e's own entries (new keys: a prefix-done
    /// entry of e, a core-done entry of an extra core, or of a (core, state) the plain parse never
    /// reached) and through what they relax: an existing core-done entry, or a B score. So e's
    /// arcs are replayed at each of its occurrences, exactly as parse_pair relaxes them (same
    /// helpers), against the plain final values: while none is better, the DP with e ends with the
    /// plain parse's scores (up to ties within better's 1e-9). Conservative: any win (even one a
    /// later plain entry would undo) sends the segment to a full parse with e.
    fn extra_pair_wins(&self, x: &[u32], sc: &Scratch, e: &Extra, cm: &mut Vec<(usize, u32, u8, f64, f64)>,
                       sm: &mut Vec<(usize, u32, f64, f64, u8)>) -> bool {
        let n = x.len();
        let l = e.s.len();
        let (delta, lam, mu) = (self.delta, self.lam, self.mu);
        let ex = Some(e);
        // be_permille: an entry at position k carries k's boundary cost once final (pair_step), so a
        // replayed relaxation into k is compared with the final entries with it added
        let bat = |k: usize| if sc.bc.is_empty() { 0.0 } else { sc.bc[k] };
        let at = |c: (f64, f64), k: usize| (c.0, c.1 + bat(k));
        // a new core-done entry at k (its score with k's boundary cost): does it improve a B score
        // (empty suffix, or a suffix arc)?
        let private_ce = |ce: &CEnt, k: usize, sm: &mut Vec<(usize, u32, f64, f64, u8)>| -> bool {
            if better(self.pair_close(ce), sc.bs[k]) {
                return true;
            }
            if k < n {
                self.pair_suffix_arcs(x, k, None, ex, sm);
                for &arc in sm.iter() {
                    if self.pair_bs(ce, arc, ex).is_some_and(|c| better(at(c, arc.0), sc.bs[arc.0])) {
                        return true;
                    }
                }
            }
            false
        };
        for i in 0..n {
            if i + l > n {
                break;
            }
            // e's arcs out of i, exactly as parse_pair makes them
            let fits = if self.ca {
                self.extra_tm(x, i, e).is_some()
            } else {
                match e.t {
                    0 => self.extra_core_var(x, i, e).is_some(),
                    1 => x[i..i + l] == e.s[..] && !self.sym.mark_at(x[i]),
                    _ => x[i..i + l] == e.s[..],
                }
            };
            if !fits {
                continue;
            }
            match e.t {
                1 => {
                    // e's prefix-done entries at a = i + l, then every core arc out of a
                    let b = sc.bs[i];
                    if b.0.is_infinite() {
                        continue;
                    }
                    let s = (b.0 + delta + lam * e.cost + mu * e.spec + e.usec, b.1 + if self.cl.on { 0.0 } else { e.cost });
                    let s = at(s, i + l);
                    let states: u16 = if self.ca { vs_start(self.extra_tm(x, i, e).unwrap()) } else { 1 };
                    let a = i + l;
                    if a == n {
                        continue; // (no core arc leaves the end)
                    }
                    self.pair_core_arcs(x, a, None, ex, cm);
                    for vs in 0..if self.ca { NVS } else { 1 } {
                        if states >> vs & 1 == 0 {
                            continue;
                        }
                        let pe = PEnt { pid: EXTRA, vs: vs as u8, s, from: i as u32 };
                        for &arc in cm.iter() {
                            let Some((v, vs2, c)) = self.pair_ce(&pe, arc, ex, || self.pair_pb(EXTRA, arc.1, ex)) else { continue };
                            let k = arc.0;
                            let c = at(c, k);
                            match sc.ce[k].iter().find(|o| o.cid == arc.1 && o.vs == vs2) {
                                Some(o) => {
                                    if better(c, o.s) {
                                        return true;
                                    }
                                }
                                None => {
                                    let ce = CEnt { cid: arc.1, v, vs: vs2, s: c, fpos: 0, fidx: 0 };
                                    if private_ce(&ce, k, sm) {
                                        return true;
                                    }
                                }
                            }
                        }
                    }
                }
                0 => {
                    // e's core-done entries (new keys) from every plain prefix-done entry at i
                    self.pair_core_arcs(x, i, None, ex, cm);
                    let Some(&arc) = cm.iter().find(|a| a.1 == EXTRA) else { continue };
                    for pe in &sc.pe[i] {
                        let Some((v, vs2, c)) = self.pair_ce(pe, arc, ex, || self.pair_pb(pe.pid, EXTRA, ex)) else { continue };
                        let ce = CEnt { cid: EXTRA, v, vs: vs2, s: at(c, arc.0), fpos: 0, fidx: 0 };
                        if private_ce(&ce, arc.0, sm) {
                            return true;
                        }
                    }
                }
                _ => {
                    // e's suffix arc from every plain core-done entry at i
                    self.pair_suffix_arcs(x, i, None, ex, sm);
                    let Some(&arc) = sm.iter().find(|a| a.1 == EXTRA) else { continue };
                    for ce in &sc.ce[i] {
                        if self.pair_bs(ce, arc, ex).is_some_and(|c| better(at(c, arc.0), sc.bs[arc.0])) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// the text of a token (prefix, written core, suffix) into x
    fn token_text_into(&self, t: &Tok, x: &mut Vec<u32>) {
        x.clear();
        if self.ca {
            let part = |k: usize, r: u32| (&self.tabs[k].strs[r as usize][..], self.tabs[k].umask[r as usize]);
            self.write_parts(&[part(1, t.p), part(0, t.c), part(2, t.s)], true, t.v, x);
            return;
        }
        x.extend_from_slice(&self.tabs[1].strs[t.p as usize]);
        self.write_core(t.c as usize, t.v, x);
        x.extend_from_slice(&self.tabs[2].strs[t.s as usize]);
    }
}

// ------------------------------------------------------------------ corpus passes

struct Corpus {
    segs: Vec<Vec<u32>>,
    freqs: Vec<f64>,
}

/// Frequency-weighted sample (sample_t = t): every segment with count >= t is kept as it is; one
/// with count f < t is kept with probability f / t and then counts t (inverse inclusion
/// probability), so every sum over the corpus keeps its expectation while the long tail of rare
/// segments, which dominates the parse work, shrinks by about t. Inclusion is a fixed hash of the
/// segment's text, so it does not depend on segment order.
fn sample_corpus(c: &Corpus, t: f64) -> Corpus {
    let (mut segs, mut freqs) = (Vec::new(), Vec::new());
    for (s, &f) in c.segs.iter().zip(&c.freqs) {
        if f >= t {
            segs.push(s.clone());
            freqs.push(f);
        } else if ((seq_hash(s.iter().copied()) >> 11) as f64 / (1u64 << 53) as f64) < f / t {
            segs.push(s.clone());
            freqs.push(t);
        }
    }
    Corpus { segs, freqs }
}

struct ParseStats {
    tokens: f64,
    cost: f64,
    seg: Vec<f64>, // primary cost of every segment (only when types are collected)
    use_: [Vec<f64>; 3],
    v: [f64; NVAR_MAX],
    types: HashMap<Tok, f64, Fast>,
    bytes: f64, // byte-fallback tokens (one per UTF-8 byte)
}

/// per block of parse_corpus: primary cost of every segment, token ends, tokens
type Parsed = (Vec<f64>, Vec<u32>, Vec<Tok>);
/// parse_corpus's block size (segments)
const PC_BLOCK: usize = 512;

fn parse_corpus(d: &Dict, c: &Corpus, threads: usize, want_types: bool) -> ParseStats {
    parse_corpus_k(d, c, threads, want_types, false).0
}

/// parse_corpus, also returning every block's tokens if `keep` (prune_reuse)
fn parse_corpus_k(d: &Dict, c: &Corpus, threads: usize, want_types: bool, keep: bool) -> (ParseStats, Option<Vec<Parsed>>) {
    // The statistics are those of `threads` fixed ranges of segments, each accumulated in order and
    // then merged in range order (sums and the types map, whose iteration order later steps
    // depend on, come out bit-identical to one thread per range). The parsing itself runs on small
    // blocks handed out on demand (segment costs vary a lot); a range's statistics are built as
    // soon as all its blocks are parsed, and merged by this thread while the others still parse.
    let n = c.segs.len();
    let threads = threads.max(1);
    const B: usize = PC_BLOCK;
    let nb = n.div_ceil(B);
    let chunk = n.div_ceil(threads).max(1);
    let range = |t: usize| (t * chunk).min(n)..((t + 1) * chunk).min(n);
    let blocks_of = |t: usize| {
        let r = range(t);
        if r.is_empty() { 0..0 } else { r.start / B..r.end.div_ceil(B) }
    };
    let slots: Vec<std::sync::OnceLock<Parsed>> = (0..nb).map(|_| std::sync::OnceLock::new()).collect();
    let new_st = || ParseStats {
        tokens: 0.0,
        cost: 0.0,
        seg: Vec::new(),
        use_: [0, 1, 2].map(|k| vec![0.0; d.tabs[k].strs.len()]),
        v: [0.0; NVAR_MAX],
        types: HashMap::default(),
        bytes: 0.0,
    };
    // a range's statistics over its segments lo..hi (fed in segment order)
    let feed = |st: &mut ParseStats, lo: usize, hi: usize| {
        for i in lo..hi {
            let f = c.freqs[i];
            let (ks, ends, all) = slots[i / B].get().unwrap();
            let j = i % B;
            let k = ks[j];
            let toks = &all[if j == 0 { 0 } else { ends[j - 1] as usize }..ends[j] as usize];
            let real: u32 = toks.iter().map(|t| if t.c == NONE { t.v } else { 1 }).sum();
            st.cost += f * k;
            st.tokens += f * real as f64;
            if want_types {
                st.seg.push(k);
            }
            for tk in toks {
                if tk.c == NONE {
                    st.bytes += f * tk.v as f64;
                    continue;
                }
                st.use_[0][tk.c as usize] += f;
                st.use_[1][tk.p as usize] += f;
                st.use_[2][tk.s as usize] += f;
                st.v[tk.v as usize] += f;
                if want_types {
                    *st.types.entry(*tk).or_insert(0.0) += f;
                }
            }
        }
    };
    let build = |t: usize| {
        let mut st = new_st();
        let r = range(t);
        feed(&mut st, r.start, r.end);
        st
    };
    // Every range's statistics are built block by block as its blocks come in (in block order,
    // so exactly as build would): a thread that parsed a block feeds it, and any later blocks
    // already parsed, to its ranges unless another thread is feeding that range right now (that
    // one then sees the block when it re-checks after letting go). The range's last block sends
    // it on.
    let states: Vec<std::sync::Mutex<(usize, Option<ParseStats>)>> =
        (0..threads).map(|t| std::sync::Mutex::new((blocks_of(t).start, (!blocks_of(t).is_empty()).then(new_st)))).collect();
    let advance = |t: usize, tx: &std::sync::mpsc::Sender<(usize, ParseStats)>| {
        let (bl, r) = (blocks_of(t), range(t));
        loop {
            let Ok(mut g) = states[t].try_lock() else { return };
            let mut b = g.0;
            while b < bl.end && slots[b].get().is_some() {
                let st = g.1.as_mut().unwrap();
                feed(st, (b * B).max(r.start), ((b + 1) * B).min(r.end));
                b += 1;
            }
            g.0 = b;
            if b == bl.end {
                if let Some(st) = g.1.take() {
                    tx.send((t, st)).unwrap();
                }
                return;
            }
            drop(g);
            // (fences on both sides: a block set while we held the lock is seen here, or its
            // thread's try_lock gets the lock)
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            if slots[b].get().is_none() {
                return;
            }
        }
    };
    let tq = Instant::now();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let parsed = std::sync::atomic::AtomicUsize::new(0);
    let t_last: std::sync::Mutex<Option<Instant>> = std::sync::Mutex::new(None);
    let (tx, rx) = std::sync::mpsc::channel::<(usize, ParseStats)>();
    let st = std::thread::scope(|s| {
        for _ in 0..threads {
            let (tx, next, slots, advance, blocks_of) = (tx.clone(), &next, &slots, &advance, &blocks_of);
            let (parsed, t_last) = (&parsed, &t_last);
            s.spawn(move || {
                let (mut sc, mut toks) = (Scratch::new(), Vec::new());
                loop {
                    let b = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if b >= nb {
                        break;
                    }
                    let r = b * B..((b + 1) * B).min(n);
                    let (mut ks, mut ends, mut all) = (Vec::with_capacity(r.len()), Vec::with_capacity(r.len()), Vec::new());
                    for i in r {
                        ks.push(d.parse(&c.segs[i], &mut sc, Some(&mut toks), None).0);
                        all.extend_from_slice(&toks);
                        ends.push(all.len() as u32);
                    }
                    let _ = slots[b].set((ks, ends, all));
                    if parsed.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1 == nb {
                        *t_last.lock().unwrap() = Some(Instant::now());
                    }
                    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
                    // the ranges this block belongs to
                    let t0 = b * B / chunk;
                    for t in t0..threads {
                        if !blocks_of(t).contains(&b) {
                            break;
                        }
                        advance(t, &tx);
                    }
                }
            });
        }
        drop(tx);
        // ranges without segments are complete from the start
        let mut ready: Vec<Option<ParseStats>> = (0..threads).map(|t| if blocks_of(t).is_empty() { Some(build(t)) } else { None }).collect();
        let mut acc: Option<ParseStats> = None;
        let mut tm = 0.0;
        for t in 0..threads {
            while ready[t].is_none() {
                let (u, st) = rx.recv().unwrap();
                ready[u] = Some(st);
            }
            let tw = Instant::now();
            let p = ready[t].take().unwrap();
            match acc.as_mut() {
                None => acc = Some(p),
                Some(a) => merge_stats(a, p),
            }
            tm += tw.elapsed().as_secs_f64();
        }
        prof_add("pc_merge", tm);
        if let Some(tl) = *t_last.lock().unwrap() {
            prof("pc_tail", tl);
            prof_add("pc_body", tl.duration_since(tq).as_secs_f64());
        }
        prof("pc_all", tq);
        acc.unwrap()
    });
    let kept = keep.then(|| slots.into_iter().map(|b| b.into_inner().unwrap()).collect());
    (st, kept)
}

/// add a later part's statistics (in part order: the types map depends on the insertion order)
fn merge_stats(acc: &mut ParseStats, p: ParseStats) {
    acc.tokens += p.tokens;
    acc.cost += p.cost;
    acc.seg.extend(p.seg);
    for k in 0..3 {
        for (a, b) in acc.use_[k].iter_mut().zip(&p.use_[k]) {
            *a += b;
        }
    }
    for k in 0..NVAR_MAX {
        acc.v[k] += p.v[k];
    }
    acc.bytes += p.bytes;
    for (t, n) in p.types {
        *acc.types.entry(t).or_insert(0.0) += n;
    }
}

/// q = (n + 1/2) / (sum n + |D|/2), secondary cost = -log2 q (bits)
/// (code_len: the cores' distribution also has the BYTE pseudo-core, n = byte tokens)
fn update_costs(d: &mut Dict, st: &ParseStats) {
    for k in 0..3 {
        let live = d.tabs[k].live() as f64;
        let tot: f64 = (0..d.tabs[k].strs.len()).filter(|&i| d.tabs[k].alive[i]).map(|i| st.use_[k][i]).sum();
        if k == 0 && (d.cl.on || d.cl.wu > 0.0) {
            let den = tot + st.bytes + (live + 1.0) / 2.0;
            for i in 0..d.tabs[k].strs.len() {
                d.tabs[k].cost[i] = -((st.use_[k][i] + 0.5) / den).log2();
            }
            d.cl.byte = 8.0 - ((st.bytes + 0.5) / den).log2();
            continue;
        }
        for i in 0..d.tabs[k].strs.len() {
            d.tabs[k].cost[i] = -((st.use_[k][i] + 0.5) / (tot + live / 2.0)).log2();
        }
    }
    let tot: f64 = st.v.iter().sum(); // (unused variations count 0)
    let nv = d.nvar();
    for k in 0..nv {
        d.vcost[k] = -((st.v[k] + 0.5) / (tot + nv as f64 / 2.0)).log2();
    }
    d.refresh_arcs();
}

/// Partner prices (partner_n0 > 0): affix price = pi x L0 x (1 + k / E_w), E_w from the parse.
/// Signature prices (update_signature_prices, sig_k > 0) then multiply these for affixes.
/// Both come out exactly as computed one after the other: the partner prices of prefixes and
/// suffixes (one thread each, every sum in the same order) and the signature counts (which do not
/// depend on any price) are computed at the same time.
fn update_prices(d: &mut Dict, st: &ParseStats) {
    let partner = d.pricing.n0 != 0.0 && d.pricing.pi != 0.0;
    let sig = d.pricing.sig_k != 0.0 && d.pricing.pi_core != 0.0;
    let dd: &Dict = d;
    let (prices, maps) = std::thread::scope(|sc| {
        let hs: Vec<_> = if partner { (1..3).map(|k| sc.spawn(move || partner_prices(dd, st, k))).collect() } else { Vec::new() };
        let maps = sig.then(|| sig_maps(dd, st));
        (hs.into_iter().map(|h| h.join().unwrap()).collect::<Vec<Vec<f64>>>(), maps)
    });
    for (k, p) in (1..3).zip(prices) {
        for a in 1..p.len() {
            d.tabs[k].price[a] = p[a];
        }
    }
    if let Some(m) = maps {
        update_signature_prices(d, st, m);
    }
}

/// the partner price of every row of affix table k (update_prices; row 0: unused)
fn partner_prices(d: &Dict, st: &ParseStats, k: usize) -> Vec<f64> {
    let n0 = d.pricing.n0;
    let mut uc = vec![0.0f64; d.tabs[0].strs.len()];
    for (t, &n) in &st.types {
        if t.c != NONE {
            uc[t.c as usize] += n;
        }
    }
    {
        let mut pair: HashMap<(u32, u32), f64, Fast> = HashMap::default();
        for (t, &n) in &st.types {
            let a = if k == 1 { t.p } else { t.s };
            if t.c != NONE && a != 0 {
                *pair.entry((a, t.c)).or_insert(0.0) += n;
            }
        }
        let len = d.tabs[k].strs.len();
        let (mut tot, mut wtot) = (vec![0.0f64; len], vec![0.0f64; len]);
        for (&(a, c), &n) in &pair {
            let w = uc[c as usize] / (uc[c as usize] + n0);
            tot[a as usize] += n;
            wtot[a as usize] += n * w;
        }
        let mut h = vec![0.0f64; len];
        for (&(a, c), &n) in &pair {
            let w = uc[c as usize] / (uc[c as usize] + n0);
            let q = n * w / wtot[a as usize];
            if q > 0.0 {
                h[a as usize] -= q * q.log2();
            }
        }
        let mut out = vec![0.0f64; len];
        for a in 1..len {
            let e = if tot[a] > 0.0 { h[a].exp2() * wtot[a] / tot[a] } else { 0.0 };
            let l0 = d.l0_of(&d.stat_str(k, &d.tabs[k].strs[a]));
            out[a] = d.pricing.pi * l0 * (1.0 + d.pricing.k / e.max(1e-3));
        }
        out
    }
}

type SigMaps = ([Vec<u32>; 2], Vec<f64>, [HashMap<(u32, u32), f64, Fast>; 2]);

/// update_signature_prices' counts: the letter piece of every affix row, uses per core, and
/// (core, piece) -> uses per side
fn sig_maps(d: &Dict, st: &ParseStats) -> SigMaps {
    let an = |c: u32| !is_byte(c) && d.sym.alnum[c as usize] != 0;
    // letter piece of every affix row: the letter run at its inner end (0 = none)
    let mut piece_ids: HashMap<Vec<u32>, u32, Fast> = HashMap::default();
    let mut piece: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
    for k in 1..3 {
        piece[k - 1] = d.tabs[k]
            .strs
            .iter()
            .map(|a| {
                let run: Vec<u32> = if k == 1 {
                    let n = a.iter().rev().take_while(|&&c| an(c)).count();
                    a[a.len() - n..].to_vec()
                } else {
                    a.iter().copied().take_while(|&c| an(c)).collect()
                };
                if run.is_empty() {
                    0
                } else {
                    let n = piece_ids.len() as u32 + 1;
                    *piece_ids.entry(run).or_insert(n)
                }
            })
            .collect();
    }
    let nc = d.tabs[0].strs.len();
    // (core, piece) -> uses per side, and uses per core: the two sides' maps are built on two
    // threads, each from the types in the same order as one loop would
    let (uses, by) = {
        let (piece, types) = (&piece, &st.types);
        std::thread::scope(|sc| {
            let h1 = sc.spawn(move || {
                let mut by1: HashMap<(u32, u32), f64, Fast> = HashMap::default();
                for (t, &n) in types {
                    if t.c != NONE {
                        *by1.entry((t.c, piece[1][t.s as usize])).or_insert(0.0) += n;
                    }
                }
                by1
            });
            let mut uses = vec![0.0f64; nc];
            let mut by0: HashMap<(u32, u32), f64, Fast> = HashMap::default();
            for (t, &n) in types {
                if t.c == NONE {
                    continue;
                }
                uses[t.c as usize] += n;
                *by0.entry((t.c, piece[0][t.p as usize])).or_insert(0.0) += n;
            }
            (uses, [by0, h1.join().unwrap()])
        })
    };
    (piece, uses, by)
}

/// Paradigm (signature) prices of the cores (sig_k > 0): see the header. (Counts from sig_maps.)
fn update_signature_prices(d: &mut Dict, st: &ParseStats, (piece, uses, by): SigMaps) {
    let nc = d.tabs[0].strs.len();
    let mut mult = vec![1.0f64; nc];
    // per side, 1 / (paradigm size) of every core whose signature has a letter piece (0 otherwise):
    // an affix pays by the average of it over its uses (see below)
    let mut inv_shared: [Vec<f64>; 2] = [vec![0.0; nc], vec![0.0; nc]];
    if d.pricing.sig_soft {
        // SOFT: per ending, not per exact set. N(p) = cores (>= 20 uses) taking piece p with at
        // least sig_share of their uses; a core pays 1 + sig_k x sum_p share(c, p) / N(p) per side.
        // Erfolg {e, s, t, reich, Miss} is a rare exact set but every ending is shared, so it pays
        // about its spelling; Broadc {ast} still pays 1 + sig_k / N(ast).
        // (the sides on two threads; their factors are applied in side order)
        let softs: Vec<Vec<f64>> = std::thread::scope(|sc| {
            let dd: &Dict = d;
            let hs: Vec<_> = (0..2).map(|side| {
                let (by, uses, d) = (&by, &uses, dd);
                sc.spawn(move || {
                    let mut n_of: HashMap<u32, f64, Fast> = HashMap::default();
                    for (&(c, pc), &n) in &by[side] {
                        if pc != 0 && uses[c as usize] >= 20.0 && n >= d.pricing.sig_share * uses[c as usize] {
                            *n_of.entry(pc).or_insert(0.0) += 1.0;
                        }
                    }
                    let mut soft = vec![0.0f64; nc];
                    let mut top = vec![0.0f64; nc]; // share of the core's most frequent letter piece
                    let mut bound = vec![0.0f64; nc]; // share of its uses with any letter piece on this side
                    for (&(c, pc), &n) in &by[side] {
                        if pc != 0 && uses[c as usize] > 0.0 {
                            bound[c as usize] += n / uses[c as usize];
                        }
                        if pc != 0 && uses[c as usize] > 0.0 && n >= d.pricing.sig_share * uses[c as usize] {
                            let sh = n / uses[c as usize];
                            soft[c as usize] += sh / n_of.get(&pc).copied().unwrap_or(0.0).max(1.0);
                            top[c as usize] = top[c as usize].max(sh);
                        }
                    }
                    // LEFT-BOUND: on the prefix side a core should be a free stem. One that nearly always
                    // follows letters (uch in m:uch / s:uch, heir in t:heir, rès in t:rès) is a word's
                    // tail, however many different letters come before it. (Not on the suffix side: bound
                    // stems + endings, कर:ते or mach:en, are ordinary morphology.)
                    if side == 0 {
                        for c in 0..nc {
                            top[c] = top[c].max(bound[c]);
                        }
                    }
                    // CONCENTRATION: a core that nearly always takes the same letter piece is really core +
                    // piece (migh + t, olar after s): 0 up to a 50% share, rising to 1 at 100%. Common
                    // endings alone do not make such a core look regular.
                    for c in 0..nc {
                        soft[c] += (2.0 * top[c] - 1.0).max(0.0);
                    }
                    soft
                })
            }).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (side, soft) in softs.into_iter().enumerate() {
            for c in 0..nc {
                mult[c] *= 1.0 + d.pricing.sig_k * soft[c];
            }
            inv_shared[side] = soft;
        }
    }
    for side in 0..2 {
        if d.pricing.sig_soft {
            break;
        }
        let mut sig: Vec<Vec<u32>> = vec![Vec::new(); nc];
        for (&(c, pc), &n) in &by[side] {
            if n >= d.pricing.sig_share * uses[c as usize] {
                sig[c as usize].push(pc);
            }
        }
        for g in sig.iter_mut() {
            g.sort_unstable();
        }
        let mut count: HashMap<&[u32], f64, Fast> = HashMap::default();
        for (c, g) in sig.iter().enumerate() {
            if uses[c] >= 20.0 && g.iter().any(|&p| p != 0) {
                *count.entry(&g[..]).or_insert(0.0) += 1.0;
            }
        }
        for c in 0..nc {
            let g = &sig[c];
            if g.iter().any(|&p| p != 0) {
                let shared = count.get(&g[..]).copied().unwrap_or(0.0).max(1.0);
                mult[c] *= 1.0 + d.pricing.sig_k / shared;
                inv_shared[side][c] = 1.0 / shared;
            }
        }
    }
    for c in 0..nc {
        let base = d.price_of(0, &d.tabs[0].strs[c]);
        d.tabs[0].price[c] = base * mult[c];
    }
    // the affix side of the same test: an affix carrying letters is part of the paradigms of the
    // cores it attaches to. s, -ing, ा attach to stems of paradigms shared by thousands and pay about
    // their spelling; a word piece (Broadc|ast, パッケ|ージ) attaches to one-off stems and pays up to
    // 1 + sig_k times more, so moving a fragment from the core into the affix no longer dodges the
    // price. (Multiplies the partner price set by update_prices this pass.)
    // (per affix table: its sums on its own thread, same order)
    let sums: Vec<(Vec<f64>, Vec<f64>)> = {
        let dd: &Dict = d;
        let (piece, inv_shared) = (&piece, &inv_shared);
        std::thread::scope(|sc| {
            let hs: Vec<_> = (1..3)
                .map(|k| {
                    sc.spawn(move || {
                        let side = k - 1;
                        let len = dd.tabs[k].strs.len();
                        let (mut tot, mut acc) = (vec![0.0f64; len], vec![0.0f64; len]);
                        for (t, &n) in &st.types {
                            let a = if k == 1 { t.p } else { t.s } as usize;
                            if t.c == NONE || a == 0 || piece[side][a] == 0 {
                                continue;
                            }
                            tot[a] += n;
                            acc[a] += n * inv_shared[side][t.c as usize];
                        }
                        (tot, acc)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        })
    };
    for (k, (tot, acc)) in (1..3).zip(sums) {
        let len = d.tabs[k].strs.len();
        for a in 1..len {
            if tot[a] > 0.0 {
                // start from this pass's partner price, or the fixed price if that is off (so the
                // factor never compounds across passes)
                let base = if d.pricing.n0 > 0.0 { d.tabs[k].price[a] } else { d.price_of(k, &d.tabs[k].strs[a]) };
                d.tabs[k].price[a] = base * (1.0 + d.pricing.sig_k * acc[a] / tot[a]);
            }
        }
    }
}

/// code_len: the pair statistics of the parse (hard EM; see the header), after update_costs
/// (which sets the marginals they back off to). Every value depends only on its own counts, so
/// the maps' order does not matter.
fn update_codelen(d: &mut Dict, st: &ParseStats) {
    if !d.cl.on {
        return;
    }
    let nc = d.tabs[0].strs.len();
    let mut n = vec![0.0f64; nc];
    let mut cnt: [HashMap<(u32, u32), f64, Fast>; 3] = Default::default();
    for (t, &f) in &st.types {
        if t.c == NONE {
            continue;
        }
        n[t.c as usize] += f;
        *cnt[0].entry((t.p, t.c)).or_insert(0.0) += f;
        *cnt[1].entry((t.c, t.s)).or_insert(0.0) += f;
        *cnt[2].entry((t.c, t.v)).or_insert(0.0) += f;
    }
    // beta per core and side: fixed, or Witten-Bell (distinct partners, at least 1)
    let mut beta = vec![[1.0f64; 3]; nc];
    if d.cl.beta > 0.0 {
        beta.iter_mut().for_each(|b| *b = [d.cl.beta; 3]);
    } else {
        let mut tt = vec![[0u32; 3]; nc];
        for k in 0..3 {
            for &(a, b) in cnt[k].keys() {
                tt[if k == 0 { b } else { a } as usize][k] += 1;
            }
        }
        for (b, t) in beta.iter_mut().zip(&tt) {
            for k in 0..3 {
                b[k] = (t[k] as f64).max(1.0);
            }
        }
    }
    d.cl.bo = (0..nc).map(|c| std::array::from_fn(|k| cl_q(((n[c] + beta[c][k]) / beta[c][k]).log2()))).collect();
    for k in 0..3 {
        let marg = |a: u32| -> f64 {
            match k {
                0 => d.tabs[1].cost[a as usize],
                1 => d.tabs[2].cost[a as usize],
                _ => d.vcost[a as usize],
            }
        };
        d.cl.pair[k] = cnt[k]
            .iter()
            .filter(|&(_, &f)| f >= CL_MIN_PAIR)
            .map(|(&(a, b), &f)| {
                let (c, x) = if k == 0 { (b, a) } else { (a, b) };
                let bt = beta[c as usize][k];
                ((a, b), cl_q(-((f + bt * (-marg(x)).exp2()) / (n[c as usize] + bt)).log2()))
            })
            .collect();
    }
    d.cl.vb = codelen_vb(&d.cl.bo, &d.cl.pair[2], &d.vcost, d.nvar());
    d.cl.index();
}

fn update_cond(d: &mut Dict, st: &ParseStats) {
    if d.lamc == 0.0 || st.types.is_empty() {
        return;
    }
    let mut nc = vec![0.0f32; d.tabs[0].strs.len()];
    let mut cn: [HashMap<(u32, u32), f32, Fast>; 2] = Default::default();
    for (t, &f) in &st.types {
        if t.c == NONE {
            continue;
        }
        nc[t.c as usize] += f as f32;
        *cn[0].entry((t.p, t.c)).or_insert(0.0) += f as f32;
        *cn[1].entry((t.c, t.s)).or_insert(0.0) += f as f32;
    }
    d.cn = cn;
    d.ncore = nc;
}

/// Excess PMI (bits above tau) of every (prefix, core) and (core, suffix) pair in the parse.
fn update_pmi(d: &mut Dict, st: &ParseStats) {
    if d.mu2 == 0.0 || st.types.is_empty() {
        return;
    }
    let mut nc = vec![0.0f64; d.tabs[0].strs.len()];
    let mut np = vec![0.0f64; d.tabs[1].strs.len()];
    let mut ns = vec![0.0f64; d.tabs[2].strs.len()];
    let mut npc: HashMap<(u32, u32), f64, Fast> = HashMap::default();
    let mut ncs: HashMap<(u32, u32), f64, Fast> = HashMap::default();
    let mut n = 0.0;
    for (t, &f) in &st.types {
        if t.c == NONE {
            continue;
        }
        n += f;
        nc[t.c as usize] += f;
        if t.p != 0 {
            np[t.p as usize] += f;
            *npc.entry((t.p, t.c)).or_insert(0.0) += f;
        }
        if t.s != 0 {
            ns[t.s as usize] += f;
            *ncs.entry((t.c, t.s)).or_insert(0.0) += f;
        }
    }
    let tau = d.tau;
    d.pmi[0] = npc
        .into_iter()
        .filter_map(|((p, c), f)| {
            let v = (f * n / (np[p as usize] * nc[c as usize])).log2() - tau;
            (v > 0.0).then_some(((p, c), v as f32))
        })
        .collect();
    d.pmi[1] = ncs
        .into_iter()
        .filter_map(|((c, s), f)| {
            let v = (f * n / (nc[c as usize] * ns[s as usize])).log2() - tau;
            (v > 0.0).then_some(((c, s), v as f32))
        })
        .collect();
}

/// Affix productivity deficit: spec(a) = max(0, h0 - H(core | a)) in bits, from the parse's token
/// types. Affixes without uses get 0 (they are unmeasured, and cost nothing until used).
fn update_spec(d: &mut Dict, st: &ParseStats) {
    if d.mu == 0.0 || st.types.is_empty() {
        return;
    }
    for k in 1..3 {
        let mut pair: HashMap<(u32, u32), f64, Fast> = HashMap::default();
        for (t, &n) in &st.types {
            let a = if k == 1 { t.p } else { t.s };
            *pair.entry((a, t.c)).or_insert(0.0) += n;
        }
        let mut na = vec![0.0f64; d.tabs[k].strs.len()];
        for (&(a, _), &n) in &pair {
            na[a as usize] += n;
        }
        let mut h = vec![0.0f64; d.tabs[k].strs.len()];
        for (&(a, _), &n) in &pair {
            let q = n / na[a as usize];
            h[a as usize] -= q * q.log2();
        }
        for (i, v) in h.into_iter().enumerate() {
            let word = d.tabs[k].strs[i].iter().any(|&c| !is_byte(c) && d.sym.alnum[c as usize] != 0);
            d.tabs[k].spec[i] = if na[i] > 0.0 && (word || !d.alnum_only) { (d.h0 - v).max(0.0) } else { 0.0 };
        }
    }
    d.refresh_arcs();
}

/// description length R (bits): table spelling + smoothed usage code lengths
fn description_length(d: &Dict, st: &ParseStats, char_bits: &[f64], gamma: f64) -> f64 {
    let mut r = 0.0;
    for k in 0..3 {
        let t = &d.tabs[k];
        for i in 0..t.strs.len() {
            if !t.alive[i] || t.strs[i].is_empty() {
                continue;
            }
            let s = &t.strs[i];
            let l0 = 2.0 * ((s.len() + 1) as f64).log2() + s.iter().map(|&c| char_bits[c as usize]).sum::<f64>();
            r += if k == 0 { l0 } else { (1.0 + gamma) * l0 };
            r += (st.use_[k][i] + 0.5) * t.cost[i];
        }
    }
    r
}

// ------------------------------------------------------------------ pruning

/// Lists of ids per key, stored flat: items[start[i]..start[i + 1]] for key i, in input order.
struct Csr {
    start: Vec<u32>,
    items: Vec<u32>,
}

impl Csr {
    /// from the key (or None) of every id 0, 1, 2, ...
    fn new(n: usize, keys: impl Iterator<Item = Option<u32>> + Clone) -> Csr {
        let mut start = vec![0u32; n + 1];
        for k in keys.clone().flatten() {
            start[k as usize + 1] += 1;
        }
        for i in 0..n {
            start[i + 1] += start[i];
        }
        let mut fill = start.clone();
        let mut items = vec![0u32; start[n] as usize];
        for (id, k) in keys.enumerate() {
            if let Some(k) = k {
                items[fill[k as usize] as usize] = id as u32;
                fill[k as usize] += 1;
            }
        }
        Csr { start, items }
    }
    fn get(&self, i: usize) -> &[u32] {
        &self.items[self.start[i] as usize..self.start[i + 1] as usize]
    }
}

/// prune_reuse: the parse in `blocks` (with its statistics st) after the rows flagged in `gone`
/// were dropped: every segment whose tokens use a dropped row is parsed again (it has to change),
/// every other one keeps its tokens, and st is updated by the difference (a token a segment keeps
/// cancels out; in block and segment order; a type whose count falls to zero is removed; counts
/// are sums of whole segment frequencies, so they stay exact). Returns the number of segments
/// parsed.
fn reparse_dirty(d: &Dict, c: &Corpus, threads: usize, blocks: &mut [Parsed], gone: &[Vec<bool>; 3], st: &mut ParseStats) -> usize {
    let is_gone = |t: &Tok| t.c != NONE && (gone[0][t.c as usize] || gone[1][t.p as usize] || gone[2][t.s as usize]);
    let seg_of = |p: &Parsed, j: usize| -> std::ops::Range<usize> { (if j == 0 { 0 } else { p.1[j - 1] as usize })..p.1[j] as usize };
    let bl: &[Parsed] = blocks;
    let tq = Instant::now();
    // per block: the new block, the number of segments parsed again, the changed tokens (count
    // differences) and the differences of cost, real tokens and byte tokens
    type Redo = (Parsed, usize, Vec<(Tok, f64)>, [f64; 3]);
    let res: Vec<Option<Redo>> = par_map(bl.len(), threads, 1, || (Scratch::new(), Vec::new(), Vec::new()), |(sc, toks, old), b| {
        let p = &bl[b];
        let dirty: Vec<usize> = (0..p.0.len()).filter(|&j| p.2[seg_of(p, j)].iter().any(is_gone)).collect();
        if dirty.is_empty() {
            return None;
        }
        let lo = b * PC_BLOCK;
        let (mut ks, mut ends, mut all) = (Vec::with_capacity(p.0.len()), Vec::with_capacity(p.0.len()), Vec::with_capacity(p.2.len()));
        let (mut delta, mut sums) = (Vec::new(), [0.0f64; 3]);
        let mut q = 0;
        for j in 0..p.0.len() {
            if q < dirty.len() && dirty[q] == j {
                q += 1;
                let f = c.freqs[lo + j];
                let k = d.parse(&c.segs[lo + j], sc, Some(toks), None).0;
                ks.push(k);
                all.extend_from_slice(toks);
                sums[0] += f * (k - p.0[j]);
                old.clear();
                old.extend_from_slice(&p.2[seg_of(p, j)]);
                let real = |ts: &[Tok]| ts.iter().map(|t| if t.c == NONE { t.v } else { 1 }).sum::<u32>() as f64;
                sums[1] += f * (real(toks) - real(old));
                for (s, ts) in [(-f, &old[..]), (f, &toks[..])] {
                    for t in ts.iter().filter(|t| t.c == NONE) {
                        sums[2] += s * t.v as f64;
                    }
                }
                // tokens the segment keeps cancel out
                for t in toks.iter().filter(|t| t.c != NONE) {
                    match old.iter().position(|o| o == t) {
                        Some(i) => {
                            old.swap_remove(i);
                        }
                        None => delta.push((*t, f)),
                    }
                }
                delta.extend(old.iter().filter(|t| t.c != NONE).map(|t| (*t, -f)));
            } else {
                ks.push(p.0[j]);
                all.extend_from_slice(&p.2[seg_of(p, j)]);
            }
            ends.push(all.len() as u32);
        }
        Some(((ks, ends, all), dirty.len(), delta, sums))
    });
    prof("rd_parse", tq);
    let tq = Instant::now();
    let mut n = 0;
    for (b, r) in res.into_iter().enumerate() {
        let Some((newp, nd, delta, sums)) = r else { continue };
        blocks[b] = newp;
        n += nd;
        st.cost += sums[0];
        st.tokens += sums[1];
        st.bytes += sums[2];
        for (t, s) in delta {
            st.use_[0][t.c as usize] += s;
            st.use_[1][t.p as usize] += s;
            st.use_[2][t.s as usize] += s;
            st.v[t.v as usize] += s;
            let e = st.types.entry(t).or_insert(0.0);
            *e += s;
            if e.abs() < 1e-6 {
                st.types.remove(&t);
            }
        }
    }
    prof("rd_apply", tq);
    n
}

/// Returns true if prune_reuse is on and the dictionary and costs are those the last pass's
/// (re-used) parse was re-estimated with: an EM step on the re-used parse would change nothing.
fn prune_to(d: &mut Dict, c: &Corpus, budget: usize, threads: usize, protect_chars: bool, step: f64, reuse: bool, tail: f64) -> bool {
    let mut pass = 0;
    let priced = d.pricing.pi > 0.0 || d.pricing.pi_core > 0.0 || d.pricing.conc > 0.0;
    // prune_reuse: the last pass's parse (statistics, tokens) and the rows it dropped
    let mut cache: Option<(ParseStats, Vec<Parsed>)> = None;
    let mut last_dropped: Vec<(usize, u32)> = Vec::new();
    while (d.rows() > budget || priced) && pass < 100 {
        pass += 1;
        let tq = Instant::now();
        let mut blocks: Option<Vec<Parsed>> = None;
        let mut reparsed = None;
        let st = match cache.take() {
            Some((mut st, mut bl)) => {
                let mut gone: [Vec<bool>; 3] = std::array::from_fn(|k| vec![false; d.tabs[k].strs.len()]);
                for &(k, i) in &last_dropped {
                    gone[k][i as usize] = true;
                }
                reparsed = Some(reparse_dirty(d, c, threads, &mut bl, &gone, &mut st));
                blocks = Some(bl);
                st
            }
            None if reuse => {
                let (st, bl) = parse_corpus_k(d, c, threads, true, true);
                blocks = bl;
                st
            }
            None => parse_corpus(d, c, threads, true),
        };
        prof("prune_parse", tq);
        let tq = Instant::now();
        update_costs(d, &st);
        update_spec(d, &st);
        update_pmi(d, &st);
        update_cond(d, &st);
        update_codelen(d, &st);
        update_prices(d, &st);
        prof("prune_update", tq);
        let tq = Instant::now();
        let types: Vec<(Tok, f64)> = st.types.iter().map(|(t, n)| (*t, *n)).collect();
        prof("pt_types", tq);
        let tq2 = Instant::now();
        // the types using each row (cores, non-empty prefixes, non-empty suffixes), in the types' order
        let row_of = |t: &Tok, k: usize| match k {
            0 => Some(t.c),
            1 => (t.p != 0).then_some(t.p),
            _ => (t.s != 0).then_some(t.s),
        };
        let dd: &Dict = d;
        let uses: [Csr; 3] = std::thread::scope(|s| {
            let hs: Vec<_> = (0..3)
                .map(|k| {
                    let (types, row_of) = (&types, &row_of);
                    s.spawn(move || Csr::new(dd.tabs[k].strs.len(), types.iter().map(|(t, _)| row_of(t, k))))
                })
                .collect();
            let mut v = hs.into_iter().map(|h| h.join().unwrap());
            [v.next().unwrap(), v.next().unwrap(), v.next().unwrap()]
        });
        prof("pt_csr", tq2);
        // loss term of type ti when row `ban` is gone: its count x (best parse of its text without
        // the row - what it pays as the single token it is now), texts built on the fly (only the
        // types actually reached are parsed)
        let own_of = |d: &Dict, t: &Tok| -> f64 {
            1.0 + (d.tabs[0].usec[t.c as usize] + d.tabs[1].usec[t.p as usize] + d.tabs[2].usec[t.s as usize]) + d.delta * ((t.p != 0) as u32 + (t.s != 0) as u32) as f64
                + d.lam * (d.tabs[1].cost[t.p as usize] + d.tabs[2].cost[t.s as usize])
                + d.mu * (d.tabs[1].spec[t.p as usize] + d.tabs[2].spec[t.s as usize])
                + if d.mu2 > 0.0 { d.mu2 * (d.pmi_of(0, t.p, t.c) + d.pmi_of(1, t.c, t.s)) } else { 0.0 }
                + if d.lamc > 0.0 {
                    d.lamc * (d.cond(0, t.p, t.c, (-d.tabs[1].cost[t.p as usize]).exp2())
                        + d.cond(1, t.c, t.s, (-d.tabs[2].cost[t.s as usize]).exp2()))
                } else {
                    0.0
                }
                + if d.cl.on { d.cl.w * d.cl_bits(t) } else { 0.0 }
                + if d.cl.wu > 0.0 {
                    // (code_w without code_len: the token's unconditional bits)
                    d.cl.wu * (d.tabs[0].cost[t.c as usize] + d.vcost[t.v as usize] + d.tabs[1].cost[t.p as usize] + d.tabs[2].cost[t.s as usize])
                } else {
                    0.0
                }
        };
        let term_of = |d: &Dict, (sc, text): &mut (Scratch, Vec<u32>), ban: (usize, u32), ti: u32| -> f64 {
            let (t, n) = &types[ti as usize];
            d.token_text_into(t, text);
            let alt = d.parse_prim(text, sc, Some(ban));
            n * (if alt.is_infinite() { 1e6 } else { (alt - own_of(d, t)).max(0.0) })
        };
        let new_sc = || (Scratch::new(), Vec::new());
        prof("prune_types_texts", tq);
        let tq = Instant::now();
        let mut cand: Vec<(usize, u32)> = Vec::new();
        for k in 0..3 {
            for i in 0..d.tabs[k].strs.len() {
                let single = k == 0 && d.tabs[0].strs[i].len() == 1;
                if d.tabs[k].alive[i] && !(k > 0 && i == 0) && !(protect_chars && single) {
                    cand.push((k, i as u32));
                }
            }
        }
        // affixes concentrated on one core (conc): treated as worthless, so they go first
        let mut conc_bad: [Vec<bool>; 2] = [vec![false; d.tabs[1].strs.len()], vec![false; d.tabs[2].strs.len()]];
        if d.pricing.conc > 0.0 {
            // per affix: its uses in total and with its most frequent core (sums of counts, so exact
            // in any order)
            for k in 1..3 {
                let conc = d.pricing.conc;
                let n_cores = d.tabs[0].strs.len();
                conc_bad[k - 1] = par_map(d.tabs[k].strs.len(), threads, 64, || (Vec::new(), Vec::new()), |(pc, dense): &mut (Vec<(u32, f64)>, Vec<f64>), a| {
                    let u = uses[k].get(a);
                    let (mut tot, mut mx) = (0.0f64, 0.0f64);
                    if u.len() > 4096 {
                        // a very common affix: per-core sums in a dense array, not a sort
                        dense.resize(n_cores, 0.0);
                        for &ti in u {
                            let (t, n) = &types[ti as usize];
                            dense[t.c as usize] += n;
                            tot += n;
                        }
                        for &ti in u {
                            let c = types[ti as usize].0.c as usize;
                            mx = mx.max(dense[c]);
                        }
                        for &ti in u {
                            dense[types[ti as usize].0.c as usize] = 0.0;
                        }
                    } else {
                        pc.clear();
                        pc.extend(u.iter().map(|&ti| (types[ti as usize].0.c, types[ti as usize].1)));
                        pc.sort_unstable_by_key(|e| e.0);
                        let mut run = 0.0f64;
                        for (j, &(c, n)) in pc.iter().enumerate() {
                            run = if j > 0 && pc[j - 1].0 == c { run + n } else { n };
                            mx = mx.max(run);
                            tot += n;
                        }
                    }
                    a > 0 && tot > 0.0 && mx >= conc * tot
                });
            }
        }
        prof("prune_cand_conc", tq);
        let tq = Instant::now();
        // Losses. Priced, rows are dropped in order of net value = loss - price, and only rows near
        // or below zero are normally reached; so a row's sum stops once loss > price + lazy, and the
        // stopped sums are only continued (with lazy x 3, in the state before this pass) if the
        // dropping needs rows beyond those known to be below lazy / 2. Rows used by more than G types
        // finish in a flat parallel pass, so no row holds up a thread. Every loss that is used is the
        // full sum of its terms in the types' order, i.e. exactly what the plain sum gives.
        const LAZY0: f64 = 8.0;
        const G: usize = 64;
        let prices: Vec<f64> = cand.iter().map(|&(k, i)| d.tabs[k].price[i as usize]).collect();
        let is_conc = |j: usize| cand[j].0 > 0 && conc_bad[cand[j].0 - 1][cand[j].1 as usize];
        // per row: (sum of its first `next` terms, next, complete)
        let mut acc: Vec<(f64, usize, bool)> = (0..cand.len()).map(|j| (0.0, 0, is_conc(j))).collect(); // conc: net -inf whatever the loss
        // continue the incomplete sums until loss > price + lazy or complete
        let advance = |d: &Dict, acc: &mut Vec<(f64, usize, bool)>, lazy: f64| {
            let open: Vec<usize> = (0..cand.len()).filter(|&j| !acc[j].2).collect();
            let step: Vec<(f64, usize, u8)> = par_map(open.len(), threads, 16, new_sc, |sc, q| {
                let j = open[q];
                let (k, i) = cand[j];
                let u = uses[k].get(i as usize);
                let limit = prices[j] + lazy;
                let (mut loss, mut next) = (acc[j].0, acc[j].1);
                for _ in 0..G {
                    if next == u.len() {
                        return (loss, next, 0); // complete
                    }
                    if loss > limit {
                        return (loss, next, 1); // stopped
                    }
                    loss += term_of(d, sc, (k, i), u[next]);
                    next += 1;
                }
                (loss, next, if next == u.len() { 0 } else if loss > limit { 1 } else { 2 })
            });
            // rows with more than G terms left under the limit: all their remaining terms, flat
            let mut off = vec![0usize];
            let mut pairs: Vec<(u32, u32)> = Vec::new();
            for (q, &j) in open.iter().enumerate() {
                if step[q].2 == 2 {
                    let (k, i) = cand[j];
                    pairs.extend(uses[k].get(i as usize)[step[q].1..].iter().map(|&ti| (j as u32, ti)));
                }
                off.push(pairs.len());
            }
            let t = par_map(pairs.len(), threads, 256, new_sc, |sc, q| term_of(d, sc, cand[pairs[q].0 as usize], pairs[q].1));
            for (q, &j) in open.iter().enumerate() {
                let (mut loss, next, state) = step[q];
                for x in &t[off[q]..off[q + 1]] {
                    loss += x;
                }
                acc[j] = (loss, next, state != 1);
            }
        };
        let mut lazy = if priced { LAZY0 } else { f64::INFINITY };
        advance(dd, &mut acc, lazy);
        prof("prune_losses", tq);
        let tq = Instant::now();
        // net value of each row = loss - price (concentrated affixes: -inf); rows with net < 0 do not
        // pay for themselves and go even under budget. Stopped rows: +inf for now (their net value
        // is above lazy - rounding).
        let net_of = |acc: &[(f64, usize, bool)]| -> Vec<f64> {
            (0..cand.len())
                .map(|j| if is_conc(j) { f64::NEG_INFINITY } else if acc[j].2 { acc[j].0 - prices[j] } else { f64::INFINITY })
                .collect()
        };
        let mut net = net_of(&acc);
        let below = net.iter().filter(|&&v| v < 0.0).count();
        let cap = ((d.rows() as f64) * step).ceil() as usize;
        let need = (d.rows().saturating_sub(budget)).max(below).min(cap);
        if need == 0 {
            return reuse;
        }
        if tail > 0.0 && d.rows() <= budget && (need as f64) < tail * d.rows() as f64 {
            // prune_tail_ppm (lossy): the last few rows below their price stay
            eprintln!("      prune pass {pass}: stopping, {need} rows below price (prune_tail_ppm)");
            return reuse;
        }
        let sorted = |net: &[f64]| {
            let mut order: Vec<usize> = (0..cand.len()).collect();
            order.sort_by(|&a, &b| net[a].partial_cmp(&net[b]).unwrap());
            order
        };
        let mut order = sorted(&net);
        // the first `safe` rows of the order are exactly those of the order by true net value
        let safe_of = |acc: &[(f64, usize, bool)], net: &[f64], lazy: f64| {
            if acc.iter().all(|a| a.2) {
                cand.len()
            } else {
                net.iter().filter(|&&v| v < lazy / 2.0).count()
            }
        };
        let mut safe = safe_of(&acc, &net, lazy);
        // widen: continue the stopped sums (in the state before this pass: dropped rows come back)
        let widen = |d: &mut Dict, dropped: &[(usize, u32)], acc: &mut Vec<(f64, usize, bool)>, lazy: &mut f64| {
            prof_add("prune_widen", 1.0);
            *lazy = if *lazy > 1e6 { f64::INFINITY } else { *lazy * 3.0 };
            for &(k, i) in dropped {
                d.tabs[k].alive[i as usize] = true;
            }
            advance(d, acc, *lazy);
            for &(k, i) in dropped {
                d.tabs[k].alive[i as usize] = false;
            }
        };
        while need > safe {
            widen(d, &[], &mut acc, &mut lazy);
            net = net_of(&acc);
            order = sorted(&net);
            safe = safe_of(&acc, &net, lazy);
        }
        let cutoff = order.get(need.saturating_sub(1)).map_or(0.0, |&o| net[o]);
        prof("prune_sort", tq);
        let tq = Instant::now();
        let mut removed = [0usize; 3];
        let mut done = 0;
        let mut dropped: Vec<(usize, u32)> = Vec::new();
        let mut sc = new_sc();
        let mut pos = 0;
        // rows that the types of an already dropped row now parse with: this pass's losses were
        // counted from the parse before any drop, so such a row looks unused by those types and
        // would be dropped too (Präsident goes because Prä + sident is one token, and then Prä or
        // sident goes because Präsident had all the uses). They are kept until the next pass
        // re-parses and counts them right.
        let mut pinned: [Vec<bool>; 3] = std::array::from_fn(|k| vec![false; d.tabs[k].strs.len()]);
        // The re-check of candidate o against the rows dropped so far: whether it is dropped, and the
        // tokens its uses' texts are parsed with once it is gone (pinned below). A candidate with a
        // finite net value has its uses' losses summed again with the row banned, up to its limit;
        // one without (conc) is dropped and its uses parsed with the row gone (banning a row parses
        // exactly as dropping it). Also returns every text it parsed (folded).
        let recheck = |d: &Dict, (sc, text): &mut (Scratch, Vec<u32>), net: &[f64], o: usize| -> (bool, Vec<Tok>, Vec<Vec<u32>>) {
            let (k, i) = cand[o];
            let (mut pin, mut texts, mut toks) = (Vec::new(), Vec::new(), Vec::new());
            let limit = prices[o] + net[o].max(cutoff) + 1e-9;
            let mut loss = 0.0;
            for &ti in uses[k].get(i as usize) {
                let (t, n) = &types[ti as usize];
                d.token_text_into(t, text);
                toks.clear();
                let (alt, _) = d.parse(text, sc, Some(&mut toks), Some((k, i)));
                pin.extend_from_slice(&toks);
                texts.push(text.iter().map(|&c| if is_byte(c) { c } else { d.sym.fold[c as usize] }).collect());
                if net[o].is_finite() {
                    loss += n * (if alt.is_infinite() { 1e6 } else { (alt - own_of(d, t)).max(0.0) });
                    if loss > limit {
                        return (false, pin, texts);
                    }
                }
            }
            (true, pin, texts)
        };
        // Re-checks are computed ahead in parallel, a batch of the next candidates at a time,
        // against the rows dropped when the batch starts. One is then used as it is unless a row
        // dropped since occurs (folded) in a text it parsed; otherwise it is redone. Exact: a row
        // whose string occurs nowhere in a text has no arc in its parse, so every parse, loss and
        // token is what the re-check at its turn computes.
        const SPEC_W: usize = 256;
        let mut spec: HashMap<usize, (bool, Vec<Tok>, Vec<Vec<u32>>), Fast> = HashMap::default();
        let mut spec_until = 0usize;
        // the rows dropped since the batch was computed (folded strings, by first symbol)
        let mut since: HashMap<u32, Vec<Vec<u32>>, Fast> = HashMap::default();
        let occurs = |since: &HashMap<u32, Vec<Vec<u32>>, Fast>, texts: &[Vec<u32>]| -> bool {
            !since.is_empty()
                && texts.iter().any(|t| (0..t.len()).any(|p| since.get(&t[p]).is_some_and(|v| v.iter().any(|s| t[p..].starts_with(s)))))
        };
        while pos < order.len() {
            if done >= need {
                break;
            }
            while pos >= safe {
                widen(d, &dropped, &mut acc, &mut lazy);
                net = net_of(&acc);
                order = sorted(&net);
                safe = safe_of(&acc, &net, lazy);
                spec.clear();
                spec_until = pos;
            }
            if pos >= spec_until {
                let hi = (pos + SPEC_W).min(safe).min(order.len());
                let batch: Vec<usize> = order[pos..hi].iter().copied().filter(|&o| !pinned[cand[o].0][cand[o].1 as usize]).collect();
                let dd: &Dict = d;
                let res = par_map(batch.len(), threads, 1, new_sc, |sc, q| recheck(dd, sc, &net, batch[q]));
                spec = batch.into_iter().zip(res).collect();
                since.clear();
                spec_until = hi;
            }
            let o = order[pos];
            pos += 1;
            let (k, i) = cand[o];
            if pinned[k][i as usize] {
                continue;
            }
            let (drop, pin_toks) = match spec.remove(&o) {
                Some((drop, pin, texts)) if !occurs(&since, &texts) => (drop, pin),
                _ => {
                    let (drop, pin, _) = recheck(d, &mut sc, &net, o);
                    (drop, pin)
                }
            };
            if !drop {
                continue;
            }
            d.tabs[k].alive[i as usize] = false;
            let f: Vec<u32> = d.tabs[k].strs[i as usize].iter().map(|&c| d.sym.fold[c as usize]).collect();
            if let Some(&c0) = f.first() {
                since.entry(c0).or_default().push(f);
            }
            dropped.push((k, i));
            removed[k] += 1;
            done += 1;
            for t in &pin_toks {
                if t.c != NONE && (t.c as usize) < pinned[0].len() {
                    pinned[0][t.c as usize] = true;
                }
                if t.p != 0 && (t.p as usize) < pinned[1].len() {
                    pinned[1][t.p as usize] = true;
                }
                if t.s != 0 && (t.s as usize) < pinned[2].len() {
                    pinned[2][t.s as usize] = true;
                }
            }
        }
        prof("prune_recheck", tq);
        let tq = Instant::now();
        d.rebuild();
        prof("prune_rebuild", tq);
        eprintln!("      prune pass {pass}: -{} cores -{} prefixes -{} suffixes (cutoff {:.0} tokens net, {below} rows below price){}",
                  removed[0], removed[1], removed[2], cutoff,
                  reparsed.map_or(String::new(), |r| format!(" (reuse: {r} of {} segments parsed again)", c.segs.len())));
        if let Some(bl) = blocks {
            cache = Some((st, bl));
            last_dropped = dropped;
        }
    }
    false
}

// ------------------------------------------------------------------ substring index

/// Frequent substrings of the raw text (count >= min_freq, 2 <= len <= max_len), counted
/// level by level: a length-l string is only counted where both of its (l-1)-substrings are
/// frequent. Each position counts with weight wt[i] (the frequency of its segment, since
/// repeated segments are stored once). Returns (position, length, weighted count).
fn mine(x: &[u32], wt: &[u32], min_freq: u32, max_len: usize, threads: usize) -> Vec<(u32, u16, u32)> {
    let n = x.len();
    let min_freq = min_freq as u64;
    let ok = |c: u32| c != NONE && !is_byte(c);
    let mut unigram: HashMap<u32, u64, Fast> = HashMap::default();
    for (i, &c) in x.iter().enumerate() {
        if ok(c) {
            *unigram.entry(c).or_insert(0) += wt[i] as u64;
        }
    }
    let mut freq: Vec<bool> = x.iter().map(|&c| ok(c) && unigram[&c] >= min_freq).collect();
    let mut h: Vec<u64> = x.iter().map(|&c| c as u64 + 1).collect();
    let mut out = Vec::new();
    let threads = threads.max(1);
    let shard_of = |h: u64| (h >> 40) as usize % threads;
    // the position-wise steps run on contiguous ranges of positions in parallel
    let span = n.div_ceil(threads).max(1);
    for l in 2..=max_len {
        if n < l {
            break;
        }
        // rolling hash of x[i..i+l]
        let m = n - l + 1;
        let mut cand = vec![false; m];
        std::thread::scope(|s| {
            for ((r, cand), h) in cand.chunks_mut(span).enumerate().zip(h.chunks_mut(span)) {
                let freq = &freq;
                s.spawn(move || {
                    for (j, c) in cand.iter_mut().enumerate() {
                        let i = r * span + j;
                        *c = freq[i] && freq[i + 1];
                        if *c {
                            h[j] = h[j].wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(x[i + l - 1] as u64 + 1);
                        }
                    }
                });
            }
        });
        // sharded counting: (count, first position). Each shard's positions are gathered range by
        // range, so every shard map sees its positions in increasing order, as a scan would.
        let (h, cand) = (&h, &cand);
        let lists: Vec<Vec<Vec<u32>>> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..m.div_ceil(span))
                .map(|r| {
                    s.spawn(move || {
                        let mut ls = vec![Vec::new(); threads];
                        for i in r * span..((r + 1) * span).min(m) {
                            if cand[i] {
                                ls[shard_of(h[i])].push(i as u32);
                            }
                        }
                        ls
                    })
                })
                .collect();
            hs.into_iter().map(|j| j.join().unwrap()).collect()
        });
        let shards: Vec<HashMap<u64, (u64, u32), Fast>> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..threads)
                .map(|t| {
                    let lists = &lists;
                    s.spawn(move || {
                        let mut mp: HashMap<u64, (u64, u32), Fast> = HashMap::default();
                        for ls in lists {
                            for &i in &ls[t] {
                                let i = i as usize;
                                mp.entry(h[i]).or_insert((0, i as u32)).0 += wt[i] as u64;
                            }
                        }
                        mp
                    })
                })
                .collect();
            hs.into_iter().map(|j| j.join().unwrap()).collect()
        });
        drop(lists);
        let mut nf = 0usize;
        std::thread::scope(|s| {
            for (r, fr) in freq.chunks_mut(span).enumerate() {
                let shards = &shards;
                s.spawn(move || {
                    for (j, f) in fr.iter_mut().enumerate() {
                        let i = r * span + j;
                        *f = i < m && cand[i] && shards[shard_of(h[i])][&h[i]].0 >= min_freq;
                    }
                });
            }
        });
        for sh in &shards {
            for (_, &(c, p)) in sh {
                if c >= min_freq {
                    out.push((p, l as u16, c.min(u32::MAX as u64) as u32));
                    nf += 1;
                }
            }
        }
        eprintln!("    substrings of length {l}: {nf} frequent");
        if nf == 0 {
            break;
        }
    }
    out
}

// ------------------------------------------------------------------ proposals

/// Exclusive count of every frequent substring: its count minus that of its more frequent
/// one-symbol extension (left or right, as found at its stored occurrence). Package values use it:
/// every substring of a frequent word is itself a frequent substring (msel, emsel, hemse ... of
/// themselves), and with full counts the same occurrences were valued once per overlapping
/// substring, so mid-word fragments collected many times the demand of the word itself. A
/// substring that almost only occurs inside one longer one (closed-substring test) now brings
/// almost nothing; them (in them, themselves, ...) keeps its own occurrences.
fn exclusive_counts(sym: &Symbols, x: &[u32], subs: &[(u32, u16, u32)], threads: usize) -> Vec<u32> {
    let mut cnt: HashMap<u64, u32, Fast> = HashMap::default();
    cnt.reserve(subs.len());
    for &(p, l, c) in subs {
        cnt.insert(seq_hash(x[p as usize..p as usize + l as usize].iter().copied()), c);
    }
    let out = par_map(subs.len(), threads, 4096, || (), |_, k| {
        let (p, l, c) = subs[k];
        let (p, l) = (p as usize, l as usize);
        let mut ext = 0u32;
        // only extensions that continue a word (letter / digit / mark next to one) count: a
        // space or punctuation before or after a word becomes an affix, not part of the core
        let an = |c: u32| c != NONE && !is_byte(c) && sym.alnum.get(c as usize).is_some_and(|&a| a != 0);
        if p > 0 && an(x[p - 1]) && an(x[p]) {
            ext = ext.max(cnt.get(&seq_hash(x[p - 1..p + l].iter().copied())).copied().unwrap_or(0));
        }
        if p + l < x.len() && an(x[p + l]) && an(x[p + l - 1]) {
            ext = ext.max(cnt.get(&seq_hash(x[p..p + l + 1].iter().copied())).copied().unwrap_or(0));
        }
        c.saturating_sub(ext)
    });
    eprintln!("  exclusive counts: {} of {} substrings keep >= 10% of their occurrences",
              out.iter().zip(subs).filter(|(e, s)| **e as f64 >= 0.1 * s.2 as f64).count(), subs.len());
    out
}

/// Row key: (table, string)
type RowKey = (u8, Vec<u32>);

/// Score every missing row by package demand; returns (demand, row) in no particular order.
/// `failed` holds demand keys ([table, symbols..]) of rows that were added and pruned in the last
/// round: a cut needing one of them is skipped, and so is a cut needing a dead row whose price is
/// above the demand it has. Otherwise a package's value keeps going to the cut of the most
/// widely shared fragments (suffix ोशिश, core ␣क), which are pruned again every round, and the
/// word's own core (कोशिश) never gets the demand it would earn.
fn propose(d: &Dict, x: &[u32], subs: &[(u32, u16, u32)], excl: &[u32], use_direct: bool, max_packages: usize, threads: usize,
           failed: &std::collections::HashSet<Vec<u32>, Fast>) -> Vec<(f64, RowKey)> {
    let tr_words = load_trace();
    let tr_words = &tr_words;
    // value of each substring under the current dictionary
    let tq = Instant::now();
    let mut vals: Vec<(f64, usize)> = par_map(subs.len(), threads, 1024, Scratch::new, |sc, k| {
        let (p, l, _) = subs[k];
        let t = d.parse_prim(&x[p as usize..p as usize + l as usize], sc, None);
        (t > 1.0 + 1e-9 && excl[k] > 0).then(|| (excl[k] as f64 * (t - 1.0), k))
    })
    .into_iter()
    .flatten()
    .collect();
    prof("prop_vals", tq);
    let tq = Instant::now();
    vals.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    vals.truncate(max_packages);
    prof("prop_vals_sort", tq);
    let tq = Instant::now();

    // cuts of w with their missing rows: (number of missing rows, rows in prefix, core, suffix
    // order). Row membership comes from trie walks (the tries hold exactly the live rows), so a
    // cut costs no string allocation or hashing.
    let cuts = |w: &[u32], cs: &mut CutScratch| {
        if d.ca {
            cuts_ca(d, w, cs);
            return;
        }
        let l = w.len();
        cs.cuts.clear();
        cs.p_ok.clear();
        cs.p_ok.resize(l + 1, false);
        cs.s_ok.clear();
        cs.s_ok.resize(l + 1, false);
        cs.p_ok[0] = true;
        cs.s_ok[l] = true;
        let live = |k: usize, n: u32| {
            let r = d.tries[k].term(n);
            r != NONE && d.tabs[k].alive[r as usize]
        };
        let mut node = 0u32;
        for a in 1..=l {
            let Some(m) = d.tries[1].step(node, w[a - 1]) else { break };
            node = m;
            cs.p_ok[a] = live(1, node);
        }
        for b in 0..l {
            let mut node = 0u32;
            let mut ok = true;
            for &ch in &w[b..] {
                match d.tries[2].step(node, ch) {
                    Some(m) => node = m,
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            cs.s_ok[b] = ok && live(2, node);
        }
        for a in 0..l {
            // mark_rule: no core, and no prefix, starts at a combining mark (such rows can never be
            // added, so demand routed to them would be lost)
            if d.sym.mark_at(w[a]) || (a > 0 && d.sym.mark_at(w[0])) {
                continue;
            }
            let p = &w[..a];
            let p_ok = cs.p_ok[a];
            let mut m = 0u8;
            let (mut ca, mut h) = (CaseAcc::default(), SEQ_H0);
            let mut node = Some(0u32);
            cs.folded.clear();
            for b in a + 1..=l {
                let ch = w[b - 1];
                if d.case {
                    ca.push(d.sym.cf[ch as usize], b - a - 1);
                    if ca.dead() {
                        break;
                    }
                } else {
                    m = if b == a + 1 { d.sym.mask(ch) } else { combine(m, d.sym.mask(ch)) };
                    if variation(m).is_none() {
                        break;
                    }
                }
                let f = d.sym.fold[ch as usize];
                cs.folded.push(f);
                node = node.and_then(|n| d.tries[0].step(n, f));
                if d.case {
                    // the core must be writable from its canonical spelling (a live row's is the same)
                    // (the lookup only matters for mixed-case spans: without cased chars the mask is 0,
                    // and an all-lowercase span is always written by lower)
                    h = seq_step(h, f);
                    let lookup = ca.hasu != 0 && (ca.up != 0 || ca.trad);
                    let uc = if lookup { d.canon.get(&h).copied().unwrap_or(0) & ca.hasu } else { 0 };
                    if ca.best(uc, &[0.0; NVAR_MAX]).is_none() {
                        continue;
                    }
                }
                let s = &w[b..];
                if d.classes && !(d.row_ok(1, p) && d.row_ok(0, &cs.folded) && d.row_ok(2, s)) {
                    continue;
                }
                let mut miss = [Miss { t: 0, a: 0, b: 0 }; 3];
                let mut nm = 0;
                if !p_ok {
                    miss[nm] = Miss { t: 1, a: 0, b: a as u16 };
                    nm += 1;
                }
                if !node.is_some_and(|n| live(0, n)) {
                    miss[nm] = Miss { t: 0, a: a as u16, b: b as u16 };
                    nm += 1;
                }
                if !cs.s_ok[b] {
                    miss[nm] = Miss { t: 2, a: b as u16, b: l as u16 };
                    nm += 1;
                }
                if nm > 0 {
                    cs.cuts.push((nm, miss));
                }
            }
        }
    };
    // Demand maps are sharded by key (shard_of): each part of the packages fills one map per
    // shard, and the shards are merged in parallel, each key's values still added part by part in
    // order, so every value is the one a single merged map would get.
    type Demand = Vec<HashMap<Vec<u32>, f64, Fast>>;
    let nsh = threads.max(1);
    let merge = |parts: Vec<Demand>| -> Demand {
        let mut by_shard: Vec<Vec<HashMap<Vec<u32>, f64, Fast>>> = (0..nsh).map(|_| Vec::new()).collect();
        for p in parts {
            for (h, m) in p.into_iter().enumerate() {
                by_shard[h].push(m);
            }
        }
        std::thread::scope(|sc| {
            let hs: Vec<_> = by_shard
                .into_iter()
                .map(|ms| {
                    sc.spawn(move || {
                        let mut acc: HashMap<Vec<u32>, f64, Fast> = HashMap::default();
                        for m in ms {
                            for (k, v) in m {
                                *acc.entry(k).or_insert(0.0) += v;
                            }
                        }
                        acc
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        })
    };
    // accumulate into a demand map without allocating a key for rows already in it
    let add = |dm: &mut Demand, key: &[u32], v: f64| {
        let dm = &mut dm[shard_of(key, nsh)];
        match dm.get_mut(key) {
            Some(e) => *e += v,
            None => {
                dm.insert(key.to_vec(), 0.0 + v);
            }
        }
    };
    // The sums run over fixed parts of the packages (one thread each), in order.
    let chunk = vals.len().div_ceil(threads).max(1);
    const PB: usize = 256;
    let nv = vals.len();
    let vals = &vals;
    let word = |pk: usize| {
        let (p, l, _) = subs[vals[pk].1];
        &x[p as usize..p as usize + l as usize]
    };
    let accumulate = |contrib: &Vec<Vec<(u32, Miss, f64)>>| -> Demand {
        merge(std::thread::scope(|s| {
            let hs: Vec<_> = (0..nv.div_ceil(chunk))
                .map(|pt| {
                    let (add, word) = (&add, &word);
                    s.spawn(move || {
                        let mut dm: Demand = (0..nsh).map(|_| HashMap::default()).collect();
                        let mut key = Vec::new();
                        let (lo, hi) = (pt * chunk, ((pt + 1) * chunk).min(nv));
                        for blk in &contrib[lo / PB..hi.div_ceil(PB)] {
                            for &(pk, m, v) in blk {
                                if (lo..hi).contains(&(pk as usize)) {
                                    miss_key(d, word(pk as usize), m, &mut key);
                                    add(&mut dm, &key, v);
                                }
                            }
                        }
                        dm
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        }))
    };
    // pass 1: spread over the cheapest cuts (fewest missing rows, +1)
    let demand1 = merge(std::thread::scope(|s| {
        let hs: Vec<_> = vals
            .chunks(chunk)
            .map(|part| {
                let (cuts, add) = (&cuts, &add);
                s.spawn(move || {
                    let mut dm: Demand = (0..nsh).map(|_| HashMap::default()).collect();
                    let mut cs = CutScratch::default();
                    let mut key = Vec::new();
                    for &(g, si) in part {
                        let (p, l, _) = subs[si];
                        let w = &x[p as usize..p as usize + l as usize];
                        cuts(w, &mut cs);
                        if !failed.is_empty() {
                            cs.cuts.retain(|c| c.1[..c.0].iter().all(|&m| {
                                miss_key(d, w, m, &mut key);
                                !failed.contains(&key[..])
                            }));
                        }
                        let Some(min) = cs.cuts.iter().map(|c| c.0).min() else { continue };
                        let n_good = cs.cuts.iter().filter(|c| c.0 <= min + 1).count();
                        let share = g / n_good as f64;
                        for c in cs.cuts.iter().filter(|c| c.0 <= min + 1) {
                            for &m in &c.1[..c.0] {
                                miss_key(d, w, m, &mut key);
                                add(&mut dm, &key, share / c.0 as f64);
                            }
                        }
                    }
                    dm
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    }));
    prof("prop_pass1", tq);
    let tq = Instant::now();
    // pass 2: each package goes to the cut whose missing rows are most in demand. Choosing the cut
    // (a demand lookup per missing row of every cut) runs on blocks handed out on demand; only the
    // few rows of the chosen cuts are then added up part by part.
    // (the direct values below come from the same cuts: collected here, block by block, in the
    // same order as a separate pass over the packages would)
    let an = |x: u32| !is_byte(x) && d.sym.alnum.get(x as usize).is_some_and(|&a| a != 0);
    let both: Vec<(Vec<(u32, Miss, f64)>, Vec<(Vec<u32>, f64)>)> = par_map(vals.len().div_ceil(PB), threads, 1, || (CutScratch::default(), Vec::new(), Vec::new()), |(cs, key, score), b| {
        let mut out = Vec::new();
        let mut dout = Vec::new();
        for pk in b * PB..((b + 1) * PB).min(vals.len()) {
            let g = vals[pk].0;
            let w = word(pk);
            cuts(w, cs);
            if use_direct {
                for c in &cs.cuts {
                    if c.1[0].t == 0 && c.0 == 1 {
                        miss_key(d, w, c.1[0], key);
                        // not a core that swallows the space or punctuation next to a word ( iPhone,
                        // GitHub.): the affixes carry those, and the bare word is the core
                        let n = key.len();
                        let glued = n > 2 && ((!an(key[1]) && an(key[2])) || (!an(key[n - 1]) && an(key[n - 2])));
                        if !glued && !failed.contains(&key[..]) {
                            dout.push((key.clone(), g));
                        }
                    }
                }
            }
            score.clear();
            for c in &cs.cuts {
                let mut hopeless = false;
                // the cut only happens if all its missing rows are added: its weakest row bounds it
                // (a sum let one very popular row, core té, carry a one-off partner, possibili)
                let weakest: f64 = c.1[..c.0]
                    .iter()
                    .map(|&m| {
                        miss_key(d, w, m, key);
                        let v = demand1[shard_of(key, nsh)].get(&key[..]).copied().unwrap_or(0.0);
                        if failed.contains(&key[..]) {
                            hopeless = true;
                        } else if let Some(&r) = d.tabs[m.t as usize].map.get(&key[1..]) {
                            hopeless |= !d.tabs[m.t as usize].alive[r as usize] && d.tabs[m.t as usize].price[r as usize] > v;
                        }
                        v
                    })
                    .fold(f64::INFINITY, f64::min);
                score.push(if hopeless { -1.0 } else { weakest / c.0 as f64 });
            }
            // the last of equally good cuts, as Iterator::max_by
            let best = (0..cs.cuts.len()).max_by(|&a, &b| score[a].partial_cmp(&score[b]).unwrap()).filter(|&b| score[b] >= 0.0);
            if tr_words.iter().any(|t| t[..] == w[..]) {
                let name = |m: Miss, key: &mut Vec<u32>| { miss_key(d, w, m, key); format!("{}[{}..{}]", ["core", "pre", "suf"][m.t as usize], m.a, m.b) };
                let mut lines = Vec::new();
                for (ci, c) in cs.cuts.iter().enumerate() {
                    let rows: Vec<String> = c.1[..c.0].iter().map(|&m| { let n = name(m, key); format!("{n} d1={:.0}", demand1[shard_of(key, nsh)].get(&key[..]).copied().unwrap_or(0.0)) }).collect();
                    lines.push(format!("      cut score {:.0}{}: {}", score[ci], if Some(ci) == best { " <== CHOSEN" } else { "" }, rows.join(" + ")));
                }
                eprintln!("    trace package {:?} value {g:.0}, {} cuts:
{}", w, cs.cuts.len(), lines.join("
"));
            }
            if let Some(bi) = best {
                let c = cs.cuts[bi];
                for &m in &c.1[..c.0] {
                    out.push((pk as u32, m, g / c.0 as f64));
                }
            }
        }
        (out, dout)
    });
    let (contrib, direct): (Vec<_>, Vec<_>) = both.into_iter().unzip();
    let mut demand2 = accumulate(&contrib);
    drop(contrib);
    prof("prop_pass2", tq);
    let tq = Instant::now();
    // DIRECT value: a core that alone completes a package (its only missing row; the package is
    // then one token) is worth at least that package's value, whatever the routing gave it. Shared
    // short pieces (tou + jo + urs) collect demand from hundreds of packages and win the routing,
    // so a frequent word (toujours) got almost nothing; the exact re-scoring then checks the real
    // gain. (Exclusive counts keep mid-word fragments' own values small.) Max over packages.
    // (each shard's map on its own thread, its keys in the same order as one pass would take them)
    let mut by_shard: Vec<Vec<(Vec<u32>, f64)>> = (0..nsh).map(|_| Vec::new()).collect();
    for blk in direct {
        for (k, g) in blk {
            by_shard[shard_of(&k, nsh)].push((k, g));
        }
    }
    let n_direct: usize = std::thread::scope(|sc| {
        let hs: Vec<_> = demand2
            .iter_mut()
            .zip(by_shard)
            .map(|(m, ks)| {
                sc.spawn(move || {
                    let mut n = 0usize;
                    for (k, g) in ks {
                        match m.get_mut(&k) {
                            Some(v) => {
                                if g > *v {
                                    *v = g;
                                    n += 1;
                                }
                            }
                            None => {
                                m.insert(k, g);
                                n += 1;
                            }
                        }
                    }
                    n
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).sum()
    });
    eprintln!("    direct values raised {n_direct} core proposals");
    prof("prop_direct", tq);
    let tq = Instant::now();
    let rows: Vec<(f64, RowKey)> = demand2.into_iter().flatten().map(|(k, v)| (v, (k[0] as u8, k[1..].to_vec()))).collect();
    prof("prop_sort", tq);
    rows
}

/// A row a cut still lacks: table t and the span w[a..b] of the substring it spells.
#[derive(Clone, Copy)]
struct Miss {
    t: u8,
    a: u16,
    b: u16,
}

#[derive(Default)]
struct CutScratch {
    cuts: Vec<(usize, [Miss; 3])>,
    p_ok: Vec<bool>,
    s_ok: Vec<bool>,
    folded: Vec<u32>,
    p_tm: Vec<u8>, // case_affixes: transition masks of the prefix w[..a] / suffix w[b..] (0 = unwritable)
    s_tm: Vec<u8>,
}

/// propose's cuts with case_affixes: every cut w = prefix + core + suffix that one variation
/// writes as a whole token (the parts' canonical spellings: the rows' if they exist, else the
/// ones they would get), with its missing rows (all keyed folded)
fn cuts_ca(d: &Dict, w: &[u32], cs: &mut CutScratch) {
    let l = w.len();
    cs.cuts.clear();
    for v in [&mut cs.p_ok, &mut cs.s_ok] {
        v.clear();
        v.resize(l + 1, false);
    }
    for v in [&mut cs.p_tm, &mut cs.s_tm] {
        v.clear();
        v.resize(l + 1, 0);
    }
    cs.p_ok[0] = true;
    cs.s_ok[l] = true;
    cs.p_tm[0] = TM_EMPTY;
    cs.s_tm[l] = TM_EMPTY;
    let live = |k: usize, n: Option<u32>| {
        n.is_some_and(|n| {
            let r = d.tries[k].term(n);
            r != NONE && d.tabs[k].alive[r as usize]
        })
    };
    let fold = |c: u32| d.sym.fold[c as usize];
    // every prefix w[..a] and suffix w[b..]: transition mask and whether it is a live row
    let (mut ca, mut h, mut node) = (CaseAcc::default(), SEQ_H0, Some(0u32));
    for a in 1..=l {
        let ch = w[a - 1];
        ca.push(d.sym.cf[ch as usize], a - 1);
        if ca.dead() {
            break;
        }
        h = seq_step(h, fold(ch));
        node = node.and_then(|n| d.tries[1].step(n, fold(ch)));
        cs.p_ok[a] = live(1, node);
        cs.p_tm[a] = d.span_tm(&ca, h);
    }
    for b in 0..l {
        let (mut ca, mut h, mut node) = (CaseAcc::default(), SEQ_H0, Some(0u32));
        let mut dead = false;
        for (k, &ch) in w[b..].iter().enumerate() {
            ca.push(d.sym.cf[ch as usize], k);
            if ca.dead() {
                dead = true;
                break;
            }
            h = seq_step(h, fold(ch));
            node = node.and_then(|n| d.tries[2].step(n, fold(ch)));
        }
        if !dead {
            cs.s_ok[b] = live(2, node);
            cs.s_tm[b] = d.span_tm(&ca, h);
        }
    }
    for a in 0..l {
        // mark_rule: no core, and no prefix, starts at a combining mark
        if d.sym.mark_at(w[a]) || (a > 0 && d.sym.mark_at(w[0])) {
            continue;
        }
        let after_p = vs_start(cs.p_tm[a]);
        if after_p == 0 {
            continue;
        }
        let (mut ca, mut h, mut node) = (CaseAcc::default(), SEQ_H0, Some(0u32));
        cs.folded.clear();
        for b in a + 1..=l {
            let ch = w[b - 1];
            ca.push(d.sym.cf[ch as usize], b - a - 1);
            if ca.dead() {
                break;
            }
            let f = fold(ch);
            cs.folded.push(f);
            h = seq_step(h, f);
            node = node.and_then(|n| d.tries[0].step(n, f));
            let after_c = vs_after(after_p, d.span_tm(&ca, h));
            if vs_after(after_c, cs.s_tm[b]) == 0 {
                continue; // no single variation writes this cut
            }
            if d.classes {
                let (p, s): (Vec<u32>, Vec<u32>) = (w[..a].iter().map(|&c| fold(c)).collect(), w[b..].iter().map(|&c| fold(c)).collect());
                if !(d.row_ok(1, &p) && d.row_ok(0, &cs.folded) && d.row_ok(2, &s)) {
                    continue;
                }
            }
            let mut miss = [Miss { t: 0, a: 0, b: 0 }; 3];
            let mut nm = 0;
            if !cs.p_ok[a] {
                miss[nm] = Miss { t: 1, a: 0, b: a as u16 };
                nm += 1;
            }
            if !live(0, node) {
                miss[nm] = Miss { t: 0, a: a as u16, b: b as u16 };
                nm += 1;
            }
            if !cs.s_ok[b] {
                miss[nm] = Miss { t: 2, a: b as u16, b: l as u16 };
                nm += 1;
            }
            if nm > 0 {
                cs.cuts.push((nm, miss));
            }
        }
    }
}

/// Demand-map key of a missing row: [table, symbols..] (cores folded, affixes too with
/// case_affixes); orders like RowKey.
fn miss_key(d: &Dict, w: &[u32], m: Miss, key: &mut Vec<u32>) {
    key.clear();
    key.push(m.t as u32);
    let r = &w[m.a as usize..m.b as usize];
    if m.t == 0 || d.ca {
        key.extend(r.iter().map(|&c| d.sym.fold[c as usize]));
    } else {
        key.extend_from_slice(r);
    }
}

// ------------------------------------------------------------------ refactoring proposals

/// Shard of a string key in a sharded map (see refactor_proposals).
fn shard_of(key: &[u32], n: usize) -> usize {
    (seq_hash(key.iter().copied()) % n as u64) as usize
}

/// For every observed token type with a non-empty affix, propose the cores pc, cs and pcs (the
/// affixes absorbed into the core, built from the raw surface and re-folded; only if that is a
/// legal core). Value = uses x lambda x the affix information the absorption saves.
/// Returned as `threads` maps sharded by shard_of (folded core string -> value). The candidates
/// are found in parallel and every shard then sums its keys' values in the token types' order,
/// so each value is exactly what one sequential pass over the types would give.
fn refactor_proposals(d: &Dict, st: &ParseStats, threads: usize) -> Vec<HashMap<Vec<u32>, f64, Fast>> {
    let nsh = threads.max(1);
    let tq = Instant::now();
    let types: Vec<(Tok, f64)> = st.types.iter().map(|(t, &n)| (*t, n)).collect();
    prof("rf_types", tq);
    let tq = Instant::now();
    let (cp, cs) = (&d.tabs[1].cost, &d.tabs[2].cost);
    // legal core spelled by raw: its folded form into f
    let legal = |raw: &[u32], f: &mut Vec<u32>| -> bool { d.legal_core(raw, f) };
    // per block of types: the candidate strings, and per shard (start, len, value) in order
    const G: usize = 4096;
    let blocks: Vec<(Vec<u32>, Vec<Vec<(u32, u32, f64)>>)> = par_map(types.len().div_ceil(G), threads, 1, || (Vec::new(), Vec::new(), Vec::new()), |(raw, f, core), b| {
        let mut syms: Vec<u32> = Vec::new();
        let mut ents: Vec<Vec<(u32, u32, f64)>> = vec![Vec::new(); nsh];
        for &(t, n) in &types[b * G..((b + 1) * G).min(types.len())] {
            if t.c == NONE || (t.p == 0 && t.s == 0) {
                continue;
            }
            // the token's text, split into its written prefix, core and suffix (case_affixes: the
            // variation applies to all three)
            core.clear();
            let (lp, ls) = (d.tabs[1].strs[t.p as usize].len(), d.tabs[2].strs[t.s as usize].len());
            if d.ca {
                d.token_text_into(&t, core);
            } else {
                core.extend_from_slice(&d.tabs[1].strs[t.p as usize]);
                d.write_core(t.c as usize, t.v, core);
                core.extend_from_slice(&d.tabs[2].strs[t.s as usize]);
            }
            let (p, rest) = core.split_at(lp);
            let (core, s) = rest.split_at(rest.len() - ls);
            let save_p = (cp[t.p as usize] - cp[0]).max(0.0);
            let save_s = (cs[t.s as usize] - cs[0]).max(0.0);
            let (sp, ss) = (&d.tabs[1].spec, &d.tabs[2].spec);
            let spec_p = (sp[t.p as usize] - sp[0]).max(0.0);
            let spec_s = (ss[t.s as usize] - ss[0]).max(0.0);
            let pair_p = if d.mu2 > 0.0 { d.pmi_of(0, t.p, t.c) } else { 0.0 };
            let pair_s = if d.mu2 > 0.0 { d.pmi_of(1, t.c, t.s) } else { 0.0 };
            // absorbing an affix also saves its information beyond the core (the new core's empty affix
            // is charged at its unconditional cost)
            let cond_p = if d.lamc > 0.0 && t.p != 0 { (d.cond(0, t.p, t.c, (-cp[t.p as usize]).exp2()) - cp[0]).max(0.0) } else { 0.0 };
            let cond_s = if d.lamc > 0.0 && t.s != 0 { (d.cond(1, t.c, t.s, (-cs[t.s as usize]).exp2()) - cs[0]).max(0.0) } else { 0.0 };
            // code_w: the bits of the affix given the core, beyond an empty affix's marginal
            // (without code_len: the affix's unconditional bits beyond an empty affix's, save_p / save_s)
            let cw = (d.cl.on && d.cl.w > 0.0) || d.cl.wu > 0.0;
            let clb_p = if cw && t.p != 0 { if d.cl.on { (d.cl_abits(0, t.p, t.c, cp[t.p as usize]) - cp[0]).max(0.0) } else { save_p } } else { 0.0 };
            let clb_s = if cw && t.s != 0 { if d.cl.on { (d.cl_abits(1, t.s, t.c, cs[t.s as usize]) - cs[0]).max(0.0) } else { save_s } } else { 0.0 };
            let (gc, gp, gs) = (d.tabs[0].glue[t.c as usize], d.tabs[1].glue[t.p as usize], d.tabs[2].glue[t.s as usize]);
            for kind in 0..3 {
                let (save, spec, pair, condg, glue, clb) = match kind {
                    0 if t.p != 0 => (save_p, spec_p, pair_p, cond_p, gc + gp, clb_p),
                    1 if t.s != 0 => (save_s, spec_s, pair_s, cond_s, gc + gs, clb_s),
                    2 if t.p != 0 && t.s != 0 => (save_p + save_s, spec_p + spec_s, pair_p + pair_s, cond_p + cond_s, gc + gp + gs, clb_p + clb_s),
                    _ => continue,
                };
                let value = n * d.lam * save + n * d.mu * spec + if d.mu2 > 0.0 { n * d.mu2 * pair } else { 0.0 }
                    + if d.lamc > 0.0 { n * d.lamc * condg } else { 0.0 }
                    + if cw { n * d.cl.w * clb } else { 0.0 };
                if value <= 0.0 && d.nu == 0.0 {
                    continue;
                }
                raw.clear();
                if kind != 1 {
                    raw.extend_from_slice(p);
                }
                raw.extend_from_slice(core);
                if kind != 0 {
                    raw.extend_from_slice(s);
                }
                if legal(raw, f) && !d.tabs[0].has(f) && d.row_ok(0, f) {
                    let value = value + n * d.nu * (glue - d.glue_of(0, f)).max(0.0);
                    if value > 0.0 {
                        ents[shard_of(f, nsh)].push((syms.len() as u32, f.len() as u32, value));
                        syms.extend_from_slice(f);
                    }
                }
            }
        }
        (syms, ents)
    });
    prof("rf_cands", tq);
    let tq = Instant::now();
    let r = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..nsh)
            .map(|h| {
                let blocks = &blocks;
                sc.spawn(move || {
                    let mut out: HashMap<Vec<u32>, f64, Fast> = HashMap::default();
                    for (syms, ents) in blocks {
                        for &(a, l, v) in &ents[h] {
                            let key = &syms[a as usize..(a + l) as usize];
                            match out.get_mut(key) {
                                Some(e) => *e += v,
                                None => {
                                    out.insert(key.to_vec(), 0.0 + v);
                                }
                            }
                        }
                    }
                    out
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    prof("rf_accum", tq);
    r
}

/// Debug trace (env DS_TRACE_FILE: u32 n, then n x (u32 len, u32 ids[len]) raw symbol strings):
/// the state of every traced string as a core (folded) and as a prefix / suffix (raw).
fn load_trace() -> Vec<Vec<u32>> {
    let Ok(path) = std::env::var("DS_TRACE_FILE") else { return Vec::new() };
    let mut r = Reader::open(&path);
    let n = r.u32() as usize;
    (0..n).map(|_| { let l = r.u32() as usize; r.u32s(l) }).collect()
}

fn fold_str(d: &Dict, s: &[u32]) -> Vec<u32> {
    s.iter().map(|&c| d.sym.fold[c as usize]).collect()
}

fn trace_state(d: &Dict, st: &ParseStats, tr: &[Vec<u32>], label: &str) {
    for (i, s) in tr.iter().enumerate() {
        let mut parts = Vec::new();
        for k in 0..3 {
            let key = if k == 0 || d.ca { fold_str(d, s) } else { s.clone() };
            if let Some(&r) = d.tabs[k].map.get(&key) {
                let r = r as usize;
                parts.push(format!("{}:{} uses {:.0} price {:.1} glue {:.2}", ["core", "pre", "suf"][k],
                    if d.tabs[k].alive[r] { "live" } else { "dead" }, st.use_[k].get(r).copied().unwrap_or(0.0),
                    d.tabs[k].price[r], d.tabs[k].glue[r]));
            }
        }
        eprintln!("    trace[{i}] {label}: {}", if parts.is_empty() { "no row".to_string() } else { parts.join(" | ") });
    }
}

/// Swap proposals (row prices on): core + affix piece combinations that dominate (>= half) the
/// core's uses or the whole affix's uses, where the merged core (pc / cs / pcs, legal, not yet a row) is cheaper than the
/// share of the old rows' prices it would free: happines + s -> happiness. Such a merge saves no
/// tokens, so neither the substring proposals nor exact re-scoring can see its value; these rows
/// are added directly and pruning then drops whichever rows no longer pay for themselves.
fn swap_proposals(d: &Dict, st: &ParseStats, threads: usize) -> Vec<Vec<u32>> {
    if d.pricing.pi == 0.0 && d.pricing.pi_core == 0.0 {
        return Vec::new();
    }
    let tq = Instant::now();
    let types: Vec<(Tok, f64)> = st.types.iter().filter(|(t, _)| t.c != NONE).map(|(t, &n)| (*t, n)).collect();
    let mut uses: [Vec<f64>; 3] = [0, 1, 2].map(|k| vec![0.0; d.tabs[k].strs.len()]);
    for &(t, n) in &types {
        uses[0][t.c as usize] += n;
        uses[1][t.p as usize] += n;
        uses[2][t.s as usize] += n;
    }
    // uses of each merge: (side, core, variation, absorbed piece) -> (uses, uses where the piece is
    // the whole affix). The piece is the affix's inner part (the end of a prefix, the start of a
    // suffix) whose outer rest is itself a row, so happines + s / s, / s. all count as
    // happiness + (empty / , / .)
    // Which pieces an affix row offers depends only on the row, so they are found once per used
    // row (not once per token type), numbered per side: pieces[k - 1][row] = [(piece id, whole)].
    prof("sw_uses", tq);
    let tq = Instant::now();
    let mut pieces: [Vec<Vec<(u32, bool)>>; 2] = [Vec::new(), Vec::new()];
    let mut piece_strs: [Vec<&[u32]>; 2] = [Vec::new(), Vec::new()];
    for k in 1..3 {
        let mut ids: HashMap<&[u32], u32, Fast> = HashMap::default();
        let t = &d.tabs[k];
        pieces[k - 1] = (0..t.strs.len())
            .map(|r| {
                let a = &t.strs[r][..];
                let l = a.len();
                if uses[k][r] == 0.0 {
                    return Vec::new();
                }
                (1..=l)
                    .filter(|&j| j == l || if k == 1 { t.has(&a[..l - j]) } else { t.has(&a[j..]) })
                    .map(|j| {
                        let s = if k == 1 { &a[l - j..] } else { &a[..j] };
                        let n = ids.len() as u32;
                        let id = *ids.entry(s).or_insert(n);
                        if id == n {
                            piece_strs[k - 1].push(s);
                        }
                        (id, j == l)
                    })
                    .collect()
            })
            .collect();
    }
    prof("sw_pieces", tq);
    let tq = Instant::now();
    // The merge sums are sharded by core: every shard scans the types in order and adds up the
    // merges of its own cores, so each key sees its terms in the types' order, exactly as one
    // sequential pass over the types (the result is a sorted set anyway).
    type Key = (u32, u32, u32); // (core, side x 8 + variation, piece id)
    let nsh = (4 * threads).max(1);
    let found: Vec<Vec<(f64, Vec<u32>)>> = par_map(nsh, threads, 1, || Vec::new(), |raw: &mut Vec<u32>, h| {
        let mut merges: HashMap<Key, (f64, f64), Fast> = HashMap::default();
        for &(t, n) in &types {
            if t.c as usize % nsh != h {
                continue;
            }
            for (side, r) in [(0u32, t.p), (1, t.s)] {
                for &(pid, whole) in &pieces[side as usize][r as usize] {
                    let e = merges.entry((t.c, side * 8 + t.v, pid)).or_insert((0.0, 0.0));
                    e.0 += n;
                    if whole {
                        e.1 += n;
                    }
                }
            }
        }
        let mut out = Vec::new();
        for (&(c, sv, pid), &(n, n_whole)) in &merges {
            let (side, v) = (sv / 8, sv % 8);
            let piece = piece_strs[side as usize][pid as usize];
            let k = if side == 0 { 1 } else { 2 };
            let arow = if n_whole > 0.0 { d.tabs[k].map.get(piece).copied() } else { None };
            // dominated from either side: the core is mostly used with this piece (happines + s),
            // or the whole affix is mostly used with this core (priorit + ize_: once prioritize_
            // exists, priorit is left to priority, which the core side then proposes)
            let core_dom = n >= d.pricing.swap_share * uses[0][c as usize];
            let affix_dom = arow.is_some_and(|r| n_whole >= 0.5 * uses[k][r as usize]);
            // SELF-PAYING (swap_self): the merged core saves, on every use where the piece was the
            // whole affix, that affix's information cost (lambda x its bits + delta). That is what
            // pruning weighs against the price, so a merge whose saving is above its price would be
            // kept once added; but it saves no token (Got:t, hi:m, du:e are one token either way),
            // so no other proposal ever adds it.
            // (code_w: also w x the affix's bits given the core, beyond an empty affix's marginal)
            // (without code_len: w x its unconditional bits beyond an empty affix's)
            let clw = |r: u32| -> f64 {
                if d.cl.on && d.cl.w > 0.0 {
                    let ct = &d.tabs[k].cost;
                    d.cl.w * (d.cl_abits(k - 1, r, c, ct[r as usize]) - ct[0]).max(0.0)
                } else if d.cl.wu > 0.0 {
                    let ct = &d.tabs[k].cost;
                    d.cl.wu * (ct[r as usize] - ct[0]).max(0.0)
                } else {
                    0.0
                }
            };
            let self_pay = d.pricing.swap_self > 0.0
                && arow.is_some_and(|r| n_whole * (d.lam * d.tabs[k].cost[r as usize] + d.delta + clw(r)) > 0.0);
            if !core_dom && !affix_dom && !self_pay {
                continue;
            }
            raw.clear();
            if d.ca {
                // (the piece with its own canonical spelling, the variation over both)
                let (pc, cr) = ((piece, d.canon_mask(piece)), (&d.tabs[0].strs[c as usize][..], d.tabs[0].umask[c as usize]));
                d.write_parts(&if side == 0 { [pc, cr] } else { [cr, pc] }, side == 0, v, raw);
            } else {
                if side == 0 {
                    raw.extend_from_slice(piece);
                }
                d.write_core(c as usize, v, raw);
                if side == 1 {
                    raw.extend_from_slice(piece);
                }
            }
            let mut f = Vec::with_capacity(raw.len());
            if !d.legal_core(raw, &mut f) || d.tabs[0].has(&f) || !d.row_ok(0, &f) {
                continue;
            }
            // freed: the core's price in proportion to the uses that move, plus the piece's own price
            // (as an affix row) for the uses where it was the whole affix
            let piece_price = d.tabs[k].map.get(piece).map_or(0.0, |&r| n_whole / uses[k][r as usize].max(1e-9) * d.tabs[k].price[r as usize]);
            let freed = n / uses[0][c as usize] * d.tabs[0].price[c as usize] + piece_price;
            let saving = arow.map_or(0.0, |r| n_whole * (d.lam * d.tabs[k].cost[r as usize] + d.delta + clw(r)));
            let pf = d.price_of(0, &f).max(1e-9);
            let by_freed = (core_dom || affix_dom) && freed > pf;
            let by_self = d.pricing.swap_self > 0.0 && saving > d.pricing.swap_self * pf;
            if by_freed || by_self {
                // value: how many times over the merge pays its price (freed merges first)
                out.push((if by_freed { f64::INFINITY } else { saving / pf }, f));
            }
        }
        out
    });
    prof("sw_merge", tq);
    // best first (ties by string, a total order); a merge found from several pieces keeps its best
    let mut v: Vec<(f64, Vec<u32>)> = found.into_iter().flatten().collect();
    v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then_with(|| a.1.cmp(&b.1)));
    let mut seen: std::collections::HashSet<Vec<u32>, Fast> = Default::default();
    v.into_iter().filter(|(_, f)| seen.insert(f.clone())).map(|(_, f)| f).collect()
}

// ------------------------------------------------------------------ exact candidate scoring

/// Exact gain of each candidate row on its own: sum over segments containing it of
/// f(x) (J_D(x) - J_{D+row}(x)), from up to `occ` sampled segments, scaled to all of them.
fn exact_gains(d: &Dict, c: &Corpus, base: &[f64], cands: &[RowKey], occ: usize, threads: usize) -> Vec<f64> {
    // tries over the candidate strings: raw text for affixes (folded with case_affixes), folded
    // text for cores
    let tq = Instant::now();
    let tries: [Trie; 2] = [0, 1].map(|k| {
        Trie::from_strs(cands.iter().filter(|(t, _)| (*t == 0) == (k == 1)).map(|(_, s)| (&s[..], 0)).collect())
    });
    // candidates ending at each node (a prefix and a suffix can share a string)
    let mut term: [Vec<Vec<u32>>; 2] = [0, 1].map(|k| vec![Vec::new(); tries[k].node.len()]);
    for (ci, (t, s)) in cands.iter().enumerate() {
        let k = if *t == 0 { 1 } else { 0 };
        term[k][tries[k].find(s).unwrap() as usize].push(ci as u32);
    }
    // postings: a uniform random sample of the segments containing each candidate, stratified by
    // thread (each thread's range is a stratum); weight = containing segments / sampled segments
    let ns = c.segs.len();
    let chunk = ns.div_ceil(threads).max(1);
    let cap = occ.div_ceil(threads).max(1);
    let key = |ci: u32, si: u32| {
        let mut h = Fx::default();
        h.write_u32(ci);
        h.write_u32(si);
        h.write_u32(0x9e37_79b9);
        h.finish()
    };
    // 1. the candidates occurring in every segment (blocks of segments handed out on demand)
    const B: usize = 1024;
    let found: Vec<(Vec<u32>, Vec<u32>)> = par_map(ns.div_ceil(B), threads, 1, Vec::new, |hits: &mut Vec<u32>, b| {
        let (mut all, mut ends) = (Vec::new(), Vec::new());
        for x in &c.segs[b * B..((b + 1) * B).min(ns)] {
            hits.clear();
            for i in 0..x.len() {
                for k in 0..2 {
                    let mut n = 0u32;
                    for &ch in &x[i..] {
                        if is_byte(ch) {
                            break;
                        }
                        let sym = if k == 1 || d.ca { d.sym.fold[ch as usize] } else { ch };
                        let Some(m) = tries[k].step(n, sym) else { break };
                        n = m;
                        hits.extend(&term[k][n as usize]);
                    }
                }
            }
            hits.sort_unstable();
            hits.dedup();
            all.extend_from_slice(hits);
            ends.push(all.len() as u32);
        }
        (all, ends)
    });
    prof("eg_find", tq);
    // 2. per stratum, the sample heaps fed in segment order as before (their internal order is the
    //    order of the postings, so it must not change)
    let found = &found;
    let tq = Instant::now();
    // per stratum and range of candidates (so a stratum with many hits is spread over threads):
    // every candidate's sampled segments (flat, candidate by candidate) and ends. A candidate's
    // heap sees the same pushes and pops in the same order as with one thread per stratum.
    const R: usize = 8;
    let per = cands.len().div_ceil(R).max(1);
    let parts: Vec<(Vec<(u32, f64)>, Vec<u32>)> = par_map(threads * R, threads, 1, || (), |_, task| {
        let (t, r) = (task / R, task % R);
        let (lo, hi) = ((r * per) as u32, ((r + 1) * per).min(cands.len()) as u32);
        let n_r = hi.saturating_sub(lo) as usize;
        let mut heaps: Vec<std::collections::BinaryHeap<(u64, u32)>> = (0..n_r).map(|_| Default::default()).collect();
        let mut m_t = vec![0u32; n_r];
        for si in (t * chunk)..((t + 1) * chunk).min(ns) {
            let (all, ends) = &found[si / B];
            let j = si % B;
            let hits = &all[if j == 0 { 0 } else { ends[j - 1] as usize }..ends[j] as usize];
            let from = hits.partition_point(|&h| h < lo);
            for &h in &hits[from..] {
                if h >= hi {
                    break;
                }
                let q = (h - lo) as usize;
                m_t[q] += 1;
                let hp = &mut heaps[q];
                hp.push((key(h, si as u32), si as u32));
                if hp.len() > cap {
                    hp.pop(); // keep the `cap` smallest keys: a uniform sample
                }
            }
        }
        let (mut flat, mut ends) = (Vec::new(), Vec::with_capacity(n_r));
        for (hp, m) in heaps.into_iter().zip(m_t) {
            let w = m as f64 / hp.len().max(1) as f64;
            flat.extend(hp.into_iter().map(|(_, si)| (si, w)));
            ends.push(flat.len() as u32);
        }
        (flat, ends)
    });
    prof("eg_heaps", tq);
    let tq = Instant::now();
    // a candidate's postings: its samples from stratum 0, 1, ... in order
    let parts = &parts;
    let post_of = move |i: usize| {
        let (r, q) = (i / per, i % per);
        (0..threads).flat_map(move |t| {
            let (flat, ends) = &parts[t * R + r];
            flat[if q == 0 { 0 } else { ends[q - 1] as usize }..ends[q] as usize].iter().copied()
        })
    };
    prof("eg_postings", tq);
    let tq = Instant::now();
    // the cost of a new row: the median cost of the live rows in its table
    let med: Vec<f64> = (0..3)
        .map(|k| {
            let mut v: Vec<f64> = (0..d.tabs[k].strs.len()).filter(|&i| d.tabs[k].alive[i]).map(|i| d.tabs[k].cost[i]).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v.get(v.len() / 2).copied().unwrap_or(20.0)
        })
        .collect();
    let med_spec: Vec<f64> = (0..3)
        .map(|k| {
            if k == 0 || d.mu == 0.0 {
                return 0.0;
            }
            let mut v: Vec<f64> = (0..d.tabs[k].strs.len()).filter(|&i| d.tabs[k].alive[i]).map(|i| d.tabs[k].spec[i]).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v.get(v.len() / 2).copied().unwrap_or(0.0)
        })
        .collect();
    prof("eg_median", tq);
    let tq = Instant::now();
    // Segment by segment, the plain parse once, then every candidate sampled there re-scored
    // from it: long segments (whole paragraphs) by parse_x_window, which re-runs only the DP around
    // its occurrences, short ones (chunks) by resuming the plain DP's state where its arcs first
    // change a score (see exact_gains_windowed). Same values as the full re-parses below, so the
    // choice is only about speed.
    let mean_len = c.segs.iter().map(|s| s.len()).sum::<usize>() as f64 / ns.max(1) as f64;
    if d.cl.on {
        let r = exact_gains_pair(d, c, base, cands, &post_of, &med, &med_spec, threads);
        prof("eg_score", tq);
        return r;
    }
    if d.mu2 == 0.0 && d.lamc == 0.0 {
        let r = exact_gains_windowed(d, c, base, cands, &tries, &term, &post_of, &med, &med_spec, threads, mean_len < WINDOW_MIN_LEN);
        prof("eg_score", tq);
        return r;
    }
    let r = par_map(cands.len(), threads, 4, Scratch::new, |sc, i| {
        let (t, st) = &cands[i];
        let e = Extra { t: *t as usize, s: st.clone(), cost: med[*t as usize], spec: med_spec[*t as usize], usec: d.usec_of(*t as usize, st),
                        umask: if *t == 0 || d.ca { d.canon_mask(st) } else { 0 } };
        let mut g = 0.0;
        for (si, w) in post_of(i) {
            let f = c.freqs[si as usize];
            let j = d.parse_x(&c.segs[si as usize], sc, None, None, Some(&e)).0;
            g += w * f * (base[si as usize] - j).max(0.0);
        }
        g - d.price_of(*t as usize, st)
    });
    prof("eg_score", tq);
    r
}

/// exact_gains uses windowed re-scoring when segments are this long on average (chars), else the
/// resumed DP.
const WINDOW_MIN_LEN: f64 = 48.0;

type WinScratch = (Arcs, [Vec<(f64, f64)>; 3], [Vec<(f64, f64)>; 3], [Vec<Back>; 3], Vec<(u32, u32)>, Vec<u32>, [Vec<(f64, f64)>; 3], Vec<f64>);

/// The last step of exact_gains for parse_x (mu2 = lamc = 0), by segment instead of by candidate:
/// the same values (parse_x_window, or with `snap` the resumed DP for segments of up to 256 chars),
/// summed per candidate in the same order.
#[allow(clippy::too_many_arguments)]
fn exact_gains_windowed<P: Iterator<Item = (u32, f64)>>(d: &Dict, c: &Corpus, base: &[f64], cands: &[RowKey], tries: &[Trie; 2],
                                                         term: &[Vec<Vec<u32>>; 2], post_of: &(impl Fn(usize) -> P + Sync),
                                                         med: &[f64], med_spec: &[f64], threads: usize, snap: bool) -> Vec<f64> {
    let ns = c.segs.len();
    let extras: Vec<Extra> = cands
        .iter()
        .map(|(t, s)| Extra { t: *t as usize, s: s.clone(), cost: med[*t as usize], spec: med_spec[*t as usize], usec: d.usec_of(*t as usize, s),
                              umask: if *t == 0 || d.ca { d.canon_mask(s) } else { 0 } })
        .collect();
    let mut out = Vec::with_capacity(cands.len());
    // candidates in batches of about BATCH postings, so memory stays bounded
    const BATCH: usize = 1 << 24;
    let mut lo = 0;
    while lo < cands.len() {
        // the batch's postings, candidate by candidate: (segment, weight, candidate)
        let mut post: Vec<(u32, f64, u32)> = Vec::new();
        let mut ends: Vec<usize> = Vec::new();
        let mut hi = lo;
        while hi < cands.len() && (hi == lo || post.len() < BATCH) {
            post.extend(post_of(hi).map(|(si, w)| (si, w, hi as u32)));
            ends.push(post.len());
            hi += 1;
        }
        // every segment's postings, in posting order (so by candidate; a candidate samples a
        // segment at most once)
        let by_seg = Csr::new(ns, post.iter().map(|p| Some(p.0)));
        let segs: Vec<u32> = (0..ns as u32).filter(|&si| !by_seg.get(si as usize).is_empty()).collect();
        let new_ws = || -> WinScratch { Default::default() };
        let js: Vec<Vec<f64>> = par_map(segs.len(), threads, 1, new_ws, |(arcs, fin, w, nob, hits, occ, snaps, bc): &mut WinScratch, q| {
            let si = segs[q] as usize;
            let x = &c.segs[si];
            let ps = by_seg.get(si);
            // occurrences of the sampled candidates: (index in ps, start), sorted, as exact_gains
            // finds them (its tries: cores folded, affixes raw unless case_affixes; no byte); in a
            // short text by matching each sampled candidate's string directly
            hits.clear();
            if x.len() <= 256 {
                for (r, &p) in ps.iter().enumerate() {
                    let (t, cs) = &cands[post[p as usize].2 as usize];
                    let raw = *t != 0 && !d.ca;
                    let l = cs.len();
                    if l == 0 || l > x.len() {
                        continue;
                    }
                    for i in 0..=x.len() - l {
                        let ok = x[i..i + l].iter().zip(cs).all(|(&ch, &c)| !is_byte(ch) && (if raw { ch } else { d.sym.fold[ch as usize] }) == c);
                        if ok {
                            hits.push((r as u32, i as u32));
                        }
                    }
                }
            } else {
                for i in 0..x.len() {
                    for k in 0..2 {
                        let mut n = 0u32;
                        for &ch in &x[i..] {
                            if is_byte(ch) {
                                break;
                            }
                            let sym = if k == 1 || d.ca { d.sym.fold[ch as usize] } else { ch };
                            let Some(m) = tries[k].step(n, sym) else { break };
                            n = m;
                            for &ci in &term[k][n as usize] {
                                if (lo as u32..hi as u32).contains(&ci) {
                                    if let Ok(r) = ps.binary_search_by_key(&ci, |&p| post[p as usize].2) {
                                        hits.push((r as u32, i as u32));
                                    }
                                }
                            }
                        }
                    }
                }
                hits.sort_unstable();
            }
            // the plain parse, from the text's arcs (walked once for all its candidates)
            d.arcs_x(x, arcs);
            d.be.fill(x, bc);
            let bc = &bc[..];
            if snap && x.len() <= 256 {
                // Short segments. The DP with e does exactly what the plain DP does as long as none
                // of e's arcs wins a relaxation (extra_wins): so where that never happens its score
                // is the plain parse's, fin[0][n], and otherwise its state just before the first
                // position a whose e arcs win is the plain DP's state then. The plain DP runs once
                // from the arcs, testing e's arcs at each of its occurrences as step_x would relax
                // them, and keeps its pending scores (positions >= a; earlier ones are never read
                // again) at every such a; the DP with e then runs on from there: the same operations
                // in the same order as parse_x(x, extra = e), so bit-identical. It stops early, as
                // parse_x_window does, once it is past e's last occurrence and its arcs and the final
                // scores of all positions with arcs into the rest equal the plain parse's (the rest
                // is then the plain DP, whose score is fin[0][n]).
                let n = x.len();
                const NOT: u32 = u32::MAX;
                // occurrences (position, index in ps), by position; per posting its last one
                let mut occs: Vec<(u32, u32)> = hits.iter().map(|&(r, i)| (i, r)).collect();
                occs.sort_unstable();
                occs.dedup();
                let mut last = vec![0u32; ps.len()];
                for &(i, r) in &occs {
                    last[r as usize] = last[r as usize].max(i);
                }
                let mut won = vec![NOT; ps.len()];
                for s in 0..3 {
                    fin[s].clear();
                    fin[s].resize((n + 1) * d.wid(s), INF);
                    snaps[s].clear();
                }
                fin[0][0] = (0.0, 0.0);
                // the plain DP to the end, keeping its pending scores at every position i where some
                // posting's e first wins (at: (i, start in snaps in units of positions)). Only
                // positions i .. i + lx - 1 can have any at time i (lx = the longest arc of the
                // text): the rest are still INF.
                let lx = (0..n)
                    .map(|i| arcs.arcs[arcs.start[3 * i] as usize..arcs.start[3 * i + 3] as usize].iter().map(|a| a.0 as usize - i).max().unwrap_or(0))
                    .max()
                    .unwrap_or(0)
                    .max(1);
                let span = |i: usize| (i + lx - 1).min(n) - i + 1;
                let mut at: Vec<(u32, usize)> = Vec::new();
                let mut o = 0;
                for i in 0..=n {
                    let o0 = o;
                    while o < occs.len() && occs[o].0 as usize == i {
                        o += 1;
                    }
                    let open = occs[o0..o].iter().any(|&(_, r)| won[r as usize] == NOT);
                    let start = snaps[0].len();
                    if open {
                        for s in 0..3 {
                            let k = d.wid(s);
                            let (src, dst) = (&fin[s], &mut snaps[s]);
                            dst.extend_from_slice(&src[i * k..(i + span(i)) * k]);
                        }
                    }
                    let mut any = false;
                    let occ_i = &occs[o0..o];
                    let won_ref = &mut won;
                    d.step_xp::<false>(x, i, fin, nob, 0, None, None, true, Some(arcs), bc, |dd| {
                        for &(_, r) in occ_i {
                            if won_ref[r as usize] == NOT {
                                let e = &extras[post[ps[r as usize] as usize].2 as usize];
                                if d.extra_wins(x, i, dd, 0, e) {
                                    won_ref[r as usize] = i as u32;
                                    any = true;
                                }
                            }
                        }
                    });
                    if any {
                        at.push((i as u32, start));
                    } else if open {
                        for s in 0..3 {
                            let k = d.wid(s);
                            snaps[s].truncate(start * k);
                        }
                    }
                }
                let same = |w: &[Vec<(f64, f64)>; 3], p: usize, off: usize| {
                    (0..3).all(|s| {
                        let k = d.wid(s);
                        (0..k).all(|v| {
                            let (a, b) = (w[s][(p - off) * k + v], fin[s][p * k + v]);
                            a.0.to_bits() == b.0.to_bits() && a.1.to_bits() == b.1.to_bits()
                        })
                    })
                };
                // minsrc[t] = the first position with an arc (a byte, or a trie arc) reaching t or
                // beyond: the pending scores at time t depend only on the final scores of
                // positions minsrc[t] .. t - 1 (and on e's arcs)
                let mut minsrc = vec![0u32; n + 1];
                {
                    let maxreach = |i: usize| {
                        arcs.arcs[arcs.start[3 * i] as usize..arcs.start[3 * i + 3] as usize].iter().map(|a| a.0 as usize).max().unwrap_or(0).max(i + 1)
                    };
                    let (mut i, mut mr) = (0usize, maxreach(0));
                    for t in 1..=n {
                        while mr < t {
                            i += 1;
                            mr = maxreach(i);
                        }
                        minsrc[t] = i as u32;
                    }
                }
                let plain = fin[0][n].0;
                let mut res = vec![plain; ps.len()];
                for r in 0..ps.len() {
                    if won[r] == NOT {
                        continue; // e never wins: the plain parse
                    }
                    let (a, z) = (won[r] as usize, last[r] as usize);
                    let q = at.partition_point(|p| (p.0 as usize) < a);
                    let e = &extras[post[ps[r] as usize].2 as usize];
                    let base = at[q].1;
                    for s in 0..3 {
                        let k = d.wid(s);
                        w[s].clear();
                        w[s].extend_from_slice(&snaps[s][base * k..(base + span(a)) * k]);
                        w[s].resize((n - a + 1) * k, INF);
                    }
                    let reach = z + e.s.len();
                    // run = number of consecutive positions just processed (ending at i) whose final
                    // scores equal the plain parse's (those before a all do)
                    let mut run = a;
                    let mut i = a;
                    res[r] = loop {
                        d.step_x::<false>(x, i, w, nob, a, None, Some(e), true, Some(arcs), bc);
                        if i == n {
                            break w[0][n - a].0;
                        }
                        run = if same(w, i, a) { run + 1 } else { 0 };
                        i += 1;
                        if i > z && reach < i && run >= i - minsrc[i] as usize {
                            break plain; // rejoined at time i
                        }
                    };
                }
                return res;
            }
            // the longest arc of this text (walks and a byte): bounds how far back the pending
            // scores at any time come from, so parse_x_window may use it for lmax
            let seg_lmax = (0..x.len())
                .map(|i| arcs.arcs[arcs.start[3 * i] as usize..arcs.start[3 * i + 3] as usize].iter().map(|a| a.0 as usize - i).max().unwrap_or(0))
                .max()
                .unwrap_or(0)
                .max(1);
            d.scores_x(x, arcs, fin, nob, bc);
            let mut h = 0;
            (0..ps.len())
                .map(|r| {
                    occ.clear();
                    while h < hits.len() && hits[h].0 as usize == r {
                        occ.push(hits[h].1);
                        h += 1;
                    }
                    let e = &extras[post[ps[r] as usize].2 as usize];
                    d.parse_x_window(x, arcs, fin, occ, seg_lmax.max(e.s.len()), e, w, nob, bc)
                })
                .collect()
        });
        let mut jv = vec![0.0f64; post.len()];
        for (q, &si) in segs.iter().enumerate() {
            for (&p, &j) in by_seg.get(si as usize).iter().zip(&js[q]) {
                jv[p as usize] = j;
            }
        }
        let mut a = 0;
        for (i, &b) in (lo..hi).zip(&ends) {
            let mut g = 0.0;
            for p in a..b {
                let (si, w, _) = post[p];
                let f = c.freqs[si as usize];
                g += w * f * (base[si as usize] - jv[p]).max(0.0);
            }
            a = b;
            let (t, st) = &cands[i];
            out.push(g - d.price_of(*t as usize, st));
        }
        lo = hi;
    }
    out
}

/// The last step of exact_gains with code_len (the pairwise DP), by segment: every sampled
/// segment is parsed once, and a candidate is re-parsed with its row only where its arcs would
/// change a score (Dict::extra_pair_wins); elsewhere its score is the plain parse's. Summed per
/// candidate in the postings' order, as exact_gains_windowed.
fn exact_gains_pair<P: Iterator<Item = (u32, f64)>>(d: &Dict, c: &Corpus, base: &[f64], cands: &[RowKey],
                                                     post_of: &(impl Fn(usize) -> P + Sync), med: &[f64], med_spec: &[f64],
                                                     threads: usize) -> Vec<f64> {
    let ns = c.segs.len();
    let extras: Vec<Extra> = cands
        .iter()
        .map(|(t, s)| Extra { t: *t as usize, s: s.clone(), cost: med[*t as usize], spec: med_spec[*t as usize], usec: d.usec_of(*t as usize, s),
                              umask: if *t == 0 || d.ca { d.canon_mask(s) } else { 0 } })
        .collect();
    let mut out = Vec::with_capacity(cands.len());
    // the longest arc of any row (every pending entry at a position lies within this of the
    // positions already processed)
    let lmax = (0..3).flat_map(|k| d.tabs[k].strs.iter().map(|s| s.len())).max().unwrap_or(1).max(1);
    // do two parse_pair states hold the same entries at position p (all that later positions read)?
    let pair_same = |a: &Scratch, b: &Scratch, p: usize| -> bool {
        let eq = |x: (f64, f64), y: (f64, f64)| x.0.to_bits() == y.0.to_bits() && x.1.to_bits() == y.1.to_bits();
        eq(a.bs[p], b.bs[p])
            && a.pe[p].len() == b.pe[p].len()
            && a.pe[p].iter().zip(&b.pe[p]).all(|(x, y)| x.pid == y.pid && x.vs == y.vs && eq(x.s, y.s))
            && a.ce[p].len() == b.ce[p].len()
            && a.ce[p].iter().zip(&b.ce[p]).all(|(x, y)| x.cid == y.cid && x.v == y.v && x.vs == y.vs && eq(x.s, y.s))
    };
    const BATCH: usize = 1 << 24;
    let mut lo = 0;
    let (mut n_post, n_full) = (0usize, std::sync::atomic::AtomicUsize::new(0));
    while lo < cands.len() {
        let mut post: Vec<(u32, f64, u32)> = Vec::new();
        let mut ends: Vec<usize> = Vec::new();
        let mut hi = lo;
        while hi < cands.len() && (hi == lo || post.len() < BATCH) {
            post.extend(post_of(hi).map(|(si, w)| (si, w, hi as u32)));
            ends.push(post.len());
            hi += 1;
        }
        n_post += post.len();
        let by_seg = Csr::new(ns, post.iter().map(|p| Some(p.0)));
        let segs: Vec<u32> = (0..ns as u32).filter(|&si| !by_seg.get(si as usize).is_empty()).collect();
        type PairScratch = (Scratch, Scratch, Vec<(usize, u32, u8, f64, f64)>, Vec<(usize, u32, f64, f64, u8)>);
        let new_ps = || -> PairScratch { (Scratch::new(), Scratch::new(), Vec::new(), Vec::new()) };
        let n_full = &n_full;
        let js: Vec<Vec<f64>> = par_map(segs.len(), threads, 1, new_ps, |(sc, sc2, cm, sm): &mut PairScratch, q| {
            let si = segs[q] as usize;
            let x = &c.segs[si];
            let n = x.len();
            let ps = by_seg.get(si);
            // every posting's first and last position where its row has arcs
            let occ: Vec<Option<(usize, usize)>> = ps
                .iter()
                .map(|&p| {
                    let e = &extras[post[p as usize].2 as usize];
                    let mut it = (0..n).filter(|&i| d.extra_fits(x, i, e));
                    let a = it.next()?;
                    Some((a, it.last().unwrap_or(a)))
                })
                .collect();
            let mut firsts: Vec<usize> = occ.iter().flatten().map(|o| o.0).collect();
            firsts.sort_unstable();
            firsts.dedup();
            let lmax = if firsts.is_empty() { lmax } else { d.longest_arc(x) };
            // the plain parse (exactly parse_pair), keeping its pending state (the entries of every
            // position the arcs of the positions before can reach) at every first position
            type Snap = (Vec<(f64, f64)>, Vec<Vec<PEnt>>, Vec<Vec<CEnt>>);
            let mut snaps: Vec<Snap> = Vec::with_capacity(firsts.len());
            d.be.fill(x, &mut sc.bc);
            sc2.bc.clone_from(&sc.bc);
            d.pair_init(n, sc);
            let mut f = 0;
            for i in 0..=n {
                if f < firsts.len() && firsts[f] == i {
                    let hi = (i + lmax).min(n);
                    snaps.push((sc.bs[i..=hi].to_vec(), sc.pe[i..=hi].to_vec(), sc.ce[i..=hi].to_vec()));
                    f += 1;
                }
                d.pair_step(x, i, sc, None, None);
            }
            let plain = d.pair_finish(x, sc, None).0;
            ps.iter()
                .zip(&occ)
                .map(|(&p, o)| {
                    let e = &extras[post[p as usize].2 as usize];
                    let Some((a, z)) = *o else { return plain };
                    if !d.extra_pair_wins(x, sc, e, cm, sm) {
                        return plain;
                    }
                    n_full.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    // parse_pair(x, extra = e): the same DP as the plain one until position a, e's
                    // first, so it resumes from the plain one's pending state there; and once past
                    // e's last arcs, if the final entries of every position whose arcs reach further
                    // are the plain parse's, the rest is the plain DP too (its score is plain)
                    let (bs, pe, ce) = &snaps[firsts.binary_search(&a).unwrap()];
                    d.pair_init(n, sc2);
                    for (k, q) in (a..).zip(0..bs.len()) {
                        sc2.bs[k] = bs[q];
                        sc2.pe[k].clone_from(&pe[q]);
                        sc2.ce[k].clone_from(&ce[q]);
                    }
                    let reach = z + e.s.len();
                    for i in a..=n {
                        d.pair_step(x, i, sc2, None, Some(e));
                        if i >= reach && i < n && (a.max((i + 1).saturating_sub(lmax))..=i).all(|p| pair_same(sc2, sc, p)) {
                            return plain;
                        }
                    }
                    d.pair_finish(x, sc2, None).0
                })
                .collect()
        });
        let mut jv = vec![0.0f64; post.len()];
        for (q, &si) in segs.iter().enumerate() {
            for (&p, &j) in by_seg.get(si as usize).iter().zip(&js[q]) {
                jv[p as usize] = j;
            }
        }
        let mut a = 0;
        for (i, &b) in (lo..hi).zip(&ends) {
            let mut g = 0.0;
            for p in a..b {
                let (si, w, _) = post[p];
                let f = c.freqs[si as usize];
                g += w * f * (base[si as usize] - jv[p]).max(0.0);
            }
            a = b;
            let (t, st) = &cands[i];
            out.push(g - d.price_of(*t as usize, st));
        }
        lo = hi;
    }
    eprintln!("    (code_len re-scoring: {} of {n_post} sampled (candidate, segment) pairs re-parsed with the row)", n_full.load(std::sync::atomic::Ordering::Relaxed));
    out
}

// ------------------------------------------------------------------ io

struct Reader {
    buf: Vec<u8>,
    pos: usize,
}

impl Reader {
    fn open(path: &str) -> Self {
        Reader { buf: std::fs::read(path).expect("read input"), pos: 0 }
    }
    fn take<const N: usize>(&mut self) -> [u8; N] {
        let out: [u8; N] = self.buf[self.pos..self.pos + N].try_into().unwrap();
        self.pos += N;
        out
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take::<4>())
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.take::<8>())
    }
    fn i64(&mut self) -> i64 {
        i64::from_le_bytes(self.take::<8>())
    }
    fn f64(&mut self) -> f64 {
        f64::from_le_bytes(self.take::<8>())
    }
    fn u32s(&mut self, n: usize) -> Vec<u32> {
        (0..n).map(|_| self.u32()).collect()
    }
    fn symbols(&mut self, n: usize) -> Symbols {
        let mut s = Symbols { fold: self.u32s(n), flag: self.u32s(n), upper_of: self.u32s(n), trad_of: self.u32s(n), nbytes: vec![], alnum: vec![], cf: vec![], mark_rule: false };
        s.init_case();
        s
    }
    fn table(&mut self) -> Vec<Vec<u32>> {
        let n = self.u32() as usize;
        (0..n)
            .map(|_| {
                let l = self.u32() as usize;
                self.u32s(l)
            })
            .collect()
    }
    fn corpus(&mut self, with_counts: bool) -> Corpus {
        let n = self.u64() as usize;
        let _ = self.u64();
        let freqs = if with_counts { (0..n).map(|_| self.i64() as f64).collect() } else { vec![1.0; n] };
        let lens: Vec<usize> = (0..n).map(|_| self.u32() as usize).collect();
        let segs = lens.iter().map(|&l| self.u32s(l)).collect();
        Corpus { segs, freqs }
    }
}

fn put(w: &mut impl Write, xs: &[u32]) {
    for x in xs {
        w.write_all(&x.to_le_bytes()).unwrap();
    }
}

fn save(d: &Dict, n_sym: usize, path: &str) {
    let mut w = BufWriter::new(std::fs::File::create(path).expect("create model"));
    if d.case {
        put(&mut w, &[if d.ca { CAFX_MAGIC } else { CASE_MAGIC }]);
    }
    put(&mut w, &[n_sym as u32]);
    for a in [&d.sym.fold, &d.sym.flag, &d.sym.upper_of, &d.sym.trad_of] {
        put(&mut w, a);
    }
    let keep: Vec<Vec<usize>> =
        (0..3).map(|k| (0..d.tabs[k].strs.len()).filter(|&i| d.tabs[k].alive[i]).collect()).collect();
    let mut can = Vec::new();
    for k in 0..3 {
        put(&mut w, &[keep[k].len() as u32]);
        for &i in &keep[k] {
            put(&mut w, &[d.tabs[k].strs[i].len() as u32]);
            if k == 0 && d.case {
                // the canonical spelling (variation 0)
                can.clear();
                d.write_core(i, 0, &mut can);
                put(&mut w, &can);
            } else if d.ca {
                // case_affixes: affixes store their canonical spellings too
                can.clear();
                d.write_parts(&[(&d.tabs[k].strs[i], d.tabs[k].umask[i])], false, 0, &mut can);
                put(&mut w, &can);
            } else {
                put(&mut w, &d.tabs[k].strs[i]);
            }
        }
    }
    for k in 0..3 {
        for &i in &keep[k] {
            w.write_all(&d.tabs[k].cost[i].to_le_bytes()).unwrap();
        }
    }
    for c in &d.vcost[..d.nvar()] {
        w.write_all(&c.to_le_bytes()).unwrap();
    }
    w.write_all(&d.delta.to_le_bytes()).unwrap();
    put(&mut w, &d.sym.nbytes);
    w.write_all(&d.lam.to_le_bytes()).unwrap();
    w.write_all(&d.mu.to_le_bytes()).unwrap();
    for k in 1..3 {
        for &i in &keep[k] {
            w.write_all(&d.tabs[k].spec[i].to_le_bytes()).unwrap();
        }
    }
    // pair tables, renumbered like the rows
    let new_id: Vec<HashMap<u32, u32, Fast>> =
        (0..3).map(|k| keep[k].iter().enumerate().map(|(n, &i)| (i as u32, n as u32)).collect()).collect();
    w.write_all(&d.mu2.to_le_bytes()).unwrap();
    w.write_all(&d.tau.to_le_bytes()).unwrap();
    for (k, (ta, tb)) in [(1usize, 0usize), (0, 2)].into_iter().enumerate() {
        let mut rows: Vec<(u32, u32, f32)> = d.pmi[k]
            .iter()
            .filter_map(|(&(a, b), &v)| Some((*new_id[ta].get(&a)?, *new_id[tb].get(&b)?, v)))
            .collect();
        rows.sort_by(|x, y| (x.0, x.1).cmp(&(y.0, y.1)));
        put(&mut w, &[rows.len() as u32]);
        for (a, b, v) in rows {
            put(&mut w, &[a, b]);
            w.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    // conditional affix statistics
    w.write_all(&d.lamc.to_le_bytes()).unwrap();
    let nc: Vec<f32> = keep[0].iter().map(|&i| d.ncore.get(i).copied().unwrap_or(0.0)).collect();
    put(&mut w, &[nc.len() as u32]);
    for v in nc {
        w.write_all(&v.to_le_bytes()).unwrap();
    }
    for (k, (ta, tb)) in [(1usize, 0usize), (0, 2)].into_iter().enumerate() {
        let mut rows: Vec<(u32, u32, f32)> = d.cn[k]
            .iter()
            .filter_map(|(&(a, b), &v)| Some((*new_id[ta].get(&a)?, *new_id[tb].get(&b)?, v)))
            .collect();
        rows.sort_by(|x, y| (x.0, x.1).cmp(&(y.0, y.1)));
        put(&mut w, &[rows.len() as u32]);
        for (a, b, v) in rows {
            put(&mut w, &[a, b]);
            w.write_all(&v.to_le_bytes()).unwrap();
        }
    }
    w.write_all(&d.nu.to_le_bytes()).unwrap();
    for k in 0..3 {
        for &i in &keep[k] {
            w.write_all(&d.tabs[k].glue[i].to_le_bytes()).unwrap();
        }
    }
    if d.sym.mark_rule {
        put(&mut w, &[MARK_MAGIC]);
        put(&mut w, &d.sym.alnum);
    }
    if !d.cl.on && d.cl.wu > 0.0 {
        // code_w without code_len: only w and the bits per fallback byte
        put(&mut w, &[CODE_MAGIC, 0]);
        w.write_all(&d.cl.wu.to_le_bytes()).unwrap();
        w.write_all(&d.cl.byte.to_le_bytes()).unwrap();
    }
    if d.cl.on {
        // code length statistics (live rows renumbered; pairs of dead rows dropped)
        put(&mut w, &[CODE_MAGIC, 1]);
        w.write_all(&d.cl.w.to_le_bytes()).unwrap();
        w.write_all(&d.cl.byte.to_le_bytes()).unwrap();
        // (all values are multiples of 1/1024 bit below 64: u16, exact)
        let q16 = |v: f32| ((v as f64 * CL_Q).round() as u16).to_le_bytes();
        for &i in &keep[0] {
            for b in d.cl.bo.get(i).copied().unwrap_or([0.0; 3]) {
                w.write_all(&q16(b)).unwrap();
            }
        }
        let mut buf: Vec<u8> = Vec::new();
        for k in 0..3 {
            // (core, affix, bits) with the new ids, by core then affix
            let mut rows: Vec<(u32, u32, f32)> = d.cl.pair[k]
                .iter()
                .filter_map(|(&(a, b), &v)| {
                    Some(match k {
                        0 => (*new_id[0].get(&b)?, *new_id[1].get(&a)?, v),
                        1 => (*new_id[0].get(&a)?, *new_id[2].get(&b)?, v),
                        _ => (*new_id[0].get(&a)?, b, v),
                    })
                })
                .collect();
            rows.sort_by(|x, y| (x.0, x.1).cmp(&(y.0, y.1)));
            // per core row: varint count, then per pair varint affix-id gap and u16 bits
            buf.clear();
            let mut r = 0;
            for c in 0..keep[0].len() as u32 {
                let from = r;
                while r < rows.len() && rows[r].0 == c {
                    r += 1;
                }
                put_var(&mut buf, (r - from) as u32);
                let mut prev = 0u32;
                for (j, &(_, a, v)) in rows[from..r].iter().enumerate() {
                    put_var(&mut buf, if j == 0 { a } else { a - prev - 1 });
                    buf.extend_from_slice(&q16(v));
                    prev = a;
                }
            }
            put(&mut w, &[buf.len() as u32]);
            w.write_all(&buf).unwrap();
        }
    }
    if d.be.w > 0.0 {
        // boundary statistics: w, order, the two fallbacks, the percentile table, then the kept
        // contexts by increasing key as (varint key gap, u8 entropy)
        let be = d.be;
        put(&mut w, &[BE_MAGIC]);
        w.write_all(&be.w.to_le_bytes()).unwrap();
        put(&mut w, &[be.order as u32, be.med[0] as u32, be.med[1] as u32, be.ncls as u32, BE_NS as u32]);
        for v in &be.cdf {
            w.write_all(&v.to_le_bytes()).unwrap();
        }
        put(&mut w, &[be.grp.len() as u32]);
        w.write_all(&be.grp).unwrap();
        let mut rows: Vec<(u32, u8)> = be.tab.iter().map(|(&k, &v)| (k, v)).collect();
        rows.sort_unstable();
        let mut buf: Vec<u8> = Vec::with_capacity(3 * rows.len());
        for (j, &(k, v)) in rows.iter().enumerate() {
            put_var(&mut buf, if j == 0 { k } else { k - rows[j - 1].0 - 1 });
            buf.push(v);
        }
        put(&mut w, &[rows.len() as u32, buf.len() as u32]);
        w.write_all(&buf).unwrap();
    }
}

/// LEB128 varint
fn put_var(buf: &mut Vec<u8>, mut x: u32) {
    while x >= 0x80 {
        buf.push((x as u8) | 0x80);
        x >>= 7;
    }
    buf.push(x as u8);
}

/// LEB128 varint at buf[*at..]
fn get_var(buf: &[u8], at: &mut usize) -> u32 {
    let (mut x, mut sh) = (0u32, 0);
    loop {
        let b = buf[*at];
        *at += 1;
        x |= ((b & 0x7f) as u32) << sh;
        if b < 0x80 {
            return x;
        }
        sh += 7;
    }
}

/// code_len: the dense per-core variation bits from the backoff offsets and the observed pairs
/// (exactly as update_codelen builds them)
fn codelen_vb(bo: &[[f32; 3]], pair: &HashMap<(u32, u32), f32, Fast>, vcost: &[f64; NVAR_MAX], nv: usize) -> Vec<[f64; NVAR_MAX]> {
    let mut vb: Vec<[f64; NVAR_MAX]> =
        bo.iter().map(|b| std::array::from_fn(|v| if v < nv { vcost[v] + b[2] as f64 } else { f64::INFINITY })).collect();
    for (&(c, v), &b) in pair {
        vb[c as usize][v as usize] = b as f64;
    }
    vb
}

// ------------------------------------------------------------------ main loops

fn train(args: &[String]) {
    let tr0 = Instant::now();
    let mut r = Reader::open(&args[2]);
    let n_sym = r.u32() as usize;
    let mut sym = r.symbols(n_sym);
    sym.nbytes = r.u32s(n_sym);
    sym.alnum = r.u32s(n_sym);
    let budget = r.u32() as usize;
    let expand = r.u32() as f64 / 1000.0;
    let min_freq = r.u32();
    let max_len = r.u32() as usize;
    let em_iters = r.u32() as usize;
    let max_rounds = r.u32() as usize;
    let threads = r.u32() as usize;
    let gamma = r.u32() as f64 / 1000.0;
    let max_packages = r.u32() as usize;
    let protect_chars = r.u32() != 0;
    let first_expand = r.u32() as f64 / 1000.0;
    let step = r.u32() as f64 / 1000.0;
    let delta = r.u32() as f64 / 1000.0;
    let lam = r.u32() as f64 / 1000.0;
    let rerank_mult = r.u32() as usize;
    let rerank_occ = r.u32() as usize;
    let refactor = r.u32() != 0;
    let mu = r.u32() as f64 / 1000.0;
    let h0 = r.u32() as f64 / 1000.0;
    let mu2 = r.u32() as f64 / 1000.0;
    let tau = r.u32() as f64 / 1000.0;
    let alnum_only = r.u32() != 0;
    let lamc = r.u32() as f64 / 1000.0;
    let nu = r.u32() as f64 / 1000.0;
    let frag_t = r.u32() as f64 / 1000.0;
    let glue_h0 = r.u32() as f64 / 1000.0;
    let price_pi = r.u32() as f64 / 1000.0;
    let prod_k = r.u32() as f64;
    let conc = r.u32() as f64 / 1000.0;
    let pi_core = r.u32() as f64 / 1000.0;
    let min_gain = r.u32() as f64;
    let rare = r.u32() as f64 / 1000.0;
    let partner_n0 = r.u32() as f64;
    let sample_t = r.u32() as f64;
    let fine_rounds = r.u32() as usize;
    let coarse_gain = r.u32() as f64;
    let sig_k = r.u32() as f64;
    let sig_share = r.u32() as f64 / 1000.0;
    let swap_share = r.u32() as f64 / 1000.0;
    let case = r.u32() != 0;
    sym.mark_rule = r.u32() != 0;
    let sig_soft = r.u32() != 0;
    let swap_self = r.u32() as f64 / 1000.0;
    let ca = r.u32() != 0;
    assert!(!ca || case, "case_affixes needs case_cores");
    let code_len = r.u32();
    let code_w = r.u32() as f64 / 1000.0;
    let code_beta = r.u32() as f64 / 1000.0;
    let prune_reuse = r.u32() != 0;
    let prune_tail = r.u32() as f64 / 1e6;
    let be_w = r.u32() as f64 / 1000.0;
    let be_order = match r.u32() {
        0 => 1,
        o => o as usize,
    };
    assert!(be_order <= 8, "be_order is at most 8");
    assert!(code_len <= 1, "code_len is 0 or 1");
    // code_w without code_len: w x the unconditional bits on the 3-state DP (no pairwise terms)
    assert!(code_w == 0.0 || code_len == 1 || (mu2 == 0.0 && lamc == 0.0), "code_w_permille without code_len needs mu2 = lamc = 0");
    let cl0 = CodeLen { on: code_len == 1, w: code_w, beta: code_beta, wu: if code_len == 1 { 0.0 } else { code_w }, ..Default::default() };
    let cl_on = cl0.on;
    let full = r.corpus(true);
    let dev = r.corpus(true);
    let classes = r.u32() != 0;
    let mut allow: [std::collections::HashSet<Vec<u32>, Fast>; 3] = Default::default();
    for k in 1..3 {
        // (case_affixes: affix rows are folded strings, and so are their allowed parts)
        allow[k] = r.table().into_iter().map(|s| if ca { s.iter().map(|&c| sym.fold[c as usize]).collect() } else { s }).collect();
    }
    let init: Option<[Vec<Vec<u32>>; 3]> = if r.u32() != 0 { Some([r.table(), r.table(), r.table()]) } else { None };
    let script: Vec<u8> = if r.pos < r.buf.len() && r.u32() != 0 { r.u32s(n_sym).into_iter().map(|g| g as u8).collect() } else { Vec::new() };
    drop(r);
    eprintln!("  input read in {:.1}s", tr0.elapsed().as_secs_f64());
    let t0 = Instant::now();
    let chars_of = |c: &Corpus| -> f64 { c.segs.iter().zip(&c.freqs).map(|(s, f)| s.len() as f64 * f).sum() };
    let dev_chars = chars_of(&dev);

    // raw text for the substring index (segments separated by NONE; repeated segments once);
    // always the full corpus, so proposals and raw-text statistics are exact in every phase
    let mut x: Vec<u32> = Vec::new();
    let mut wt: Vec<u32> = Vec::new();
    for (s, &f) in full.segs.iter().zip(&full.freqs) {
        x.extend(s);
        x.push(NONE);
        wt.extend(std::iter::repeat(f as u32).take(s.len() + 1));
    }
    // coarse phase corpus (sample_t): see sample_corpus
    let sampled = (sample_t > 1.0).then(|| sample_corpus(&full, sample_t));
    let mut train: &Corpus = sampled.as_ref().unwrap_or(&full);
    let mut chars = chars_of(train);
    if let Some(s) = &sampled {
        eprintln!("  coarse phase on a sample (t {sample_t}): {} of {} unique segments, {:.1}% of their chars; then up to {fine_rounds} rounds on the full corpus",
                  s.segs.len(), full.segs.len(),
                  100.0 * s.segs.iter().map(|x| x.len()).sum::<usize>() as f64 / full.segs.iter().map(|x| x.len()).sum::<usize>().max(1) as f64);
    }
    let mut char_n = vec![1.0f64; n_sym];
    for (s, f) in full.segs.iter().zip(&full.freqs) {
        for &c in s {
            if !is_byte(c) {
                char_n[c as usize] += f;
            }
        }
    }
    let tot: f64 = char_n.iter().sum();
    let char_bits: Vec<f64> = char_n.iter().map(|n| -(n / tot).log2()).collect();
    eprintln!("  mining substrings (min count {min_freq}, max length {max_len})");
    let subs = mine(&x, &wt, min_freq, max_len, threads);
    eprintln!("  {} frequent substrings in {:.0}s", subs.len(), t0.elapsed().as_secs_f64());
    let excl = exclusive_counts(&sym, &x, &subs, threads);
    let frag = if nu > 0.0 || price_pi > 0.0 || pi_core > 0.0 { fragment_scores(&sym, &x, &wt, &subs, frag_t, glue_h0, min_freq) } else { Default::default() };
    let canon = if case { canonical_masks(&sym, &x, &wt, &subs, ca) } else { HashMap::default() };
    if case {
        eprintln!("  case_cores: {} folded strings have a capitalised canonical spelling{}", canon.len(),
                  if ca { " (case_affixes: affixes too, whole-token variations)" } else { "" });
    }
    let mut pricing = Pricing {
        pi: price_pi, pi_core, k: prod_k, conc, char_bits: char_bits.clone(), dstems: Default::default(), rare, n0: partner_n0,
        script: Vec::new(), script_chars: Vec::new(), rawcnt: HashMap::default(), min_freq: min_freq as f64,
        sig_k, sig_share, sig_soft, swap_self, swap_share: if swap_share > 0.0 { swap_share } else { 0.5 },
    };
    if rare > 0.0 {
        // raw counts of every char and frequent substring, and chars per script group
        let groups = script.iter().map(|&g| g as usize + 1).max().unwrap_or(1);
        let mut sc = vec![0.0f64; groups];
        let mut uni = vec![0u64; n_sym];
        for (i, &c) in x.iter().enumerate() {
            if c != NONE && !is_byte(c) {
                uni[c as usize] += wt[i] as u64;
            }
        }
        for (c, &n) in uni.iter().enumerate() {
            if n > 0 {
                pricing.rawcnt.insert(seq_hash(std::iter::once(c as u32)), n);
                sc[script.get(c).copied().unwrap_or(0) as usize] += n as f64;
            }
        }
        for &(p, l, c) in &subs {
            pricing.rawcnt.insert(seq_hash(x[p as usize..p as usize + l as usize].iter().copied()), c as u64);
        }
        eprintln!("  rarity cost (rare {rare}): {} scripts, chars per script {:?}", groups,
                  sc.iter().map(|v| format!("{:.1}M", v / 1e6)).collect::<Vec<_>>());
        pricing.script = if script.is_empty() { vec![0; n_sym] } else { script.clone() };
        pricing.script_chars = sc;
    }
    if price_pi > 0.0 || pi_core > 0.0 {
        let tp = Instant::now();
        pricing.dstems = affix_stems(&x, &subs, threads);
        eprintln!("  row prices (pi {price_pi}, pi_core {pi_core}, k {prod_k}, conc {conc}): free-stem counts for {} prefix / {} suffix strings ({:.0}s)",
                  pricing.dstems[0].len(), pricing.dstems[1].len(), tp.elapsed().as_secs_f64());
    }
    if nu > 0.0 || price_pi > 0.0 || pi_core > 0.0 {
        eprintln!("  glue scores (nu {nu}, t {frag_t}, affix h0 {glue_h0}): {} core strings, {} prefix strings, {} suffix strings glued to one neighbour",
                  frag[0].len(), frag[1].len(), frag[2].len());
    }

    // branching-entropy boundary statistics (be_permille), from the full training corpus
    let be = if be_w > 0.0 { build_be(&full, be_order, be_w, &script, threads) } else { BoundEnt::default() };

    // initial dictionary: the alphabet as cores (folded), empty affixes
    let mut tabs = [Table::new(false), Table::new(true), Table::new(true)];
    for i in 0..n_sym as u32 {
        // (mark_rule: not a combining mark, which no core may start with: such a row could never be
        // used, and single chars are protected from pruning, so it held its slot to the end)
        if sym.fold[i as usize] == i && !sym.mark_at(i) {
            tabs[0].add(&[i]);
        }
    }
    if let Some(init) = &init {
        for k in 0..3 {
            for row in &init[k] {
                // cores are keyed folded (a case_cores model's table holds canonical spellings), and
                // affixes too with case_affixes
                let f: Vec<u32> = if k == 0 || ca { row.iter().map(|&c| sym.fold[c as usize]).collect() } else { row.clone() };
                tabs[k].add(&f);
            }
        }
        eprintln!("  starting from {} cores, {} prefixes, {} suffixes", tabs[0].live(), tabs[1].live(), tabs[2].live());
    }
    let empty = Trie::empty;
    let mut inc = Dict {
        sym: &sym, tabs, vcost: [0.0; NVAR_MAX], case, ca, canon: &canon, delta, lam, mu, h0, alnum_only, classes, allow: allow.clone(), mu2, tau,
        lamc, cn: Default::default(), ncore: Vec::new(),
        pmi: [HashMap::default(), HashMap::default()], tries: [empty(), empty(), empty()], nu, frag: &frag, pricing: &pricing,
        cl: cl0, be: &be,
    };
    inc.rebuild();
    let st = parse_corpus(&inc, &train, threads, cl_on);
    update_costs(&mut inc, &st);
    update_codelen(&mut inc, &st);
    let st = parse_corpus(&inc, &train, threads, true);
    let mut inc_score = (st.cost + inc.total_price(), description_length(&inc, &st, &char_bits, gamma));
    eprintln!("  start: rows {} chars/token {:.3}", inc.rows(), chars / st.tokens);
    eprintln!("    prof (setup, {:.1}s since input read): {}", t0.elapsed().as_secs_f64(), prof_dump());
    // the incumbent's parse under its final costs: exactly what a fresh parse of the incumbent at
    // the start of the next round would give, so it is reused there (lossless)
    let mut inc_st = st;
    let mut best_dev = f64::INFINITY;
    let mut fails = 0;
    let mut k_extra = (expand * budget as f64) as usize;

    // rows that were added and then pruned in the last round are not proposed again next round,
    // so the expansion budget goes to new candidates instead of re-trying the same rejects
    let mut tabu: std::collections::HashSet<RowKey, Fast> = Default::default();
    let trace = load_trace();
    // coarse phase (on the sample): until max_rounds or a stop; then fine_rounds on the full corpus
    let mut coarse = sampled.is_some();
    let mut fine_left = fine_rounds;
    for round in 0..max_rounds + if coarse { fine_rounds } else { 0 } {
        let tr = Instant::now();
        let tp = Instant::now();
        let failed: std::collections::HashSet<Vec<u32>, Fast> =
            tabu.iter().map(|(t, s)| std::iter::once(*t as u32).chain(s.iter().copied()).collect()).collect();
        let mut props = propose(&inc, &x, &subs, &excl, round > 0, max_packages, threads, &failed);
        let t_prop = tp.elapsed().as_secs_f64();
        if refactor && (lam > 0.0 || nu > 0.0 || code_w > 0.0) {
            let tq = Instant::now();
            let mut extra = refactor_proposals(&inc, &inc_st, threads);
            prof("refactor_props", tq);
            let tq = Instant::now();
            let n_extra: usize = extra.iter().map(|m| m.len()).sum();
            // merge: a core proposed both ways gets the sum of its values
            let nsh = extra.len();
            for (v, k) in props.iter_mut() {
                if k.0 == 0 {
                    if let Some(e) = extra[shard_of(&k.1, nsh)].remove(&k.1) {
                        *v += e;
                    }
                }
            }
            for m in extra {
                props.extend(m.into_iter().map(|(k, v)| (0.0 + v, (0u8, k))));
            }
            prof("refactor_merge", tq);
            eprintln!("    {n_extra} refactoring proposals (absorb affixes into the core)");
        }
        let tq = Instant::now();
        let swaps = swap_proposals(&inc, &inc_st, threads);
        for (i, s) in trace.iter().enumerate() {
            let f = fold_str(&inc, s);
            if swaps.contains(&f) {
                eprintln!("    trace[{i}] swap proposed{}", if tabu.contains(&(0u8, f.clone())) { " (but tabu)" } else { "" });
            }
        }
        prof("swap_props", tq);
        let tq = Instant::now();
        let mut work = Dict {
            sym: &sym, tabs: inc.tabs.clone(), vcost: inc.vcost, case, ca, canon: &canon, delta, lam, mu, h0, alnum_only, classes, allow: allow.clone(), mu2, tau,
            lamc, cn: inc.cn.clone(), ncore: inc.ncore.clone(),
            pmi: inc.pmi.clone(), tries: [empty(), empty(), empty()], nu, frag: &frag, pricing: &pricing,
            cl: inc.cl.clone(), be: &be,
        };
        prof("clone_dict", tq);
        let tq = Instant::now();
        let target = if round == 0 { (budget as f64 * first_expand) as usize } else { budget + k_extra };
        let need = target.saturating_sub(inc.rows());
        let rerank = rerank_mult > 0 && round > 0;
        // proposals best first (value, then key: a total order). Every proposal is a row the incumbent
        // lacks, so the adding below stops after `need` of them (after the re-scored shortlist when
        // re-ranking): only that many have to be found and sorted, not the millions proposed.
        for (i, t) in trace.iter().enumerate() {
            let f = fold_str(&inc, t);
            let rank = props.iter().filter(|p| p.1 .0 == 0 && p.1 .1 == f).map(|p| p.0).next();
            let better = rank.map(|v| props.iter().filter(|p| p.0 > v).count());
            eprintln!("    trace[{i}] as core: incumbent row {}, proposed {:?} (rank {:?} of {}), tabu {}, need {}",
                      inc.tabs[0].has(&f), rank, better, props.len(), tabu.contains(&(0u8, f.clone())), target.saturating_sub(inc.rows()));
        }
        if !tabu.is_empty() {
            let keep = par_map(props.len(), threads, 4096, || (), |_, i| !tabu.contains(&props[i].1));
            let mut i = 0;
            props.retain(|_| {
                i += 1;
                keep[i - 1]
            });
        }
        let by_value = |a: &(f64, RowKey), b: &(f64, RowKey)| b.0.partial_cmp(&a.0).unwrap().then_with(|| a.1.cmp(&b.1));
        let n_top = (if rerank { need * rerank_mult } else { 0 }).max(need).min(props.len());
        if n_top < props.len() {
            props.select_nth_unstable_by(n_top, by_value);
        }
        let (top, rest) = props.split_at_mut(n_top);
        top.sort_by(by_value);
        let mut order: Vec<&RowKey> = top.iter().map(|(_, k)| k).collect();
        prof("tabu_filter", tq);
        // exact in-context re-scoring of the shortlist (rounds >= 1)
        if rerank {
            let m = (need * rerank_mult).min(order.len());
            let te = Instant::now();
            let short: Vec<RowKey> = order[..m].iter().map(|k| (*k).clone()).collect();
            let gains = exact_gains(&inc, &train, &inc_st.seg, &short, rerank_occ, threads);
            let tq = Instant::now();
            let mut idx: Vec<usize> = (0..m).collect();
            idx.sort_by(|&a, &b| gains[b].partial_cmp(&gains[a]).unwrap().then(a.cmp(&b)));
            let zero = gains.iter().filter(|&&g| g <= 1e-9).count();
            for (i, t) in trace.iter().enumerate() {
                let f = fold_str(&inc, t);
                if let Some(j) = short.iter().position(|k| k.0 == 0 && k.1 == f) {
                    let pos = idx.iter().position(|&q| q == j).unwrap();
                    eprintln!("    trace[{i}] re-scored: exact gain {:.1}, position {pos} of {m} after re-rank (need {need})", gains[j]);
                }
            }
            eprintln!("    re-scored {m} proposals exactly: {zero} save nothing ({:.0}s)", te.elapsed().as_secs_f64());
            let tail: Vec<&RowKey> = order[m..].to_vec();
            order = idx.iter().map(|&i| order[i]).chain(tail).collect();
            prof("rerank_sort", tq);
        }
        let tq = Instant::now();
        let mut added = [0usize; 3];
        let mut added_keys: Vec<RowKey> = Vec::new();
        let mut rows = work.rows(); // kept up to date: counting live rows per add is quadratic
        // swap proposals first (best first), but at most a quarter of the slots: uncapped, a flood of
        // self-paying merges filled the whole target, so no re-scored proposal (a word saving
        // thousands of tokens, बिल्कुल) got in that round
        let cap = (target.saturating_sub(rows) / 4).max(1000);
        let mut n_swaps = 0;
        for f in swaps {
            if n_swaps >= cap {
                break;
            }
            let key: RowKey = (0, f);
            if !tabu.contains(&key) && work.tabs[0].add(&key.1) {
                added[0] += 1;
                rows += 1;
                n_swaps += 1;
                added_keys.push(key);
            }
        }
        if n_swaps > 0 {
            eprintln!("    {n_swaps} swap proposals (merged cores cheaper than the fragments they replace)");
        }
        let n_order = order.len();
        // (the rest is only needed if a proposal were not new after all: then it follows, sorted)
        let mut rest_sorted = false;
        let mut pos = 0;
        while rows < target {
            let key: &RowKey = if pos < n_order {
                order[pos]
            } else {
                if !rest_sorted {
                    rest.sort_by(by_value);
                    rest_sorted = true;
                }
                match rest.get(pos - n_order) {
                    Some((_, k)) => k,
                    None => break,
                }
            };
            pos += 1;
            if work.tabs[key.0 as usize].add(&key.1) {
                rows += 1;
                added[key.0 as usize] += 1;
                added_keys.push(key.clone());
            }
        }
        prof("add_rows", tq);
        for (i, t) in trace.iter().enumerate() {
            let f = fold_str(&inc, t);
            if added_keys.iter().any(|k| k.0 == 0 && k.1 == f) {
                eprintln!("    trace[{i}] added as core this round");
            }
        }
        let tq = Instant::now();
        work.rebuild();
        prof("rebuild", tq);
        let tw = Instant::now();
        for _ in 0..em_iters {
            let st = parse_corpus(&work, &train, threads, mu > 0.0 || mu2 > 0.0 || lamc > 0.0 || cl_on);
            update_costs(&mut work, &st);
            update_spec(&mut work, &st);
            update_pmi(&mut work, &st);
            update_cond(&mut work, &st);
            update_codelen(&mut work, &st);
        }
        let t_em = tw.elapsed().as_secs_f64();
        let tw = Instant::now();
        let em_done = prune_to(&mut work, &train, budget, threads, protect_chars, step, prune_reuse, prune_tail);
        let t_prune = tw.elapsed().as_secs_f64();
        for (i, t) in trace.iter().enumerate() {
            let f = fold_str(&inc, t);
            if let Some(&r) = work.tabs[0].map.get(&f) {
                eprintln!("    trace[{i}] after prune: core {} price {:.1}", if work.tabs[0].alive[r as usize] { "live" } else { "dead" }, work.tabs[0].price[r as usize]);
            }
        }
        let tw = Instant::now();
        let tq = Instant::now();
        tabu = added_keys.into_iter().filter(|(t, s)| !work.tabs[*t as usize].has(s)).collect();
        prof("tabu", tq);
        let tq = Instant::now();
        // (prune_reuse: the first step would re-estimate from the prune loop's last parse, re-used as
        // is since no row was dropped after it, i.e. repeat that pass's re-estimation: skipped)
        for _ in usize::from(em_done)..em_iters {
            let st = parse_corpus(&work, &train, threads, mu > 0.0 || mu2 > 0.0 || lamc > 0.0 || cl_on);
            update_costs(&mut work, &st);
            update_spec(&mut work, &st);
            update_pmi(&mut work, &st);
            update_cond(&mut work, &st);
            update_codelen(&mut work, &st);
        }
        prof("em2", tq);
        let tq = Instant::now();
        let st = parse_corpus(&work, &train, threads, true);
        prof("final_parse", tq);
        trace_state(&work, &st, &trace, &format!("round {round} end"));
        let tq = Instant::now();
        let score = (st.cost + work.total_price(), description_length(&work, &st, &char_bits, gamma));
        let accept = score.0 < inc_score.0 || (score.0 == inc_score.0 && score.1 < inc_score.1);
        prof("score", tq);
        let tq = Instant::now();
        let dv = parse_corpus(&work, &dev, threads, false).tokens;
        prof("dev_parse", tq);
        eprintln!("    time: proposals {t_prop:.0}s, EM {t_em:.0}s, prune {t_prune:.0}s, final+dev parse {:.0}s",
                  tw.elapsed().as_secs_f64());
        eprintln!("  round {round}: +{} cores +{} prefixes +{} suffixes -> rows {} (core {} prefix {} suffix {}), \
                   chars/token train {:.3} dev {:.3}, R {:.3e} bits, {} ({:.0}s)",
                  added[0], added[1], added[2], work.rows(), work.tabs[0].live(), work.tabs[1].live(),
                  work.tabs[2].live(), chars / st.tokens, dev_chars / dv, score.1,
                  if accept { "accepted" } else { "rejected" }, tr.elapsed().as_secs_f64());
        let tq = Instant::now();
        let prev_best = best_dev;
        let mut stop = false;
        if accept {
            inc = work;
            inc_st = st;
            inc_score = score;
            fails = 0;
            if dv < best_dev {
                best_dev = dv;
                save(&inc, n_sym, &args[3]);
            }
        } else {
            fails += 1;
            k_extra *= 2; // explore a larger expansion next time
            stop = fails >= 2;
        }
        prof("accept_save", tq);
        eprintln!("    prof: {}", prof_dump());
        // early stop (min_gain_ppm): the round improved the best dev chars/token too little (in the
        // fine phase only accepted rounds count: a rejection there is retried with a larger expansion)
        let fine = sampled.is_some() && !coarse;
        let min_gain_now = if coarse && coarse_gain > 0.0 { coarse_gain } else { min_gain };
        if !stop && min_gain_now > 0.0 && round > 0 && prev_best.is_finite() && (accept || !fine) {
            let gain = (prev_best / best_dev - 1.0) * 1e6;
            if gain < min_gain_now {
                eprintln!("  stopping{}: best dev chars/token improved by {gain:.0} ppm < {min_gain_now} ppm this round",
                          if coarse { " the coarse phase" } else { "" });
                stop = true;
            }
        }
        if !coarse && sampled.is_some() {
            fine_left = fine_left.saturating_sub(1);
            stop |= fine_left == 0;
        }
        if coarse && (stop || round + 1 >= max_rounds) {
            // switch to the full corpus: the incumbent re-estimated and re-scored there (its sample
            // score is not comparable), then refined by the fine rounds
            coarse = false;
            if fine_rounds == 0 {
                break;
            }
            let tq = Instant::now();
            train = &full;
            chars = chars_of(train);
            let st = parse_corpus(&inc, train, threads, true);
            update_costs(&mut inc, &st);
            update_spec(&mut inc, &st);
            update_pmi(&mut inc, &st);
            update_cond(&mut inc, &st);
            update_codelen(&mut inc, &st);
            // (partner and signature prices; the signature factor too: without it the incumbent was scored with its affixes'
            // bare partner prices, cheaper than the fine round's fully priced rows, so the fine round
            // was rejected even when it improved dev chars/token)
            update_prices(&mut inc, &st);
            let st = parse_corpus(&inc, train, threads, true);
            inc_score = (st.cost + inc.total_price(), description_length(&inc, &st, &char_bits, gamma));
            inc_st = st;
            fails = 0;
            // a coarse phase run to convergence stopped at this expansion (a first fine round there
            // was mostly rejected in tests): refine with a larger one, as after a rejection
            if coarse_gain == 0.0 {
                k_extra *= 2;
            }
            eprintln!("  fine phase: full corpus, incumbent chars/token train {:.3}, R {:.4e} bits (parse {:.4e} + prices {:.4e}) ({:.0}s)", chars / inc_st.tokens, inc_score.0, inc_st.cost, inc.total_price(), tq.elapsed().as_secs_f64());
            prof_dump();
            continue;
        }
        if stop {
            break;
        }
    }
    if best_dev == f64::INFINITY {
        save(&inc, n_sym, &args[3]);
    }
    eprintln!("  done in {:.0}s; best dev chars/token {:.3}", t0.elapsed().as_secs_f64(), dev_chars / best_dev);
}

fn encode(args: &[String]) {
    let mut m = Reader::open(&args[2]);
    let mut n_sym = m.u32() as usize;
    let ca = n_sym == CAFX_MAGIC as usize;
    let case = ca || n_sym == CASE_MAGIC as usize;
    if case {
        n_sym = m.u32() as usize;
    }
    let mut sym = m.symbols(n_sym);
    let mut strs = [m.table(), m.table(), m.table()];
    let costs: Vec<Vec<f64>> = (0..3).map(|k| (0..strs[k].len()).map(|_| m.f64()).collect()).collect();
    let mut vcost = [0.0; NVAR_MAX];
    for v in vcost.iter_mut().take(if ca { NVAR_MAX } else if case { NVAR_CASE } else { 4 }) {
        *v = m.f64();
    }
    // case_cores: cores are stored as canonical spellings; the trie is keyed by the folded form and
    // the canonical upper masks are looked up by it (Dict::canon_mask)
    // (case_affixes: the affix tables too; their masks are also set directly on the rows below,
    // so that encoding always writes exactly the stored spellings)
    let mut canon: HashMap<u64, u64, Fast> = HashMap::default();
    let mut masks: [Vec<u64>; 3] = Default::default();
    if case {
        for k in 0..if ca { 3 } else { 1 } {
            for s in strs[k].iter_mut() {
                let um = s.iter().take(64).enumerate().fold(0u64, |a, (k, &c)| a | (((sym.cf[c as usize] & CF_UP != 0) as u64) << k));
                for c in s.iter_mut() {
                    *c = sym.fold[*c as usize];
                }
                if um != 0 {
                    canon.insert(seq_hash(s.iter().copied()), um);
                }
                masks[k].push(um);
            }
        }
    }
    let delta = m.f64();
    sym.nbytes = m.u32s(n_sym);
    let lam = m.f64();
    let mu = if m.pos + 8 <= m.buf.len() { m.f64() } else { 0.0 };
    let mut tabs = [Table::new(false), Table::new(false), Table::new(false)];
    for k in 0..3 {
        for (i, s) in strs[k].iter().enumerate() {
            tabs[k].map.insert(s.clone(), i as u32);
            tabs[k].strs.push(s.clone());
            tabs[k].alive.push(true);
            tabs[k].cost.push(costs[k][i]);
            tabs[k].spec.push(0.0);
            tabs[k].glue.push(0.0);
            tabs[k].usec.push(0.0);
            tabs[k].umask.push(0);
            tabs[k].price.push(0.0);
        }
    }
    if m.pos < m.buf.len() {
        for k in 1..3 {
            for i in 0..strs[k].len() {
                tabs[k].spec[i] = m.f64();
            }
        }
    }
    let (mut mu2, mut tau) = (0.0, 0.0);
    let mut pmi: [HashMap<(u32, u32), f32, Fast>; 2] = [HashMap::default(), HashMap::default()];
    if m.pos + 16 <= m.buf.len() {
        mu2 = m.f64();
        tau = m.f64();
        for k in 0..2 {
            let n = m.u32() as usize;
            for _ in 0..n {
                let (a, b) = (m.u32(), m.u32());
                let v = f32::from_le_bytes(m.take::<4>());
                pmi[k].insert((a, b), v);
            }
        }
    }
    let (mut lamc, mut ncore, mut cn) = (0.0, Vec::new(), <[HashMap<(u32, u32), f32, Fast>; 2]>::default());
    if m.pos + 12 <= m.buf.len() {
        lamc = m.f64();
        let n = m.u32() as usize;
        ncore = (0..n).map(|_| f32::from_le_bytes(m.take::<4>())).collect();
        for k in 0..2 {
            let n = m.u32() as usize;
            for _ in 0..n {
                let (a, b) = (m.u32(), m.u32());
                cn[k].insert((a, b), f32::from_le_bytes(m.take::<4>()));
            }
        }
    }
    let mut nu = 0.0;
    if m.pos + 8 <= m.buf.len() {
        nu = m.f64();
        for k in 0..3 {
            if k > 0 && m.pos >= m.buf.len() {
                break; // older models: core scores only
            }
            for i in 0..strs[k].len() {
                tabs[k].glue[i] = m.f64();
            }
        }
    }
    // trailing sections: mark_rule, then code_len (each optional, tagged)
    let mut tag = if m.pos + 4 <= m.buf.len() { m.u32() } else { 0 };
    if tag == MARK_MAGIC {
        sym.alnum = m.u32s(n_sym);
        sym.mark_rule = true;
        tag = if m.pos + 4 <= m.buf.len() { m.u32() } else { 0 };
    }
    let mut cl = CodeLen::default();
    let had_code = tag == CODE_MAGIC;
    if tag == CODE_MAGIC {
        cl.on = m.u32() == 1;
        cl.w = m.f64();
        cl.byte = m.f64();
        cl.wu = if cl.on { 0.0 } else { cl.w };
    }
    if cl.on {
        let nc = strs[0].len();
        let mut q = || (u16::from_le_bytes(m.take::<2>()) as f64 / CL_Q) as f32;
        cl.bo = (0..nc).map(|_| [q(), q(), q()]).collect();
        for k in 0..3 {
            let len = m.u32() as usize;
            let buf = &m.buf[m.pos..m.pos + len];
            m.pos += len;
            let mut at = 0usize;
            for c in 0..nc as u32 {
                let cnt = get_var(buf, &mut at);
                let mut a = 0u32;
                for j in 0..cnt {
                    let g = get_var(buf, &mut at);
                    a = if j == 0 { g } else { a + g + 1 };
                    let v = (u16::from_le_bytes([buf[at], buf[at + 1]]) as f64 / CL_Q) as f32;
                    at += 2;
                    cl.pair[k].insert(if k == 0 { (a, c) } else { (c, a) }, v);
                }
            }
        }
        let nv = if ca { NVAR_MAX } else if case { NVAR_CASE } else { 4 };
        cl.vb = codelen_vb(&cl.bo, &cl.pair[2], &vcost, nv);
        cl.index();
    }
    if had_code {
        tag = if m.pos + 4 <= m.buf.len() { m.u32() } else { 0 };
    }
    let mut be = BoundEnt::default();
    if tag == BE_MAGIC {
        be.w = m.f64();
        be.order = m.u32() as usize;
        be.med = [m.u32() as u8, m.u32() as u8];
        be.ncls = m.u32() as usize;
        let nq = m.u32() as usize;
        assert_eq!(nq, BE_NS, "boundary percentile table");
        be.cdf = (0..be.ncls * nq).map(|_| f32::from_le_bytes(m.take::<4>())).collect();
        let ng = m.u32() as usize;
        be.grp = m.buf[m.pos..m.pos + ng].to_vec();
        m.pos += ng;
        let (n, len) = (m.u32() as usize, m.u32() as usize);
        let buf = &m.buf[m.pos..m.pos + len];
        m.pos += len;
        let (mut at, mut k) = (0usize, 0u32);
        be.tab.reserve(n);
        for j in 0..n {
            let g = get_var(buf, &mut at);
            k = if j == 0 { g } else { k + g + 1 };
            be.tab.insert(k, buf[at]);
            at += 1;
        }
    }
    let empty = Trie::empty;
    let no_frag: [HashMap<u64, f32, Fast>; 3] = Default::default();
    let no_pricing = Pricing::default();
    let mut d = Dict { sym: &sym, tabs, vcost, case, ca, canon: &canon, delta, lam, mu, h0: 0.0, alnum_only: false, classes: false, allow: Default::default(), mu2, tau, pmi, lamc, cn, ncore, tries: [empty(), empty(), empty()], nu, frag: &no_frag, pricing: &no_pricing, cl, be: &be };
    d.rebuild();
    if ca {
        for k in 0..3 {
            d.tabs[k].umask.clone_from(&masks[k]);
        }
    }
    let c = Reader::open(&args[3]).corpus(false);
    let mut w = BufWriter::new(std::fs::File::create(&args[4]).expect("create output"));
    // segments are parsed independently: on several threads (env DS_THREADS, default up to 8),
    // in chunks whose outputs are written in segment order (the same bytes as one thread)
    let threads = std::env::var("DS_THREADS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or_else(|| {
        std::thread::available_parallelism().map_or(1, |n| n.get()).min(8)
    }).max(1);
    const CHUNK: usize = 1 << 16;
    const BLOCK: usize = 256;
    let d = &d;
    for lo in (0..c.segs.len()).step_by(CHUNK) {
        let hi = (lo + CHUNK).min(c.segs.len());
        let out: Vec<Vec<u8>> = par_map((hi - lo).div_ceil(BLOCK), threads, 1, || (Scratch::new(), Vec::new()), |(sc, toks), b| {
            let mut o = Vec::new();
            for s in &c.segs[lo + b * BLOCK..(lo + (b + 1) * BLOCK).min(hi)] {
                d.parse(s, sc, Some(toks), None);
                put(&mut o, &[toks.len() as u32]);
                for t in toks.iter() {
                    put(&mut o, &[t.v, t.p, t.c, t.s]);
                }
            }
            o
        });
        for o in out {
            w.write_all(&o).unwrap();
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("train") if args.len() == 4 => train(&args),
        Some("encode") if args.len() == 5 => encode(&args),
        _ => panic!("usage: dict_search train INPUT MODEL | encode MODEL WORDS OUTPUT"),
    }
}
