//! Reversible factored dictionary search for unrestricted Combinatorial BPE.
//! (Implements the design in docs/unrestricted_trainer_problem.md's answer by GPT: optimise the
//! dictionary directly, re-parse from scratch after every change, never commit merges.)
//!
//! A token is (variation, prefix, core, suffix); all three parts are arbitrary strings. Cores are
//! case/Han-folded and carry the variation; prefixes and suffixes match the original text exactly.
//!
//! ENCODER: exact minimum-token parse with three states per position,
//!     B[i] between tokens --prefix--> P[j] --core,variation (1 token)--> C[k] --suffix--> B[l]
//! (empty prefix/suffix allowed, core never empty), comparing (tokens, secondary cost)
//! lexicographically. Secondary cost = -log2 q(prefix) - log2 q(core) - log2 q(variation)
//! - log2 q(suffix). Cost is linear in the number of trie matches (no prefix x core x suffix
//! enumeration). Out-of-alphabet chars are byte-fallback tokens (one per UTF-8 byte).
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
//!     rejected round improves nothing); the saved model is the best so far, as always)
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
//!     suffix row
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

struct Symbols {
    fold: Vec<u32>,
    flag: Vec<u32>,
    upper_of: Vec<u32>,
    trad_of: Vec<u32>,
    nbytes: Vec<u32>,
    alnum: Vec<u32>,
}

impl Symbols {
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
    scored: usize,   // rows whose glue and price are set (rebuild)
}

impl Table {
    fn new(with_empty: bool) -> Self {
        let mut t = Table { strs: vec![], alive: vec![], map: HashMap::default(), cost: vec![], spec: vec![], glue: vec![], price: vec![], usec: vec![], scored: 0 };
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
    vcost: [f64; 4],
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
}

/// 64-bit hash of a symbol string (splitmix64 steps; symbol 0 and leading symbols all count,
/// unlike Fx, which maps a leading 0 to the empty state)
fn seq_hash(s: impl Iterator<Item = u32>) -> u64 {
    let mut h: u64 = 0x9e37_79b9_7f4a_7c15;
    for c in s {
        h = (h ^ (c as u64 + 1)).wrapping_add(0x9e37_79b9_7f4a_7c15);
        h = (h ^ (h >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        h = (h ^ (h >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        h ^= h >> 31;
    }
    h
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

/// The trie arcs of a text (Dict::arcs_x): arcs[start[3i + k]..start[3i + k + 1]] leave position i
/// by walk k (0 prefixes, 1 cores, 2 suffixes).
#[derive(Default)]
struct Arcs {
    start: Vec<u32>,
    arcs: Vec<(u32, f64, f64)>,
}

impl Arcs {
    #[inline]
    fn get(&self, i: usize, k: usize) -> &[(u32, f64, f64)] {
        &self.arcs[self.start[3 * i + k] as usize..self.start[3 * i + k + 1] as usize]
    }
}

/// A row that is not in the dictionary, enabled for one parse (exact candidate scoring).
struct Extra {
    t: usize,
    s: Vec<u32>,
    cost: f64,
    spec: f64,
    usec: f64,
}

const EXTRA: u32 = u32::MAX - 1;

#[derive(Clone, Copy)]
struct Back {
    state: u8,
    from: u32,
    id: u32,
    v: u8,
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
    sm: Vec<(usize, u32, f64, f64)>,
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
            sm: vec![],
        }
    }
}

/// prefix-done entry: which prefix (id), score, and the between-tokens position it started at
#[derive(Clone, Copy)]
struct PEnt {
    pid: u32,
    s: (f64, f64),
    from: u32,
}

/// core-done entry: which core and variation, score, and the prefix-done entry it came from
#[derive(Clone, Copy)]
struct CEnt {
    cid: u32,
    v: u8,
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
    match v.iter_mut().find(|o| o.pid == e.pid) {
        Some(o) => {
            if better(e.s, o.s) {
                *o = e;
            }
        }
        None => v.push(e),
    }
}

fn add_c(v: &mut Vec<CEnt>, e: CEnt) {
    match v.iter_mut().find(|o| o.cid == e.cid) {
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
            self.tabs[k].scored = self.tabs[k].strs.len();
        }
        self.refresh_arcs();
    }
    /// (primary, secondary) cost of the arc of the row at every trie node, exactly as parse_x adds
    /// it (cores: without the variation cost), in node order: a trie walk then reads it next to
    /// the node instead of from four row arrays. Refreshed whenever tries, costs or specs change.
    fn refresh_arcs(&mut self) {
        let (delta, lam, mu) = (self.delta, self.lam, self.mu);
        for k in 0..3 {
            let t = &self.tabs[k];
            for nd in self.tries[k].node.iter_mut() {
                let r = nd.term as usize;
                nd.arc = match (nd.term == NONE, k) {
                    (true, _) => INF,
                    (_, 0) => (1.0 + t.usec[r], t.cost[r]),
                    _ => (delta + lam * t.cost[r] + mu * t.spec[r] + t.usec[r], t.cost[r]),
                };
            }
        }
    }
    /// one-time price of a row string of table k, in tokens
    fn price_of(&self, k: usize, s: &[u32]) -> f64 {
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
        self.frag[k].get(&seq_hash(s.iter().copied())).map_or(0.0, |&v| v as f64)
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

    /// The exact DP with pairwise costs: prefix-done states are kept per prefix id and core-done
    /// states per core id, so a token can pay mu2 x excess PMI of its (prefix, core) and
    /// (core, suffix) pairs. Same arcs and costs as parse_x otherwise.
    fn parse_pair(&self, x: &[u32], sc: &mut Scratch, out: Option<&mut Vec<Tok>>, ban: Ban, extra: Option<&Extra>)
        -> (f64, f64) {
        let n = x.len();
        let (delta, lam, mu, mu2, lamc) = (self.delta, self.lam, self.mu, self.mu2, self.lamc);
        let (cc, cp, cs) = (&self.tabs[0].cost, &self.tabs[1].cost, &self.tabs[2].cost);
        let (sp, ss) = (&self.tabs[1].spec, &self.tabs[2].spec);
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
        let empty_p = (lam * cp[0] + mu * sp[0], cp[0]);
        let empty_s = (lam * cs[0] + mu * ss[0], cs[0]);
        for i in 0..=n {
            // empty suffix: core-done -> between tokens
            for idx in 0..sc.ce[i].len() {
                let e = sc.ce[i][idx];
                let cond = if lamc > 0.0 { lamc * self.cond(1, e.cid, 0, (-cs[0]).exp2()) } else { 0.0 };
                let c = (e.s.0 + empty_s.0 + cond, e.s.1 + empty_s.1);
                if better(c, sc.bs[i]) {
                    sc.bs[i] = c;
                    sc.bb[i] = BBack { kind: 1, pos: i as u32, idx: idx as u32, sid: 0 };
                }
            }
            let b = sc.bs[i];
            if !b.0.is_infinite() {
                add_p(&mut sc.pe[i], PEnt { pid: 0, s: (b.0 + empty_p.0, b.1 + empty_p.1), from: i as u32 });
            }
            if i == n {
                break;
            }
            if !b.0.is_infinite() {
                let nb = self.sym.byte_len(x[i]) as f64;
                let c = (b.0 + nb * (1.0 + (lam + lamc) * (cp[0] + cs[0]) + mu * (sp[0] + ss[0])), b.1 + 1000.0 * nb);
                if better(c, sc.bs[i + 1]) {
                    sc.bs[i + 1] = c;
                    sc.bb[i + 1] = BBack { kind: 2, pos: i as u32, idx: 0, sid: 0 };
                }
                let mut node = 0u32;
                for j in i..n {
                    let Some(m) = self.tries[1].step(node, x[j]) else { break };
                    node = m;
                    let r = self.tries[1].term(node);
                    if self.usable(1, r, ban) {
                        let e = PEnt {
                            pid: r,
                            s: (b.0 + delta + lam * cp[r as usize] + mu * sp[r as usize] + self.tabs[1].usec[r as usize], b.1 + cp[r as usize]),
                            from: i as u32,
                        };
                        add_p(&mut sc.pe[j + 1], e);
                    }
                }
                if let Some(e) = extra {
                    let l = e.s.len();
                    if e.t == 1 && i + l <= n && x[i..i + l] == e.s[..] {
                        let pe = PEnt { pid: EXTRA, s: (b.0 + delta + lam * e.cost + mu * e.spec + e.usec, b.1 + e.cost), from: i as u32 };
                        add_p(&mut sc.pe[i + l], pe);
                    }
                }
            }
            if !sc.pe[i].is_empty() {
                sc.cm.clear();
                let mut node = 0u32;
                let mut m = 0u8;
                for j in i..n {
                    let c = x[j];
                    if is_byte(c) {
                        break;
                    }
                    let Some(nn) = self.tries[0].step(node, self.sym.fold[c as usize]) else { break };
                    node = nn;
                    m = if j == i { self.sym.mask(c) } else { combine(m, self.sym.mask(c)) };
                    let Some(v) = variation(m) else { break };
                    let r = self.tries[0].term(node);
                    if self.usable(0, r, ban) {
                        sc.cm.push((j + 1, r, v as u8, 1.0 + self.tabs[0].usec[r as usize], cc[r as usize] + self.vcost[v]));
                    }
                }
                if let Some(e) = extra {
                    let l = e.s.len();
                    if e.t == 0 && i + l <= n {
                        let mut m = 0u8;
                        let mut ok = true;
                        for k in 0..l {
                            let c = x[i + k];
                            if is_byte(c) || self.sym.fold[c as usize] != e.s[k] {
                                ok = false;
                                break;
                            }
                            m = if k == 0 { self.sym.mask(c) } else { combine(m, self.sym.mask(c)) };
                        }
                        if let Some(v) = if ok { variation(m) } else { None } {
                            sc.cm.push((i + l, EXTRA, v as u8, 1.0 + e.usec, e.cost + self.vcost[v]));
                        }
                    }
                }
                for pidx in 0..sc.pe[i].len() {
                    let pe = sc.pe[i][pidx];
                    for q in 0..sc.cm.len() {
                        let (k, cid, v, c1, c2) = sc.cm[q];
                        let mut pair = mu2 * self.pmi_of(0, pe.pid, cid);
                        if lamc > 0.0 {
                            let qa = if pe.pid == EXTRA { (-extra.map_or(20.0, |e| e.cost)).exp2() } else { (-cp[pe.pid as usize]).exp2() };
                            pair += lamc * self.cond(0, pe.pid, cid, qa);
                        }
                        let e = CEnt { cid, v, s: (pe.s.0 + c1 + pair, pe.s.1 + c2), fpos: i as u32, fidx: pidx as u32 };
                        add_c(&mut sc.ce[k], e);
                    }
                }
            }
            if !sc.ce[i].is_empty() {
                sc.sm.clear();
                let mut node = 0u32;
                for j in i..n {
                    let Some(m) = self.tries[2].step(node, x[j]) else { break };
                    node = m;
                    let r = self.tries[2].term(node);
                    if self.usable(2, r, ban) {
                        sc.sm.push((j + 1, r, delta + lam * cs[r as usize] + mu * ss[r as usize] + self.tabs[2].usec[r as usize], cs[r as usize]));
                    }
                }
                if let Some(e) = extra {
                    let l = e.s.len();
                    if e.t == 2 && i + l <= n && x[i..i + l] == e.s[..] {
                        sc.sm.push((i + l, EXTRA, delta + lam * e.cost + mu * e.spec + e.usec, e.cost));
                    }
                }
                for cidx in 0..sc.ce[i].len() {
                    let ce = sc.ce[i][cidx];
                    for q in 0..sc.sm.len() {
                        let (j, sid, c1, c2) = sc.sm[q];
                        let mut pair = mu2 * self.pmi_of(1, ce.cid, sid);
                        if lamc > 0.0 {
                            let qa = if sid == EXTRA { (-extra.map_or(20.0, |e| e.cost)).exp2() } else { (-cs[sid as usize]).exp2() };
                            pair += lamc * self.cond(1, ce.cid, sid, qa);
                        }
                        let c = (ce.s.0 + c1 + pair, ce.s.1 + c2);
                        if better(c, sc.bs[j]) {
                            sc.bs[j] = c;
                            sc.bb[j] = BBack { kind: 1, pos: i as u32, idx: cidx as u32, sid };
                        }
                    }
                }
            }
        }
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
                                ban: Ban, extra: Option<&Extra>, close: bool, arcs: Option<&Arcs>) {
        let n = x.len();
        let (cp, cs) = (&self.tabs[1].cost, &self.tabs[2].cost);
        let (delta, lam, mu) = (self.delta, self.lam, self.mu);
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
                    back[t][j - off] = Back { state: s as u8, from: i as u32, id, v };
                }
            }
        };
        if close {
            // empty closure, order C -> B -> P
            relax(d, back, 2, i, 0, i, lam * cs[0] + mu * ss[0], cs[0], 0, 0);
            relax(d, back, 0, i, 1, i, lam * cp[0] + mu * sp[0], cp[0], 0, 0);
        }
        if i == n {
            return;
        }
        // byte fallback: each byte token pays what a token with empty affixes pays, and a high
        // secondary cost, so a char core always wins a tie
        let nb = self.sym.byte_len(x[i]) as f64;
        relax(d, back, 0, i, 0, i + 1, nb * (1.0 + lam * (cp[0] + cs[0]) + mu * (sp[0] + ss[0])), 1000.0 * nb, NONE, 0);
        if let Some(e) = extra {
            let l = e.s.len();
            if i + l <= n {
                if e.t == 1 && x[i..i + l] == e.s[..] {
                    relax(d, back, 0, i, 1, i + l, delta + lam * e.cost + mu * e.spec + e.usec, e.cost, EXTRA, 0);
                } else if e.t == 2 && x[i..i + l] == e.s[..] {
                    relax(d, back, 2, i, 0, i + l, delta + lam * e.cost + mu * e.spec + e.usec, e.cost, EXTRA, 0);
                } else if e.t == 0 {
                    let mut m = 0u8;
                    let mut ok = true;
                    for k in 0..l {
                        let c = x[i + k];
                        if is_byte(c) || self.sym.fold[c as usize] != e.s[k] {
                            ok = false;
                            break;
                        }
                        m = if k == 0 { self.sym.mask(c) } else { combine(m, self.sym.mask(c)) };
                    }
                    if let Some(v) = if ok { variation(m) } else { None } {
                        relax(d, back, 1, i, 2, i + l, 1.0 + e.usec, e.cost + self.vcost[v], EXTRA, v as u8);
                    }
                }
            }
        }
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
        if !o.0.is_infinite() {
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
                            bt[j + 1 - off] = Back { state: 0, from: i as u32, id: nd.term, v: 0 };
                        }
                    }
                }
            }
        }
        let o = d[1][i - off];
        if !o.0.is_infinite() {
            let tr = &self.tries[0];
            let (dt, bt) = (&mut d[2], &mut back[2]);
            let mut node = 0u32;
            let mut m = 0u8;
            for j in i..n {
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
                    let c = (o.0 + nd.arc.0, o.1 + (nd.arc.1 + self.vcost[v]));
                    if better(c, dt[j + 1 - off]) {
                        dt[j + 1 - off] = c;
                        if BACK {
                            bt[j + 1 - off] = Back { state: 1, from: i as u32, id: nd.term, v: v as u8 };
                        }
                    }
                }
            }
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
                            bt[j + 1 - off] = Back { state: 2, from: i as u32, id: nd.term, v: 0 };
                        }
                    }
                }
            }
        }
    }

    /// Every arc of step_x's trie walks in x (no ban), whatever the source scores: per position and
    /// walk, (target, primary, secondary) in walk order, as step_x adds them.
    fn arcs_x(&self, x: &[u32], a: &mut Arcs) {
        let n = x.len();
        a.start.clear();
        a.arcs.clear();
        for i in 0..n {
            a.start.push(a.arcs.len() as u32);
            let tr = &self.tries[1];
            let mut node = 0u32;
            for j in i..n {
                let Some(m) = tr.step(node, x[j]) else { break };
                node = m;
                let nd = &tr.node[node as usize];
                if self.usable(1, nd.term, None) {
                    a.arcs.push(((j + 1) as u32, nd.arc.0, nd.arc.1));
                }
            }
            a.start.push(a.arcs.len() as u32);
            let tr = &self.tries[0];
            let mut node = 0u32;
            let mut m = 0u8;
            for j in i..n {
                let c = x[j];
                if is_byte(c) {
                    break;
                }
                let Some(nn) = tr.step(node, self.sym.fold[c as usize]) else { break };
                node = nn;
                m = if j == i { self.sym.mask(c) } else { combine(m, self.sym.mask(c)) };
                let Some(v) = variation(m) else { break };
                let nd = &tr.node[node as usize];
                if self.usable(0, nd.term, None) {
                    a.arcs.push(((j + 1) as u32, nd.arc.0, nd.arc.1 + self.vcost[v]));
                }
            }
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
    fn scores_x(&self, x: &[u32], a: &Arcs, d: &mut [Vec<(f64, f64)>; 3], nob: &mut [Vec<Back>; 3]) {
        let n = x.len();
        for s in 0..3 {
            d[s].clear();
            d[s].resize(n + 1, INF);
        }
        d[0][0] = (0.0, 0.0);
        for i in 0..=n {
            self.step_x::<false>(x, i, d, nob, 0, None, None, true, Some(a));
        }
    }

    /// Longest arc of parse_x under the live rows (trie matches, a byte) plus a row of length l.
    fn max_arc(&self, l: usize) -> usize {
        let live = |k: usize| (0..self.tabs[k].strs.len()).filter(|&i| self.tabs[k].alive[i]).map(|i| self.tabs[k].strs[i].len()).max().unwrap_or(0);
        (0..3).map(live).max().unwrap_or(0).max(l).max(1)
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
                      w: &mut [Vec<(f64, f64)>; 3], nob: &mut [Vec<Back>; 3]) -> f64 {
        let n = x.len();
        let same = |w: &[Vec<(f64, f64)>; 3], p: usize, off: usize| {
            (0..3).all(|s| w[s][p - off].0.to_bits() == fin[s][p].0.to_bits() && w[s][p - off].1.to_bits() == fin[s][p].1.to_bits())
        };
        let mut k = 0;
        while k < occ.len() {
            let a = occ[k] as usize;
            let off = a.saturating_sub(lmax);
            // window scores cover [off, min(n, i + lmax)] when position i is processed
            let mut end = (a + lmax).min(n);
            for s in 0..3 {
                w[s].clear();
                w[s].resize(end - off + 1, INF);
            }
            if off == 0 {
                w[0][0] = (0.0, 0.0); // the start state, as parse_x seeds it
            }
            // pending scores at time a: the plain arcs out of [off, a), from their final scores
            for i in off..a {
                for s in 0..3 {
                    w[s][i - off] = fin[s][i];
                }
                self.step_x::<false>(x, i, w, nob, off, None, None, false, Some(arcs));
            }
            // the DP with e from a on; `run` = number of positions just processed whose final scores
            // equal the plain parse's (those before a all do), `reach` = end of e's arcs so far
            let (mut run, mut reach) = (lmax, 0usize);
            let mut i = a;
            loop {
                let want = (i + lmax).min(n);
                if want > end {
                    for s in 0..3 {
                        w[s].resize(want - off + 1, INF);
                    }
                    end = want;
                }
                self.step_x::<false>(x, i, w, nob, off, None, Some(e), true, Some(arcs));
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

    fn parse_x(&self, x: &[u32], sc: &mut Scratch, out: Option<&mut Vec<Tok>>, ban: Ban, extra: Option<&Extra>)
        -> (f64, f64) {
        if self.mu2 > 0.0 || self.lamc > 0.0 {
            return self.parse_pair(x, sc, out, ban, extra);
        }
        let n = x.len();
        for k in 0..3 {
            sc.d[k].clear();
            sc.d[k].resize(n + 1, INF);
            sc.back[k].clear();
            sc.back[k].resize(n + 1, Back { state: 9, from: 0, id: 0, v: 0 });
        }
        sc.d[0][0] = (0.0, 0.0);
        for i in 0..=n {
            self.step_x::<true>(x, i, &mut sc.d, &mut sc.back, 0, ban, extra, true, None);
        }
        let res = sc.d[0][n];
        if let Some(out) = out {
            out.clear();
            if res.0.is_infinite() {
                return res;
            }
            // backtrace: labels come in suffix, core, prefix order (reversed)
            let (mut st, mut pos) = (0usize, n);
            let mut cur = Tok { v: 0, p: 0, c: NONE, s: 0 };
            while !(st == 0 && pos == 0) {
                let b = sc.back[st][pos];
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
            }
            out.reverse();
        }
        res
    }

    /// the text of a token (prefix, written core, suffix) into x
    fn token_text_into(&self, t: &Tok, x: &mut Vec<u32>) {
        x.clear();
        x.extend_from_slice(&self.tabs[1].strs[t.p as usize]);
        x.extend(self.sym.written(&self.tabs[0].strs[t.c as usize], t.v));
        x.extend_from_slice(&self.tabs[2].strs[t.s as usize]);
    }
}

// ------------------------------------------------------------------ corpus passes

struct Corpus {
    segs: Vec<Vec<u32>>,
    freqs: Vec<f64>,
}

struct ParseStats {
    tokens: f64,
    cost: f64,
    seg: Vec<f64>, // primary cost of every segment (only when types are collected)
    use_: [Vec<f64>; 3],
    v: [f64; 4],
    types: HashMap<Tok, f64, Fast>,
}

fn parse_corpus(d: &Dict, c: &Corpus, threads: usize, want_types: bool) -> ParseStats {
    // The statistics are those of `threads` fixed ranges of segments, each accumulated in order and
    // then merged in range order (sums and the types map, whose iteration order later steps
    // depend on, come out bit-identical to one thread per range). The parsing itself runs on small
    // blocks handed out on demand (segment costs vary a lot); a range's statistics are built as
    // soon as all its blocks are parsed, and merged by this thread while the others still parse.
    let n = c.segs.len();
    let threads = threads.max(1);
    const B: usize = 512;
    let nb = n.div_ceil(B);
    let chunk = n.div_ceil(threads).max(1);
    let range = |t: usize| (t * chunk).min(n)..((t + 1) * chunk).min(n);
    let blocks_of = |t: usize| {
        let r = range(t);
        if r.is_empty() { 0..0 } else { r.start / B..r.end.div_ceil(B) }
    };
    type Parsed = (Vec<f64>, Vec<u32>, Vec<Tok>); // per block: primary costs, token ends, tokens
    let slots: Vec<std::sync::OnceLock<Parsed>> = (0..nb).map(|_| std::sync::OnceLock::new()).collect();
    let left: Vec<std::sync::atomic::AtomicUsize> = (0..threads).map(|t| std::sync::atomic::AtomicUsize::new(blocks_of(t).len())).collect();
    let build = |t: usize| {
        let mut st = ParseStats {
            tokens: 0.0,
            cost: 0.0,
            seg: Vec::new(),
            use_: [0, 1, 2].map(|k| vec![0.0; d.tabs[k].strs.len()]),
            v: [0.0; 4],
            types: HashMap::default(),
        };
        for i in range(t) {
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
        st
    };
    let tq = Instant::now();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let (tx, rx) = std::sync::mpsc::channel::<(usize, ParseStats)>();
    std::thread::scope(|s| {
        for _ in 0..threads {
            let (tx, next, slots, left, build, blocks_of) = (tx.clone(), &next, &slots, &left, &build, &blocks_of);
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
                    // the ranges this block belongs to: the one it completes is built here
                    let t0 = b * B / chunk;
                    for t in t0..threads {
                        if !blocks_of(t).contains(&b) {
                            break;
                        }
                        if left[t].fetch_sub(1, std::sync::atomic::Ordering::AcqRel) == 1 {
                            tx.send((t, build(t))).unwrap();
                        }
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
        prof("pc_all", tq);
        acc.unwrap()
    })
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
    for k in 0..4 {
        acc.v[k] += p.v[k];
    }
    for (t, n) in p.types {
        *acc.types.entry(t).or_insert(0.0) += n;
    }
}

/// q = (n + 1/2) / (sum n + |D|/2), secondary cost = -log2 q (bits)
fn update_costs(d: &mut Dict, st: &ParseStats) {
    for k in 0..3 {
        let live = d.tabs[k].live() as f64;
        let tot: f64 = (0..d.tabs[k].strs.len()).filter(|&i| d.tabs[k].alive[i]).map(|i| st.use_[k][i]).sum();
        for i in 0..d.tabs[k].strs.len() {
            d.tabs[k].cost[i] = -((st.use_[k][i] + 0.5) / (tot + live / 2.0)).log2();
        }
    }
    let tot: f64 = st.v.iter().sum();
    for k in 0..4 {
        d.vcost[k] = -((st.v[k] + 0.5) / (tot + 2.0)).log2();
    }
    d.refresh_arcs();
}

/// Counts behind the conditional affix cost: n(prefix, core), n(core, suffix), n(core).
/// Partner prices (partner_n0 > 0): affix price = pi x L0 x (1 + k / E_w), E_w from the parse.
fn update_partner_prices(d: &mut Dict, st: &ParseStats) {
    let n0 = d.pricing.n0;
    if n0 == 0.0 || d.pricing.pi == 0.0 {
        return;
    }
    let mut uc = vec![0.0f64; d.tabs[0].strs.len()];
    for (t, &n) in &st.types {
        if t.c != NONE {
            uc[t.c as usize] += n;
        }
    }
    for k in 1..3 {
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
        for a in 1..len {
            let e = if tot[a] > 0.0 { h[a].exp2() * wtot[a] / tot[a] } else { 0.0 };
            let l0 = d.l0_of(&d.tabs[k].strs[a]);
            d.tabs[k].price[a] = d.pricing.pi * l0 * (1.0 + d.pricing.k / e.max(1e-3));
        }
    }
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

fn prune_to(d: &mut Dict, c: &Corpus, budget: usize, threads: usize, protect_chars: bool, step: f64) {
    let mut pass = 0;
    let priced = d.pricing.pi > 0.0 || d.pricing.pi_core > 0.0 || d.pricing.conc > 0.0;
    while (d.rows() > budget || priced) && pass < 100 {
        pass += 1;
        let tq = Instant::now();
        let st = parse_corpus(d, c, threads, true);
        prof("prune_parse", tq);
        let tq = Instant::now();
        update_costs(d, &st);
        update_spec(d, &st);
        update_pmi(d, &st);
        update_cond(d, &st);
        update_partner_prices(d, &st);
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
        };
        let term_of = |d: &Dict, (sc, text): &mut (Scratch, Vec<u32>), ban: (usize, u32), ti: u32| -> f64 {
            let (t, n) = &types[ti as usize];
            d.token_text_into(t, text);
            let (alt, _) = d.parse(text, sc, None, Some(ban));
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
        let loss_of = |d: &Dict, k: usize, i: u32, limit: f64, sc: &mut (Scratch, Vec<u32>)| -> f64 {
            let mut loss = 0.0;
            for &ti in uses[k].get(i as usize) {
                loss += term_of(d, sc, (k, i), ti);
                if loss > limit {
                    break;
                }
            }
            loss
        };
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
            break;
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
        while pos < order.len() {
            if done >= need {
                break;
            }
            while pos >= safe {
                widen(d, &dropped, &mut acc, &mut lazy);
                net = net_of(&acc);
                order = sorted(&net);
                safe = safe_of(&acc, &net, lazy);
            }
            let o = order[pos];
            pos += 1;
            let (k, i) = cand[o];
            if net[o].is_finite() {
                // re-check against the rows already dropped
                let limit = prices[o] + net[o].max(cutoff) + 1e-9;
                if loss_of(d, k, i, limit, &mut sc) > limit {
                    continue;
                }
            }
            d.tabs[k].alive[i as usize] = false;
            dropped.push((k, i));
            removed[k] += 1;
            done += 1;
        }
        prof("prune_recheck", tq);
        let tq = Instant::now();
        d.rebuild();
        prof("prune_rebuild", tq);
        eprintln!("      prune pass {pass}: -{} cores -{} prefixes -{} suffixes (cutoff {:.0} tokens net, {below} rows below price)",
                  removed[0], removed[1], removed[2], cutoff);
    }
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

/// Row key: (table, string)
type RowKey = (u8, Vec<u32>);

/// Score every missing row by package demand; returns (demand, row) in no particular order.
fn propose(d: &Dict, x: &[u32], subs: &[(u32, u16, u32)], max_packages: usize, threads: usize) -> Vec<(f64, RowKey)> {
    // value of each substring under the current dictionary
    let tq = Instant::now();
    let mut vals: Vec<(f64, usize)> = par_map(subs.len(), threads, 1024, Scratch::new, |sc, k| {
        let (p, l, c) = subs[k];
        let (t, _) = d.parse(&x[p as usize..p as usize + l as usize], sc, None, None);
        (t > 1.0 + 1e-9).then(|| (c as f64 * (t - 1.0), k))
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
            let p = &w[..a];
            let p_ok = cs.p_ok[a];
            let mut m = 0u8;
            let mut node = Some(0u32);
            cs.folded.clear();
            for b in a + 1..=l {
                let ch = w[b - 1];
                m = if b == a + 1 { d.sym.mask(ch) } else { combine(m, d.sym.mask(ch)) };
                if variation(m).is_none() {
                    break;
                }
                let f = d.sym.fold[ch as usize];
                cs.folded.push(f);
                node = node.and_then(|n| d.tries[0].step(n, f));
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
    let contrib: Vec<Vec<(u32, Miss, f64)>> = par_map(vals.len().div_ceil(PB), threads, 1, || (CutScratch::default(), Vec::new(), Vec::new()), |(cs, key, score), b| {
        let mut out = Vec::new();
        for pk in b * PB..((b + 1) * PB).min(vals.len()) {
            let g = vals[pk].0;
            let w = word(pk);
            cuts(w, cs);
            score.clear();
            for c in &cs.cuts {
                let sum: f64 = c.1[..c.0]
                    .iter()
                    .map(|&m| {
                        miss_key(d, w, m, key);
                        demand1[shard_of(key, nsh)].get(&key[..]).copied().unwrap_or(0.0)
                    })
                    .sum::<f64>();
                score.push(sum / (c.0 * c.0) as f64);
            }
            // the last of equally good cuts, as Iterator::max_by
            let best = (0..cs.cuts.len()).max_by(|&a, &b| score[a].partial_cmp(&score[b]).unwrap());
            if let Some(bi) = best {
                let c = cs.cuts[bi];
                for &m in &c.1[..c.0] {
                    out.push((pk as u32, m, g / c.0 as f64));
                }
            }
        }
        out
    });
    let demand2 = accumulate(&contrib);
    drop(contrib);
    prof("prop_pass2", tq);
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
}

/// Demand-map key of a missing row: [table, symbols..] (cores folded); orders like RowKey.
fn miss_key(d: &Dict, w: &[u32], m: Miss, key: &mut Vec<u32>) {
    key.clear();
    key.push(m.t as u32);
    let r = &w[m.a as usize..m.b as usize];
    if m.t == 0 {
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
    let legal = |raw: &[u32], f: &mut Vec<u32>| -> bool {
        let mut m = 0u8;
        f.clear();
        for (i, &ch) in raw.iter().enumerate() {
            if is_byte(ch) {
                return false;
            }
            m = if i == 0 { d.sym.mask(ch) } else { combine(m, d.sym.mask(ch)) };
            f.push(d.sym.fold[ch as usize]);
        }
        variation(m).is_some()
    };
    // per block of types: the candidate strings, and per shard (start, len, value) in order
    const G: usize = 4096;
    let blocks: Vec<(Vec<u32>, Vec<Vec<(u32, u32, f64)>>)> = par_map(types.len().div_ceil(G), threads, 1, || (Vec::new(), Vec::new(), Vec::new()), |(raw, f, core), b| {
        let mut syms: Vec<u32> = Vec::new();
        let mut ents: Vec<Vec<(u32, u32, f64)>> = vec![Vec::new(); nsh];
        for &(t, n) in &types[b * G..((b + 1) * G).min(types.len())] {
            if t.c == NONE || (t.p == 0 && t.s == 0) {
                continue;
            }
            let p = &d.tabs[1].strs[t.p as usize];
            core.clear();
            core.extend(d.sym.written(&d.tabs[0].strs[t.c as usize], t.v));
            let s = &d.tabs[2].strs[t.s as usize];
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
            let (gc, gp, gs) = (d.tabs[0].glue[t.c as usize], d.tabs[1].glue[t.p as usize], d.tabs[2].glue[t.s as usize]);
            for kind in 0..3 {
                let (save, spec, pair, condg, glue) = match kind {
                    0 if t.p != 0 => (save_p, spec_p, pair_p, cond_p, gc + gp),
                    1 if t.s != 0 => (save_s, spec_s, pair_s, cond_s, gc + gs),
                    2 if t.p != 0 && t.s != 0 => (save_p + save_s, spec_p + spec_s, pair_p + pair_s, cond_p + cond_s, gc + gp + gs),
                    _ => continue,
                };
                let value = n * d.lam * save + n * d.mu * spec + if d.mu2 > 0.0 { n * d.mu2 * pair } else { 0.0 }
                    + if d.lamc > 0.0 { n * d.lamc * condg } else { 0.0 };
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
            let key = if k == 0 { fold_str(d, s) } else { s.clone() };
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
    type Key = (u32, u32, u32); // (core, side x 4 + variation, piece id)
    let nsh = (4 * threads).max(1);
    let found: Vec<Vec<Vec<u32>>> = par_map(nsh, threads, 1, || Vec::new(), |raw: &mut Vec<u32>, h| {
        let mut merges: HashMap<Key, (f64, f64), Fast> = HashMap::default();
        for &(t, n) in &types {
            if t.c as usize % nsh != h {
                continue;
            }
            for (side, r) in [(0u32, t.p), (1, t.s)] {
                for &(pid, whole) in &pieces[side as usize][r as usize] {
                    let e = merges.entry((t.c, side * 4 + t.v, pid)).or_insert((0.0, 0.0));
                    e.0 += n;
                    if whole {
                        e.1 += n;
                    }
                }
            }
        }
        let mut out = Vec::new();
        for (&(c, sv, pid), &(n, n_whole)) in &merges {
            let (side, v) = (sv / 4, sv % 4);
            let piece = piece_strs[side as usize][pid as usize];
            let k = if side == 0 { 1 } else { 2 };
            let arow = if n_whole > 0.0 { d.tabs[k].map.get(piece).copied() } else { None };
            // dominated from either side: the core is mostly used with this piece (happines + s),
            // or the whole affix is mostly used with this core (priorit + ize_: once prioritize_
            // exists, priorit is left to priority, which the core side then proposes)
            let core_dom = n >= 0.5 * uses[0][c as usize];
            let affix_dom = arow.is_some_and(|r| n_whole >= 0.5 * uses[k][r as usize]);
            if !core_dom && !affix_dom {
                continue;
            }
            raw.clear();
            if side == 0 {
                raw.extend_from_slice(piece);
            }
            raw.extend(d.sym.written(&d.tabs[0].strs[c as usize], v));
            if side == 1 {
                raw.extend_from_slice(piece);
            }
            let mut m = 0u8;
            let mut f = Vec::with_capacity(raw.len());
            let mut ok = true;
            for (i, &ch) in raw.iter().enumerate() {
                if is_byte(ch) {
                    ok = false;
                    break;
                }
                m = if i == 0 { d.sym.mask(ch) } else { combine(m, d.sym.mask(ch)) };
                f.push(d.sym.fold[ch as usize]);
            }
            if !ok || variation(m).is_none() || d.tabs[0].has(&f) || !d.row_ok(0, &f) {
                continue;
            }
            // freed: the core's price in proportion to the uses that move, plus the piece's own price
            // (as an affix row) for the uses where it was the whole affix
            let piece_price = d.tabs[k].map.get(piece).map_or(0.0, |&r| n_whole / uses[k][r as usize].max(1e-9) * d.tabs[k].price[r as usize]);
            let freed = n / uses[0][c as usize] * d.tabs[0].price[c as usize] + piece_price;
            if freed > d.price_of(0, &f) {
                out.push(f);
            }
        }
        out
    });
    prof("sw_merge", tq);
    let mut v: Vec<Vec<u32>> = found.into_iter().flatten().collect();
    v.sort_unstable();
    v.dedup();
    v
}

// ------------------------------------------------------------------ exact candidate scoring

/// Exact gain of each candidate row on its own: sum over segments containing it of
/// f(x) (J_D(x) - J_{D+row}(x)), from up to `occ` sampled segments, scaled to all of them.
fn exact_gains(d: &Dict, c: &Corpus, base: &[f64], cands: &[RowKey], occ: usize, threads: usize) -> Vec<f64> {
    // tries over the candidate strings: raw text for affixes, folded text for cores
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
                        let sym = if k == 1 { d.sym.fold[ch as usize] } else { ch };
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
    // Long segments (whole paragraphs): segment by segment, the plain parse once, then every
    // candidate sampled there re-scored by parse_x_window, which re-runs only the DP around its
    // occurrences. Same values as the full re-parses below (see parse_x_window), so the choice is
    // only about speed; short segments (chunks) are cheaper to re-parse whole.
    let mean_len = c.segs.iter().map(|s| s.len()).sum::<usize>() as f64 / ns.max(1) as f64;
    if d.mu2 == 0.0 && d.lamc == 0.0 && mean_len >= WINDOW_MIN_LEN {
        let r = exact_gains_windowed(d, c, base, cands, &tries, &term, &post_of, &med, &med_spec, threads);
        prof("eg_score", tq);
        return r;
    }
    let r = par_map(cands.len(), threads, 4, Scratch::new, |sc, i| {
        let (t, st) = &cands[i];
        let e = Extra { t: *t as usize, s: st.clone(), cost: med[*t as usize], spec: med_spec[*t as usize], usec: d.usec_of(*t as usize, st) };
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

/// exact_gains switches to windowed re-scoring when segments are this long on average (chars).
const WINDOW_MIN_LEN: f64 = 48.0;

type WinScratch = (Arcs, [Vec<(f64, f64)>; 3], [Vec<(f64, f64)>; 3], [Vec<Back>; 3], Vec<(u32, u32)>, Vec<u32>);

/// The last step of exact_gains for parse_x (mu2 = lamc = 0), by segment instead of by candidate:
/// the same values (parse_x_window), summed per candidate in the same order.
#[allow(clippy::too_many_arguments)]
fn exact_gains_windowed<P: Iterator<Item = (u32, f64)>>(d: &Dict, c: &Corpus, base: &[f64], cands: &[RowKey], tries: &[Trie; 2],
                                                         term: &[Vec<Vec<u32>>; 2], post_of: &(impl Fn(usize) -> P + Sync),
                                                         med: &[f64], med_spec: &[f64], threads: usize) -> Vec<f64> {
    let ns = c.segs.len();
    let extras: Vec<Extra> = cands
        .iter()
        .map(|(t, s)| Extra { t: *t as usize, s: s.clone(), cost: med[*t as usize], spec: med_spec[*t as usize], usec: d.usec_of(*t as usize, s) })
        .collect();
    let lmax = d.max_arc(0);
    let mut out = Vec::with_capacity(cands.len());
    // candidates in batches of about BATCH postings, so memory stays bounded
    const BATCH: usize = 1 << 23;
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
        let js: Vec<Vec<f64>> = par_map(segs.len(), threads, 1, new_ws, |(arcs, fin, w, nob, hits, occ): &mut WinScratch, q| {
            let si = segs[q] as usize;
            let x = &c.segs[si];
            let ps = by_seg.get(si);
            // occurrences of the sampled candidates: (index in ps, start), as exact_gains finds them
            hits.clear();
            for i in 0..x.len() {
                for k in 0..2 {
                    let mut n = 0u32;
                    for &ch in &x[i..] {
                        if is_byte(ch) {
                            break;
                        }
                        let sym = if k == 1 { d.sym.fold[ch as usize] } else { ch };
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
            // the plain parse, from the text's arcs (walked once for all its candidates)
            d.arcs_x(x, arcs);
            d.scores_x(x, arcs, fin, nob);
            let mut h = 0;
            (0..ps.len())
                .map(|r| {
                    occ.clear();
                    while h < hits.len() && hits[h].0 as usize == r {
                        occ.push(hits[h].1);
                        h += 1;
                    }
                    let e = &extras[post[ps[r] as usize].2 as usize];
                    d.parse_x_window(x, arcs, fin, occ, lmax.max(e.s.len()), e, w, nob)
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
        Symbols { fold: self.u32s(n), flag: self.u32s(n), upper_of: self.u32s(n), trad_of: self.u32s(n), nbytes: vec![], alnum: vec![] }
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
    put(&mut w, &[n_sym as u32]);
    for a in [&d.sym.fold, &d.sym.flag, &d.sym.upper_of, &d.sym.trad_of] {
        put(&mut w, a);
    }
    let keep: Vec<Vec<usize>> =
        (0..3).map(|k| (0..d.tabs[k].strs.len()).filter(|&i| d.tabs[k].alive[i]).collect()).collect();
    for k in 0..3 {
        put(&mut w, &[keep[k].len() as u32]);
        for &i in &keep[k] {
            put(&mut w, &[d.tabs[k].strs[i].len() as u32]);
            put(&mut w, &d.tabs[k].strs[i]);
        }
    }
    for k in 0..3 {
        for &i in &keep[k] {
            w.write_all(&d.tabs[k].cost[i].to_le_bytes()).unwrap();
        }
    }
    for c in d.vcost {
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
    let train = r.corpus(true);
    let dev = r.corpus(true);
    let classes = r.u32() != 0;
    let mut allow: [std::collections::HashSet<Vec<u32>, Fast>; 3] = Default::default();
    for k in 1..3 {
        allow[k] = r.table().into_iter().collect();
    }
    let init: Option<[Vec<Vec<u32>>; 3]> = if r.u32() != 0 { Some([r.table(), r.table(), r.table()]) } else { None };
    let script: Vec<u8> = if r.pos < r.buf.len() && r.u32() != 0 { r.u32s(n_sym).into_iter().map(|g| g as u8).collect() } else { Vec::new() };
    drop(r);
    eprintln!("  input read in {:.1}s", tr0.elapsed().as_secs_f64());
    let t0 = Instant::now();
    let chars: f64 = train.segs.iter().zip(&train.freqs).map(|(s, f)| s.len() as f64 * f).sum();
    let dev_chars: f64 = dev.segs.iter().zip(&dev.freqs).map(|(s, f)| s.len() as f64 * f).sum();

    // raw text for the substring index (segments separated by NONE; repeated segments once)
    let mut x: Vec<u32> = Vec::new();
    let mut wt: Vec<u32> = Vec::new();
    for (s, &f) in train.segs.iter().zip(&train.freqs) {
        x.extend(s);
        x.push(NONE);
        wt.extend(std::iter::repeat(f as u32).take(s.len() + 1));
    }
    let mut char_n = vec![1.0f64; n_sym];
    for (s, f) in train.segs.iter().zip(&train.freqs) {
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
    let frag = if nu > 0.0 || price_pi > 0.0 || pi_core > 0.0 { fragment_scores(&sym, &x, &wt, &subs, frag_t, glue_h0, min_freq) } else { Default::default() };
    let mut pricing = Pricing {
        pi: price_pi, pi_core, k: prod_k, conc, char_bits: char_bits.clone(), dstems: Default::default(), rare, n0: partner_n0,
        script: Vec::new(), script_chars: Vec::new(), rawcnt: HashMap::default(), min_freq: min_freq as f64,
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

    // initial dictionary: the alphabet as cores (folded), empty affixes
    let mut tabs = [Table::new(false), Table::new(true), Table::new(true)];
    for i in 0..n_sym as u32 {
        if sym.fold[i as usize] == i {
            tabs[0].add(&[i]);
        }
    }
    if let Some(init) = &init {
        for k in 0..3 {
            for row in &init[k] {
                tabs[k].add(row);
            }
        }
        eprintln!("  starting from {} cores, {} prefixes, {} suffixes", tabs[0].live(), tabs[1].live(), tabs[2].live());
    }
    let empty = Trie::empty;
    let mut inc = Dict {
        sym: &sym, tabs, vcost: [0.0; 4], delta, lam, mu, h0, alnum_only, classes, allow: allow.clone(), mu2, tau,
        lamc, cn: Default::default(), ncore: Vec::new(),
        pmi: [HashMap::default(), HashMap::default()], tries: [empty(), empty(), empty()], nu, frag: &frag, pricing: &pricing,
    };
    inc.rebuild();
    let st = parse_corpus(&inc, &train, threads, false);
    update_costs(&mut inc, &st);
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
    for round in 0..max_rounds {
        let tr = Instant::now();
        let tp = Instant::now();
        let mut props = propose(&inc, &x, &subs, max_packages, threads);
        let t_prop = tp.elapsed().as_secs_f64();
        if refactor && (lam > 0.0 || nu > 0.0) {
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
            if swaps.binary_search(&f).is_ok() {
                eprintln!("    trace[{i}] swap proposed{}", if tabu.contains(&(0u8, f.clone())) { " (but tabu)" } else { "" });
            }
        }
        prof("swap_props", tq);
        let tq = Instant::now();
        let mut work = Dict {
            sym: &sym, tabs: inc.tabs.clone(), vcost: inc.vcost, delta, lam, mu, h0, alnum_only, classes, allow: allow.clone(), mu2, tau,
            lamc, cn: inc.cn.clone(), ncore: inc.ncore.clone(),
            pmi: inc.pmi.clone(), tries: [empty(), empty(), empty()], nu, frag: &frag, pricing: &pricing,
        };
        prof("clone_dict", tq);
        let tq = Instant::now();
        let target = if round == 0 { (budget as f64 * first_expand) as usize } else { budget + k_extra };
        let need = target.saturating_sub(inc.rows());
        let rerank = rerank_mult > 0 && round > 0;
        // proposals best first (value, then key: a total order). Every proposal is a row the incumbent
        // lacks, so the adding below stops after `need` of them (after the re-scored shortlist when
        // re-ranking): only that many have to be found and sorted, not the millions proposed.
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
            eprintln!("    re-scored {m} proposals exactly: {zero} save nothing ({:.0}s)", te.elapsed().as_secs_f64());
            let tail: Vec<&RowKey> = order[m..].to_vec();
            order = idx.iter().map(|&i| order[i]).chain(tail).collect();
            prof("rerank_sort", tq);
        }
        let tq = Instant::now();
        let mut added = [0usize; 3];
        let mut added_keys: Vec<RowKey> = Vec::new();
        let mut rows = work.rows(); // kept up to date: counting live rows per add is quadratic
        let mut n_swaps = 0;
        for f in swaps {
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
        let tq = Instant::now();
        work.rebuild();
        prof("rebuild", tq);
        let tw = Instant::now();
        for _ in 0..em_iters {
            let st = parse_corpus(&work, &train, threads, mu > 0.0 || mu2 > 0.0 || lamc > 0.0);
            update_costs(&mut work, &st);
            update_spec(&mut work, &st);
            update_pmi(&mut work, &st);
            update_cond(&mut work, &st);
        }
        let t_em = tw.elapsed().as_secs_f64();
        let tw = Instant::now();
        prune_to(&mut work, &train, budget, threads, protect_chars, step);
        let t_prune = tw.elapsed().as_secs_f64();
        let tw = Instant::now();
        let tq = Instant::now();
        tabu = added_keys.into_iter().filter(|(t, s)| !work.tabs[*t as usize].has(s)).collect();
        prof("tabu", tq);
        let tq = Instant::now();
        for _ in 0..em_iters {
            let st = parse_corpus(&work, &train, threads, mu > 0.0 || mu2 > 0.0 || lamc > 0.0);
            update_costs(&mut work, &st);
            update_spec(&mut work, &st);
            update_pmi(&mut work, &st);
            update_cond(&mut work, &st);
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
            if fails >= 2 {
                prof("accept_save", tq);
                eprintln!("    prof: {}", prof_dump());
                break;
            }
        }
        prof("accept_save", tq);
        eprintln!("    prof: {}", prof_dump());
        // early stop (min_gain_ppm): the round improved the best dev chars/token too little
        if min_gain > 0.0 && round > 0 && prev_best.is_finite() {
            let gain = (prev_best / best_dev - 1.0) * 1e6;
            if gain < min_gain {
                eprintln!("  stopping: best dev chars/token improved by {gain:.0} ppm < {min_gain} ppm this round");
                break;
            }
        }
    }
    if best_dev == f64::INFINITY {
        save(&inc, n_sym, &args[3]);
    }
    eprintln!("  done in {:.0}s; best dev chars/token {:.3}", t0.elapsed().as_secs_f64(), dev_chars / best_dev);
}

fn encode(args: &[String]) {
    let mut m = Reader::open(&args[2]);
    let n_sym = m.u32() as usize;
    let mut sym = m.symbols(n_sym);
    let strs = [m.table(), m.table(), m.table()];
    let costs: Vec<Vec<f64>> = (0..3).map(|k| (0..strs[k].len()).map(|_| m.f64()).collect()).collect();
    let vcost = [m.f64(), m.f64(), m.f64(), m.f64()];
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
    let empty = Trie::empty;
    let no_frag: [HashMap<u64, f32, Fast>; 3] = Default::default();
    let no_pricing = Pricing::default();
    let mut d = Dict { sym: &sym, tabs, vcost, delta, lam, mu, h0: 0.0, alnum_only: false, classes: false, allow: Default::default(), mu2, tau, pmi, lamc, cn, ncore, tries: [empty(), empty(), empty()], nu, frag: &no_frag, pricing: &no_pricing };
    d.rebuild();
    let c = Reader::open(&args[3]).corpus(false);
    let mut w = BufWriter::new(std::fs::File::create(&args[4]).expect("create output"));
    let mut sc = Scratch::new();
    let mut toks = Vec::new();
    for s in &c.segs {
        d.parse(s, &mut sc, Some(&mut toks), None);
        put(&mut w, &[toks.len() as u32]);
        for t in &toks {
            put(&mut w, &[t.v, t.p, t.c, t.s]);
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
