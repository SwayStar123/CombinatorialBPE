//! BPE merge loop for `cbpe.CharBPE.train`.
//!
//! usage: bpe_train INPUT OUTPUT
//!
//! INPUT (little endian):
//!     u32 n_symbols      alphabet size; symbol ids 0..n_symbols, u32::MAX = never merges
//!     u32 n_merges       number of merges to learn
//!     u64 n_words, u64 n_ids
//!     i64 counts[n_words], u32 lens[n_words], u32 ids[n_ids]   (words concatenated)
//! OUTPUT:
//!     u32 n_learned, then n_learned x (u32 a, u32 b, i64 count); merge k creates symbol n_symbols + k
//!
//! Same algorithm and tie-breaking as the pure-Python trainer (highest count first, then the
//! smallest (a, b) id pair), so both produce identical merges.
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{BufWriter, Write};

#[derive(Default)]
struct Fx(u64);

impl Hasher for Fx {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.write_u64(*b as u64);
        }
    }
    fn write_u32(&mut self, i: u32) {
        self.write_u64(i as u64);
    }
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

type Fast = BuildHasherDefault<Fx>;
type Pair = (u32, u32);
const NONE: u32 = u32::MAX;

struct Reader {
    buf: Vec<u8>,
    pos: usize,
}

impl Reader {
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
}

fn pairs(ids: &[u32]) -> impl Iterator<Item = Pair> + '_ {
    ids.windows(2).map(|w| (w[0], w[1])).filter(|p| p.0 != NONE && p.1 != NONE)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(args.len() == 3, "usage: bpe_train INPUT OUTPUT");
    let mut r = Reader { buf: std::fs::read(&args[1]).expect("read input"), pos: 0 };
    let n_symbols = r.u32();
    let n_merges = r.u32() as usize;
    let n_words = r.u64() as usize;
    let _n_ids = r.u64() as usize;
    let freqs: Vec<i64> = (0..n_words).map(|_| r.i64()).collect();
    let lens: Vec<usize> = (0..n_words).map(|_| r.u32() as usize).collect();
    let mut words: Vec<Vec<u32>> = lens.iter().map(|&n| (0..n).map(|_| r.u32()).collect()).collect();

    let mut pair_counts: HashMap<Pair, i64, Fast> = HashMap::default();
    let mut locs: HashMap<Pair, HashSet<u32, Fast>, Fast> = HashMap::default();
    for (wi, ids) in words.iter().enumerate() {
        for p in pairs(ids) {
            *pair_counts.entry(p).or_insert(0) += freqs[wi];
            locs.entry(p).or_default().insert(wi as u32);
        }
    }
    let mut heap: BinaryHeap<(i64, Reverse<Pair>)> = pair_counts.iter().map(|(&p, &c)| (c, Reverse(p))).collect();

    let mut merges: Vec<(u32, u32, i64)> = Vec::with_capacity(n_merges);
    while merges.len() < n_merges {
        let Some((c, Reverse(pair))) = heap.pop() else { break };
        let cur = *pair_counts.get(&pair).unwrap_or(&0);
        if cur <= 0 {
            continue;
        }
        if c != cur {
            // stale entry: counts of existing pairs only ever decrease
            heap.push((cur, Reverse(pair)));
            continue;
        }
        let (a, b) = pair;
        let new_id = n_symbols + merges.len() as u32;
        merges.push((a, b, cur));

        let mut touched: HashMap<Pair, i64, Fast> = HashMap::default();
        let ws = locs.remove(&pair).unwrap_or_default();
        let mut ws: Vec<u32> = ws.into_iter().collect();
        ws.sort_unstable();
        for wi in ws {
            let ids = &words[wi as usize];
            if ids.len() < 2 || !ids.windows(2).any(|w| w[0] == a && w[1] == b) {
                continue;
            }
            let f = freqs[wi as usize];
            for p in pairs(ids) {
                *touched.entry(p).or_insert(0) -= f;
            }
            let mut out = Vec::with_capacity(ids.len());
            let mut i = 0;
            while i < ids.len() {
                if i + 1 < ids.len() && ids[i] == a && ids[i + 1] == b {
                    out.push(new_id);
                    i += 2;
                } else {
                    out.push(ids[i]);
                    i += 1;
                }
            }
            for p in pairs(&out) {
                *touched.entry(p).or_insert(0) += f;
                locs.entry(p).or_default().insert(wi);
            }
            words[wi as usize] = out;
        }
        for (p, d) in touched {
            if d == 0 {
                continue;
            }
            let e = pair_counts.entry(p).or_insert(0);
            *e += d;
            if d > 0 {
                heap.push((*e, Reverse(p)));
            }
        }
        pair_counts.remove(&pair);
    }

    let mut w = BufWriter::new(std::fs::File::create(&args[2]).expect("create output"));
    w.write_all(&(merges.len() as u32).to_le_bytes()).unwrap();
    for (a, b, c) in merges {
        w.write_all(&a.to_le_bytes()).unwrap();
        w.write_all(&b.to_le_bytes()).unwrap();
        w.write_all(&c.to_le_bytes()).unwrap();
    }
}
