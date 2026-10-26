"""Pure-Python encoder for unrestricted Combinatorial BPE models: the same segmentation as
`dict_search encode` (native/dict_search), for machines without the Rust binary. Slower (roughly
10-20x), identical output: tests/test_unrestricted_py.py compares them chunk by chunk.

It ports the DP of the binary's `encode` for case_affixes models ("CAFX" files) trained without the
pairwise / code-length terms (mu2 = lamc = 0, no CLEN section), which is what the published
Qwen3 model uses; other model kinds raise NotImplementedError (use the binary for them). A chunk is
parsed by the exact minimum-cost DP over three states (B: at a token boundary, P: prefix done, C:
core done), with one score per variation state in P and C (see the header of main.rs): primary cost
= tokens + sum over tokens of [delta x non-empty affixes + lambda x affix bits], secondary cost =
sum of the rows' bits, ties within 1e-9 broken exactly as `better` does, relaxations in the order of
`step_xa`, so that equal-cost parses resolve to the same one.
"""
from __future__ import annotations

import struct

NONE = 0xFFFFFFFF
NONE_BASE = 0xFFFFFFF0
INF = float("inf")
NVS = 10
NVAR_MAX = 8
CF_UP, CF_TRAD, CF_HS, CF_HASU, CF_BAD = 1, 2, 4, 8, 16
TM_AS, TM_CAP, TM_LOW, TM_TITLE, TM_ALL, TM_CAPS = 1, 2, 16, 32, 63, 64
TM_EMPTY = TM_ALL
VS_CAP_PENDING, VS_TITLE_PENDING, VS_CAP_DONE, VS_TITLE_DONE = 1, 7, 8, 9
CAFX_MAGIC = int.from_bytes(b"CAFX", "little")
MARK_MAGIC = int.from_bytes(b"MARK", "little")
CODE_MAGIC = int.from_bytes(b"CLEN", "little")
U64 = (1 << 64) - 1


def _vs_move(m, tm):
    if tm & TM_CAPS == 0:
        return m
    if m & 1 << VS_CAP_PENDING:
        m = (m & ~(1 << VS_CAP_PENDING)) | 1 << VS_CAP_DONE
    if m & 1 << VS_TITLE_PENDING:
        m = (m & ~(1 << VS_TITLE_PENDING)) | 1 << VS_TITLE_DONE
    return m


def _vs_allow(tm):
    m = tm & 31
    if tm & TM_CAP:
        m |= 3 << 5
    if tm & TM_TITLE:
        m |= 1 << VS_TITLE_PENDING
    if tm & TM_AS:
        m |= 1 << VS_CAP_DONE
    if tm & TM_LOW:
        m |= 1 << VS_TITLE_DONE
    return m


def _vs_start(tm):
    m = tm & 31
    if tm & TM_AS:
        m |= 1 << 5
    if tm & TM_CAP:
        m |= 1 << 6
    if tm & TM_TITLE:
        m |= 1 << VS_TITLE_PENDING
    return _vs_move(m, tm)


def _vs_to(vs, tm):
    if tm & TM_CAPS == 0:
        return vs
    if vs == VS_CAP_PENDING:
        return VS_CAP_DONE
    if vs == VS_TITLE_PENDING:
        return VS_TITLE_DONE
    return vs


def _vs_var(vs):
    return 1 if vs == VS_CAP_DONE else 7 if vs == VS_TITLE_DONE else vs


ALLOW = [_vs_allow(tm) for tm in range(128)]
START = [_vs_start(tm) for tm in range(128)]
# per tm: [(vs, vs_to(vs, tm), vs_var(vs))] for the states a core / suffix with that mask can follow
ALLOW_STATES = [[(vs, _vs_to(vs, tm), _vs_var(vs)) for vs in range(NVS) if ALLOW[tm] >> vs & 1] for tm in range(128)]
START_STATES = [[vs for vs in range(NVS) if START[tm] >> vs & 1] for tm in range(128)]
START_EMPTY = START_STATES[TM_EMPTY]


def _better(a0, a1, b0, b1):
    """parse score (a0, a1) strictly better than (b0, b1): primary within 1e-9, then secondary"""
    return a0 < b0 - 1e-9 or (a0 <= b0 + 1e-9 and a1 < b1)


def _tm(up, hasu, trad, hs, hi_up, hi_v2, hi_h, bad, uc):
    """CaseAcc::tm: the transition mask of a span written from canonical upper mask uc"""
    if bad:
        return 0
    plain = not trad and not hi_up
    ok0 = plain and up == uc
    m = 0
    if ok0:
        m |= TM_AS
    low = hasu & -hasu & U64  # the first char with an uppercase form
    if hasu or hi_h:
        m |= TM_CAPS
        if hasu and plain and up == (uc | low):
            m |= TM_CAP
    elif ok0:
        m |= TM_CAP
    if not trad and not hi_v2 and up == hasu:
        m |= 1 << 2
    if not hs and not hi_up and up == uc:
        m |= 1 << 3
    if plain and up == 0:
        m |= TM_LOW
    if hasu or hi_h:
        if hasu and plain and up == low:
            m |= TM_TITLE
    elif plain and up == 0:
        m |= TM_TITLE
    return m


class PyEncoder:
    def __init__(self, model_path: str):
        data = open(model_path, "rb").read()
        pos = 0

        def u32s(n):
            nonlocal pos
            out = struct.unpack_from(f"<{n}I", data, pos)
            pos += 4 * n
            return out

        def f64s(n):
            nonlocal pos
            out = struct.unpack_from(f"<{n}d", data, pos)
            pos += 8 * n
            return out

        if u32s(1)[0] != CAFX_MAGIC:
            raise NotImplementedError("the Python encoder supports case_affixes (CAFX) models only")
        n_sym = u32s(1)[0]
        fold, _flag, upper_of, trad_of = (u32s(n_sym) for _ in range(4))
        tables = [[u32s(u32s(1)[0]) for _ in range(u32s(1)[0])] for _ in range(3)]
        costs = [f64s(len(t)) for t in tables]
        vcost = f64s(NVAR_MAX)
        delta = f64s(1)[0]
        self.nbytes = u32s(n_sym)
        lam = f64s(1)[0]
        mu = f64s(1)[0] if pos + 8 <= len(data) else 0.0
        spec = [[0.0] * len(t) for t in tables]
        if pos < len(data):
            spec[1] = list(f64s(len(tables[1])))
            spec[2] = list(f64s(len(tables[2])))
        if pos + 16 <= len(data):  # mu2, tau, the PMI pair tables
            mu2, tau = f64s(2)
            for _ in range(2):
                m = u32s(1)[0]
                pos += 12 * m
            if mu2 > 0:
                raise NotImplementedError("pairwise (mu2) models need the binary")
        if pos + 12 <= len(data):  # lamc, n(core), the pair counts
            lamc = f64s(1)[0]
            m = u32s(1)[0]
            pos += 4 * m
            for _ in range(2):
                m = u32s(1)[0]
                pos += 12 * m
            if lamc > 0:
                raise NotImplementedError("pairwise (lamc) models need the binary")
        nu = 0.0
        if pos + 8 <= len(data):
            nu = f64s(1)[0]
            for k in range(3):
                if k > 0 and pos >= len(data):
                    break
                pos += 8 * len(tables[k])
        tag = u32s(1)[0] if pos + 4 <= len(data) else 0
        self.alnum = ()
        if tag == MARK_MAGIC:
            self.alnum = u32s(n_sym)
            tag = u32s(1)[0] if pos + 4 <= len(data) else 0
        if tag == CODE_MAGIC:
            raise NotImplementedError("code_len models need the binary")
        # (usec = nu x glue, with no glue statistics at encoding time: 0)
        self.fold, self.upper_of, self.trad_of = fold, upper_of, trad_of
        # per-symbol case classes (Symbols::init_case)
        cf = []
        for c in range(n_sym):
            f = fold[c]
            up, tr = upper_of[f], trad_of[f]
            m = CF_HASU if up != NONE else 0
            if f == c:
                if tr != NONE:
                    m |= CF_HS
            elif up == c:
                m |= CF_UP
            elif tr == c:
                m |= CF_TRAD
            else:
                m |= CF_BAD
            cf.append(m)
        self.cf = cf
        self.mark = [a == 3 for a in self.alnum] if self.alnum else None
        self.lam, self.mu, self.delta, self.vcost = lam, mu, delta, vcost
        # tries: folded string (chr per symbol) -> (row id, arc primary, arc secondary, upper mask)
        self.tries, self.maxlen = [], []  # (maxlen[k]: first folded char -> the longest row k starting with it)
        for k in range(3):
            d = {}
            for i, s in enumerate(tables[k]):
                um = 0
                for j, c in enumerate(s[:64]):
                    um |= (cf[c] & CF_UP != 0) << j
                if not s:
                    continue
                key = "".join(chr(fold[c]) for c in s)
                if key in d:
                    raise NotImplementedError("duplicate folded rows")
                a0 = 1.0 + 0.0 + 0.0 * costs[k][i] if k == 0 else delta + lam * costs[k][i] + mu * spec[k][i] + 0.0
                d[key] = (i, a0, costs[k][i], um)
            self.tries.append(d)
            ml = {}
            for key in d:
                if ml.get(key[0], 0) < len(key):
                    ml[key[0]] = len(key)
            self.maxlen.append(ml)
        self.cp0, self.cs0 = costs[1][0], costs[2][0]
        self.sp0, self.ss0 = spec[1][0], spec[2][0]
        assert not tables[1][0] and not tables[2][0]

    # -------------------------------------------------------------------- walk
    def _walk(self, k, x, fx, i):
        """[(j + 1, row, arc primary, arc secondary, tm)] for every row spelling x[i..=j] folded (walk_ca)"""
        if k < 2 and self.mark is not None and x[i] < NONE_BASE and self.mark[x[i]]:
            return ()  # mark_rule: no core or prefix starts at a combining mark
        trie, cf = self.tries[k], self.cf
        n = len(x)
        lim = self.maxlen[k].get(fx[i], 0)
        if not lim:
            return ()
        out = []
        up = hasu = 0
        trad = hs = hi_up = hi_v2 = hi_h = bad = False
        key = ""
        for j in range(i, min(n, i + lim)):
            c = x[j]
            if c >= NONE_BASE:
                break
            key += fx[j]
            f = cf[c]
            u, h = f & CF_UP != 0, f & CF_HASU != 0
            q = j - i
            if q < 64:
                up |= u << q
                hasu |= h << q
            else:
                hi_up |= u
                hi_v2 |= u != h
                hi_h |= h
            trad |= f & CF_TRAD != 0
            hs |= f & CF_HS != 0
            bad |= f & CF_BAD != 0
            if bad or (trad and hs):
                break
            e = trie.get(key)
            if e is not None:
                tm = _tm(up, hasu, trad, hs, hi_up, hi_v2, hi_h, bad, e[3])
                if tm & TM_ALL:
                    out.append((j + 1, e[0], e[1], e[2], tm))
        return out

    # ------------------------------------------------------------------ parse
    def parse(self, x):
        """tokens (v, p, c, s) of the chunk x (symbol ids; NONE_BASE + n for a char outside the alphabet)
        in the binary's output format: c == NONE for a byte-fallback char, whose v is its number of bytes"""
        n = len(x)
        fold, lam, mu, delta = self.fold, self.lam, self.mu, self.delta
        fx = ["" if c >= NONE_BASE else chr(fold[c]) for c in x]
        cp0, cs0, sp0, ss0 = self.cp0, self.cs0, self.sp0, self.ss0
        vcost = self.vcost
        # scores per position: d0[pos] = (a, b); d1[pos][vs], d2[pos][vs] (state P / C)
        d0 = [INF] * (n + 1), [INF] * (n + 1)
        d1 = [[(INF, INF)] * NVS for _ in range(n + 1)]
        d2 = [[(INF, INF)] * NVS for _ in range(n + 1)]
        # back pointers: (state, from, id, v, fvs)
        b0 = [None] * (n + 1)
        b1 = [[None] * NVS for _ in range(n + 1)]
        b2 = [[None] * NVS for _ in range(n + 1)]
        d0a, d0b = d0
        d0a[0] = 0.0
        d0b[0] = 0.0
        for i in range(n + 1):
            # empty closures, order C -> B -> P
            c2 = d2[i]
            ca, cb = lam * cs0 + mu * ss0, cs0
            for vs in range(NVS):
                o = c2[vs]
                if o[0] != INF:
                    c0, c1 = o[0] + ca, o[1] + cb
                    if _better(c0, c1, d0a[i], d0b[i]):
                        d0a[i], d0b[i] = c0, c1
                        b0[i] = (2, i, 0, 0, vs)
            o0, o1 = d0a[i], d0b[i]
            if o0 != INF:
                pa, pb = lam * cp0 + mu * sp0, cp0
                for vs in START_EMPTY:
                    c0, c1 = o0 + pa, o1 + pb
                    cur = d1[i][vs]
                    if _better(c0, c1, cur[0], cur[1]):
                        d1[i][vs] = (c0, c1)
                        b1[i][vs] = (0, i, 0, 0, 0)
            if i == n:
                break
            # byte fallback
            o0, o1 = d0a[i], d0b[i]
            if o0 != INF:
                nb = self.nbytes[x[i]] if x[i] < NONE_BASE else x[i] - NONE_BASE
                dt = nb * (1.0 + lam * (cp0 + cs0) + mu * (sp0 + ss0) + 0.0)
                c0, c1 = o0 + dt, o1 + 1000.0 * nb
                if _better(c0, c1, d0a[i + 1], d0b[i + 1]):
                    d0a[i + 1], d0b[i + 1] = c0, c1
                    b0[i + 1] = (0, i, NONE, 0, 0)
            # prefixes B -> P
            o0, o1 = d0a[i], d0b[i]
            if o0 != INF:
                for j1, row, a0, a1, tm in self._walk(1, x, fx, i):
                    c0, c1 = o0 + a0, o1 + a1
                    t = d1[j1]
                    for vs in START_STATES[tm]:
                        cur = t[vs]
                        if _better(c0, c1, cur[0], cur[1]):
                            t[vs] = (c0, c1)
                            b1[j1][vs] = (0, i, row, 0, 0)
            # cores P -> C
            src = d1[i]
            if any(o[0] != INF for o in src):
                for j1, row, a0, a1, tm in self._walk(0, x, fx, i):
                    t = d2[j1]
                    for vs, vto, v in ALLOW_STATES[tm]:
                        o = src[vs]
                        if o[0] != INF:
                            c0, c1 = o[0] + a0, o[1] + (a1 + vcost[v])
                            cur = t[vto]
                            if _better(c0, c1, cur[0], cur[1]):
                                t[vto] = (c0, c1)
                                b2[j1][vto] = (1, i, row, v, vs)
            # suffixes C -> B
            src = d2[i]
            if any(o[0] != INF for o in src):
                for j1, row, a0, a1, tm in self._walk(2, x, fx, i):
                    for vs, _vto, _v in ALLOW_STATES[tm]:
                        o = src[vs]
                        if o[0] != INF:
                            c0, c1 = o[0] + a0, o[1] + a1
                            if _better(c0, c1, d0a[j1], d0b[j1]):
                                d0a[j1], d0b[j1] = c0, c1
                                b0[j1] = (2, i, row, 0, vs)
        if d0a[n] == INF:
            return []
        out = []
        st, pos, vs = 0, n, 0
        cur = [0, 0, NONE, 0]
        while not (st == 0 and pos == 0):
            b = (b0[pos] if st == 0 else b1[pos][vs] if st == 1 else b2[pos][vs])
            state, frm, bid, bv, fvs = b
            if (state, st) == (0, 0):
                xi = x[frm]
                out.append((self.nbytes[xi] if xi < NONE_BASE else xi - NONE_BASE, 0, NONE, 0))
            elif (state, st) == (2, 0):
                cur[3] = bid
            elif (state, st) == (1, 2):
                cur[2], cur[0] = bid, bv
            elif (state, st) == (0, 1):
                cur[1] = bid
                out.append(tuple(cur))
                cur = [0, 0, NONE, 0]
            st, pos, vs = state, frm, fvs
        out.reverse()
        return out

    def encode_chunks(self, chunks_ids):
        """list of symbol-id lists -> list of token lists (the binary's output)"""
        return [self.parse(x) for x in chunks_ids]
