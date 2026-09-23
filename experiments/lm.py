"""Does the combinatorial tokenizer help an actual language model?

Trains the same small GPT on English Wikipedia with a standard BPE and with the
combinatorial BPE (equal embedding budget), and compares validation bits-per-byte,
which is tokenizer-independent.

Combinatorial model:
  input  embedding  = E_var[v] + E_pre[p] + E_core[c] + E_suf[s]
  output            = p(c | h) * p(v | h, c) * p(p | h, c) * p(s | h, c)
The second factor group is conditioned on the (teacher-forced) core via
g = LN(h + W E_core[c]); all output layers are tied to the input tables.
This is a proper distribution over 4-tuples, so -log p of the canonical encoding is an
upper bound on the code length of the text -> the bpb comparison is fair.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cbpe import CombinatorialBPE, load  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE = os.path.join(ROOT, "data", "cache")

# ---------------------------------------------------------------- tokenise
def encode_file(tok_path, text_path, max_chars=None):
    """Tokenise in a torch-free subprocess (encode.py) and cache the result as .npy."""
    key = f"{os.path.basename(tok_path)[:-5]}__{os.path.basename(text_path)[:-4]}_{max_chars}.npy"
    out = os.path.join(CACHE, key)
    if not os.path.exists(out):
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "encode.py"), tok_path,
                        text_path, out, str(max_chars or 0)], check=True)
    with open(text_path, encoding="utf-8") as f:
        text = f.read(max_chars) if max_chars else f.read()
    return np.load(out), len(text.encode("utf-8"))


# ------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d, bias=False), nn.GELU(), nn.Linear(4 * d, d, bias=False))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    """sizes: list of table sizes; 1 table = standard LM, 4 tables = combinatorial."""

    def __init__(self, sizes, d=512, n_layer=8, n_head=8, ctx=512, head="linear", head_hidden=1,
                 order=(2, 1, 0, 3)):
        super().__init__()
        self.sizes = sizes
        self.head = head
        self.order = tuple(order)  # chain head: factor prediction order (indices into var, pre, core, suf)
        self.emb = nn.ModuleList(nn.Embedding(n, d) for n in sizes)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d)
        if len(sizes) == 4:  # factor order in tuples: (var, pre, core, suf)
            if head == "linear":
                self.cond = nn.Linear(d, d, bias=False)
                self.ln_c = nn.LayerNorm(d)
            else:  # "mlp": one residual MLP on [h; e(core)]; "chain": core -> prefix -> var -> suffix
                n = 1 if head == "mlp" else 3
                hid = int(head_hidden * d)
                self.steps = nn.ModuleList(nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, hid, bias=False),
                                                         nn.GELU(), nn.Linear(hid, d, bias=False))
                                           for _ in range(n))
                self.ln_steps = nn.ModuleList(nn.LayerNorm(d) for _ in range(n))
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("o.weight") or n.endswith("mlp.2.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, y):
        """x, y: [B, T, F] int. Returns summed NLL (nats) over all target tokens, and per-factor sums."""
        B, T, _ = x.shape
        h = self.pos(torch.arange(T, device=x.device))
        for i, e in enumerate(self.emb):
            h = h + e(x[..., i])
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        if len(self.sizes) == 1:
            nll = F.cross_entropy((h @ self.emb[0].weight.T).float().flatten(0, 1), y[..., 0].flatten(),
                                  reduction="sum")
            return nll, [nll.detach()]
        core_e = self.emb[2]
        parts = [None] * 4

        def ce(g, i):
            return F.cross_entropy((g @ self.emb[i].weight.T).float().flatten(0, 1), y[..., i].flatten(),
                                   reduction="sum")

        if self.head != "chain":
            parts[2] = ce(h, 2)

        if self.head == "linear":
            g = self.ln_c(h + self.cond(core_e(y[..., 2])))
            for i in (0, 1, 3):
                parts[i] = ce(g, i)
        elif self.head == "mlp":
            g = h + self.steps[0](torch.cat([h, core_e(y[..., 2])], -1))
            g = self.ln_steps[0](g)
            for i in (0, 1, 3):
                parts[i] = ce(g, i)
        else:  # chain: each factor conditioned on all previously predicted ones, in self.order
            first = self.order[0]
            parts[first] = ce(h, first)
            g, prev = h, self.emb[first](y[..., first])
            for k, i in enumerate(self.order[1:]):
                g = g + self.steps[k](torch.cat([g, prev], -1))
                parts[i] = ce(self.ln_steps[k](g), i)
                prev = self.emb[i](y[..., i])
        return sum(parts), [p.detach() for p in parts]


# ---------------------------------------------------------------- training
@torch.no_grad()
def evaluate(model, val, ctx, n_bytes, bs=8):
    model.eval()
    n = (len(val) - 1) // ctx
    tot, parts, count = 0.0, None, 0
    for i in range(0, n, bs):
        idx = torch.arange(i, min(i + bs, n)) * ctx
        w = torch.stack([val[j:j + ctx + 1] for j in idx.tolist()]).cuda().long()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            nll, p = model(w[:, :-1], w[:, 1:])
        tot += nll.item()
        parts = [a.item() for a in p] if parts is None else [a + b.item() for a, b in zip(parts, p)]
        count += w.shape[0] * ctx
    model.train()
    # bytes covered by the evaluated targets (proportional share of the val text)
    covered = n_bytes * count / (len(val) - 1)
    return tot / math.log(2) / covered, [q / math.log(2) / covered for q in parts]


def train(tok_path, args):
    tok = load(tok_path)
    sizes = list(tok.sizes.values()) if isinstance(tok, CombinatorialBPE) else [tok.vocab_size]
    data, train_bytes = encode_file(tok_path, args.train_text)
    val, val_bytes = encode_file(tok_path, args.val_text, max_chars=args.val_chars)
    bytes_per_tok = train_bytes / len(data)
    print(f"[{args.tag or args.head}] {os.path.basename(tok_path)}: tables {sizes} (sum {sum(sizes)}), train tokens {len(data) / 1e6:.1f}M, "
          f"{bytes_per_tok:.2f} bytes/token", flush=True)
    data, val = torch.from_numpy(data), torch.from_numpy(val)

    torch.manual_seed(args.seed)
    model = GPT(sizes, args.d, args.layers, args.heads, args.ctx, args.head, args.head_hidden,
                [int(c) for c in args.order.split(",")]).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    n_emb = sum(e.weight.numel() for e in model.emb)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "emb" not in n and "pos" not in n]
    other = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "emb" not in n and "pos" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), fused=True)
    tokens_per_step = args.bs * args.ctx
    assert args.steps * tokens_per_step <= len(data), "would repeat data"
    # single pass over a random permutation of non-overlapping windows
    g = torch.Generator().manual_seed(args.seed)
    n_windows = (len(data) - 1) // args.ctx
    order = torch.randperm(n_windows, generator=g)[: args.steps * args.bs] * args.ctx

    log = {"tokenizer": os.path.basename(tok_path), "tag": args.tag or args.head, "sizes": sizes, "params": n_params, "emb_params": n_emb,
           "bytes_per_token": bytes_per_tok, "curve": []}
    t0 = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            bpb, parts = evaluate(model, val, args.ctx, val_bytes)
            seen = step * tokens_per_step
            log["curve"].append({"step": step, "tokens": seen, "bytes": seen * bytes_per_tok, "val_bpb": bpb,
                                 "val_bpb_parts": parts})
            print(f"  step {step:5d}  val bpb {bpb:.4f}  parts {[round(p, 4) for p in parts]}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if step == args.steps:
            break
        lr = args.lr * min(1, (step + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for grp in opt.param_groups:
            grp["lr"] = max(lr, args.lr * 0.1 * min(1, (step + 1) / args.warmup))
        idx = order[step * args.bs:(step + 1) * args.bs]
        opt.zero_grad(set_to_none=True)
        for chunk in idx.split(args.bs // args.accum):  # gradient accumulation (large vocabs)
            w = torch.stack([data[j:j + args.ctx + 1] for j in chunk.tolist()]).cuda(non_blocking=True).long()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                nll, _ = model(w[:, :-1], w[:, 1:])
            (nll / (args.bs * args.ctx)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    log["train_s"] = time.time() - t0
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizers", nargs="+")
    ap.add_argument("--train_text", default=os.path.join(ROOT, "data", "wiki_en_lm.txt"))
    ap.add_argument("--val_text", default=os.path.join(ROOT, "data", "wiki_en.test.txt"))
    ap.add_argument("--val_chars", type=int, default=2_000_000)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1, help="split each batch into this many micro-batches")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head", default="linear", choices=["linear", "mlp", "chain"])
    ap.add_argument("--head_hidden", type=float, default=1.0)
    ap.add_argument("--order", default="2,1,0,3", help="chain head order; 0=var 1=prefix 2=core 3=suffix")
    ap.add_argument("--tag", default="", help="label stored with the run")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "lm.json"))
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    results = json.load(open(args.out)) if os.path.exists(args.out) else []
    for t in args.tokenizers:
        r = train(t, args)
        r["args"] = {k: os.path.relpath(v, ROOT).replace("\\", "/") if k in ("train_text", "val_text", "out") else v
                     for k, v in vars(args).items()}
        results = [x for x in results if not (x["tokenizer"] == r["tokenizer"] and x["tag"] == r["tag"]
                                              and x["args"]["seed"] == args.seed)]
        results.append(r)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
