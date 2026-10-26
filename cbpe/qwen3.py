"""A Qwen3 model on a factored (variation, prefix, core, suffix) tokenizer: the PyTorch port of the model
trained by experiments/jax_lm.py on the jax-lm branch (Qwen3-1.7B-Base healed onto an unrestricted
Combinatorial BPE vocabulary). Inference only.

* Input: the embedding of a token is the sum of its four table rows.
* Body: transformers' Qwen3Model, unchanged, fed with those embeddings.
* Output ("chain head"): a token is predicted field by field in `order` (core, prefix, variation,
  suffix for the published model), each field conditioned on all the ones before it:
      logits[order[0]] = h @ table[order[0]].T                       (h: final hidden state, after the norm)
      g = h, prev = table[order[0]][chosen id]
      for k, i in enumerate(order[1:]):
          g = g + W3 @ gelu(W1 @ LayerNorm([g; prev]))               (steps.k)
          logits[i] = LayerNorm_k(g) @ table[i].T                    (ln_steps.k)
          prev = table[i][chosen id]
  The input and output tables are the same (tied). A token's log-probability is the sum of its fields'.
* Numerics: the body runs in `dtype` (bf16 by default); the tables, the chain head and the logits are
  fp32 (the weights are stored in bf16), like the fp32 residual / logits of the JAX trainer.

    from cbpe.qwen3 import load_model, generate
    model, tok = load_model("SwayStar123/...")        # a directory or a Hugging Face repo, see load_model
    print(generate(model, tok, "The capital of France is")[0])

or `python -m cbpe.qwen3 --model DIR --prompt "..."` (also `--bpb file.txt` for bits per byte). Encoding
needs the Rust binary: `cargo build --release` in native/dict_search (or set DS_BIN).
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = "config.json"
WEIGHTS = "model.safetensors"
# keys of the full config (ours, or ones that only make sense with a vocabulary) the Qwen3 body does not take
SKIP_KEYS = ("cbpe", "torch_dtype", "dtype", "bos_token_id", "eos_token_id", "pad_token_id", "model_type", "auto_map",
             "architectures", "_name_or_path", "transformers_version")
GEN_CHUNK = 256  # positions per cross-entropy chunk (bounds the [chunk, vocab] fp32 logits)


class FactoredQwen3(nn.Module):
    """config: the Hugging Face Qwen3 config.json dict; sizes: rows of the four tables (variation, prefix,
    core, suffix); order: the chain head's field order."""

    def __init__(self, config: dict, sizes=(8, 1469, 128561, 1038), order=(2, 1, 0, 3), dtype=torch.bfloat16):
        super().__init__()
        from transformers import Qwen3Config, Qwen3Model
        self.sizes, self.order = tuple(sizes), tuple(order)
        assert len(self.sizes) == 4 and sorted(self.order) == [0, 1, 2, 3]
        self.body_dtype = dtype
        body = Qwen3Config.from_dict({**{k: v for k, v in config.items() if k not in SKIP_KEYS},
                                       "vocab_size": 1})
        self.model = Qwen3Model(body).to(dtype)  # (its 1-row embed_tokens is never used: inputs_embeds)
        d = body.hidden_size
        self.d = d
        self.emb = nn.ModuleList(nn.Embedding(n, d) for n in self.sizes)
        self.steps = nn.ModuleList(nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d, bias=False), nn.GELU(),
                                                 nn.Linear(d, d, bias=False)) for _ in range(3))
        self.ln_steps = nn.ModuleList(nn.LayerNorm(d) for _ in range(3))
        for m in (self.emb, self.steps, self.ln_steps):
            m.float()

    # ------------------------------------------------------------------ input
    def embed(self, ids):
        """ids [B, T, 4] -> [B, T, d] fp32: the sum of the four rows"""
        return sum(self.emb[i](ids[..., i]) for i in range(4))

    def hidden(self, ids, past_key_values=None):
        """ids [B, T, 4] -> final hidden states [B, T, d] (fp32, after the final norm); with a
        transformers Cache, the keys / values of these positions are appended to it"""
        out = self.model(inputs_embeds=self.embed(ids).to(self.body_dtype), past_key_values=past_key_values,
                         use_cache=past_key_values is not None)
        return out.last_hidden_state.float()

    # ------------------------------------------------------------ chain head
    def step(self, k, g, prev):
        s = self.steps[k]
        return s[3](s[2](s[1](s[0](torch.cat([g, prev], -1)))))

    def logits(self, i, g):
        return F.linear(g, self.emb[i].weight)

    def nll_fields(self, h, y):
        """negative log-likelihood in nats of the target tokens y [B, T, 4], each field given the true
        ones before it: [B, T, 4] (a token's NLL is the sum over the last dim)"""
        out = torch.empty(y.shape, dtype=torch.float32, device=h.device)
        for s in range(0, h.shape[1], GEN_CHUNK):
            hs, ys = h[:, s:s + GEN_CHUNK], y[:, s:s + GEN_CHUNK]

            def ce(i, g):
                lg = self.logits(i, g)
                out[:, s:s + GEN_CHUNK, i] = torch.logsumexp(lg, -1) - lg.gather(-1, ys[..., i:i + 1])[..., 0]

            first = self.order[0]
            ce(first, hs)
            g, prev = hs, self.emb[first](ys[..., first])
            for k, i in enumerate(self.order[1:]):
                g = g + self.step(k, g, prev)
                ce(i, self.ln_steps[k](g))
                prev = self.emb[i](ys[..., i])
        return out

    @staticmethod
    def pick(logits, temperature, top_p, gen):
        """a row's id from logits [B, V]: argmax (temperature 0) or a draw from the top-p nucleus of
        softmax(logits / temperature); also returns log softmax(logits) of the id"""
        lp = torch.log_softmax(logits, -1)
        if temperature <= 0:
            i = logits.argmax(-1)
        else:
            p, order = torch.softmax(logits / temperature, -1).sort(-1, descending=True)
            p = torch.where(p.cumsum(-1) - p < top_p, p, torch.zeros_like(p))  # the smallest set with mass >= top_p
            i = order.gather(-1, torch.multinomial(p / p.sum(-1, keepdim=True), 1, generator=gen))[..., 0]
        return i, lp.gather(-1, i[:, None])[:, 0]

    @torch.no_grad()
    def sample(self, h, temperature=0.0, top_p=1.0, gen=None):
        """next tokens from hidden states h [B, d]: (ids [B, 4], log-probabilities [B]); the fields are
        drawn one by one in `order`, each conditioned on the ones already drawn"""
        got, first = [None] * 4, self.order[0]
        got[first], lp = self.pick(self.logits(first, h), temperature, top_p, gen)
        g, prev = h, self.emb[first](got[first])
        for k, i in enumerate(self.order[1:]):
            g = g + self.step(k, g, prev)
            got[i], li = self.pick(self.logits(i, self.ln_steps[k](g)), temperature, top_p, gen)
            lp = lp + li
            prev = self.emb[i](got[i])
        return torch.stack(got, -1), lp

    @torch.no_grad()
    def candidates(self, h, n=64, gen=None):
        """the greedy token and n draws from the model's distribution at h [1, d] (for token healing):
        ids [n + 1, 4], log-probabilities [n + 1]"""
        gi, gl = self.sample(h, 0.0, 1.0, gen)
        di, dl = self.sample(h.expand(n, -1), 1.0, 1.0, gen)
        return torch.cat([gi, di]), torch.cat([gl, dl])

    # ----------------------------------------------------------------- loading
    def load_weights(self, path):
        from safetensors.torch import load_file
        sd = load_file(path)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        missing = [k for k in missing if k != "model.embed_tokens.weight"]
        if missing or unexpected:
            raise ValueError(f"checkpoint mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
        return self


def load_model(path, dtype=torch.bfloat16, device=None, tokenizer=True):
    """(model, tokenizer) from a directory (or a Hugging Face repo id, downloaded) holding config.json,
    model.safetensors and the tokenizer files named by config["cbpe"]["tokenizer"]."""
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download
        path = snapshot_download(path)
    with open(os.path.join(path, CONFIG), encoding="utf-8") as f:
        config = json.load(f)
    c = config["cbpe"]
    model = FactoredQwen3(config, c["sizes"], c["order"], dtype).load_weights(os.path.join(path, WEIGHTS))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    if not tokenizer:
        return model, None
    from . import load
    tok = load(os.path.join(path, c["tokenizer"]))
    assert tuple(tok.sizes.values()) == model.sizes, (tok.sizes, model.sizes)
    return model, tok


# ------------------------------------------------------------------- scoring
@torch.no_grad()
def score(model, ids, ctx=2048, bs=4):
    """NLL in nats of every token of ids [N, 4] but the first given the ones before it, in
    non-overlapping windows of `ctx` tokens (as the trainer's evaluation): [n * ctx, 4], n = (N - 1) // ctx"""
    dev = next(model.parameters()).device
    ids = torch.as_tensor(np.asarray(ids), dtype=torch.long)
    n = (len(ids) - 1) // ctx
    out = []
    for i in range(0, n, bs):
        w = torch.stack([ids[j * ctx:j * ctx + ctx + 1] for j in range(i, min(i + bs, n))]).to(dev)
        out.append(model.nll_fields(model.hidden(w[:, :-1]), w[:, 1:]).reshape(-1, 4).cpu())
    return torch.cat(out) if out else torch.zeros(0, 4)


def bits_per_byte(model, tok, text, ctx=2048, bs=4):
    """bits per UTF-8 byte of `text` over its non-overlapping windows of ctx tokens (the trainer's
    metric: the bytes are the text's, scaled by the share of tokens the windows cover)"""
    ids = tok.encode(text)
    nll = score(model, ids, ctx, bs).sum().item()
    covered = len(text.encode("utf-8")) * (((len(ids) - 1) // ctx) * ctx) / (len(ids) - 1)
    return nll / math.log(2) / covered


# ---------------------------------------------------------------- generation
def heal_choice(cands, logps, need, tok, temperature, rng):
    """Token healing: among the candidate tokens, those whose text agrees with the bytes `need` that
    the prompt's removed last token spelled (the token starts with them, or is a prefix of them); the
    most likely one (temperature 0) or one drawn by probability. Returns (token, remaining need) or
    None if no candidate fits."""
    uniq = {}
    for t, lp in zip(cands, logps):
        t = tuple(int(x) for x in t)
        b = tok.token_bytes(t)
        if b and (b.startswith(need) or need.startswith(b)):
            uniq[t] = (float(lp), b)
    if not uniq:
        return None
    items = sorted(uniq.items(), key=lambda kv: -kv[1][0])
    if temperature > 0:
        lps = np.array([v[0] for _, v in items]) / temperature
        p = np.exp(lps - lps.max())
        t, (lp, b) = items[rng.choice(len(items), p=p / p.sum())]
    else:
        t, (lp, b) = items[0]
    return t, (need[len(b):] if need.startswith(b) else b"")


@torch.no_grad()
def generate_tokens(model, ids, tok=None, max_new=80, temperature=0.0, top_p=0.9, heal=False, seed=0, n_cand=64,
                    stop=()):
    """Continue the token list `ids` ((v, p, c, s) tuples): (the prompt tokens kept, the new tokens, the
    number of leading bytes of the new tokens' text that respell the prompt's removed last token).
    Greedy for temperature 0. With `heal` (needs `tok`, anything with token_bytes(t)) the prompt's last
    token is dropped and the first new token(s) must respell its bytes (among the greedy token and
    n_cand draws of each step), so the continuation does not start at an unnatural token boundary (a
    prompt ending in "http:" or a space). `stop`: strings that end the generation (needs `tok`).
    max_new counts the healed tokens. Healing picks among sampled candidates, so when the greedy token
    does not fit the removed bytes the healed token depends on `seed` (and on n_cand)."""
    from transformers import DynamicCache
    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev).manual_seed(seed)
    rng = np.random.default_rng(seed)
    ids = [tuple(int(x) for x in t) for t in ids]
    if not ids:
        raise ValueError("empty prompt")
    if (heal or stop) and tok is None:
        raise ValueError("token healing and stop strings need the tokenizer")
    removed, need = None, b""
    if heal and len(ids) > 1:
        removed = ids.pop()
        need = tok.token_bytes(removed)
    skip = len(need)
    cache = DynamicCache()
    h = model.hidden(torch.tensor([ids], device=dev), cache)[:, -1]
    new, data = [], b""
    while len(new) < max_new:
        if need:
            c, lp = model.candidates(h, n_cand, gen)
            r = heal_choice(c.cpu().numpy(), lp.cpu().numpy(), need, tok, temperature, rng)
            t, need = (removed, b"") if r is None else r
        else:
            t = tuple(model.sample(h, temperature, top_p, gen)[0][0].tolist())
        new.append(t)
        if stop:
            data += tok.token_bytes(t)
            if any(s in data[skip:].decode("utf-8", errors="ignore") for s in stop):
                break
        h = model.hidden(torch.tensor([[t]], device=dev), cache)[:, -1]
    return ids, new, skip


def generate(model, tok, prompt, max_new=80, temperature=0.0, top_p=0.9, heal=True, seed=0, n_cand=64, stop=()):
    """Continue `prompt`: (text after the prompt, the new tokens as (v, p, c, s) tuples); see
    generate_tokens. `stop`: the text is cut at the first of these strings."""
    ids = [tuple(t) for t in tok.encode(prompt)]
    kept, new, skip = generate_tokens(model, ids, tok, max_new, temperature, top_p, heal, seed, n_cand, stop)
    text = b"".join(tok.token_bytes(t) for t in new)[skip:].decode("utf-8", errors="replace")
    for s in stop:
        if s in text:
            text = text[:text.index(s)]
    return text, new


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sample from / score a Qwen3 model on a factored tokenizer")
    ap.add_argument("--model", required=True, help="directory or Hugging Face repo id")
    ap.add_argument("--prompt", action="append", default=[])
    ap.add_argument("--bpb", nargs="*", default=[], help="text files to score (bits per byte)")
    ap.add_argument("--max_new", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heal", action=argparse.BooleanOptionalAction, default=True, help="token healing")
    ap.add_argument("--ctx", type=int, default=2048, help="window of --bpb")
    ap.add_argument("--max_chars", type=int, default=0, help="--bpb: only the first N characters of each file")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--stop", action="append", default=[])
    a = ap.parse_args(argv)
    model, tok = load_model(a.model, getattr(torch, a.dtype), a.device)
    for p in a.prompt:
        text, new = generate(model, tok, p, a.max_new, a.temperature, a.top_p, a.heal, a.seed, stop=a.stop)
        print(f"=== {p!r} ({len(new)} new tokens)\n{p}{text}\n", flush=True)
    for path in a.bpb:
        with open(path, encoding="utf-8") as f:
            text = f.read(a.max_chars or -1)
        print(f"{os.path.basename(path)}: {bits_per_byte(model, tok, text, a.ctx):.4f} bits/byte", flush=True)


if __name__ == "__main__":
    main()
