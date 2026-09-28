"""JAX port of experiments/lm.py for large-scale training (Cloud TPU v4 / v5e / v5p / v6e, also CPU/GPU).

Same model, optimiser, schedule, data order, evaluation (bits-per-byte) and JSON log format as lm.py,
so results are directly comparable (see lm.py's docstring for the model). Additions for scale:

  * jit-compiled train step; layers are stacked and run with lax.scan (compile time independent of
    depth), optional rematerialisation (--remat);
  * a device mesh (data, fsdp, tensor): the batch is split over data x fsdp, parameters and optimiser
    state are sharded over fsdp (ZeRO-3 style) and optionally over tensor (vocabulary / heads / MLP
    hidden); XLA's SPMD partitioner inserts the collectives;
  * multi-host: jax.distributed.initialize() on pods, every host builds only the rows of each
    global batch that live on its own devices (deterministic: all hosts compute the same order);
  * bf16 compute with fp32 master weights and fp32 optimiser state;
  * the output layers compute the cross-entropy in chunks of positions under jax.checkpoint, so
    [B, T, V] logits are never materialised (only [B, chunk, V]);
  * orbax checkpoints (local or gs://) with automatic resume, throughput / MFU logging.

Parameters are a flat dict keyed by lm.py's state_dict names and stored in PyTorch layout
((out, in) for linear weights); the per-layer tensors "blocks.<i>.<name>" are stacked into one
"blocks.<name>" array with a leading layer axis, so converting a PyTorch checkpoint is a stack
(see from_torch_state_dict / to_torch_state_dict).
"""
import argparse
import dataclasses
import json
import math
import os
import queue
import sys
import threading
import time
from functools import partial

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lm_common import CACHE, ROOT, encode_file, rel  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

F32 = jnp.float32
EPS32 = float(np.finfo(np.float32).eps)  # nn.RMSNorm(eps=None) uses the fp32 machine epsilon
BATCH = ("data", "fsdp")  # mesh axes the batch dimension is split over


# ------------------------------------------------------------------- config
@dataclasses.dataclass(frozen=True)
class Config:
    sizes: tuple            # table sizes; 1 = standard LM, 4 = combinatorial (var, pre, core, suf)
    d: int = 512
    n_layer: int = 8
    n_head: int = 8
    ctx: int = 512
    head: str = "linear"    # linear | mlp | chain (factored vocab only)
    head_hidden: float = 1.0
    order: tuple = (2, 1, 0, 3)
    arch: str = "gpt2"      # gpt2 | modern
    compute_dtype: str = "bfloat16"
    remat: str = "none"     # none | full | dots
    ce_chunk: int = 128     # positions per cross-entropy chunk (0 = no chunking)
    mesh: object = None     # jax.sharding.Mesh or None (single device / let XLA decide)

    @property
    def cdt(self):
        return jnp.dtype(self.compute_dtype)

    @property
    def factored(self):
        return len(self.sizes) == 4

    @property
    def mlp_hidden(self):
        return 4 * self.d if self.arch == "gpt2" else 32 * round(8 * self.d / 3 / 32)


def param_shapes(cfg):
    """{name: shape} in lm.py's naming / PyTorch layout, block tensors stacked over layers."""
    d, L = cfg.d, cfg.n_layer
    s = {f"emb.{i}.weight": (n, d) for i, n in enumerate(cfg.sizes)}
    if cfg.arch == "modern":
        hd, hid = d // cfg.n_head, cfg.mlp_hidden
        blk = {"ln1.weight": (d,), "ln2.weight": (d,), "q_norm.weight": (hd,), "k_norm.weight": (hd,),
               "qkv.weight": (3 * d, d), "o.weight": (d, d), "gate_up.weight": (2 * hid, d), "down.weight": (d, hid)}
        s["ln_f.weight"] = (d,)
    else:
        s["pos.weight"] = (cfg.ctx, d)
        blk = {"ln1.weight": (d,), "ln1.bias": (d,), "ln2.weight": (d,), "ln2.bias": (d,), "qkv.weight": (3 * d, d),
               "o.weight": (d, d), "mlp.0.weight": (4 * d, d), "mlp.2.weight": (d, 4 * d)}
        s["ln_f.weight"], s["ln_f.bias"] = (d,), (d,)
    s.update({f"blocks.{k}": (L,) + v for k, v in blk.items()})
    if cfg.factored:
        if cfg.head == "linear":
            s.update({"cond.weight": (d, d), "ln_c.weight": (d,), "ln_c.bias": (d,)})
        else:
            hid = int(cfg.head_hidden * d)
            for k in range(1 if cfg.head == "mlp" else 3):
                s.update({f"steps.{k}.0.weight": (2 * d,), f"steps.{k}.0.bias": (2 * d,),
                          f"steps.{k}.1.weight": (hid, 2 * d), f"steps.{k}.3.weight": (d, hid),
                          f"ln_steps.{k}.weight": (d,), f"ln_steps.{k}.bias": (d,)})
    return s


def layer_shape(name, shape):
    return shape[1:] if name.startswith("blocks.") else shape


def is_matrix(name, shape):
    """lm.py's grouping: 2-D weights except the embedding tables / positions (-> Muon or weight decay)."""
    return len(layer_shape(name, shape)) == 2 and "emb" not in name and "pos" not in name


def init_params(key, cfg):
    """Same distributions as lm.py: N(0, 0.02) for linear / embedding weights, N(0, 0.02/sqrt(2L)) for
    the residual output projections, ones / zeros for norm weights / biases."""
    params = {}
    for j, (name, shape) in enumerate(sorted(param_shapes(cfg).items())):
        if len(layer_shape(name, shape)) == 1:
            params[name] = (jnp.zeros if name.endswith("bias") else jnp.ones)(shape, F32)
            continue
        std = 0.02
        if name.endswith("o.weight") or name.endswith("mlp.2.weight") or name.endswith("down.weight"):
            std = 0.02 / math.sqrt(2 * cfg.n_layer)
        params[name] = std * jax.random.normal(jax.random.fold_in(key, j), shape, F32)
    return params


def from_torch_state_dict(sd, cfg):
    """lm.py GPT.state_dict() (tensors or arrays) -> JAX params (fp32)."""
    to_np = lambda t: t.detach().float().cpu().numpy() if hasattr(t, "detach") else np.asarray(t, np.float32)
    out = {}
    for name in param_shapes(cfg):
        if name.startswith("blocks."):
            sub = name[len("blocks."):]
            out[name] = jnp.asarray(np.stack([to_np(sd[f"blocks.{l}.{sub}"]) for l in range(cfg.n_layer)]))
        else:
            out[name] = jnp.asarray(to_np(sd[name]))
    return out


def to_torch_state_dict(params, cfg):
    """JAX params -> {name: np.ndarray} loadable into lm.py's GPT (load_state_dict(strict=False) ignores rope buffers)."""
    out = {}
    for name, v in params.items():
        v = np.asarray(jax.device_get(v))
        if name.startswith("blocks."):
            for l in range(cfg.n_layer):
                out[f"blocks.{l}.{name[len('blocks.'):]}"] = v[l]
        else:
            out[name] = v
    return out


# --------------------------------------------------------------- sharding
def make_mesh(fsdp=1, tensor=1):
    devs = np.array(jax.devices())
    n = len(devs)
    assert n % (fsdp * tensor) == 0, f"{n} devices not divisible by fsdp={fsdp} x tensor={tensor}"
    shape = (n // (fsdp * tensor), fsdp, tensor)
    try:
        from jax.experimental import mesh_utils
        devs = mesh_utils.create_device_mesh(shape, allow_split_physical_axes=True)
    except Exception:  # noqa: BLE001  (e.g. CPU / odd topologies)
        devs = devs.reshape(shape)
    return Mesh(devs, ("data", "fsdp", "tensor"))


COLUMN = ("qkv.weight", "mlp.0.weight", "gate_up.weight", "cond.weight", ".1.weight")  # (out, in): out over tensor
ROW = ("o.weight", "mlp.2.weight", "down.weight", ".3.weight")                         # (out, in): in over tensor


def param_spec(name, shape, mesh):
    """PartitionSpec for one parameter: 1-D tensors replicated; matrices: one dim over fsdp, the other
    over tensor (Megatron-style column / row split; vocabulary split for embedding tables)."""
    if mesh is None:
        return P()
    sz = dict(zip(mesh.axis_names, mesh.devices.shape))
    lead = (None,) if name.startswith("blocks.") else ()
    ls = layer_shape(name, shape)
    if len(ls) != 2:
        return P(*(lead + (None,) * len(ls)))
    out_ax, in_ax = "fsdp", "tensor"  # row-parallel default: (out, in) = (fsdp, tensor)
    if name.startswith("emb.") or name.endswith(COLUMN):
        out_ax, in_ax = "tensor", "fsdp"
    elif name == "pos.weight":
        out_ax, in_ax = None, "fsdp"
    ok = lambda ax, n: ax if ax is not None and sz[ax] > 1 and n % sz[ax] == 0 else None
    return P(*(lead + (ok(out_ax, ls[0]), ok(in_ax, ls[1]))))


def param_shardings(cfg):
    if cfg.mesh is None:
        return None
    return {n: NamedSharding(cfg.mesh, param_spec(n, s, cfg.mesh)) for n, s in param_shapes(cfg).items()}


def constrain_batch(x, cfg):
    if cfg.mesh is None:
        return x
    return lax.with_sharding_constraint(x, NamedSharding(cfg.mesh, P(BATCH, *([None] * (x.ndim - 1)))))


# ------------------------------------------------------------------- model
def _ln(x, w, b, eps=1e-5):
    """LayerNorm in fp32 (autocast runs layer_norm in fp32 and returns fp32)."""
    x = x.astype(F32)
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    return xc * lax.rsqrt((xc * xc).mean(-1, keepdims=True) + eps) * w + b


def _rms(x, w):
    """RMSNorm computed in fp32, returned in the input dtype (as torch.rms_norm does)."""
    x32 = x.astype(F32)
    return (x32 * lax.rsqrt((x32 * x32).mean(-1, keepdims=True) + EPS32) * w).astype(x.dtype)


def _dense(x, w, cfg):
    """x @ w.T in the compute dtype (w in PyTorch (out, in) layout, fp32 master copy)."""
    return jnp.einsum("...i,oi->...o", x.astype(cfg.cdt), w.astype(cfg.cdt))


def _gelu(x):
    return jax.nn.gelu(x, approximate=False)


def rope_tables(T, head_dim, base=10000.0):
    """Same as lm.py (fp32 maths, stored in bf16): cos/sin shaped (T, 1, head_dim/2)."""
    inv = 1.0 / base ** (np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim))
    ang = np.outer(np.arange(T, dtype=np.float32), inv.astype(np.float32))[:, None, :]
    return jnp.asarray(np.cos(ang), jnp.bfloat16), jnp.asarray(np.sin(ang), jnp.bfloat16)


def _rope(x, cos, sin):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def _attention(q, k, v):
    """Causal softmax attention on (B, T, H, hd); scores and softmax in fp32 like SDPA's kernels."""
    T, hd = q.shape[1], q.shape[-1]
    s = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=F32) * (1.0 / math.sqrt(hd))
    s = jnp.where(jnp.tril(jnp.ones((T, T), bool)), s, -jnp.inf)
    p = jax.nn.softmax(s, axis=-1).astype(v.dtype)
    return jnp.einsum("bhqk,bkhd->bqhd", p, v)


def _block(x, p, cfg, rope):
    B, T, D = x.shape
    H = cfg.n_head
    if cfg.arch == "modern":
        qkv = _dense(_rms(x, p["ln1.weight"]), p["qkv.weight"], cfg).reshape(B, T, 3, H, D // H)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = _rope(_rms(q, p["q_norm.weight"]), *rope).astype(v.dtype)
        k = _rope(_rms(k, p["k_norm.weight"]), *rope).astype(v.dtype)
        x = x + _dense(_attention(q, k, v).reshape(B, T, D), p["o.weight"], cfg)
        g, u = jnp.split(_dense(_rms(x, p["ln2.weight"]), p["gate_up.weight"], cfg), 2, axis=-1)
        return x + _dense(jax.nn.silu(g) * u, p["down.weight"], cfg)
    qkv = _dense(_ln(x, p["ln1.weight"], p["ln1.bias"]), p["qkv.weight"], cfg).reshape(B, T, 3, H, D // H)
    a = _attention(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2])
    x = x + _dense(a.reshape(B, T, D), p["o.weight"], cfg)
    hmid = _gelu(_dense(_ln(x, p["ln2.weight"], p["ln2.bias"]), p["mlp.0.weight"], cfg))
    return x + _dense(hmid, p["mlp.2.weight"], cfg)


def _ce_sum(g, table, y, wts, cfg):
    """sum_t wts * -log softmax(g @ table.T)[y], never materialising more than [B, ce_chunk, V] logits.
    Logits come out of the matmul in fp32 (lm.py rounds them to bf16 first; this is slightly more exact)."""
    B, T, D = g.shape
    tab = table.astype(cfg.cdt)

    def chunk_nll(gi, yi, wi):
        logits = jnp.einsum("btd,vd->btv", gi.astype(cfg.cdt), tab, preferred_element_type=F32)
        tgt = jnp.take_along_axis(logits, yi[..., None], axis=-1)[..., 0]
        return ((jax.nn.logsumexp(logits, axis=-1) - tgt) * wi).sum()

    c = cfg.ce_chunk
    if not c or c >= T or T % c:
        return chunk_nll(g, y, wts)
    n = T // c
    split = lambda a: jnp.moveaxis(a.reshape((B, n, c) + a.shape[2:]), 1, 0)

    def body(acc, inp):
        return acc + jax.checkpoint(chunk_nll)(*inp), None

    tot, _ = lax.scan(body, jnp.zeros((), F32), (split(g), split(y), split(wts)))
    return tot


def forward(params, w, cfg, wts=None):
    """w: [B, T+1, F] int windows (inputs w[:, :-1], targets w[:, 1:]); wts: optional [B] window weights
    (eval padding). Returns (summed NLL in nats, list of per-factor sums) like GPT.forward in lm.py."""
    x, y = w[:, :-1], w[:, 1:]
    B, T, _ = x.shape
    wts = jnp.ones((B, T), F32) if wts is None else jnp.broadcast_to(wts.astype(F32)[:, None], (B, T))
    emb = lambda i, idx: jnp.take(params[f"emb.{i}.weight"], idx, axis=0)
    h = 0
    for i in range(len(cfg.sizes)):
        h = h + emb(i, x[..., i])
    rope = None
    if cfg.arch == "modern":
        cos, sin = rope_tables(cfg.ctx, cfg.d // cfg.n_head)
        rope = (cos[:T], sin[:T])
    else:
        h = h + params["pos.weight"][:T]
    h = constrain_batch(h, cfg)

    def body(h, p):
        return constrain_batch(_block(h, p, cfg, rope), cfg), None

    if cfg.remat == "full":
        body = jax.checkpoint(body)
    elif cfg.remat == "dots":
        body = jax.checkpoint(body, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
    blocks = {k[len("blocks."):]: v for k, v in params.items() if k.startswith("blocks.")}
    h, _ = lax.scan(body, h, blocks)
    if cfg.arch == "modern":
        h = _rms(h, params["ln_f.weight"])
    else:
        h = _ln(h, params["ln_f.weight"], params["ln_f.bias"])

    ce = lambda g, i: _ce_sum(g, params[f"emb.{i}.weight"], y[..., i], wts, cfg)
    if not cfg.factored:
        nll = ce(h, 0)
        return nll, [nll]

    def step(k, a, b):
        z = _ln(jnp.concatenate([a, b], -1), params[f"steps.{k}.0.weight"], params[f"steps.{k}.0.bias"])
        return _dense(_gelu(_dense(z, params[f"steps.{k}.1.weight"], cfg)), params[f"steps.{k}.3.weight"], cfg)

    ln_step = lambda k, g: _ln(g, params[f"ln_steps.{k}.weight"], params[f"ln_steps.{k}.bias"])
    parts = [None] * 4
    if cfg.head != "chain":
        parts[2] = ce(h, 2)
    if cfg.head == "linear":
        g = _ln(h + _dense(emb(2, y[..., 2]), params["cond.weight"], cfg), params["ln_c.weight"], params["ln_c.bias"])
        for i in (0, 1, 3):
            parts[i] = ce(g, i)
    elif cfg.head == "mlp":
        g = ln_step(0, h + step(0, h, emb(2, y[..., 2])))
        for i in (0, 1, 3):
            parts[i] = ce(g, i)
    else:  # chain: each factor conditioned on all previously predicted ones, in cfg.order
        first = cfg.order[0]
        parts[first] = ce(h, first)
        g, prev = h, emb(first, y[..., first])
        for k, i in enumerate(cfg.order[1:]):
            g = g + step(k, g, prev)
            parts[i] = ce(ln_step(k, g), i)
            prev = emb(i, y[..., i])
    return sum(parts), parts


# ---------------------------------------------------------------- optimizer
MUON_NS_DTYPE = jnp.bfloat16  # lm.py orthogonalises in bf16 (tests switch both sides to fp32)


def newton_schulz(G, steps=5, eps=1e-7, coeffs=(3.4445, -4.775, 2.0315), dtype=None):
    """Batched quintic Newton-Schulz orthogonalisation over the last two axes, in bf16 (as lm.py)."""
    a, b, c = coeffs
    dt = dtype or MUON_NS_DTYPE
    X = G.astype(dt)
    tall = X.shape[-2] > X.shape[-1]
    if tall:
        X = jnp.swapaxes(X, -1, -2)
    nrm = jnp.sqrt(jnp.sum(jnp.square(X.astype(F32)), axis=(-2, -1), keepdims=True)).astype(dt)
    X = X / (nrm + jnp.asarray(eps, dt))
    for _ in range(steps):
        A = X @ jnp.swapaxes(X, -1, -2)
        X = a * X + (b * A + c * A @ A) @ X
    return jnp.swapaxes(X, -1, -2) if tall else X


def opt_groups(params, opt):
    """name -> (rule, weight decay), mirroring lm.py: Muon (wd 0.1) for hidden matrices + AdamW (wd 0) for
    the rest, or AdamW with wd 0.1 on matrices and 0 elsewhere."""
    g = {}
    for n, v in params.items():
        m = is_matrix(n, v.shape)
        g[n] = ("muon", 0.1) if (m and opt == "muon") else ("adamw", 0.1 if m else 0.0)
    return g


def opt_init(params, groups):
    z = lambda v: jnp.zeros(v.shape, F32)
    adam = {n: v for n, v in params.items() if groups[n][0] == "adamw"}
    return {"t": jnp.zeros((), jnp.int32),
            "m": {n: z(v) for n, v in adam.items()}, "v": {n: z(v) for n, v in adam.items()},
            "buf": {n: z(v) for n, v in params.items() if groups[n][0] == "muon"}}


def clip_by_global_norm(grads, max_norm=1.0):
    """torch.nn.utils.clip_grad_norm_: g *= min(1, max_norm / (||g|| + 1e-6))."""
    norm = jnp.sqrt(sum(jnp.sum(jnp.square(g.astype(F32))) for g in jax.tree.leaves(grads)))
    coef = jnp.minimum(max_norm / (norm + 1e-6), 1.0)
    return jax.tree.map(lambda g: g * coef, grads), norm


def opt_update(params, grads, state, lr, groups, betas=(0.9, 0.95), eps=1e-8, momentum=0.95):
    """One step of torch.optim.AdamW (decoupled decay, bias correction) / lm.py BatchedMuon (Nesterov)."""
    t = state["t"] + 1
    b1, b2 = betas
    bc1 = 1 - b1 ** t.astype(F32)
    bc2 = 1 - b2 ** t.astype(F32)
    new_p, m, v, buf = {}, dict(state["m"]), dict(state["v"]), dict(state["buf"])
    for n, p in params.items():
        g = grads[n].astype(F32)
        rule, wd = groups[n]
        p = p * (1 - lr * wd)
        if rule == "muon":
            buf[n] = momentum * buf[n] + g
            O = newton_schulz(g + momentum * buf[n])
            scale = 0.2 * max(layer_shape(n, p.shape)) ** 0.5  # match the update RMS of AdamW
            new_p[n] = p - (lr * scale) * O.astype(F32)
        else:
            m[n] = b1 * m[n] + (1 - b1) * g
            v[n] = b2 * v[n] + (1 - b2) * g * g
            new_p[n] = p - (lr / bc1) * m[n] / (jnp.sqrt(v[n]) / jnp.sqrt(bc2) + eps)
    return new_p, {"t": t, "m": m, "v": v, "buf": buf}


def lr_at(step, args):
    """lm.py's schedule: linear warmup, cosine decay, floored at 10% of the (warmed-up) peak."""
    warm = min(1, (step + 1) / args.warmup)
    lr = args.lr * warm * 0.5 * (1 + math.cos(math.pi * step / args.steps))
    return max(lr, args.lr * 0.1 * warm)


def make_train_step(cfg, groups, tokens_per_step):
    """batch: [accum, micro_bs, ctx+1, F]. Loss per micro-batch = nll / (bs * ctx), gradients summed
    over micro-batches (lm.py's accumulation), clipped to global norm 1, then the optimiser step."""

    def loss(params, w):
        nll, _ = forward(params, w, cfg)
        return nll / tokens_per_step

    grad_fn = jax.value_and_grad(loss)

    def train_step(params, opt_state, batch, lr):
        if batch.shape[0] == 1:
            l, grads = grad_fn(params, batch[0])
        else:
            def micro(carry, w):
                l, g = grad_fn(params, w)
                return (carry[0] + l, jax.tree.map(jnp.add, carry[1], g)), None
            zero = (jnp.zeros((), F32), jax.tree.map(lambda p: jnp.zeros(p.shape, F32), params))
            (l, grads), _ = lax.scan(micro, zero, batch)
        grads, gnorm = clip_by_global_norm(grads, 1.0)
        params, opt_state = opt_update(params, grads, opt_state, lr, groups)
        return params, opt_state, {"loss": l, "grad_norm": gnorm}

    return train_step


# ---------------------------------------------------------------------- data
def window_order(n_windows, need, seed, how="auto"):
    """lm.py's order of window indices: torch.randperm epochs with a seeded CPU generator (identical to
    lm.py when torch is importable); numpy permutations otherwise."""
    reps = -(-need // n_windows)
    if how in ("auto", "torch"):
        try:
            import torch
            g = torch.Generator().manual_seed(seed)
            return torch.cat([torch.randperm(n_windows, generator=g) for _ in range(reps)])[:need].numpy(), "torch"
        except ImportError:
            if how == "torch":
                raise
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.permutation(n_windows) for _ in range(reps)])[:need], "numpy"


def as_2d(a):
    return a.reshape(len(a), -1)


def make_global_batch(data, starts, ctx, shape, sharding, weights=None):
    """Build a global jax.Array of windows data[s:s+ctx+1] for starts shaped `shape[:-2]`; each process
    reads only the rows of its own addressable shards (multi-host safe)."""
    starts = np.asarray(starts).reshape(shape[:-2])
    if sharding is None:
        win = np.stack([data[s:s + ctx + 1] for s in starts.ravel()]).reshape(shape)
        return jnp.asarray(win), None if weights is None else jnp.asarray(weights)
    arrs, warrs = [], []
    for dev, idx in sharding.addressable_devices_indices_map(shape).items():
        sub = starts[idx[:len(shape) - 2]]
        win = np.stack([data[s:s + ctx + 1] for s in sub.ravel()]).reshape(sub.shape + shape[-2:])
        arrs.append(jax.device_put(win, dev))
        if weights is not None:
            warrs.append(jax.device_put(np.asarray(weights)[idx[:1]], dev))
    out = jax.make_array_from_single_device_arrays(shape, sharding, arrs)
    if weights is None:
        return out, None
    wsh = NamedSharding(sharding.mesh, P(sharding.spec[0]))
    return out, jax.make_array_from_single_device_arrays((shape[0],), wsh, warrs)


class Prefetcher:
    """Builds the next global batches on a background thread."""

    def __init__(self, fn, steps, depth=2):
        self.q = queue.Queue(depth)
        self.t = threading.Thread(target=self._run, args=(fn, steps), daemon=True)
        self.t.start()

    def _run(self, fn, steps):
        try:
            for s in steps:
                self.q.put((s, fn(s)))
        except BaseException as e:  # noqa: BLE001  (re-raised in the training loop)
            self.q.put((None, e))

    def get(self):
        s, item = self.q.get()
        if s is None:
            raise item
        return s, item


# ---------------------------------------------------------------- training
def n_batch_shards(cfg):
    if cfg.mesh is None:
        return 1
    sz = dict(zip(cfg.mesh.axis_names, cfg.mesh.devices.shape))
    return sz["data"] * sz["fsdp"]


def make_eval_fn(cfg):
    @jax.jit
    def ev(params, w, wts):
        nll, parts = forward(params, w, cfg, wts)
        return nll, jnp.stack(parts)
    return ev


def evaluate(ev, params, val, ctx, n_bytes, cfg, bs=8):
    """lm.py's evaluate(): non-overlapping windows of the validation ids, bits per byte (total and per factor)."""
    n = (len(val) - 1) // ctx
    k = n_batch_shards(cfg)
    bs = -(-bs // k) * k
    F = as_2d(val[:1]).shape[1]
    sh = None if cfg.mesh is None else NamedSharding(cfg.mesh, P(BATCH, None, None))
    tot, parts, count = 0.0, None, 0
    for i in range(0, n, bs):
        m = min(bs, n - i)
        starts = np.zeros(bs, np.int64)
        starts[:m] = np.arange(i, i + m) * ctx
        wts = (np.arange(bs) < m).astype(np.float32)
        w, wv = make_global_batch(val, starts, ctx, (bs, ctx + 1, F), sh, wts)
        nll, p = ev(params, w, wv)
        tot += float(nll)
        p = [float(a) for a in np.asarray(jax.device_get(p))]
        parts = p if parts is None else [a + b for a, b in zip(parts, p)]
        count += m * ctx
    covered = n_bytes * count / (len(val) - 1)
    return tot / math.log(2) / covered, [q / math.log(2) / covered for q in parts]


PEAK_BF16 = {"v4": 275e12, "v5 lite": 197e12, "v5e": 197e12, "v5p": 459e12, "v5": 459e12,
             "v6 lite": 918e12, "v6e": 918e12}


def peak_flops(args):
    if args.peak_tflops:
        return args.peak_tflops * 1e12
    kind = jax.devices()[0].device_kind.lower()
    for k in sorted(PEAK_BF16, key=len, reverse=True):
        if k in kind:
            return PEAK_BF16[k]
    return None


def train_flops_per_token(params, cfg):
    """6 x (non-embedding matmul params + tied output heads) + attention (12 L d ctx), as in PaLM's MFU."""
    mat = sum(int(np.prod(v.shape)) for n, v in params.items() if is_matrix(n, v.shape))
    heads = cfg.d * sum(cfg.sizes)
    return 6 * (mat + heads) + 12 * cfg.n_layer * cfg.d * cfg.ctx


def train(tok_path, args, mesh):
    from cbpe import load
    main = jax.process_index() == 0
    say = (lambda *a: print(*a, flush=True)) if main else (lambda *a: None)
    tok = load(tok_path)
    sizes = list(tok.sizes.values()) if hasattr(tok, "sizes") else [tok.vocab_size]
    del tok
    data, train_bytes = encode_file(tok_path, args.train_text, cache=args.cache_dir, mmap_mode="r", write_meta=True)
    val, val_bytes = encode_file(tok_path, args.val_text, max_chars=args.val_chars, cache=args.cache_dir,
                                 write_meta=True)
    data, val = as_2d(data), as_2d(val)
    bytes_per_tok = train_bytes / len(data)
    say(f"[{args.tag or args.head}] {os.path.basename(tok_path)}: tables {sizes} (sum {sum(sizes)}), train tokens "
        f"{len(data) / 1e6:.1f}M, {bytes_per_tok:.2f} bytes/token")
    extra = {}
    for path in args.extra_val:
        v, vb = encode_file(tok_path, path, cache=args.cache_dir, write_meta=True)
        extra[os.path.basename(path)] = (as_2d(v), vb)
    if args.prepare_only:
        say("  --prepare_only: .npy caches and .meta.json byte counts are in place")
        return None

    cfg = Config(tuple(sizes), args.d, args.layers, args.heads, args.ctx, args.head, args.head_hidden,
                 tuple(int(c) for c in args.order.split(",")), args.arch, args.compute_dtype, args.remat,
                 args.ce_chunk, mesh)
    shardings = param_shardings(cfg)
    params = jax.jit(partial(init_params, cfg=cfg), out_shardings=shardings)(jax.random.PRNGKey(args.seed))
    groups = opt_groups(params, args.opt)
    opt_state = jax.jit(partial(opt_init, groups=groups),
                        out_shardings=None if shardings is None else
                        {"t": NamedSharding(mesh, P()),
                         "m": {n: shardings[n] for n in groups if groups[n][0] == "adamw"},
                         "v": {n: shardings[n] for n in groups if groups[n][0] == "adamw"},
                         "buf": {n: shardings[n] for n in groups if groups[n][0] == "muon"}})(params)
    n_params = sum(int(np.prod(v.shape)) for v in params.values())
    n_emb = sum(int(np.prod(params[f"emb.{i}.weight"].shape)) for i in range(len(sizes)))

    assert args.bs % args.accum == 0, "--bs must be divisible by --accum"
    mb = args.bs // args.accum
    assert mb % n_batch_shards(cfg) == 0, f"micro-batch {mb} not divisible by data x fsdp = {n_batch_shards(cfg)}"
    tokens_per_step = args.bs * args.ctx
    assert args.allow_repeat or args.steps * tokens_per_step <= len(data), "would repeat data (see --allow_repeat)"
    n_windows = (len(data) - 1) // args.ctx
    need = args.steps * args.bs
    order, order_how = window_order(n_windows, need, args.seed, args.data_order)
    order = order.astype(np.int64) * args.ctx
    if need > n_windows:
        say(f"  --allow_repeat: {need / n_windows:.2f} epochs over the training data")
    F = data.shape[1]
    bshape = (args.accum, mb, args.ctx + 1, F)
    bsh = None if mesh is None else NamedSharding(mesh, P(None, BATCH, None, None))

    step_fn = jax.jit(make_train_step(cfg, groups, tokens_per_step), donate_argnums=(0, 1))
    ev = make_eval_fn(cfg)
    fpt = train_flops_per_token(params, cfg)
    peak = peak_flops(args)
    n_dev = jax.device_count()
    say(f"  params {n_params / 1e6:.2f}M (embeddings {n_emb / 1e6:.2f}M), devices {n_dev} "
        f"({jax.devices()[0].device_kind}), mesh {None if mesh is None else dict(zip(mesh.axis_names, mesh.devices.shape))}, "
        f"{fpt / 1e6:.1f} MFLOP/token, data order {order_how}")

    log = {"tokenizer": os.path.basename(tok_path), "tag": args.tag or args.head, "sizes": sizes, "params": n_params,
           "emb_params": n_emb, "bytes_per_token": bytes_per_tok, "curve": [], "backend": "jax"}
    start, prev_s = 0, 0.0  # prev_s: wall time of earlier (pre-resume) segments
    mngr = None
    if args.ckpt_dir:
        import orbax.checkpoint as ocp
        d = args.ckpt_dir if "://" in args.ckpt_dir else os.path.abspath(args.ckpt_dir)
        d = d.rstrip("/") + "/" + f"{log['tokenizer'][:-5]}__{log['tag']}__seed{args.seed}"
        mngr = ocp.CheckpointManager(d, options=ocp.CheckpointManagerOptions(max_to_keep=args.keep_ckpts, create=True))
        last = mngr.latest_step()
        if last is not None:
            abstract = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=a.sharding),
                                    {"params": params, "opt": opt_state})
            r = mngr.restore(last, args=ocp.args.Composite(state=ocp.args.StandardRestore(abstract),
                                                           meta=ocp.args.JsonRestore()))
            params, opt_state = r["state"]["params"], r["state"]["opt"]
            log, start, prev_s = r["meta"]["log"], last, r["meta"]["train_s"]
            say(f"  resumed from {d} at step {start}")

    def get_batch(step):
        return make_global_batch(data, order[step * args.bs:(step + 1) * args.bs], args.ctx, bshape, bsh)[0]

    pre = Prefetcher(get_batch, range(start, args.steps), depth=2)
    t0 = time.time()
    wall = lambda: prev_s + time.time() - t0  # lm.py's train_s: wall time incl. eval
    t_mark, s_mark, compiled = t0, start, False  # throughput windows exclude eval and compilation
    for step in range(start, args.steps + 1):
        if (step % args.eval_every == 0 or step == args.steps) and not (step == start and start > 0):
            bpb, parts = evaluate(ev, params, val, args.ctx, val_bytes, cfg, args.eval_bs)
            seen = step * tokens_per_step
            point = {"step": step, "tokens": seen, "bytes": seen * bytes_per_tok, "val_bpb": bpb, "val_bpb_parts": parts}
            if extra and step > 0:
                point["by_source"] = {name: evaluate(ev, params, v, args.ctx, vb, cfg, args.eval_bs)[0]
                                      for name, (v, vb) in extra.items()}
            log["curve"].append(point)
            say(f"  step {step:5d}  val bpb {bpb:.4f}  parts {[round(p, 4) for p in parts]}  ({wall():.0f}s)")
            if mngr is not None and step > start and (step % args.ckpt_every == 0 or step == args.steps):
                mngr.save(step, args=ocp.args.Composite(state=ocp.args.StandardSave({"params": params, "opt": opt_state}),
                                                        meta=ocp.args.JsonSave({"log": log, "train_s": wall()})))
            t_mark, s_mark = time.time(), step
        elif mngr is not None and step > start and step % args.ckpt_every == 0:
            mngr.save(step, args=ocp.args.Composite(state=ocp.args.StandardSave({"params": params, "opt": opt_state}),
                                                    meta=ocp.args.JsonSave({"log": log, "train_s": wall()})))
        if step == args.steps:
            break
        s, batch = pre.get()
        assert s == step
        params, opt_state, metrics = step_fn(params, opt_state, batch, jnp.float32(lr_at(step, args)))
        if not compiled:
            jax.block_until_ready(metrics)
            compiled = True
            say(f"  compiled + first step: {time.time() - t_mark:.1f}s")
            t_mark, s_mark = time.time(), step + 1
        elif (step + 1) % args.log_every == 0:
            m = jax.device_get(metrics)
            now = time.time()
            tps = (step + 1 - s_mark) * tokens_per_step / max(now - t_mark, 1e-9)
            mfu = f"  MFU {100 * tps * fpt / (peak * n_dev):.1f}%" if peak else ""
            log.setdefault("throughput", []).append({"step": step + 1, "tokens_per_s": tps,
                                                     "mfu": tps * fpt / (peak * n_dev) if peak else None})
            say(f"  step {step + 1:5d}  loss {float(m['loss']):.4f}  |g| {float(m['grad_norm']):.3f}  "
                f"lr {lr_at(step, args):.2e}  {tps:,.0f} tok/s ({tps / n_dev:,.0f}/device){mfu}")
            t_mark, s_mark = now, step + 1
    log["train_s"] = wall()
    log["final_by_source"] = {}
    for name, (v, vb) in extra.items():
        bpb, parts = evaluate(ev, params, v, args.ctx, vb, cfg, args.eval_bs)
        log["final_by_source"][name] = {"val_bpb": bpb, "val_bpb_parts": parts}
        say(f"  {name}: bpb {bpb:.4f}")
    if mngr is not None:
        mngr.wait_until_finished()
    return log


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tokenizers", nargs="+")
    # --- identical to lm.py
    ap.add_argument("--train_text", default=os.path.join(ROOT, "data", "wiki_en_lm.txt"))
    ap.add_argument("--val_text", default=os.path.join(ROOT, "data", "wiki_en.test.txt"))
    ap.add_argument("--val_chars", type=int, default=2_000_000)
    ap.add_argument("--extra_val", nargs="*", default=[], help="extra validation files, scored once at the end")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--allow_repeat", action="store_true",
                    help="repeat the training data if steps x tokens exceed it (to match compute)")
    ap.add_argument("--bs", type=int, default=32, help="global batch size (sequences)")
    ap.add_argument("--accum", type=int, default=1, help="split each batch into this many micro-batches")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--arch", default="gpt2", choices=["gpt2", "modern"])
    ap.add_argument("--opt", default="adamw", choices=["adamw", "muon"])
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head", default="linear", choices=["linear", "mlp", "chain"])
    ap.add_argument("--head_hidden", type=float, default=1.0)
    ap.add_argument("--order", default="2,1,0,3", help="chain head order; 0=var 1=prefix 2=core 3=suffix")
    ap.add_argument("--tag", default="", help="label stored with the run")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "lm_jax.json"))
    # --- JAX / scale-out only
    ap.add_argument("--compute_dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--fsdp", type=int, default=1, help="mesh axis sharding params + optimiser state (and batch)")
    ap.add_argument("--tensor", type=int, default=1, help="mesh axis for tensor parallelism (vocab / heads / MLP)")
    ap.add_argument("--remat", default="none", choices=["none", "full", "dots"])
    ap.add_argument("--ce_chunk", type=int, default=128, help="positions per chunk of the output cross-entropy (0 = off)")
    ap.add_argument("--eval_bs", type=int, default=8, help="eval windows per batch (rounded up to the batch shards)")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--data_order", default="auto", choices=["auto", "torch", "numpy"],
                    help="torch = exactly lm.py's window order (needs CPU torch); auto = torch if installed")
    ap.add_argument("--cache_dir", default=CACHE, help="where the encode.py .npy caches live")
    ap.add_argument("--ckpt_dir", default="", help="orbax checkpoint root (local path or gs://...); resumes if present")
    ap.add_argument("--ckpt_every", type=int, default=0, help="steps between checkpoints (0 = at every eval)")
    ap.add_argument("--keep_ckpts", type=int, default=2)
    ap.add_argument("--peak_tflops", type=float, default=0, help="per-device peak for MFU (auto for TPUs)")
    ap.add_argument("--prepare_only", action="store_true",
                    help="only tokenise / cache the data (+ .meta.json byte counts for machines without the texts)")
    ap.add_argument("--distributed", default="auto", choices=["auto", "on", "off"],
                    help="jax.distributed.initialize() (auto: on multi-host Cloud TPU)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    # auto: on a TPU VM (TPU device files present) or when a multi-worker TPU env is advertised;
    # initialize() then discovers the other hosts itself (a single-host VM is a 1-process cluster)
    import glob
    on_tpu_vm = bool(glob.glob("/dev/accel*") or glob.glob("/dev/vfio/[0-9]*"))
    multi_host = "," in os.environ.get("TPU_WORKER_HOSTNAMES", "")
    if args.distributed == "on" or (args.distributed == "auto" and (multi_host or on_tpu_vm)):
        jax.distributed.initialize()
    args.ckpt_every = args.ckpt_every or args.eval_every
    mesh = make_mesh(args.fsdp, args.tensor) if (jax.device_count() > 1 or args.fsdp * args.tensor > 1) else None
    results = json.load(open(args.out)) if os.path.exists(args.out) else []
    for t in args.tokenizers:
        r = train(t, args, mesh)
        if r is None:
            continue
        r["args"] = {k: rel(v) if k in ("train_text", "val_text", "out", "cache_dir") else v
                     for k, v in vars(args).items() if k != "extra_val"}
        r["args"]["extra_val"] = [os.path.basename(v) for v in args.extra_val]
        r["args"]["devices"] = jax.device_count()
        r["args"]["device_kind"] = jax.devices()[0].device_kind
        results = [x for x in results if not (x["tokenizer"] == r["tokenizer"] and x["tag"] == r["tag"]
                                              and x["args"]["seed"] == args.seed)]
        results.append(r)
        if jax.process_index() == 0:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
