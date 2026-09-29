"""JAX port of experiments/lm.py for large-scale training (Cloud TPU v4 / v5e / v5p / v6e, also CPU/GPU).

Same model, optimiser, schedule, data order, evaluation (bits-per-byte) and JSON log format as lm.py,
so results are directly comparable (see lm.py's docstring for the model). Additions for scale
(docs/tpu.md has the how-to):

  * jit-compiled train step; layers are stacked and run with lax.scan (compile time independent of
    depth), optional rematerialisation (--remat);
  * a device mesh (data, fsdp, tensor): the batch is split over data x fsdp; fsdp shards weights,
    gradients and optimiser state (ZeRO-3: each layer gathers its bf16 weights just before use);
    tensor splits heads, MLP hidden units and vocabularies (Megatron column / row parallel);
  * splash (flash) attention on TPU, per device under shard_map;
  * Muon orthogonalises whole matrices spread over all devices along the layer axis, so its
    Newton-Schulz iterations need no communication;
  * the output layers compute the cross-entropy in chunks of positions under jax.checkpoint, so
    [B, T, V] logits are never materialised;
  * multi-host: jax.distributed.initialize() on pods, every host builds only the rows of each
    global batch that live on its own devices (deterministic: all hosts compute the same order);
  * bf16 compute with fp32 master weights and fp32 optimiser state;
  * orbax checkpoints (local or gs://) with automatic resume and a save on preemption notices;
    throughput / MFU logging; the JSON log is rewritten after every evaluation.

Parameters are a flat dict keyed by lm.py's state_dict names in PyTorch layout ((out, in) for linear
weights); the per-layer tensors "blocks.<i>.<name>" are stacked into one "blocks.<name>" array with a
leading layer axis, and fused projections are split by part (qkv: (3, d, d), gate_up: (2, hidden, d)),
so converting a PyTorch checkpoint is a stack and a reshape (from_torch_state_dict / to_torch_state_dict).
"""
import argparse
import dataclasses
import functools
import glob
import json
import math
import os
import queue
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cbpe import load  # noqa: E402
from lm_common import CACHE, ROOT, encode_file, lr_at, rel, window_order  # noqa: E402

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
    mesh: object = None     # jax.sharding.Mesh or None (single device)
    attention: str = "xla"  # xla (scores materialised) | splash (Pallas flash attention; interpreted off TPU)

    @property
    def cdt(self):
        return jnp.dtype(self.compute_dtype)

    @property
    def factored(self):
        return len(self.sizes) == 4

    @property
    def mlp_hidden(self):
        return 4 * self.d if self.arch == "gpt2" else 32 * round(8 * self.d / 3 / 32)

    @property
    def tp(self):
        """the mesh axis heads, hidden units and vocabularies are split over (None: no tensor parallelism)"""
        return "tensor" if self.mesh is not None and self.mesh.shape["tensor"] > 1 else None


# fused projections are stored with a leading part axis, (parts, out / parts, in), so that splitting
# the output over the tensor axis splits heads / hidden units instead of cutting across q|k|v, gate|up
FUSED = {"qkv.weight": 3, "gate_up.weight": 2}


def param_shapes(cfg):
    """{name: shape} in lm.py's naming and PyTorch (out, in) layout; block tensors stacked over layers,
    fused projections split by part: qkv (3, d, d), gate_up (2, hidden, d)."""
    d, L = cfg.d, cfg.n_layer
    s = {f"emb.{i}.weight": (n, d) for i, n in enumerate(cfg.sizes)}
    if cfg.arch == "modern":
        hd, hid = d // cfg.n_head, cfg.mlp_hidden
        blk = {"ln1.weight": (d,), "ln2.weight": (d,), "q_norm.weight": (hd,), "k_norm.weight": (hd,),
               "qkv.weight": (3, d, d), "o.weight": (d, d), "gate_up.weight": (2, hid, d), "down.weight": (d, hid)}
        s["ln_f.weight"] = (d,)
    else:
        s["pos.weight"] = (cfg.ctx, d)
        blk = {"ln1.weight": (d,), "ln1.bias": (d,), "ln2.weight": (d,), "ln2.bias": (d,), "qkv.weight": (3, d, d),
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
    """shape of one layer's tensor (without the stacked layer axis)"""
    return shape[1:] if name.startswith("blocks.") else shape


def torch_shape(name, shape):
    """one layer's tensor in lm.py's layout (a fused projection is one (parts * out, in) matrix)"""
    ls = layer_shape(name, shape)
    return (ls[0] * ls[1], ls[2]) if name.split(".", 1)[-1] in FUSED else ls


def is_matrix(name, shape):
    """lm.py's grouping: 2-D weights except the embedding tables / positions (-> Muon or weight decay)."""
    return len(torch_shape(name, shape)) == 2 and "emb" not in name and "pos" not in name


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
    for name, shape in param_shapes(cfg).items():
        if name.startswith("blocks."):
            sub = name[len("blocks."):]
            a = np.stack([to_np(sd[f"blocks.{l}.{sub}"]) for l in range(cfg.n_layer)])
        else:
            a = to_np(sd[name])
        out[name] = jnp.asarray(a.reshape(shape))
    return out


def to_torch_state_dict(params, cfg):
    """JAX params -> {name: np.ndarray} loadable into lm.py's GPT (load_state_dict(strict=False) ignores rope buffers)."""
    out = {}
    for name, v in params.items():
        v = np.asarray(jax.device_get(v))
        ts = torch_shape(name, v.shape)
        if name.startswith("blocks."):
            for l in range(cfg.n_layer):
                out[f"blocks.{l}.{name[len('blocks.'):]}"] = v[l].reshape(ts)
        else:
            out[name] = v.reshape(ts)
    return out


# --------------------------------------------------------------- sharding
def make_mesh(fsdp=1, tensor=1):
    devs = np.array(jax.devices())
    n = len(devs)
    assert n % (fsdp * tensor) == 0, f"{n} devices not divisible by fsdp={fsdp} x tensor={tensor}"
    shape = (n // (fsdp * tensor), fsdp, tensor)
    try:  # topology-aware: the last (tensor) axis gets the closest chips
        from jax.experimental import mesh_utils
        devs = mesh_utils.create_device_mesh(shape, allow_split_physical_axes=True)
    except Exception:  # noqa: BLE001  (e.g. CPU / odd topologies)
        devs = devs.reshape(shape)
    return Mesh(devs, ("data", "fsdp", "tensor"))


COLUMN = ("qkv.weight", "mlp.0.weight", "gate_up.weight", ".1.weight")  # output dim over tensor
# every other matrix (o, mlp.2, down, steps.<k>.3, cond) is row-parallel: input dim over tensor


def param_spec(name, shape, mesh, fsdp=True):
    """PartitionSpec of a parameter. Vectors are replicated. Matrices (out, in): one dim over fsdp, the
    other over tensor (Megatron column / row split by heads and hidden units; embedding tables split
    along the vocabulary). A dim that does not divide evenly stays unsplit. fsdp=False: the layout a
    layer computes with (gathered over fsdp, still split over tensor)."""
    if mesh is None:
        return P()
    lead = (None,) if name.startswith("blocks.") else ()
    ls = layer_shape(name, shape)
    if len(ls) == 1:
        return P(*lead, None)
    out_ax, in_ax = "fsdp", "tensor"
    if name.startswith("emb.") or name.endswith(COLUMN):
        out_ax, in_ax = "tensor", "fsdp"
    elif name == "pos.weight":
        out_ax, in_ax = None, "fsdp"
    ok = lambda ax, n: ax if ax and (fsdp or ax != "fsdp") and mesh.shape[ax] > 1 and n % mesh.shape[ax] == 0 else None
    return P(*lead, *(None,) * (len(ls) - 2), ok(out_ax, ls[-2]), ok(in_ax, ls[-1]))


def param_shardings(cfg):
    if cfg.mesh is None:
        return None
    return {n: NamedSharding(cfg.mesh, param_spec(n, s, cfg.mesh)) for n, s in param_shapes(cfg).items()}


@functools.lru_cache(maxsize=None)
def compute_shardings(cfg):
    """{name: sharding of one layer's weight as the forward pass uses it} (param_spec with fsdp=False)"""
    out = {}
    for n, s in param_shapes(cfg).items():
        spec = param_spec(n, s, cfg.mesh, fsdp=False)
        out[n] = NamedSharding(cfg.mesh, P(*spec[1:]) if n.startswith("blocks.") else spec)
    return out


def _weight(w, name, cfg, dtype=None):
    """A weight as a layer uses it: cast to the compute dtype, then (with a mesh) gathered over fsdp
    (ZeRO-3: one all-gather per use, of the cast copy) and still split over tensor."""
    w = w.astype(dtype or cfg.cdt)
    if cfg.mesh is None:
        return w
    return lax.with_sharding_constraint(w, compute_shardings(cfg)[name])


def _act(x, cfg, *spec):
    """Pin an activation's sharding: dim 0 (batch) over data x fsdp, the other dims as in spec."""
    if cfg.mesh is None:
        return x
    return lax.with_sharding_constraint(x, NamedSharding(cfg.mesh, P(BATCH, *spec)))


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


def _dense(x, w):
    """x @ w.T with w in PyTorch (out, in) layout, already in the compute dtype (see _weight)."""
    return jnp.einsum("...i,oi->...o", x.astype(w.dtype), w)


def _gelu(x):
    return jax.nn.gelu(x, approximate=False)


def rope_tables(T, head_dim, base=10000.0):
    """Same as lm.py (fp32 maths, stored in bf16): cos/sin shaped (T, head_dim/2)."""
    inv = 1.0 / base ** (np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim))
    ang = np.outer(np.arange(T, dtype=np.float32), inv.astype(np.float32))
    return jnp.asarray(np.cos(ang), jnp.bfloat16), jnp.asarray(np.sin(ang), jnp.bfloat16)


def _rope(x, cos, sin):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def _attention(q, k, v, cfg):
    """Causal attention on (B, H, T, hd). xla: scores and softmax in fp32 like SDPA's kernels."""
    if cfg.attention == "splash":
        return _splash_attention(q, k, v, cfg)
    T, hd = q.shape[2], q.shape[3]
    s = jnp.einsum("bhqd,bhkd->bhqk", q, k, preferred_element_type=F32) * (1.0 / math.sqrt(hd))
    s = jnp.where(jnp.tril(jnp.ones((T, T), bool)), s, -jnp.inf)
    return jnp.einsum("bhqk,bhkd->bhqd", jax.nn.softmax(s, axis=-1).astype(v.dtype), v)


def _splash_attention(q, k, v, cfg):
    """Causal flash attention with the Pallas TPU splash kernel (no [T, T] scores in memory). Runs per
    device under shard_map: batch over data x fsdp, heads over tensor. Off TPU it is interpreted
    (slow; tests only). T must be a multiple of 128."""
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as sm
    T, hd = q.shape[2], q.shape[3]
    blk = next(b for b in (512, 256, 128) if T % b == 0)
    blocks = sk.BlockSizes(block_q=blk, block_kv=blk, block_kv_compute=blk, block_q_dkv=blk, block_kv_dkv=blk,
                           block_kv_dkv_compute=blk, block_q_dq=blk, block_kv_dq=blk)

    def local(q, k, v):  # this device's (b, h, T, hd)
        mask = sm.MultiHeadMask([sm.CausalMask((T, T))] * q.shape[1])
        kernel = sk.make_splash_mha(mask, block_sizes=blocks, head_shards=1, q_seq_shards=1,
                                    interpret=jax.default_backend() != "tpu")
        return jax.vmap(kernel)(q * (1.0 / math.sqrt(hd)), k, v)

    if cfg.mesh is None:
        return local(q, k, v)
    spec = P(BATCH, cfg.tp, None, None)
    return jax.shard_map(local, mesh=cfg.mesh, in_specs=(spec, spec, spec), out_specs=spec, check_vma=False)(q, k, v)


def _block(x, p, cfg, rope):
    """One transformer layer on the residual stream x [B, T, D] (fp32); p: this layer's parameters."""
    B, T, D = x.shape
    H = cfg.n_head
    w = lambda n: _weight(p[n], "blocks." + n, cfg)
    modern = cfg.arch == "modern"
    a = _rms(x, p["ln1.weight"]) if modern else _ln(x, p["ln1.weight"], p["ln1.bias"])
    qkv = jnp.einsum("btd,shed->sbhte", a.astype(cfg.cdt), w("qkv.weight").reshape(3, H, D // H, D))
    q, k, v = (_act(t, cfg, cfg.tp, None, None) for t in qkv)  # (B, H, T, hd), heads over tensor
    if modern:
        q = _rope(_rms(q, p["q_norm.weight"]), *rope).astype(v.dtype)
        k = _rope(_rms(k, p["k_norm.weight"]), *rope).astype(v.dtype)
    o = jnp.einsum("bhte,dhe->btd", _attention(q, k, v, cfg), w("o.weight").reshape(D, H, D // H))
    x = _act(x + o, cfg, None, None)
    if modern:
        g, u = jnp.einsum("btd,shd->sbth", _rms(x, p["ln2.weight"]).astype(cfg.cdt), w("gate_up.weight"))
        hmid, down = jax.nn.silu(g) * u, "down.weight"
    else:
        hmid, down = _gelu(_dense(_ln(x, p["ln2.weight"], p["ln2.bias"]), w("mlp.0.weight"))), "mlp.2.weight"
    return _act(x + _dense(_act(hmid, cfg, None, cfg.tp), w(down)), cfg, None, None)


def _ce_sum(g, table, y, wts, cfg):
    """sum_t wts * -log softmax(g @ table.T)[y], never materialising more than [B, ce_chunk, V] logits.
    Logits come out of the matmul in fp32 (lm.py rounds them to bf16 first; this is slightly more exact)."""
    B, T, D = g.shape
    tab = table.astype(cfg.cdt)

    def chunk_nll(gi, yi, wi):
        logits = jnp.einsum("btd,vd->btv", gi.astype(cfg.cdt), tab, preferred_element_type=F32)
        logits = _act(logits, cfg, None, cfg.tp)  # vocabulary over tensor
        tgt = jnp.take_along_axis(logits, yi[..., None], axis=-1)[..., 0]
        return ((jax.nn.logsumexp(logits, axis=-1) - tgt) * wi).sum()

    c = cfg.ce_chunk
    if not c or c >= T:
        return chunk_nll(g, y, wts)
    n, rest = divmod(T, c)
    split = lambda a: jnp.moveaxis(a[:, :n * c].reshape((B, n, c) + a.shape[2:]), 1, 0)
    nll = jax.checkpoint(chunk_nll)

    def body(acc, inp):
        return acc + nll(*inp), None

    tot, _ = lax.scan(body, jnp.zeros((), F32), (split(g), split(y), split(wts)))
    if rest:
        tot = tot + nll(g[:, n * c:], y[:, n * c:], wts[:, n * c:])
    return tot


def forward(params, w, cfg, wts=None):
    """w: [B, T+1, F] int windows (inputs w[:, :-1], targets w[:, 1:]); wts: optional [B] window weights
    (eval padding). Returns (summed NLL in nats, list of per-factor sums) like GPT.forward in lm.py."""
    x, y = w[:, :-1], w[:, 1:]
    B, T, _ = x.shape
    wts = jnp.ones((B, T), F32) if wts is None else jnp.broadcast_to(wts.astype(F32)[:, None], (B, T))
    W = lambda n: _weight(params[n], n, cfg)
    # embedding tables are gathered once, in fp32: lookups stay fp32 as in lm.py, the output layers cast
    tables = [_weight(params[f"emb.{i}.weight"], f"emb.{i}.weight", cfg, F32) for i in range(len(cfg.sizes))]
    emb = lambda i, idx: jnp.take(tables[i], idx, axis=0)
    h = sum(emb(i, x[..., i]) for i in range(len(cfg.sizes)))
    rope = None
    if cfg.arch == "modern":
        cos, sin = rope_tables(cfg.ctx, cfg.d // cfg.n_head)
        rope = (cos[:T], sin[:T])
    else:
        h = h + _weight(params["pos.weight"], "pos.weight", cfg, F32)[:T]
    h = _act(h, cfg, None, None)

    def body(h, p):
        return _block(h, p, cfg, rope), None

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

    ce = lambda g, i: _ce_sum(g, tables[i], y[..., i], wts, cfg)
    if not cfg.factored:
        nll = ce(h, 0)
        return nll, [nll]

    def step(k, a, b):
        z = _ln(jnp.concatenate([a, b], -1), params[f"steps.{k}.0.weight"], params[f"steps.{k}.0.bias"])
        hmid = _act(_gelu(_dense(z, W(f"steps.{k}.1.weight"))), cfg, None, cfg.tp)
        return _dense(hmid, W(f"steps.{k}.3.weight"))

    ln_step = lambda k, g: _ln(g, params[f"ln_steps.{k}.weight"], params[f"ln_steps.{k}.bias"])
    parts = [None] * 4
    if cfg.head != "chain":
        parts[2] = ce(h, 2)
    if cfg.head == "linear":
        g = _ln(h + _dense(emb(2, y[..., 2]), W("cond.weight")), params["ln_c.weight"], params["ln_c.bias"])
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


def newton_schulz(G, steps=5, eps=1e-7, coeffs=(3.4445, -4.775, 2.0315)):
    """Batched quintic Newton-Schulz orthogonalisation over the last two axes, in bf16 (as lm.py)."""
    a, b, c = coeffs
    dt = MUON_NS_DTYPE
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


def _spread_layers(X, spec, mesh):
    """Reshard X (PartitionSpec spec, dim 0 unsplit, its length a multiple of mesh.size) so that dim 0
    is split over every mesh axis and no other dim is split: one all-to-all per split dim (moving its
    axis onto dim 0), then local slices for the remaining axes. Returns X and the specs passed through
    (undone in reverse by _unspread_layers)."""
    specs = [tuple(spec)]
    for k in range(1, X.ndim):
        if specs[-1][k] is not None:
            s = list(specs[-1])
            s[0], s[k] = (s[0] or ()) + (s[k],), None
            specs.append(tuple(s))
    rest = tuple(a for a in mesh.axis_names if a not in (specs[-1][0] or ()))
    specs.append(((specs[-1][0] or ()) + rest,) + specs[-1][1:])
    for s in specs:  # (the first pins X itself, so the moves happen after any padding)
        X = lax.with_sharding_constraint(X, NamedSharding(mesh, P(*s)))
    return X, specs


def _unspread_layers(X, specs, mesh):
    for s in reversed(specs[:-1]):
        X = lax.with_sharding_constraint(X, NamedSharding(mesh, P(*s)))
    return X


def orthogonalise(u, name, mesh=None):
    """Muon's direction for one parameter: Newton-Schulz on each layer's matrix in lm.py's layout.
    With a mesh the matrices are first spread whole over all devices along the layer axis (padded with
    zero matrices to a multiple of the device count), so the iterations need no communication: a few
    all-to-alls in and out instead of collectives inside every matmul of a sharded matrix."""
    blocks = name.startswith("blocks.")
    X = u.astype(MUON_NS_DTYPE)
    if not blocks:
        X = X[None]
    as_matrices = lambda A: A.reshape((A.shape[0],) + tuple(torch_shape(name, u.shape)))  # (layers, out, in)
    if mesh is None:
        return newton_schulz(as_matrices(X)).reshape(u.shape)
    n = X.shape[0]
    X = jnp.pad(X, ((0, -n % mesh.size),) + ((0, 0),) * (X.ndim - 1))
    spec = tuple(param_spec(name, u.shape, mesh))
    X, specs = _spread_layers(X, spec if blocks else (None,) + spec, mesh)
    O = newton_schulz(as_matrices(X)).reshape(X.shape)  # whole matrices on each device: local reshapes
    return _unspread_layers(O, specs, mesh)[:n].reshape(u.shape)


def opt_update(params, grads, state, lr, groups, cfg=None, betas=(0.9, 0.95), eps=1e-8, momentum=0.95):
    """One step of torch.optim.AdamW (decoupled decay, bias correction) / lm.py BatchedMuon (Nesterov)."""
    mesh = None if cfg is None else cfg.mesh
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
            O = orthogonalise(g + momentum * buf[n], n, mesh).astype(F32)
            scale = 0.2 * max(torch_shape(n, p.shape)) ** 0.5  # match the update RMS of AdamW
            new_p[n] = p - (lr * scale) * O
        else:
            m[n] = b1 * m[n] + (1 - b1) * g
            v[n] = b2 * v[n] + (1 - b2) * g * g
            new_p[n] = p - (lr / bc1) * m[n] / (jnp.sqrt(v[n]) / jnp.sqrt(bc2) + eps)
    return new_p, {"t": t, "m": m, "v": v, "buf": buf}


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
        params, opt_state = opt_update(params, grads, opt_state, lr, groups, cfg)
        return params, opt_state, {"loss": l, "grad_norm": gnorm}

    return train_step


# ---------------------------------------------------------------------- data
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
    return cfg.mesh.shape["data"] * cfg.mesh.shape["fsdp"]


def make_eval_fn(cfg):
    @jax.jit
    def ev(params, w, wts):
        nll, parts = forward(params, w, cfg, wts)
        return nll, jnp.stack(parts)
    return ev


def evaluate(ev, params, val, ctx, n_bytes, cfg, bs=8):
    """lm.py's evaluate(): non-overlapping windows of the validation ids [N, F], bits per byte (total and
    per factor)."""
    n = (len(val) - 1) // ctx
    k = n_batch_shards(cfg)
    bs = -(-bs // k) * k
    F = val.shape[1]
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


def train(tok_path, args, mesh, publish=lambda log: None):
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

    attention = args.attention
    if attention == "auto":
        attention = "splash" if jax.default_backend() == "tpu" and args.ctx % 128 == 0 else "xla"
    assert attention != "splash" or args.ctx % 128 == 0, "splash attention needs --ctx divisible by 128"
    cfg = Config(tuple(sizes), args.d, args.layers, args.heads, args.ctx, args.head, args.head_hidden,
                 tuple(int(c) for c in args.order.split(",")), args.arch, args.compute_dtype, args.remat,
                 args.ce_chunk, mesh, attention)
    shardings = param_shardings(cfg)
    if mesh is not None and mesh.shape["fsdp"] > 1:
        whole = [n for n, s in param_shapes(cfg).items()
                 if len(layer_shape(n, s)) > 1 and "fsdp" not in tuple(param_spec(n, s, mesh))]
        if whole:
            say(f"  not sharded over fsdp (no dim divisible by {mesh.shape['fsdp']}): {', '.join(whole)}")
    params = jax.jit(functools.partial(init_params, cfg=cfg), out_shardings=shardings)(jax.random.PRNGKey(args.seed))
    groups = opt_groups(params, args.opt)
    opt_state = jax.jit(functools.partial(opt_init, groups=groups),
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
        f"({jax.devices()[0].device_kind}), mesh {None if mesh is None else dict(mesh.shape)}, "
        f"{fpt / 1e6:.1f} MFLOP/token, attention {attention}, data order {order_how}")

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

    def save(step):
        mngr.save(step, args=ocp.args.Composite(state=ocp.args.StandardSave({"params": params, "opt": opt_state}),
                                                meta=ocp.args.JsonSave({"log": log, "train_s": wall()})))

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
            if step < args.steps:
                publish(dict(log, train_s=wall(), partial=True))  # the curve so far survives a crash
            t_mark, s_mark = time.time(), step
        if mngr is not None and step > start:
            if step % args.ckpt_every == 0 or step == args.steps:
                save(step)
                t_mark, s_mark = time.time(), step
            elif mngr.reached_preemption(step):  # all hosts agree on this step after a preemption notice
                save(step)
                mngr.wait_until_finished()
                raise SystemExit(f"preempted: saved step {step}; re-run the same command to resume")
        if step == args.steps:
            break
        s, batch = pre.get()
        assert s == step
        lr = lr_at(step, args.lr, args.warmup, args.steps)
        params, opt_state, metrics = step_fn(params, opt_state, batch, jnp.float32(lr))
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
                f"lr {lr:.2e}  {tps:,.0f} tok/s ({tps / n_dev:,.0f}/device){mfu}")
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
    ap.add_argument("--attention", default="auto", choices=["auto", "splash", "xla"],
                    help="splash = Pallas flash attention (TPU); auto = splash on TPU when --ctx %% 128 == 0")
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
    on_tpu_vm = bool(glob.glob("/dev/accel*") or glob.glob("/dev/vfio/[0-9]*"))
    multi_host = "," in os.environ.get("TPU_WORKER_HOSTNAMES", "")
    if args.distributed == "on" or (args.distributed == "auto" and (multi_host or on_tpu_vm)):
        jax.distributed.initialize()
    args.ckpt_every = args.ckpt_every or args.eval_every
    assert args.heads % args.tensor == 0, "--heads must be divisible by --tensor (heads are split over it)"
    assert not args.ckpt_dir or jax.process_count() == 1 or "://" in args.ckpt_dir, \
        "multi-host runs need a shared --ckpt_dir (gs://...): every host writes its own shards"
    mesh = make_mesh(args.fsdp, args.tensor) if (jax.device_count() > 1 or args.fsdp * args.tensor > 1) else None
    results = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
    run_args = {k: rel(v) if k in ("train_text", "val_text", "out", "cache_dir") else v
                for k, v in vars(args).items() if k != "extra_val"}
    run_args.update(extra_val=[os.path.basename(v) for v in args.extra_val], devices=jax.device_count(),
                    device_kind=jax.devices()[0].device_kind)

    def publish(log):
        """write the run's log into --out (process 0), replacing an earlier entry of the same run"""
        nonlocal results
        r = dict(log, args=run_args)
        results = [x for x in results if not (x["tokenizer"] == r["tokenizer"] and x["tag"] == r["tag"]
                                              and x["args"]["seed"] == args.seed)] + [r]
        if jax.process_index() == 0:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out + ".tmp", "w") as f:
                json.dump(results, f, indent=1)
            os.replace(args.out + ".tmp", args.out)

    for t in args.tokenizers:
        log = train(t, args, mesh, publish)
        if log is not None:
            publish(log)


if __name__ == "__main__":
    main()
