"""Numerical parity of experiments/jax_lm.py with experiments/lm.py (tiny configs, CPU only).

Identical initial weights (lm.py's state_dict converted to JAX params) and identical batches:
  * forward loss and per-factor parts match in fp32 (standard vocab and factored heads, both archs);
  * bf16 compute (JAX) vs torch CPU autocast bf16 agree to bf16 accuracy;
  * a few optimiser steps (AdamW and Muon, incl. schedule and clipping) track PyTorch;
  * sharded execution on 4 fake CPU devices (data / fsdp / tensor meshes) matches one device.
"""
import math
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
import jax.numpy as jnp  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import jax_lm  # noqa: E402
import lm  # noqa: E402

torch.set_num_threads(4)
D, L, H, CTX, B = 64, 2, 4, 32, 4
FACTORED = [5, 11, 97, 13]  # (var, pre, core, suf)
STANDARD = [211]


def build(sizes, arch, head="chain", order=(1, 2, 0, 3), seed=0, dtype="float32", ce_chunk=8):
    torch.manual_seed(seed)
    model = lm.GPT(sizes, D, L, H, CTX, head, 1.0, list(order), arch)
    cfg = jax_lm.Config(tuple(sizes), D, L, H, CTX, head, 1.0, tuple(order), arch, dtype, "none", ce_chunk)
    return model, cfg, jax_lm.from_torch_state_dict(model.state_dict(), cfg)


def batch(sizes, seed=0, b=B):
    rng = np.random.default_rng(seed)
    return np.stack([rng.integers(0, n, size=(b, CTX + 1)) for n in sizes], -1).astype(np.int32)


def torch_loss(model, w):
    w = torch.from_numpy(w).long()
    nll, parts = model(w[:, :-1], w[:, 1:])
    return nll.item(), [p.item() for p in parts]


CASES = [(STANDARD, "gpt2", "linear", (2, 1, 0, 3)), (STANDARD, "modern", "linear", (2, 1, 0, 3)),
         (FACTORED, "gpt2", "chain", (2, 1, 0, 3)), (FACTORED, "modern", "chain", (1, 2, 0, 3)),
         (FACTORED, "gpt2", "linear", (2, 1, 0, 3)), (FACTORED, "modern", "mlp", (2, 1, 0, 3))]


@pytest.mark.parametrize("sizes,arch,head,order", CASES, ids=lambda c: str(c))
def test_forward_fp32(sizes, arch, head, order):
    model, cfg, params = build(sizes, arch, head, order)
    w = batch(sizes)
    ref, ref_parts = torch_loss(model, w)
    nll, parts = jax.jit(lambda p, w: jax_lm.forward(p, w, cfg))(params, jnp.asarray(w))
    assert abs(float(nll) - ref) / ref < 2e-6, (float(nll), ref)
    np.testing.assert_allclose([float(p) for p in parts], ref_parts, rtol=2e-6)
    # chunked and unchunked cross-entropy are the same function
    cfg0 = jax_lm.Config(**{**cfg.__dict__, "ce_chunk": 0})
    assert abs(float(jax_lm.forward(params, jnp.asarray(w), cfg0)[0]) - float(nll)) / ref < 1e-6


@pytest.mark.parametrize("sizes,arch", [(FACTORED, "modern"), (STANDARD, "gpt2")])
def test_forward_bf16(sizes, arch):
    model, cfg, params = build(sizes, arch, dtype="bfloat16")
    w = batch(sizes, seed=1)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        ref, ref_parts = torch_loss(model, w)
    nll, parts = jax_lm.forward(params, jnp.asarray(w), cfg)
    assert abs(float(nll) - ref) / ref < 3e-3, (float(nll), ref)
    np.testing.assert_allclose([float(p) for p in parts], ref_parts, rtol=5e-3)


def torch_train(model, opt_name, batches, args):
    """lm.py's training loop body (parameter groups, schedule, clipping), on CPU."""
    named = list(model.named_parameters())
    matrices = [p for n, p in named if p.dim() == 2 and "emb" not in n and "pos" not in n]
    other = [p for n, p in named if not (p.dim() == 2 and "emb" not in n and "pos" not in n)]
    if opt_name == "muon":
        opts = [lm.BatchedMuon(matrices, lr=args.lr, weight_decay=0.1),
                torch.optim.AdamW(other, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))]
    else:
        opts = [torch.optim.AdamW([{"params": matrices, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                                  lr=args.lr, betas=(0.9, 0.95))]
    losses = []
    for step, w in enumerate(batches):
        lr = args.lr * min(1, (step + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for opt in opts:
            for grp in opt.param_groups:
                grp["lr"] = max(lr, args.lr * 0.1 * min(1, (step + 1) / args.warmup))
            opt.zero_grad(set_to_none=True)
        w = torch.from_numpy(w).long()
        nll, _ = model(w[:, :-1], w[:, 1:])
        (nll / (w.shape[0] * CTX)).backward()
        losses.append(nll.item() / (w.shape[0] * CTX))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in opts:
            opt.step()
    return losses


def newton_schulz_fp32(G, steps=5, eps=1e-7, coeffs=(3.4445, -4.775, 2.0315)):
    """lm.newton_schulz with the bf16 cast replaced by fp32 (to test Muon's maths without bf16 noise)."""
    a, b, c = coeffs
    X = G.float()
    tall = X.size(-2) > X.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return X.mT if tall else X


@pytest.mark.parametrize("opt_name", ["adamw", "muon", "muon_fp32"])
@pytest.mark.parametrize("sizes,arch", [(FACTORED, "modern"), (STANDARD, "gpt2")])
def test_optimizer_steps(opt_name, sizes, arch, monkeypatch):
    if opt_name == "muon_fp32":
        monkeypatch.setattr(lm, "newton_schulz", newton_schulz_fp32)
        monkeypatch.setattr(jax_lm, "MUON_NS_DTYPE", jnp.float32)
        opt_name = "muon"
        tol_loss, tol_update = 1e-5, 1e-4
    else:
        tol_loss, tol_update = (5e-5, 0.1) if opt_name == "muon" else (1e-5, 1e-4)
    args = SimpleNamespace(lr=3e-3, warmup=2, steps=4)
    model, cfg, params = build(sizes, arch)
    init = jax_lm.to_torch_state_dict(params, cfg)
    batches = [batch(sizes, seed=10 + s) for s in range(args.steps)]
    ref_losses = torch_train(model, opt_name, batches, args)
    groups = jax_lm.opt_groups(params, opt_name)
    state = jax_lm.opt_init(params, groups)
    step_fn = jax.jit(jax_lm.make_train_step(cfg, groups, B * CTX))
    losses = []
    for s, w in enumerate(batches):
        params, state, m = step_fn(params, state, jnp.asarray(w)[None], jnp.float32(jax_lm.lr_at(s, args)))
        losses.append(float(m["loss"]))
    np.testing.assert_allclose(losses, ref_losses, rtol=tol_loss)
    ref = {n: v.numpy() for n, v in model.state_dict().items()}
    got = jax_lm.to_torch_state_dict(params, cfg)
    # L1 distance between the two trajectories relative to the total L1 size of the updates.
    # bf16 Newton-Schulz rounds differently in XLA and ATen, and 5 iterations amplify it (~3%).
    err = sum(np.abs(got[n] - ref[n]).sum() for n in got) / sum(np.abs(ref[n] - init[n]).sum() for n in got)
    assert err < tol_update, err


def test_grad_accumulation():
    _, cfg, params = build(FACTORED, "gpt2")
    groups = jax_lm.opt_groups(params, "adamw")
    state = jax_lm.opt_init(params, groups)
    w = jnp.asarray(batch(FACTORED, seed=3))
    step_fn = jax.jit(jax_lm.make_train_step(cfg, groups, B * CTX))
    p1, _, m1 = step_fn(params, state, w[None], jnp.float32(1e-3))
    p2, _, m2 = step_fn(params, state, w.reshape(2, B // 2, CTX + 1, -1), jnp.float32(1e-3))
    assert abs(float(m1["loss"]) - float(m2["loss"])) < 1e-5
    assert abs(float(m1["grad_norm"]) - float(m2["grad_norm"])) < 1e-4 * float(m1["grad_norm"])


def test_window_order_matches_lm():
    n_windows, need, seed = 37, 100, 5
    g = torch.Generator().manual_seed(seed)
    ref = torch.cat([torch.randperm(n_windows, generator=g) for _ in range(-(-need // n_windows))])[:need]
    got, how = jax_lm.window_order(n_windows, need, seed)
    assert how == "torch" and (got == ref.numpy()).all()


SHARDED = textwrap.dedent("""
    import os, sys
    sys.path.insert(0, sys.argv[1])
    import numpy as np, jax, jax.numpy as jnp
    import jax_lm
    from jax.sharding import NamedSharding, PartitionSpec as P
    assert jax.device_count() == 4
    jax_lm.MUON_NS_DTYPE = jnp.float32  # compare sharded vs unsharded without bf16 rounding noise
    rng = np.random.default_rng(0)
    sizes = (5, 11, 96, 13)
    w = np.stack([rng.integers(0, n, size=(8, 33)) for n in sizes], -1).astype(np.int32)
    out = {}
    for arch in ("gpt2", "modern"):
        for fsdp, tensor in ((0, 0), (1, 1), (2, 1), (4, 1), (1, 2), (2, 2)):
            mesh = None if fsdp == 0 else jax_lm.make_mesh(fsdp, tensor)
            cfg = jax_lm.Config(sizes, 64, 2, 4, 32, "chain", 1.0, (1, 2, 0, 3), arch, "float32", "full", 8, mesh)
            params = jax.jit(lambda k: jax_lm.init_params(k, cfg),
                             out_shardings=jax_lm.param_shardings(cfg))(jax.random.PRNGKey(0))
            groups = jax_lm.opt_groups(params, "muon")
            state = jax_lm.opt_init(params, groups)
            sh = None if mesh is None else NamedSharding(mesh, P(None, jax_lm.BATCH, None, None))
            data = w.reshape(-1, 4)  # windows of 33 rows: row b*33 .. b*33+32
            wb, _ = jax_lm.make_global_batch(data, np.arange(8) * 33, 32, (1, 8, 33, 4), sh)
            assert (np.asarray(wb)[0] == w).all()
            step = jax.jit(jax_lm.make_train_step(cfg, groups, 8 * 32))
            params, state, m = step(params, state, wb, jnp.float32(1e-2))
            params, state, m = step(params, state, wb, jnp.float32(1e-2))
            got = (float(m["loss"]), {n: np.asarray(v) for n, v in params.items()})
            ref = out.setdefault(arch, got)
            assert abs(got[0] - ref[0]) < 1e-5 * abs(ref[0]), (arch, fsdp, tensor, got[0], ref[0])
            diff = max(np.abs(got[1][n] - ref[1][n]).max() for n in ref[1])
            assert diff < 1e-5, (arch, fsdp, tensor, diff)
            print(arch, fsdp, tensor, got[0], diff)
    print("OK")
""")


def test_sharded_matches_single_device():
    env = dict(os.environ, JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=4")
    r = subprocess.run([sys.executable, "-c", SHARDED, os.path.join(ROOT, "experiments")], env=env,
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]
