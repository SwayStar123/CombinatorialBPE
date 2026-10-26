"""Regenerates tests/fixtures/qwen3_tiny.{safetensors,json,npz}: a random tiny FactoredQwen3 and the
per-token NLL of two random windows computed by the JAX trainer (experiments/jax_lm.py of the jax-lm
branch, fp32, CPU), the reference of tests/test_qwen3.py. Needs jax, torch and transformers, and that
branch's experiments/ directory:

    python tests/fixtures/make_qwen3_tiny.py /path/to/jax-lm-checkout/experiments
"""
import json
import os
import sys

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, sys.argv[1])
from cbpe.qwen3 import FactoredQwen3  # noqa: E402  (before jax_lm, which puts its own checkout's cbpe first)
import jax.numpy as jnp  # noqa: E402
import jax_lm  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

CONFIG = {"model_type": "qwen3", "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 2,
          "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 24, "rope_theta": 1e6,
          "rms_norm_eps": 1e-6, "tie_word_embeddings": True, "max_position_embeddings": 32,
          "attention_bias": False, "hidden_act": "silu"}
SIZES, ORDER, T = (4, 6, 40, 5), (2, 1, 0, 3), 16

torch.manual_seed(0)
model = FactoredQwen3(CONFIG, SIZES, ORDER, torch.float32)
with torch.no_grad():  # non-trivial norms / head (they start at 1 / 0)
    for n, p in model.named_parameters():
        if "norm" in n or n.startswith("ln_steps") or (n.startswith("steps") and p.dim() == 1):
            p.add_(0.3 * torch.randn_like(p))
        elif n.startswith("steps"):
            p.mul_(3.0)
        elif n.startswith("emb"):
            p.mul_(0.2)
sd = {k: v.contiguous() for k, v in model.state_dict().items() if k != "model.embed_tokens.weight"}
save_file(sd, os.path.join(HERE, "qwen3_tiny.safetensors"))
json.dump({**CONFIG, "cbpe": {"sizes": list(SIZES), "order": list(ORDER)}}, open(os.path.join(HERE, "qwen3_tiny.json"), "w"))

# the same weights as JAX params (the mapping of experiments/export_j300.py, inverted)
cfg = jax_lm.Config(SIZES, 64, 2, 4, 32, "chain", 1.0, ORDER, "qwen3", "float32", n_kv_head=2, head_dim_=24,
                    mlp_dim=96, rope_theta=1e6, norm_eps=1e-6, ce_chunk=8)
g = lambda k: sd[k].numpy()
L = lambda f: np.stack([f(l) for l in range(2)])
P = lambda l, s: g(f"model.layers.{l}.{s}")
params = {
    "blocks.ln1.weight": L(lambda l: P(l, "input_layernorm.weight")),
    "blocks.ln2.weight": L(lambda l: P(l, "post_attention_layernorm.weight")),
    "blocks.q_norm.weight": L(lambda l: P(l, "self_attn.q_norm.weight")),
    "blocks.k_norm.weight": L(lambda l: P(l, "self_attn.k_norm.weight")),
    "blocks.q.weight": L(lambda l: P(l, "self_attn.q_proj.weight")),
    "blocks.o.weight": L(lambda l: P(l, "self_attn.o_proj.weight")),
    "blocks.kv.weight": L(lambda l: np.stack([P(l, "self_attn.k_proj.weight"), P(l, "self_attn.v_proj.weight")])),
    "blocks.gate_up.weight": L(lambda l: np.stack([P(l, "mlp.gate_proj.weight"), P(l, "mlp.up_proj.weight")])),
    "blocks.down.weight": L(lambda l: P(l, "mlp.down_proj.weight")),
    "ln_f.weight": g("model.norm.weight"),
}
params.update({k: v.numpy() for k, v in sd.items() if k.startswith(("emb.", "steps.", "ln_steps."))})
params = {k: jnp.asarray(v) for k, v in params.items()}
assert {k: tuple(v.shape) for k, v in params.items()} == {k: tuple(v) for k, v in jax_lm.param_shapes(cfg).items()}


def fp32_rope(T, hd, base=10000.0):  # (jax_lm stores the RoPE tables in bf16, as Hugging Face casts them to the bf16 compute dtype)
    inv = 1.0 / base ** (np.arange(0, hd, 2, dtype=np.float32) / np.float32(hd))
    ang = np.outer(np.arange(T, dtype=np.float32), inv)
    return jnp.asarray(np.cos(ang)), jnp.asarray(np.sin(ang))


jax_lm.rope_tables = fp32_rope
rng = np.random.default_rng(0)
ids = np.stack([rng.integers(0, n, (2, T + 1)) for n in SIZES], -1).astype(np.int32)
out = jax_lm.forward(params, jnp.asarray(ids), cfg, per_position="fields")
np.savez(os.path.join(HERE, "qwen3_tiny.npz"), ids=ids, nll=np.asarray(out[3], np.float32))
print("nll per field", np.asarray(out[3]).mean((0, 1)))
