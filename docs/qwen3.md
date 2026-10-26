# Qwen3-1.7B on a Combinatorial BPE vocabulary: running the converted model

`cbpe/qwen3.py` runs **Qwen3-1.7B-Base healed onto a 131,072-row unrestricted Combinatorial BPE
vocabulary** (tables of 8 variations, 1,469 prefixes, 128,561 cores and 1,038 suffixes). Training
(the JAX / TPU / GPU trainer, the embedding transplant and the data pipeline) lives on the `jax-lm`
branch; this directory only needs what is required to *use* the finished model. The weights are on
Hugging Face: https://huggingface.co/SwayStar123/Qwen3-1.7B-CBPE

## Quickest: from the Hugging Face repository

```bash
pip install torch transformers safetensors huggingface_hub regex
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
repo = "SwayStar123/Qwen3-1.7B-CBPE"
tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)    # bf16, on the GPU if there is one
inputs = tok("The capital of France is", return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=40, tokenizer=tok)             # tokenizer= turns token healing on
print(tok.decode(out[0]))
```

`input_ids` are one integer per token (the four fields packed in mixed radix), so `tok(...)`, batches
with padding, `tok.decode` and `model(input_ids, labels=input_ids).loss` work as for any model. There
is no flat vocabulary, hence no `logits` and no transformers pipelines / vLLM: `model.generate` is
our own field-by-field sampler (greedy, or `do_sample=True` with `temperature` / `top_p`), with a KV
cache. `model.token_nll(ids)` gives the per-field NLL (variation, prefix, core, suffix).

The tokenizer needs **no Rust**: without the `dict_search` binary it uses a pure-Python encoder with
identical output (roughly 1-9k words per second; see below). For bulk tokenisation build the binary
(below) and point `DS_BIN` at it, or have it in `native/dict_search/target/release`.

## From a clone of this repository

```bash
git clone https://github.com/SwayStar123/CombinatorialBPE && cd CombinatorialBPE
pip install -e ".[qwen3]"
cargo build --release --manifest-path native/dict_search/Cargo.toml   # optional: the fast tokenizer encoder
```

```python
from cbpe.qwen3 import load_model, generate, bits_per_byte
model, tok = load_model("SwayStar123/Qwen3-1.7B-CBPE")      # or a local directory
text, tokens = generate(model, tok, "The capital of France is", max_new=40)
```

```bash
python -m cbpe.qwen3 --model SwayStar123/Qwen3-1.7B-CBPE --prompt "def fibonacci(n):" --max_new 60
python -m cbpe.qwen3 --model DIR --bpb test_en.txt          # bits per byte, 2048-token windows
```

`generate` uses token healing by default (the prompt's last token is dropped and the first new
tokens must respell its bytes, so a prompt that ends mid-word or in a space is not forced onto an
unnatural token boundary); `--no-heal` / `heal=False` turns it off. The model has no special tokens:
chat markers (`<|im_start|>`, ...) are ordinary text, and `stop=[...]` ends a generation at a string.

## What the model is

* **Input**: the embedding of a token is the sum of its four table rows (variation, prefix, core, suffix).
* **Body**: transformers' `Qwen3Model`, architecture unchanged, fully fine-tuned.
* **Output**: a *chain head*. The tables are tied with the input; the core is predicted first from the
  final hidden state, then prefix, variation and suffix, each conditioned on the ones already chosen
  through a small residual MLP step. A token's log-probability is the sum of its four fields'.
  The exact maths is in the docstring of `cbpe/qwen3.py`.
* **Numerics**: the body runs in bf16; the tables, the head and the logits in fp32 (the stored weights are bf16).
* Trained with 2048-token contexts (about 12k characters); longer contexts are untested.

## The pure-Python encoder

`cbpe/unrestricted_py.py` ports the DP of `dict_search encode` for case_affixes models without the
pairwise / code-length terms (what this tokenizer is; other model kinds raise `NotImplementedError`
and need the binary). `UnrestrictedBPE` uses it when no binary is found (`CBPE_ENCODER=python` /
`rust` forces one). It is verified against the binary: `tests/test_unrestricted_py.py` compares them
on models trained in the test, and for the published tokenizer they agreed on every chunk of the ten
test files and the mixed test set (about 150k distinct chunks), and of 1.1M synthetic chunks (case perturbations, glued and cut
words, random alphabet characters, characters outside the alphabet, combining marks).

## The Hugging Face repository

`scripts/make_hf_repo.py` assembles the directory that `AutoModelForCausalLM` / `AutoTokenizer` load
(`trust_remote_code=True`): the weights, `config.json` with `auto_map`, the tokenizer files, and the
code the remote loading needs (the files of `hf_remote/` plus copies of `cbpe/qwen3.py` and the
tokenizer modules). Tested with transformers 4.57 and 5.19.

## Correctness

`tests/test_qwen3.py` checks the port against the JAX trainer: a random tiny model (the fixture in
`tests/fixtures/`, regenerated by `make_qwen3_tiny.py` from the `jax-lm` branch) must give the same
per-token, per-field NLL as `experiments/jax_lm.py` in fp32 (tolerance 2e-4), the KV-cached forward
must equal the full forward, and greedy / healed generation is checked against uncached recomputation.
`tests/test_hf_remote.py` does the same through `AutoModelForCausalLM`. The real weights were compared
with the JAX reference on the test files, see the model card.
