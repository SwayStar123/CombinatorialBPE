# Training the LM on Cloud TPUs (`experiments/jax_lm.py`)

`experiments/jax_lm.py` is a JAX port of `experiments/lm.py`: same model (both `--arch`es, all
`--head`s, factored and standard vocabularies, tied output layers), same AdamW / Muon, LR schedule,
clipping, data order and bits-per-byte evaluation, same CLI flags and the same JSON log format
(default `results/lm_jax.json`, so a TPU run never overwrites a GPU log). `tests/test_jax_lm.py`
checks parity with the PyTorch code on CPU.

> Status: developed and tested on CPU only (incl. 4 simulated devices for the sharded paths).
> Nothing here has been run on a real TPU yet; treat the first TPU run as a shake-down.

## 1. Prepare data locally

The TPU hosts read the same `.npy` caches as `lm.py` (written by `experiments/encode.py`). A TPU VM
does not need the raw texts if each cache has a `.meta.json` sidecar with the byte count, which
`jax_lm.py` writes. Create caches + sidecars without training:

```bash
python experiments/jax_lm.py results/tokenizers/mix_32768_bpe_gpt2.json results/tokenizers/mix_32768_comb.json \
  --train_text data/mix_lm3x.txt --val_text data/mix.test.txt --extra_val data/val_mix_*.txt --prepare_only
```

Upload them (one bucket in the same region as the TPUs; the bucket name is yours to choose):

```bash
gsutil mb -l us-central2 gs://MY_BUCKET                  # once
gsutil -m cp data/cache/mix_32768_*__{mix_lm3x_None,mix.test_2000000,val_mix_*_None}.{npy,meta.json} gs://MY_BUCKET/cbpe/cache/
gsutil -m cp results/tokenizers/mix_32768_{bpe_gpt2,comb}.json gs://MY_BUCKET/cbpe/tokenizers/   # git-ignored, so copy them
```

Keep the paths you pass as `--train_text/--val_text/--extra_val` the same on the TPU (the files need
not exist there; only their basenames are used to find the cache and sidecar).

## 2. Create a TPU VM (examples; nothing is created by this repo)

Use the zones listed in your TRC grant e-mail. `--spot` gives preemptible capacity (resume works, see 5).

```bash
# v4-8: one host, 4 chips (JAX sees 4 devices), 32 GiB HBM each
gcloud compute tpus tpu-vm create cbpe-v4 --zone=us-central2-b --accelerator-type=v4-8 --version=tpu-ubuntu2204-base
# v5e (v5litepod-8): one host, 8 chips, 16 GiB each
gcloud compute tpus tpu-vm create cbpe-v5e --zone=us-west4-a --accelerator-type=v5litepod-8 --version=v2-alpha-tpuv5-lite
# v6e-8: one host, 8 chips, 32 GiB each
gcloud compute tpus tpu-vm create cbpe-v6e --zone=us-east1-d --accelerator-type=v6e-8 --version=v2-alpha-tpuv6e
# pods (several hosts), e.g. v4-32 = 4 hosts x 4 chips; large slices are easier to get as queued resources
gcloud compute tpus queued-resources create cbpe-q --node-id=cbpe-v4-32 --zone=us-central2-b \
  --accelerator-type=v4-32 --runtime-version=tpu-ubuntu2204-base --spot
```

## 3. Set up every worker

`jax 0.11` needs Python >= 3.12; the TPU images ship an older one, so use `uv`:

```bash
TPU=cbpe-v4; ZONE=us-central2-b
gcloud compute tpus tpu-vm ssh $TPU --zone=$ZONE --worker=all --command='
  curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env &&
  git clone -b jax-lm https://github.com/SwayStar123/CombinatorialBPE.git && cd CombinatorialBPE &&
  uv venv -p 3.12 .venv && uv pip install -p .venv -r requirements-tpu.txt &&
  mkdir -p data/cache results/tokenizers && gsutil -m cp "gs://MY_BUCKET/cbpe/cache/*" data/cache/ &&
  gsutil -m cp "gs://MY_BUCKET/cbpe/tokenizers/*" results/tokenizers/'
```

(Optional) `uv pip install -p .venv torch --index-url https://download.pytorch.org/whl/cpu` makes the
training-window order identical to `lm.py` (`--data_order torch`); without it a numpy permutation is
used (the log records which). The tokenizer JSON is only read for its table sizes; an
`unrestricted` tokenizer also needs its model file next to the JSON.

Check the devices: `.venv/bin/python -c "import jax; print(jax.device_count(), jax.devices()[0].device_kind)"`.

## 4. Launch

The same command runs on every worker; with `--distributed auto` (default) it calls
`jax.distributed.initialize()` on TPU VMs, every host then feeds only its own devices' rows of each
global batch. `--bs` is the global batch; it (divided by `--accum`) must be divisible by the number
of batch shards (`devices / tensor`).

```bash
gcloud compute tpus tpu-vm ssh $TPU --zone=$ZONE --worker=all --command='
  cd CombinatorialBPE && nohup .venv/bin/python experiments/jax_lm.py results/tokenizers/mix_32768_comb.json \
    --train_text data/mix_lm3x.txt --val_text data/mix.test.txt --extra_val data/val_mix_{cpp,de,en,fr,go,ja,java,javascript,python,ru,zh}.txt \
    --arch modern --opt muon --head chain --order 1,2,0,3 --lr 2e-3 \
    --d 1024 --layers 16 --heads 16 --ctx 1024 --bs 256 --steps 20000 --warmup 500 --eval_every 1000 --eval_bs 64 \
    --fsdp 4 --ckpt_dir gs://MY_BUCKET/cbpe/ckpt --tag modern_1024x16 --out results/lm_tpu.json > train.log 2>&1 &'
```

Only process 0 prints and writes the JSON; copy it back with
`gcloud compute tpus tpu-vm scp $TPU:CombinatorialBPE/results/lm_tpu.json results/ --zone=$ZONE --worker=0`.
`experiments/report.py`-style readers work on it unchanged (extra keys: `backend`, `throughput`, more `args`).

Throughput lines (`--log_every`) show tokens/s overall and per device and an MFU estimate
(6 x matmul params incl. the tied output heads + 12 L d ctx per token, against the bf16 peak of the
detected chip: v4 275, v5e 197, v5p 459, v6e 918 TFLOP/s; override with `--peak_tflops`).

## 5. Sharding, memory and batch size

Mesh = `(data, fsdp, tensor)` with `data = devices / (fsdp x tensor)`. The batch is split over
`data x fsdp`; `fsdp` shards every weight matrix and its optimiser state (ZeRO-3 style, gathered per
layer by XLA); `tensor` additionally splits embedding tables along the vocabulary and the attention /
MLP matrices Megatron-style (experimental: correct, but XLA reports involuntary resharding for some
tensors, so only use it if fsdp alone doesn't fit). Training state is ~16 bytes/param (fp32 weights,
grads, two Adam moments or one Muon buffer) before activations.

| model (non-emb.) | example flags | suggested layout |
|---|---|---|
| ~50-150M | `--d 768 --layers 12 --heads 12 --ctx 1024` | pure data parallel (`--fsdp 1`); bs 256-512 |
| ~300-500M | `--d 1024 --layers 24 --heads 16` | `--fsdp` = chips per host (4 on v4/v5p, 8 on v5e/v6e) |
| ~1B | `--d 2048 --layers 16 --heads 16` or `--d 1536 --layers 32` | `--fsdp` = all chips of the slice, `--remat dots` (or `full` on 16 GiB v5e) |
| huge vocab (>= 256k) on small-HBM chips | | add `--tensor 2` |

Other knobs: `--remat none|dots|full` (activation memory vs ~20-30% recompute); `--accum` for
bigger batches than fit; `--ce_chunk` (positions per chunk of the output cross-entropy, default 128:
only `[B/shards, 128, V]` logits exist at a time, per factor head); `--compute_dtype float32` for
debugging. Attention is plain XLA softmax attention (scores are materialised); for `--ctx >= 2048`
use `--remat full` (a splash/flash-attention kernel is not wired in yet).

## 6. Checkpoints and preemption

`--ckpt_dir` (local or `gs://`; must be a shared `gs://` path on multi-host slices) saves params +
optimiser state + the log so far with orbax, at every eval (or every `--ckpt_every` steps), keeping
`--keep_ckpts` (2). Each run gets `<ckpt_dir>/<tokenizer>__<tag>__seed<seed>/`. Re-running the
identical command resumes from the latest checkpoint (the data order is a pure function of the seed,
so a resumed run reproduces an uninterrupted one). The JSON is written at the end of the run.

## 7. Differences from lm.py worth knowing

* Output logits come out of the bf16 matmul in fp32 (PyTorch autocast rounds them to bf16 first):
  slightly more exact, not bit-identical.
* Muon's Newton-Schulz runs in bf16 as in `lm.py`; XLA and ATen round differently, so runs agree to
  bf16 noise (the maths matches to ~1e-6 when both sides run it in fp32; see the tests).
* Evaluation batches can be larger (`--eval_bs`); padding windows are masked, sums are the same.
* `train_s` is wall time including evaluation (as in `lm.py`); throughput numbers exclude
  compilation and evaluation.
