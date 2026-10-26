# Training the LM on Cloud TPUs (`experiments/jax_lm.py`)

`experiments/jax_lm.py` is a JAX port of `experiments/lm.py`: same model (both `--arch`es, all
`--head`s, factored and standard vocabularies, tied output layers), same AdamW / Muon, LR schedule,
clipping, data order and bits-per-byte evaluation, same CLI flags and the same JSON log format
(default `results/lm_jax.json`, so a TPU run never overwrites a GPU log). It scales to multi-host
slices (written for up to 64 chips) with FSDP, tensor parallelism and a flash-attention kernel.

> Status: tested on CPU only. `tests/test_jax_lm.py` checks parity with the PyTorch code, the
> splash-attention kernel (interpreted) against plain attention, and that every mesh layout trains
> exactly like one device on 4 simulated devices; all layouts also compile without resharding
> fallbacks on 64 simulated devices. Nothing has run on a real TPU yet, so start with the
> shake-down in section 4.

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
not exist there; only their basenames are used to find the cache and sidecar). Every host reads
the caches from its own disk (memory-mapped; each host only reads the rows of its own chips).

## 2. Create a TPU VM (examples; nothing is created by this repo)

Use the zones listed in your TRC grant e-mail. `--spot` gives preemptible capacity (see section 6).
Slice names count TensorCores on v4 / v5p (v4-128 = 64 chips) and chips on v5e / v6e.

| 64 chips | hosts | HBM / chip | bf16 peak / chip |
|---|---|---|---|
| `v4-128` | 16 x 4 chips | 32 GiB | 275 TFLOP/s |
| `v5p-128` | 16 x 4 chips | 95 GiB | 459 TFLOP/s |
| `v5litepod-64` (v5e) | 16 x 4 chips | 16 GiB | 197 TFLOP/s |
| `v6e-64` | 16 x 4 chips | 32 GiB | 918 TFLOP/s |

```bash
# single host first (shake-down): v4-8 = 4 chips
gcloud compute tpus tpu-vm create cbpe-v4 --zone=us-central2-b --accelerator-type=v4-8 --version=tpu-ubuntu2204-base
# 64 chips; large slices are easier to get as queued resources
gcloud compute tpus queued-resources create cbpe-q --node-id=cbpe-v4-128 --zone=us-central2-b \
  --accelerator-type=v4-128 --runtime-version=tpu-ubuntu2204-base --spot
# v5e / v6e runtime versions: v2-alpha-tpuv5-lite / v2-alpha-tpuv6e
```

## 3. Set up every worker

`jax 0.11` needs Python >= 3.12; the TPU images ship an older one, so use `uv`:

```bash
TPU=cbpe-v4-128; ZONE=us-central2-b
gcloud compute tpus tpu-vm ssh $TPU --zone=$ZONE --worker=all --command='
  curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env &&
  git clone https://github.com/SwayStar123/CombinatorialBPE.git && cd CombinatorialBPE &&
  uv venv -p 3.12 .venv && uv pip install -p .venv -r requirements-tpu.txt &&
  mkdir -p data/cache results/tokenizers && gsutil -m cp "gs://MY_BUCKET/cbpe/cache/*" data/cache/ &&
  gsutil -m cp "gs://MY_BUCKET/cbpe/tokenizers/*" results/tokenizers/'
```

(Optional) `uv pip install -p .venv torch --index-url https://download.pytorch.org/whl/cpu` makes the
training-window order identical to `lm.py` (`--data_order torch`); without it a numpy permutation is
used (the log records which; install it on all hosts or none). The tokenizer JSON is only read for
its table sizes; an `unrestricted` tokenizer also needs its model file next to the JSON.

Check the devices: `.venv/bin/python -c "import jax; print(jax.device_count(), jax.devices()[0].device_kind)"`
(on a multi-host slice run it on all workers at once; each sees its 4 local chips until
`jax.distributed.initialize()` joins them).

## 4. Launch

The same command runs on every worker; with `--distributed auto` (default) it calls
`jax.distributed.initialize()` on TPU VMs and every host then feeds only its own chips' rows of
each global batch. Only process 0 prints and writes the JSON.

Shake-down on one host (a few hundred steps; check the loss falls, the MFU line, and a resume):

```bash
.venv/bin/python experiments/jax_lm.py results/tokenizers/mix_32768_comb.json \
  --train_text data/mix_lm3x.txt --val_text data/mix.test.txt --arch modern --opt muon --head chain --order 1,2,0,3 \
  --lr 2e-3 --d 1024 --layers 16 --heads 16 --ctx 1024 --bs 64 --steps 300 --eval_every 100 --fsdp 4 \
  --ckpt_dir gs://MY_BUCKET/cbpe/ckpt --tag shakedown --out results/lm_tpu.json
```

64 chips:

```bash
gcloud compute tpus tpu-vm ssh $TPU --zone=$ZONE --worker=all --command='
  cd CombinatorialBPE && nohup .venv/bin/python experiments/jax_lm.py results/tokenizers/mix_32768_comb.json \
    --train_text data/mix_lm3x.txt --val_text data/mix.test.txt --extra_val data/val_mix_{cpp,de,en,fr,go,ja,java,javascript,python,ru,zh}.txt \
    --arch modern --opt muon --head chain --order 1,2,0,3 --lr 2e-3 \
    --d 2048 --layers 24 --heads 16 --ctx 2048 --bs 256 --steps 20000 --warmup 500 --eval_every 1000 --eval_bs 64 \
    --fsdp 64 --ckpt_dir gs://MY_BUCKET/cbpe/ckpt --tag modern_2048x24 --out results/lm_tpu.json > train.log 2>&1 &'
```

Copy the JSON back with
`gcloud compute tpus tpu-vm scp $TPU:CombinatorialBPE/results/lm_tpu.json results/ --zone=$ZONE --worker=0`.
It is rewritten after every evaluation (entries of unfinished runs have `"partial": true`).
`experiments/report.py`-style readers work on it unchanged (extra keys: `backend`, `throughput`,
more `args`).

Throughput lines (`--log_every`) show tokens/s overall and per chip and an MFU estimate
(6 x matmul params incl. the tied output heads + 12 L d ctx per token, against the bf16 peak of the
detected chip; override with `--peak_tflops`).

## 5. Sharding, memory and batch size

Mesh = `(data, fsdp, tensor)` with `data = chips / (fsdp x tensor)`.

- **Batch** is split over `data x fsdp`: `--bs` (global, in sequences) divided by `--accum` must be
  a multiple of `data x fsdp` (so at least 64 on 64 chips with `--tensor 1`).
- **fsdp** (ZeRO-3) shards every weight matrix, its gradient and its optimiser state. Each layer
  gathers its weights just before use, as a bf16 copy (half the traffic of fp32); the embedding
  tables are gathered once per step in fp32 (lookups stay fp32, as in `lm.py`). The startup log
  lists any matrix left unsharded because no dim divides by `fsdp`.
- **tensor** (Megatron) splits attention by heads, the MLP by hidden units and the embedding / output
  tables by vocabulary, so logits are never replicated. `--heads` must be divisible by `--tensor`.
  Use it when the per-chip batch would otherwise drop below one sequence, or when HBM is short (big
  vocabularies on 16 GiB v5e). Keep it within a host (`--tensor` <= 4 on 4-chip hosts); the mesh puts
  it on the closest chips.
- **Muon** orthogonalises whole matrices: each parameter's per-layer matrices are spread across all
  chips along the layer axis (a few all-to-alls), so Newton-Schulz needs no communication.
- **Attention** (`--attention auto`): the Pallas splash (flash) kernel on TPU when `--ctx` is a
  multiple of 128, so memory is linear in the context; plain XLA attention otherwise.
- **Output layers** compute the cross-entropy in chunks of `--ce_chunk` (128) positions, so only
  `[batch per chip, 128, vocab / tensor]` logits exist at a time.

Training state is ~16 bytes/param (fp32 weights, gradients, two Adam moments or one Muon buffer),
divided by `fsdp x tensor`, plus activations. Starting points for 64 chips:

| model | example flags | layout |
|---|---|---|
| ~100-500M | `--d 1024 --layers 16-24 --heads 16 --ctx 1024-2048` | `--fsdp 64` (or `--fsdp 4` = data parallel across hosts), `--bs 128-256` |
| ~1-2B | `--d 2048 --layers 24 --heads 16 --ctx 2048` | `--fsdp 64`, `--remat dots` (`full` on 16 GiB v5e) |
| ~3-8B | `--d 3072-4096 --layers 32 --heads 32` | `--fsdp 16 --tensor 4`, `--remat full`; 32 GiB+ chips (on v5e add `--accum`) |

XLA's per-chip memory estimates (64 simulated devices, 128k factored vocabulary, one 2048-token
sequence per chip, plain attention): ~6-8 GB for the 1.4B example above (`--remat dots` / `full`;
each extra layer adds only its saved residual, so weights are gathered one layer at a time), ~13 GB
for d 4096 x 32 layers on `--fsdp 16 --tensor 4` with 4 sequences per chip. A few GB of that is the
gathered embedding tables and their gradients, which `--tensor` divides.

Other knobs: `--remat none|dots|full` (activation memory vs ~20-30% recompute); `--accum` for
batches that don't fit; `--compute_dtype float32` for debugging.

## 6. Checkpoints and preemption

`--ckpt_dir` (local or `gs://`; must be `gs://` on multi-host slices) saves params + optimiser state
+ the log so far with orbax, at every eval (or every `--ckpt_every` steps), keeping `--keep_ckpts`
(2). Each run gets `<ckpt_dir>/<tokenizer>__<tag>__seed<seed>/`. Re-running the identical command
resumes from the latest checkpoint; the data order is a pure function of the seed, so a resumed run
reproduces an uninterrupted one.

On a preemption notice (spot / queued capacity) all hosts agree on a step, save a checkpoint there
and exit with "preempted: saved step N"; re-run the same command (e.g. in a retry loop, or after
recreating the queued resource) to continue.

## 7. Differences from lm.py worth knowing

* Parameters use lm.py's names; fused projections are stored split by part, `qkv` as `(3, d, d)`
  and `gate_up` as `(2, hidden, d)`, so `from_torch_state_dict` / `to_torch_state_dict` reshape.
* Output logits come out of the bf16 matmul in fp32 (PyTorch autocast rounds them to bf16 first):
  slightly more exact, not bit-identical.
* Muon's Newton-Schulz runs in bf16 as in `lm.py`; XLA and ATen round differently, so runs agree to
  bf16 noise (the maths matches to ~1e-6 when both sides run it in fp32; see the tests).
* Splash attention scales queries before the kernel (in bf16) instead of the scores (in fp32).
* Evaluation batches can be larger (`--eval_bs`); padding windows are masked, sums are the same.
* `train_s` is wall time including evaluation (as in `lm.py`); throughput numbers exclude
  compilation and evaluation.
