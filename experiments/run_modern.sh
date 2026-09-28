#!/usr/bin/env bash
# Modern recipe (RoPE, RMSNorm, QK-norm, SwiGLU, Muon) on the mixed prose + code data, run strictly
# one job at a time: (1) learning-rate sweep, (2) pick each tokenizer's best LR, (3) 22,500-step runs.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}
T=results/tokenizers
COMMON="--arch modern --opt muon --head chain --order 1,2,0,3 --train_text data/mix_lm3x.txt --val_text data/mix.test.txt"

echo "== sweep $(date)"
for lr in 1e-3 2e-3 4e-3; do
  for tok in mix_32768_bpe_gpt2 mix_32768_comb; do
    WORKERS=6 $PY experiments/lm.py $T/$tok.json $COMMON --lr $lr --tag modern_muon_lr$lr \
      --steps 3000 --eval_every 1000 --out results/lm_sweep.json
  done
done

echo "== picking learning rates $(date)"
read LR_STD LR_COMB < <($PY - <<'EOF' | tr -d '\r'
import json
runs = json.load(open("results/lm_sweep.json"))
best = {}
for r in runs:
    lr, bpb = r["args"]["lr"], r["curve"][-1]["val_bpb"]
    k = r["tokenizer"]
    if k not in best or bpb < best[k][1]:
        best[k] = (lr, bpb)
print(best["mix_32768_bpe_gpt2.json"][0], best["mix_32768_comb.json"][0])
EOF
)
echo "best lr: standard $LR_STD, combinatorial $LR_COMB"

echo "== long runs $(date)"
V=$(ls data/val_mix_*.txt | tr '\n' ' ')
for pair in "mix_32768_bpe_gpt2:$LR_STD" "mix_32768_comb:$LR_COMB"; do
  tok=${pair%%:*}; lr=${pair##*:}
  WORKERS=6 $PY experiments/lm.py $T/$tok.json $COMMON --lr $lr --tag modern_muon_lr$lr \
    --steps 22500 --eval_every 1500 --extra_val $V --out results/lm_mix3x_modern.json
done
echo "== done $(date)"
