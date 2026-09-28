#!/usr/bin/env bash
# 128k unrestricted Combinatorial BPE vs standard BPE trained on the same text (320M characters
# per source, curated data), same modern recipe as run_modern.sh. Strictly one job at a time:
# (1) short learning-rate sweep, (2) each tokenizer's best LR, (3) 15,000-step runs.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}
T=results/tokenizers
D=data/curated
COMB=mix_131072_dict_search_hybrid_caff_d320
STD=mix_131072_bpe_gpt2_d320
COMMON="--arch modern --opt muon --train_text $D/mix_lm3x.txt --val_text $D/mix.test.txt --accum 4"
HEAD_COMB="--head chain --order 1,2,0,3"
export DS_THREADS=${DS_THREADS:-2}

echo "== sweep $(date)"
for lr in 4e-3 8e-3; do
  WORKERS=6 $PY experiments/lm.py $T/$STD.json $COMMON --lr $lr --tag lr$lr --steps 1500 --eval_every 500 --out results/lm_128k_sweep.json
  WORKERS=6 $PY experiments/lm.py $T/$COMB.json $COMMON $HEAD_COMB --lr $lr --tag lr$lr --steps 1500 --eval_every 500 --out results/lm_128k_sweep.json
done

echo "== picking learning rates $(date)"
read LR_STD LR_COMB < <($PY - <<EOF | tr -d '\r'
import json
runs = json.load(open("results/lm_128k_sweep.json"))
best = {}
for r in runs:
    lr, bpb = r["args"]["lr"], r["curve"][-1]["val_bpb"]
    k = r["tokenizer"]
    if k not in best or bpb < best[k][1]:
        best[k] = (lr, bpb)
print(best["$STD.json"][0], best["$COMB.json"][0])
EOF
)
echo "best lr: standard $LR_STD, combinatorial $LR_COMB"

echo "== long runs $(date)"
V=$(ls $D/val_mix_*.txt | tr '\n' ' ')
WORKERS=6 $PY experiments/lm.py $T/$STD.json $COMMON --lr $LR_STD --tag long_lr$LR_STD \
  --steps 15000 --eval_every 1500 --extra_val $V --out results/lm_128k.json
WORKERS=6 $PY experiments/lm.py $T/$COMB.json $COMMON $HEAD_COMB --lr $LR_COMB --tag long_lr$LR_COMB \
  --steps 15000 --eval_every 1500 --extra_val $V --out results/lm_128k.json
echo "== done $(date)"
