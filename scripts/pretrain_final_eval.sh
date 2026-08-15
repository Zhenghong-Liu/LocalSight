#!/usr/bin/env bash
# pretrain 收尾评测：对 model soup 权重跑 MMLU/C-Eval/GSM8K/NIAH/IFEval。
set -euo pipefail
cd ~/project/LocalSight
CKPT=${1:-artifacts/pretrain/soup}
export PYTHONPATH=src

.venv/bin/python scripts/run_evals.py --checkpoint "$CKPT" \
  --bench mmlu,ceval,gsm8k --limit 100 2>&1 | tee artifacts/pretrain_final_evals.log
.venv/bin/python scripts/run_niah.py --checkpoint "$CKPT" 2>&1 | tee -a artifacts/pretrain_final_evals.log
.venv/bin/python scripts/run_ifeval.py --checkpoint "$CKPT" 2>&1 | tee -a artifacts/pretrain_final_evals.log
echo "FINAL_EVAL_DONE"
