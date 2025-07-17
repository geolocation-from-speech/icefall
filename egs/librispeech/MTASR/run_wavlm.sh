#!/bin/bash

. /ocean/projects/cis210027p/mwiesner/jsalt2025/activate_python.sh

python wavlm_mdctc/train.py \
  --lang-dir data/lang_bpe_5000 \
  --exp-dir wavlm_mdctc/exp1 \
  --num-workers 14 \
  --world-size 2 \
  --use-fp16 True \
  --max-duration 450 \
  --base-lr 0.0001 \
  --collar 64000 \
  --freeze-lr 0.005 \
  --freeze-iters 2000
