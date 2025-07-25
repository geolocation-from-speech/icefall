#!/bin/bash

num_gpu=2
exp_name=exp8spk
max_duration=350
lr=0.00005
collar=48000

. ./shared/parse_options.sh


. /ocean/projects/cis210027p/mwiesner/jsalt2025/activate_python.sh

python wavlm_spashl/train.py \
  --lang-dir data/lang_bpe_5000 \
  --exp-dir wavlm_spashl/${exp_name} --num-workers 14 \
  --world-size ${num_gpu} --use-fp16 True \
  --max-duration ${max_duration} \
  --base-lr ${lr} \
  --collar ${collar} \
  --freeze-lr 0.005 \
  --freeze-iters 2000 \
  --max-num-spks 8 \
  --use-hat True \
  --use-layer-norm False \
  --use-large True

