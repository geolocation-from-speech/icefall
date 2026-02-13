#!/bin/bash

num_gpu=1
exp_name=exp3spk
max_duration=320
lr=0.0001
freeze_iters=2000
freeze_lr=5e-03
pct_start=0.08
collar=64000
total_max_overlaps=4000
total_steps=100000
master_port=12354
hat=True
large=False
downsample=True
langdir=data/lang_bpe_5000
original_topo=False
beam_size=10
sort_strategy=start_time
num_spks=3
layer_norm=False
max_splices=4
max_splice_duration=30
max_unique=3
overlap=0.4
drift=1.0
max_snr=30.0
reverb=True
normalize_loudness=False
duration_increment=1e-03
config=

. ./shared/parse_options.sh

if [[ ! -z $config ]]; then
  . ./${config}
fi

. /ocean/projects/cis210027p/mwiesner/jsalt2025/activate_python.sh

python wavlm_shuffle_ctc/train.py \
  --lang-dir ${langdir} \
  --exp-dir wavlm_shuffle_ctc/${exp_name} --num-workers 5 \
  --world-size ${num_gpu} --use-fp16 True \
  --max-duration ${max_duration} \
  --base-lr ${lr} \
  --pct-start ${pct_start} \
  --collar ${collar} \
  --freeze-lr ${freeze_lr} \
  --freeze-iters ${freeze_iters} \
  --max-num-spks ${num_spks} \
  --max-overlap 3 \
  --overlap ${overlap} \
  --min-overlap ${min_overlap} \
  --drift ${drift} \
  --normalize-loudness ${normalize_loudness} \
  --reverb ${reverb} \
  --max-snr ${max_snr} \
  --total-max-overlaps ${total_max_overlaps} \
  --total-steps ${total_steps} \
  --use-layer-norm ${layer_norm} \
  --use-large ${large} \
  --downsample ${downsample} \
  --original-topo ${original_topo} \
  --master-port ${master_port} \
  --beam-size ${beam_size} \
  --sort-strategy ${sort_strategy} \
  --max-splices ${max_splices} \
  --max-splice-duration ${max_splice_duration} \
  --duration-increment ${duration_increment} \
  --max-unique ${max_unique} \
  --allow-self-overlap ${allow_self_overlap}


