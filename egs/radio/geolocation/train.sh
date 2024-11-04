#!/bin/bash
master_port=12354
ngpu=4
master_port=12354
fp16=True
exp_dir=w2v2_angular_distance/exp
max_duration=400
lr=3e-06
pct_start=0.08
use_feats="False"
pooling_type=att # max
pooling_loc=0
freeze_iters=1000
freeze_lr=0.00001
cuts=data/manifests/radio_cuts_10_shuf.jsonl.gz
modelpath="facebook/mms-300m"
freeze_feat_extractor=True
num_workers=8
weight_decay=1e-06
pooling_heads=1

. ./shared/parse_options.sh

. /expscratch/mwiesner/geolocation/activate_python.sh
module load ffmpeg

mkdir -p ${exp_name} 

export PYTHONUNBUFFERED=1
python w2v2_angular_distance/train.py \
  --master-port ${master_port} \
  --world-size ${ngpu} \
  --num-workers ${num_workers} \
  --max-duration ${max_duration} \
  --exp-dir "${exp_dir}" \
  --num-buckets 30 \
  --num-workers ${num_workers} \
  --use-feats False \
  --use-fp16 "${fp16}" \
  --lr ${lr} \
  --pct-start ${pct_start} \
  --weight-decay ${weight_decay} \
  --cuts ${cuts} \
  --num-epochs 70 \
  --total-steps 800000 \
  --freeze-iters ${freeze_iters} \
  --freeze-lr ${freeze_lr} \
  --modelpath ${modelpath} \
  --pooling-type ${pooling_type} \
  --pooling-loc ${pooling_loc} \
  --pooling-heads ${pooling_heads} \
  --freeze-feat-extractor ${freeze_feat_extractor} \

