#!/bin/bash
master_port=12354
ngpu=8
master_port=12354
fp16=True
exp_dir=w2v2_angular_distance/exp
max_duration=900
valid_cuts=data/manifests/radio_cuts_all.jsonl.gz
num_workers=12
checkpoint=200000
suffix="cuts_all_200k"

. ./shared/parse_options.sh

. /expscratch/mwiesner/geolocation/activate_python.sh

module load ffmpeg

export PYTHONUNBUFFERED=1
python w2v2_angular_distance/predict.py \
  --master-port ${master_port} \
  --world-size ${ngpu} \
  --max-duration ${max_duration} \
  --exp-dir "${exp_dir}" \
  --num-buckets 100 \
  --num-workers 12 \
  --use-fp16 True \
  --valid-cuts ${valid_cuts} \
  --suffix "${suffix}" \
  --checkpoint checkpoint-${checkpoint}
