#!/bin/bash
master_port=12354
ngpu=4
fp16=True
exp_dir=w2v2/exp_mms_dev_p1
pooling_type=post
modelpath="facebook/mms-300m" # mshwiesner/speech-geolocation-300m"
narrowband=False
train_cuts=data/manifests/lre_dev_p1.jsonl.gz
valid_cuts=data/manifests/lre_dev_p2.jsonl.gz
freeze_feat_extractor="True"

. ./shared/parse_options.sh

. /expscratch/mwiesner/geolocation/activate_python.sh
module load ffmpeg

python w2v2/train_mean_prop.py \
  --train-cuts ${train_cuts} \
  --valid-cuts ${valid_cuts} \
  --world-size 4 \
  --exp-dir "${exp_dir}" \
  --lr 1e-05 \
  --freeze-lr 1e-06 \
  --total-steps 60000 \
  --freeze-iters 500 \
  --modelpath $modelpath \
  --freeze-feat-extractor ${freeze_feat_extractor} \
  --pooling-type ${pooling_type} \
  --cache-dir /expscratch/mwiesner/testthis \
  --num-buckets 30 \
  --num-workers 8 \
  --use-fp16 True \
  --narrowband ${narrowband} \
  --max-duration 400
