#!/bin/bash

world_size=4
master_port=12355
modelpath=mshwiesner/speech-geolocation-300m
expname=exp
lr=0.045

. ./shared/parse_options.sh

. /expscratch/mwiesner/geolocation/activate_python.sh
module load ffmpeg

python w2v2_zipformer/train.py \
  --master-port ${master_port} \
  --world-size ${world_size} \
  --num-workers 12 \
  --base-lr ${lr} \
  --use-fp16 True \
  --exp-dir w2v2_zipformer/${expname} \
  --max-duration 400 \
  --modelpath ${modelpath} \
  --bpe-model data/lang_bpe_10000/bpe.model \
  --input-strategy AudioSamples \
  --enable-musan False \
  --enable-spec-aug False
