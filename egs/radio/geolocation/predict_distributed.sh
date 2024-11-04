#!/bin/bash
cuts=( "$@" )
index=$(($SGE_TASK_ID - 1))
name=`basename ${recosdirs[$index]}`
suffix="radio_all_${index}"

python w2v2_angular_distance/predict.py \
  --world-size 1 \
  --exp-dir w2v2_angular_distance/exp_10_big \
  --use-fp16 True \
  --suffix "${suffix}" \
  --valid-cuts ${cuts[$index]} \
  --checkpoint checkpoint-230000
