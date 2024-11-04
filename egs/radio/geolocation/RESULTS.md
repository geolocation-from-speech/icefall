## Results

### Wav2Vec2.0 MMS-300M parameter model Angular Distance Regression Loss

The following model achieves an average distance error of 867 km on
an 11-language subset of FLEURS languages as reported in 
<https://aclanthology.org/2024.naacl-long.286.pdf>

python w2v2_angular_distance/train.py \
  --world-size 4 \
  --max-duration 400 \
  --exp-dir w2v2_angular_distance/exp_5 \
  --num-buckets 30 \
  --use-feats False \
  --use-fp16 True \
  --lr 1e-06 \
  --pct-start 0.02 \
  --weight-decay 1e-08 \
  --cuts data/manifests/radio_cuts_5_shuf.jsonl.gz \
  --num-epochs 70 \
  --total-steps 800000 \
  --freeze-iters 1000 \
  --freeze-lr 0.000001 \
  --modelpath facebook/mms-300m \
  --pooling-type att \
  --pooling-loc 0 \
  --pooling-heads 1 \
  --freeze-feat-extractor True


Some other results are reported below:
Some different model configs:
| Name |Pretrained model | # Pooling Heads | Max Chunk Width | lr | pct_start | total_steps | weight_decay | freeze_lr | freeze_iters |
|------|-----------------|-----------------|-----------------|----|-----------|-------------|--------------|-----------|--------------|
| Rand |  -              | - | -     | -     | -    | -      | -  |     -    | -   |
| V0   | MMS-300M        | 1 | 5 sec | 3e-06 | 0.08 | 800000 | 1e-06 | 1e-05 | 1000| 
| V1   | MMS-300M        | 1 | 10 sec | 3e-06 | 0.08 | 800000 | 1e-06 | 1e-05 | 1000| 
| V2   | MMS-300M        | 4 | 5 sec | 1e-06 | 0.02 | 800000 | 1e-08 | 1e-06 | 1000| 
| V3   | MMS-300M        | 1 | 5 sec | 1e-06 | 0.02 | 800000 | 1e-08 | 1e-06 | 1000| 


|Model | Checkpoint | FLEURS 11 DEV Avg Distance | FLEURS 11 TEST Avg Distance |
|------|------------|----------------------------|-----------------------------|
| Rand |     -      |   8624 km                  |                             |
| V0   |   260000   |   1012 km                  |                             |
| V1   |   168000   |   941.8 km                 |                             |
| V2   |   216000   |   966.5 km                 |                             |
| V2   |   404000   |   924.4 km                 |                             |
| V2   |   496000   |   904.2 km                 |                             |
| V3   |   404000   |   867.1 km                 |  844.1 km                   |
