#!/bin/bash

stage=0
iter=84000
avg=1
expdir=wavlm_spashl_ctc/exp3spk_1gpu_config14
langdir=data/lang_bpe_5000
bw=0.0
spkw=1.0
max_num_spks=4
use_large=True
downsample=False
test_set="synth2"
sort_strategy="speech_time"
max_duration=400
collar=32000
oracle_num_spks=
ignore_speaker=False
search_beam=32
lattice_beam=10
genjiko=
extra_suffix=
ref=data/manifests2/alignments_dev_synth_2spk_8splices

. ./shared/parse_options.sh

ref=${ref}_${sort_strategy}.json

suffix=bw${bw}_spkw${spkw}_lbeam${lattice_beam}_sbeam${search_beam}
if [ $ignore_speaker == "True" ]; then
  suffix=${suffix}_nospk
  if [[ ! -z $genjiko ]]; then
    suffix=${suffix}_genjiko${genjiko}
  fi
fi


if [ $downsample == "False" ]; then
  frame_duration=0.02
else
  frame_duration=0.04
fi 

if [[ -z $oracle_num_spks ]]; then
  oracle_num_spks=${max_num_spks}
fi

suffix=${suffix}_nspks${oracle_num_spks}

suffix=${suffix}${extra_suffix}

if [ $stage -le 0 ]; then
  if [[ $ignore_speaker == "True" ]] && [[ ! -z $genjiko ]]; then
    echo "genjiko ${genjiko}"
    python wavlm_spashl_ctc/align_genjiko3.py \
      --iter ${iter} \
      --avg ${avg} \
      --use-averaged-model False \
      --exp-dir ${expdir} \
      --lang-dir data/lang_bpe_5000 \
      --suffix ${suffix} \
      --blank-weight ${bw} \
      --max-num-spks ${max_num_spks} \
      --use-hat True \
      --use-layer-norm False \
      --use-large ${use_large} \
      --downsample ${downsample} \
      --test-sets "${test_set}" \
      --sort-strategy "${sort_strategy}" \
      --frame-duration ${frame_duration} \
      --num-workers 5 \
      --num-buckets 100 \
      --max-duration ${max_duration} \
      --speaker-weight ${spkw} \
      --normalize-loudness False \
      --collar ${collar} \
      --total-max-overlaps 10000000 \
      --beam-size ${lattice_beam} \
      --search-beam ${search_beam} \
      --genjiko ${genjiko} \
      --oracle-num-spks ${oracle_num_spks}
  else
    python wavlm_spashl_ctc/align_tokens.py \
      --iter ${iter} \
      --avg ${avg} \
      --use-averaged-model False \
      --exp-dir ${expdir} \
      --lang-dir data/lang_bpe_5000 \
      --suffix ${suffix} \
      --blank-weight ${bw} \
      --max-num-spks ${max_num_spks} \
      --use-hat True \
      --use-layer-norm False \
      --use-large ${use_large} \
      --downsample ${downsample} \
      --test-sets "${test_set}" \
      --sort-strategy "${sort_strategy}" \
      --frame-duration ${frame_duration} \
      --num-workers 5 \
      --num-buckets 100 \
      --max-duration ${max_duration} \
      --speaker-weight ${spkw} \
      --normalize-loudness False \
      --collar ${collar} \
      --total-max-overlaps 10000000 \
      --beam-size 40 \
      --ignore-speaker ${ignore_speaker} \
      --oracle-num-spks ${oracle_num_spks}
  fi
fi

if [ $stage -le 1 ]; then
  alidir=${expdir}/align
  hyp_name=alignments_chkpt${iter}_avg${avg}_${test_set}_collar${collar}_${suffix}.json
  word_hyp_name=word_${hyp_name}
  python local/ali_to_word_ali.py ${alidir}/${hyp_name}
  
  python local/score_alignments.py --relabel --calibrate \
    -h ${alidir}/${word_hyp_name} \
    -r ${ref} 
fi 
