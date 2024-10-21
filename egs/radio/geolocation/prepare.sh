#!/usr/bin/env bash
# Copyright 2023 Johns Hopkins University  (Matthew Wiesner)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# fix segmentation fault reported in https://github.com/k2-fsa/icefall/issues/674
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

set -eou pipefail

use_musan=false
corpus=
fleurs=/export/common/data/corpora/fleurs
stage=1
stop_stage=1

. shared/parse_options.sh || exit 1

dl_dir=$PWD/corpus

# The cuts will be stored here.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Download data"
  # If you do not have musan, then download it
  if $use_musan; then
    if [ ! -d ${dl_dir}/musan ]; then
      lhotse download musan $dl_dir
    fi
  fi

  # Check if the radio corpus is available. This could be modified to any
  # data that has a geolocation (lat, lon). Just put them in the custom field
  # i.e., supervisions[0].custom['lat'], supervisions[0].custom['lon'] and
  # it should work. 
  if [ ! -d ${corpus}/recos/recos.1 ]; then
    log "${corpus} not found. To obtain access, email wiesner@jhu.edu.\n The" \
    "release of the data, which is clips of radio broadcasts, is subject to" \
    "various copyright laws to US export control.\n" \
    "At this time it is for strictly non-commerical, academic use.\n"\
    "Please include your academic institution and brief description of its use"\
    "to obtain the data."
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  manifests=data/manifests
  mkdir -p ${manifests}
  lhotse prepare radio -j 8 ${corpus} ${manifests}
  
  ./local/prepare_cuts.py \
    --recordings ${manifests}/radio_recordings.jsonl.gz \
    --supervisions ${manifests}/radio_supervisions.jsonl.gz \
    ${manifests}/radio_cuts2.jsonl.gz

  ./local/prepare_fleurs.py \
    data/manifests/fleurs_dev ${fleurs}/*_*/dev

  ./local/prepare_fleurs.py \
    data/manifests/fleurs_test ${fleurs}/*_*/test

fi
