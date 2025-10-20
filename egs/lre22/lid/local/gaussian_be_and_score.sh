#!/bin/bash

embeds_dir=
backend_data=
test_data=
pca_dim=700
normalize=True
stage=1
stop_stage=10
out_scores=results
cov_type="global"
config="config_eval.ini"

. ./shared/parse_options.sh

#if [[ $# -eq 0 ]]; then
#  echo "Submit with arguments. Usage: ./local/gaussian_be_and_score.sh --embed-dir <embed_dir> --backend-data <backend_data_suffix> --test-data <test_data_suffix>"
#  echo ""
#  echo "eg: ./local/gaussian_be_and_score.sh w2v2/exp_geo_fleurs_prepool/embeds --backend-data lre22_dev_p1 --test-data data/manifests/lre_dev_p2"
#  exit 1
#fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ] ; then
  echo "------------------- Training Gaussian Backend ------------------"
  backend_name=gaussian_be_${backend_data}_norm${normalize}_pcadim${pca_dim}_cov${cov_type}
  python local/gaussian_backend.py \
    --norm ${normalize} \
    --cov-type ${cov_type} \
    --embeds ${embeds_dir}/embeds_${backend_data}.npy \
    --tgts ${embeds_dir}/tgts_${backend_data}.npy \
    --pca-dim ${pca_dim} \
    --out ${embeds_dir}/${backend_name}
fi


if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "------------------- Applying Gaussian Backend ------------------"
  backend_name=gaussian_be_${backend_data}_norm${normalize}_pcadim${pca_dim}_cov${cov_type}
  #python local/apply_gaussian_backend.py \
  #  --embeds ${embeds_dir}/embeds_${test_data}.npy \
  #  --backend ${embeds_dir}/${backend_name}.npz \
  #  --out ${embeds_dir}/scores_${backend_name}_${test_data}.npy
  python local/apply_gaussian_backend.py \
    --test-mean \
    --embeds ${embeds_dir}/embeds_${test_data}.npy \
    --backend ${embeds_dir}/${backend_name}.npz \
    --out ${embeds_dir}/scores_${backend_name}_${test_data}.npy

fi


if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo "------------------- Converting scores to NIST format ------------------"
  backend_name=gaussian_be_${backend_data}_norm${normalize}_pcadim${pca_dim}_cov${cov_type}
  python local/numpy_to_nist.py \
    --numpy-scores ${embeds_dir}/scores_${backend_name}_${test_data}.npy \
    --numpy-ids ${embeds_dir}/ids_${test_data}.npy \
    --nist ${embeds_dir}/scores_${backend_name}_${test_data}.tsv
fi


if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  echo "------------------- Scoring ------------------"
  backend_name=gaussian_be_${backend_data}_norm${normalize}_pcadim${pca_dim}_cov${cov_type}
  #python local/score.py \
  #  --skip-cpmin \
  #  --scores ${embeds_dir}/scores_${backend_name}_${test_data}.npy \
  #  --tgts ${embeds_dir}/tgts_${test_data}.npy \
  #  --ids ${embeds_dir}/ids_${test_data}.npy \
  #  --nist ${embeds_dir}/scores_${backend_name}_${test_data}.tsv
  python local/score.py \
    --skip-cpmin \
    --scores ${embeds_dir}/scores_${backend_name}_${test_data}.npy \
    --tgts ${embeds_dir}/tgts_${test_data}.npy \
    --nist ${embeds_dir}/scores_${backend_name}_${test_data}.tsv

fi

exit
if [ $stage -le 5 ]; then
  backend_name=gaussian_be_${backend_data}_norm${normalize}_pcadim${pca_dim}
  embeds_name=`dirname ${embeds_dir}`
  embeds_name=`basename ${embeds_name}`
  echo "---------------- Official Scoring ------------"
  cd lre-scorer
  python scoreit.py -e \
    -s ../${embeds_dir}/scores_${backend_name}_${test_data}.tsv \
    -o ../${results}/${embeds_name}_${backend_name}_${test_data} \
    ${config}
  cd -
fi
