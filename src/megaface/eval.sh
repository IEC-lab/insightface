#!/bin/bash

Megaface_dataset=/home/data/insight/megaface/megaface/daniel/FlickrFinal2
Facescrub_dataset=/home/data/insight/megaface/facescrub
Align_out=aligned
Align_megaface=$Align_out/megaface
Align_facescrub=$Align_out/facescrub

# align datasets
if [ ! -d $Align_megaface ]; then
  python ../align/align_megaface.py --input-dir $Megaface_dataset --name megaface --output-dir $Align_megaface
fi
if [ ! -d $Align_facescrub ]; then
  python ../align/align_facescrub.py --input-dir $Facescrub_dataset --output-dir $Align_facescrub
fi

Algo='amr34'

# generate feature files
if [ ! -d ${Align_megaface}_features ]; then
  python gen_megaface.py \
    --batch_size 500 \
    --image_size '3,112,112' \
    --gpus '0' \
    --mean 1 --seed 727 \
    --fsall 1 --mf 1 \
    --algo $Algo \
    --megaface_lst $Align_megaface/lst \
    --megaface_out ${Align_megaface}_features \
    --facescrub_lst $Align_facescrub/lst \
    --facescrub_out ${Align_facescrub}_features \
    --model '../../models/emore/am_r34_1024_tune/model,33'
fi

# remove noise
if [ ! -d ${Align_megaface}_features_cm ]; then
  python -u remove_noises.py \
    --facescrub-noises ./facescrub_noises.txt \
    --megaface-noises ./megaface_noises.txt \
    --suffix ${Algo}_112x112 \
    --algo ${Algo}_112x112 \
    --megaface-lst $Align_megaface/lst \
    --facescrub-lst $Align_facescrub/lst \
    --megaface-feature-dir ${Align_megaface}_features \
    --facescrub-feature-dir ${Align_facescrub}_features \
    --megaface-feature-dir-out ${Align_megaface}_features_cm \
    --facescrub-feature-dir-out ${Align_facescrub}_features_cm \
    --feature_dim 256
fi

# run devkit
if [ ! -d ${Align_out}/result ]; then
  cd devkit/experiments
  python run_experiment.py \
    ../../${Align_megaface}_features_cm \
    ../../${Align_facescrub}_features_cm \
    _${Algo}_112x112.bin \
    ../../${Align_out}/result
fi
  
