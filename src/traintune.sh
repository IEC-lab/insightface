#!/usr/bin/env bash
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

DATA_DIR=/home/data/insight/insight/faces_emore

NETWORK=r50
PREFIX_DIR=../models/emore/am_${NETWORK}_1024_tune
if [ ! -d $PREFIX_DIR ]; then
    mkdir -p $PREFIX_DIR
fi

CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' \
    python -u train_softmax.py \
	--network "$NETWORK" \
    --pretrained '../models/am_r50/model,115' \
	--loss-type 4 \
	--margin-m 0.5 \
	--data-dir $DATA_DIR \
	--prefix $PREFIX_DIR/model \
	--per-batch-size 128 \
    --target 'lfw,cfp_fp,cfp_ff,agedb_30,vgg2_fp,cplfw,calfw' \
	--lr 0.01 \
	--lr-steps '20000,40000,60000,80000,100000' \
	--emb-size 256 \
	2>&1 | tee $PREFIX_DIR/train.log

