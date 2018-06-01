#!/usr/bin/env bash
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

DATA_DIR=/home/data/insight/insight/faces_ms1m_112x112

NETWORK=r50
PREFIX_DIR=../models/am_$NETWORK
if [ ! -d $PREFIX_DIR ]; then
    mkdir -p $PREFIX_DIR
fi

# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' \
CUDA_VISIBLE_DEVICES='0,1,2,3' \
    python -u train_softmax.py \
	--network "$NETWORK" \
	--loss-type 4 \
	--margin-m 0.5 \
    --data-dir $DATA_DIR \
	--prefix $PREFIX_DIR/model \
	--per-batch-size 110 \
	--lr 0.1 \
    --lr-steps '100000,160000,180000,200000' \
	--emb-size 256 \
	2>&1 | tee $PREFIX_DIR/train.log

