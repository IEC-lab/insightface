#!/usr/bin/env bash
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
# export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice

DATA_DIR=/home/data/insight/insight/faces_emore

NETWORK=y1


PREFIX_DIR=../models/emore/$NETWORK/softmax
if [ ! -d $PREFIX_DIR ]; then
    mkdir -p $PREFIX_DIR
fi
## Train with softmax
CUDA_VISIBLE_DEVICES='0,1,2,3' \
    python -u train_softmax.py \
        --network $NETWORK \
        --loss-type 0 \
        --ckpt 2 \
        --data-dir $DATA_DIR \
        --prefix $PREFIX_DIR/model \
        --per-batch-size 250 \
        --target 'lfw,cfp_fp,cfp_ff,agedb_30,vgg2_fp' \
        --lr 0.1 \
        --lr-steps '30000,40000,50000' \
        --max-steps 60001 \
        --emb-size 128 \
        --fc7-wd-mult 10.0 \
        --wd 0.00004 \
        2>&1 | tee $PREFIX_DIR/train.log


## Finetune with Arcface 0.3
MarginM=0.3
PREFIX_DIR=../models/emore/$NETWORK/$MarginM
if [ ! -d $PREFIX_DIR ]; then
    mkdir -p $PREFIX_DIR
fi
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' \
    python -u train_softmax.py \
        --network $NETWORK \
        --loss-type 4 \
        --ckpt 2 \
        --margin-m $MarginM \
        --data-dir $DATA_DIR \
        --prefix $PREFIX_DIR/model \
        --per-batch-size 128 \
        --target 'lfw,cfp_fp,cfp_ff,agedb_30,vgg2_fp' \
        --lr 0.1 \
        --lr-steps '60000,80000,90000' \
        --max-steps 100001 \
        --emb-size 128 \
        --fc7-wd-mult 10.0 \
        --wd 0.00004 \
        --pretrained "../models/emore/$NETWORK/softmax/model,30" \
        2>&1 | tee $PREFIX_DIR/train.log


# ## Finetune with Arcface 0.5
# MarginM=0.5
# PREFIX_DIR=../models/emore/$NETWORK/$MarginM
# if [ ! -d $PREFIX_DIR ]; then
#     mkdir -p $PREFIX_DIR
# fi
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' \
#     python -u train_softmax.py \
#         --network $NETWORK \
#         --loss-type 4 \
#         --margin-m $MarginM \
#         --data-dir $DATA_DIR \
#         --prefix $PREFIX_DIR/model \
#         --per-batch-size 128 \
#         --target 'lfw,cfp_fp,cfp_ff,agedb_30,vgg2_fp' \
#         --lr 0.1 \
#         --lr-steps '90000,120000,140000,150000' \
#         --max-steps 160001 \
#         --emb-size 128 \
#         --fc7-wd-mult 10.0 \
#         --wd 0.00004 \
#         --pretrained "../models/emore/$NETWORK/0.3/model,40" \
#         2>&1 | tee $PREFIX_DIR/train.log


# ## Finetune with Triplet Loss
# PREFIX_DIR=../models/emore/$NETWORK/triplet
# if [ ! -d $PREFIX_DIR ]; then
#     mkdir -p $PREFIX_DIR
# fi
# CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' \
#     python -u train_softmax.py \
#         --network $NETWORK \
#         --loss-type 12 \
#         --data-dir $DATA_DIR \
#         --prefix $PREFIX_DIR/model \
#         --per-batch-size 150 \
#         --target 'lfw,cfp_fp,cfp_ff,agedb_30,vgg2_fp' \
#         --lr 0.005 \
#         --mom 0.0 \
#         --lr-steps '90000,120000,140000,150000' \
#         --max-steps 160001 \
#         --emb-size 128 \
#         --fc7-wd-mult 10.0 \
#         --wd 0.00004 \
#         --pretrained "../models/emore/$NETWORK/0.5/model,50" \
#         2>&1 | tee $PREFIX_DIR/train.log

