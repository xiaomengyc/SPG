#!/bin/sh

cd ../exper/


#v5_v3

ROOT_DIR=/home/xiaolin/eccv18

CUDA_VISIBLE_DEVICES=1 python val_frame.py --arch=inceptionv3_spg \
	--batch_size=1 \
	--img_dir=${ROOT_DIR}/data/ILSVRC/Data/CLS-LOC/val \
	--train_list=${ROOT_DIR}/data/ILSVRC/list/train_list.txt \
	--test_list=${ROOT_DIR}/data/ILSVRC/list/val_list.txt \
	--num_gpu=1 \
	--dataset=imagenet \
	--num_classes=1000 \
	--snapshot_dir=../snapshots/inception_imagenet_full_spg/

