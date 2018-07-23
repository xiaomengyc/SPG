#!/bin/sh

cd ../exper/


#v5_v3

ROOT_DIR=/home/xiaolin/eccv18
CUDA_VISIBLE_DEVICES=1 python val_vgg_imagenet_single_img.py --arch=vgg_v5_4 \
	--batch_size=1 \
	--img_dir=${ROOT_DIR}/data/ILSVRC/Data/CLS-LOC/val \
	--train_list=${ROOT_DIR}/data/ILSVRC/list/train_list.txt \
	--test_list=${ROOT_DIR}/data/ILSVRC/list/val_list.txt \
	--num_gpu=1 \
	--dataset=imagenet \
	--num_classes=1000 \
	--snapshot_dir=../snapshots/vgg16_imagenet_full_v5_4/

#	--img_dir=${ROOT_DIR}/data/ILSVRC/Data/CLS-LOC/train \
