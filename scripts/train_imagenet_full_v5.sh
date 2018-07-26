#!/bin/sh

cd ../exper/


ROOT_DIR=/home/xiaolin/eccv18


CUDA_VISIBLE_DEVICES=0,1 python train_frame.py 
	--arch=inceptionv3_spg --epoch=5 --lr=0.001 --batch_size=30 
	--num_gpu=2 --dataset=imagenet  \
	--img_dir=${ROOT_DIR}/data/ILSVRC/Data/CLS-LOC/train \
	--train_list=${ROOT_DIR}/data/ILSVRC/list/train_list.txt \
	--num_classes=1000 \
	--snapshot_dir=../snapshots/inception_imagenet_full_spg/  \
	
	
	#--restore_from=/home/xiaolin/.torch/models/vgg16-397923af.pth \

