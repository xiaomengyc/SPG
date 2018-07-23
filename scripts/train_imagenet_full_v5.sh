#!/bin/sh

cd ../exper/

#CUDA_VISIBLE_DEVICES=0 python train_vgg_imagenet_new_frame.py --arch=vgg_v5 --epoch=26 --lr=0.001 --batch_size=20 --num_gpu=1 --dataset=imagenet  \
#	--snapshot_dir=../snapshots/vgg16_imagenet_v5/  \
#	--restore_from=/home/xiaolin/.torch/models/vgg16-397923af.pth \

ROOT_DIR=/home/xiaolin/eccv18

#CUDA_VISIBLE_DEVICES=0,1 python train_vgg_imagenet_new_frame.py --arch=vgg_v5_3_1 --epoch=26 --lr=0.001 --batch_size=20 --num_gpu=2 --dataset=imagenet  \
CUDA_VISIBLE_DEVICES=0,1 python train_vgg_imagenet_new_frame.py --arch=vgg_v5_4 --epoch=26 --lr=0.0001 --batch_size=20 --num_gpu=2 --dataset=imagenet  \
	--img_dir=${ROOT_DIR}/data/ILSVRC/Data/CLS-LOC/train \
	--train_list=${ROOT_DIR}/data/ILSVRC/list/train_list.txt \
	--num_classes=1000 \
	--snapshot_dir=../snapshots/vgg16_imagenet_full_v5_4/  \
	#--restore_from=/home/xiaolin/.torch/models/vgg16-397923af.pth \

	#--snapshot_dir=../snapshots/vgg16_imagenet_full_v5_4/  \
