#!/bin/bash

for lr in {1e-4,5e-3,5e-5,1e-3}
do
	for weight_decay in {1e-2,1e-4,0,1e-6}
	do
		CUDA_VISIBLE_DEVICES=1,2,3,4 python scripts/train.py train.lr=$lr train.max_epochs=20 train.weight_decay=$weight_decay train.scheduler.lr_cosine_warmup_epochs=0 train.devices=4 train.scheduler.lr_cosine_epochs=100
	done
done

# for lr in {2e-4,3e-4,4e-4,5e-4,6e-4,7e-4,8e-4,9e-4,5e-6,1e-5,3e-5,5e-5,7e-5,2e-3,3e-3,5e-3}
# do
#	CUDA_VISIBLE_DEVICES=1,2,3,4 python scripts/train.py train.lr=$lr train.max_epochs=15 train.scheduler.lr_cosine_warmup_epochs=0 train.devices=4 train.scheduler.lr_cosine_epochs=100
# done