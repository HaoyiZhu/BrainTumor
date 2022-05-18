#!/bin/bash

for lr in {1e-4,1e-3,1e-2}
do
	for lr_cosine_min in {1e-7,1e-6,1e-5}
	do
		CUDA_VISIBLE_DEVICES=1 python scripts/train.py train.lr=$lr train.max_epochs=20 train.scheduler.lr_cosine_min=$lr_cosine_min train.scheduler.lr_cosine_warmup_epochs=0
	done
done