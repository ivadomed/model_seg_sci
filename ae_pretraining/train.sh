#!/bin/bash

# run commands in parallel; '&' runs the command in background
# 1. to see the effect of learning rate
CUDA_VISIBLE_DEVICES=0 python pretraining.py -bnf 8 -ne 500 -bs 4 -lr 1e-4 -se 42 -mn unet3d &
# 2. to see the effect of subvolume and stride size with the same learning rate as above
CUDA_VISIBLE_DEVICES=1 python pretraining.py -bnf 8 -svs 64 128 48 -srs 64 128 48 -ne 500 -bs 4 -lr 1e-4 -se 42 -mn unet3d &
CUDA_VISIBLE_DEVICES=2 python pretraining.py -bnf 8 -svs 32 64 48 -srs 32 64 48 -ne 500 -bs 4 -lr 1e-4 -se 42 -mn unet3d &
# 3. 
# CUDA_VISIBLE_DEVICES=3 python pretraining.py -bnf 8 -svs 64 128 48 -srs 64 128 48 -ne 500 -bs 16 -lr 5e-4 -se 42 -mn unet3d &
