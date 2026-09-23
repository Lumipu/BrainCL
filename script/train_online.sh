#!/bin/bash
cd ..
python train.py \
--cuda 1 \
--batch_size 32 \
--seed 63 \
--n_splits 10 \
--n_repeats 10 \
--lr 1e-4 \
--weight_decay 1e-4 \
--max_epochs 200