#!/bin/bash
cd ..
nohup python -u train.py \
--cuda 1 \
--batch_size 32 \
--seed 63 \
--n_splits 10 \
--n_repeats 10 \
--lr 1e-4 \
--weight_decay 1e-4 \
--max_epochs 200 \
> ./log/terminal/myrecord.log 2>&1 &