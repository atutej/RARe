#!/bin/bash

conda activate ret
torchrun --nnodes 1 --nproc_per_node 8 experiments/run_supervised.py train_configs/supervised/E5-Instruct.json