#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Roller-py37

# Setup GV100 GPU
export CUDA_VISIBLE_DEVICES=4

# Run benchmark (run this script in the root directory)
python -u test_op.py --code_dir generated_source/conv --smem_tiling --reg_tiling --op fused_conv_expr_S1D1P1 --shape 128 128 28 28 128 3 3
python -u test_op.py --code_dir generated_source/matmul --smem_tiling --reg_tiling --op matmul_expr --shape 4096 4096 4096
