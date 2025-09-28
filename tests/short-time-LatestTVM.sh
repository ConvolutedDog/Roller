#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tvm-latest

# Function to run benchmarks on a specific GPU
run_benchmarks() {
    local gpu_id=$1
    local device_name=$2
    
    echo "========================================"
    echo "Running benchmarks on $device_name (CUDA_VISIBLE_DEVICES=$gpu_id)"
    echo "========================================"
    
    export CUDA_VISIBLE_DEVICES=$gpu_id
    
    # Run conv benchmark
    echo "Running fused_conv_expr_S1D1P1 (float16)..."
    python -u test_op.py --gen_check_code --backend tvm --topk 1 --code_dir generated_source/conv --smem_tiling --reg_tiling --codegen_input_reg_tiling --shared_fetch_vectorize --data_type "float16" --op fused_conv_expr_S1D1P1 --shape 128 128 28 28 128 3 3
    
    # Run matmul benchmarks
    echo "Running matmul_expr (float16) with tensor cores..."
    python -u test_op.py --gen_check_code --backend tvm --topk 1 --code_dir generated_source/matmul --smem_tiling --reg_tiling --codegen_input_reg_tiling --shared_fetch_vectorize --use_tc --data_type "float16" --op matmul_expr --shape 4096 4096 4096
    
    echo "Running matmul_expr (float32)..."
    python -u test_op.py --gen_check_code --backend tvm --topk 1 --code_dir generated_source/matmul --smem_tiling --reg_tiling --codegen_input_reg_tiling --shared_fetch_vectorize --data_type "float32" --op matmul_expr --shape 4096 4096 4096
    
    echo ""
}

# Run benchmarks on all GPUs
run_benchmarks 0 "NVIDIA H100"
run_benchmarks 5 "NVIDIA RTX 4090"
run_benchmarks 4 "NVIDIA GV100"

echo "All benchmarks completed!"
