# training bs 128
mkdir -p ./logs/roller/wo_storage_align/elementwise
LOG_DIR=./logs/roller/wo_storage_align/elementwise
CODE_DIR=.
# BUG!!! yangjianchao16
# Namespace(arch='V100', backend='tvm', code_dir='generated_source/elementwise', data_type='float32', eval_bar=[1, 5, 10, 20, 50], fuse=False, keep_tiny=False, num_threads=1, op='relu_expr', padding_threshold_cap=1.0, reg_tiling=True, rtile0_shape=[64, 128, 32], rtile1_shape=[8, 8, 1], rtile2_shape=[1, 1, 1], schedule_fuse=False, shape=[227598336], smem_tiling=True, st_align=False, topk=10, use_artificial_rtile=False, use_tc=False)
# IODependent:  False
# threshold 0
# found 10 results in first round with threshold 0
# evaluating top 10 configs
# {'i0': [32, 2]}
# {'i0': [2, 32, 1]}
# Exception in thread Thread-1:
# Traceback (most recent call last):
#   File "/home/yangjianchao/miniconda3/envs/Roller-py37/lib/python3.7/threading.py", line 926, in _bootstrap_inner
    # self.run()
#   File "/home/yangjianchao/Github/Roller/artifacts/roller/utils/commons.py", line 118, in run
    # self.result = self.func(*self.args)
#   File "./test_op_mp.py", line 340, in eval_thread
    # rprog, op, arch, policy, device_id, idx
#   File "./test_op_mp.py", line 272, in compile_and_run_kernel
    # source = get_tvm_source(rprog, arch, policy, args.data_type)
#   File "./test_op_mp.py", line 258, in get_tvm_source
    # bank_size=arch.smem_bank_size,
#   File "/home/yangjianchao/Github/Roller/artifacts/roller/codegen/op_impl/codegenR.py", line 245, in rewrite_schedule
    # self.sche[rt].compute_at(self.sche[reg_tile], reduce_axis[-1])
# IndexError: list index out of range
python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 227598336 2>&1 |tee $LOG_DIR/elementwise0_128_1008_42_42.log
python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 6422528 2>&1 |tee $LOG_DIR/elementwise1_128_256_14_14.log
python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 25690112 2>&1 |tee $LOG_DIR/elementwise2_128_1024_14_14.log
python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 12845056 2>&1 |tee $LOG_DIR/elementwise3_128_512_14_14.log
python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 334540800 2>&1 |tee $LOG_DIR/elementwise4_128_96_165_165.log
python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 75866112 2>&1 |tee $LOG_DIR/elementwise5_128_1344_21_21.log
python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 41631744 2>&1 |tee $LOG_DIR/elementwise6_128_2688_11_11.log
python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 102760448 2>&1 |tee $LOG_DIR/elementwise7_128_64_112_112.log
python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 102760448 2>&1 |tee $LOG_DIR/elementwise8_128_256_56_56.log
python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 12845056 2>&1 |tee $LOG_DIR/elementwise9_128_128_28_28.log
python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 51380224 2>&1 |tee $LOG_DIR/elementwise10_128_512_28_28.log
python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 25690112 2>&1 |tee $LOG_DIR/elementwise11_128_256_28_28.log
python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 113799168 2>&1 |tee $LOG_DIR/elementwise12_128_2016_21_21.log
python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 10407936 2>&1 |tee $LOG_DIR/elementwise13_128_672_11_11.log
python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 148141056 2>&1 |tee $LOG_DIR/elementwise14_128_168_83_83.log
python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 25690112 2>&1 |tee $LOG_DIR/elementwise15_128_64_56_56.log
python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 37933056 2>&1 |tee $LOG_DIR/elementwise16_128_168_42_42.log
python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 18966528 2>&1 |tee $LOG_DIR/elementwise17_128_336_21_21.log
python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 62447616 2>&1 |tee $LOG_DIR/elementwise18_128_4032_11_11.log
python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 3211264 2>&1 |tee $LOG_DIR/elementwise19_128_512_7_7.log
python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 12845056 2>&1 |tee $LOG_DIR/elementwise20_128_2048_7_7.log
python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 74070528 2>&1 |tee $LOG_DIR/elementwise21_128_84_83_83.log
python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 75866112 2>&1 |tee $LOG_DIR/elementwise22_128_336_42_42.log
python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 146361600 2>&1 |tee $LOG_DIR/elementwise23_128_42_165_165.log
python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 37933056 2>&1 |tee $LOG_DIR/elementwise24_128_672_21_21.log
python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 18966528 2>&1 |tee $LOG_DIR/elementwise25_128_84_42_42.log
python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 37035264 2>&1 |tee $LOG_DIR/elementwise26_128_42_83_83.log
python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 51380224 2>&1 |tee $LOG_DIR/elementwise27_128_128_56_56.log

# python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 128 1008 42 42 2>&1 |tee $LOG_DIR/elementwise_128_1008_42_42.log
# python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 128 256 14 14 2>&1 |tee $LOG_DIR/elementwise_128_256_14_14.log
# python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 128 1024 14 14 2>&1 |tee $LOG_DIR/elementwise_128_1024_14_14.log
# python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 128 512 14 14 2>&1 |tee $LOG_DIR/elementwise_128_512_14_14.log
# python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 128 96 165 165 2>&1 |tee $LOG_DIR/elementwise_128_96_165_165.log
# python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 128 1344 21 21 2>&1 |tee $LOG_DIR/elementwise_128_1344_21_21.log
# python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 128 2688 11 11 2>&1 |tee $LOG_DIR/elementwise_128_2688_11_11.log
# python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 128 64 112 112 2>&1 |tee $LOG_DIR/elementwise_128_64_112_112.log
# python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 128 256 56 56 2>&1 |tee $LOG_DIR/elementwise_128_256_56_56.log
# python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 128 128 28 28 2>&1 |tee $LOG_DIR/elementwise_128_128_28_28.log
# python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 128 512 28 28 2>&1 |tee $LOG_DIR/elementwise_128_512_28_28.log
# python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 128 256 28 28 2>&1 |tee $LOG_DIR/elementwise_128_256_28_28.log
# python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 128 2016 21 21 2>&1 |tee $LOG_DIR/elementwise_128_2016_21_21.log
# python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 128 672 11 11 2>&1 |tee $LOG_DIR/elementwise_128_672_11_11.log
# python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 128 168 83 83 2>&1 |tee $LOG_DIR/elementwise_128_168_83_83.log
# python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 128 64 56 56 2>&1 |tee $LOG_DIR/elementwise_128_64_56_56.log
# python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 128 168 42 42 2>&1 |tee $LOG_DIR/elementwise_128_168_42_42.log
# python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 128 336 21 21 2>&1 |tee $LOG_DIR/elementwise_128_336_21_21.log
# python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 128 4032 11 11 2>&1 |tee $LOG_DIR/elementwise_128_4032_11_11.log
# python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 128 512 7 7 2>&1 |tee $LOG_DIR/elementwise_128_512_7_7.log
# python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 128 2048 7 7 2>&1 |tee $LOG_DIR/elementwise_128_2048_7_7.log
# python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 128 84 83 83 2>&1 |tee $LOG_DIR/elementwise_128_84_83_83.log
# python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 128 336 42 42 2>&1 |tee $LOG_DIR/elementwise_128_336_42_42.log
# python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 128 42 165 165 2>&1 |tee $LOG_DIR/elementwise_128_42_165_165.log
# python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 128 672 21 21 2>&1 |tee $LOG_DIR/elementwise_128_672_21_21.log
# python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 128 84 42 42 2>&1 |tee $LOG_DIR/elementwise_128_84_42_42.log
# python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 128 42 83 83 2>&1 |tee $LOG_DIR/elementwise_128_42_83_83.log
# python -u $CODE_DIR/test_op_mp.py --code_dir generated_source/elementwise --op relu_expr --shape 128 128 56 56 2>&1 |tee $LOG_DIR/elementwise_128_128_56_56.log