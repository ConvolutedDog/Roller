from arch import *
from op import *
from config import *
import codegen.op_impl.codegenR
from codegen.op_impl.codegenR import *
from tvm import te
import time
from policy import *
import os
from utils import *
from test_config import *
import argparse
from typing import Union
from dbg_logger import dbglogger
from typing import Union, Tuple

parser = argparse.ArgumentParser()
parser.add_argument("--op", type=str, default="matmul_expr")
# parser.add_argument("--op", type=str, default="fused_conv_expr_S1D1P1")
# M, N, K for matmul_expr, K is reduce axis.
parser.add_argument("--shape", nargs="*", type=int, default=[5120, 5120, 5120])
# parser.add_argument(
#    "--shape", nargs="*", type=int, default=[128, 128, 28, 28, 128, 3, 3]
# )
parser.add_argument("--rtile2_shape", nargs="*", type=int, default=[1, 1, 1])
parser.add_argument("--rtile1_shape", nargs="*", type=int, default=[16, 8, 1])
parser.add_argument("--rtile0_shape", nargs="*", type=int, default=[64, 256, 16])
parser.add_argument("--backend", type=str, default="tvm")
parser.add_argument(
    "--smem_tiling", dest="smem_tiling", action="store_true", default=True
)
parser.add_argument(
    "--reg_tiling", dest="reg_tiling", action="store_true", default=True
)
# When generating CUDA code, if the input tensor should be fetched to local memory.
parser.add_argument(
    "--codegen_input_reg_tiling",
    dest="codegen_input_reg_tiling",
    action="store_true",
    default=False,
)
# When fetching shared memory, vectorize the shared memory load code (use float4)
# to improve performance.
parser.add_argument(
    "--shared_fetch_vectorize",
    dest="shared_fetch_vectorize",
    action="store_true",
    default=True,
)
# This st_align argument has not been used in the experiments of the original paper.
# Therefore, the related code has not been adapted for newer versions of TVM.
parser.add_argument("--st_align", dest="st_align", action="store_true", default=False)
# This fuse argument has not been used in the experiments of the original paper.
# Therefore, the related code has not been adapted for newer versions of TVM.
parser.add_argument("--fuse", dest="fuse", action="store_true", default=False)
# For matmul_expr, the rewrite_schedule_fuse func has bugs. Therefore, the related
# code has not been adapted for newer versions of TVM.
parser.add_argument(
    "--schedule_fuse", dest="schedule_fuse", action="store_true", default=False
)
# Only test the tile size specified in parser.
parser.add_argument(
    "--use_artificial_rtile",
    dest="use_artificial_rtile",
    action="store_true",
    default=False,
)
# Generate result checking code for each kernel.
parser.add_argument(
    "--gen_check_code", dest="gen_check_code", action="store_true", default=False
)
parser.add_argument("--code_dir", type=str, default="./tmp_dir")
parser.add_argument("--topk", type=int, default=10)
parser.add_argument("--eval_bar", nargs="*", type=int, default=[1, 5, 10, 20, 50])
# When using TC, data type should be "float16".
parser.add_argument("--use_tc", dest="use_tc", action="store_true", default=False)
# "float32" for CUDA Core, "float16" for Tensor Core.
parser.add_argument("--data_type", type=str, default="float32")
parser.add_argument("--padding_threshold_cap", type=float, default=1.0)
parser.add_argument("--keep_tiny", dest="keep_tiny", action="store_true")
# If you have several GPUs with the same architecture, you can change
# the num_threads to run them in parallel.
parser.add_argument("--num_threads", type=int, default=1)
parser.add_argument(
    "--verbose_cuda_code", dest="verbose_cuda_code", action="store_true", default=False
)
parser.add_argument(
    "--verbose_irmodule", dest="verbose_irmodule", action="store_true", default=False
)

args = parser.parse_args()
top1_time = 0


def main_template(
    backend: Backend,
    source: str,
    op: Op,
    grids: Tuple[int, int, int],
    blocks: Tuple[int, int, int],
    times: int,
) -> str:
    input_tensors = op.GetInputTensors(args.fuse or args.schedule_fuse)
    input_tensors_name = ["input" + str(i) for i in range(len(input_tensors))]
    output_tensors = op.GetOutputTensors()
    output_tensors_name = ["output" + str(i) for i in range(len(output_tensors))]
    all_tensors_name = input_tensors_name + output_tensors_name
    tensor_type_size = op.TensorTypeSize(args.fuse or args.schedule_fuse)
    tensor_dim = op.TensorDim(args.fuse or args.schedule_fuse)
    s_size, s_hmalloc, s_dmalloc, s_feed, s_memcpyh2d = "", "", "", "", ""
    s_hfree, s_dfree, s_memcpyd2h = "", "", ""
    s_simple_check = ""
    s_htensor = (
        "    float " + ", ".join(["*" + n + "h" for n in all_tensors_name]) + ";\n"
    )
    s_dtensor = (
        "    float " + ", ".join(["*" + n + "d" for n in all_tensors_name]) + ";\n"
    )
    # hot fix for conv_bias fused kernel, where tvm generate wrong arg order
    for i in range(len(input_tensors)):
        if input_tensors[i].name == "bias":
            all_tensors_name.append(all_tensors_name[i])
            all_tensors_name.remove(all_tensors_name[i])
    s_parameters = ", ".join(["(float*)" + n + "d" for n in all_tensors_name])

    # NOTE: Fix conv parameters bug, temporary solution, should be fixed later.
    if LatestTVM:
        if "conv" in output_tensors[0].op.name:
            s_parameters = ", ".join(
                [
                    "(float*)" + n + "d"
                    for n in [
                        all_tensors_name[-1],
                    ]
                    + all_tensors_name[:-1]
                ]
            )

    for i in range(len(input_tensors_name)):
        name = input_tensors_name[i]
        dim = tensor_dim[0][i]
        type_size = tensor_type_size[0][i]
        size = 1
        for d in dim:
            size *= d
        byte = size * type_size
        s_size += "    int input_size" + str(i) + " = " + str(size) + ";\n"
        s_hmalloc += "    " + name + "h = (float*)malloc(" + str(byte) + ");\n"
        s_hfree += "    free(" + name + "h);\n"
        s_dmalloc += "    cudaMalloc((void **)&" + name + "d, " + str(byte) + ");\n"
        s_dfree += "    cudaFree(" + name + "d);\n"
        s_feed += (
            "    for (int i = 0; i < input_size"
            + str(i)
            + "; ++ i)\n"
            + "        "
            + name
            + "h[i] = 1;\n"
        )
        s_memcpyh2d += (
            "    cudaMemcpy("
            + name
            + "d, "
            + name
            + "h, "
            + str(byte)
            + ", cudaMemcpyHostToDevice);\n"
        )

    for i in range(len(output_tensors_name)):
        name = output_tensors_name[i]
        dim = tensor_dim[1][i]
        type_size = tensor_type_size[1][i]
        size = 1
        for d in dim:
            size *= d
        byte = size * type_size
        s_size += "    int output_size" + str(i) + " = " + str(size) + ";\n"
        s_hmalloc += "    " + name + "h = (float*)malloc(" + str(byte) + ");\n"
        s_hfree += "    free(" + name + "h);\n"
        s_dmalloc += "    cudaMalloc((void **)&" + name + "d, " + str(byte) + ");\n"
        s_dfree += "    cudaFree(" + name + "d);\n"
        s_memcpyd2h += (
            "    cudaMemcpy("
            + name
            + "h, "
            + name
            + "d, "
            + str(byte)
            + ", cudaMemcpyDeviceToHost);\n"
        )
        s_simple_check += (
            "    float same_res = "
            + name
            + "h[0];\n"
            + "    for (int i = 1; i < output_size"
            + str(i)
            + "; ++i)\n"
            + "    {\n"
            "        if ("
            + name
            + "h[i] != same_res)\n"
            + "        {\n"
            + '            printf("output[%d] = %f\\n", i, '
            + name
            + "h[i]);\n"
            + "            exit(1);\n"
            + "        }\n"
            + "    }\n"
        )

    if backend == "antares":
        kernel_name = "template_op_kernel0"
    if backend == "tvm":
        kernel_name = "default_function_kernel0" if not LatestTVM else "main_kernel"

    return (
        "#include <cuda_runtime.h>\n"
        "#include <stdio.h>\n"
        "#include <stdlib.h>\n"
        '#include "cu_helper.h"\n'
        "#include <cuda_fp16.h>\n"
        "#include <mma.h>\n"
        "#include <string>\n"
        "\n"
        "//full_dimensions: {}"
        "\n"
        "{}"
        "\n"
        "int main(int argc, char *argv[])\n"
        "{{\n"
        "    std::string path;\n"
        "{}"
        "\n"
        "    checkCudaErrors(cuInit(0));\n"
        "    CUdevice device;\n"
        "    checkCudaErrors(cuDeviceGet(&device, 0));\n"
        "    CUcontext context;\n"
        "    checkCudaErrors(cuCtxCreate(&context, CU_CTX_SCHED_AUTO/*CU_CTX_SCHED_YIELD*/ | CU_CTX_MAP_HOST, device));\n"
        "\n"
        "{}"
        "{}"
        "{}"
        "\n"
        "{}"
        "\n"
        "    srand(1);\n"
        "{}"
        "\n"
        "{}"
        "\n"
        "    dim3 grid{};\n"
        "    dim3 block{};\n"
        "\n"
        "    for (int i = 0; i < {}; ++i)\n"
        "    {{\n"
        "        {}<<<grid, block>>>({});\n"
        "        cudaDeviceSynchronize();\n"
        "    }}\n"
        "\n"
        "{}"
        "\n"
        "{}"
        "\n"
        "{}"
        "\n"
        "{}"
        "}}\n".format(
            op.Dimensions(),
            source,
            s_size,
            s_htensor,
            s_dtensor,
            s_hmalloc,
            s_dmalloc,
            s_feed,
            s_memcpyh2d,
            grids,
            blocks,
            times,
            kernel_name,
            s_parameters,
            s_memcpyd2h,
            s_simple_check if args.gen_check_code else "",
            s_dfree,
            s_hfree,
        )
    )


def get_pad(rprog: rProg, out_tensor: tvm.te.Tensor):
    smem_tile_shape = rprog.GetTile(0).Dimensions()
    shape = rprog.op.Dimensions()
    saxis_name, raxis_name = get_axis_names(out_tensor)
    all_axis_name = saxis_name + raxis_name
    assert len(smem_tile_shape) == len(shape) == len(all_axis_name)
    pad = {}
    for d in range(len(shape)):
        s = shape[d]
        t = smem_tile_shape[d]
        aligned_s = ((s - 1) // t + 1) * t
        assert aligned_s >= 0
        pad[all_axis_name[d]] = aligned_s - s
    return pad


def get_tvm_source(
    rprog: rProg,
    arch: Union[Arch, IPU, K80, MI50, V100, RTX4090, H100],
    policy: Union[
        NaivePolicy,
        BuildingBlockPolicy,
        ConstructionPolicyV0,
        ConstructionPolicyV1,
        ConstructionPolicyPlain,
        ConstructionPolicyRT,
    ],
    dtype: str,
) -> str:
    expr = rprog.Expression()
    # shape = rprog.Dimensions()
    shape = args.shape
    expr_out = expr(shape, dtype, False)
    in_tensors, out_tensors = expr_out[0], expr_out[1]
    out_tensor = out_tensors[0]
    if args.fuse or args.schedule_fuse:
        pad = get_pad(rprog, out_tensor)
        expr_out = expr(shape, dtype, False, pad)
        in_tensors, out_tensors = expr_out[0], expr_out[1]
        ori_in = []
        pad_in = []
        for ins in in_tensors:
            if "_pad" in ins.name:
                pad_in.append(ins)
            else:
                ori_in.append(ins)
        out_tensor = out_tensors[0]
        write_tensor = out_tensors[-1]
        if LatestTVM:
            # NOTE: Here, specifying the `for_tvm_source` parameter is a
            # last-resort approach, because newer versions of TVM do not
            # support performing compute_inline on the output block in an
            # IRModule. Therefore, we need to set non-output blocks such
            # as data_pad/kernel_pad for operations like convolution.
            import inspect

            if "for_tvm_source" in inspect.signature(expr).parameters:
                expr_out = expr(shape, dtype, False, {}, for_tvm_source=True)
                in_tensors, out_tensors = expr_out[0], expr_out[1]
                out_tensor = out_tensors[0]

            pf = te.create_prim_func(in_tensors + out_tensors)
            mod = tvm.IRModule({"main": pf})
            # Create a TIR schedule
            s = tvm.tir.Schedule(mod)
        else:
            s = te.create_schedule(write_tensor.op)
        align_info = policy.get_align_info_fuse(
            rprog,
            arch,
            args.smem_tiling,
            args.reg_tiling,
            target_stage=out_tensor.name,
            write_stage=write_tensor.name,
            st_align=args.st_align,
        )
        cgen = CodeGeneratorR()
        cgen.rewrite_schedule_fuse(
            s,
            rprog,
            args.smem_tiling,
            args.reg_tiling,
            pad_in,
            out_tensors[:-1],
            write_tensor,
            target_stage=out_tensor.name,
            write_stage=write_tensor.name,
            align_info=align_info,
            bank_size=arch.smem_bank_size,
        )
        func = tvm.build(s, ori_in + out_tensors, "cuda")
        return func.imported_modules[0].get_source()
    else:
        if LatestTVM:
            pf = te.create_prim_func(in_tensors + out_tensors)
            mod = tvm.IRModule({"main": pf})
            # Create a TIR schedule
            s = tvm.tir.Schedule(mod)
        else:
            s = te.create_schedule(out_tensor.op)
        align_info = policy.get_align_info(
            rprog,
            arch,
            args.smem_tiling,
            args.reg_tiling,
            target_stage=out_tensor.name,
            st_align=args.st_align,
        )
        cgen = CodeGeneratorR()
        cgen.rewrite_schedule(
            s,
            rprog,
            args.smem_tiling,
            args.reg_tiling,
            target_stage=out_tensor.name,
            align_info=align_info,
            bank_size=arch.smem_bank_size,
            in_tensors=in_tensors,
            out_tensors=out_tensors,
            shared_fetch_vectorize=args.shared_fetch_vectorize,
            codegen_input_reg_tiling=args.codegen_input_reg_tiling,
        )
        if LatestTVM:
            target = tvm.target.Target("cuda")
            mod = tvm.build(s.mod, target=target)

            return mod.imported_modules[0].get_source()
        else:
            s.normalize()
            mod = tvm.lower(s, in_tensors + out_tensors, simple_mode=False)

            func = tvm.build(s, in_tensors + out_tensors, "cuda")
            return func.imported_modules[0].get_source()


def compile_and_run_kernel(
    rprog: rProg,
    op: Op,
    arch: Union[Arch, IPU, K80, MI50, V100, RTX4090, H100],
    policy: Union[
        NaivePolicy,
        BuildingBlockPolicy,
        ConstructionPolicyV0,
        ConstructionPolicyV1,
        ConstructionPolicyPlain,
        ConstructionPolicyRT,
    ],
    device_id,
    idx,
):
    print(f"rProg[{idx}]: {rprog.Dump()}")
    block_size = rprog.GetParallelism(1) * (32 if args.use_tc else 1)
    grid_size = rprog.GetParallelism(0)
    blocks = (block_size, 1, 1)
    grids = (grid_size, 1, 1)

    times = 10
    if not args.use_tc:
        source = get_tvm_source(rprog, arch, policy, args.data_type)
        main_source = main_template(args.backend, source, op, grids, blocks, times)
    else:
        source = get_tc_mm_source(
            op.GetInputTensors()[0],
            op.GetInputTensors()[1],
            op.GetOutputTensors()[0],
            rprog,
        )

        M, N, K = args.shape
        block_x, block_y, block_z = get_tc_block_size(
            rprog.GetTile(0), rprog.GetTile(1)
        )
        grid_x, grid_y = get_tc_grid_size(M, N, rprog.GetTile(0))
        main_source = tc_mm_main_template(
            source,
            M,
            K,
            N,
            grid_x,
            grid_y,
            block_x,
            block_y,
            block_z,
            10,
            args.gen_check_code,
        )

        blocks = (block_x, block_y, block_z)
        grids = (grid_x, grid_y, 1)

    file_name = "{}_{}_{}_{}_{}_{}".format(
        args.op,
        "_".join([str(d) for d in args.shape]),
        device_id,
        idx,
        "_".join([str(d) for d in grids]),
        "_".join([str(d) for d in blocks]),
    )
    log_name = "tmp_{}_{}_{}_{}".format(
        args.op, "_".join([str(d) for d in args.shape]), device_id, idx
    )
    with open("{}.cu".format(file_name), "w") as ouf:
        ouf.write(main_source)

        if args.verbose_cuda_code:
            print("v" * 40)
            print(main_source)
            print("^" * 40)

    os.system(
        "nvcc {}.cu -lcuda -gencode=arch=compute_{},code=compute_{} -o {}".format(
            file_name, compute_capability, compute_capability, file_name
        )
    )

    # NOTE: nvprof is not supported on devices with compute capability
    # 8.0 and higher.
    if compute_capability >= "80":
        os.system(
            "nsys nvprof -o /tmp/tmp-nsys-rep --force-overwrite ./{} &> {}".format(
                file_name, log_name
            )
        )
    else:
        os.system("nvprof ./{} &> {}".format(file_name, log_name))

    os.system("rm {}".format(file_name))
    os.system("rm {}.cu".format(file_name))

    with open(log_name, "r") as f:
        for line in f.readlines():
            print(line, end="")

    exec_time = get_time_from_nvprof_file(log_name)
    os.system("rm {}".format(log_name))
    return exec_time, source, blocks, grids


def eval_thread(
    rprogs: rProg,
    rprog_idx: int,
    device_id: int,
    op: Op,
    arch: Union[Arch, IPU, K80, MI50, V100, RTX4090, H100],
    policy: Union[
        NaivePolicy,
        BuildingBlockPolicy,
        ConstructionPolicyV0,
        ConstructionPolicyV1,
        ConstructionPolicyPlain,
        ConstructionPolicyRT,
    ],
):
    best_time = 1e100
    best_idx = 0
    top1_time = -1
    best_source = None
    best_blocks = None
    best_grids = None
    for idx in range(rprog_idx[device_id], rprog_idx[device_id + 1]):
        rprog = rprogs[idx]
        exec_time, source, blocks, grids = compile_and_run_kernel(
            rprog, op, arch, policy, device_id, idx
        )
        if exec_time < best_time:
            best_idx = idx
            best_time = exec_time
            best_source = source
            best_blocks = blocks
            best_grids = grids
        if idx == 0 and device_id == 0:
            top1_time = exec_time
    return best_time, best_idx, top1_time, best_source, best_blocks, best_grids


if __name__ == "__main__":
    print(args)
    expr = globals()[args.op]
    if args.fuse:
        expr = rewrite_expr(expr, args.shape, "fused_" + args.op)
    arch = GPUDetector.create_gpu_instance()
    if args.use_tc:
        assert args.op == "matmul_expr"
    op = Op(expr, args.shape, args.data_type, args.use_tc)
    print("IODependent: ", op.IODependent())
    start_time = time.time()
    if op.IODependent():
        policy = ConstructionPolicyRT(
            op,
            arch,
            args.smem_tiling,
            args.reg_tiling,
            args.st_align,
            args.padding_threshold_cap,
            shrink_tiny=not args.keep_tiny,
        )
    else:
        policy = ConstructionPolicyPlainRT(
            op,
            arch,
            args.smem_tiling,
            args.reg_tiling,
            args.st_align,
            args.padding_threshold_cap,
        )
    emit_time = time.time() - start_time

    if args.use_artificial_rtile and len(op.Dimensions()) == len(
        args.rtile2_shape
    ) == len(args.rtile1_shape) == len(args.rtile0_shape):
        rTile2 = rTile(
            expr, args.rtile2_shape, op.SAxis(), op.RAxis(), op.GetTvmOutTensor()
        )
        rTile1 = rTile(
            expr, args.rtile1_shape, op.SAxis(), op.RAxis(), op.GetTvmOutTensor()
        )
        rTile0 = rTile(
            expr, args.rtile0_shape, op.SAxis(), op.RAxis(), op.GetTvmOutTensor()
        )
        rprog = rProg(arch.num_level, op)
        rprog.AddTile(2, rTile2)
        rprog.AddTile(1, rTile1)
        rprog.AddTile(0, rTile0)

        rprogs = [rprog]
        print("-------------------use artificial rtile---------------------------")
    else:
        rprogs = policy.emit_config_without_trails(args.topk)

    print("Evaluating top {} configs".format(len(rprogs)))

    rprog_idx = alloc_configs_for_subprocess(args.num_threads, len(rprogs))
    threads = []
    for device_id in range(args.num_threads):
        thread = MyThread(
            func=eval_thread, args=(rprogs, rprog_idx, device_id, op, arch, policy)
        )
        threads.append(thread)
        thread.start()

    best_time = 1e100
    for thread in threads:
        thread.join()
        (
            local_best_time,
            local_best_idx,
            this_top1_time,
            local_best_source,
            local_best_blocks,
            local_best_grids,
        ) = thread.get_result()
        if local_best_time < best_time:
            best_time = local_best_time
            best_idx = local_best_idx
            best_source = local_best_source
            best_block_size = local_best_blocks
            best_grid_size = local_best_grids
        if this_top1_time > -1:
            top1_time = this_top1_time

    eval_time = time.time() - start_time

    print("Top1 time: {} ms".format(top1_time))
    print("Top10 time: {} ms".format(best_time))
    print("Best idx: {}".format(best_idx))
    print("Best config: {}".format(rprogs[best_idx].Dump()))
    print("Top1 compile time: {} s".format(emit_time))
    print("Top10 compile time: {} s".format(eval_time))

    cu_file_name = "roller_{}_{}.cu".format(
        args.op, "_".join([str(d) for d in args.shape])
    )
    os.system("mkdir -p " + args.code_dir)
    with open(os.path.join(args.code_dir, cu_file_name), "w") as f:
        f.write(best_source)
        f.write("dim3 grid{};\n".format(best_grid_size))
        f.write("dim3 block{};\n".format(best_block_size))
