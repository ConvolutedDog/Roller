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
from typing import Union, Tuple
from dbg_logger import dbglogger


parser = argparse.ArgumentParser()
parser.add_argument("--op", type=str, default="matmul_expr")
# M, N, K for matmul_expr, K is reduce axis.
parser.add_argument("--shape", nargs="*", type=int, default=[5120, 5120, 5120])
parser.add_argument("--rtile2_shape", nargs="*", type=int, default=[1, 1, 1])
parser.add_argument("--rtile1_shape", nargs="*", type=int, default=[8, 8, 1])
parser.add_argument("--rtile0_shape", nargs="*", type=int, default=[128, 64, 8])
parser.add_argument("--arch", type=str, default="V100")
parser.add_argument("--backend", type=str, default="tvm")
parser.add_argument(
    "--smem_tiling", dest="smem_tiling", action="store_true", default=True
)
parser.add_argument(
    "--reg_tiling", dest="reg_tiling", action="store_true", default=True
)
# This st_align argument has not been used in the experiments of the original paper.
# Therefore, the related code has not been adapted for newer versions of TVM.
parser.add_argument("--st_align", dest="st_align", action="store_true", default=False)
# This fuse argument has not been used in the experiments of the original paper.
# Therefore, the related code has not been adapted for newer versions of TVM.
parser.add_argument("--fuse", dest="fuse", action="store_true", default=False)
# For maymul_expr, the rewrite_schedule_fuse func has bugs. Therefore, the related
# code has not been adapted for newer versions of TVM.
parser.add_argument(
    "--schedule_fuse", dest="schedule_fuse", action="store_true", default=False
)
# Only test the tile size specified in parser.
parser.add_argument(
    "--use_artificial_rtile ",
    dest="use_artificial_rtile",
    action="store_true",
    default=False,
)
parser.add_argument("--code_dir", type=str, default="./tmp_dir")
parser.add_argument("--topk", type=int, default=10)
parser.add_argument("--eval_bar", nargs="*", type=int, default=[1, 5, 10, 20, 50])
# When using TC, data type should be "float16".
parser.add_argument("--use_tc", dest="use_tc", action="store_true", default=True)
# "float32" for CUDA Core, "float16" for Tensor Core.
parser.add_argument("--data_type", type=str, default="float16")
parser.add_argument("--padding_threshold_cap", type=float, default=1.0)
parser.add_argument("--keep_tiny", dest="keep_tiny", action="store_true")


args = parser.parse_args()


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
    arch: Union[Arch, IPU, K80, MI50, V100],
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
        print("pad: ", pad)
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
        )
        if LatestTVM:
            print(s.mod)

            target = tvm.target.Target("cuda")
            mod = tvm.build(s.mod, target=target)

            return mod.imported_modules[0].get_source()
        else:
            s.normalize()
            mod = tvm.lower(s, in_tensors + out_tensors, simple_mode=False)

            func = tvm.build(s, in_tensors + out_tensors, "cuda")
            return func.imported_modules[0].get_source()


if __name__ == "__main__":
    print(args)
    expr = globals()[args.op]
    if args.fuse:
        expr = rewrite_expr(expr, args.shape, "fused_" + args.op)
    arch = globals()[args.arch]()
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

    print("evaluating top {} configs".format(len(rprogs)))
    best_idx = -1
    best_time = 1e100
    idx = 0

    eval_bar = args.eval_bar
    evals = []
    bar_id = 0
    dtype = "float16" if args.use_tc else "float32"
    for rprog in rprogs:
        print("id: {}".format(idx))
        print(rprog.Dump())
        block_size = rprog.GetParallelism(1) * (32 if args.use_tc else 1)
        grid_size = rprog.GetParallelism(0)
        blocks = (block_size, 1, 1)
        grids = (grid_size, 1, 1)

        file_name = "{}_{}_{}_{}_{}_{}".format(
            args.op,
            "_".join([str(d) for d in args.shape]),
            0,  # device_id
            idx,
            "_".join([str(d) for d in grids]),
            "_".join([str(d) for d in blocks]),
        )
        log_name = "tmp_{}_{}_{}_{}".format(
            args.op, "_".join([str(d) for d in args.shape]), 0, idx
        )
        times = 10
        if not args.use_tc:
            source = get_tvm_source(rprog, arch, policy, dtype)
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
                source, M, K, N, grid_x, grid_y, block_x, block_y, block_z, 10
            )

        with open("{}.cu".format(file_name), "w") as ouf:
            ouf.write(main_source)

            print("v" * 40)
            print(main_source)
            print("^" * 40)

        if LatestTVM:
            os.system(
                "/usr/local/cuda-12.4/bin/nvcc {}.cu -lcuda -gencode=arch=compute_70,code=compute_70 -o {}".format(
                    file_name, file_name
                )
            )
        else:
            os.system(
                "/usr/local/cuda-10.2/bin/nvcc {}.cu -lcuda -gencode=arch=compute_70,code=compute_70 -o {}".format(
                    file_name, file_name
                )
            )

        if LatestTVM:
            os.system(
                "/usr/local/cuda-12.4/bin/nvprof ./{} &> {}".format(file_name, log_name)
            )
        else:
            os.system(
                "/usr/local/cuda-10.2/bin/nvprof ./{} &> {}".format(file_name, log_name)
            )

        os.system("rm {}".format(file_name))
        os.system("rm {}.cu".format(file_name))

        print("LOG_NAME: {}".format(log_name))
        with open(log_name, "r") as f:
            for line in f.readlines():
                print(line, end="")

        exec_time = get_time_from_nvprof_file(log_name)
        os.system("rm {}".format(log_name))
        if exec_time < best_time:
            best_idx = idx
            best_rprog = rprog
            best_time = exec_time

            best_source = source
            best_block_size = block_size
            best_grid_size = grid_size

        idx += 1
        print(idx, bar_id)
        if idx == eval_bar[bar_id]:
            cur_time = time.time()
            eval_results = {}
            eval_results["best time"] = best_time
            eval_results["best idx"] = best_idx
            eval_results["best config"] = best_rprog.Dump()
            eval_results["compilation time"] = cur_time - start_time
            evals.append(eval_results)
            bar_id += 1

    for topx, eval_results in zip(eval_bar, evals):
        print("Eval top {} configs ======================".format(topx))
        print("compilation time: {}s".format(eval_results["compilation time"]))
        print("best time: {}ms".format(eval_results["best time"]))
        print("best config: {}".format(eval_results["best config"]))
        print("best idx: {}".format(eval_results["best idx"]))

    cu_file_name = "roller_{}_{}.cu".format(
        args.op, "_".join([str(d) for d in args.shape])
    )
    os.system("mkdir -p " + args.code_dir)
    with open(os.path.join(args.code_dir, cu_file_name), "w") as f:
        f.write(best_source)
        f.write("dim3 grid({}, 1, 1);\n".format(best_grid_size))
        f.write("dim3 block({}, 1, 1);\n".format(best_block_size))
