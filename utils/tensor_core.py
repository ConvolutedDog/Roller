import tvm
from tvm import te
from .tc_intrin import (
    init_intrin_strides,
    intrin_wmma_load_matrix,
    intrin_wmma_store_matrix,
    intrin_wmma_gemm,
    register_wmma_load_intrin,
    register_wmma_fill_intrin,
    register_wmma_intrin,
    register_wmma_store_intrin,
)
import math
import numpy as np
from utils import (
    LatestTVM,
    showmod,
    getSpatialLoopRVs,
    getReduceLoopRVs,
    calculate_factors,
)
from typing import Tuple, TYPE_CHECKING, Union, Optional
from config import rProg, rTile

if TYPE_CHECKING:
    # Type checking only
    from tvm.te import Schedule as TeSchedule
    from tvm.tir import Schedule as TirSchedule

    ScheduleType = Union[TeSchedule, TirSchedule]
else:
    # Runtime only
    if not LatestTVM:
        from tvm.te import Schedule as TeSchedule

        ScheduleType = TeSchedule
    else:
        from tvm.tir import Schedule as TirSchedule

        ScheduleType = TirSchedule


def schedule_tensorcore(
    tvm_schedule: ScheduleType,
    rprog: rProg,
    A: Optional[tvm.te.Tensor],
    B: Optional[tvm.te.Tensor],
    C: tvm.te.Tensor,
    verbose: bool = False,
) -> ScheduleType:
    """
    Schedule dense operator using Tensorcore
    """
    Mdim, Ndim = C.shape
    s = tvm_schedule
    if not LatestTVM:
        A, B = s[C].op.input_tensors
    else:
        assert A is not None
        assert B is not None
    data_dtype = A.dtype
    out_dtype = C.dtype

    # Explicit memory access
    if not LatestTVM:
        AS = s.cache_read(A, "shared", [C])
        BS = s.cache_read(B, "shared", [C])
        AF = s.cache_read(AS, "wmma.matrix_a", [C])
        BF = s.cache_read(BS, "wmma.matrix_b", [C])
        CF = s.cache_write(C, "wmma.accumulator")
        # CS = s.cache_read(CF, "shared", [C])
    else:
        block_compute = s.get_block("compute")
        AS: tvm.tir.BlockRV = s.cache_read(
            block_compute, read_buffer_index=0, storage_scope="shared"
        )
        BS: tvm.tir.BlockRV = s.cache_read(
            block_compute, read_buffer_index=1, storage_scope="shared"
        )
        AF: tvm.tir.BlockRV = s.cache_read(
            block_compute, read_buffer_index=0, storage_scope="wmma.matrix_a"
        )
        BF: tvm.tir.BlockRV = s.cache_read(
            block_compute, read_buffer_index=1, storage_scope="wmma.matrix_b"
        )
        CF: tvm.tir.BlockRV = s.cache_write(
            block_compute, write_buffer_index=0, storage_scope="wmma.accumulator"
        )

    if verbose:
        showmod(s, [A, B, C])

    # Extract Sokoban scheduling information
    #
    #  -------------------               -------             ---
    # |    |    |    |    |             |---|-`-|`--------> |   | wmma_m
    # |    |    |    |  `-|-----------> |---|---| warp_m     ---
    # |    |    |    |    |             |---|---|           wmma_n
    # |----|----|----|----| block_m      -------
    # |    |    |    |    |              warp_n
    # |    |    |    |    |
    # |    |    |    |    |
    #  -------------------
    #       block_n
    #
    warp_size = 32
    wmma_m, wmma_n = rprog.GetTile(2).SDimensions()
    wmma_k = rprog.GetTile(2).RDimensions()[0]
    warp_m, warp_n = rprog.GetTile(1).SDimensions()
    block_m, block_n = rprog.GetTile(0).SDimensions()
    rstep_size = rprog.GetTile(0).RDimensions()[0] // wmma_k

    block_row_warps = block_m // warp_m
    block_col_warps = block_n // warp_n
    warp_row_tiles = warp_m // wmma_m
    warp_col_tiles = warp_n // wmma_n
    offset = 8
    offsetCS = 0
    vec = 1

    # Define the stride of intrin functions
    # CS_align = warp_col_tiles * block_col_warps * wmma_n + offsetCS
    # AS_stride = [AS_align, 1]
    # BS_stride = [BS_align, 1]
    # AF_stride = [wmma_k, 1]
    # BF_stride = [wmma_k, 1]
    AF_stride, AS_stride = init_intrin_strides(
        wmma_shape=[wmma_m, wmma_k],
        warp_s=warp_row_tiles,
        block_s=block_row_warps,
        rstep_size=rstep_size,
        offset=offset,
        layout="row_major",
    )
    BF_stride, BS_stride = init_intrin_strides(
        wmma_shape=[wmma_k, wmma_n],
        warp_s=warp_col_tiles,
        block_s=block_col_warps,
        rstep_size=rstep_size,
        offset=offset,
        layout="col_major",
    )

    AS_align = AS_stride[0]
    BS_align = BS_stride[0]
    CF_stride = [warp_col_tiles * wmma_n, 1]
    C_stride = [Ndim, 1]

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")

    if not LatestTVM:
        # Schedule for dense computation
        block_factor_m = wmma_m * warp_row_tiles * block_row_warps
        block_factor_n = wmma_n * warp_col_tiles * block_col_warps
        m, n = C.op.axis

        block_i, mc = s[C].split(m, factor=block_factor_m)
        block_j, nc = s[C].split(n, factor=block_factor_n)
        mm, mmi = s[C].split(mc, factor=wmma_m)
        nn, nni = s[C].split(nc, factor=wmma_n)
        mm, mmii = s[C].split(mm, factor=warp_row_tiles)
        nn, nnii = s[C].split(nn, factor=warp_col_tiles)
        s[C].reorder(block_i, block_j, mm, nn, mmii, nnii, mmi, nni)
        s[C].bind(block_i, block_x)
        s[C].bind(block_j, block_y)
    else:
        block_factor_m = wmma_m * warp_row_tiles * block_row_warps
        block_factor_n = wmma_n * warp_col_tiles * block_col_warps

        loops_wmma_acc = s.get_loops(CF)

        m, n = loops_wmma_acc

        block_i, mc = s.split(m, factors=calculate_factors(s, m, block_factor_m))
        block_j, nc = s.split(n, factors=calculate_factors(s, n, block_factor_n))

        mm, mmi = s.split(mc, factors=calculate_factors(s, mc, wmma_m))
        nn, nni = s.split(nc, factors=calculate_factors(s, nc, wmma_n))
        mm, mmii = s.split(mm, factors=calculate_factors(s, mm, warp_row_tiles))
        nn, nnii = s.split(nn, factors=calculate_factors(s, nn, warp_col_tiles))

        s.reorder(block_i, block_j, mm, nn, mmii, nnii, mmi, nni)
        s.bind(block_i, "blockIdx.x")
        s.bind(block_j, "blockIdx.y")

    if verbose:
        showmod(s, [A, B, C])

    if not LatestTVM:
        s[C].bind(mm, thread_y)
        s[C].bind(nn, thread_z)
        s[C].unroll(mmii)
        s[C].unroll(nnii)
    else:
        s.bind(mm, "threadIdx.y")
        s.bind(nn, "threadIdx.z")
        s.unroll(mmii)
        s.unroll(nnii)

    if verbose:
        showmod(s, [A, B, C])

    if not LatestTVM:
        # Schedule for wmma computation
        s[CF].compute_at(s[C], nn)
        warp_i, warp_j = CF.op.axis
        warp_i, _ii = s[CF].split(warp_i, factor=wmma_m)
        warp_j, _jj = s[CF].split(warp_j, factor=wmma_n)
        (k,) = CF.op.reduce_axis
        k, _k = s[CF].split(k, factor=wmma_k)
        ko, ki = s[CF].split(k, factor=rstep_size)
        s[CF].reorder(ko, ki, warp_i, warp_j, _ii, _jj, _k)
        s[CF].unroll(ki)
        s[CF].unroll(warp_i)
        s[CF].unroll(warp_j)
    else:
        block_compute = s.get_block("compute")
        s.compute_at(block_compute, nn)
        warp_i, warp_j = getSpatialLoopRVs(s, block_compute)
        warp_i, _ii = s.split(warp_i, factors=calculate_factors(s, warp_i, wmma_m))
        warp_j, _jj = s.split(warp_j, factors=calculate_factors(s, warp_j, wmma_n))
        (k,) = getReduceLoopRVs(s, block_compute)
        k, _k = s.split(k, factors=calculate_factors(s, k, wmma_k))
        ko, ki = s.split(k, factors=calculate_factors(s, k, rstep_size))
        s.reorder(ko, ki, warp_i, warp_j, _ii, _jj, _k)
        s.unroll(ki)
        s.unroll(warp_i)
        s.unroll(warp_j)

        # Purpose: Optimize reduction operations by hoisting initialization
        #          out of hot loops
        # Ref: https://discuss.tvm.apache.org/t/optimizing-reduction-initialization-in-generated-cuda-code/18613
        # ------------------------------------------------------------
        # Original Problem:
        #   In reduction patterns (e.g., matrix multiplication), initialization
        #   (e.g., C[i,j] = 0) typically occurs INSIDE the reduction loop nest,
        #   guarded by a condition like:
        #     "if (k == 0): C[i,j] = 0"
        #   This causes two performance issues:
        #     1. Branch Divergence: Threads waste cycles checking the condition
        #        every iteration.
        #     2. Redundant Checks: Initialization only needs to run once, but
        #        the condition is evaluated N times.
        #
        # Solution: Decompose the reduction into two phases:
        #   1. Initialization Phase (runs once):
        #      - Sets output buffers to zero (e.g., C_init block)
        #      - Positioned OUTSIDE all computation loops
        #   2. Update Phase (runs in hot loops):
        #      - Pure computation without initialization checks (e.g., C_update
        #        block)
        #
        # Performance Impact:
        #   - Eliminates branch mispredictions (~20% speedup observed in generated
        #     CUDA kernels for a 5120 x 5120 x 5120 matrix multiplication operation).
        # ------------------------------------------------------------
        # How this code achieves it:
        #   - decompose_reduction(block_compute, loops[3]):
        #     * Identifies the reduction block (e.g., matrix multiply accumulation)
        #     * Splits it at the SECOND-TO-LAST loop level
        #     * Creates two blocks:
        #       - {block_name}_init: Initializes output
        #       - {block_name}_update: Pure computation
        initial_block = s.decompose_reduction(block_compute, nn)

    if verbose:
        showmod(s, [A, B, C])

    if not LatestTVM:
        # Schedule for  wmma_matrix_a load
        s[AF].compute_at(s[CF], ki)
        m, i = AF.op.axis
        m, m_ii = s[AF].split(m, factor=wmma_m)
        i, i_jj = s[AF].split(i, factor=wmma_k)
        s[AF].reorder(m, i, m_ii, i_jj)
        s[AF].unroll(m)
        s[AF].unroll(i)
    else:
        s.compute_at(AF, ki)
        m, i = getSpatialLoopRVs(s, AF)
        m, m_ii = s.split(m, factors=calculate_factors(s, m, wmma_m))
        i, i_jj = s.split(i, factors=calculate_factors(s, i, wmma_k))
        s.reorder(m, i, m_ii, i_jj)
        s.unroll(m)
        s.unroll(i)

    if verbose:
        showmod(s, [A, B, C])

    if not LatestTVM:
        # Schedule for  wmma_matrix_b load
        s[BF].compute_at(s[CF], ki)
        i, n = BF.op.axis
        i, i_ii = s[BF].split(i, factor=wmma_k)
        n, n_ii = s[BF].split(n, factor=wmma_n)
        s[BF].reorder(i, n, i_ii, n_ii)
        s[BF].unroll(i)
        s[BF].unroll(n)
    else:
        s.compute_at(BF, ki)
        i, n = getSpatialLoopRVs(s, BF)
        i, i_ii = s.split(i, factors=calculate_factors(s, i, wmma_k))
        n, n_ii = s.split(n, factors=calculate_factors(s, n, wmma_n))
        s.reorder(i, n, i_ii, n_ii)
        s.unroll(i)
        s.unroll(n)

    if verbose:
        showmod(s, [A, B, C])

    # Schedule for A's(B's) shared memory load
    def shared_shedule(stage, strides):
        if not LatestTVM:
            s[stage].compute_at(s[CF], ko)
            xo, yo = stage.op.axis
            s[stage].storage_align(xo, strides - 1, strides)
            t = s[stage].fuse(xo, yo)
            t, vi = s[stage].split(t, factor=vec)
            t, tx = s[stage].split(t, factor=warp_size)
            t, ty = s[stage].split(t, factor=block_row_warps)
            t, tz = s[stage].split(t, factor=block_col_warps)
            s[stage].bind(ty, thread_y)
            s[stage].bind(tz, thread_z)
            s[stage].bind(tx, thread_x)
            s[stage].unroll(t)
            s[stage].vectorize(vi)
        else:
            s.compute_at(stage, ko)
            xo, yo = getSpatialLoopRVs(s, stage)
            # NOTE: `index` of xo in getSpatialLoopRVs(s, stage) equals to 0.
            s.storage_align(
                stage, buffer_index=0, axis=0, factor=strides - 1, offset=strides
            )
            t = s.fuse(xo, yo)
            t, vi = s.split(t, factors=calculate_factors(s, t, vec))
            t, tx = s.split(t, factors=calculate_factors(s, t, warp_size))
            t, ty = s.split(t, factors=calculate_factors(s, t, block_row_warps))
            t, tz = s.split(t, factors=calculate_factors(s, t, block_col_warps))
            s.bind(ty, "threadIdx.y")
            s.bind(tz, "threadIdx.z")
            s.bind(tx, "threadIdx.x")
            s.unroll(t)
            s.vectorize(vi)

    shared_shedule(AS, AS_align)
    shared_shedule(BS, BS_align)

    if verbose:
        showmod(s, [A, B, C])

    shape = (wmma_m, wmma_n, wmma_k)
    AL_gemm = te.placeholder((wmma_m, wmma_k), name="AL_gemm", dtype=data_dtype)
    BL_gemm = te.placeholder((wmma_k, wmma_n), name="BL_gemm", dtype=data_dtype)
    k_gemm = te.reduce_axis((0, wmma_k), name="k_gemm")
    CL_compute = te.compute(
        (wmma_m, wmma_n),
        lambda ii, jj: te.sum(
            AL_gemm[ii, k_gemm].astype(out_dtype)
            * BL_gemm[k_gemm, jj].astype(out_dtype),
            axis=k_gemm,
        ),
        name="CL_compute",
    )

    if not LatestTVM:
        # lower the computation loops down to TensorCore hardware intrinsics
        # by mapping the dense tensorcore to tensor intrinsics
        s[AF].tensorize(
            m_ii,
            intrin_wmma_load_matrix(
                (wmma_m, wmma_n, wmma_k),
                (warp_row_tiles, warp_col_tiles),
                (block_row_warps, block_col_warps),
                rstep_size,
                AF_stride,
                AS_stride,
                "wmma.matrix_a",
                "row_major",
                data_dtype,
            ),
        )
    else:
        intrin_name = register_wmma_load_intrin(
            (wmma_m, wmma_n, wmma_k),
            (warp_row_tiles, warp_col_tiles),
            (block_row_warps, block_col_warps),
            rstep_size,
            AF_stride,
            AS_stride,
            "wmma.matrix_a",
            "row_major",
            data_dtype,
        )
        s.tensorize(m_ii, intrin_name)

    if verbose:
        showmod(s, [A, B, C])

    if not LatestTVM:
        s[BF].tensorize(
            i_ii,
            intrin_wmma_load_matrix(
                (wmma_m, wmma_n, wmma_k),
                (warp_row_tiles, warp_col_tiles),
                (block_row_warps, block_col_warps),
                rstep_size,
                BF_stride,
                BS_stride,
                "wmma.matrix_b",
                "col_major",
                data_dtype,
            ),
        )
    else:
        intrin_name = register_wmma_load_intrin(
            (wmma_m, wmma_n, wmma_k),
            (warp_row_tiles, warp_col_tiles),
            (block_row_warps, block_col_warps),
            rstep_size,
            BF_stride,
            BS_stride,
            "wmma.matrix_b",
            "col_major",
            data_dtype,
        )
        s.tensorize(i_ii, intrin_name)

    if verbose:
        showmod(s, [A, B, C])

    if not LatestTVM:
        s[CF].tensorize(
            _ii,
            intrin_wmma_gemm(
                AL_gemm, BL_gemm, CL_compute, AF_stride, BF_stride, CF_stride, shape
            ),
        )
    else:
        intrin_name = register_wmma_intrin(
            AL_gemm, BL_gemm, CL_compute, AF_stride, BF_stride, CF_stride, shape
        )
        s.tensorize(_ii, intrin_name)

        intrin_name = register_wmma_fill_intrin(
            (wmma_m, wmma_n, wmma_k), CF_stride, out_dtype
        )
        s.tensorize(s.get_loops(initial_block)[-2], intrin_name)

    if verbose:
        showmod(s, [A, B, C])

    if not LatestTVM:
        s[C].tensorize(
            mmi,
            intrin_wmma_store_matrix(C_stride, CF_stride, shape, out_dtype),
        )
    else:
        intrin_name = register_wmma_store_intrin(C_stride, CF_stride, shape, out_dtype)
        s.tensorize(mmi, intrin_name)

    if verbose:
        showmod(s, [A, B, C])

    return s


BACKEND = "tvm"


def tc_mm_main_template(
    source: str,
    M: int,
    K: int,
    N: int,
    grid_x: int,
    grid_y: int,
    block_x: int,
    block_y: int,
    block_z: int,
    times: int,
) -> str:
    if BACKEND == "antares":
        kernel_name = "template_op_kernel0"
    if BACKEND == "tvm":
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
        "const int M = {}, K = {}, N = {};\n"
        "\n"
        "{}"
        "\n"
        "int main(int argc, char *argv[])\n"
        "{{\n"
        "    const int input_size0 = M * K;\n"
        "    const int input_size1 = K * N;\n"
        "    const int output_size = N * M;\n"
        "\n"
        "    checkCudaErrors(cuInit(0));\n"
        "    CUdevice device;\n"
        "    checkCudaErrors(cuDeviceGet(&device, 0));\n"
        "    CUcontext context;\n"
        "    checkCudaErrors(cuCtxCreate(&context, CU_CTX_SCHED_AUTO/*CU_CTX_SCHED_YIELD*/ | CU_CTX_MAP_HOST, device));\n"
        "\n"
        "    half *Ah, *Bh, *Ch;\n"
        "    half *Ad, *Bd, *Cd;\n"
        "    Ah = (half*)malloc(input_size0 * sizeof(half));\n"
        "    Bh = (half*)malloc(input_size1 * sizeof(half));\n"
        "    Ch = (half*)malloc(output_size * sizeof(half));\n"
        "\n"
        "    cudaMalloc((void **)&Ad, input_size0 * sizeof(half));\n"
        "    cudaMalloc((void **)&Bd, input_size1 * sizeof(half));\n"
        "    cudaMalloc((void **)&Cd, output_size * sizeof(half));\n"
        "\n"
        "    srand(1);\n"
        "    for (int i = 0; i < input_size0; ++ i)\n"
        "        Ah[i] = __float2half(1);\n"
        "    for (int i = 0; i < input_size1; ++ i)\n"
        "        Bh[i] = __float2half(1);\n"
        "\n"
        "    cudaMemcpy(Ad, Ah, input_size0 * sizeof(half), cudaMemcpyHostToDevice);\n"
        "    cudaMemcpy(Bd, Bh, input_size1 * sizeof(half), cudaMemcpyHostToDevice);\n"
        "\n"
        "    dim3 grid({}, {}, 1);\n"
        "    dim3 block({}, {}, {});\n"
        "\n"
        "    int numBlocks;\n"
        "    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, {}, {}, 0);\n"
        '    fprintf(stderr, "Active blocks per SM = %d\\n", numBlocks);\n '
        "\n"
        "    for (int i = 0; i < {}; ++i)\n"
        "    {{\n"
        "        {}<<<grid, block>>>((half*)Ad, (half*)Bd, (half*)Cd);\n"
        "        cudaDeviceSynchronize();\n"
        "    }}\n"
        "\n"
        "    cudaMemcpy(Ch, Cd, output_size * sizeof(half), cudaMemcpyDeviceToHost);\n"
        "\n"
        "    free(Ah);\n"
        "    free(Bh);\n"
        "    free(Ch);\n"
        "\n"
        "    cudaFree(Ad);\n"
        "    cudaFree(Bd);\n"
        "    cudaFree(Cd);\n"
        "\n"
        "    return 0;\n"
        "}}\n".format(
            M,
            K,
            N,
            source,
            grid_x,
            grid_y,
            block_x,
            block_y,
            block_z,
            kernel_name,
            block_x * block_y * block_z,
            times,
            kernel_name,
        )
    )


def get_tc_mm_source(
    A: tvm.te.Tensor, B: tvm.te.Tensor, C: tvm.te.Tensor, rprog: rProg
) -> str:
    if not LatestTVM:
        s = te.create_schedule(C.op)
        s = schedule_tensorcore(s, rprog, None, None, C, False)
        func = tvm.build(s, [A, B, C], "cuda")
        source = func.imported_modules[0].get_source()
    else:
        pf = te.create_prim_func([A, B, C])
        mod = tvm.IRModule({"main": pf})
        # Create a TIR schedule
        s = tvm.tir.Schedule(mod)
        s = schedule_tensorcore(s, rprog, A, B, C, False)
        # Build
        target = tvm.target.Target("cuda")
        func = tvm.build(s.mod, target=target)
        source = func.imported_modules[0].get_source()

    # get rid of prior definitions
    start_pos = source.find("extern")
    return source[start_pos:]


def get_tc_block_size(block_rTile: rTile, warp_rTile: rTile) -> Tuple[int, int, int]:
    block_x = 32
    block_y = block_rTile.SDimensions()[0] // warp_rTile.SDimensions()[0]
    block_z = block_rTile.SDimensions()[1] // warp_rTile.SDimensions()[1]
    return block_x, block_y, block_z


def get_tc_grid_size(M: int, N: int, block_rTile: rTile) -> Tuple[int, int]:
    m = math.ceil(M / block_rTile.SDimensions()[0])
    n = math.ceil(N / block_rTile.SDimensions()[1])
    return m, n
