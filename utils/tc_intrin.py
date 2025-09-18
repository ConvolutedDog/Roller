import tvm
from tvm import te
import sys
from typing import List, Tuple
from utils import LatestTVM
from tvm.script import tir as T


if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

if LatestTVM:
    from tvm.tir import TensorIntrin


def init_intrin_strides(
    wmma_shape: Tuple[int, int],
    warp_s: int,
    block_s: int,
    rstep_size: int,
    offset: int,
    layout: Literal["row_major", "col_major"],
) -> Tuple[List[int], List[int]]:
    wmma_x, wmma_y = wmma_shape
    # warp_m, warp_n = warp_shape
    # block_m, block_n = block_shape
    if layout == "row_major":
        # row-major input tensor
        # block-level subtensor shape = [block_size_m, wmma_k]
        # warp-level subtensor shape = [warp_size_m, wmma_k]
        # fragment shape = [warp_m, wmma_k]
        wmma_m, wmma_k = wmma_x, wmma_y
        F_stride = [wmma_k, 1]
        S_stride = [rstep_size * wmma_k + offset, 1]
    if layout == "col_major":
        # col-major input tensor
        # block-level subtensor shape = [wmma_k, block_size_n]
        # warp-level subtensor shape = [wmma_k, warp_size_n]
        # fragment shape = [wmma_k, warp_n]
        wmma_k, wmma_n = wmma_x, wmma_y
        F_stride = [wmma_n * warp_s, 1]
        S_stride = [wmma_n * warp_s * block_s + offset, 1]
    return F_stride, S_stride


def intrin_wmma_load_matrix(
    wmma_shape: Tuple[int, int, int],
    warp_shape: Tuple[int, int],
    block_shape: Tuple[int, int],
    rstep: int,
    stride_dst: int,
    stride_src: int,
    scope: Literal["wmma.matrix_a", "wmma.matrix_b"],
    layout: Literal["row_major", "col_major"],
    data_dtype: Literal["float16"],
):
    wmma_m, wmma_n, wmma_k = wmma_shape
    warp_m, warp_n = warp_shape
    block_m, block_n = block_shape
    if layout == "row_major":
        buffer_shape = (wmma_m, wmma_k)
        # stride_dst = [wmma_k, 1]
        # stride_src = [wmma_k, 1]
    elif layout == "col_major":
        buffer_shape = (wmma_k, wmma_n)
        # stride_dst = [wmma_n * warp_n, 1]
        # stride_src = [wmma_n * warp_n * block_n, 1]

    A = te.placeholder(buffer_shape, name="A", dtype=data_dtype)
    BA = tvm.tir.decl_buffer(
        A.shape,
        A.dtype,
        scope="shared",
        strides=stride_src,
        data_alignment=32,
        offset_factor=8,
    )
    C = te.compute(buffer_shape, lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(
        C.shape,
        C.dtype,
        scope=scope,
        strides=stride_dst,
        data_alignment=32,
        offset_factor=8,
    )

    if layout == "row_major":
        warp_index = (BC.elem_offset % (warp_m * wmma_m * wmma_k * rstep)) // (
            wmma_m * wmma_k
        )
    elif layout == "col_major":
        warp_index = (BC.elem_offset % (warp_n * wmma_n * rstep)) // wmma_n

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_load_matrix_sync",
                BC.data,
                wmma_m,
                wmma_n,
                wmma_k,
                warp_index,
                BA.access_ptr("r"),
                stride_src[0],
                layout,
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_store_matrix(
    strides_dst: Tuple[int, int],
    strides_from: Tuple[int, int],
    F_shape: Tuple[int, int, int],
    out_dtype: Literal["float16", "float32"],
):
    """Intrin function for storing the results from wmma.accumulator to global"""
    wmma_m, wmma_n, wmma_k = F_shape
    A = te.placeholder((wmma_m, wmma_n), name="A", dtype=out_dtype)
    BA = tvm.tir.decl_buffer(
        A.shape,
        A.dtype,
        scope="wmma.accumulator",
        data_alignment=32,
        offset_factor=8,
        strides=strides_from,
    )
    C = te.compute((wmma_m, wmma_n), lambda *i: A(*i), name="C")
    BC = tvm.tir.decl_buffer(
        C.shape,
        C.dtype,
        scope="global",
        data_alignment=32,
        offset_factor=8,
        strides=strides_dst,
    )

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        BA = ins[0]
        BC = outs[0]
        row = wmma_m * wmma_n
        warp_index = BA.elem_offset // row + BA.elem_offset % row // wmma_n
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_store_matrix_sync",
                BA.data,
                wmma_m,
                wmma_n,
                wmma_k,
                warp_index,
                BC.access_ptr("w"),
                strides_dst[0],
                "row_major",
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_gemm(
    AL_gemm: tvm.te.Tensor,
    BL_gemm: tvm.te.Tensor,
    CL_compute: tvm.te.Tensor,
    strides_A: Tuple[int, int],
    strides_B: Tuple[int, int],
    strides_C: Tuple[int, int],
    shape: Tuple[int, int, int],
):
    """Intrin for wmma fill_fragment and mma_sync
    Parameters
    ----------
    AL_gemm : tvm.te.placeholder
        wmma matrix A
    WL_gemm : tvm.te.placeholder
        wmma matrix B
    CL_compute : tvm.te.compute
        The definition of wmma gemm
    """
    wmma_m, wmma_n, wmma_k = shape
    A = AL_gemm
    B = BL_gemm
    C = CL_compute

    BA = tvm.tir.decl_buffer(
        A.shape,
        A.dtype,
        name="BA",
        scope="wmma.matrix_a",
        data_alignment=32,
        offset_factor=8,
        strides=strides_A,
    )
    BB = tvm.tir.decl_buffer(
        B.shape,
        B.dtype,
        name="BB",
        scope="wmma.matrix_b",
        data_alignment=32,
        offset_factor=8,
        strides=strides_B,
    )
    BC = tvm.tir.decl_buffer(
        C.shape,
        C.dtype,
        name="BC",
        scope="wmma.accumulator",
        data_alignment=32,
        offset_factor=8,
        strides=strides_C,
    )

    def intrin_func(ins, outs):
        BA, BB = ins
        (BC,) = outs

        def warp_idnex(offset, row, col):
            row = row * col
            return offset // row + offset % row // col

        warp_index_A = warp_idnex(BA.elem_offset, wmma_m, wmma_k)
        warp_index_B = warp_idnex(BB.elem_offset, wmma_k, wmma_n)
        warp_index_C = warp_idnex(BC.elem_offset, wmma_m, wmma_n)

        def init():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle",
                    "tir.tvm_fill_fragment",
                    BC.data,
                    wmma_m,
                    wmma_n,
                    wmma_k,
                    warp_index_C,
                    0.0,
                )
            )
            return ib.get()

        def update():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle",
                    "tir.tvm_mma_sync",
                    BC.data,
                    warp_index_C,
                    BA.data,
                    warp_index_A,
                    BB.data,
                    warp_index_B,
                    BC.data,
                    warp_index_C,
                )
            )
            return ib.get()

        return update(), init(), update()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, B: BB, C: BC})


def register_wmma_load_intrin(
    wmma_shape: Tuple[int, int, int],
    warp_shape: Tuple[int, int],
    block_shape: Tuple[int, int],
    rstep: int,
    stride_dst: Tuple[int, int],
    stride_src: Tuple[int, int],
    scope: Literal["wmma.matrix_a", "wmma.matrix_b"],
    layout: Literal["row_major", "col_major"],
    data_dtype: Literal["float16"],
):
    wmma_m, wmma_n, wmma_k = wmma_shape
    warp_m, warp_n = warp_shape
    block_m, block_n = block_shape

    intrin_name = (
        f"wmma_load_{wmma_m}x{wmma_n}x{wmma_k}_{warp_n}x{warp_n}_"
        + f"{block_m}x{block_n}_{rstep}_{stride_dst[0]}x{stride_dst[1]}_"
        + f"{stride_src[0]}x{stride_src[1]}_{scope}_{layout}_{data_dtype}"
    )

    if TensorIntrin.get(intrin_name, allow_missing=True):
        return intrin_name

    if layout == "row_major":
        buffer_shape = (wmma_m, wmma_k)
        read_range = (wmma_m, wmma_k)
        write_range = (wmma_m, wmma_k)
    elif layout == "col_major":
        buffer_shape = (wmma_k, wmma_n)
        read_range = (wmma_k, wmma_n)
        write_range = (wmma_k, wmma_n)

    @T.prim_func
    def wmma_load_desc(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a,
            buffer_shape,
            dtype=data_dtype,
            scope="shared",
            align=32,
            offset_factor=8,
            strides=stride_src,
        )
        C = T.match_buffer(
            c,
            buffer_shape,
            dtype=data_dtype,
            scope=scope,
            align=32,
            offset_factor=8,
            strides=stride_dst,
        )

        with T.block("root"):
            T.reads(A[0 : read_range[0], 0 : read_range[1]])
            T.writes(C[0 : write_range[0], 0 : write_range[1]])
            for i, j in T.grid(buffer_shape[0], buffer_shape[1]):
                with T.block("load"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    T.reads(A[vii, vjj])
                    T.writes(C[vii, vjj])
                    C[vii, vjj] = A[vii, vjj]

    @T.prim_func
    def wmma_load_impl(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a,
            buffer_shape,
            dtype=data_dtype,
            scope="shared",
            align=32,
            offset_factor=8,
            strides=stride_src,
        )
        C = T.match_buffer(
            c,
            buffer_shape,
            dtype=data_dtype,
            scope=scope,
            align=32,
            offset_factor=8,
            strides=stride_dst,
        )

        with T.block("root"):
            T.reads(A[0 : read_range[0], 0 : read_range[1]])
            T.writes(C[0 : write_range[0], 0 : write_range[1]])

            warp_index = T.if_then_else(
                layout == "row_major",
                T.floordiv(
                    T.floormod(C.elem_offset, warp_m * wmma_m * wmma_k * rstep),
                    wmma_m * wmma_k,
                ),
                T.floordiv(T.floormod(C.elem_offset, warp_n * wmma_n * rstep), wmma_n),
            )

            T.evaluate(
                T.tvm_load_matrix_sync(
                    C.data,
                    T.int32(wmma_m),
                    T.int32(wmma_n),
                    T.int32(wmma_k),
                    T.int32(warp_index),
                    A.access_ptr("r"),
                    T.int32(stride_src[0]),
                    T.StringImm(layout),
                    dtype=data_dtype,
                )
            )

    TensorIntrin.register(intrin_name, wmma_load_desc, wmma_load_impl)
    return intrin_name


def register_wmma_fill_intrin(
    wmma_shape: Tuple[int, int, int], strides_C: int, dtype: str
):
    """Register WMMA fill intrinsic for initializing accumulator fragments."""

    wmma_m, wmma_n, wmma_k = wmma_shape
    buffer_shape = (wmma_m, wmma_n)

    intrin_name = f"wmma_fill_{wmma_m}x{wmma_n}x{wmma_k}_{strides_C}_{dtype}"

    if TensorIntrin.get(intrin_name, allow_missing=True):
        return intrin_name

    @T.prim_func
    def wmma_fill_desc(a: T.handle) -> None:
        A = T.match_buffer(
            a,
            buffer_shape,
            dtype=dtype,
            scope="wmma.accumulator",
            align=32,
            offset_factor=8,
            strides=strides_C,
        )
        with T.block("root"):
            T.reads()
            T.writes(A[0:wmma_m, 0:wmma_n])
            for i, j in T.grid(wmma_m, wmma_n):
                with T.block("fill"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    T.writes(A[vii, vjj])
                    A[vii, vjj] = T.cast(0, dtype)

    @T.prim_func
    def wmma_fill_impl(a: T.handle) -> None:
        A = T.match_buffer(
            a,
            buffer_shape,
            dtype=dtype,
            scope="wmma.accumulator",
            align=32,
            offset_factor=8,
            strides=strides_C,
        )
        with T.block("root"):
            T.reads()
            T.writes(A[0:wmma_m, 0:wmma_n])

            # Calculate warp index
            warp_index = T.floordiv(A.elem_offset, wmma_m * wmma_n) + T.floordiv(
                T.floormod(A.elem_offset, wmma_m * wmma_n), wmma_n
            )

            T.evaluate(
                T.tvm_fill_fragment(
                    A.data,
                    wmma_m,
                    wmma_n,
                    wmma_k,
                    warp_index,
                    T.cast(0, dtype),
                    dtype="handle",
                )
            )

    TensorIntrin.register(intrin_name, wmma_fill_desc, wmma_fill_impl)
    return intrin_name


def register_wmma_intrin(
    AL_gemm: tvm.te.Tensor,
    BL_gemm: tvm.te.Tensor,
    CL_compute: tvm.te.Tensor,
    strides_A: Tuple[int, int],
    strides_B: Tuple[int, int],
    strides_C: Tuple[int, int],
    shape: Tuple[int, int, int],
):
    """Intrin for wmma fill_fragment and mma_sync
    Parameters
    ----------
    AL_gemm : tvm.te.placeholder
        wmma matrix A
    WL_gemm : tvm.te.placeholder
        wmma matrix B
    CL_compute : tvm.te.compute
        The definition of wmma gemm
    strides_A : Tuple[int, int]
        The strides of matrix A
    strides_B : Tuple[int, int]
        The strides of matrix B
    strides_C : Tuple[int, int]
        The strides of matrix C
    shape : Tuple[int, int, int]
        The shape of wmma gemm
    """
    wmma_m, wmma_n, wmma_k = shape
    A = AL_gemm
    B = BL_gemm
    C = CL_compute

    intrin_name = (
        f"wmma_{wmma_m}x{wmma_n}x{wmma_k}_{strides_A[0]}x{strides_A[1]}"
        + f"_{strides_B[0]}x{strides_B[1]}_{strides_C[0]}x{strides_C[1]}"
    )

    if TensorIntrin.get(intrin_name, allow_missing=True):
        return intrin_name

    @T.prim_func
    def wmma_gemm_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
        A_buffer = T.match_buffer(
            a,
            (wmma_m, wmma_k),
            dtype=A.dtype,
            scope="wmma.matrix_a",
            align=32,
            offset_factor=8,
            strides=strides_A,
        )
        B_buffer = T.match_buffer(
            b,
            (wmma_k, wmma_n),
            dtype=B.dtype,
            scope="wmma.matrix_b",
            align=32,
            offset_factor=8,
            strides=strides_B,
        )
        C_buffer = T.match_buffer(
            c,
            (wmma_m, wmma_n),
            dtype=C.dtype,
            scope="wmma.accumulator",
            align=32,
            offset_factor=8,
            strides=strides_C,
        )

        with T.block("root"):
            T.reads(
                C_buffer[0:wmma_m, 0:wmma_n],
                A_buffer[0:wmma_m, 0:wmma_k],
                B_buffer[0:wmma_k, 0:wmma_n],
            )
            T.writes(C_buffer[0:wmma_m, 0:wmma_n])
            for i, j, k in T.grid(wmma_m, wmma_n, wmma_k):
                with T.block("wmma"):
                    vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                    T.reads(C_buffer[vii, vjj], A_buffer[vii, vkk], B_buffer[vkk, vjj])
                    T.writes(C_buffer[vii, vjj])
                    with T.init():
                        C_buffer[vii, vjj] = T.cast(0, C.dtype)
                    C_buffer[vii, vjj] = (
                        C_buffer[vii, vjj] + A_buffer[vii, vkk] * B_buffer[vkk, vjj]
                    )

    @T.prim_func
    def wmma_gemm_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
        A_buffer = T.match_buffer(
            a,
            (wmma_m, wmma_k),
            dtype=A.dtype,
            scope="wmma.matrix_a",
            align=32,
            offset_factor=8,
            strides=strides_A,
        )
        B_buffer = T.match_buffer(
            b,
            (wmma_k, wmma_n),
            dtype=B.dtype,
            scope="wmma.matrix_b",
            align=32,
            offset_factor=8,
            strides=strides_B,
        )
        C_buffer = T.match_buffer(
            c,
            (wmma_m, wmma_n),
            dtype=C.dtype,
            scope="wmma.accumulator",
            align=32,
            offset_factor=8,
            strides=strides_C,
        )

        with T.block("root"):
            T.reads(
                C_buffer[0:wmma_m, 0:wmma_n],
                A_buffer[0:wmma_m, 0:wmma_k],
                B_buffer[0:wmma_k, 0:wmma_n],
            )
            T.writes(C_buffer[0:wmma_m, 0:wmma_n])

            warp_index_A = T.floordiv(
                A_buffer.elem_offset, wmma_m * wmma_k
            ) + T.floordiv(T.floormod(A_buffer.elem_offset, wmma_m * wmma_k), wmma_k)
            warp_index_B = T.floordiv(
                B_buffer.elem_offset, wmma_k * wmma_n
            ) + T.floordiv(T.floormod(B_buffer.elem_offset, wmma_k * wmma_n), wmma_n)
            warp_index_C = T.floordiv(
                C_buffer.elem_offset, wmma_m * wmma_n
            ) + T.floordiv(T.floormod(C_buffer.elem_offset, wmma_m * wmma_n), wmma_n)

            # Fill fragment with zeros
            # T.evaluate(
            #     T.tvm_fill_fragment(
            #         C_buffer.data,
            #         wmma_m,
            #         wmma_n,
            #         wmma_k,
            #         warp_index_C,
            #         T.cast(0, C.dtype),
            #         dtype="handle",
            #     )
            # )

            # MMA sync operation
            T.evaluate(
                T.tvm_mma_sync(
                    C_buffer.data,
                    warp_index_C,
                    A_buffer.data,
                    warp_index_A,
                    B_buffer.data,
                    warp_index_B,
                    C_buffer.data,
                    warp_index_C,
                    dtype="handle",
                )
            )

    TensorIntrin.register(intrin_name, wmma_gemm_desc, wmma_gemm_impl)
    return intrin_name


def register_wmma_store_intrin(
    strides_dst: Tuple[int, int],
    strides_from: Tuple[int, int],
    F_shape: Tuple[int, int, int],
    out_dtype: Literal["float16", "float32"],
):
    wmma_m, wmma_n, wmma_k = F_shape
    buffer_shape = (wmma_m, wmma_n)
    read_range = buffer_shape
    write_range = buffer_shape

    intrin_name = (
        f"wmma_store_{wmma_m}x{wmma_n}x{wmma_k}_{strides_dst[0]}"
        + f"x{strides_dst[1]}_{strides_from[0]}x{strides_from[1]}_{out_dtype}_row_major"
    )

    if TensorIntrin.get(intrin_name, allow_missing=True):
        return intrin_name

    @T.prim_func
    def wmma_store_desc(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a,
            buffer_shape,
            dtype=out_dtype,
            scope="wmma.accumulator",
            align=32,
            offset_factor=8,
            strides=strides_from,
        )
        C = T.match_buffer(
            c,
            buffer_shape,
            dtype=out_dtype,
            scope="global",
            align=32,
            offset_factor=8,
            strides=strides_dst,
        )

        with T.block("root"):
            T.reads(A[0 : read_range[0], 0 : read_range[1]])
            T.writes(C[0 : write_range[0], 0 : write_range[1]])
            for i, j in T.grid(buffer_shape[0], buffer_shape[1]):
                with T.block("store"):
                    vii, vjj = T.axis.remap("SS", [i, j])
                    T.reads(A[vii, vjj])
                    T.writes(C[vii, vjj])
                    C[vii, vjj] = A[vii, vjj]

    @T.prim_func
    def wmma_store_impl(a: T.handle, c: T.handle) -> None:
        A = T.match_buffer(
            a,
            buffer_shape,
            dtype=out_dtype,
            scope="wmma.accumulator",
            align=32,
            offset_factor=8,
            strides=strides_from,
        )
        C = T.match_buffer(
            c,
            buffer_shape,
            dtype=out_dtype,
            scope="global",
            align=32,
            offset_factor=8,
            strides=strides_dst,
        )

        with T.block("root"):
            T.reads(A[0 : read_range[0], 0 : read_range[1]])
            T.writes(C[0 : write_range[0], 0 : write_range[1]])

            warp_index = T.floordiv(A.elem_offset, wmma_m * wmma_n) + T.floordiv(
                T.floormod(A.elem_offset, wmma_m * wmma_n), wmma_n
            )

            T.evaluate(
                T.tvm_store_matrix_sync(
                    A.data,
                    T.int32(wmma_m),
                    T.int32(wmma_n),
                    T.int32(wmma_k),
                    T.int32(warp_index),
                    C.access_ptr("w"),
                    T.int32(strides_dst[0]),
                    T.StringImm("row_major"),
                    dtype=out_dtype,
                )
            )

    TensorIntrin.register(intrin_name, wmma_store_desc, wmma_store_impl)
    return intrin_name
