"""
codegen implementation for rprog
"""

from os import close
import tvm
from tvm import te
import numpy as np
import math
import copy
from .tc_intrin import (
    init_intrin_strides,
    intrin_wmma_load_matrix,
    intrin_wmma_store_matrix,
    intrin_wmma_gemm,
)
from utils import LatestTVM, get_axis_names, get_blocks


class CodeGeneratorR:
    def get_codegen_dict(self, rprog):
        """
        convert a rprog to tiling, results stored to self.tiling
        """
        self.tiling = {}
        for axis_name in rprog.saxis:
            self.tiling[axis_name] = []
            axis_cfg = rprog.GetAxisConfig(axis_name)
            for i in range(rprog.num_level):
                self.tiling[axis_name].append(math.ceil(axis_cfg[i] / axis_cfg[i + 1]))
        for axis_name in rprog.raxis:
            self.tiling[axis_name] = []
            axis_cfg = rprog.GetAxisConfig(axis_name)
            for i in range(rprog.num_level):
                self.tiling[axis_name].append(math.ceil(axis_cfg[i] / axis_cfg[i + 1]))
        return self.tiling

    def split_axis(
        self,
        op,
        axis,
        sche=None,
        target_stage="compute_local",
        all_tensors=None,
        split_reduce=False,
    ):
        if sche == None:
            sche = self.sche

        factors = None

        # If axis is loop_var, it has no attr 'var'.
        if (
            not isinstance(axis, tvm.tir.schedule.schedule.LoopRV)
            and axis.var.name in self.tiling
        ):
            factors = self.tiling[axis.var.name]
        elif (
            isinstance(axis, tvm.tir.schedule.schedule.LoopRV)
            and str(self.sche.get(axis).loop_var) in self.tiling
        ):
            factors = self.tiling[str(self.sche.get(axis).loop_var)]
        elif (
            isinstance(axis, tvm.tir.schedule.schedule.LoopRV)
            and str(self.sche.get(axis).loop_var) not in self.tiling
        ):
            if not hasattr(self, "iter_vars_map"):
                # Should rename it to block_compute_local
                block_compute_local = self.sche.get_block(target_stage)
                loops = self.sche.get_loops(block_compute_local)

                iter_vars_compute_local = [
                    self.sche.get(loops[i]).loop_var for i in range(len(loops))
                ]
                iter_vars_compute = [
                    self.sche.get(
                        self.sche.get_loops(
                            self.sche.get_block(target_stage.split("_local")[0])
                        )[i]
                    ).loop_var
                    for i in range(
                        len(
                            self.sche.get_loops(
                                self.sche.get_block(target_stage.split("_local")[0])
                            )
                        )
                    )
                ]

                self.iter_vars_map = {}
                self.iter_vars_map_reverse = {}
                for i in range(len(iter_vars_compute_local)):
                    var = iter_vars_compute_local[i]
                    self.iter_vars_map[iter_vars_compute[i].name] = var.name
                    self.iter_vars_map_reverse[var.name] = iter_vars_compute[i].name

            if not split_reduce:
                # Only split space axes
                factor_key = self.iter_vars_map_reverse[
                    str(self.sche.get(axis).loop_var)
                ]
                factors = self.tiling[str(factor_key)]
            else:
                # Only split reduce axes
                for key in self.tiling.keys():
                    if not key in self.iter_vars_map_reverse.values():
                        # NOTE: This can only be useful when there is only one reduce axis.
                        factor_key = key
                        break
                factors = self.tiling[factor_key]
        else:
            raise Exception("Can't find factors")

        assert factors is not None

        if LatestTVM:
            # Should rename it to block_compute_local
            block_compute_local = self.sche.get_block(target_stage)
            loops = self.sche.get_loops(block_compute_local)

            target_loop = axis
            if target_loop is None:
                raise ValueError(f"Unfound axis")

            if not isinstance(axis, tvm.tir.schedule.schedule.LoopRV):
                new_factors = [
                    math.ceil(int(axis.dom.extent) / int(np.prod(factors))),
                ] + factors[:]
            else:
                new_factors = [
                    math.ceil(
                        int(self.sche.get(target_loop).extent) / int(np.prod(factors))
                    ),
                ] + factors[:]

            axises = sche.split(target_loop, factors=new_factors)

            return axises

        else:
            ret = []

            for i in range(0, len(factors)):
                ax0, ax1 = sche[op].split(axis, factor=int(np.prod(factors[i:])))
                ret.append(ax0)
                axis = ax1
            return ret + [axis]

    def update_thread_per_block(self, stage, sche=None, vthread=True):
        num = 1
        if LatestTVM:
            saxis, _ = get_axis_names(stage)  # stage is out
            for name in saxis:
                num = num * self.tiling[name][1 if vthread else 0]
        else:
            if sche == None:
                sche = self.sche
            for axis in sche[stage].op.axis:
                num = num * self.tiling[axis.var.name][1 if vthread else 0]
        self.thread_per_block = num

    def cooperative_fetch(
        self, shared, sch, old_axis=None, shared_fetch_vectorize=False
    ):
        if LatestTVM:
            new_looprvs = [
                self.sche.get_loops(shared)[i]
                for i in range(len(self.sche.get_loops(shared)))
            ]
            fused_loops = []
            for looprv in new_looprvs:
                loop = sch.get(looprv)
                if str(loop.loop_var) in old_axis:
                    fused_loops.append(looprv)

            fused = sch.fuse(*fused_loops)

            if not isinstance(sch.get(fused).extent, int):
                assert self.bank_size % 4 == 0 and self.bank_size >= 4
                if not shared_fetch_vectorize:
                    new_factors = [None, self.bank_size // 4]
                else:
                    new_factors = [None, 4]
            else:
                raise NotImplementedError("Not implemented.")
                new_factors = [
                    math.ceil(int(sch.get(fused).extent) / int(self.bank_size // 4)),
                ] + [
                    self.bank_size // 4
                ]  # right?

            fused, ii_n = sch.split(fused, factors=new_factors)

            if not isinstance(sch.get(fused).extent, int):
                new_factors = [None, self.thread_per_block]
            else:
                new_factors = [
                    math.ceil(int(sch.get(fused).extent) / self.thread_per_block),
                ] + [
                    self.thread_per_block
                ]  # right?
            oo, ii = sch.split(fused, factors=new_factors)

            sch.vectorize(ii_n)
            sch.reorder(oo, ii, ii_n)
            sch.unroll(oo)
            sch.bind(ii, thread_axis="threadIdx.x")

        else:
            axes = sch[shared].op.axis
            fused = sch[shared].fuse(*axes)
            if not shared_fetch_vectorize:
                fused, ii_n = sch[shared].split(fused, factor=self.bank_size // 4)
            else:
                fused, ii_n = sch[shared].split(fused, factor=4)
            oo, ii = sch[shared].split(fused, factor=self.thread_per_block)
            # ii, ii_n = sch[shared].split(ii, factor=2)
            sch[shared].vectorize(ii_n)
            sch[shared].reorder(oo, ii, ii_n)
            sch[shared].unroll(oo)
            # sch[shared].unroll(ii_n)
            sch[shared].bind(ii, te.thread_axis("threadIdx.x"))

    def calc_grid(self, reduce_iters, space_iters, vthread=True):
        blck_dict = {"blockIdx.x": 1, "blockIdx.y": 1, "blockIdx.z": 1}
        thrd_dict = {"threadIdx.x": 1, "threadIdx.y": 1, "threadIdx.z": 1}

        for iter in space_iters:
            if iter.var.name in self.tiling:
                factors = self.tiling[iter.var.name]
                length = iter.dom.extent
                blck = max(length // int(np.prod(factors[0:])), 1)
                thrd = factors[1 if vthread else 0]
                if self.binding["space"][0] in blck_dict:
                    blck_dict[self.binding["space"][0]] *= blck
                if self.binding["space"][2 if vthread else 1] in thrd_dict:
                    thrd_dict[self.binding["space"][2 if vthread else 1]] *= thrd

        for iter in reduce_iters:
            if iter.var.name in self.tiling:
                factors = self.tiling[iter.var.name]
                length = iter.dom.extent
                blck = max(length // int(np.prod(factors[0:])), 1)
                thrd = factors[0]
                if self.binding["reduce"][0] in blck_dict:
                    blck_dict[self.binding["reduce"][0]] *= blck
                if self.binding["reduce"][1] in thrd_dict:
                    thrd_dict[self.binding["reduce"][1]] *= thrd

        self.blck_grid = [
            blck_dict["blockIdx.x"],
            blck_dict["blockIdx.y"],
            blck_dict["blockIdx.z"],
        ]
        self.thrd_grid = [
            thrd_dict["threadIdx.x"],
            thrd_dict["threadIdx.y"],
            thrd_dict["threadIdx.z"],
        ]

    def adjust_format(self, out):
        if LatestTVM:
            saxis, raxis = get_axis_names(out)
            for name in saxis:
                if len(self.tiling[name]) == 2:
                    vthrd = self.tiling[name][1]
                    thrd = self.tiling[name][0]
                    self.tiling[name] = [vthrd, thrd, 1]
        else:
            for axis in self.sche[out].op.axis:
                name = axis.var.name
                if len(self.tiling[name]) == 2:
                    vthrd = self.tiling[name][1]
                    thrd = self.tiling[name][0]
                    self.tiling[name] = [vthrd, thrd, 1]

    # [Parameters]
    #   schedule: the original TVM schedule of an op
    #   rprog: roller's rprog configuration
    #   smem_bool: True if we need tiling at shared memory
    #   reg_bool: True if we need tiling at register files
    #
    # [Return]
    #   new_s: an optimized TVM schedule
    def rewrite_schedule(
        self,
        schedule,
        rprog,
        smem_bool,
        reg_bool,
        target_stage="compute",
        align_info=[],
        bank_size=4,
        in_tensors=None,
        out_tensors=None,
        shared_fetch_vectorize=False,
        codegen_input_reg_tiling=False,
    ):
        self.bank_size = bank_size
        self.binding = {
            "space": ["blockIdx.x", "vthread", "threadIdx.x"],
            "reduce": [None, None],
        }
        self.get_codegen_dict(rprog)
        print("Tiling of rewrite_schedule: ", self.tiling)
        self.need_smem_tiling = smem_bool
        self.need_reg_tiling = reg_bool
        self.sche = schedule

        input_tensors = []
        output_num = 0
        output_tensors = []

        if LatestTVM:
            assert in_tensors is not None and out_tensors is not None
            for tensor in in_tensors + out_tensors:
                if isinstance(tensor.op, tvm.te.tensor.ComputeOp):
                    if tensor.op.name != target_stage:
                        # NOTE: This makes no sense and will never be executed, so it
                        # should be removed later.
                        # NOTE: The code related to the following code is moved to the
                        # end of rewrite_schedule function. You can search this:
                        #     for block_name in old_blocks:
                        #         if block_name not in op_names:
                        #             self.sche.compute_inline(block_name)
                        self.sche.compute_inline(self.sche.get_block(tensor.op.name))
                    else:
                        input_tensors = list(tensor.op.input_tensors)
                        output_tensors.append(tensor)

            op_names = [tensor.op.name for tensor in in_tensors + out_tensors]
            old_blocks = get_blocks(self.sche, without_root=True).keys()
        else:
            for item in self.sche.stage_map.items():
                if isinstance(item[0], tvm.te.tensor.ComputeOp):
                    output_num = item[0].num_outputs
                    for i in range(output_num):
                        if item[0].name != target_stage:
                            out = item[0].output(i)
                            self.sche[out].compute_inline()
                        else:
                            input_tensors = list(item[0].input_tensors)
                            output_tensors.append(item[0].output(i))

        for out in output_tensors:
            self.adjust_format(out)
            # TVM only allows binding reduce axis if it's the only one
            if self.binding["reduce"][1] is not None:
                assert len(self.sche[out].op.reduce_axis) == 1

            self.update_thread_per_block(out)
            if LatestTVM:
                saxis = list(out.op.axis)
                raxis = list(out.op.reduce_axis)
                all_iters = tvm.ir.container.Array(saxis + raxis)
                reduce_iters = tvm.ir.container.Array(raxis)
                space_iters = [
                    all_iters[i]
                    for i in range(len(all_iters))
                    if all_iters[i] not in raxis
                ]
            else:
                all_iters = self.sche[out].all_iter_vars
                reduce_iters = out.op.reduce_axis
                space_iters = list(set(all_iters) - set(reduce_iters))
            self.calc_grid(reduce_iters, space_iters)

            smem_tensor = []
            reg_tensor = []
            reg_tile = None

            if LatestTVM:
                if self.need_smem_tiling:
                    read_buffer_index = 0
                    for input_tensor in input_tensors:
                        block_compute = self.sche.get_block(target_stage)
                        shared_tensor = self.sche.cache_read(
                            block_compute, read_buffer_index, "shared"
                        )
                        read_buffer_index += 1
                        smem_tensor.append(shared_tensor)
                if self.need_reg_tiling:
                    read_buffer_index = 0
                    if codegen_input_reg_tiling:
                        for input_tensor in input_tensors:
                            block_compute = self.sche.get_block(target_stage)
                            local_tensor = self.sche.cache_read(
                                block_compute, read_buffer_index, "local"
                            )
                            read_buffer_index += 1
                            reg_tensor.append(local_tensor)
                    reg_tile = self.sche.cache_write(
                        block_compute, write_buffer_index=0, storage_scope="local"
                    )
            else:
                if self.need_smem_tiling:
                    for input_tensor in input_tensors:
                        shared_tensor = self.sche.cache_read(
                            input_tensor, "shared", [out]
                        )
                        smem_tensor.append(shared_tensor)
                if self.need_reg_tiling:
                    if codegen_input_reg_tiling:
                        for shared_tensor in smem_tensor:
                            local_tensor = self.sche.cache_read(
                                shared_tensor, "local", [out]
                            )
                            reg_tensor.append(local_tensor)
                    reg_tile = self.sche.cache_write(out, "local")

            blck_axis = []
            vthd_axis = []
            thrd_axis = []
            tile_axis = []

            if not LatestTVM:
                for axis in self.sche[out].op.axis:
                    # adjust self.tiling's space axis for proper smem load
                    # TODO: what if not two-level tiling structure?
                    if self.bank_size != 4:
                        assert len(self.tiling[axis.var.name]) == 3
                        if self.tiling[axis.var.name][-3] >= (self.bank_size // 4):
                            self.tiling[axis.var.name][-3] = self.tiling[axis.var.name][
                                -3
                            ] // (self.bank_size // 4)
                            self.tiling[axis.var.name][-1] = self.tiling[axis.var.name][
                                -1
                            ] * (self.bank_size // 4)
                        else:
                            print("Shared mem tiling is too small.")
                            self.tiling[axis.var.name][-1] = (
                                self.tiling[axis.var.name][-1]
                                * self.tiling[axis.var.name][-3]
                            )
                            self.tiling[axis.var.name][-3] = 1
                        print("Updated self.tiling: ", self.tiling)

                    bx, vx, tx, tn = self.split_axis(
                        out,
                        axis,
                        sche=None,
                        target_stage=target_stage + ("_local" if reg_bool else ""),
                        all_tensors=in_tensors + out_tensors,
                    )

                    blck_axis.append(bx)
                    vthd_axis.append(vx)
                    thrd_axis.append(tx)
                    tile_axis.append(tn)
            else:
                block_compute = self.sche.get_block(
                    target_stage + ("_local" if reg_bool else "")
                )
                looprvs = self.sche.get_loops(block_compute)

                iter_vars = self.sche.get(block_compute).iter_vars
                iter_types = [iter_var.iter_type for iter_var in iter_vars]

                # Now the IRModule is the initial state so we can get the iter_type using the loop index.
                assert len(looprvs) == len(iter_vars)

                for looprv in looprvs:
                    # Only split the space axis
                    if iter_types[looprvs.index(looprv)] == tvm.tir.IterVar.CommReduce:
                        continue

                    loop_var = str(self.sche.get(looprv).loop_var)
                    if self.bank_size != 4:
                        raise NotImplementedError(
                            "Only support bank size 4 for LatestTVM"
                        )
                        assert len(self.tiling[loop_var]) == 3
                        if self.tiling[loop_var][-3] >= (self.bank_size // 4):
                            self.tiling[loop_var][-3] = self.tiling[loop_var][-3] // (
                                self.bank_size // 4
                            )
                            self.tiling[loop_var][-1] = self.tiling[loop_var][-1] * (
                                self.bank_size // 4
                            )
                        else:
                            print("Shared mem tiling is too small.")
                            self.tiling[loop_var][-1] = (
                                self.tiling[loop_var][-1] * self.tiling[loop_var][-3]
                            )
                            self.tiling[loop_var][-3] = 1
                        print("Updated self.tiling: ", self.tiling)

                    bx, vx, tx, tn = self.split_axis(
                        None,
                        looprv,
                        sche=None,
                        target_stage=target_stage + ("_local" if reg_bool else ""),
                        all_tensors=in_tensors + out_tensors,
                    )

                    blck_axis.append(bx)
                    vthd_axis.append(vx)
                    thrd_axis.append(tx)
                    tile_axis.append(tn)

            axis_order = blck_axis + vthd_axis + thrd_axis + tile_axis

            if LatestTVM:
                self.sche.reorder(*axis_order)
            else:
                self.sche[out].reorder(*axis_order)

            if LatestTVM:
                blck_fused = self.sche.fuse(*blck_axis)
                thrd_fused = self.sche.fuse(*thrd_axis)
            else:
                blck_fused = self.sche[out].fuse(*blck_axis)
                thrd_fused = self.sche[out].fuse(*thrd_axis)

            if self.binding["space"][0] is not None:
                if LatestTVM:
                    self.sche.bind(blck_fused, self.binding["space"][0])
                else:
                    self.sche[out].bind(
                        blck_fused, te.thread_axis(self.binding["space"][0])
                    )

            if self.binding["space"][1] is not None:
                if LatestTVM:
                    vthd_map = {0: ".x", 1: ".y", 2: ".z"}
                    for i in range(len(vthd_axis)):
                        va = vthd_axis[i]
                        self.sche.bind(va, self.binding["space"][1] + vthd_map[i])
                else:
                    for va in vthd_axis:
                        self.sche[out].bind(
                            va, te.thread_axis(self.binding["space"][1])
                        )

            if self.binding["space"][2] is not None:
                if LatestTVM:
                    self.sche.bind(thrd_fused, self.binding["space"][2])
                else:
                    self.sche[out].bind(
                        thrd_fused, te.thread_axis(self.binding["space"][2])
                    )

            reduce_axis = []
            if reg_tile is not None:
                if LatestTVM:
                    self.sche.compute_at(target_stage, thrd_fused)

                    space_axis = []
                    block_compute = self.sche.get_block(target_stage)

                    loops = self.sche.get_loops(block_compute)

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
                    #   - target_stage += "_update":
                    #     * Labels the current scheduling focus as the update phase
                    #     * Ensures subsequent optimizations apply to right update phase only
                    self.sche.get(
                        self.sche.decompose_reduction(block_compute, loops[3])
                    )
                    target_stage += "_update"

                    for i in range(len(loops)):
                        if tvm.tir.ForKind.SERIAL == self.sche.get(loops[i]).kind:
                            reduce_axis.append(loops[i])
                        else:
                            assert self.sche.get(loops[i]).kind in [
                                tvm.tir.ForKind.UNROLLED,
                                tvm.tir.ForKind.VECTORIZED,
                                tvm.tir.ForKind.PARALLEL,
                                tvm.tir.ForKind.THREAD_BINDING,
                            ]
                            space_axis.append(loops[i])

                    new_reduce_axis = []

                    for axis in reduce_axis.copy():
                        if self.sche.get(axis).extent == 1:
                            continue
                        res = self.split_axis(
                            None,
                            axis,
                            sche=None,
                            target_stage=target_stage,
                            all_tensors=in_tensors + out_tensors,
                            split_reduce=True,
                        )
                        reduce_axis = reduce_axis + res
                        new_reduce_axis = new_reduce_axis + res

                    axis_order = new_reduce_axis + space_axis

                    # self.sche.reorder(*axis_order) # TODO: ERROR
                    # space_fused = self.sche.fuse(*space_axis) # TODO: ERROR
                    # self.sche.unroll(space_fused) # TODO: ERROR
                else:
                    self.sche[reg_tile].compute_at(self.sche[out], thrd_fused)
                    space_axis = []
                    for axis in self.sche[reg_tile].op.axis:
                        space_axis.append(axis)
                    for axis in self.sche[reg_tile].op.reduce_axis:
                        res = self.split_axis(reg_tile, axis)
                        reduce_axis = reduce_axis + res
                    axis_order = reduce_axis + space_axis
                    self.sche[reg_tile].reorder(*axis_order)
                    space_fused = self.sche[reg_tile].fuse(*space_axis)
                    self.sche[reg_tile].unroll(space_fused)
            else:
                if LatestTVM:
                    space_axis = []
                    block_compute = self.sche.get_block(target_stage)

                    loops = self.sche.get_loops(block_compute)

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
                    #   - target_stage += "_update":
                    #     * Labels the current scheduling focus as the update phase
                    #     * Ensures subsequent optimizations apply to right update phase only
                    self.sche.get(
                        self.sche.decompose_reduction(block_compute, loops[3])
                    )
                    target_stage += "_update"

                    block_compute = self.sche.get_block(target_stage)
                    iter_vars = self.sche.get(block_compute).iter_vars
                    iter_types = [iter_var.iter_type for iter_var in iter_vars]

                    for i in range(len(loops)):
                        if i >= len(loops) - len(iter_vars):
                            if (
                                iter_types[i - (len(loops) - len(iter_vars))]
                                == tvm.tir.IterVar.CommReduce
                            ):
                                reduce_axis.append(loops[i])
                            else:
                                space_axis.append(loops[i])
                        else:
                            if tvm.tir.ForKind.SERIAL == self.sche.get(loops[i]).kind:
                                reduce_axis.append(loops[i])
                            else:
                                assert self.sche.get(loops[i]).kind in [
                                    tvm.tir.ForKind.UNROLLED,
                                    tvm.tir.ForKind.VECTORIZED,
                                    tvm.tir.ForKind.PARALLEL,
                                    tvm.tir.ForKind.THREAD_BINDING,
                                ]
                                space_axis.append(loops[i])

                    new_reduce_axis = []

                    for axis in reduce_axis.copy():
                        if self.sche.get(axis).extent == 1:
                            continue
                        res = self.split_axis(
                            None,
                            axis,
                            sche=None,
                            target_stage=target_stage,
                            all_tensors=in_tensors + out_tensors,
                        )
                        reduce_axis = reduce_axis + res
                        new_reduce_axis = new_reduce_axis + res

                    # axis_order = reduce_axis + space_axis
                    axis_order = new_reduce_axis + space_axis

                    # self.sche.reorder(*axis_order) # TODO: ERROR
                    # space_fused = self.sche.fuse(*space_axis) # TODO: ERROR
                    # self.sche.unroll(space_fused) # TODO: ERROR
                else:
                    for axis in self.sche[out].op.reduce_axis:
                        res = self.split_axis(out, axis)
                        reduce_axis = reduce_axis + res
                    if self.binding["reduce"][1] is not None:
                        bind_idx = te.thread_axis(self.binding["reduce"][1])
                        self.sche[out].bind(reduce_axis[1], bind_idx)
                        self.sche[out].set_store_predicate(bind_idx.var.equal(0))

            if reg_tile is not None:
                if LatestTVM:
                    for rt in reg_tensor:
                        self.sche.compute_at(rt, new_reduce_axis[-1])
                    for st in smem_tensor:
                        old_axis = [
                            str(self.sche.get(self.sche.get_loops(st)[i]).loop_var)
                            for i in range(len(self.sche.get_loops(st)))
                        ]
                        self.sche.compute_at(st, new_reduce_axis[0])
                        self.cooperative_fetch(
                            st,
                            self.sche,
                            old_axis,
                            shared_fetch_vectorize=shared_fetch_vectorize,
                        )
                else:
                    for rt in reg_tensor:
                        self.sche[rt].compute_at(self.sche[reg_tile], reduce_axis[-1])
                    for st in smem_tensor:
                        self.sche[st].compute_at(self.sche[reg_tile], reduce_axis[0])
                        self.cooperative_fetch(
                            st,
                            self.sche,
                            None,
                            shared_fetch_vectorize=shared_fetch_vectorize,
                        )
            else:
                if LatestTVM:
                    for rt in reg_tensor:
                        self.sche.compute_at(rt, new_reduce_axis[-1])
                    for st in smem_tensor:
                        old_axis = [
                            str(self.sche.get(self.sche.get_loops(st)[i]).loop_var)
                            for i in range(len(self.sche.get_loops(st)))
                        ]
                        self.sche.compute_at(st, new_reduce_axis[0])
                        self.cooperative_fetch(
                            st,
                            self.sche,
                            old_axis,
                            shared_fetch_vectorize=shared_fetch_vectorize,
                        )
                else:
                    for rt in reg_tensor:
                        self.sche[rt].compute_at(self.sche[out], reduce_axis[-1])
                    for st in smem_tensor:
                        self.sche[st].compute_at(self.sche[out], reduce_axis[0])
                        self.cooperative_fetch(
                            st,
                            self.sche,
                            None,
                            shared_fetch_vectorize=shared_fetch_vectorize,
                        )

        if LatestTVM:
            for block_name in old_blocks:
                if block_name not in op_names:
                    self.sche.compute_inline(block_name)

        for info in align_info:
            if LatestTVM:
                raise NotImplementedError("Latest TVM doesn't support align info")
            else:
                idx, factor, offset = info
                st = smem_tensor[idx]
                # st_size = tvm.runtime.DataType(st.dtype).bits // 8
                # num_ele = bank_size // st_size
                # assert num_ele > 0
                # factor = factor * num_ele
                # offset = math.ceil(offset/num_ele) * num_ele
                self.sche[st].storage_align(st.op.axis[-2], factor, offset)

        return self.sche

    # [Parameters]
    #   schedule: the original TVM schedule of an op
    #   tile_dict: a dictionary holding split factors of each axis,
    #              e.g., {"i": [8, 16, 1], "j": [8, 16, 1], "k": [32]}.
    #              For spacial axes, the format is "axis_name": [thread_tile_size, thread_num, 1].
    #              For reduce axes, the format is "axis_name": [step_size].
    #   bind_dict: a dictionary indicating which GPU index an axis should be bound to.
    #              Since we'll fuse spatial and reduction axes respectively, it's sufficient
    #              to just provide binding information for spatial and reduction axes,
    #              e.g., {"space": ["blockIdx.x", "threadIdx.y", None], "reduce": [None, "threadIdx.x"]}.
    #   smem_bool: True if we need tiling at shared memory
    #   reg_bool: True if we need tiling at register files
    #
    # [Return]
    #   new_s: an optimized TVM schedule

    def rewrite_schedule_fuse(
        self,
        schedule,
        rprog,
        smem_bool,
        reg_bool,
        input_tensors,
        output_tensors,
        write_tensor,
        target_stage="conv2d_nchw_implicit_gemm",
        write_stage="output",
        align_info=[],
        bank_size=4,
    ):
        # self.storage_align_on = st_align
        self.bank_size = bank_size
        # self.bank_number = bank_number
        self.binding = {
            "space": ["blockIdx.x", "vthread", "threadIdx.x"],
            "reduce": [None, None],
        }
        self.get_codegen_dict(rprog)
        print("Tiling of rewrite_schedule: ", self.tiling)
        self.need_smem_tiling = smem_bool
        self.need_reg_tiling = reg_bool
        self.sche = schedule
        # align_info = self.get_align_info_fuse(schedule, rprog, smem_bool, reg_bool, target_stage, st_align, bank_size, bank_number)

        for out in output_tensors:
            self.adjust_format(out)
            # TVM only allows binding reduce axis if it's the only one
            if self.binding["reduce"][1] is not None:
                assert len(self.sche[out].op.reduce_axis) == 1

            self.update_thread_per_block(out)
            all_iters = self.sche[out].all_iter_vars
            reduce_iters = out.op.reduce_axis
            space_iters = list(set(all_iters) - set(reduce_iters))
            self.calc_grid(reduce_iters, space_iters)

            smem_tensor = []
            reg_tensor = []
            reg_tile = self.sche.cache_write(out, "local")

            if self.need_smem_tiling:
                for input_tensor in input_tensors:
                    self.sche[input_tensor].compute_inline()
                    shared_tensor = self.sche.cache_read(
                        input_tensor, "shared", [reg_tile]
                    )
                    smem_tensor.append(shared_tensor)

                for shared_tensor in smem_tensor:
                    local_tensor = self.sche.cache_read(
                        shared_tensor, "local", [reg_tile]
                    )
                    reg_tensor.append(local_tensor)

            blck_axis = []
            vthd_axis = []
            thrd_axis = []
            tile_axis = []
            self.sche[out].compute_inline()
            out = write_tensor
            for axis in self.sche[out].op.axis:
                if self.bank_size != 4:
                    assert len(self.tiling[axis.var.name]) == 3
                    if self.tiling[axis.var.name][-3] >= (self.bank_size // 4):
                        self.tiling[axis.var.name][-3] = self.tiling[axis.var.name][
                            -3
                        ] // (self.bank_size // 4)
                        self.tiling[axis.var.name][-1] = self.tiling[axis.var.name][
                            -1
                        ] * (self.bank_size // 4)
                    else:
                        print("Shared mem tiling is too small.")
                        self.tiling[axis.var.name][-1] = (
                            self.tiling[axis.var.name][-1]
                            * self.tiling[axis.var.name][-3]
                        )
                        self.tiling[axis.var.name][-3] = 1
                    print("Updated self.tiling: ", self.tiling)
                bx, vx, tx, tn = self.split_axis(out, axis)
                # bx, tx, tn = self.split_axis(out, axis)
                blck_axis.append(bx)
                vthd_axis.append(vx)
                thrd_axis.append(tx)
                tile_axis.append(tn)
            axis_order = blck_axis + vthd_axis + thrd_axis + tile_axis

            self.sche[out].reorder(*axis_order)
            blck_fused = self.sche[out].fuse(*blck_axis)
            thrd_fused = self.sche[out].fuse(*thrd_axis)
            if self.binding["space"][0] is not None:
                self.sche[out].bind(
                    blck_fused, te.thread_axis(self.binding["space"][0])
                )
            if self.binding["space"][1] is not None:
                for va in vthd_axis:
                    self.sche[out].bind(va, te.thread_axis(self.binding["space"][1]))
            if self.binding["space"][2] is not None:
                self.sche[out].bind(
                    thrd_fused, te.thread_axis(self.binding["space"][2])
                )

            reduce_axis = []
            if reg_tile is not None:
                self.sche[reg_tile].compute_at(self.sche[out], thrd_fused)
                space_axis = []
                for axis in self.sche[reg_tile].op.axis:
                    space_axis.append(axis)
                for axis in self.sche[reg_tile].op.reduce_axis:
                    res = self.split_axis(reg_tile, axis)
                    reduce_axis = reduce_axis + res
                axis_order = reduce_axis + space_axis
                self.sche[reg_tile].reorder(*axis_order)
                reg_fused = self.sche[reg_tile].fuse(*space_axis)
                self.sche[reg_tile].unroll(reg_fused)
            else:
                for axis in self.sche[out].op.reduce_axis:
                    res = self.split_axis(out, axis)
                    reduce_axis = reduce_axis + res
                if self.binding["reduce"][1] is not None:
                    bind_idx = te.thread_axis(self.binding["reduce"][1])
                    self.sche[out].bind(reduce_axis[1], bind_idx)
                    self.sche[out].set_store_predicate(bind_idx.var.equal(0))

            if reg_tile is not None:
                for rt in reg_tensor:
                    self.sche[rt].compute_at(self.sche[reg_tile], reduce_axis[-1])
                for st in smem_tensor:
                    self.sche[st].compute_at(self.sche[reg_tile], reduce_axis[0])
                    self.cooperative_fetch(st, self.sche)
            else:
                for rt in reg_tensor:
                    self.sche[rt].compute_at(self.sche[out], reduce_axis[-1])
                for st in smem_tensor:
                    self.sche[st].compute_at(self.sche[out], reduce_axis[0])
                    self.cooperative_fetch(st, self.sche)
        for info in align_info:
            idx, factor, offset = info
            st = smem_tensor[idx]
            # st_size = tvm.runtime.DataType(st.dtype).bits // 8
            # num_ele = bank_size // st_size
            # assert num_ele > 0
            # factor = factor * num_ele
            # offset = math.ceil(offset/num_ele) * num_ele
            self.sche[st].storage_align(st.op.axis[-2], factor, offset)
        # assert False
        return self.sche
