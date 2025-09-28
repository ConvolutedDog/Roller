import tvm
from tvm import te
from utils import *
import math
from config import *
from test_config import *


class Op:
    def __init__(self, expr, shape, data_type, use_tc=False) -> None:
        self.expr = expr
        self.shape = shape
        self.use_tc = use_tc
        self.ori_in = []
        self.pad_in = []
        self.outs = []
        self.unpad_outs = []
        self.expr_out = self.expr(self.shape, dataType=data_type, for_rtile=False)
        self.input_tensors = self.expr_out[0]
        self.output_tensors = self.expr_out[1]
        self.fused_shape = []
        self.data_type = data_type

        for it in self.input_tensors:
            if "_pad" in it.name:
                self.pad_in.append(it)
            else:
                self.ori_in.append(it)

        for ot in self.output_tensors:
            if "_unpad" in ot.name:
                self.unpad_outs.append(ot)
            else:
                self.outs.append(ot)
        # todo: what if multi-output op?
        self.saxis, self.raxis = get_axis_names(self.output_tensors[0])

        if LatestTVM:
            # if len(self.unpad_outs) > 0:
            #     import warnings
            #     warnings.warn(
            #         "The unpad_outs length > 0, please check here."
            #     )
            fadd_pf = te.create_prim_func(self.input_tensors + self.output_tensors)
            mod = tvm.IRModule({"main": fadd_pf})
            # Create a TIR schedule
            self.sche = tvm.tir.Schedule(mod)
        else:
            if len(self.unpad_outs) > 0:
                self.sche = tvm.te.create_schedule(self.unpad_outs[0].op)
            else:
                self.sche = tvm.te.create_schedule(self.output_tensors[0].op)

        if len(self.expr_out) == 3:
            # Exprs defined in artifacts/roller/test_config/conv_expr.py may have the 3rd output:
            # E.g. return [data, kernel, data_pad, kernel_pad], [conv, conv_unpad],
            #             {'sa_fused0': [0, 2, 3], 'ra_fused0': [4, 5, 6], 'f': [1]}
            self.fused_shape_map = self.expr_out[2]
            for a in self.saxis:
                fs = self.fused_shape_map[a]
                dim = 1
                for d in fs:
                    dim *= self.shape[d]
                self.fused_shape.append(dim)
            for a in self.raxis:
                fs = self.fused_shape_map[a]
                dim = 1
                for d in fs:
                    dim *= self.shape[d]
                self.fused_shape.append(dim)

        self.spatial_dim = len(self.saxis)
        self._axis_id = {}
        aid = 0
        for axis_name in self.saxis:
            self._axis_id[axis_name] = aid
            aid += 1
        for axis_name in self.raxis:
            self._axis_id[axis_name] = aid
            aid += 1

    def GetUniSchedule(self):
        if not self.use_tc:
            return [1 for _ in range(len(self.SAxis()) + len(self.RAxis()))]
        else:
            return [16 for _ in range(len(self.SAxis()) + len(self.RAxis()))]

    def GetAxisLen(self, axis_name):
        assert axis_name in self._axis_id
        aid = self._axis_id[axis_name]
        return self.Dimensions()[aid]

    def TensorTypeSize(self, tvm_codegen=False):
        tensor_type_size = [[], []]  # input, output
        for t in self.GetInputTensors(tvm_codegen):
            assert isinstance(t, tvm.te.Tensor)
            tensor_type_size[0].append(tvm.runtime.DataType(t.dtype).bits // 8)
        for t in self.GetOutputTensors():
            assert isinstance(t, tvm.te.Tensor)
            tensor_type_size[1].append(tvm.runtime.DataType(t.dtype).bits // 8)
        return tensor_type_size

    def InputTypeSize(self):
        return self.TensorTypeSize()[0][0]

    def OutputTypeSize(self):
        return self.TensorTypeSize()[1][0]

    def TensorDim(self, tvm_codegen=False):
        tensor_dim = [[], []]
        for t in self.GetInputTensors(tvm_codegen):
            assert isinstance(t, tvm.te.Tensor)
            tensor_dim[0].append([int(d) for d in t.shape])
        for t in self.GetOutputTensors():
            assert isinstance(t, tvm.te.Tensor)
            tensor_dim[1].append([int(d) for d in t.shape])
        return tensor_dim

    def ComputeWorkload(self, rtile: rTile):
        wk = 1
        for d in rtile.GetOutputDataTiles()[0]:
            wk *= d
        op_rdim = self.RDimensions()
        tile_rdim = rtile.RDimensions()
        assert len(op_rdim) == len(tile_rdim)
        for i in range(len(op_rdim)):
            aligned_r = math.ceil(op_rdim[i] / tile_rdim[i]) * tile_rdim[i]
            wk *= aligned_r
        tensor_type_size = self.OutputTypeSize()
        return (
            wk * tensor_type_size / 2
        )  # Each ffma instruction executes 2 flops in one cycle.

    def MemWorkload(self, rtile: rTile, tile_tensor="output"):  # todo
        op_rdim = self.RDimensions()
        tile_sdim = rtile.SDimensions()
        tile_rdim = rtile.RDimensions()
        aligned_op_rdim = []
        assert len(op_rdim) == len(tile_rdim)
        for i in range(len(op_rdim)):
            aligned_r = math.ceil(op_rdim[i] / tile_rdim[i]) * tile_rdim[i]
            aligned_op_rdim.append(aligned_r)

        # Why here merge spatial_dim of tile and reduce_dim of op?
        # Although we set tile size to [409, 409, 409] for matmul, we still need
        # loop over the reduce axis of tile, the loop times is ceil(4096 / 409) *
        # 409 / 409 times. And the memory workload should be calculated as the
        # byte size of [409, 409, ceil(4096 / 409) * 409] items.
        merge_dim = tile_sdim + aligned_op_rdim
        tmp_rtile = rTile(
            self.expr, merge_dim, self.SAxis(), self.RAxis(), self.GetTvmOutTensor()
        )

        input_data_tiles = tmp_rtile.GetInputDataTiles()
        output_data_tiles = tmp_rtile.GetOutputDataTiles()

        tensor_type_size = self.TensorTypeSize()
        storage_padding = rtile.GetStoragePadding()

        ret = [[], []]  # inputs, outputs

        for i in range(len(input_data_tiles)):
            shape = input_data_tiles[i]
            padding = storage_padding[i]
            area = 1
            for d in range(len(shape)):
                area *= shape[d] + padding[d]
            ret[0].append(int(area * tensor_type_size[0][i]))

        for i in range(len(output_data_tiles)):
            shape = output_data_tiles[i]
            area = 1
            for d in shape:
                area *= d
            ret[1].append(int(area * tensor_type_size[1][i]))
        return ret

    def MemFootprint(self, rtile, tile_tensor="output"):
        input_data_tiles = rtile.GetInputDataTiles()
        tensor_type_size = self.TensorTypeSize()
        storage_padding = rtile.GetStoragePadding()
        inputs_size = []  # inputs
        for i in range(len(input_data_tiles)):
            shape = input_data_tiles[i]
            padding = storage_padding[i]
            area = 1
            assert len(shape) == len(padding)
            for d in range(len(shape)):
                area *= shape[d] + padding[d]
            inputs_size.append(area * tensor_type_size[0][i])

        ret = 0
        for t in inputs_size:
            ret += t
        return ret

    def Dimensions(self):
        return self.fused_shape if len(self.fused_shape) > 0 else self.shape

    def SAxis(self):
        return self.saxis

    def RAxis(self):
        return self.raxis

    def SDimensions(self):
        return self.Dimensions()[: len(self.saxis)]

    def RDimensions(self):
        return self.Dimensions()[len(self.saxis) :]

    def ReductionAxisLen(self):
        ret = 1
        for rn in self.RDimensions():
            ret *= rn
        return ret

    def RegUsage(self, rtile: rTile, tile_tensor="output"):
        # reduction axis of reg rtile is 1, which should be defined in rtile
        in_datatile = rtile.GetInputDataTiles()
        out_datatiles = rtile.GetOutputDataTiles()
        ret = 0
        for ins in in_datatile:
            area = 1
            for d in ins:
                area *= d
            ret += area * self.InputTypeSize() / 4  # Each register stores 4 byte.
        for outs in out_datatiles:
            area = 1
            for d in outs:
                area *= d
            ret += area * self.OutputTypeSize() / 4
        if self.use_tc:
            ret /= (
                32  # Data used by TC is shared among registers of 32 threads (1 warp).
            )
        if tile_tensor == "output":
            return ret

    def GetGridSize(self, rtile, tile_tensor="output"):
        if tile_tensor == "output":
            output_data_tile = rtile.GetOutputDataTiles()[0]
            output_tensor_shape = self.GetOutputTensors()[0].shape
            assert len(output_data_tile) == len(output_tensor_shape)
            grid_size = 1
            for i in range(len(output_data_tile)):
                grid_i = int(
                    (output_tensor_shape[i] + (output_data_tile[i] - 1))
                    // output_data_tile[i]
                )
                grid_size *= grid_i
            return grid_size

    def GetInputTensors(self, tvm_codegen=False):
        if tvm_codegen and len(self.ori_in) > 0:
            return self.ori_in
        else:
            return self.pad_in if len(self.pad_in) > 0 else self.input_tensors

    def GetOutputTensors(self):
        return self.unpad_outs if self.unpad_outs else self.output_tensors

    def IODependent(self):  # todo
        slopes = []
        # xs = [0.25, 0.5, 0.75]
        # Now, xs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        xs = [0.1 * s for s in range(1, 10)]
        ys = []
        for x in xs:
            shape = [int(x * d) if int(x * d) > 0 else 1 for d in self.Dimensions()]
            rtile = rTile(
                self.expr, shape, self.SAxis(), self.RAxis(), self.GetTvmOutTensor()
            )
            compute = 1
            for d in shape:
                compute *= d
            # Maybe this io should be total io of inputs and outputs? But here it only
            # use the memory footprint of inputs.
            # And the former compute only calculates [409, 409, 409] for matmul, but the
            # latter memory footprint calculates [409, 409, ceil(4096 / 409) * 409]. Is
            # this a bug?
            io = sum(self.MemWorkload(rtile)[0])
            y = compute / io
            ys.append(y)
        for i in range(len(xs) - 1):
            dy = ys[i + 1] - ys[i]
            dx = xs[i + 1] - xs[i]
            s = dy / dx
            slopes.append(s)
        # print("avg_s=", sum_s/len(slopes))
        # print(min(slopes), sum(slopes)/len(slopes))
        for s in slopes:
            # if abs(s) > 3:

            # High slope: Efficiency improves rapidly as scale increases -> I/O bottleneck (the
            #             larger the scale, the higher the computation/I/O ratio).
            # Low slope:  Efficiency changes little as scale increases -> Computation bottleneck.

            # Why here set the threshold to 2 and 4?
            # 1. High Slope (True branch in the code - I/O Bound):
            #    Phenomenon: When the tile size (x) increases, the computational intensity (y)
            #                improves rapidly (high slope).
            #    Explanation: The computation volume compute grows cubically with the size (e.g.,
            #                 x^3), while the data movement volume io grows quadratically (e.g.,
            #                 x^2, because data being moved is reused by an algorithm structured
            #                 similarly to matrix multiplication. The ratio y = compute / io
            #                 approximates x^3/x^2 = x. Therefore, as x increases, y increases
            #                 almost linearly, resulting in a stable and relatively high positive
            #                 slope.
            #    Conclusion: The operation is I/O-bound (I/O-Dependent). Its performance primarily
            #                depends on the speed of moving data from memory to the computational
            #                units. Increasing the data scale (using larger tiles or batch sizes)
            #                can more effectively hide the data movement latency, thereby improving
            #                computational efficiency (higher FLOPs/Byte).
            # 2. Low Slope (False branch in the code - Compute Bound):
            #    Phenomenon: When the tile size (x) increases, the computational intensity (y)
            #                improves very slowly or even flattens out (very low slope).
            #    Explanation: At this point, the tile is large enough that all the required data
            #                 fits into the high-speed cache (Cache). The computational units can
            #                 operate at full speed with almost no waiting for data. Both compute
            #                 and io grow at similar rates, causing their ratio y to approach a
            #                 constant. The curve becomes flat, and the slope approaches zero.
            #    Conclusion: The operation is compute-bound. Its performance has reached the
            #                theoretical peak computational capacity of the hardware units (e.g.,
            #                GPU's CUDA Cores). Further increasing the scale cannot significantly
            #                improve efficiency.
            if min(slopes) > 2 and sum(slopes) / len(slopes) > 4:
                return True
        return False

    def GetTvmOutTensor(self):
        return self.output_tensors[0]
