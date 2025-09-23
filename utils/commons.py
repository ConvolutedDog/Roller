import tvm
from threading import Thread
from . import LatestTVM
import sys
import math
from typing import Tuple, TYPE_CHECKING, Union, Optional
from tvm.tir.stmt_functor import post_order_visit
import warnings
from functools import wraps


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

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


def deprecated(exit_immediately=True):
    """Deprecated decorator"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warning_message = f"\033[91mFunction {func.__name__} is deprecated and cannot be used anymore.\033[0m"
            if exit_immediately:
                print(warning_message)
                sys.exit()
            else:
                warnings.warn(
                    warning_message,
                    DeprecationWarning,
                    stacklevel=2,
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_blocks(sch: tvm.tir.Schedule, without_root=False):
    """Get all blocks from a schedule"""
    assert LatestTVM

    blocks = {}

    def update_blocks(node):
        if isinstance(node, tvm.tir.Block):
            if not (without_root and node.name_hint == "root"):
                blocks[node.name_hint] = node

    for gv in sch.mod.get_global_vars():
        func = sch.mod[gv]
        if isinstance(func, tvm.tir.PrimFunc):
            post_order_visit(func.body, update_blocks)

    return blocks


def get_axis_names(Tensor: tvm.te.Tensor):
    """Get axis names"""
    if LatestTVM:
        saxis = [str(axis.var.name) for axis in Tensor.op.axis]
        raxis = [str(axis.var.name) for axis in Tensor.op.reduce_axis]
    else:
        s: tvm.te.Schedule = tvm.te.create_schedule(Tensor.op)
        saxis = [axis.var.name for axis in s[Tensor].op.axis]
        raxis = [axis.var.name for axis in s[Tensor].op.reduce_axis]
    return saxis, raxis


def showmod(sch: ScheduleType, tensor_list: Optional[Tuple[tvm.te.Tensor]]):
    """Show the IRModule of schedule"""
    print("=" * 80)
    if LatestTVM:
        print(tvm.lower(sch, tensor_list).script())
    else:
        sch.mod.show()
    print("=" * 80)


def getSpecifiedLoopRVs(
    sch: ScheduleType, block: "BlockRV", iter_type
) -> Tuple["LoopRV"]:
    """Helper function to get specified loop RVs"""
    alllooprvs = sch.get_loops(block)
    targetlooprvs = []
    itervars = sch.get(block).iter_vars
    for i in range(len(itervars)):
        itervar = itervars[i]
        if itervar.iter_type == iter_type:
            targetlooprvs.append(alllooprvs[len(alllooprvs) - len(itervars) + i])
    return targetlooprvs


def getSpatialLoopRVs(sch: ScheduleType, block: "BlockRV") -> Tuple["LoopRV"]:
    """Helper function to get spatial LoopRVs"""
    return getSpecifiedLoopRVs(sch, block, iter_type=tvm.tir.IterVar.DataPar)


def getReduceLoopRVs(sch: ScheduleType, block: "BlockRV") -> Tuple["LoopRV"]:
    """Helper function to get reduce LoopRVs"""
    return getSpecifiedLoopRVs(sch, block, iter_type=tvm.tir.IterVar.CommReduce)


def calculate_factors(
    sch: ScheduleType, looprv: "LoopRV", factor: int
) -> Tuple[int, int]:
    """Helper function to transform old `factor` argument of `sch.split`
    in TVM 0.8 to new `factors` argument of `sch.split` in LatestTVM"""
    return [math.ceil(int(sch.get(looprv).extent) / factor), factor]


def str_to_ms(string):
    if string.endswith("ms"):
        return float(string[:-2])
    elif string.endswith("us"):
        return float(string[:-2]) / 1000
    elif string.endswith("s"):
        return float(string[:-1]) * 1000


def get_time_from_nvprof_file(out, backend="tvm"):
    with open(out, "r") as inf:
        lines = inf.readlines()
        if backend == "tvm":
            kernel_name = "default_function_kernel0" if not LatestTVM else "main_kernel"
        if backend == "antares":
            kernel_name = "template_op_kernel0"
        for line in lines:
            if kernel_name in line:
                breaks = line.split()
                return str_to_ms(breaks[-4])
    # kernel does not execute
    return 1e100


def get_time_from_rocm_file(file_name="_tmp"):
    try:
        with open(file_name) as f:
            for line in f.readlines():
                if "- TPR" in line:
                    t_ms = float(line.rstrip()[7:])
                    return t_ms
        return 1e100
    except:
        return 1e100


def simplify(expr):
    """Simplify the expression if it is Expr, directly return if it is int.
    Parameters
    ----------
    expr : Expr or int
        The input.
    Returns
    -------
    out : Expr or int
        The simplified output
    """
    return (
        tvm.arith.Analyzer().simplify(expr)
        if isinstance(expr, tvm.tir.PrimExpr)
        else expr
    )


def get_pad_tuple(padding, kernel):
    """Common code to get the pad option
    Parameters
    ----------
    padding : int or str
        Padding size, or ['VALID', 'SAME']
    kernel : tuple of int
        Conv kernel size
    Returns
    -------
    pad_top : int
        Padding size on top
    pad_left : int
        Padding size on left
    pad_down : int
        Padding size on down.
    pad_right : int
        Padding size on right.
    """
    # compute the padding size
    if isinstance(padding, (tuple, list)):
        if len(padding) == 2:
            pad_h = padding[0] * 2
            pad_w = padding[1] * 2
        elif len(padding) == 4:
            return padding[0], padding[1], padding[2], padding[3]
        else:
            raise ValueError("Size of padding can only be 2 or 4")
    elif isinstance(padding, int):
        pad_h = pad_w = padding * 2
    elif padding == "VALID":
        pad_h = 0
        pad_w = 0
    elif padding == "SAME":
        pad_h = kernel[0] - 1
        pad_w = kernel[1] - 1
    else:
        raise ValueError("Unknown padding option %s" % padding)
    pad_top = (pad_h + 1) // 2
    pad_left = (pad_w + 1) // 2
    return pad_top, pad_left, pad_h - pad_top, pad_w - pad_left


def alloc_configs_for_subprocess(parallel, configs_num):
    num_process = [
        int(configs_num // parallel) + 1 for i in range(configs_num % parallel)
    ]
    num_process = num_process + [
        int(configs_num // parallel) for i in range(parallel - configs_num % parallel)
    ]
    idx = 0
    process_idx = [0]
    for num in num_process:
        process_idx.append(idx + num)
        idx += num
    return process_idx


class MyThread(Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


Backend = Literal["tvm", "antares"]
