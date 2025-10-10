"""
CUDA Kernel Dimension Analyzer

This module provides functionality to analyze TVM IRModule and extract
CUDA kernel launch configurations (gridDim and blockDim) from thread binding patterns.
"""

import tvm
from tvm.tir import PyStmtExprVisitor
from tvm.tir import For, ForKind
from dataclasses import dataclass


@dataclass
class SupportedThreadTags:
    """Supported CUDA Thread Tags"""

    supported_block_tags = ["blockIdx.x", "blockIdx.y", "blockIdx.z"]
    supported_thread_tags = ["threadIdx.x", "threadIdx.y", "threadIdx.z"]


class Dim3:
    """CUDA dim3 Structure Representation"""

    def __init__(self):
        self.x: int = 1
        self.y: int = 1
        self.z: int = 1

    def set_val(self, tag: str, val: int):
        """Set dimension value based on thread tag"""
        if not isinstance(val, int):
            raise ValueError(f"Extent value must be integer, got {type(val)}")

        if (
            tag in SupportedThreadTags.supported_block_tags
            or tag in SupportedThreadTags.supported_thread_tags
        ):
            if ".x" in tag:
                self.x = val
            elif ".y" in tag:
                self.y = val
            elif ".z" in tag:
                self.z = val
            else:
                raise RuntimeError(f"Unsupported thread tag: {tag}")
        else:
            raise RuntimeError(f"Unsupported thread tag: {tag}")

        self.validate()

    def validate(self):
        """Validate dimension values"""
        if self.x <= 0 or self.y <= 0 or self.z <= 0:
            raise ValueError(
                f"Invalid dimension values: ({self.x}, {self.y}, {self.z})"
            )
        if self.x * self.y * self.z > 1024:
            raise ValueError(
                f"Too max dimension values: ({self.x}, {self.y}, {self.z})"
            )

    def __str__(self):
        return f"dim3({self.x}, {self.y}, {self.z})"


@tvm.tir.functor.visitor
class SeekBlockThreadExtent(PyStmtExprVisitor):
    """Thread Binding Extent Visitor

    This visitor traverses TIR statements to extract thread binding extents
    for both block-level (gridDim) and thread-level (blockDim) dimensions.
    """

    def __init__(self):
        super().__init__()
        self.block_extent = {}
        self.thread_extent = {}

    def visit_for(self, stmt: For):
        """Visit For nodes to extract thread binding information"""
        if stmt.kind == ForKind.THREAD_BINDING:
            thread_tag = stmt.thread_binding.thread_tag
            extent_value = stmt.extent.value

            if thread_tag in SupportedThreadTags.supported_thread_tags:
                if not thread_tag in self.thread_extent.keys():
                    self.thread_extent[thread_tag] = extent_value
                else:
                    raise RuntimeError(f"Duplicated thread binding tag: {thread_tag}")
            elif thread_tag in SupportedThreadTags.supported_block_tags:
                if not thread_tag in self.block_extent.keys():
                    self.block_extent[thread_tag] = extent_value
                else:
                    raise RuntimeError(f"Duplicated thread binding tag: {thread_tag}")
            elif not "vthread" in thread_tag:
                raise RuntimeError(
                    f"Unsupported thread binding tag: {stmt.thread_binding.thread_tag}"
                )


def cuda_launch_params(mod: tvm.IRModule):
    """Extract CUDA Kernel Launch Configuration

    Parameters
    ----------
    mod : tvm.IRModule
        The IRModule containing CUDA kernel functions

    Returns
    -------
    gridDim : Dim3
        Grid dimension configuration (blockIdx ranges)
    blockDim : Dim3
        Block dimension configuration (threadIdx ranges)

    Raises
    ------
    RuntimeError
        If duplicate thread bindings or unsupported thread tags are found
    """
    visitor = SeekBlockThreadExtent()

    for gv, func in mod.functions.items():
        tvm.tir.stmt_functor.post_order_visit(
            func.body,
            lambda stmt: (
                visitor.visit_for(stmt) if isinstance(stmt, tvm.tir.For) else None
            ),
        )

    gridDim = Dim3()
    for tag, val in visitor.block_extent.items():
        gridDim.set_val(tag, val)

    blockDim = Dim3()
    for tag, val in visitor.thread_extent.items():
        blockDim.set_val(tag, val)

    return gridDim, blockDim
