from .Arch import *


class H100(Arch):
    # TODO: Need to adjust the parameters
    def __init__(self, para_opt=True):
        super().__init__()
        # Only for tagging.
        self.name = "NVIDIA H100 80GB HBM3"
        self.global_mem_capacity = "80GB"

        # DRAM: memory level 0
        # SMEM: memory level 1
        # REG: memory level 2
        self.num_level = 2

        # NOTE: The memory interface of H100 (HBM2e) is 5120-bit wide.
        #       HBM2e supports speeds of 2039 GB/s on H100.
        #
        #       The GPU boost clock of H100 is 1755 MHz, and the bank
        #       size of the shared memory is 32 with 32-bit (4 bytes) data
        #       width per bank. In fact, the number of banks per shared
        #       memory is usually designed to 32 for the reason that it
        #       could provide 4-byte data for each of the the 32 threads
        #       in a warp. So ecifically, each shared memory can provide
        #       4 bytes * 32 banks = 128 bytes per cycle. So the memory
        #       bandwidth of the shared memory is 128 bytes * 1755 MHz
        #       = 224.64 GB/s, and the 114 SMs in H100 can provide
        #       224.64 GB/s * 114 = 25608 GB/s.
        # REF: https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c
        #
        # bandwidth in GB/second
        self.bandwidth = [2039, 25608]
        # compute throughput in GFLOPS
        self.peak_flops = 52428
        self.peak_tc_flops = 774144
        self.limit = []
        # NOTE: The max register number that can be used by a single thread
        #       is 255, but the max number of registers that can be used by a
        #       single block can be accessed by:
        #       import pycuda.driver as cuda
        #       import pycuda.autoinit
        #       cuda.Context.get_device().get_attribute(
        #           cuda.device_attribute.MAX_REGISTERS_PER_BLOCK
        #       ),
        #       which is 65536 for the H100 GPU.
        self.reg_cap = [65536, 255]
        # NOTE: The max shared memory size that can be used by a single block
        #       can be accessed by:
        #       import pycuda.driver as cuda
        #       import pycuda.autoinit
        #       cuda.Context.get_device().get_attribute(
        #           cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK
        #       ),
        #       which is 49152 for the H100 GPU.
        self.smem_cap = [49152]
        # NOTE: There are 114 SMs on the H100 GPU, and each SM has 4 sub-cores.
        self.compute_max_core = [114]
        self.mem_max_core = [114]
        self.para_opt = para_opt

        self.warp_size = 32
        # NOTE: There are 114 SMs on the H100 GPU, and each SM has 4 sub-cores.
        self.compute_sm_partition = [114, 4]
        self.smem_sm_partition = [114, 4]
        self.compute_block_schedule_way = ["warp", "active block"]
        self.smem_block_schedule_way = ["warp", "active block"]
        # transaction size in bytes.
        self.transaction_size = [32]

        # The number of banks per shared memory is usually designed to 32 for
        # the reason that it could provide 4-byte data for each of the the 32
        # threads in a warp.
        self.smem_bank_size = 4
        self.bank_number = 32
        self.compute_capability = "compute_90"

        # for active block estimation
        #
        # not used in latest Roller.
        self.max_active_blocks = 32
        # Each shared memory/L1 cache has 228 KB capacity on H100.
        #
        # not used in latest Roller.
        self.max_smem_usage = 228 * 1024 - 1
        # The max number of threads per SM can be accessed by:
        #       import pycuda.driver as cuda
        #       import pycuda.autoinit
        #       cuda.Context.get_device().get_attribute(
        #           cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR
        #       ),
        # which is 2048 for the H100 GPU.
        #
        # not used in latest Roller.
        self.max_threads_per_sm = 2048
