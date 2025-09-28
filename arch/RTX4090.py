from .Arch import *


class RTX4090(Arch):
    # TODO: Need to adjust the parameters
    def __init__(self, para_opt=True):
        super().__init__()
        # Only for tagging.
        self.name = "NVIDIA GeForce RTX 4090"
        self.global_mem_capacity = "46GB"

        # DRAM: memory level 0
        # SMEM: memory level 1
        # REG: memory level 2
        self.num_level = 2

        # NOTE: The memory interface of RTX 4090 (GDDR6X) is 384-bit wide.
        #       GDDR6X supports speeds of 21 Gbps on RTX 4090. So the
        #       total bandwidth of RTX 4090 is 384*21/8 = 1008 GB/s.
        #
        #       The GPU boost clock of RTX 4090 is 2520 MHz, and the bank
        #       size of the shared memory is 32 with 32-bit (4 bytes) data 
        #       width per bank. In fact, the number of banks per shared 
        #       memory is usually designed to 32 for the reason that it 
        #       could provide 4-byte data for each of the the 32 threads 
        #       in a warp. So ecifically, each shared memory can provide
        #       4 bytes * 32 banks = 128 bytes per cycle. So the memory
        #       bandwidth of the shared memory is 128 bytes * 2520 MHz
        #       = 322.56 GB/s, and the 128 SMs in RTX 4090 can provide
        #       322.56 GB/s * 128 = 41287 GB/s.
        # REF: https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
        #
        # bandwidth in GB/second
        self.bandwidth = [1008, 41287]
        # compute throughput in GFLOPS
        self.peak_flops = 84582
        self.peak_tc_flops = 338227
        self.limit = []
        # NOTE: The max register number that can be used by a single thread
        #       is 255, but the max number of registers that can be used by a
        #       single block can be accessed by:
        #       import pycuda.driver as cuda
        #       import pycuda.autoinit
        #       cuda.Context.get_device().get_attribute(
        #           cuda.device_attribute.MAX_REGISTERS_PER_BLOCK
        #       ),
        #       which is 65536 for the RTX 4090 GPU.
        self.reg_cap = [65536, 255]
        # NOTE: The max shared memory size that can be used by a single block
        #       can be accessed by:
        #       import pycuda.driver as cuda
        #       import pycuda.autoinit
        #       cuda.Context.get_device().get_attribute(
        #           cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK
        #       ),
        #       which is 49152 for the RTX 4090 GPU.
        self.smem_cap = [49152]
        # NOTE: There are 128 SMs on the RTX 4090 GPU, and each SM has 4 sub-cores.
        self.compute_max_core = [128]
        self.mem_max_core = [128]
        self.para_opt = para_opt

        self.warp_size = 32
        # NOTE: There are 128 SMs on the RTX 4090 GPU, and each SM has 4 sub-cores.
        self.compute_sm_partition = [128, 4]
        self.smem_sm_partition = [128, 4]
        self.compute_block_schedule_way = ["warp", "active block"]
        self.smem_block_schedule_way = ["warp", "active block"]
        # transaction size in bytes.
        self.transaction_size = [32]

        # The number of banks per shared memory is usually designed to 32 for
        # the reason that it could provide 4-byte data for each of the the 32
        # threads in a warp.
        self.smem_bank_size = 4
        self.bank_number = 32
        self.compute_capability = "compute_89"

        # for active block estimation
        #
        # not used in latest Roller.
        self.max_active_blocks = 32
        # Each shared memory/L1 cache has 128 KB capacity on RTX 4090.
        #
        # not used in latest Roller.
        self.max_smem_usage = 128 * 1024 - 1
        # The max number of threads per SM can be accessed by:
        #       import pycuda.driver as cuda
        #       import pycuda.autoinit
        #       cuda.Context.get_device().get_attribute(
        #           cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR
        #       ),
        # which is 1536 for the RTX 4090 GPU.
        #
        # not used in latest Roller.
        self.max_threads_per_sm = 1536
