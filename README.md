# Unofficial ROLLER Implementation Using ​TVM 0.21+

This repository contains an unofficial modern implementation of the paper named *ROLLER: Fast and Efficient Tensor Compilation for Deep Learning*, rebuilt using the latest version of TVM (Apache TVM 0.21+).

## A. About ROLLER

ROLLER is a fast and efficient tensor compilation system for deep learning workloads. Unlike search-based approaches that can take hours to find optimal kernels, ROLLER uses a construction-based approach that generates highly efficient kernels in seconds.

#### ​​1. Original Paper:​

- [ROLLER: Fast and Efficient Tensor Compilation for Deep Learning](https://www.usenix.org/conference/osdi22/presentation/zhu) (OSDI'22)

#### 2. Key Features:​​

- Novel rTile abstraction that encapsulates tensor shapes aligned with accelerator characteristics.
- Recursive construction algorithm for generating efficient rTile-based programs.
- Micro-performance model for rapid evaluation without device execution.
- Support for various accelerators including GPUs and emerging AI chips.

## B. Installation

- This implementation requires ​​TVM 0.21 or newer​​. Please install the latest version of TVM following the official [installation guide](https://tvm.apache.org/docs/install/index.html).

## C. Usage

```python3
python test_op.py
```

## D. Citation

If you use ROLLER in your research, please cite the original paper:

```bibtex
@inproceedings {280896,
  author = {Hongyu Zhu and Ruofan Wu and Yijia Diao and Shanbin Ke and Haoyu Li and Chen Zhang and Jilong Xue and Lingxiao Ma and Yuqing Xia and Wei Cui and Fan Yang and Mao Yang and Lidong Zhou and Asaf Cidon and Gennady Pekhimenko},
  title = {{ROLLER}: Fast and Efficient Tensor Compilation for Deep Learning},
  booktitle = {16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22)},
  year = {2022},
  isbn = {978-1-939133-28-1},
  address = {Carlsbad, CA},
  pages = {233--248},
  url = {https://www.usenix.org/conference/osdi22/presentation/zhu},
  publisher = {USENIX Association},
  month = jul
}
```

**If you wish**, you may also cite this repository:

```bibtex
@misc{unofficial_roller_impl,
  author = {Jianchao Yang},
  title = {Unofficial ROLLER Implementation Using TVM 0.21+},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ConvolutedDog/Roller}}
}
```

## E. Roadmap & Contributing

1. The codebase has not been fully organized and refactored.
2. Scripts in the original Roller repository are not included and have not been thoroughly tested in this new environment.

This repo is currently in its early stages. We are actively working towards a stable and feature-complete release. We welcome and appreciate any contributions!

## F. Acknowledgments

This implementation is based on the original ROLLER research from Microsoft Research and collaborating institutions. The original implementation can be found at: [microsoft/nnfusion](https://github.com/microsoft/nnfusion/tree/osdi22_artifact/artifacts).
