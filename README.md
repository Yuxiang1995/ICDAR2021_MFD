# About The Project
This project releases our 1st place solution on **ICDAR 2021 Competition on Mathematical Formula Detection**.
We implement our solution based on [MMDetection](https://github.com/open-mmlab/mmdetection), which is an open source object detection toolbox based on PyTorch.
 You can click [here](http://transcriptorium.eu/~htrcontest/MathsICDAR2021/) for more details about this competition.

# Getting Start

## Prerequisites

- Linux or macOS (Windows is in experimental support)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

This project is based on MMDetection-v2.7.0, mmcv-full>=1.1.5, <1.3 is needed.
Note: You need to run `pip uninstall mmcv` first if you have mmcv installed.
If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.

## Installation

1. Install PyTorch and torchvision following the [official instructions
](https://pytorch.org/), e.g.,
   
   1.3.1=<pytorch version<=1.6.0

    ```shell
    pip install pytorch torchvision -c pytorch
    ```

    Note: Make sure that your compilation CUDA version and runtime CUDA version match.
    You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

    `E.g.1` If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install
    PyTorch 1.5, you need to install the prebuilt PyTorch with CUDA 10.1.

    ```shell
    pip install pytorch cudatoolkit=10.1 torchvision -c pytorch
    ```

    `E.g. 2` If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install
    PyTorch 1.3.1., you need to install the prebuilt PyTorch with CUDA 9.2.

    ```shell
    pip install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
    ```

    If you build PyTorch from source instead of installing the prebuilt pacakge,
    you can use more CUDA versions such as 9.0.
   
2. Install mmcv-full, we recommend you to install the pre-build package as below.

    ```shell
    pip install mmcv-full==latest+torch1.6.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html
    ```

    See [here](https://github.com/open-mmlab/mmcv#install-with-pip) for different versions of MMCV compatible to different PyTorch and CUDA versions.
    Optionally you can choose to compile mmcv from source by the following command

    ```shell
    git clone https://github.com/open-mmlab/mmcv.git
    cd mmcv
    MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full will be installed after this step
    cd ..
    ```

    Or directly run

    ```shell
    pip install mmcv-full
    ```
   
3. Install build requirements and then install MMDetection.

    ```shell
    pip install -r requirements/build.txt
    pip install ensemble-boxes
    pip install -v -e .  # or "python setup.py develop"
    ```
   
## Data Preparation