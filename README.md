# About The Project
This project releases our 1st place solution on **ICDAR 2021 Competition on Mathematical Formula Detection**.
We implement our solution based on [MMDetection](https://github.com/open-mmlab/mmdetection), which is an open source object detection toolbox based on PyTorch.
 You can click [here](http://transcriptorium.eu/~htrcontest/MathsICDAR2021/) for more details about this competition.

## Method Description
We built our approach on [FCOS](https://arxiv.org/abs/1904.01355), A simple and strong anchor-free object detector, with [ResNeSt](https://arxiv.org/abs/2004.08955) as our backbone, to detect embedded and isolated formulas. 
We employed [ATSS](https://arxiv.org/abs/1912.02424) as our sampling strategy instead of random sampling to eliminate the effects of sample imbalance. Moreover, we observed and revealed the influence of different FPN levels on the detection result. 
[Generalized Focal Loss](https://arxiv.org/abs/2006.04388) is adopted to our loss.
Finally, with a series of useful tricks and model ensembles, our method was ranked 1st in the MFD task.

![Random Sampling(left) ATSS(right)](https://github.com/Yuxiang1995/ICDAR2021_MFD/blob/main/resources/sampling_strategy.png)
**Random Sampling(left) ATSS(right)**


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
   
3. Install build requirements and then compile MMDetection.

    ```shell
    pip install -r requirements.txt
    pip install tensorboard
    pip install ensemble-boxes
    pip install -v -e .  # or "python setup.py develop"
    ```

# Usage

## Data Preparation
Firstly, Firstly, you need to put the image files and the GT files into two separate folders as below.

```shell
Tr01
├── gt
│   ├── 0001125-color_page02.txt
│   ├── 0001125-color_page05.txt
│   ├── ...
│   └── 0304067-color_page08.txt
├── img
    ├── 0001125-page02.jpg
    ├── 0001125-page05.jpg
    ├── ...
    └── 0304067-page08.jpg
```

Secondly, run [data_preprocess.py](https://github.com/Yuxiang1995/ICDAR2021_MFD/blob/main/tools/data_preprocess.py) to get coco format label. 
Remember to change **'img_path'**, **'txt_path'**, **'dst_path'** and **'train_path'** to your own path.  

```shell
python ./tools/data_preprocess.py
```

The new structure of data folder will become,
```shell
Tr01
├── gt
│   ├── 0001125-color_page02.txt
│   ├── 0001125-color_page05.txt
│   ├── ...
│   └── 0304067-color_page08.txt
│
├── gt_icdar
│   ├── 0001125-color_page02.txt
│   ├── 0001125-color_page05.txt
│   ├── ...
│   └── 0304067-color_page08.txt
│   
├── img
│   ├── 0001125-page02.jpg
│   ├── 0001125-page05.jpg
│   ├── ...
│   └── 0304067-page08.jpg
│
└── train_coco.json
```

Finally, change **'data_root'** in ./configs/_base_/datasets/formula_detection.py to your path.

## Train

1. train with single gpu on ResNeSt50
   ```shell
   python tools/train.py configs/gfl/gfl_s50_fpn_2x_coco.py --gpus 1 --work-dir ${Your Dir}
   ```
   
2. train with 8 gpus on ResNeSt101
   ```shell
   ./tools/dist_train.sh configs/gfl/gfl_s101_fpn_2x_coco.py 8 --work-dir ${Your Dir}
   ```
   
## Inference
Run [tools/test_formula.py](https://github.com/Yuxiang1995/ICDAR2021_MFD/blob/main/tools/test_formula.py)
```shell
python tools/test_formula.py configs/gfl/gfl_s101_fpn_2x_coco.py ${checkpoint path} 
```
It will generate a 'result' file at the same level with work-dir in default. You can specify the output path of the result file in line 231.

## Model Ensemble
Specify the paths of the results in tools/model_fusion_test.py, and run
```shell
python tools/model_fusion_test.py
```

## Evaluation
[evaluate.py](https://github.com/Yuxiang1995/ICDAR2021_MFD/blob/main/evaluate.py) is the officially provided evaluation tool. Run
```shell
python evaluate.py ${GT_DIR} ${CSV_Pred_File}
```
Note: GT_DIR is the path of the original data folder which contains both the image and the GT files. 
CSV_Pred_File is the path of the final prediction csv file.

# Result
Train on Tr00, Tr01, Va00 and Va01, and test on Ts01. Some results are as follows,
**F1-score**
    <table>
        <tr>
            <th>Method</th>
            <th>embedded</th>
            <th>isolated</th>
            <th>total</th>
        </tr>
        <tr>
            <th>ResNeSt50-DCN</th>
            <th>95.67</th>
            <th>97.67</th>
            <th>96.03</th>
        </tr>
        <tr>
            <th>ResNeSt101-DCN</th>
            <th>96.11</th>
            <th>97.75</th>
            <th>96.41</th>
        </tr>
    </table>

Our final result, that was ranked 1st place in the competition, was obtained by fusing two Resnest101+GFL models trained with two different random seeds and all labeled data.
The final ranking can be seen in our [technical report](https://arxiv.org/abs/2107.05534).

# License
This project is licensed under the MIT License. See LICENSE for more details.

# Citations
```shell
@article{zhong20211st,
  title={1st Place Solution for ICDAR 2021 Competition on Mathematical Formula Detection},
  author={Zhong, Yuxiang and Qi, Xianbiao and Li, Shanjun and Gu, Dengyi and Chen, Yihao and Ning, Peiyang and Xiao, Rong},
  journal={arXiv preprint arXiv:2107.05534},
  year={2021}
}
@article{GFLli2020generalized,
  title={Generalized focal loss: Learning qualified and distributed bounding boxes for dense object detection},
  author={Li, Xiang and Wang, Wenhai and Wu, Lijun and Chen, Shuo and Hu, Xiaolin and Li, Jun and Tang, Jinhui and Yang, Jian},
  journal={arXiv preprint arXiv:2006.04388},
  year={2020}
}
@inproceedings{ATSSzhang2020bridging,
  title={Bridging the gap between anchor-based and anchor-free detection via adaptive training sample selection},
  author={Zhang, Shifeng and Chi, Cheng and Yao, Yongqiang and Lei, Zhen and Li, Stan Z},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9759--9768},
  year={2020}
}
@inproceedings{FCOStian2019fcos,
  title={Fcos: Fully convolutional one-stage object detection},
  author={Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9627--9636},
  year={2019}
}
@article{solovyev2019weighted,
  title={Weighted boxes fusion: ensembling boxes for object detection models},
  author={Solovyev, Roman and Wang, Weimin and Gabruseva, Tatiana},
  journal={arXiv preprint arXiv:1910.13302},
  year={2019}
}
@article{ResNestzhang2020resnest,
  title={Resnest: Split-attention networks},
  author={Zhang, Hang and Wu, Chongruo and Zhang, Zhongyue and Zhu, Yi and Lin, Haibin and Zhang, Zhi and Sun, Yue and He, Tong and Mueller, Jonas and Manmatha, R and others},
  journal={arXiv preprint arXiv:2004.08955},
  year={2020}
}
@article{MMDetectionchen2019mmdetection,
  title={MMDetection: Open mmlab detection toolbox and benchmark},
  author={Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and Liu, Ziwei and Xu, Jiarui and others},
  journal={arXiv preprint arXiv:1906.07155},
  year={2019}
}
```

# Acknowledgements
* [MMDetection](https://github.com/open-mmlab/mmdetection/tree/master)
* [Weighted Box Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)
* [Ranger](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)
* [Solution Slides](https://github.com/Yuxiang1995/ICDAR2021_MFD/blob/main/resources/MFD.pdf)