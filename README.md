![Python >=3.5](https://img.shields.io/badge/Python->=3.6-blue.svg)
![PyTorch >=1.6](https://img.shields.io/badge/PyTorch->=1.6-yellow.svg)

# Neighbour Consistency Guided Pseudo-Label Refinement for Unsupervised Person Re-Identification

## Requirements

### Installation

```shell
cd NCPLR
python setup.py develop
```

### Prepare Datasets

```shell
mkdir data
```
Download the person datasets Market-1501,MSMT17,PersonX,DukeMTMC-reID and the vehicle datasets VeRi-776. Then unzip them under the directory like

```
ClusterContrast/examples/data
├── market1501
│   └── Market-1501-v15.09.15
├── msmt17
│   └── MSMT17_V1
├── personx
│   └── PersonX
├── dukemtmcreid
│   └── DukeMTMC-reID
└── veri
    └── VeRi
```

### Prepare ImageNet Pre-trained Models for ResNet50

ImageNet-pretrained models for **ResNet-50** will be automatically downloaded in the python script.

## Training

We utilize 4 GTX-1080TI GPUs for training. For more parameter configuration, please check **`run.sh`**.

**examples:**

Market-1501:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python ncplr_train.py --logs logs/market
```

personX
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python ncplr_train.py -d personx --eps 0.7 --rampup-value 0.8 --eps-neighbor-gap 0.1 --logs logs/personx
```

MSMT17:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python ncplr_train.py -d msmt17 --iters 400 --eps 0.7 --eps-neighbor 0.3 --logs logs/dukemtmcreid
```

DukeMTMC-reID:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python ncplr_train.py -d dukemtmcreid --eps 0.7  --eps-neighbor 0.3 --logs logs/msmt17
```

VeRi-776

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python ncplr_train.py -d veri --iters 400 --eps 0.6 --height 224 --width 224 --eps-neighbor 0.3 --logs logs/veri
```

## Evaluation

We utilize 1 GTX-1080TI GPU for testing. **Note that**

+ use `--width 128 --height 256` (default) for person datasets, and `--height 224 --width 224` for vehicle datasets;

+ use `-a resnet50_neughbor` (default) for the backbone of ResNet-50_Neighbor.

To evaluate the model, run:
```shell
CUDA_VISIBLE_DEVICES=0 python test.py -d $DATASET --resume $PATH_FOR_RESUME
```