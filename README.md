# DTMD
This is the PyTorch implementation of paper: DTMD: Learning Improvement of Spiking Neural Networks with Dynamic Thresholding Neurons and Moderate Dropout  **(submitted to NeurIPS 2022)**.

Here we provide code for CIFAR10 dataset as a pre-release version, trained model is also provided. More datasets will be added later.

## Dependencies and Installation
- Python 3.8.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- PyThorch 1.9.0
- [PyTorch, torchvision](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

## Dataset Preparation
As for MNIST and CIFAR10, the data can be downloaded by torchvision as in the code. The DVS-CIFAR10 is avaliable at [here](https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671/2). The N-MNIST dataset is avaliable at [here](https://www.garrickorchard.com/datasets/n-mnist).

Preprocessing of splitting images of N-MNIST and DVS-CIFAR10 are provided as [dataset.py](https://github.com/sq171717/DTMT/blob/main/nmnist/dataset.py) and [processing_data.py](https://github.com/sq171717/DTMT/blob/main/dvscifar10/processing_data.py) in the corresponding folders.

## Training and Testing
In the corresponding directory, run following command with GPU IDs identification:

	CUDA_VISIBLE_DEVICE=0,1,2,3 python trialnew.py

Network will be trained and tested iteratively. Highest performance model and the final model generated in the last epoch will be automatically saved in tmp folder.

You can also find our trained models in tmp folder.

All the hyperparameters are initialized same as the values mentioned in the paper.

## Load and Test the Existing Model
In the corresponding directory, run following command with GPU IDs identification:

	CUDA_VISIBLE_DEVICE=0,1,2,3 python test.py

Model with the highest accuracy will be loaded and run testing process.

## Results
The results of our DTMD method on MNIST, N-MNIST, CIFAR10 and DVS-CIFAR10 are:

|    Model    | Time steps |  Best  |  Mean   |   Std   |
|:-----------:|:----------:|:------:|:-------:|:-------:|
|    MNIST    |     4      | 99.60% | 99.584% | 0.012%  |
|   CIFAR10   |     4      | 94.19% | 94.154% | 0.0403% |
|   N-MNIST   |     15     | 99.65% | 99.614% | 0.0206% |
| DVS-CIFAR10 |     7      | 73.30% | 72.92% | 0.319%  |


Results of ablation study on the effect of our proposed methods are:

|    Model    | Time steps | LIF (plain) | DT (Dynamic thresholding) | DTMD (Dynamic threshodling&Moderate dropout) |
|:-----------:|:----------:|:-----------:|:-------------------------:|:--------------------------------------------:|
|    MNIST    |     4      |   99.53%    |          99.57%           |                    99.60%                    |
|   CIFAR10   |     2      |   92.88%    |          93.51%           |                    93.75%                    |
|   N-MNIST   |     15     |   99.55%    |          99.58%           |                    99.65%                    |
| DVS-CIFAR10 |     7      |   71.30%    |          72.30%           |                    73.30%                    |

