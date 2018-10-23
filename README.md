# Cifar-10_PyTorch

Personal practice on CIFAR10 with PyTorch
Inspired by [pytorch-cifar10](https://github.com/icpm/pytorch-cifar10) 

## Result
Models | Best Accuracy | Comments
:---:|:---:|:---:
[AlexNet](https://github.com/zhang-zx/cifar10_pytorch/master/models/AlexNet.py) | 82.18% | BatchNorm and learning rate adjustment is added to make an improvment. 
[ResNet18](https://github.com/zhang-zx/cifar10_pytorch/master/models/ResNet.py) | 89.57% |From the picture, one can tell that this model is convergent before 50 epochs and the rest training is just in vain. :) It seems that this ResNet is not as good as expected. Maybe I made something wrong.



## Usage

1. Requirements

```shell
pip install -r requirements.txt
```

2. Run

```shell
python main.py --lr learning_rate --epoch epochs_num --trainBatchSize train_batch_size --testBatchSize test_batch_size --net network
```

# Training Procedure

## AlexNet

1. Training Procedure 

![image](./Img/AlexNet_Train.png)

2. Learning Rate Decay 

![image](./Img/AlexNet_Learning_Rate.png)

## ResNet 18

1. Training Procedure 

![image](./Img/ResNet18_Train.png)

2. Learning Rate Decay 

![image](./Img/ResNet18_Learning_Rate.png)
