# DS-Pruning: Harnessing Dying Neurons for Sparsity-Driven Structured Pruning in Neural Networks
This is the official implementation for DS-Pruning: Harnessing Dying Neurons for Sparsity-Driven Structured Pruning in Neural Networks

# Overview
In this paper, we challenge the traditional view of "dying neurons" in deep neural network training. Conventionally, dying neurons—neurons that cease to activate—are seen as problematic, often linked to optimization challenges and a reduction in model adaptability during continual learning. However, we offer a novel perspective by investigating how dying neurons can contribute to network sparsity, particularly in structured pruning. Through a systematic exploration of hyperparameter configurations, we reveal that dying neurons can be harnessed to enhance structured pruning algorithms effectively. Our approach, termed "DS-Pruning," introduces a method to regulate the occurrence of dying neurons, enabling dynamic sparsification during training. This method is both straightforward and broadly applicable, outperforming existing structured pruning techniques while achieving competitive results with popular state-of-the-art methods. These findings suggest that dying neurons can serve as an efficient mechanism in sparsity-aware pruning for network compression and resource optimization via the behaviour analysis of individual neurons, paving a new pathway for more efficient and performant deep learning models.

# Results

![BS](https://github.com/wangbst/ExplainableP/assets/97005040/ed999e78-f198-42fb-a556-6f308ac0a163)![LR](https://github.com/wangbst/ExplainableP/assets/97005040/ac4abc77-595f-4d42-9a1f-4e81b2bb2432)![Regularization](https://github.com/wangbst/ExplainableP/assets/97005040/2c054748-7efc-434c-b321-90650f35ded3) 

# Dependencies
```shell
conda create -n myenv python=3.7
conda activate myenv
conda install -c pytorch pytorch==1.9.0 torchvision==0.10.0
pip install scipy
```

# Datasets
Please download the Imagenet Dataset. 

# ResNet18 and Leaky ReLU
All used ResNet18 and Leaky ReLU models can be downloaded from here. Please put them in ResNet18().

# Run dying neurons accumulation for a ResNet-18 trained on CIFAR-10.
 ```shell
$ python Resnet18.py
$ python Leaky ReLU.py
$ python Swish.py
```
- In Leaky ReLU.py, replace activation functions ReLU with LeakyReLU.
- In Swish.py, replace activation functions ReLU with Swish.

# Run SGD noise and SGD for a ResNet-18 trained on CIFAR-10.
 ```shell
$ python SGD noise.py
$ python SGD.py
```

# Run Neural sparsity, structured methods for ResNet-18 on CIFAR-10.
 ```shell
$ python Resnet18.py
$ python Leaky ReLU.py
```

 # Run Weight sparsity, structured methods for ResNet-18 on CIFAR-10.
 ```shell
$ python Resnet18.py
$ python Leaky ReLU.py
```

# Run ResNet-50 model trained on ImageNet using different criteria when pruning at approximately 80% and 90% weight sparsity.
 ```shell
$ python Imagenet.py
```
Set a new download directory for `'model = torchvision.models.resnet50(pretrained=True)'`, we need to export `'TORCH_HOME=/torch_cache'`
