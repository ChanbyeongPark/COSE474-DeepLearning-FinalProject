# Multi-Residual Networks with ReLU Dropping

## Objective
기존의 residual block에 ReLU dropped residual function을 추가한 모델이 이미지 분류에서 성능 향상을 보일 수 있는지 확인한다.

## Related Work

### Residual Networks
- Identity mapping을 통한 residual learning 제안  
  &rightarrow; Plain 모델에 비해 깊이가 깊어도 잘 학습할 수 있으며, 깊이 증가에 따른 성능 향상을 얻을 수 있다.
- 50-layer 이상의 깊은 모델은 building block 대신 1x1 conv를 적용한 bottleneck block 사용
- 간단한 augmentation 적용(4 pixel padding 후 random crop, horizontal flip)
- He initialization, SGD, mini-batch size는 128, learning rate는 0.1로 시작해서 필요할 때 10을 나눔, weight decay는 0.0001, momentum은 0.9로 적용
- Pre-activation residual unit을 사용하면 더욱 깊은 모델에서 학습이 잘 되는 효과를 얻는다.

### Multi-Residual Networks
- Residual network가 ensemble처럼 동작한다는 관점에 근거하여 residual block의 residual function을 증가시키는 multi-residual network를 제안한다.
- 이러한 multi-residual network는 모델을 깊고 좁게보다는 얕고 넓게 만들어 이점을 얻을 수 있다.
- 병렬적인 구조이므로 model parallelism technique을 통해 15%의 계산 복잡도를 향상시켰다.

### ReLU Proportional Module
- Convolution과 ReLU의 수가 N:M (N>M)의 비율을 이루는 proportional module 제안
- 이는 추가적인 연산 없이 더 일반화된 특성을 얻도록 하여 성능 향상을 이끌 수 있다.
- 각 conv 사이에는 batch normalization이 존재하기 때문에 하나의 conv로 합쳐지지 않는다.

## Introduction

### Motivation
1. Residual networks는 ensemble과 비슷한 특성을 보인다.
2. Ensemble의 개수(multiplicity)가 많을수록 성능이 향상될 수 있다.
3. Ensemble learning은 같은 데이터를 다른 알고리즘으로 사용하거나, 다른 데이터를 같은 알고리즘으로 사용한다.
4. ReLU dropping은 추가적인 연산 없이 더 일반화된 특성을 얻을 수 있도록 한다.

### Problem Definition
이미지 분류

### Contribution
1. 동일한 구조의 residual function을 여러 개 사용하는 multi-residual block은 각 residual function이 동일한 데이터와 알고리즘으로 학습하기 때문에 multiplicity가 증가한다는 이점을 얻기 어려울 수 있다.
따라서 ensemble의 다양성을 위해 ReLU dropped residual function을 사용하며, 이를 통해 multiplicity의 증가와 ReLU dropped의 일반화 특성으로 성능이 향상될 수 있는지 확인한다.
2. Multi-residual block은 1개의 identity edge와 n개의 function edge로 구성된다.
전체 edge의 개수가 일정하다면 block의 edge의 개수가 3일 때 가장 많은 ensemble의 개수(multiplicity)를 얻으며, 
따라서 function edge가 2개인 경우 기존에 비해 더 뛰어난 성능을 보일 수 있는지 확인한다.

## Methods

### Figure
<img src="https://github.com/ChanbyeongPark/COSE474-DeepLearning-FinalProject/assets/78645777/de4ce6f6-de4d-4fe0-b716-f3478e494f09"  width="665" height="595">

ReLU Dropped Basic Block(좌), ReLU Dropped Bottleneck Block(우)

## Experiments

### Dataset
기존 모델과의 용이한 성능 비교를 위해 이미지 분류에서 주로 사용되는 CIFAR-10을 사용한다.

### Computer Resource & Experimental Design
Google Colab Pro  
CPU: Intel Xeon 2.20 GHz  
GPU: NVIDIA V100

ResNet 논문과 유사한 실험 환경 사용  
&rightarrow; SGD(lr=0.1, momentum=0.9, weight decay=0.0001)  
&rightarrow; learning rate warm up으로 첫번째 epoch는 0.01을 사용한 뒤 80 epoch까지 0.1, 120 epoch까지 0.01, 이후 0.001 사용하여 165 epoch까지 진행.  
&rightarrow; He initialization 사용, batch size는 128  
&rightarrow; flip과 translation이라는 기본적인 augmentation만 사용

기존 모델과의 비교를 위해 parameter의 개수를 1.7M로 통일  
My Models: ReLU Dropped Multi-ResNet-56(basic block), ReLU Dropped Multi-ResNet-83(bottleneck block)

기존 모델  
Basic block: ResNet-110, PreActResNet-110, Multi-ResNet-8, Multi-ResNet14, Multi-ResNet30  
Bottleneck block: ResNet-164, PreActResNet-164

CIFAR-10에 대해 RDM-ResNet-56와 RDM-ResNet-83을 각각 5번 수행

median(mean&pm;std)로 모델 성능 비교

### Results

r은 block을 구성하는 residual function의 개수를 나타낸다.

| Model             | Depth | r     | Error Rate(%)      |
| :---:             | :---: | :---: | :---:              |
| PreActResNet      | 110   | 1     | 6.37               |
| PreActResNet      | 164   | 1     | 5.46               |
| Multi-ResNet      | 8     | 23    | 7.37               |
| Multi-ResNet      | 14    | 10    | 6.42               |
| Multi-ResNet      | 30    | 4     | 5.89               |
| RDM-ResNet(Mine)  | 56    | 2     | 5.98(5.98&pm;0.20) |
| RDM-ResNet(Mine)  | 83    | 2     | 5.41(5.34&pm;0.12) |

<img src="https://github.com/ChanbyeongPark/COSE474-DeepLearning-FinalProject/assets/78645777/9f684ae5-8b9e-4192-b1a3-a35c2022c84c">
