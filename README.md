# Multi-Residual Networks with ReLU Dropping

## Objective
기존의 residual block에 ReLU dropped function edge를 추가한 block의 모델이 이미지 분류에서 성능 향상을 보일 수 있는지 확인한다.

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
1. Multi-residual block은 1개의 identity edge와 n개의 function edge로 구성된다.
전체 edge의 개수가 일정하다면 block의 edge의 개수가 3일 때 가장 많은 ensemble의 개수(multiplicity)를 얻으며, 
따라서 function edge가 2개인 경우 기존에 비해 더 뛰어난 성능을 보일 수 있는지 확인한다.
2. 동일한 형태의 function edge를 사용하는 multi-residual block은 동일한 데이터가 동일한 알고리즘으로 작동하기 때문에 multiplicity가 증가한다는 이점을 얻기 어려울 수 있다.
그대신 ensemble의 다양성을 얻기 위해 ReLU dropping을 적용한 function edge을 사용하며, 이를 통해 multiplicity의 증가와 ReLU dropping의 일반화된 특성으로 이점을 얻을 수 있는지 확인한다.

## Methods

### Figure
<img src="https://github.com/ChanbyeongPark/COSE474-DeepLearning-FinalProject/assets/78645777/de4ce6f6-de4d-4fe0-b716-f3478e494f09"  width="665" height="595">

ReLU Dropped Basic Block(좌), ReLU Dropped Bottleneck Block(우)

## Experiments

### Dataset
기존 모델과의 용이한 성능 비교를 위해 이미지 분류에서 주로 사용되는 CIFAR-10과 CIFAR-100을 사용한다.

### Computer Resource & Experimental Design
Google Colab (Pro) - 사용한 CPU와 GPU는 실험 중 확인 후 작성  

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

Basic block 모델은 CIFAR-10에만 사용(마지막 Conv 채널 개수가 64라서 CIFAR-100에 사용하기에는 부적절)
Bottleneck block 모델은 CIFAR-10과 CIFAR-100에 사용

RDM-ResNet-56을 CIFAR-10에 3번 수행  
RDM-ResNet-83을 CIFAR-10와 CIFAR-100에 각각 3번 수행

진행 과정 비교 용으로 PreActResNet-164를 CIFAR-10, CIFAR-100에 한 번씩 수행

median(mean&pm;std)로 모델 성능 비교

### Quantitative Results
실험 후 작성

### Qualitative Results
실험 후 작성

### Figures(Plots) / Tables and Analysis
추가 예정

RDM-ResNet-83(median) vs PreActResNet-164 for CIFAR-10 and CIFAR-100 (Training curve)  
성능 비교 총 정리 Table (top-1 error)

### Discussion(Why method is successful or unsuccessful)
실험 후 작성

## Future Direction
Multi-Residual bottleneck block의 function edge가 n개, Multi-ResNet의 depth가 d라고 했을 때, 해당 모델의 ensemble path가 거치는 평균적인 function edge의 개수는 $\frac{d}{3(1+n)}$이다.
Residual network가 ensemble의 형태로 작동한다는 논문에 따르면 학습에 가장 큰 영향을 미치는 effective path는 상대적으로 적은 module을 지난 경우이며, 대부분의 gradient도 얕은 layer에서 얻어진다.  
따라서 모델의 parameter 개수가 일정한 경우, depth가 증가할 때 function edge의 개수도 증가시켜 거치는 평균 function edge의 개수가 해당 범위에 속하도록 하면 효율적인 학습이 가능할 것으로 보인다.  
(layer가 충분히 많아 첫번째 layer와 마지막 layer를 무시할 수 있으며, 각 function edge가 모두 같은 개수의 parameter를 사용한다고 가정)  