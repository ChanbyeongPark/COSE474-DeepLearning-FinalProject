# Multi-Residual Networks with ReLU Dropping

## 목표
기존의 ResNet block에 `BN-Conv-BN-Conv` 형태의 ReLU dropping residual path를 추가한 block으로 이루어진 모델이 이미지 분류에서 성능 향상을 보일 수 있는지 확인한다.

## 기존 연구 분석

### ResNet



### ReLU Dropping


### Multi-Residual Networks




## 차례
추후 작성 예정

## Introduction

### Motivation

### Problem Definition

### Contribution


## Methods

### Significance / Novelty

### Figure

### Reproducibility


## Experiments

### Dataset
Proposal에서는 데이터셋으로 Tiny ImageNet을 사용한다고 작성하였으나, Tiny ImageNet은 training data가 너무 적어 모델 자체의 성능보다 augmentation에 크게 의존하는 것으로 보인다.
따라서 데이터셋으로는 이미지 분류에서 주로 사용되는 CIFAR-10과 CIFAR-100을 사용할 예정이다.

### Computer Resource & Experimental Design
Google Colab (Pro) - 사용한 CPU와 GPU는 실험 중 확인 후 작성 예정

### Quantitative Results

### Qualitative Results

### Figures(Plots) / Tables and Analysis

### Discussion(Why method is successful or unsuccessful)

## Future Direction
