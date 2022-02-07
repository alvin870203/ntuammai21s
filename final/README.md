 # Cross-Domain Few-Shot Learning
高等多媒體資訊分析與檢索 | Advanced Topics in Multimedia Analysis and Indexing | NTU  | 徐宏民教授 | 2021 Spring | https://winstonhsu.info/ammai-21s/

## Goal
Improve some few-shot learning models' performance using metric learning technique, and test them on three task: **Few-shot Learning**, **Cross-domain Few-shot Learning**, and **Cross-domain Few-shot Learning with Multiple Source Domains**.

![intro](https://user-images.githubusercontent.com/57071722/152838746-a1d5e107-9b02-4282-a95f-9f2bf59adc4e.png)
(Ref:  Tseng et al., “Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation, ” ICLR, 2020

## Tasks
### Task 1 - few-shot learning (fsl):
- source domain: mini-ImageNet
- target domain: mini-ImageNet
- description: train (meta-train) the model on base classes from mini-ImageNet, and meta-test the model on novel classes also from mini-ImageNet. Please note that the base classes and novel classes are disjoint.

### Task 2 - cross-domain few-shot learning with single source domain (cdfsl-single):
- source domain: mini-ImageNet
- target domain: CropDisease, EuroSAT, ISIC
- description:  train (meta-train) the model on base classes from mini-ImageNet, and meta-test the model on novel classes sampled from three different datasets. The domain gap between source and three target domains are different. (CropDisease (small) -> ISIC (large)).

### Task 3 - cross-domain few-shot learning with multi source domain (cdfsl-multi):
- source domain: mini-ImageNet, cifar-100
- target domain: CropDisease, EuroSAT, ISIC
- description: train (meta-train) the model on base classes from two different dataset, and meta-test the model on novel classes sampled from three different datasets. 

## Result
| | fsl | cdfsl-single | cdfsl-single | cdfsl-single | cdfsl-multi | cdfsl-multi | cdfsl-multi |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Model | mini-ImageNet | Crop Disease | EuroSAT | ISIC | Crop Disease | EuroSAT | ISIC |
| Baseline  | 75.09% | 88.94% | 77.93% | 49.08% | 86.94% | 78.63% | 48.59% |
| ProtoNet | 64.16% | 85.18% | 76.19% | 41.51% | 84.07% | 77.61% | 42.74% |
| MyNet | 65.13% | 85.04% | 75.42% | 42.61% | 88.48% | 80.64% | 45.61% |
