# WaterMamba: Visual State Space Model for UnderWater Image Enhancement
Meisheng Guan, Haiyong Xu, Gangyi Jiang, Mei Yu, Yeyao Chen, Ting Luoand Yang Song

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2405.08419)

<img src="assets/model.png" alt="Demo" style="zoom🕙%;" />


<hr />

> **Abstract:** Underwater imaging often suffers from low quality and lack of ffne details due to various physical factors affecting light propagation, scattering, and absorption in water. To improve the quality of underwater images, some underwater image enhancement (UIE) methods based on convolutional neural networks (CNN) and Transformer have been proposed. However, CNN-based UIE methods are limited in modeling long-range dependencies and are often applied to speciffc underwater environments and scenes with poor generalizability. Transformer-based UIE methods excel at longrange modeling, which typically involves a large number of parameters and complex self-attention mechanisms, posing challenges for efffciency due to the quadratic computational complexity of image size. Considering computational complexity and the severe degradation of underwater images, the state space model (SSM) with linear computational complexity for UIE, named WaterMamba, is proposed. Considering the challenges of non-uniform degradation and color channel loss in underwater image processing, we propose a spatialchannel omnidirectional selective scan (SCOSS) blocks consisting of the spatial-channel coordinate omnidirectional selective scan (SCCOSS) modules module and a multi-scale feedforward network (MSFFN). The SCOSS block effectively models pixel information ffow in four directions and channel information ffow in four directions, addressing the issues of pixel and channel dependencies. The MSFFN facilitates information ffow adjustment and promotes synchronized operations within SCCOSS modules. Extensive experiments on four datasets showcase the cutting-edge performance of the WaterMamba while employing reduced parameters and computational resources. The WaterMamba outperforms the state-of-the-art method, achieving a PSNR of 24.7dB and an SSIM of 0.93 on the UIEB dataset. On the UCIOD dataset, the WaterMamba achieves a PSNR of 21.9dB and an SSIM of 0.84. Additionally, the UCIQE of the WaterMamba on the SQUID dataset is 2.77, while the UIQM is 0.56, which further validated its effectiveness and generalizability. The code is publicly available at: https://github.com/Guan-MS/WaterMamba

<hr />

## Network Architecture


[Pre-trained model](https://drive.google.com/drive/folders/1UPsBqRFNToAzvTeF-w66Wr41K3m6ZxH9?usp=sharing)
