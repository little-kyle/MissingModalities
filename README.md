# Multi-Modal Brain Tumor Segmentation with Missing modalities

<font face=STKaiti>该项目主要对于缺失模态脑肿瘤MRI分割任务</font>
1. 常用数据集

| 数据集名称 | Training Data |
|:---:|:---:|
| BraTS2018 | 285 |
| BraTS2019 | 335 |
| BraTS2020 | 369 |

2.常用评价指标

3.MR图像的原理及其特点：
&emsp;MR核磁共振成像技术是常用的成像技术，它可以更好的显示出软组织的对比来可视化脑组织，捕获多种对比度可以提供互补的信息；
&emsp;常用的MR的四种模态分别是T1-weighted(T1)、T2-weighted(T2)、enhanced T1-weighted(T1ce)、Fluid Attenuation Inversion Recovery(FLAIR)。每一种模态对于肿瘤的不同的区域具有不同的敏感度，T2和FLAIR可以突出肿瘤和非肿瘤区域，T1ce可以更清楚显示肿瘤核心边界。

4.相关工作
合成缺失模态：从已有的模态合成缺失的模态，可以有一对一方法（从一种模态合成另一种模态）；多对一方法（从多种模态合成另一种模态）；多对多方法（多种已知模态合成多个缺失的模态）。但是这种方法可能会需要对每一种缺失模态的情况进行模型训练，会造成极大的复杂度；

5.方法论文总结
&emsp;对于缺失模态的学习方法主要可以分为：①对于每一种的缺失模态情况都进行一个model的训练；②合成缺失的模态；③将所有模态映射到他同一个潜在的表征空间内（只需要训练一个model）。对每一种模态都进行一个单独的model的训练需要一共训练 $2^n-1$ 个处理缺失模态的模型；对比之下，将模态编码到一个共同的特征空间只需要一个模型；
&emsp;对于编码到同一个表征空间中，目标是能将模态信息提取出来，最小化确实模态所损失的信息。

[HVED](https://arxiv.org/abs/1907.11150)
<font face="楷体">采用数据集BraTS2018，将MVAE扩展到含有缺失模态的多模态数据集上的3D分割任务上；基于mixture sampling procedure，提出了一个优化过程的principled formulation；将3D Unet应用到一个变分框架中。</font>

[mmFormer](https://arxiv.org/abs/2206.02425)
下面是mmFormer网络的整体结构图：
<div align=center>
<img src="Materials\pictures\mmformer.png" style="zoom:40%"/><br/>
</div>
该方法是第一次将Transformer应用到缺失模态情况下的脑部肿瘤分割任务上；主要包括三个部分：

>混合具体模态编码器，每个模态的图像对应一个编码器进行特征提取，且每一个编码器同时结合了CNN和Transformer（局部和全局性优点）；所用的Transformer分为模态内部（关注每一个模态内部的上下文关联）和模态间（模态之间的上下文关联）；

>一个含有逐步进行上采样和融合模态不变性特征的解码器用来生成鲁棒分割结果；

>在encoder和decoder中引入附加的正则化器，用于进一步增强模态缺失时的鲁棒性；

&emsp;
&emsp;
&emsp;
&emsp;
&emsp;
&emsp;


