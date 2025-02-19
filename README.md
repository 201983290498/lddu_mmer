# Latent Distribution Decoupling: A Probabilistic Framework for Uncertainty-Aware Multimodal Emotion Recognition
[\[Paper\]](https://arxiv.org/abs/2312.08379)

This is the homepage for Latent Distribution Decoupling: A Probabilistic Framework for Uncertainty-Aware Multimodal Emotion Recognition.

We propose a new framework name lddu based on the latent emotion space modeling and uncertainty learning. the proposed framework consists of three components: (1) the transformer-base unimodal extractor (2) a contrastive learning-based emotion space decomposition module and (3) an uncertainty-aware fusion and uncertainty calibration module.*

![LDDU Framework](./images/framework_v2.png)

Authors: [Jiangwang Huang](huangjiangwang@stu.cqu.edu.cn), [KaiWen Wei](weikaiwen@cqu.edu.cn), [Jiang Zhong](zhongjiang@cqu.edu.cn) at el.

## Quick Links:
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Training and Inference](#training-and-inference)
- [Citation](#citation)

## Release Progress
- Training code
- Data and data preparation
- Training readme

## Requirements

## Dataset

the dataset can be downloaded from [here](https://drive.google.com/drive/folders/1umLIjIlL8Y1oWYzU2L6UyPTHFQx7RREB). Please download the aligned data and unaligned data to "./dataset/dataset1/". Then run the script:
```bash
cd ./dataset
python data_process.py
```

the dataset is without labels, to get ground truth, you can process by yourself, or use processed groud truth file from "./dataset/dataset1/label.csv"


More details about CMU-MOSEI and M3ED datasets will be released soon after the paper is officially published.

## Training and Inference 
```bash
pip install -r requirement
sh trian.sh
```

## Citation
```code

```