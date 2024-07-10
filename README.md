# Retain and Recover: Delving into Information Loss for Few-Shot Segmentation

This repository contains the source code for our paper "*Retain and Recover: Delving into Information Loss for Few-Shot Segmentation*" by Chunbo Lang, Gong Cheng, Binfei Tu, Chao Li, and Junwei Han.

> **Abstract:** *Benefiting from advances in few-shot learning techniques, their application to dense prediction tasks (e.g., segmentation) has also made great strides in the past few years. However, most existing few-shot segmentation (FSS) approaches follow a similar pipeline to that of few-shot classification, where some core components are directly exploited regardless of various properties between tasks. We note that such an ill-conceived framework introduces unnecessary information loss, which is clearly unacceptable given the already very limited training sample. To this end, we delve into the typical types of information loss and provide a reasonably effective way, namely Retain And REcover (RARE). The main focus of this paper can be summarized as follows: (i) the loss of spatial information due to global pooling; (ii) the loss of boundary information due to mask interpolation; (iii) the degradation of representational power due to sample averaging. Accordingly, we propose a series of strategies to retain/recover the avoidable/unavoidable information, such as unidirectional pooling, error-prone region focusing, and adaptive integration. Extensive experiments on two popular benchmarks (i.e., PASCAL-5<sup>i</sup> and COCO-20<sup>i</sup>) demonstrate the effectiveness of our scheme, which is not restricted to a particular baseline approach. The ultimate goal of our work is to address different information loss problems within a unified framework, and it also exhibits superior performance compared to other methods with similar motivations.*

## ▶️ Getting Started

Please refer to our [BAM](https://github.com/chunbolang/BAM) code repository for environment setup and result reproduction.

## 📖 BibTex

If you find this repository useful for your publications, please consider citing our paper.

```bibtex
@article{lang2023rare,
	title={Retain and Recover: Delving into Information Loss for Few-Shot Segmentation},
	author={Lang, Chunbo and Cheng, Gong and Tu, Binfei and Li, Chao and Han, Junwei},
	journal={IEEE Transactions on Image Processing},
	volume={32},
	pages={5353-5365},
	year={2023},
}
  
@article{lang2023bam,
	title={Base and Meta: A New Perspective on Few-shot Segmentation},
	author={Lang, Chunbo and Cheng, Gong and Tu, Binfei and Li, Chao and Han, Junwei},
	journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
	volume={45},
	number={9},
	pages={10669-10686},
	year={2023},
}
```
