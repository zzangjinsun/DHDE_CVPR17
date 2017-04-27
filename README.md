A Unified Approach of Multi-scale Deep and Hand-crafted Features for Defocus Estimation
----------
#### [Jinsun Park](https://sites.google.com/site/zzangjinsun/), Yu-Wing Tai, [Donghyeon Cho](https://sites.google.com/site/donghyeonchocvip/) and In So Kweon
#### _IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Jul 2017_

### Introduction

![Teaser](./images/teaser.png)

In this paper, we introduce robust and synergetic hand-crafted features and a simple but efficient deep feature from a convolutional neural network (CNN) architecture for defocus estimation. This paper systematically analyzes the effectiveness of different features, and shows how each feature can compensate for the weaknesses of other features when they are concatenated. For a full defocus map estimation, we extract image patches on strong edges sparsely, after which we use them for deep and hand-crafted feature extraction. In order to reduce the degree of patch-scale dependency, we also propose a multi-scale patch extraction strategy. A sparse defocus map is generated using a neural network classifier followed by a probability-joint bilateral filter. The final defocus map is obtained from the sparse defocus map with guidance from an edge-preserving filtered input image. Experimental results show that our algorithm is superior to state-of-the-art algorithms in terms of defocus estimation. Our work can be used for applications such as segmentation, blur magnification, all-in-focus image generation, and 3-D estimation.

### Results

![Results](./images/results.png)

Defocus map estimation and binary blurry region segmentation results. (a) Input images. (b) Results of Shi et al.[^1] (c) Results of Shi et al.[^2] (d) Results of Shi et al.[^3] (Inverted for visualization) (e) Results of Zhuo and Sim.[^4] (f) Our defocus maps and (g) corresponding binary masks. (h) Ground truth binary masks.

### Citation

### Dependencies
Our current implementation is tested on:
- Ubuntu 14.04.5 LTS
- [Caffe](http://caffe.berkeleyvision.org/)
- MATLAB R2016a / R2017a

### Usage



[^1]: Shi, Jianping, et al. "Break ames room illusion: depth from general single images." ACM Transactions on Graphics (TOG) 34.6 (2015): 225.
[^2]: Shi, Jianping, Li Xu, and Jiaya Jia. "Discriminative blur detection features." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2014.
[^3]: Shi, Jianping, Li Xu, and Jiaya Jia. "Just noticeable defocus blur detection and estimation." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.
[^4]: Zhuo, Shaojie, and Terence Sim. "Defocus map estimation from a single image." Pattern Recognition 44.9 (2011): 1852-1858.


