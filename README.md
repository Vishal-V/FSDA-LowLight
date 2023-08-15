# Few-Shot Domain Adaptation for Low Light RAW Image Enhancement ([Project Page](https://val.cds.iisc.ac.in/HDR/BMVC21/index.html))
  
## Abstract
Enhancing practical low light raw images is a difficult task due to severe noise and color distortions from short exposure time and limited illumination. Despite the success of existing Convolutional Neural Network (CNN) based methods, their performance is not adaptable to different camera domains. In addition, such methods also require large datasets with short-exposure and corresponding long-exposure ground truth raw images for each camera domain, which is tedious to compile. To address this issue, we present a novel few-shot domain adaptation method to utilize the existing source camera labeled data with few labeled samples from the target camera to improve the target domain's enhancement quality in extreme low-light imaging. Our experiments show that only ten or fewer labeled samples from the target camera domain are sufficient to achieve similar or better enhancement performance than training a model with a large labeled target camera dataset. To support research in this direction, we also present a new low-light raw image dataset captured with a Nikon camera, comprising short-exposure and their corresponding long-exposure ground truth images.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/few-shot-domain-adaptation-for-low-light-raw/domain-adaptation-on-canon-raw-low-light)](https://paperswithcode.com/sota/domain-adaptation-on-canon-raw-low-light?p=few-shot-domain-adaptation-for-low-light-raw)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/few-shot-domain-adaptation-for-low-light-raw/domain-adaptation-on-nikon-raw-low-light)](https://paperswithcode.com/sota/domain-adaptation-on-nikon-raw-low-light?p=few-shot-domain-adaptation-for-low-light-raw)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/few-shot-domain-adaptation-for-low-light-raw/low-light-image-enhancement-on-nikon-raw-low)](https://paperswithcode.com/sota/low-light-image-enhancement-on-nikon-raw-low?p=few-shot-domain-adaptation-for-low-light-raw)


## Links
Nikon Camera Dataset: [Kaggle Link](https://www.kaggle.com/datasets/razorblade/nikon-camera-dataset)
- Sony Camera Dataset (SID): [Drive](https://storage.googleapis.com/isl-datasets/SID/Sony.zip)
- Fuji Camera Dataset (SID): [Drive](https://storage.googleapis.com/isl-datasets/SID/Fuji.zip)
- Canon Camera Dataset: [GitHub](https://github.com/jconenna/Canon-6D-Datasets-For-Learning-to-See-in-the-Dark)

Environments:
1. Python 3.7
2. Pytorch 1.0.0 + Rawpy + Numpy + Scipy
3. Trained and tested on Tesla V100 32GB GPU. Cuda(>=10.0)
4. CPU requirement >= 64 GB

Docker Image: [Link](https://hub.docker.com/r/vvinodhub/midnight)

Testing :
1. Run "python test.py". We can change "input_dir", "gt_dir", "result_dir" at the beginning.
2. We use pack_nikon for Nikon validation and pack_canon for Canon validation.

Model:
1. UnetConverter is for the source Grayscale SSIM 16-to-8 bit conversion. The pretrained Sony 16-to-8 bit converter can be downloaded here: [Google Drive](https://drive.google.com/file/d/17BphXVzG9YKaWxW5qjNZdzV8dqcS-E_S/view?usp=sharing)
2. UNetSony is used for shared U-net model (N).
