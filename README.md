# FSDA LL

Code for "Few Shot Domain Adaptation for Low Light RAW Image Enhancmenet" - BMVC 2021 Best Student Paper Award Runner Up

Environments:
1. Python 3.7
2. Pytorch 1.0.0 + Rawpy + Numpy + Scipy
3. Trained and tested on Tesla V100 32GB GPU. Cuda(>=10.0)
4. CPU requirement >= 64 GB

Testing :
1. Run "python test.py". We can change "input_dir", "gt_dir", "result_dir" at the beginning.
2. We use pack_nikon for Nikon validation and pack_canon for Canon validation.

Model:
1. UnetConverter is used for 16-to-8 bit converter.
2. UNetSony is used for shared U-net model (N).
