# MindSpore implementation of IRNet

### Lightweight Improved Residual Network for Efficient Inverse Tone Mapping
[[paper]](https://arxiv.org/abs/2307.03998)

![image](https://github.com/ThisisVikki/IRNet_mindspore/blob/main/figure/Net%26block_structure_newcrop.png)
**Figure:** *Architecture of the proposed Improved Residual Network (IRNet) and the Improved Residual Block (IRB)*

## Installation and Dependencies
Clone this github repository
```
git clone https://github.com/ThisisVikki/IRNet_mindspore.git
cd IRNet_mindspore
```
Create a new environment and install the dependencies
```
conda create -n IRNet_ms python=3.8 -y
conda activate IRNet_ms

pip install -r requirements.txt
```
Install MindSpore: please refer to the offical website of [MindSpore](https://www.mindspore.cn/install/)

## Getting Started
### Dataset
We use the [HDRTV1K](https://github.com/chxy95/HDRTVNet#configuration) dataset for both training and testing. The test set of [Deep SR-ITM](https://github.com/sooyekim/Deep-SR-ITM) and our ITM-4K test dataset are also used for testing. The ITM-4K test dataset can be downloaded from [Google Drive](https://drive.google.com/file/d/1AJqz2ILR9XD_UO4lFklBOoKpj4ZOrRmo/view?usp=drive_link) and [Baidu NetDisk](https://pan.baidu.com/s/1KK9s6DU-ZAKIEydMnFfClQ) (access code: t7iy). It contains 160 pairs of SDR and HDR images of size $3840\times2160\times3$.

### How to test
The `model` is the model that needs to be tested, e.g. IRNet_2, IRNet_1 or SRITM_IRNet_5. Please make sure the path of the test data `testdata_path` and pretrained model `model_path` in `photo_mindspore.py` are correct, then run the code:
```
python photo_mindspore.py
```
The test result will be saved to `./IRNet-2/results`. We also offer the Pytorch code in `photo.py`.

**The pretrained model will be released soon.**

### How to Train
The training settings can be found at `./experiments/IRNet_COSINE_mindspore.yaml`, please check the settings and make sure `TRAIN_DATAROOT_GT`, `TRAIN_DATAROOT_LQ`, `VALID_DATAROOT_GT` and `VALID_DATAROOT_LQ` are in the right paths.

Then run the code:
```
python train_mindspore.py --model [model name] --channels [model_channels]
```
We also offer the Pytorch training code in `train.py`.


## References
If our work is helpful for you, please cite our paper:
```BibTeX
@article{xue2024lightweight,
  title={Lightweight improved residual network for efficient inverse tone mapping},
  author={Xue, Liqi and Xu, Tianyi and Song, Yongbao and Liu, Yan and Zhang, Lei and Zhen, Xiantong and Xu, Jun},
  journal={Multimedia Tools and Applications},
  pages={1--24},
  year={2024},
  publisher={Springer}
}
```
The code and the proposed test dataset is released for academic research only.
