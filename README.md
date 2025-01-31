# FireSeg
Official code for 'FireSeg: A Weakly Supervised Fire Segmentation Framework via Pre-trained Latent Diffusion Models'


### Installation
```sh
conda create -n FireSeg python=3.10
```
CUDA==12.1
Then install other packages:
```sh
python -m pip install -r requirements.txt
```

Select one of the following pre-training weight files: [Stable Diffusion XL Base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main) and place them in the ```./dataset/ckpts``` directory.


### Training Dataset Prepare
- I have placed the [BoWFire](https://ieeexplore.ieee.org/abstract/document/7314551) dataset used for training in a standardized format under the [Link](https://drive.google.com/file/d/1lilG-1MkrV6wLILSVZDuMMz-4wQiqjCb/view?usp=sharing), and you can follow this example to configure the contents of the dataset yourself to fit your project.

The overall format arrangement is consistent with the VOC dataset:


```
data/
    PascalVOC12/
	JPEGImages
	SegmentationClassAug
	splits/
	     train_aug.txt
```


### Running Command
Training the Flame-Decoder


```sh
# For Training Mask-Decoder
sh train.sh
```


```sh
# Inference
sh inference.sh
```



