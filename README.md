# Semantic Image Inpainting TensorFlow
This repository is a Tensorflow implementation of the [Semantic Image Inpainting with Deep Generative Models, CVPR2017](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yeh_Semantic_Image_Inpainting_CVPR_2017_paper.pdf).

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/43243280-d4e8a3c0-90e0-11e8-8495-b768427019bb.png")
</p>
  
## Requirements
- tensorflow 1.9.0
- python 3.5.3
- numpy 1.14.2
- pillow 5.0.0
- matplotlib 2.0.2
- scipy 0.19.0
- opencv 3.2.0
- pyamg 3.3.2

## Semantic Image Inpainting
1. **celebA**  
- Note: The following resutls are cherry-picked images

<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/43244581-48360a66-90e6-11e8-823c-a71d957ed73b.png">
</p>

2. **SVHN**  
- Note: The following resutls are cherry-picked images

<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/43244654-98d56cdc-90e6-11e8-8f0f-4695d3d3ebe4.png">
</p>

3. Failure Examples
<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/43245170-4eefe500-90e8-11e8-8f49-a47680de2efe.png">
</p>

## Documentation
### Download Dataset
1. **celebA Dataset**
Use the following command to download `CelebA` dataset and copy the `CelebA` dataset on the corresponding file as introduced in **Directory Hierarchy** information. Manually remove approximately 2,000 images from the dataset for testing, put them on the `val` folder and others in the `train' folder.
```
python download.py celebA
```

2. **SVHN Dataset**  
Download SVHN data from [The Street View House Numbers (SVHN) Dataset](http://ufldl.stanford.edu/housenumbers/) website. Two mat files you need to download are `train_32x32.mat` and `test_32x32.mat` in Cropped Digits Format 2.

### Directory Hierarchy
``` 
.
│   semantic_image_inpainting
│   ├── src
│   │   ├── dataset.py
│   │   ├── dcgan.py
│   │   ├── download.py
│   │   ├── inpaint_main.py
│   │   ├── inpaint_model.py
│   │   ├── inpaint_solver.py
│   │   ├── main.py
│   │   ├── solver.py
│   │   ├── mask_generator.py
│   │   ├── poissonblending.py
│   │   ├── tensorflow_utils.py
│   │   └── utils.py
│   Data
│   ├── celebA
│   │   ├── train
│   │   └── val
│   ├── svhn
│   │   ├── test_32x32.mat
│   │   └── train_32x32.mat
```  
**src**: source codes of the `Semantic-image-inpainting`

### Implementation Details
We need two sperate stages to utilize semantic image inpainting model. 
- First, independently train DCGAN on your dataset as the original DCGAN process. 
- Second, use pretrained DCGAN and semantic-image-inpainting model to restore the corrupt images. 

Same generator and discriminator networks of the DCGAN are used as described in [Alec Radford's paper](https://arxiv.org/pdf/1511.06434.pdf), except that batch normalization of training mode is used in training and test mode that we found to get more stalbe results. Semantic image inpainting model is implemented as moodoki's [semantic_image_inpainting](https://github.com/moodoki/semantic_image_inpainting). Some bugs and different implementations of the [original paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yeh_Semantic_Image_Inpainting_CVPR_2017_paper.pdf) are fixed.

### Stage 1: Training DCGAN
Use `main.py` to train a DCGAN network. Example usage:

```
python main.py --is_train=true
```

 - `gpu_index`: gpu index, default: `0`
 - `batch_size`: batch size for one feed forward, default: `256`
 - `dataset`: dataset name for choice [celebA|svhn], default: `celebA`
 - `is_train`: training or inference mode, default: `False`
 - `learning_rate`: initial learning rate, default: `0.0002`
 - `beta1`: momentum term of Adam, default: `0.5`
 - `z_dim`: dimension of z vector, default: `100`
 - `iters`: number of interations, default: `200000`
 - `print_freq`: print frequency for loss, default: `100`
 - `save_freq`: save frequency for model, default: `10000`
 - `sample_freq`: sample frequency for saving image, default: `500`
 - `sample_size`: sample size for check generated image quality, default: `64`
 - `load_model`: folder of save model that you wish to test, (e.g. 20180704-1736). default: `None`
 
### Evaluate DCGAN
Use `main.py` to evaluate a DCGAN network. Example usage:

```
python main.py --is_train=false --load_model=folder/you/wish/to/test/e.g./20180704-1746
```
Please refer to the above arguments.

### Stage 2: Utilize Semantic-image-inpainting Model
Use `inpaint_main.py` to utilize semantic-image-inpainting model. Example usage:

```
python inpaint_main.py --dataset=celebA \
    --load_model=DCGAN/model/you/want/to/use/e.g./20180704-1746 \
    --mask_type=center
```

- `gpu_index': gpu index, default: `0`
- `dataset`: dataset name for choice [celebA|svhn], default: `celebA`
- `learning_rate`: learning rate to update latent vector z, default: `0.01`
- `momentum`: momentum term of the NAG optimizer for latent vector, default: `0.9`
- `z_dim`: dimension of z vector, default: `100`
- `lamb`: hyper-parameter for prior loss, default: `3`
- `is_blend`: blend predicted image to original image, default: `true`
- `mask_type`: mask type choice in [center|random|half|pattern], default: `center`
- `img_size`: image height or width, default: `64`
- `iters`: number of iterations to optimize latent vector, default: `1500`
- `num_try`: number of random samples, default: `20`
- `print_freq`: print frequency for loss, default: `100`
- `sample_batch`: number of sampling images, default: `2`
- `load_model`: saved DCGAN model that you with to test, (e.g. 20180705-1736), default: `None`

### Loss for Optimizing Latent Vector
- Content Loss
<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/43247920-201307b8-90f1-11e8-8b22-8ecb3ebc734d.png", width=800>
</p>

- Prior Loss
<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/43247983-560194d4-90f1-11e8-9d8f-a7435cb21885.png", width=800>
</p>

- Total Loss
<p align='center'>
<img src="https://user-images.githubusercontent.com/37034031/43247998-677ea8c8-90f1-11e8-9564-12ffa3117b9b.png", width=800>
</p>

### Citation
```
  @misc{chengbinjin2018semantic-image-inpainting,
    author = {Cheng-Bin Jin},
    title = {semantic-image-inpainting},
    year = {2018},
    howpublished = {\url{https://github.com/ChengBinJin/semantic-image-inpainting}},
    note = {commit xxxxxxx}
  }
```

### Attributions/Thanks
- This project borrowed some code from [carpedm20](https://github.com/carpedm20/DCGAN-tensorflow) and [moodoki](https://github.com/moodoki/semantic_image_inpainting).
- Some readme formatting was borrowed from [Logan Engstrom](https://github.com/lengstrom/fast-style-transfer)

## License
Copyright (c) 2018 Cheng-Bin Jin. Contact me for commercial use (or rather any use that is not academic research) (email: sbkim0407@gmail.com). Free for research use, as long as proper attribution is given and this copyright notice is retained.

## Related Projects
- [DCGAN](https://github.com/ChengBinJin/DCGAN-TensorFlow)

