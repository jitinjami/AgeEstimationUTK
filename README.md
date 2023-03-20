# Age Estimation PyTorch
PyTorch-based CNN implementation for [estimating age from face images](https://link.springer.com/article/10.1007/s11263-016-0940-3) using [UTK Face](https://susanqq.github.io/UTKFace/) and [UTK Face cropped](https://www.kaggle.com/datasets/abhikjha/utk-face-cropped) dataset. 

Similar Keras-based project can be found [here](https://github.com/yu4u/age-gender-estimation). This PyTorch implementation is forked and modified from [here](https://github.com/yu4u/age-estimation-pytorch).


## Requirements

```bash
pip install -r requirements.txt
```

## Data Download

To download [UTK Face](https://susanqq.github.io/UTKFace/)

```bash
python3 data_download_UTK.py
```

To download [UTK Face cropped](https://www.kaggle.com/datasets/abhikjha/utk-face-cropped/)

```bash
python3 data_download_UTK_crop.py
```

The above 2 scripts download the zip files and extract them into a directory [UTK](UTK).

## Create Dataset

The follwing script:
1. Splits the images into training, texting and validation dataset.
2. Extracts age from file name.
3. Filter out corrupted files
4. Place images and csv files in [UTK_db](UTK_db) directory

See `python create_utk_dataset.py -h` for detailed options.

```bash
python3 create_utk_dataset.py
```

## Train
Train a model using the UTK Face or UTK Face cropped dataset.
See `python train.py -h` for detailed options.

```bash
python train.py
```

#### Training Options
You can change training parameters including model architecture using additional arguments like this:

```bash
python train.py --data_dir [PATH/TO/UTK_db] --tensorboard tf_log MODEL.ARCH se_resnet50 TRAIN.OPT sgd TRAIN.LR 0.1
```

All default parameters defined in [defaults.py](defaults.py) can be changed using this style.

Best models is saved in [checkpoint](checkpoint)

## Test Trained Model
Evaluate the trained model using the APPA-REAL test dataset.

```bash
python test.py --data_dir [PATH/TO/UTK_db] --resume [PATH/TO/BEST_MODEL.pth]
```

## Demo
The following script reads images from input directory, predicts age of the person and saves the output image (with age and bounding box on the image) to the output directory.
See `python demo.py -h` for detailed options.

```bash
python demo.py --resume [PATH/TO/BEST_MODEL.pth] --img_dir [PATH/TO/IMAGE_DIRECTORY] --output_dir [PATH/TO/OUTPUT_DIRECTORY]
```

## Trained Models

The [checkpoint](checkpoint/) directory has models that were trained on different datasets with or without decaying learning rate.

The naming convention is `<case>_<val_mae>_<test_mae>.pth`.

The cases are:
1. UTK with no decay: `utk_no_decay_6.5340_6.510.pth`
2. UTK with decay: `utk_6.3537_6.393.pth`
3. UTK_crop with no decay: `utk_crop_no_decay_6.0225_5.729.pth`
4. UTK_crop with decay: `utk_crop_5.7053_5.682.pth`
