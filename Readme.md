# Pneumonia Classification
## _Using several image classification algorithms_

&nbsp;
## Features

- Accuracy upto 98.56%
- Comparision of several transformer models for pneumonia classification
- Reproducable results
- Publicly available dataset
&nbsp;
&nbsp;

## About the project:

This project aims to compare different machine learning algorithms for the classificarion of pneumonia.
The following are some of the major libararies used in this project.

- PyTorch
- Datasets by HuggingFace
- Transformers by HuggingFace
- Pillow
- Timm (PyTorch Image Models)
- Torchvision

&nbsp;

## About the files
 
&nbsp;
This file contains a list of libraries used in the project

Use the following command in command-line to install them together
> pip install -r requirements.txt

After installing all the libraries you will also need to download the models which will automatically download once you run the ipynb files

#### ipynb files

These files contain the which was used for fine-tuining the pretrained models used in this project
> Note: During the 1st run the there must be an active internet connection for downloading the pretrained models once they have been downloaded there is no need to have an active internet connection
##

#### Tensorboard folders
Tensorboard dirs for training/testting metrics
To see metrics use this command in command-line:
> tensorbaord --logdir=<path>
e.g: tensorboard --logdir=ViT_base_384_tensorboard

&nbsp;

### Results:

