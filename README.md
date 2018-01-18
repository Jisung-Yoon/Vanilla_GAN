# Vanilla_GAN
Tensorflow implementation of [Generative Adversarial Network(GAN)](https://arxiv.org/abs/1406.2661). <br/>
This model generates MNIST images using GAN. 

## File discription
- main.py: Main function of implemenation, construct and train the model, generates images
- model.py: GAN class
- downlad.py: Files for downlading MNIST data sets
- ops.py: Operation functions
- utils.py: Functions dealing with images processing.
## Prerequisites (my environments)
- Python 3.5.2
- Tensorflow > 0.14
- Numpy

## Usage
First, download dataset with:

    $ python download.py mnist

Second, write the main function with configuration you want.

## Results

![result](assets/result.png)
