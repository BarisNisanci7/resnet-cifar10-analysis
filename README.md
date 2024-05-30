# CIFAR-10 Classification using ResNet Bottleneck Residual Blocks

This repository contains my implementation of CIFAR-10 classification using a ResNet model with bottleneck residual blocks, as well as experiments with different training dynamics and transfer learning using a pre-trained ResNet-50.

## Introduction
This project aims to classify images from the CIFAR-10 dataset using a Convolutional Neural Network (CNN) with bottleneck residual blocks, inspired by the ResNet architecture. Additionally, different training dynamics and transfer learning approaches are explored to optimize performance.

## Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is divided into 50,000 training images and 10,000 test images. In this project, the dataset is further split into training and validation sets.

## Model Architecture

### Bottleneck Residual Block
The bottleneck residual block used in ResNet is designed to reduce the computational complexity of the network while maintaining performance. It consists of three convolutional layers:
1. A 1x1 convolutional layer that reduces the dimensionality.
2. A 3x3 convolutional layer.
3. A 1x1 convolutional layer that restores the original dimensionality.

Each convolutional layer is followed by Batch Normalization and a ReLU activation function. A residual connection adds the input of the block to the output.

### CNN Model
The CNN model includes several bottleneck residual blocks, along with additional layers to handle the CIFAR-10 classification task.

## Training

### SGD with Momentum and L2 Regularization
The initial training is done using Stochastic Gradient Descent (SGD) with momentum and L2 regularization.

- **Batch Size:** 32
- **Epochs:** 10
- **Learning Rate:** Optimized through experiments
- **Momentum:** Optimized through experiments
- **L2 Regularization:** Implemented via loss function

### ADAM with Weight Decay
In the second part, ADAM optimizer with weight decay is used instead of SGD.

- **Weight Decay:** Optimized through experiments
- **Learning Rate:** Optimized through experiments

### Learning Rate Scheduling
A learning rate scheduler (`ReduceLROnPlateau`) is used to reduce the learning rate when the validation accuracy stops improving.

- **Patience:** 2 epochs

## Transfer Learning
Transfer learning is applied using a pre-trained ResNet-50 model. All layers are frozen except for the last layer, which is modified to classify CIFAR-10 classes.

## Results
The training and validation performance are monitored and visualized through learning and generalization curves. The best architecture and hyperparameters are reported.

## Usage
To run the code and train the model, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cifar10-resnet.git
   cd cifar10-resnet
   
2. Install the required packages:
  pip install -r requirements.txt

3. Run the training script:
  python train.py

