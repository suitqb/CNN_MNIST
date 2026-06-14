# MNIST Digit Classifier — From Scratch to PyTorch

Implementation of a neural network for handwritten digit recognition on the MNIST dataset, built progressively from raw NumPy to PyTorch.

## Overview

This project explores the fundamentals of deep learning by implementing the same architecture at two levels of abstraction:

- **NumPy** — fully manual implementation: forward pass, backpropagation, and weight updates written by hand
- **PyTorch** — higher-level reimplementation using autograd and built-in layers *(in progress)*

The goal is to understand what happens under the hood before relying on a framework to abstract it away.

## Architecture

A fully connected deep neural network (DNN) with the following structure:

```
Input (784)  →  Hidden (256, Sigmoid)  →  Hidden (128, Sigmoid)  →  Output (10, Softmax)
```

- Input: 28×28 grayscale pixel values normalized to `[0.01, 1.00]`
- Output: probability distribution over 10 digit classes (0–9)
- Loss: mean squared error
- Optimizer: stochastic gradient descent (SGD)

## Project Structure

```
CNN_MNIST/
├── numpy/
│   ├── convert.py   # converts MNIST binary files (.idx) to CSV
│   └── main.py      # DNN class with manual forward/backward pass
├── pytorch/
│   └── main.py      # PyTorch implementation (in progress)
└── mnist/           # raw MNIST dataset (not tracked)
```

## Getting Started

### Prerequisites

```bash
pip install numpy
```

### 1. Download the MNIST dataset

Download the four binary files from [yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist/) and place them in `mnist/`:

```
mnist/
├── train-images.idx3-ubyte
├── train-labels.idx1-ubyte
├── t10k-images.idx3-ubyte
└── t10k-labels.idx1-ubyte
```

### 2. Convert to CSV

```bash
python numpy/convert.py
```

This generates `train.csv` (60 000 samples) and `test.csv` (10 000 samples).

### 3. Train the model

```bash
python numpy/main.py
```

Training output example:

```
Epoch 1,  Time Spent: 12.4s,  Accuracy: 87.32%
Epoch 5,  Time Spent: 61.8s,  Accuracy: 94.11%
Epoch 20, Time Spent: 247.3s, Accuracy: 97.60%
```

## Key Concepts Implemented

| Concept | Location |
|---|---|
| Weight initialization (He-style scaling) | `DNN.__init__` |
| Sigmoid activation + derivative | `DNN.sigmoid` |
| Stable Softmax | `DNN.softmax` |
| Forward propagation | `DNN.forward_pass` |
| Backpropagation | `DNN.backward_pass` |
| SGD weight update | `DNN.update_weights` |

## Hyperparameters

| Parameter | Value |
|---|---|
| Architecture | `[784, 256, 128, 10]` |
| Epochs | 20 |
| Learning rate | 0.001 |

## What's Next

- [ ] Complete the PyTorch implementation
- [ ] Add convolutional layers (CNN)
- [ ] Plot training curves (loss & accuracy)
- [ ] Export and visualize misclassified samples