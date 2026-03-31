# Digit Recognizer CNN

## Overview

This project trains a CNN to recognize handwritten digits (0-9) using the MNIST dataset, a benchmark collection of 70,000 grayscale images commonly used in computer vision and machine learning. Each image is 28x28 pixels and labeled with the digit it represents (0-9).

---

## Key Concepts Implemented

- **Convolutional Layers:** Two `Conv2d` layers extract spatial features from the images. Without padding, each convolution reduces the spatial dimensions by 2 pixels on each side.

- **Max Pooling:** Applied after each convolutional layer with a 2x2 kernel to downsample the feature maps, halving spatial dimensions each time.

- **Dimension Tracking"** The spatial flow through the network:
```
28x28 -> 26x26 -> 13x13 -> 11x11 -> 5x5
```
This leads to a flattened feature vector of `16 x 5 x 5 = 400` values.

- **Fully Connected Layers:** Three linear layers `(400 -> 120 -> 84 -> 10)` map the extracted features to class scores.

- **Activation Functions:** ReLU is applied after each convolution and hidden fully connected layer. log_softmax is applied at the output.

- **Loss and Optimizer:** CrossEntropyLoss for multi-class classification with an Adam optimizer at a learning rate of 0.001.

- **Training Loop:** 5 epochs, with batch-level logging every 600 batches and per-epoch loss/accuracy tracking.

---

## Tools and Libraries

| Library | Purpose |
|---|---|
| `torch` / `torch.nn` | Model definition and training |
| `torchvision` | MNIST dataset loading and transforms |
| `torch.optim` | Adam optimizer |
| `matplotlib` | Loss and accuracy plots |
| `numpy` / `pandas` | General data utilities |

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **98.89%** |
| Correct Predictions | 9,889 / 10,000 |
| Training Time | ~3.06 minutes (T4 GPU) |
| Epochs | 5 |

Loss decreased steadily across epochs, and validation accuracy closely tracked training accuracy, indicating good generalization with minimal overfitting.
