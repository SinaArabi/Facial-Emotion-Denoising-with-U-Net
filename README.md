# Facial Emotion Denoising with U-Net

## Overview
This project utilizes a U-Net model to denoise facial emotion images from the FER2013 dataset. The goal is to reconstruct clean facial expressions from noisy inputs, improving the performance of facial emotion recognition systems.

## Dataset
The FER2013 dataset consists of 48x48 pixel grayscale images labeled with one of seven emotions:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

The dataset includes approximately 35,000 training images and 7,000 test images. You can download the dataset from [Kaggle - FER2013](https://www.kaggle.com/datasets/msambare/fer2013). Add the dataset files to the ./dataset directory for more compatability.

## Model Architecture
The model is based on the U-Net architecture, which is effective for image-to-image tasks. It consists of:
- **Encoder**: A series of convolutional layers with max-pooling to capture context.
- **Bottleneck**: A bridge between the encoder and decoder.
- **Decoder**: A series of up-convolutional layers to enable precise localization.
- **Skip Connections**: To retain high-resolution features from the encoder.

## Preprocessing
- Images are resized to 48x48 pixels.
- The dataset is split into training and testing sets.
- Images are normalized to a range of 0 to 1.
- Salt-and-Pepper noise is added to the images to simulate noisy conditions.

## Training
- **Optimizer**: Adam optimizer is used to minimize the Mean Squared Error (MSE).
- **Loss Function**: MSE is used as the loss function for training the denoising model.
- **Epochs**: The model is trained for a specified number of epochs.
- **Batch Size**: A batch size of 64 is used during training.

## Evaluation
The model's performance is evaluated using several metrics:
- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values.
- **Peak Signal-to-Noise Ratio (PSNR)**: Measures the quality of the reconstructed images.
- **Structural Similarity Index (SSIM)**: Measures the similarity between two images.

## Installation
To run this project, you need the following dependencies:

```bash
pip install tensorflow numpy matplotlib scikit-image pandas
```
Then you can use the notebook in order to train your model.
## Results
The model achieves a PSNR of approximately 30 dB and an SSIM of 0.95 on the test set, indicating effective denoising capabilities.
