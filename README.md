# Traffic Sign Recognition System

A deep-learning traffic sign classification system built with **TensorFlow, OpenCV, and NumPy** using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

The project implements an end-to-end computer vision workflow covering image preprocessing, CNN training, evaluation, and prediction on unseen traffic sign images.

## Features

- Multiclass classification across **43 traffic sign categories**
- OpenCV-based image loading and preprocessing
- Image resizing and normalization with NumPy
- TensorFlow/Keras Convolutional Neural Network (CNN)
- Stratified training, validation, and test splits
- Model checkpointing and training callbacks
- Classification report generation
- Confusion matrix visualization
- Command-line prediction for new traffic sign images
- Automated tests with pytest

## Dataset

The project uses the **German Traffic Sign Recognition Benchmark (GTSRB)**.

Dataset used during training:

- Total images: **26,640**
- Traffic sign classes: **43**
- Training samples: **21,312**
- Validation samples: **2,664**
- Test samples: **2,664**
- Input image size: **32 × 32 × 3**

The dataset itself is excluded from Git to keep the repository lightweight.

## Model Performance

The trained CNN achieved:

| Metric | Result |
|---|---:|
| Test Accuracy | **99.85%** |
| Test Loss | **0.0172** |
| Evaluation Accuracy | **99.81%** |

Performance can vary when the model is retrained because of training randomness and environment differences.

## Project Structure

```text
traffic-sign-recognition/
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
├── data/
├── models/
├── reports/
│   └── figures/
├── requirements.txt
├── .gitignore
└── README.md
