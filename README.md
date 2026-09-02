# Traffic Sign Recognition System

A computer vision project that classifies German traffic signs into 43 classes using a TensorFlow Convolutional Neural Network (CNN) with OpenCV and NumPy preprocessing.

## Features

- Downloads and loads the GTSRB traffic-sign dataset.
- Preprocesses images using OpenCV and NumPy.
- Resizes images to 32 × 32 RGB format.
- Normalizes pixel values to the range [0, 1].
- Trains a TensorFlow CNN for 43-class image classification.
- Uses batch normalization, max pooling, dropout, and dense layers.
- Uses early stopping, learning-rate reduction, and model checkpointing.
- Generates evaluation metrics and a classification report.
- Generates a confusion matrix for model analysis.
- Supports command-line prediction for individual traffic-sign images.
- Includes automated preprocessing tests using pytest.

## Tech Stack

- Python 3.11
- TensorFlow
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- pytest

## Dataset

The downloaded GTSRB archive used by this project contained:

- **26,640 images**
- **43 traffic-sign classes**
- Image size after preprocessing: **32 × 32 × 3**

A reproducible stratified split using `random_state=42` produced:

| Split | Images |
|---|---:|
| Training | 21,312 |
| Validation | 2,664 |
| Held-out Test | 2,664 |

The dataset itself is excluded from Git to keep the repository lightweight.

## Image Preprocessing

Each image follows this preprocessing pipeline:

```text
Input Image
    |
    v
OpenCV Image Loading
    |
    v
Resize to 32 x 32
    |
    v
Convert to RGB
    |
    v
Normalize Pixels to [0, 1]
    |
    v
TensorFlow CNN
    |
    v
43-Class Prediction
```

## CNN Architecture

The classification model includes:

- Convolutional layers
- Batch normalization
- Max-pooling layers
- Dropout regularization
- Dense classification layer
- 43-class softmax output layer

The model is trained using the Adam optimizer and sparse categorical cross-entropy loss.

## Model Performance

The training run achieved:

| Metric | Result |
|---|---:|
| Test Accuracy | **99.85%** |
| Test Loss | **0.0172** |

The persisted model checkpoint was separately reloaded and evaluated:

| Metric | Result |
|---|---:|
| Persisted Model Accuracy | **99.81%** |

These results are based on a held-out stratified image-level split of the downloaded archive and should not be interpreted as official GTSRB benchmark test-set accuracy.

## Example Prediction

A class-14 traffic-sign image was tested using the saved model:

```text
Predicted class ID: 14
Traffic sign: Stop
Confidence: 100.00%
```

## Project Structure

```text
traffic-sign-recognition/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── download_data.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   ├── preprocess.py
│   └── train.py
├── tests/
│   └── test_preprocess.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
│   ├── metrics.json
│   ├── classification_report.json
│   └── figures/
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Lahari76/traffic-sign-recognition.git
cd traffic-sign-recognition
```

### 2. Create a Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## Download the Dataset

```bash
python -m src.download_data
```

The downloaded dataset is extracted under:

```text
data/raw/gtsrb/
```

## Train the Model

```bash
python -m src.train
```

Training generates the saved model, metrics, and training figures.

## Evaluate the Model

```bash
python -m src.evaluate
```

Evaluation generates the classification report and confusion matrix.

## Predict a Traffic Sign

Run:

```bash
python -m src.predict path/to/image.ppm
```

Example:

```bash
python -m src.predict data/raw/gtsrb/14/00015_00010.ppm
```

## Run Tests

```bash
pytest -q
```

Verified test result:

```text
6 passed
```

## Generated Outputs

The project generates artifacts including:

```text
models/traffic_sign_cnn.keras
reports/metrics.json
reports/classification_report.json
reports/figures/training_accuracy.png
reports/figures/training_loss.png
reports/figures/confusion_matrix.png
```

Large datasets, trained model files, and generated figures are excluded from Git to keep the repository lightweight. Small JSON evaluation reports are retained as evidence of the verified model run.