# Digit Classifier with Streamlit and Scikit-learn

This project is a simple image classification app that recognizes handwritten digits using a Random Forest classifier trained on the scikit-learn digits dataset.

You can try the live app here:  
[Digit Classifier Streamlit App](https://digit-classifier-app-evuz7rad2o9dlwappadirqc.streamlit.app/)

## Project Overview

- Model: Random Forest classifier trained on the `sklearn.datasets.load_digits` dataset.
- Test Accuracy: **97.22%**
- The dataset contains 8x8 pixel grayscale images of handwritten digits.
- The app allows users to upload images of digits, preprocesses them to the required format, and predicts the digit class.

## Important Note on Input Images

**Please upload only 8x8 pixel grayscale images** of handwritten digits.

The model was trained on the [scikit-learn digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html), which consists of small 8x8 images. Uploading images with different sizes or styles may cause inaccurate predictions.

### How to get 8x8 images to test:

- Use digit images directly from the sklearn digits dataset (can be saved using Python).
- Resize your own digit images to 8x8 pixels using an image editor or script (make sure they are grayscale).
- For best results, images should be centered, simple, and have high contrast (white digit on black background).

If you want, you can generate and save sample 8x8 digit images from sklearn using a small Python script.

## Features

- Fast, lightweight model without TensorFlow dependency.
- Simple and intuitive Streamlit web interface.
- Preprocessing includes grayscale conversion, resizing to 8x8, and scaling pixel values to match the training data distribution.

## Getting Started

### Requirements

- Python 3.7+
- Packages listed in `requirements.txt`

### Installation

Clone the repository:

```bash
git clone https://github.com/haroooru/digit-classifier-streamlit.git
cd digit-classifier-streamlit
