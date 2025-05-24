import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import joblib
import cv2

# Load model
clf = joblib.load('digit_classifier_rf.joblib')

st.title("Digit Classifier (sklearn RandomForest)")

uploaded_file = st.file_uploader("Upload an image of a digit", type=['png', 'jpg', 'jpeg'])

def preprocess(image):
    # Convert to grayscale
    image = ImageOps.grayscale(image)
    # Resize to 8x8 pixels (like sklearn digits dataset)
    image = image.resize((8, 8), Image.Resampling.LANCZOS)
    # Convert to numpy array
    image_np = np.array(image)
    # Binarize the image to remove noise (threshold at 128)
    image_np = (image_np > 128).astype(float) * 16  # pixels 0 or 16
    # Invert colors (digits dataset has white on black)
    image_np = 16 - image_np
    # Flatten to 1D array
    return image_np.flatten().reshape(1, -1)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    input_data = preprocess(image)
    prediction = clf.predict(input_data)
    st.write(f"Predicted digit: {prediction[0]}")



