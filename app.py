import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ------------------------------
# Page Configuration
# ------------------------------

st.set_page_config(
    page_title="Solar Panel Defect Detection",
    layout="centered"
)

st.title("🔆 Solar Panel Defect Detection")
st.write("Upload an EL image to classify defect severity")

# ------------------------------
# Load Model
# ------------------------------

MODEL_PATH = "saved_models/xception.h5"

model = tf.keras.models.load_model(MODEL_PATH)

# ------------------------------
# Class Labels
# ------------------------------

class_names = [
    "functional",
    "mild",
    "moderate",
    "severe"
]

# ------------------------------
# Image Preprocessing
# ------------------------------

IMG_SIZE = 75

def preprocess_image(image):

    # Convert to RGB
    image = image.convert("RGB")

    # Resize to 75×75
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # Convert to numpy array
    image = np.array(image)

    # Normalize
    image = image / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image

# ------------------------------
# Upload Image
# ------------------------------

uploaded_file = st.file_uploader(
    "Upload EL Image",
    type=["jpg", "jpeg", "png"]
)

# ------------------------------
# Prediction Section
# ------------------------------

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Show uploaded image
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Preprocess image
    processed = preprocess_image(image)

    # Debug shape
    st.write("Processed Shape:", processed.shape)

    # Predict
    prediction = model.predict(processed)

    # Predicted class
    pred_index = np.argmax(prediction)

    pred_class = class_names[pred_index]

    # Confidence score
    confidence = np.max(prediction) * 100

    # ------------------------------
    # Results
    # ------------------------------

    st.subheader("Prediction Result")

    st.success(f"Predicted Class: {pred_class}")

    st.info(f"Confidence Score: {confidence:.2f}%")