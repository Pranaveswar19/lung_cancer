import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import gdown

# Title
st.set_page_config(page_title="Lung Cancer Detection", layout="centered")
st.title("🫁 Lung Cancer Detection Web App")
st.markdown("Upload a **chest X-ray or CT scan** to check for signs of **lung cancer** using a pretrained deep learning model.")

# Load model
model_path = "model/lung_model.h5"
gdrive_url = "https://drive.google.com/file/d/1Qyp8Mnc0W87veyZb4JU5-eucMGFwjkh3/view?usp=sharing"

if not os.path.exists(model_path):
    st.error("⚠️ Trained model not found! Please add `lung_model.h5` in the `model/` directory.")
    st.stop()

model = load_model(model_path)

# Image uploader
uploaded_file = st.file_uploader("Upload Lung X-ray or CT Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    def preprocess_image(img):
        img = img.resize((224, 224))
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # normalize
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    processed_img = preprocess_image(image)

    # Prediction
    with st.spinner("Analyzing..."):
        prediction = model.predict(processed_img)[0][0]

    # Display result
    st.subheader("Prediction Result:")
    if prediction > 0.5:
        st.error(f"🧬 **Prediction: Lung Cancer Detected**\n\nConfidence: {prediction:.2%}")
    else:
        st.success(f"✅ **Prediction: Normal / No Lung Cancer Detected**\n\nConfidence: {100 - prediction * 100:.2f}%")

    st.markdown("---")
    st.caption("⚠️ This tool is for educational purposes only and not a substitute for medical advice.")
