import streamlit as st 
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import gdown

# Setup page
st.set_page_config(page_title="Lung Cancer Detection", layout="centered")
st.title("🫁 Lung Cancer Detection Web App")
st.markdown("Upload a **chest X-ray or CT scan** to check for signs of **lung cancer** using a pretrained deep learning model.")

# Model setup
model_dir = "model"
model_file = "best_model.hdf5"
model_path = os.path.join(model_dir, model_file)

# Google Drive model file
gdrive_file_id = "1Qyp8Mnc0W87veyZb4JU5-eucMGFwjkh3"
gdrive_url = f"https://drive.google.com/uc?id={gdrive_file_id}"

# Download model if it doesn't exist
if not os.path.exists(model_path):
    st.warning("⚠️ Model not found locally. Downloading from Google Drive...")
    os.makedirs(model_dir, exist_ok=True)
    try:
        gdown.download(gdrive_url, model_path, quiet=False)
        st.success("✅ Model downloaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")
        st.stop()

# Load the model
model = load_model(model_path)

# File uploader
uploaded_file = st.file_uploader("Upload Lung X-ray or CT Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    def preprocess_image(img):
        img = img.resize((224, 224))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    processed_img = preprocess_image(image)

    with st.spinner("Analyzing..."):
        prediction = model.predict(processed_img)[0][0]

    st.subheader("Prediction Result:")
    if prediction > 0.5:
        st.error(f"🧬 **Lung Cancer Detected**\n\nConfidence: {prediction:.2%}")
    else:
        st.success(f"✅ **No Lung Cancer Detected**\n\nConfidence: {100 - prediction * 100:.2f}%")

    st.caption("⚠️ This tool is for educational purposes only and not a substitute for medical advice.")
