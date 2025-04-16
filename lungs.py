import streamlit as st 
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import gdown

# Set page config
st.set_page_config(page_title="Lung Cancer Detection", layout="centered")
st.title("🫁 Lung Cancer Detection Web App")
st.markdown("Upload a **chest X-ray or CT scan** to check for signs of **lung cancer** using a pretrained deep learning model.")

# Define model path and Google Drive file ID
model_dir = "model"
model_filename = "lung_model.h5"
model_path = os.path.join(model_dir, model_filename)

gdrive_file_id = "1Qyp8Mnc0W87veyZb4JU5-eucMGFwjkh3"
gdrive_url = f"https://drive.google.com/uc?id={gdrive_file_id}"

# Check if model exists, otherwise try to download
if not os.path.exists(model_path):
    st.warning("⚠️ Model file not found locally. Attempting to download from Google Drive...")
    os.makedirs(model_dir, exist_ok=True)

    try:
        gdown.download(gdrive_url, model_path, quiet=False)
    except Exception as e:
        st.error(f"❌ Failed to download model: {e}")
        st.stop()

    # Confirm if the file downloaded
    if not os.path.exists(model_path):
        st.error("❌ Model download failed. Please download `lung_model.h5` manually and place it in the `model/` folder.")
        st.stop()

# Load the model
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
        img_array = img_array / 255.0  # Normalize
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
