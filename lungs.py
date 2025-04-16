import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Page setup
st.set_page_config(page_title="Lung Cancer Detection", layout="centered")
st.title("🫁 Lung Cancer Detection (Demo Model)")

# Load pretrained model
model_path = "model/model.pkl"
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("🚫 Model file not found. Please upload `model.pkl` to the `model/` folder.")
    st.stop()

# Image uploader
uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Fake feature extraction (just for demo)
    image = image.resize((64, 64))  # Downsize
    img_array = np.array(image).flatten()[:1024]  # Take first 1024 pixels
    features = np.pad(img_array, (0, max(0, 1024 - len(img_array))), mode='constant')  # Pad to 1024 if needed
    features = features.reshape(1, -1)

    # Predict
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][prediction]

    # Output
    st.subheader("🧪 Prediction Result:")
    if prediction == 1:
        st.error(f"🧬 Lung Cancer Detected\nConfidence: {proba:.2%}")
    else:
        st.success(f"✅ No Lung Cancer Detected\nConfidence: {proba:.2%}")
