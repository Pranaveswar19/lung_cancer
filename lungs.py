import streamlit as st
import numpy as np
from tensorflow.keras.models import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Title
st.set_page_config(page_title="Lung Cancer Detection", layout="centered")
st.title("🫁 Lung Cancer Detection Web App")
st.markdown("Upload a **chest X-ray or CT scan** to check for signs of **lung cancer** using a pretrained deep learning model.")

# Load pre-trained ResNet50 model + higher level layers
base_model = ResNet50(weights='imagenet', include_top=False)  # Exclude final fully connected layers

# Adding custom layers for lung cancer classification
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())  # Global average pooling layer
model.add(Dense(1024, activation='relu'))  # Fully connected layer
model.add(Dense(1, activation='sigmoid'))  # Sigmoid activation for binary classification (cancer vs normal)

# Freeze the layers of ResNet50 model to retain pre-trained features
for layer in base_model.layers:
    layer.trainable = False

# Image uploader
uploaded_file = st.file_uploader("Upload Lung X-ray or CT Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image with PIL and display it
    image = load_img(uploaded_file, target_size=(224, 224))  # Resizing image to 224x224
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = preprocess_input(image_array)  # Preprocessing for ResNet50

    # Prediction
    with st.spinner("Analyzing..."):
        prediction = model.predict(image_array)[0][0]

    # Display result
    st.subheader("Prediction Result:")
    if prediction > 0.5:
        st.error(f"🧬 **Prediction: Lung Cancer Detected**\n\nConfidence: {prediction:.2%}")
    else:
        st.success(f"✅ **Prediction: Normal / No Lung Cancer Detected**\n\nConfidence: {100 - prediction * 100:.2f}%")

    st.markdown("---")
    st.caption("⚠️ This tool is for educational purposes only and not a substitute for medical advice.")
