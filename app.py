import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load your trained model
model = tf.keras.models.load_model('autismodel.h5')

# App title
st.title("Anokhi - Autism Detection App")

# Upload image
uploaded_image = st.file_uploader("Upload an image of the child", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display uploaded image
    st.image(uploaded_image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    image = Image.open(uploaded_image)
    image = image.resize((224, 224))  # Resize to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values

    # Make a prediction
    prediction = model.predict(np.expand_dims(image, axis=0))
    prediction_value = prediction[0]  # Get the prediction value from the array

    # Compare the prediction value with the threshold
    prediction_text = "Child has Autism" if prediction_value < 0.5 else "Child does not have Autism"


    # Display prediction
    st.write(f"Prediction: {prediction_text}")
