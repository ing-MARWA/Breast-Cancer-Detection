import streamlit as st
import numpy as np
from keras.models import load_model
import tensorflow as tf
from PIL import Image, ImageOps
import cv2

st.title('Breast Cancer Classification App')
st.header("Breast Cancer Prediction")
# Open the image
image = Image.open('Breast.jpg')

# Resize the image
new_size = (350, 250)
resized_image = image.resize(new_size)

# Save the resized image to a new file
resized_image.save('smaller_Breast.jpg')
image = Image.open('smaller_Breast.jpg')
st.image(image, caption='Breast Breast Ultrasound Images scans')
# Load the trained model
model = load_model("D:\Graduation Project\Breast Cancer\model.keras")


# Add a file uploader to get the user's input image
uploaded_file = st.file_uploader("Choose an image...", type="png")


def import_and_predict(image_data, model):
    size = (224, 224) 
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction


if uploaded_file is None:
    st.text("No image file has been uploaded.")
else:
    image = Image.open(uploaded_file)
    predictions = import_and_predict(image, model)
    class_names = ['Benign', 'Malignant', 'Normal']
    string = "The patient is predicted to be: " + class_names[np.argmax(predictions)]
    st.success(string)
    st.image(image)