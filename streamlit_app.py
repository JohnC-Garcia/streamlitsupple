import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_weather_model():
    return load_model('weather_classifier.h5')

model = load_weather_model()

st.write("""# Weather Detection System""")
file = st.file_uploader("Choose weather photo from computer", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (64, 64)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    image = image.convert("RGB")  # make sure it's in RGB
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...] / 255.0  # normalize
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['cloudy', 'rain', 'shine', 'sunrise']
    string = "OUTPUT : " + class_names[np.argmax(prediction)]
    st.success(string)
