import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

st.title("Weather Image Classifier 🌦️")

uploaded_file = st.file_uploader("Upload a weather image...", type=["jpg", "png", "jpeg"])
model = load_model("weather_classifier.h5")

class_names = ['cloudy', 'rain', 'shine', 'sunrise']

def import_and_predict(image_data, model):
      size = (150, 150)
      image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
      img = np.asarray(image)
      img_reshape = img[np.newaxis, ...] / 255.0
      prediction = model.predict(img_reshape)
      return prediction

if uploaded_file is not None:
      image = Image.open(uploaded_file)
      st.image(image, caption='Uploaded Image', use_column_width=True)
      prediction = import_and_predict(image, model)
      label = class_names[np.argmax(prediction)]
      st.subheader(f"Prediction: **{label.upper()}**")
