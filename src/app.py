import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from src.utils import inv_class_mapping  

model = tf.keras.models.load_model('models/best_model.keras')

def preprocess_image(img):
    img = img.resize((168, 168))
    img = img.convert('L')  # grayscale
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0) 
    img = np.expand_dims(img, axis=-1)  # channel
    return img

st.title('Brain Tomur Classification')
uploaded_file = st.file_uploader("Upload your image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='uploaded image', width='stretch')
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    class_idx = np.argmax(prediction)
    class_name = inv_class_mapping[class_idx]
    st.write(f'prediction: {class_name}')