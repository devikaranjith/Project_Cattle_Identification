import streamlit as st
st.title("Cattle Identification")
st.text("Upload an image of a cattle nose")

import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import tensorflow_hub as hub


def teachable_machine_classification(img, weights_file):
    # Load the model
    #model = keras.models.load_model(weights_file)

    model = tf.keras.models.load_model(
       (weights_file),
       custom_objects={'KerasLayer':hub.KerasLayer})

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction) # return position of the highest probability



uploaded_file = st.file_uploader("Choose an image of cow's nose ...", type=["jpg","jpeg"])
st.write("")
if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width='auto')
        st.write("")
        st.write("RESULT")
        label = teachable_machine_classification(image, r"D:\DUK\Sem3\Mini_Project\mod1.hdf5")
        tx=('COW '+str(label))

        st.markdown(f'<h1 style="color:#006600;font-size:24px;">{tx}</h1>', unsafe_allow_html=True)
        

        