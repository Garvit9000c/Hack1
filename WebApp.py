import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="Brain Tumor Delector", page_icon="favicon.jpeg", layout='centered', initial_sidebar_state='auto')

URL='[Download Sample Testing Data](https://drive.google.com/drive/folders/1L--z7-1LftMeWfxjgWpEe7qo6r_BqEYi?usp=sharing)'

def import_and_predict(image_data, model):
        size = (100,100)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(100, 100),interpolation=cv2.INTER_CUBIC))/255.
        img_reshape = img_resize[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction

model = tf.keras.models.load_model('/content/Tumor_CNN.hdf5')
class_names = ['Normal Brain', 'Tumorous Brain']
st.write("""
         # Early Stage Brain Tumor Detection Using Convoluted Neural Network
         """
         )
file = st.file_uploader("Please upload an image file", type=["jpg"])
st.markdown(URL, unsafe_allow_html=True)
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    score = tf.nn.softmax(prediction)
    if np.argmax(prediction) == 0:
        st.write("The Brain is Normal!")
    else:
        st.write("Tumor Detected!")
    st.write("This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score)], 100 * np.max(score)))


#Styling Code
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Sans&display=swap');
.css-1v3fvcr{
    background: linear-gradient(90deg, rgba(12,2,179,1) 0%, rgba(20,20,202,1) 35%, rgba(0,212,255,1) 100%);
}
h1:first-of-type, .css-j8zjtb, .css-145kmo2, p{
    color: #FFF3C2;
    font-family: 'Fira Sans', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)
