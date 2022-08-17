import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import streamlit as st

st.set_page_config(
     page_title="Data For Thought: Neural Style Transfer",
)

@st.cache
def load_model():
    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    return hub_module

st.title('Neural Style Transfer')

st.image("https://images.unsplash.com/photo-1655635643568-f30d5abc618a?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2500&h=640&q=80")

st.markdown('### What is neural style transfer: A little theory')

st.write('Neural style transfer is a way of manipulating images or other media to alter their appearance to that of another image.')

st.write('The neural network takes two images as inputs (one referred to as the content image and the other as the style image) - the content image acts as the base image to which we will apply the style from the style image.')

st.write('As for the underlying network itself, the network contains a bottleneck in the middle-most hidden layer. This so-called bottleneck acts as the boundary between the encoding and decoding processes - the encoding process is where the style image is broken down to its features and the decoding process is where the content image is built up using such features.')

st.write('For a deeper explanation of the archiecture - [Towards Data Science article](https://towardsdatascience.com/light-on-math-machine-learning-intuitive-guide-to-neural-style-transfer-ef88e46697ee)')

st.markdown('### Demo using Google Magenta')

uploaded_content_file = st.file_uploader("Choose a content file")
uploaded_style_file = st.file_uploader("Choose a style file")
if uploaded_content_file and uploaded_style_file is not None:
    content_image = plt.imread(uploaded_content_file)
    style_image = plt.imread(uploaded_style_file)
    content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
    hub_model = load_model()
    outputs = hub_model(tf.constant(content_image), tf.constant(style_image))
    stylised_image = outputs[0]
    cv2.imwrite("./cache/image.jpg", cv2.cvtColor(np.squeeze(stylised_image)*255, cv2.COLOR_BGR2RGB))
    st.image("./cache/image.jpg")
    st.write('Above you can see the output of the content image being re-imagined by the Google Magenta algorithm in the style of the style image you uploaded!')
    with st.expander("See code"):
        code = '''def neural_style_transfer():
          import tensorflow_hub as hub
          import tensorflow as tf
          import numpy as np
          import matplotlib.pyplot as plt
          import cv2
          
          content_image = plt.imread(uploaded_content_file)
          style_image = plt.imread(uploaded_style_file)
          content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
          style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
          hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
          outputs = hub_model(tf.constant(content_image), tf.constant(style_image))
          stylised_image = outputs[0]
          cv2.imwrite("./cache/image.jpg", cv2.cvtColor(np.squeeze(stylised_image)*255, cv2.COLOR_BGR2RGB))'''
        st.code(code, language='python')
    st.header('~ Ethan')