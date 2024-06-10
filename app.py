import streamlit as st
import requests
from PIL import Image
import io

st.title("Neural Style Transfer")

content_file = st.file_uploader("Upload Content Image", type=["png", "jpg", "jpeg"])
style_file = st.file_uploader("Upload Style Image", type=["png", "jpg", "jpeg"])

size = st.number_input("Enter Image Size (between 256 and 512):", min_value=256, max_value=512, value=356, step=1)
num_epochs = st.number_input("Enter Number of Epochs (between 1 and 500):", min_value=1, max_value=500, value=101, step=1)
alpha = st.number_input("Enter Content Weight (alpha) (between 0.1 and 10.0):", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
beta = st.number_input("Enter Style Weight (beta) (between 100000 and 10000000):", min_value=1e5, max_value=1e7, value=1e6, step=1e5, format="%f")

if st.button("Generate Image"):
    if content_file and style_file:
        files = {
            'content': content_file.getvalue(),
            'style': style_file.getvalue(),
        }
        data = {
            'size': size,
            'num_epochs': num_epochs,
            'alpha': alpha,
            'beta': beta
        }
        response = requests.post(" https://2ae8-59-178-29-120.ngrok-free.app/style_transfer", files=files, data=data)
        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content))
            st.image(image, caption="Generated Image")
        else:
            st.error("Failed to generate image.")
    else:
        st.error("Please upload both content and style images.")
